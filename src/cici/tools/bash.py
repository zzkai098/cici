"""bash tool — run a shell command in the workspace.

Stays on Tool rather than SyncTool because the work here is genuinely async:
asyncio.timeout around an asyncio subprocess can actually kill a hung command,
which subprocess.run(timeout=) cannot — it blocks the whole process for the
duration and leaves the child's process group behind.

Not a sandbox, and unlike the file tools it is not confined to the workspace.
Confirmation is the only thing standing between the model and the machine, so
it is on by default; CICI_YOLO=1 turns it off for evals and batch runs.
Anything claiming this is sandboxed would be a lie.
"""

import asyncio
import os
import signal
import sys
from typing import ClassVar

from .base import Tool
from .workspace import Workspace

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
# Enough to see what happened, not enough to bury the context window.
MAX_OUTPUT_CHARS = 30_000


def _truncate(text, budget=MAX_OUTPUT_CHARS):
    """Keep the head and the tail — the error is usually in one or the other."""
    if len(text) <= budget:
        return text
    half = budget // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n\n… [{dropped} chars omitted] …\n\n{text[-half:]}"


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command from the workspace root and return its stdout, "
        "stderr and exit code. Use it for things the file tools do not cover: "
        "listing directories, searching with grep or rg, running tests, git. "
        "To read or change a file's contents, prefer read / write / edit — they "
        "give better errors and can be undone."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "The shell command to run, e.g. 'ls -la src' or "
                    "'uv run pytest -q'. Runs through the shell, so pipes, "
                    "redirection and && work. The working directory is the "
                    "workspace root and does not persist between calls — chain "
                    "with && instead of relying on a previous cd."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Seconds to wait before killing the command. Defaults to "
                    f"{DEFAULT_TIMEOUT}, maximum {MAX_TIMEOUT}. Raise it for "
                    "test suites or installs."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()

    def _confirm(self, command):
        """Ask the user before running. Returns None to allow, or a reason to refuse.

        A plain blocking input(), for the same reason cli/repl.py uses one: this
        runs inside asyncio.Runner, so a Ctrl-C here cancels the agent task, and
        a task parked in a to_thread(input) could never receive that
        cancellation. Nothing else is waiting on the loop at this moment anyway.
        """
        if os.getenv("CICI_YOLO"):
            return None
        if not sys.stdin.isatty():
            return (
                "refused: cici cannot ask for confirmation because stdin is not "
                "a terminal. Set CICI_YOLO=1 to allow commands without asking."
            )
        print(f"\n  run: {command}")
        try:
            answer = input("  allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "refused: the user declined to run this command"
        if answer in ("y", "yes"):
            return None
        return "refused: the user declined to run this command"

    async def run(self, command=None, timeout=None):
        if not command or not command.strip():
            raise ValueError("command is required")

        refusal = self._confirm(command)
        if refusal:
            # Returned, not raised: the model should see a refusal as a normal
            # outcome it can work around, not as a tool that broke.
            return refusal

        seconds = min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace.root,
            # Own process group, so the timeout kills the whole tree rather
            # than just the shell and orphaning whatever it spawned.
            start_new_session=True,
        )
        try:
            async with asyncio.timeout(seconds):
                stdout, stderr = await proc.communicate()
        except TimeoutError:
            self._kill(proc)
            await proc.wait()
            raise TimeoutError(
                f"command exceeded {seconds}s and was killed: {command}"
            ) from None

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")

        parts = [f"exit code: {proc.returncode}"]
        if out.strip():
            parts.append(f"stdout:\n{_truncate(out)}")
        if err.strip():
            parts.append(f"stderr:\n{_truncate(err)}")
        if not out.strip() and not err.strip():
            parts.append("(no output)")
        return "\n\n".join(parts)

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
