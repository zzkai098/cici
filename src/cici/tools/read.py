"""read tool — view file contents with line numbers.

Line numbers are not decoration: they are the coordinate system edit's insert
command uses, so what read prints and what insert accepts have to agree.
"""

from typing import ClassVar

from .base import SyncTool
from .workspace import Workspace

# A model that reads a 40k-line file has spent its context on one tool call.
# Cap here and say so out loud, so it knows to narrow the range rather than
# assuming it saw everything.
MAX_LINES = 2000
MAX_LINE_CHARS = 2000


class ReadTool(SyncTool):
    name = "read"
    description = (
        "Read a text file from the workspace and return its contents with line "
        "numbers. Use this before editing a file, so the exact text passed to "
        "edit matches what is on disk. For directory listings or file search, "
        "use bash instead."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file, relative to the workspace root "
                    "(e.g. 'src/cici/agent.py'). Paths outside the workspace "
                    "are rejected."
                ),
            },
            "offset": {
                "type": "integer",
                "description": (
                    "1-based line number to start reading from. Omit to start "
                    f"at line 1. Use with limit to page through a file longer "
                    f"than {MAX_LINES} lines."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    f"How many lines to return. Defaults to {MAX_LINES}, which "
                    "is also the maximum."
                ),
            },
        },
        "required": ["path"],
    }
    parallel_safe = True

    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()

    def _run(self, path, offset=None, limit=None):
        resolved = self.workspace.resolve(path)
        shown = self.workspace.relative(resolved)

        if resolved.is_dir():
            raise IsADirectoryError(
                f"{shown} is a directory, not a file — use bash (ls) to list it"
            )
        if not resolved.exists():
            raise FileNotFoundError(f"no such file: {shown}")

        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(
                f"{shown} is not UTF-8 text and cannot be shown as source"
            ) from e

        if text == "":
            return f"{shown} is empty (0 bytes)"

        lines = text.split("\n")
        # A trailing newline splits into a final empty element that is not a
        # line; dropping it keeps the count honest.
        if lines and lines[-1] == "":
            lines.pop()

        total = len(lines)
        start = max(1, offset or 1)
        if start > total:
            raise ValueError(
                f"offset {start} is past the end of {shown} ({total} lines)"
            )
        count = min(limit or MAX_LINES, MAX_LINES)
        window = lines[start - 1 : start - 1 + count]

        width = len(str(start + len(window) - 1))
        out = []
        for i, line in enumerate(window, start):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + f"… [line truncated, {len(line)} chars]"
            out.append(f"{i:>{width}}\t{line}")

        end = start + len(window) - 1
        if end < total:
            out.append(
                f"\n[showing lines {start}-{end} of {total}; "
                f"call read again with offset={end + 1} for the rest]"
            )
        return "\n".join(out)
