"""Terminal rendering. Ported from cici_101/webagent_stream.py.

Terminal output is plain text — markdown renders as literal characters here,
so no bold, no headers, no bullet syntax.
"""

import sys
import threading

CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"

JELLYFISH = r"""
       .-~~~~~-.
     .'  o   o  '.
    /       -      \
    '.___________.'
      ~  ~  ~  ~  ~
"""


def _tty():
    return sys.stdout.isatty()


class Tentacles:
    """Jellyfish tentacles that drift while we wait on the model or a tool."""

    FRAMES = ("∿  ∿  ∿  ∿  ∿", " ∿  ∿  ∿  ∿  ∿", "  ∿  ∿  ∿  ∿  ∿", " ∿  ∿  ∿  ∿  ∿")

    def __init__(self, label=""):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def start(self, label=None):
        if label is not None:
            self.label = label
        if self._thread or not _tty():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r{CYAN}  {frame}{RESET} {DIM}{self.label}{RESET}\033[K")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.15)

    def stop(self):
        """Halt the animation and wipe the line so real output starts clean."""
        if not self._thread:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def banner(model):
    cyan = CYAN if _tty() else ""
    dim = DIM if _tty() else ""
    reset = RESET if _tty() else ""
    print(f"{cyan}{JELLYFISH}{reset}")
    print("  Hi, I'm cici, how can I help you today :D")
    print(f"{dim}  {model} · Ctrl-C to exit{reset}\n")


def describe_tool_input(tool_input):
    """Render a tool's input for display. Never render as blank — an empty
    line reads like a bug, so say what was actually there."""
    if not tool_input:
        return "(empty)"
    if not isinstance(tool_input, dict):
        return str(tool_input)
    for key in ("query", "command", "path", "file_path"):
        val = tool_input.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ", ".join(f"{k}={v!r}" for k, v in tool_input.items()) or "(empty)"


def dim_open():
    """Open a dim span — no-op when not a TTY so piped output stays clean."""
    return DIM if _tty() else ""


def dim_close():
    return RESET if _tty() else ""


def rule(width=50):
    print("\n" + "─" * width)
