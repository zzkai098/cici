"""ls tool — list what is in a directory."""

from typing import ClassVar

from .base import SyncTool
from .workspace import PRUNED, Workspace

MAX_ENTRIES = 500
MAX_DEPTH = 3


class LsTool(SyncTool):
    name = "ls"
    description = (
        "List the files and directories inside a directory in the workspace. "
        "Use it to find out what exists before reading or editing something. "
        "Dependency and build directories (.git, .venv, node_modules, "
        "__pycache__ and similar) are never listed — use bash if you genuinely "
        "need to look inside one. To search file contents rather than names, "
        "use grep."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory to list, relative to the workspace root "
                    "(e.g. 'src/cici/tools'). Defaults to the workspace root "
                    "itself. Paths outside the workspace are rejected."
                ),
            },
            "depth": {
                "type": "integer",
                "description": (
                    "How many levels to descend. 1, the default, lists only "
                    f"the directory's own contents; 2 also lists what is inside "
                    f"each subdirectory, up to a maximum of {MAX_DEPTH}. Start "
                    "at 1 and go deeper only when you actually need the layout."
                ),
            },
        },
        "required": [],
    }
    parallel_safe = True

    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()

    def _run(self, path=None, depth=None):
        root = self.workspace.resolve(path or ".")
        shown = self.workspace.relative(root)

        if not root.exists():
            raise FileNotFoundError(f"no such directory: {shown}")
        if not root.is_dir():
            raise NotADirectoryError(
                f"{shown} is a file, not a directory — use read to see its contents"
            )

        # The model's depth is a request; this is the ceiling.
        levels = max(1, min(depth or 1, MAX_DEPTH))
        entries, unreadable = self._walk(root, levels)

        if not entries:
            return f"{shown} is empty"

        header = f"{shown} — {len(entries)} entries, depth {levels}"
        body = entries[:MAX_ENTRIES]
        out = [header, *body]

        if len(entries) > MAX_ENTRIES:
            out.append(
                f"[showing {MAX_ENTRIES} of {len(entries)}; narrow the path or "
                f"lower depth to see the rest]"
            )
        for d in unreadable:
            out.append(f"[{d} could not be read: permission denied]")
        return "\n".join(out)

    def _walk(self, root, levels):
        """Depth-first walk, pruning noise directories BEFORE descending.

        Pruning on the way in rather than filtering the result is the whole
        point: a .venv is skipped without ever paying to walk its 4000 files.
        That is also what keeps collecting every entry before truncating cheap
        enough to be honest about how many were omitted.
        """
        entries = []
        unreadable = []

        def visit(directory, prefix, level):
            try:
                children = sorted(directory.iterdir(), key=lambda p: p.name)
            except PermissionError:
                unreadable.append(prefix.rstrip("/") or ".")
                return

            for child in children:
                # A symlink is rendered but never followed: resolve() guards the
                # path the model asked for, and descending through a link could
                # walk straight back out of the workspace.
                if child.is_symlink():
                    entries.append(f"{prefix}{child.name}@")
                    continue

                is_dir = child.is_dir()
                if is_dir and child.name in PRUNED:
                    continue

                rendered = f"{prefix}{child.name}{'/' if is_dir else ''}"
                entries.append(rendered)
                if is_dir and level < levels:
                    visit(child, rendered, level + 1)

        visit(root, "", 1)
        return entries, unreadable
