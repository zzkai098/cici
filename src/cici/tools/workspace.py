"""Shared filesystem access for the file tools.

Every path a tool touches goes through Workspace.resolve, which confines it to
the directory cici was launched in. cici is meant to run inside other people's
projects, so that root is captured once at startup — not read from the cwd on
each call, which a tool could otherwise move.

Ported from cici_101/005_text_editor_tool.ipynb::TextEditorTool, with two
changes that are corrections rather than preferences — see resolve() and
backup().
"""

import shutil
import time
from pathlib import Path

# Directories no tool ever descends into. Vendored dependencies and build
# output tell the model nothing, and walking them costs real time on a large
# repo. Shared by ls and grep so the two agree on what the workspace "is" —
# a listing that hides .venv and a search that finds things in it would be
# incoherent.
# TODO (roadmap 3): read .gitignore instead of hard-coding this.
PRUNED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cici",
}


class Workspace:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else Path.cwd().resolve()
        # roadmap 3 puts sessions under .cici/ too, so the agent gets one
        # namespaced directory in the user's project instead of several.
        self.backup_dir = self.root / ".cici" / "backups"

    def resolve(self, path):
        """Resolve a tool-supplied path and confine it to the workspace root.

        The original used `abs_path.startswith(self.base_dir)`, which is a
        prefix test, not a path test: with root /home/me/app it accepts
        /home/me/app-secrets. Path.is_relative_to compares path components, so
        it does not. Resolving first also means a symlink pointing outside the
        root is caught by its target, not waved through by its name.

        An absolute path is not silently reinterpreted as relative: Path.__truediv__
        discards the left side when the right side is absolute, so it lands
        outside the root and is rejected here like any other escape.
        """
        if not path or not str(path).strip():
            raise ValueError("path is required")
        resolved = (self.root / path).resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise ValueError(
                f"path escapes the workspace: {path!r} resolves outside {self.root}"
            )
        return resolved

    def relative(self, resolved):
        """Render a resolved path the way the user typed it, for messages."""
        try:
            return str(resolved.relative_to(self.root))
        except ValueError:
            return str(resolved)

    def walk_files(self, start):
        """Yield every file under start, deepest-first within each directory.

        Shared by the tools that need to traverse rather than list. Two rules
        are enforced here rather than in each caller, because getting either
        wrong is a security or performance bug: PRUNED directories are never
        descended into, and symlinks are never followed — resolve() guards the
        path the model named, but a link met mid-walk would lead straight back
        out of the workspace.

        Children are sorted at every level: iterdir() order is not stable, and
        an unstable result reads to the model as the files having changed.
        """
        try:
            children = sorted(start.iterdir(), key=lambda p: p.name)
        except PermissionError:
            return
        for child in children:
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name not in PRUNED:
                    yield from self.walk_files(child)
            elif child.is_file():
                yield child

    def backup(self, resolved):
        """Snapshot a file before it is modified. Returns None if it is new.

        The backup directory is created here rather than in __init__: cici gets
        launched in directories it never writes to, and the original made a
        .backups/ in every one of them on startup.
        """
        if not resolved.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time():.6f}"
        target = self.backup_dir / f"{resolved.name}.{stamp}"
        shutil.copy2(resolved, target)
        return target

    def restore(self, resolved):
        """Put back the most recent snapshot of a file and consume it.

        Consuming it makes repeated undo_edit walk back through the history one
        edit at a time, instead of restoring the same snapshot forever.
        """
        prefix = resolved.name + "."
        backups = sorted(
            (p for p in self.backup_dir.glob(f"{prefix}*") if p.is_file()),
            key=lambda p: p.name,
        )
        if not backups:
            raise FileNotFoundError(
                f"no edit to undo for {self.relative(resolved)} — cici only "
                "tracks files it changed in this session"
            )
        latest = backups[-1]
        shutil.copy2(latest, resolved)
        latest.unlink()
        return latest
