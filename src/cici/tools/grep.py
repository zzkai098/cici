"""grep tool — search file contents by regular expression.

bash can already run grep or rg, so this only earns its place by being better
at it: no confirmation prompt on a read-only action, no dependency on which
search binary happens to be installed, noise directories skipped without the
model having to remember, and output capped so one broad pattern cannot eat the
context window.

Pure Python re rather than shelling out to rg. cici runs inside other people's
projects and cannot assume rg is installed; a tool that works everywhere beats
one that is faster where it happens to exist. Pruning is what keeps that
affordable.
"""

import re
from fnmatch import fnmatch
from typing import ClassVar

from .base import SyncTool
from .workspace import Workspace

MAX_MATCHES = 200
MAX_FILES = 200
MAX_LINE_CHARS = 500
# Past this a file is data, not source. Reading it would cost more than the
# match could possibly be worth.
MAX_FILE_BYTES = 2_000_000


class GrepTool(SyncTool):
    name = "grep"
    description = (
        "Search the contents of files in the workspace with a regular "
        "expression. Use it to find where something is defined or used before "
        "reading or editing. Dependency and build directories (.git, .venv, "
        "node_modules and similar) are skipped, as are binary files. To list "
        "file and directory NAMES rather than search inside files, use ls."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Python regular expression to search for, e.g. "
                    "'def run\\\\(' or 'TODO|FIXME'. Matched against each line "
                    "independently. Prefix with '(?i)' to make it "
                    "case-insensitive."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "File or directory to search, relative to the workspace "
                    "root. A directory is searched recursively. Defaults to the "
                    "whole workspace."
                ),
            },
            "glob": {
                "type": "string",
                "description": (
                    "Only search files whose name matches this shell pattern, "
                    "e.g. '*.py' or 'test_*.py'. Include a '/' to match against "
                    "the whole relative path instead, e.g. 'src/*/*.py'. Use "
                    "this first when a pattern is common — it is much cheaper "
                    "than filtering the results afterwards."
                ),
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files"],
                "description": (
                    "content, the default, returns each matching line with its "
                    "file and line number — use it when you need to see the "
                    "code. files returns just the file paths and how many times "
                    "each matched — use it when a pattern is widespread and you "
                    "only want to know where to look."
                ),
            },
        },
        "required": ["pattern"],
    }
    parallel_safe = True

    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()

    def _run(self, pattern, path=None, glob=None, output_mode="content"):
        if output_mode not in ("content", "files"):
            raise ValueError(
                f"unknown output_mode {output_mode!r} — expected 'content' or 'files'"
            )
        try:
            regex = re.compile(pattern)
        except re.error as e:
            # Raised back to the model, which can rewrite the pattern; a bad
            # regex is a normal thing for it to get wrong once.
            raise ValueError(f"invalid regular expression {pattern!r}: {e}") from e

        root = self.workspace.resolve(path or ".")
        if not root.exists():
            raise FileNotFoundError(
                f"no such file or directory: {self.workspace.relative(root)}"
            )

        files = [root] if root.is_file() else self.workspace.walk_files(root)
        hits, files_seen, matched_files = [], 0, 0

        for f in files:
            shown = self.workspace.relative(f)
            if glob and not self._matches_glob(f.name, shown, glob):
                continue
            lines = self._read_lines(f)
            if lines is None:
                continue
            files_seen += 1

            found = [(n, ln) for n, ln in enumerate(lines, 1) if regex.search(ln)]
            if not found:
                continue
            matched_files += 1

            if output_mode == "files":
                plural = "" if len(found) == 1 else "es"
                hits.append(f"{shown} ({len(found)} match{plural})")
            else:
                hits.extend(f"{shown}:{n}: {self._clip(ln)}" for n, ln in found)

        return self._render(hits, pattern, output_mode, files_seen, matched_files)

    @staticmethod
    def _matches_glob(name, relative_path, glob):
        """A glob with a separator filters on the path; otherwise on the name."""
        return fnmatch(relative_path, glob) if "/" in glob else fnmatch(name, glob)

    @staticmethod
    def _read_lines(f):
        """Return the file's lines, or None if it is not searchable text."""
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                return None
            return f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            # Binary, or vanished/unreadable mid-walk. Skipping is right: the
            # model asked to search source, not to be told about every blob.
            return None

    @staticmethod
    def _clip(line):
        line = line.rstrip()
        if len(line) <= MAX_LINE_CHARS:
            return line
        return line[:MAX_LINE_CHARS] + f"… [line truncated, {len(line)} chars]"

    def _render(self, hits, pattern, output_mode, files_seen, matched_files):
        if not hits:
            return (
                f"no matches for {pattern!r} in {files_seen} files searched. "
                "Widen the path, drop the glob, or try a looser pattern."
            )

        if output_mode == "files":
            cap = MAX_FILES
            header = f"{matched_files} files matched ({files_seen} files searched)"
        else:
            cap = MAX_MATCHES
            header = (
                f"{len(hits)} matches in {matched_files} files "
                f"({files_seen} files searched)"
            )
        out = [header, *hits[:cap]]
        if len(hits) > cap:
            out.append(
                f"[showing {cap} of {len(hits)}; narrow the pattern, add a glob, "
                f"or use output_mode='files' to see where to look]"
            )
        return "\n".join(out)
