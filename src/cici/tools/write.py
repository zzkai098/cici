"""write tool — create a file, or replace one wholesale.

Overwriting is allowed but always snapshots first, so edit's undo_edit can walk
back a clobbered file the same way it walks back a replacement. cici_101's
create() refused to touch an existing file at all; that is safer in isolation
but pushes the model into faking a rewrite as a chain of str_replace calls.
"""

from typing import ClassVar

from .base import SyncTool
from .workspace import Workspace


class WriteTool(SyncTool):
    name = "write"
    description = (
        "Write text to a file in the workspace, creating parent directories as "
        "needed. Creates the file if it does not exist, and replaces its entire "
        "contents if it does. To change part of an existing file, use edit "
        "instead — it is far cheaper than restating the whole file here."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file, relative to the workspace root. Parent "
                    "directories are created automatically. Paths outside the "
                    "workspace are rejected."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "The complete new contents of the file. This replaces "
                    "everything already there — it is not appended. Pass an "
                    "empty string to truncate the file."
                ),
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()

    def _run(self, path, content):
        resolved = self.workspace.resolve(path)
        shown = self.workspace.relative(resolved)

        if resolved.is_dir():
            raise IsADirectoryError(f"{shown} is a directory, not a file")
        if content is None:
            raise ValueError('content is required (pass "" to truncate the file)')

        existed = resolved.exists()
        # Snapshot before clobbering, so undo_edit can bring it back.
        self.workspace.backup(resolved)

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        n_lines = content.count("\n") + (
            1 if content and not content.endswith("\n") else 0
        )
        verb = "overwrote" if existed else "created"
        return f"{verb} {shown} ({n_lines} lines, {len(content)} chars)"
