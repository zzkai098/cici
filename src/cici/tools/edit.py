"""edit tool — targeted changes to an existing file.

Ported from cici_101/005_text_editor_tool.ipynb::TextEditorTool, minus its view
and create commands: read and write own those now, and two tools that can do
the same job is the classic reason a model picks the wrong one.

What survives the port is the part that earns its keep — the uniqueness check
on str_replace. Replacing a string that appears twice silently corrupts the
file in a way the model cannot see, so an ambiguous match is an error, not a
best guess.
"""

from typing import ClassVar

from .base import SyncTool
from .workspace import Workspace


class EditTool(SyncTool):
    name = "edit"
    description = (
        "Change part of an existing file: replace an exact string, insert lines "
        "at a position, or undo the last change cici made to it. Read the file "
        "first — old_str has to match the file byte for byte."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["str_replace", "insert", "undo_edit"],
                "description": (
                    "str_replace: swap one exact, unique block of text for "
                    "another. insert: add new lines after a given line number. "
                    "undo_edit: restore the file to before cici's last change "
                    "to it."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Path to the file, relative to the workspace root. The file "
                    "must already exist — use write to create one."
                ),
            },
            "old_str": {
                "type": "string",
                "description": (
                    "str_replace only. The exact text to replace, including "
                    "indentation and line breaks. It must appear EXACTLY ONCE "
                    "in the file: if it appears more than once the call is "
                    "rejected, so include enough surrounding lines to make the "
                    "match unique."
                ),
            },
            "new_str": {
                "type": "string",
                "description": (
                    "str_replace: the text to put in place of old_str; pass an "
                    "empty string to delete it. insert: the text to insert, "
                    "without a trailing newline."
                ),
            },
            "insert_line": {
                "type": "integer",
                "description": (
                    "insert only. The 1-based line number to insert AFTER, as "
                    "numbered by the read tool. Use 0 to insert at the very top "
                    "of the file."
                ),
            },
        },
        "required": ["command", "path"],
    }

    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()

    def _run(self, command, path, old_str=None, new_str=None, insert_line=None):
        resolved = self.workspace.resolve(path)
        shown = self.workspace.relative(resolved)

        if command == "undo_edit":
            self.workspace.restore(resolved)
            return f"restored {shown} to before cici's last change"

        if resolved.is_dir():
            raise IsADirectoryError(f"{shown} is a directory, not a file")
        if not resolved.exists():
            raise FileNotFoundError(f"no such file: {shown} — use write to create it")

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{shown} is not UTF-8 text and cannot be edited") from e

        if command == "str_replace":
            return self._str_replace(resolved, shown, content, old_str, new_str)
        if command == "insert":
            return self._insert(resolved, shown, content, insert_line, new_str)
        raise ValueError(
            f"unknown command {command!r} — expected str_replace, insert or undo_edit"
        )

    def _str_replace(self, resolved, shown, content, old_str, new_str):
        if not old_str:
            raise ValueError("old_str is required for str_replace")
        if new_str is None:
            raise ValueError('new_str is required (pass "" to delete old_str)')

        matches = content.count(old_str)
        if matches == 0:
            raise ValueError(
                f"old_str not found in {shown}. Read the file again — it must "
                "match byte for byte, including indentation."
            )
        if matches > 1:
            raise ValueError(
                f"old_str appears {matches} times in {shown}. Include more "
                "surrounding lines so exactly one location matches."
            )

        self.workspace.backup(resolved)
        resolved.write_text(content.replace(old_str, new_str), encoding="utf-8")
        line = content[: content.index(old_str)].count("\n") + 1
        return f"replaced 1 occurrence in {shown} at line {line}"

    def _insert(self, resolved, shown, content, insert_line, new_str):
        if insert_line is None:
            raise ValueError("insert_line is required for insert")
        if new_str is None:
            raise ValueError("new_str is required for insert")

        lines = content.splitlines(keepends=True)
        if not 0 <= insert_line <= len(lines):
            raise IndexError(
                f"insert_line {insert_line} is out of range — {shown} has "
                f"{len(lines)} lines (use 0 to insert at the top)"
            )

        # If the file does not end in a newline, inserting at the end would
        # otherwise glue the new text onto the last line.
        if lines and insert_line == len(lines) and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        self.workspace.backup(resolved)
        lines.insert(insert_line, new_str + "\n")
        resolved.write_text("".join(lines), encoding="utf-8")
        where = "at the top" if insert_line == 0 else f"after line {insert_line}"
        return f"inserted 1 line in {shown} {where}"
