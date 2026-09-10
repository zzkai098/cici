"""Tool surface.

Roadmap 1: get read / write / edit / bash working. Until all four land this is
not a coding agent — cici_101 had the file primitives sitting in a notebook,
never wired into the CLI. That gap is the whole point of this rewrite.

Port targets:
    edit.py  <- cici_101/005_text_editor_tool.ipynb (TextEditorTool)
                view / create / str_replace / insert / undo_edit,
                _validate_path, _backup_file/_restore_backup, _count_matches
"""

from .base import Registry, SyncTool, Tool  # noqa: F401


def default_registry():
    """Build the standard tool surface. Fill in as tools land."""
    reg = Registry()
    # TODO roadmap 1:
    # from .read import ReadTool; reg.register(ReadTool())
    # from .write import WriteTool; reg.register(WriteTool())
    # from .edit import EditTool; reg.register(EditTool())
    # from .bash import BashTool; reg.register(BashTool())
    return reg
