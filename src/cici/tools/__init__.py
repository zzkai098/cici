"""Tool surface.

Five tools, with deliberately non-overlapping jobs — ls finds what exists, read
views a file, write creates or replaces one, edit changes part of one, and bash
does everything else. cici_101's TextEditorTool folded view and create into the
editor because it was the only tool in the notebook; keeping that here would
give the model two ways to do the same thing, which is a reliable way to make it
pick the wrong one.

Ported from cici_101/005_text_editor_tool.ipynb::TextEditorTool:
    _validate_path   -> workspace.py::Workspace.resolve   (prefix bug fixed)
    _backup_file     -> workspace.py::Workspace.backup    (created lazily)
    _restore_backup  -> workspace.py::Workspace.restore   (consumes the snapshot)
    _count_matches   -> edit.py::EditTool._str_replace    (uniqueness check)
"""

from .base import Registry, SyncTool, Tool  # noqa: F401
from .workspace import Workspace


def default_registry(workspace=None):
    """Build the standard tool surface.

    They all share one Workspace, so they agree on the root they are confined
    to and on where undo snapshots live.

    Registration order is the order the model sees the tools in. ls first, then
    read, follows the order a task actually goes in: find the file, look at it,
    then change it.
    """
    workspace = workspace or Workspace()
    reg = Registry()

    from .bash import BashTool
    from .edit import EditTool
    from .ls import LsTool
    from .read import ReadTool
    from .write import WriteTool

    reg.register(LsTool(workspace))
    reg.register(ReadTool(workspace))
    reg.register(WriteTool(workspace))
    reg.register(EditTool(workspace))
    reg.register(BashTool(workspace))
    return reg
