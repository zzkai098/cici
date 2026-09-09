"""edit tool — TODO (roadmap 1). See tools/__init__.py for port targets."""
from .base import Tool


class EditTool(Tool):
    name = "edit"
    description = "Edit a file: view / create / str_replace / insert / undo_edit. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise NotImplementedError("edit tool not implemented yet")
