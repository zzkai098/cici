"""read tool — TODO (roadmap 1). See tools/__init__.py for port targets."""
from .base import Tool


class ReadTool(Tool):
    name = "read"
    description = "Read a file from disk. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise NotImplementedError("read tool not implemented yet")
