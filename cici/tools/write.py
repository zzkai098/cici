"""write tool — TODO (roadmap 1). See tools/__init__.py for port targets."""
from .base import Tool


class WriteTool(Tool):
    name = "write"
    description = "Write a file to disk, creating it if absent. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise NotImplementedError("write tool not implemented yet")
