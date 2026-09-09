"""ls tool — TODO (roadmap 1). See tools/__init__.py for port targets."""
from .base import Tool


class LsTool(Tool):
    name = "ls"
    description = "List a directory. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise NotImplementedError("ls tool not implemented yet")
