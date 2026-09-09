"""bash tool — TODO (roadmap 1). See tools/__init__.py for port targets."""
from .base import Tool


class BashTool(Tool):
    name = "bash"
    description = "Run a shell command with a timeout and truncated output. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise NotImplementedError("bash tool not implemented yet")
