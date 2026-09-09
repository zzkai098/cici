"""grep tool — TODO (roadmap 1). See tools/__init__.py for port targets."""

from typing import ClassVar

from .base import Tool


class GrepTool(Tool):
    name = "grep"
    description = "Search file contents by regex. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise NotImplementedError("grep tool not implemented yet")
