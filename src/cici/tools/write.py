"""write tool — TODO (roadmap 1). See tools/__init__.py for port targets."""

from typing import ClassVar

from .base import SyncTool


class WriteTool(SyncTool):
    name = "write"
    description = "Write a file to disk, creating it if absent. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def _run(self, **kwargs):
        raise NotImplementedError("write tool not implemented yet")
