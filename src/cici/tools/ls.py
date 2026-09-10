"""ls tool — TODO (roadmap 1). See tools/__init__.py for port targets."""

from typing import ClassVar

from .base import SyncTool


class LsTool(SyncTool):
    name = "ls"
    description = "List a directory. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}
    # Read-only: safe to gather with other read-only tools once
    # Registry.run_all learns to batch them.
    parallel_safe = True

    def _run(self, **kwargs):
        raise NotImplementedError("ls tool not implemented yet")
