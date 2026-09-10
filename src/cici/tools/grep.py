"""grep tool — TODO (roadmap 1). See tools/__init__.py for port targets."""

from typing import ClassVar

from .base import SyncTool


class GrepTool(SyncTool):
    name = "grep"
    description = "Search file contents by regex. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}
    # Read-only: safe to gather with other read-only tools once
    # Registry.run_all learns to batch them.
    parallel_safe = True

    def _run(self, **kwargs):
        raise NotImplementedError("grep tool not implemented yet")
