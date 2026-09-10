"""bash tool — TODO (roadmap 1). See tools/__init__.py for port targets."""

from typing import ClassVar

from .base import Tool


class BashTool(Tool):
    """Stays on Tool, not SyncTool: the work here is genuinely async.

    Implement with asyncio.create_subprocess_exec inside `async with
    asyncio.timeout(...)`, so a hung command is actually killable instead of
    blocking the process the way subprocess.run(timeout=) does. Truncate stdout
    and stderr before returning, and gate the command behind a confirmation.
    The TimeoutError that escapes on expiry is an Exception, so Registry.run_all
    contains it as an is_error tool_result — which is what we want the model to
    see.
    """

    name = "bash"
    description = "Run a shell command with a timeout and truncated output. TODO: hand-write the full schema, per-parameter descriptions included."
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs):
        raise NotImplementedError("bash tool not implemented yet")
