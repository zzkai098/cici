"""Tool interface and registry.

Schemas are hand-written on purpose — per-parameter descriptions are where tool
reliability actually comes from, and an interviewer can ask about them. Do not
swap this for auto-generation from type hints.

The execution contract is async. Tools whose work is plain blocking local I/O
subclass SyncTool and implement _run; to_thread keeps them off the event loop.
Tools with genuinely async work (bash, via asyncio.create_subprocess_exec)
implement run directly.
"""

import asyncio
import json
from typing import ClassVar


class Tool:
    """Subclass, set name/description/input_schema, implement run().

    For plain blocking local I/O, subclass SyncTool instead.
    """

    name = ""
    description = ""
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}
    # Parallel seam. Read-only tools are safe to gather; anything that mutates
    # the filesystem or shells out is not. run_all is strictly sequential today
    # and ignores this flag — see the TODO there for the intended batching.
    parallel_safe = False

    async def run(self, **kwargs):
        raise NotImplementedError

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class SyncTool(Tool):
    """Base for tools whose work is plain blocking local I/O.

    Subclasses implement _run synchronously and never think about the loop;
    to_thread keeps a slow stat() or a big read from stalling everything else.
    """

    def _run(self, **kwargs):
        raise NotImplementedError

    async def run(self, **kwargs):
        return await asyncio.to_thread(self._run, **kwargs)


class Registry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool
        return tool

    def schemas(self, extra=None):
        out = [t.schema() for t in self._tools.values()]
        if extra:
            out.extend(extra)
        return out

    async def dispatch(self, name, tool_input):
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return await tool.run(**(tool_input or {}))

    async def run_all(self, message):
        """Execute every tool_use block in a message.

        HARD RULE: a failing tool returns an is_error tool_result to the model.
        It must never crash the loop — the model gets a chance to correct itself.

        The except clause must stay `Exception`, never `BaseException`:
        asyncio.CancelledError derives from BaseException, so cancelling the
        agent propagates instead of being swallowed into a fake tool_result.
        A bash timeout raises TimeoutError, which IS an Exception, so it is
        correctly contained and reported back to the model. That split is the
        whole point.

        TODO (parallel tools): scan blocks in order, gather each run of
        consecutive parallel_safe=True tools (read/grep/ls) with asyncio.gather,
        run each parallel_safe=False tool (write/edit/bash) alone, and refill
        results by original block index so the order the model sees is stable.
        """
        blocks = []
        for req in [b for b in message.content if b.type == "tool_use"]:
            try:
                output = await self.dispatch(req.name, req.input)
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": req.id,
                        "content": output
                        if isinstance(output, str)
                        else json.dumps(output),
                        "is_error": False,
                    }
                )
            except Exception as e:  # noqa: BLE001 - containment is the point
                # Some exceptions stringify to nothing (TimeoutError is the one
                # that matters here — it is what a bash timeout raises). Naming
                # the class keeps the model from receiving a bare "Error: ".
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": req.id,
                        "content": f"Error: {str(e) or type(e).__name__}",
                        "is_error": True,
                    }
                )
        return blocks
