"""Tool interface and registry.

Schemas are hand-written on purpose — per-parameter descriptions are where tool
reliability actually comes from, and an interviewer can ask about them. Do not
swap this for auto-generation from type hints.
"""

import json
from typing import ClassVar


class Tool:
    """Subclass, set name/description/input_schema, implement run()."""

    name = ""
    description = ""
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise NotImplementedError

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


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

    def dispatch(self, name, tool_input):
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool.run(**(tool_input or {}))

    def run_all(self, message):
        """Execute every tool_use block in a message.

        HARD RULE: a failing tool returns an is_error tool_result to the model.
        It must never crash the loop — the model gets a chance to correct itself.
        """
        blocks = []
        for req in [b for b in message.content if b.type == "tool_use"]:
            try:
                output = self.dispatch(req.name, req.input)
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
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": req.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    }
                )
        return blocks
