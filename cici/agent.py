"""Agent loop (pi-agent-core equivalent).

Ported and restructured from cici_101/webagent_stream.py::run_conversation_stream.

Two invariants this loop exists to hold:
  1. Error containment — a tool exception becomes an is_error tool_result, never
     a crash. Handled in tools/base.py::Registry.run_all.
  2. Observability — every turn emits a telemetry line.

Known streaming gotchas (already paid for once in cici_101):
  - server_tool_use has input == {} at content_block_start; it only lands at
    content_block_stop.
  - web_search_20260209 also grants code_execution, so dispatch on block.name
    rather than assuming search.
"""

from . import tui
from .telemetry import Recorder

SYSTEM_PROMPT = """You are cici, a coding agent.

You run in a terminal. Write plain text only — markdown renders as literal
characters here, so no bold, no headers, no bullet syntax. Keep answers to a
few sentences unless the user asks for more.
"""


class Agent:
    def __init__(self, provider, registry, session, max_turns=20, server_tools=None):
        self.provider = provider
        self.registry = registry
        self.session = session
        self.max_turns = max_turns
        self.server_tools = server_tools or []
        self.telemetry = Recorder()

    def _tools(self):
        return self.registry.schemas(extra=self.server_tools)

    def run(self):
        """Drive turns until the model stops asking for tools."""
        response = None
        for _ in range(self.max_turns):
            turn = self.telemetry.begin()
            spinner = tui.Tentacles()
            spinner.start("thinking…")

            with self.provider.stream(
                self.session.messages,
                system=self.session.system_prompt or SYSTEM_PROMPT,
                tools=self._tools(),
            ) as stream:
                try:
                    for chunk in stream:
                        self._on_chunk(chunk, spinner)
                finally:
                    spinner.stop()
                response = stream.get_final_message()

            self.telemetry.end(turn, response)
            tui.rule()
            print(turn.line())

            self.session.add_assistant(response)
            if response.stop_reason != "tool_use":
                break
            self.session.add_user(self.registry.run_all(response))
        else:
            print(f"\n[stopped: reached {self.max_turns} turns]")
        return response

    def _on_chunk(self, chunk, spinner):
        if chunk.type == "text":
            spinner.stop()
            print(chunk.text, end="", flush=True)
            return

        if chunk.type == "thinking":
            spinner.start("thinking…")
            return

        if chunk.type == "content_block_start":
            block = chunk.content_block
            if block.type == "thinking":
                spinner.start("thinking…")
            elif block.type == "tool_use":
                spinner.stop()
                print(f"\n[tool] {block.name}", end="", flush=True)
            elif block.type == "server_tool_use":
                # input is still {} here — it only lands at block stop
                spinner.start("working…")
            return

        if chunk.type == "content_block_stop":
            block = getattr(chunk, "content_block", None)
            if block is None:
                return
            if block.type == "tool_use":
                args = tui.describe_tool_input(block.input) if block.input else ""
                print(f"({args})", flush=True)
            elif block.type == "server_tool_use":
                spinner.stop()
                label = block.name
                print(f"\n[{label}] {tui.describe_tool_input(block.input)}", flush=True)
                spinner.start("working…")
