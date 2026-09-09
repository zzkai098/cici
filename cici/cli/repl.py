"""Interactive CLI (pi-coding-agent equivalent).

TODO (roadmap 3/5): port the prompt_toolkit shell from
cici_101/cli_project/core/cli.py — /command completion with descriptions,
@resource mentions, custom key bindings that pop completion on / or @.
For now this is a plain input() loop so the spine runs end to end.
"""
import os

from .. import tui
from ..agent import Agent, SYSTEM_PROMPT
from ..llm import AnthropicProvider
from ..session import Session
from ..tools import default_registry


def run():
    provider = AnthropicProvider()
    tui.banner(provider.model)

    session = Session(system_prompt=SYSTEM_PROMPT)
    registry = default_registry()
    agent = Agent(provider, registry, session)

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input.strip():
            continue
        if user_input.strip() in ("/exit", "/quit"):
            print("Bye!")
            break

        session.add_user(user_input)
        agent.run()
        if os.getenv("CICI_DEBUG"):
            print("\n" + "-" * 60)
            print(session.messages)
