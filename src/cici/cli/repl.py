"""Interactive CLI (pi-coding-agent equivalent).

TODO (roadmap 3/5): port the prompt_toolkit shell from
cici_101/cli_project/core/cli.py — /command completion with descriptions,
@resource mentions, custom key bindings that pop completion on / or @.
For now this is a plain input() loop so the spine runs end to end.
"""

import asyncio
import os

from .. import tui
from ..agent import SYSTEM_PROMPT, Agent
from ..llm import Claude
from ..session import Session
from ..tools import default_registry


def run():
    """Own the event loop explicitly, and read the prompt outside of it.

    This is asyncio.Runner rather than asyncio.run(...) for one measured reason.
    asyncio.run installs a SIGINT handler that turns the first Ctrl-C into
    main_task.cancel() instead of raising KeyboardInterrupt. A task parked in a
    blocking input() can never receive that cancellation, so the process hangs
    until a second Ctrl-C. Runner installs the same handler only for the
    duration of each run() call, so at the prompt Ctrl-C is the plain default
    handler again — identical to the behaviour before the harness went async.
    Measured on a pty: sync input() exits, asyncio.run + input() hangs,
    Runner + input() exits.

    Wrapping input() in asyncio.to_thread is not the fix either: Ctrl-C is
    delivered to the main thread, so a worker thread's input() just retries its
    read (PEP 475) and never returns, and shutdown_default_executor() then
    blocks forever trying to join it.

    Once prompt_toolkit lands (roadmap 3), `await session.prompt_async("> ")`
    takes over stdin from inside the loop and handles Ctrl-C itself; at that
    point this function becomes `async def` and the Runner goes away.
    """
    provider = Claude()
    tui.banner(provider.model)

    session = Session(system_prompt=SYSTEM_PROMPT)
    registry = default_registry()
    agent = Agent(provider, registry, session)

    with asyncio.Runner() as runner:
        try:
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

                # The return value is already on screen — Agent streams as it
                # goes — so printing it here would double-print. It exists for
                # programmatic callers (evals, sub-agents).
                runner.run(agent.run(user_input))
                if os.getenv("CICI_DEBUG"):
                    print("\n" + "-" * 60)
                    print(session.messages)
        finally:
            # A Ctrl-C mid-turn surfaces here as KeyboardInterrupt (Runner
            # re-raises it after the cancellation lands); close the pool anyway.
            runner.run(provider.aclose())
