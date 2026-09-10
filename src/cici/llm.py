"""Unified LLM layer (pi-ai equivalent).

Only Anthropic is implemented. The seam is here so a second provider can be
added without the agent loop knowing — do not let provider-specific types leak
into agent.py beyond the content-block shapes the loop already handles.

Async because the harness above it is async. `Claude.stream()` is deliberately
NOT a coroutine: the SDK's `messages.stream()` is a plain method returning an
async context manager, so the seam keeps the same signature it had when sync.

Pinned to anthropic 1.x. Things 1.x removed that 0.x-era code (cici_101) used:
  - temperature / top_p / top_k are gone from messages.create/.stream (TypeError).
  - assistant prefill returns 400 on every current model. To force structured
    output, use output_config={"format": {...}} — not prefill + stop_sequence.
  - httpx objects handed to the SDK must come from httpx2 (we hand it none).
"""

from pathlib import Path

from anthropic import AnthropicError, AsyncAnthropic
from dotenv import load_dotenv

# cici is meant to be run inside OTHER projects, so .env.local is resolved
# against the repo root, never the cwd.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.local"

DEFAULT_MODEL = "claude-opus-5"
# Streaming has no HTTP-timeout pressure, so give a coding agent real room.
DEFAULT_MAX_TOKENS = 64000


class Claude:
    """Anthropic implementation of the LLM seam.

    Named after cici_101/cli_project/core/claude.py::Claude, which plays the
    same role there. The message-history helpers that class carried live on
    session.py::Session here — cici_101 hung them off Claude only because it
    had no conversation object.
    """

    def __init__(self, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
        load_dotenv(_ENV_FILE)
        # Don't gate on ANTHROPIC_API_KEY: the SDK also accepts
        # ANTHROPIC_AUTH_TOKEN and an OAuth profile from `ant auth login`.
        # Let it resolve credentials, and translate its error into one line.
        try:
            # Construction is still synchronous; only the calls are awaited.
            self.client = AsyncAnthropic()
        except (AnthropicError, TypeError) as e:
            raise RuntimeError(
                f"no Anthropic credentials found ({e}). "
                f"Put ANTHROPIC_API_KEY in {_ENV_FILE}, export it, or run `ant auth login`."
            ) from e
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, messages, system=None, tools=None, stop_sequences=None):
        """Return the async streaming context manager. Caller drives the events.

        This is NOT a coroutine — do not await this call. The SDK returns an
        AsyncMessageStreamManager, so the caller uses `async with`, iterates
        with `async for`, and finishes with `await stream.get_final_message()`.
        """
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            # Adaptive thinking is the only on-mode on current models; budget_tokens
            # is a 400. display defaults to "omitted", which reads as a dead pause —
            # ask for the summary so the CLI has something to show.
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if system:
            # TODO (roadmap 2): cache_control breakpoint on the system prompt.
            params["system"] = system
        if tools:
            # TODO (roadmap 2): cache_control breakpoint on the last tool schema.
            params["tools"] = tools
        if stop_sequences:
            params["stop_sequences"] = stop_sequences
        return self.client.messages.stream(**params)

    @staticmethod
    def text_from_message(message):
        """Concatenate the text blocks of a Message.

        Ported from cici_101/cli_project/core/claude.py::text_from_message.
        """
        return "\n".join(b.text for b in message.content if b.type == "text")

    async def aclose(self):
        """Release the HTTP pool.

        The sync client gets away without this because GC closes it; the async
        client does not, and leaks a warning on exit if never closed.
        """
        await self.client.close()
