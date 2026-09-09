"""Unified LLM layer (pi-ai equivalent).

Only Anthropic is implemented. The seam is here so a second provider can be
added without the agent loop knowing — do not let provider-specific types leak
into agent.py beyond the content-block shapes the loop already handles.

Pinned to anthropic 1.x. Things 1.x removed that 0.x-era code (cici_101) used:
  - temperature / top_p / top_k are gone from messages.create/.stream (TypeError).
  - assistant prefill returns 400 on every current model. To force structured
    output, use output_config={"format": {...}} — not prefill + stop_sequence.
  - httpx objects handed to the SDK must come from httpx2 (we hand it none).
"""

from pathlib import Path

from anthropic import Anthropic, AnthropicError
from dotenv import load_dotenv

# cici is meant to be run inside OTHER projects, so .env.local is resolved
# against the repo root, never the cwd.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.local"

DEFAULT_MODEL = "claude-opus-5"
# Streaming has no HTTP-timeout pressure, so give a coding agent real room.
DEFAULT_MAX_TOKENS = 64000


class AnthropicProvider:
    def __init__(self, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
        load_dotenv(_ENV_FILE)
        # Don't gate on ANTHROPIC_API_KEY: the SDK also accepts
        # ANTHROPIC_AUTH_TOKEN and an OAuth profile from `ant auth login`.
        # Let it resolve credentials, and translate its error into one line.
        try:
            self.client = Anthropic()
        except (AnthropicError, TypeError) as e:
            raise RuntimeError(
                f"no Anthropic credentials found ({e}). "
                f"Put ANTHROPIC_API_KEY in {_ENV_FILE}, export it, or run `ant auth login`."
            ) from e
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, messages, system=None, tools=None, stop_sequences=None):
        """Return the streaming context manager. Caller drives the events."""
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
