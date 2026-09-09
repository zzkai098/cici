"""Unified LLM layer (pi-ai equivalent).

Only Anthropic is implemented. The seam is here so a second provider can be
added without the agent loop knowing — do not let provider-specific types leak
into agent.py beyond the content-block shapes the loop already handles.
"""
import os

from dotenv import load_dotenv
from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(object):
    def __init__(self, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
        load_dotenv(".env.local")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. cp .env.local.example .env.local and fill it in."
            )
        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, messages, system=None, tools=None, stop_sequences=None):
        """Return the streaming context manager. Caller drives the events."""
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
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
