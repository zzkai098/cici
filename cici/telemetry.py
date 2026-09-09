"""Per-turn telemetry.

When an agent misbehaves, the conversation record is the only evidence you have.
Every turn gets a line.

TODO (roadmap 2): once prompt caching lands, add the cache hit rate —
    usage.cache_read_input_tokens / (cache_read + input)
That is what turns "context management" from a claim into a defensible number.
"""

import time


class Turn:
    __slots__ = (
        "cache_read_tokens",
        "index",
        "input_tokens",
        "latency_s",
        "n_tools",
        "output_tokens",
        "stop_reason",
    )

    def __init__(self, index):
        self.index = index
        self.stop_reason = None
        self.n_tools = 0
        self.latency_s = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    def line(self):
        parts = [
            f"[turn {self.index}]",
            f"stop={self.stop_reason}",
            f"tools={self.n_tools}",
            f"{self.latency_s:.1f}s",
        ]
        if self.input_tokens or self.output_tokens:
            parts.append(f"in={self.input_tokens} out={self.output_tokens}")
        if self.cache_read_tokens:
            denom = self.cache_read_tokens + self.input_tokens
            if denom:
                parts.append(f"cache={self.cache_read_tokens / float(denom):.0%}")
        return " ".join(parts)


class Recorder:
    def __init__(self):
        self.turns = []
        self._t0 = None

    def begin(self):
        turn = Turn(len(self.turns) + 1)
        self._t0 = time.time()
        self.turns.append(turn)
        return turn

    def end(self, turn, response):
        turn.latency_s = time.time() - (self._t0 or time.time())
        turn.stop_reason = getattr(response, "stop_reason", None)
        turn.n_tools = sum(1 for b in response.content if b.type == "tool_use")
        usage = getattr(response, "usage", None)
        if usage is not None:
            turn.input_tokens = getattr(usage, "input_tokens", 0) or 0
            turn.output_tokens = getattr(usage, "output_tokens", 0) or 0
            turn.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        return turn

    def total_tokens(self):
        return sum(t.input_tokens + t.output_tokens for t in self.turns)
