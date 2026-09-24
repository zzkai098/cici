"""Unified LLM layer (pi-ai equivalent).

Three implementations of one seam:

  Claude            — Anthropic Messages API.
  OpenAICompatible  — any OpenAI-compatible /v1/chat/completions endpoint.
  StubLLM           — scripted, no network; for wiring tests.

The seam was written for Anthropic with the note that "a second provider can be
added without the agent loop knowing". OpenAICompatible is that second provider,
and it cashes the note in: agent.py, session.py and tools/ are untouched. The
price of that is paid here, in _ChatStream, which normalises OpenAI's chunk
stream into the Anthropic content-block shapes the loop already handles.

Async because the harness above it is async. `.stream()` is deliberately NOT a
coroutine: the SDK's `messages.stream()` is a plain method returning an async
context manager, so the seam keeps the same signature it had when sync. Every
provider here matches that — `async with`, `async for`, then
`await stream.get_final_message()`.

Pinned to anthropic 1.x. Things 1.x removed that 0.x-era code (cici_101) used:
  - temperature / top_p / top_k are gone from messages.create/.stream (TypeError).
  - assistant prefill returns 400 on every current model. To force structured
    output, use output_config={"format": {...}} — not prefill + stop_sequence.
  - httpx objects handed to the SDK must come from httpx2 (we hand it none).
"""

import json
import os
from pathlib import Path

# httpx2 rather than httpx: it is what anthropic 1.x pins, so using it directly
# adds no dependency and cannot drift into two HTTP stacks in one process.
import httpx2
from anthropic import AnthropicError, AsyncAnthropic
from anthropic.types import Message
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
    same role there. Its add_user_message/add_assistant_message are split in
    two here: unwrapping a response into content stays on this class, because
    knowing what a Message looks like is provider knowledge; appending to the
    history belongs to session.py::Session, which cici_101 had no equivalent of.
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
    def content_from_message(message):
        """Unwrap a response into the content that goes back into history.

        Stores the blocks verbatim rather than flattening to text. That is
        load-bearing, not stylistic: with thinking enabled the content carries
        ThinkingBlocks, and they must be echoed back unchanged on the same
        model or the next request after a tool_use turn is rejected.

        isinstance, not getattr(message, "content", message): the duck-typed
        version unwraps anything with a .content attribute — an httpx2.Response
        has one — and would quietly put the wrong bytes into the conversation
        instead of failing.
        """
        return message.content if isinstance(message, Message) else message

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


# --- Anthropic-shaped values the other providers hand back --------------------
#
# agent.py, telemetry.py and tools/base.py all read these by attribute
# (block.type, block.name, block.input, block.id, response.stop_reason,
# response.usage.input_tokens). Hand-rolled instead of reusing anthropic.types:
# those are pydantic models with required fields we have no honest value for,
# and constructing one to fake a non-Anthropic response would put a lie behind
# the seam that isinstance checks elsewhere would then believe.


class TextBlock:
    type = "text"
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    type = "tool_use"
    __slots__ = ("id", "input", "name")

    def __init__(self, id, name, input):  # mirrors the SDK field names
        self.id = id
        self.name = name
        self.input = input


class Usage:
    __slots__ = ("cache_read_input_tokens", "input_tokens", "output_tokens")

    def __init__(self, input_tokens=0, output_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class ShimMessage:
    """What get_final_message() returns for the non-Anthropic providers."""

    __slots__ = ("content", "model", "stop_reason", "usage")

    def __init__(self, content, stop_reason, usage=None, model=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or Usage()
        self.model = model


class StreamEvent:
    """One event in the shape agent.py::_on_chunk switches on.

    Every attribute exists on every instance so the handler's mix of direct
    access (chunk.content_block) and getattr-with-default never has to care
    which kind of event it got.
    """

    __slots__ = ("content_block", "text", "thinking", "type")

    def __init__(self, type, text="", content_block=None, thinking=""):
        self.type = type
        self.text = text
        self.content_block = content_block
        self.thinking = thinking


def _content_from_shim(message):
    """Shared content_from_message for the non-Anthropic providers.

    Same isinstance discipline as Claude's: unwrap what we know is ours, pass
    anything else through untouched rather than duck-typing on `.content`.
    """
    return message.content if isinstance(message, ShimMessage) else message


def _text_from_shim(message):
    return "\n".join(b.text for b in message.content if b.type == "text")


# --- Anthropic history <-> OpenAI chat messages -------------------------------

# Anthropic stop_reason values, keyed by OpenAI finish_reason.
_FINISH_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _block_field(block, name, default=None):
    """Read a content block that may be one of our objects or a plain dict.

    session.py stores whatever content_from_message returned, and
    Registry.run_all appends tool_results as dicts, so both shapes are in the
    history by design.
    """
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def to_openai_tools(tools):
    """Anthropic {name, description, input_schema} -> OpenAI function schema.

    tools/base.py::Tool.schema() stays Anthropic-shaped; translating here keeps
    the six hand-written schemas as the single source of truth.
    """
    out = []
    for tool in tools or []:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": _block_field(tool, "name"),
                    "description": _block_field(tool, "description", "") or "",
                    "parameters": _block_field(tool, "input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def to_openai_messages(messages, system=None):
    """Anthropic-shaped history -> OpenAI chat messages.

    The three shape changes that matter:
      - system moves from a top-level parameter into messages[0].
      - assistant tool_use blocks become a `tool_calls` array, with the input
        re-serialised to the JSON string OpenAI expects.
      - tool_result blocks, which Anthropic carries inside a *user* message,
        become standalone role="tool" messages.

    Ordering is load-bearing on that last one: OpenAI requires each tool
    message to follow the assistant message that requested it, so tool results
    in a mixed user turn are emitted before any text from the same turn.
    """
    out = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        blocks = content or []
        if role == "assistant":
            text = "".join(
                _block_field(b, "text", "") or ""
                for b in blocks
                if _block_field(b, "type") == "text"
            )
            tool_calls = [
                {
                    "id": _block_field(b, "id"),
                    "type": "function",
                    "function": {
                        "name": _block_field(b, "name"),
                        "arguments": _dump_args(_block_field(b, "input")),
                    },
                }
                for b in blocks
                if _block_field(b, "type") == "tool_use"
            ]
            entry = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        # user turn: tool_results first, then whatever text came with them.
        texts = []
        for b in blocks:
            btype = _block_field(b, "type")
            if btype == "tool_result":
                body = _block_field(b, "content", "")
                if not isinstance(body, str):
                    body = json.dumps(body)
                if _block_field(b, "is_error"):
                    # OpenAI's tool message has no is_error flag, and dropping
                    # the distinction is how a model ends up believing a failed
                    # tool succeeded. Say it in the content instead.
                    body = f"Error: {body}" if not body.startswith("Error:") else body
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": _block_field(b, "tool_use_id"),
                        "content": body,
                    }
                )
            elif btype == "text":
                texts.append(_block_field(b, "text", "") or "")
        if texts:
            out.append({"role": "user", "content": "".join(texts)})
    return out


def _dump_args(value):
    if isinstance(value, str):
        return value
    return json.dumps(value or {})


def _parse_args(raw):
    """Turn accumulated argument fragments into a tool input.

    On malformed JSON the raw string is handed through rather than silently
    replaced with {}: Registry.dispatch's `**tool_input` then raises TypeError,
    run_all contains it as an is_error tool_result, and the model gets to try
    again. Substituting {} would instead run the tool with no arguments, which
    is the same failure wearing a success costume.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return parsed if isinstance(parsed, dict) else raw


def _usage_from(raw):
    if not isinstance(raw, dict):
        return Usage()
    prompt = raw.get("prompt_tokens") or 0
    details = raw.get("prompt_tokens_details") or {}
    cached = (details.get("cached_tokens") if isinstance(details, dict) else 0) or 0
    # OpenAI's prompt_tokens INCLUDES cached tokens; Anthropic's input_tokens
    # excludes them, and telemetry.py's cache% divides by (cache_read + input).
    # Subtract so the ratio means the same thing whichever provider produced it.
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=raw.get("completion_tokens") or 0,
        cache_read_input_tokens=cached,
    )


# --- OpenAI-compatible provider ----------------------------------------------

# 4000, not the 4096 the official Agenthon example passes. The two numbers
# disagree (main doc says 4,000, baselines/ says 4096) and card.toml carries no
# max_tokens field, so there is nothing authoritative to settle it. The risk is
# asymmetric: 96 extra tokens of headroom buys almost nothing, while an
# over-limit request still spends one of the 25 admitted requests for the unit.
# Take the conservative number and let the environment raise it if a run ever
# proves it safe.
OPENAI_DEFAULT_MAX_TOKENS = 4000
MAX_TOKENS_ENV = "CICI_MAX_TOKENS"


def _resolve_max_tokens(explicit=None):
    """Explicit argument wins, then CICI_MAX_TOKENS, then the conservative default.

    Same precedence as the credentials on OpenAICompatible. Read at construction
    rather than at import so setting the variable after import still takes.
    """
    if explicit is not None:
        return explicit
    raw = os.getenv(MAX_TOKENS_ENV)
    if not raw:
        return OPENAI_DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError as e:
        raise RuntimeError(f"{MAX_TOKENS_ENV} must be an integer, got {raw!r}") from e
    if value <= 0:
        raise RuntimeError(f"{MAX_TOKENS_ENV} must be positive, got {value}")
    return value


# Generous read timeout: a long tool-writing turn streams for minutes, and
# httpx2's 5s default would kill it mid-sentence.
OPENAI_DEFAULT_TIMEOUT = httpx2.Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0)


class OpenAICompatible:
    """OpenAI-compatible /v1/chat/completions implementation of the LLM seam.

    Credentials come from the environment the way the Agenthon harness injects
    them — MODEL_ENDPOINT / MODEL_TOKEN / MODEL_NAME — with explicit arguments
    winning so the class stays usable outside that harness.

    trust_env is left at its default on purpose: the harness routes every call
    through an auditing proxy advertised in HTTP_PROXY / HTTPS_PROXY, and
    turning it off would take cici off the audited path.
    """

    def __init__(
        self,
        model=None,
        max_tokens=None,
        base_url=None,
        api_key=None,
        timeout=OPENAI_DEFAULT_TIMEOUT,
    ):
        # In a container there is no .env.local; load_dotenv just returns False
        # for a missing path, so this is safe to call unconditionally.
        load_dotenv(_ENV_FILE)

        endpoint = base_url or os.getenv("MODEL_ENDPOINT") or ""
        endpoint = endpoint.rstrip("/")
        if not endpoint:
            raise RuntimeError(
                "no model endpoint found. Set MODEL_ENDPOINT "
                "(e.g. http://model:8443) or pass base_url=."
            )
        # The harness documents the endpoint as a host with the API under /v1,
        # but tolerate someone handing us the /v1 URL directly.
        self.base_url = endpoint if endpoint.endswith("/v1") else endpoint + "/v1"

        self.api_key = api_key or os.getenv("MODEL_TOKEN") or ""
        self.model = model or os.getenv("MODEL_NAME") or ""
        if not self.model:
            raise RuntimeError("no model name found. Set MODEL_NAME or pass model=.")
        self.max_tokens = _resolve_max_tokens(max_tokens)

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.client = httpx2.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

        # Counted at the request layer, not the turn layer: a unit is allowed
        # 25 admitted requests and retries count, so turns are the wrong unit.
        #
        # ⚠️ The increment lives in stream(), one frame above the POST in
        # _ChatStreamManager.__aenter__. That is correct today precisely
        # because there is no retry anywhere below it: one stream() call is one
        # HTTP request, and a 4xx is counted because the increment happens
        # first. Add a retry loop inside __aenter__ and this silently
        # under-counts every retry — each of which the grader admits against
        # the same budget of 25. If retries are ever added, the counter must
        # move down to wrap the actual send, not stay here.
        #
        # TODO (step 3): act on this — inject a "N calls left, write the final
        # deliverable now" note as the budget runs down.
        self.request_count = 0

    def stream(self, messages, system=None, tools=None, stop_sequences=None):
        """Return an async context manager over the chat-completions stream.

        NOT a coroutine, matching Claude.stream: the caller does `async with`,
        `async for`, then `await stream.get_final_message()`.

        No `thinking` parameter — that one is Anthropic-only. Reasoning that an
        OpenAI-compatible server volunteers in `reasoning_content` is surfaced
        as a thinking event by the stream shim instead.
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": to_openai_messages(messages, system=system),
            "stream": True,
            # Usage is omitted from a stream unless asked for. Servers that do
            # not know this option ignore it; the shim tolerates no usage at all.
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        if stop_sequences:
            payload["stop"] = stop_sequences

        self.request_count += 1
        return _ChatStreamManager(self.client, payload)

    content_from_message = staticmethod(_content_from_shim)
    text_from_message = staticmethod(_text_from_shim)

    async def aclose(self):
        await self.client.aclose()


class _ChatStreamManager:
    """Mirror of anthropic's AsyncMessageStreamManager."""

    def __init__(self, client, payload):
        self._client = client
        self._payload = payload
        self._cm = None

    async def __aenter__(self):
        self._cm = self._client.stream("POST", "/chat/completions", json=self._payload)
        response = await self._cm.__aenter__()
        if response.status_code >= 400:
            # The body has not been read yet on a streaming response, and it is
            # where the useful half of the error lives.
            body = (await response.aread()).decode("utf-8", "replace")[:500]
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            raise RuntimeError(
                f"model endpoint returned {response.status_code}: {body.strip()}"
            )
        return _ChatStream(response)

    async def __aexit__(self, *exc_info):
        if self._cm is None:
            return False
        cm, self._cm = self._cm, None
        return await cm.__aexit__(*exc_info)


class _ChatStream:
    """Normalise an OpenAI SSE chunk stream into Anthropic-shaped events.

    Three things OpenAI does differently that this class absorbs:
      - tool calls arrive as fragments keyed by index, with the name in the
        first fragment and the arguments split across the rest, so a content
        block cannot be emitted whole and has to be opened, fed, and closed.
      - there is no get_final_message(), so the message is accumulated here.
      - finish_reason is not stop_reason.
    """

    def __init__(self, response):
        self._response = response
        self._gen = None
        self._text_parts = []
        self._calls = {}  # index -> {"id", "name", "args"}
        self._order = []  # indices, in arrival order
        self._open = None  # index of the tool call whose start we emitted
        self._finish_reason = None
        self._usage = None
        self._model = None

    def __aiter__(self):
        # Cached so that get_final_message()'s drain resumes the same generator
        # rather than restarting the read and hanging on an exhausted body.
        if self._gen is None:
            self._gen = self._events()
        return self._gen

    async def get_final_message(self):
        # Drain anything the caller did not consume — the same courtesy the
        # SDK's stream helper does — so an early `break` still yields a usable
        # message instead of a truncated one.
        async for _ in self:
            pass
        return self._build()

    async def _events(self):
        async for raw_line in self._response.aiter_lines():
            line = raw_line.strip()
            # Blank lines separate SSE events; ":" lines are keep-alive comments.
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for event in self._consume(chunk):
                yield event
        # The last tool call has no successor to close it.
        for event in self._close_open():
            yield event

    def _consume(self, chunk):
        events = []
        self._model = chunk.get("model") or self._model
        if chunk.get("usage"):
            self._usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            # The usage-only chunk that include_usage appends has no choices.
            return events
        choice = choices[0]
        delta = choice.get("delta") or {}

        # Not in the OpenAI spec, but every reasoning model served behind an
        # OpenAI-compatible façade puts its summary in one of these two.
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            events.append(StreamEvent("thinking", thinking=reasoning))

        text = delta.get("content")
        if isinstance(text, list):
            # A few servers echo the multimodal part-list shape back.
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        if text:
            events.extend(self._close_open())
            self._text_parts.append(text)
            events.append(StreamEvent("text", text=text))

        for fragment in delta.get("tool_calls") or []:
            events.extend(self._consume_tool_fragment(fragment))

        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]
        return events

    def _consume_tool_fragment(self, fragment):
        events = []
        index = fragment.get("index")
        if index is None:
            # Servers that stream a single tool call sometimes omit the index.
            index = self._order[-1] if self._order else 0
        if index not in self._calls:
            events.extend(self._close_open())
            self._calls[index] = {"id": None, "name": "", "args": ""}
            self._order.append(index)
        call = self._calls[index]

        if fragment.get("id"):
            call["id"] = fragment["id"]
        function = fragment.get("function") or {}
        if function.get("name"):
            # Concatenated, not assigned: the name is usually whole in the first
            # fragment, but nothing in the protocol promises it.
            call["name"] += function["name"]
        if function.get("arguments"):
            call["args"] += function["arguments"]

        # Hold the start event until the name is known — agent.py prints
        # block.name there, and "[tool] " with nothing after it reads as a bug.
        if self._open is None and call["name"]:
            self._open = index
            events.append(
                StreamEvent(
                    "content_block_start",
                    content_block=ToolUseBlock(self._call_id(index), call["name"], {}),
                )
            )
        return events

    def _close_open(self):
        if self._open is None:
            return []
        index, self._open = self._open, None
        return [StreamEvent("content_block_stop", content_block=self._block_for(index))]

    def _call_id(self, index):
        # Some servers omit the id when there is only one call, but the loop
        # needs one: it is what pairs the tool_result back to the request.
        return self._calls[index]["id"] or f"call_{index}"

    def _block_for(self, index):
        call = self._calls[index]
        return ToolUseBlock(
            self._call_id(index), call["name"], _parse_args(call["args"])
        )

    def _build(self):
        content = []
        if self._text_parts:
            content.append(TextBlock("".join(self._text_parts)))
        content.extend(self._block_for(i) for i in self._order)

        stop_reason = _FINISH_REASONS.get(self._finish_reason, "end_turn")
        if self._order:
            # Load-bearing override. agent.py only runs tools when stop_reason
            # is "tool_use", and servers have been seen reporting finish_reason
            # "stop" on a chunk that carried tool_calls. Trust the content.
            stop_reason = "tool_use"
        return ShimMessage(
            content, stop_reason, usage=_usage_from(self._usage), model=self._model
        )


# --- Stub provider ------------------------------------------------------------


class StubLLM:
    """Scripted provider. Same seam, no network, no credentials.

    Exists so the whole spine — solve entrypoint, task-dir read, tool dispatch,
    deliverable on disk — can be verified without a model. A stub that only
    returned text would not exercise the part most likely to break, so the
    script speaks in turns of blocks and tool calls.

    A turn is a list of blocks, each one of:
        "some text"                 -> a text block
        ("write", {"path": "a.py"}) -> a tool_use block
        {"type": "tool_use", "name": ..., "input": ..., "id": ...}
        {"type": "text", "text": ...}

    Turns with a tool_use block stop with "tool_use", so the agent loop runs the
    tool and comes back for the next turn. When the script runs out the stub
    ends the conversation rather than looping forever.
    """

    def __init__(self, script=None, model="stub", max_tokens=None):
        self.script = list(script or [])
        self.model = model
        self.max_tokens = _resolve_max_tokens(max_tokens)
        self.turn = 0
        # Every request the agent made, for assertions about what it sent.
        self.requests = []
        self.request_count = 0

    def stream(self, messages, system=None, tools=None, stop_sequences=None):
        self.requests.append(
            {
                "messages": list(messages),
                "system": system,
                "tools": tools,
                "stop_sequences": stop_sequences,
            }
        )
        self.request_count += 1
        blocks = self._next_turn()
        return _StubStreamManager(blocks)

    def _next_turn(self):
        if self.turn >= len(self.script):
            return [TextBlock("stub: script exhausted")]
        spec = self.script[self.turn]
        self.turn += 1
        if isinstance(spec, (str, tuple, dict)):
            spec = [spec]
        return [self._block(i, b) for i, b in enumerate(spec)]

    def _block(self, position, spec):
        if isinstance(spec, str):
            return TextBlock(spec)
        if isinstance(spec, tuple):
            name, tool_input = spec
            return ToolUseBlock(f"stub_{self.turn}_{position}", name, tool_input)
        if isinstance(spec, dict):
            if spec.get("type") == "text":
                return TextBlock(spec.get("text", ""))
            return ToolUseBlock(
                spec.get("id") or f"stub_{self.turn}_{position}",
                spec["name"],
                spec.get("input") or {},
            )
        raise TypeError(f"unsupported stub block: {spec!r}")

    content_from_message = staticmethod(_content_from_shim)
    text_from_message = staticmethod(_text_from_shim)

    async def aclose(self):
        return None


class _StubStreamManager:
    def __init__(self, blocks):
        self._blocks = blocks

    async def __aenter__(self):
        return _StubStream(self._blocks)

    async def __aexit__(self, *exc_info):
        return False


class _StubStream:
    def __init__(self, blocks):
        self._blocks = blocks
        self._gen = None

    def __aiter__(self):
        if self._gen is None:
            self._gen = self._events()
        return self._gen

    async def _events(self):
        for block in self._blocks:
            if block.type == "text":
                yield StreamEvent("text", text=block.text)
            else:
                # Start carries no input, exactly as the real stream does —
                # the TUI must not come to depend on having it early.
                yield StreamEvent(
                    "content_block_start",
                    content_block=ToolUseBlock(block.id, block.name, {}),
                )
                yield StreamEvent("content_block_stop", content_block=block)

    async def get_final_message(self):
        async for _ in self:
            pass
        has_tools = any(b.type == "tool_use" for b in self._blocks)
        return ShimMessage(
            list(self._blocks),
            "tool_use" if has_tools else "end_turn",
            usage=Usage(),
            model="stub",
        )
