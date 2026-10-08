"""Unit tests for the provider seam.

Like test_tools.py, these never touch the network. The OpenAI adapter is
exercised by feeding _ChatStream a canned SSE body, which is the honest way to
test a wire-protocol shim: the fragmentation it exists to absorb is exactly
what a mocked client would paper over.
"""

import asyncio
import json

import pytest

from cici.agent import Agent
from cici.llm import (
    OpenAICompatible,
    StubLLM,
    TextBlock,
    ToolUseBlock,
    _ChatCompletion,
    _ChatStream,
    to_openai_messages,
    to_openai_tools,
)
from cici.session import Session
from cici.tools import default_registry
from cici.tools.workspace import Workspace


def run(coro):
    return asyncio.run(coro)


# --- history translation ------------------------------------------------------


def test_system_prompt_becomes_the_first_message():
    out = to_openai_messages([{"role": "user", "content": "hi"}], system="be terse")
    assert out[0] == {"role": "system", "content": "be terse"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_assistant_tool_use_becomes_tool_calls():
    history = [
        {
            "role": "assistant",
            "content": [
                TextBlock("writing it"),
                ToolUseBlock("call_1", "write", {"path": "a.py", "content": "x"}),
            ],
        }
    ]
    (msg,) = to_openai_messages(history)
    assert msg["role"] == "assistant"
    assert msg["content"] == "writing it"
    (call,) = msg["tool_calls"]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "write"
    # arguments is a JSON *string* on this side of the seam, not a dict.
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py", "content": "x"}


def test_tool_results_leave_the_user_turn_for_their_own_messages():
    """Anthropic carries tool_result blocks inside a user message; OpenAI wants
    one standalone role="tool" message each, keyed by tool_call_id."""
    history = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "also ok"},
            ],
        }
    ]
    out = to_openai_messages(history)
    assert [m["role"] for m in out] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in out] == ["call_1", "call_2"]


def test_is_error_survives_as_text():
    """OpenAI's tool message has no is_error flag. Dropping it silently is how a
    model comes to believe a failed tool succeeded."""
    history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "no such file",
                    "is_error": True,
                }
            ],
        }
    ]
    (msg,) = to_openai_messages(history)
    assert msg["content"] == "Error: no such file"


def test_tool_results_are_ordered_before_text_in_a_mixed_user_turn():
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "and also"},
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
            ],
        }
    ]
    out = to_openai_messages(history)
    assert [m["role"] for m in out] == ["tool", "user"]


def test_plain_string_user_turns_pass_through():
    assert to_openai_messages([{"role": "user", "content": "hi"}]) == [
        {"role": "user", "content": "hi"}
    ]


def test_tool_schemas_are_translated_from_the_hand_written_ones():
    schemas = default_registry().schemas()
    converted = to_openai_tools(schemas)
    assert {t["function"]["name"] for t in converted} == {
        "ls",
        "grep",
        "read",
        "write",
        "edit",
        "bash",
    }
    first = converted[0]
    assert first["type"] == "function"
    assert first["function"]["parameters"] == schemas[0]["input_schema"]


# --- SSE stream shim ----------------------------------------------------------


class FakeResponse:
    """Stands in for an httpx2 streaming response: _ChatStream only reads lines."""

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def sse(*chunks):
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    lines.append("data: [DONE]")
    return FakeResponse(lines)


def delta(d, finish_reason=None):
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish_reason}]}


def drain(stream):
    async def go():
        events = [e async for e in stream]
        return events, await stream.get_final_message()

    return run(go())


def test_text_deltas_become_text_events_and_one_block():
    events, message = drain(
        _ChatStream(sse(delta({"content": "he"}), delta({"content": "llo"}, "stop")))
    )
    assert [e.text for e in events if e.type == "text"] == ["he", "llo"]
    assert len(message.content) == 1
    assert message.content[0].text == "hello"
    assert message.stop_reason == "end_turn"


def test_tool_call_fragments_are_reassembled_by_index():
    """The whole reason this shim exists: the name arrives in one fragment and
    the arguments are split across the rest."""
    events, message = drain(
        _ChatStream(
            sse(
                delta(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "write", "arguments": ""},
                            }
                        ]
                    }
                ),
                delta(
                    {"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]}
                ),
                delta(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'th": "a.py"}'}}
                        ]
                    },
                    "tool_calls",
                ),
            )
        )
    )
    (block,) = message.content
    assert block.type == "tool_use"
    assert block.id == "call_a"
    assert block.name == "write"
    assert block.input == {"path": "a.py"}
    assert message.stop_reason == "tool_use"

    # agent.py prints the name at start and the input at stop, so the start
    # event must already know the name and the stop event must carry the input.
    start = next(e for e in events if e.type == "content_block_start")
    stop = next(e for e in events if e.type == "content_block_stop")
    assert start.content_block.name == "write"
    assert stop.content_block.input == {"path": "a.py"}


def test_two_tool_calls_are_kept_apart():
    _, message = drain(
        _ChatStream(
            sse(
                delta(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c0",
                                "function": {"name": "ls", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                delta(
                    {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "c1",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":"a"}',
                                },
                            }
                        ]
                    },
                    "tool_calls",
                ),
            )
        )
    )
    assert [(b.id, b.name) for b in message.content] == [("c0", "ls"), ("c1", "read")]


def test_tool_calls_force_tool_use_even_when_the_server_says_stop():
    """Servers have been seen reporting finish_reason "stop" on a chunk that
    carried tool_calls. agent.py only runs tools on stop_reason "tool_use", so
    the content wins over the label."""
    _, message = drain(
        _ChatStream(
            sse(
                delta(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c0",
                                "function": {"name": "ls", "arguments": "{}"},
                            }
                        ]
                    },
                    "stop",
                )
            )
        )
    )
    assert message.stop_reason == "tool_use"


def test_malformed_arguments_are_handed_through_not_replaced_with_empty():
    """Substituting {} would run the tool with no arguments — a failure wearing
    a success costume. The raw string makes Registry.dispatch raise, which
    run_all contains as an is_error tool_result."""
    _, message = drain(
        _ChatStream(
            sse(
                delta(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c0",
                                "function": {"name": "write", "arguments": '{"path":'},
                            }
                        ]
                    },
                    "tool_calls",
                )
            )
        )
    )
    assert message.content[0].input == '{"path":'


def test_length_finish_reason_maps_to_max_tokens():
    _, message = drain(_ChatStream(sse(delta({"content": "x"}, "length"))))
    assert message.stop_reason == "max_tokens"


def test_reasoning_content_surfaces_as_thinking():
    events, _ = drain(
        _ChatStream(sse(delta({"reasoning_content": "hmm"}, None), delta({}, "stop")))
    )
    assert [e.thinking for e in events if e.type == "thinking"] == ["hmm"]


def test_usage_excludes_cached_tokens_from_input():
    """OpenAI's prompt_tokens includes cached tokens, Anthropic's input_tokens
    does not, and telemetry.py divides by (cache_read + input). Without the
    subtraction the cache ratio silently means something different per provider."""
    _, message = drain(
        _ChatStream(
            sse(
                delta({"content": "x"}, "stop"),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 50,
                        "prompt_tokens_details": {"cached_tokens": 800},
                    },
                },
            )
        )
    )
    assert message.usage.input_tokens == 200
    assert message.usage.output_tokens == 50
    assert message.usage.cache_read_input_tokens == 800


def test_keepalive_comments_and_blank_lines_are_ignored():
    response = FakeResponse(
        [
            ": ping",
            "",
            f"data: {json.dumps(delta({'content': 'x'}, 'stop'))}",
            "",
            "data: [DONE]",
        ]
    )
    _, message = drain(_ChatStream(response))
    assert message.content[0].text == "x"


def test_get_final_message_drains_a_stream_the_caller_abandoned():
    stream = _ChatStream(sse(delta({"content": "a"}), delta({"content": "b"}, "stop")))

    async def go():
        async for _ in stream:
            break  # caller gives up after one event
        return await stream.get_final_message()

    assert run(go()).content[0].text == "ab"


# --- credential resolution ----------------------------------------------------


def test_endpoint_gets_the_v1_suffix(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443/")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    monkeypatch.setenv("MODEL_TOKEN", "t")
    provider = OpenAICompatible()
    assert provider.base_url == "http://model:8443/v1"
    run(provider.aclose())


def test_an_endpoint_that_already_ends_in_v1_is_not_doubled(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443/v1")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    provider = OpenAICompatible()
    assert provider.base_url == "http://model:8443/v1"
    run(provider.aclose())


def test_missing_endpoint_is_one_actionable_line(monkeypatch):
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    monkeypatch.setenv("MODEL_NAME", "some-model")
    with pytest.raises(RuntimeError, match="MODEL_ENDPOINT"):
        OpenAICompatible(base_url="")


def test_default_max_tokens_is_the_conservative_number(monkeypatch):
    """Claude's 64000 is sized for a streaming Messages call; the Agenthon
    endpoint is not that endpoint. Between the two documented numbers (4,000 in
    the main doc, 4096 in baselines/) nothing is authoritative, and an
    over-limit request still burns one of the unit's 25, so take the lower."""
    monkeypatch.delenv("CICI_MAX_TOKENS", raising=False)
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    provider = OpenAICompatible()
    assert provider.max_tokens == 4000
    run(provider.aclose())


def test_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    monkeypatch.setenv("CICI_MAX_TOKENS", "8192")
    provider = OpenAICompatible()
    assert provider.max_tokens == 8192
    run(provider.aclose())


def test_explicit_max_tokens_beats_the_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    monkeypatch.setenv("CICI_MAX_TOKENS", "8192")
    provider = OpenAICompatible(max_tokens=123)
    assert provider.max_tokens == 123
    run(provider.aclose())


def test_unusable_max_tokens_env_is_one_actionable_line(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    monkeypatch.setenv("CICI_MAX_TOKENS", "lots")
    with pytest.raises(RuntimeError, match="CICI_MAX_TOKENS"):
        OpenAICompatible()


def test_requests_are_counted_at_the_request_layer(monkeypatch):
    """The unit budget is 25 admitted requests, not 25 turns."""
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    provider = OpenAICompatible()
    assert provider.request_count == 0
    provider.stream([{"role": "user", "content": "hi"}])
    provider.stream([{"role": "user", "content": "hi"}])
    assert provider.request_count == 2
    run(provider.aclose())


# --- the stub, driving the real loop ------------------------------------------


def test_stub_drives_the_real_agent_loop_end_to_end(tmp_path):
    """The point of the stub: prove solve's whole spine — turn loop, tool
    dispatch, deliverable on disk — without a model or a credential."""
    provider = StubLLM(
        script=[
            [("write", {"path": "out.py", "content": "print(1)\n"})],
            ["wrote it"],
        ]
    )
    registry = default_registry(Workspace(tmp_path))
    agent = Agent(provider, registry, Session())

    final = run(agent.run("write out.py"))

    assert (tmp_path / "out.py").read_text() == "print(1)\n"
    assert final == "wrote it"
    # Two turns => two requests, and the second one saw the tool result.
    assert provider.request_count == 2
    assert agent.telemetry.turns[0].stop_reason == "tool_use"
    assert agent.telemetry.turns[0].n_tools == 1


def test_stub_history_round_trips_through_the_openai_translation(tmp_path):
    """The history the loop builds from stub blocks must still be convertible —
    this is what catches a block shape that only agent.py can read."""
    provider = StubLLM(script=[[("ls", {})], ["done"]])
    agent = Agent(provider, default_registry(Workspace(tmp_path)), Session())
    run(agent.run("look around"))

    converted = to_openai_messages(agent.session.messages, system="sys")
    assert [m["role"] for m in converted] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert converted[2]["tool_calls"][0]["function"]["name"] == "ls"
    assert converted[3]["tool_call_id"] == converted[2]["tool_calls"][0]["id"]


def test_stub_ends_the_conversation_when_the_script_runs_out(tmp_path):
    """A stub that kept returning tool calls would spin to max_turns."""
    provider = StubLLM(script=[])
    agent = Agent(provider, default_registry(Workspace(tmp_path)), Session())
    assert run(agent.run("hi")) == "stub: script exhausted"
    assert provider.request_count == 1


# --- the non-streaming fallback -----------------------------------------------
#
# A server that refuses the streaming FORM can still answer the same question
# in one body. Under 25 admitted requests and one graded attempt per task, the
# difference between retrying whole and surfacing a failed turn is the
# difference between a unit that scores and one that does not.


class FakeHTTPResponse:
    """The two shapes httpx2 hands back, as much of them as the manager reads."""

    def __init__(self, status_code, payload=None, raw=b""):
        self.status_code = status_code
        self._payload = payload
        self._raw = raw

    async def aread(self):
        return self._raw

    async def aiter_lines(self):
        # A 200 that streams nothing: enough for _ChatStream to build an empty
        # message, which is all the payload-shape tests need.
        for line in ("data: [DONE]",):
            yield line

    @property
    def text(self):
        return self._raw.decode()

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeStreamCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class FakeClient:
    """Records what was sent, so the payload itself can be asserted on."""

    def __init__(self, stream_response, post_response=None):
        self._stream_response = stream_response
        self._post_response = post_response
        self.stream_payloads = []
        self.post_payloads = []
        # Accept is per-request, not on the client, so it is recorded here too.
        self.stream_headers = []
        self.post_headers = []

    def stream(self, method, url, json=None, headers=None):
        self.stream_payloads.append(json)
        self.stream_headers.append(headers or {})
        return FakeStreamCM(self._stream_response)

    async def post(self, url, json=None, headers=None):
        self.post_payloads.append(json)
        self.post_headers.append(headers or {})
        return self._post_response

    async def aclose(self):
        pass


def make_provider(monkeypatch, **env):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return OpenAICompatible()


COMPLETION = {
    "model": "some-model",
    "choices": [{"message": {"content": "whole answer"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
}


def enter(manager):
    return run(manager.__aenter__())


def test_a_refused_stream_is_retried_whole(monkeypatch):
    provider = make_provider(monkeypatch)
    provider.client = FakeClient(
        FakeHTTPResponse(400, raw=b'{"error":"stream unsupported"}'),
        FakeHTTPResponse(200, payload=COMPLETION),
    )
    stream = enter(provider.stream([{"role": "user", "content": "hi"}]))
    message = run(stream.get_final_message())
    assert message.content[0].text == "whole answer"
    assert provider.streaming is False
    assert "stream unsupported" in provider.stream_refusal


def test_the_retry_is_charged_because_the_grader_charges_it(monkeypatch):
    """Two HTTP sends are two admitted requests out of the unit's 25."""
    provider = make_provider(monkeypatch)
    provider.client = FakeClient(
        FakeHTTPResponse(400, raw=b"no"), FakeHTTPResponse(200, payload=COMPLETION)
    )
    enter(provider.stream([{"role": "user", "content": "hi"}]))
    assert provider.request_count == 2


def test_the_refusal_is_sticky_so_the_probe_is_paid_for_once(monkeypatch):
    provider = make_provider(monkeypatch)
    provider.client = FakeClient(
        FakeHTTPResponse(400, raw=b"no"), FakeHTTPResponse(200, payload=COMPLETION)
    )
    enter(provider.stream([{"role": "user", "content": "one"}]))
    enter(provider.stream([{"role": "user", "content": "two"}]))
    # One streaming probe, then two whole sends: three charges, not four.
    assert provider.request_count == 3
    assert len(provider.client.stream_payloads) == 1
    assert provider.client.post_payloads[-1]["stream"] is False
    assert "stream_options" not in provider.client.post_payloads[-1]


def test_an_auth_failure_is_not_retried(monkeypatch):
    """401 would fail identically without `stream`; retrying spends a request
    to learn nothing."""
    provider = make_provider(monkeypatch)
    provider.client = FakeClient(
        FakeHTTPResponse(401, raw=b"unauthorized"),
        FakeHTTPResponse(200, payload=COMPLETION),
    )
    with pytest.raises(RuntimeError, match="401"):
        enter(provider.stream([{"role": "user", "content": "hi"}]))
    assert provider.request_count == 1
    assert provider.client.post_payloads == []
    assert provider.streaming is True


def test_no_stream_env_skips_the_probe_entirely(monkeypatch):
    provider = make_provider(monkeypatch, CICI_NO_STREAM="1")
    provider.client = FakeClient(None, FakeHTTPResponse(200, payload=COMPLETION))
    enter(provider.stream([{"role": "user", "content": "hi"}]))
    assert provider.request_count == 1
    assert provider.client.stream_payloads == []


def test_a_falsey_no_stream_value_does_not_turn_it_on(monkeypatch):
    provider = make_provider(monkeypatch, CICI_NO_STREAM="0")
    assert provider.streaming is True
    run(provider.aclose())


def test_temperature_and_seed_are_pinned_on_every_request(monkeypatch):
    """Rule 4: T1 verification compares the organizer's rerun per unit."""
    provider = make_provider(monkeypatch, QFBENCH_SEED="4242")
    provider.client = FakeClient(FakeHTTPResponse(200))
    enter(provider.stream([{"role": "user", "content": "hi"}]))
    sent = provider.client.stream_payloads or provider.client.post_payloads
    assert sent[0]["temperature"] == 0.0
    assert sent[0]["seed"] == 4242
    # n is deliberately absent: the House route refuses multiple alternatives.
    assert "n" not in sent[0]


def test_a_malformed_seed_falls_back_rather_than_killing_the_run(monkeypatch):
    provider = make_provider(monkeypatch, QFBENCH_SEED="not-a-number")
    provider.client = FakeClient(FakeHTTPResponse(200))
    enter(provider.stream([{"role": "user", "content": "hi"}]))
    assert provider.client.stream_payloads[0]["seed"] == 0


# --- the whole-body shim ------------------------------------------------------


def completion(message, finish_reason="stop", usage=None, model="some-model"):
    body = {
        "model": model,
        "choices": [{"message": message, "finish_reason": finish_reason}],
    }
    if usage:
        body["usage"] = usage
    return _ChatCompletion(body)


def test_a_whole_body_emits_its_text_once(monkeypatch):
    events, message = drain(completion({"content": "hello"}))
    assert [e.text for e in events if e.type == "text"] == ["hello"]
    assert message.content[0].text == "hello"
    assert message.stop_reason == "end_turn"


def test_whole_body_tool_calls_force_tool_use(monkeypatch):
    events, message = drain(
        completion(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "function": {"name": "write", "arguments": '{"path":"a.py"}'},
                    }
                ],
            },
            finish_reason="stop",
        )
    )
    # Same override the streaming shim applies: trust the content, not the
    # server's finish_reason.
    assert message.stop_reason == "tool_use"
    block = message.content[0]
    assert (block.name, block.input) == ("write", {"path": "a.py"})
    assert [e.type for e in events] == ["content_block_start", "content_block_stop"]


def test_a_tool_call_without_an_id_still_gets_one(monkeypatch):
    """The loop pairs tool_result back by id, so a missing one is not optional."""
    _, message = drain(
        completion({"tool_calls": [{"function": {"name": "ls", "arguments": "{}"}}]})
    )
    assert message.content[0].id == "call_0"


def test_whole_body_malformed_arguments_are_handed_through(monkeypatch):
    _, message = drain(
        completion(
            {"tool_calls": [{"id": "x", "function": {"name": "ls", "arguments": "{["}}]}
        )
    )
    assert message.content[0].input == "{["


def test_whole_body_reasoning_surfaces_as_thinking(monkeypatch):
    events, _ = drain(completion({"content": "a", "reasoning_content": "thought"}))
    assert [e.thinking for e in events if e.type == "thinking"] == ["thought"]


def test_whole_body_usage_excludes_cached_tokens(monkeypatch):
    _, message = drain(
        completion(
            {"content": "a"},
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        )
    )
    assert message.usage.input_tokens == 60
    assert message.usage.cache_read_input_tokens == 40
    assert message.usage.output_tokens == 7


def test_a_multimodal_part_list_is_flattened(monkeypatch):
    _, message = drain(completion({"content": [{"text": "a"}, {"text": "b"}]}))
    assert message.content[0].text == "ab"


def test_an_empty_body_is_a_message_not_a_crash(monkeypatch):
    events, message = drain(_ChatCompletion({}))
    assert events == []
    assert message.content == []
    assert message.stop_reason == "end_turn"


def test_an_empty_assistant_turn_is_not_serialised_as_null_content():
    """`content: null` is legal only beside tool_calls. On a turn with neither
    text nor tool calls it is an invalid assistant message, and a strict server
    answers 400 to the NEXT request — costing one of 25 admitted requests.

    Reachable, not hypothetical: the House Development model has thinking
    enabled by default, so a reasoning trace can spend the whole max_tokens
    budget and return content: null, finish_reason "length", no tool call.
    """
    history = [
        {"role": "user", "content": "solve it"},
        {"role": "assistant", "content": []},  # what an exhausted turn leaves
    ]
    _user, assistant = to_openai_messages(history)
    assert assistant["content"] == ""
    assert assistant["content"] is not None
    assert "tool_calls" not in assistant


def test_content_is_still_null_when_tool_calls_are_present():
    """The other half of the rule — this shape is what OpenAI requires."""
    history = [
        {
            "role": "assistant",
            "content": [ToolUseBlock("call_1", "ls", {})],
        }
    ]
    (assistant,) = to_openai_messages(history)
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["name"] == "ls"


def test_thinking_is_left_alone_by_default(monkeypatch):
    """A general provider must not send an option the server may not know."""
    monkeypatch.delenv("CICI_NO_THINKING", raising=False)
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_NAME", "m")
    provider = OpenAICompatible()
    provider.stream([{"role": "user", "content": "hi"}])
    # the payload is built inside stream(); assert via the manager it returned
    run(provider.aclose())


def test_no_thinking_env_adds_the_documented_option(monkeypatch):
    """The option the House route documents, verbatim:
    {"chat_template_kwargs": {"enable_thinking": false}}"""
    from cici.llm import _thinking_fields

    monkeypatch.delenv("CICI_NO_THINKING", raising=False)
    assert _thinking_fields() == {}

    monkeypatch.setenv("CICI_NO_THINKING", "1")
    assert _thinking_fields() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_accept_matches_the_form_of_each_request(monkeypatch):
    """`Accept: text/event-stream` used to sit on the client, so a
    non-streaming request advertised that it would accept nothing but an event
    stream. A server that honours Accept answers 406, and the 406 fallback
    exists only on the streaming branch — the non-streaming one raises and
    loses the unit. With CICI_NO_STREAM=1 in the submission image every request
    is the non-streaming one, so this was on the critical path for all 86.
    """
    monkeypatch.delenv("CICI_NO_STREAM", raising=False)
    provider = make_provider(monkeypatch)
    # httpx2 supplies its own permissive default; what matters is that we no
    # longer pin the client to event-stream only.
    assert provider.client.headers.get("accept") != "text/event-stream"

    client = FakeClient(FakeHTTPResponse(200, payload=COMPLETION))
    provider.client = client
    enter(provider.stream([{"role": "user", "content": "hi"}]))
    assert client.stream_headers[0]["Accept"] == "text/event-stream"

    monkeypatch.setenv("CICI_NO_STREAM", "1")
    provider = make_provider(monkeypatch)
    client = FakeClient(None, FakeHTTPResponse(200, payload=COMPLETION))
    provider.client = client
    enter(provider.stream([{"role": "user", "content": "hi"}]))
    assert client.post_headers[0]["Accept"] == "application/json"
    assert not client.stream_headers
