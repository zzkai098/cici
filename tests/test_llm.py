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
