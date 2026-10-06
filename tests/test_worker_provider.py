"""worker._call_model through the provider abstraction.

Proves the Anthropic request the worker sends is the pre-abstraction request
(real system prompt, real tools, adaptive thinking, effort, bounded timeout),
that ``provider`` is audited, and that ``provider_reasoning`` blocks survive
the loop, the save, and the next request.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import worker
from utils import clients, llm
from utils.llm import anthropic_provider as ap
from utils.llm.types import ModelResponse, Usage, provider_reasoning_block
from utils.slack_helpers import MessageDelivery
from utils.system_prompt import load_system_prompt


class _Block(SimpleNamespace):
    def model_dump(self):
        return dict(vars(self))


class _Response(SimpleNamespace):
    pass


class _Client:
    def __init__(self, responses):
        self.messages = self
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.responses.pop(0)


@pytest.fixture
def quiet(monkeypatch):
    state = {"audits": [], "saved": [], "posts": []}
    monkeypatch.setattr(worker, "post_message", lambda channel, text, thread_ts=None, blocks=None: state["posts"].append(text) or "ts")
    monkeypatch.setattr(worker, "update_message", lambda *a, **k: None)
    monkeypatch.setattr(
        worker,
        "post_message_chunks",
        lambda channel, text, thread_ts=None: MessageDelivery((text,), ("answer-ts",), 1),
    )
    monkeypatch.setattr(worker, "_post_feedback_prompt", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_attach_final_sql", lambda *a, **k: True)
    monkeypatch.setattr(worker, "save_session", lambda thread_ts, messages: state["saved"].append(deepcopy(messages)))
    monkeypatch.setattr(worker, "_dispatch_tool", lambda name, inp: {"result": 2, "result_display": "2"})
    monkeypatch.setattr(worker, "audit", lambda event, **fields: state["audits"].append((event, fields)))
    monkeypatch.setenv("NUMERIC_GUARD_MODE", "off")
    llm.reset_provider()
    yield state
    llm.reset_provider()


def test_worker_sends_the_exact_baseline_anthropic_request(monkeypatch, quiet):
    client = _Client(
        [
            _Response(
                stop_reason="end_turn",
                content=[_Block(type="text", text="Done")],
                model="claude-sonnet-4-6",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )
        ]
    )
    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    messages = [{"role": "user", "content": "hi"}]

    assert worker.run_agent(deepcopy(messages), "C1", "t1") == "answered"

    (kwargs,) = client.calls
    timeout = kwargs.pop("timeout")
    # Identical to the pre-abstraction request except for the deliberate
    # prompt-caching delta: the user's text is sent as a text block carrying
    # the moving ``cache_control`` breakpoint (see utils/llm/anthropic_provider.py).
    assert kwargs == {
        "model": worker.MODEL,
        "max_tokens": worker.MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": load_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": worker.ALL_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            }
        ],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": worker.EFFORT},
    }
    assert ap.count_cache_breakpoints(kwargs) == 2
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["max_tokens"] == 24_000
    assert kwargs["output_config"] == {"effort": "high"}
    assert timeout.connect == clients.ANTHROPIC_CONNECT_TIMEOUT_SECONDS
    assert timeout.read == clients.ANTHROPIC_TIMEOUT_SECONDS


def test_model_call_and_run_summary_carry_provider(monkeypatch, quiet):
    client = _Client(
        [
            _Response(
                stop_reason="end_turn",
                content=[_Block(type="text", text="Done")],
                model="claude-sonnet-4-6-20260101",
                usage=SimpleNamespace(input_tokens=5, output_tokens=2),
            )
        ]
    )
    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    metrics = worker.RunMetrics()

    worker.run_agent([{"role": "user", "content": "hi"}], "C1", "t1", metrics=metrics)

    (model_call,) = [f for e, f in quiet["audits"] if e == "model_call"]
    assert model_call["provider"] == "anthropic"
    assert model_call["model"] == "claude-sonnet-4-6-20260101"
    assert model_call["requested_model"] == worker.MODEL
    assert model_call["stop_reason"] == "end_turn"
    assert model_call["transport"] == "create"  # 16k cap -> non-streaming
    assert metrics.summary()["provider"] == "anthropic"
    assert worker.PROVIDER_NAME == "anthropic"


class _ReasoningProvider:
    """A fake non-Anthropic provider that emits provider_reasoning like OpenAI would."""

    name = "openai"
    model = "gpt-5.6-terra"

    def __init__(self):
        self.requests = []

    def supports_effort(self, level):
        return True

    def map_effort(self, level):
        return level

    def create(self, request):
        self.requests.append(deepcopy(request.messages))
        if len(self.requests) == 1:
            return ModelResponse(
                model="gpt-5.6-terra",
                stop_reason="tool_use",
                content=[
                    provider_reasoning_block("openai", {"type": "reasoning", "id": "rs_1", "encrypted_content": "abc"}),
                    {"type": "tool_use", "id": "call_1", "name": "calculate", "input": {"expression": "1+1"}},
                ],
                usage=Usage(input_tokens=10, output_tokens=3),
                raw_stop_reason="completed",
                provider="openai",
            )
        return ModelResponse(
            model="gpt-5.6-terra",
            stop_reason="end_turn",
            content=[
                provider_reasoning_block("openai", {"type": "reasoning", "id": "rs_2", "encrypted_content": "def"}),
                {"type": "text", "text": "The answer is 2."},
            ],
            usage=Usage(input_tokens=20, cache_read_input_tokens=9, output_tokens=5),
            raw_stop_reason="completed",
            provider="openai",
        )


def test_provider_reasoning_blocks_are_replayed_and_persisted(monkeypatch, quiet):
    provider = _ReasoningProvider()
    monkeypatch.setattr(worker, "get_provider", lambda: provider)
    messages = [{"role": "user", "content": "add one and one"}]

    outcome = worker.run_agent(messages, "C1", "t1")

    assert outcome == "answered"
    # Second request replays the first turn's reasoning + tool_use + tool_result.
    second = provider.requests[1]
    assert second[1]["role"] == "assistant"
    assert second[1]["content"][0]["type"] == "provider_reasoning"
    assert second[1]["content"][1]["type"] == "tool_use"
    assert second[2]["content"][0]["type"] == "tool_result"
    assert second[2]["content"][0]["tool_use_id"] == "call_1"
    # Saved history keeps the final turn's reasoning alongside the delivered text.
    saved = quiet["saved"][-1]
    assert [b["type"] for b in saved[-1]["content"]] == ["provider_reasoning", "text"]
    assert saved[-1]["content"][1]["text"] == "The answer is 2."
    (model_call, *_rest) = [f for e, f in quiet["audits"] if e == "model_call"]
    assert model_call["provider"] == "openai"
    assert model_call["model"] == "gpt-5.6-terra"


def test_unexpected_stop_posts_the_raw_provider_reason(monkeypatch, quiet):
    class _OtherProvider(_ReasoningProvider):
        def create(self, request):
            return ModelResponse(
                model="gpt-5.6-terra",
                stop_reason="other",
                content=[{"type": "text", "text": "x"}],
                raw_stop_reason="incomplete:content_filter",
                provider="openai",
            )

    monkeypatch.setattr(worker, "get_provider", lambda: _OtherProvider())

    assert worker.run_agent([{"role": "user", "content": "q"}], "C1", "t1") == "unexpected_stop"
    assert quiet["posts"] == ["_Stopped unexpectedly: incomplete:content_filter_"]


def test_extract_text_skips_non_text_blocks_in_dict_and_object_form():
    blocks = [
        {"type": "provider_reasoning", "provider": "openai", "payload": {"text": "12345"}},
        {"type": "text", "text": "a"},
        _Block(type="tool_use", id="t", name="n", input={}),
        _Block(type="text", text="b"),
        {"type": "text", "text": None},
    ]
    assert worker._extract_text(blocks) == "ab"


def test_replace_assistant_text_keeps_provider_reasoning_block():
    message = {
        "role": "assistant",
        "content": [
            provider_reasoning_block("openai", {"id": "rs"}),
            {"type": "text", "text": "draft"},
        ],
    }
    worker._replace_assistant_text(message, "final")
    assert message["content"] == [
        provider_reasoning_block("openai", {"id": "rs"}),
        {"type": "text", "text": "final"},
    ]


def test_worker_accepts_xhigh_effort_env():
    assert worker._resolve_effort("xhigh") == "xhigh"


def _no_cache_control(messages):
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                assert "cache_control" not in block, message
    return True


def test_two_iteration_run_caches_the_prefix_and_never_persists_the_marker(monkeypatch, quiet):
    """End-to-end through ``worker.run_agent``: iteration 0 marks the user text,
    iteration 1 marks the new tool_result, iteration 0's assistant turn (with
    its thinking) reappears byte-identical in iteration 1's request, and the
    session saved to DynamoDB never carries ``cache_control``."""
    client = _Client(
        [
            _Response(
                stop_reason="tool_use",
                content=[
                    _Block(type="thinking", thinking="need to add", signature="sig-1"),
                    _Block(type="tool_use", id="t1", name="calculate", input={"expression": "1+1"}),
                ],
                model="claude-sonnet-5",
                usage=SimpleNamespace(input_tokens=100, cache_creation_input_tokens=100, cache_read_input_tokens=56_000, output_tokens=20),
            ),
            _Response(
                stop_reason="end_turn",
                content=[_Block(type="thinking", thinking="got it", signature="sig-2"), _Block(type="text", text="It is 2.")],
                model="claude-sonnet-5",
                usage=SimpleNamespace(input_tokens=30, cache_creation_input_tokens=120, cache_read_input_tokens=56_100, output_tokens=10),
            ),
        ]
    )
    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    messages = [{"role": "user", "content": "add one and one"}]

    assert worker.run_agent(messages, "C1", "t1") == "answered"

    first, second = (call["messages"] for call in client.calls)
    assert first == [
        {"role": "user", "content": [{"type": "text", "text": "add one and one", "cache_control": {"type": "ephemeral"}}]}
    ]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    # The iteration-0 assistant turn is replayed with its thinking intact.
    assert second[1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "need to add", "signature": "sig-1"},
            {"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expression": "1+1"}},
        ],
    }
    # Exactly one marker in ``messages`` per request, on the newest block.
    for call in client.calls:
        assert ap.count_cache_breakpoints(call) == 2
    (result_block,) = second[2]["content"]
    assert result_block["type"] == "tool_result" and result_block["cache_control"] == {"type": "ephemeral"}
    # The user text that carried the marker on call 1 is sent unmarked on call 2.
    assert second[0]["content"] == "add one and one"

    # Nothing the worker persists carries the annotation.
    assert quiet["saved"], "the run must have saved the session"
    for saved in quiet["saved"]:
        assert _no_cache_control(saved)
    assert _no_cache_control(messages)
    assert messages[0] == {"role": "user", "content": "add one and one"}
    assert [b["type"] for b in messages[1]["content"]] == ["thinking", "tool_use"]
