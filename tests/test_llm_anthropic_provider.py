"""AnthropicProvider: request identity with the deployed baseline, effort table, mapping."""

import logging
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from anthropic.types.parsed_message import ParsedContentBlock, ParsedMessage

from utils import clients
from utils.llm import anthropic_provider as ap
from utils.llm.anthropic_provider import AnthropicProvider
from utils.llm.types import ModelRequest, ToolSpec, provider_reasoning_block


class _Block(SimpleNamespace):
    def model_dump(self):
        return dict(vars(self))


class _Client:
    def __init__(self, response):
        self.messages = self
        self.calls = []
        self.response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _response(stop_reason="end_turn", content=None, usage=None, model="claude-sonnet-4-6-20260101"):
    resp = SimpleNamespace(stop_reason=stop_reason, content=content or [], model=model)
    if usage is not None:
        resp.usage = SimpleNamespace(**usage)
    return resp


TOOLS = [
    ToolSpec("calculate", "Deterministic arithmetic.", {"type": "object", "properties": {}}),
    ToolSpec("github_find_files", "Find files.", {"type": "object", "properties": {"name": {"type": "string"}}}),
]


def _request(**overrides):
    values = dict(
        system="SYSTEM PROMPT",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        effort="high",
        max_tokens=16_000,
        timeout_s=180.0,
    )
    values.update(overrides)
    return ModelRequest(**values)


CACHE = {"type": "ephemeral"}


class TestRequestShape:
    def test_matches_pre_abstraction_messages_create_call_exactly(self):
        client = _Client(_response(content=[_Block(type="text", text="ok")]))
        provider = AnthropicProvider("claude-sonnet-4-6", client=client)
        messages = [{"role": "user", "content": "hi"}]

        provider.create(_request(messages=messages))

        (kwargs,) = client.calls
        timeout = kwargs.pop("timeout")
        # The only deliberate delta from the pre-abstraction request: the last
        # message's text is a block carrying the moving cache breakpoint.
        assert kwargs == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 16_000,
            "system": [
                {"type": "text", "text": "SYSTEM PROMPT", "cache_control": {"type": "ephemeral"}}
            ],
            "tools": [
                {
                    "name": "calculate",
                    "description": "Deterministic arithmetic.",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "github_find_files",
                    "description": "Find files.",
                    "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                },
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": CACHE}]}
            ],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        assert messages == [{"role": "user", "content": "hi"}]  # caller's list untouched
        # Same bounded SDK Timeout the worker always sent (short connect, full read).
        assert timeout.connect == clients.ANTHROPIC_CONNECT_TIMEOUT_SECONDS
        assert timeout.read == 180.0
        assert timeout.write == 180.0
        assert timeout.pool == 180.0
        # No extra keys sneak in (e.g. metadata, stream, extra_body).
        assert "extra_body" not in kwargs

    def test_prefix_messages_are_passed_through_untouched_when_nothing_to_strip(self):
        client = _Client(_response())
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "a"}]},
            {"role": "user", "content": "follow-up"},
        ]
        snapshot = deepcopy(messages)

        AnthropicProvider("claude-sonnet-4-6", client=client).create(_request(messages=messages))

        sent = client.calls[0]["messages"]
        # Every message but the last is the caller's own object (exact prefix);
        # the last is a copy carrying the breakpoint. Input never mutated.
        assert sent[0] is messages[0] and sent[1] is messages[1]
        assert sent[2] == {
            "role": "user",
            "content": [{"type": "text", "text": "follow-up", "cache_control": CACHE}],
        }
        assert messages == snapshot

    def test_request_model_override_wins(self):
        client = _Client(_response())
        AnthropicProvider("claude-sonnet-4-6", client=client).create(_request(model="claude-sonnet-5"))
        assert client.calls[0]["model"] == "claude-sonnet-5"

    def test_no_timeout_kwarg_when_request_has_none(self):
        client = _Client(_response())
        AnthropicProvider("claude-sonnet-4-6", client=client).create(_request(timeout_s=None))
        assert "timeout" not in client.calls[0]

    def test_thinking_none_omits_parameter_and_empty_effort_omits_output_config(self):
        client = _Client(_response())
        AnthropicProvider("claude-sonnet-4-6", client=client, thinking=None).create(
            _request(effort="")
        )
        assert "thinking" not in client.calls[0]
        assert "output_config" not in client.calls[0]

    def test_extra_body_fallback_for_legacy_sdks(self):
        client = _Client(_response())
        AnthropicProvider("claude-sonnet-4-6", client=client, effort_via_extra_body=True).create(
            _request()
        )
        assert client.calls[0]["extra_body"] == {"output_config": {"effort": "high"}}
        assert "output_config" not in client.calls[0]

    def test_invalid_thinking_mode_rejected(self):
        with pytest.raises(ValueError):
            AnthropicProvider("claude-sonnet-4-6", thinking="bogus")

    def test_uses_shared_lazy_client_when_none_injected(self, monkeypatch):
        client = _Client(_response())
        monkeypatch.setitem(clients._clients, "anthropic_client", client)

        AnthropicProvider("claude-sonnet-4-6").create(_request())

        assert len(client.calls) == 1


class TestForeignBlockStripping:
    def test_provider_reasoning_removed_without_mutating_input(self):
        client = _Client(_response())
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    provider_reasoning_block("openai", {"type": "reasoning", "id": "rs_1"}),
                    {"type": "text", "text": "answer"},
                ],
            },
            {"role": "user", "content": "next"},
        ]
        snapshot = deepcopy(messages)

        AnthropicProvider("claude-sonnet-4-6", client=client).create(_request(messages=messages))

        sent = client.calls[0]["messages"]
        assert sent[1]["content"] == [{"type": "text", "text": "answer"}]
        assert sent[0] is messages[0]
        assert sent[2] == {"role": "user", "content": [{"type": "text", "text": "next", "cache_control": CACHE}]}
        assert messages == snapshot

    def test_reasoning_only_turn_gets_placeholder_text(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [provider_reasoning_block("openai", {"id": "rs"})]},
        ]
        stripped = ap.strip_foreign_blocks(messages)
        assert stripped[1]["content"] == [{"type": "text", "text": ap.FOREIGN_REASONING_PLACEHOLDER}]

    def test_user_turns_and_string_content_are_left_alone(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": "plain"},
        ]
        assert ap.strip_foreign_blocks(messages) is messages


class TestResponseMapping:
    def test_blocks_stop_reason_usage_and_model(self):
        client = _Client(
            _response(
                stop_reason="tool_use",
                content=[
                    _Block(type="thinking", thinking="hmm", signature="sig"),
                    _Block(type="tool_use", id="t1", name="calculate", input={"expression": "1+1"}),
                ],
                usage={
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 7000,
                    "output_tokens": 30,
                },
            )
        )

        resp = AnthropicProvider("claude-sonnet-4-6", client=client).create(_request())

        assert resp.provider == "anthropic"
        assert resp.model == "claude-sonnet-4-6-20260101"
        assert resp.stop_reason == "tool_use"
        assert resp.raw_stop_reason == "tool_use"
        assert resp.content == [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expression": "1+1"}},
        ]
        assert resp.usage.as_dict() == {
            "input_tokens": 100,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 7000,
            "output_tokens": 30,
        }

    @pytest.mark.parametrize("raw", ["end_turn", "tool_use", "max_tokens", "refusal", "pause_turn"])
    def test_known_stop_reasons_pass_through(self, raw):
        assert ap.normalize_stop_reason(raw) == raw

    def test_unknown_stop_reason_normalizes_to_other_but_keeps_raw(self):
        client = _Client(_response(stop_reason="stop_sequence"))
        resp = AnthropicProvider("claude-sonnet-4-6", client=client).create(_request())
        assert resp.stop_reason == "other"
        assert resp.raw_stop_reason == "stop_sequence"

    def test_missing_usage_and_model_default(self):
        client = _Client(SimpleNamespace(stop_reason="end_turn", content=[]))
        resp = AnthropicProvider("claude-sonnet-4-6", client=client).create(_request())
        assert resp.model == "claude-sonnet-4-6"
        assert resp.usage.as_dict() == {
            "input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
        }

    def test_usage_tolerates_none_and_non_numeric(self):
        usage = ap.usage_from_response(
            SimpleNamespace(usage=SimpleNamespace(input_tokens="n/a", cache_read_input_tokens=None, output_tokens=3))
        )
        assert usage.as_dict() == {
            "input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 3,
        }


class TestEffort:
    @pytest.mark.parametrize("level", ["low", "medium", "high", "max"])
    def test_sonnet_46_passthrough(self, level):
        provider = AnthropicProvider("claude-sonnet-4-6")
        assert provider.supports_effort(level)
        assert provider.map_effort(level) == level

    def test_sonnet_46_maps_xhigh_to_max(self, caplog):
        caplog.set_level("INFO")
        provider = AnthropicProvider("claude-sonnet-4-6")
        assert not provider.supports_effort("xhigh")
        assert provider.map_effort("xhigh") == "max"
        assert "xhigh" in caplog.text

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5", "claude-opus-5-20260301"])
    def test_newer_models_pass_xhigh_through(self, model):
        provider = AnthropicProvider(model)
        assert provider.supports_effort("xhigh")
        assert provider.map_effort("xhigh") == "xhigh"

    def test_unknown_model_uses_default_table(self):
        assert ap.effort_levels_for("claude-something-new") == ap.DEFAULT_EFFORT_LEVELS
        assert ap.effort_levels_for("claude-sonnet-4-6-20260115") == ap.MODEL_EFFORT_LEVELS["claude-sonnet-4-6"]

    def test_unsupported_level_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            AnthropicProvider("claude-sonnet-4-6").map_effort("turbo")

    def test_mapped_effort_is_what_gets_sent(self):
        client = _Client(_response())
        AnthropicProvider("claude-sonnet-4-6", client=client).create(_request(effort="xhigh"))
        assert client.calls[0]["output_config"] == {"effort": "max"}

    def test_effort_table_has_sonnet_5_with_xhigh(self):
        assert ap.MODEL_EFFORT_LEVELS["claude-sonnet-5"] == ("low", "medium", "high", "xhigh", "max")
        assert ap.MODEL_EFFORT_LEVELS["claude-sonnet-4-6"] == ("low", "medium", "high", "max")


class TestRecommendedMaxTokens:
    @pytest.mark.parametrize(
        ("model", "effort", "expected"),
        [
            ("claude-sonnet-4-6", "low", 16_000),
            ("claude-sonnet-4-6", "medium", 16_000),
            ("claude-sonnet-4-6", "high", 16_000),
            ("claude-sonnet-4-6", "xhigh", 32_000),  # xhigh -> max on Sonnet 4.6
            ("claude-sonnet-4-6", "max", 32_000),
            ("claude-sonnet-5", "low", 24_000),
            ("claude-sonnet-5", "medium", 24_000),
            ("claude-sonnet-5", "high", 24_000),
            ("claude-sonnet-5", "xhigh", 48_000),
            ("claude-sonnet-5", "max", 64_000),
            ("claude-opus-5", "high", 24_000),
            ("claude-opus-5", "xhigh", 48_000),
            ("claude-opus-5", "max", 64_000),
        ],
    )
    def test_table(self, model, effort, expected):
        assert ap.recommended_max_tokens(model, effort) == expected

    def test_dated_snapshot_and_case_are_tolerated(self):
        assert ap.recommended_max_tokens("claude-sonnet-5-20260301", "xhigh") == 48_000
        assert ap.recommended_max_tokens(" Claude-Opus-5 ", "MAX") == 64_000

    def test_unknown_model_or_effort_returns_baseline(self):
        assert ap.recommended_max_tokens("claude-something-new", "max") == 16_000
        assert ap.recommended_max_tokens("gpt-5.6-terra", "high") == 16_000
        assert ap.recommended_max_tokens("claude-sonnet-5", "turbo") == 16_000
        assert ap.recommended_max_tokens(None, None) == 16_000

    def test_baseline_constant_is_the_production_default(self):
        assert ap.DEFAULT_MAX_TOKENS == 16_000


class _StreamHandle:
    """Stand-in for the SDK's ``MessageStreamManager``: a context manager whose
    ``get_final_message()`` hands back the accumulated ``Message``."""

    def __init__(self, response, log):
        self.response = response
        self.log = log

    def __enter__(self):
        self.log.append("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log.append("exit")
        return False

    def get_final_message(self):
        self.log.append("get_final_message")
        return self.response


class _StreamingClient(_Client):
    """Fake client exposing both ``messages.create`` and ``messages.stream``."""

    def __init__(self, response):
        super().__init__(response)
        self.stream_calls = []
        self.stream_log = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return _StreamHandle(self.response, self.stream_log)


def _timeout_fields(timeout):
    return (timeout.connect, timeout.read, timeout.write, timeout.pool)


class TestStreaming:
    def test_default_mode_is_auto_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(ap.STREAM_ENV_VAR, raising=False)
        assert AnthropicProvider("claude-sonnet-4-6").stream_mode == "auto"

    @pytest.mark.parametrize("raw", ["always", "never", "auto", " ALWAYS "])
    def test_mode_read_from_env(self, monkeypatch, raw):
        monkeypatch.setenv(ap.STREAM_ENV_VAR, raw)
        assert AnthropicProvider("claude-sonnet-4-6").stream_mode == raw.strip().lower()

    def test_invalid_env_value_warns_and_uses_auto(self, monkeypatch, caplog):
        monkeypatch.setenv(ap.STREAM_ENV_VAR, "sometimes")
        caplog.set_level("WARNING")
        assert AnthropicProvider("claude-sonnet-4-6").stream_mode == "auto"
        assert "ANTHROPIC_STREAM" in caplog.text

    def test_explicit_constructor_mode_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(ap.STREAM_ENV_VAR, "always")
        assert AnthropicProvider("claude-sonnet-4-6", stream="never").stream_mode == "never"

    def test_invalid_constructor_mode_rejected(self):
        with pytest.raises(ValueError, match="stream must be one of"):
            AnthropicProvider("claude-sonnet-4-6", stream="bogus")

    @pytest.mark.parametrize(
        ("mode", "max_tokens", "expected"),
        [
            ("auto", 16_000, False),
            ("auto", 21_000, False),
            ("auto", 21_001, True),
            ("auto", 25_000, True),
            ("always", 16_000, True),
            ("always", 1, True),
            ("never", 25_000, False),
            ("never", 128_000, False),
        ],
    )
    def test_decision_logic(self, mode, max_tokens, expected):
        assert ap.should_stream(mode, max_tokens) is expected
        assert AnthropicProvider("claude-sonnet-5", stream=mode).should_stream(max_tokens) is expected

    def test_auto_threshold_sits_under_the_sdk_nonstreaming_limit(self):
        # The SDK refuses non-streaming requests implying >10 min at 128k tok/h.
        sdk_limit = 128_000 * 600 / 3600
        assert ap.AUTO_STREAM_MAX_TOKENS == 21_000
        assert ap.AUTO_STREAM_MAX_TOKENS < sdk_limit

    def test_production_default_streams_and_the_sonnet_4_6_floor_does_not(self):
        # claude-sonnet-5 / high resolves to 24,000 (what Terraform sets), which is
        # above the auto threshold: production runs on the streaming transport.
        from utils.llm import DEFAULT_ANTHROPIC_MODEL

        production_cap = ap.recommended_max_tokens(DEFAULT_ANTHROPIC_MODEL, "high")
        assert production_cap == 24_000
        assert ap.should_stream("auto", production_cap) is True
        assert ap.should_stream("auto", ap.recommended_max_tokens("claude-sonnet-4-6", "high")) is False

    def test_auto_streams_at_25000_and_not_at_16000(self):
        client = _StreamingClient(_response(content=[_Block(type="text", text="ok")]))
        provider = AnthropicProvider("claude-sonnet-5", client=client, stream="auto")

        provider.create(_request(max_tokens=16_000))
        assert len(client.calls) == 1 and client.stream_calls == []

        provider.create(_request(max_tokens=25_000))
        assert len(client.calls) == 1 and len(client.stream_calls) == 1
        assert client.stream_log == ["enter", "get_final_message", "exit"]

    def test_kwargs_identical_between_modes_except_the_method(self):
        response = _response(content=[_Block(type="text", text="ok")])
        never = _StreamingClient(response)
        always = _StreamingClient(response)
        request = _request(max_tokens=25_000, model="claude-sonnet-5")

        AnthropicProvider("claude-sonnet-5", client=never, stream="never").create(request)
        AnthropicProvider("claude-sonnet-5", client=always, stream="always").create(request)

        assert never.stream_calls == [] and len(never.calls) == 1
        assert always.calls == [] and len(always.stream_calls) == 1
        create_kwargs = dict(never.calls[0])
        stream_kwargs = dict(always.stream_calls[0])
        assert _timeout_fields(create_kwargs.pop("timeout")) == _timeout_fields(
            stream_kwargs.pop("timeout")
        )
        assert create_kwargs == stream_kwargs
        assert create_kwargs["max_tokens"] == 25_000
        assert create_kwargs["thinking"] == {"type": "adaptive"}
        assert create_kwargs["output_config"] == {"effort": "high"}
        # No ``stream=True`` flag or other transport marker leaks into the body.
        assert "stream" not in create_kwargs

    def test_timeout_kwarg_applies_the_same_way_when_streaming(self):
        client = _StreamingClient(_response())
        AnthropicProvider("claude-sonnet-5", client=client, stream="always").create(
            _request(timeout_s=90.0)
        )
        timeout = client.stream_calls[0]["timeout"]
        assert timeout.read == 90.0 and timeout.connect == clients.ANTHROPIC_CONNECT_TIMEOUT_SECONDS

        client = _StreamingClient(_response())
        AnthropicProvider("claude-sonnet-5", client=client, stream="always").create(
            _request(timeout_s=None)
        )
        assert "timeout" not in client.stream_calls[0]

    def test_final_message_from_stream_maps_identically_to_create_response(self):
        response = _response(
            stop_reason="tool_use",
            content=[
                _Block(type="thinking", thinking="hmm", signature="sig"),
                _Block(type="tool_use", id="t1", name="calculate", input={"expression": "1+1"}),
            ],
            usage={
                "input_tokens": 100,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 7000,
                "output_tokens": 30,
            },
            model="claude-sonnet-5-20260301",
        )
        via_create = AnthropicProvider(
            "claude-sonnet-5", client=_StreamingClient(response), stream="never"
        ).create(_request())
        via_stream = AnthropicProvider(
            "claude-sonnet-5", client=_StreamingClient(response), stream="always"
        ).create(_request())

        # Identical except for the transport marker, which is the whole point of that field.
        assert via_create.transport == "create" and via_stream.transport == "stream"
        assert replace(via_stream, transport="") == replace(via_create, transport="")
        assert via_stream.provider == "anthropic"
        assert via_stream.model == "claude-sonnet-5-20260301"
        assert via_stream.stop_reason == "tool_use" and via_stream.raw_stop_reason == "tool_use"
        assert via_stream.content == [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expression": "1+1"}},
        ]
        assert via_stream.usage.as_dict() == {
            "input_tokens": 100,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 7000,
            "output_tokens": 30,
        }

    @pytest.mark.parametrize("raw", ["refusal", "max_tokens", "stop_sequence"])
    def test_stop_reasons_normalize_the_same_when_streaming(self, raw):
        response = _response(stop_reason=raw)
        via_stream = AnthropicProvider(
            "claude-sonnet-5", client=_StreamingClient(response), stream="always"
        ).create(_request())
        assert via_stream.stop_reason == ap.normalize_stop_reason(raw)
        assert via_stream.raw_stop_reason == raw

    def test_never_mode_works_with_a_client_that_has_no_stream_method(self):
        client = _Client(_response())  # no ``stream`` attribute at all
        AnthropicProvider("claude-sonnet-5", client=client, stream="never").create(
            _request(max_tokens=64_000)
        )
        assert len(client.calls) == 1

    def test_production_default_request_never_streams(self, monkeypatch):
        monkeypatch.delenv(ap.STREAM_ENV_VAR, raising=False)
        client = _Client(_response())  # would raise AttributeError if ``.stream`` were used
        AnthropicProvider("claude-sonnet-4-6", client=client).create(_request(max_tokens=16_000))
        assert len(client.calls) == 1


# --------------------------------------------------------------------------- #
# Incident regression: client-side block fields must never be echoed to the API
# --------------------------------------------------------------------------- #
#
# Production (claude-sonnet-5, ANTHROPIC_MAX_TOKENS=24000 -> ``messages.stream``)
# failed every thread follow-up with
#   400 invalid_request_error: messages.3.content.0.text.parsed_output:
#       Extra inputs are not permitted
# because the SDK's streaming accumulator returns ``ParsedTextBlock`` objects
# whose ``parsed_output`` (and ``citations=None``) survived ``model_dump()``,
# were appended to the assistant turn, saved, and replayed.


def _sdk_message(content, model="claude-sonnet-5", stop_reason="end_turn"):
    """A real ``anthropic.types.Message`` (what ``create`` / ``get_final_message`` return)."""
    return Message(
        id="msg_01",
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _streamed_text_block(text):
    """The exact object the streaming accumulator builds for a text block
    (``anthropic/lib/streaming/_messages.py`` -> ``construct_type(ParsedContentBlock, ...)``)."""
    from anthropic._models import construct_type

    block = construct_type(type_=ParsedContentBlock, value={"type": "text", "text": text})
    assert block.model_dump() == {"citations": None, "text": text, "type": "text", "parsed_output": None}
    return block


class TestClientSideFieldsNeverReachTheApi:
    def test_text_block_with_parsed_output_attribute_is_reduced_to_type_and_text(self):
        # ``TextBlock`` has ``extra="allow"``; this is how the SDK attaches the field.
        block = TextBlock(type="text", text="Here is the plan", parsed_output=None)
        assert block.model_dump() == {"citations": None, "text": "Here is the plan", "type": "text", "parsed_output": None}
        response = _sdk_message([block])
        client = _StreamingClient(response)

        resp = AnthropicProvider("claude-sonnet-5", client=client, stream="always").create(
            _request(max_tokens=24_000)
        )

        assert resp.content == [{"type": "text", "text": "Here is the plan"}]
        assert resp.transport == "stream"

    def test_end_to_end_streaming_at_production_max_tokens_yields_sanitized_blocks(self, caplog):
        # ``ParsedMessage`` holding a real accumulator-built text block plus a tool call,
        # returned through ``messages.stream(...).get_final_message()`` at the
        # production cap that selects the streaming transport in ``auto`` mode.
        final = ParsedMessage.construct(
            id="msg_01",
            type="message",
            role="assistant",
            model="claude-sonnet-5",
            content=[
                _streamed_text_block("Let me check the table first."),
                ToolUseBlock(type="tool_use", id="toolu_1", name="calculate", input={"expression": "1+1"}),
            ],
            stop_reason="tool_use",
            stop_sequence=None,
            usage=Usage(input_tokens=100, output_tokens=40),
        )
        client = _StreamingClient(final)
        caplog.set_level(logging.INFO, logger=ap.__name__)

        resp = AnthropicProvider("claude-sonnet-5", client=client, stream="auto").create(
            _request(max_tokens=24_000, model="claude-sonnet-5")
        )

        assert client.calls == [] and len(client.stream_calls) == 1
        assert client.stream_calls[0]["max_tokens"] == 24_000
        assert resp.content == [
            {"type": "text", "text": "Let me check the table first."},
            {"type": "tool_use", "id": "toolu_1", "name": "calculate", "input": {"expression": "1+1"}},
        ]
        for block in resp.content:
            assert "parsed_output" not in block and "citations" not in block
        assert resp.stop_reason == "tool_use"
        assert resp.transport == "stream"
        # One INFO line naming the transport, model and cap.
        assert "messages.stream" in caplog.text
        assert "claude-sonnet-5" in caplog.text and "24000" in caplog.text

    def test_create_transport_is_reported_and_logged(self, caplog):
        client = _Client(_sdk_message([TextBlock(type="text", text="ok")]))
        caplog.set_level(logging.INFO, logger=ap.__name__)

        resp = AnthropicProvider("claude-sonnet-4-6", client=client).create(_request(max_tokens=16_000))

        assert resp.transport == "create"
        assert resp.content == [{"type": "text", "text": "ok"}]
        assert "messages.create" in caplog.text and "16000" in caplog.text

    def test_poisoned_history_is_sanitized_before_sending_without_mutating_input(self):
        # Defense in depth: a session saved before the fix still replays cleanly.
        client = _Client(_response())
        messages = [
            {"role": "user", "content": "what is the count?"},
            {
                "role": "assistant",
                "content": [
                    {"citations": None, "text": "Checking.", "type": "text", "parsed_output": None},
                    {"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expression": "1"}, "caller": None},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is 1.", "citations": None, "parsed_output": None}]},
            {"role": "user", "content": "and now?"},
        ]
        snapshot = deepcopy(messages)

        AnthropicProvider("claude-sonnet-5", client=client).create(_request(messages=messages))

        sent = client.calls[0]["messages"]
        assert sent[1]["content"] == [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expression": "1"}},
        ]
        assert sent[3]["content"] == [{"type": "text", "text": "It is 1."}]
        assert sent[0] is messages[0] and sent[2] is messages[2]
        assert sent[4] == {"role": "user", "content": [{"type": "text", "text": "and now?", "cache_control": CACHE}]}
        assert messages == snapshot
        for message in sent:
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    assert "parsed_output" not in block

    def test_sanitizer_composes_with_foreign_block_stripping(self):
        client = _Client(_response())
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    provider_reasoning_block("openai", {"id": "rs_1"}),
                    {"type": "text", "text": "a", "parsed_output": None},
                ],
            },
            {"role": "user", "content": "next"},
        ]
        AnthropicProvider("claude-sonnet-5", client=client).create(_request(messages=messages))
        assert client.calls[0]["messages"][1]["content"] == [{"type": "text", "text": "a"}]

    def test_thinking_and_tool_use_blocks_keep_exactly_their_allowed_keys(self):
        # Streamed thinking / tool_use blocks are the SDK's own classes; a bare
        # dump of a stand-in with stray attributes must still come out exact.
        response = _response(
            stop_reason="tool_use",
            content=[
                _Block(type="thinking", thinking="hmm", signature="sig", parsed_output=None),
                _Block(type="tool_use", id="t1", name="calculate", input={"expression": "1+1"}, caller=None),
                _Block(type="redacted_thinking", data="opaque", parsed_output=None),
            ],
        )
        resp = AnthropicProvider("claude-sonnet-5", client=_StreamingClient(response), stream="always").create(
            _request(max_tokens=24_000)
        )
        assert resp.content == [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expression": "1+1"}},
            {"type": "redacted_thinking", "data": "opaque"},
        ]

    def test_citations_survive_when_present(self):
        citation = {"type": "char_location", "cited_text": "x", "document_index": 0, "document_title": None, "start_char_index": 0, "end_char_index": 1}
        blocks = ap.content_to_blocks([{"type": "text", "text": "t", "citations": [citation], "parsed_output": None}])
        assert blocks == [{"type": "text", "text": "t", "citations": [citation]}]


# --------------------------------------------------------------------------- #
# Prompt caching: the moving breakpoint on the conversation tail
# --------------------------------------------------------------------------- #
#
# Production (2026-09-08, Sonnet 5, 8 calls) showed ``cache_read_input_tokens``
# constant at the system-prompt size and ``cache_creation_input_tokens`` 0 on
# every call: only the system block had a breakpoint, so nothing after it was
# cache-eligible and the growing conversation was re-sent uncached each
# iteration. The provider now marks the last block of the last message too.


def _tool_result(tool_use_id, content, **extra):
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, **extra}


class TestConversationCacheBreakpoint:
    def test_string_user_text_becomes_a_marked_text_block(self):
        out = ap.with_cache_breakpoint([{"role": "user", "content": "hi"}])
        assert out == [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": CACHE}]}]

    def test_marker_lands_on_the_final_tool_result_block(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "calculate", "input": {}}, {"type": "tool_use", "id": "t2", "name": "calculate", "input": {}}]},
            {"role": "user", "content": [_tool_result("t1", "1", is_error=False), _tool_result("t2", "2", is_error=True)]},
        ]
        out = ap.with_cache_breakpoint(messages)
        assert out[-1]["content"] == [
            _tool_result("t1", "1", is_error=False),
            _tool_result("t2", "2", is_error=True, cache_control=CACHE),
        ]
        assert out[0] is messages[0] and out[1] is messages[1]
        assert "cache_control" not in messages[-1]["content"][1]

    def test_marker_lands_on_a_trailing_image_block(self):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
        messages = [{"role": "user", "content": [{"type": "text", "text": "what is this"}, image]}]
        out = ap.with_cache_breakpoint(messages)
        assert out[0]["content"] == [{"type": "text", "text": "what is this"}, {**image, "cache_control": CACHE}]
        assert "cache_control" not in image

    def test_text_last_block_in_a_list_is_marked(self):
        out = ap.with_cache_breakpoint([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
        assert out[0]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": CACHE}]

    def test_ineligible_trailing_block_walks_back_to_the_nearest_cacheable_one(self):
        # ``thinking`` / ``redacted_thinking`` cannot carry cache_control.
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "done"},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            },
        ]
        out = ap.with_cache_breakpoint(messages)
        assert out[-1]["content"] == [
            {"type": "text", "text": "done", "cache_control": CACHE},
            {"type": "redacted_thinking", "data": "opaque"},
        ]

    def test_no_cacheable_block_or_empty_history_is_left_alone(self):
        only_thinking = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "s"}]}]
        assert ap.with_cache_breakpoint(only_thinking) is only_thinking
        empty_content = [{"role": "user", "content": []}]
        assert ap.with_cache_breakpoint(empty_content) is empty_content
        assert ap.with_cache_breakpoint([]) == []

    def test_cacheable_block_types_match_the_api_and_exclude_thinking(self):
        assert ap.CACHEABLE_BLOCK_TYPES == {"text", "image", "tool_use", "tool_result", "document"}
        assert "thinking" not in ap.CACHEABLE_BLOCK_TYPES
        assert "redacted_thinking" not in ap.CACHEABLE_BLOCK_TYPES

    def test_request_carries_exactly_two_breakpoints_well_under_the_limit(self):
        client = _Client(_response())
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}, {"type": "tool_use", "id": "t1", "name": "calculate", "input": {}}]},
            {"role": "user", "content": [_tool_result("t1", "1")]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "t2", "signature": "s2"}, {"type": "tool_use", "id": "t2", "name": "calculate", "input": {}}]},
            {"role": "user", "content": [_tool_result("t2", "2")]},
        ]
        AnthropicProvider("claude-sonnet-5", client=client).create(_request(messages=messages))

        (kwargs,) = client.calls
        assert ap.count_cache_breakpoints(kwargs) == 2 <= ap.MAX_CACHE_BREAKPOINTS
        assert "cache_control" in kwargs["system"][0]
        assert all("cache_control" not in tool for tool in kwargs["tools"])  # tools ride on the system marker
        marked = [
            (i, j)
            for i, message in enumerate(kwargs["messages"])
            for j, block in enumerate(message["content"])
            if "cache_control" in block
        ]
        assert marked == [(4, 0)]
        # Retained thinking blocks are sent as-is (Sonnet 4.6+/5 keep prior-turn thinking server-side).
        assert kwargs["messages"][1]["content"][0] == {"type": "thinking", "thinking": "t", "signature": "s"}

    def test_moving_marker_between_iterations_leaves_the_overlap_byte_identical(self):
        # The previous request's messages must reappear unchanged as a prefix
        # of the next one (cache_control markers excluded), or the cache misses.
        def strip(messages):
            return [
                {**m, "content": [{k: v for k, v in b.items() if k != "cache_control"} for b in m["content"]]}
                for m in messages
            ]

        history = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}, {"type": "tool_use", "id": "t1", "name": "calculate", "input": {}}]},
            {"role": "user", "content": [_tool_result("t1", "1")]},
        ]
        first = ap.with_cache_breakpoint(history)
        history = history + [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "calculate", "input": {}}]},
            {"role": "user", "content": [_tool_result("t2", "2")]},
        ]
        second = ap.with_cache_breakpoint(history)

        assert strip(second[: len(first)]) == strip(first)
        assert "cache_control" in first[2]["content"][0]
        assert "cache_control" not in second[2]["content"][0]
        assert "cache_control" in second[4]["content"][0]

    def test_input_history_never_acquires_cache_control(self):
        client = _Client(_response())
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "calculate", "input": {}}]},
            {"role": "user", "content": [_tool_result("t1", "1")]},
        ]
        snapshot = deepcopy(messages)

        AnthropicProvider("claude-sonnet-5", client=client).create(_request(messages=messages))

        assert messages == snapshot
        assert "cache_control" not in repr(messages)

    def test_system_marker_stays_on_the_five_minute_ttl(self):
        # ~2 conversations/day: the 1-hour TTL (2x write) would rarely be read
        # back; within a run every iteration refreshes the 5-minute entry.
        assert ap.EPHEMERAL_CACHE_CONTROL == {"type": "ephemeral"}
        client = _Client(_response())
        AnthropicProvider("claude-sonnet-5", client=client).create(_request())
        assert client.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert client.calls[0]["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
