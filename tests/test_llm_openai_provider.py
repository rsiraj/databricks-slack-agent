"""OpenAI Responses translation: both directions, stop reasons, usage, effort, budget."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from utils.llm import openai_provider as op
from utils.llm.openai_provider import OpenAIProvider
from utils.llm.types import ModelRequest, ToolSpec, provider_reasoning_block

TOOLS = [
    ToolSpec("calculate", "Deterministic arithmetic.", {"type": "object", "properties": {"expression": {"type": "string"}}}),
]

REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_123",
    "summary": [],
    "encrypted_content": "gAAAAB-opaque",
}


def _request(**overrides):
    values = dict(
        system="SYSTEM PROMPT",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        effort="high",
        max_tokens=16_000,
        timeout_s=120.0,
    )
    values.update(overrides)
    return ModelRequest(**values)


class TestMessageTranslation:
    def test_string_user_and_assistant_turns(self):
        items = op.translate_messages(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
        )
        assert items == [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]},
        ]

    def test_user_text_and_image_blocks_become_one_message(self):
        items = op.translate_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
                        },
                        {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                    ],
                }
            ]
        )
        assert items == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this"},
                    {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
                    {"type": "input_image", "image_url": "https://x/y.png"},
                ],
            }
        ]

    def test_tool_use_and_tool_result_become_function_call_items(self):
        items = op.translate_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "call_1", "name": "calculate", "input": {"expression": "1+1"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_1", "content": '{"result": 2}', "is_error": False}
                    ],
                },
            ]
        )
        assert items == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]},
            {"type": "function_call", "call_id": "call_1", "name": "calculate", "arguments": '{"expression": "1+1"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"result": 2}'},
        ]

    def test_tool_result_error_is_wrapped_unless_already_an_error_object(self):
        wrapped = op.tool_result_output(
            {"type": "tool_result", "tool_use_id": "c", "content": "boom", "is_error": True}
        )
        assert json.loads(wrapped) == {"error": "boom"}

        already = op.tool_result_output(
            {"type": "tool_result", "tool_use_id": "c", "content": '{"error": "x", "status_code": 500}', "is_error": True}
        )
        assert already == '{"error": "x", "status_code": 500}'

        plain = op.tool_result_output({"type": "tool_result", "tool_use_id": "c", "content": "fine"})
        assert plain == "fine"

    def test_tool_result_list_content_is_flattened(self):
        text = op.tool_result_output(
            {
                "type": "tool_result",
                "tool_use_id": "c",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        )
        assert text == "a\nb"

    def test_user_text_after_tool_result_keeps_order(self):
        items = op.translate_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": "r1"},
                        {"type": "text", "text": "also this"},
                    ],
                }
            ]
        )
        assert [item.get("type") or item["role"] for item in items] == ["function_call_output", "user"]

    def test_openai_reasoning_is_replayed_verbatim_and_anthropic_thinking_dropped(self):
        items = op.translate_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private", "signature": "s"},
                        {"type": "redacted_thinking", "data": "xxx"},
                        provider_reasoning_block("openai", REASONING_ITEM),
                        provider_reasoning_block("someone-else", {"type": "reasoning", "id": "no"}),
                        {"type": "text", "text": "done"},
                    ],
                }
            ]
        )
        assert items == [
            REASONING_ITEM,
            {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ]
        assert items[0] is not REASONING_ITEM  # copied, not aliased

    def test_translation_does_not_mutate_input(self):
        messages = [
            {"role": "assistant", "content": [provider_reasoning_block("openai", REASONING_ITEM), {"type": "text", "text": "t"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "x", "is_error": True}]},
        ]
        snapshot = deepcopy(messages)
        op.translate_messages(messages)
        assert messages == snapshot

    def test_anthropic_cache_control_markers_never_reach_the_responses_input(self):
        # The Anthropic provider marks the last block of the last message; the
        # Responses API has automatic caching and no such field. Translation
        # rebuilds every item from the block's own fields, so the marker is dropped.
        cache = {"cache_control": {"type": "ephemeral"}}
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q", **cache}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "calculate", "input": {"expression": "1"}, **cache}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "1", **cache}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}, **cache},
                    {"type": "text", "text": "and this?", **cache},
                ],
            },
        ]
        kwargs = op.build_request(_request(messages=messages), "gpt-5.6-terra")
        assert "cache_control" not in json.dumps(kwargs)
        assert kwargs["input"][0] == {"role": "user", "content": [{"type": "input_text", "text": "q"}]}
        assert kwargs["input"][2]["type"] == "function_call_output"


class TestToolsAndRequest:
    def test_tools_become_function_tools(self):
        assert op.translate_tools(TOOLS) == [
            {
                "type": "function",
                "name": "calculate",
                "description": "Deterministic arithmetic.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
            }
        ]

    def test_full_request_shape(self, caplog):
        caplog.set_level("INFO")
        kwargs = op.build_request(_request(), "gpt-5.6-terra")
        assert kwargs == {
            "model": "gpt-5.6-terra",
            "instructions": "SYSTEM PROMPT",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "tools": op.translate_tools(TOOLS),
            "reasoning": {"effort": "high"},
            "max_output_tokens": 25_000,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "timeout": 120.0,
        }
        assert "Raising max_output_tokens from 16000 to 25000" in caplog.text

    def test_max_output_tokens_not_lowered_and_timeout_optional(self):
        kwargs = op.build_request(_request(max_tokens=40_000, timeout_s=None), "gpt-5.6-terra")
        assert kwargs["max_output_tokens"] == 40_000
        assert "timeout" not in kwargs

    def test_request_model_override_wins(self):
        assert op.build_request(_request(model="gpt-5.6-mini"), "gpt-5.6-terra")["model"] == "gpt-5.6-mini"

    @pytest.mark.parametrize("level", ["none", "low", "medium", "high", "xhigh", "max"])
    def test_effort_levels_pass_through(self, level):
        provider = OpenAIProvider()
        assert provider.supports_effort(level)
        assert provider.map_effort(level) == level

    def test_unknown_effort_rejected(self):
        with pytest.raises(ValueError, match="not supported by OpenAI"):
            op.map_effort("turbo")


class TestResponseTranslation:
    def _resp(self, output, status="completed", incomplete_reason=None, usage=None, model="gpt-5.6-terra-2026-08"):
        return {
            "model": model,
            "status": status,
            "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
            "output": output,
            "usage": usage,
        }

    def test_message_output_becomes_text_end_turn(self):
        resp = op.translate_response(
            self._resp([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]}]),
            "gpt-5.6-terra",
        )
        assert resp.provider == "openai"
        assert resp.model == "gpt-5.6-terra-2026-08"
        assert resp.stop_reason == "end_turn"
        assert resp.raw_stop_reason == "completed"
        assert resp.content == [{"type": "text", "text": "Hello"}]

    def test_function_call_and_reasoning_items(self):
        resp = op.translate_response(
            self._resp(
                [
                    REASONING_ITEM,
                    {
                        "type": "function_call",
                        "id": "fc_9",
                        "call_id": "call_9",
                        "name": "calculate",
                        "arguments": '{"expression": "2*3"}',
                        "status": "completed",
                    },
                ]
            ),
            "gpt-5.6-terra",
        )
        assert resp.stop_reason == "tool_use"
        assert resp.content == [
            provider_reasoning_block("openai", REASONING_ITEM),
            {"type": "tool_use", "id": "call_9", "name": "calculate", "input": {"expression": "2*3"}},
        ]

    def test_pydantic_like_items_are_dumped(self):
        class Item(SimpleNamespace):
            def model_dump(self, exclude_none=False):
                return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}

        output = [
            Item(type="function_call", call_id="c", name="calculate", arguments="{}", id=None),
        ]
        resp = op.translate_response(SimpleNamespace(output=output, status="completed", usage=None, model="m", incomplete_details=None), "m")
        assert resp.content == [{"type": "tool_use", "id": "c", "name": "calculate", "input": {}}]

    def test_unparseable_arguments_become_empty_input(self, caplog):
        caplog.set_level("WARNING")
        resp = op.translate_response(
            self._resp([{"type": "function_call", "call_id": "c", "name": "calculate", "arguments": "{not json"}]),
            "m",
        )
        assert resp.content[0]["input"] == {}
        assert "Unparseable" in caplog.text

    def test_truncation_maps_to_max_tokens_unless_a_function_call_exists(self):
        truncated = op.translate_response(
            self._resp(
                [{"type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
                status="incomplete",
                incomplete_reason="max_output_tokens",
            ),
            "m",
        )
        assert truncated.stop_reason == "max_tokens"
        assert truncated.raw_stop_reason == "incomplete:max_output_tokens"

        with_call = op.translate_response(
            self._resp(
                [{"type": "function_call", "call_id": "c", "name": "calculate", "arguments": "{}"}],
                status="incomplete",
                incomplete_reason="max_output_tokens",
            ),
            "m",
        )
        assert with_call.stop_reason == "tool_use"

    def test_refusal_part(self):
        resp = op.translate_response(
            self._resp([{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}]),
            "m",
        )
        assert resp.stop_reason == "refusal"
        assert resp.content == [{"type": "text", "text": "No."}]

    def test_other_incomplete_reason_and_unknown_status_are_other(self):
        assert op.map_stop_reason("incomplete", "content_filter", saw_function_call=False, saw_refusal=False) == (
            "other",
            "incomplete:content_filter",
        )
        assert op.map_stop_reason("failed", None, saw_function_call=False, saw_refusal=False) == ("other", "failed")
        assert op.map_stop_reason(None, None, saw_function_call=False, saw_refusal=False) == ("end_turn", "completed")

    def test_usage_subtracts_cached_tokens_from_input(self):
        usage = op.translate_usage(
            {"input_tokens": 12_000, "input_tokens_details": {"cached_tokens": 10_000}, "output_tokens": 800}
        )
        assert usage.as_dict() == {
            "input_tokens": 2_000,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10_000,
            "output_tokens": 800,
        }

    def test_usage_from_objects_and_missing_details(self):
        usage = op.translate_usage(SimpleNamespace(input_tokens=50, input_tokens_details=None, output_tokens=5))
        assert usage.as_dict()["input_tokens"] == 50
        assert usage.as_dict()["cache_read_input_tokens"] == 0
        assert op.translate_usage(None).as_dict()["output_tokens"] == 0

    def test_unknown_output_item_types_are_dropped(self):
        resp = op.translate_response(self._resp([{"type": "web_search_call", "id": "ws"}]), "m")
        assert resp.content == []
        assert resp.stop_reason == "end_turn"


class TestProviderRoundTrip:
    def test_create_uses_injected_client_and_replays_reasoning_next_turn(self):
        class FakeResponses:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return {
                    "model": "gpt-5.6-terra",
                    "status": "completed",
                    "output": [
                        REASONING_ITEM,
                        {"type": "function_call", "call_id": "call_1", "name": "calculate", "arguments": '{"expression": "1+1"}'},
                    ],
                    "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 4},
                }

        client = SimpleNamespace(responses=FakeResponses())
        provider = OpenAIProvider("gpt-5.6-terra", client=client)
        messages = [{"role": "user", "content": "add"}]

        first = provider.create(_request(messages=messages))
        messages.append({"role": "assistant", "content": first.content})
        messages.append(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"result": 2}'}]}
        )
        provider.create(_request(messages=messages))

        second_input = client.responses.calls[1]["input"]
        assert second_input == [
            {"role": "user", "content": [{"type": "input_text", "text": "add"}]},
            REASONING_ITEM,
            {"type": "function_call", "call_id": "call_1", "name": "calculate", "arguments": '{"expression": "1+1"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"result": 2}'},
        ]
        assert client.responses.calls[0]["store"] is False
