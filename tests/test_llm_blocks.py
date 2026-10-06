"""``utils.llm.blocks``: the per-type allowlist that keeps client-side SDK fields
(``parsed_output``, ``citations: None``) out of replayed assistant turns."""

import logging

import pytest

from utils.llm import blocks
from utils.llm.blocks import (
    sanitize_assistant_messages,
    sanitize_content_block,
    sanitize_content_blocks,
)
from utils.llm.types import provider_reasoning_block


class TestTextBlocks:
    def test_streamed_text_block_dump_is_reduced_to_type_and_text(self):
        # Exactly what ``ParsedTextBlock.model_dump()`` yields after streaming.
        raw = {"citations": None, "text": "hello", "type": "text", "parsed_output": None}
        assert sanitize_content_block(raw) == {"type": "text", "text": "hello"}

    def test_non_null_parsed_output_is_dropped_too(self):
        raw = {"type": "text", "text": "{}", "parsed_output": {"a": 1}}
        assert sanitize_content_block(raw) == {"type": "text", "text": "{}"}

    def test_empty_citations_list_is_dropped(self):
        assert sanitize_content_block({"type": "text", "text": "t", "citations": []}) == {
            "type": "text",
            "text": "t",
        }

    def test_non_empty_citations_are_kept(self):
        citation = {"type": "char_location", "cited_text": "x", "document_index": 0}
        raw = {"type": "text", "text": "t", "citations": [citation]}
        assert sanitize_content_block(raw) == raw

    def test_input_is_not_mutated(self):
        raw = {"type": "text", "text": "t", "parsed_output": None}
        sanitize_content_block(raw)
        assert raw == {"type": "text", "text": "t", "parsed_output": None}


class TestOtherKnownBlocks:
    def test_tool_use_keeps_exactly_its_allowed_keys(self):
        raw = {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "calculate",
            "input": {"expression": "1+1"},
            "caller": None,
            "parsed_output": None,
        }
        assert sanitize_content_block(raw) == {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "calculate",
            "input": {"expression": "1+1"},
        }

    def test_thinking_keeps_exactly_its_allowed_keys(self):
        raw = {"type": "thinking", "thinking": "hmm", "signature": "sig", "extra": None}
        assert sanitize_content_block(raw) == {"type": "thinking", "thinking": "hmm", "signature": "sig"}

    def test_thinking_signature_is_kept_even_when_empty(self):
        # The API requires ``signature``; an empty string is the model's business.
        raw = {"type": "thinking", "thinking": "", "signature": ""}
        assert sanitize_content_block(raw) == raw

    def test_redacted_thinking(self):
        raw = {"type": "redacted_thinking", "data": "opaque", "parsed_output": None}
        assert sanitize_content_block(raw) == {"type": "redacted_thinking", "data": "opaque"}

    def test_server_tool_use(self):
        raw = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}, "x": 1}
        assert sanitize_content_block(raw) == {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "q"},
        }

    @pytest.mark.parametrize(
        "block_type",
        ["web_search_tool_result", "web_fetch_tool_result", "code_execution_tool_result", "tool_result"],
    )
    def test_tool_result_blocks_keep_type_tool_use_id_and_content(self, block_type):
        raw = {"type": block_type, "tool_use_id": "srvtoolu_1", "content": [{"type": "x"}], "junk": None}
        assert sanitize_content_block(raw) == {
            "type": block_type,
            "tool_use_id": "srvtoolu_1",
            "content": [{"type": "x"}],
        }

    def test_missing_allowed_keys_are_not_fabricated(self):
        assert sanitize_content_block({"type": "tool_use", "id": "t"}) == {"type": "tool_use", "id": "t"}


class TestUnknownAndInternalBlocks:
    def test_unknown_type_keeps_non_null_keys_and_warns(self, caplog):
        caplog.set_level(logging.WARNING, logger=blocks.__name__)
        raw = {"type": "mystery", "keep": 1, "drop": None}
        assert sanitize_content_block(raw) == {"type": "mystery", "keep": 1}
        assert "mystery" in caplog.text

    def test_provider_reasoning_passes_through_without_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=blocks.__name__)
        raw = provider_reasoning_block("openai", {"type": "reasoning", "id": "rs_1", "summary": None})
        assert sanitize_content_block(raw) == raw
        assert sanitize_content_block(raw) is not raw
        assert caplog.text == ""

    def test_block_without_type_is_copied_as_is(self):
        raw = {"text": "no type"}
        assert sanitize_content_block(raw) == raw


class TestMessageLevel:
    def test_unchanged_history_returns_the_same_list_object(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "s"}, {"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]},
        ]
        assert sanitize_assistant_messages(messages) is messages

    def test_only_assistant_turns_are_touched_and_input_is_not_mutated(self):
        user_turn = {"role": "user", "content": [{"type": "text", "text": "q", "parsed_output": None}]}
        poisoned = {"role": "assistant", "content": [{"type": "text", "text": "a", "parsed_output": None, "citations": None}]}
        messages = [user_turn, poisoned]

        result = sanitize_assistant_messages(messages)

        assert result is not messages
        assert result[0] is user_turn  # user turns are never rewritten
        assert result[1] == {"role": "assistant", "content": [{"type": "text", "text": "a"}]}
        assert poisoned["content"][0]["parsed_output"] is None  # original intact

    def test_string_assistant_content_is_left_alone(self):
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "plain"}]
        assert sanitize_assistant_messages(messages) is messages
        assert sanitize_content_blocks("plain") == "plain"
