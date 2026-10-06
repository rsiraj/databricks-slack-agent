"""Session sanitization — utils/sessions.py.

These guard the self-healing logic that recovers a thread whose first message
is an orphan ``tool_result`` (a state the older trim logic could leave behind,
and the exact failure mode that wedged a production thread). The Anthropic API
rejects a first message that isn't a ``tool_result``-free user message, so a
regression here would crash the agent on the next turn of any affected thread.

Ported from the former ``scripts/verify_session_fix.py`` so the cases run in CI
instead of relying on someone remembering to run the script by hand.
"""

import json

import pytest
from botocore.exceptions import ClientError

from utils import sessions
from utils.sessions import (
    MAX_ITEM_BYTES,
    SESSION_RETENTION_DAYS,
    TOOL_RESULT_TRUNCATE_CHARS,
    TTL_SECONDS,
    _is_invalid_first_message,
    _strip_invalid_leading_messages,
    load_session,
    save_session,
)


def test_session_retention_is_one_year():
    assert SESSION_RETENTION_DAYS == 365
    assert TTL_SECONDS == 365 * 24 * 3600


class TestIsInvalidFirstMessage:
    def test_text_user_first_is_valid(self):
        assert _is_invalid_first_message({"role": "user", "content": "hello"}) is False

    def test_user_with_text_blocks_is_valid(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        assert _is_invalid_first_message(msg) is False

    def test_assistant_first_is_invalid(self):
        assert _is_invalid_first_message({"role": "assistant", "content": "hi"}) is True

    def test_orphan_tool_result_is_invalid(self):
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "..."}],
        }
        assert _is_invalid_first_message(msg) is True

    def test_mixed_text_and_tool_result_is_invalid(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "follow-up"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "..."},
            ],
        }
        assert _is_invalid_first_message(msg) is True


class TestStripInvalidLeadingMessages:
    def test_healthy_session_untouched(self):
        msgs = [
            {"role": "user", "content": "what's the weather?"},
            {"role": "assistant", "content": "Sunny."},
        ]
        assert len(_strip_invalid_leading_messages(list(msgs))) == 2

    def test_orphan_tool_result_at_start(self):
        # Trim dropped [user_text, assistant_with_tool_use], leaving the
        # tool_result-bearing user message orphaned at index 0.
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "..."}]},
            {"role": "assistant", "content": "Here are the results..."},
            {"role": "user", "content": "great, what about X?"},
            {"role": "assistant", "content": "Let me check."},
        ]
        out = _strip_invalid_leading_messages(list(msgs))
        assert len(out) == 2
        assert out[0]["content"] == "great, what about X?"

    def test_two_orphan_cycles_in_a_row(self):
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "..."}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "..."}]},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "ok thanks"},
        ]
        out = _strip_invalid_leading_messages(list(msgs))
        assert len(out) == 1
        assert out[0]["content"] == "ok thanks"

    def test_assistant_first_is_stripped(self):
        msgs = [
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Hi"},
        ]
        out = _strip_invalid_leading_messages(list(msgs))
        assert len(out) == 1
        assert out[0]["role"] == "user"

    def test_all_invalid_returns_empty(self):
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "..."}]},
        ]
        assert _strip_invalid_leading_messages(list(msgs)) == []

    def test_mixed_text_and_tool_result_at_start(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "follow-up"},
                    {"type": "tool_result", "tool_use_id": "x", "content": "..."},
                ],
            },
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        out = _strip_invalid_leading_messages(list(msgs))
        assert len(out) == 1
        assert out[0]["content"] == "next"


# --------------------------------------------------------------------------- #
# save_session / load_session against a stub table
# --------------------------------------------------------------------------- #


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "PutItem")


class _StubTable:
    """Minimal DynamoDB Table stand-in: one item, records every put and its kwargs."""

    def __init__(self, item=None, put_failures=()):
        self.item = item
        self.puts = []
        self.get_calls = 0
        self.put_failures = list(put_failures)

    def get_item(self, Key, ConsistentRead=False):
        assert ConsistentRead is True
        self.get_calls += 1
        return {"Item": dict(self.item)} if self.item else {}

    def put_item(self, Item, **kwargs):
        self.puts.append((Item, kwargs))
        if self.put_failures:
            failure = self.put_failures.pop(0)
            if failure is not None:
                raise failure
        self.item = Item

    def stored_messages(self):
        return json.loads(self.item["messages"])


@pytest.fixture
def stub(monkeypatch):
    table = _StubTable()
    audits = []
    monkeypatch.setattr(sessions, "_table", table)
    monkeypatch.setattr(sessions, "audit", lambda event, **fields: audits.append((event, fields)))
    sessions._loaded_sessions.clear()
    yield table, audits
    sessions._loaded_sessions.clear()


def _events(audits, name):
    return [fields for event, fields in audits if event == name]


def _image_block(size_bytes, media_type="image/png"):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": "A" * size_bytes},
    }


def _tool_loop(index, payload_chars):
    """user question -> assistant tool_use -> user tool_result -> assistant answer."""
    return [
        {"role": "user", "content": [{"type": "text", "text": f"question {index}"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"t{index}", "name": "databricks_execute_sql", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"t{index}", "content": "r" * payload_chars}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {index}"}]},
    ]


class TestSaveSessionSizeHandling:
    def test_tiny_session_is_stored_byte_for_byte(self, stub):
        table, audits = stub
        messages = [
            {"role": "user", "content": "what's the weather?"},
            {"role": "assistant", "content": [{"type": "text", "text": "Sunny."}]},
        ]

        save_session("thread-1", messages)

        item, _ = table.puts[0]
        assert item["messages"] == json.dumps(messages, default=str)
        assert item["thread_ts"] == "thread-1"
        assert len(item["version"]) == 32
        assert item["ttl"] == item["updated_at"] + TTL_SECONDS
        assert audits == []

    def test_oversized_image_becomes_placeholder_and_session_saves(self, stub):
        table, audits = stub
        image = _image_block(400_000)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "what is this chart?"}, image]},
            {"role": "assistant", "content": [{"type": "text", "text": "A revenue chart."}]},
        ]
        original = json.dumps(messages, default=str)

        save_session("thread-1", messages)

        assert len(table.puts) == 1
        stored = table.stored_messages()
        assert len(stored) == 2
        assert stored[0]["content"][0] == {"type": "text", "text": "what is this chart?"}
        placeholder = stored[0]["content"][1]
        assert placeholder["type"] == "text"
        assert placeholder["text"].startswith("[Image attachment (image/png, 293 KB) was shared here;")
        assert placeholder["text"].endswith("it is no longer available in thread memory.]")
        assert stored[1] == messages[1]
        assert len(table.item["messages"].encode("utf-8")) <= MAX_ITEM_BYTES
        # The caller's list is untouched (the model still sees the image this turn).
        assert json.dumps(messages, default=str) == original
        trimmed = _events(audits, "session_trimmed")
        assert len(trimmed) == 1
        assert trimmed[0]["images_replaced"] == 1
        assert trimmed[0]["trimmed_messages"] == 0
        assert trimmed[0]["tool_results_truncated"] == 0
        assert trimmed[0]["bytes"] <= MAX_ITEM_BYTES
        assert _events(audits, "session_save_failed") == []

    def test_images_are_replaced_oldest_first_and_only_as_needed(self, stub):
        table, audits = stub
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "first"}, _image_block(200_000)]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "second"}, _image_block(200_000, "image/jpeg")]},
            {"role": "assistant", "content": "ok again"},
        ]

        save_session("thread-1", messages)

        stored = table.stored_messages()
        assert stored[0]["content"][1]["type"] == "text"  # oldest replaced
        assert stored[2]["content"][1]["type"] == "image"  # newest kept
        assert _events(audits, "session_trimmed")[0]["images_replaced"] == 1

    def test_tool_loop_trims_leading_messages_one_at_a_time(self, stub):
        table, audits = stub
        messages = _tool_loop(0, 150_000) + _tool_loop(1, 150_000) + _tool_loop(2, 100_000)
        assert len(json.dumps(messages).encode("utf-8")) > MAX_ITEM_BYTES

        save_session("thread-1", messages)

        stored = table.stored_messages()
        # Dropping only the first user message orphans loop 0's tool_use /
        # tool_result / answer, which are stripped; loop 1 becomes the head.
        assert stored == messages[4:]
        assert _is_invalid_first_message(stored[0]) is False
        assert stored[0]["content"][0]["text"] == "question 1"
        trimmed = _events(audits, "session_trimmed")[0]
        assert trimmed["trimmed_messages"] == 4
        assert trimmed["images_replaced"] == 0
        assert trimmed["tool_results_truncated"] == 0

    def test_text_only_history_drops_one_exchange_not_everything(self, stub):
        table, audits = stub
        messages = []
        for index in range(3):
            messages.append({"role": "user", "content": f"q{index}"})
            messages.append({"role": "assistant", "content": "a" * 120_000})

        save_session("thread-1", messages)

        assert table.stored_messages() == messages[2:]
        assert _events(audits, "session_trimmed")[0]["trimmed_messages"] == 2

    def test_never_trims_below_final_exchange_and_truncates_tool_results_instead(self, stub):
        table, audits = stub
        # A single tool loop that is itself over the cap. The old pair-dropping
        # loop cascaded this to [] and saved an empty session.
        messages = _tool_loop(0, 400_000)

        save_session("thread-1", messages)

        stored = table.stored_messages()
        assert len(stored) == 4
        assert _is_invalid_first_message(stored[0]) is False
        tool_result = stored[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["content"].endswith("…[truncated for storage]")
        assert len(tool_result["content"]) == TOOL_RESULT_TRUNCATE_CHARS + len("…[truncated for storage]")
        assert stored[3] == messages[3]
        trimmed = _events(audits, "session_trimmed")[0]
        assert trimmed["trimmed_messages"] == 0
        assert trimmed["tool_results_truncated"] == 1
        assert _events(audits, "session_save_failed") == []
        # The caller's list still holds the full payload.
        assert len(messages[2]["content"][0]["content"]) == 400_000

    def test_tool_results_truncate_oldest_first(self, stub):
        table, audits = stub
        # Two loops; the final exchange (loop 1) cannot be dropped, and dropping
        # loop 0 alone still leaves loop 1 over the cap, so truncation kicks in
        # oldest-first and stops as soon as the item fits.
        messages = _tool_loop(0, 100_000) + _tool_loop(1, 400_000)

        save_session("thread-1", messages)

        stored = table.stored_messages()
        assert stored == [
            *messages[4:6],
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "r" * TOOL_RESULT_TRUNCATE_CHARS + "…[truncated for storage]",
                    }
                ],
            },
            messages[7],
        ]
        trimmed = _events(audits, "session_trimmed")[0]
        assert trimmed["trimmed_messages"] == 4
        assert trimmed["tool_results_truncated"] == 1

    def test_lone_final_user_message_is_kept(self, stub):
        table, audits = stub
        messages = [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "a" * 200_000},
            {"role": "user", "content": "u" * 200_000},
        ]

        save_session("thread-1", messages)

        assert table.stored_messages() == [messages[2]]
        assert _events(audits, "session_trimmed")[0]["trimmed_messages"] == 2

    def test_still_oversize_is_audited_and_put_is_attempted(self, stub, caplog):
        table, audits = stub
        table.put_failures = [_client_error("ValidationException")]
        messages = [
            {"role": "user", "content": "x" * 400_000},
            {"role": "assistant", "content": "short"},
        ]
        caplog.set_level("ERROR")

        save_session("thread-1", messages)  # must not raise

        assert len(table.puts) == 1
        failed = _events(audits, "session_save_failed")
        assert len(failed) == 1
        assert failed[0]["reason"] == "oversize"
        assert failed[0]["bytes"] > MAX_ITEM_BYTES
        assert _events(audits, "session_trimmed") == []
        assert "attempting save anyway" in caplog.text
        assert "Failed to save session for thread-1" in caplog.text


class TestOptimisticLocking:
    def _seed(self, table, messages, version="v1"):
        item = {"thread_ts": "thread-1", "messages": json.dumps(messages), "updated_at": 1, "ttl": 2}
        if version is not None:
            item["version"] = version
        table.item = item

    def test_save_without_load_requires_item_to_not_exist(self, stub):
        table, _ = stub

        save_session("thread-1", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])

        _, kwargs = table.puts[0]
        assert kwargs == {"ConditionExpression": "attribute_not_exists(thread_ts)"}

    def test_load_then_save_conditions_on_loaded_version(self, stub):
        table, _ = stub
        prior = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        self._seed(table, prior, version="v1")

        loaded = load_session("thread-1")
        assert loaded == prior
        messages = loaded + [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
        save_session("thread-1", messages)

        item, kwargs = table.puts[0]
        assert kwargs["ConditionExpression"] == "attribute_not_exists(thread_ts) OR #ver = :ver"
        assert kwargs["ExpressionAttributeNames"] == {"#ver": "version"}
        assert kwargs["ExpressionAttributeValues"] == {":ver": "v1"}
        assert item["version"] != "v1" and len(item["version"]) == 32
        assert table.stored_messages() == messages

    def test_second_save_in_same_invocation_uses_fresh_version(self, stub):
        table, _ = stub
        self._seed(table, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
        messages = load_session("thread-1") + [{"role": "user", "content": "q2"}]

        save_session("thread-1", messages)
        first_version = table.puts[0][0]["version"]
        messages.append({"role": "assistant", "content": "a2"})
        save_session("thread-1", messages)

        _, kwargs = table.puts[1]
        assert kwargs["ExpressionAttributeValues"] == {":ver": first_version}
        assert table.stored_messages() == messages

    def test_legacy_item_without_version_conditions_on_missing_version(self, stub):
        table, _ = stub
        self._seed(table, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], version=None)

        messages = load_session("thread-1") + [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
        save_session("thread-1", messages)

        _, kwargs = table.puts[0]
        assert kwargs["ConditionExpression"] == "attribute_not_exists(#ver)"
        assert kwargs["ExpressionAttributeNames"] == {"#ver": "version"}
        assert "version" in table.item

    def test_load_of_missing_session_records_absence(self, stub):
        table, _ = stub

        assert load_session("thread-1") is None
        save_session("thread-1", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])

        assert table.puts[0][1] == {"ConditionExpression": "attribute_not_exists(thread_ts)"}

    def test_conflict_merges_stored_messages_with_new_ones(self, stub):
        table, audits = stub
        prior = [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]
        self._seed(table, prior, version="v1")
        messages = load_session("thread-1") + [
            {"role": "user", "content": "q1 (this invocation)"},
            {"role": "assistant", "content": "a1 (this invocation)"},
        ]

        # Another invocation wrote in the meantime.
        other = prior + [
            {"role": "user", "content": "qX (other invocation)"},
            {"role": "assistant", "content": "aX (other invocation)"},
        ]
        self._seed(table, other, version="v2")
        table.put_failures = [_client_error("ConditionalCheckFailedException")]

        save_session("thread-1", messages)

        assert len(table.puts) == 2
        _, first_kwargs = table.puts[0]
        assert first_kwargs["ExpressionAttributeValues"] == {":ver": "v1"}
        merged_item, second_kwargs = table.puts[1]
        assert second_kwargs["ExpressionAttributeValues"] == {":ver": "v2"}
        assert table.stored_messages() == other + messages[2:]
        assert merged_item["version"] not in {"v1", "v2"}
        conflicts = _events(audits, "session_write_conflict")
        assert conflicts == [
            {
                "thread_ts": "thread-1",
                "stored_messages": 4,
                "new_messages": 2,
                "merged_messages": 6,
            }
        ]

    def test_later_saves_after_a_conflict_keep_the_merge(self, stub):
        table, _ = stub
        prior = [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]
        self._seed(table, prior, version="v1")
        messages = load_session("thread-1") + [{"role": "user", "content": "q1"}]
        other = prior + [{"role": "user", "content": "qX"}, {"role": "assistant", "content": "aX"}]
        self._seed(table, other, version="v2")
        table.put_failures = [_client_error("ConditionalCheckFailedException")]

        save_session("thread-1", messages)  # conflict + merge
        merged_version = table.item["version"]
        messages.append({"role": "assistant", "content": "a1"})
        save_session("thread-1", messages)  # normal conditional save

        _, kwargs = table.puts[-1]
        assert kwargs["ExpressionAttributeValues"] == {":ver": merged_version}
        assert table.stored_messages() == other + [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]

    def test_conflict_when_session_was_never_loaded_merges_too(self, stub):
        table, audits = stub
        other = [{"role": "user", "content": "qX"}, {"role": "assistant", "content": "aX"}]
        self._seed(table, other, version="v9")
        table.put_failures = [_client_error("ConditionalCheckFailedException")]
        messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]

        save_session("thread-1", messages)

        assert table.puts[0][1] == {"ConditionExpression": "attribute_not_exists(thread_ts)"}
        assert table.stored_messages() == other + messages
        assert _events(audits, "session_write_conflict")[0]["new_messages"] == 2

    def test_merged_list_satisfies_first_message_invariant(self, stub):
        table, _ = stub
        # The other writer left an orphan tool_result at the front; the merge
        # must still start with a valid user message.
        self._seed(
            table,
            [
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]},
                {"role": "assistant", "content": "aX"},
                {"role": "user", "content": "qY"},
                {"role": "assistant", "content": "aY"},
            ],
            version="v2",
        )
        table.put_failures = [_client_error("ConditionalCheckFailedException")]

        save_session("thread-1", [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}])

        stored = table.stored_messages()
        assert _is_invalid_first_message(stored[0]) is False
        assert [m["content"] for m in stored] == ["qY", "aY", "q1", "a1"]

    def test_merged_result_still_goes_through_size_handling(self, stub):
        table, audits = stub
        self._seed(
            table,
            [{"role": "user", "content": "qX"}, {"role": "assistant", "content": "a" * 200_000}],
            version="v2",
        )
        table.put_failures = [_client_error("ConditionalCheckFailedException")]
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q1"}, _image_block(300_000)]},
            {"role": "assistant", "content": "a1"},
        ]

        save_session("thread-1", messages)

        stored = table.stored_messages()
        assert len(table.item["messages"].encode("utf-8")) <= MAX_ITEM_BYTES
        assert stored[-2]["content"][1]["type"] == "text"
        assert _events(audits, "session_trimmed")[0]["images_replaced"] == 1
        assert _events(audits, "session_write_conflict")

    def test_non_conditional_client_error_is_logged_not_merged(self, stub, caplog):
        table, audits = stub
        table.put_failures = [_client_error("ProvisionedThroughputExceededException")]
        caplog.set_level("ERROR")

        save_session("thread-1", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])

        assert len(table.puts) == 1
        assert table.get_calls == 0
        assert _events(audits, "session_write_conflict") == []
        assert "Failed to save session for thread-1" in caplog.text


# --------------------------------------------------------------------------- #
# Self-heal: assistant blocks stored with client-side SDK fields
# --------------------------------------------------------------------------- #


class TestLoadSanitizesStoredAssistantBlocks:
    """Threads poisoned by the Sonnet 5 streaming incident (``parsed_output`` /
    ``citations: null`` persisted in assistant turns) recover on the next load."""

    POISONED = [
        {"role": "user", "content": "how many rows?"},
        {
            "role": "assistant",
            "content": [
                {"citations": None, "text": "Let me check.", "type": "text", "parsed_output": None},
                {"type": "tool_use", "id": "t1", "name": "databricks_execute_sql", "input": {"sql": "SELECT 1"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "1"}]},
        {"role": "assistant", "content": [{"citations": None, "text": "One row.", "type": "text", "parsed_output": None}]},
    ]

    def test_load_returns_clean_blocks_and_does_not_write_back(self, stub, caplog):
        table, audits = stub
        table.item = {"thread_ts": "t1", "messages": json.dumps(self.POISONED), "version": "v1"}
        caplog.set_level("WARNING")

        loaded = load_session("t1")

        assert loaded[1]["content"] == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t1", "name": "databricks_execute_sql", "input": {"sql": "SELECT 1"}},
        ]
        assert loaded[3]["content"] == [{"type": "text", "text": "One row."}]
        assert loaded[0] == self.POISONED[0] and loaded[2] == self.POISONED[2]
        for message in loaded:
            if isinstance(message["content"], list):
                for block in message["content"]:
                    assert "parsed_output" not in block and "citations" not in block
        assert table.puts == []  # no write on load
        assert "sanitized on load" in caplog.text
        # Stored item untouched until the next save.
        assert json.loads(table.item["messages"]) == self.POISONED

    def test_next_save_persists_the_clean_version(self, stub):
        table, audits = stub
        table.item = {"thread_ts": "t1", "messages": json.dumps(self.POISONED), "version": "v1"}

        loaded = load_session("t1")
        loaded.append({"role": "user", "content": "thanks"})
        save_session("t1", loaded)

        stored = table.stored_messages()
        assert stored[-1] == {"role": "user", "content": "thanks"}
        assert stored[3]["content"] == [{"type": "text", "text": "One row."}]
        assert "parsed_output" not in json.dumps(stored)
        assert _events(audits, "session_trimmed") == []

    def test_clean_session_is_returned_unchanged_without_warning(self, stub, caplog):
        table, _ = stub
        clean = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        ]
        table.item = {"thread_ts": "t1", "messages": json.dumps(clean), "version": "v1"}
        caplog.set_level("WARNING")

        assert load_session("t1") == clean
        assert "sanitized" not in caplog.text

    def test_provider_reasoning_blocks_survive_load(self, stub):
        table, _ = stub
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "provider_reasoning", "provider": "openai", "payload": {"id": "rs_1"}},
                    {"type": "text", "text": "a", "parsed_output": None},
                ],
            },
        ]
        table.item = {"thread_ts": "t1", "messages": json.dumps(messages), "version": "v1"}

        loaded = load_session("t1")

        assert loaded[1]["content"] == [
            {"type": "provider_reasoning", "provider": "openai", "payload": {"id": "rs_1"}},
            {"type": "text", "text": "a"},
        ]


def test_conflict_retries_against_latest_version_without_losing_another_writer(stub, monkeypatch):
    table, _ = stub
    prior = [{"role": "user", "content": "original"}]
    table.item = {"thread_ts": "t", "version": "v1", "messages": json.dumps(prior)}
    messages = load_session("t") + [{"role": "assistant", "content": "our answer"}]
    other = prior + [{"role": "assistant", "content": "other answer"}]
    table.item = {"thread_ts": "t", "version": "v2", "messages": json.dumps(other)}
    original_put = table.put_item
    raced = False

    def concurrent_put(Item, **kwargs):
        nonlocal raced
        expected = kwargs.get("ExpressionAttributeValues", {}).get(":ver")
        if expected == "v2" and not raced:
            raced = True
            table.item = {"thread_ts": "t", "version": "v3", "messages": json.dumps(
                other + [{"role": "user", "content": "newer question"}])}
        if expected != table.item["version"]:
            raise _client_error("ConditionalCheckFailedException")
        original_put(Item=Item, **kwargs)

    monkeypatch.setattr(table, "put_item", concurrent_put)
    save_session("t", messages)
    assert raced
    assert [m["content"] for m in table.stored_messages()] == [
        "original", "other answer", "newer question", "our answer"]


def test_repeated_conflict_does_not_duplicate_already_saved_messages(stub):
    table, _ = stub
    prior = [{"role": "user", "content": "original"}]
    table.item = {"thread_ts": "t", "version": "v1", "messages": json.dumps(prior)}
    messages = load_session("t") + [{"role": "assistant", "content": "our first answer"}]
    other = prior + [{"role": "user", "content": "other question"}]
    table.item = {"thread_ts": "t", "version": "v2", "messages": json.dumps(other)}
    table.put_failures = [_client_error("ConditionalCheckFailedException")]
    save_session("t", messages)
    first_saved = table.stored_messages()
    table.item = {"thread_ts": "t", "version": "v3", "messages": json.dumps(
        first_saved + [{"role": "user", "content": "another question"}])}
    messages.append({"role": "user", "content": "our follow-up"})
    table.put_failures = [_client_error("ConditionalCheckFailedException")]
    save_session("t", messages)
    assert [m["content"] for m in table.stored_messages()] == [
        "original", "other question", "our first answer", "another question", "our follow-up"]


def test_conflict_retry_exhaustion_never_overwrites_stored_history(stub):
    table, audits = stub
    original = {"thread_ts": "t", "version": "new", "messages": json.dumps(
        [{"role": "user", "content": "other writer"}])}
    table.item = original
    table.put_failures = [_client_error("ConditionalCheckFailedException")] * (sessions.MAX_CONFLICT_RETRIES + 1)
    save_session("t", [{"role": "user", "content": "ours"}])
    assert table.item == original
    assert len(table.puts) == sessions.MAX_CONFLICT_RETRIES + 1
    assert _events(audits, "session_save_failed")[-1]["reason"] == "conflict_retries_exhausted"
