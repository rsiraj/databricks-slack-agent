"""Message compaction — worker.py.

Not security-critical, but `_prepare_messages_for_api` runs on every API call and
a regression that drops the wrong block (or breaks tool_use/tool_result pairing)
would break the live agent loop, so it's worth pinning down. `_compact_tool_result_content`
must pass SQL results and file content through untouched (the model grounded prior
answers on them) while stripping boilerplate from other tool shapes.
"""

import copy
import json

from utils.compaction import _compact_tool_result_content, _prepare_messages_for_api
from worker import _is_status_message, _strip_mention


class TestCompactToolResult:
    def test_sql_result_passes_through(self):
        raw = json.dumps(
            {"status": "SUCCEEDED", "columns": ["a"], "sample_rows": [[1], [2]]}
        )
        assert _compact_tool_result_content(raw) == raw

    def test_file_content_passes_through(self):
        raw = json.dumps({"content": "x" * 5000, "path": "models/a.sql", "ref": "main"})
        assert _compact_tool_result_content(raw) == raw

    def test_running_sql_passes_through(self):
        raw = json.dumps({"status": "RUNNING", "statement_id": "abc"})
        assert _compact_tool_result_content(raw) == raw

    def test_github_search_drops_sha_and_url(self):
        raw = json.dumps(
            {
                "total_count": 2,
                "items": [
                    {"path": "a.sql", "sha": "deadbeef", "html_url": "http://x"},
                    {"path": "b.sql", "sha": "cafebabe", "html_url": "http://y"},
                ],
            }
        )
        out = _compact_tool_result_content(raw)
        assert json.loads(out) == {
            "total_count": 2,
            "items": [{"path": "a.sql"}, {"path": "b.sql"}],
        }
        assert "sha" not in out and "html_url" not in out

    def test_github_compare_drops_patches(self):
        raw = json.dumps(
            {
                "ahead_by": 3,
                "total_commits": 3,
                "files": [
                    {
                        "filename": "a.sql",
                        "status": "modified",
                        "additions": 5,
                        "deletions": 2,
                        "patch": "@@ a very large diff body @@",
                    }
                ],
            }
        )
        out = _compact_tool_result_content(raw)
        assert "patch" not in out
        assert json.loads(out)["files"][0]["filename"] == "a.sql"

    def test_job_run_listing_keeps_summary_and_run_essentials_as_valid_json(self):
        runs = [
            {
                "run_id": 1000 + index,
                "job_id": 42,
                "run_name": f"Coverage {index}",
                "life_cycle_state": "TERMINATED",
                "result_state": "SUCCESS",
                "state_message": "",
                "start_time_ms": 1_786_556_730_000 + index,
                "setup_duration_ms": 1_000,
                "execution_duration_ms": 358_000,
                "cleanup_duration_ms": 1_000,
                "end_time_ms": 1_786_557_090_000 + index,
                "run_page_url": f"https://dbx.example.test/#job/42/run/{1000 + index}",
                "run_type": "JOB_RUN",
                "trigger": "PERIODIC",
                "start_time": "2026-08-12T08:05:30-04:00",
                "end_time": "2026-08-12T08:11:30-04:00",
            }
            for index in range(30)
        ]
        raw = json.dumps(
            {
                "date": "2026-08-12",
                "scope": "started_on_date",
                "timezone": "America/New_York",
                "active_only": False,
                "fetched_run_count": 30,
                "truncated": False,
                "interpretation_note": "These are observed executions only.",
                "runs": runs,
            }
        )
        assert len(raw) > 1000  # the generic fallback would have cut this mid-string

        out = _compact_tool_result_content(raw)
        data = json.loads(out)  # must be valid JSON, not a truncated string

        assert data["date"] == "2026-08-12"
        assert data["fetched_run_count"] == 30
        assert data["truncated"] is False
        assert "interpretation_note" not in data
        assert len(data["runs"]) == 30
        assert data["runs"][0] == {
            "run_name": "Coverage 0",
            "job_id": 42,
            "run_id": 1000,
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "start_time": "2026-08-12T08:05:30-04:00",
            "end_time": "2026-08-12T08:11:30-04:00",
            "setup_duration_ms": 1_000,
            "execution_duration_ms": 358_000,
            "cleanup_duration_ms": 1_000,
        }
        assert "run_page_url" not in out
        assert "trigger" not in out
        assert len(out) < len(raw)

    def test_active_run_listing_without_date_is_recognised(self):
        raw = json.dumps(
            {
                "date": None,
                "scope": "currently_active",
                "timezone": "America/New_York",
                "active_only": True,
                "fetched_run_count": 1,
                "truncated": False,
                "interpretation_note": "x" * 1200,
                "runs": [
                    {
                        "run_id": 7,
                        "job_id": 1,
                        "run_name": "Email Merge",
                        "life_cycle_state": "RUNNING",
                        "result_state": None,
                        "run_page_url": "https://dbx.example.test/#job/1/run/7",
                    }
                ],
            }
        )
        data = json.loads(_compact_tool_result_content(raw))
        assert data["scope"] == "currently_active"
        assert data["runs"] == [
            {
                "run_name": "Email Merge",
                "job_id": 1,
                "run_id": 7,
                "life_cycle_state": "RUNNING",
            }
        ]

    def test_recent_runs_summary_shape_is_unchanged(self):
        raw = json.dumps(
            {
                "job_id": 1,
                "fetched_run_count": 2,
                "successful_run_summary": {"successful_run_count": 2},
                "runs": [{"run_id": 1}, {"run_id": 2}],
            }
        )
        assert json.loads(_compact_tool_result_content(raw)) == {
            "job_id": 1,
            "fetched_run_count": 2,
            "successful_run_summary": {"successful_run_count": 2},
        }

    def test_small_error_passes_through(self):
        raw = json.dumps({"error": "nope"})
        assert _compact_tool_result_content(raw) == raw

    def test_unrecognised_long_payload_truncated(self):
        raw = "x" * 1500
        out = _compact_tool_result_content(raw)
        assert out.endswith("…[truncated]")
        assert len(out) < len(raw)

    def test_non_dict_json_passes_through(self):
        raw = json.dumps([1, 2, 3])
        assert _compact_tool_result_content(raw) == raw


def _sample_messages():
    """A two-run thread: user → assistant(tool_use) → tool_result → assistant(answer)."""
    return [
        {"role": "user", "content": [{"type": "text", "text": "question 1"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "old reasoning"},
                {"type": "tool_use", "id": "t1", "name": "github_search_code", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": json.dumps(
                        {
                            "total_count": 1,
                            "items": [{"path": "a.sql", "sha": "x", "html_url": "u"}],
                        }
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "new reasoning"},
                {"type": "text", "text": "the answer"},
            ],
        },
    ]


class TestPrepareMessagesForApi:
    def test_does_not_mutate_original(self):
        msgs = _sample_messages()
        snapshot = copy.deepcopy(msgs)
        _prepare_messages_for_api(msgs, current_run_start=3)
        assert msgs == snapshot

    def test_strips_thinking_from_all_but_last_assistant(self):
        out = _prepare_messages_for_api(_sample_messages(), current_run_start=3)
        assert all(b.get("type") != "thinking" for b in out[1]["content"])
        assert any(b.get("type") == "thinking" for b in out[3]["content"])

    def test_compacts_tool_results_before_current_run(self):
        out = _prepare_messages_for_api(_sample_messages(), current_run_start=3)
        compacted = out[2]["content"][0]["content"]
        assert "sha" not in compacted

    def test_preserves_tool_results_in_current_run(self):
        out = _prepare_messages_for_api(_sample_messages(), current_run_start=2)
        preserved = out[2]["content"][0]["content"]
        assert "sha" in preserved

    def test_single_message_returned_as_is(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert _prepare_messages_for_api(msgs) is msgs

    def test_tool_use_result_pairing_preserved(self):
        out = _prepare_messages_for_api(_sample_messages(), current_run_start=3)
        use_ids = {
            b["id"]
            for m in out
            if m["role"] == "assistant"
            for b in m["content"]
            if b.get("type") == "tool_use"
        }
        result_ids = {
            b["tool_use_id"]
            for m in out
            if m["role"] == "user"
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        assert use_ids == result_ids


def _reasoning(block_id):
    return {
        "type": "provider_reasoning",
        "provider": "openai",
        "payload": {"type": "reasoning", "id": block_id, "encrypted_content": "opaque"},
    }


def _two_run_thread_with_provider_reasoning():
    """Run 1 (indices 0-3) then run 2 (index 4 onward), OpenAI-style history."""
    return [
        {"role": "user", "content": "question 1"},
        {
            "role": "assistant",
            "content": [
                _reasoning("rs_old_1"),
                {"type": "tool_use", "id": "c1", "name": "calculate", "input": {"expression": "1+1"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": '{"result": 2}'}]},
        {"role": "assistant", "content": [_reasoning("rs_old_2"), {"type": "text", "text": "2"}]},
        {"role": "user", "content": "question 2"},
        {
            "role": "assistant",
            "content": [
                _reasoning("rs_new_1"),
                {"type": "tool_use", "id": "c2", "name": "calculate", "input": {"expression": "2+2"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c2", "content": '{"result": 4}'}]},
        {"role": "assistant", "content": [_reasoning("rs_new_2"), {"type": "text", "text": "4"}]},
    ]


class TestProviderReasoningCompaction:
    def test_stripped_from_previous_runs_only(self):
        out = _prepare_messages_for_api(_two_run_thread_with_provider_reasoning(), current_run_start=4)

        def reasoning_ids(message):
            return [b["payload"]["id"] for b in message["content"] if b.get("type") == "provider_reasoning"]

        assert reasoning_ids(out[1]) == []
        assert reasoning_ids(out[3]) == []
        # Current run keeps every reasoning item so function_call <-> reasoning
        # pairing stays intact for the active tool loop (not just the last turn).
        assert reasoning_ids(out[5]) == ["rs_new_1"]
        assert reasoning_ids(out[7]) == ["rs_new_2"]
        # Tool calls and text in the older turns survive.
        assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
        assert [b["type"] for b in out[3]["content"]] == ["text"]

    def test_original_is_not_mutated(self):
        msgs = _two_run_thread_with_provider_reasoning()
        snapshot = copy.deepcopy(msgs)
        _prepare_messages_for_api(msgs, current_run_start=4)
        assert msgs == snapshot

    def test_offline_fallback_uses_last_user_turn_boundary(self):
        out = _prepare_messages_for_api(_two_run_thread_with_provider_reasoning())
        # Boundary is the last user turn (index 6): everything before it loses reasoning.
        assert all(b.get("type") != "provider_reasoning" for b in out[5]["content"])
        assert any(b.get("type") == "provider_reasoning" for b in out[7]["content"])

    def test_thinking_rule_is_unchanged_alongside(self):
        msgs = _sample_messages()
        msgs[1]["content"].insert(0, _reasoning("rs_x"))
        out = _prepare_messages_for_api(msgs, current_run_start=3)
        assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
        assert [b["type"] for b in out[3]["content"]] == ["thinking", "text"]


def _thinking(text, signature="sig"):
    return {"type": "thinking", "thinking": text, "signature": signature}


def _tool_use(tool_id, expression):
    return {"type": "tool_use", "id": tool_id, "name": "calculate", "input": {"expression": expression}}


def _tool_result(tool_id, content):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": False}]}


class TestCurrentRunPrefixIsCacheStable:
    """Prompt caching is a prefix match: the compacted request at iteration N
    must reappear byte-for-byte as a prefix of the request at iteration N+1.
    Before this rule, iteration N's assistant turn lost its ``thinking`` block
    at iteration N+1 (only the *last* assistant turn kept it), which rewrote
    the prefix and invalidated the conversation cache on every call."""

    PRIOR_RUN = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": [_thinking("prior-run think", "s0"), {"type": "text", "text": "earlier answer"}]},
    ]
    CURRENT_RUN_START = len(PRIOR_RUN)

    def _iteration(self, n):
        """Message list as sent on iteration ``n`` (0-based) of the current run."""
        messages = copy.deepcopy(self.PRIOR_RUN) + [{"role": "user", "content": "current question"}]
        for k in range(n):
            messages.append({"role": "assistant", "content": [_thinking(f"iter-{k} think", f"s{k + 1}"), _tool_use(f"t{k}", f"{k}+{k}")]})
            messages.append(_tool_result(f"t{k}", json.dumps({"result": 2 * k})))
        return messages

    def test_two_iteration_overlap_is_byte_identical(self):
        first = _prepare_messages_for_api(self._iteration(1), self.CURRENT_RUN_START)
        second = _prepare_messages_for_api(self._iteration(2), self.CURRENT_RUN_START)
        assert json.dumps(second[: len(first)]) == json.dumps(first)

    def test_every_iteration_extends_the_previous_one(self):
        prepared = [_prepare_messages_for_api(self._iteration(n), self.CURRENT_RUN_START) for n in range(1, 5)]
        for earlier, later in zip(prepared, prepared[1:]):
            assert later[: len(earlier)] == earlier
            assert len(later) == len(earlier) + 2

    def test_current_run_assistant_turns_keep_their_thinking(self):
        out = _prepare_messages_for_api(self._iteration(3), self.CURRENT_RUN_START)
        current_run_assistants = [m for m in out[self.CURRENT_RUN_START:] if m["role"] == "assistant"]
        assert len(current_run_assistants) == 3
        for turn in current_run_assistants:
            assert turn["content"][0]["type"] == "thinking"

    def test_prior_run_thinking_is_still_stripped_and_stable_across_iterations(self):
        first = _prepare_messages_for_api(self._iteration(1), self.CURRENT_RUN_START)
        second = _prepare_messages_for_api(self._iteration(2), self.CURRENT_RUN_START)
        for out in (first, second):
            assert [b["type"] for b in out[1]["content"]] == ["text"]
        assert first[: self.CURRENT_RUN_START] == second[: self.CURRENT_RUN_START]

    def test_prior_run_last_turn_is_stripped_on_the_first_call_too(self):
        # A history that ends in the previous run's final assistant turn with
        # the boundary past it (call 1 of a follow-up). That turn is the last
        # assistant turn in the list, but it is prior-run history: it must be
        # stripped now exactly as it will be on call 2, or the prefix rewrites.
        messages = copy.deepcopy(self.PRIOR_RUN) + [{"role": "user", "content": "new question"}]
        out = _prepare_messages_for_api(messages, current_run_start=2)
        assert [b["type"] for b in out[1]["content"]] == ["text"]

    def test_offline_fallback_behaviour_is_unchanged(self):
        # No boundary: thinking survives only on the most recent assistant turn.
        out = _prepare_messages_for_api(self._iteration(2))
        assistants = [m for m in out if m["role"] == "assistant"]
        assert [b["type"] for b in assistants[0]["content"]] == ["text"]
        assert [b["type"] for b in assistants[1]["content"]] == ["tool_use"]
        assert [b["type"] for b in assistants[2]["content"]] == ["thinking", "tool_use"]

    def test_never_emits_cache_control_and_never_mutates_input(self):
        messages = self._iteration(2)
        snapshot = copy.deepcopy(messages)
        out = _prepare_messages_for_api(messages, self.CURRENT_RUN_START)
        assert messages == snapshot
        assert "cache_control" not in json.dumps(out)


class TestFollowUpTurnPrefixIsCacheStable:
    """A follow-up turn appends a new run to a stored session. On the run's
    first call the previous run's final assistant turn is the last assistant
    turn in the list; on the second call the current run has its own. The
    prior-run prefix must be byte-identical between those two calls — before
    this rule the last-assistant exemption kept the prior turn's ``thinking``
    on call 1 and stripped it on call 2, rewriting the prefix once per
    follow-up (production: ``cache_read_input_tokens`` fell back to the
    system-prompt-only value on iteration 1 and recovered on iteration 2)."""

    PRIOR_RUN = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": [_thinking("prior think 1", "p1"), _tool_use("p0", "1+1")]},
        _tool_result("p0", json.dumps({"total_count": 1, "items": [{"path": "a.sql", "sha": "x", "html_url": "u"}]})),
        {"role": "assistant", "content": [_thinking("prior think 2", "p2"), {"type": "text", "text": "earlier answer"}]},
    ]
    CURRENT_RUN_START = len(PRIOR_RUN)

    def _call(self, n):
        """Message list as sent on call ``n`` (1-based) of the follow-up run."""
        messages = copy.deepcopy(self.PRIOR_RUN) + [{"role": "user", "content": "follow-up question"}]
        for k in range(n - 1):
            messages.append({"role": "assistant", "content": [_thinking(f"cur think {k}", f"c{k}"), _tool_use(f"c{k}", f"{k}+{k}")]})
            messages.append(_tool_result(f"c{k}", json.dumps({"result": 2 * k})))
        return messages

    def test_first_and_second_call_share_a_byte_identical_prefix(self):
        first = _prepare_messages_for_api(self._call(1), self.CURRENT_RUN_START)
        second = _prepare_messages_for_api(self._call(2), self.CURRENT_RUN_START)
        assert json.dumps(second[: len(first)]) == json.dumps(first)
        assert len(second) == len(first) + 2

    def test_prior_run_thinking_is_gone_on_every_call(self):
        for n in (1, 2, 3):
            out = _prepare_messages_for_api(self._call(n), self.CURRENT_RUN_START)
            assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
            assert [b["type"] for b in out[3]["content"]] == ["text"]
            # Prior-run tool results are compacted identically too.
            assert "sha" not in out[2]["content"][0]["content"]

    def test_current_run_assistant_turns_keep_their_thinking(self):
        out = _prepare_messages_for_api(self._call(3), self.CURRENT_RUN_START)
        current = [m for m in out[self.CURRENT_RUN_START:] if m["role"] == "assistant"]
        assert len(current) == 2
        for turn in current:
            assert [b["type"] for b in turn["content"]] == ["thinking", "tool_use"]

    def test_tool_use_result_pairing_survives(self):
        out = _prepare_messages_for_api(self._call(2), self.CURRENT_RUN_START)
        use_ids = {b["id"] for m in out if m["role"] == "assistant" for b in m["content"] if b.get("type") == "tool_use"}
        result_ids = {
            b["tool_use_id"]
            for m in out
            if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == "tool_result"
        }
        assert use_ids == result_ids == {"p0", "c0"}

    def test_offline_fallback_still_keeps_only_the_last_assistant_thinking(self):
        # No boundary: unchanged — the most recent assistant turn keeps its
        # thinking even when it is the previous run's final turn.
        out = _prepare_messages_for_api(self._call(1))
        assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
        assert [b["type"] for b in out[3]["content"]] == ["thinking", "text"]
        out = _prepare_messages_for_api(self._call(2))
        assistants = [m for m in out if m["role"] == "assistant"]
        assert [b["type"] for b in assistants[0]["content"]] == ["tool_use"]
        assert [b["type"] for b in assistants[1]["content"]] == ["text"]
        assert [b["type"] for b in assistants[2]["content"]] == ["thinking", "tool_use"]


class TestSlackTextHelpers:
    def test_strip_mention_removes_leading_mention(self):
        assert _strip_mention("<@U123ABC> hello there") == "hello there"

    def test_strip_mention_passthrough_when_absent(self):
        assert _strip_mention("hello there") == "hello there"

    def test_is_status_message_detects_status(self):
        assert _is_status_message("*Working on it...*\nRunning SQL") is True

    def test_is_status_message_false_for_answer(self):
        assert _is_status_message("Here is your answer.") is False

    def test_is_status_message_false_for_empty(self):
        assert _is_status_message("") is False


class TestDescribeObjectCompaction:
    @staticmethod
    def _describe_payload(comment_length: int) -> dict:
        return {
            "status": "SUCCEEDED",
            "statement_id": "stmt-1",
            "object": {
                "catalog": "acme_prod",
                "schema": "reporting",
                "name": "signups",
                "full_name": "acme_prod.reporting.signups",
                "object_type": "TABLE",
                "table_type": "MANAGED",
                "comment": "T" * comment_length,
                "data_source_format": "DELTA",
                "created": "2024-01-01T00:00:00Z",
                "last_altered": "2026-08-28T09:00:00Z",
            },
            "column_count": 60,
            "columns": [
                {
                    "ordinal_position": index,
                    "name": f"column_{index}",
                    "data_type": "STRING",
                    "is_nullable": index % 2 == 0,
                    "comment": "C" * comment_length if index == 1 else None,
                }
                for index in range(1, 61)
            ],
            "columns_truncated": False,
            "note": "column_count is the authoritative number of columns.",
        }

    def test_structured_describe_keeps_every_column_and_count_as_valid_json(self):
        raw = json.dumps(self._describe_payload(50))
        assert len(raw) > 1000  # the generic fallback would cut this mid-JSON

        data = json.loads(_compact_tool_result_content(raw))

        assert data["column_count"] == 60 == len(data["columns"])
        assert data["object"]["full_name"] == "acme_prod.reporting.signups"
        assert data["object"]["comment"] == "T" * 50
        assert data["columns"][0] == {
            "ordinal_position": 1,
            "name": "column_1",
            "data_type": "STRING",
            "is_nullable": False,
            "comment": "C" * 50,
        }
        assert data["note"] == "column_count is the authoritative number of columns."

    def test_structured_describe_trims_comments_longer_than_200_chars(self):
        raw = json.dumps(self._describe_payload(500))

        data = json.loads(_compact_tool_result_content(raw))

        assert data["object"]["comment"].startswith("T" * 200)
        assert len(data["object"]["comment"]) < 230
        assert data["columns"][0]["comment"].startswith("C" * 200)
        assert data["columns"][0]["comment"].endswith("[truncated]")
        assert data["columns"][1]["comment"] is None
        assert data["column_count"] == 60 == len(data["columns"])
        assert len(json.dumps(data)) < len(raw)
