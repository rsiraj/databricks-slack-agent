from copy import deepcopy

import pytest

import worker
from utils import clients
from utils.slack_helpers import MessageDelivery


class _FakeBlock:
    def __init__(self, block_type, **values):
        self.type = block_type
        for key, value in values.items():
            setattr(self, key, value)

    def model_dump(self):
        result = {"type": self.type}
        for key in ("id", "name", "input", "text"):
            if hasattr(self, key):
                result[key] = getattr(self, key)
        return result


class _FakeResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeAnthropicClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = self

    def create(self, **kwargs):
        return self.responses.pop(0)


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "tool_result", "expected_outcome", "expected_event"),
    [
        (
            "databricks_execute_sql",
            {"sql": "DELETE FROM acme_prod.reporting.signups"},
            {
                "error": "Only read-only statements are allowed.",
                "validation_failure": "disallowed_statement_type",
            },
            "guard_precondition",
            "guard_precondition",
        ),
        (
            "databricks_get_statement",
            {"statement_id": "statement-1"},
            {"error": "Databricks unavailable", "status_code": 503},
            "databricks_failure",
            "databricks_failure",
        ),
    ],
)
def test_worker_audits_tool_outcomes_without_changing_status_rendering(
    monkeypatch,
    tool_name,
    tool_input,
    tool_result,
    expected_outcome,
    expected_event,
):
    client = _FakeAnthropicClient(
        [
            _FakeResponse(
                "tool_use",
                [
                    _FakeBlock(
                        "tool_use",
                        id="tool-1",
                        name=tool_name,
                        input=tool_input,
                    )
                ],
            ),
            _FakeResponse("end_turn", [_FakeBlock("text", text="Done")]),
        ]
    )
    audits = []
    rendered_statuses = []
    saved = []

    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(worker, "_dispatch_tool", lambda name, inp: deepcopy(tool_result))
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: "status-ts",
    )
    monkeypatch.setattr(
        worker,
        "update_message",
        lambda channel, ts, text: rendered_statuses.append(text),
    )
    monkeypatch.setattr(
        worker,
        "post_message_chunks",
        lambda channel, text, thread_ts=None: MessageDelivery((text,), ("answer-ts",), 1),
    )
    monkeypatch.setattr(worker, "_post_feedback_prompt", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "save_session",
        lambda thread_ts, messages: saved.append(deepcopy(messages)),
    )
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    worker.run_agent(
        [{"role": "user", "content": "Test the tool."}],
        "C123",
        "thread-1",
    )

    assert any(event == expected_event for event, _ in audits)
    tool_call = next(fields for event, fields in audits if event == "tool_call")
    assert tool_call["outcome"] == expected_outcome
    assert tool_call["is_error"] is True
    assert any("Had trouble" in status for status in rendered_statuses)
    assert saved


def test_tool_outcome_keeps_rate_limit_separate_from_databricks_failure():
    assert worker._tool_outcome(
        "databricks_execute_sql",
        {"error": "cool-down", "rate_limit_failure": "daily limit"},
    ) == "tool_failure"


def test_databricks_error_code_is_extracted_without_full_error_text():
    assert worker._databricks_error_code(
        {
            "error": (
                "[CAST_INVALID_INPUT] The value cannot be cast. "
                "SQLSTATE: 22018"
            )
        }
    ) == "CAST_INVALID_INPUT"
    assert worker._databricks_error_code({"error": "plain failure"}) is None


def test_auto_metadata_then_column_rejection_emits_both_guard_preconditions(monkeypatch):
    client = _FakeAnthropicClient(
        [
            _FakeResponse(
                "tool_use",
                [
                    _FakeBlock(
                        "tool_use",
                        id="tool-1",
                        name="databricks_execute_sql",
                        input={
                            "sql": (
                                "SELECT internal_sum "
                                "FROM acme_prod.reporting.signups"
                            )
                        },
                    )
                ],
            ),
            _FakeResponse("end_turn", [_FakeBlock("text", text="Done")]),
        ]
    )
    metadata_rows = [
        ["table", None, "signups", "MANAGED", None, "test object"],
        ["column", 1, "_anonymousid", "STRING", "YES", None],
    ]
    metadata_result = {
        "status": "SUCCEEDED",
        "row_count": len(metadata_rows),
        "sample_row_count": len(metadata_rows),
        "truncated": False,
        "columns": [
            "metadata_kind",
            "ordinal_position",
            "metadata_name",
            "data_type",
            "is_nullable",
            "comment",
        ],
        "sample_rows": metadata_rows,
    }
    dispatches = []
    audits = []

    def fake_dispatch(name, inp):
        dispatches.append((name, inp))
        assert name == "databricks_describe_object"
        return deepcopy(metadata_result)

    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(worker, "_dispatch_tool", fake_dispatch)
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: "status-ts",
    )
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "post_message_chunks",
        lambda channel, text, thread_ts=None: MessageDelivery((text,), ("answer-ts",), 1),
    )
    monkeypatch.setattr(worker, "_post_feedback_prompt", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "save_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    worker.run_agent(
        [{"role": "user", "content": "Query the stored internal_sum column."}],
        "C123",
        "thread-1",
    )

    guard_events = [fields for event, fields in audits if event == "guard_precondition"]
    assert [event["validation_failure"] for event in guard_events] == [
        "object_metadata_required",
        "column_not_in_live_schema",
    ]
    assert guard_events[0]["auto_satisfied"] is True
    assert guard_events[1]["auto_satisfied"] is False
    assert guard_events[1]["source"] == "sql_run_guard"
    assert [name for name, _ in dispatches] == ["databricks_describe_object"]
    tool_call = next(fields for event, fields in audits if event == "tool_call")
    assert tool_call["outcome"] == "guard_precondition"
