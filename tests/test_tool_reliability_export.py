import json
from datetime import datetime, timezone

from scripts import export_sessions


def _event(event_id, timestamp, message):
    return {
        "eventId": event_id,
        "timestamp": timestamp,
        "message": message,
        "logStreamName": "2026/08/28/[$LATEST]abc",
    }


def test_parse_reliability_event_from_plain_lambda_log_message():
    payload = {
        "audit_event": "guard_precondition",
        "audit_timestamp": "2026-08-28T20:15:00+00:00",
        "thread_ts": "1787948143.137669",
        "validation_failure": "object_metadata_required",
        "auto_satisfied": True,
    }
    event = _event(
        "event-1",
        1_788_000_000_000,
        f"[INFO]\trequest-id\taudit {json.dumps(payload)}",
    )

    parsed = export_sessions.parse_tool_reliability_log_event(event)

    assert parsed["event_id"] == "event-1"
    assert parsed["audit_event"] == "guard_precondition"
    assert parsed["validation_failure"] == "object_metadata_required"
    assert parsed["thread_ts"] == "1787948143.137669"


def test_parse_reliability_event_from_json_formatted_log_message():
    payload = {
        "audit_event": "databricks_recovery_succeeded",
        "operation": "sql.execute",
        "attempts": 2,
    }
    event = _event(
        "event-2",
        1_788_000_001_000,
        json.dumps({"level": "INFO", "message": f"audit {json.dumps(payload)}"}),
    )

    parsed = export_sessions.parse_tool_reliability_log_event(event)

    assert parsed["audit_event"] == "databricks_recovery_succeeded"
    assert parsed["operation"] == "sql.execute"


def test_parser_ignores_unrelated_or_malformed_audit_events():
    unrelated = _event(
        "event-3",
        1,
        'audit {"audit_event":"answer_feedback","rating":"helpful"}',
    )
    malformed = _event("event-4", 2, 'audit {"audit_event":')

    assert export_sessions.parse_tool_reliability_log_event(unrelated) is None
    assert export_sessions.parse_tool_reliability_log_event(malformed) is None


def test_parser_includes_agent_errors_for_reference_lookup():
    payload = {
        "audit_event": "agent_error",
        "error_id": "agent-6932fbda6013",
        "error_type": "OverloadedError",
        "failure_type": "anthropic_overloaded",
        "status_code": 529,
        "thread_ts": "1788460921.921689",
    }
    event = _event(
        "event-agent-error",
        1_788_464_164_139,
        f"[INFO]\trequest-id\taudit {json.dumps(payload)}",
    )

    parsed = export_sessions.parse_tool_reliability_log_event(event)

    assert parsed["audit_event"] == "agent_error"
    assert parsed["error_id"] == "agent-6932fbda6013"
    assert parsed["failure_type"] == "anthropic_overloaded"


def test_fetch_reliability_events_paginates_and_deduplicates():
    first = _event(
        "event-1",
        1000,
        'audit {"audit_event":"tool_call","outcome":"success"}',
    )
    second = _event(
        "event-2",
        2000,
        'audit {"audit_event":"databricks_failure","failure_type":"statement"}',
    )

    class FakeLogsClient:
        def __init__(self):
            self.calls = []

        def filter_log_events(self, **kwargs):
            self.calls.append(kwargs)
            if "nextToken" not in kwargs:
                return {"events": [first], "nextToken": "next"}
            return {"events": [first, second], "nextToken": "next"}

    client = FakeLogsClient()

    events = export_sessions.fetch_tool_reliability_events(client, 500)

    assert len(client.calls) == 2
    assert client.calls[0]["filterPattern"] == '"audit_event"'
    assert [event["event_id"] for event in events] == ["event-2", "event-1"]


def test_reliability_report_counts_events_and_tool_outcomes(monkeypatch):
    monkeypatch.setattr(export_sessions, "GUARD_EVENT_LOOKBACK_DAYS", 7)
    events = [
        {
            "audit_event": "tool_call",
            "outcome": "success",
            "timestamp_iso": "2026-08-28T20:15:00+00:00",
        },
        {
            "audit_event": "tool_call",
            "outcome": "guard_precondition",
            "timestamp_iso": "2026-08-28T20:15:01+00:00",
            "thread_ts": "1787948143.137669",
        },
        {
            "audit_event": "guard_precondition",
            "validation_failure": "object_metadata_required",
            "timestamp_iso": "2026-08-28T20:15:01+00:00",
        },
        {
            "audit_event": "databricks_transient_failure",
            "failure_type": "http",
            "timestamp_iso": "2026-08-28T20:15:02+00:00",
        },
        {
            "audit_event": "databricks_recovery_succeeded",
            "operation": "sql.execute",
            "timestamp_iso": "2026-08-28T20:15:03+00:00",
        },
        {
            "audit_event": "databricks_failure",
            "failure_type": "statement",
            "timestamp_iso": "2026-08-28T20:15:04+00:00",
        },
        {
            "audit_event": "agent_error",
            "error_id": "agent-6932fbda6013",
            "error_type": "OverloadedError",
            "failure_type": "anthropic_overloaded",
            "timestamp_iso": "2026-08-28T20:15:05+00:00",
            "thread_ts": "1787948143.137669",
        },
    ]

    report = export_sessions.render_tool_reliability_report(
        events,
        datetime(2026, 8, 28, 20, 20, tzinfo=timezone.utc),
    )

    assert "Tool calls: 2" in report
    assert "Guard preconditions: 1" in report
    assert "Auto-satisfied metadata preflights: 0" in report
    assert "Blocking guard results: 1" in report
    assert "Databricks transient failures: 1" in report
    assert "Databricks recoveries: 1" in report
    assert "Final Databricks failures: 1" in report
    assert "Agent errors: 1" in report
    assert "`success`: 1" in report
    assert "`guard_precondition`: 1" in report
    assert "thread `1787948143.137669`" in report
    assert "`anthropic_overloaded`" in report
    assert "reference `agent-6932fbda6013`" in report
