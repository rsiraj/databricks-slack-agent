"""Production model-usage report built from ``model_call`` / ``run_summary`` audit events."""

import json
from datetime import datetime, timezone

from evals.pricing import estimate_cost_usd
from scripts import export_sessions


EXPORTED_AT = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
AUG_31 = int(datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
SEP_02 = int(datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
SEP_03 = int(datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _log_event(event_id, timestamp, payload):
    return {
        "eventId": event_id,
        "timestamp": timestamp,
        "message": f"[INFO]\t2026-09-03T12:00:00.000Z\trequest-id\taudit {json.dumps(payload)}",
        "logStreamName": "2026/09/03/[$LATEST]abc",
    }


def _model_call(ts, *, model="claude-sonnet-4-6", effort="high", latency=1000, stop="tool_use",
                input_tokens=1000, cache_write=0, cache_read=30000, output=500, iteration=0):
    return {
        "audit_event": "model_call",
        "model": model,
        "requested_model": model,
        "effort": effort,
        "iteration": iteration,
        "latency_ms": latency,
        "stop_reason": stop,
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output,
        "max_tokens": 16000,
        "timestamp_ms": ts,
    }


def _run_summary(ts, *, model="claude-sonnet-4-6", effort="high", api_calls=3, tool_calls=2,
                 total_latency=4000, outcome="delivered", input_tokens=3000, cache_write=1000,
                 cache_read=90000, output=1500):
    return {
        "audit_event": "run_summary",
        "model": model,
        "effort": effort,
        "api_calls": api_calls,
        "tool_calls": tool_calls,
        "total_latency_ms": total_latency,
        "total_input_tokens": input_tokens,
        "total_cache_creation_input_tokens": cache_write,
        "total_cache_read_input_tokens": cache_read,
        "total_output_tokens": output,
        "outcome": outcome,
        "timestamp_ms": ts,
    }


def test_parse_model_usage_events_from_rendered_audit_lines():
    call = _log_event(
        "e1",
        SEP_03,
        {
            "audit_event": "model_call",
            "model": "claude-sonnet-4-6",
            "requested_model": "claude-sonnet-4-6",
            "effort": "high",
            "iteration": 2,
            "latency_ms": 1234.5,
            "stop_reason": "end_turn",
            "input_tokens": 12,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 45000,
            "output_tokens": 800,
            "max_tokens": 16000,
        },
    )
    summary = _log_event(
        "e2",
        SEP_03,
        json.loads(json.dumps(_run_summary(SEP_03))) | {"thread_ts": "1.2"},
    )
    unrelated = _log_event("e3", SEP_03, {"audit_event": "tool_call", "outcome": "success"})

    parsed_call = export_sessions.parse_model_usage_log_event(call)
    parsed_summary = export_sessions.parse_model_usage_log_event(summary)

    assert parsed_call["audit_event"] == "model_call"
    assert parsed_call["stop_reason"] == "end_turn"
    assert parsed_call["cache_read_input_tokens"] == 45000
    assert parsed_call["timestamp_ms"] == SEP_03
    assert parsed_call["timestamp_iso"].startswith("2026-09-03T12:00:00")
    assert parsed_summary["audit_event"] == "run_summary"
    assert parsed_summary["outcome"] == "delivered"
    assert export_sessions.parse_model_usage_log_event(unrelated) is None
    assert export_sessions.parse_model_usage_log_event({"message": "audit {bad"}) is None


def test_fetch_model_usage_events_paginates_and_filters_on_audit_marker():
    events = [
        _log_event("e1", 1000, _model_call(1000)),
        _log_event("e2", 2000, {"audit_event": "tool_call"}),
        _log_event("e3", 3000, _run_summary(3000)),
    ]

    class FakeLogs:
        def __init__(self):
            self.calls = []

        def filter_log_events(self, **kwargs):
            self.calls.append(kwargs)
            if "nextToken" not in kwargs:
                return {"events": events[:2], "nextToken": "n"}
            return {"events": events, "nextToken": "n"}

    logs = FakeLogs()
    parsed = export_sessions.fetch_model_usage_events(logs, 0)

    assert len(logs.calls) == 2
    assert logs.calls[0]["filterPattern"] == '"audit_event"'
    assert [event["event_id"] for event in parsed] == ["e3", "e1"]


def test_aggregate_model_usage_per_model_and_effort():
    events = [
        _model_call(SEP_03, latency=1000, stop="tool_use", iteration=0),
        _model_call(SEP_03, latency=2000, stop="tool_use", iteration=1),
        _model_call(SEP_03, latency=3000, stop="end_turn", iteration=2, cache_write=1000, cache_read=59000),
        _run_summary(SEP_03, total_latency=6000),
        _run_summary(SEP_02, total_latency=8000, outcome="max_tokens_truncated", api_calls=5),
        _run_summary(AUG_31, total_latency=2000),  # previous month: excluded from MTD
        _model_call(SEP_03, model="claude-sonnet-5", effort="medium", latency=500, stop="end_turn"),
        _run_summary(SEP_03, model="claude-sonnet-5", effort="medium", api_calls=1, total_latency=500),
        _model_call(SEP_03, model="claude-mystery", effort="low", latency=10),
    ]

    summary = export_sessions.aggregate_model_usage(events, EXPORTED_AT)

    assert summary["model_call_events"] == 5
    assert summary["run_summary_events"] == 4
    by_key = {(row["model"], row["effort"]): row for row in summary["groups"]}
    assert set(by_key) == {
        ("claude-sonnet-4-6", "high"),
        ("claude-sonnet-5", "medium"),
        ("claude-mystery", "low"),
    }

    sonnet = by_key[("claude-sonnet-4-6", "high")]
    assert sonnet["model_calls"] == 3
    assert sonnet["runs"] == 3
    assert sonnet["api_calls_per_run"] == round((3 + 5 + 3) / 3, 3)
    assert sonnet["tool_calls_per_run"] == 2.0
    # cache_read / (cache_read + input + cache_write) over the three calls
    assert sonnet["cache_hit_rate"] == round(119000 / (119000 + 3000 + 1000), 4)
    assert sonnet["tokens_per_call"]["output_tokens"] == 500.0
    assert sonnet["tokens_per_run"] == {
        "input_tokens": 3000.0,
        "cache_creation_input_tokens": 1000.0,
        "cache_read_input_tokens": 90000.0,
        "output_tokens": 1500.0,
    }
    per_run_cost = estimate_cost_usd(
        "claude-sonnet-4-6",
        input_tokens=3000,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=90000,
        output_tokens=1500,
    )
    assert sonnet["estimated_cost_per_run_usd"] == per_run_cost
    assert sonnet["estimated_cost_total_usd"] == round(per_run_cost * 3, 6)
    assert sonnet["month_to_date"] == {"runs": 2, "estimated_cost_usd": round(per_run_cost * 2, 6)}
    assert sonnet["call_latency_ms"] == {"mean": 2000.0, "p50": 2000.0, "p95": 3000.0}
    assert sonnet["run_latency_ms"] == {"mean": round(16000 / 3, 3), "p50": 6000.0, "p95": 8000.0}
    assert sonnet["outcomes"] == {"delivered": 2, "max_tokens_truncated": 1}
    assert sonnet["stop_reasons"] == {"tool_use": 2, "end_turn": 1}
    assert sonnet["max_tokens_hits"] == 0

    mystery = by_key[("claude-mystery", "low")]
    assert mystery["runs"] == 0
    assert mystery["estimated_cost_per_call_usd"] is None
    assert mystery["estimated_cost_total_usd"] is None
    assert mystery["api_calls_per_run"] is None
    assert mystery["run_latency_ms"] == {"mean": None, "p50": None, "p95": None}


def test_aggregate_uses_run_totals_for_cache_rate_when_no_model_calls():
    summary = export_sessions.aggregate_model_usage([_run_summary(SEP_03)], EXPORTED_AT)

    row = summary["groups"][0]
    assert row["model_calls"] == 0
    assert row["cache_hit_rate"] == round(90000 / (90000 + 3000 + 1000), 4)
    assert row["call_latency_ms"]["p50"] is None


def test_aggregate_falls_back_to_audit_timestamp_and_requested_model():
    event = _run_summary(0)
    event.pop("timestamp_ms")
    event["audit_timestamp"] = "2026-09-03T12:00:00+00:00"
    event["model"] = None
    event["requested_model"] = "claude-sonnet-4-6"

    summary = export_sessions.aggregate_model_usage([event], EXPORTED_AT)

    row = summary["groups"][0]
    assert row["model"] == "claude-sonnet-4-6"
    assert row["month_to_date"]["runs"] == 1


def test_render_model_usage_report_with_events(monkeypatch):
    monkeypatch.setattr(export_sessions, "GUARD_EVENT_LOOKBACK_DAYS", 7)
    events = [
        _model_call(SEP_03, stop="max_tokens"),
        _run_summary(SEP_03, outcome="delivered"),
        _model_call(SEP_03, model="claude-mystery", effort="low"),
    ]

    report = export_sessions.render_model_usage_report(events, EXPORTED_AT)

    assert "# Model usage report" in report
    assert "`model_call` events: 2" in report
    assert "`run_summary` events: 1" in report
    assert "| `claude-sonnet-4-6` | high | 1 | 1 | 3.0 | 2.0 |" in report
    assert "| `claude-mystery` | low | 0 | 1 | n/a | n/a |" in report
    assert "Calls that hit `max_tokens`: 1" in report
    assert "- `delivered`: 1" in report
    assert "- `max_tokens`: 1" in report
    assert "re-verify against the pricing page" in report
    assert "$n/a" not in report


def test_render_model_usage_report_handles_zero_events(monkeypatch):
    monkeypatch.setattr(export_sessions, "GUARD_EVENT_LOOKBACK_DAYS", 7)

    report = export_sessions.render_model_usage_report([], EXPORTED_AT)

    assert "`model_call` events: 0" in report
    assert "## No usage events" in report
    assert "intentionally empty" in report
    assert "| `" not in report


def test_export_model_usage_events_writes_both_files(tmp_path, monkeypatch):
    monkeypatch.setattr(export_sessions, "MODEL_USAGE_EVENTS_OUT", tmp_path / "events.json")
    monkeypatch.setattr(export_sessions, "MODEL_USAGE_REPORT_OUT", tmp_path / "report.md")

    class FakeLogs:
        def filter_log_events(self, **kwargs):
            return {"events": [_log_event("e1", SEP_03, _run_summary(SEP_03))]}

    events = export_sessions.export_model_usage_events(FakeLogs(), EXPORTED_AT)

    assert len(events) == 1
    assert json.loads((tmp_path / "events.json").read_text())[0]["audit_event"] == "run_summary"
    assert "# Model usage report" in (tmp_path / "report.md").read_text()
