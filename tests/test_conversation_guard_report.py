from datetime import datetime, timezone

from scripts import export_sessions


def _event(event_id, timestamp, message):
    return {
        "eventId": event_id,
        "timestamp": timestamp,
        "message": message,
        "logStreamName": "2026/08/12/[$LATEST]abc",
    }


def test_parse_guard_event_with_thread_and_categories():
    event = _event(
        "event-1",
        1_786_563_841_000,
        (
            "[WARNING]\t2026-08-12T19:24:01.123Z\t"
            "a347bc31-426a-4f19-adb3-33ea60bfd280\t"
            "Blocked unsupported job-run interpretation: "
            "thread_ts=1786562460.989019 "
            "violations=recent_run_baseline_required,runtime_comparison_required"
        ),
    )

    parsed = export_sessions.parse_guard_log_event(event)

    assert parsed["event_id"] == "event-1"
    assert parsed["thread_ts"] == "1786562460.989019"
    assert parsed["request_id"] == "a347bc31-426a-4f19-adb3-33ea60bfd280"
    assert parsed["violations"] == [
        "recent_run_baseline_required",
        "runtime_comparison_required",
    ]


def test_parse_legacy_guard_event_without_thread():
    event = _event(
        "event-2",
        1_786_563_716_000,
        (
            "[WARNING]\t2026-08-12T19:21:56.495Z\t"
            "347c6cda-857c-4b1f-bedf-a53d789b0408\t"
            "Blocked unsupported job-run interpretation: "
            "schedule_configuration_required"
        ),
    )

    parsed = export_sessions.parse_guard_log_event(event)

    assert parsed["thread_ts"] is None
    assert parsed["violations"] == ["schedule_configuration_required"]


def test_fetch_guard_events_follows_tokens_and_deduplicates():
    class FakeLogsClient:
        def __init__(self):
            self.calls = []

        def filter_log_events(self, **kwargs):
            self.calls.append(kwargs)
            if "nextToken" not in kwargs:
                return {
                    "events": [
                        _event(
                            "event-1",
                            2000,
                            (
                                "Blocked unsupported job-run interpretation: "
                                "schedule_configuration_required"
                            ),
                        )
                    ],
                    "nextToken": "next",
                }
            return {
                "events": [
                    _event(
                        "event-1",
                        2000,
                        (
                            "Blocked unsupported job-run interpretation: "
                            "schedule_configuration_required"
                        ),
                    ),
                    _event(
                        "event-2",
                        3000,
                        (
                            "Blocked unsupported job-run interpretation: "
                            "recent_run_baseline_required"
                        ),
                    ),
                ],
                "nextToken": "next",
            }

    client = FakeLogsClient()
    events = export_sessions.fetch_guard_events(client, 1000)

    assert len(client.calls) == 2
    assert client.calls[0]["startTime"] == 1000
    assert client.calls[1]["nextToken"] == "next"
    assert [event["event_id"] for event in events] == ["event-2", "event-1"]


def test_guard_report_counts_categories_and_thread_correlation(monkeypatch):
    monkeypatch.setattr(export_sessions, "GUARD_EVENT_LOOKBACK_DAYS", 7)
    events = [
        {
            "timestamp_iso": "2026-08-12T19:24:01+00:00",
            "thread_ts": "1786562460.989019",
            "request_id": "request-1",
            "violations": ["recent_run_baseline_required"],
        },
        {
            "timestamp_iso": "2026-08-12T19:21:56+00:00",
            "thread_ts": None,
            "request_id": "request-2",
            "violations": [
                "schedule_configuration_required",
                "recent_run_baseline_required",
            ],
        },
    ]

    report = export_sessions.render_guard_report(
        events,
        datetime(2026, 8, 12, 19, 25, tzinfo=timezone.utc),
    )

    assert "Guard events: 2" in report
    assert "Events correlated to a Slack thread: 1" in report
    assert "`recent_run_baseline_required`: 2" in report
    assert "thread `1786562460.989019`" in report
    assert "thread unavailable" in report
