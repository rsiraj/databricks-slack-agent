from datetime import datetime, timezone

from scripts import export_sessions


NOW = datetime(2026, 8, 18, 19, 30, tzinfo=timezone.utc)


def _record(user_id, rating, updated_at, comment="", updated_at_ms=None):
    record = {
        "answer_key": "C123#answer-1",
        "user_id": user_id,
        "channel_id": "C123",
        "thread_ts": "thread-1",
        "answer_ts": "answer-1",
        "rating": rating,
        "comment": comment,
        "updated_at": updated_at,
        "updated_at_iso": datetime.fromtimestamp(
            updated_at,
            tz=timezone.utc,
        ).isoformat(),
    }
    if updated_at_ms is not None:
        record["updated_at_ms"] = updated_at_ms
    return record


def test_feedback_catalog_preserves_multiple_raters_and_latest_revision():
    catalog = {}
    now_iso = NOW.isoformat()

    assert (
        export_sessions.merge_feedback_record(
            catalog,
            _record("U1", "helpful", 100),
            now_iso,
        )
        == "added"
    )
    assert (
        export_sessions.merge_feedback_record(
            catalog,
            _record("U2", "needs_fix", 110, "Wrong source."),
            now_iso,
        )
        == "added"
    )
    assert (
        export_sessions.merge_feedback_record(
            catalog,
            _record("U1", "somewhat_helpful", 120, "Show SQL."),
            now_iso,
        )
        == "updated"
    )

    assert len(catalog) == 2
    u1 = catalog["C123#answer-1|U1"]
    assert u1["rating"] == "somewhat_helpful"
    assert u1["comment"] == "Show SQL."


def test_feedback_catalog_uses_milliseconds_for_same_second_revision():
    catalog = {}
    now_iso = NOW.isoformat()
    export_sessions.merge_feedback_record(
        catalog,
        _record("U1", "helpful", 100, updated_at_ms=100_100),
        now_iso,
    )

    action = export_sessions.merge_feedback_record(
        catalog,
        _record(
            "U1",
            "needs_fix",
            100,
            "Changed my mind.",
            updated_at_ms=100_900,
        ),
        now_iso,
    )

    assert action == "updated"
    assert catalog["C123#answer-1|U1"]["rating"] == "needs_fix"


def test_readable_feedback_exports_omit_responder_identity():
    records = [
        _record("U-sensitive", "helpful", 100),
        _record(
            "U-other",
            "needs_fix",
            110,
            "The metric definition is wrong.",
        ),
    ]

    detail = export_sessions.render_feedback_export(records, NOW)
    summary = export_sessions.render_feedback_summary(records, NOW)

    assert "U-sensitive" not in detail
    assert "U-other" not in detail
    assert "Needs a Fix" in detail
    assert "The metric definition is wrong." in detail
    assert "Responses: 2" in summary
    assert "Answers rated: 1" in summary
    assert "Written comments: 1" in summary
