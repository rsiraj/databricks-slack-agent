"""Export bot conversations from DynamoDB into a growing local catalog.

Sessions in DynamoDB have a 365-day TTL, so any single scan only sees the last
~365 days of live threads. This script merges each scan into a persistent local
catalog so old conversations are never lost as they expire, and threads are
deduped by ``thread_ts`` as they grow.

Merge policy — KEEP-FULLEST:
  A thread is keyed by ``thread_ts``. When the same thread is already in the
  catalog and also in a fresh scan, we keep whichever copy has MORE messages
  (tie-break: newer ``updated_at``). This protects history: the bot's own
  size-cap and self-heal logic can make a *newer* saved record SHORTER than one
  we captured earlier, and newest-wins would silently drop those older
  messages. Keep-fullest never shrinks a thread we've already archived.

  Known limitation: if a thread both grows at the end AND gets trimmed at the
  front between scans (net fewer messages but with new tail content), the newer
  tail is not merged in — we keep the longer earlier snapshot. Running the
  export regularly and more often than the 365-day TTL avoids this in practice.

Threads in the catalog that are absent from the current scan (already expired)
are left untouched. NOTE: a thread born and expired entirely between two runs
is never seen — run this more often than the TTL to stay complete.

Outputs (all under scripts/exports/, gitignored — they contain real user data):
  catalog.json            - the persistent store: {thread_ts: record}
  sessions_export.json    - flat list of all catalog threads (newest first)
  sessions_export.md      - readable transcripts; tool_result payloads truncated
  sessions_compacted.json - each thread with raw + compacted messages + savings
  compaction_report.md    - per-thread byte-savings table
  guard_events.json       - evidence-guard violations from worker CloudWatch logs
  guard_report.md         - violation counts and recent thread-correlated events
  tool_reliability_events.json - classified tool/Databricks reliability events
  tool_reliability_report.md   - compact outcome and recovery counts
  model_usage_events.json - raw ``model_call`` / ``run_summary`` audit events
  model_usage_report.md   - per (model, effort) calls, tokens, cache hit rate,
                            latency percentiles, estimated cost, outcomes
  feedback_catalog.json   - persistent latest rating per answer and Slack user
  feedback_export.json    - machine-readable feedback joined to conversation IDs
  feedback_export.md      - readable feedback with responder identities omitted
  feedback_summary.md     - rating totals and recent written feedback

Run:
    python3 scripts/export_sessions.py
"""

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3

# Import the compaction functions directly. utils.compaction depends only on the
# stdlib (json, copy), so this pulls in no boto3/anthropic/slack at import time.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from utils.compaction import _prepare_messages_for_api  # noqa: E402
from evals.aggregate import percentile  # noqa: E402  (stdlib-only module)
from evals.pricing import estimate_cost_usd  # noqa: E402  (stdlib-only module)

TABLE_NAME = os.environ.get("SESSIONS_TABLE_NAME", "databricks-slack-agent-sessions")
FEEDBACK_TABLE_NAME = os.environ.get(
    "FEEDBACK_TABLE_NAME",
    "databricks-slack-agent-feedback",
)
REGION = os.environ.get("AWS_REGION", "us-east-1")
WORKER_LOG_GROUP = os.environ.get(
    "WORKER_LOG_GROUP",
    "/aws/lambda/databricks-slack-agent-worker",
)
GUARD_EVENT_LOOKBACK_DAYS = max(1, int(os.environ.get("GUARD_EVENT_LOOKBACK_DAYS", "7")))

OUT_DIR = Path(__file__).parent
EXPORT_DIR = OUT_DIR / "exports"
CATALOG_PATH = EXPORT_DIR / "catalog.json"
JSON_OUT = EXPORT_DIR / "sessions_export.json"
MD_OUT = EXPORT_DIR / "sessions_export.md"
COMPACTED_OUT = EXPORT_DIR / "sessions_compacted.json"
REPORT_OUT = EXPORT_DIR / "compaction_report.md"
GUARD_EVENTS_OUT = EXPORT_DIR / "guard_events.json"
GUARD_REPORT_OUT = EXPORT_DIR / "guard_report.md"
TOOL_RELIABILITY_EVENTS_OUT = EXPORT_DIR / "tool_reliability_events.json"
TOOL_RELIABILITY_REPORT_OUT = EXPORT_DIR / "tool_reliability_report.md"
MODEL_USAGE_EVENTS_OUT = EXPORT_DIR / "model_usage_events.json"
MODEL_USAGE_REPORT_OUT = EXPORT_DIR / "model_usage_report.md"
FEEDBACK_CATALOG_PATH = EXPORT_DIR / "feedback_catalog.json"
FEEDBACK_JSON_OUT = EXPORT_DIR / "feedback_export.json"
FEEDBACK_MD_OUT = EXPORT_DIR / "feedback_export.md"
FEEDBACK_SUMMARY_OUT = EXPORT_DIR / "feedback_summary.md"

# Legacy pre-catalog dump (scripts/ root); seeded into the catalog on first run.
LEGACY_EXPORT = OUT_DIR / "sessions_export.json"

TOOL_RESULT_TRUNCATE = 500
GUARD_LOG_MARKER = "Blocked unsupported job-run interpretation"
TOOL_RELIABILITY_EVENT_NAMES = frozenset(
    {
        "guard_precondition",
        "databricks_transient_failure",
        "databricks_recovery_succeeded",
        "databricks_failure",
        "agent_error",
        "tool_call",
    }
)
# Worker audit events describing model spend. ``model_call`` is emitted once per
# ``messages.create``; ``run_summary`` once per completed agent run.
MODEL_USAGE_EVENT_NAMES = frozenset({"model_call", "run_summary"})
MODEL_CALL_USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
RUN_SUMMARY_USAGE_FIELDS = {
    "input_tokens": "total_input_tokens",
    "cache_creation_input_tokens": "total_cache_creation_input_tokens",
    "cache_read_input_tokens": "total_cache_read_input_tokens",
    "output_tokens": "total_output_tokens",
}
_THREAD_TS_RE = re.compile(r"\bthread_ts=([0-9.]+)")
_VIOLATIONS_RE = re.compile(r"\bviolations=([a-z0-9_,]+)", re.IGNORECASE)
_REQUEST_ID_RE = re.compile(
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DynamoDB scan + parsing
# ---------------------------------------------------------------------------

def scan_all(table) -> list[dict]:
    items: list[dict] = []
    kwargs: dict = {}
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def parse_item(item: dict) -> dict:
    """Turn a raw DynamoDB session item into a catalog record (no provenance yet)."""
    ts = int(item.get("updated_at", 0))
    try:
        messages = json.loads(item["messages"])
    except (KeyError, json.JSONDecodeError):
        messages = []
    return {
        "thread_ts": item.get("thread_ts"),
        "updated_at": ts,
        "updated_at_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
        "message_count": len(messages),
        "messages": messages,
    }


def parse_feedback_item(item: dict) -> dict:
    """Normalize one feedback-table item for catalog storage and rendering."""
    updated_at = int(item.get("updated_at", 0) or 0)
    updated_at_ms = int(item.get("updated_at_ms", 0) or updated_at * 1000)
    return {
        "answer_key": str(item.get("answer_key") or ""),
        "user_id": str(item.get("user_id") or ""),
        "channel_id": str(item.get("channel_id") or ""),
        "thread_ts": str(item.get("thread_ts") or ""),
        "answer_ts": str(item.get("answer_ts") or ""),
        "rating": str(item.get("rating") or ""),
        "comment": str(item.get("comment") or ""),
        "updated_at": updated_at,
        "updated_at_ms": updated_at_ms,
        "updated_at_iso": (
            datetime.fromtimestamp(updated_at, tz=timezone.utc).isoformat()
            if updated_at
            else ""
        ),
    }


# ---------------------------------------------------------------------------
# Evidence-guard events from CloudWatch
# ---------------------------------------------------------------------------

def parse_guard_log_event(event: dict) -> dict | None:
    """Parse one worker warning into a reportable guard event."""
    message = str(event.get("message", ""))
    if GUARD_LOG_MARKER not in message:
        return None

    payload = message.split(GUARD_LOG_MARKER, 1)[1].lstrip(": ")
    thread_match = _THREAD_TS_RE.search(payload)
    violations_match = _VIOLATIONS_RE.search(payload)
    if violations_match:
        violations = [
            value for value in violations_match.group(1).split(",") if value
        ]
    else:
        legacy_payload = payload.strip().split()[0] if payload.strip() else ""
        violations = [
            value
            for value in legacy_payload.split(",")
            if re.fullmatch(r"[a-z0-9_]+", value, flags=re.IGNORECASE)
        ]

    timestamp_ms = int(event.get("timestamp", 0) or 0)
    request_match = _REQUEST_ID_RE.search(message)
    return {
        "event_id": event.get("eventId"),
        "timestamp_ms": timestamp_ms,
        "timestamp_iso": (
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
            if timestamp_ms
            else ""
        ),
        "thread_ts": thread_match.group(1) if thread_match else None,
        "request_id": request_match.group(1) if request_match else None,
        "violations": violations,
        "log_stream_name": event.get("logStreamName"),
    }


def fetch_guard_events(logs_client, start_time_ms: int) -> list[dict]:
    """Fetch and parse all guard warnings in the requested CloudWatch window."""
    kwargs = {
        "logGroupName": WORKER_LOG_GROUP,
        "startTime": start_time_ms,
        "filterPattern": f'"{GUARD_LOG_MARKER}"',
    }
    parsed: dict[str, dict] = {}
    previous_token = None
    while True:
        response = logs_client.filter_log_events(**kwargs)
        for event in response.get("events", []):
            item = parse_guard_log_event(event)
            if item is None:
                continue
            key = str(
                item.get("event_id")
                or f"{item.get('timestamp_ms')}:{item.get('request_id')}:{item.get('violations')}"
            )
            parsed[key] = item

        next_token = response.get("nextToken")
        if not next_token or next_token == previous_token:
            break
        previous_token = next_token
        kwargs["nextToken"] = next_token

    return sorted(parsed.values(), key=lambda item: item.get("timestamp_ms", 0), reverse=True)


def render_guard_report(events: list[dict], exported_at: datetime) -> str:
    """Render a compact review companion for the conversation export."""
    counts = Counter(
        violation
        for event in events
        for violation in event.get("violations", [])
    )
    correlated = sum(bool(event.get("thread_ts")) for event in events)
    lines = [
        "# Job-run evidence guard report",
        "",
        f"_Exported {exported_at.isoformat()} from the last "
        f"{GUARD_EVENT_LOOKBACK_DAYS} day(s)_",
        "",
        f"- Guard events: {len(events)}",
        f"- Events correlated to a Slack thread: {correlated}",
        f"- Events without thread correlation: {len(events) - correlated}",
        "",
        "## Violation counts",
        "",
    ]
    if counts:
        lines.extend(
            f"- `{violation}`: {count}"
            for violation, count in counts.most_common()
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Recent events", ""])
    if not events:
        lines.append("- None")
    for event in events:
        thread = (
            f"thread `{event['thread_ts']}`"
            if event.get("thread_ts")
            else "thread unavailable"
        )
        violations = ", ".join(
            f"`{violation}`" for violation in event.get("violations", [])
        ) or "category unavailable"
        request = (
            f"; request `{event['request_id']}`"
            if event.get("request_id")
            else ""
        )
        lines.append(
            f"- {event.get('timestamp_iso') or 'unknown time'} — {thread} — "
            f"{violations}{request}"
        )
    return "\n".join(lines) + "\n"


def export_guard_events(logs_client, exported_at: datetime) -> list[dict]:
    start_time_ms = int(
        (exported_at.timestamp() - GUARD_EVENT_LOOKBACK_DAYS * 86_400) * 1000
    )
    events = fetch_guard_events(logs_client, start_time_ms)
    GUARD_EVENTS_OUT.write_text(json.dumps(events, indent=2, default=str))
    GUARD_REPORT_OUT.write_text(render_guard_report(events, exported_at))
    print(
        f"Wrote {GUARD_EVENTS_OUT} ({GUARD_EVENTS_OUT.stat().st_size:,} bytes, "
        f"{len(events)} events)"
    )
    print(f"Wrote {GUARD_REPORT_OUT} ({GUARD_REPORT_OUT.stat().st_size:,} bytes)")
    return events


# ---------------------------------------------------------------------------
# Tool reliability events from CloudWatch
# ---------------------------------------------------------------------------

def _audit_payload_from_message(message: str) -> dict | None:
    """Extract an audit payload from plain Lambda or JSON-formatted log output."""
    try:
        outer = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        outer = None

    if isinstance(outer, dict):
        if outer.get("audit_event"):
            return outer
        nested_message = outer.get("message")
        if isinstance(nested_message, str) and nested_message != message:
            nested = _audit_payload_from_message(nested_message)
            if nested is not None:
                return nested

    marker = "audit {"
    marker_index = message.find(marker)
    if marker_index == -1:
        return None
    json_start = marker_index + len("audit ")
    try:
        payload, _ = json.JSONDecoder().raw_decode(message[json_start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_tool_reliability_log_event(event: dict) -> dict | None:
    """Parse one relevant rendered audit event into an export-safe record."""
    payload = _audit_payload_from_message(str(event.get("message", "")))
    if not payload or payload.get("audit_event") not in TOOL_RELIABILITY_EVENT_NAMES:
        return None

    timestamp_ms = int(event.get("timestamp", 0) or 0)
    return {
        **payload,
        "event_id": event.get("eventId"),
        "timestamp_ms": timestamp_ms,
        "timestamp_iso": (
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
            if timestamp_ms
            else ""
        ),
        "log_stream_name": event.get("logStreamName"),
    }


def _fetch_audit_events(logs_client, start_time_ms: int, parser) -> list[dict]:
    """Page through ``audit_event`` log lines, parse, and deduplicate them."""
    kwargs = {
        "logGroupName": WORKER_LOG_GROUP,
        "startTime": start_time_ms,
        "filterPattern": '"audit_event"',
    }
    parsed: dict[str, dict] = {}
    previous_token = None
    while True:
        response = logs_client.filter_log_events(**kwargs)
        for event in response.get("events", []):
            item = parser(event)
            if item is None:
                continue
            key = str(
                item.get("event_id")
                or f"{item.get('timestamp_ms')}:{item.get('audit_event')}:{item.get('thread_ts')}"
            )
            parsed[key] = item

        next_token = response.get("nextToken")
        if not next_token or next_token == previous_token:
            break
        previous_token = next_token
        kwargs["nextToken"] = next_token

    return sorted(parsed.values(), key=lambda item: item.get("timestamp_ms", 0), reverse=True)


def fetch_tool_reliability_events(logs_client, start_time_ms: int) -> list[dict]:
    """Fetch, parse, and deduplicate recent tool-reliability audit events."""
    return _fetch_audit_events(logs_client, start_time_ms, parse_tool_reliability_log_event)


def render_tool_reliability_report(events: list[dict], exported_at: datetime) -> str:
    """Render compact counts plus recent non-success tool reliability events."""
    event_counts = Counter(str(event.get("audit_event")) for event in events)
    guard_events = [
        event for event in events if event.get("audit_event") == "guard_precondition"
    ]
    auto_satisfied_guards = sum(
        event.get("auto_satisfied") is True for event in guard_events
    )
    blocking_guards = len(guard_events) - auto_satisfied_guards
    outcome_counts = Counter(
        str(event.get("outcome") or "unknown")
        for event in events
        if event.get("audit_event") == "tool_call"
    )
    notable = [
        event
        for event in events
        if event.get("audit_event") != "tool_call" or event.get("outcome") != "success"
    ]
    lines = [
        "# Tool reliability report",
        "",
        f"_Exported {exported_at.isoformat()} from the last "
        f"{GUARD_EVENT_LOOKBACK_DAYS} day(s)_",
        "",
        f"- Tool calls: {event_counts.get('tool_call', 0)}",
        f"- Guard preconditions: {event_counts.get('guard_precondition', 0)}",
        f"  - Auto-satisfied metadata preflights: {auto_satisfied_guards}",
        f"  - Blocking guard results: {blocking_guards}",
        f"- Databricks transient failures: {event_counts.get('databricks_transient_failure', 0)}",
        f"- Databricks recoveries: {event_counts.get('databricks_recovery_succeeded', 0)}",
        f"- Final Databricks failures: {event_counts.get('databricks_failure', 0)}",
        f"- Agent errors: {event_counts.get('agent_error', 0)}",
        "",
        "## Tool-call outcomes",
        "",
    ]
    if outcome_counts:
        lines.extend(
            f"- `{outcome}`: {count}"
            for outcome, count in outcome_counts.most_common()
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Recent non-success and recovery events", ""])
    if not notable:
        lines.append("- None")
    for event in notable[:50]:
        event_name = str(event.get("audit_event") or "unknown")
        detail = (
            event.get("outcome")
            or event.get("validation_failure")
            or event.get("error_code")
            or event.get("failure_type")
            or event.get("error_type")
            or event.get("operation")
            or "uncategorized"
        )
        thread = f"; thread `{event['thread_ts']}`" if event.get("thread_ts") else ""
        error_ref = f"; reference `{event['error_id']}`" if event.get("error_id") else ""
        lines.append(
            f"- {event.get('timestamp_iso') or 'unknown time'} — `{event_name}`: "
            f"`{detail}`{thread}{error_ref}"
        )
    return "\n".join(lines) + "\n"


def export_tool_reliability_events(logs_client, exported_at: datetime) -> list[dict]:
    start_time_ms = int(
        (exported_at.timestamp() - GUARD_EVENT_LOOKBACK_DAYS * 86_400) * 1000
    )
    events = fetch_tool_reliability_events(logs_client, start_time_ms)
    TOOL_RELIABILITY_EVENTS_OUT.write_text(json.dumps(events, indent=2, default=str))
    TOOL_RELIABILITY_REPORT_OUT.write_text(
        render_tool_reliability_report(events, exported_at)
    )
    print(
        f"Wrote {TOOL_RELIABILITY_EVENTS_OUT} "
        f"({TOOL_RELIABILITY_EVENTS_OUT.stat().st_size:,} bytes, {len(events)} events)"
    )
    print(
        f"Wrote {TOOL_RELIABILITY_REPORT_OUT} "
        f"({TOOL_RELIABILITY_REPORT_OUT.stat().st_size:,} bytes)"
    )
    return events


# ---------------------------------------------------------------------------
# Model usage (cost / latency) events from CloudWatch
# ---------------------------------------------------------------------------

def parse_model_usage_log_event(event: dict) -> dict | None:
    """Parse one ``model_call`` or ``run_summary`` audit line into a record."""
    payload = _audit_payload_from_message(str(event.get("message", "")))
    if not payload or payload.get("audit_event") not in MODEL_USAGE_EVENT_NAMES:
        return None

    timestamp_ms = int(event.get("timestamp", 0) or 0)
    return {
        **payload,
        "event_id": event.get("eventId"),
        "timestamp_ms": timestamp_ms,
        "timestamp_iso": (
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
            if timestamp_ms
            else ""
        ),
        "log_stream_name": event.get("logStreamName"),
    }


def fetch_model_usage_events(logs_client, start_time_ms: int) -> list[dict]:
    """Fetch, parse, and deduplicate recent model-usage audit events."""
    return _fetch_audit_events(logs_client, start_time_ms, parse_model_usage_log_event)


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _event_model(event: dict) -> str:
    return str(event.get("model") or event.get("requested_model") or "unknown")


def _event_effort(event: dict) -> str:
    return str(event.get("effort") or "unknown")


def _event_datetime(event: dict) -> datetime | None:
    timestamp_ms = _int(event.get("timestamp_ms"))
    if timestamp_ms:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    raw = event.get("audit_timestamp")
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _cache_hit_rate(usage: dict) -> float | None:
    denominator = (
        usage.get("cache_read_input_tokens", 0)
        + usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )
    if denominator <= 0:
        return None
    return round(usage.get("cache_read_input_tokens", 0) / denominator, 4)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _sum_cost(costs: list[float | None]) -> float | None:
    known = [cost for cost in costs if cost is not None]
    return round(sum(known), 6) if known else None


def aggregate_model_usage(events: list[dict], exported_at: datetime) -> dict:
    """Aggregate ``model_call`` and ``run_summary`` events per (model, effort).

    ``month_to_date`` cost sums ``run_summary`` events dated in ``exported_at``'s
    UTC calendar month. Missing pricing yields ``None`` rather than a guess.
    """
    groups: dict[tuple[str, str], dict] = {}

    def group_for(event: dict) -> dict:
        key = (_event_model(event), _event_effort(event))
        if key not in groups:
            groups[key] = {
                "model": key[0],
                "effort": key[1],
                "calls": [],
                "runs": [],
            }
        return groups[key]

    for event in events:
        name = event.get("audit_event")
        if name == "model_call":
            group_for(event)["calls"].append(event)
        elif name == "run_summary":
            group_for(event)["runs"].append(event)

    month_start = exported_at.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )

    rows: list[dict] = []
    for _key, group in sorted(groups.items()):
        calls = group["calls"]
        runs = group["runs"]
        model = group["model"]

        call_usage = {
            field: sum(_int(call.get(field)) for call in calls)
            for field in MODEL_CALL_USAGE_FIELDS
        }
        call_latencies = [
            latency
            for latency in (_float_or_none(call.get("latency_ms")) for call in calls)
            if latency is not None
        ]
        call_costs = [
            estimate_cost_usd(
                _event_model(call),
                **{field: _int(call.get(field)) for field in MODEL_CALL_USAGE_FIELDS},
            )
            for call in calls
        ]

        run_usage_totals = {
            field: sum(_int(run.get(source)) for run in runs)
            for field, source in RUN_SUMMARY_USAGE_FIELDS.items()
        }
        run_latencies = [
            latency
            for latency in (_float_or_none(run.get("total_latency_ms")) for run in runs)
            if latency is not None
        ]
        run_costs = [
            estimate_cost_usd(
                _event_model(run),
                **{field: _int(run.get(source)) for field, source in RUN_SUMMARY_USAGE_FIELDS.items()},
            )
            for run in runs
        ]
        mtd_runs = [
            (run, cost)
            for run, cost in zip(runs, run_costs)
            if (moment := _event_datetime(run)) is not None and moment >= month_start
        ]

        # Cache hit rate prefers per-call usage (exact); falls back to run totals.
        usage_for_rate = call_usage if calls else run_usage_totals
        rows.append(
            {
                "model": model,
                "effort": group["effort"],
                "model_calls": len(calls),
                "runs": len(runs),
                "api_calls_per_run": _mean([float(_int(run.get("api_calls"))) for run in runs]),
                "tool_calls_per_run": _mean([float(_int(run.get("tool_calls"))) for run in runs]),
                "cache_hit_rate": _cache_hit_rate(usage_for_rate),
                "call_usage_totals": call_usage,
                "tokens_per_call": {
                    field: _mean([float(_int(call.get(field))) for call in calls])
                    for field in MODEL_CALL_USAGE_FIELDS
                },
                "tokens_per_run": {
                    field: _mean([float(_int(run.get(source))) for run in runs])
                    for field, source in RUN_SUMMARY_USAGE_FIELDS.items()
                },
                "estimated_cost_total_usd": _sum_cost(run_costs if runs else call_costs),
                "estimated_cost_per_run_usd": (
                    round(_sum_cost(run_costs) / len(runs), 6)
                    if runs and _sum_cost(run_costs) is not None
                    else None
                ),
                "estimated_cost_per_call_usd": (
                    round(_sum_cost(call_costs) / len(calls), 6)
                    if calls and _sum_cost(call_costs) is not None
                    else None
                ),
                "month_to_date": {
                    "runs": len(mtd_runs),
                    "estimated_cost_usd": _sum_cost([cost for _run, cost in mtd_runs]),
                },
                "call_latency_ms": {
                    "mean": _mean(call_latencies),
                    "p50": percentile(call_latencies, 50),
                    "p95": percentile(call_latencies, 95),
                },
                "run_latency_ms": {
                    "mean": _mean(run_latencies),
                    "p50": percentile(run_latencies, 50),
                    "p95": percentile(run_latencies, 95),
                },
                "outcomes": dict(
                    Counter(str(run.get("outcome") or "unknown") for run in runs).most_common()
                ),
                "stop_reasons": dict(
                    Counter(
                        str(call.get("stop_reason") or "unknown") for call in calls
                    ).most_common()
                ),
                "max_tokens_hits": sum(
                    1 for call in calls if str(call.get("stop_reason")) == "max_tokens"
                ),
            }
        )

    return {
        "exported_at": exported_at.isoformat(),
        "lookback_days": GUARD_EVENT_LOOKBACK_DAYS,
        "model_call_events": sum(row["model_calls"] for row in rows),
        "run_summary_events": sum(row["runs"] for row in rows),
        "groups": rows,
    }


def _fmt_num(value, digits: int = 0, prefix: str = "", suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{prefix}{value:,.{digits}f}{suffix}"


def render_model_usage_report(events: list[dict], exported_at: datetime) -> str:
    """Render the per-(model, effort) production usage table."""
    summary = aggregate_model_usage(events, exported_at)
    lines = [
        "# Model usage report",
        "",
        f"_Exported {exported_at.isoformat()} from the last "
        f"{GUARD_EVENT_LOOKBACK_DAYS} day(s)_",
        "",
        f"- `model_call` events: {summary['model_call_events']}",
        f"- `run_summary` events: {summary['run_summary_events']}",
        "- Costs use the list-price snapshot in `evals/pricing.py`; re-verify against "
        "the pricing page before quoting. Unknown model ids show `n/a`.",
        "",
    ]
    if not summary["groups"]:
        lines.extend(
            [
                "## No usage events",
                "",
                "- No `model_call` or `run_summary` audit events were found in the "
                "window. The worker emits them once the model-usage instrumentation is "
                "deployed; until then this report is intentionally empty.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Per model and effort",
            "",
            "| Model | Effort | Runs | Calls | Calls/run | Tool calls/run | Cache hit rate | "
            "Input/run | Cache write/run | Cache read/run | Output/run | Cost/run | "
            "Cost MTD (runs) | Call p50/p95 (ms) | Run p50/p95 (ms) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in summary["groups"]:
        per_run = row["tokens_per_run"]
        rate = row["cache_hit_rate"]
        mtd = row["month_to_date"]
        lines.append(
            f"| `{row['model']}` | {row['effort']} | {row['runs']} | {row['model_calls']} | "
            f"{_fmt_num(row['api_calls_per_run'], 1)} | {_fmt_num(row['tool_calls_per_run'], 1)} | "
            f"{_fmt_num(rate * 100 if rate is not None else None, 1, suffix='%')} | "
            f"{_fmt_num(per_run['input_tokens'])} | "
            f"{_fmt_num(per_run['cache_creation_input_tokens'])} | "
            f"{_fmt_num(per_run['cache_read_input_tokens'])} | "
            f"{_fmt_num(per_run['output_tokens'])} | "
            f"{_fmt_num(row['estimated_cost_per_run_usd'], 4, '$')} | "
            f"{_fmt_num(mtd['estimated_cost_usd'], 2, '$')} ({mtd['runs']}) | "
            f"{_fmt_num(row['call_latency_ms']['p50'])}/{_fmt_num(row['call_latency_ms']['p95'])} | "
            f"{_fmt_num(row['run_latency_ms']['p50'])}/{_fmt_num(row['run_latency_ms']['p95'])} |"
        )

    for row in summary["groups"]:
        lines.extend(["", f"### `{row['model']}` / {row['effort']}", ""])
        lines.append(
            f"- Estimated cost per call: {_fmt_num(row['estimated_cost_per_call_usd'], 4, '$')}; "
            f"total in window: {_fmt_num(row['estimated_cost_total_usd'], 2, '$')}"
        )
        lines.append(f"- Calls that hit `max_tokens`: {row['max_tokens_hits']}")
        lines.append("- Run outcomes:")
        if row["outcomes"]:
            lines.extend(f"  - `{outcome}`: {count}" for outcome, count in row["outcomes"].items())
        else:
            lines.append("  - None")
        lines.append("- Stop reasons:")
        if row["stop_reasons"]:
            lines.extend(
                f"  - `{reason}`: {count}" for reason, count in row["stop_reasons"].items()
            )
        else:
            lines.append("  - None")
    return "\n".join(lines) + "\n"


def export_model_usage_events(logs_client, exported_at: datetime) -> list[dict]:
    start_time_ms = int(
        (exported_at.timestamp() - GUARD_EVENT_LOOKBACK_DAYS * 86_400) * 1000
    )
    events = fetch_model_usage_events(logs_client, start_time_ms)
    MODEL_USAGE_EVENTS_OUT.write_text(json.dumps(events, indent=2, default=str))
    MODEL_USAGE_REPORT_OUT.write_text(render_model_usage_report(events, exported_at))
    print(
        f"Wrote {MODEL_USAGE_EVENTS_OUT} "
        f"({MODEL_USAGE_EVENTS_OUT.stat().st_size:,} bytes, {len(events)} events)"
    )
    print(
        f"Wrote {MODEL_USAGE_REPORT_OUT} "
        f"({MODEL_USAGE_REPORT_OUT.stat().st_size:,} bytes)"
    )
    return events


# ---------------------------------------------------------------------------
# Catalog load / merge (keep-fullest) / save
# ---------------------------------------------------------------------------

def load_catalog() -> dict[str, dict]:
    """Load the persistent catalog, or seed it from the legacy dump on first run."""
    if CATALOG_PATH.exists():
        return json.loads(CATALOG_PATH.read_text())

    if LEGACY_EXPORT.exists():
        print(f"No catalog yet; seeding from legacy export {LEGACY_EXPORT}")
        seeded: dict[str, dict] = {}
        now_iso = datetime.now(timezone.utc).isoformat()
        for rec in json.loads(LEGACY_EXPORT.read_text()):
            ts = rec.get("thread_ts")
            if not ts:
                continue
            rec.setdefault("message_count", len(rec.get("messages", [])))
            rec["first_seen_iso"] = now_iso
            rec["last_seen_iso"] = now_iso
            seeded[ts] = rec
        return seeded

    return {}


def merge_record(catalog: dict[str, dict], rec: dict, now_iso: str) -> str:
    """Merge one scanned record into the catalog (keep-fullest). Returns the action."""
    ts = rec.get("thread_ts")
    if not ts:
        return "skipped"

    existing = catalog.get(ts)
    if existing is None:
        rec["first_seen_iso"] = now_iso
        rec["last_seen_iso"] = now_iso
        catalog[ts] = rec
        return "added"

    # Seen live again regardless of whether content changes.
    existing["last_seen_iso"] = now_iso

    new_n = rec.get("message_count", len(rec.get("messages", [])))
    old_n = existing.get("message_count", len(existing.get("messages", [])))
    newer = rec.get("updated_at", 0) > existing.get("updated_at", 0)

    # Keep-fullest: take the new copy only if it has more messages, or the same
    # count but a newer timestamp. Never replace a longer archived thread.
    if new_n > old_n or (new_n == old_n and newer):
        rec["first_seen_iso"] = existing.get("first_seen_iso", now_iso)
        rec["last_seen_iso"] = now_iso
        catalog[ts] = rec
        return "updated"

    return "unchanged"


def save_catalog(catalog: dict[str, dict]) -> None:
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, default=str))
    print(f"Wrote {CATALOG_PATH} ({CATALOG_PATH.stat().st_size:,} bytes, {len(catalog)} threads)")


def _feedback_catalog_key(record: dict) -> str:
    answer_key = str(record.get("answer_key") or "")
    user_id = str(record.get("user_id") or "")
    return f"{answer_key}|{user_id}" if answer_key and user_id else ""


def load_feedback_catalog() -> dict[str, dict]:
    if FEEDBACK_CATALOG_PATH.exists():
        return json.loads(FEEDBACK_CATALOG_PATH.read_text())
    return {}


def merge_feedback_record(
    catalog: dict[str, dict],
    record: dict,
    now_iso: str,
) -> str:
    """Keep the latest rating for each answer/user pair."""
    key = _feedback_catalog_key(record)
    if not key:
        return "skipped"

    existing = catalog.get(key)
    if existing is None:
        record["first_seen_iso"] = now_iso
        record["last_seen_iso"] = now_iso
        catalog[key] = record
        return "added"

    existing["last_seen_iso"] = now_iso
    record_order = record.get(
        "updated_at_ms",
        record.get("updated_at", 0) * 1000,
    )
    existing_order = existing.get(
        "updated_at_ms",
        existing.get("updated_at", 0) * 1000,
    )
    if record_order > existing_order:
        record["first_seen_iso"] = existing.get("first_seen_iso", now_iso)
        record["last_seen_iso"] = now_iso
        catalog[key] = record
        return "updated"
    return "unchanged"


def save_feedback_catalog(catalog: dict[str, dict]) -> None:
    FEEDBACK_CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, default=str)
    )
    print(
        f"Wrote {FEEDBACK_CATALOG_PATH} "
        f"({FEEDBACK_CATALOG_PATH.stat().st_size:,} bytes, "
        f"{len(catalog)} response(s))"
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…\n[truncated, {len(text):,} chars total]"


def render_block(block: dict, role: str) -> str:
    btype = block.get("type")

    if btype == "text":
        return block.get("text", "").strip()

    if btype == "tool_use":
        name = block.get("name", "?")
        input_repr = json.dumps(block.get("input", {}), indent=2, default=str)
        return f"**→ tool_use: `{name}`**\n```json\n{input_repr}\n```"

    if btype == "tool_result":
        is_error = block.get("is_error", False)
        content = block.get("content", "")
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
                else:
                    parts.append(json.dumps(c, default=str))
            content = "\n".join(parts)
        elif not isinstance(content, str):
            content = json.dumps(content, default=str)

        if is_error:
            return f"**← tool_result (ERROR):**\n```\n{content}\n```"
        return f"**← tool_result:**\n```\n{truncate(content, TOOL_RESULT_TRUNCATE)}\n```"

    if btype == "provider_reasoning":
        # Opaque reasoning replayed by a non-Anthropic provider (see
        # utils/llm/types.py); the payload is encrypted/provider-specific.
        provider = block.get("provider", "?")
        size = len(json.dumps(block.get("payload", {}), default=str))
        return f"_(provider reasoning from {provider}, {size:,} chars, not shown)_"

    return f"_(unknown block type: {btype})_"


def render_message(msg: dict) -> str:
    role = msg.get("role", "?")
    content = msg.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return ""

    rendered = [render_block(b, role) for b in content]
    rendered = [r for r in rendered if r]
    if not rendered:
        return ""
    body = "\n\n".join(rendered)
    return f"### {role}\n\n{body}"


def render_thread(thread_ts: str, updated_at_iso: str, messages: list[dict]) -> str:
    out = [f"## Thread `{thread_ts}` — {updated_at_iso}", f"_{len(messages)} messages_", ""]
    for msg in messages:
        rendered = render_message(msg)
        if rendered:
            out.append(rendered)
            out.append("")
    out.append("---\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Compaction analysis
# ---------------------------------------------------------------------------

def _json_size(obj) -> int:
    return len(json.dumps(obj, default=str).encode("utf-8"))


def _count_blocks(messages: list[dict], block_type: str) -> int:
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            count += sum(1 for b in content if b.get("type") == block_type)
    return count


def _build_compaction_outputs(parsed: list[dict]) -> None:
    """Write sessions_compacted.json and compaction_report.md from the catalog."""
    report_lines = [
        "# Compaction report",
        "",
        "Per-thread size of the raw message list vs the compacted version that",
        "would be sent to the Anthropic API.",
        "",
        "| Thread | Updated | Msgs | Thinking blks | Tool results | Raw bytes | Compacted bytes | Saved | % saved |",
        "|--------|---------|------|---------------|--------------|-----------|-----------------|-------|---------|",
    ]

    total_raw = 0
    total_compact = 0
    compacted_threads: list[dict] = []

    for thread in parsed:
        messages = thread.get("messages", [])
        if not messages:
            continue

        thread_ts = thread.get("thread_ts", "?")
        updated = thread.get("updated_at_iso", "?")

        raw_bytes = _json_size(messages)
        compacted = _prepare_messages_for_api(messages)
        compact_bytes = _json_size(compacted)
        saved = raw_bytes - compact_bytes

        total_raw += raw_bytes
        total_compact += compact_bytes
        pct = (saved / raw_bytes * 100) if raw_bytes > 0 else 0

        report_lines.append(
            f"| `{thread_ts}` | {updated[:10]} | {len(messages)} | "
            f"{_count_blocks(messages, 'thinking')} | "
            f"{_count_blocks(messages, 'tool_result')} | "
            f"{raw_bytes:,} | {compact_bytes:,} | {saved:,} | {pct:.1f}% |"
        )

        compacted_threads.append({
            "thread_ts": thread_ts,
            "updated_at_iso": updated,
            "message_count": len(messages),
            "raw_bytes": raw_bytes,
            "compact_bytes": compact_bytes,
            "saved_bytes": saved,
            "saved_pct": round(pct, 1),
            "raw_messages": messages,
            "compacted_messages": compacted,
        })

    total_saved = total_raw - total_compact
    total_pct = (total_saved / total_raw * 100) if total_raw > 0 else 0

    report_lines.extend([
        "",
        "## Totals",
        "",
        f"- **Raw total**: {total_raw:,} bytes",
        f"- **Compacted total**: {total_compact:,} bytes",
        f"- **Saved**: {total_saved:,} bytes ({total_pct:.1f}%)",
    ])

    COMPACTED_OUT.write_text(json.dumps(compacted_threads, indent=2, default=str))
    print(f"Wrote {COMPACTED_OUT} ({COMPACTED_OUT.stat().st_size:,} bytes)")

    REPORT_OUT.write_text("\n".join(report_lines))
    print(f"Wrote {REPORT_OUT} ({REPORT_OUT.stat().st_size:,} bytes)")

    print(f"\nCompaction: {total_raw:,} → {total_compact:,} bytes "
          f"(saved {total_saved:,}, {total_pct:.1f}%)")


# ---------------------------------------------------------------------------
# Output regeneration (from the full catalog)
# ---------------------------------------------------------------------------

def regenerate_outputs(catalog: dict[str, dict]) -> None:
    parsed = sorted(catalog.values(), key=lambda r: r.get("updated_at", 0), reverse=True)

    JSON_OUT.write_text(json.dumps(parsed, indent=2, default=str))
    print(f"Wrote {JSON_OUT} ({JSON_OUT.stat().st_size:,} bytes)")

    sections = [
        "# Sessions export",
        f"_{len(parsed)} threads, exported {datetime.now(timezone.utc).isoformat()}_",
        f"_tool_result content truncated to {TOOL_RESULT_TRUNCATE} chars; everything else verbatim_",
        "",
    ]
    for thread in parsed:
        if not thread.get("messages"):
            continue
        sections.append(
            render_thread(thread.get("thread_ts"), thread.get("updated_at_iso", ""), thread["messages"])
        )
    MD_OUT.write_text("\n".join(sections))
    print(f"Wrote {MD_OUT} ({MD_OUT.stat().st_size:,} bytes)")

    _build_compaction_outputs(parsed)


def _feedback_label(rating: str) -> str:
    return {
        "helpful": "Helpful",
        "somewhat_helpful": "Somewhat Helpful",
        "needs_fix": "Needs a Fix",
    }.get(rating, rating or "Unknown")


def _quote_feedback(comment: str) -> list[str]:
    return [f"> {line}" for line in comment.splitlines() if line.strip()]


def render_feedback_export(records: list[dict], exported_at: datetime) -> str:
    lines = [
        "# Answer feedback export",
        "",
        f"_{len(records)} response(s), exported {exported_at.isoformat()}_",
        "_Responder identities are intentionally omitted from this readable export._",
        "",
    ]
    current_thread = None
    for record in records:
        thread_ts = record.get("thread_ts") or "unavailable"
        if thread_ts != current_thread:
            lines.extend([f"## Thread `{thread_ts}`", ""])
            current_thread = thread_ts
        lines.append(
            f"- Answer `{record.get('answer_ts') or 'unavailable'}` — "
            f"**{_feedback_label(record.get('rating', ''))}** — "
            f"{record.get('updated_at_iso') or 'unknown time'}"
        )
        if record.get("comment"):
            lines.extend(_quote_feedback(record["comment"]))
        lines.append("")
    if not records:
        lines.append("- No feedback exported.")
    return "\n".join(lines).rstrip() + "\n"


def render_feedback_summary(records: list[dict], exported_at: datetime) -> str:
    counts = Counter(record.get("rating") or "unknown" for record in records)
    answers = {record.get("answer_key") for record in records if record.get("answer_key")}
    threads = {record.get("thread_ts") for record in records if record.get("thread_ts")}
    written = [record for record in records if record.get("comment")]
    lines = [
        "# Answer feedback summary",
        "",
        f"_Exported {exported_at.isoformat()}_",
        "",
        f"- Responses: {len(records)}",
        f"- Answers rated: {len(answers)}",
        f"- Threads represented: {len(threads)}",
        f"- Written comments: {len(written)}",
        "",
        "## Ratings",
        "",
    ]
    for rating in ("helpful", "somewhat_helpful", "needs_fix", "unknown"):
        if counts.get(rating):
            lines.append(f"- {_feedback_label(rating)}: {counts[rating]}")
    if not counts:
        lines.append("- None")

    lines.extend(["", "## Recent written feedback", ""])
    for record in written[:50]:
        lines.append(
            f"- Thread `{record.get('thread_ts') or 'unavailable'}`, "
            f"answer `{record.get('answer_ts') or 'unavailable'}` — "
            f"**{_feedback_label(record.get('rating', ''))}**"
        )
        lines.extend(_quote_feedback(record["comment"]))
    if not written:
        lines.append("- None")
    return "\n".join(lines).rstrip() + "\n"


def regenerate_feedback_outputs(
    feedback_catalog: dict[str, dict],
    session_catalog: dict[str, dict],
    exported_at: datetime,
) -> None:
    records = sorted(
        feedback_catalog.values(),
        key=lambda record: record.get(
            "updated_at_ms",
            record.get("updated_at", 0) * 1000,
        ),
        reverse=True,
    )
    correlated = [
        {
            **record,
            "conversation_exported": record.get("thread_ts") in session_catalog,
        }
        for record in records
    ]
    FEEDBACK_JSON_OUT.write_text(
        json.dumps(correlated, indent=2, default=str)
    )
    FEEDBACK_MD_OUT.write_text(render_feedback_export(correlated, exported_at))
    FEEDBACK_SUMMARY_OUT.write_text(
        render_feedback_summary(correlated, exported_at)
    )
    print(
        f"Wrote {FEEDBACK_JSON_OUT} "
        f"({FEEDBACK_JSON_OUT.stat().st_size:,} bytes)"
    )
    print(
        f"Wrote {FEEDBACK_MD_OUT} "
        f"({FEEDBACK_MD_OUT.stat().st_size:,} bytes)"
    )
    print(
        f"Wrote {FEEDBACK_SUMMARY_OUT} "
        f"({FEEDBACK_SUMMARY_OUT.stat().st_size:,} bytes)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog()
    feedback_catalog = load_feedback_catalog()
    print(f"Catalog starts with {len(catalog)} thread(s)")
    print(f"Feedback catalog starts with {len(feedback_catalog)} response(s)")

    boto3.setup_default_session(region_name=REGION)
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(TABLE_NAME)
    print(f"Scanning {TABLE_NAME} in {REGION}...")
    items = scan_all(table)
    print(f"Scan found {len(items)} live thread(s)")

    exported_at = datetime.now(timezone.utc)
    now_iso = exported_at.isoformat()
    stats = {"added": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for item in items:
        stats[merge_record(catalog, parse_item(item), now_iso)] += 1

    print(
        f"Merge: +{stats['added']} new, {stats['updated']} grew, "
        f"{stats['unchanged']} unchanged → catalog now {len(catalog)} thread(s)"
    )

    save_catalog(catalog)
    regenerate_outputs(catalog)

    feedback_table = dynamodb.Table(FEEDBACK_TABLE_NAME)
    print(f"Scanning {FEEDBACK_TABLE_NAME} in {REGION}...")
    feedback_items = scan_all(feedback_table)
    print(f"Scan found {len(feedback_items)} live feedback response(s)")
    feedback_stats = {
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
    }
    for item in feedback_items:
        action = merge_feedback_record(
            feedback_catalog,
            parse_feedback_item(item),
            now_iso,
        )
        feedback_stats[action] += 1
    print(
        f"Feedback merge: +{feedback_stats['added']} new, "
        f"{feedback_stats['updated']} revised, "
        f"{feedback_stats['unchanged']} unchanged → catalog now "
        f"{len(feedback_catalog)} response(s)"
    )
    save_feedback_catalog(feedback_catalog)
    regenerate_feedback_outputs(feedback_catalog, catalog, exported_at)

    print(
        f"Scanning {WORKER_LOG_GROUP} for evidence-guard events from the last "
        f"{GUARD_EVENT_LOOKBACK_DAYS} day(s)..."
    )
    logs_client = boto3.client("logs")
    export_guard_events(logs_client, exported_at)
    print(
        f"Scanning {WORKER_LOG_GROUP} for tool-reliability events from the last "
        f"{GUARD_EVENT_LOOKBACK_DAYS} day(s)..."
    )
    export_tool_reliability_events(logs_client, exported_at)
    print(
        f"Scanning {WORKER_LOG_GROUP} for model-usage events from the last "
        f"{GUARD_EVENT_LOOKBACK_DAYS} day(s)..."
    )
    export_model_usage_events(logs_client, exported_at)


if __name__ == "__main__":
    main()
