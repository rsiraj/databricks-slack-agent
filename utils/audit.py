"""Structured audit logging for security-relevant events.

The ASR review asked for "user identity, Slack channel context, prompts,
query execution, Databricks job actions, GitHub interactions, and
operational errors" to be captured in an auditable form, with ~1 year
retention. Retention is set on the CloudWatch log groups; this module is
how events get emitted.

Output shape — one JSON line per audit event, riding on the same
python-json-logger handler that the rest of the app uses, so events
land in the existing CloudWatch log groups alongside diagnostic logs but
are trivially filterable by the `audit_event` field:

    fields @timestamp, audit_event, user_id, channel_id, tool_name
    | filter audit_event = "tool_call"
    | sort @timestamp desc

Ambient context — `set_context(...)` populates `user_id`, `channel_id`,
and `thread_ts` for the current Lambda invocation via contextvars. Once
set, every subsequent `audit(...)` call in the same invocation auto-
includes them, so call sites deep in tool code (databricks_tool,
github_tool) don't need to plumb the identifiers through their
signatures. Lambda runs each invocation as a single Python thread, so
contextvars scope cleanly to the invocation lifetime.

Events emitted today (see README "Security controls" for the full
catalogue):

  mention_received       — every app_mention before any agent work
  tool_call              — every tool dispatch in the agent loop
  channel_access_denied  — channel allowlist or privacy check rejected
  sql_rejected           — SQL guard rejected a statement pre-flight
  sql_repaired           — a safe semantics-preserving SQL correction was applied
  job_triggered          — databricks_run_job succeeded
  job_cancelled          — databricks_cancel_run succeeded
  agent_error            — unhandled exception in the agent loop
  answer_delivery_failed — Slack did not confirm every final-answer part
  rate_limit_blocked     — Item 5 (rate limits)
  guard_precondition     — a tool call was blocked pending safer input/metadata
  databricks_transient_failure  — a retryable Databricks request attempt failed
  databricks_recovery_succeeded — a later Databricks request attempt succeeded
  databricks_failure     — a Databricks tool call ended in failure
  final_response_guard_revision — an unsupported draft received a private revision
  final_response_guard_fallback — an unsupported draft was replaced safely
  model_call             — one model call: provider, model, latency, usage, stop_reason
  run_summary            — per-run totals (api/tool calls, tokens, latency, outcome, provider)
  agent_deadline_reached — the loop stopped early to respect the Lambda deadline
  model_refusal          — the model returned stop_reason=refusal
  admin_check_unverifiable — member_left_channel could not list members

All field values are serialized with ``json.dumps(default=str)``, so ints,
floats, bools, lists, and None are emitted natively; anything else falls
back to ``str()``.
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any

_audit_logger = logging.getLogger("audit")

# Per-invocation context. Defaults to None so unset values are simply
# omitted from the emitted record.
_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "audit_user_id", default=None
)
_channel_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "audit_channel_id", default=None
)
_thread_ts: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "audit_thread_ts", default=None
)


def set_context(
    *,
    user_id: str | None = None,
    channel_id: str | None = None,
    thread_ts: str | None = None,
) -> None:
    """Set ambient audit context for the current Lambda invocation.

    Subsequent `audit(...)` calls auto-include these fields. Passing None
    for any keyword is a no-op (keeps a previously-set value).
    """
    if user_id is not None:
        _user_id.set(user_id)
    if channel_id is not None:
        _channel_id.set(channel_id)
    if thread_ts is not None:
        _thread_ts.set(thread_ts)


def audit(event: str, **fields: Any) -> None:
    """Emit one structured audit log line.

    Schema:
      audit_event      — required, event name (see module docstring)
      audit_timestamp  — ISO 8601 UTC, added automatically
      user_id          — from context if set
      channel_id       — from context if set
      thread_ts        — from context if set
      <event-specific> — whatever the caller passes as kwargs

    Caller-provided kwargs override ambient context if the same key is
    used (so an explicit `user_id=...` wins over the contextvar). This
    helps `channel_access_denied` events that need to log a denial for a
    user before any context is set for them.
    """
    payload: dict[str, Any] = {
        "audit_event": event,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for key, var in (
        ("user_id", _user_id),
        ("channel_id", _channel_id),
        ("thread_ts", _thread_ts),
    ):
        val = var.get()
        if val is not None and key not in fields:
            payload[key] = val
    payload.update(fields)

    # Include the JSON in the rendered message as well as LogRecord ``extra``.
    # AWS Lambda installs its own handler before application import, and that
    # handler may omit custom record attributes. Keeping the payload in the
    # message makes every event searchable and parseable in either runtime.
    rendered = json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True)
    _audit_logger.info("audit %s", rendered, extra=payload)


def tool_input_summary(inp: dict[str, Any] | None) -> str:
    """JSON-serialize a tool input for inclusion in a tool_call audit event.

    Full fidelity — CloudWatch logs aren't a token-cost surface, and full
    inputs are useful for forensics. Returns a sentinel string if the input
    isn't JSON-serializable rather than raising, so a malformed input never
    crashes the audit emission.
    """
    if not inp:
        return ""
    try:
        return json.dumps(inp, default=str)
    except (TypeError, ValueError):
        return "(unserialisable input)"
