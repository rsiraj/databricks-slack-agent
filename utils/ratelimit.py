"""DynamoDB-backed rate limiting.

The ASR review asked for rate limiting against abuse, excessive cost, and
denial-of-service scenarios. Counters live in a dedicated DDB table
(`databricks-slack-agent-ratelimit`) — kept separate from the sessions
table so the two unrelated workloads don't share a key space.

Schema:
    PK: key  (string)        — e.g. "user#U00000000001#hour#2026052215"
    count   (number)         — incremented atomically per request
    ttl     (number)         — DDB-TTL attribute, auto-expires the row

Atomicity: each check is a single `UpdateItem` with a conditional
expression — the increment only lands if `count < limit`, otherwise DDB
raises `ConditionalCheckFailedException` and we treat it as a block. The
TTL is set via `if_not_exists` so the first request in a window
establishes the expiry; subsequent requests in the same window don't
slide it forward (each window has a stable expiry; the next window gets
its own item under a different key).

Failure mode: a DDB outage causes any rate-limit check to FAIL OPEN.
This is deliberate — rate limiting is one of several layers (channel
allowlist, SQL guards, audit logging are the others), and we'd rather
keep the bot answering than block all queries because DDB is sick. The
failure is logged at exception level so on-call notices.

Limits (configurable via env vars, sensible defaults baked in):
    user_per_hour    — 60     (a real analyst burst is well below this)
    user_per_day     — 200    (~4× sustained burst, still well above typical)
    channel_per_hour — 200    (multi-user channel, headroom)
    sql_per_day      — 500    (global SQL submission cap; cost protection)

When a check blocks, callers emit a `rate_limit_blocked` audit event so
the trip is queryable in CloudWatch Logs Insights.

The same table also backs `claim_once`, a one-shot idempotency claim the
receiver uses to dedupe Slack event redeliveries by `event_id` (rows are
keyed "dedupe#..." and carry only a short `ttl`).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Literal

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("RATELIMIT_TABLE_NAME", "databricks-slack-agent-ratelimit")

USER_PER_HOUR = int(os.environ.get("RATELIMIT_USER_PER_HOUR", "60"))
USER_PER_DAY = int(os.environ.get("RATELIMIT_USER_PER_DAY", "200"))
CHANNEL_PER_HOUR = int(os.environ.get("RATELIMIT_CHANNEL_PER_HOUR", "200"))
SQL_PER_DAY = int(os.environ.get("RATELIMIT_SQL_PER_DAY", "500"))

_table = boto3.resource("dynamodb").Table(TABLE_NAME)

Window = Literal["hour", "day"]


def _window_bucket(window: Window) -> tuple[str, int]:
    """Return (bucket_label, ttl_seconds_from_now) for the current window."""
    now = datetime.now(timezone.utc)
    if window == "hour":
        return now.strftime("%Y%m%d%H"), 7200  # 2 h covers window + delay
    if window == "day":
        return now.strftime("%Y%m%d"), 90000  # 25 h
    raise ValueError(f"Unknown window: {window}")


def _check_and_increment(
    scope: str, identifier: str, window: Window, limit: int
) -> bool:
    """Atomically increment the counter and return True iff under the limit.

    Single round-trip — DDB does the compare-and-update server-side.
    """
    bucket, ttl_delta = _window_bucket(window)
    key = f"{scope}#{identifier}#{window}#{bucket}"
    ttl_value = int(time.time()) + ttl_delta

    try:
        _table.update_item(
            Key={"key": key},
            UpdateExpression="ADD #c :one SET #ttl = if_not_exists(#ttl, :ttl)",
            ConditionExpression="attribute_not_exists(#c) OR #c < :limit",
            ExpressionAttributeNames={"#c": "count", "#ttl": "ttl"},
            ExpressionAttributeValues={":one": 1, ":limit": limit, ":ttl": ttl_value},
            ReturnValues="NONE",
        )
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        logger.exception(
            "Rate-limit DDB error; failing open",
            extra={"key": key, "limit": limit},
        )
        return True
    except BotoCoreError:
        # Endpoint/connection/credential failures never reach the service and
        # surface as BotoCoreError, not ClientError. Same fail-open policy.
        logger.exception(
            "Rate-limit DDB transport error; failing open",
            extra={"key": key, "limit": limit},
        )
        return True


def claim_once(key: str, ttl_seconds: int = 600) -> bool | None:
    """Atomically claim ``key`` once; used by the receiver to dedupe Slack retries.

    Single conditional ``UpdateItem`` (the execution role has UpdateItem but
    not PutItem on this table). The item is created with only a ``ttl``
    attribute so DynamoDB expires it after ``ttl_seconds``.

    Returns:
        True  — first claim; the caller should process the event.
        False — the key was already claimed; the caller should skip.
        None  — DynamoDB could not be reached or rejected the request for a
                reason other than the condition. The caller decides how to
                degrade (the receiver falls back to retry-number dedupe).

    A claim that was explicitly released (see ``release_claim``) can be
    re-claimed, so a delivery whose downstream dispatch failed is retried by
    Slack's next redelivery instead of being swallowed as a duplicate.
    """
    ttl_value = int(time.time()) + ttl_seconds
    try:
        _table.update_item(
            Key={"key": key},
            UpdateExpression="SET #ttl = :ttl, #rel = :no",
            ConditionExpression="attribute_not_exists(#k) OR #rel = :yes",
            ExpressionAttributeNames={"#k": "key", "#ttl": "ttl", "#rel": "released"},
            ExpressionAttributeValues={":ttl": ttl_value, ":no": False, ":yes": True},
            ReturnValues="NONE",
        )
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        logger.exception("Dedupe claim DDB error", extra={"key": key})
        return None
    except BotoCoreError:
        logger.exception("Dedupe claim DDB transport error", extra={"key": key})
        return None


def release_claim(key: str) -> bool:
    """Mark a claim as released so the next ``claim_once`` on ``key`` succeeds.

    Called when the caller claimed the key but then failed to do the work
    (e.g. the async worker invoke failed), so Slack's redelivery is processed
    rather than deduped away. Best effort: returns False on any DynamoDB error.
    """
    try:
        _table.update_item(
            Key={"key": key},
            UpdateExpression="SET #rel = :yes",
            ExpressionAttributeNames={"#rel": "released"},
            ExpressionAttributeValues={":yes": True},
            ReturnValues="NONE",
        )
        return True
    except (ClientError, BotoCoreError):
        logger.exception("Dedupe claim release failed", extra={"key": key})
        return False


def check_user(user_id: str | None) -> tuple[bool, str | None]:
    """Return (allowed, reason_if_blocked) for per-user hour + day caps.

    A missing user_id fails open — the event shouldn't have arrived without
    one, and we'd rather not block a legitimate (but anomalous) request.
    """
    if not user_id:
        return True, None
    if not _check_and_increment("user", user_id, "hour", USER_PER_HOUR):
        return False, f"per_user_hour ({USER_PER_HOUR}/hour)"
    if not _check_and_increment("user", user_id, "day", USER_PER_DAY):
        return False, f"per_user_day ({USER_PER_DAY}/day)"
    return True, None


def check_channel(channel_id: str | None) -> tuple[bool, str | None]:
    if not channel_id:
        return True, None
    if not _check_and_increment("channel", channel_id, "hour", CHANNEL_PER_HOUR):
        return False, f"per_channel_hour ({CHANNEL_PER_HOUR}/hour)"
    return True, None


def check_sql_global() -> tuple[bool, str | None]:
    """Global cap on databricks_execute_sql submissions per day.

    Cost protection — bounds Anthropic + Databricks spend if a runaway
    loop or sustained probe slips past the other guards.
    """
    if not _check_and_increment("sql", "global", "day", SQL_PER_DAY):
        return False, f"global_sql_day ({SQL_PER_DAY}/day)"
    return True, None
