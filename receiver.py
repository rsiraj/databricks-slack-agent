"""Slack receiver Lambda — verifies signatures, acks within 3s, async-invokes the worker.

Forwards the following event types to the worker:
- `app_mention` — user @-mentioned the bot
- `member_joined_channel` — someone (possibly the bot) joined a channel
- `member_left_channel` — someone (possibly an admin) left a channel

Worker filters/handles each based on the event payload. Receiver stays light.

Other Slack delivery types handled here:
- `url_verification` — Slack app config challenge
- Slack retries (`X-Slack-Retry-Num` header): deduped by `event_id` through a
  one-shot DynamoDB claim (`utils.ratelimit.claim_once`). The first delivery
  to win the claim is dispatched whether or not it is a retry, so a redelivery
  after a lost original is still processed. If DynamoDB is unavailable the
  receiver falls back to dispatching only `X-Slack-Retry-Num` == 0 so an
  outage cannot create duplicate worker runs.
"""

import base64
import json
import logging
from urllib.parse import parse_qs

from utils import clients
from utils.job_approvals import CONFIRM_ACTION
from utils.feedback import (
    FEEDBACK_CALLBACK_ID,
    RATINGS_REQUIRING_COMMENT,
    feedback_modal_view,
    parse_button_value,
    parse_modal_submission,
)
from utils.ratelimit import claim_once, release_claim
from utils.slack_receiver import open_modal, verify_slack_signature

logger = logging.getLogger(__name__)

TRIGGERING_EVENT_TYPES = {
    "app_mention",
    "member_joined_channel",
    "member_left_channel",
}

DEDUPED_INTERACTION_TYPES = {"block_actions", "view_submission"}
DEDUPE_TTL_SECONDS = 600


def _event_dedupe_key(payload: dict) -> str | None:
    """Stable key for an Events API delivery: `event_id`, else channel+event_ts."""
    event_id = payload.get("event_id")
    if event_id:
        return f"dedupe#event#{event_id}"
    ev = payload.get("event") or {}
    event_ts = ev.get("event_ts") or ev.get("ts")
    channel = ev.get("channel")
    if event_ts and channel:
        return f"dedupe#event#{channel}#{event_ts}"
    return None


def _interaction_dedupe_key(payload: dict) -> str | None:
    """Stable key for an interactivity payload: `trigger_id`, else action_ts."""
    trigger_id = payload.get("trigger_id")
    if trigger_id:
        return f"dedupe#interaction#{trigger_id}"
    action = next(iter(payload.get("actions") or []), None) or {}
    action_ts = action.get("action_ts")
    if action_ts:
        return f"dedupe#interaction#{action_ts}"
    return None


def _should_dispatch(dedupe_key: str | None, retry_num: str | None) -> bool:
    """Decide whether this delivery does the work or is a duplicate.

    One DynamoDB call. When no key can be derived or DynamoDB errors, use the
    legacy rule (only the original, non-retry delivery dispatches) so a
    dependency failure degrades to at-most-once rather than duplicates.
    """
    is_retry = bool(retry_num) and retry_num != "0"
    if dedupe_key is None:
        return not is_retry
    claimed = claim_once(dedupe_key, ttl_seconds=DEDUPE_TTL_SECONDS)
    if claimed is None:
        logger.warning(
            "Dedupe claim unavailable; using retry-number fallback",
            extra={"retry_num": retry_num, "dispatch": not is_retry},
        )
        return not is_retry
    if not claimed:
        logger.info(
            "Skipping duplicate Slack delivery",
            extra={"retry_num": retry_num, "dedupe_key": dedupe_key},
        )
    return claimed


def _ok(body: dict | str = "") -> dict:
    if isinstance(body, dict):
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}
    return {"statusCode": 200, "body": body}


def _headers_lower(event: dict) -> dict:
    return {k.lower(): v for k, v in (event.get("headers") or {}).items()}


def _text_preview(text: str | None, limit: int = 200) -> str:
    text = (text or "").replace("\n", "\\n")
    return text[:limit] + "..." if len(text) > limit else text


def _invoke_worker(payload: dict) -> bool:
    try:
        clients.lambda_client.invoke(
            FunctionName=clients.WORKER_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        return True
    except Exception:
        logger.exception("Failed to async-invoke worker")
        return False


def _parse_interaction(raw_body: str) -> dict | None:
    form = parse_qs(raw_body, keep_blank_values=True)
    encoded = (form.get("payload") or [None])[0]
    if not encoded:
        return None
    try:
        return json.loads(encoded)
    except json.JSONDecodeError:
        return None


def _handle_interaction(payload: dict) -> dict:
    interaction_type = payload.get("type")
    user_id = (payload.get("user") or {}).get("id")

    if interaction_type == "block_actions":
        action = next(iter(payload.get("actions") or []), None)
        if action and action.get("action_id") == CONFIRM_ACTION:
            container = payload.get("container") or {}
            message = payload.get("message") or {}
            event = {
                "approval_id": action.get("value"),
                "user_id": user_id,
                "channel_id": (payload.get("channel") or {}).get("id") or container.get("channel_id"),
                "thread_ts": message.get("thread_ts") or container.get("thread_ts"),
                "message_ts": message.get("ts") or container.get("message_ts"),
            }
            if not all(isinstance(value, str) and value for value in event.values()):
                return _ok()
            accepted = _invoke_worker({"job_confirmation_event": event})
            return _ok() if accepted else {"statusCode": 500, "body": "invoke failed"}
        if not action or not str(action.get("action_id", "")).startswith("feedback_"):
            return _ok()
        try:
            metadata = parse_button_value(action.get("value", ""))
        except ValueError:
            logger.warning("Ignoring malformed feedback button payload")
            return _ok()

        container = payload.get("container") or {}
        channel_id = (
            (payload.get("channel") or {}).get("id")
            or container.get("channel_id")
            or ""
        )
        thread_ts = (
            container.get("thread_ts")
            or (payload.get("message") or {}).get("thread_ts")
            or ""
        )
        if not all((user_id, channel_id, thread_ts)):
            logger.warning("Ignoring incomplete feedback action payload")
            return _ok()

        rating = metadata["rating"]
        if rating in RATINGS_REQUIRING_COMMENT:
            trigger_id = payload.get("trigger_id")
            if not trigger_id:
                return _ok()
            opened = open_modal(
                trigger_id,
                feedback_modal_view(
                    rating=rating,
                    answer_ts=metadata["answer_ts"],
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                ),
            )
            return _ok() if opened else {"statusCode": 500, "body": "modal failed"}

        accepted = _invoke_worker(
            {
                "feedback_event": {
                    **metadata,
                    "channel_id": channel_id,
                    "thread_ts": thread_ts,
                    "user_id": user_id,
                    "comment": "",
                }
            }
        )
        return _ok() if accepted else {"statusCode": 500, "body": "invoke failed"}

    if interaction_type == "view_submission":
        view = payload.get("view") or {}
        if view.get("callback_id") != FEEDBACK_CALLBACK_ID:
            return _ok()
        try:
            feedback = parse_modal_submission(view)
        except ValueError:
            logger.warning("Rejecting malformed feedback modal submission")
            return _ok(
                {
                    "response_action": "errors",
                    "errors": {
                        "feedback_comment_block": "Please provide brief feedback."
                    },
                }
            )
        accepted = _invoke_worker(
            {"feedback_event": {**feedback, "user_id": user_id}}
        )
        return _ok() if accepted else {"statusCode": 500, "body": "invoke failed"}

    return _ok()


def lambda_handler(event, context):
    headers = _headers_lower(event)
    raw_body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        try:
            raw_body = base64.b64decode(raw_body, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            logger.warning("Invalid base64-encoded request body")
            return {"statusCode": 400, "body": "bad encoded body"}

    if not verify_slack_signature(headers, raw_body):
        logger.warning("Invalid Slack signature")
        return {"statusCode": 401, "body": "invalid signature"}

    # Slack redelivers when we don't ack fast enough. Dedupe by event identity
    # (see module docstring) instead of dropping every retry, so a retry whose
    # original was lost still gets handled exactly once.
    retry_num = headers.get("x-slack-retry-num")
    if retry_num:
        logger.info(
            "Slack retry received",
            extra={"retry_num": retry_num, "reason": headers.get("x-slack-retry-reason")},
        )

    content_type = headers.get("content-type", "")
    if content_type.startswith("application/x-www-form-urlencoded"):
        interaction = _parse_interaction(raw_body)
        if interaction is None:
            return {"statusCode": 400, "body": "bad interaction payload"}
        interaction_key = None
        if interaction.get("type") in DEDUPED_INTERACTION_TYPES:
            interaction_key = _interaction_dedupe_key(interaction)
            if not _should_dispatch(interaction_key, retry_num):
                return _ok()
        response = _handle_interaction(interaction)
        if interaction_key and response.get("statusCode", 200) >= 500:
            # We claimed the key but failed to hand off the work; release it so
            # Slack's redelivery is processed instead of deduped away.
            release_claim(interaction_key)
        return response

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": "bad json"}

    # URL verification handshake
    if payload.get("type") == "url_verification":
        return _ok({"challenge": payload.get("challenge")})

    if payload.get("type") == "event_callback":
        ev = payload.get("event") or {}
        ev_type = ev.get("type")

        if ev_type not in TRIGGERING_EVENT_TYPES:
            return _ok()

        # For app_mention only: ignore mentions sourced from bots (prevents loops)
        if ev_type == "app_mention" and (
            ev.get("bot_id")
            or ev.get("subtype") in {"bot_message", "message_changed", "message_deleted"}
        ):
            logger.warning(
                "Ignoring bot-sourced app_mention: %s",
                json.dumps(
                    {
                        "bot_id": ev.get("bot_id"),
                        "subtype": ev.get("subtype"),
                        "user": ev.get("user"),
                        "channel": ev.get("channel"),
                        "ts": ev.get("ts"),
                        "text_preview": _text_preview(ev.get("text")),
                    },
                    default=str,
                ),
                extra={
                    "bot_id": ev.get("bot_id"),
                    "subtype": ev.get("subtype"),
                    "user": ev.get("user"),
                    "channel": ev.get("channel"),
                    "ts": ev.get("ts"),
                    "text_preview": _text_preview(ev.get("text")),
                },
            )
            return _ok()

        dedupe_key = _event_dedupe_key(payload)
        if not _should_dispatch(dedupe_key, retry_num):
            return _ok()

        if not _invoke_worker({"slack_event": payload}):
            # Claimed but not dispatched: release so the Slack retry gets through.
            if dedupe_key:
                release_claim(dedupe_key)
            return {"statusCode": 500, "body": "invoke failed"}

        return _ok()

    return _ok()
