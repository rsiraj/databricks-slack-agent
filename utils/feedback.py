"""Slack answer-feedback payloads and DynamoDB persistence."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get(
    "FEEDBACK_TABLE_NAME",
    "databricks-slack-agent-feedback",
)
FEEDBACK_RETENTION_DAYS = 365
TTL_SECONDS = FEEDBACK_RETENTION_DAYS * 24 * 3600

RATING_LABELS = {
    "helpful": "Helpful",
    "somewhat_helpful": "Somewhat Helpful",
    "needs_fix": "Needs a Fix",
}
RATINGS_REQUIRING_COMMENT = frozenset({"somewhat_helpful", "needs_fix"})
FEEDBACK_CALLBACK_ID = "answer_feedback_detail"
FEEDBACK_COMMENT_BLOCK_ID = "feedback_comment_block"
FEEDBACK_COMMENT_ACTION_ID = "feedback_comment"
FEEDBACK_PROMPT_TEXT = "How useful was this answer?"

_table = None


def _get_table():
    global _table
    if _table is None:
        import boto3

        _table = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _table


def _button_value(rating: str, answer_ts: str) -> str:
    return json.dumps(
        {"rating": rating, "answer_ts": answer_ts},
        separators=(",", ":"),
    )


def feedback_prompt_blocks(answer_ts: str) -> list[dict]:
    """Build the shared feedback panel displayed after a bot answer."""
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{FEEDBACK_PROMPT_TEXT}*"},
        },
        {
            "type": "actions",
            "block_id": "answer_feedback_actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": "feedback_helpful",
                    "text": {"type": "plain_text", "text": "Helpful"},
                    "style": "primary",
                    "value": _button_value("helpful", answer_ts),
                },
                {
                    "type": "button",
                    "action_id": "feedback_somewhat_helpful",
                    "text": {"type": "plain_text", "text": "Somewhat Helpful"},
                    "value": _button_value("somewhat_helpful", answer_ts),
                },
                {
                    "type": "button",
                    "action_id": "feedback_needs_fix",
                    "text": {"type": "plain_text", "text": "Needs a Fix"},
                    "value": _button_value("needs_fix", answer_ts),
                },
            ],
        },
    ]


def parse_button_value(value: str) -> dict[str, str]:
    """Validate and return feedback metadata encoded in a Slack button."""
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid feedback button value") from exc

    rating = parsed.get("rating")
    answer_ts = str(parsed.get("answer_ts") or "")
    if rating not in RATING_LABELS or not answer_ts:
        raise ValueError("Incomplete feedback button value")
    return {"rating": rating, "answer_ts": answer_ts}


def feedback_modal_view(
    *,
    rating: str,
    answer_ts: str,
    channel_id: str,
    thread_ts: str,
) -> dict:
    """Build the private follow-up modal for a partial or negative rating."""
    if rating not in RATINGS_REQUIRING_COMMENT:
        raise ValueError(f"Rating does not require a comment: {rating}")

    prompt = (
        "What was missing?"
        if rating == "somewhat_helpful"
        else "What should we correct?"
    )
    metadata = json.dumps(
        {
            "rating": rating,
            "answer_ts": answer_ts,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
        },
        separators=(",", ":"),
    )
    return {
        "type": "modal",
        "callback_id": FEEDBACK_CALLBACK_ID,
        "private_metadata": metadata,
        "title": {"type": "plain_text", "text": "Help us improve"},
        "submit": {"type": "plain_text", "text": "Submit"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": FEEDBACK_COMMENT_BLOCK_ID,
                "label": {"type": "plain_text", "text": prompt},
                "element": {
                    "type": "plain_text_input",
                    "action_id": FEEDBACK_COMMENT_ACTION_ID,
                    "multiline": True,
                    "max_length": 500,
                },
            }
        ],
    }


def parse_modal_submission(view: dict) -> dict[str, str]:
    """Extract validated metadata and the required written feedback."""
    try:
        metadata = json.loads(view.get("private_metadata") or "{}")
        comment = (
            view["state"]["values"][FEEDBACK_COMMENT_BLOCK_ID][
                FEEDBACK_COMMENT_ACTION_ID
            ]["value"]
            or ""
        ).strip()
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid feedback modal submission") from exc

    rating = metadata.get("rating")
    result = {
        "rating": rating,
        "answer_ts": str(metadata.get("answer_ts") or ""),
        "channel_id": str(metadata.get("channel_id") or ""),
        "thread_ts": str(metadata.get("thread_ts") or ""),
        "comment": comment,
    }
    if (
        rating not in RATINGS_REQUIRING_COMMENT
        or not result["answer_ts"]
        or not result["channel_id"]
        or not result["thread_ts"]
        or not comment
    ):
        raise ValueError("Incomplete feedback modal submission")
    return result


def save_feedback(
    *,
    channel_id: str,
    thread_ts: str,
    answer_ts: str,
    user_id: str,
    rating: str,
    comment: str = "",
) -> dict[str, Any]:
    """Upsert one user's latest feedback for an answer."""
    if rating not in RATING_LABELS:
        raise ValueError(f"Unknown feedback rating: {rating}")
    comment = comment.strip()
    if rating in RATINGS_REQUIRING_COMMENT and not comment:
        raise ValueError(f"Feedback comment is required for {rating}")
    if len(comment) > 500:
        raise ValueError("Feedback comment exceeds 500 characters")
    if not all((channel_id, thread_ts, answer_ts, user_id)):
        raise ValueError("Feedback identifiers are required")

    now_value = time.time()
    now = int(now_value)
    item = {
        "answer_key": f"{channel_id}#{answer_ts}",
        "user_id": user_id,
        "channel_id": channel_id,
        "thread_ts": thread_ts,
        "answer_ts": answer_ts,
        "rating": rating,
        "comment": comment,
        "updated_at": now,
        "updated_at_ms": int(now_value * 1000),
        "ttl": now + TTL_SECONDS,
    }
    _get_table().put_item(Item=item)
    return item
