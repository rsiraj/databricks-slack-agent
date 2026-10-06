"""Lightweight Slack helpers used on the receiver's three-second request path."""

import hashlib
import hmac
import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils.secrets import get_secret

logger = logging.getLogger(__name__)

SLACK_API_URL = "https://slack.com/api/views.open"


def verify_slack_signature(headers: dict, body: str) -> bool:
    """Verify a Slack request signature and reject stale requests."""
    timestamp = headers.get("x-slack-request-timestamp")
    signature = headers.get("x-slack-signature")
    if not timestamp or not signature:
        return False
    try:
        request_time = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - request_time) > 60 * 5:
        return False

    basestring = f"v0:{timestamp}:{body}".encode("utf-8")
    expected = "v0=" + hmac.new(
        get_secret("slack_signing_secret").encode("utf-8"),
        basestring,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def open_modal(trigger_id: str, view: dict) -> bool:
    """Open a Slack modal without loading the full Slack SDK in the receiver."""
    body = json.dumps(
        {"trigger_id": trigger_id, "view": view},
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        SLACK_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {get_secret('slack_bot_token')}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Slack modal open failed: %s", exc)
        return False

    if not result.get("ok"):
        logger.error("Slack modal open failed: %s", result.get("error", "unknown_error"))
        return False
    return True
