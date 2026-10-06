"""Single-use Slack job approvals, persisted using the existing UpdateItem grant.

The short-lived rate-limit table also holds approval rows under a separate key
prefix. All state transitions are conditional; expiry is checked in the write
condition rather than relying on asynchronous DynamoDB TTL deletion.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from botocore.exceptions import ClientError

from utils import ratelimit

CONFIRM_ACTION = "job_run_confirm"
APPROVAL_TTL_SECONDS = 600
_context: ContextVar[dict | None] = ContextVar("job_request_context", default=None)


@contextmanager
def request_context(user_id: str, channel_id: str, thread_ts: str):
    token = _context.set({"user_id": user_id, "channel_id": channel_id,
                          "thread_ts": thread_ts, "requests": {}})
    try:
        yield
    finally:
        _context.reset(token)


def current_request() -> dict | None:
    return _context.get()


def normalize_parameters(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("Notebook parameters must be string key/value pairs")
    if len(json.dumps(value, ensure_ascii=False)) > 2000:
        raise ValueError("Notebook parameters are too long to display for confirmation")
    return dict(value)


def snapshot(payload: dict) -> dict:
    """Validate current Jobs API configuration and fingerprint exactly that job."""
    if payload.get("next_page_token") or payload.get("has_more"):
        raise ValueError("The full job configuration could not be verified")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("The job configuration could not be verified")
    name = settings.get("name")
    if not isinstance(name, str) or not name or len(name) > 200:
        raise ValueError("The job name could not be displayed for confirmation")
    if name.startswith("PAUSED_USE_BUNDLE_VERSION_"):
        raise ValueError("This is a retired predecessor; request the current bundle job")
    configured = [settings[key] for key in ("schedule", "continuous", "trigger")
                  if key in settings]
    if not configured or any(
        not isinstance(config, dict) or config.get("pause_status") != "UNPAUSED"
        for config in configured
    ):
        raise ValueError("The job must have an explicitly unpaused schedule or trigger")
    source = settings.get("git_source")
    if source is not None and (
        not isinstance(source, dict) or source.get("git_branch") != "main"
    ):
        raise ValueError("Jobs using Git must run from the main branch")
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    return {"job_id": int(payload["job_id"]), "job_name": name,
            "settings_fingerprint": fingerprint}


def create(snapshot: dict, params: dict[str, str], context: dict) -> str:
    approval_id = uuid.uuid4().hex
    record = {**snapshot, "notebook_params": params}
    ratelimit._table.update_item(
        Key={"key": f"job_approval#{approval_id}"},
        UpdateExpression="SET #state=:preparing, #ttl=:ttl, #payload=:payload, "
                         "#channel=:channel, #thread=:thread, #requester=:requester",
        ConditionExpression="attribute_not_exists(#key)",
        ExpressionAttributeNames={"#key": "key", "#state": "state", "#ttl": "ttl",
                                  "#payload": "payload", "#channel": "channel_id",
                                  "#thread": "thread_ts", "#requester": "requester"},
        ExpressionAttributeValues={":preparing": "preparing", ":ttl": int(time.time()) + APPROVAL_TTL_SECONDS,
                                   ":payload": json.dumps(record), ":channel": context["channel_id"],
                                   ":thread": context["thread_ts"], ":requester": context["user_id"]},
    )
    return approval_id


def activate(approval_id: str, message_ts: str) -> None:
    ratelimit._table.update_item(
        Key={"key": f"job_approval#{approval_id}"},
        UpdateExpression="SET #state=:pending, #message=:message",
        ConditionExpression="#state=:preparing",
        ExpressionAttributeNames={"#state": "state", "#message": "message_ts"},
        ExpressionAttributeValues={":pending": "pending", ":preparing": "preparing", ":message": message_ts},
    )


def claim(event: dict) -> dict | None:
    """Claim a displayed approval once; any verified current member may claim it."""
    approval_id = event.get("approval_id", "")
    if not isinstance(approval_id, str) or not re.fullmatch(r"[0-9a-f]{32}", approval_id):
        return None
    try:
        response = ratelimit._table.update_item(
            Key={"key": f"job_approval#{approval_id}"},
            UpdateExpression="SET #state=:executing, #confirmer=:confirmer",
            ConditionExpression="#state=:pending AND #ttl>:now AND #channel=:channel "
                                "AND #thread=:thread AND #message=:message",
            ExpressionAttributeNames={"#state": "state", "#ttl": "ttl", "#channel": "channel_id",
                                      "#thread": "thread_ts", "#message": "message_ts", "#confirmer": "confirmer"},
            ExpressionAttributeValues={":executing": "executing", ":pending": "pending", ":now": int(time.time()),
                                       ":channel": event["channel_id"], ":thread": event["thread_ts"],
                                       ":message": event["message_ts"], ":confirmer": event["user_id"]},
            ReturnValues="ALL_NEW",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return None
        raise
    return json.loads(response["Attributes"]["payload"])


def complete(approval_id: str, result: dict) -> None:
    ratelimit._table.update_item(
        Key={"key": f"job_approval#{approval_id}"},
        UpdateExpression="SET #state=:done, #result=:result",
        ConditionExpression="#state=:executing",
        ExpressionAttributeNames={"#state": "state", "#result": "result"},
        ExpressionAttributeValues={":done": "complete", ":executing": "executing",
                                   ":result": json.dumps(result, default=str)},
    )


def confirmation_blocks(approval_id: str, job: dict, params: dict) -> list[dict]:
    # Plain text prevents job names or parameter values from injecting Slack
    # formatting, mentions, or a different-looking action into the card.
    return [
        {"type": "section", "text": {"type": "plain_text", "text":
            f"Confirm job run\n{job['job_name']} (job_id {job['job_id']})\n"
            f"Parameters: {json.dumps(params, ensure_ascii=False, sort_keys=True)}"}},
        {"type": "context", "elements": [{"type": "plain_text", "text":
            "Any current channel member can confirm once. Expires in 10 minutes. "
            "Confirmation and the resulting run will be visible to everyone here."}]},
        {"type": "actions", "elements": [{"type": "button", "action_id": CONFIRM_ACTION,
            "text": {"type": "plain_text", "text": "Confirm run"}, "style": "primary", "value": approval_id}]},
    ]
