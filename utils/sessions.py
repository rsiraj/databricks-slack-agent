"""DynamoDB-backed session storage for the agent.

Each Slack thread maps to one session item, keyed by `thread_ts`. The session
stores the full Anthropic-format message list — including `tool_use` and
`tool_result` blocks — so the worker has perfect memory of prior turns even
across cold-started Lambda containers.

Sessions auto-expire 365 days after their last update via DynamoDB TTL.

Two protections wrap every write:

* **Size handling without silent loss.** DynamoDB rejects items above 400 KB.
  Before writing, the message list is reduced in a fixed order until it fits:
  base64 image blocks become text placeholders, leading messages are dropped
  one at a time (never below the current turn's final exchange), and finally
  old ``tool_result`` payloads are truncated. Anything that changed the stored
  history is audited as ``session_trimmed``; a list that still cannot fit is
  audited as ``session_save_failed`` before the write is attempted anyway.

* **Optimistic locking.** Every item carries a random ``version``. ``load_session``
  remembers the version it read; ``save_session`` writes conditionally against
  it. A concurrent writer (two @-mentions in the same thread racing through
  two Lambda invocations) triggers a reload-and-merge instead of a lost update,
  audited as ``session_write_conflict``.
"""

import copy
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from utils.audit import audit
from utils.llm.blocks import sanitize_assistant_messages

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("SESSIONS_TABLE_NAME", "databricks-slack-agent-sessions")
SESSION_RETENTION_DAYS = 365
TTL_SECONDS = SESSION_RETENTION_DAYS * 24 * 3600
# DDB item limit is 400 KB; leave headroom for attribute overhead and tool result
# payloads we haven't seen yet.
MAX_ITEM_BYTES = 350_000
MAX_CONFLICT_RETRIES = 3
# Step (c) of the size handling: oldest tool_result payloads are cut to this
# many characters once image replacement and message dropping were not enough.
TOOL_RESULT_TRUNCATE_CHARS = 2_000
_TOOL_RESULT_TRUNCATE_SUFFIX = "…[truncated for storage]"

_table = boto3.resource("dynamodb").Table(TABLE_NAME)

# Optimistic-locking bookkeeping, keyed by thread_ts. Populated by
# ``load_session`` and refreshed by every successful ``save_session``.
#
#   version    — the ``version`` attribute of the item we last read or wrote
#                (None for a legacy item written before versions existed)
#   exists     — whether an item was present when we loaded it
#   loaded_len — how many entries of the caller's list have been loaded or
#                successfully saved; ``messages[loaded_len:]`` are its unsaved
#                additions and are what gets merged on a write conflict
#   prefix     — after a conflict merge, the complete last persisted history;
#                later saves append only newly added messages on top of it so
#                the caller's (unmerged) local list can't clobber the merge
_loaded_sessions: dict[str, dict[str, Any]] = {}


def _is_invalid_first_message(msg: dict) -> bool:
    """True if `msg` cannot legally be the first message in an Anthropic API call.

    The API requires the first message to be `role=user` and free of any
    `tool_result` blocks (since there is no preceding assistant message
    holding the matching `tool_use`).
    """
    if msg.get("role") != "user":
        return True
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


def _strip_invalid_leading_messages(messages: list[dict]) -> list[dict]:
    """Drop messages from the front until `messages[0]` is a valid first message.

    Self-heals sessions that were left in an invalid state by the older trim
    logic (which dropped two messages at a time without checking whether the
    new first message was an orphan `tool_result`).
    """
    while messages and _is_invalid_first_message(messages[0]):
        messages = messages[1:]
    return messages


def _serialize(messages: list[dict]) -> str:
    return json.dumps(messages, default=str)


def _byte_size(serialized: str) -> int:
    return len(serialized.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Size handling
# --------------------------------------------------------------------------- #


def _is_base64_image_block(block: Any) -> bool:
    if not isinstance(block, dict) or block.get("type") != "image":
        return False
    source = block.get("source")
    return isinstance(source, dict) and source.get("type") == "base64"


def _image_placeholder(block: dict) -> dict:
    source = block.get("source") or {}
    media_type = source.get("media_type") or "image"
    data = source.get("data") or ""
    # base64 expands bytes by 4/3; report the approximate decoded size.
    decoded_bytes = (len(data) * 3) // 4 if isinstance(data, str) else 0
    size_kb = max(1, round(decoded_bytes / 1024)) if decoded_bytes else 0
    return {
        "type": "text",
        "text": (
            f"[Image attachment ({media_type}, {size_kb} KB) was shared here; "
            "it is no longer available in thread memory.]"
        ),
    }


def _replace_images_in_message(msg: dict) -> int:
    """Replace base64 image blocks in one message in place. Returns the count."""
    content = msg.get("content")
    if not isinstance(content, list):
        return 0
    replaced = 0
    for index, block in enumerate(content):
        if _is_base64_image_block(block):
            content[index] = _image_placeholder(block)
            replaced += 1
    return replaced


def _truncate_tool_results_in_message(msg: dict) -> int:
    """Cut oversized tool_result payloads in one message in place. Returns the count."""
    content = msg.get("content")
    if not isinstance(content, list):
        return 0
    truncated = 0
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        payload = block.get("content")
        if isinstance(payload, str):
            if len(payload) > TOOL_RESULT_TRUNCATE_CHARS:
                block["content"] = (
                    payload[:TOOL_RESULT_TRUNCATE_CHARS] + _TOOL_RESULT_TRUNCATE_SUFFIX
                )
                truncated += 1
        elif isinstance(payload, list):
            for inner in payload:
                if (
                    isinstance(inner, dict)
                    and inner.get("type") == "text"
                    and isinstance(inner.get("text"), str)
                    and len(inner["text"]) > TOOL_RESULT_TRUNCATE_CHARS
                ):
                    inner["text"] = (
                        inner["text"][:TOOL_RESULT_TRUNCATE_CHARS]
                        + _TOOL_RESULT_TRUNCATE_SUFFIX
                    )
                    truncated += 1
    return truncated


def _can_drop_leading_message(messages: list[dict]) -> Optional[list[dict]]:
    """Return the list after dropping one leading message, or None if that would
    cut into the current turn's final exchange.

    Dropping one message may orphan `tool_result` blocks at the new front, so
    the result is re-sanitized. The result is accepted only if it still holds
    at least two messages, or exactly the final message when that message is a
    lone user message.
    """
    if len(messages) <= 1:
        return None
    candidate = _strip_invalid_leading_messages(messages[1:])
    if len(candidate) >= 2:
        return candidate
    if len(candidate) == 1 and messages[-1].get("role") == "user":
        return candidate
    return None


def _fit_messages_for_storage(
    messages: list[dict], thread_ts: str
) -> tuple[list[dict], str, dict[str, int]]:
    """Reduce `messages` until its JSON fits `MAX_ITEM_BYTES`.

    The caller's list is never mutated; a deep copy is made only when a
    reduction is actually needed so the common small case is stored exactly
    as ``json.dumps(messages)`` would render it.

    Returns ``(messages, serialized, stats)`` where stats has keys
    ``trimmed_messages``, ``images_replaced``, ``tool_results_truncated`` and
    ``bytes``. The result may still exceed the cap; the caller decides what to
    do in that case.
    """
    stats = {"trimmed_messages": 0, "images_replaced": 0, "tool_results_truncated": 0}
    serialized = _serialize(messages)
    size = _byte_size(serialized)
    if size <= MAX_ITEM_BYTES:
        stats["bytes"] = size
        return messages, serialized, stats

    messages = copy.deepcopy(messages)

    # (a) Base64 images become placeholders, oldest message first.
    for msg in messages:
        if size <= MAX_ITEM_BYTES:
            break
        replaced = _replace_images_in_message(msg)
        if replaced:
            stats["images_replaced"] += replaced
            serialized = _serialize(messages)
            size = _byte_size(serialized)

    # (b) Drop leading messages one at a time, never below the final exchange.
    while size > MAX_ITEM_BYTES:
        candidate = _can_drop_leading_message(messages)
        if candidate is None:
            break
        stats["trimmed_messages"] += len(messages) - len(candidate)
        messages = candidate
        serialized = _serialize(messages)
        size = _byte_size(serialized)

    # (c) Truncate old tool_result payloads, oldest message first.
    for msg in messages:
        if size <= MAX_ITEM_BYTES:
            break
        truncated = _truncate_tool_results_in_message(msg)
        if truncated:
            stats["tool_results_truncated"] += truncated
            serialized = _serialize(messages)
            size = _byte_size(serialized)

    stats["bytes"] = size
    if stats["trimmed_messages"]:
        logger.warning(
            "Trimmed session %s by %d messages to fit DDB; %d remain",
            thread_ts,
            stats["trimmed_messages"],
            len(messages),
        )
    return messages, serialized, stats


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #


def _parse_item_messages(item: Optional[dict], thread_ts: str) -> Optional[list[dict]]:
    """Decode and sanitize the `messages` attribute of a stored item."""
    if not item:
        return None
    raw = item.get("messages")
    if not raw:
        return None
    try:
        messages = json.loads(raw)
    except json.JSONDecodeError:
        logger.exception("Corrupt session JSON for %s; ignoring", thread_ts)
        return None

    sanitized = _strip_invalid_leading_messages(messages)
    if len(sanitized) < len(messages):
        logger.warning(
            "Session %s had %d invalid leading message(s); recovered with %d remaining",
            thread_ts,
            len(messages) - len(sanitized),
            len(sanitized),
        )

    # Self-heal assistant turns stored with client-side SDK fields
    # (``parsed_output``, ``citations: null``) that the API rejects on replay.
    # Nothing is written back here; the next ``save_session`` persists the
    # clean version.
    clean = sanitize_assistant_messages(sanitized)
    if clean is not sanitized:
        logger.warning(
            "Session %s had assistant blocks with non-API fields; sanitized on load",
            thread_ts,
        )
    return clean or None


def load_session(thread_ts: str) -> Optional[list[dict]]:
    """Return the saved message list for this thread, or None if not found / corrupt.

    Self-heals sessions whose first message is an orphan `tool_result` block —
    a state the older trim logic could leave behind — and assistant turns whose
    blocks carry client-side SDK fields (``parsed_output``) that the API
    rejects on replay. Wedged threads recover automatically on the next user
    message.

    Also records the item's ``version`` (and how many messages were loaded) so
    the next ``save_session`` for this thread can write conditionally.
    """
    _loaded_sessions.pop(thread_ts, None)
    try:
        resp = _table.get_item(Key={"thread_ts": thread_ts}, ConsistentRead=True)
    except Exception:
        logger.exception("Failed to load session for %s", thread_ts)
        return None

    item = resp.get("Item")
    messages = _parse_item_messages(item, thread_ts)
    _loaded_sessions[thread_ts] = {
        "version": item.get("version") if item else None,
        "exists": bool(item),
        "loaded_len": len(messages) if messages else 0,
        "prefix": None,
    }
    return messages


def _condition_for(record: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build the conditional-write kwargs for `put_item` from the load record.

    * nothing loaded (or loaded and absent) — the item must not exist yet;
    * a legacy item without a ``version`` — it must still have no version;
    * a versioned item — the stored version must be the one we read.
    """
    if not record or not record.get("exists"):
        return {"ConditionExpression": "attribute_not_exists(thread_ts)"}
    version = record.get("version")
    if version is None:
        return {
            "ConditionExpression": "attribute_not_exists(#ver)",
            "ExpressionAttributeNames": {"#ver": "version"},
        }
    return {
        "ConditionExpression": "attribute_not_exists(thread_ts) OR #ver = :ver",
        "ExpressionAttributeNames": {"#ver": "version"},
        "ExpressionAttributeValues": {":ver": version},
    }


def _put(thread_ts: str, serialized: str, version: str, condition: dict[str, Any]) -> None:
    now = int(time.time())
    _table.put_item(
        Item={
            "thread_ts": thread_ts,
            "messages": serialized,
            "version": version,
            "updated_at": now,
            "ttl": now + TTL_SECONDS,
        },
        **condition,
    )


def _is_conditional_check_failure(exc: Exception) -> bool:
    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"


def _prepare_and_audit(messages: list[dict], thread_ts: str) -> tuple[str, int]:
    """Run the size handling and emit its audit events. Returns (json, bytes)."""
    _, serialized, stats = _fit_messages_for_storage(messages, thread_ts)
    size = stats["bytes"]
    if stats["trimmed_messages"] or stats["images_replaced"] or stats["tool_results_truncated"]:
        audit(
            "session_trimmed",
            thread_ts=thread_ts,
            trimmed_messages=stats["trimmed_messages"],
            images_replaced=stats["images_replaced"],
            tool_results_truncated=stats["tool_results_truncated"],
            bytes=size,
        )
    if size > MAX_ITEM_BYTES:
        logger.error(
            "Session %s is still %d bytes after trimming (cap %d); attempting save anyway",
            thread_ts,
            size,
            MAX_ITEM_BYTES,
        )
        audit("session_save_failed", thread_ts=thread_ts, reason="oversize", bytes=size)
    return serialized, size


def _messages_to_store(record: Optional[dict[str, Any]], messages: list[dict]) -> list[dict]:
    """Apply an earlier in-invocation conflict merge, if there was one."""
    if not record or record.get("prefix") is None:
        return messages
    loaded_len = int(record.get("loaded_len", 0) or 0)
    return _strip_invalid_leading_messages(list(record["prefix"]) + messages[loaded_len:])


def _resolve_conflict(thread_ts: str, messages: list[dict], record: Optional[dict[str, Any]]) -> None:
    """Retry a versioned merge, preserving writes made during conflict recovery."""
    loaded_len = int((record or {}).get("loaded_len", 0) or 0)
    new_messages = messages[loaded_len:] if loaded_len <= len(messages) else []
    for _ in range(MAX_CONFLICT_RETRIES):
        try:
            resp = _table.get_item(Key={"thread_ts": thread_ts}, ConsistentRead=True)
        except Exception:
            logger.exception("Failed to reload session %s after write conflict", thread_ts)
            return

        item = resp.get("Item")
        stored = _parse_item_messages(item, thread_ts) or []
        merged = _strip_invalid_leading_messages(list(stored) + list(new_messages))
        audit(
            "session_write_conflict",
            thread_ts=thread_ts,
            stored_messages=len(stored),
            new_messages=len(new_messages),
            merged_messages=len(merged),
        )
        serialized, size = _prepare_and_audit(merged, thread_ts)
        version = uuid.uuid4().hex
        current = {"exists": bool(item), "version": item.get("version") if item else None}
        try:
            _put(thread_ts, serialized, version, _condition_for(current))
        except Exception as exc:
            if _is_conditional_check_failure(exc):
                continue
            logger.exception("Failed to save merged session for %s (%d bytes)", thread_ts, size)
            return
        _loaded_sessions[thread_ts] = {
            "version": version,
            "exists": True,
            "loaded_len": len(messages),
            "prefix": merged,
        }
        return

    logger.error(
        "Session %s still conflicted after %d retries; preserving stored history",
        thread_ts,
        MAX_CONFLICT_RETRIES,
    )
    audit("session_save_failed", thread_ts=thread_ts, reason="conflict_retries_exhausted")


def save_session(thread_ts: str, messages: list[dict]) -> None:
    """Persist the message list.

    Reduces the list to fit DDB's per-item size limit without dropping the
    current turn (see module docstring), preserving the Anthropic invariant
    that ``messages[0]`` is a user message free of ``tool_result`` blocks, and
    writes with optimistic locking against the version read by ``load_session``.
    The caller's list is never mutated.
    """
    record = _loaded_sessions.get(thread_ts)
    to_store = _messages_to_store(record, messages)
    serialized, size = _prepare_and_audit(to_store, thread_ts)
    version = uuid.uuid4().hex

    try:
        _put(thread_ts, serialized, version, _condition_for(record))
    except Exception as exc:
        if _is_conditional_check_failure(exc):
            _resolve_conflict(thread_ts, messages, record)
            return
        logger.exception("Failed to save session for %s (%d bytes)", thread_ts, size)
        return

    prefix = record.get("prefix") if record else None
    _loaded_sessions[thread_ts] = {
        "version": version,
        "exists": True,
        "loaded_len": len(messages),
        "prefix": to_store if prefix is not None else None,
    }
