"""Slack helpers for the worker: message posting, channel guards, thread history, images.

Signature verification and modal opening live in ``utils.slack_receiver`` (the
receiver's lightweight request path) and are intentionally not duplicated here.
"""

import base64
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from slack_sdk.errors import SlackApiError

from utils import clients, deadline as invocation_deadline
from utils.secrets import get_secret
from utils.slack_formatting import split_slack_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MessageDelivery:
    """Confirmed Slack delivery state for a possibly multi-part response."""

    chunks: tuple[str, ...]
    timestamps: tuple[str, ...]
    total_chunks: int

    @property
    def complete(self) -> bool:
        return len(self.timestamps) == self.total_chunks

    @property
    def delivered_text(self) -> str:
        return "\n\n".join(self.chunks[: len(self.timestamps)])


def post_message(
    channel: str,
    text: str,
    thread_ts: Optional[str] = None,
    blocks: Optional[list[dict[str, Any]]] = None,
) -> Optional[str]:
    """Post a message; return ts on success."""
    try:
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        if blocks is not None:
            kwargs["blocks"] = blocks
        resp = clients.slack_client.chat_postMessage(**kwargs)
        return resp.get("ts")
    except SlackApiError as e:
        logger.error(f"Slack post failed: {e.response['error']}")
        return None
    except Exception:
        # Transport errors (timeouts, DNS, connection resets) are not
        # SlackApiError. A failed post must never take the whole run down.
        logger.exception("Slack post failed before completion")
        return None


def post_message_chunks(
    channel: str,
    text: str,
    *,
    thread_ts: Optional[str] = None,
) -> MessageDelivery:
    """Post all response chunks, stopping at the first unconfirmed delivery."""
    chunks = tuple(split_slack_message(text))
    timestamps: list[str] = []
    for chunk in chunks:
        timestamp = post_message(channel, chunk, thread_ts=thread_ts)
        if not timestamp:
            break
        timestamps.append(timestamp)
    return MessageDelivery(
        chunks=chunks,
        timestamps=tuple(timestamps),
        total_chunks=len(chunks),
    )


_SQL_FENCE_RE = re.compile(
    r"^[ \t]*```[ \t]*(?:sql|databricks[-_ ]sql|spark[-_ ]sql)[ \t]*\r?\n"
    r"(.*?)"
    r"^[ \t]*```[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def extract_sql_blocks(text: str) -> tuple[str, ...]:
    """Return non-empty SQL-language fenced blocks in answer order."""
    return tuple(
        sql
        for match in _SQL_FENCE_RE.finditer(text)
        if (sql := match.group(1).strip())
    )


def render_sql_attachment(sql_blocks: tuple[str, ...]) -> str:
    """Combine one or more SQL blocks into a runnable attachment."""
    if len(sql_blocks) == 1:
        return f"{sql_blocks[0]}\n"
    return "\n\n".join(
        f"-- Query {index}\n{sql}"
        for index, sql in enumerate(sql_blocks, start=1)
    ) + "\n"


def upload_sql_file(
    channel: str,
    sql: str,
    *,
    thread_ts: str,
    filename: str,
) -> Optional[str]:
    """Upload SQL into a Slack thread; return the Slack file ID on success."""
    try:
        resp = clients.slack_client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            content=sql,
            filename=filename,
            title="SQL from Databricks Slack Agent",
            snippet_type="sql",
        )
        uploaded = resp.get("files") or []
        if uploaded and uploaded[0].get("id"):
            return uploaded[0]["id"]
        file_obj = resp.get("file") or {}
        return file_obj.get("id")
    except SlackApiError as e:
        logger.error("Slack SQL upload failed: %s", e.response["error"])
        return None
    except Exception:
        logger.exception("Slack SQL upload failed before completion")
        return None


def update_message(channel: str, ts: str, text: str, blocks: list[dict] | None = None) -> bool:
    try:
        kwargs = {"channel": channel, "ts": ts, "text": text}
        if blocks is not None:
            kwargs["blocks"] = blocks
        clients.slack_client.chat_update(**kwargs)
        return True
    except SlackApiError as e:
        logger.error(f"Slack update failed: {e.response['error']}")
    except Exception:
        logger.exception("Slack update failed before completion")
    return False


def post_ephemeral(
    channel: str,
    user: str,
    text: str,
    thread_ts: Optional[str] = None,
) -> None:
    """Post a private acknowledgement visible only to one Slack user."""
    try:
        clients.slack_client.chat_postEphemeral(
            channel=channel,
            user=user,
            text=text,
            thread_ts=thread_ts,
        )
    except SlackApiError as e:
        logger.error(f"Slack ephemeral post failed: {e.response['error']}")
    except Exception:
        logger.exception("Slack ephemeral post failed before completion")


_bot_user_id_cache: Optional[str] = None


def get_bot_user_id() -> str:
    """Return the bot's own Slack user_id (cached for the Lambda container lifetime)."""
    global _bot_user_id_cache
    if _bot_user_id_cache is None:
        resp = clients.slack_client.auth_test()
        _bot_user_id_cache = resp["user_id"]
    return _bot_user_id_cache


MAX_THREAD_REPLIES = 1000


def get_thread_replies(channel: str, thread_ts: str, limit: int = 200) -> list[dict]:
    """Fetch all messages in a thread, oldest-first, following pagination.

    ``limit`` is the per-page size. Collection stops at ``MAX_THREAD_REPLIES``
    messages so a pathological thread cannot stall the worker. Returns the
    pages fetched so far on an error part-way through, and an empty list if
    the very first page fails.
    """
    messages: list[dict] = []
    cursor: Optional[str] = None
    try:
        while True:
            kwargs: dict[str, Any] = {"channel": channel, "ts": thread_ts, "limit": limit}
            if cursor:
                kwargs["cursor"] = cursor
            resp = clients.slack_client.conversations_replies(**kwargs)
            messages.extend(resp.get("messages", []) or [])
            if len(messages) >= MAX_THREAD_REPLIES:
                logger.warning(
                    "Thread reply history capped at %d messages", MAX_THREAD_REPLIES
                )
                return messages[:MAX_THREAD_REPLIES]
            cursor = (resp.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                return messages
    except SlackApiError as e:
        logger.error(f"conversations.replies failed: {e.response['error']}")
        return messages
    except Exception:
        logger.exception("conversations.replies failed before completion")
        return messages


def get_channel_members(channel: str) -> frozenset[str]:
    """Return all member user_ids in the channel as a frozenset.

    Empty set on any failure (API error, bot not in channel, etc.). Callers
    treating empty as 'couldn't verify' should fail closed."""
    try:
        members: list[str] = []
        cursor = None
        while True:
            kwargs: dict = {"channel": channel, "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            resp = clients.slack_client.conversations_members(**kwargs)
            members.extend(resp.get("members", []) or [])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
        return frozenset(members)
    except SlackApiError as e:
        logger.error(f"conversations.members failed: {e.response['error']}")
        return frozenset()
    except Exception:
        logger.exception("conversations.members failed before completion")
        return frozenset()


def leave_channel(channel: str) -> bool:
    """Have the bot leave the given channel."""
    try:
        clients.slack_client.conversations_leave(channel=channel)
        return True
    except SlackApiError as e:
        logger.error(f"conversations.leave failed: {e.response['error']}")
        return False
    except Exception:
        logger.exception("conversations.leave failed before completion")
        return False


def is_private_channel(channel: str) -> Optional[bool]:
    """Return True/False if the channel privacy can be determined; None on error.

    Callers treating None as "couldn't verify" should fail closed (refuse to
    operate). Used by the channel-access guard alongside the channel-ID
    allowlist as defense-in-depth — we never want the bot acting in a public
    channel even if its ID somehow ended up on the allowlist.
    """
    try:
        resp = clients.slack_client.conversations_info(channel=channel)
    except SlackApiError as e:
        logger.error(f"conversations.info failed: {e.response['error']}")
        return None
    except Exception:
        logger.exception("conversations.info failed before completion")
        return None
    info = resp.get("channel") or {}
    if "is_private" in info:
        return bool(info["is_private"])
    return None


SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE = 5
MAX_MESSAGE_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_DOWNLOAD_BUDGET_SECONDS = 30.0
IMAGE_CHUNK_BYTES = 64 * 1024


def download_slack_image(
    file_obj: dict, *, max_bytes: int = MAX_IMAGE_BYTES, deadline: float | None = None
) -> Optional[dict]:
    """Download a Slack-hosted image file and return a Claude `image` content block.

    Returns None for non-images, unsupported mimetypes, or download failures.
    """
    mimetype = file_obj.get("mimetype")
    if mimetype not in SUPPORTED_IMAGE_MIMES:
        return None
    url = file_obj.get("url_private_download") or file_obj.get("url_private")
    if not url:
        return None
    max_bytes = min(max_bytes, MAX_IMAGE_BYTES)
    if max_bytes <= 0:
        return None
    if deadline is None:
        deadline = time.monotonic() + IMAGE_DOWNLOAD_BUDGET_SECONDS
    try:
        advertised_size = int(file_obj.get("size") or 0)
    except (TypeError, ValueError):
        advertised_size = 0
    if advertised_size > max_bytes:
        logger.warning("Skipping oversized image before download: %d bytes", advertised_size)
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    try:
        with requests.get(
            url,
            headers={"Authorization": f"Bearer {get_secret('slack_bot_token')}"},
            timeout=(min(5.0, remaining), min(5.0, remaining)),
            stream=True,
        ) as response:
            response.raise_for_status()
            try:
                content_length = int(response.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                content_length = 0
            if content_length > max_bytes:
                logger.warning("Skipping oversized image response: %d bytes", content_length)
                return None
            data = bytearray()
            for chunk in response.iter_content(chunk_size=IMAGE_CHUNK_BYTES):
                if time.monotonic() >= deadline:
                    logger.warning("Image download exceeded elapsed time budget")
                    return None
                if len(data) + len(chunk) > max_bytes:
                    logger.warning("Skipping image that exceeded %d bytes", max_bytes)
                    return None
                data.extend(chunk)
    except Exception:
        logger.exception("Failed to download Slack file")
        return None

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mimetype,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def extract_image_blocks(slack_message: dict) -> list[dict]:
    """Turn any image files attached to a Slack message into Claude image content blocks."""
    files = slack_message.get("files") or []
    blocks: list[dict] = []
    remaining_bytes = MAX_MESSAGE_IMAGE_BYTES
    deadline = time.monotonic() + IMAGE_DOWNLOAD_BUDGET_SECONDS
    # Bound attempted downloads as well as accepted images: invalid files must
    # not turn one Slack message into an unbounded series of network requests.
    images = [f for f in files if f.get("mimetype") in SUPPORTED_IMAGE_MIMES]
    if not images:
        return blocks
    try:
        # Socket inactivity timeouts cannot stop a server that continuously
        # trickles bytes. Bound the entire batch with the Lambda alarm too.
        with invocation_deadline.enforce_remaining(max_seconds=IMAGE_DOWNLOAD_BUDGET_SECONDS):
            for index, file_obj in enumerate(images):
                if index >= MAX_IMAGES_PER_MESSAGE or remaining_bytes <= 0 or time.monotonic() >= deadline:
                    break
                block = download_slack_image(file_obj, max_bytes=remaining_bytes, deadline=deadline)
                if block:
                    encoded = block["source"]["data"]
                    padding = len(encoded) - len(encoded.rstrip("="))
                    remaining_bytes -= (len(encoded) // 4) * 3 - padding
                    blocks.append(block)
    except invocation_deadline.DeadlineExceeded:
        logger.warning("Stopped image downloads at elapsed time budget")
    return blocks
