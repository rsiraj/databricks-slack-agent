"""Request-safe serialization of assistant content blocks.

The Anthropic SDK's response objects carry *client-side* attributes that the
Messages API does not accept on input. The streaming accumulator
(``anthropic/lib/streaming/_messages.py``) builds every text block as a
``ParsedTextBlock`` whose ``parsed_output`` field is meant for the SDK's
structured-output helpers; ``TextBlock`` also always carries
``citations=None``. Both survive a bare ``model_dump()``. Once such a dict is
appended to the assistant turn and persisted, the next request replays it and
the API answers::

    400 invalid_request_error: messages.N.content.0.text.parsed_output:
        Extra inputs are not permitted

This module is the single place that knows which keys each block type may
carry on the way back in. It is deliberately dependency-free so both the
provider (``utils.llm.anthropic_provider``) and session storage
(``utils.sessions``) can import it without cycles.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Block types the agent stores for its own bookkeeping. They are not sent to
# Anthropic (``strip_foreign_blocks`` removes them) but they must survive the
# sanitizer untouched so the provider that wrote them can replay them.
INTERNAL_BLOCK_TYPES: frozenset[str] = frozenset({"provider_reasoning"})

# Per block type: the keys the Messages API accepts on *input*. ``required``
# keys are emitted whenever present (the API needs them even if the value is
# odd); ``optional`` keys are emitted only when their value is not ``None``.
_ALLOWED_KEYS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # type: (required, optional)
    "text": (("type", "text"), ("citations",)),
    "tool_use": (("type", "id", "name", "input"), ()),
    "thinking": (("type", "thinking", "signature"), ()),
    "redacted_thinking": (("type", "data"), ()),
    "server_tool_use": (("type", "id", "name", "input"), ()),
}
# Server-side tool results (``web_search_tool_result``, ``web_fetch_tool_result``,
# ``code_execution_tool_result``, ...) all share one input shape.
_TOOL_RESULT_KEYS: tuple[str, ...] = ("type", "tool_use_id", "content")


def _is_tool_result_type(block_type: str) -> bool:
    return block_type.endswith("_tool_result") or block_type == "tool_result"


def sanitize_content_block(block: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``block`` holding only keys the Messages API accepts.

    Pure: the input is never mutated. Known block types are reduced to their
    allowlist; ``text`` keeps ``citations`` only when it is a non-empty list.
    Internal block types pass through untouched. Unknown block types keep
    every key whose value is not ``None`` and log a warning naming the type.
    """
    block_type = block.get("type")
    if not isinstance(block_type, str) or block_type in INTERNAL_BLOCK_TYPES:
        return dict(block)

    if block_type in _ALLOWED_KEYS:
        required, optional = _ALLOWED_KEYS[block_type]
    elif _is_tool_result_type(block_type):
        required, optional = _TOOL_RESULT_KEYS, ()
    else:
        logger.warning(
            "Unknown content block type %r; keeping non-null keys only", block_type
        )
        return {key: value for key, value in block.items() if value is not None}

    clean: dict[str, Any] = {key: block[key] for key in required if key in block}
    for key in optional:
        value = block.get(key)
        if value is None:
            continue
        if key == "citations" and not (isinstance(value, list) and value):
            continue
        clean[key] = value
    return clean


def sanitize_content_blocks(content: Any) -> Any:
    """Sanitize a list of block dicts; non-list content (plain strings) is returned as-is."""
    if not isinstance(content, list):
        return content
    return [
        sanitize_content_block(block) if isinstance(block, dict) else block
        for block in content
    ]


def sanitize_assistant_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run every assistant turn's blocks through ``sanitize_content_block``.

    Returns the very same list object when nothing changed so callers that
    forward ``messages`` to the API keep sending exactly what they were given.
    Messages are never mutated; a changed turn is replaced by a shallow copy
    with new ``content``.
    """
    changed = False
    result: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if message.get("role") != "assistant" or not isinstance(content, list):
            result.append(message)
            continue
        clean = sanitize_content_blocks(content)
        if clean == content:
            result.append(message)
            continue
        changed = True
        result.append({**message, "content": clean})
    return result if changed else messages
