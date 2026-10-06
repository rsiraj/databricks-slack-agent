"""Anthropic Messages API provider.

This is the call that used to live inline in ``worker._call_model``. The
request it builds is what the deployed baseline sends:

    model, max_tokens,
    system=[{"type": "text", "text": ..., "cache_control": {"type": "ephemeral"}}],
    tools=[{name, description, input_schema}],
    messages=<compacted history, last block marked cache_control ephemeral>,
    thinking={"type": "adaptive"},
    output_config={"effort": <effort>},
    timeout=anthropic_request_timeout(<bounded seconds>)

so the ``model_call`` / ``run_summary`` measurements stay comparable.

Prompt caching uses two explicit breakpoints. The one on the system block
covers ``tools`` + ``system`` (tools render first; a marker on the last system
block caches both). The second is a *moving* marker on the last content block
of the last message — the newest ``tool_result`` inside an agent run, the
user's text on the first call — so each iteration reads the whole prefix the
previous iteration wrote and writes only the newly appended turn. Without it
nothing after the system block is cache-eligible and the growing conversation
is re-sent uncached on every iteration (observed in production as a constant
``cache_read_input_tokens`` and ``cache_creation_input_tokens: 0``). The
marker is written to a copy of the last message; the caller's list — and so
the stored session — never carries ``cache_control``.

Transport is the one thing that may differ: ``ANTHROPIC_STREAM`` (``auto`` /
``always`` / ``never``) selects ``client.messages.stream(**kwargs)`` +
``get_final_message()`` instead of ``client.messages.create(**kwargs)``. The
kwargs, the returned ``Message`` and therefore usage, content, stop reason and
the ``model_call`` audit are identical either way; only the HTTP mode changes.

One thing the two transports do *not* share is the Python type of the text
blocks: the streaming accumulator returns ``ParsedTextBlock`` objects carrying
the client-side ``parsed_output`` attribute, which a bare ``model_dump()``
leaks into the stored assistant turn and the API then rejects on replay
(``messages.N.content.0.text.parsed_output: Extra inputs are not permitted``).
Every block is therefore reduced to the keys the API accepts on input via
``utils.llm.blocks.sanitize_content_block`` — on the way out of the response
and, as defense in depth for history that was stored before this existed, on
every assistant turn of the outgoing request.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from utils.llm.blocks import sanitize_assistant_messages, sanitize_content_block
from utils.llm.types import (
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolSpec,
    Usage,
    is_provider_reasoning,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "anthropic"

# Effort levels each model family accepts natively. Anything else is mapped by
# ``AnthropicProvider.map_effort`` (``xhigh`` -> ``max`` on Sonnet 4.6).
MODEL_EFFORT_LEVELS: dict[str, tuple[str, ...]] = {
    "claude-sonnet-4-6": ("low", "medium", "high", "max"),
    "claude-sonnet-5": ("low", "medium", "high", "xhigh", "max"),
    "claude-opus-5": ("low", "medium", "high", "xhigh", "max"),
}
DEFAULT_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "max")

KNOWN_STOP_REASONS = frozenset({"end_turn", "tool_use", "max_tokens", "refusal", "pause_turn"})
THINKING_MODES = ("adaptive", "disabled", None)

# Transport selection (``ANTHROPIC_STREAM``). ``auto`` streams only when the
# output cap is large enough that the SDK would otherwise refuse the request:
# ``Messages.create`` rejects a non-streaming call whose ``max_tokens`` implies
# more than 10 minutes of generation at its assumed 128k tokens/hour, i.e.
# ``max_tokens > 21,333``. 21,000 keeps a small margin under that line. The
# production default (claude-sonnet-5 at ``high`` -> 24,000) therefore streams;
# the Sonnet 4.6 baseline (16,000) stays on the non-streaming path.
STREAM_MODES: tuple[str, ...] = ("auto", "always", "never")
DEFAULT_STREAM_MODE = "auto"
STREAM_ENV_VAR = "ANTHROPIC_STREAM"
AUTO_STREAM_MAX_TOKENS = 21_000
# ``ModelResponse.transport`` / ``model_call.transport`` values: which
# ``client.messages`` method carried the request.
TRANSPORT_STREAM = "stream"
TRANSPORT_CREATE = "create"

# ``recommended_max_tokens``: output cap per model family and effort. The
# Sonnet 5 / Opus 5 tokenizer spends ~30% more tokens on the same text and the
# top effort levels think longer, so the 16,000 cap that fits Sonnet 4.6 at
# ``high`` truncates them. Values above ``AUTO_STREAM_MAX_TOKENS`` stream.
# ``DEFAULT_MAX_TOKENS`` is the hard floor (unknown models, invalid env values).
DEFAULT_MAX_TOKENS = 16_000
EFFORT_ORDER: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
RECOMMENDED_MAX_TOKENS: dict[str, dict[str, int]] = {
    "claude-sonnet-4-6": {"low": 16_000, "medium": 16_000, "high": 16_000, "xhigh": 32_000, "max": 32_000},
    "claude-sonnet-5": {"low": 24_000, "medium": 24_000, "high": 24_000, "xhigh": 48_000, "max": 64_000},
    "claude-opus-5": {"low": 24_000, "medium": 24_000, "high": 24_000, "xhigh": 48_000, "max": 64_000},
}

# Shown in place of an assistant turn that held only another provider's
# reasoning; Anthropic rejects an assistant message with empty content.
FOREIGN_REASONING_PLACEHOLDER = "[reasoning from another model provider omitted]"

# Prompt caching. ``EPHEMERAL_CACHE_CONTROL`` is the 5-minute TTL: the loop's
# iterations start well under five minutes apart, so every request refreshes
# the entry and the 1-hour TTL (2x write price) would buy nothing.
# ``CACHEABLE_BLOCK_TYPES`` are the content block types the API accepts a
# ``cache_control`` marker on; thinking / redacted_thinking are not among them.
# ``MAX_CACHE_BREAKPOINTS`` is the API's per-request ceiling; this provider
# places exactly two (system block + last message block).
EPHEMERAL_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}
CACHEABLE_BLOCK_TYPES: frozenset[str] = frozenset(
    {"text", "image", "tool_use", "tool_result", "document"}
)
MAX_CACHE_BREAKPOINTS = 4

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def base_model_id(model: str | None) -> str:
    """Lower-cased model id with any ``-YYYYMMDD`` snapshot suffix removed."""
    candidate = (model or "").strip().lower()
    return _DATE_SUFFIX_RE.sub("", candidate)


def effort_levels_for(model: str | None) -> tuple[str, ...]:
    """Native effort levels for ``model`` (tolerating dated snapshot ids)."""
    candidate = (model or "").strip().lower()
    if candidate in MODEL_EFFORT_LEVELS:
        return MODEL_EFFORT_LEVELS[candidate]
    return MODEL_EFFORT_LEVELS.get(base_model_id(candidate), DEFAULT_EFFORT_LEVELS)


def recommended_max_tokens(model: str | None, effort: str | None) -> int:
    """Suggested ``max_tokens`` for ``model`` at ``effort``.

    Unknown models and unknown effort levels fall back to the production
    baseline (``DEFAULT_MAX_TOKENS``); an effort above ``high`` on a model that
    has no such level (``xhigh`` on Sonnet 4.6) uses that model's ``max`` row,
    matching ``AnthropicProvider.map_effort``.
    """
    table = RECOMMENDED_MAX_TOKENS.get(base_model_id(model))
    if table is None:
        return DEFAULT_MAX_TOKENS
    level = (effort or "").strip().lower()
    if level not in EFFORT_ORDER:
        return DEFAULT_MAX_TOKENS
    return table[level]


def stream_mode_from_env(environ: Any | None = None) -> str:
    """Read ``ANTHROPIC_STREAM``; unknown values warn and fall back to ``auto``."""
    env = os.environ if environ is None else environ
    raw = (env.get(STREAM_ENV_VAR) or "").strip().lower()
    if not raw:
        return DEFAULT_STREAM_MODE
    if raw in STREAM_MODES:
        return raw
    logger.warning(
        "Invalid %s=%r; using %r (valid: %s)",
        STREAM_ENV_VAR,
        raw,
        DEFAULT_STREAM_MODE,
        ", ".join(STREAM_MODES),
    )
    return DEFAULT_STREAM_MODE


def should_stream(mode: str, max_tokens: int) -> bool:
    """Decide the transport: ``always`` / ``never`` are literal; ``auto`` streams iff
    ``max_tokens > AUTO_STREAM_MAX_TOKENS``."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode != "auto":
        raise ValueError(f"stream mode must be one of {STREAM_MODES}, got {mode!r}")
    return int(max_tokens) > AUTO_STREAM_MAX_TOKENS


def tools_to_anthropic(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
        for tool in tools
    ]


def strip_foreign_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove ``provider_reasoning`` blocks from assistant turns.

    Returns the very same list object when nothing needed stripping so the
    common (Anthropic-only history) path sends exactly what it was given.
    """
    changed = False
    result: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if message.get("role") != "assistant" or not isinstance(content, list):
            result.append(message)
            continue
        kept = [block for block in content if not is_provider_reasoning(block)]
        if len(kept) == len(content):
            result.append(message)
            continue
        changed = True
        if not kept:
            kept = [{"type": "text", "text": FOREIGN_REASONING_PLACEHOLDER}]
        result.append({**message, "content": kept})
    return result if changed else messages


def with_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``messages`` with ``cache_control`` on the last cacheable block of the last message.

    The marker goes on the last content block when its type accepts one
    (``text``, ``image``, ``tool_use``, ``tool_result``, ``document``);
    otherwise on the nearest earlier cacheable block of that same message;
    if the message has none, the list is returned unchanged. String content
    is promoted to a single ``text`` block, which the API treats identically.

    Never mutates its input: only the last message is replaced, by a shallow
    copy whose ``content`` list and marked block are new objects. Every other
    message is the caller's own object, so the outgoing prefix is exactly what
    was passed in and the caller's history (the stored session) never acquires
    a ``cache_control`` key.
    """
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, dict):
        return messages
    content = last.get("content")
    if isinstance(content, str):
        blocks: list[Any] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = list(content)
    else:
        return messages

    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if isinstance(block, dict) and block.get("type") in CACHEABLE_BLOCK_TYPES:
            blocks[index] = {**block, "cache_control": dict(EPHEMERAL_CACHE_CONTROL)}
            return [*messages[:-1], {**last, "content": blocks}]
    return messages


def count_cache_breakpoints(kwargs: dict[str, Any]) -> int:
    """Number of ``cache_control`` markers in a built request (system + tools + messages)."""
    count = 0
    for block in kwargs.get("system") or []:
        if isinstance(block, dict) and "cache_control" in block:
            count += 1
    for tool in kwargs.get("tools") or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            count += 1
    for message in kwargs.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                count += 1
    return count


def usage_from_response(resp: Any) -> Usage:
    """Read token usage off a Messages response, defaulting every field to 0."""
    usage = getattr(resp, "usage", None)
    values: dict[str, int] = {}
    for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
        raw = getattr(usage, name, 0) or 0
        try:
            values[name] = int(raw)
        except (TypeError, ValueError):
            values[name] = 0
    return Usage(**values)


def normalize_stop_reason(raw: Any) -> StopReason:
    return raw if raw in KNOWN_STOP_REASONS else "other"  # type: ignore[return-value]


def content_to_blocks(content: Any) -> list[dict[str, Any]]:
    """``resp.content`` -> list of request-safe plain dicts.

    SDK objects are ``model_dump``-ed and then reduced to the keys the Messages
    API accepts on input (see ``utils.llm.blocks``). This is what drops the
    streaming accumulator's ``parsed_output`` and the always-present
    ``citations: None`` before the block reaches the stored assistant turn.
    """
    blocks: list[dict[str, Any]] = []
    for block in content or []:
        if isinstance(block, dict):
            raw = block
        elif hasattr(block, "model_dump"):
            raw = block.model_dump()
        else:  # pragma: no cover - defensive; SDK blocks are pydantic models
            raw = dict(vars(block))
        blocks.append(sanitize_content_block(raw))
    return blocks


class AnthropicProvider:
    name = PROVIDER_NAME

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        thinking: str | None = "adaptive",
        effort_via_extra_body: bool = False,
        stream: str | None = None,
    ) -> None:
        if thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {THINKING_MODES}, got {thinking!r}")
        if stream is None:
            stream = stream_mode_from_env()
        elif stream not in STREAM_MODES:
            raise ValueError(f"stream must be one of {STREAM_MODES}, got {stream!r}")
        self.model = model
        self._client_override = client
        self.thinking = thinking
        # Older SDKs without a first-class ``output_config`` kwarg need
        # ``extra_body``; the evaluation harness detects this per client.
        self.effort_via_extra_body = effort_via_extra_body
        # ``auto`` (default) / ``always`` / ``never``; see ``should_stream``.
        self.stream_mode = stream

    # ------------------------------------------------------------------ client

    def _client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        from utils import clients

        return clients.anthropic_client

    # ------------------------------------------------------------------ effort

    def supports_effort(self, level: str) -> bool:
        return level in effort_levels_for(self.model)

    def map_effort(self, level: str) -> str:
        if self.supports_effort(level):
            return level
        if level == "xhigh":
            logger.info("Effort 'xhigh' is not available on %s; sending 'max'", self.model)
            return "max"
        raise ValueError(
            f"Effort {level!r} is not supported by {self.model} "
            f"(valid: {', '.join(effort_levels_for(self.model))})"
        )

    # ----------------------------------------------------------------- request

    def build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        model = request.model or self.model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            # Breakpoint 1: tools render before system, so this one marker
            # caches the whole static prefix (tools + system).
            "system": [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": dict(EPHEMERAL_CACHE_CONTROL),
                }
            ],
            "tools": tools_to_anthropic(request.tools),
            # The two cleaning passes return the caller's list object when
            # nothing changed, so a clean Anthropic-only history is sent
            # exactly as given; breakpoint 2 then replaces only the last
            # message with an annotated copy (see ``with_cache_breakpoint``).
            "messages": with_cache_breakpoint(
                sanitize_assistant_messages(strip_foreign_blocks(request.messages))
            ),
        }
        if self.thinking is not None:
            kwargs["thinking"] = {"type": self.thinking}
        if request.effort:
            effort = self.map_effort(request.effort)
            if self.effort_via_extra_body:
                kwargs["extra_body"] = {"output_config": {"effort": effort}}
            else:
                kwargs["output_config"] = {"effort": effort}
        if request.timeout_s is not None:
            from utils.clients import anthropic_request_timeout

            kwargs["timeout"] = anthropic_request_timeout(request.timeout_s)
        return kwargs

    def should_stream(self, max_tokens: int) -> bool:
        return should_stream(self.stream_mode, max_tokens)

    def create(self, request: ModelRequest) -> ModelResponse:
        kwargs = self.build_kwargs(request)
        messages_api = self._client().messages
        transport = TRANSPORT_STREAM if self.should_stream(kwargs["max_tokens"]) else TRANSPORT_CREATE
        logger.info(
            "Anthropic request via messages.%s (model=%s, max_tokens=%s)",
            transport,
            kwargs["model"],
            kwargs["max_tokens"],
        )
        if transport == TRANSPORT_STREAM:
            # Same kwargs (including ``timeout``); the SDK accumulates the
            # events into a ``Message`` whose text blocks are ``ParsedTextBlock``
            # objects — ``content_to_blocks`` strips their client-side fields.
            with messages_api.stream(**kwargs) as stream:
                resp = stream.get_final_message()
        else:
            resp = messages_api.create(**kwargs)
        return self._to_response(resp, request, transport)

    def _to_response(self, resp: Any, request: ModelRequest, transport: str = "") -> ModelResponse:
        raw_stop = getattr(resp, "stop_reason", None)
        return ModelResponse(
            model=str(getattr(resp, "model", None) or request.model or self.model),
            stop_reason=normalize_stop_reason(raw_stop),
            content=content_to_blocks(getattr(resp, "content", None)),
            usage=usage_from_response(resp),
            raw_stop_reason="" if raw_stop is None else str(raw_stop),
            provider=self.name,
            transport=transport,
        )
