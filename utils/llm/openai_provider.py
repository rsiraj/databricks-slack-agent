"""OpenAI Responses API provider (translation layer; network call gated).

Everything here except ``OpenAIProvider._client`` is a pure function over
plain dicts so the Anthropic <-> Responses translation is unit-tested without
the ``openai`` package. The client is constructed only when the provider is
actually used, and only if the package imports *and* the app secret carries an
``openai_api_key``; otherwise ``ProviderNotConfigured`` is raised. Nothing in
this module is reached while ``LLM_PROVIDER`` is ``anthropic``.

Target model facts (verified 2026-09-04, see docs/PROVIDERS.md): ``gpt-5.6-terra``
on the Responses API; ``reasoning={"effort": ...}`` with levels
none/low/medium/high/xhigh/max; ``max_output_tokens`` caps reasoning + output
(reserve >= 25k); with ``store=False`` every prior output item, including
``reasoning`` items with ``encrypted_content``, must be replayed in ``input``;
function tools are ``{"type": "function", "name", "description", "parameters"}``
and results are ``{"type": "function_call_output", "call_id", "output"}``;
caching is automatic on a stable prefix; ``status == "incomplete"`` with
``incomplete_details.reason == "max_output_tokens"`` signals truncation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.llm.base import ProviderNotConfigured
from utils.llm.types import (
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolSpec,
    Usage,
    is_provider_reasoning,
    provider_reasoning_block,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-5.6-terra"
EFFORT_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")
# ``max_output_tokens`` caps reasoning *and* visible output; below this the
# model can spend the whole budget thinking and return nothing.
MIN_MAX_OUTPUT_TOKENS = 25_000
REASONING_INCLUDE = ["reasoning.encrypted_content"]
NOT_CONFIGURED_MESSAGE = (
    "OpenAI provider is not enabled: install the openai package and add "
    "openai_api_key to the app secret"
)

_ANTHROPIC_ONLY_BLOCKS = frozenset({"thinking", "redacted_thinking"})


# --------------------------------------------------------------------------- #
# Request translation (Anthropic-shaped history -> Responses ``input``)
# --------------------------------------------------------------------------- #


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-key access so SDK objects and plain dicts translate alike."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(vars(obj))


def _text_of(block: dict[str, Any]) -> str:
    return str(block.get("text") or "")


def _image_to_input(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") or {}
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64":
        media_type = source.get("media_type") or "image/png"
        return {
            "type": "input_image",
            "image_url": f"data:{media_type};base64,{source.get('data') or ''}",
        }
    if source.get("type") == "url" and source.get("url"):
        return {"type": "input_image", "image_url": source["url"]}
    return None


def _tool_result_text(content: Any) -> str:
    """Flatten a ``tool_result`` payload (string or text blocks) to one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for inner in content:
            if isinstance(inner, dict) and inner.get("type") == "text":
                parts.append(str(inner.get("text") or ""))
            elif isinstance(inner, dict):
                parts.append(json.dumps(inner, default=str))
            else:
                parts.append(str(inner))
        return "\n".join(parts)
    return json.dumps(content, default=str)


def tool_result_output(block: dict[str, Any]) -> str:
    """Tool result text for ``function_call_output``; ``is_error`` becomes explicit.

    Anthropic carries the error flag out of band. The Responses API has no
    such flag, so an error result whose text is not already a JSON object with
    an ``error`` key is wrapped as ``{"error": <text>}``.
    """
    text = _tool_result_text(block.get("content"))
    if not block.get("is_error"):
        return text
    try:
        decoded = json.loads(text) if text.strip() else None
    except ValueError:
        decoded = None
    if isinstance(decoded, dict) and "error" in decoded:
        return text
    return json.dumps({"error": text}, default=str)


def _user_items(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": [{"type": "input_text", "text": content}]}]
    parts: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "input_text", "text": _text_of(block)})
        elif block_type == "image":
            image = _image_to_input(block)
            if image is not None:
                parts.append(image)
        elif block_type == "tool_result":
            # Tool results are top-level items, not message parts. Preserve
            # order: flush any accumulated user parts first.
            if parts:
                items.append({"role": "user", "content": parts})
                parts = []
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id") or ""),
                    "output": tool_result_output(block),
                }
            )
    if parts:
        items.append({"role": "user", "content": parts})
    return items


def _assistant_items(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "assistant", "content": [{"type": "output_text", "text": content}]}]
    items: list[dict[str, Any]] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in _ANTHROPIC_ONLY_BLOCKS:
            continue
        if block_type == "text":
            items.append(
                {"role": "assistant", "content": [{"type": "output_text", "text": _text_of(block)}]}
            )
        elif block_type == "tool_use":
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(block.get("id") or ""),
                    "name": str(block.get("name") or ""),
                    "arguments": json.dumps(block.get("input") or {}, default=str),
                }
            )
        elif is_provider_reasoning(block, PROVIDER_NAME):
            payload = block.get("payload")
            if isinstance(payload, dict) and payload:
                items.append(dict(payload))
        # provider_reasoning from any other provider is opaque here: skip it.
    return items


def translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped user/assistant turns -> Responses API ``input`` items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            items.extend(_user_items(content))
        elif role == "assistant":
            items.extend(_assistant_items(content))
    return items


def translate_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]


def map_effort(level: str) -> str:
    if level not in EFFORT_LEVELS:
        raise ValueError(f"Effort {level!r} is not supported by OpenAI (valid: {', '.join(EFFORT_LEVELS)})")
    return level


def max_output_tokens_for(max_tokens: int) -> int:
    if max_tokens < MIN_MAX_OUTPUT_TOKENS:
        logger.info(
            "Raising max_output_tokens from %d to %d so reasoning cannot starve the answer",
            max_tokens,
            MIN_MAX_OUTPUT_TOKENS,
        )
        return MIN_MAX_OUTPUT_TOKENS
    return max_tokens


def build_request(request: ModelRequest, model: str) -> dict[str, Any]:
    """Kwargs for ``client.responses.create``."""
    kwargs: dict[str, Any] = {
        "model": request.model or model,
        "instructions": request.system,
        "input": translate_messages(request.messages),
        "tools": translate_tools(request.tools),
        "reasoning": {"effort": map_effort(request.effort)},
        "max_output_tokens": max_output_tokens_for(request.max_tokens),
        "store": False,
        "include": list(REASONING_INCLUDE),
    }
    if request.timeout_s is not None:
        kwargs["timeout"] = float(request.timeout_s)
    return kwargs


# --------------------------------------------------------------------------- #
# Response translation (Responses ``output`` -> Anthropic-shaped blocks)
# --------------------------------------------------------------------------- #


def _parse_arguments(raw: Any, name: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Unparseable function_call arguments for %s; sending empty input", name)
        return {}
    return decoded if isinstance(decoded, dict) else {}


def translate_output(output: Any) -> tuple[list[dict[str, Any]], bool, bool]:
    """Responses ``output`` items -> (blocks, saw_function_call, saw_refusal)."""
    blocks: list[dict[str, Any]] = []
    saw_function_call = False
    saw_refusal = False
    for raw_item in output or []:
        item = _as_dict(raw_item)
        item_type = item.get("type")
        if item_type == "message":
            for raw_part in item.get("content") or []:
                part = _as_dict(raw_part)
                part_type = part.get("type")
                if part_type == "output_text":
                    blocks.append({"type": "text", "text": str(part.get("text") or "")})
                elif part_type == "refusal":
                    saw_refusal = True
                    refusal = str(part.get("refusal") or "")
                    if refusal:
                        blocks.append({"type": "text", "text": refusal})
        elif item_type == "function_call":
            saw_function_call = True
            name = str(item.get("name") or "")
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "name": name,
                    "input": _parse_arguments(item.get("arguments"), name),
                }
            )
        elif item_type == "reasoning":
            blocks.append(provider_reasoning_block(PROVIDER_NAME, dict(item)))
        # Other item types (web_search_call, etc.) are not requested and are dropped.
    return blocks, saw_function_call, saw_refusal


def map_stop_reason(
    status: Any,
    incomplete_reason: Any,
    *,
    saw_function_call: bool,
    saw_refusal: bool,
) -> tuple[StopReason, str]:
    """Normalize a Responses ``status`` into the worker's stop reasons.

    A function call wins because the loop must answer it; truncation is next
    so a cut-off answer is delivered with a note; refusal content parts map to
    ``refusal``; a completed response is ``end_turn``.
    """
    raw = str(status or "")
    if incomplete_reason:
        raw = f"{raw}:{incomplete_reason}"
    if saw_function_call:
        return "tool_use", raw
    if status == "incomplete" and incomplete_reason == "max_output_tokens":
        return "max_tokens", raw
    if saw_refusal:
        return "refusal", raw
    if status in (None, "", "completed"):
        return "end_turn", raw or "completed"
    return "other", raw


def translate_usage(usage: Any) -> Usage:
    """Responses ``usage`` -> Anthropic accounting.

    OpenAI's ``input_tokens`` *includes* cached tokens; Anthropic's excludes
    cache reads. Subtract so cost math and cache-hit rates stay comparable.
    There is no cache-write charge, so ``cache_creation_input_tokens`` is 0.
    """
    if usage is None:
        return Usage()
    input_tokens = int(_get(usage, "input_tokens", 0) or 0)
    details = _get(usage, "input_tokens_details", None)
    cached = int(_get(details, "cached_tokens", 0) or 0) if details is not None else 0
    cached = min(cached, input_tokens)
    return Usage(
        input_tokens=input_tokens - cached,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached,
        output_tokens=int(_get(usage, "output_tokens", 0) or 0),
    )


def translate_response(resp: Any, requested_model: str) -> ModelResponse:
    blocks, saw_function_call, saw_refusal = translate_output(_get(resp, "output", None))
    incomplete = _get(resp, "incomplete_details", None)
    incomplete_reason = _get(incomplete, "reason", None) if incomplete is not None else None
    stop_reason, raw_stop = map_stop_reason(
        _get(resp, "status", None),
        incomplete_reason,
        saw_function_call=saw_function_call,
        saw_refusal=saw_refusal,
    )
    return ModelResponse(
        model=str(_get(resp, "model", None) or requested_model),
        stop_reason=stop_reason,
        content=blocks,
        usage=translate_usage(_get(resp, "usage", None)),
        raw_stop_reason=raw_stop,
        provider=PROVIDER_NAME,
    )


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class OpenAIProvider:
    name = PROVIDER_NAME

    def __init__(self, model: str = DEFAULT_MODEL, *, client: Any | None = None) -> None:
        self.model = model
        self._client_override = client
        self._managed_client = None
        self._managed_api_key = None

    def _client(self) -> Any:
        """Build the SDK client lazily; refuse clearly when the provider is not enabled."""
        if self._client_override is not None:
            return self._client_override
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ProviderNotConfigured(NOT_CONFIGURED_MESSAGE) from exc
        from utils.secrets import get_secret

        try:
            api_key = get_secret("openai_api_key")
        except Exception as exc:  # noqa: BLE001 - missing key is a configuration error
            raise ProviderNotConfigured(NOT_CONFIGURED_MESSAGE) from exc
        if not api_key:
            raise ProviderNotConfigured(NOT_CONFIGURED_MESSAGE)
        if self._managed_client is None or self._managed_api_key != api_key:
            self._managed_client = openai.OpenAI(api_key=api_key)
            self._managed_api_key = api_key
        return self._managed_client

    def supports_effort(self, level: str) -> bool:
        return level in EFFORT_LEVELS

    def map_effort(self, level: str) -> str:
        return map_effort(level)

    def build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        return build_request(request, self.model)

    def create(self, request: ModelRequest) -> ModelResponse:
        kwargs = self.build_kwargs(request)
        resp = self._client().responses.create(**kwargs)
        return translate_response(resp, kwargs["model"])
