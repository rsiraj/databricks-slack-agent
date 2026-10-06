"""Provider-neutral request/response types for the agent loop.

The canonical *stored* message format stays the Anthropic block format
(``text`` / ``tool_use`` / ``tool_result`` / ``image`` / ``thinking``); DynamoDB
holds a year of it. Providers translate to and from that shape at the edge.

One addition lets a non-Anthropic provider carry state across turns:

    {"type": "provider_reasoning", "provider": "openai", "payload": {...}}

The payload is opaque to everything except the provider that wrote it. Every
other consumer (compaction, guards, rendering) must tolerate and skip it, and
``AnthropicProvider`` strips it before sending (Anthropic rejects unknown block
types).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PROVIDER_REASONING_BLOCK_TYPE = "provider_reasoning"

StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "pause_turn", "other"]

USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model may call; ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ModelRequest:
    """One model call, expressed without provider-specific wrapping.

    ``messages`` are Anthropic-shaped (already compacted by the caller).
    ``timeout_s`` of ``None`` lets the provider's client default apply (the
    evaluation harness configures its client-level timeout instead).
    ``model`` overrides the provider's configured model for this call only;
    the eval matrix uses it to run several models through one provider.
    """

    system: str
    messages: list[dict[str, Any]]
    tools: list[ToolSpec]
    effort: str
    max_tokens: int
    timeout_s: float | None
    model: str | None = None


@dataclass
class Usage:
    """Token usage in Anthropic's accounting: ``input_tokens`` excludes cache reads."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name) or 0) for name in USAGE_FIELDS}


@dataclass
class ModelResponse:
    """A provider response normalized to the stored block format.

    ``content`` blocks are Anthropic-shaped dicts (``text``, ``tool_use``,
    ``thinking``, ``redacted_thinking``) plus ``provider_reasoning`` for
    providers that need to replay opaque state. ``stop_reason`` is normalized;
    ``raw_stop_reason`` keeps the provider's own value for audit logs.
    ``transport`` names the HTTP mode the provider used (Anthropic:
    ``"stream"`` or ``"create"``); empty when the provider does not report it.
    """

    model: str
    stop_reason: StopReason
    content: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw_stop_reason: str = ""
    provider: str = ""
    transport: str = ""


def provider_reasoning_block(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": PROVIDER_REASONING_BLOCK_TYPE, "provider": provider, "payload": payload}


def is_provider_reasoning(block: Any, provider: str | None = None) -> bool:
    if not isinstance(block, dict) or block.get("type") != PROVIDER_REASONING_BLOCK_TYPE:
        return False
    return provider is None or block.get("provider") == provider
