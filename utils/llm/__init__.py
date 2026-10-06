"""Model provider selection.

Environment:

    LLM_PROVIDER     ``anthropic`` (default) or ``openai``
    LLM_MODEL        provider-neutral model override; wins when set
    ANTHROPIC_MODEL  Anthropic model id (default ``claude-sonnet-5``)
    OPENAI_MODEL     OpenAI model id (default ``gpt-5.6-terra``)

``get_provider()`` is cached per process (one Lambda container); tests call
``reset_provider()`` after changing the environment.
"""

from __future__ import annotations

import os

from utils.llm.base import Provider, ProviderNotConfigured
from utils.llm.types import ModelRequest, ModelResponse, ToolSpec, Usage

DEFAULT_PROVIDER = "anthropic"
SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai")
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"

_cached: dict[str, Provider] = {}


def provider_name() -> str:
    """Read and validate ``LLM_PROVIDER``."""
    raw = (os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if raw not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER {raw!r}; expected one of {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return raw


def resolve_model(provider: str | None = None) -> str:
    """Model id for ``provider``: ``LLM_MODEL`` wins, else the provider's own variable."""
    override = (os.environ.get("LLM_MODEL") or "").strip()
    if override:
        return override
    name = provider or provider_name()
    if name == "openai":
        return (os.environ.get("OPENAI_MODEL") or "").strip() or DEFAULT_OPENAI_MODEL
    return (os.environ.get("ANTHROPIC_MODEL") or "").strip() or DEFAULT_ANTHROPIC_MODEL


def build_provider(name: str | None = None, model: str | None = None) -> Provider:
    """Construct (uncached) the provider ``name`` for ``model``."""
    name = name or provider_name()
    if name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {name!r}; expected one of {', '.join(SUPPORTED_PROVIDERS)}"
        )
    model = model or resolve_model(name)
    if name == "openai":
        from utils.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(model)
    from utils.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(model)


def get_provider() -> Provider:
    """The process-wide provider selected by ``LLM_PROVIDER``."""
    if "provider" not in _cached:
        _cached["provider"] = build_provider()
    return _cached["provider"]


def reset_provider() -> None:
    _cached.clear()


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "ModelRequest",
    "ModelResponse",
    "Provider",
    "ProviderNotConfigured",
    "ToolSpec",
    "Usage",
    "build_provider",
    "get_provider",
    "provider_name",
    "reset_provider",
    "resolve_model",
]
