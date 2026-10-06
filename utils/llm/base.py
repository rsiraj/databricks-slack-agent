"""Provider protocol shared by the worker and the evaluation harness."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from utils.llm.types import ModelRequest, ModelResponse


class ProviderNotConfigured(RuntimeError):
    """Raised when a provider is selected but its dependency or credential is missing."""


@runtime_checkable
class Provider(Protocol):
    name: str
    model: str

    def create(self, request: ModelRequest) -> ModelResponse:
        """Perform one model call and normalize the response."""

    def supports_effort(self, level: str) -> bool:
        """True when ``level`` is accepted natively by the configured model."""

    def map_effort(self, level: str) -> str:
        """Translate a requested effort level to what the provider will be sent."""
