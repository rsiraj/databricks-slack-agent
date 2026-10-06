"""Provider selection (LLM_PROVIDER / LLM_MODEL / *_MODEL) and the not-configured path."""

import sys

import pytest

from utils import llm
from utils.llm.anthropic_provider import AnthropicProvider
from utils.llm.base import ProviderNotConfigured
from utils.llm.openai_provider import OpenAIProvider
from utils.llm.types import ModelRequest


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    llm.reset_provider()
    yield
    llm.reset_provider()


class TestProviderName:
    def test_defaults_to_anthropic(self):
        assert llm.provider_name() == "anthropic"

    @pytest.mark.parametrize("raw", ["openai", " OpenAI ", "OPENAI"])
    def test_openai_is_normalized(self, monkeypatch, raw):
        monkeypatch.setenv("LLM_PROVIDER", raw)
        assert llm.provider_name() == "openai"

    def test_unknown_provider_is_rejected_loudly(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER 'gemini'"):
            llm.provider_name()


class TestResolveModel:
    def test_anthropic_default_and_env(self, monkeypatch):
        assert llm.resolve_model("anthropic") == "claude-sonnet-5"
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        assert llm.resolve_model("anthropic") == "claude-sonnet-4-6"

    def test_openai_default_and_env(self, monkeypatch):
        assert llm.resolve_model("openai") == "gpt-5.6-terra"
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-mini")
        assert llm.resolve_model("openai") == "gpt-5.6-mini"

    def test_llm_model_wins_for_any_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-mini")
        monkeypatch.setenv("LLM_MODEL", "override-model")
        assert llm.resolve_model("anthropic") == "override-model"
        assert llm.resolve_model("openai") == "override-model"

    def test_blank_values_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "   ")
        monkeypatch.setenv("ANTHROPIC_MODEL", "")
        assert llm.resolve_model("anthropic") == "claude-sonnet-5"

    def test_uses_env_provider_when_not_given(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert llm.resolve_model() == "gpt-5.6-terra"


class TestGetProvider:
    def test_default_is_anthropic_with_default_model_and_is_cached(self):
        provider = llm.get_provider()
        assert isinstance(provider, AnthropicProvider)
        assert provider.name == "anthropic"
        assert provider.model == "claude-sonnet-5"
        assert llm.get_provider() is provider

    def test_env_switch_selects_openai_after_reset(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        llm.reset_provider()
        provider = llm.get_provider()
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-5.6-terra"

    def test_cache_survives_env_change_until_reset(self, monkeypatch):
        first = llm.get_provider()
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert llm.get_provider() is first
        llm.reset_provider()
        assert isinstance(llm.get_provider(), OpenAIProvider)

    def test_build_provider_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            llm.build_provider("gemini")

    def test_build_provider_accepts_explicit_model(self):
        provider = llm.build_provider("openai", "gpt-5.6-mini")
        assert provider.model == "gpt-5.6-mini"


def _request():
    return ModelRequest(
        system="sys", messages=[{"role": "user", "content": "hi"}], tools=[], effort="high",
        max_tokens=16_000, timeout_s=30.0,
    )


class TestOpenAINotConfigured:
    def test_missing_package_raises_before_any_network_call(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)  # makes ``import openai`` fail
        with pytest.raises(ProviderNotConfigured, match="install the openai package"):
            OpenAIProvider("gpt-5.6-terra").create(_request())

    def test_missing_secret_key_raises(self, monkeypatch):
        from types import SimpleNamespace

        from utils import secrets

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: object()))
        monkeypatch.setattr(secrets, "get_secret", lambda name: None)
        with pytest.raises(ProviderNotConfigured, match="openai_api_key"):
            OpenAIProvider("gpt-5.6-terra").create(_request())

    def test_secret_lookup_failure_is_reported_as_not_configured(self, monkeypatch):
        from types import SimpleNamespace

        from utils import secrets

        def boom(name):
            raise KeyError(name)

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: object()))
        monkeypatch.setattr(secrets, "get_secret", boom)
        with pytest.raises(ProviderNotConfigured):
            OpenAIProvider("gpt-5.6-terra").create(_request())

    def test_configured_path_builds_client_with_secret_key(self, monkeypatch):
        from types import SimpleNamespace

        from utils import secrets

        constructed = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                constructed.append(kwargs)
                self.responses = self

            def create(self, **kwargs):
                return {"status": "completed", "output": [], "usage": None}

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
        monkeypatch.setattr(secrets, "get_secret", lambda name: "sk-test" if name == "openai_api_key" else None)

        response = OpenAIProvider("gpt-5.6-terra").create(_request())

        assert constructed == [{"api_key": "sk-test"}]
        assert response.provider == "openai"
        assert response.stop_reason == "end_turn"
