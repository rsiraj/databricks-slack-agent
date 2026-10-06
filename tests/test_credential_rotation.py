"""Warm SDK clients follow the actual Secrets Manager cache TTL, offline."""

import json
import sys
from types import SimpleNamespace

import pytest

from utils import clients, secrets
from utils.llm.openai_provider import OpenAIProvider


@pytest.fixture
def rotating_secret(monkeypatch):
    state = SimpleNamespace(now=1000.0, value="old-credential", reads=0)

    def read_secret(**kwargs):
        state.reads += 1
        return {"SecretString": json.dumps({
            "anthropic_api_key": state.value,
            "slack_bot_token": state.value,
            "openai_api_key": state.value,
        })}

    monkeypatch.setattr(secrets, "_cache", {"data": None, "expires_at": 0.0})
    monkeypatch.setattr(secrets, "_secrets_client", SimpleNamespace(get_secret_value=read_secret))
    monkeypatch.setattr(secrets.time, "time", lambda: state.now)
    monkeypatch.setattr(clients, "_clients", {})
    monkeypatch.setattr(clients, "_client_credentials", {})
    return state


@pytest.mark.parametrize("provider", ["slack", "anthropic", "openai"])
def test_warm_client_refreshes_rotated_credential_after_secret_ttl(
    monkeypatch, rotating_secret, provider
):
    constructed = []

    def sdk_client(**kwargs):
        client = SimpleNamespace(credential=kwargs.get("token", kwargs.get("api_key")))
        constructed.append(client)
        return client

    monkeypatch.setitem(sys.modules, "slack_sdk", SimpleNamespace(WebClient=sdk_client))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=sdk_client))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=sdk_client))
    if provider == "openai":
        accessor = OpenAIProvider()._client
    else:
        accessor = lambda: getattr(clients, f"{provider}_client")

    first = accessor()
    assert first.credential == "old-credential"
    rotating_secret.value = "rotated-credential"
    rotating_secret.now += secrets.CACHE_TTL_SECONDS - 1
    assert accessor() is first
    assert rotating_secret.reads == 1

    rotating_secret.now += 1
    refreshed = accessor()
    assert refreshed is not first
    assert refreshed.credential == "rotated-credential"
    assert rotating_secret.reads == 2
    assert len(constructed) == 2

    # Further TTL refreshes with the same value preserve the connection pool.
    rotating_secret.now += secrets.CACHE_TTL_SECONDS
    assert accessor() is refreshed
    assert rotating_secret.reads == 3
    assert len(constructed) == 2


def test_explicit_openai_client_does_not_read_production_secrets(monkeypatch):
    def unexpected_secret(name):
        raise AssertionError("Injected evaluation clients must remain isolated")

    monkeypatch.setattr(secrets, "get_secret", unexpected_secret)
    fake = object()
    assert OpenAIProvider(client=fake)._client() is fake
