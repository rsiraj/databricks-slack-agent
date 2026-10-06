import sys
from types import SimpleNamespace

from utils import clients, secrets


def test_anthropic_client_uses_configured_bounded_retries(monkeypatch):
    constructed = []

    class FakeAnthropic:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=FakeAnthropic),
    )
    monkeypatch.setattr(secrets, "get_secret", lambda name: "test-key")
    monkeypatch.setattr(clients, "_clients", {})
    monkeypatch.setattr(clients, "ANTHROPIC_MAX_RETRIES", 5)

    client = clients.__getattr__("anthropic_client")

    assert isinstance(client, FakeAnthropic)
    assert constructed == [{"api_key": "test-key", "max_retries": 5}]
