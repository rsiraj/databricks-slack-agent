"""Channel allowlist enforcement — utils/channel_access.py.

`is_channel_allowed` is one of the three checks gating where the bot will
operate. `allowed_channel_ids` is `@lru_cache`d and reads a config file, so each
test points the module at a temp fixture and clears the cache.
"""

import json

import pytest

from utils import channel_access


@pytest.fixture
def set_allowlist(tmp_path, monkeypatch):
    """Return a setter that writes a temp allowlist and wires the module to it."""

    def _set(mapping):
        f = tmp_path / "allowed_channels.json"
        f.write_text(json.dumps({"allowed_channels": mapping}))
        monkeypatch.setattr(channel_access, "ALLOWED_CHANNELS_FILE", f)
        channel_access.allowed_channel_ids.cache_clear()
        return f

    channel_access.allowed_channel_ids.cache_clear()
    yield _set
    channel_access.allowed_channel_ids.cache_clear()


class TestIsChannelAllowed:
    def test_on_list(self, set_allowlist):
        set_allowlist({"C123": "data-team", "C456": "analytics"})
        assert channel_access.is_channel_allowed("C123") is True

    def test_off_list(self, set_allowlist):
        set_allowlist({"C123": "data-team"})
        assert channel_access.is_channel_allowed("C999") is False

    def test_none_short_circuits_false(self, set_allowlist):
        set_allowlist({"C123": "data-team"})
        assert channel_access.is_channel_allowed(None) is False

    def test_empty_string_false(self, set_allowlist):
        set_allowlist({"C123": "data-team"})
        assert channel_access.is_channel_allowed("") is False


class TestAllowedChannelIds:
    def test_returns_keys_only(self, set_allowlist):
        set_allowlist({"C1": "one", "C2": "two"})
        assert channel_access.allowed_channel_ids() == frozenset({"C1", "C2"})

    def test_empty_map_fails_loud(self, set_allowlist):
        set_allowlist({})
        with pytest.raises(RuntimeError):
            channel_access.allowed_channel_ids()
