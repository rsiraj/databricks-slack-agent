"""Admin-presence guard — utils/admins.py.

`channel_has_admin` decides whether the bot stays in a channel. `admin_user_ids`
is `@lru_cache`d and reads a config file, so tests point it at a temp fixture and
clear the cache.
"""

import json

import pytest

from utils import admins


@pytest.fixture
def set_admins(tmp_path, monkeypatch):
    def _set(ids):
        f = tmp_path / "admins.json"
        f.write_text(json.dumps({"admin_user_ids": ids}))
        monkeypatch.setattr(admins, "ADMINS_FILE", f)
        admins.admin_user_ids.cache_clear()
        return f

    admins.admin_user_ids.cache_clear()
    yield _set
    admins.admin_user_ids.cache_clear()


class TestChannelHasAdmin:
    def test_admin_present(self, set_admins):
        set_admins(["U_ADMIN", "U_OTHER_ADMIN"])
        assert admins.channel_has_admin(["U_RANDOM", "U_ADMIN"]) is True

    def test_no_admin_present(self, set_admins):
        set_admins(["U_ADMIN"])
        assert admins.channel_has_admin(["U_RANDOM", "U_NOBODY"]) is False

    def test_empty_member_list(self, set_admins):
        set_admins(["U_ADMIN"])
        assert admins.channel_has_admin([]) is False


class TestAdminUserIds:
    def test_returns_frozenset(self, set_admins):
        set_admins(["U1", "U2"])
        assert admins.admin_user_ids() == frozenset({"U1", "U2"})

    def test_empty_list_fails_loud(self, set_admins):
        set_admins([])
        with pytest.raises(RuntimeError):
            admins.admin_user_ids()
