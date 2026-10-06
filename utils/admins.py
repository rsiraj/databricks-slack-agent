"""Admin allowlist for channel access.

The bot stays in any channel where at least one allowlisted admin is a member,
and auto-leaves any channel where none are. Enforced at:

- `member_joined_channel` (when the bot itself is added)
- `member_left_channel` (when an admin leaves; if they were the last admin, bot follows)

Updates: edit `config/admins.json` and PR. Find your Slack user ID via
profile → ⋮ → Copy member ID.
"""

import json
from functools import lru_cache
from pathlib import Path

ADMINS_FILE = Path(__file__).resolve().parent.parent / "config" / "admins.json"


@lru_cache(maxsize=1)
def admin_user_ids() -> frozenset[str]:
    """Return the set of allowlisted admin user IDs.

    Raises if the file is missing or has an empty list — fail-loud is the
    correct behavior here, since silently treating the list as empty would
    cause the bot to leave every channel it's added to.
    """
    blob = json.loads(ADMINS_FILE.read_text())
    ids = blob.get("admin_user_ids") or []
    if not ids:
        raise RuntimeError(
            "config/admins.json must contain a non-empty admin_user_ids list"
        )
    return frozenset(ids)


def channel_has_admin(channel_member_ids) -> bool:
    """True iff at least one admin is among the given channel members."""
    return bool(set(channel_member_ids) & admin_user_ids())
