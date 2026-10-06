"""Channel allowlist + privacy enforcement.

The bot only operates in:

1. Channels whose ID is listed in `config/allowed_channels.json`, AND
2. Channels that are PRIVATE (verified at runtime via `conversations.info`).

Both checks fail closed: if the allowlist file is missing/empty, or if the
privacy check errors out, the bot refuses to operate. Together with the
existing admin-presence guard in `utils/admins.py` this gives layered access
control:

  - Channel allowlist   — only PR-approved channel IDs are honored. The
                          primary enforcement boundary.
  - Privacy check       — even if a public channel were accidentally added
                          to the allowlist, runtime refuses it via
                          conversations.info.
  - Admin-presence      — at least one allowlisted admin must be a member
                          (defense-in-depth on top of the channel allowlist).

Note: Slack's platform requires several public-channel scopes
(`channels:read`, `channels:history`, `channels:manage`) as long as the
app subscribes to `app_mention` / `member_joined_channel` /
`member_left_channel`, so the bot token will retain those scopes. The
code-level allowlist + is_private check is the real security boundary;
the scopes are necessary but not sufficient.

Updates: edit `config/allowed_channels.json` and PR. Pull channel IDs from
Slack via channel name → ⓘ → "About" → bottom of the pane, or the URL slug.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_CHANNELS_FILE = (
    Path(__file__).resolve().parent.parent / "config" / "allowed_channels.json"
)


@lru_cache(maxsize=1)
def allowed_channel_ids() -> frozenset[str]:
    """Return the set of allowlisted channel IDs.

    The config stores a `{channel_id: human_name}` map; only the keys are
    functional, the names are documentation for PR review.

    Raises if the file is missing or has an empty map. Fail-loud is correct:
    an empty map would mean "the bot can operate in no channel at all", which
    is almost certainly a misconfiguration we want surfaced immediately on
    cold start rather than silently masking every mention.
    """
    blob = json.loads(ALLOWED_CHANNELS_FILE.read_text())
    channels = blob.get("allowed_channels") or {}
    if not channels:
        raise RuntimeError(
            "config/allowed_channels.json must contain a non-empty "
            "allowed_channels map"
        )
    return frozenset(channels.keys())


def is_channel_allowed(channel_id: str | None) -> bool:
    """True iff `channel_id` is on the allowlist."""
    if not channel_id:
        return False
    return channel_id in allowed_channel_ids()
