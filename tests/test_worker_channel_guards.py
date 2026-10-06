"""Channel-access guards in worker.py: admin re-check on app_mention, visible
member_left verification failures, leave-even-if-post-fails, and bridge-mode
image reuse.
"""

from copy import deepcopy

import pytest

import worker


@pytest.fixture
def guard_env(monkeypatch):
    state = {
        "audits": [],
        "posts": [],
        "left": [],
        "runs": [],
        "members": frozenset({"UADMIN", "U1"}),
        "has_admin": True,
    }
    monkeypatch.setattr(worker, "is_channel_allowed", lambda channel: True)
    monkeypatch.setattr(worker, "is_private_channel", lambda channel: True)
    monkeypatch.setattr(worker, "get_channel_members", lambda channel: state["members"])
    monkeypatch.setattr(worker, "channel_has_admin", lambda members: state["has_admin"])
    monkeypatch.setattr(worker, "admin_user_ids", lambda: frozenset({"UADMIN"}))
    monkeypatch.setattr(worker, "get_bot_user_id", lambda: "UBOT")
    monkeypatch.setattr(worker, "check_user", lambda user_id: (True, None))
    monkeypatch.setattr(worker, "check_channel", lambda channel: (True, None))
    monkeypatch.setattr(worker, "load_session", lambda thread_ts: None)
    monkeypatch.setattr(worker, "extract_image_blocks", lambda message: [])
    monkeypatch.setattr(worker, "save_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker, "audit", lambda event, **fields: state["audits"].append((event, fields))
    )
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: state["posts"].append(text) or "ts",
    )
    monkeypatch.setattr(worker, "leave_channel", lambda channel: state["left"].append(channel) or True)

    def fake_run_agent(messages, channel, thread_ts, metrics=None):
        state["runs"].append(deepcopy(messages))
        if metrics is not None:
            metrics.outcome = "answered"
        return "answered"

    monkeypatch.setattr(worker, "run_agent", fake_run_agent)
    return state


def _mention(**overrides):
    ev = {"type": "app_mention", "channel": "C1", "user": "U1", "ts": "2.0", "text": "<@UBOT> hello"}
    ev.update(overrides)
    return ev


def _denials(state):
    return [fields for event, fields in state["audits"] if event == "channel_access_denied"]


class TestAppMentionAdminRecheck:
    def test_admin_present_proceeds_to_agent(self, guard_env):
        worker._handle_app_mention(_mention())

        assert len(guard_env["runs"]) == 1
        assert _denials(guard_env) == []

    def test_unverifiable_members_fails_closed_without_leaving(self, guard_env, caplog):
        guard_env["members"] = frozenset()
        caplog.set_level("WARNING")

        result = worker._handle_app_mention(_mention())

        assert result == {"ok": True}
        assert guard_env["runs"] == []
        assert guard_env["posts"] == []
        assert guard_env["left"] == []
        assert _denials(guard_env) == [{"reason": "admin_unverifiable", "trigger": "app_mention"}]
        assert "Could not verify admin presence" in caplog.text

    def test_no_admin_posts_notice_and_leaves(self, guard_env):
        guard_env["has_admin"] = False

        worker._handle_app_mention(_mention())

        assert guard_env["runs"] == []
        assert guard_env["posts"] == [worker.NO_ADMIN_LEFT_MESSAGE]
        assert guard_env["left"] == ["C1"]
        assert _denials(guard_env) == [{"reason": "no_admin", "trigger": "app_mention"}]

    def test_admin_check_runs_after_privacy_check(self, guard_env, monkeypatch):
        calls = []
        monkeypatch.setattr(
            worker, "is_private_channel", lambda channel: calls.append("private") or True
        )
        monkeypatch.setattr(
            worker, "get_channel_members", lambda channel: calls.append("members") or frozenset({"UADMIN"})
        )

        worker._handle_app_mention(_mention())

        assert calls == ["private", "members"]


class TestMemberLeft:
    def test_unverifiable_members_stays_but_audits(self, guard_env, caplog):
        guard_env["members"] = frozenset()
        caplog.set_level("WARNING")

        result = worker._handle_member_left(
            {"type": "member_left_channel", "channel": "C1", "user": "UADMIN"}
        )

        assert result == {"ok": True}
        assert guard_env["left"] == []
        assert guard_env["posts"] == []
        assert guard_env["audits"] == [
            (
                "admin_check_unverifiable",
                {"channel_id": "C1", "departed_admin": "UADMIN", "trigger": "member_left_channel"},
            )
        ]
        assert "Could not verify admin presence" in caplog.text

    def test_last_admin_departure_posts_and_leaves(self, guard_env):
        guard_env["has_admin"] = False

        worker._handle_member_left(
            {"type": "member_left_channel", "channel": "C1", "user": "UADMIN"}
        )

        assert guard_env["posts"] == [worker.NO_ADMIN_LEFT_MESSAGE]
        assert guard_env["left"] == ["C1"]

    def test_non_admin_departure_is_ignored(self, guard_env):
        worker._handle_member_left(
            {"type": "member_left_channel", "channel": "C1", "user": "U1"}
        )

        assert guard_env["audits"] == []
        assert guard_env["left"] == []


class TestLeaveEvenIfPostFails:
    def test_member_joined_leaves_when_farewell_post_raises(self, guard_env, monkeypatch):
        guard_env["has_admin"] = False

        def failing_post(*args, **kwargs):
            raise ConnectionError("slack down")

        monkeypatch.setattr(worker, "post_message", failing_post)

        result = worker._handle_member_joined(
            {"type": "member_joined_channel", "channel": "C1", "user": "UBOT"}
        )

        assert result == {"ok": True}
        assert guard_env["left"] == ["C1"]

    def test_member_joined_leaves_when_farewell_post_returns_none(self, guard_env, monkeypatch, caplog):
        monkeypatch.setattr(worker, "is_channel_allowed", lambda channel: False)
        monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: None)
        caplog.set_level("WARNING")

        worker._handle_member_joined(
            {"type": "member_joined_channel", "channel": "C1", "user": "UBOT"}
        )

        assert guard_env["left"] == ["C1"]
        assert "leaving anyway" in caplog.text

    def test_member_joined_with_admin_stays(self, guard_env):
        worker._handle_member_joined(
            {"type": "member_joined_channel", "channel": "C1", "user": "UBOT"}
        )

        assert guard_env["left"] == []
        assert guard_env["posts"] == []


class TestBridgeModeImages:
    def test_trigger_message_reuses_downloaded_images(self, monkeypatch):
        extracted = []
        monkeypatch.setattr(
            worker,
            "extract_image_blocks",
            lambda message: extracted.append(message["ts"]) or [{"type": "image", "source": {"ts": message["ts"]}}],
        )
        trigger_image = {"type": "image", "source": {"type": "base64", "data": "already"}}

        history = worker._build_message_history(
            [
                {"user": "U1", "text": "<@UBOT> first", "ts": "1", "files": [{}]},
                {"user": "UBOT", "text": "Answer one.", "ts": "2"},
                {"user": "U1", "text": "<@UBOT> follow-up", "ts": "3", "files": [{}]},
            ],
            "UBOT",
            "3",
            trigger_image_blocks=[trigger_image],
        )

        # Only the earlier user message was re-downloaded; the trigger reused its blocks.
        assert extracted == ["1"]
        assert history[-1]["role"] == "user"
        assert history[-1]["content"] == [
            {"type": "text", "text": "follow-up"},
            trigger_image,
        ]

    def test_trigger_without_images_still_keeps_text(self, monkeypatch):
        monkeypatch.setattr(worker, "extract_image_blocks", lambda message: [])

        history = worker._build_message_history(
            [
                {"user": "U1", "text": "<@UBOT> q", "ts": "1"},
                {"user": "UBOT", "text": "a", "ts": "2"},
                {"user": "U1", "text": "<@UBOT> q2", "ts": "3"},
            ],
            "UBOT",
            "3",
        )

        assert history[-1]["content"] == [{"type": "text", "text": "q2"}]

    def test_handle_app_mention_passes_trigger_images_when_bridging(self, guard_env, monkeypatch):
        image = {"type": "image", "source": {"type": "base64", "data": "x"}}
        downloads = []
        monkeypatch.setattr(
            worker, "extract_image_blocks", lambda message: downloads.append(message.get("ts")) or [image]
        )
        monkeypatch.setattr(
            worker,
            "get_thread_replies",
            lambda channel, thread_ts: [
                {"user": "U1", "text": "<@UBOT> first", "ts": "1.0"},
                {"user": "UBOT", "text": "Answer.", "ts": "1.5"},
                {"user": "U1", "text": "<@UBOT> hello", "ts": "2.0", "files": [{}]},
            ],
        )
        captured = {}

        def fake_history(replies, bot_user_id, trigger_ts, trigger_image_blocks=None):
            captured["trigger_ts"] = trigger_ts
            captured["trigger_image_blocks"] = trigger_image_blocks
            return [{"role": "user", "content": "bridged"}]

        monkeypatch.setattr(worker, "_build_message_history", fake_history)

        worker._handle_app_mention(_mention(thread_ts="1.0"))

        assert captured["trigger_ts"] == "2.0"
        assert captured["trigger_image_blocks"] == [image]
        # The event itself was downloaded exactly once, up front.
        assert downloads == ["2.0"]
        assert guard_env["runs"] == [[{"role": "user", "content": "bridged"}]]


@pytest.mark.parametrize("blocked_guard", ["channel", "privacy", "admin", "user_rate", "channel_rate"])
def test_blocked_mentions_never_download_images(guard_env, monkeypatch, blocked_guard):
    def unexpected_download(message):
        pytest.fail("Blocked mentions must not download attachments")
    monkeypatch.setattr(worker, "extract_image_blocks", unexpected_download)
    if blocked_guard == "channel":
        monkeypatch.setattr(worker, "is_channel_allowed", lambda channel: False)
    elif blocked_guard == "privacy":
        monkeypatch.setattr(worker, "is_private_channel", lambda channel: False)
    elif blocked_guard == "admin":
        guard_env["has_admin"] = False
    elif blocked_guard == "user_rate":
        monkeypatch.setattr(worker, "check_user", lambda user: (False, "limit"))
    else:
        monkeypatch.setattr(worker, "check_channel", lambda channel: (False, "limit"))
    worker._handle_app_mention(_mention(files=[{"mimetype": "image/png"}]))
    assert guard_env["runs"] == []
