import base64
import json
from urllib.parse import urlencode

import receiver
import worker
from utils import feedback, slack_receiver


class _FakeTable:
    def __init__(self):
        self.items = []

    def put_item(self, *, Item):
        self.items.append(Item)
        return {}


def _block_action(user_id: str, rating: str) -> dict:
    return {
        "type": "block_actions",
        "user": {"id": user_id},
        "channel": {"id": "C123"},
        "container": {
            "channel_id": "C123",
            "thread_ts": "thread-1",
            "message_ts": "feedback-panel-1",
        },
        "trigger_id": "trigger-1",
        "actions": [
            {
                "action_id": f"feedback_{rating}",
                "value": json.dumps(
                    {"rating": rating, "answer_ts": "answer-1"}
                ),
            }
        ],
    }


def test_receiver_decodes_base64_interaction_before_signature(monkeypatch):
    raw_body = urlencode({"payload": json.dumps({"type": "unknown"})})
    verified = []
    monkeypatch.setattr(
        receiver,
        "verify_slack_signature",
        lambda headers, body: verified.append(body) or True,
    )

    response = receiver.lambda_handler(
        {
            "headers": {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            "body": base64.b64encode(raw_body.encode()).decode(),
            "isBase64Encoded": True,
        },
        None,
    )

    assert response["statusCode"] == 200
    assert verified == [raw_body]


def test_lightweight_receiver_opens_modal_with_slack_api(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(
        slack_receiver,
        "get_secret",
        lambda key: "xoxb-test" if key == "slack_bot_token" else "secret",
    )
    monkeypatch.setattr(slack_receiver, "urlopen", fake_urlopen)

    assert slack_receiver.open_modal("trigger-1", {"type": "modal"})
    assert captured["timeout"] == 2.0
    assert captured["request"].get_header("Authorization") == "Bearer xoxb-test"
    assert json.loads(captured["request"].data) == {
        "trigger_id": "trigger-1",
        "view": {"type": "modal"},
    }


def test_feedback_panel_keeps_all_three_shared_actions():
    blocks = feedback.feedback_prompt_blocks("answer-1")
    actions = blocks[1]["elements"]

    assert [button["text"]["text"] for button in actions] == [
        "Helpful",
        "Somewhat Helpful",
        "Needs a Fix",
    ]
    assert all(
        json.loads(button["value"])["answer_ts"] == "answer-1"
        for button in actions
    )


def test_helpful_action_is_sent_directly_to_worker(monkeypatch):
    invoked = []
    monkeypatch.setattr(
        receiver,
        "_invoke_worker",
        lambda payload: invoked.append(payload) or True,
    )

    response = receiver._handle_interaction(
        _block_action("U1", "helpful")
    )

    assert response["statusCode"] == 200
    assert invoked[0]["feedback_event"] == {
        "rating": "helpful",
        "answer_ts": "answer-1",
        "channel_id": "C123",
        "thread_ts": "thread-1",
        "user_id": "U1",
        "comment": "",
    }


def test_partial_rating_opens_private_comment_modal(monkeypatch):
    opened = []
    invoked = []
    monkeypatch.setattr(
        receiver,
        "open_modal",
        lambda trigger_id, view: opened.append((trigger_id, view)) or True,
    )
    monkeypatch.setattr(
        receiver,
        "_invoke_worker",
        lambda payload: invoked.append(payload) or True,
    )

    response = receiver._handle_interaction(
        _block_action("U2", "somewhat_helpful")
    )

    assert response["statusCode"] == 200
    assert not invoked
    assert opened[0][0] == "trigger-1"
    view = opened[0][1]
    assert view["callback_id"] == feedback.FEEDBACK_CALLBACK_ID
    assert view["blocks"][0]["label"]["text"] == "What was missing?"


def test_modal_submission_sends_private_comment_to_worker(monkeypatch):
    invoked = []
    monkeypatch.setattr(
        receiver,
        "_invoke_worker",
        lambda payload: invoked.append(payload) or True,
    )
    view = feedback.feedback_modal_view(
        rating="needs_fix",
        answer_ts="answer-1",
        channel_id="C123",
        thread_ts="thread-1",
    )
    view["state"] = {
        "values": {
            feedback.FEEDBACK_COMMENT_BLOCK_ID: {
                feedback.FEEDBACK_COMMENT_ACTION_ID: {
                    "value": "The date range is wrong."
                }
            }
        }
    }

    response = receiver._handle_interaction(
        {
            "type": "view_submission",
            "user": {"id": "U3"},
            "view": view,
        }
    )

    assert response["statusCode"] == 200
    assert invoked[0]["feedback_event"]["user_id"] == "U3"
    assert (
        invoked[0]["feedback_event"]["comment"]
        == "The date range is wrong."
    )


def test_multiple_users_get_distinct_records_and_can_update(monkeypatch):
    table = _FakeTable()
    monkeypatch.setattr(feedback, "_table", table)
    monkeypatch.setattr(feedback.time, "time", lambda: 1_800_000_000)

    common = {
        "channel_id": "C123",
        "thread_ts": "thread-1",
        "answer_ts": "answer-1",
    }
    feedback.save_feedback(**common, user_id="U1", rating="helpful")
    feedback.save_feedback(
        **common,
        user_id="U2",
        rating="needs_fix",
        comment="Wrong source.",
    )
    feedback.save_feedback(
        **common,
        user_id="U1",
        rating="somewhat_helpful",
        comment="Show the SQL.",
    )

    assert len(table.items) == 3
    assert {item["user_id"] for item in table.items} == {"U1", "U2"}
    assert table.items[0]["answer_key"] == table.items[1]["answer_key"]
    assert table.items[-1]["rating"] == "somewhat_helpful"


def test_worker_saves_feedback_and_sends_private_ack(monkeypatch):
    saved = []
    audits = []
    ephemeral = []
    monkeypatch.setattr(
        worker,
        "save_feedback",
        lambda **kwargs: saved.append(kwargs),
    )
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )
    monkeypatch.setattr(
        worker,
        "post_ephemeral",
        lambda channel, user, text, thread_ts=None: ephemeral.append(
            (channel, user, text, thread_ts)
        ),
    )

    result = worker._handle_feedback_event(
        {
            "channel_id": "C123",
            "thread_ts": "thread-1",
            "answer_ts": "answer-1",
            "user_id": "U1",
            "rating": "helpful",
            "comment": "",
        }
    )

    assert result == {"ok": True}
    assert saved[0]["user_id"] == "U1"
    assert audits[0][0] == "answer_feedback"
    assert ephemeral[0][0:2] == ("C123", "U1")
    assert "Choose another rating anytime" in ephemeral[0][2]


def test_feedback_panel_is_excluded_from_slack_history(monkeypatch):
    monkeypatch.setattr(worker, "extract_image_blocks", lambda message: [])
    history = worker._build_message_history(
        [
            {"user": "U1", "text": "<@BOT> first question", "ts": "1"},
            {"user": "BOT", "text": "The answer.", "ts": "2"},
            {
                "user": "BOT",
                "text": feedback.FEEDBACK_PROMPT_TEXT,
                "ts": "3",
            },
            {"user": "U1", "text": "<@BOT> follow-up", "ts": "4"},
        ],
        "BOT",
        "4",
    )

    assert all(
        feedback.FEEDBACK_PROMPT_TEXT not in str(message)
        for message in history
    )
    assert len(history) == 3
