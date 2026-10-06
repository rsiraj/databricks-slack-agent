from copy import deepcopy
from types import SimpleNamespace

import pytest

import worker
from utils import clients
from utils import slack_helpers
from utils.slack_helpers import MessageDelivery


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text

    def model_dump(self):
        return {"type": "text", "text": self.text}


class _FakeResponse:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeAnthropicClient:
    def __init__(self, text):
        self.messages = self
        self.response = _FakeResponse(text)

    def create(self, **kwargs):
        return self.response


def test_chunk_delivery_stops_at_first_missing_timestamp(monkeypatch):
    attempts = []
    timestamps = iter(("part-1-ts", None, "should-not-be-used"))
    monkeypatch.setattr(
        slack_helpers,
        "post_message",
        lambda channel, text, thread_ts=None: (
            attempts.append(text) or next(timestamps)
        ),
    )

    delivery = slack_helpers.post_message_chunks(
        "C123",
        "\n\n".join("x" * 1_700 for _ in range(5)),
        thread_ts="thread-1",
    )

    assert delivery.total_chunks >= 3
    assert delivery.timestamps == ("part-1-ts",)
    assert not delivery.complete
    assert len(attempts) == 2
    assert delivery.delivered_text == delivery.chunks[0]


def test_sql_blocks_are_combined_and_uploaded_to_the_answer_thread(monkeypatch):
    uploads = []
    monkeypatch.setattr(
        slack_helpers.clients,
        "slack_client",
        SimpleNamespace(
            files_upload_v2=lambda **kwargs: uploads.append(kwargs)
            or {"files": [{"id": "file-1"}]}
        ),
    )

    blocks = slack_helpers.extract_sql_blocks(
        """First query:
```sql
SELECT 1;
```

Not SQL:
```python
print(1)
```

```databricks-sql
SELECT 2;
```"""
    )
    content = slack_helpers.render_sql_attachment(blocks)
    file_id = slack_helpers.upload_sql_file(
        "C123",
        content,
        thread_ts="thread-1",
        filename="queries.sql",
    )

    assert blocks == ("SELECT 1;", "SELECT 2;")
    assert content == "-- Query 1\nSELECT 1;\n\n-- Query 2\nSELECT 2;\n"
    assert file_id == "file-1"
    assert uploads == [
        {
            "channel": "C123",
            "thread_ts": "thread-1",
            "content": content,
            "filename": "queries.sql",
            "title": "SQL from Databricks Slack Agent",
            "snippet_type": "sql",
        }
    ]


def test_worker_attaches_sql_after_complete_answer_delivery(monkeypatch):
    delivery = MessageDelivery(("answer",), ("answer-ts",), 1)
    uploads = []
    feedback = []
    audits = []

    monkeypatch.setitem(
        clients._clients,
        "anthropic_client",
        _FakeAnthropicClient("Here it is:\n```sql\nSELECT 1;\n```"),
    )
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(worker, "post_message_chunks", lambda *args, **kwargs: delivery)
    monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: "notice-ts")
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "save_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "upload_sql_file",
        lambda *args, **kwargs: uploads.append((args, kwargs)) or "file-1",
    )
    monkeypatch.setattr(
        worker,
        "_post_feedback_prompt",
        lambda *args, **kwargs: feedback.append((args, kwargs)),
    )
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    worker.run_agent([{"role": "user", "content": "Write SQL"}], "C123", "123.456")

    assert uploads == [
        (
            ("C123", "SELECT 1;\n"),
            {
                "thread_ts": "123.456",
                "filename": "query-123-456.sql",
            },
        )
    ]
    assert feedback
    assert ("sql_attachment_uploaded", {
        "file_id": "file-1",
        "sql_block_count": 1,
        "sql_character_count": 10,
    }) in audits


def test_worker_does_not_attach_sql_after_partial_answer_delivery(monkeypatch):
    delivery = MessageDelivery(("part one", "part two"), ("part-1-ts",), 2)
    uploads = []

    monkeypatch.setitem(
        clients._clients,
        "anthropic_client",
        _FakeAnthropicClient("```sql\nSELECT 1;\n```"),
    )
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(worker, "post_message_chunks", lambda *args, **kwargs: delivery)
    monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: "notice-ts")
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "save_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "upload_sql_file", lambda *args, **kwargs: uploads.append(1))
    monkeypatch.setattr(worker, "_post_feedback_prompt", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "audit", lambda *args, **kwargs: None)

    worker.run_agent([{"role": "user", "content": "Write SQL"}], "C123", "123.456")

    assert uploads == []


@pytest.mark.parametrize(
    ("delivery", "expected_saves"),
    [
        (MessageDelivery(("part one", "part two"), ("part-1-ts",), 2), 1),
        (MessageDelivery(("part one",), (), 1), 0),
    ],
)
def test_worker_memory_and_feedback_follow_confirmed_delivery(
    monkeypatch,
    delivery,
    expected_saves,
):
    saved = []
    feedback = []
    audits = []
    notices = []
    monkeypatch.setitem(clients._clients, "anthropic_client", _FakeAnthropicClient("Complete answer"))
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(worker, "post_message_chunks", lambda *args, **kwargs: delivery)
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: (
            notices.append(text) or "notice-ts"
        ),
    )
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "save_session",
        lambda thread_ts, messages: saved.append(deepcopy(messages)),
    )
    monkeypatch.setattr(
        worker,
        "_post_feedback_prompt",
        lambda *args, **kwargs: feedback.append((args, kwargs)),
    )
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    messages = [{"role": "user", "content": "Question"}]
    worker.run_agent(messages, "C123", "thread-1")

    assert len(saved) == expected_saves
    assert not feedback
    assert audits[-1] == (
        "answer_delivery_failed",
        {
            "delivered_parts": len(delivery.timestamps),
            "total_parts": delivery.total_chunks,
        },
    )
    if delivery.timestamps:
        assert saved[-1][-1]["content"][0]["text"] == delivery.delivered_text
    else:
        assert any("Answer delivery stopped after 0 of 1 part(s)" in text for text in notices)
