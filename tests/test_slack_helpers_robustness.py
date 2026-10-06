"""Slack helper robustness — utils/slack_helpers.py.

Transport-level failures (timeouts, connection resets) raise generic exceptions,
not SlackApiError. Every helper must swallow them and return its documented
failure value. Also covers conversations.replies pagination and its hard cap.
"""

import pytest
from slack_sdk.errors import SlackApiError

from utils import slack_helpers


class _BrokenSlack:
    """Every API method raises a non-Slack exception."""

    def __getattr__(self, name):
        def _raise(**kwargs):
            raise ConnectionError(f"{name}: connection reset")

        return _raise


class _SlackApiFailure:
    def __getattr__(self, name):
        def _raise(**kwargs):
            raise SlackApiError("failed", {"ok": False, "error": "ratelimited"})

        return _raise


@pytest.fixture(params=[_BrokenSlack, _SlackApiFailure], ids=["transport_error", "slack_api_error"])
def broken(monkeypatch, request):
    monkeypatch.setattr(slack_helpers.clients, "slack_client", request.param())


class TestFailureValues:
    def test_post_message_returns_none(self, broken):
        assert slack_helpers.post_message("C1", "hi") is None

    def test_update_message_does_not_raise(self, broken):
        assert slack_helpers.update_message("C1", "1.0", "hi") is False

    def test_post_ephemeral_does_not_raise(self, broken):
        assert slack_helpers.post_ephemeral("C1", "U1", "hi") is None

    def test_leave_channel_returns_false(self, broken):
        assert slack_helpers.leave_channel("C1") is False

    def test_get_channel_members_returns_empty(self, broken):
        assert slack_helpers.get_channel_members("C1") == frozenset()

    def test_is_private_channel_returns_none(self, broken):
        assert slack_helpers.is_private_channel("C1") is None

    def test_get_thread_replies_returns_empty(self, broken):
        assert slack_helpers.get_thread_replies("C1", "1.0") == []


def test_transport_errors_are_logged_at_exception_level(monkeypatch, caplog):
    monkeypatch.setattr(slack_helpers.clients, "slack_client", _BrokenSlack())
    caplog.set_level("ERROR")

    slack_helpers.post_message("C1", "hi")

    record = next(r for r in caplog.records if "Slack post failed" in r.getMessage())
    assert record.exc_info is not None


class _PagedReplies:
    def __init__(self, pages, fail_on_page=None):
        self.pages = pages
        self.fail_on_page = fail_on_page
        self.calls = []

    def conversations_replies(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if self.fail_on_page is not None and index == self.fail_on_page:
            raise ConnectionError("boom")
        return self.pages[index]


class TestThreadRepliesPagination:
    def test_follows_next_cursor_until_exhausted(self, monkeypatch):
        client = _PagedReplies(
            [
                {"messages": [{"ts": "1"}, {"ts": "2"}], "response_metadata": {"next_cursor": "c2"}},
                {"messages": [{"ts": "3"}], "response_metadata": {"next_cursor": ""}},
            ]
        )
        monkeypatch.setattr(slack_helpers.clients, "slack_client", client)

        replies = slack_helpers.get_thread_replies("C1", "1", limit=2)

        assert [m["ts"] for m in replies] == ["1", "2", "3"]
        assert "cursor" not in client.calls[0]
        assert client.calls[0]["limit"] == 2
        assert client.calls[1]["cursor"] == "c2"

    def test_caps_at_max_thread_replies(self, monkeypatch):
        page = {
            "messages": [{"ts": str(i)} for i in range(400)],
            "response_metadata": {"next_cursor": "more"},
        }
        client = _PagedReplies([page] * 10)
        monkeypatch.setattr(slack_helpers.clients, "slack_client", client)

        replies = slack_helpers.get_thread_replies("C1", "1")

        assert len(replies) == slack_helpers.MAX_THREAD_REPLIES == 1000
        assert len(client.calls) == 3

    def test_partial_pages_are_returned_when_a_later_page_fails(self, monkeypatch):
        client = _PagedReplies(
            [
                {"messages": [{"ts": "1"}], "response_metadata": {"next_cursor": "c2"}},
                None,
            ],
            fail_on_page=1,
        )
        monkeypatch.setattr(slack_helpers.clients, "slack_client", client)

        assert slack_helpers.get_thread_replies("C1", "1") == [{"ts": "1"}]


def test_dead_helpers_are_gone():
    assert not hasattr(slack_helpers, "verify_slack_signature")
    assert not hasattr(slack_helpers, "open_modal")


class _ImageResponse:
    def __init__(self, chunks, headers=None):
        self.chunks = chunks
        self.headers = headers or {}
        self.closed = False
        self.read_chunks = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for chunk in self.chunks:
            self.read_chunks += 1
            yield chunk


def _image_download(monkeypatch, response):
    monkeypatch.setattr(slack_helpers, "get_secret", lambda name: "dummy")
    monkeypatch.setattr(slack_helpers.requests, "get", lambda *args, **kwargs: response)
    return {"mimetype": "image/png", "url_private": "https://files.slack.com/test"}


def test_image_download_stops_stream_at_byte_limit_and_closes(monkeypatch):
    response = _ImageResponse([b"1234", b"5678", b"must not read"])
    file_obj = _image_download(monkeypatch, response)
    assert slack_helpers.download_slack_image(file_obj, max_bytes=6) is None
    assert response.read_chunks == 2
    assert response.closed


def test_image_response_length_rejected_before_body_read(monkeypatch):
    response = _ImageResponse([b"must not read"], {"Content-Length": "100"})
    file_obj = _image_download(monkeypatch, response)
    assert slack_helpers.download_slack_image(file_obj, max_bytes=6) is None
    assert response.read_chunks == 0
    assert response.closed


def test_image_stream_budget_expiry_closes_response(monkeypatch):
    response = _ImageResponse([b"1234", b"must not read"])
    file_obj = _image_download(monkeypatch, response)
    times = iter([0.0, 31.0])
    monkeypatch.setattr(slack_helpers.time, "monotonic", lambda: next(times))
    assert slack_helpers.download_slack_image(file_obj, deadline=30.0) is None
    assert response.read_chunks == 1
    assert response.closed


def test_image_download_valid_bytes_are_preserved(monkeypatch):
    import base64
    response = _ImageResponse([b"1234", b"56"])
    file_obj = _image_download(monkeypatch, response)
    block = slack_helpers.download_slack_image(file_obj, max_bytes=6)
    assert base64.b64decode(block["source"]["data"]) == b"123456"
    assert response.closed


def test_image_download_closes_when_stream_raises(monkeypatch):
    def broken_stream():
        yield b"first"
        raise ConnectionError("stream failed")
    response = _ImageResponse(broken_stream())
    file_obj = _image_download(monkeypatch, response)
    assert slack_helpers.download_slack_image(file_obj) is None
    assert response.closed


def test_message_limits_attempted_images_and_aggregate_bytes(monkeypatch):
    import base64
    downloads = []
    def download(file_obj, *, max_bytes, deadline):
        downloads.append(max_bytes)
        return {"source": {"data": base64.b64encode(b"1234").decode()}}
    monkeypatch.setattr(slack_helpers, "download_slack_image", download)
    monkeypatch.setattr(slack_helpers, "MAX_MESSAGE_IMAGE_BYTES", 8)
    files = [{"mimetype": "image/png"}] * 20
    assert len(slack_helpers.extract_image_blocks({"files": files})) == 2
    assert downloads == [8, 4]
    downloads.clear()
    monkeypatch.setattr(slack_helpers, "MAX_MESSAGE_IMAGE_BYTES", 1000)
    assert len(slack_helpers.extract_image_blocks({"files": files})) == slack_helpers.MAX_IMAGES_PER_MESSAGE


def test_image_batch_interrupts_trickle_before_first_chunk_and_closes(monkeypatch):
    import signal
    import time
    if not hasattr(signal, "setitimer"):
        pytest.skip("Lambda timer requires POSIX")

    def trickle_before_chunk():
        # Emulate a stream receiving bytes continuously without filling the
        # first iter_content chunk: ordinary read timeouts never fire here.
        while True:
            time.sleep(0.001)
        yield b"unreachable"

    response = _ImageResponse(trickle_before_chunk())
    file_obj = _image_download(monkeypatch, response)
    monkeypatch.setattr(slack_helpers, "IMAGE_DOWNLOAD_BUDGET_SECONDS", 0.02)
    started = time.monotonic()
    assert slack_helpers.extract_image_blocks({"files": [file_obj]}) == []
    assert time.monotonic() - started < 1.0
    assert response.closed
    assert response.read_chunks == 0
