import re

import worker


def test_error_reference_is_short_and_non_sensitive():
    error_id = worker._new_error_id()

    assert re.fullmatch(r"agent-[0-9a-f]{12}", error_id)


def test_safe_error_message_contains_reference_not_exception_details():
    raw_error = "ReadTimeoutError: internal.service.example/path?token=secret"

    message = worker._safe_agent_error_message("agent-0123456789ab")

    assert message == (
        ":warning: Something went wrong while processing this request. "
        "Reference: `agent-0123456789ab`."
    )
    assert raw_error not in message


class OverloadedError(Exception):
    status_code = 529
    request_id = "req-provider-123"


def test_overload_error_is_specific_actionable_and_searchable():
    error = OverloadedError("provider detail that should not reach Slack")

    message = worker._safe_agent_error_message("agent-0123456789ab", error)
    fields = worker._agent_error_fields(error)

    assert message == (
        ":warning: The AI service is temporarily overloaded. I retried "
        "automatically but couldn't complete this request. Please try again "
        "in a minute. Reference: `agent-0123456789ab`."
    )
    assert "provider detail" not in message
    assert fields == {
        "failure_type": "anthropic_overloaded",
        "status_code": 529,
        "provider_request_id": "req-provider-123",
    }


def test_failed_user_turn_is_persisted_with_visible_error(monkeypatch):
    saved = []
    messages = [{"role": "user", "content": "Please run it."}]
    monkeypatch.setattr(
        worker,
        "save_session",
        lambda thread_ts, value: saved.append((thread_ts, list(value))),
    )

    persisted = worker._persist_agent_failure_if_balanced(
        messages,
        "thread-1",
        "Temporary failure. Reference: agent-1.",
    )

    assert persisted is True
    assert messages[-1] == {
        "role": "assistant",
        "content": "Temporary failure. Reference: agent-1.",
    }
    assert saved == [("thread-1", messages)]


def test_failure_is_not_persisted_from_unbalanced_assistant_state(monkeypatch):
    monkeypatch.setattr(
        worker,
        "save_session",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not save")),
    )

    persisted = worker._persist_agent_failure_if_balanced(
        [{"role": "assistant", "content": [{"type": "tool_use"}]}],
        "thread-1",
        "Temporary failure.",
    )

    assert persisted is False
