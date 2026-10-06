import json

from utils import audit as audit_module


def test_audit_payload_is_present_in_rendered_log_message(monkeypatch):
    calls = []
    monkeypatch.setattr(
        audit_module._audit_logger,
        "info",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    audit_module.set_context(
        user_id="U123",
        channel_id="C123",
        thread_ts="1787948143.137669",
    )

    audit_module.audit(
        "tool_call",
        tool_name="databricks_execute_sql",
        outcome="success",
    )

    args, kwargs = calls[0]
    assert args[0] == "audit %s"
    rendered = json.loads(args[1])
    assert rendered["audit_event"] == "tool_call"
    assert rendered["outcome"] == "success"
    assert rendered["user_id"] == "U123"
    assert rendered["thread_ts"] == "1787948143.137669"
    assert kwargs["extra"] == rendered


def test_explicit_context_field_overrides_ambient_value(monkeypatch):
    calls = []
    monkeypatch.setattr(
        audit_module._audit_logger,
        "info",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    audit_module.set_context(user_id="ambient-user")

    audit_module.audit("channel_access_denied", user_id="explicit-user")

    rendered = json.loads(calls[0][0][1])
    assert rendered["user_id"] == "explicit-user"

