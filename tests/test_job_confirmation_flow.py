"""Offline receiver-to-worker job approval regressions, with real approval state."""

import copy
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest

import receiver
import worker
from tools import databricks_tool
from utils import job_approvals, slack_receiver
from test_job_approvals import ConditionalTable


@pytest.fixture
def flow(monkeypatch):
    table = ConditionalTable()
    state = {
        "table": table, "posts": [], "updates": [], "ephemeral": [], "runs": [],
        "saved": [], "lookups": [], "members": {"UREQUESTER", "UCONFIRMER", "UADMIN"},
        "allowed": True, "private": True, "has_admin": True,
        "payload": {"job_id": 123, "settings": {
            "name": "Merge signups", "schedule": {"pause_status": "UNPAUSED"},
            "git_source": {"git_branch": "main"},
            "tasks": [{"task_key": "merge", "notebook_task": {"notebook_path": "merge"}}],
        }},
    }
    monkeypatch.setattr(job_approvals.ratelimit, "_table", table)
    monkeypatch.setattr(worker, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "is_channel_allowed", lambda channel: state["allowed"])
    monkeypatch.setattr(worker, "is_private_channel", lambda channel: state["private"])
    monkeypatch.setattr(worker, "get_channel_members", lambda channel: state["members"])
    monkeypatch.setattr(worker, "channel_has_admin", lambda members: state["has_admin"])
    monkeypatch.setattr(worker, "load_session", lambda thread: [{"role": "user", "content": "prior question"}])
    monkeypatch.setattr(worker, "save_session", lambda thread, messages: state["saved"].append(copy.deepcopy(messages)))

    def post(channel, text, thread_ts=None, blocks=None):
        state["posts"].append({"channel": channel, "text": text, "thread_ts": thread_ts,
                               "blocks": copy.deepcopy(blocks)})
        return f"{100 + len(state['posts'])}.001"
    def update(channel, message_ts, text, blocks=None):
        state["updates"].append({"channel": channel, "message_ts": message_ts, "text": text,
                                 "blocks": copy.deepcopy(blocks)})
        return True
    def lookup(job_id):
        state["lookups"].append(job_id)
        return copy.deepcopy(state["payload"])
    def run(job_id, params, *, idempotency_token):
        state["runs"].append((job_id, copy.deepcopy(params), idempotency_token))
        if state.get("run_exception"):
            raise state["run_exception"]
        return state.get("run_result", {"run_id": 987})

    monkeypatch.setattr(worker, "post_message", post)
    monkeypatch.setattr(worker, "update_message", update)
    monkeypatch.setattr(worker, "post_ephemeral", lambda *args, **kwargs: state["ephemeral"].append((args, kwargs)))
    monkeypatch.setattr(databricks_tool, "get_job_configuration", lookup)
    monkeypatch.setattr(databricks_tool, "run_job", run)
    return state


def _request_card(state):
    with job_approvals.request_context("UREQUESTER", "CPRIVATE", "100.001"):
        result = worker._dispatch_tool("databricks_run_job", {"job_id": 123, "notebook_params": {"day": "2026-09-08"}})
    assert result["status"] == "confirmation_pending"
    key, row = next(iter(state["table"].rows.items()))
    return {"approval_id": key.removeprefix("job_approval#"), "user_id": "UCONFIRMER",
            "channel_id": "CPRIVATE", "thread_ts": "100.001", "message_ts": row["message_ts"]}


def _row(state, event):
    return state["table"].rows[f"job_approval#{event['approval_id']}"]


def test_tool_requests_exact_shared_card_without_launching_and_coalesces_duplicates(flow):
    params = {"day": "2026-09-08"}
    with job_approvals.request_context("UREQUESTER", "CPRIVATE", "100.001"):
        first = worker._dispatch_tool("databricks_run_job", {"job_id": 123, "notebook_params": params})
        second = worker._dispatch_tool("databricks_run_job", {"job_id": 123, "notebook_params": params})
    assert first == second and first["status"] == "confirmation_pending"
    assert "run_id" not in first
    assert flow["runs"] == []
    assert len(flow["posts"]) == 1
    card = flow["posts"][0]
    assert card["channel"] == "CPRIVATE" and card["thread_ts"] == "100.001"
    description = card["blocks"][0]["text"]
    assert description["type"] == "plain_text"
    assert "Merge signups (job_id 123)" in description["text"]
    assert json.loads(description["text"].split("Parameters: ")[1]) == params


def test_direct_databricks_dispatch_and_contextless_requests_cannot_run(flow):
    result = databricks_tool.dispatch("databricks_run_job", {"job_id": 123})
    assert result["validation_failure"] == "job_confirmation_required"
    result = worker._dispatch_tool("databricks_run_job", {"job_id": 123})
    assert result["validation_failure"] == "job_confirmation_required"
    assert flow["runs"] == flow["posts"] == []


def test_current_member_other_than_requester_confirms_once_with_visible_identity_and_persistence(flow):
    event = _request_card(flow)
    assert worker._handle_job_confirmation(event) == {"ok": True}
    assert flow["runs"] == [(123, {"day": "2026-09-08"}, event["approval_id"])]
    assert _row(flow, event)["confirmer"] == "UCONFIRMER"
    assert _row(flow, event)["requester"] == "UREQUESTER"
    assert _row(flow, event)["state"] == "complete"
    assert json.loads(_row(flow, event)["result"])["run_id"] == 987
    assert len(flow["updates"]) == 2
    for update in flow["updates"]:
        assert "<@UCONFIRMER>" in update["text"]
        assert update["message_ts"] == event["message_ts"]
        assert all(block["type"] != "actions" for block in update["blocks"])
    assert "Started run `987`" in flow["posts"][-1]["text"]
    assert flow["saved"][0][0]["content"] == "prior question"
    assert any("UCONFIRMER clicked Confirm run" in message["content"]
               for message in flow["saved"][0] if isinstance(message["content"], str))
    assert "987" in flow["saved"][0][-1]["content"]
    results = [block for message in flow["saved"][0] if isinstance(message["content"], list)
               for block in message["content"] if block["type"] == "tool_result"]
    assert json.loads(results[0]["content"])["run_id"] == 987
    worker._handle_job_confirmation({**event, "user_id": "UREQUESTER"})
    assert len(flow["runs"]) == 1


@pytest.mark.parametrize("guard", ["allowlist", "public", "privacy_unavailable", "nonmember", "no_admin", "membership_unavailable"])
def test_current_access_must_be_reverified_before_consuming_or_launching(flow, guard):
    event = _request_card(flow)
    if guard == "allowlist":
        flow["allowed"] = False
    elif guard == "public":
        flow["private"] = False
    elif guard == "privacy_unavailable":
        flow["private"] = None
    elif guard == "nonmember":
        flow["members"].remove("UCONFIRMER")
    elif guard == "no_admin":
        flow["has_admin"] = False
    else:
        flow["members"] = set()
    assert worker._handle_job_confirmation(event) == {"ok": False}
    assert flow["runs"] == []
    assert _row(flow, event)["state"] == "pending"


@pytest.mark.parametrize("invalid", ["expired", "already_executing", "wrong_message", "wrong_thread", "bad_id"])
def test_expired_duplicate_or_unbound_confirmation_never_launches(flow, invalid):
    event = _request_card(flow)
    if invalid == "expired":
        _row(flow, event)["ttl"] = 0
    elif invalid == "already_executing":
        _row(flow, event)["state"] = "executing"
    elif invalid == "wrong_message":
        event["message_ts"] = "999.001"
    elif invalid == "wrong_thread":
        event["thread_ts"] = "999.001"
    else:
        event["approval_id"] = "invalid"
    worker._handle_job_confirmation(event)
    assert flow["runs"] == []
    assert flow["ephemeral"]


@pytest.mark.parametrize("change", ["task", "paused", "nonmain", "lookup_failure", "wrong_job", "partial_configuration"])
def test_configuration_changes_or_failed_lookup_consume_button_without_launching(flow, change):
    event = _request_card(flow)
    if change == "task":
        flow["payload"]["settings"]["tasks"][0]["notebook_task"]["notebook_path"] = "different"
    elif change == "paused":
        flow["payload"]["settings"]["schedule"]["pause_status"] = "PAUSED"
    elif change == "nonmain":
        flow["payload"]["settings"]["git_source"]["git_branch"] = "feature/test"
    elif change == "lookup_failure":
        flow["payload"] = {"error": "lookup unavailable"}
    elif change == "wrong_job":
        flow["payload"]["job_id"] = 999
    else:
        flow["payload"]["next_page_token"] = "more"
    worker._handle_job_confirmation(event)
    assert flow["runs"] == []
    assert _row(flow, event)["state"] == "complete"
    assert "No run was submitted" in flow["posts"][-1]["text"]
    worker._handle_job_confirmation(event)
    assert flow["runs"] == []


@pytest.mark.parametrize("uncertain", ["exception", "error_result", "missing_run_id"])
def test_uncertain_launch_result_never_retries_or_reactivates_button(flow, uncertain):
    event = _request_card(flow)
    if uncertain == "exception":
        flow["run_exception"] = ConnectionError("connection reset after send")
    elif uncertain == "error_result":
        flow["run_result"] = {"error": "read timed out"}
    else:
        flow["run_result"] = {}
    worker._handle_job_confirmation(event)
    worker._handle_job_confirmation({**event, "user_id": "UREQUESTER"})
    assert len(flow["runs"]) == 1
    assert flow["runs"][0][2] == event["approval_id"]
    assert _row(flow, event)["state"] == "complete"
    assert "Check its recent runs" in flow["posts"][-1]["text"]
    assert "Started run" not in flow["posts"][-1]["text"]


def test_dynamodb_claim_failure_never_launches(flow):
    event = _request_card(flow)
    flow["table"].failure = "InternalServerError"
    worker._handle_job_confirmation(event)
    assert flow["runs"] == []
    assert _row(flow, event)["state"] == "pending"
    assert flow["ephemeral"]


def _signed_click(event):
    payload = {"type": "block_actions", "trigger_id": "trigger-123", "user": {"id": event["user_id"]},
               "channel": {"id": event["channel_id"]},
               "message": {"ts": event["message_ts"], "thread_ts": event["thread_ts"]},
               "actions": [{"action_id": job_approvals.CONFIRM_ACTION, "value": event["approval_id"]}]}
    body = urlencode({"payload": json.dumps(payload)})
    timestamp = str(int(time.time()))
    signature = "v0=" + hmac.new(b"test-signing-secret", f"v0:{timestamp}:{body}".encode(), hashlib.sha256).hexdigest()
    return {"headers": {"content-type": "application/x-www-form-urlencoded",
                         "x-slack-request-timestamp": timestamp, "x-slack-signature": signature}, "body": body}


@pytest.mark.parametrize("valid_signature", [True, False])
def test_receiver_authenticates_button_and_forwards_bound_event(flow, monkeypatch, valid_signature):
    event = _request_card(flow)
    request = _signed_click(event)
    if not valid_signature:
        request["headers"]["x-slack-signature"] = "v0=invalid"
    monkeypatch.setattr(slack_receiver, "get_secret", lambda name: "test-signing-secret")
    monkeypatch.setattr(receiver, "claim_once", lambda *args, **kwargs: True)
    forwarded = []
    def invoke(payload):
        forwarded.append(payload)
        worker._handle_job_confirmation(payload["job_confirmation_event"])
        return True
    monkeypatch.setattr(receiver, "_invoke_worker", invoke)
    result = receiver.lambda_handler(request, None)
    assert result["statusCode"] == (200 if valid_signature else 401)
    assert forwarded == ([{"job_confirmation_event": event}] if valid_signature else [])
    assert len(flow["runs"]) == int(valid_signature)


def test_receiver_failed_referral_returns_retryable_status_and_releases_delivery_claim(flow, monkeypatch):
    event = _request_card(flow)
    monkeypatch.setattr(slack_receiver, "get_secret", lambda name: "test-signing-secret")
    monkeypatch.setattr(receiver, "claim_once", lambda *args, **kwargs: True)
    monkeypatch.setattr(receiver, "_invoke_worker", lambda payload: False)
    released = []
    monkeypatch.setattr(receiver, "release_claim", lambda key: released.append(key))
    result = receiver.lambda_handler(_signed_click(event), None)
    assert result["statusCode"] == 500
    assert released == ["dedupe#interaction#trigger-123"]
    assert flow["runs"] == []


def test_initial_lookup_failure_creates_no_card_or_approval(flow):
    flow["payload"] = {"error": "lookup unavailable"}
    with job_approvals.request_context("UREQUESTER", "CPRIVATE", "100.001"):
        result = worker._dispatch_tool("databricks_run_job", {"job_id": 123})
    assert "error" in result
    assert flow["runs"] == flow["posts"] == []
    assert flow["table"].rows == {}


def test_card_delivery_failure_does_not_leave_an_executable_approval(flow, monkeypatch):
    monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: None)
    with job_approvals.request_context("UREQUESTER", "CPRIVATE", "100.001"):
        result = worker._dispatch_tool("databricks_run_job", {"job_id": 123})
    assert "error" in result
    assert flow["runs"] == []
    assert all(row["state"] == "preparing" for row in flow["table"].rows.values())


def test_failed_outcome_persistence_does_not_reopen_consumed_approval(flow, monkeypatch):
    event = _request_card(flow)
    def unavailable(*args, **kwargs):
        raise ConnectionError("storage unavailable")
    monkeypatch.setattr(job_approvals, "complete", unavailable)
    worker._handle_job_confirmation(event)
    worker._handle_job_confirmation(event)
    assert len(flow["runs"]) == 1
    assert _row(flow, event)["state"] == "executing"


def test_cannot_launch_when_shared_confirmation_cannot_be_published(flow, monkeypatch):
    event = _request_card(flow)
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: False)
    monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: None)
    assert worker._handle_job_confirmation(event) == {"ok": False}
    assert flow["runs"] == []
    assert _row(flow, event)["state"] == "complete"
    assert "no run submitted" in _row(flow, event)["result"]


def test_shared_fallback_post_can_publish_confirmation_when_card_update_fails(flow, monkeypatch):
    event = _request_card(flow)
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: False)
    assert worker._handle_job_confirmation(event) == {"ok": True}
    assert len(flow["runs"]) == 1
    assert any("<@UCONFIRMER> confirmed" in post["text"] for post in flow["posts"])


def test_confirmed_card_update_is_persisted_even_if_new_reply_fails(flow, monkeypatch):
    event = _request_card(flow)
    monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: None)
    worker._handle_job_confirmation(event)
    assert len(flow["runs"]) == 1
    assert "Started run `987`" in flow["saved"][0][-1]["content"]


@pytest.mark.parametrize("job_id", [123.9, True, None, [], "not-a-job", 0, -1])
def test_invalid_job_id_never_resolves_a_different_job(flow, job_id):
    with job_approvals.request_context("UREQUESTER", "CPRIVATE", "100.001"):
        result = worker._dispatch_tool("databricks_run_job", {"job_id": job_id})
    assert result["validation_failure"] == "job_confirmation_required"
    assert flow["lookups"] == flow["runs"] == flow["posts"] == []
