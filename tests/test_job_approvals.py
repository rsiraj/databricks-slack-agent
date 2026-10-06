"""Approval state transitions against a conditional, atomic UpdateItem fake."""

import copy
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from utils import job_approvals as approvals


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "UpdateItem")


class ConditionalTable:
    """Interpret the expression subset used here, enforcing it under one lock.

    This intentionally evaluates supplied conditions rather than implementing
    approval states itself: removing a condition from production code causes
    the corresponding tampering/replay test to fail.
    """

    def __init__(self):
        self.rows = {}
        self.calls = []
        self.failure = None
        self.lock = threading.Lock()

    def update_item(self, **request):
        with self.lock:
            self.calls.append(copy.deepcopy(request))
            if self.failure:
                raise _client_error(self.failure)
            key = request["Key"]["key"]
            row = copy.deepcopy(self.rows.get(key, {}))
            names = request["ExpressionAttributeNames"]
            values = request["ExpressionAttributeValues"]
            condition = request["ConditionExpression"]
            for clause in condition.split(" AND "):
                missing = re.fullmatch(r"attribute_not_exists\((#\w+)\)", clause)
                if missing:
                    passed = names[missing[1]] not in row
                else:
                    comparison = re.fullmatch(r"(#\w+)\s*(=|>)\s*(:\w+)", clause)
                    assert comparison, f"Unsupported condition: {clause}"
                    field, operator, value = comparison.groups()
                    actual = row.get(names[field])
                    expected = values[value]
                    passed = actual is not None and (
                        actual == expected if operator == "=" else actual > expected
                    )
                if not passed:
                    raise _client_error("ConditionalCheckFailedException")
            expression = request["UpdateExpression"]
            assert expression.startswith("SET ")
            for assignment in expression[4:].split(","):
                match = re.fullmatch(r"\s*(#\w+)\s*=\s*(:\w+)\s*", assignment)
                assert match, f"Unsupported update: {assignment}"
                row[names[match[1]]] = copy.deepcopy(values[match[2]])
            row.update(request["Key"])
            self.rows[key] = row
            if request.get("ReturnValues") == "ALL_NEW":
                return {"Attributes": copy.deepcopy(row)}
            return {}


@pytest.fixture
def table(monkeypatch):
    table = ConditionalTable()
    monkeypatch.setattr(approvals.ratelimit, "_table", table)
    monkeypatch.setattr(approvals, "time", SimpleNamespace(time=lambda: 1_000_000))
    return table


@pytest.fixture
def job_payload():
    return {
        "job_id": 123,
        "settings": {
            "name": "[DAB] Signups - Merge",
            "schedule": {"pause_status": "UNPAUSED", "quartz_cron_expression": "0 0 7 * * ?"},
            "git_source": {"git_branch": "main", "git_url": "https://github.com/example/data"},
            "tasks": [{"task_key": "merge", "notebook_task": {"notebook_path": "notebooks/merge"}}],
        },
    }


def _pending(job_payload, params=None):
    job = approvals.snapshot(job_payload)
    params = params or {"partition": "2026-09-08"}
    context = {"user_id": "U_REQUESTER", "channel_id": "C_PRIVATE", "thread_ts": "100.001"}
    approval_id = approvals.create(job, params, context)
    approvals.activate(approval_id, "101.001")
    event = {"approval_id": approval_id, "user_id": "U_OTHER_MEMBER",
             "channel_id": "C_PRIVATE", "thread_ts": "100.001", "message_ts": "101.001"}
    return event, {**job, "notebook_params": params}


@pytest.mark.parametrize("mode", ["schedule", "continuous", "trigger"])
def test_snapshot_accepts_explicitly_unpaused_automation(job_payload, mode):
    config = job_payload["settings"].pop("schedule")
    job_payload["settings"][mode] = config
    result = approvals.snapshot(job_payload)
    assert result["job_id"] == 123
    assert result["job_name"] == "[DAB] Signups - Merge"
    assert re.fullmatch(r"[0-9a-f]{64}", result["settings_fingerprint"])


@pytest.mark.parametrize("mode", ["schedule", "continuous", "trigger"])
@pytest.mark.parametrize("config", [None, {}, {"pause_status": "PAUSED"}, {"pause_status": "unpaused"}])
def test_snapshot_refuses_paused_or_unverifiable_automation(job_payload, mode, config):
    job_payload["settings"].pop("schedule")
    job_payload["settings"][mode] = config
    with pytest.raises(ValueError, match="explicitly unpaused"):
        approvals.snapshot(job_payload)


def test_snapshot_refuses_unscheduled_and_mixed_paused_configuration(job_payload):
    job_payload["settings"].pop("schedule")
    with pytest.raises(ValueError):
        approvals.snapshot(job_payload)
    job_payload["settings"].update(schedule={"pause_status": "UNPAUSED"}, trigger={"pause_status": "PAUSED"})
    with pytest.raises(ValueError):
        approvals.snapshot(job_payload)


@pytest.mark.parametrize("source", ["main", {}, {"git_branch": "feature/test"},
                                    {"git_tag": "main"}, {"git_commit": "abc123"}])
def test_snapshot_refuses_nonmain_or_unverified_git_source(job_payload, source):
    job_payload["settings"]["git_source"] = source
    with pytest.raises(ValueError, match="main branch"):
        approvals.snapshot(job_payload)


def test_snapshot_allows_current_ui_managed_job_without_git(job_payload):
    job_payload["settings"].pop("git_source")
    assert approvals.snapshot(job_payload)["job_id"] == 123


@pytest.mark.parametrize("field,value", [("next_page_token", "next"), ("has_more", True)])
def test_snapshot_refuses_partial_configuration(job_payload, field, value):
    job_payload[field] = value
    with pytest.raises(ValueError, match="full job configuration"):
        approvals.snapshot(job_payload)


@pytest.mark.parametrize("name", [None, "", "x" * 201, "PAUSED_USE_BUNDLE_VERSION_Signups"])
def test_snapshot_refuses_invalid_names_and_retired_predecessor(job_payload, name):
    job_payload["settings"]["name"] = name
    with pytest.raises(ValueError):
        approvals.snapshot(job_payload)


def test_snapshot_fingerprint_is_order_stable_but_covers_execution_settings(job_payload):
    original = approvals.snapshot(job_payload)
    reordered = copy.deepcopy(job_payload)
    reordered["settings"] = dict(reversed(list(reordered["settings"].items())))
    assert approvals.snapshot(reordered) == original
    job_payload["settings"]["tasks"][0]["notebook_task"]["notebook_path"] = "notebooks/different"
    assert approvals.snapshot(job_payload)["settings_fingerprint"] != original["settings_fingerprint"]


def test_claim_returns_only_stored_payload_and_records_different_confirmer(table, job_payload):
    event, expected = _pending(job_payload)
    event.update(job_id=999, notebook_params={"partition": "tampered"}, settings_fingerprint="tampered")
    assert approvals.claim(event) == expected
    row = table.rows[f"job_approval#{event['approval_id']}"]
    assert row["requester"] == "U_REQUESTER"
    assert row["confirmer"] == "U_OTHER_MEMBER"
    assert row["state"] == "executing"
    assert approvals.claim(event) is None


def test_first_concurrent_member_claim_wins_once(table, job_payload):
    event, expected = _pending(job_payload)
    barrier = threading.Barrier(8)

    def claim_as(index):
        barrier.wait()
        return approvals.claim({**event, "user_id": f"U_MEMBER_{index}"})

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(claim_as, range(8)))
    assert results.count(expected) == 1
    assert results.count(None) == 7
    assert table.rows[f"job_approval#{event['approval_id']}"]["confirmer"].startswith("U_MEMBER_")


@pytest.mark.parametrize("field,value", [("channel_id", "C_DIFFERENT"), ("thread_ts", "999.001"),
                                         ("message_ts", "999.002"), ("approval_id", "f" * 32)])
def test_tampered_binding_is_refused_without_consuming_original(table, job_payload, field, value):
    event, expected = _pending(job_payload)
    assert approvals.claim({**event, field: value}) is None
    assert approvals.claim(event) == expected


@pytest.mark.parametrize("approval_id", [None, 123, "", "f" * 31, "F" * 32, "../approval"])
def test_malformed_approval_ids_never_reach_dynamodb(table, approval_id):
    assert approvals.claim({"approval_id": approval_id}) is None
    assert table.calls == []


@pytest.mark.parametrize("elapsed,allowed", [(599, True), (600, False), (601, False)])
def test_expiry_is_enforced_even_while_row_still_exists(table, job_payload, monkeypatch, elapsed, allowed):
    event, expected = _pending(job_payload)
    monkeypatch.setattr(approvals, "time", SimpleNamespace(time=lambda: 1_000_000 + elapsed))
    assert approvals.claim(event) == (expected if allowed else None)
    assert f"job_approval#{event['approval_id']}" in table.rows


@pytest.mark.parametrize("state", ["preparing", "executing", "complete"])
def test_only_displayed_pending_approvals_can_be_claimed(table, job_payload, state):
    event, _ = _pending(job_payload)
    table.rows[f"job_approval#{event['approval_id']}"]["state"] = state
    assert approvals.claim(event) is None


def test_complete_records_result_and_cannot_complete_or_activate_twice(table, job_payload):
    event, expected = _pending(job_payload)
    with pytest.raises(ClientError):
        approvals.complete(event["approval_id"], {"run_id": 42})
    assert approvals.claim(event) == expected
    approvals.complete(event["approval_id"], {"run_id": 42})
    row = table.rows[f"job_approval#{event['approval_id']}"]
    assert row["state"] == "complete"
    assert json.loads(row["result"]) == {"run_id": 42}
    with pytest.raises(ClientError):
        approvals.complete(event["approval_id"], {"run_id": 43})
    with pytest.raises(ClientError):
        approvals.activate(event["approval_id"], "999.001")
    assert approvals.claim(event) is None


@pytest.mark.parametrize("code", ["AccessDeniedException", "ProvisionedThroughputExceededException", "InternalServerError"])
def test_dynamodb_claim_failure_never_returns_approval(table, job_payload, code):
    event, expected = _pending(job_payload)
    table.failure = code
    with pytest.raises(ClientError):
        approvals.claim(event)
    table.failure = None
    assert approvals.claim(event) == expected


def test_storage_failure_cannot_create_or_activate_executable_approval(table, job_payload):
    job = approvals.snapshot(job_payload)
    context = {"user_id": "U_REQUESTER", "channel_id": "C_PRIVATE", "thread_ts": "100.001"}
    table.failure = "AccessDeniedException"
    with pytest.raises(ClientError):
        approvals.create(job, {}, context)
    assert table.rows == {}

    table.failure = None
    approval_id = approvals.create(job, {}, context)
    event = {**context, "approval_id": approval_id, "message_ts": "101.001"}
    assert approvals.claim(event) is None
    table.failure = "InternalServerError"
    with pytest.raises(ClientError):
        approvals.activate(approval_id, "101.001")
    table.failure = None
    assert approvals.claim(event) is None
    assert table.rows[f"job_approval#{approval_id}"]["state"] == "preparing"


def test_stored_confirmation_payload_does_not_follow_mutated_input_objects(table, job_payload):
    event, original = _pending(job_payload, {"partition": "original"})
    original["notebook_params"]["partition"] = "changed after card creation"
    job_payload["settings"]["name"] = "Different job"
    claimed = approvals.claim(event)
    assert claimed["job_name"] == "[DAB] Signups - Merge"
    assert claimed["notebook_params"] == {"partition": "original"}


def test_create_collision_does_not_overwrite_existing_approval(table, job_payload, monkeypatch):
    fixed = SimpleNamespace(hex="a" * 32)
    monkeypatch.setattr(approvals, "uuid", SimpleNamespace(uuid4=lambda: fixed))
    event, expected = _pending(job_payload)
    original = copy.deepcopy(table.rows)
    with pytest.raises(ClientError):
        _pending(job_payload, {"partition": "different"})
    assert table.rows == original
    assert approvals.claim(event) == expected


def test_request_context_restores_outer_and_clears_after_exception():
    assert approvals.current_request() is None
    with approvals.request_context("U1", "C1", "1.1"):
        outer = approvals.current_request()
        with pytest.raises(RuntimeError), approvals.request_context("U2", "C2", "2.2"):
            assert approvals.current_request()["user_id"] == "U2"
            raise RuntimeError("abort")
        assert approvals.current_request() is outer
    assert approvals.current_request() is None


@pytest.mark.parametrize("params", [[], "bad", {"flag": True}, {1: "value"}, {"huge": "x" * 2001}])
def test_normalize_parameters_rejects_ambiguous_or_undisplayable_values(params):
    with pytest.raises(ValueError):
        approvals.normalize_parameters(params)


def test_confirmation_displays_exact_job_and_parameters_as_plain_text():
    params = {"message": "<@U123> *override*\n\"quoted\"", "Unicode": "✓"}
    job = {"job_id": 123, "job_name": "<@U456> *job*"}
    normalized = approvals.normalize_parameters(params)
    assert normalized == params and normalized is not params
    blocks = approvals.confirmation_blocks("a" * 32, job, normalized)
    assert blocks[0]["text"]["type"] == "plain_text"
    assert job["job_name"] in blocks[0]["text"]["text"]
    assert "job_id 123" in blocks[0]["text"]["text"]
    displayed = blocks[0]["text"]["text"].split("Parameters: ", 1)[1]
    assert json.loads(displayed) == params
    button = blocks[2]["elements"][0]
    assert button["action_id"] == approvals.CONFIRM_ACTION
    assert button["value"] == "a" * 32
