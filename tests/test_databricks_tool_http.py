"""HTTP boundary behavior for tools/databricks_tool.py.

Covers actionable 4xx error bodies, statement_id path safety, deadline-aware
polling and retry sleeps, structured credential errors, and global SQL cap
accounting for metadata queries.
"""

import json

import pytest
import requests

from tools import databricks_tool as dt
from utils import deadline as run_deadline


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload

    def raise_for_status(self):  # pragma: no cover - the tool no longer calls this
        raise AssertionError("raise_for_status must not be used; parse the error body instead")


SUCCEEDED = {
    "statement_id": "01ef0000-aaaa-bbbb-cccc-000000000001",
    "status": {"state": "SUCCEEDED"},
    "manifest": {"total_row_count": 0, "schema": {"columns": []}},
    "result": {"data_array": []},
}


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def clear_deadline():
    run_deadline.clear()
    yield
    run_deadline.clear()


@pytest.fixture
def request_environment(monkeypatch):
    monkeypatch.setattr(dt, "_build_url", lambda api, path="": f"https://dbx.test/{api}/{path}")
    monkeypatch.setattr(dt, "_headers", lambda: {"Authorization": "Bearer super-secret-token"})
    monkeypatch.setattr(dt.secrets, "randbelow", lambda upper: 0)
    monkeypatch.setattr(dt, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(dt, "check_sql_global", lambda: (True, None))
    return monkeypatch


# ---------------------------------------------------------------------------
# 4xx error bodies
# ---------------------------------------------------------------------------


def test_execute_sql_surfaces_error_code_and_message_without_token(request_environment):
    request_environment.setattr(
        dt.requests,
        "request",
        lambda *a, **k: FakeResponse(
            400,
            {"error_code": "INVALID_PARAMETER_VALUE", "message": "Warehouse abc does not exist"},
        ),
    )

    result = dt.execute_sql("SELECT 1")

    assert result["status_code"] == 400
    assert result["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "INVALID_PARAMETER_VALUE" in result["error"]
    assert "Warehouse abc does not exist" in result["error"]
    assert "super-secret-token" not in json.dumps(result)
    assert "Bearer" not in json.dumps(result)


def test_execute_sql_without_a_configured_warehouse_fails_before_any_request(request_environment):
    request_environment.setattr(dt, "WAREHOUSE_ID", "")

    def forbidden_request(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("no HTTP request may be sent without a warehouse id")

    request_environment.setattr(dt.requests, "request", forbidden_request)

    result = dt.execute_sql("SELECT 1")

    assert result["configuration_failure"] == "missing_warehouse_id"
    assert "DATABRICKS_WAREHOUSE_ID" in result["error"]


def test_job_status_4xx_returns_structured_error_instead_of_raising(request_environment):
    request_environment.setattr(
        dt.requests,
        "request",
        lambda *a, **k: FakeResponse(
            400, {"error_code": "INVALID_PARAMETER_VALUE", "message": "Run 42 does not exist."}
        ),
    )

    result = dt.get_run_status(42)

    assert result == {
        "error": (
            "Databricks jobs.runs.get request failed (HTTP 400): [INVALID_PARAMETER_VALUE] "
            "Run 42 does not exist."
        ),
        "status_code": 400,
        "error_code": "INVALID_PARAMETER_VALUE",
    }


def test_non_json_error_body_is_truncated_text(request_environment):
    request_environment.setattr(
        dt.requests,
        "request",
        lambda *a, **k: FakeResponse(404, payload=None, text="<html>" + "x" * 1000),
    )

    result = dt.cancel_run(7)

    assert result["status_code"] == 404
    assert "error_code" not in result
    assert len(result["error"]) < 400


def test_exhausted_5xx_is_returned_as_structured_error(request_environment):
    request_environment.setattr(dt.time, "sleep", lambda s: None)
    request_environment.setattr(
        dt.requests,
        "request",
        lambda *a, **k: FakeResponse(503, {"error_code": "TEMPORARILY_UNAVAILABLE", "message": "busy"}),
    )

    result = dt.get_recent_runs(5)

    assert result["status_code"] == 503
    assert "temporarily unavailable" in result["error"]
    assert result["error_code"] == "TEMPORARILY_UNAVAILABLE"


# ---------------------------------------------------------------------------
# statement_id path safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement_id",
    ["../jobs/run-now", "abc/def12345", "short", "", None, "id with spaces", "a" * 129, "x?y=1234567"],
)
def test_get_statement_rejects_unsafe_statement_ids_without_a_request(
    request_environment, statement_id
):
    request_environment.setattr(
        dt.requests,
        "request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call API")),
    )

    result = dt.get_statement(statement_id)

    assert result["validation_failure"] == "invalid_argument"
    assert result["argument"] == "statement_id"


def test_get_statement_passes_valid_id_as_path(request_environment):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen["url"] = url
        return FakeResponse(200, SUCCEEDED)

    request_environment.setattr(dt.requests, "request", fake_request)

    result = dt.get_statement("01ef0000-aaaa-bbbb-cccc-000000000001")

    assert seen["url"] == "https://dbx.test/sql/01ef0000-aaaa-bbbb-cccc-000000000001"
    assert result["status"] == "SUCCEEDED"


def test_dispatch_returns_structured_error_for_bad_statement_id(request_environment):
    result = dt.dispatch("databricks_get_statement", {"statement_id": "../x"})

    assert result["validation_failure"] == "invalid_argument"


# ---------------------------------------------------------------------------
# Poll budget and deadline
# ---------------------------------------------------------------------------


def _pending_then_running(clock, polls):
    def fake_request(method, url, **kwargs):
        if method == "POST":
            return FakeResponse(
                200, {"statement_id": "stmt-000000001", "status": {"state": "PENDING"}}
            )
        polls.append(kwargs["timeout"])
        clock.sleep(1.0)  # each status poll takes one simulated second
        return FakeResponse(200, {"statement_id": "stmt-000000001", "status": {"state": "RUNNING"}})

    return fake_request


def test_poll_loop_uses_full_internal_budget_without_a_deadline(request_environment):
    clock = FakeClock()
    polls = []
    request_environment.setattr(dt.time, "time", clock.time)
    request_environment.setattr(dt.time, "sleep", clock.sleep)
    request_environment.setattr(dt.requests, "request", _pending_then_running(clock, polls))

    result = dt.execute_sql("SELECT 1")

    # 3s sleep + 1s request per poll; stop once fewer than 8s remain of the 90s cap.
    assert len(polls) == 21
    assert result["status"] == "RUNNING"
    assert result["statement_id"] == "stmt-000000001"
    assert "still running" in result["note"]
    assert "databricks_get_statement" in result["note"]


def test_poll_loop_stops_early_to_leave_response_reserve(request_environment):
    clock = FakeClock()
    polls = []
    request_environment.setattr(dt.time, "time", clock.time)
    request_environment.setattr(dt.time, "sleep", clock.sleep)
    request_environment.setattr(dt.requests, "request", _pending_then_running(clock, polls))
    # 20s of usable budget beyond the 30s reserve → 5 polls of ~4s would overrun; expect 4.
    run_deadline.set_deadline_from_remaining(dt.RESPONSE_RESERVE_SECONDS + 20)

    result = dt.execute_sql("SELECT 1")

    assert len(polls) == 4
    assert result["status"] == "RUNNING"
    assert run_deadline.remaining_seconds() >= dt.RESPONSE_RESERVE_SECONDS


def test_poll_loop_returns_immediately_when_budget_is_nearly_gone(request_environment):
    clock = FakeClock()
    polls = []
    request_environment.setattr(dt.time, "time", clock.time)
    request_environment.setattr(dt.time, "sleep", clock.sleep)
    request_environment.setattr(dt.requests, "request", _pending_then_running(clock, polls))
    run_deadline.set_deadline_from_remaining(dt.RESPONSE_RESERVE_SECONDS + 5)

    result = dt.execute_sql("SELECT 1")

    assert polls == []
    assert result["status"] == "PENDING"
    assert result["statement_id"] == "stmt-000000001"
    assert "still running" in result["note"]


def test_request_timeout_is_bounded_by_deadline(request_environment):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return FakeResponse(200, SUCCEEDED)

    request_environment.setattr(dt.requests, "request", fake_request)
    run_deadline.set_deadline_from_remaining(dt.RESPONSE_RESERVE_SECONDS + 12)

    dt.execute_sql("SELECT 1")

    assert 11.5 <= seen["timeout"] <= 12.0


def test_request_timeout_unchanged_without_deadline(request_environment):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return FakeResponse(200, SUCCEEDED)

    request_environment.setattr(dt.requests, "request", fake_request)

    dt.execute_sql("SELECT 1")

    assert seen["timeout"] == 30


def test_retry_backoff_never_sleeps_past_the_deadline(request_environment):
    sleeps = []
    calls = []
    request_environment.setattr(dt.time, "sleep", sleeps.append)
    request_environment.setattr(
        dt.requests, "request", lambda *a, **k: calls.append(1) or FakeResponse(503)
    )
    # ~1.5s of budget after the reserve: the first retry (1s) fits, the second (2s) does not.
    run_deadline.set_deadline_from_remaining(dt.RESPONSE_RESERVE_SECONDS + 1.5)

    response = dt._request_with_retry("GET", "jobs", path="runs/list")

    assert response.status_code == 503
    assert sleeps == [1.0]
    assert len(calls) == 2


def test_network_retry_is_skipped_when_deadline_has_no_room(request_environment):
    sleeps = []
    request_environment.setattr(dt.time, "sleep", sleeps.append)
    request_environment.setattr(
        dt.requests,
        "request",
        lambda *a, **k: (_ for _ in ()).throw(requests.ReadTimeout("slow")),
    )
    run_deadline.set_deadline_from_remaining(dt.RESPONSE_RESERVE_SECONDS)

    with pytest.raises(requests.ReadTimeout):
        dt._request_with_retry("GET", "sql", path="stmt-000000001")

    assert sleeps == []


# ---------------------------------------------------------------------------
# Global SQL cap accounting
# ---------------------------------------------------------------------------


def test_metadata_queries_do_not_consume_the_global_sql_cap(request_environment):
    checks = []
    request_environment.setattr(dt, "check_sql_global", lambda: checks.append(1) or (True, None))
    request_environment.setattr(dt.requests, "request", lambda *a, **k: FakeResponse(200, SUCCEEDED))

    described = dt.describe_object("acme_prod", "reporting", "signups")
    found = dt.find_objects("signups")

    assert described["status"] == "SUCCEEDED"
    assert found["status"] == "SUCCEEDED"
    assert checks == []

    dt.execute_sql("SELECT 1")
    assert checks == [1]


def test_dispatch_describe_does_not_consume_the_global_sql_cap(request_environment):
    request_environment.setattr(
        dt,
        "check_sql_global",
        lambda: (_ for _ in ()).throw(AssertionError("metadata query must not be counted")),
    )
    request_environment.setattr(dt.requests, "request", lambda *a, **k: FakeResponse(200, SUCCEEDED))

    result = dt.dispatch(
        "databricks_describe_object",
        {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
    )

    assert result["status"] == "SUCCEEDED"


def test_model_sql_is_still_blocked_by_the_global_cap(request_environment):
    request_environment.setattr(dt, "check_sql_global", lambda: (False, "global_sql_day (500/day)"))
    request_environment.setattr(
        dt.requests,
        "request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call API")),
    )

    result = dt.execute_sql("SELECT 1")

    assert result["rate_limit_failure"] == "global_sql_day (500/day)"


# ---------------------------------------------------------------------------
# Credential secret shape
# ---------------------------------------------------------------------------


class FakeSecretsClient:
    def __init__(self, secret_string):
        self.secret_string = secret_string

    def get_secret_value(self, SecretId):
        return {"SecretString": self.secret_string}


@pytest.fixture
def reset_creds_cache(monkeypatch):
    monkeypatch.setattr(dt, "_creds_cache", {"host": None, "token": None, "expires_at": 0.0})
    yield


def test_missing_credential_keys_return_structured_error(monkeypatch, reset_creds_cache):
    monkeypatch.setattr(dt, "_secrets_client", FakeSecretsClient(json.dumps({"host": "dbx.test"})))
    monkeypatch.setattr(
        dt.requests,
        "request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call API")),
    )

    result = dt.get_statement("stmt-000000001")

    assert result["config_failure"] == "missing_credential_keys"
    assert result["missing_keys"] == ["token"]
    assert "token" in result["error"]


def test_malformed_credential_secret_returns_structured_error(monkeypatch, reset_creds_cache):
    monkeypatch.setattr(dt, "_secrets_client", FakeSecretsClient("not json"))

    result = dt.dispatch("databricks_get_run_status", {"run_id": 1})

    assert result["config_failure"] == "malformed_credential_secret"


def test_valid_credentials_are_normalized(monkeypatch, reset_creds_cache):
    monkeypatch.setattr(
        dt,
        "_secrets_client",
        FakeSecretsClient(json.dumps({"host": "https://dbx.test/", "token": " dapi123 "})),
    )

    assert dt._load_creds() == ("dbx.test", "dapi123")
