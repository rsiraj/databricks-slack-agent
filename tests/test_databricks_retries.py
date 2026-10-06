"""Bounded retry behavior for Databricks HTTP requests."""

from copy import deepcopy

import pytest
import requests

from tools import databricks_tool as dt


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


@pytest.fixture
def request_environment(monkeypatch):
    monkeypatch.setattr(dt, "_build_url", lambda api, path="": f"https://dbx.test/{api}/{path}")
    monkeypatch.setattr(dt, "_headers", lambda: {"Authorization": "Bearer test"})
    monkeypatch.setattr(dt.secrets, "randbelow", lambda upper: 0)
    sleeps = []
    monkeypatch.setattr(dt.time, "sleep", sleeps.append)
    return sleeps


def test_retries_429_and_5xx_then_returns_success(monkeypatch, request_environment):
    responses = [
        FakeResponse(429, headers={"Retry-After": "3"}),
        FakeResponse(503),
        FakeResponse(200, {"ok": True}),
    ]
    calls = []
    audits = []

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(dt.requests, "request", fake_request)
    monkeypatch.setattr(dt, "audit", lambda event, **fields: audits.append((event, fields)))

    result = dt._request_with_retry("GET", "jobs", path="runs/list", timeout=30)

    assert result.json() == {"ok": True}
    assert len(calls) == dt.MAX_HTTP_ATTEMPTS
    assert request_environment == [3.0, 2.0]
    assert [event for event, _ in audits] == [
        "databricks_transient_failure",
        "databricks_transient_failure",
        "databricks_recovery_succeeded",
    ]
    assert audits[0][1]["status_code"] == 429
    assert audits[-1][1]["transient_failure_count"] == 2


def test_retries_network_timeout_then_returns_success(monkeypatch, request_environment):
    outcomes = [requests.ReadTimeout("slow response"), FakeResponse(200)]
    calls = 0
    audits = []

    def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(dt.requests, "request", fake_request)
    monkeypatch.setattr(dt, "audit", lambda event, **fields: audits.append((event, fields)))

    result = dt._request_with_retry("GET", "sql", path="statement-1")

    assert result.status_code == 200
    assert calls == 2
    assert request_environment == [1.0]
    assert audits[0][0] == "databricks_transient_failure"
    assert audits[0][1]["failure_type"] == "ReadTimeout"
    assert audits[1][0] == "databricks_recovery_succeeded"


def test_stops_after_bounded_retryable_http_attempts(monkeypatch, request_environment):
    calls = 0
    audits = []

    def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse(599)

    monkeypatch.setattr(dt.requests, "request", fake_request)
    monkeypatch.setattr(dt, "audit", lambda event, **fields: audits.append((event, fields)))

    result = dt._request_with_retry("GET", "jobs", path="runs/list")

    assert result.status_code == 599
    assert calls == dt.MAX_HTTP_ATTEMPTS
    assert len(request_environment) == dt.MAX_HTTP_ATTEMPTS - 1
    assert [event for event, _ in audits] == [
        "databricks_transient_failure",
        "databricks_transient_failure",
        "databricks_transient_failure",
    ]
    assert audits[-1][1]["will_retry"] is False


def test_ordinary_4xx_is_not_retried(monkeypatch, request_environment):
    calls = 0

    def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse(403)

    monkeypatch.setattr(dt.requests, "request", fake_request)

    result = dt._request_with_retry("GET", "jobs", path="runs/list")

    assert result.status_code == 403
    assert calls == 1
    assert request_environment == []


def test_tls_error_is_not_retried(monkeypatch, request_environment):
    calls = 0

    def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise requests.exceptions.SSLError("bad certificate")

    monkeypatch.setattr(dt.requests, "request", fake_request)

    with pytest.raises(requests.exceptions.SSLError):
        dt._request_with_retry("GET", "sql", path="statement-1")

    assert calls == 1
    assert request_environment == []


def test_401_refreshes_credentials_at_most_once(monkeypatch, request_environment):
    responses = [FakeResponse(401), FakeResponse(401)]
    refreshes = []
    monkeypatch.setattr(dt.requests, "request", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(
        dt,
        "_load_creds",
        lambda force_refresh=False: refreshes.append(force_refresh) or ("dbx.test", "token"),
    )

    result = dt._request_with_retry("GET", "sql", path="statement-1")

    assert result.status_code == 401
    assert refreshes == [True]
    assert request_environment == []


def test_retry_after_is_capped(monkeypatch, request_environment):
    responses = [
        FakeResponse(429, headers={"Retry-After": "120"}),
        FakeResponse(200),
    ]
    monkeypatch.setattr(dt.requests, "request", lambda *args, **kwargs: responses.pop(0))

    dt._request_with_retry("GET", "jobs", path="runs/list")

    assert request_environment == [dt.MAX_RETRY_DELAY_SECONDS]


def test_run_job_reuses_one_idempotency_token_across_retries(
    monkeypatch,
    request_environment,
):
    responses = [FakeResponse(503), FakeResponse(200, {"run_id": 456, "number_in_job": 7})]
    request_bodies = []

    def fake_request(*args, **kwargs):
        request_bodies.append(deepcopy(kwargs["json"]))
        return responses.pop(0)

    monkeypatch.setattr(dt.requests, "request", fake_request)
    monkeypatch.setattr(dt, "audit", lambda *args, **kwargs: None)

    approval_id = "0123456789abcdef0123456789abcdef"
    result = dt.run_job(123, {"date": "2026-08-28"}, idempotency_token=approval_id)

    assert result["run_id"] == 456
    assert len(request_bodies) == 2
    assert request_bodies[0] == request_bodies[1]
    assert request_bodies[0]["idempotency_token"] == approval_id
    assert request_bodies[0]["notebook_params"] == {"date": "2026-08-28"}


def test_http_retries_consume_one_sql_rate_limit_check(monkeypatch, request_environment):
    payload = {
        "statement_id": "statement-1",
        "status": {"state": "SUCCEEDED"},
        "manifest": {"total_row_count": 0, "schema": {"columns": []}},
        "result": {"data_array": []},
    }
    responses = [FakeResponse(503), FakeResponse(200, payload)]
    checks = 0

    def fake_check_sql_global():
        nonlocal checks
        checks += 1
        return True, "ok"

    monkeypatch.setattr(dt.requests, "request", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(dt, "check_sql_global", fake_check_sql_global)

    result = dt.execute_sql("SELECT 1")

    assert result["status"] == "SUCCEEDED"
    assert checks == 1
