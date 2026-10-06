"""Regression checks derived from recent production tool failures."""

from tools import databricks_tool as dt
from tools import github_tool as gt
from utils.system_prompt import load_system_prompt


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP {self.status_code}")


def test_github_search_removes_bare_hyphen_before_request(monkeypatch):
    requests = []

    def fake_request(method, url, **kwargs):
        requests.append(kwargs["params"]["q"])
        return FakeResponse(200, {"total_count": 0, "items": []})

    monkeypatch.setattr(gt, "_headers", lambda *args, **kwargs: {})
    monkeypatch.setattr(gt.requests, "request", fake_request)

    result = gt.search_code("Orders - Merge path:notebooks")

    assert requests == ["Orders Merge path:notebooks repo:example-org/example-data-repo"]
    assert result["normalized_query"] == "Orders Merge path:notebooks"


def test_github_search_retries_422_with_conservative_terms(monkeypatch):
    requests = []
    responses = [
        FakeResponse(422),
        FakeResponse(
            200,
            {
                "total_count": 1,
                "items": [
                    {
                        "path": "notebooks/Orders - Merge.sql",
                        "sha": "abc",
                        "html_url": "https://example.test/file",
                    }
                ],
            },
        ),
    ]

    def fake_request(method, url, **kwargs):
        requests.append(kwargs["params"]["q"])
        return responses.pop(0)

    monkeypatch.setattr(gt, "_headers", lambda *args, **kwargs: {})
    monkeypatch.setattr(gt.requests, "request", fake_request)

    result = gt.search_code("Orders ::: Merge path:notebooks")

    assert len(requests) == 2
    assert requests[1] == "Orders Merge path:notebooks repo:example-org/example-data-repo"
    assert result["total_count"] == 1
    assert result["normalized_query"] == "Orders Merge path:notebooks"


def test_github_file_discovery_returns_paths_without_code_search(monkeypatch):
    gt._tree_cache.update({"paths": None, "truncated": False, "expires_at": 0.0})

    def fake_request(method, url, **kwargs):
        assert "/git/trees/main" in url
        assert kwargs["params"] == {"recursive": "1"}
        return FakeResponse(
            200,
            {
                "tree": [
                    {"type": "blob", "path": "notebooks/Orders - Merge.sql"},
                    {"type": "blob", "path": "resources/jobs/adt.job.yml"},
                    {"type": "tree", "path": "notebooks"},
                ]
            },
        )

    monkeypatch.setattr(gt, "_headers", lambda *args, **kwargs: {})
    monkeypatch.setattr(gt.requests, "request", fake_request)

    result = gt.find_files("Orders - Merge", path_prefix="notebooks/")

    assert result["matches"] == ["notebooks/Orders - Merge.sql"]
    assert result["match_count"] == 1


def test_typed_object_discovery_builds_exact_metadata_sql(monkeypatch):
    captured = {}

    def fake_execute(sql, **kwargs):
        captured["sql"] = sql
        return {"status": "SUCCEEDED", "row_count": 1}

    monkeypatch.setattr(dt, "execute_sql", fake_execute)

    result = dt.find_objects("signup_unified")

    assert "FROM system.information_schema.tables" in captured["sql"]
    assert "LOWER(table_name) = LOWER(CAST(unhex(" in captured["sql"]
    assert "signup_unified" not in captured["sql"]
    assert result["metadata_discovery"]["match_count"] == 1


def test_typed_object_description_escapes_literal_inputs(monkeypatch):
    captured = {}

    def fake_execute(sql, **kwargs):
        captured["sql"] = sql
        return {"status": "SUCCEEDED", "row_count": 2}

    monkeypatch.setattr(dt, "execute_sql", fake_execute)

    result = dt.describe_object("acme_prod", "report'ing", "signups")

    assert "report'ing" not in captured["sql"]
    assert "table_schema = CAST(unhex(" in captured["sql"]
    assert result["object"]["full_name"] == "acme_prod.report'ing.signups"
    assert result["object"]["schema"] == "report'ing"


def test_metadata_literals_encode_backslashes_and_quotes_without_sql_metacharacters():
    encoded = dt._metadata_hex_literal(r"report\'ing", "schema")

    assert encoded == r"report\'ing".encode("utf-8").hex()
    assert "\\" not in encoded
    assert "'" not in encoded


def test_limited_sql_preflight_catches_high_confidence_shape_errors():
    metadata_error = dt._validate_sql("DESCRIBE TABLE system.access.audit LIMIT 10")
    window_error = dt._validate_sql(
        "SELECT id FROM source WHERE ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts) = 1"
    )

    assert metadata_error["validation_failure"] == "invalid_metadata_limit"
    assert window_error["validation_failure"] == "window_function_in_where"
    assert dt._validate_sql(
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts) AS rn "
        "FROM source QUALIFY rn = 1"
    ) is None


def test_execute_sql_repairs_trailing_metadata_limit_before_dispatch(monkeypatch):
    captured = {}
    audits = []

    def fake_request(method, api, **kwargs):
        captured.update({"method": method, "api": api, **kwargs})
        return FakeResponse(
            200,
            {"statement_id": "statement-1", "status": {"state": "SUCCEEDED"}},
        )

    monkeypatch.setattr(dt, "check_sql_global", lambda: (True, None))
    monkeypatch.setattr(dt, "_request_with_retry", fake_request)
    monkeypatch.setattr(
        dt,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    result = dt.execute_sql("DESCRIBE HISTORY acme_prod.reporting.signups LIMIT 5")

    assert captured["json"]["statement"] == (
        "DESCRIBE HISTORY acme_prod.reporting.signups"
    )
    assert result["status"] == "SUCCEEDED"
    assert result["automatic_repair"] == "removed_invalid_metadata_limit"
    assert ("sql_repaired", {"repair": "removed_invalid_metadata_limit", "sql": (
        "DESCRIBE HISTORY acme_prod.reporting.signups"
    )}) in audits


def test_semantic_check_runs_only_for_explicit_strict_funnel_aliases():
    warning = dt._funnel_semantic_check(
        ["segment", "funnel_numerator", "funnel_denominator", "funnel_rate"],
        [["mobile", "120", "100", "1.2"]],
    )

    assert warning["status"] == "WARNING"
    assert len(warning["violations"]) == 2
    assert dt._funnel_semantic_check(
        ["segment", "stage_one", "stage_two"],
        [["mobile", "120", "100"]],
    ) is None


def test_semantic_check_warns_when_final_funnel_aliases_are_renamed():
    warning = dt._funnel_semantic_check(
        ["coverage_view_week", "form_starts", "form_submits", "funnel_rate"],
        [["2026-08-02", "100", "30", "0.3"]],
    )

    assert warning["status"] == "WARNING"
    assert "funnel_denominator" in warning["violations"][0]
    assert "funnel_numerator" in warning["violations"][0]
    assert "outermost SELECT" in warning["instruction"]


def test_workspace_run_history_uses_jobs_api(monkeypatch):
    captured = {}

    def fake_request(method, api, **kwargs):
        captured.update({"method": method, "api": api, **kwargs})
        return FakeResponse(
            200,
            {
                "runs": [
                    {
                        "run_id": 456,
                        "job_id": 123,
                        "run_name": "[DAB] Example",
                        "run_type": "JOB_RUN",
                        "trigger": "PERIODIC",
                        "state": {
                            "life_cycle_state": "TERMINATED",
                            "result_state": "SUCCESS",
                        },
                        "start_time": 1786521600000,
                        "end_time": 1786521660000,
                        "run_page_url": "https://example.test/run/456",
                    }
                ]
            },
        )

    monkeypatch.setattr(dt, "_request_with_retry", fake_request)

    result = dt.list_job_runs("2026-08-12")

    assert captured["api"] == "jobs"
    assert captured["path"] == "runs/list"
    assert captured["params"]["limit"] == 25
    assert result["date"] == "2026-08-12"
    assert result["runs"][0]["run_name"] == "[DAB] Example"
    assert result["runs"][0]["result_state"] == "SUCCESS"
    assert "observed executions only" in result["interpretation_note"]


def test_workspace_run_history_follows_page_tokens(monkeypatch):
    calls = []
    responses = [
        FakeResponse(
            200,
            {
                "runs": [{"run_id": 1, "state": {}}],
                "has_more": True,
                "next_page_token": "next",
            },
        ),
        FakeResponse(200, {"runs": [{"run_id": 2, "state": {}}], "has_more": False}),
    ]

    def fake_request(method, api, **kwargs):
        calls.append(kwargs["params"])
        return responses.pop(0)

    monkeypatch.setattr(dt, "_request_with_retry", fake_request)

    result = dt.list_job_runs("2026-08-12", limit=50)

    assert calls[0]["limit"] == 25
    assert calls[1]["page_token"] == "next"
    assert [run["run_id"] for run in result["runs"]] == [1, 2]
    assert result["truncated"] is False


def test_active_run_listing_without_date_includes_older_active_runs(monkeypatch):
    captured = {}

    def fake_request(method, api, **kwargs):
        captured.update(kwargs["params"])
        return FakeResponse(200, {"runs": []})

    monkeypatch.setattr(dt, "_request_with_retry", fake_request)

    result = dt.list_job_runs(active_only=True)

    assert captured["active_only"] == "true"
    assert "start_time_from" not in captured
    assert "start_time_to" not in captured
    assert result["date"] is None
    assert result["scope"] == "currently_active"


def test_workspace_run_history_rejects_invalid_date_without_api_call(monkeypatch):
    monkeypatch.setattr(
        dt,
        "_request_with_retry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call API")),
    )

    result = dt.list_job_runs("08/12/2026")

    assert result["validation_failure"] == "invalid_date"


def test_prompt_routes_run_history_and_unqualified_objects_safely():
    prompt = load_system_prompt()

    assert "`databricks_list_job_runs(date=<YYYY-MM-DD>)`" in prompt
    assert "Do not query guessed `system.lakeflow` tables" in prompt
    assert "locate its definition first" in prompt
    assert "`FIELD_NOT_FOUND`, `TABLE_OR_VIEW_NOT_FOUND`" in prompt
    assert "`databricks_find_objects` and `databricks_describe_object`" in prompt
    assert "Strict funnel semantic check (opt-in)" in prompt
    assert "Do not use these aliases" in prompt
    assert "outermost final SELECT and returned column names" in prompt
    assert "Run timestamps prove only" in prompt
    assert "infer schedule cadence" in prompt
    assert "Do not infer processed data volume" in prompt
    assert '"3× longer"' in prompt
    assert "Before querying any `system.*` lineage table" in prompt
    assert "rely on the SQL guard to reject it" in prompt
