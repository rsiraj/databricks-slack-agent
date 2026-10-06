import pytest

from utils.job_run_evidence_guard import JobRunEvidenceGuard


def _guard_with_jobs(*jobs):
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, {
        "runs": [{"job_id": job_id, "run_name": name} for job_id, name in jobs],
    })
    return guard


def _baseline(guard, job_id):
    guard.record_tool_result("databricks_get_recent_runs", {"job_id": job_id}, {
        "job_id": job_id,
        "successful_run_summary": {"successful_run_count": 5},
        "runs": [{"duration_ms": 5000}],
    })


def _configuration(guard, name, path):
    guard.record_tool_result("github_get_file_content", {"path": f"resources/jobs/{name}.job.yml"}, {
        "content": f"resources:\n  jobs:\n    job:\n      name: {name}\n"
        f"      tasks:\n      - task_key: merge\n        notebook_task:\n          notebook_path: {path}\n"
        "      schedule:\n        quartz_cron_expression: 0 0 8 * * ?",
    })


def _notebook(guard, path, table):
    guard.record_tool_result("github_get_file_content", {"path": path + ".sql"}, {
        "content": f"SELECT * FROM {table}",
    })


def _volume(guard, table):
    guard.record_tool_result("databricks_execute_sql", {
        "sql": f"SELECT event_date, COUNT(*) FROM {table} GROUP BY event_date",
    }, {"row_count": 2, "sample_rows": [["2026-09-01", 5], ["2026-09-02", 50]]})


@pytest.mark.parametrize(("claim", "violation"), [
    ("{name} looks normal.", "recent_run_baseline_required"),
    ("{name} runs daily.", "schedule_configuration_required"),
    ("{name} takes time because of a full scan.", "structural_logic_required"),
    ("{name} took longer likely due to data volume.", "runtime_comparison_required"),
])
def test_evidence_for_one_job_cannot_authorize_another_job(claim, violation):
    guard = _guard_with_jobs((1, "Coverage"), (2, "Email"))
    _baseline(guard, 1)
    _configuration(guard, "Coverage", "notebooks/coverage")
    _notebook(guard, "notebooks/coverage", "catalog.coverage.events")
    _volume(guard, "catalog.coverage.events")

    assert guard.validate(claim.format(name="Coverage")) == []
    assert guard.validate(claim.format(name="`Coverage`")) == []
    assert guard.validate(claim.format(name="Email")) == [violation]
    assert guard.validate(claim.format(name="`Email`")) == [violation]
    assert guard.validate(claim.format(name="UnknownJob")) == [violation]
    assert guard.validate(claim.format(name="job_id 999")) == [violation]
    assert guard.validate(claim.format(name="Coverage and Email")) == [violation]
    assert guard.validate(claim.format(name="It")) == [violation]
    assert guard.validate("Coverage ran for 5m; Email ran for 9m.") == []


def test_single_job_implicit_baseline_is_supported_but_unknown_name_is_not():
    guard = _guard_with_jobs((1, "Coverage"))
    _baseline(guard, 1)
    assert guard.validate("Today's duration looks normal.") == []
    assert guard.validate("It looks normal.") == []
    assert guard.validate("The Coverage job looks normal.") == []
    assert guard.validate("UnseenPipeline looks normal.") == ["recent_run_baseline_required"]
    assert guard.validate("job_id 2 looks normal.") == ["recent_run_baseline_required"]
    assert guard.validate("Coverage looks normal; UnknownJob looks normal.") == ["recent_run_baseline_required"]
    assert guard.validate("Coverage looks normal and UnknownJob looks normal.") == ["recent_run_baseline_required"]


def test_same_display_name_for_two_jobs_requires_both_baselines():
    guard = _guard_with_jobs((1, "Shared Name"), (2, "Shared Name"))
    _baseline(guard, 1)
    assert guard.validate("Shared Name looks normal.") == ["recent_run_baseline_required"]
    assert guard.validate("job_id 1 looks normal.") == []


def test_unrelated_notebook_and_sql_do_not_supply_evidence_even_for_single_job():
    guard = _guard_with_jobs((1, "Coverage"))
    _notebook(guard, "notebooks/email", "catalog.email.events")
    _volume(guard, "catalog.email.events")
    assert guard.validate("Coverage took longer because of a full scan and data volume.") == [
        "structural_logic_required", "runtime_comparison_required",
    ]


def test_configuration_read_after_notebook_links_evidence_to_matching_job():
    guard = _guard_with_jobs((1, "Coverage"), (2, "Email"))
    _notebook(guard, "notebooks/coverage", "catalog.coverage.events")
    _volume(guard, "catalog.coverage.events")
    _configuration(guard, "Coverage", "notebooks/coverage")
    assert guard.validate("Coverage took longer because of a full scan and data volume.") == []
    assert guard.validate("Email took longer because of a full scan and data volume.") == [
        "structural_logic_required", "runtime_comparison_required",
    ]


def test_schedule_rows_only_authorize_the_job_returned_in_that_row():
    guard = _guard_with_jobs((1, "Coverage"), (2, "Email"))
    guard.record_tool_result("databricks_execute_sql", {
        "sql": "SELECT job_id, job_name, schedule FROM databricks_workspace_jobs",
    }, {"row_count": 2, "columns": ["job_id", "job_name", "schedule"],
        "sample_rows": [[1, "Coverage", {"quartz_cron_expression": "0 0 8 * * ?"}], [2, "Email", None]]})
    assert guard.validate("Coverage runs daily.") == []
    assert guard.validate("Email runs daily.") == ["schedule_configuration_required"]


def test_schedule_without_job_identifier_remains_unattributed():
    guard = _guard_with_jobs((1, "Coverage"))
    guard.record_tool_result("github_get_file_content", {"path": "resources/jobs/unknown.job.yml"}, {
        "content": "schedule:\n  quartz_cron_expression: 0 0 8 * * ?",
    })
    assert guard.validate("Coverage runs daily.") == ["schedule_configuration_required"]


@pytest.mark.parametrize("predicate", ["job_id = 1 OR job_id = 2", "EXISTS (SELECT 1 FROM other WHERE job_id = 1)"])
def test_sql_scope_cannot_be_inferred_from_disjunction_or_subquery(predicate):
    guard = _guard_with_jobs((1, "Coverage"), (2, "Email"))
    guard.record_tool_result("databricks_execute_sql", {
        "sql": f"SELECT event_date, COUNT(*) FROM run_history WHERE {predicate} GROUP BY event_date",
    }, {"row_count": 2})
    assert guard.validate("Coverage's runtime increased likely due to compute.") == ["runtime_comparison_required"]


def test_job_scoped_runtime_query_supports_only_that_job():
    guard = _guard_with_jobs((1, "Coverage"), (2, "Email"))
    guard.record_tool_result("databricks_execute_sql", {
        "sql": "SELECT event_date, COUNT(*) FROM run_history WHERE job_id = 1 GROUP BY event_date",
    }, {"row_count": 2})
    assert guard.validate("Coverage's runtime increased likely due to compute.") == []
    assert guard.validate("Email's runtime increased likely due to compute.") == ["runtime_comparison_required"]


# Source shape from resources/jobs/adt_email_session_rts_merge.job.yml in the
# data repository: the YAML emitter wraps this notebook's path after the hyphen.
_ADT_PATH = (
    "notebooks/Data Team/Gold Table Creation/Email Gold Table Creation/"
    "PartnerA Email Session - Merge"
)
_PARTNER_NAME = "[DAB] PartnerA Email Session with RTS - Merge"


@pytest.mark.parametrize("scalar", [
    "notebooks/Data Team/Gold Table Creation/Email Gold Table Creation/PartnerA Email Session -\n            Merge",
    "'notebooks/Data Team/Gold Table Creation/Email Gold Table Creation/PartnerA Email Session -\n            Merge'",
    '"notebooks/Data Team/Gold Table Creation/Email Gold Table Creation/PartnerA Email Session -\n            Merge"',
    ">-\n            notebooks/Data Team/Gold Table Creation/Email Gold Table Creation/PartnerA Email Session -\n            Merge",
])
def test_wrapped_yaml_notebook_paths_link_real_repository_source(scalar):
    guard = _guard_with_jobs((1, _PARTNER_NAME), (2, "Other Job"))
    guard.record_tool_result("github_get_file_content", {
        "path": "resources/jobs/adt_email_session_rts_merge.job.yml",
    }, {"content": (
        "resources:\n  jobs:\n    adt_email_session_rts_merge:\n"
        f"      name: '{_PARTNER_NAME}'\n"
        "      tasks:\n      - task_key: ADT_Email_Session\n        notebook_task:\n"
        f"          notebook_path: {scalar}\n          source: GIT\n"
        "      schedule:\n        quartz_cron_expression: 0 0 8 * * ?\n"
    )})
    _notebook(guard, _ADT_PATH, "acme_prod.reporting.adt_email_events")
    _volume(guard, "acme_prod.reporting.adt_email_events")
    assert guard.validate(f"{_PARTNER_NAME} runs daily.") == []
    assert guard.validate(f"{_PARTNER_NAME} took longer because of a full scan and data volume.") == []
    assert guard.validate("Other Job took longer because of a full scan and data volume.") == [
        "structural_logic_required", "runtime_comparison_required",
    ]


def test_yaml_jobs_and_nested_task_fields_do_not_share_evidence():
    guard = _guard_with_jobs((1, "Coverage"), (2, "Email"))
    guard.record_tool_result("github_get_file_content", {"path": "resources/jobs/multiple.job.yml"}, {
        "content": """resources:
  jobs:
    coverage:
      name: Coverage
      tasks:
      - task_key: merge
        notebook_task:
          notebook_path: notebooks/coverage
        # A task parameter called schedule is not a job's schedule.
        parameters:
          schedule: nightly
    email:
      name: Email
      schedule: {quartz_cron_expression: '0 0 8 * * ?'}
      tasks:
      - task_key: partitions
        for_each_task:
          task:
            task_key: merge
            notebook_task:
              notebook_path: notebooks/email
""",
    })
    _notebook(guard, "notebooks/email", "catalog.email.events")
    assert guard.validate("Email runs daily.") == []
    assert guard.validate("Coverage runs daily.") == ["schedule_configuration_required"]
    assert guard.validate("Email takes time because of a full scan.") == []
    assert guard.validate("Coverage takes time because of a full scan.") == ["structural_logic_required"]


@pytest.mark.parametrize("content", ["resources: [broken", "- not a bundle", "resources: {jobs: []}"])
def test_invalid_or_non_bundle_yaml_does_not_supply_evidence(content):
    guard = _guard_with_jobs((1, "Coverage"))
    guard.record_tool_result("github_get_file_content", {"path": "resources/jobs/bad.job.yml"}, {"content": content})
    assert guard.validate("Coverage runs daily.") == ["schedule_configuration_required"]


def test_comparative_cte_alias_is_not_mistaken_for_an_unrelated_source_table():
    guard = _guard_with_jobs((1, "Signups"), (2, "Email"))
    _configuration(guard, "Signups", "notebooks/Signups - Merge")
    _notebook(guard, "notebooks/Signups - Merge", "acme_prod.app_server.signup_unified")
    guard.record_tool_result("databricks_execute_sql", {
        "sql": (
            "WITH daily AS (SELECT date(event_date) AS day, COUNT(*) AS n "
            "FROM acme_prod.app_server.signup_unified GROUP BY 1) "
            "SELECT * FROM daily"
        ),
    }, {"row_count": 2, "sample_rows": [["2026-09-01", 5], ["2026-09-02", 50]]})
    assert guard.validate("Signups took longer likely due to data volume.") == []
    assert guard.validate("Email took longer likely due to data volume.") == ["runtime_comparison_required"]


def test_scala_notebook_file_matches_extensionless_bundle_path():
    guard = _guard_with_jobs((1, "MOAR"))
    _configuration(guard, "MOAR", "notebooks/Projects/Acme/MOAR")
    guard.record_tool_result("github_get_file_content", {
        "path": "notebooks/Projects/Acme/MOAR.scala",
    }, {"content": 'spark.table("catalog.schema.source").write.mode("overwrite").saveAsTable("target")'})
    assert guard.validate("MOAR takes time because of an overwrite.") == []
