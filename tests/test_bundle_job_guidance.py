"""Regression checks for bundle-managed Databricks job guidance."""

from tools import databricks_tool as dt
from utils.system_prompt import load_system_prompt


RETIRED_MIGRATION_JOB_IDS = {
    "47507576483886",
    "36579880797445",
    "1115167999804151",
    "1118874364574382",
    "900791718310120",
    "634686517299048",
}


def test_system_prompt_separates_dab_and_ui_managed_jobs():
    prompt = load_system_prompt()

    assert "conf/bundle-job-clones.json" not in prompt
    assert "resources/jobs/" in prompt
    assert "acme_prod.acme_history.databricks_workspace_jobs" in prompt
    assert "latest row per `job_id`" in prompt
    assert "git_source.git_branch" in prompt
    assert "non-main" in prompt
    assert "PAUSED_USE_BUNDLE_VERSION_" in prompt
    assert "00000000-0000-4000-8000-000000000001" in prompt

    for job_id in RETIRED_MIGRATION_JOB_IDS:
        assert job_id not in prompt


def test_system_prompt_keeps_bundle_configuration_out_of_ui():
    prompt = load_system_prompt()

    assert "Do not suggest editing a bundle-managed job in the Databricks UI" in prompt
    assert "bundle job changes must go through `resources/jobs/` in GitHub" in prompt


def test_all_prompt_sections_require_button_confirmation_and_no_typed_yes():
    prompt = load_system_prompt()
    assert "Confirm run" in prompt
    assert "Any current channel member can confirm once" in prompt
    assert "Typed yes/go does not authorize execution" in prompt
    assert "never call `databricks_run_job` without an explicit user yes" not in prompt.lower()
    assert "trigger the next job immediately" not in prompt.lower()


def test_run_job_403_reports_authorization_without_guessing_schedule(monkeypatch):
    class ForbiddenResponse:
        status_code = 403

    monkeypatch.setattr(dt, "_request_with_retry", lambda *args, **kwargs: ForbiddenResponse())

    result = dt.run_job(123, idempotency_token="confirmed-action")

    assert result["status_code"] == 403
    assert "not authorized" in result["error"]
    assert "does not prove the job is paused" in result["error"]
