import re
from copy import deepcopy

import worker
from evals.synthetic_tools import load_fixture
from utils import clients
from utils.job_run_evidence_guard import JobRunEvidenceGuard
from utils.slack_helpers import MessageDelivery


def _list_runs_result():
    return {
        "date": "2026-08-12",
        "timezone": "America/New_York",
        "fetched_run_count": 3,
        "runs": [
            {
                "job_id": 1,
                "run_name": "Coverage",
                "result_state": "SUCCESS",
                "life_cycle_state": "TERMINATED",
                "start_time_ms": 1_786_556_730_000,
                "end_time_ms": 1_786_557_090_000,
            },
            {
                "job_id": 1,
                "run_name": "Coverage",
                "result_state": "SUCCESS",
                "life_cycle_state": "TERMINATED",
                "start_time_ms": 1_786_563_930_000,
                "end_time_ms": 1_786_564_110_000,
            },
            {
                "job_id": 2,
                "run_name": "Ops Email",
                "result_state": "SUCCESS",
                "life_cycle_state": "TERMINATED",
                "start_time_ms": 1_786_532_400_000,
                "end_time_ms": 1_786_536_391_371,
            },
        ],
    }


def _recent_runs_result():
    return {
        "job_id": 1,
        "successful_run_summary": {
            "successful_run_count": 15,
            "median_duration_ms": 2_684_037,
            "min_duration_ms": 2_489_119,
            "max_duration_ms": 4_101_060,
        },
        "runs": [{"duration_ms": 3_991_371}],
    }


def test_observed_job_run_comparison_needs_no_extra_evidence():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())

    assert guard.validate("Ops Email ran longest at 66m; PartnerB was next at 46m.") == []


def test_job_run_interpretations_require_matching_evidence():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())

    violations = guard.validate(
        "It runs every two hours. Today's duration looks normal, probably because of "
        "higher data volume."
    )

    assert violations == [
        "schedule_configuration_required",
        "recent_run_baseline_required",
        "runtime_comparison_required",
    ]


def test_recent_runs_support_normality_but_not_cadence_or_cause():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())
    guard.record_tool_result(
        "databricks_get_recent_runs",
        {"job_id": 1},
        _recent_runs_result(),
    )

    assert guard.validate("Coverage's duration is slower than normal.") == []
    assert guard.validate("It is on a daily schedule.") == [
        "schedule_configuration_required"
    ]
    assert guard.validate("Coverage was likely slow because of higher data volume.") == [
        "runtime_comparison_required"
    ]


def test_schedule_structural_and_runtime_checks_satisfy_their_claim_categories():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())
    guard.record_tool_result(
        "github_get_file_content",
        {"path": "resources/jobs/example.job.yml"},
        {"content": (
            "resources:\n  jobs:\n    email:\n      name: Ops Email\n"
            "      tasks:\n      - task_key: merge\n        notebook_task:\n"
            "          notebook_path: notebooks/Email Merge\n"
            "      schedule:\n        quartz_cron_expression: 0 0 8 * * ?"
        )},
    )
    guard.record_tool_result(
        "github_get_file_content",
        {"path": "notebooks/Email Merge.py"},
        {"content": "INSERT OVERWRITE acme_prod.email.target SELECT * FROM acme_prod.email.events"},
    )
    guard.record_tool_result(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT event_date, COUNT(*) AS source_rows "
                "FROM acme_prod.email.events "
                "GROUP BY event_date"
            )
        },
        {
            "status": "SUCCEEDED",
            "row_count": 2,
            "sample_rows": [["2026-08-11", 100], ["2026-08-12", 200]],
        },
    )

    assert guard.validate(
        "Ops Email's configured daily schedule fired. Ops Email is expensive because it performs a full scan. "
        "Ops Email's higher source volume likely contributed."
    ) == []


def test_notebook_logic_alone_does_not_verify_a_specific_runtime_cause():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())
    guard.record_tool_result(
        "github_get_file_content",
        {"path": "resources/jobs/email.job.yml"},
        {"content": "resources:\n  jobs:\n    email:\n      name: Ops Email\n"
         "      tasks:\n      - task_key: merge\n        notebook_task:\n"
         "          notebook_path: notebooks/Email Merge"},
    )
    guard.record_tool_result(
        "github_get_file_content",
        {"path": "notebooks/Email Merge.py"},
        {"content": "The job scans all email events and aggregates every campaign."},
    )

    assert guard.validate(
        "Higher email volume was likely the cause of today's duration."
    ) == ["runtime_comparison_required"]
    assert guard.validate(
        "Ops Email is expensive because it aggregates every campaign."
    ) == []


def test_single_current_count_is_not_comparative_runtime_evidence():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())
    guard.record_tool_result(
        "databricks_execute_sql",
        {"sql": "SELECT COUNT(*) FROM acme_prod.email.events"},
        {"status": "SUCCEEDED", "row_count": 1, "sample_rows": [[200]]},
    )

    assert guard.validate("Higher volume likely caused the duration.") == [
        "runtime_comparison_required"
    ]


def test_failed_or_empty_evidence_does_not_satisfy_guard():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())
    guard.record_tool_result(
        "databricks_get_recent_runs",
        {"job_id": 1},
        {"error": "forbidden"},
    )
    guard.record_tool_result(
        "github_get_file_content",
        {"path": "resources/jobs/example.job.yml"},
        {"content": ""},
    )

    assert guard.validate("The usual daily cadence looks normal.") == [
        "schedule_configuration_required",
        "recent_run_baseline_required",
    ]


def test_schedule_claim_requires_configuration_without_job_run_history():
    guard = JobRunEvidenceGuard()

    violations = guard.validate("This daily table is normally refreshed upstream.")

    assert violations == ["schedule_configuration_required"]
    assert "Remove that schedule/cadence claim" in guard.revision_instruction(violations)


def test_daily_date_grain_is_not_mistaken_for_job_cadence():
    guard = JobRunEvidenceGuard()

    assert guard.validate("Daily signup trend with flow and autofill mix.") == []
    assert guard.validate("The Signups job runs daily.") == [
        "schedule_configuration_required"
    ]


def test_cadence_claim_with_article_and_code_span_noun_is_blocked():
    """Regression for the signups_table_explanation eval failure: the run verb
    was separated from the cadence token by an article and the noun it modified
    was an inline code span, which ``_prose_without_code`` strips before matching."""
    guard = JobRunEvidenceGuard()
    # Reading the notebook proves logic, not schedule; schedule stays unverified.
    guard.record_tool_result(
        "github_get_file_content",
        {"path": "notebooks/Data Team/Gold Table Creation/Signups - Merge.sql"},
        {"content": "MERGE INTO acme_prod.reporting.signups AS target USING (...)"},
    )

    assert guard.validate(
        "The notebook runs a daily `MERGE` with a 30-day lookback window on the source side."
    ) == ["schedule_configuration_required"]
    assert guard.validate("The notebook performs a nightly load into the table.") == [
        "schedule_configuration_required"
    ]
    assert guard.validate("This is the daily merge that populates signups.") == [
        "schedule_configuration_required"
    ]
    # Date-grain language and negations are still not cadence claims.
    assert guard.validate("signup_date is the safe grain for daily analysis.") == []
    assert guard.validate("-- Signup volume trend (daily, last 90 days)") == []
    assert guard.validate("There is no daily merge schedule declared for this notebook.") == []


# The exact validation-SQL block from an earlier eval run, signups_table_explanation
# rep 3: the cadence claim is a SQL comment inside a fenced code block, and the
# sentence uses a bare "should". The scorer's forbidden pattern matched it; the
# guard did not (final_revisions = 0).
_DAILY_REFRESH_SQL_BLOCK = (
    "*2. Date coverage and recency:*\n"
    "```sql\n"
    "SELECT MIN(signup_date) AS earliest, MAX(signup_date) AS latest,\n"
    "       DATEDIFF(CURRENT_DATE, MAX(signup_date)) AS days_since_latest\n"
    "FROM acme_prod.reporting.signups;\n"
    "-- days_since_latest should be <= 2 for a healthy daily refresh\n"
    "```\n"
    "\n"
    "*3. DOB placeholder detection — rows broken for age analysis:*\n"
)


def test_cadence_claim_in_a_sql_comment_inside_a_code_fence_is_blocked():
    guard = JobRunEvidenceGuard()

    assert guard.validate(_DAILY_REFRESH_SQL_BLOCK) == ["schedule_configuration_required"]
    # The same sentence as plain prose, and its obvious siblings.
    assert guard.validate(
        "-- days_since_latest should be <= 2 for a healthy daily refresh"
    ) == ["schedule_configuration_required"]
    assert guard.validate("```sql\n# expect a one-day lag given the nightly load\nSELECT 1\n```") == [
        "schedule_configuration_required"
    ]
    assert guard.validate("The job runs each day; the merge runs every night.") == [
        "schedule_configuration_required"
    ]
    assert guard.validate("An every-day refresh keeps the table current.") == [
        "schedule_configuration_required"
    ]
    # Code itself is still not prose: a column alias or literal is not a claim.
    assert guard.validate(
        "```sql\nSELECT daily_refresh_flag, 'daily job' AS label FROM t\n```"
    ) == []
    # A fence with only code contributes nothing (and does not glue sentences).
    assert guard.validate("Row counts:\n```sql\nSELECT COUNT(*) FROM t\n```\nDone.") == []


def test_guard_and_fixture_forbidden_patterns_agree_on_the_daily_refresh_comment():
    """The eval scorer and the production guard must judge the same text alike:
    the fixture forbids the sentence, so the guard must request a revision."""
    fixture = load_fixture("signups_table_explanation")
    forbidden = [re.compile(pattern) for pattern in fixture["forbidden_final_patterns"]]
    guard = JobRunEvidenceGuard()

    for text in (
        _DAILY_REFRESH_SQL_BLOCK,
        "-- days_since_latest should be <= 2 for a healthy daily refresh",
        "The job runs each day.",
        "The MERGE is executed every day at 06:00.",
    ):
        assert any(pattern.search(text) for pattern in forbidden), text
        assert guard.validate(text) == ["schedule_configuration_required"], text

    # ...and both stay quiet on date-grain language.
    for text in (
        "signup_date is the safe grain for daily analysis.",
        "-- Signup volume trend (daily, last 90 days)",
    ):
        assert not any(pattern.search(text) for pattern in forbidden), text
        assert guard.validate(text) == [], text


def test_schedule_word_alone_is_not_a_cadence_claim():
    guard = JobRunEvidenceGuard()

    assert guard.validate("The schedule block lives in resources/jobs/signups.job.yml.") == []
    assert guard.validate("I did not check the schedule configuration for this job.") == []
    assert guard.validate("Cadence was not part of this question.") == []


def test_instructional_and_hypothetical_schedule_text_is_not_blocked():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())

    assert guard.validate(
        "To add a schedule to the job, set quartz_cron_expression under schedule: in the "
        "job YAML and open a PR."
    ) == []
    assert guard.validate("How to add a daily schedule: edit the resource YAML.") == []
    assert guard.validate(
        "If you want it scheduled daily, you can add a schedule block with a cron expression."
    ) == []
    assert guard.validate("You could schedule it hourly if the upstream table allows.") == []
    assert guard.validate("The job should probably be scheduled nightly, but I have not checked.") == []


def test_negated_schedule_statements_are_not_blocked():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())

    assert guard.validate("There is no schedule configured for this job; it ran once at 8:05 AM.") == []
    assert guard.validate("The job is not scheduled daily; the run was triggered manually.") == []
    assert guard.validate("It isn't scheduled to run hourly.") == []
    assert guard.validate("This job is unscheduled and only runs on demand.") == []
    assert guard.validate("It has no cron schedule.") == []


def test_positive_cadence_assertions_are_still_blocked_sentence_by_sentence():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())

    assert guard.validate("The job is scheduled to run daily at 8:00 AM.") == [
        "schedule_configuration_required"
    ]
    assert guard.validate("Its schedule is a nightly cron.") == [
        "schedule_configuration_required"
    ]
    assert guard.validate("It runs on a weekly schedule.") == [
        "schedule_configuration_required"
    ]
    # An instructional sentence does not launder a separate positive claim.
    assert guard.validate(
        "To add a schedule, edit the YAML. Today the job is on an hourly schedule."
    ) == ["schedule_configuration_required"]


def test_revision_instruction_covers_every_violation_category():
    guard = JobRunEvidenceGuard()

    baseline = guard.revision_instruction(["recent_run_baseline_required"])
    assert baseline is not None
    assert "baseline judgement" in baseline
    assert "internal correction" in baseline

    structural = guard.revision_instruction(["structural_logic_required"])
    assert structural is not None and "structural explanation" in structural

    runtime = guard.revision_instruction(["runtime_comparison_required"])
    assert runtime is not None and "causal claim" in runtime

    combined = guard.revision_instruction(
        ["schedule_configuration_required", "recent_run_baseline_required"]
    )
    assert "Remove that schedule/cadence claim" in combined
    assert "baseline judgement" in combined

    assert guard.revision_instruction([]) is None
    assert guard.revision_instruction(["unknown_category"]) is None


def test_list_runs_fallback_reports_observations_without_inference():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result("databricks_list_job_runs", {}, _list_runs_result())

    fallback = guard.safe_fallback()

    assert "*Observed job runs — 2026-08-12*" in fallback
    assert "3 executions across 2 jobs" in fallback
    assert "`Coverage` — 2 starts" in fallback
    assert "`Ops Email` — 1h 6m" in fallback
    assert "Configured cadence, normal duration, workload, and causes were not inferred." in fallback


def test_recent_runs_fallback_reports_direct_comparison_and_no_cause():
    guard = JobRunEvidenceGuard()
    guard.record_tool_result(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT job_id, job_name FROM "
                "acme_prod.acme_history.databricks_workspace_jobs"
            )
        },
        {
            "status": "SUCCEEDED",
            "row_count": 1,
            "columns": ["job_id", "job_name"],
            "sample_rows": [[1, "Ops Email"]],
        },
    )
    guard.record_tool_result(
        "databricks_get_recent_runs",
        {"job_id": 1},
        _recent_runs_result(),
    )

    fallback = guard.safe_fallback()

    assert "Job: `Ops Email`" in fallback
    assert "Latest observed duration: 1h 6m" in fallback
    assert "Latest vs. median: 48.7% above" in fallback
    assert "does not establish the cause" in fallback


class _FakeBlock:
    def __init__(self, block_type, **values):
        self.type = block_type
        for key, value in values.items():
            setattr(self, key, value)

    def model_dump(self):
        result = {"type": self.type}
        for key in ("id", "name", "input", "text"):
            if hasattr(self, key):
                result[key] = getattr(self, key)
        return result


class _FakeResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeAnthropicClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs["messages"]))
        return self.responses.pop(0)


def test_worker_replaces_unsupported_claim_with_observed_summary_after_failed_revision(
    monkeypatch,
    caplog,
):
    """Baseline/structural violations now get one private revision pass; the
    deterministic fallback replaces the answer only when the revision still
    violates (previously these categories fell back immediately)."""
    client = _FakeAnthropicClient(
        [
            _FakeResponse(
                "tool_use",
                [
                    _FakeBlock(
                        "tool_use",
                        id="tool-1",
                        name="databricks_list_job_runs",
                        input={},
                    )
                ],
            ),
            _FakeResponse(
                "end_turn",
                [
                    _FakeBlock(
                        "text",
                        text="Ops Email looks normal because it aggregates many campaigns.",
                    )
                ],
            ),
            _FakeResponse(
                "end_turn",
                [
                    _FakeBlock(
                        "text",
                        text="Ops Email is slow because it aggregates many campaigns.",
                    )
                ],
            ),
        ]
    )
    posted = []
    saved = []
    audits = []

    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(
        worker,
        "_dispatch_tool",
        lambda name, inp: _list_runs_result(),
    )
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: posted.append(text)
        or "message-ts",
    )
    monkeypatch.setattr(
        worker,
        "post_message_chunks",
        lambda channel, text, thread_ts=None: (
            posted.append(text)
            or MessageDelivery((text,), ("answer-ts",), 1)
        ),
    )
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "save_session",
        lambda thread_ts, messages: saved.append(deepcopy(messages)),
    )
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    messages = [{"role": "user", "content": "Compare today's email jobs."}]
    caplog.set_level("WARNING")
    worker.run_agent(messages, "C123", "thread-1")

    assert len(client.requests) == 3
    assert "internal correction" in str(client.requests[2][-1]).lower()
    assert "baseline judgement" in str(client.requests[2][-1])
    assert "structural explanation" in str(client.requests[2][-1])
    assert "*Observed job runs — 2026-08-12*" in posted[-2]
    assert "3 executions across 2 jobs" in posted[-2]
    assert posted[-1] == "How useful was this answer?"
    assert all("aggregates many campaigns" not in str(message) for message in messages)
    assert all("internal correction" not in str(message) for message in messages)
    assert saved[-1] == messages
    assert any(event == "final_response_guard_revision" for event, _ in audits)
    assert any(event == "final_response_guard_fallback" for event, _ in audits)
    assert "thread_ts=thread-1" in caplog.text
    assert "recent_run_baseline_required" in caplog.text
    assert "structural_logic_required" in caplog.text


def test_worker_accepts_revised_baseline_claim_without_fallback(monkeypatch):
    client = _FakeAnthropicClient(
        [
            _FakeResponse(
                "tool_use",
                [
                    _FakeBlock(
                        "tool_use",
                        id="tool-1",
                        name="databricks_list_job_runs",
                        input={},
                    )
                ],
            ),
            _FakeResponse(
                "end_turn",
                [_FakeBlock("text", text="Ops Email ran for 66m, which looks slow.")],
            ),
            _FakeResponse(
                "end_turn",
                [_FakeBlock("text", text="Ops Email ran for 66m; Coverage ran twice.")],
            ),
        ]
    )
    posted = []
    saved = []
    audits = []

    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(
        worker,
        "_dispatch_tool",
        lambda name, inp: _list_runs_result(),
    )
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: posted.append(text)
        or "message-ts",
    )
    monkeypatch.setattr(
        worker,
        "post_message_chunks",
        lambda channel, text, thread_ts=None: (
            posted.append(text)
            or MessageDelivery((text,), ("answer-ts",), 1)
        ),
    )
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "save_session",
        lambda thread_ts, messages: saved.append(deepcopy(messages)),
    )
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    messages = [{"role": "user", "content": "Compare today's email jobs."}]
    worker.run_agent(messages, "C123", "thread-1")

    assert len(client.requests) == 3
    assert posted[-2] == "Ops Email ran for 66m; Coverage ran twice."
    assert posted[-1] == "How useful was this answer?"
    assert all("looks slow" not in str(message) for message in messages)
    assert all("internal correction" not in str(message) for message in messages)
    assert saved[-1] == messages
    assert any(event == "final_response_guard_revision" for event, _ in audits)
    assert not any(event == "final_response_guard_fallback" for event, _ in audits)


def test_worker_privately_revises_unverified_schedule_claim(monkeypatch):
    client = _FakeAnthropicClient(
        [
            _FakeResponse(
                "end_turn",
                [_FakeBlock("text", text="The Signups job runs daily. Grain is one row.")],
            ),
            _FakeResponse(
                "end_turn",
                [_FakeBlock("text", text="Grain is one row per browser identity.")],
            ),
        ]
    )
    posted = []
    saved = []
    audits = []

    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: posted.append(text)
        or "message-ts",
    )
    monkeypatch.setattr(
        worker,
        "post_message_chunks",
        lambda channel, text, thread_ts=None: (
            posted.append(text)
            or MessageDelivery((text,), ("answer-ts",), 1)
        ),
    )
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "save_session",
        lambda thread_ts, messages: saved.append(deepcopy(messages)),
    )
    monkeypatch.setattr(
        worker,
        "audit",
        lambda event, **fields: audits.append((event, fields)),
    )

    messages = [{"role": "user", "content": "Explain the Signups table."}]
    worker.run_agent(messages, "C123", "thread-1")

    assert len(client.requests) == 2
    assert "internal correction" in str(client.requests[1][-1]).lower()
    assert posted[-2] == "Grain is one row per browser identity."
    assert posted[-1] == "How useful was this answer?"
    assert all("runs daily" not in str(message) for message in messages)
    assert all("internal correction" not in str(message) for message in messages)
    assert saved[-1] == messages
    assert any(event == "final_response_guard_revision" for event, _ in audits)
