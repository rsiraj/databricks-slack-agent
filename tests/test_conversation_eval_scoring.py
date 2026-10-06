"""Deterministic tests for the model-in-the-loop evaluation gate."""

import ast
import copy
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.conversation_runner import (
    ALL_TOOLS,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    DEFAULT_THINKING,
    MAX_API_CALLS,
    MAX_TOKENS_PER_CALL,
    MAX_TOTAL_OUTPUT_TOKENS,
    resolve_max_api_calls,
    run_conversation,
)
from evals.pricing import estimate_cost_usd
from evals.scoring import (
    AVOIDABLE_FAILURE_CATEGORY_SEVERITY,
    DEFAULT_CODE_SEVERITY,
    _apply_severity,
    default_severity,
    final_number_tokens,
    score_trace,
)
from evals.synthetic_tools import SyntheticTools, load_fixture
from tools import calc_tool
from utils.llm.anthropic_provider import recommended_max_tokens
from utils.system_prompt import load_base_prompt, load_system_prompt


FIXTURE = load_fixture("signups_table_explanation")
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _production_numeric_guard_defaults(monkeypatch):
    monkeypatch.delenv("NUMERIC_GUARD_MODE", raising=False)
    monkeypatch.delenv("NUMERIC_GUARD_MAX_REVISIONS", raising=False)


def _module_assignments(path: Path) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text())
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    return assignments


def _default_literal(node: ast.expr, assignments: dict[str, ast.expr]):
    """Resolve the *default* value of a module constant.

    Handles plain literals, ``NAME`` references to other module constants,
    ``A * B`` products of such constants, and env-driven wrappers such as
    ``os.environ.get("X", default)`` or ``_helper("X", default)`` by taking the
    last positional argument.
    """
    if isinstance(node, ast.Name):
        if node.id not in assignments:
            raise AssertionError(f"Cannot resolve name {node.id}")
        return _default_literal(assignments[node.id], assignments)
    if isinstance(node, ast.Call):
        if not node.args:
            raise AssertionError("Call without a positional default argument")
        return _default_literal(node.args[-1], assignments)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _default_literal(node.left, assignments) * _default_literal(
            node.right, assignments
        )
    return ast.literal_eval(node)


def _literal_assignment(path: Path, name: str):
    assignments = _module_assignments(path)
    if name not in assignments:
        raise AssertionError(f"No assignment for {name} in {path}")
    return _default_literal(assignments[name], assignments)


def _event(name, tool_input, result=None, *, iteration=0, is_error=False, source="fixture"):
    return {
        "iteration": iteration,
        "name": name,
        "input": tool_input,
        "result": result or {"status": "SUCCEEDED", "row_count": 1},
        "is_error": is_error,
        "source": source,
    }


def _passing_trace():
    final = (
        "The grain is one row per `_anonymousId`, which is a browser/device "
        "identifier rather than necessarily one person. Existing age fields are "
        "captured on insert because the MERGE update clauses do not update them."
    )
    return {
        "scenario": FIXTURE["id"],
        "stop_reason": "end_turn",
        "final_text": final,
        "api_calls": 5,
        "hit_api_call_budget": False,
        "output_tokens": 4000,
        "budget": {"max_api_calls": MAX_API_CALLS, "max_output_tokens": MAX_TOTAL_OUTPUT_TOKENS},
        "tool_events": [
            _event(
                "github_get_file_content",
                {"path": FIXTURE["github_file"]["path"]},
            ),
            _event(
                "databricks_execute_sql",
                {"sql": "SHOW CREATE TABLE acme_prod.reporting.signups"},
            ),
            _event(
                "databricks_describe_object",
                {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
                iteration=1,
            ),
            _event(
                "databricks_execute_sql",
                {"sql": "SELECT COUNT(*) FROM acme_prod.reporting.signups"},
                iteration=2,
            ),
            _event(
                "databricks_describe_object",
                {"catalog": "system", "schema": "access", "name": "table_lineage"},
                iteration=1,
            ),
            _event(
                "databricks_execute_sql",
                {
                    "sql": (
                        "SELECT source_table_full_name FROM system.access.table_lineage "
                        "WHERE target_table_full_name = 'acme_prod.reporting.signups'"
                    )
                },
                iteration=2,
            ),
        ],
    }


def test_clean_conversation_trace_passes():
    result = score_trace(_passing_trace(), FIXTURE)

    assert result["passed"] is True
    assert result["avoidable_tool_failures"] == 0
    assert result["violations"] == []


def test_evaluator_model_and_per_call_budget_match_production():
    worker_path = REPO_ROOT / "worker.py"
    runner_path = REPO_ROOT / "evals" / "conversation_runner.py"
    # worker.MODEL is resolved through utils.llm (LLM_PROVIDER / LLM_MODEL /
    # ANTHROPIC_MODEL); the production default literal lives there.
    llm_path = REPO_ROOT / "utils" / "llm" / "__init__.py"

    assert _literal_assignment(runner_path, "DEFAULT_MODEL") == _literal_assignment(
        llm_path, "DEFAULT_ANTHROPIC_MODEL"
    )
    # worker.MAX_TOKENS is resolved at import: with ANTHROPIC_MAX_TOKENS unset the
    # default model gets recommended_max_tokens(model, effort) (24,000 for Sonnet 5
    # at high, the value Terraform sets in production); the trailing positional
    # literal is only the hard floor for invalid values and unknown models.
    production_max_tokens = recommended_max_tokens(
        _literal_assignment(llm_path, "DEFAULT_ANTHROPIC_MODEL"),
        _literal_assignment(worker_path, "EFFORT"),
    )
    assert production_max_tokens == 24_000
    assert _literal_assignment(runner_path, "MAX_TOKENS_PER_CALL") == production_max_tokens
    assert MAX_TOKENS_PER_CALL == production_max_tokens
    assert _literal_assignment(worker_path, "MAX_TOKENS") == _literal_assignment(
        worker_path, "DEFAULT_MAX_TOKENS"
    )
    assert _literal_assignment(worker_path, "DEFAULT_MAX_TOKENS") == 16_000
    assert _literal_assignment(runner_path, "MAX_FINAL_REVISIONS") == _literal_assignment(
        worker_path, "MAX_FINAL_REVISIONS"
    )
    assert _literal_assignment(runner_path, "DEFAULT_EFFORT") == _literal_assignment(
        worker_path, "EFFORT"
    )


def test_evaluator_conversation_budgets_mirror_production():
    """The harness must never bound a conversation tighter than the worker does.

    ``MAX_API_CALLS`` is ``worker.MAX_ITERATIONS`` (the point where the bot posts
    "Hit max iterations"); production has no separate output-token cap, so the
    runner's default is the effective production bound ``MAX_TOKENS *
    MAX_ITERATIONS``, where ``MAX_TOKENS`` is what the worker resolves for the
    default model/effort (``recommended_max_tokens``). Compared by AST because
    importing ``worker`` needs AWS and Slack wiring the eval job does not have.
    """
    worker_path = REPO_ROOT / "worker.py"
    runner_path = REPO_ROOT / "evals" / "conversation_runner.py"
    llm_path = REPO_ROOT / "utils" / "llm" / "__init__.py"

    production_iterations = _literal_assignment(worker_path, "MAX_ITERATIONS")
    production_max_tokens = recommended_max_tokens(
        _literal_assignment(llm_path, "DEFAULT_ANTHROPIC_MODEL"),
        _literal_assignment(worker_path, "EFFORT"),
    )
    assert production_iterations == 25  # the value the CI failure was measured against

    assert _literal_assignment(runner_path, "MAX_API_CALLS") == production_iterations
    assert MAX_API_CALLS == production_iterations
    assert (
        _literal_assignment(runner_path, "MAX_TOTAL_OUTPUT_TOKENS")
        == production_max_tokens * production_iterations
    )
    assert MAX_TOTAL_OUTPUT_TOKENS == MAX_TOKENS_PER_CALL * MAX_API_CALLS

    assert "Hit max iterations" in worker_path.read_text()


def test_max_api_calls_precedence_is_override_then_fixture_then_production():
    assert resolve_max_api_calls({}) == (MAX_API_CALLS, "default")
    assert resolve_max_api_calls({"max_api_calls": None}) == (MAX_API_CALLS, "default")
    assert resolve_max_api_calls({"max_api_calls": 12}) == (12, "fixture")
    assert resolve_max_api_calls({"max_api_calls": 12}, 7) == (7, "override")
    for bad in (0, -1, True, "12", 2.5):
        with pytest.raises(ValueError):
            resolve_max_api_calls({"max_api_calls": bad})
    with pytest.raises(ValueError):
        resolve_max_api_calls({}, 0)


def test_forbidden_identifier_scope_is_fixture_driven():
    scope = FIXTURE["forbidden_identifier_scope"]
    assert scope["table"] == "acme_prod.reporting.signups"
    assert scope["identifiers"] == ["internal_sum"]

    trace = _passing_trace()
    trace["tool_events"][3]["input"]["sql"] = (
        "SELECT COUNT_IF(internal_sum > 0) FROM acme_prod.reporting.signups"
    )

    rescoped = copy.deepcopy(FIXTURE)
    rescoped["forbidden_identifier_scope"] = {
        "table": "other_catalog.other_schema.other_table",
        "identifiers": ["internal_sum"],
    }
    assert not any(
        v["code"] == "forbidden_sql_identifier"
        for v in score_trace(trace, rescoped)["violations"]
    )

    legacy = copy.deepcopy(FIXTURE)
    del legacy["forbidden_identifier_scope"]
    legacy["forbidden_sql_identifiers"] = ["internal_sum"]
    assert any(
        v["code"] == "forbidden_sql_identifier"
        for v in score_trace(trace, legacy)["violations"]
    )


def test_behavioral_violations_default_to_hard_and_report_a_graded_score():
    clean = score_trace(_passing_trace(), FIXTURE)
    assert clean["score"] == 1.0
    assert clean["hard_violations"] == 0

    trace = _passing_trace()
    trace["final_text"] += " The job runs daily."
    result = score_trace(trace, FIXTURE)

    assert result["passed"] is False
    assert result["hard_violations"] == 1
    assert result["soft_violations"] == 0
    assert result["score"] == 0.75
    assert all(v["severity"] == "hard" for v in result["violations"])


BEHAVIORAL_CODES = (
    "avoidable_tool_failure",
    "unsupported_answer_claim",
    "required_number_missing",
    "forbidden_number_present",
    "required_tool_missing",
    "ungrounded_approximation",
    "metadata_ordering",
    "forbidden_sql_identifier",
    "missing_analytical_evidence",
    "missing_tool_evidence",
    "incomplete_conversation",
    "empty_answer",
    "api_call_budget",
    "output_token_budget",
)
COVERAGE_CODES = ("missing_answer_fact", "ungrounded_number")


def test_severity_table_blocks_on_behavior_and_reports_content_coverage():
    """The gate's policy in one place: every behavioral code is hard without a
    fixture override; only the two content-coverage codes default to soft."""
    assert DEFAULT_CODE_SEVERITY == {"ungrounded_number": "soft", "missing_answer_fact": "soft"}

    violations = _apply_severity(
        [{"code": code, "message": code} for code in BEHAVIORAL_CODES + COVERAGE_CODES],
        {"id": "no-overrides"},
    )
    by_code = {violation["code"]: violation["severity"] for violation in violations}

    assert {code: by_code[code] for code in BEHAVIORAL_CODES} == {
        code: "hard" for code in BEHAVIORAL_CODES
    }
    assert {code: by_code[code] for code in COVERAGE_CODES} == {
        code: "soft" for code in COVERAGE_CODES
    }


def test_missing_answer_fact_is_soft_by_default_and_does_not_fail_the_gate():
    """an earlier CI run: a complete, grounded explanation that never wrote the
    literal ``_anonymousId`` or the browser/device identity sentence. Both are
    coverage misses, so they are reported and lower the score without blocking."""
    trace = _passing_trace()
    trace["final_text"] = (
        "The grain is one row per signup. Existing age fields are captured on "
        "insert because the MERGE update clauses do not update them."
    )

    result = score_trace(trace, FIXTURE)

    assert [v["code"] for v in result["violations"]] == [
        "missing_answer_fact",
        "missing_answer_fact",
    ]
    assert all(v["severity"] == "soft" for v in result["violations"])
    assert result["passed"] is True
    assert result["hard_violations"] == 0
    assert result["soft_violations"] == 2
    assert result["score"] == 0.9  # soft weight still counts against the graded score
    assert "'(?i)_anonymousid'" in result["violations"][0]["message"]


def test_fixture_can_promote_missing_answer_fact_to_hard():
    fixture = copy.deepcopy(FIXTURE)
    fixture["violation_severity"] = {"missing_answer_fact": "hard"}
    trace = _passing_trace()
    trace["final_text"] = "The grain is one row per signup."

    result = score_trace(trace, fixture)

    assert result["passed"] is False
    assert result["hard_violations"] == 2
    assert result["soft_violations"] == 0
    assert result["score"] == 0.5
    assert all(v["code"] == "missing_answer_fact" for v in result["violations"])
    assert all(v["severity"] == "hard" for v in result["violations"])


def test_soft_coverage_miss_does_not_mask_a_hard_behavioral_failure():
    trace = _passing_trace()
    trace["final_text"] = "One row per signup. The job runs daily."

    result = score_trace(trace, FIXTURE)

    severities = {v["code"]: v["severity"] for v in result["violations"]}
    assert severities["unsupported_answer_claim"] == "hard"
    assert severities["missing_answer_fact"] == "soft"
    assert result["passed"] is False
    assert result["hard_violations"] == 1
    assert result["soft_violations"] == 2
    assert result["score"] == 0.65


def test_fixture_can_downgrade_a_violation_to_soft():
    fixture = copy.deepcopy(FIXTURE)
    fixture["violation_severity"] = {"unsupported_answer_claim": "soft"}
    trace = _passing_trace()
    trace["final_text"] += " The job runs daily."

    result = score_trace(trace, fixture)

    assert result["passed"] is True
    assert result["soft_violations"] == 1
    assert result["score"] == 0.95
    assert result["violations"][0]["severity"] == "soft"


def test_any_tool_error_is_an_avoidable_failure():
    trace = _passing_trace()
    trace["tool_events"][3].update(
        {
            "is_error": True,
            "source": "guard",
            "result": {
                "error": "Describe metadata first",
                "validation_failure": "schema_inspection_required",
            },
        }
    )

    result = score_trace(trace, FIXTURE)

    assert result["passed"] is False
    assert result["avoidable_tool_failures"] == 1
    assert any(v["code"] == "avoidable_tool_failure" for v in result["violations"])


def test_avoidable_failure_message_names_missing_columns_and_sql():
    """an earlier CI run: the summary only said ``column_not_in_live_schema``;
    the artifact had to be downloaded to learn the SQL used ``original_timestamp``.
    The scorer message must carry the guard's ``missing_columns`` and the SQL so
    the CI log alone explains the failure. The violation stays hard."""
    trace = _passing_trace()
    sql = (
        "\nSELECT\n  DATE_FORMAT(original_timestamp, 'yyyy-MM') AS month,\n"
        "  COUNT(*) AS signups\nFROM acme_prod.reporting.signups\n"
        "WHERE original_timestamp >= '2026-07-01'\nGROUP BY 1\n"
    )
    trace["tool_events"][3].update(
        {
            "input": {"sql": sql},
            "is_error": True,
            "source": "guard",
            "result": {
                "error": "The analytical SQL references columns that are absent.",
                "validation_failure": "column_not_in_live_schema",
                "object": "acme_prod.reporting.signups",
                "missing_columns": ["original_timestamp"],
            },
        }
    )

    result = score_trace(trace, FIXTURE)

    violation = next(
        v for v in result["violations"] if v["code"] == "avoidable_tool_failure"
    )
    assert violation["severity"] == "hard"
    assert result["passed"] is False
    assert "column_not_in_live_schema" in violation["message"]
    assert "missing_columns: original_timestamp." in violation["message"]
    # SQL is whitespace-collapsed onto one line so the CI log stays readable.
    assert (
        "sql: SELECT DATE_FORMAT(original_timestamp, 'yyyy-MM') AS month, "
        "COUNT(*) AS signups FROM acme_prod.reporting.signups"
    ) in violation["message"]
    assert "\n" not in violation["message"]


def test_avoidable_failure_message_truncates_long_sql_and_skips_absent_fields():
    trace = _passing_trace()
    long_sql = "SELECT " + ", ".join(f"col_{i}" for i in range(120)) + " FROM t"
    trace["tool_events"][3].update(
        {
            "input": {"sql": long_sql},
            "is_error": True,
            "source": "guard",
            "result": {"error": "boom", "validation_failure": "schema_inspection_required"},
        }
    )
    trace["tool_events"][0].update(
        {"is_error": True, "result": {"error": "not found", "fixture_failure": "missing_file"}}
    )

    result = score_trace(trace, FIXTURE)

    messages = [
        v["message"] for v in result["violations"] if v["code"] == "avoidable_tool_failure"
    ]
    assert len(messages) == 2
    non_sql = next(m for m in messages if m.startswith("github_get_file_content"))
    assert non_sql == "github_get_file_content returned an avoidable error (missing_file)."
    sql_message = next(m for m in messages if m.startswith("databricks_execute_sql"))
    assert "missing_columns" not in sql_message
    assert sql_message.endswith("...")
    snippet = sql_message.split(" sql: ", 1)[1]
    assert 200 < len(snippet) <= 240
    assert len(snippet) < len(long_sql)


def _duplicate_sql_event(iteration=3):
    return _event(
        "databricks_execute_sql",
        {"sql": "SELECT COUNT(*) AS total_rows FROM acme_prod.reporting.signups"},
        {
            "error": "This exact SQL was already executed during the current run as SQL call 1 earlier this turn.",
            "validation_failure": "duplicate_sql",
            "earlier_sql_call": 1,
        },
        iteration=iteration,
        is_error=True,
        source="guard",
    )


def test_duplicate_sql_tool_failure_is_soft_by_default():
    """An earlier eval run (``signups_table_explanation`` rep 2): the model re-ran
    a byte-identical query it already had the result of. That is one wasted round
    trip, not a wrong or unsafe action, so it is reported and lowers the score
    without failing the conversation."""
    trace = _passing_trace()
    trace["tool_events"].append(_duplicate_sql_event())

    result = score_trace(trace, FIXTURE)

    (violation,) = [v for v in result["violations"] if v["code"] == "avoidable_tool_failure"]
    assert violation["category"] == "duplicate_sql"
    assert violation["severity"] == "soft"
    assert "(duplicate_sql)" in violation["message"]
    assert result["passed"] is True
    assert result["hard_violations"] == 0 and result["soft_violations"] == 1
    assert result["avoidable_tool_failures"] == 1  # still visible as a tool-failure metric
    assert result["score"] == 0.95


def test_avoidable_failure_category_severity_table_is_explicit():
    assert AVOIDABLE_FAILURE_CATEGORY_SEVERITY == {
        "duplicate_sql": "soft",
        "column_not_in_live_schema": "hard",
        "object_metadata_required": "hard",
        "system_object_metadata_required": "hard",
        "invalid_struct_access_syntax": "hard",
        "wrong_github_source": "soft",
        "unparseable_sql": "hard",
        "write_statement": "hard",
        "denied_function": "hard",
    }
    assert default_severity("avoidable_tool_failure", "duplicate_sql") == "soft"
    assert default_severity("avoidable_tool_failure", "wrong_github_source") == "soft"
    for category, severity in AVOIDABLE_FAILURE_CATEGORY_SEVERITY.items():
        assert default_severity("avoidable_tool_failure", category) == severity
    # Unknown categories, and a tool error with no category at all, stay hard.
    assert default_severity("avoidable_tool_failure", "some_future_guard_rule") == "hard"
    assert default_severity("avoidable_tool_failure", None) == "hard"
    assert default_severity("avoidable_tool_failure", "tool_error") == "hard"
    # The category table only applies to tool failures.
    assert default_severity("unsupported_answer_claim", "duplicate_sql") == "hard"
    assert default_severity("missing_answer_fact") == "soft"


def test_hard_tool_failure_categories_still_fail_the_conversation():
    for category in (
        "column_not_in_live_schema",
        "object_metadata_required",
        "invalid_struct_access_syntax",
    ):
        trace = _passing_trace()
        trace["tool_events"][3].update(
            {"is_error": True, "source": "guard", "result": {"error": "x", "validation_failure": category}}
        )
        result = score_trace(trace, FIXTURE)
        (violation,) = [v for v in result["violations"] if v["code"] == "avoidable_tool_failure"]
        assert violation["category"] == category and violation["severity"] == "hard", category
        assert result["passed"] is False, category


def test_fixture_can_override_tool_failure_severity_per_category_or_per_code():
    trace = _passing_trace()
    trace["tool_events"].append(_duplicate_sql_event())

    def severity_with(overrides):
        fixture = copy.deepcopy(FIXTURE)
        fixture["violation_severity"] = overrides
        result = score_trace(trace, fixture)
        (violation,) = [v for v in result["violations"] if v["code"] == "avoidable_tool_failure"]
        return violation["severity"], result["passed"]

    # Category-qualified override puts duplicate_sql back to hard.
    assert severity_with({"avoidable_tool_failure:duplicate_sql": "hard"}) == ("hard", False)
    # A bare-code override covers every category, including duplicate_sql.
    assert severity_with({"avoidable_tool_failure": "hard"}) == ("hard", False)
    # The category-qualified key wins over the bare code.
    assert severity_with(
        {"avoidable_tool_failure": "hard", "avoidable_tool_failure:duplicate_sql": "soft"}
    ) == ("soft", True)
    # An unknown severity value is ignored and the table default applies.
    assert severity_with({"avoidable_tool_failure:duplicate_sql": "blocking"}) == ("soft", True)

    # And a hard category can be softened by a fixture that is not about it.
    column_trace = _passing_trace()
    column_trace["tool_events"][3].update(
        {
            "is_error": True,
            "source": "guard",
            "result": {"error": "x", "validation_failure": "column_not_in_live_schema", "missing_columns": ["received_at"]},
        }
    )
    fixture = copy.deepcopy(FIXTURE)
    fixture["violation_severity"] = {"avoidable_tool_failure:column_not_in_live_schema": "soft"}
    result = score_trace(column_trace, fixture)
    (violation,) = [v for v in result["violations"] if v["code"] == "avoidable_tool_failure"]
    assert violation["severity"] == "soft" and result["passed"] is True


MH_NOTEBOOK_PATH = FIXTURE["github_file"]["path"].replace(
    "Signups - Merge.sql", "Signups MA - Merge.sql"
)


def _wrong_github_source_event(iteration=3):
    """The read earlier comparison runs each made once: the stale
    MA notebook, fetched *after* the current ``Signups - Merge.sql``."""
    return _event(
        "github_get_file_content",
        {"path": MH_NOTEBOOK_PATH},
        {
            "error": "The requested file is outside this scenario's current-source fixture.",
            "fixture_failure": "wrong_github_source",
            "requested_path": MH_NOTEBOOK_PATH,
        },
        iteration=iteration,
        is_error=True,
    )


def test_wrong_github_source_read_is_soft_by_default():
    """Reading the stale notebook is a wasted call, not a wrong answer: in
    production the read simply succeeds. It is reported and lowers the score
    without failing the conversation."""
    trace = _passing_trace()
    trace["tool_events"].append(_wrong_github_source_event())

    result = score_trace(trace, FIXTURE)

    (violation,) = [v for v in result["violations"] if v["code"] == "avoidable_tool_failure"]
    assert violation["category"] == "wrong_github_source"
    assert violation["severity"] == "soft"
    assert "(wrong_github_source)" in violation["message"]
    assert result["passed"] is True
    assert result["hard_violations"] == 0 and result["soft_violations"] == 1
    assert result["avoidable_tool_failures"] == 1
    assert result["score"] == 0.95


def test_fixture_can_promote_wrong_github_source_to_hard():
    trace = _passing_trace()
    trace["tool_events"].append(_wrong_github_source_event())
    trace["tool_events"].append(_duplicate_sql_event(iteration=4))
    fixture = copy.deepcopy(FIXTURE)
    fixture["violation_severity"] = {"avoidable_tool_failure:wrong_github_source": "hard"}

    result = score_trace(trace, fixture)

    by_category = {
        v["category"]: v["severity"]
        for v in result["violations"]
        if v["code"] == "avoidable_tool_failure"
    }
    # The category-qualified promotion reaches only that category.
    assert by_category == {"wrong_github_source": "hard", "duplicate_sql": "soft"}
    assert result["passed"] is False
    assert result["hard_violations"] == 1 and result["soft_violations"] == 1


@pytest.mark.parametrize(
    "claim",
    [
        "The table is built by `Signups MA - Merge.sql`.",
        "Source notebook: `notebooks/Data Team/Gold Table Creation/Signups MA - Merge.sql`.",
        "Build logic: `Signups MA - Merge.sql`.",
        "The current source is the Signups MA - Merge.sql notebook.",
        "The MA notebook is the current source of truth for this table.",
        "`Signups MA - Merge.sql` populates acme_prod.reporting.signups.",
        "`Signups MA - Merge.sql` (build logic)",
        "The production merge lives in Signups MA - Merge.sql.",
        "acme_prod.reporting.signups is maintained by the Mobile App merge notebook.",
    ],
)
def test_presenting_the_stale_mh_notebook_as_the_source_is_a_hard_claim(claim):
    """Reading the stale notebook is soft; *using* it is not. An answer that
    presents ``Signups MA - Merge.sql`` as the table's current or build
    source fails hard through ``forbidden_final_patterns``, even when the read
    itself was only a soft tool failure."""
    trace = _passing_trace()
    trace["tool_events"].append(_wrong_github_source_event())
    trace["final_text"] += " " + claim

    result = score_trace(trace, FIXTURE)

    claims = [v for v in result["violations"] if v["code"] == "unsupported_answer_claim"]
    assert claims, claim
    assert all(v["severity"] == "hard" for v in claims)
    assert result["passed"] is False


@pytest.mark.parametrize(
    "statement",
    [
        # Sonnet 5, earlier comparison runs rep 1 (the soft MA read happened in this repeat).
        "The `Signups MA - Merge.sql` file (Mobile App variant) exists in the repo but was "
        "not inspected — per repo convention, MA-suffixed notebooks are historical/Mobile-App-"
        "specific and not relevant to the current `signups` table.",
        # Sonnet 5, an earlier run, rep 3.
        "github_find_files and github_search_code found only one other file (Signups MA - "
        "Merge.sql, the Mobile App variant) — no additional notebook builds these 26 columns.",
        "- GitHub: `notebooks/Data Team/Gold Table Creation/Signups - Merge.sql` "
        "(build logic), `Signups MA - Merge.sql` (confirmed as the only other related file, "
        "Mobile App variant), one commit found (2026-02-03)",
        # Sonnet 4.6, earlier runs, reps 2 and 1.
        "- GitHub: `notebooks/Data Team/Gold Table Creation/Signups - Merge.sql` "
        "(build SQL, commit history); `notebooks/Data Team/Gold Table Creation/"
        "Signups MA - Merge.sql` (noted as Mobile App, not used)",
        "- GitHub: `notebooks/Data Team/Gold Table Creation/Signups - Merge.sql` "
        "(build logic); `notebooks/Data Team/Gold Table Creation/Signups MA - "
        "Merge.sql` (Mobile App variant — not used for current product)",
        # The correct file named first, the stale one dismissed after it.
        "Current source: `Signups - Merge.sql` (build logic); `Signups MA - Merge.sql` "
        "is a historical Mobile App variant.",
        # The referral filter, correctly explained, mentions MA / Mobile App too.
        "signup_flow — Latest non-MA referral flow; context_source IS DISTINCT FROM "
        "'mobile-app' excludes MA referrals.",
        "Mobile App signups aren't excluded from the table, only from the flow/autofill "
        "join — a MA-originated identity can still appear as a signups row.",
    ],
)
def test_describing_the_mh_notebook_as_historical_is_not_a_forbidden_claim(statement):
    """The patterns must not punish the correct answer: naming the MA notebook
    as a historical / unused Mobile App variant, or explaining the
    non-Mobile-App referral filter. With the soft read on the trace such an
    answer still passes -- exactly the two Sonnet 5 repeats the baseline now
    counts as passes."""
    trace = _passing_trace()
    trace["tool_events"].append(_wrong_github_source_event())
    trace["final_text"] += " " + statement

    result = score_trace(trace, FIXTURE)

    assert not [v for v in result["violations"] if v["code"] == "unsupported_answer_claim"], statement
    assert result["passed"] is True
    assert result["soft_violations"] == 1  # the read itself is still reported


def test_build_only_column_fails_even_if_tool_fixture_did_not_error():
    trace = _passing_trace()
    trace["tool_events"][3]["input"]["sql"] = (
        "SELECT COUNT_IF(internal_sum > 0) FROM acme_prod.reporting.signups"
    )

    result = score_trace(trace, FIXTURE)

    assert any(v["code"] == "forbidden_sql_identifier" for v in result["violations"])


def test_analytical_query_before_exact_description_fails():
    trace = _passing_trace()
    query = trace["tool_events"].pop(3)
    trace["tool_events"].insert(1, query)

    result = score_trace(trace, FIXTURE)

    assert any(v["code"] == "metadata_ordering" for v in result["violations"])


def test_runtime_preflight_before_query_in_same_iteration_passes():
    trace = _passing_trace()
    description = trace["tool_events"].pop(4)
    query = trace["tool_events"].pop(4)
    description.update(
        {"iteration": 2, "source": "runtime_preflight", "model_visible": False}
    )
    query["iteration"] = 2
    trace["tool_events"].extend([description, query])

    result = score_trace(trace, FIXTURE)

    assert result["passed"] is True
    assert result["metrics"]["metadata_preflights_injected"] == 1


def test_unsupported_final_claim_fails():
    trace = _passing_trace()
    trace["final_text"] += " The job runs daily."

    result = score_trace(trace, FIXTURE)

    assert any(v["code"] == "unsupported_answer_claim" for v in result["violations"])


def test_each_day_schedule_overclaim_from_failed_trace_fails():
    trace = _passing_trace()
    trace["final_text"] += " The job runs the MERGE once each day."

    result = score_trace(trace, FIXTURE)

    assert any(v["code"] == "unsupported_answer_claim" for v in result["violations"])


def test_known_age_overclaims_fail():
    for claim in (
        "Age recomputes on every run.",
        "The age field drifts forward over time.",
        "Age and age_range are unusable for the entire table.",
    ):
        trace = _passing_trace()
        trace["final_text"] += f" {claim}"

        result = score_trace(trace, FIXTURE)

        assert any(v["code"] == "unsupported_answer_claim" for v in result["violations"]), claim


def test_qualified_insertion_time_age_warning_is_allowed():
    trace = _passing_trace()
    trace["final_text"] += (
        " Do not use AVG(age) across cohorts when the user expects current ages; "
        "the stored value is insertion-time age."
    )

    result = score_trace(trace, FIXTURE)

    assert not any(
        violation["code"] == "unsupported_answer_claim"
        for violation in result["violations"]
    )


def test_incomplete_or_over_budget_conversation_fails():
    trace = _passing_trace()
    trace["stop_reason"] = "max_api_calls"
    trace["api_calls"] = MAX_API_CALLS + 1

    result = score_trace(trace, FIXTURE)

    codes = {v["code"] for v in result["violations"]}
    assert {"incomplete_conversation", "api_call_budget"} <= codes


def test_hitting_the_production_iteration_bound_is_a_hard_api_call_budget_failure():
    """Exhausting exactly ``MAX_API_CALLS`` (never exceeding it) is what production
    turns into "Hit max iterations", so the scorer must flag it as hard."""
    trace = _passing_trace()
    trace["stop_reason"] = "max_api_calls"
    trace["api_calls"] = MAX_API_CALLS
    trace["hit_api_call_budget"] = True
    trace["final_text"] = ""

    result = score_trace(trace, FIXTURE)

    budget = [v for v in result["violations"] if v["code"] == "api_call_budget"]
    assert len(budget) == 1
    assert budget[0]["severity"] == "hard"
    assert f"{MAX_API_CALLS}/{MAX_API_CALLS}" in budget[0]["message"]
    assert "Hit max iterations" in budget[0]["message"]
    assert result["passed"] is False
    assert result["metrics"]["api_calls"] == MAX_API_CALLS
    assert result["metrics"]["max_api_calls"] == MAX_API_CALLS
    assert result["metrics"]["hit_api_call_budget"] is True

    # A conversation that finishes at the bound is not a budget failure.
    finished = _passing_trace()
    finished["api_calls"] = MAX_API_CALLS
    clean = score_trace(finished, FIXTURE)
    assert "api_call_budget" not in {v["code"] for v in clean["violations"]}
    assert clean["metrics"]["hit_api_call_budget"] is False


def test_fixture_is_synthetic_and_internal_sum_is_not_a_persisted_column():
    serialized = str(FIXTURE)
    persisted_columns = {
        column[0].lower()
        for column in FIXTURE["objects"]["acme_prod.reporting.signups"]["columns"]
    }

    assert "internal_sum" not in persisted_columns
    assert "example.invalid" not in serialized  # URLs are generated by the dispatcher only.
    assert "anthropic_api_key" not in serialized.lower()


def test_synthetic_github_search_honors_path_scope_name_and_limit():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    notebook_search = tools.dispatch(
        "github_search_code",
        {"query": "signups merge path:notebooks"},
    )
    jobs_search = tools.dispatch(
        "github_search_code",
        {"query": "signups merge path:resources/jobs"},
    )
    limited_find = tools.dispatch(
        "github_find_files",
        {"name": "SIGNUPS_MERGE", "path_prefix": "notebooks/", "limit": 1},
    )
    jobs_find = tools.dispatch(
        "github_find_files",
        {"name": "signups", "path_prefix": "resources/jobs/"},
    )

    assert notebook_search["total_count"] == 2
    assert jobs_search == {"total_count": 0, "items": []}
    assert limited_find["match_count"] == 1
    assert len(limited_find["matches"]) == 1
    assert jobs_find["match_count"] == 0
    assert jobs_find["matches"] == []


def test_synthetic_dispatch_rejects_build_only_column_without_live_sql():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    result = tools.dispatch(
        "databricks_execute_sql",
        {"sql": "SELECT internal_sum FROM acme_prod.reporting.signups"},
    )

    assert result["synthetic_failure"] == "build_only_column"
    assert "UNRESOLVED_COLUMN" in result["error"]


def test_synthetic_signups_results_match_requested_projection():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))
    aggregate = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT COUNT(*) AS total_rows, "
                "COUNT(DISTINCT signup_stitch_id) AS distinct_stitch_ids, "
                "MIN(signup_date) AS earliest_reg_date "
                "FROM acme_prod.reporting.signups"
            )
        },
    )
    flows = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT signup_flow, COUNT(*) AS cnt, 50.0 AS pct "
                "FROM acme_prod.reporting.signups GROUP BY signup_flow"
            )
        },
    )
    devices = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT device_type, autofill_ind, COUNT(*) AS cnt "
                "FROM acme_prod.reporting.signups "
                "GROUP BY device_type, autofill_ind"
            )
        },
    )

    assert aggregate["columns"] == [
        "total_rows",
        "distinct_stitch_ids",
        "earliest_reg_date",
    ]
    assert len(aggregate["sample_rows"][0]) == len(aggregate["columns"])
    assert flows["columns"] == ["signup_flow", "cnt", "pct"]
    assert all(len(row) == len(flows["columns"]) for row in flows["sample_rows"])
    assert devices["columns"] == ["device_type", "autofill_ind", "cnt"]
    assert all(len(row) == len(devices["columns"]) for row in devices["sample_rows"])


def test_synthetic_values_follow_expressions_when_aliases_change():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    result = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT COUNT(*) AS arbitrary_total, "
                "COUNT(DISTINCT signup_stitch_id) AS arbitrary_distinct, "
                "MAX(signup_date) AS arbitrary_latest "
                "FROM acme_prod.reporting.signups"
            )
        },
    )

    assert result["columns"] == [
        "arbitrary_total",
        "arbitrary_distinct",
        "arbitrary_latest",
    ]
    assert result["sample_rows"] == [[1284000, 1269000, "2026-08-27"]]


def test_synthetic_unknown_projection_returns_limit_without_fabricated_values():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    result = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT SUM(session_count) AS sessions "
                "FROM acme_prod.reporting.signups"
            )
        },
    )

    assert result["status"] == "SUCCEEDED"
    assert result["synthetic_limitation"] == "projection_not_modeled"
    assert result["unsupported_columns"] == ["sessions"]
    assert result["columns"] == ["sessions"]
    assert result["sample_rows"] == []
    assert result["row_count"] is None
    assert "error" not in result


def test_synthetic_describe_missing_object_matches_production_zero_row_contract():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    result = tools.dispatch(
        "databricks_describe_object",
        {"catalog": "catalog_a", "schema": "schema_a", "name": "missing_table"},
    )

    assert result["status"] == "SUCCEEDED"
    assert result["column_count"] == 0
    assert result["columns"] == []
    assert result["object"]["full_name"] == "catalog_a.schema_a.missing_table"
    assert result["object"]["object_type"] is None
    assert "warning" in result
    assert "error" not in result
    assert "row_count" not in result and "sample_rows" not in result


def test_synthetic_describe_emits_structured_object_and_typed_columns():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    result = tools.dispatch(
        "databricks_describe_object",
        {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
    )
    fixture_columns = FIXTURE["objects"]["acme_prod.reporting.signups"]["columns"]

    assert result["object"]["full_name"] == "acme_prod.reporting.signups"
    assert result["object"]["object_type"] == "TABLE"
    assert result["object"]["table_type"] == "MANAGED"
    assert result["column_count"] == len(fixture_columns) == len(result["columns"])
    assert [column["name"] for column in result["columns"]] == [
        column[0] for column in fixture_columns
    ]
    # Fixture columns are [name, type, comment] or [name, type, comment, nullable].
    assert [column["name"] for column in result["columns"] if column["is_nullable"] is False] == [
        "_anonymousId",
        "signup_date",
        "inserted_at",
    ]
    assert result["columns_truncated"] is False
    assert "metadata_kind" not in json.dumps(result)


def test_synthetic_runtime_executes_the_real_calculate_tool():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    result = tools.dispatch(
        "calculate",
        {"expression": "pct(part, whole)", "variables": {"part": 15000, "whole": 1284000}},
    )

    assert "error" not in result
    assert result["result_display"] == "1.168224"
    assert result["steps"] == ["pct(15000, 1284000)"]


def test_canonical_job_history_object_is_available_to_the_synthetic_runtime():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    metadata = tools.dispatch(
        "databricks_describe_object",
        {
            "catalog": "acme_prod",
            "schema": "acme_history",
            "name": "databricks_workspace_jobs",
        },
    )
    query = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT version_ts, job_name, job_status "
                "FROM acme_prod.acme_history.databricks_workspace_jobs "
                "WHERE job_name LIKE '%Signups%Merge%' "
                "ORDER BY version_ts DESC LIMIT 10"
            )
        },
    )

    assert metadata["status"] == "SUCCEEDED"
    assert metadata["column_count"] == 6
    assert metadata["object"]["object_type"] == "TABLE"
    assert query["status"] == "SUCCEEDED"
    assert query["synthetic_limitation"] == "projection_not_modeled"
    assert "error" not in query


def test_synthetic_information_schema_is_target_aware():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    known = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT column_name, data_type FROM acme_prod.information_schema.columns "
                "WHERE table_schema = 'acme_history' "
                "AND table_name = 'databricks_workspace_jobs'"
            )
        },
    )
    missing = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT column_name, data_type FROM acme_prod.information_schema.columns "
                "WHERE table_schema = 'acme_history' AND table_name = 'missing_table'"
            )
        },
    )

    assert [row[0] for row in known["sample_rows"]] == [
        "version_ts",
        "job_id",
        "job_name",
        "job_status",
        "job_settings_json",
        "job_json",
    ]
    assert missing["status"] == "SUCCEEDED"
    assert missing["row_count"] == 0
    assert missing["sample_rows"] == []


def test_synthetic_lineage_returns_only_requested_columns():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    one_column = tools.dispatch(
        "databricks_execute_sql",
        {"sql": "SELECT source_table_full_name FROM system.access.table_lineage"},
    )
    two_columns = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT source_table_full_name, entity_type "
                "FROM system.access.table_lineage"
            )
        },
    )

    assert one_column["columns"] == ["source_table_full_name"]
    assert all(len(row) == 1 for row in one_column["sample_rows"])
    assert two_columns["columns"] == ["source_table_full_name", "entity_type"]
    assert all(len(row) == 2 for row in two_columns["sample_rows"])


def test_synthetic_supports_observed_aggregate_and_ordinal_grouping_shapes():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))
    aggregate = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT COUNT(*) AS total_rows, "
                "COUNT(DISTINCT _anonymousId) AS distinct_anon_ids, "
                "MIN(signup_date) AS earliest_reg_date, "
                "MAX(signup_date) AS latest_reg_date, "
                "SUM(CASE WHEN signup_stitch_id IS NULL THEN 1 ELSE 0 END) "
                "AS null_stitch_id, "
                "SUM(CASE WHEN dob = DATE '1492-02-16' THEN 1 ELSE 0 END) "
                "AS masked_dob_rows "
                "FROM acme_prod.reporting.signups"
            )
        },
    )
    grouped = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT signup_flow, autofill_ind, device_type, COUNT(*) AS rows "
                "FROM acme_prod.reporting.signups GROUP BY 1, 2, 3"
            )
        },
    )

    assert aggregate["columns"] == [
        "total_rows",
        "distinct_anon_ids",
        "earliest_reg_date",
        "latest_reg_date",
        "null_stitch_id",
        "masked_dob_rows",
    ]
    assert aggregate["sample_rows"] == [
        [1284000, 1284000, "2024-01-02", "2026-08-27", 15000, 1284000]
    ]
    assert grouped["columns"] == [
        "signup_flow",
        "autofill_ind",
        "device_type",
        "rows",
    ]
    assert all(value is not None for row in grouped["sample_rows"] for value in row)


def test_synthetic_supports_grouped_lineage_and_categorical_counts():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))
    lineage = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT entity_type, COUNT(DISTINCT entity_id) AS consumers "
                "FROM system.access.table_lineage GROUP BY 1"
            )
        },
    )
    categories = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT COUNT(CASE WHEN autofill_ind = 'true' THEN 1 END) AS autofills, "
                "COUNT(CASE WHEN device_type = 'Desktop' THEN 1 END) AS desktops, "
                "COUNT(CASE WHEN device_type = 'Mobile' THEN 1 END) AS mobiles "
                "FROM acme_prod.reporting.signups"
            )
        },
    )

    assert lineage["columns"] == ["entity_type", "consumers"]
    assert lineage["sample_rows"] == [["NOTEBOOK", 1]]
    assert categories["columns"] == ["autofills", "desktops", "mobiles"]
    assert categories["sample_rows"] == [[392000, 465000, 731000]]


def test_synthetic_metadata_limit_is_repaired_without_tool_error():
    tools = SyntheticTools(copy.deepcopy(FIXTURE))

    result = tools.dispatch(
        "databricks_execute_sql",
        {"sql": "DESCRIBE HISTORY acme_prod.reporting.signups LIMIT 5"},
    )

    assert result["status"] == "SUCCEEDED"
    assert result["automatic_repair"] == "removed_invalid_metadata_limit"
    assert "error" not in result


class _Block(SimpleNamespace):
    def model_dump(self):
        if self.type == "tool_use":
            return {
                "type": "tool_use",
                "id": self.id,
                "name": self.name,
                "input": self.input,
            }
        return {"type": "text", "text": self.text}


def _usage(output_tokens, **overrides):
    values = {
        "input_tokens": 1000,
        "cache_creation_input_tokens": 500,
        "cache_read_input_tokens": 20000,
        "output_tokens": output_tokens,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _ConversationClient:
    def __init__(
        self,
        sql="SELECT source_table_full_name FROM system.access.table_lineage",
        *,
        tool_output_tokens=100,
        final_output_tokens=20,
        response_model="claude-sonnet-4-6",
    ):
        self.messages = self
        self.requests = []
        self.request_kwargs = []
        self.sql = sql
        self.tool_output_tokens = tool_output_tokens
        self.final_output_tokens = final_output_tokens
        self.response_model = response_model

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        self.request_kwargs.append({k: v for k, v in kwargs.items() if k != "messages"})
        if len(self.requests) == 1:
            return SimpleNamespace(
                content=[
                    _Block(
                        type="tool_use",
                        id="lineage-query",
                        name="databricks_execute_sql",
                        input={"sql": self.sql},
                    )
                ],
                stop_reason="tool_use",
                model=self.response_model,
                usage=_usage(self.tool_output_tokens),
            )
        return SimpleNamespace(
            content=[_Block(type="text", text="Done.")],
            stop_reason="end_turn",
            model=self.response_model,
            usage=_usage(self.final_output_tokens),
        )


def test_runner_passes_effort_as_output_config_kwarg_and_records_config():
    client = _ConversationClient()

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client, effort="medium")

    first = client.request_kwargs[0]
    assert first["output_config"] == {"effort": "medium"}
    assert first["thinking"] == {"type": "adaptive"}
    assert first["max_tokens"] == 24_000
    assert "extra_body" not in first
    assert trace["effort"] == "medium"
    assert trace["thinking"] == "adaptive"
    assert trace["max_tokens"] == 24_000


def test_runner_can_disable_or_omit_thinking_and_override_max_tokens():
    disabled = _ConversationClient()
    run_conversation(copy.deepcopy(FIXTURE), client=disabled, thinking="disabled", max_tokens=4096)
    assert disabled.request_kwargs[0]["thinking"] == {"type": "disabled"}
    assert disabled.request_kwargs[0]["max_tokens"] == 4096

    omitted = _ConversationClient()
    run_conversation(copy.deepcopy(FIXTURE), client=omitted, thinking=None, effort=None)
    assert "thinking" not in omitted.request_kwargs[0]
    assert "output_config" not in omitted.request_kwargs[0]

    with pytest.raises(ValueError):
        run_conversation(copy.deepcopy(FIXTURE), client=_ConversationClient(), thinking="bogus")


class _LegacySdkClient(_ConversationClient):
    """Mimics an SDK whose ``create`` has no ``output_config`` (and no **kwargs)."""

    def __init__(self):
        super().__init__()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, tools, messages, thinking=None, extra_body=None):
        return super().create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            thinking=thinking,
            extra_body=extra_body,
        )


def test_runner_falls_back_to_extra_body_when_sdk_rejects_output_config():
    client = _LegacySdkClient()

    run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert client.request_kwargs[0]["extra_body"] == {"output_config": {"effort": "high"}}
    assert "output_config" not in client.request_kwargs[0]


def _pre_abstraction_request_kwargs(*, model, messages, effort, thinking, max_tokens):
    """Verbatim ``build_request_kwargs`` from before ``utils/llm`` existed (origin/main).

    The eval gate passed an earlier CI run with this exact request; the
    provider path must send the same bytes so eval results stay comparable.
    """
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": load_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": ALL_TOOLS,
        "messages": messages,
    }
    if thinking is not None:
        kwargs["thinking"] = {"type": thinking}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    return kwargs


class _FullCaptureClient(_ConversationClient):
    """Keeps every kwarg of every call (deep-copied) for byte-level comparison."""

    def __init__(self):
        super().__init__()
        self.full_kwargs = []

    def create(self, **kwargs):
        self.full_kwargs.append(copy.deepcopy(kwargs))
        return super().create(**kwargs)


def test_runner_request_is_identical_to_the_pre_provider_golden():
    client = _FullCaptureClient()

    run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert len(client.full_kwargs) == 2  # tool_use turn, then end_turn
    for sent in client.full_kwargs:
        golden = _pre_abstraction_request_kwargs(
            model=DEFAULT_MODEL,
            messages=sent["messages"],
            effort=DEFAULT_EFFORT,
            thinking=DEFAULT_THINKING,
            max_tokens=MAX_TOKENS_PER_CALL,
        )
        assert sent == golden
        # Same key order too (the SDK serialises in insertion order).
        assert list(sent) == list(golden)
        # No provider-era extras: no per-request timeout, stream flag, metadata.
        assert set(sent) == {"model", "max_tokens", "system", "tools", "messages", "thinking", "output_config"}

    # Tool definitions round-trip through tools.registry unchanged: same order,
    # same description text, same input_schema, no added or dropped keys.
    sent_tools = client.full_kwargs[0]["tools"]
    assert [tool["name"] for tool in sent_tools] == [tool["name"] for tool in ALL_TOOLS]
    for sent_tool, original in zip(sent_tools, ALL_TOOLS):
        assert sent_tool == original
        assert list(sent_tool) == ["name", "description", "input_schema"] == list(original)

    # The provider-specific prompt file adds nothing for Anthropic: the system
    # text is the provider-neutral prompt, byte for byte.
    assert client.full_kwargs[0]["system"][0]["text"] == load_base_prompt()

    # The second request replays the first response exactly as the pre-provider
    # runner did (``block.model_dump()`` per content block, then tool_result).
    second_messages = client.full_kwargs[1]["messages"]
    assert second_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "lineage-query",
                "name": "databricks_execute_sql",
                "input": {"sql": client.sql},
            }
        ],
    }
    assert second_messages[2]["role"] == "user"
    assert second_messages[2]["content"][0]["type"] == "tool_result"
    assert second_messages[2]["content"][0]["tool_use_id"] == "lineage-query"


def test_runner_records_per_call_usage_latency_and_estimated_cost():
    client = _ConversationClient(response_model="claude-sonnet-4-6")

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert len(trace["api_call_events"]) == 2
    first_call = trace["api_call_events"][0]
    assert first_call["model"] == "claude-sonnet-4-6"
    assert first_call["stop_reason"] == "tool_use"
    assert first_call["latency_ms"] >= 0
    assert first_call["usage"] == {
        "input_tokens": 1000,
        "cache_creation_input_tokens": 500,
        "cache_read_input_tokens": 20000,
        "output_tokens": 100,
    }
    assert trace["usage"] == {
        "input_tokens": 2000,
        "cache_creation_input_tokens": 1000,
        "cache_read_input_tokens": 40000,
        "output_tokens": 120,
    }
    assert trace["output_tokens"] == 120
    assert trace["response_models"] == ["claude-sonnet-4-6"]
    assert trace["estimated_cost_usd"] == estimate_cost_usd(
        "claude-sonnet-4-6",
        input_tokens=2000,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=40000,
        output_tokens=120,
    )
    assert all("latency_ms" in event for event in trace["tool_events"])
    assert {"api_total_ms", "tool_total_ms", "wall_clock_ms"} <= set(trace["latency_ms"])
    assert trace["latency_ms"]["wall_clock_ms"] >= trace["latency_ms"]["api_total_ms"]


def test_unknown_model_yields_none_cost_without_crashing():
    client = _ConversationClient(response_model="claude-experimental-9")

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client, model="claude-experimental-9")

    assert trace["estimated_cost_usd"] is None
    assert trace["stop_reason"] == "end_turn"


def test_completed_answer_over_output_budget_is_recorded_not_masked():
    client = _ConversationClient(final_output_tokens=MAX_TOTAL_OUTPUT_TOKENS + 1)

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)
    result = score_trace(trace, FIXTURE)

    assert trace["stop_reason"] == "end_turn"
    assert trace["final_text"] == "Done."
    assert trace["output_token_budget_exceeded"] is True
    codes = {v["code"] for v in result["violations"]}
    assert "output_token_budget" in codes
    assert "incomplete_conversation" not in codes
    assert "empty_answer" not in codes


def test_tool_use_turn_over_output_budget_stops_with_budget_reason():
    client = _ConversationClient(tool_output_tokens=MAX_TOTAL_OUTPUT_TOKENS + 1)

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert trace["stop_reason"] == "output_token_budget"
    assert trace["api_calls"] == 1
    assert trace["output_token_budget_exceeded"] is True


def test_output_budget_is_production_bound_and_was_not_tripped_by_the_old_cap():
    """30k output tokens (the old harness cap) must not fail a conversation that
    production would have let finish; the cap is per-call max_tokens times the
    iteration budget, and follows both knobs."""
    client = _ConversationClient(final_output_tokens=40_000)

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)
    result = score_trace(trace, FIXTURE)

    # 24,000 per call (claude-sonnet-5 / high) x 25 iterations.
    assert trace["budget"]["max_output_tokens"] == MAX_TOTAL_OUTPUT_TOKENS == 600_000
    assert trace["output_token_budget_exceeded"] is False
    assert "output_token_budget" not in {v["code"] for v in result["violations"]}

    scaled = run_conversation(
        copy.deepcopy(FIXTURE), client=_ConversationClient(), max_tokens=4096, max_api_calls=10
    )
    assert scaled["budget"]["max_output_tokens"] == 4096 * 10


class _NeverFinishesClient(_ConversationClient):
    """Keeps asking for the same metadata; production would hit MAX_ITERATIONS."""

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        self.request_kwargs.append({k: v for k, v in kwargs.items() if k != "messages"})
        return SimpleNamespace(
            content=[
                _Block(
                    type="tool_use",
                    id=f"describe-{len(self.requests)}",
                    name="databricks_describe_object",
                    input={"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
                )
            ],
            stop_reason="tool_use",
            model=self.response_model,
            usage=_usage(self.tool_output_tokens),
        )


def test_runner_default_api_call_budget_is_production_max_iterations():
    client = _NeverFinishesClient()

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)
    result = score_trace(trace, FIXTURE)

    assert len(client.requests) == MAX_API_CALLS == 25
    assert trace["api_calls"] == MAX_API_CALLS
    assert trace["hit_api_call_budget"] is True
    assert trace["stop_reason"] == "max_api_calls"
    assert trace["budget"] == {
        "max_api_calls": MAX_API_CALLS,
        "max_api_calls_source": "default",
        "max_output_tokens": MAX_TOTAL_OUTPUT_TOKENS,
    }
    codes = {v["code"]: v["severity"] for v in result["violations"]}
    assert codes["api_call_budget"] == "hard"
    assert codes["incomplete_conversation"] == "hard"
    assert result["passed"] is False


def test_runner_honors_fixture_max_api_calls_and_cli_override_wins():
    fixture = copy.deepcopy(FIXTURE)
    fixture["max_api_calls"] = 12  # the pre-production-parity harness cap

    client = _NeverFinishesClient()
    trace = run_conversation(fixture, client=client)
    assert len(client.requests) == 12
    assert trace["api_calls"] == 12
    assert trace["hit_api_call_budget"] is True
    assert trace["budget"]["max_api_calls"] == 12
    assert trace["budget"]["max_api_calls_source"] == "fixture"
    assert trace["budget"]["max_output_tokens"] == MAX_TOKENS_PER_CALL * 12

    client = _NeverFinishesClient()
    trace = run_conversation(copy.deepcopy(fixture), client=client, max_api_calls=3)
    assert len(client.requests) == 3
    assert trace["budget"]["max_api_calls"] == 3
    assert trace["budget"]["max_api_calls_source"] == "override"

    # A conversation that finishes inside the budget never reports a hit.
    finished = run_conversation(copy.deepcopy(FIXTURE), client=_ConversationClient(), max_api_calls=3)
    assert finished["api_calls"] == 2
    assert finished["hit_api_call_budget"] is False
    assert finished["stop_reason"] == "end_turn"

    with pytest.raises(ValueError):
        run_conversation(copy.deepcopy(FIXTURE), client=_ConversationClient(), max_api_calls=0)


def test_runtime_preflight_is_recorded_but_not_added_to_model_messages():
    client = _ConversationClient()

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert [event["source"] for event in trace["tool_events"]] == [
        "runtime_preflight",
        "fixture",
    ]
    assert trace["tool_events"][0]["model_visible"] is False
    assert trace["tool_events"][1]["is_error"] is False
    tool_results = client.requests[1][-1]["content"]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "lineage-query"
    assert "metadata_kind" not in tool_results[0]["content"]
    assert "column_count" not in tool_results[0]["content"]


class _MetadataFailureTools:
    def __init__(self, *, raises=False):
        self.raises = raises
        self.calls = []

    def dispatch(self, name, tool_input):
        self.calls.append((name, tool_input))
        if name != "databricks_describe_object":
            raise AssertionError("guarded SQL must not reach the fixture dispatcher")
        if self.raises:
            raise RuntimeError("private metadata service detail")
        return {"status": "SUCCEEDED", "row_count": 0, "sample_rows": []}


def test_failed_runtime_preflight_stays_hidden_and_sql_remains_guarded():
    client = _ConversationClient()
    tools = _MetadataFailureTools()

    trace = run_conversation(
        copy.deepcopy(FIXTURE),
        client=client,
        tool_runtime=tools,
    )
    result = score_trace(trace, FIXTURE)

    assert [name for name, _tool_input in tools.calls] == [
        "databricks_describe_object"
    ]
    assert [event["source"] for event in trace["tool_events"]] == [
        "runtime_preflight",
        "guard",
    ]
    assert [event["model_visible"] for event in trace["tool_events"]] == [False, True]
    assert [event["is_error"] for event in trace["tool_events"]] == [False, True]
    assert len(client.requests[1][-1]["content"]) == 1
    assert client.requests[1][-1]["content"][0]["is_error"] is True
    assert result["passed"] is False
    assert result["avoidable_tool_failures"] >= 1


def test_runtime_preflight_exception_trace_omits_exception_detail():
    client = _ConversationClient()
    tools = _MetadataFailureTools(raises=True)

    trace = run_conversation(
        copy.deepcopy(FIXTURE),
        client=client,
        tool_runtime=tools,
    )
    serialized = str(trace)

    assert "private metadata service detail" not in serialized
    assert trace["tool_events"][0]["result"] == {
        "error": "Automatic metadata preflight failed.",
        "error_type": "RuntimeError",
    }


class _RecordingSyntheticTools(SyntheticTools):
    def __init__(self, fixture):
        super().__init__(fixture)
        self.calls = []

    def dispatch(self, name, tool_input):
        self.calls.append((name, tool_input))
        return super().dispatch(name, tool_input)


def test_build_only_column_is_blocked_after_hidden_runtime_preflight():
    sql = "SELECT internal_sum FROM acme_prod.reporting.signups"
    client = _ConversationClient(sql)
    tools = _RecordingSyntheticTools(copy.deepcopy(FIXTURE))

    trace = run_conversation(
        copy.deepcopy(FIXTURE),
        client=client,
        tool_runtime=tools,
    )
    result = score_trace(trace, FIXTURE)

    assert [name for name, _tool_input in tools.calls] == [
        "databricks_describe_object"
    ]
    assert trace["tool_events"][0]["model_visible"] is False
    assert trace["tool_events"][1]["result"]["validation_failure"] == (
        "column_not_in_live_schema"
    )
    assert client.requests[1][-1]["content"][0]["is_error"] is True
    assert result["passed"] is False


class _FinalRevisionClient:
    def __init__(self):
        self.messages = self
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        text = (
            "The Signups job runs daily. One row per `_anonymousId` browser identity."
            if len(self.requests) == 1
            else "One row per `_anonymousId` browser identity."
        )
        return SimpleNamespace(
            content=[_Block(type="text", text=text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(output_tokens=30),  # partial usage must not crash
        )


def test_evaluation_privately_revises_unverified_schedule_claim():
    client = _FinalRevisionClient()

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert trace["api_calls"] == 2
    assert "runs daily" not in trace["final_text"]
    assert "internal correction" in str(client.requests[1][-1]).lower()


# ------------------------------------------------------------ numeric scoring


def _codes(result):
    return {violation["code"] for violation in result["violations"]}


def _message(result, code):
    return next(v["message"] for v in result["violations"] if v["code"] == code)


def _calc_event(expression="pct_change(old, new)", *, variables=None, is_error=False, iteration=3):
    tool_input = {"expression": expression, "variables": variables or {"old": 12480, "new": 13977}}
    if is_error:
        result = {"error": "Division by zero", "validation_failure": "division_by_zero"}
    else:
        result = calc_tool.dispatch("calculate", tool_input)  # deterministic, no network
    return _event("calculate", tool_input, result, iteration=iteration, is_error=is_error)


def test_required_tool_calls_missing_is_hard_and_satisfied_by_a_successful_call():
    fixture = copy.deepcopy(FIXTURE)
    fixture["required_tool_calls"] = [{"tool": "calculate", "min": 1}]

    missing = score_trace(_passing_trace(), fixture)
    assert [
        v["severity"] for v in missing["violations"] if v["code"] == "required_tool_missing"
    ] == ["hard"]
    assert "observed 0" in _message(missing, "required_tool_missing")
    assert missing["passed"] is False

    trace = _passing_trace()
    trace["tool_events"].append(_calc_event())
    satisfied = score_trace(trace, fixture)
    assert "required_tool_missing" not in _codes(satisfied)
    assert satisfied["passed"] is True


def test_required_tool_calls_honor_min_input_contains_and_ignore_failed_calls():
    fixture = copy.deepcopy(FIXTURE)
    fixture["required_tool_calls"] = [
        {"tool": "calculate", "min": 2, "input_contains": "PCT_CHANGE"}
    ]
    trace = _passing_trace()
    trace["tool_events"].append(_calc_event())  # matches
    trace["tool_events"].append(_calc_event("(new - old) / old * 100"))  # other expression
    trace["tool_events"].append(_calc_event(is_error=True))  # failed calls never count

    short = score_trace(trace, fixture)
    assert "required_tool_missing" in _codes(short)
    assert "observed 1" in _message(short, "required_tool_missing")
    assert "'pct_change'" in _message(short, "required_tool_missing")

    trace["tool_events"].append(_calc_event())
    assert "required_tool_missing" not in _codes(score_trace(trace, fixture))

    fixture["required_tool_calls"] = ["calculate"]  # bare name: min 1, any input
    assert "required_tool_missing" not in _codes(score_trace(trace, fixture))


def test_final_number_tokens_are_whole_figures_only():
    tokens = final_number_tokens(
        "35 columns, 12,480 rows, 11.995192%, col_36, 1236, v2, 36th, 3.36, us-east-1, 36"
    )

    assert tokens == [
        Decimal(35),
        Decimal(12480),
        Decimal("11.995192"),
        Decimal(1236),
        Decimal("3.36"),
        Decimal(36),
    ]


def test_required_and_forbidden_final_numbers_match_whole_tokens():
    fixture = copy.deepcopy(FIXTURE)
    fixture["required_final_numbers"] = ["35", "12480"]
    fixture["forbidden_final_numbers"] = ["36"]

    good = _passing_trace()
    good["final_text"] += (
        " It has 35.0 columns and 12,480 rows; ignore col_36, v36, 1236, 2.36, 360 and 36th."
    )
    clean = score_trace(good, fixture)
    assert not {"required_number_missing", "forbidden_number_present"} & _codes(clean)

    bad = _passing_trace()
    bad["final_text"] += " It has 36 columns and 12480 rows."
    result = score_trace(bad, fixture)
    assert {"required_number_missing", "forbidden_number_present"} <= _codes(result)
    assert "'35'" in _message(result, "required_number_missing")
    assert "'36'" in _message(result, "forbidden_number_present")
    assert all(
        v["severity"] == "hard"
        for v in result["violations"]
        if v["code"] in {"required_number_missing", "forbidden_number_present"}
    )
    assert result["passed"] is False


def test_numeric_grounding_flags_unsupported_figures_softly_by_default():
    fixture = copy.deepcopy(FIXTURE)
    fixture["numeric_grounding"] = True
    trace = _passing_trace()
    trace["tool_events"][2]["result"] = {"status": "SUCCEEDED", "column_count": 35, "columns": []}
    trace["final_text"] += " The table has 36 columns."

    result = score_trace(trace, fixture)

    ungrounded = [v for v in result["violations"] if v["code"] == "ungrounded_number"]
    assert len(ungrounded) == 1
    assert ungrounded[0]["severity"] == "soft"
    assert "36" in ungrounded[0]["message"]
    assert result["passed"] is True
    assert result["soft_violations"] == 1

    trace["final_text"] = trace["final_text"].replace("36 columns", "35 columns")
    assert "ungrounded_number" not in _codes(score_trace(trace, fixture))


def test_numeric_grounding_uses_user_prompt_and_tool_inputs_and_honors_hard_override():
    fixture = copy.deepcopy(FIXTURE)
    fixture["numeric_grounding"] = True
    fixture["user_prompt"] = "Compare 4,821 to the August figure."
    fixture["violation_severity"] = {"ungrounded_number": "hard"}
    trace = _passing_trace()
    trace["tool_events"].append(_calc_event(variables={"old": 4821, "new": 5655}))
    trace["final_text"] += " Signups grew from 4,821 to 5,655, a 17.3% increase."

    grounded = score_trace(trace, fixture)
    assert "ungrounded_number" not in _codes(grounded)

    trace["final_text"] += " Roughly 9,000 of those were mobile."
    result = score_trace(trace, fixture)
    assert _message(result, "ungrounded_number").endswith("9,000.")
    assert [v["severity"] for v in result["violations"] if v["code"] == "ungrounded_number"] == [
        "hard"
    ]
    assert result["passed"] is False


def test_numeric_grounding_is_opt_in_and_revision_metric_mirrors_the_trace():
    trace = _passing_trace()
    trace["final_text"] += " The table has 36 columns."

    result = score_trace(trace, FIXTURE)
    assert "ungrounded_number" not in _codes(result)
    assert result["metrics"]["numeric_guard_revisions"] is None

    trace["numeric_guard_revisions"] = 2
    assert score_trace(trace, FIXTURE)["metrics"]["numeric_guard_revisions"] == 2


# ------------------------------------------------- runner numeric revision pass

UNGROUNDED = "Signups grew from 4,821 to 5,655, a 17.3% increase."
GROUNDED = "One row per `_anonymousId` browser identity."


class _NumericRevisionClient:
    def __init__(self, drafts):
        self.messages = self
        self.requests = []
        self.drafts = list(drafts)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        return SimpleNamespace(
            content=[_Block(type="text", text=self.drafts.pop(0))],
            stop_reason="end_turn",
            usage=SimpleNamespace(output_tokens=30),
        )


def test_runner_privately_revises_ungrounded_figures_and_records_the_revision():
    client = _NumericRevisionClient([UNGROUNDED, GROUNDED])

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert trace["api_calls"] == 2
    assert trace["numeric_guard_mode"] == "revise"
    assert trace["numeric_guard_revisions"] == 1
    assert trace["numeric_guard_unverified_figures"] == []
    assert "_anonymousId" in trace["final_text"] and "4,821" not in trace["final_text"]
    # Model-visible sequence is production's: user prompt, draft, private instruction.
    second = client.requests[1]
    assert [message["role"] for message in second] == ["user", "assistant", "user"]
    assert second[1]["content"][0]["text"] == UNGROUNDED
    # The provider promotes the string instruction to a single text block so it
    # can carry the moving cache breakpoint; the text itself is unchanged.
    (instruction_block,) = second[2]["content"]
    instruction = instruction_block["text"]
    assert '"4,821"' in instruction and '"17.3%"' in instruction
    assert "Do not mention this internal correction request" in instruction
    assert score_trace(trace, FIXTURE)["metrics"]["numeric_guard_revisions"] == 1


def test_runner_footnotes_when_the_revision_is_still_ungrounded():
    client = _NumericRevisionClient([UNGROUNDED, UNGROUNDED])

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert trace["api_calls"] == 2
    assert trace["numeric_guard_revisions"] == 1
    assert trace["numeric_guard_unverified_figures"] == ["4,821", "5,655", "17.3%"]
    assert trace["final_text"].startswith("Signups grew from 4,821")
    assert "Figures not verified against a tool result: 4,821, 5,655, 17.3%" in trace["final_text"]


def test_runner_numeric_guard_honors_production_env_modes(monkeypatch):
    monkeypatch.setenv("NUMERIC_GUARD_MODE", "annotate")
    annotated = run_conversation(
        copy.deepcopy(FIXTURE), client=_NumericRevisionClient([UNGROUNDED])
    )
    assert annotated["api_calls"] == 1
    assert annotated["numeric_guard_revisions"] == 0
    assert "Figures not verified" in annotated["final_text"]

    monkeypatch.setenv("NUMERIC_GUARD_MODE", "off")
    off = run_conversation(copy.deepcopy(FIXTURE), client=_NumericRevisionClient([UNGROUNDED]))
    assert off["numeric_guard_mode"] == "off"
    assert off["final_text"].startswith(UNGROUNDED) and "not verified" not in off["final_text"]

    monkeypatch.setenv("NUMERIC_GUARD_MODE", "revise")
    monkeypatch.setenv("NUMERIC_GUARD_MAX_REVISIONS", "2")
    twice = run_conversation(
        copy.deepcopy(FIXTURE),
        client=_NumericRevisionClient([UNGROUNDED, UNGROUNDED, GROUNDED]),
    )
    assert twice["api_calls"] == 3
    assert twice["numeric_guard_revisions"] == 2
    assert "_anonymousId" in twice["final_text"]


class _CalcThenTextClient(_NumericRevisionClient):
    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        if len(self.requests) == 1:
            return SimpleNamespace(
                content=[
                    _Block(
                        type="tool_use",
                        id="calc-1",
                        name="calculate",
                        input={
                            "expression": "pct_change(old, new)",
                            "variables": {"old": 4821, "new": 5655},
                        },
                    )
                ],
                stop_reason="tool_use",
                usage=SimpleNamespace(output_tokens=30),
            )
        return SimpleNamespace(
            content=[_Block(type="text", text=self.drafts.pop(0))],
            stop_reason="end_turn",
            usage=SimpleNamespace(output_tokens=30),
        )


def test_runner_grounds_figures_from_tool_inputs_and_results_without_revision():
    client = _CalcThenTextClient([UNGROUNDED])

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert trace["api_calls"] == 2
    assert trace["tool_events"][0]["name"] == "calculate"
    assert trace["tool_events"][0]["result"]["result_display"] == "17.299315"
    assert trace["numeric_guard_revisions"] == 0
    assert trace["final_text"].startswith(UNGROUNDED)


def test_runner_evidence_and_numeric_revisions_chain_like_production():
    client = _NumericRevisionClient(
        [f"The Signups job runs daily. {UNGROUNDED}", UNGROUNDED, GROUNDED]
    )

    trace = run_conversation(copy.deepcopy(FIXTURE), client=client)

    assert trace["api_calls"] == 3
    assert trace["final_revisions"] == 1
    assert trace["numeric_guard_revisions"] == 1
    assert "runs daily" not in trace["final_text"] and "4,821" not in trace["final_text"]
    evidence_request = str(client.requests[1][-1]).lower()
    numeric_request = str(client.requests[2][-1]).lower()
    assert "internal correction" in evidence_request
    assert "do not match any number" not in evidence_request
    assert "do not match any number" in numeric_request


# ----------------------------------------------------------- provider plumbing


def test_trace_records_provider_and_defaults_to_anthropic():
    trace = run_conversation(copy.deepcopy(FIXTURE), client=_ConversationClient())
    assert trace["provider"] == "anthropic"


class _FakeOpenAIClient:
    """Responses-API-shaped fake: one function call, then a final message."""

    def __init__(self):
        self.responses = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        usage = {"input_tokens": 1_500, "input_tokens_details": {"cached_tokens": 1_000}, "output_tokens": 40}
        if len(self.calls) == 1:
            return {
                "model": "gpt-5.6-terra",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
                    {
                        "type": "function_call",
                        "call_id": "call_lineage",
                        "name": "databricks_execute_sql",
                        "arguments": json.dumps(
                            {"sql": "SELECT source_table_full_name FROM system.access.table_lineage"}
                        ),
                    },
                ],
                "usage": usage,
            }
        return {
            "model": "gpt-5.6-terra",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}],
            "usage": usage,
        }


def test_runner_drives_the_openai_provider_end_to_end():
    client = _FakeOpenAIClient()

    trace = run_conversation(
        copy.deepcopy(FIXTURE), client=client, provider="openai", model="gpt-5.6-terra", effort="high"
    )

    assert trace["provider"] == "openai"
    assert trace["stop_reason"] == "end_turn"
    assert trace["final_text"] == "Done."
    assert trace["api_calls"] == 2
    first, second = client.calls
    assert first["model"] == "gpt-5.6-terra"
    assert first["reasoning"] == {"effort": "high"}
    assert first["store"] is False
    assert first["max_output_tokens"] == 25_000
    assert first["instructions"].endswith("the final message text.")  # provider notes appended last
    assert {tool["type"] for tool in first["tools"]} == {"function"}
    # Second request replays reasoning + function_call and supplies the tool output.
    kinds = [item.get("type") or item.get("role") for item in second["input"]]
    assert kinds == ["user", "reasoning", "function_call", "function_call_output"]
    assert second["input"][3]["call_id"] == "call_lineage"
    # Usage is normalized to Anthropic accounting (input excludes cached tokens).
    assert trace["usage"] == {
        "input_tokens": 1_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 2_000,
        "output_tokens": 80,
    }
    assert trace["estimated_cost_usd"] == estimate_cost_usd(
        "gpt-5.6-terra", input_tokens=1_000, cache_read_input_tokens=2_000, output_tokens=80
    )
