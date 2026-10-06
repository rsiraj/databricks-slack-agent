"""The counting and arithmetic grounding scenarios: fixtures, dispatch, scoring, runner."""

import copy
import json
import re
from types import SimpleNamespace

import pytest

from evals.conversation_runner import run_conversation
from evals.run import validate_scenarios
from evals.scoring import score_trace
from evals.synthetic_tools import FIXTURES_DIR, SyntheticTools, load_fixture
from tools import calc_tool
from utils.sql_run_guard import SqlRunGuard

COLUMN_COUNT = load_fixture("signups_column_count")
PERCENT_CHANGE = load_fixture("signup_percent_change")
EXPLANATION = load_fixture("signups_table_explanation")
TABLE = "acme_prod.reporting.signups"
LINEAGE = "system.access.table_lineage"
DESCRIBE_INPUT = {"catalog": "acme_prod", "schema": "reporting", "name": "signups"}
LINEAGE_INPUT = {"catalog": "system", "schema": "access", "name": "table_lineage"}
JULY, AUGUST = 12480, 13977
EXACT_PCT = 11.995192307692308

# Verbatim model SQL from an earlier CI run (claude-sonnet-4-6 / high).
CI_LINEAGE_READS_SQL = (
    "\nSELECT entity_type, COUNT(DISTINCT event_id) AS reads\n"
    f"FROM {LINEAGE}\n"
    f"WHERE source_table_full_name = '{TABLE}'\n"
    "  AND event_date >= CURRENT_DATE - INTERVAL 30 DAYS\n"
    "GROUP BY 1\nORDER BY 2 DESC\n"
)
CI_MONTHLY_SQL = (
    "\nSELECT\n  DATE_FORMAT(signup_date, 'MMMM yyyy') AS month,\n"
    "  DATE_TRUNC('month', signup_date)       AS month_start,\n"
    "  COUNT(*)                                     AS signups\n"
    f"FROM {TABLE}\nWHERE signup_date >= '2026-07-01'\n"
    "  AND signup_date <  '2026-09-01'\nGROUP BY 1, 2\nORDER BY 2\n"
)
CI_DAILY_SANITY_CHECK_SQL = (
    "\nSELECT\n  signup_date,\n  COUNT(*) AS daily_signups\n"
    f"FROM {TABLE}\nWHERE signup_date BETWEEN '2026-07-01' AND '2026-07-07'\n"
    "GROUP BY 1\nORDER BY 1\n"
)


@pytest.fixture(autouse=True)
def _production_numeric_guard_defaults(monkeypatch):
    monkeypatch.delenv("NUMERIC_GUARD_MODE", raising=False)
    monkeypatch.delenv("NUMERIC_GUARD_MAX_REVISIONS", raising=False)


def _codes(result):
    return {violation["code"] for violation in result["violations"]}


def _severity(result, code):
    return {v["severity"] for v in result["violations"] if v["code"] == code}


def _event(name, tool_input, result, *, iteration=0, source="fixture"):
    return {
        "iteration": iteration,
        "name": name,
        "input": tool_input,
        "result": result,
        "is_error": bool(isinstance(result, dict) and "error" in result),
        "source": source,
        "model_visible": source != "runtime_preflight",
    }


def _trace(fixture, tool_events, final_text):
    return {
        "scenario": fixture["id"],
        "stop_reason": "end_turn",
        "final_text": final_text,
        "api_calls": len(tool_events) + 1,
        "output_tokens": 2000,
        "budget": {"max_api_calls": 12, "max_output_tokens": 30000},
        "tool_events": tool_events,
        "numeric_guard_revisions": 0,
    }


# --------------------------------------------------------------------- fixtures


def test_both_fixtures_load_and_validate_as_scenarios():
    names = ["signups_column_count", "signup_percent_change"]

    assert validate_scenarios(names) == names
    for name in names:
        fixture = load_fixture(name)
        assert fixture["id"] == name
        assert (FIXTURES_DIR / f"{name}.json").is_file()
        assert fixture["notes"], "each numeric fixture documents the production mistake"
        assert fixture["numeric_grounding"] is True


def test_every_fixture_describes_the_same_35_column_signups_table():
    columns = COLUMN_COUNT["objects"][TABLE]["columns"]

    assert len(columns) == 35
    assert len({column[0].lower() for column in columns}) == 35
    assert [column[0] for column in columns if column[3] is False] == [
        "_anonymousId",
        "signup_date",
        "inserted_at",
    ]
    # The loader has no include mechanism, so the list is duplicated per fixture
    # and must never drift: the three scenarios describe one production table.
    assert PERCENT_CHANGE["objects"][TABLE]["columns"] == columns
    assert EXPLANATION["objects"][TABLE]["columns"] == columns
    # SHOW CREATE TABLE derives from the shared list instead of a hand-written DDL.
    assert "create_sql" not in EXPLANATION["objects"][TABLE]
    assert COLUMN_COUNT["required_final_numbers"] == ["35"]
    assert COLUMN_COUNT["forbidden_final_numbers"] == ["36"]


def test_signups_fixture_models_the_columns_the_business_docs_name():
    """prompts/business/00_repo_context.md lists these signups columns by name.

    an earlier CI run's explanation answer flagged them as "not present in
    the live schema", a confusion the fixture should not create.
    """
    names = {column[0] for column in COLUMN_COUNT["objects"][TABLE]["columns"]}
    documented = {
        "_anonymousId",
        "signup_stitch_id",
        "signup_date",
        "dob",
        "age",
        "age_range",
        "device_type",
        "autofill_ind",
        "signup_flow",
        "loyalty_tier",
        "pre_order_ind",
        "cross_state_ship_ind",
        "shipping_distance",
        "audience",
        "identification_method",
    }

    assert documented <= names, sorted(documented - names)


def test_lineage_fixture_models_every_column_the_system_prompt_documents():
    """prompts/core/00_behavior.md tells the model which table_lineage columns to use."""
    columns = {column[0]: column[1] for column in EXPLANATION["objects"][LINEAGE]["columns"]}
    documented = {
        "source_table_full_name",
        "source_table_catalog",
        "source_table_schema",
        "target_table_full_name",
        "entity_type",
        "entity_id",
        "entity_run_id",
        "entity_metadata",
        "created_by",
        "event_time",
        "event_date",
        "event_id",
        "statement_id",
    }

    assert documented <= set(columns), sorted(documented - set(columns))
    assert columns["entity_metadata"].upper().startswith("STRUCT<")


def test_sql_run_guard_accepts_the_prompts_lineage_read_count_queries():
    """an earlier CI run rejected COUNT(DISTINCT event_id) as column_not_in_live_schema.

    The prompt mandates that expression for read counts, so the fixture's
    describe must satisfy the guard for it and for the prompt's other lineage
    patterns, and the dispatcher must then serve rows rather than a limitation.
    """
    tools = SyntheticTools(copy.deepcopy(EXPLANATION))
    guard = SqlRunGuard()
    describe = tools.dispatch("databricks_describe_object", LINEAGE_INPUT)
    guard.after_call("databricks_describe_object", LINEAGE_INPUT, describe)
    by_user = (
        f"SELECT created_by, COUNT(DISTINCT event_id) AS reads FROM {LINEAGE} "
        f"WHERE source_table_full_name = '{TABLE}' "
        "AND event_date >= CURRENT_DATE - INTERVAL 30 DAYS GROUP BY 1 ORDER BY 2 DESC"
    )
    struct_access = (
        f"SELECT entity_metadata.job_info.job_id AS job_id FROM {LINEAGE} "
        f"WHERE target_table_full_name = '{TABLE}'"
    )

    assert describe["column_count"] == len(EXPLANATION["objects"][LINEAGE]["columns"])
    for sql in (CI_LINEAGE_READS_SQL, by_user, struct_access):
        assert guard.before_call("databricks_execute_sql", {"sql": sql}) is None, sql

    reads = tools.dispatch("databricks_execute_sql", {"sql": CI_LINEAGE_READS_SQL})
    assert reads["columns"] == ["entity_type", "reads"]
    assert reads["sample_rows"] == [["NOTEBOOK", 4]]
    assert "synthetic_limitation" not in reads
    assert tools.dispatch("databricks_execute_sql", {"sql": by_user})["sample_rows"] == [
        ["sp-signups-merge", 4]
    ]


def test_synthetic_describe_exposes_column_count_and_nullability_not_a_row_count():
    tools = SyntheticTools(copy.deepcopy(COLUMN_COUNT))

    result = tools.dispatch("databricks_describe_object", DESCRIBE_INPUT)

    assert result["column_count"] == 35 == len(result["columns"])
    assert "row_count" not in result and "sample_rows" not in result
    assert [c["name"] for c in result["columns"] if c["is_nullable"] is False] == [
        "_anonymousId",
        "signup_date",
        "inserted_at",
    ]
    assert sum(1 for c in result["columns"] if c["is_nullable"] is True) == 32
    # The production bug: 35 column rows + 1 table row = 36 in the raw result set.
    assert "36" not in json.dumps(result)


def test_information_schema_columns_carry_the_fixture_nullable_flag():
    tools = SyntheticTools(copy.deepcopy(COLUMN_COUNT))

    result = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT column_name, is_nullable FROM acme_prod.information_schema.columns "
                "WHERE table_schema = 'reporting' AND table_name = 'signups'"
            )
        },
    )

    assert result["row_count"] == 35
    assert result["columns"] == ["column_name", "data_type", "comment", "is_nullable"]
    assert {row[0]: row[3] for row in result["sample_rows"]}["_anonymousId"] == "NO"
    assert {row[0]: row[3] for row in result["sample_rows"]}["shipping_zip"] == "YES"


def test_fixture_without_github_source_or_create_sql_still_dispatches():
    tools = SyntheticTools(copy.deepcopy(COLUMN_COUNT))

    search = tools.dispatch("github_search_code", {"query": "signups merge"})
    find = tools.dispatch("github_find_files", {"name": "signups"})
    read = tools.dispatch("github_get_file_content", {"path": "notebooks/x.sql"})
    ddl = tools.dispatch("databricks_execute_sql", {"sql": f"SHOW CREATE TABLE {TABLE}"})

    assert search == {"total_count": 0, "items": []}
    assert find["match_count"] == 0
    assert read["fixture_failure"] == "no_github_source"
    statement = ddl["sample_rows"][0][0]
    assert statement.startswith(f"CREATE TABLE {TABLE} (")
    assert "_anonymousId STRING NOT NULL" in statement
    assert "shipping_zip STRING," in statement


def test_reading_a_prompt_documented_repo_file_is_a_limitation_not_an_avoidable_error():
    """an earlier CI run read docs/runbooks/all-jobs.md, as the
    job-trigger workflow instructs once resources/jobs/ searches come back
    empty. The fixture does not model that file; that is a fixture gap, not a
    model mistake, so it must not score as ``avoidable_tool_failure``."""
    tools = SyntheticTools(copy.deepcopy(EXPLANATION))

    for path in ("docs/runbooks/all-jobs.md", "resources/jobs/signups_merge.job.yml"):
        read = tools.dispatch("github_get_file_content", {"path": path})
        assert "error" not in read and "fixture_failure" not in read, read
        assert read["synthetic_limitation"] == "file_not_modeled"
        assert read["path"] == path and read["content"] is None
        assert "Do not infer that it is missing" in read["result_scope"]

    events = [
        _event(
            "github_get_file_content",
            {"path": "docs/runbooks/all-jobs.md"},
            tools.dispatch("github_get_file_content", {"path": "docs/runbooks/all-jobs.md"}),
        )
    ]
    result = score_trace(_trace(EXPLANATION, events, "Some answer."), EXPLANATION)

    assert "avoidable_tool_failure" not in _codes(result)
    assert result["avoidable_tool_failures"] == 0
    assert result["metrics"]["synthetic_limitations"] == 1


@pytest.mark.parametrize(
    "path",
    [
        "notebooks/Data Team/Gold Table Creation/Signups MA - Merge.sql",
        "notebooks/guessed/Signups.sql",
        "src/signups.py",
    ],
)
def test_reading_a_stale_or_guessed_notebook_path_is_still_wrong_github_source(path):
    tools = SyntheticTools(copy.deepcopy(EXPLANATION))

    read = tools.dispatch("github_get_file_content", {"path": path})

    assert read["fixture_failure"] == "wrong_github_source"
    assert "error" in read


# ------------------------------------------------------- fixture SQL patterns


def test_sql_results_patterns_compile_and_rows_match_declared_width():
    for entry in PERCENT_CHANGE["sql_results"]:
        re.compile(entry["pattern"], re.DOTALL)
        if entry.get("unmodeled"):
            assert not entry.get("rows"), entry["description"]
            continue
        assert entry["rows"], entry["description"]
        assert all(len(row) == len(entry["columns"]) for row in entry["rows"]), entry
    # The honest-limitation catch-all is last so every month shape wins first.
    assert PERCENT_CHANGE["sql_results"][-1]["unmodeled"] is True


def test_ci_monthly_sql_with_mmmm_yyyy_format_is_served_the_monthly_rows():
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))

    result = tools.dispatch("databricks_execute_sql", {"sql": CI_MONTHLY_SQL})

    assert result["sample_rows"] == [["2026-07", JULY], ["2026-08", AUGUST]]
    assert "synthetic_limitation" not in result


@pytest.mark.parametrize(
    ("label", "sql", "columns"),
    [
        # The exact sanity check from an earlier CI run: it was served the two
        # monthly rows relabelled as daily rows, and the model then refused to
        # report the percent because the tool looked like placeholder data.
        (
            "ci_daily_sanity_check",
            CI_DAILY_SANITY_CHECK_SQL,
            ["signup_date", "daily_signups"],
        ),
        (
            "daily_date_format",
            f"SELECT DATE_FORMAT(signup_date, 'yyyy-MM-dd') AS d, COUNT(*) AS n FROM {TABLE} "
            "WHERE signup_date >= '2026-07-01' AND signup_date < '2026-09-01' GROUP BY 1",
            ["d", "n"],
        ),
        (
            "weekly_date_trunc",
            f"SELECT DATE_TRUNC('week', signup_date) AS wk, COUNT(*) AS n FROM {TABLE} "
            "WHERE signup_date >= '2026-07-01' GROUP BY 1 ORDER BY 1",
            ["wk", "n"],
        ),
        (
            "sub_month_range",
            f"SELECT COUNT(*) AS n FROM {TABLE} "
            "WHERE signup_date BETWEEN '2026-07-01' AND '2026-07-07'",
            ["n"],
        ),
        (
            "single_day",
            f"SELECT COUNT(*) AS n FROM {TABLE} WHERE signup_date = '2026-07-01'",
            ["n"],
        ),
    ],
)
def test_day_and_week_grains_get_an_honest_limitation_not_the_monthly_rows(label, sql, columns):
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))

    result = tools.dispatch("databricks_execute_sql", {"sql": sql})

    assert result["synthetic_limitation"] == "projection_not_modeled", label
    assert result["row_count"] is None and result["sample_rows"] == [], label
    assert result["columns"] == columns, label
    assert "No production data absence or zero value may be inferred" in result["result_scope"]
    assert str(JULY) not in json.dumps(result) and str(AUGUST) not in json.dumps(result), label


def test_date_scoped_coverage_checks_without_counts_still_reach_the_fallback():
    """The passing an earlier CI run checked August coverage this way; keep it answerable."""
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))

    result = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                f"SELECT MAX(signup_date) AS latest_date FROM {TABLE} "
                "WHERE signup_date >= '2026-08-01'"
            )
        },
    )

    assert result["columns"] == ["latest_date"]
    assert result["sample_rows"] == [["2026-08-27"]]
    assert "synthetic_limitation" not in result


@pytest.mark.parametrize(
    ("label", "sql", "expected_rows"),
    [
        (
            "grouped_by_month",
            f"SELECT DATE_FORMAT(signup_date, 'yyyy-MM') AS month, COUNT(*) AS signups "
            f"FROM {TABLE} WHERE signup_date >= '2026-07-01' "
            "AND signup_date < '2026-09-01' GROUP BY 1 ORDER BY 1",
            [["2026-07", JULY], ["2026-08", AUGUST]],
        ),
        (
            "grouped_by_date_trunc",
            f"SELECT date_trunc('month', signup_date) AS m, COUNT(*) AS c FROM {TABLE} "
            "WHERE signup_date BETWEEN '2026-07-01' AND '2026-08-31' GROUP BY 1",
            [["2026-07", JULY], ["2026-08", AUGUST]],
        ),
        (
            "conditional_pair",
            "SELECT COUNT(CASE WHEN signup_date BETWEEN '2026-07-01' AND '2026-07-31' "
            "THEN 1 END) AS july, COUNT(CASE WHEN signup_date BETWEEN '2026-08-01' "
            f"AND '2026-08-31' THEN 1 END) AS aug FROM {TABLE}",
            [[JULY, AUGUST]],
        ),
        (
            "count_if_pair",
            "SELECT COUNT_IF(signup_date >= '2026-07-01' AND signup_date < "
            "'2026-08-01') AS jul, COUNT_IF(signup_date >= '2026-08-01' AND "
            f"signup_date < '2026-09-01') AS aug FROM {TABLE}",
            [[JULY, AUGUST]],
        ),
        (
            "july_half_open",
            f"SELECT COUNT(*) AS signups FROM {TABLE} WHERE signup_date >= "
            "'2026-07-01' AND signup_date < '2026-08-01'",
            [[JULY]],
        ),
        (
            "july_between",
            f"SELECT COUNT(*) FROM {TABLE} WHERE signup_date BETWEEN '2026-07-01' "
            "AND '2026-07-31'",
            [[JULY]],
        ),
        (
            "august_half_open",
            f"SELECT COUNT(*) AS n FROM {TABLE} WHERE signup_date >= '2026-08-01' "
            "AND signup_date < '2026-09-01'",
            [[AUGUST]],
        ),
        (
            "august_month_function",
            f"SELECT COUNT(*) AS n FROM {TABLE} WHERE YEAR(signup_date) = 2026 "
            "AND MONTH(signup_date) = 8",
            [[AUGUST]],
        ),
        (
            "two_month_total",
            f"SELECT COUNT(*) AS n FROM {TABLE} WHERE signup_date >= '2026-07-01' "
            "AND signup_date < '2026-09-01'",
            [[JULY + AUGUST]],
        ),
    ],
)
def test_fixture_sql_patterns_serve_deterministic_monthly_rows(label, sql, expected_rows):
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))

    result = tools.dispatch("databricks_execute_sql", {"sql": sql})

    assert "error" not in result, label
    assert result["sample_rows"] == expected_rows, label
    assert result["row_count"] == len(expected_rows)
    assert "synthetic_limitation" not in result


def test_fixture_sql_pattern_uses_model_aliases_when_width_matches():
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))

    aliased = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                f"SELECT date_trunc('month', signup_date) AS reg_month, COUNT(*) AS n "
                f"FROM {TABLE} WHERE signup_date >= '2026-07-01' GROUP BY 1"
            )
        },
    )
    unaliased_star = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                f"SELECT COUNT(*) FROM {TABLE} WHERE signup_date BETWEEN "
                "'2026-08-01' AND '2026-08-31'"
            )
        },
    )

    assert aliased["columns"] == ["reg_month", "n"]
    assert unaliased_star["columns"] == ["signups"]  # declared name, never "*"


def test_percent_change_computed_in_sql_is_served_with_the_exact_value():
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))

    result = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "WITH m AS (SELECT DATE_FORMAT(signup_date, 'yyyy-MM') AS month, "
                f"COUNT(*) AS signups FROM {TABLE} WHERE signup_date >= "
                "'2026-07-01' AND signup_date < '2026-09-01' GROUP BY 1) "
                "SELECT month, signups, LAG(signups) OVER (ORDER BY month) AS prev, "
                "(signups - LAG(signups) OVER (ORDER BY month)) / "
                "LAG(signups) OVER (ORDER BY month) * 100 AS pct_change FROM m"
            )
        },
    )

    assert result["columns"] == ["month", "signups", "prev", "pct_change"]
    assert result["sample_rows"][1] == ["2026-08", AUGUST, JULY, EXACT_PCT]


def test_unmatched_sql_falls_back_to_scenario_code_and_first_pattern_wins():
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))

    fallback = tools.dispatch(
        "databricks_execute_sql",
        {"sql": f"SELECT MIN(signup_date), MAX(signup_date) FROM {TABLE}"},
    )
    metadata = tools.dispatch(
        "databricks_execute_sql",
        {
            "sql": (
                "SELECT COUNT(*) FROM acme_prod.information_schema.columns "
                "WHERE table_name = 'signups' GROUP BY table_name"
            )
        },
    )

    assert fallback["sample_rows"] == [["2024-01-02", "2026-08-27"]]
    # information_schema is excluded from every pattern, so the metadata branch answers.
    assert metadata["columns"] == ["column_name", "data_type", "comment", "is_nullable"]

    ordered = copy.deepcopy(PERCENT_CHANGE)
    ordered["sql_results"] = [
        {"pattern": "count", "columns": ["first"], "rows": [[1]]},
        {"pattern": "count", "columns": ["second"], "rows": [[2]]},
    ]
    first = SyntheticTools(ordered).dispatch(
        "databricks_execute_sql", {"sql": f"SELECT COUNT(*) AS c FROM {TABLE}"}
    )
    assert first["sample_rows"] == [[1]]


# ------------------------------------------------------------------ scoring A


def _describe_event(fixture):
    result = SyntheticTools(copy.deepcopy(fixture)).dispatch(
        "databricks_describe_object", DESCRIBE_INPUT
    )
    return _event("databricks_describe_object", dict(DESCRIBE_INPUT), result)


def test_column_count_fixture_passes_a_grounded_35_column_answer():
    final = (
        "`acme_prod.reporting.signups` has 35 columns (`column_count` from "
        "`databricks_describe_object`). All of them are nullable except `_anonymousId`, "
        "`signup_date`, and `inserted_at`, which are NOT NULL."
    )

    result = score_trace(_trace(COLUMN_COUNT, [_describe_event(COLUMN_COUNT)], final), COLUMN_COUNT)

    assert result["violations"] == []
    assert result["passed"] is True
    assert result["metrics"]["numeric_guard_revisions"] == 0


def test_column_count_fixture_catches_the_36_column_row_count_echo():
    final = (
        "The table has 36 columns. The nullable ones are every column except _anonymousId, "
        "signup_date and inserted_at."
    )

    result = score_trace(_trace(COLUMN_COUNT, [_describe_event(COLUMN_COUNT)], final), COLUMN_COUNT)

    assert {"required_number_missing", "forbidden_number_present", "ungrounded_number"} <= _codes(
        result
    )
    assert _severity(result, "ungrounded_number") == {"hard"}
    assert result["passed"] is False
    assert "36" in next(
        v["message"] for v in result["violations"] if v["code"] == "ungrounded_number"
    )


def test_column_count_fixture_reports_the_nullable_coverage_pattern_softly():
    """``(?i)nullable`` is coverage; the behavioral checks (35 stated, 36 absent,
    every figure grounded) are what gate this scenario and stay hard."""
    assert "missing_answer_fact" not in COLUMN_COUNT.get("violation_severity", {})
    final = (
        "`acme_prod.reporting.signups` has 35 columns (`column_count`). Only "
        "`_anonymousId`, `signup_date` and `inserted_at` are NOT NULL."
    )

    result = score_trace(_trace(COLUMN_COUNT, [_describe_event(COLUMN_COUNT)], final), COLUMN_COUNT)

    assert _codes(result) == {"missing_answer_fact"}
    assert _severity(result, "missing_answer_fact") == {"soft"}
    assert result["passed"] is True
    assert result["score"] == 0.95


def test_column_count_fixture_rejects_all_columns_nullable_and_missing_describe():
    all_nullable = "There are 35 columns and all 35 columns are nullable."
    result = score_trace(
        _trace(COLUMN_COUNT, [_describe_event(COLUMN_COUNT)], all_nullable), COLUMN_COUNT
    )
    assert "unsupported_answer_claim" in _codes(result)

    no_tools = score_trace(_trace(COLUMN_COUNT, [], "35 columns, all nullable."), COLUMN_COUNT)
    assert {"missing_tool_evidence", "ungrounded_number"} <= _codes(no_tools)


# ------------------------------------------------------------------ scoring B


def _grouped_sql_event(iteration=1):
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))
    sql = (
        f"SELECT DATE_FORMAT(signup_date, 'yyyy-MM') AS month, COUNT(*) AS signups "
        f"FROM {TABLE} WHERE signup_date >= '2026-07-01' AND signup_date < "
        "'2026-09-01' GROUP BY 1 ORDER BY 1"
    )
    return _event(
        "databricks_execute_sql",
        {"sql": sql},
        tools.dispatch("databricks_execute_sql", {"sql": sql}),
        iteration=iteration,
    )


def _calculate_event(iteration=2):
    tool_input = {
        "expression": "pct_change(old, new)",
        "variables": {"old": JULY, "new": AUGUST},
    }
    return _event("calculate", tool_input, calc_tool.dispatch("calculate", tool_input), iteration=iteration)


GOOD_FINAL = (
    "July 2026: 12,480 signups; August 2026: 13,977 signups (from `COUNT(*)` by "
    "month). That is a +11.995192% change (via `calculate`)."
)


def test_percent_change_fixture_passes_sql_plus_calculate():
    events = [
        _describe_event(PERCENT_CHANGE),
        _grouped_sql_event(),
        _calculate_event(),
    ]
    events[0]["source"] = "runtime_preflight"
    events[0]["model_visible"] = False
    events[0]["iteration"] = 1

    result = score_trace(_trace(PERCENT_CHANGE, events, GOOD_FINAL), PERCENT_CHANGE)

    assert result["violations"] == []
    assert result["passed"] is True


def test_percent_change_fixture_promotes_the_percent_figure_pattern_to_hard():
    """Stating the percent change is the point of the scenario, so the fixture
    promotes ``missing_answer_fact`` to hard via ``violation_severity``; both
    monthly totals without a percent figure has not answered the question."""
    assert PERCENT_CHANGE["violation_severity"]["missing_answer_fact"] == "hard"
    assert PERCENT_CHANGE["required_final_patterns"] == ["\\d+(?:\\.\\d+)?\\s?(?:%|percent)"]
    assert "percent" in PERCENT_CHANGE["notes"] and "hard" in PERCENT_CHANGE["notes"]
    events = [_describe_event(PERCENT_CHANGE), _grouped_sql_event(), _calculate_event()]
    final = (
        "July 2026 had 12,480 signups and August 2026 had 13,977 signups "
        "(from `COUNT(*)` by month), so August was higher."
    )

    result = score_trace(_trace(PERCENT_CHANGE, events, final), PERCENT_CHANGE)

    assert _codes(result) == {"missing_answer_fact"}
    assert _severity(result, "missing_answer_fact") == {"hard"}
    assert result["passed"] is False
    assert result["hard_violations"] == 1
    assert result["score"] == 0.75

    # The same answer under the default (soft) severity would only lower the score.
    unpromoted = copy.deepcopy(PERCENT_CHANGE)
    del unpromoted["violation_severity"]["missing_answer_fact"]
    relaxed = score_trace(_trace(PERCENT_CHANGE, events, final), unpromoted)
    assert _severity(relaxed, "missing_answer_fact") == {"soft"}
    assert relaxed["passed"] is True


def test_percent_change_rounded_from_calculate_result_is_grounded():
    events = [_describe_event(PERCENT_CHANGE), _grouped_sql_event(), _calculate_event()]
    final = "July 2026 had 12,480 signups and August 2026 had 13,977, a 12.0% increase."

    result = score_trace(_trace(PERCENT_CHANGE, events, final), PERCENT_CHANGE)

    assert "ungrounded_number" not in _codes(result)
    assert result["passed"] is True


def test_percent_change_computed_in_sql_is_acceptable_with_soft_penalty():
    tools = SyntheticTools(copy.deepcopy(PERCENT_CHANGE))
    sql = (
        "WITH m AS (SELECT DATE_FORMAT(signup_date, 'yyyy-MM') AS month, COUNT(*) AS n "
        f"FROM {TABLE} WHERE signup_date >= '2026-07-01' GROUP BY 1) "
        "SELECT month, n, LAG(n) OVER (ORDER BY month) AS prev, "
        "(n - LAG(n) OVER (ORDER BY month)) / LAG(n) OVER (ORDER BY month) * 100 AS pct FROM m"
    )
    events = [
        _describe_event(PERCENT_CHANGE),
        _event("databricks_execute_sql", {"sql": sql}, tools.dispatch("databricks_execute_sql", {"sql": sql}), iteration=1),
    ]
    final = "12,480 in July 2026 vs 13,977 in August 2026: a 11.995192% increase (computed in SQL)."

    result = score_trace(_trace(PERCENT_CHANGE, events, final), PERCENT_CHANGE)

    assert _codes(result) == {"required_tool_missing"}
    assert _severity(result, "required_tool_missing") == {"soft"}
    assert result["passed"] is True
    assert result["score"] == 0.95


@pytest.mark.parametrize(
    ("final", "expected_hard"),
    [
        # Without a calculate call no tool returned ~12, so the hedge is an
        # in-head estimate: the approximation check fails it explicitly.
        (
            "July 2026: 12,480. August 2026: 13,977. Roughly 12% growth month over month.",
            {"ungrounded_approximation"},
        ),
        (
            "Signups rose ~12% between July and August 2026 (12,480 -> 13,977).",
            {"ungrounded_approximation"},
        ),
        # Truncated or mis-rounded in-head figures match no tool value.
        (
            "July 2026: 12,480. August 2026: 13,977. That is an 11.99% increase.",
            {"ungrounded_number"},
        ),
        (
            "July 2026: 12,480. August 2026: 13,977. That is an 11.9% increase.",
            {"ungrounded_number"},
        ),
    ],
)
def test_percent_change_fixture_catches_in_head_arithmetic(final, expected_hard):
    events = [_describe_event(PERCENT_CHANGE), _grouped_sql_event()]

    result = score_trace(_trace(PERCENT_CHANGE, events, final), PERCENT_CHANGE)

    hard = {v["code"] for v in result["violations"] if v["severity"] == "hard"}
    assert expected_hard <= hard, result["violations"]
    assert _severity(result, "required_tool_missing") == {"soft"}
    assert result["passed"] is False


# an earlier CI run: every hedged figure rounds from a calculate
# result (28.587... -> "28.6", 402.58 -> "403", 517.67 -> "518"; "~29%" rounds
# from 28.587 at zero decimals), yet the blanket "~N%" regex failed the run.
CI_HEDGED_FINAL = (
    "*Month totals (raw):* +12.0% (via `calculate`) - not directly comparable since "
    "August has 4 fewer days. *Daily average:* July ~403/day -> August ~518/day = "
    "*+28.6%* (via `calculate`). Signups on Thu 7/2 were down ~29% vs prior "
    "Thursdays. Bottom line: August is nominally up ~28.6% per day vs. July "
    "(12,480 -> 13,977)."
)


def _ci_hedged_events():
    def calc(expression, precision):
        tool_input = {"expression": expression, "precision": precision}
        return _event("calculate", tool_input, calc_tool.dispatch("calculate", tool_input), iteration=2)

    return [
        _describe_event(PERCENT_CHANGE),
        _grouped_sql_event(),
        _calculate_event(),
        calc("pct_change(12480 / 31, 13977 / 27)", 1),
        calc("12480 / 31", 0),
        calc("13977 / 27", 0),
    ]


def test_percent_change_hedged_figures_that_round_from_calculate_results_pass():
    """``~28.6%`` for a tool value of 28.587... is presentation, not arithmetic."""
    result = score_trace(_trace(PERCENT_CHANGE, _ci_hedged_events(), CI_HEDGED_FINAL), PERCENT_CHANGE)

    assert result["violations"] == [], result["violations"]
    assert result["passed"] is True


def test_percent_change_hedged_figures_without_tool_support_fail_hard():
    """The same wording with the calculate calls removed is an in-head estimate."""
    events = [_describe_event(PERCENT_CHANGE), _grouped_sql_event()]

    result = score_trace(_trace(PERCENT_CHANGE, events, CI_HEDGED_FINAL), PERCENT_CHANGE)

    assert _severity(result, "ungrounded_approximation") == {"hard"}
    (message,) = [v["message"] for v in result["violations"] if v["code"] == "ungrounded_approximation"]
    assert "~29%" in message and "~28.6%" in message and "~403" in message and "~518" in message
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("final", "flagged"),
    [
        ("Signups rose ~12% (via calculate).", []),
        ("Roughly 12 percent growth.", []),
        ("~12.0% growth.", []),
        ("Signups rose ~11.9% between the months.", ["~11.9%"]),
        ("Approximately 15% higher.", ["Approximately 15%"]),
        ("about 1.2k more signups", ["about 1.2k"]),
        # Non-claims are never approximations: small integers, strikethrough.
        ("Only ~2 days of overlap; ~ignore~ this.", []),
    ],
)
def test_ungrounded_approximations_judge_the_hedged_figure_against_tool_evidence(final, flagged):
    from evals.scoring import ungrounded_approximations
    from utils.numeric_guard import NumericEvidenceGuard

    guard = NumericEvidenceGuard()
    guard.record_tool_result("calculate", _calculate_event()["input"], _calculate_event()["result"])

    assert ungrounded_approximations(guard, final) == flagged


def test_percent_change_fixture_relies_on_numeric_grounding_not_a_blanket_hedge_regex():
    assert PERCENT_CHANGE["numeric_grounding"] is True
    for pattern in PERCENT_CHANGE["forbidden_final_patterns"]:
        assert "~" not in pattern and "approximately" not in pattern, pattern


# ------------------------------------------------------------- runner end-to-end


class _Block(SimpleNamespace):
    def model_dump(self):
        if self.type == "tool_use":
            return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}
        return {"type": "text", "text": self.text}


def _usage():
    return SimpleNamespace(
        input_tokens=1000, cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=50
    )


class _ScriptedClient:
    """Replays scripted responses: ``("tool", name, input)`` or ``("text", text)``."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.messages = self
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        kind, *payload = self.steps.pop(0)
        if kind == "tool":
            name, tool_input = payload
            block = _Block(type="tool_use", id=f"call-{len(self.requests)}", name=name, input=tool_input)
            return SimpleNamespace(content=[block], stop_reason="tool_use", model="m", usage=_usage())
        return SimpleNamespace(
            content=[_Block(type="text", text=payload[0])], stop_reason="end_turn", model="m", usage=_usage()
        )


def test_runner_end_to_end_percent_change_scenario_passes_with_calculate():
    client = _ScriptedClient(
        [
            ("tool", "databricks_execute_sql", {"sql": _grouped_sql_event()["input"]["sql"]}),
            ("tool", "calculate", _calculate_event()["input"]),
            ("text", GOOD_FINAL),
        ]
    )

    trace = run_conversation(copy.deepcopy(PERCENT_CHANGE), client=client)
    result = score_trace(trace, PERCENT_CHANGE)

    assert [(e["name"], e["source"]) for e in trace["tool_events"]] == [
        ("databricks_describe_object", "runtime_preflight"),
        ("databricks_execute_sql", "fixture"),
        ("calculate", "fixture"),
    ]
    assert trace["tool_events"][1]["result"]["sample_rows"] == [["2026-07", JULY], ["2026-08", AUGUST]]
    assert trace["numeric_guard_revisions"] == 0
    assert result["passed"] is True and result["violations"] == []


def test_runner_end_to_end_column_count_scenario_passes_with_35():
    client = _ScriptedClient(
        [
            ("tool", "databricks_describe_object", dict(DESCRIBE_INPUT)),
            (
                "text",
                "The table has 35 columns (column_count); all but _anonymousId, "
                "signup_date and inserted_at are nullable.",
            ),
        ]
    )

    trace = run_conversation(copy.deepcopy(COLUMN_COUNT), client=client)
    result = score_trace(trace, COLUMN_COUNT)

    assert trace["api_calls"] == 2
    assert trace["tool_events"][0]["result"]["column_count"] == 35
    assert trace["numeric_guard_revisions"] == 0
    assert result["passed"] is True and result["violations"] == []


def test_runner_end_to_end_column_count_bug_fails_scoring_even_if_runtime_guard_misses_it():
    """The live "36 columns" answer, replayed through the production-shaped loop.

    The runtime numeric guard grounds the system prompt too, and that prompt
    happens to mention a "~-36%" shift, so production did not revise the
    figure. The scorer grounds only the user prompt and tool evidence, so the
    trace fails regardless of whether the runtime guard asked for a revision.
    """
    bug_answer = (
        "The table has 36 columns; all but _anonymousId, signup_date and "
        "inserted_at are nullable."
    )
    client = _ScriptedClient(
        [
            ("tool", "databricks_describe_object", dict(DESCRIBE_INPUT)),
            ("text", bug_answer),
            ("text", bug_answer),  # only consumed if the runtime guard asks for a revision
        ]
    )

    trace = run_conversation(copy.deepcopy(COLUMN_COUNT), client=client)
    result = score_trace(trace, COLUMN_COUNT)

    assert trace["numeric_guard_revisions"] in (0, 1)
    assert "36 columns" in trace["final_text"]
    assert {"forbidden_number_present", "required_number_missing", "ungrounded_number"} <= _codes(
        result
    )
    assert _severity(result, "ungrounded_number") == {"hard"}
    assert result["passed"] is False
    assert result["metrics"]["numeric_guard_revisions"] == trace["numeric_guard_revisions"]
