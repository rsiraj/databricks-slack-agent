"""Baseline-relative regression rule, the committed baseline, and the CLI round trip."""

import json
import math
from pathlib import Path

import pytest

from evals import regression
from evals import run as eval_run
from evals.regression import (
    EXIT_ERRORED,
    EXIT_PASS,
    EXIT_REGRESSION,
    NEW_CODE_MIN_FRACTION,
    NEW_CODE_MIN_REPEATS,
    BaselineError,
    build_baseline,
    entry_wide_codes,
    evaluate_regression,
    new_code_recurs,
    render_verdict_markdown,
    scenario_stats,
    tolerance_margin,
    validate_baseline,
)
from evals.synthetic_tools import FIXTURES_DIR
from test_eval_run_matrix import stubbed_runner  # noqa: F401 - pytest fixture reused here

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "evals" / "baseline.json"
CELL = {"provider": "anthropic", "model": "claude-sonnet-4-6", "effort": "high"}
CELL_KEY = "anthropic/claude-sonnet-4-6/high"
SONNET_5 = {"provider": "anthropic", "model": "claude-sonnet-5", "effort": "high"}
SONNET_5_KEY = "anthropic/claude-sonnet-5/high"


def _run(scenario, rep, *, passed=True, codes=(), status="scored", error_type=None, score=None, **cell):
    """A run record shaped like ``evals.run.run_one`` output."""
    record = {"scenario": scenario, "rep": rep, "status": status, **CELL, **cell}
    if status == "errored":
        record.update({"error_type": error_type or "APITimeoutError", "result": None, "trace": {}})
        return record
    violations = [{"code": code, "message": code, "severity": "hard"} for code in codes]
    record["result"] = {
        "scenario": scenario,
        "passed": passed and not violations,
        "score": score if score is not None else (1.0 if passed and not violations else 0.75),
        "violations": violations,
    }
    record["trace"] = {"scenario": scenario}
    return record


def _runs(scenario, outcomes, **cell):
    """``outcomes`` is a list of ``True`` (pass), a code string (fail with that code), or ``"ERR"``."""
    runs = []
    for rep, outcome in enumerate(outcomes, start=1):
        if outcome is True:
            runs.append(_run(scenario, rep, **cell))
        elif outcome == "ERR":
            runs.append(_run(scenario, rep, status="errored", **cell))
        else:
            runs.append(_run(scenario, rep, codes=(outcome,), **cell))
    return runs


def _entry(scenarios, *, cell=CELL, urls=("https://example.test/run/1",)):
    """One ``baselines`` entry; ``scenarios`` maps name -> (passed, repeats, {code: count})."""
    return {
        **cell,
        "recorded_at": "2026-09-05T01:47:44+00:00",
        "source_run_urls": list(urls),
        "scenarios": {
            name: {
                "repeats": repeats,
                "passed": passed,
                "pass_rate": round(passed / repeats, 4),
                "mean_score": 1.0,
                "hard_violation_codes": codes,
            }
            for name, (passed, repeats, codes) in scenarios.items()
        },
    }


def _baseline(**scenarios):
    """A baseline document with a single entry for ``CELL``."""
    return {"schema_version": 2, "baselines": {CELL_KEY: _entry(scenarios)}}


# --- margin arithmetic -------------------------------------------------------


def test_margin_is_one_third_for_three_repeats_against_a_three_repeat_baseline():
    # max(1/3, 1/sqrt(9)) = 1/3: a 3/3 baseline tolerates 2/3 and fails on 1/3.
    assert tolerance_margin(3, 3) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "current, baseline, expected",
    [
        (1, 3, 1.0),  # 1/N dominates: one sample can never fail the gate
        (3, 3, 1 / 3),
        (3, 9, max(1 / 3, 1 / math.sqrt(27))),  # 1/N still dominates
        (10, 10, 1 / 10),  # 1/sqrt(100) == 1/10
        (5, 20, max(1 / 5, 1 / math.sqrt(100))),  # 0.2 vs 0.1 -> 1/N
        (20, 5, max(1 / 20, 1 / math.sqrt(100))),  # 0.05 vs 0.1 -> standard error
    ],
)
def test_margin_formula(current, baseline, expected):
    assert tolerance_margin(current, baseline) == pytest.approx(expected)


def test_margin_rejects_non_positive_repeats():
    with pytest.raises(ValueError):
        tolerance_margin(0, 3)
    with pytest.raises(ValueError):
        tolerance_margin(3, 0)


# --- the rule ----------------------------------------------------------------


def test_drop_within_tolerance_passes_even_with_failed_conversations():
    baseline = _baseline(s=(3, 3, {}))
    stats = scenario_stats(_runs("s", [True, True, "avoidable_tool_failure"]))
    # A code that the baseline never recorded would fail on its own; give the
    # baseline that code so only the pass-rate rule is exercised.
    baseline["baselines"][CELL_KEY]["scenarios"]["s"]["hard_violation_codes"] = {"avoidable_tool_failure": 1}

    verdict = evaluate_regression(baseline, stats)

    assert verdict["status"] == "pass" and verdict["exit_code"] == EXIT_PASS
    (row,) = verdict["rows"]
    assert row["verdict"] == "pass"
    assert row["threshold"] == pytest.approx(0.6667, abs=1e-4)
    assert row["current_pass_rate"] == pytest.approx(0.6667, abs=1e-4)


def test_drop_beyond_margin_is_a_regression():
    baseline = _baseline(s=(3, 3, {"avoidable_tool_failure": 1}))
    stats = scenario_stats(_runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"]))

    verdict = evaluate_regression(baseline, stats)

    assert verdict["status"] == "regression" and verdict["exit_code"] == EXIT_REGRESSION
    (row,) = verdict["rows"]
    assert row["verdict"] == "regression"
    assert "pass rate 1/3 is below 0.667" in row["reason"]
    assert row["new_codes"] == []


def test_two_thirds_baseline_tolerates_one_third_but_not_zero():
    baseline = _baseline(s=(2, 3, {"avoidable_tool_failure": 1}))

    ok = evaluate_regression(baseline, scenario_stats(_runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"])))
    bad = evaluate_regression(baseline, scenario_stats(_runs("s", ["avoidable_tool_failure"] * 3)))

    assert ok["status"] == "pass"
    assert bad["status"] == "regression"


def test_one_third_baseline_cannot_regress_on_pass_rate_alone():
    # The percent-change scenario today: 1/3 in the baseline. 0/3 with the same
    # code is within tolerance; the gate only bites on a new kind of failure.
    baseline = _baseline(s=(1, 3, {"avoidable_tool_failure": 2}))

    verdict = evaluate_regression(baseline, scenario_stats(_runs("s", ["avoidable_tool_failure"] * 3)))

    assert verdict["status"] == "pass"
    assert verdict["rows"][0]["threshold"] == 0.0


def test_new_hard_violation_code_fails_even_when_pass_rate_is_within_tolerance():
    # A 1/3 baseline has threshold 0.000, so only the new-code rule can fire:
    # a code no scenario of the entry ever recorded, in two of three repeats.
    baseline = _baseline(s=(1, 3, {"avoidable_tool_failure": 2}))
    stats = scenario_stats(_runs("s", [True, "unsupported_answer_claim", "unsupported_answer_claim"]))

    verdict = evaluate_regression(baseline, stats)

    assert verdict["status"] == "regression"
    (row,) = verdict["rows"]
    assert row["new_codes"] == ["unsupported_answer_claim"]
    assert row["novel_codes"] == {"unsupported_answer_claim": 2}
    assert "never recorded in any scenario of the baseline entry recurred" in row["reason"]
    assert "`unsupported_answer_claim` in 2/3 repeats" in row["reason"]
    assert "pass rate" not in row["reason"]


def test_baseline_code_with_zero_count_still_counts_as_new():
    baseline = _baseline(s=(3, 3, {"unsupported_answer_claim": 0}))

    verdict = evaluate_regression(
        baseline, scenario_stats(_runs("s", [True, "unsupported_answer_claim", "unsupported_answer_claim"]))
    )

    assert verdict["rows"][0]["new_codes"] == ["unsupported_answer_claim"]
    assert entry_wide_codes(baseline["baselines"][CELL_KEY]) == set()


# --- the new-code rule: entry-wide, and only on recurrence -----------------------


def test_code_known_elsewhere_in_the_entry_is_not_new_for_this_scenario():
    """An earlier eval run, ``signup_percent_change``: its own baseline had no
    codes, but ``avoidable_tool_failure`` is recorded under ``signups_table_explanation``
    in the same entry. That is a known failure mode whose frequency the pass-rate
    margin governs, so it is listed under ``unseen_here`` and never fails the row."""
    baseline = _baseline(s=(6, 6, {}), t=(4, 6, {"avoidable_tool_failure": 2}))
    assert entry_wide_codes(baseline["baselines"][CELL_KEY]) == {"avoidable_tool_failure"}

    verdict = evaluate_regression(
        baseline,
        scenario_stats(_runs("s", [True, True, "avoidable_tool_failure"]) + _runs("t", [True] * 3)),
    )

    assert verdict["status"] == "pass"
    row = {r["scenario"]: r for r in verdict["rows"]}["s"]
    assert row["verdict"] == "pass"
    assert row["new_codes"] == [] and row["novel_codes"] == {}
    assert row["unseen_here"] == ["avoidable_tool_failure"]
    assert row["reason"].startswith("Informational:")
    assert "known failure mode elsewhere in the entry" in row["reason"]

    # Even recurring (2 of 3) it is never a *new-code* regression; the drop to 1/3
    # is caught by the pass-rate rule alone.
    worse = evaluate_regression(
        baseline,
        scenario_stats(
            _runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"]) + _runs("t", [True] * 3)
        ),
    )
    row = {r["scenario"]: r for r in worse["rows"]}["s"]
    assert row["verdict"] == "regression"
    assert row["new_codes"] == []
    assert "pass rate 1/3 is below 0.667" in row["reason"]
    assert "never recorded" not in row["reason"]


def test_entry_wide_novel_code_once_in_three_repeats_is_informational():
    """An earlier eval run, ``signups_table_explanation``: ``unsupported_answer_claim``
    once in three repeats against a 4/6 baseline that only ever had
    ``avoidable_tool_failure``. With R = 6 a ~10% failure mode has a ~53% chance of
    zero baseline observations, so one later sighting is not a verdict."""
    baseline = _baseline(s=(4, 6, {"avoidable_tool_failure": 2}))

    verdict = evaluate_regression(baseline, scenario_stats(_runs("s", [True, True, "unsupported_answer_claim"])))

    assert verdict["status"] == "pass" and verdict["exit_code"] == EXIT_PASS
    (row,) = verdict["rows"]
    assert row["verdict"] == "pass"
    assert row["new_codes"] == []
    assert row["novel_codes"] == {"unsupported_answer_claim": 1}
    assert row["unseen_here"] == ["unsupported_answer_claim"]
    assert row["reason"].startswith("Informational:")
    assert "`unsupported_answer_claim` is not in any scenario of the baseline entry" in row["reason"]
    assert f"below the {NEW_CODE_MIN_REPEATS}-repeat / {NEW_CODE_MIN_FRACTION:.0%} bar" in row["reason"]


def test_entry_wide_novel_code_in_two_of_three_repeats_is_a_regression():
    baseline = _baseline(s=(4, 6, {"avoidable_tool_failure": 2}))

    verdict = evaluate_regression(
        baseline, scenario_stats(_runs("s", [True, "unsupported_answer_claim", "unsupported_answer_claim"]))
    )

    assert verdict["status"] == "regression" and verdict["exit_code"] == EXIT_REGRESSION
    (row,) = verdict["rows"]
    # 1/3 is not below the 0.333 threshold: the new-code rule alone fails the row.
    assert row["threshold"] == pytest.approx(0.3333, abs=1e-4)
    assert row["new_codes"] == ["unsupported_answer_claim"]
    assert row["unseen_here"] == ["unsupported_answer_claim"]
    assert "recurred: `unsupported_answer_claim` in 2/3 repeats" in row["reason"]


def test_new_code_rule_counts_repeats_not_violations():
    # One conversation raising the novel code twice is still one repeat.
    twice = _run("s", 1, codes=("unsupported_answer_claim", "unsupported_answer_claim"))
    stats = scenario_stats([twice, _run("s", 2), _run("s", 3)])
    assert stats["s"]["hard_violation_codes"] == {"unsupported_answer_claim": 2}
    assert stats["s"]["hard_violation_code_runs"] == {"unsupported_answer_claim": 1}

    verdict = evaluate_regression(_baseline(s=(6, 6, {})), stats)

    assert verdict["status"] == "pass"
    assert verdict["rows"][0]["novel_codes"] == {"unsupported_answer_claim": 1}
    assert verdict["rows"][0]["new_codes"] == []


@pytest.mark.parametrize(
    "runs_with_code, scored, expected",
    [
        (0, 3, False),
        (1, 3, False),  # the case from an earlier run
        (2, 3, True),  # >= 2 repeats
        (3, 3, True),
        (1, 2, True),  # 50% of N = 2
        (1, 1, True),  # 100% of a single repeat
        (2, 6, True),  # >= 2 repeats even below 50%
        (1, 6, False),
        (3, 6, True),
    ],
)
def test_new_code_recurrence_bar(runs_with_code, scored, expected):
    assert new_code_recurs(runs_with_code, scored) is expected


def test_half_of_two_repeats_fires_the_new_code_rule():
    baseline = _baseline(s=(6, 6, {}))
    # N = 2: margin max(1/2, 1/sqrt(12)) = 0.5 -> threshold 0.5; 1/2 is not below it,
    # so only the >= 50% branch of the new-code rule can fail this row.
    verdict = evaluate_regression(baseline, scenario_stats(_runs("s", [True, "empty_answer"])))

    (row,) = verdict["rows"]
    assert row["threshold"] == pytest.approx(0.5, abs=1e-4)
    assert verdict["status"] == "regression"
    assert row["new_codes"] == ["empty_answer"]
    assert "in 1/2 repeats" in row["reason"]


def test_verdict_table_renders_unseen_here_without_failing():
    baseline = _baseline(s=(6, 6, {}), t=(4, 6, {"avoidable_tool_failure": 2}))
    stats = scenario_stats(
        _runs("s", [True, True, "avoidable_tool_failure"]) + _runs("t", [True, True, "unsupported_answer_claim"])
    )

    verdict = evaluate_regression(baseline, stats)
    markdown = render_verdict_markdown(verdict, baseline_path="evals/baseline.json")

    assert verdict["status"] == "pass"
    assert "## Conversation evaluation gate: PASS" in markdown
    assert "| Scenario | Baseline pass | Current pass | Threshold | New codes | Unseen here | Verdict |" in markdown
    assert "| `s` | 6/6 (100%) | 2/3 (67%) | 0.667 | - | `avoidable_tool_failure` 1/3 | **pass** |" in markdown
    assert "| `t` | 4/6 (67%) | 2/3 (67%) | 0.333 | - | `unsupported_answer_claim` 1/3 | **pass** |" in markdown
    assert "- `s`: Informational: unseen in this scenario's baseline but a known failure mode elsewhere" in markdown
    assert "- `t`: Informational: `unsupported_answer_claim` is not in any scenario of the baseline entry" in markdown
    assert "occurs in at least 2 of its N repeats (or in at least 50% of them)" in markdown


def test_rescore_runs_applies_current_scoring_to_full_traces_only():
    """``--from-results`` judges an artifact with *this* checkout's scoring rules:
    a stored ``result`` recorded when ``duplicate_sql`` was hard is re-scored soft.
    Stub traces (no ``tool_events``), unknown scenarios and errored records are untouched."""
    trace = {
        "stop_reason": "end_turn",
        "final_text": "Done.",
        "api_calls": 2,
        "hit_api_call_budget": False,
        "output_tokens": 10,
        "budget": {"max_api_calls": 25, "max_output_tokens": 400000},
        "tool_events": [
            {
                "iteration": 0,
                "name": "databricks_execute_sql",
                "input": {"sql": "SELECT 1"},
                "result": {"error": "already executed", "validation_failure": "duplicate_sql"},
                "is_error": True,
                "source": "guard",
            }
        ],
    }
    stale = {
        "scenario": "s", "status": "scored", **CELL, "trace": trace,
        "result": {"passed": False, "score": 0.75, "violations": [{"code": "avoidable_tool_failure", "severity": "hard"}]},
    }
    stub = _run("s", 2)
    stub_result = dict(stub["result"])
    other = {**stale, "scenario": "no_such_fixture", "trace": dict(trace)}
    errored = _run("s", 3, status="errored")

    count = eval_run.rescore_runs(
        [stale, stub, other, errored], fixture_loader=lambda name: {"id": name}, scenarios=["s"]
    )

    assert count == 1
    assert stale["result"]["passed"] is True
    (violation,) = stale["result"]["violations"]
    assert violation["code"] == "avoidable_tool_failure"
    assert violation["category"] == "duplicate_sql" and violation["severity"] == "soft"
    assert stub["result"] == stub_result
    assert other["result"]["passed"] is False
    assert errored["result"] is None


def test_soft_violations_are_not_codes_for_the_rule():
    baseline = _baseline(s=(3, 3, {}))
    run = _run("s", 1)
    run["result"]["violations"] = [{"code": "missing_answer_fact", "message": "x", "severity": "soft"}]
    run["result"]["score"] = 0.95

    verdict = evaluate_regression(baseline, scenario_stats([run, _run("s", 2), _run("s", 3)]))

    assert verdict["status"] == "pass"
    assert verdict["rows"][0]["new_codes"] == []


def test_infrastructure_error_makes_the_verdict_errored_with_exit_two():
    baseline = _baseline(s=(3, 3, {}), t=(3, 3, {}))
    stats = scenario_stats(_runs("s", [True, True, "ERR"]) + _runs("t", [True, True, True]))

    verdict = evaluate_regression(baseline, stats)

    assert verdict["status"] == "errored" and verdict["exit_code"] == EXIT_ERRORED == 2
    by_name = {row["scenario"]: row for row in verdict["rows"]}
    assert by_name["s"]["verdict"] == "errored"
    assert "APITimeoutError x1" in by_name["s"]["reason"]
    assert by_name["t"]["verdict"] == "pass"


def test_errored_verdict_reports_the_structured_error_code_status_and_message():
    """The verdict's infra reason uses the run record's structured ``error`` (class/code (status): message)."""
    baseline = _baseline(s=(3, 3, {}))
    error = {
        "error_type": "AuthenticationError",
        "status_code": 401,
        "code": "invalid_api_key",
        "type": "invalid_request_error",
        "message": "Incorrect API key provided: sk-….",
        "request_id": "req_abc123",
    }
    runs = [_run("s", 1)]
    for rep in (2, 3):
        errored = _run("s", rep, status="errored", error_type="AuthenticationError")
        errored["error"] = dict(error, request_id=f"{error['request_id']}_{rep}")
        runs.append(errored)

    stats = scenario_stats(runs)
    assert stats["s"]["infra_error_types"] == {"AuthenticationError/invalid_api_key (401)": 2}
    assert stats["s"]["infra_error_details"] == [
        "AuthenticationError/invalid_api_key (401): Incorrect API key provided: sk-…. [request_id=req_abc123_2]",
        "AuthenticationError/invalid_api_key (401): Incorrect API key provided: sk-…. [request_id=req_abc123_3]",
    ]

    verdict = evaluate_regression(baseline, stats)
    (row,) = verdict["rows"]
    assert verdict["exit_code"] == EXIT_ERRORED and row["verdict"] == "errored"
    assert "AuthenticationError/invalid_api_key (401) x2" in row["reason"]
    assert "Provider said: AuthenticationError/invalid_api_key (401): Incorrect API key provided: sk-…." in row["reason"]
    assert row["infra_errors"] == stats["s"]["infra_error_details"]

    markdown = render_verdict_markdown(verdict)
    assert "ERRORED (infrastructure)" in markdown
    assert "invalid_api_key (401)" in markdown and "[request_id=req_abc123_2]" in markdown

    # Records without a structured ``error`` (older artifacts) still label by class alone.
    legacy = scenario_stats(_runs("s", ["ERR", True, True]))["s"]
    assert legacy["infra_error_types"] == {"APITimeoutError": 1}
    assert legacy["infra_error_details"] == []


def test_errored_takes_precedence_over_regression_in_the_overall_status():
    baseline = _baseline(s=(3, 3, {}), t=(3, 3, {}))
    stats = scenario_stats(_runs("s", ["ERR", "ERR", "ERR"]) + _runs("t", ["empty_answer"] * 3))

    verdict = evaluate_regression(baseline, stats)

    assert verdict["status"] == "errored" and verdict["exit_code"] == EXIT_ERRORED
    by_name = {row["scenario"]: row for row in verdict["rows"]}
    assert by_name["t"]["verdict"] == "regression"


def test_scenario_missing_from_baseline_fails_with_an_actionable_reason():
    verdict = evaluate_regression(_baseline(s=(3, 3, {})), scenario_stats(_runs("new_fixture", [True] * 3)))

    assert verdict["status"] == "regression"
    (row,) = verdict["rows"]
    assert row["verdict"] == "no_baseline"
    assert "--write-baseline" in row["reason"]


def test_run_for_a_model_without_a_baseline_entry_cannot_be_judged_and_exits_two():
    verdict = evaluate_regression(
        _baseline(s=(3, 3, {})), scenario_stats(_runs("s", [True] * 3, model="claude-sonnet-5"))
    )

    assert verdict["status"] == "no_model_baseline" and verdict["exit_code"] == EXIT_ERRORED
    assert verdict["cell"] == SONNET_5_KEY
    (row,) = verdict["rows"]
    assert row["verdict"] == "no_model_baseline"
    assert "No baseline entry for anthropic/claude-sonnet-5/high" in row["reason"]
    assert f"`{CELL_KEY}`" in row["reason"]
    assert "--models anthropic:claude-sonnet-5 --efforts high" in row["reason"]
    assert "NO BASELINE FOR THIS MODEL" in render_verdict_markdown(verdict)


def test_gate_picks_the_baseline_entry_matching_the_run_cell():
    baseline = _baseline(s=(1, 3, {"avoidable_tool_failure": 2}))
    baseline["baselines"][SONNET_5_KEY] = _entry({"s": (3, 3, {})}, cell=SONNET_5)

    # 1/3 is fine against the 1/3 Sonnet 4.6 entry...
    old = evaluate_regression(baseline, scenario_stats(_runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"])))
    # ...and a regression against the 3/3 Sonnet 5 entry for the same numbers.
    new = evaluate_regression(
        baseline,
        scenario_stats(_runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"], **SONNET_5)),
    )

    assert old["status"] == "pass" and old["cell"] == CELL_KEY
    assert new["status"] == "regression" and new["cell"] == SONNET_5_KEY
    assert new["rows"][0]["baseline_passed"] == 3
    assert f"- Baseline entry: `{SONNET_5_KEY}`" in render_verdict_markdown(new)


def test_multi_cell_run_is_a_mismatch_not_a_verdict():
    verdict = evaluate_regression(
        _baseline(s=(3, 3, {})),
        scenario_stats(_runs("s", [True] * 3) + _runs("s", [True] * 3, effort="max")),
    )

    assert verdict["status"] == "regression" and verdict["cell"] is None
    (row,) = verdict["rows"]
    assert row["verdict"] == "baseline_mismatch"
    assert "anthropic:claude-sonnet-4-6/high" in row["reason"]
    assert "anthropic:claude-sonnet-4-6/max" in row["reason"]


def test_pooled_baseline_thresholds_with_six_repeats():
    # cmp1 + cmp2 pooled: 6/6, 5/6, 3/6 against N = 3 -> margin max(1/3, 1/sqrt(18)) = 1/3.
    baseline = _baseline(a=(6, 6, {}), b=(5, 6, {"avoidable_tool_failure": 1}), c=(3, 6, {"avoidable_tool_failure": 3}))
    stats = scenario_stats(
        _runs("a", [True, True, "avoidable_tool_failure"])
        + _runs("b", [True, "avoidable_tool_failure", "avoidable_tool_failure"])
        + _runs("c", ["avoidable_tool_failure"] * 3)
    )
    # Give ``a`` the code so only the pass-rate rule is exercised.
    baseline["baselines"][CELL_KEY]["scenarios"]["a"]["hard_violation_codes"] = {"avoidable_tool_failure": 1}

    verdict = evaluate_regression(baseline, stats)

    by_name = {row["scenario"]: row for row in verdict["rows"]}
    assert by_name["a"]["threshold"] == pytest.approx(0.6667, abs=1e-4) and by_name["a"]["verdict"] == "pass"
    assert by_name["b"]["threshold"] == pytest.approx(0.5, abs=1e-4) and by_name["b"]["verdict"] == "regression"
    # 3/6 no longer has a 0.000 threshold: 0/3 is a regression now.
    assert by_name["c"]["threshold"] == pytest.approx(0.1667, abs=1e-4) and by_name["c"]["verdict"] == "regression"


def test_verdict_table_has_the_required_columns_and_labels():
    baseline = _baseline(s=(3, 3, {}), t=(1, 3, {"avoidable_tool_failure": 2}))
    stats = scenario_stats(
        _runs("s", [True, "unsupported_answer_claim", "unsupported_answer_claim"])
        + _runs("t", [True, "avoidable_tool_failure", "avoidable_tool_failure"])
    )
    markdown = render_verdict_markdown(evaluate_regression(baseline, stats), baseline_path="evals/baseline.json")

    assert "## Conversation evaluation gate: REGRESSION" in markdown
    assert "| Scenario | Baseline pass | Current pass | Threshold | New codes | Unseen here | Verdict |" in markdown
    assert (
        "| `s` | 3/3 (100%) | 1/3 (33%) | 0.667 | `unsupported_answer_claim` | "
        "`unsupported_answer_claim` 2/3 | **regression** |"
    ) in markdown
    assert "| `t` | 1/3 (33%) | 1/3 (33%) | 0.000 | - | - | **pass** |" in markdown
    assert "`evals/baseline.json`" in markdown


def test_verdict_table_labels_infrastructure_errors_distinctly():
    markdown = render_verdict_markdown(
        evaluate_regression(_baseline(s=(3, 3, {})), scenario_stats(_runs("s", ["ERR"] * 3)))
    )
    assert "ERRORED (infrastructure)" in markdown
    assert "n/a +3 errored" in markdown


# --- baseline document -------------------------------------------------------


def test_build_baseline_records_counts_codes_and_provenance():
    runs = _runs("a", [True, True, True]) + _runs("b", [True, "avoidable_tool_failure", "avoidable_tool_failure"])

    document = build_baseline(runs, recorded_at="2026-09-05T00:00:00+00:00", source_run_url="https://x/1")

    validate_baseline(document)
    assert document["schema_version"] == 2
    assert list(document["baselines"]) == [CELL_KEY]
    entry = document["baselines"][CELL_KEY]
    assert {k: entry[k] for k in ("provider", "model", "effort")} == CELL
    assert entry["recorded_at"] == "2026-09-05T00:00:00+00:00"
    assert entry["source_run_urls"] == ["https://x/1"]
    assert entry["scenarios"]["b"] == {
        "repeats": 3,
        "passed": 1,
        "pass_rate": 0.3333,
        "mean_score": 0.8333,
        "hard_violation_codes": {"avoidable_tool_failure": 2},
        "sources": [{"source_run_url": "https://x/1", "repeats": 3, "passed": 1}],
    }


def test_build_baseline_pools_runs_from_several_sources_and_keeps_their_attribution():
    first = _runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"])
    second = _runs("s", [True, True, "avoidable_tool_failure"])
    for run in first:
        run["source_run_url"] = "https://x/1"
    for run in second:
        run["source_run_url"] = "https://x/2"

    document = build_baseline(first + second, recorded_at="2026-09-05T00:00:00+00:00")

    entry = document["baselines"][CELL_KEY]
    assert entry["source_run_urls"] == ["https://x/1", "https://x/2"]
    s = entry["scenarios"]["s"]
    assert (s["repeats"], s["passed"], s["pass_rate"]) == (6, 3, 0.5)
    assert s["hard_violation_codes"] == {"avoidable_tool_failure": 3}
    assert s["sources"] == [
        {"source_run_url": "https://x/1", "repeats": 3, "passed": 1},
        {"source_run_url": "https://x/2", "repeats": 3, "passed": 2},
    ]
    # And the pooled entry judges with R = 6.
    verdict = evaluate_regression(document, scenario_stats(_runs("s", ["avoidable_tool_failure"] * 3)))
    assert verdict["rows"][0]["threshold"] == pytest.approx(0.1667, abs=1e-4)
    assert verdict["status"] == "regression"


def test_merge_baselines_replaces_only_the_incoming_cell():
    existing = _baseline(s=(1, 3, {}))
    existing["baselines"][SONNET_5_KEY] = _entry({"s": (3, 3, {})}, cell=SONNET_5)
    incoming = _baseline(s=(6, 6, {}))

    merged = regression.merge_baselines(existing, incoming)

    assert sorted(merged["baselines"]) == [CELL_KEY, SONNET_5_KEY]
    assert merged["baselines"][CELL_KEY]["scenarios"]["s"]["repeats"] == 6
    assert merged["baselines"][SONNET_5_KEY] == existing["baselines"][SONNET_5_KEY]
    assert regression.merge_baselines(None, incoming) == incoming


def test_build_baseline_refuses_infrastructure_errors_and_multi_cell_runs():
    with pytest.raises(ValueError, match="infrastructure errors"):
        build_baseline(_runs("a", [True, "ERR", True]))
    with pytest.raises(ValueError, match="exactly one"):
        build_baseline(_runs("a", [True], effort="high") + _runs("a", [True], effort="max"))
    with pytest.raises(ValueError, match="No runs"):
        build_baseline([])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.pop("schema_version"), "schema_version"),
        (lambda d: d.__setitem__("schema_version", 1), "schema_version must be 2.*re-record"),
        (lambda d: d.__setitem__("baselines", {}), "non-empty"),
        (lambda d: d["baselines"][CELL_KEY].pop("source_run_urls"), "missing source_run_urls"),
        (lambda d: d["baselines"][CELL_KEY].__setitem__("source_run_urls", [""]), "list of URLs"),
        (lambda d: d["baselines"][CELL_KEY].__setitem__("model", ""), "must be a string"),
        (lambda d: d["baselines"][CELL_KEY].__setitem__("model", "other"), "key must match"),
        (lambda d: d["baselines"][CELL_KEY].__setitem__("scenarios", {}), "non-empty"),
        (lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].pop("hard_violation_codes"), "missing hard_violation_codes"),
        (lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].__setitem__("repeats", 0), "positive integer"),
        (lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].__setitem__("pass_rate", 1.5), "within \\[0, 1\\]"),
        (lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].__setitem__("hard_violation_codes", {"x": -1}), "code -> count"),
        (lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].__setitem__("passed", 4), "within \\[0, repeats\\]"),
        (lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].__setitem__("sources", "x"), "must be a list"),
        (lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].__setitem__("sources", [{"repeats": 3}]), "sources entries"),
        (
            lambda d: d["baselines"][CELL_KEY]["scenarios"]["s"].__setitem__(
                "sources", [{"source_run_url": "https://x/1", "repeats": 2, "passed": 2}]
            ),
            "add up",
        ),
        (lambda d: d["baselines"][CELL_KEY].__setitem__("prompt_version", "0123456"), "prompt_version must be an object"),
        (lambda d: d["baselines"][CELL_KEY].__setitem__("prompt_version", {"git_commit": "0123456"}), "missing system_prompt_sha256"),
        (
            lambda d: d["baselines"][CELL_KEY].__setitem__("prompt_version", {"git_commit": "not-a-sha", "system_prompt_sha256": "fedcba987654"}),
            "hex commit id",
        ),
        (
            lambda d: d["baselines"][CELL_KEY].__setitem__("prompt_version", {"git_commit": "0123456", "system_prompt_sha256": "fedc"}),
            "at least 12",
        ),
    ],
)
def test_validate_baseline_rejects_malformed_documents(mutate, message):
    document = _baseline(s=(3, 3, {}))
    mutate(document)
    with pytest.raises(BaselineError, match=message):
        validate_baseline(document)


# --- prompt_version: which prompt an entry measured -----------------------------

PROMPT_VERSION = {"git_commit": "0123456789abcdef0123456789abcdef01234567", "system_prompt_sha256": "fedcba987654"}


def test_build_baseline_records_prompt_version_only_when_given():
    runs = _runs("s", [True, True, True])
    without = build_baseline(runs, recorded_at="2026-09-08T00:00:00+00:00")
    assert "prompt_version" not in without["baselines"][CELL_KEY]

    document = build_baseline(runs, recorded_at="2026-09-08T00:00:00+00:00", prompt_version=PROMPT_VERSION)
    validate_baseline(document)
    assert document["baselines"][CELL_KEY]["prompt_version"] == PROMPT_VERSION
    with pytest.raises(BaselineError, match="hex commit id"):
        build_baseline(runs, prompt_version={"git_commit": "main", "system_prompt_sha256": "fedcba987654"})


def test_entries_without_prompt_version_still_judge_and_say_so():
    # Every entry recorded before the field existed (the _entry helper has none)
    # is judged exactly as before; the header just notes the gap.
    baseline = _baseline(s=(3, 3, {}))
    verdict = evaluate_regression(baseline, scenario_stats(_runs("s", [True] * 3)))
    assert verdict["status"] == "pass" and verdict["baseline_prompt_version"] is None
    markdown = render_verdict_markdown(verdict, current_prompt_sha256="fedcba987654aeb5")
    assert "Baseline prompt: not recorded" in markdown
    assert "differs" not in markdown


def test_verdict_header_names_the_baseline_prompt_and_flags_a_different_prompt_under_test():
    baseline = _baseline(s=(3, 3, {}))
    baseline["baselines"][CELL_KEY]["prompt_version"] = PROMPT_VERSION
    verdict = evaluate_regression(baseline, scenario_stats(_runs("s", [True] * 3)))
    assert verdict["baseline_prompt_version"] == PROMPT_VERSION

    same = render_verdict_markdown(verdict, current_prompt_sha256="fedcba987654aeb5b48d4f99085f3949116cc3b925e79afd74e9e34d4be961a3")
    assert "Baseline prompt: commit `0123456789ab`, system prompt sha256 `fedcba987654` (same prompt as this run)" in same

    other = render_verdict_markdown(verdict, current_prompt_sha256="0123456789abcdef")
    assert "differs from the prompt under test** (`0123456789ab`)" in other
    assert verdict["status"] == "pass", "the prompt note is informational, never a verdict"

    unknown = render_verdict_markdown(verdict)
    assert "sha256 `fedcba987654`" in unknown and "same prompt" not in unknown and "differs" not in unknown

    assert regression.prompt_version_matches(None, "abc") is None
    assert regression.prompt_version_matches(PROMPT_VERSION, None) is None
    assert regression.prompt_version_matches(PROMPT_VERSION, "fedcba987654ffff") is True
    assert regression.prompt_version_matches(PROMPT_VERSION, "fedcba987655") is False


def test_prompt_version_for_hashes_this_checkouts_prompt(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    digest = eval_run.system_prompt_sha256("anthropic")
    assert len(digest) == 64 and int(digest, 16) >= 0

    explicit = eval_run.prompt_version_for("anthropic", commit="0123456")
    assert explicit == {"git_commit": "0123456", "system_prompt_sha256": digest[:12]}
    with pytest.raises(ValueError, match="hex commit id"):
        eval_run.resolve_prompt_commit("feature/branch")

    monkeypatch.setenv("GITHUB_SHA", "0" * 40)
    assert eval_run.resolve_prompt_commit() == "0" * 40
    monkeypatch.setenv("GITHUB_SHA", "not a sha")
    assert eval_run.resolve_prompt_commit() is None


# The repository ships without a recorded baseline: pass rates come from paid model
# runs and must be recorded by whoever deploys it (see evals/README.md). The three
# tests below validate the committed file once one exists.
requires_committed_baseline = pytest.mark.skipif(
    not BASELINE_PATH.is_file(),
    reason="no evals/baseline.json recorded yet (see evals/README.md)",
)


@requires_committed_baseline
def test_committed_baseline_parses_and_covers_every_fixture_for_every_model():
    document = regression.load_baseline(BASELINE_PATH)
    fixtures = {path.stem for path in FIXTURES_DIR.glob("*.json")}
    production = regression.cell_key("anthropic", eval_run.DEFAULT_MODEL, eval_run.DEFAULT_EFFORT)

    assert production in document["baselines"], (
        f"evals/baseline.json has no entry for the production cell {production}; the PR gate "
        "cannot judge without one. Record it with `python -m evals.run --from-results <artifact> "
        f"--models anthropic:{eval_run.DEFAULT_MODEL} --efforts {eval_run.DEFAULT_EFFORT} "
        "--write-baseline evals/baseline.json`."
    )
    for key, entry in document["baselines"].items():
        assert set(entry["scenarios"]) == fixtures, (
            f"{key}: evals/baseline.json must have exactly one entry per fixture; re-record after "
            "adding or removing a scenario"
        )
        assert entry["source_run_urls"], f"{key}: record where the baseline came from"
        assert "prompt_version" in entry, (
            f"{key}: record which prompt the entry measured (re-record with --write-baseline "
            "--prompt-commit <sha> on the commit the comparison ran on)"
        )
        for name, scenario in entry["scenarios"].items():
            assert scenario["repeats"] >= 3, f"{key}/{name}: fewer than 3 repeats is a coin flip"
            assert scenario["passed"] / scenario["repeats"] == pytest.approx(scenario["pass_rate"], abs=1e-3)
            assert sum(source["repeats"] for source in scenario["sources"]) == scenario["repeats"]
            assert sum(source["passed"] for source in scenario["sources"]) == scenario["passed"]
            assert {source["source_run_url"] for source in scenario["sources"]} <= set(entry["source_run_urls"])


@requires_committed_baseline
def test_committed_production_baseline_pools_both_seed_runs():
    # The production entry was pooled from two 9-run comparisons (R = 6) so the
    # percent-change scenario no longer has a 0.000 threshold; with N = 3 the
    # margin is max(1/3, 1/sqrt(18)) = 1/3.
    document = regression.load_baseline(BASELINE_PATH)
    entry = document["baselines"][regression.cell_key("anthropic", eval_run.DEFAULT_MODEL, eval_run.DEFAULT_EFFORT)]

    assert len(entry["source_run_urls"]) >= 2
    for name, scenario in entry["scenarios"].items():
        assert scenario["repeats"] >= 6, f"{name}: the production baseline pools at least two runs"
        threshold = scenario["passed"] / scenario["repeats"] - tolerance_margin(3, scenario["repeats"])
        assert threshold > 0, f"{name}: a 0.000 threshold lets the scenario drop to 0/3 unnoticed"


@requires_committed_baseline
def test_committed_baseline_is_canonically_formatted():
    # Sorted keys + 2-space indent + trailing newline, as write_baseline produces,
    # so reviewers see a minimal diff on re-baseline.
    text = BASELINE_PATH.read_text()
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


# --- CLI round trip ------------------------------------------------------------


def _write_results_tree(root: Path, runs):
    for run in runs:
        path = eval_run.run_output_path(root, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run) + "\n")


def test_write_baseline_then_baseline_round_trip_from_results(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    results = tmp_path / "results"
    runs = (
        _runs("signups_column_count", [True, True, True])
        + _runs("signups_table_explanation", [True, True, "avoidable_tool_failure"])
        + _runs("signup_percent_change", [True, "avoidable_tool_failure", "avoidable_tool_failure"])
    )
    _write_results_tree(results, runs)
    baseline_path = tmp_path / "baseline.json"

    code = eval_run.main(
        [
            "--from-results", str(results),
            "--write-baseline", str(baseline_path),
            "--source-run-url", "https://example.test/run/9",
            "--prompt-commit", "0123456789abcdef0123456789abcdef01234567",
            "--output-dir", str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert f"Wrote baseline entry {CELL_KEY}" in out and "prompt 0123456789ab sha256" in out
    document = json.loads(baseline_path.read_text())
    validate_baseline(document)
    entry = document["baselines"][CELL_KEY]
    assert entry["scenarios"]["signup_percent_change"]["passed"] == 1
    assert entry["source_run_urls"] == ["https://example.test/run/9"]
    assert entry["scenarios"]["signup_percent_change"]["sources"] == [
        {"source_run_url": "https://example.test/run/9", "repeats": 3, "passed": 1}
    ]
    assert entry["recorded_at"]
    assert entry["prompt_version"] == {
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "system_prompt_sha256": eval_run.system_prompt_sha256("anthropic")[:12],
    }

    # The same results judged against the baseline they produced pass, the
    # header confirms the prompt matches, and the verdict artifacts are written
    # next to the summary.
    code = eval_run.main(
        [
            "--from-results", str(results),
            "--baseline", str(baseline_path),
            "--output-dir", str(tmp_path / "judge"),
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_PASS
    assert "Conversation evaluation gate: PASS" in out
    assert "Baseline prompt: commit `0123456789ab`" in out and "(same prompt as this run)" in out
    verdict = json.loads((tmp_path / "judge" / "verdict.json").read_text())
    assert [row["verdict"] for row in verdict["rows"]] == ["pass", "pass", "pass"]
    assert (tmp_path / "judge" / "verdict.md").read_text().startswith("## Conversation evaluation gate: PASS")


def test_from_results_selects_one_cell_and_regression_exits_one(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    results = tmp_path / "results"
    _write_results_tree(
        results,
        _runs("s", [True] * 3)
        + _runs("s", ["ERR"] * 3, provider="openai", model="gpt-5.6-terra"),
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline(s=(3, 3, {}))))

    ok = eval_run.main(
        ["--from-results", str(results), "--baseline", str(baseline_path), "--models", "anthropic:claude-sonnet-4-6", "--output-dir", str(tmp_path / "a")]
    )
    assert ok == EXIT_PASS

    errored = eval_run.main(
        ["--from-results", str(results), "--baseline", str(baseline_path), "--models", "openai:gpt-5.6-terra", "--output-dir", str(tmp_path / "b")]
    )
    assert errored == EXIT_ERRORED
    assert "INFRASTRUCTURE ERROR" in capsys.readouterr().out

    _write_results_tree(tmp_path / "worse", _runs("s", [True, "empty_answer", "empty_answer"]))
    worse = eval_run.main(
        ["--from-results", str(tmp_path / "worse"), "--baseline", str(baseline_path), "--output-dir", str(tmp_path / "c")]
    )
    out = capsys.readouterr().out
    assert worse == EXIT_REGRESSION
    assert "BEHAVIORAL REGRESSION" in out


def test_write_baseline_pools_several_result_trees_and_merges_by_cell(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    # Two comparison artifacts of the same prompts: Sonnet 4.6 in both, Sonnet 5 only in the second.
    _write_results_tree(
        tmp_path / "cmp1",
        _runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"])
        + _runs("s", ["ERR"] * 3, provider="openai", model="gpt-5.6-terra"),
    )
    _write_results_tree(
        tmp_path / "cmp2",
        _runs("s", [True, True, "avoidable_tool_failure"]) + _runs("s", [True] * 3, **SONNET_5),
    )
    baseline_path = tmp_path / "baseline.json"

    code = eval_run.main(
        [
            "--from-results", str(tmp_path / "cmp1"),
            "--from-results", str(tmp_path / "cmp2"),
            "--models", "anthropic:claude-sonnet-4-6", "--efforts", "high",
            "--source-run-url", "https://x/1",
            "--source-run-url", "https://x/2",
            "--write-baseline", str(baseline_path),
            "--output-dir", str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Loaded 6 run record(s) from 2 tree(s)" in out
    assert f"Wrote baseline entry {CELL_KEY}" in out and "s 3/6" in out
    document = validate_baseline(json.loads(baseline_path.read_text()))
    s = document["baselines"][CELL_KEY]["scenarios"]["s"]
    assert (s["repeats"], s["passed"]) == (6, 3)
    assert s["sources"] == [
        {"source_run_url": "https://x/1", "repeats": 3, "passed": 1},
        {"source_run_url": "https://x/2", "repeats": 3, "passed": 2},
    ]
    assert document["baselines"][CELL_KEY]["source_run_urls"] == ["https://x/1", "https://x/2"]

    # A second write for another model is merged in; the pooled entry is untouched.
    code = eval_run.main(
        [
            "--from-results", str(tmp_path / "cmp2"),
            "--models", "anthropic:claude-sonnet-5", "--efforts", "high",
            "--source-run-url", "https://x/2",
            "--write-baseline", str(baseline_path),
            "--output-dir", str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and f"kept {CELL_KEY}" in out
    document = validate_baseline(json.loads(baseline_path.read_text()))
    assert sorted(document["baselines"]) == [CELL_KEY, SONNET_5_KEY]
    assert document["baselines"][CELL_KEY]["scenarios"]["s"]["repeats"] == 6
    assert document["baselines"][SONNET_5_KEY]["scenarios"]["s"] == {
        "repeats": 3, "passed": 3, "pass_rate": 1.0, "mean_score": 1.0,
        "hard_violation_codes": {},
        "sources": [{"source_run_url": "https://x/2", "repeats": 3, "passed": 3}],
    }

    # Judging picks the entry for the run's own cell.
    _write_results_tree(
        tmp_path / "s5", _runs("s", [True, "avoidable_tool_failure", "avoidable_tool_failure"], **SONNET_5)
    )
    code = eval_run.main(["--from-results", str(tmp_path / "s5"), "--baseline", str(baseline_path), "--output-dir", str(tmp_path / "j")])
    verdict = json.loads((tmp_path / "j" / "verdict.json").read_text())
    assert code == EXIT_REGRESSION  # 3/3 entry never had avoidable_tool_failure; it recurred in 2 of 3
    assert verdict["cell"] == SONNET_5_KEY
    assert verdict["rows"][0]["new_codes"] == ["avoidable_tool_failure"]
    assert "BEHAVIORAL REGRESSION" in capsys.readouterr().out

    # A model with no entry cannot be judged: exit 2 with a clear message.
    _write_results_tree(tmp_path / "s6", _runs("s", [True] * 3, model="claude-sonnet-6"))
    code = eval_run.main(["--from-results", str(tmp_path / "s6"), "--baseline", str(baseline_path), "--output-dir", str(tmp_path / "k")])
    out = capsys.readouterr().out
    assert code == EXIT_ERRORED
    assert "NO BASELINE:" in out and "no entry for anthropic/claude-sonnet-6/high" in out
    assert "NO BASELINE FOR THIS MODEL" in out and "BEHAVIORAL" not in out
    assert json.loads((tmp_path / "k" / "verdict.json").read_text())["status"] == "no_model_baseline"


def test_source_run_url_count_must_match_result_trees(tmp_path):
    _write_results_tree(tmp_path / "a", _runs("s", [True] * 3))
    _write_results_tree(tmp_path / "b", _runs("s", [True] * 3))
    with pytest.raises(SystemExit, match="one per directory or one in total"):
        eval_run.main(
            [
                "--from-results", str(tmp_path / "a"), "--from-results", str(tmp_path / "b"),
                "--source-run-url", "https://x/1", "--source-run-url", "https://x/2", "--source-run-url", "https://x/3",
                "--write-baseline", str(tmp_path / "b.json"), "--output-dir", str(tmp_path),
            ]
        )
    assert not (tmp_path / "b.json").exists()


def test_write_baseline_refuses_to_merge_into_a_malformed_file(tmp_path, capsys):
    _write_results_tree(tmp_path / "a", _runs("s", [True] * 3))
    (tmp_path / "b.json").write_text('{"schema_version": 1, "scenarios": {}}')
    with pytest.raises(SystemExit, match="cannot merge into existing baseline.*delete it"):
        eval_run.main(["--from-results", str(tmp_path / "a"), "--write-baseline", str(tmp_path / "b.json"), "--output-dir", str(tmp_path)])


def test_write_baseline_from_errored_results_exits_two(tmp_path, capsys):
    results = tmp_path / "results"
    _write_results_tree(results, _runs("s", [True, "ERR", True]))

    code = eval_run.main(["--from-results", str(results), "--write-baseline", str(tmp_path / "b.json"), "--output-dir", str(tmp_path)])

    assert code == EXIT_ERRORED
    assert "INFRASTRUCTURE ERROR" in capsys.readouterr().out
    assert not (tmp_path / "b.json").exists()


def test_baseline_and_write_baseline_are_mutually_exclusive_and_baseline_must_parse(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(SystemExit, match="schema_version"):
        eval_run.main(["--baseline", str(bad), "--output-dir", str(tmp_path / "o")])
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_baseline(s=(3, 3, {}))))
    with pytest.raises(SystemExit, match="mutually exclusive"):
        eval_run.main(["--baseline", str(good), "--write-baseline", str(tmp_path / "w.json"), "--output-dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="needs --baseline or --write-baseline"):
        eval_run.main(["--from-results", str(tmp_path), "--output-dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="--prompt-commit only applies with --write-baseline"):
        eval_run.main(["--from-results", str(tmp_path), "--baseline", str(good), "--prompt-commit", "0123456", "--output-dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="hex commit id"):
        eval_run.main(["--from-results", str(tmp_path), "--write-baseline", str(tmp_path / "w.json"), "--prompt-commit", "main", "--output-dir", str(tmp_path)])
    assert not (tmp_path / "w.json").exists()


def test_live_gate_invocation_uses_the_baseline_verdict(tmp_path, stubbed_runner, capsys, monkeypatch):
    """``--repeats 3 --baseline`` on a stubbed model: one flipped conversation passes,
    an infra error exits 2, and the verdict table lands in the step summary."""
    step_summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    baseline_path = tmp_path / "baseline.json"
    # No --models/--efforts: the gate runs the production default cell, so the
    # baseline must carry that entry (pr-check.yml invokes it exactly this way).
    production = {"provider": "anthropic", "model": eval_run.DEFAULT_MODEL, "effort": eval_run.DEFAULT_EFFORT}
    assert production == SONNET_5
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "baselines": {
                    regression.cell_key(**production): _entry(
                        {"signups_table_explanation": (3, 3, {})}, cell=production
                    )
                },
            }
        )
    )

    code = eval_run.main(
        ["--output-dir", str(tmp_path / "a"), "--scenario", "signups_table_explanation", "--repeats", "3", "--baseline", str(baseline_path)],
        client_factory=lambda: object(),
        sleep=lambda _s: None,
    )
    assert code == EXIT_PASS
    assert len(stubbed_runner["calls"]) == 3
    assert "Conversation evaluation gate: PASS" in step_summary.read_text()

    stubbed_runner["outcome"] = "error"
    code = eval_run.main(
        ["--output-dir", str(tmp_path / "b"), "--scenario", "signups_table_explanation", "--repeats", "3", "--baseline", str(baseline_path)],
        client_factory=lambda: object(),
        sleep=lambda _s: None,
    )
    out = capsys.readouterr().out
    assert code == EXIT_ERRORED
    assert "ERRORED (infrastructure)" in out and "BEHAVIORAL" not in out
