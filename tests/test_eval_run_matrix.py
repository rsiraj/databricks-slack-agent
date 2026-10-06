"""Matrix runner, infrastructure-vs-behavioral separation, outputs, and CLI exit codes."""

import copy
import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from evals import run as eval_run
from evals.aggregate import aggregate_runs, percentile, render_summary_markdown
from evals.synthetic_tools import load_fixture


FIXTURE = load_fixture("signups_table_explanation")
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(status: int):
    response = httpx2.Response(status, request=_REQUEST)
    cls = {
        429: anthropic.RateLimitError,
        400: anthropic.BadRequestError,
        404: anthropic.NotFoundError,
    }.get(status, anthropic.InternalServerError if status >= 500 else anthropic.APIStatusError)
    return cls("synthetic", response=response, body=None)


class _Block(SimpleNamespace):
    def model_dump(self):
        return {"type": "text", "text": self.text}


class _FlakyClient:
    """Fails ``failures`` times with ``error`` before answering ``end_turn``."""

    def __init__(self, failures=0, error=None, text="One row per `_anonymousId` browser identity."):
        self.messages = self
        self.failures = failures
        self.error = error
        self.calls = 0
        self.text = text

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return SimpleNamespace(
            content=[_Block(type="text", text=self.text)],
            stop_reason="end_turn",
            model=kwargs["model"],
            usage=SimpleNamespace(
                input_tokens=100,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=5000,
                output_tokens=50,
            ),
        )


def test_retryable_error_classification():
    assert eval_run.is_retryable_error(_status_error(429))
    assert eval_run.is_retryable_error(_status_error(500))
    assert eval_run.is_retryable_error(_status_error(529))
    assert eval_run.is_retryable_error(anthropic.APITimeoutError(request=_REQUEST))
    assert eval_run.is_retryable_error(anthropic.APIConnectionError(request=_REQUEST))
    assert not eval_run.is_retryable_error(_status_error(400))
    assert not eval_run.is_retryable_error(_status_error(404))
    assert not eval_run.is_retryable_error(RuntimeError("config"))


def test_run_one_retries_whole_conversation_on_transient_error_then_scores():
    client = _FlakyClient(failures=1, error=_status_error(529))
    sleeps = []

    record = eval_run.run_one(
        scenario="signups_table_explanation",
        model="claude-sonnet-4-6",
        effort="high",
        rep=1,
        client=client,
        sleep=sleeps.append,
        rng=lambda: 0.5,
    )

    assert record["status"] == "scored"
    assert record["attempts"] == 2
    assert client.calls == 2
    assert sleeps == [2.5]  # base 2s * 2**0 + jitter 0.5
    assert record["result"]["passed"] is False  # fake answer lacks tool evidence
    assert record["trace"]["stop_reason"] == "end_turn"


def test_run_one_records_errored_after_exhausting_retries():
    client = _FlakyClient(failures=10, error=_status_error(429))
    sleeps = []

    record = eval_run.run_one(
        scenario="signups_table_explanation",
        model="claude-sonnet-4-6",
        effort="high",
        rep=1,
        client=client,
        sleep=sleeps.append,
        rng=lambda: 0.0,
    )

    assert record["status"] == "errored"
    assert record["attempts"] == 3
    assert record["error_type"] == "RateLimitError"
    assert record["retryable"] is True
    assert record["result"] is None
    assert record["trace"]["stop_reason"] == "infrastructure_error"
    assert sleeps == [2.0, 4.0]


def test_run_one_does_not_retry_non_transient_errors():
    client = _FlakyClient(failures=10, error=_status_error(400))
    sleeps = []

    record = eval_run.run_one(
        scenario="signups_table_explanation",
        model="claude-sonnet-4-6",
        effort="max",
        rep=2,
        client=client,
        sleep=sleeps.append,
    )

    assert record["status"] == "errored"
    assert record["attempts"] == 1
    assert record["error_type"] == "BadRequestError"
    assert record["retryable"] is False
    assert sleeps == []
    assert client.calls == 1


def test_run_matrix_covers_every_combination_and_passes_config_through():
    seen = []

    def fake_run(**kwargs):
        seen.append(kwargs)
        return {**{k: kwargs[k] for k in ("model", "effort", "scenario", "rep")}, "status": "scored"}

    runs = eval_run.run_matrix(
        models=["a", "b"],
        efforts=["low", "high"],
        scenarios=["s1"],
        repeats=3,
        client=object(),
        concurrency=4,
        thinking="disabled",
        max_tokens=2048,
        sleep=lambda _s: None,
        run_fn=fake_run,
    )

    assert len(runs) == 12
    assert {(r["model"], r["effort"], r["rep"]) for r in runs} == {
        (m, e, k) for m in ("a", "b") for e in ("low", "high") for k in (1, 2, 3)
    }
    assert all(call["thinking"] == "disabled" and call["max_tokens"] == 2048 for call in seen)
    # No override: the runner falls back to the fixture's or production's budget.
    assert all(call["max_api_calls"] is None for call in seen)


def test_run_one_passes_max_api_calls_override_to_the_runner(monkeypatch):
    seen = {}

    def fake_run_conversation(fixture, **kwargs):
        seen.update(kwargs)
        return {"scenario": fixture["id"], "api_calls": 1, "tool_events": [], "final_text": "x"}

    monkeypatch.setattr(eval_run, "run_conversation", fake_run_conversation)
    monkeypatch.setattr(eval_run, "score_trace", lambda trace, fixture: {"passed": True})

    eval_run.run_one(
        scenario="signups_table_explanation",
        model="m",
        effort="high",
        rep=1,
        client=object(),
        max_api_calls=7,
    )
    assert seen["max_api_calls"] == 7

    eval_run.run_one(
        scenario="signups_table_explanation", model="m", effort="high", rep=1, client=object()
    )
    assert seen["max_api_calls"] is None


def _scored(model, effort, rep, *, passed, wall_ms, cost, violations=(), soft_violations=()):
    recorded = [{"code": code, "message": code, "severity": "hard"} for code in violations] + [
        {"code": code, "message": code, "severity": "soft"} for code in soft_violations
    ]
    return {
        "model": model,
        "effort": effort,
        "scenario": "signups_table_explanation",
        "rep": rep,
        "status": "scored",
        "trace": {
            "api_calls": 4,
            "tool_events": [{}, {}, {}],
            "usage": {
                "input_tokens": 1000,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 50000,
                "output_tokens": 800,
            },
            "latency_ms": {"wall_clock_ms": wall_ms},
            "api_call_events": [{"latency_ms": wall_ms / 4}] * 4,
            "estimated_cost_usd": cost,
        },
        "result": {
            "passed": passed,
            "score": round(1.0 - 0.25 * len(violations) - 0.05 * len(soft_violations), 4),
            "violations": recorded,
            "metrics": {"api_calls": 4, "tool_calls": 3},
        },
    }


def _errored(model, effort, rep, error_type="RateLimitError"):
    return {
        "model": model,
        "effort": effort,
        "scenario": "signups_table_explanation",
        "rep": rep,
        "status": "errored",
        "attempts": 3,
        "error_type": error_type,
        "trace": {"stop_reason": "infrastructure_error"},
        "result": None,
    }


def test_aggregate_separates_infrastructure_errors_from_behavioral_fails():
    runs = [
        _scored("m1", "high", 1, passed=True, wall_ms=1000, cost=0.10),
        _scored("m1", "high", 2, passed=False, wall_ms=3000, cost=0.30, violations=("empty_answer",)),
        _errored("m1", "high", 3),
        _scored("m2", "low", 1, passed=True, wall_ms=500, cost=None),
    ]

    summary = aggregate_runs(runs)

    assert summary["runs"] == 4
    assert summary["scored"] == 3
    assert summary["errored"] == 1
    assert summary["failed"] == 1
    m1 = next(cell for cell in summary["cells"] if cell["model"] == "m1")
    assert m1["runs"] == 3 and m1["scored"] == 2 and m1["errored"] == 1
    assert m1["pass_rate"] == 0.5  # errored run excluded from the denominator
    assert m1["mean_score"] == 0.875
    assert m1["wall_clock"] == {"mean_ms": 2000.0, "p50_ms": 1000.0, "p95_ms": 3000.0}
    assert m1["mean_api_calls"] == 4.0
    assert m1["p95_api_calls"] == 4.0
    assert m1["api_call_budget_hits"] == 0
    assert m1["mean_tool_calls"] == 3.0
    assert m1["mean_usage"]["cache_read_input_tokens"] == 50000.0
    assert m1["mean_estimated_cost_usd"] == 0.2
    assert m1["violation_histogram"] == {"empty_answer": 1}
    assert m1["infra_error_histogram"] == {"RateLimitError": 1}
    m2 = next(cell for cell in summary["cells"] if cell["model"] == "m2")
    assert m2["pass_rate"] == 1.0
    assert m2["mean_estimated_cost_usd"] is None

    markdown = render_summary_markdown(summary, runs)
    assert "## Conversation evaluation: ERRORED" in markdown
    assert "| anthropic | `m1` | high | 2/3 | 50.0% (1/2) |" in markdown
    assert "- violation `empty_answer` [hard]: 1" in markdown
    assert "infrastructure error `RateLimitError`: 1" in markdown
    assert "rep3: ERRORED (RateLimitError, 3 attempt(s))" in markdown


def test_aggregate_reports_p95_api_calls_and_budget_hits_per_cell():
    runs = []
    for rep, calls in enumerate((3, 5, 8, 25), start=1):
        run = _scored("m1", "high", rep, passed=calls < 25, wall_ms=100, cost=0.01)
        run["trace"]["api_calls"] = calls
        run["trace"]["hit_api_call_budget"] = calls == 25
        run["trace"]["budget"] = {"max_api_calls": 25, "max_output_tokens": 400_000}
        run["result"]["metrics"]["api_calls"] = calls
        runs.append(run)

    summary = aggregate_runs(runs)
    cell = summary["cells"][0]
    assert cell["mean_api_calls"] == 10.25
    assert cell["p95_api_calls"] == 25.0
    assert cell["api_call_budget_hits"] == 1

    markdown = render_summary_markdown(summary, runs)
    header = next(line for line in markdown.splitlines() if line.startswith("| Provider | Model |"))
    assert "API calls mean/p95" in header and "Budget hits" in header
    row = next(line for line in markdown.splitlines() if line.startswith("| anthropic | `m1` |"))
    assert "| 10.2/25 | 1 |" in row
    assert "api_calls=25/25 hit_budget=yes" in markdown
    assert "api_calls=3/25 hit_budget=no" in markdown


def test_summary_reports_soft_violations_without_failing_and_lowers_the_mean_score():
    """Soft (content-coverage) violations must stay visible in summary.md and in
    the graded score so comparison runs see coverage regressions, while the
    pass rate and the gate ignore them."""
    runs = [
        _scored("m1", "high", 1, passed=True, wall_ms=1000, cost=0.10),
        _scored(
            "m1",
            "high",
            2,
            passed=True,
            wall_ms=1000,
            cost=0.10,
            soft_violations=("missing_answer_fact", "missing_answer_fact"),
        ),
        _scored(
            "m1",
            "high",
            3,
            passed=False,
            wall_ms=1000,
            cost=0.10,
            violations=("unsupported_answer_claim",),
            soft_violations=("missing_answer_fact",),
        ),
    ]

    summary = aggregate_runs(runs)
    markdown = render_summary_markdown(summary, runs)

    assert summary["failed"] == 1
    assert summary["hard_violations"] == 1
    assert summary["soft_violations"] == 3
    cell = summary["cells"][0]
    assert cell["pass_rate"] == round(2 / 3, 4)
    assert cell["hard_violations"] == 1 and cell["soft_violations"] == 3
    assert cell["mean_score"] == round((1.0 + 0.9 + 0.7) / 3, 3)
    assert cell["violation_histogram"] == {"missing_answer_fact": 3, "unsupported_answer_claim": 1}
    assert cell["violation_severity_histogram"] == {
        "missing_answer_fact [soft]": 3,
        "unsupported_answer_claim [hard]": 1,
    }

    assert "## Conversation evaluation: FAIL" in markdown
    assert "- Violations: hard 1 (block the gate), soft 3 (reported only" in markdown
    assert "- violation `missing_answer_fact` [soft]: 3" in markdown
    assert "- violation `unsupported_answer_claim` [hard]: 1" in markdown
    assert "rep1: PASS score=1.000 hard=0 soft=0 " in markdown
    assert "rep2: PASS score=0.900 hard=0 soft=2 " in markdown
    assert "rep3: FAIL score=0.700 hard=1 soft=1 " in markdown
    assert "  - `missing_answer_fact` [soft]: missing_answer_fact" in markdown
    assert "  - `unsupported_answer_claim` [hard]: unsupported_answer_claim" in markdown


def test_summary_with_only_soft_violations_passes_the_gate():
    runs = [
        _scored("m1", "high", 1, passed=True, wall_ms=1000, cost=0.10, soft_violations=("missing_answer_fact",)),
    ]

    summary = aggregate_runs(runs)
    markdown = render_summary_markdown(summary, runs)

    assert summary["failed"] == 0 and summary["soft_violations"] == 1
    assert "## Conversation evaluation: PASS" in markdown
    assert "behavioral fails 0" in markdown
    assert "soft 1 (reported only" in markdown
    assert "score=0.950 hard=0 soft=1 " in markdown


def test_percentile_is_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([10, 1, 5], 50) == 5


def test_write_outputs_layout_and_single_run_legacy_files(tmp_path):
    single = [_scored("claude-sonnet-4-6", "high", 1, passed=True, wall_ms=100, cost=0.1)]

    eval_run.write_outputs(tmp_path, single)

    rep = tmp_path / "claude-sonnet-4-6" / "high" / "signups_table_explanation" / "rep1.json"
    assert rep.is_file()
    assert json.loads(rep.read_text())["status"] == "scored"
    assert json.loads((tmp_path / "trace.json").read_text())["api_calls"] == 4
    assert json.loads((tmp_path / "result.json").read_text())["passed"] is True
    assert (tmp_path / "summary.md").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["runs"] == 1

    multi = tmp_path / "multi"
    eval_run.write_outputs(
        multi,
        [
            _scored("claude-sonnet-4-6", "high", 1, passed=True, wall_ms=100, cost=0.1),
            _scored("claude-sonnet-5", "medium", 1, passed=True, wall_ms=100, cost=0.1),
        ],
    )
    assert (multi / "claude-sonnet-5" / "medium" / "signups_table_explanation" / "rep1.json").is_file()
    assert not (multi / "trace.json").exists()


def test_write_outputs_single_errored_run_writes_infra_result(tmp_path):
    eval_run.write_outputs(tmp_path, [_errored("claude-sonnet-4-6", "high", 1, "APITimeoutError")])

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["passed"] is False
    assert result["status"] == "errored"
    assert result["violations"][0]["code"] == "evaluation_infrastructure"
    assert "APITimeoutError" in result["violations"][0]["message"]


@pytest.fixture
def stubbed_runner(monkeypatch):
    """Replace the model loop so ``main`` exercises the CLI without an API."""
    state = {"outcome": "pass", "calls": []}

    def fake_run_conversation(
        fixture,
        *,
        client,
        model,
        effort,
        thinking,
        max_tokens,
        provider="anthropic",
        max_api_calls=None,
    ):
        state["calls"].append({"model": model, "effort": effort, "thinking": thinking})
        state.setdefault("providers", []).append(provider)
        state.setdefault("max_api_calls", []).append(max_api_calls)
        if state["outcome"] == "error":
            raise anthropic.APITimeoutError(request=_REQUEST)
        return {
            "scenario": fixture["id"],
            "model": model,
            "stop_reason": "end_turn",
            "final_text": "answer",
            "tool_events": [],
            "api_calls": 1,
            "api_call_events": [{"latency_ms": 10.0}],
            "output_tokens": 10,
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 10,
            },
            "latency_ms": {"wall_clock_ms": 12.0},
            "estimated_cost_usd": 0.0001,
            "budget": {"max_api_calls": 25, "max_output_tokens": 400_000},
        }

    def fake_score_trace(trace, fixture):
        passed = state["outcome"] == "pass"
        return {
            "scenario": fixture["id"],
            "passed": passed,
            "score": 1.0 if passed else 0.75,
            "violations": [] if passed else [{"code": "empty_answer", "message": "x", "severity": "hard"}],
            "metrics": {"api_calls": 1, "tool_calls": 0},
        }

    monkeypatch.setattr(eval_run, "run_conversation", fake_run_conversation)
    monkeypatch.setattr(eval_run, "score_trace", fake_score_trace)
    monkeypatch.delenv("MODEL_EVAL_MODEL", raising=False)
    monkeypatch.delenv("MODEL_EVAL_EFFORT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    return state


def _main(tmp_path, *extra, sleep=lambda _s: None):
    # Pin one scenario so these CLI-shape assertions do not scale with the
    # number of fixtures on disk; an explicit --scenarios in ``extra`` wins.
    return eval_run.main(
        ["--output-dir", str(tmp_path), "--scenario", "signups_table_explanation", *extra],
        client_factory=lambda: object(),
        sleep=sleep,
    )


def test_default_invocation_runs_one_model_one_effort_and_exits_zero_on_pass(tmp_path, stubbed_runner, capsys):
    code = _main(tmp_path)

    assert code == 0
    # The default cell is the production model/effort (claude-sonnet-5 / high).
    assert stubbed_runner["calls"] == [
        {"model": "claude-sonnet-5", "effort": "high", "thinking": "adaptive"}
    ]
    assert (tmp_path / "trace.json").is_file()
    assert (tmp_path / "result.json").is_file()
    assert "Conversation evaluation: PASS" in capsys.readouterr().out


def test_behavioral_failure_exits_one_unless_disabled(tmp_path, stubbed_runner, capsys):
    stubbed_runner["outcome"] = "fail"

    assert _main(tmp_path) == 1
    assert "BEHAVIORAL FAILURE" in capsys.readouterr().out
    assert _main(tmp_path / "b", "--no-fail-on-violation") == 0


def test_infrastructure_error_exits_two_with_distinct_message(tmp_path, stubbed_runner, capsys):
    stubbed_runner["outcome"] = "error"
    sleeps = []

    code = _main(tmp_path, "--no-fail-on-violation", sleep=sleeps.append)

    out = capsys.readouterr().out
    assert code == eval_run.EXIT_ERRORED == 2
    assert "INFRASTRUCTURE ERROR" in out
    assert "BEHAVIORAL FAILURE" not in out
    assert len(sleeps) == 2  # two retries per conversation
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["errored"] == 1 and summary["failed"] == 0


def test_matrix_flags_and_env_defaults(tmp_path, stubbed_runner, monkeypatch):
    code = _main(
        tmp_path,
        "--models",
        "claude-sonnet-4-6,claude-sonnet-5",
        "--efforts",
        "medium,high",
        "--repeats",
        "2",
        "--thinking",
        "none",
        "--concurrency",
        "3",
    )

    assert code == 0
    assert len(stubbed_runner["calls"]) == 8
    assert all(call["thinking"] is None for call in stubbed_runner["calls"])
    for model in ("claude-sonnet-4-6", "claude-sonnet-5"):
        for effort in ("medium", "high"):
            assert (tmp_path / model / effort / "signups_table_explanation" / "rep2.json").is_file()
    assert not (tmp_path / "trace.json").exists()

    stubbed_runner["calls"].clear()
    monkeypatch.setenv("MODEL_EVAL_MODEL", "claude-opus-5")
    monkeypatch.setenv("MODEL_EVAL_EFFORT", "low")
    assert _main(tmp_path / "env") == 0
    assert stubbed_runner["calls"] == [{"model": "claude-opus-5", "effort": "low", "thinking": "adaptive"}]


def test_max_api_calls_flag_is_passed_through_and_defaults_to_none(tmp_path, stubbed_runner):
    assert _main(tmp_path / "default") == 0
    assert stubbed_runner["max_api_calls"] == [None]

    assert _main(tmp_path / "override", "--max-api-calls", "12", "--repeats", "2") == 0
    assert stubbed_runner["max_api_calls"][1:] == [12, 12]

    with pytest.raises(SystemExit):
        _main(tmp_path / "bad", "--max-api-calls", "0")
    with pytest.raises(SystemExit):
        _main(tmp_path / "bad", "--max-api-calls", "many")


def test_unknown_scenario_or_effort_is_rejected(tmp_path, stubbed_runner):
    with pytest.raises(SystemExit, match="Unknown scenario"):
        _main(tmp_path, "--scenarios", "does_not_exist")
    with pytest.raises(SystemExit, match="Unknown effort"):
        _main(tmp_path, "--efforts", "turbo")


def test_client_factory_failure_is_reported_as_infrastructure_error(tmp_path, stubbed_runner, capsys):
    def broken():
        raise RuntimeError("MODEL_EVAL_ANTHROPIC_API_KEY is required")

    code = eval_run.main(["--output-dir", str(tmp_path)], client_factory=broken)

    assert code == eval_run.EXIT_ERRORED
    assert "INFRASTRUCTURE ERROR" in capsys.readouterr().out
    assert "RuntimeError" in (tmp_path / "summary.md").read_text()


def test_provider_flag_defaults_to_anthropic_and_is_passed_through(tmp_path, stubbed_runner):
    assert _main(tmp_path) == 0
    assert stubbed_runner["providers"] == ["anthropic"]

    stubbed_runner["providers"].clear()
    stubbed_runner["calls"].clear()
    assert _main(tmp_path / "openai", "--provider", "openai") == 0
    assert stubbed_runner["providers"] == ["openai"]
    # Without --models, the provider's production default model is used.
    assert stubbed_runner["calls"] == [{"model": "gpt-5.6-terra", "effort": "high", "thinking": "adaptive"}]
    assert (tmp_path / "openai" / "gpt-5.6-terra" / "high" / "signups_table_explanation" / "rep1.json").is_file()


def test_openai_provider_without_package_is_an_infrastructure_error(tmp_path, stubbed_runner, capsys, monkeypatch):
    import sys

    monkeypatch.setenv("MODEL_EVAL_OPENAI_API_KEY", "sk-eval")
    monkeypatch.setitem(sys.modules, "openai", None)

    code = eval_run.main(["--output-dir", str(tmp_path), "--provider", "openai"])

    assert code == eval_run.EXIT_ERRORED
    assert "INFRASTRUCTURE ERROR" in capsys.readouterr().out
    assert "ProviderNotConfigured" in (tmp_path / "summary.md").read_text()


def test_openai_provider_requires_eval_key(tmp_path, stubbed_runner, monkeypatch):
    monkeypatch.delenv("MODEL_EVAL_OPENAI_API_KEY", raising=False)
    from evals.conversation_runner import build_openai_client

    with pytest.raises(RuntimeError, match="MODEL_EVAL_OPENAI_API_KEY"):
        build_openai_client()


def test_unknown_provider_flag_is_rejected(tmp_path, stubbed_runner):
    with pytest.raises(SystemExit):
        _main(tmp_path, "--provider", "gemini")


def test_retryable_classification_by_shape_for_other_sdks():
    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, status_code):
            super().__init__("synthetic")
            self.status_code = status_code

    assert eval_run.is_retryable_error(APIConnectionError())
    assert eval_run.is_retryable_error(APIStatusError(429))
    assert eval_run.is_retryable_error(APIStatusError(503))
    assert not eval_run.is_retryable_error(APIStatusError(400))


class TestProviderQualifiedModels:
    """``--models`` entries as ``<provider>:<model>`` with bare ids falling back to ``--provider``."""

    @pytest.mark.parametrize(
        ("spec", "default", "expected"),
        [
            ("claude-sonnet-4-6", "anthropic", ("anthropic", "claude-sonnet-4-6")),
            ("gpt-5.6-terra", "openai", ("openai", "gpt-5.6-terra")),
            ("anthropic:claude-sonnet-5", "openai", ("anthropic", "claude-sonnet-5")),
            ("openai:gpt-5.6-terra", "anthropic", ("openai", "gpt-5.6-terra")),
            (" OpenAI : gpt-5.6-terra ", "anthropic", ("openai", "gpt-5.6-terra")),
        ],
    )
    def test_parse_model_spec(self, spec, default, expected):
        assert eval_run.parse_model_spec(spec, default) == expected

    def test_unknown_provider_prefix_is_rejected_not_sent_as_a_model_id(self):
        with pytest.raises(ValueError, match="Unknown provider 'opnai'"):
            eval_run.parse_model_spec("opnai:gpt-5.6-terra")
        with pytest.raises(ValueError, match="no model id"):
            eval_run.parse_model_spec("openai:")
        with pytest.raises(ValueError):
            eval_run.parse_model_spec("   ")

    def test_parse_model_specs_keeps_order_and_drops_duplicates(self):
        assert eval_run.parse_model_specs(
            ["openai:gpt-5.6-terra", "claude-sonnet-4-6", "anthropic:claude-sonnet-4-6"], "anthropic"
        ) == [("openai", "gpt-5.6-terra"), ("anthropic", "claude-sonnet-4-6")]

    def test_run_matrix_selects_the_client_per_cell_provider(self):
        anthropic_client, openai_client = object(), object()
        seen = []

        def fake_run(**kwargs):
            seen.append(kwargs)
            return {**{k: kwargs[k] for k in ("provider", "model", "effort", "scenario", "rep")}, "status": "scored"}

        runs = eval_run.run_matrix(
            models=["anthropic:claude-sonnet-4-6", "openai:gpt-5.6-terra", "claude-sonnet-5"],
            efforts=["high"],
            scenarios=["s1"],
            repeats=2,
            clients={"anthropic": anthropic_client, "openai": openai_client},
            provider="anthropic",
            run_fn=fake_run,
        )

        assert len(runs) == 6
        assert {(r["provider"], r["model"]) for r in runs} == {
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-5.6-terra"),
            ("anthropic", "claude-sonnet-5"),
        }
        for call in seen:
            expected = openai_client if call["provider"] == "openai" else anthropic_client
            assert call["client"] is expected

    def test_run_matrix_without_clients_mapping_uses_the_single_client(self):
        shared = object()
        seen = []

        def fake_run(**kwargs):
            seen.append(kwargs)
            return {"status": "scored", **{k: kwargs[k] for k in ("provider", "model", "effort", "scenario", "rep")}}

        eval_run.run_matrix(
            models=["a"], efforts=["high"], scenarios=["s1"], repeats=1, client=shared, provider="openai", run_fn=fake_run
        )
        assert seen[0]["client"] is shared and seen[0]["provider"] == "openai"

    def test_run_matrix_marks_provider_without_client_unavailable(self):
        seen = []

        def fake_run(**kwargs):
            seen.append(kwargs)
            return {"status": "errored", **{k: kwargs[k] for k in ("provider", "model", "effort", "scenario", "rep")}}

        eval_run.run_matrix(
            models=["openai:gpt-5.6-terra"],
            efforts=["high"],
            scenarios=["s1"],
            repeats=1,
            clients={"anthropic": object()},
            run_fn=fake_run,
        )
        marker = seen[0]["client"]
        assert isinstance(marker, eval_run.ClientUnavailable)
        assert marker.provider == "openai"

    def test_run_one_with_unavailable_client_is_errored_without_attempting(self):
        cause = RuntimeError("MODEL_EVAL_OPENAI_API_KEY is required")
        calls = []

        record = eval_run.run_one(
            scenario="signups_table_explanation",
            provider="openai",
            model="gpt-5.6-terra",
            effort="high",
            rep=1,
            client=eval_run.ClientUnavailable("openai", cause),
            sleep=calls.append,
        )

        assert record["status"] == "errored"
        assert record["attempts"] == 0
        assert record["provider"] == "openai"
        assert record["error_type"] == "RuntimeError"  # the underlying cause, not the marker
        assert record["retryable"] is False
        assert record["result"] is None
        assert record["trace"]["provider"] == "openai"
        assert record["trace"]["stop_reason"] == "infrastructure_error"
        assert calls == []

    def test_build_clients_isolates_one_provider_failure(self):
        def openai_factory():
            raise RuntimeError("MODEL_EVAL_OPENAI_API_KEY is required")

        anthropic_client = object()
        clients = eval_run.build_clients(
            ["anthropic", "openai"],
            client_factories={"anthropic": lambda: anthropic_client, "openai": openai_factory},
        )

        assert clients["anthropic"] is anthropic_client
        assert isinstance(clients["openai"], eval_run.ClientUnavailable)
        assert type(clients["openai"].cause).__name__ == "RuntimeError"

    def test_aggregate_groups_by_provider_model_effort_and_reports_provider(self):
        runs = [
            _scored("shared-id", "high", 1, passed=True, wall_ms=100, cost=0.1),
            {**_scored("shared-id", "high", 1, passed=False, wall_ms=200, cost=0.2, violations=("empty_answer",)), "provider": "openai"},
            {**_scored("gpt-5.6-terra", "high", 1, passed=True, wall_ms=300, cost=0.3), "provider": "openai"},
            {**_errored("gpt-5.6-terra", "high", 2, "ProviderNotConfigured"), "provider": "openai"},
        ]

        summary = aggregate_runs(runs)

        keys = [(cell["provider"], cell["model"], cell["effort"]) for cell in summary["cells"]]
        # Same model id under two providers stays in two cells; legacy records are anthropic.
        assert keys == [
            ("anthropic", "shared-id", "high"),
            ("openai", "gpt-5.6-terra", "high"),
            ("openai", "shared-id", "high"),
        ]
        by_key = {key: cell for key, cell in zip(keys, summary["cells"])}
        assert by_key[("anthropic", "shared-id", "high")]["pass_rate"] == 1.0
        assert by_key[("openai", "shared-id", "high")]["pass_rate"] == 0.0
        openai_terra = by_key[("openai", "gpt-5.6-terra", "high")]
        assert openai_terra["scored"] == 1 and openai_terra["errored"] == 1
        assert openai_terra["infra_error_histogram"] == {"ProviderNotConfigured": 1}
        assert summary["failed"] == 1 and summary["errored"] == 1

        markdown = render_summary_markdown(summary, runs)
        header = next(line for line in markdown.splitlines() if line.startswith("| Provider | Model |"))
        assert header.startswith("| Provider | Model | Effort | Runs |")
        assert "| anthropic | `shared-id` | high | 1/1 | 100.0% (1/1) |" in markdown
        assert "| openai | `shared-id` | high | 1/1 | 0.0% (0/1) |" in markdown
        assert "| openai | `gpt-5.6-terra` | high | 1/2 | 100.0% (1/1) |" in markdown
        assert "### openai / `gpt-5.6-terra` / high" in markdown
        assert "- infrastructure error `ProviderNotConfigured`: 1" in markdown
        assert "- openai / `gpt-5.6-terra` / high / `signups_table_explanation` rep2: ERRORED" in markdown
        assert "- anthropic / `shared-id` / high / `signups_table_explanation` rep1: PASS" in markdown

    def test_cli_runs_a_cross_provider_matrix_and_records_provider_per_cell(self, tmp_path, stubbed_runner):
        code = _main(
            tmp_path,
            "--models",
            "anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra",
            "--efforts",
            "high",
            "--repeats",
            "2",
        )

        assert code == 0
        assert sorted(stubbed_runner["providers"]) == ["anthropic", "anthropic", "openai", "openai"]
        assert {call["model"] for call in stubbed_runner["calls"]} == {"claude-sonnet-4-6", "gpt-5.6-terra"}
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert [(cell["provider"], cell["model"]) for cell in summary["cells"]] == [
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-5.6-terra"),
        ]
        rep = json.loads((tmp_path / "gpt-5.6-terra" / "high" / "signups_table_explanation" / "rep2.json").read_text())
        assert rep["provider"] == "openai" and rep["status"] == "scored"

    def test_cli_rejects_unknown_provider_prefix(self, tmp_path, stubbed_runner):
        with pytest.raises(SystemExit, match="Unknown provider 'gemini'"):
            _main(tmp_path, "--models", "gemini:pro")

    def test_cli_bare_ids_follow_the_provider_flag(self, tmp_path, stubbed_runner):
        assert _main(tmp_path, "--provider", "openai", "--models", "gpt-5.6-terra,anthropic:claude-sonnet-4-6") == 0
        assert sorted(zip(stubbed_runner["providers"], (c["model"] for c in stubbed_runner["calls"]))) == [
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-5.6-terra"),
        ]

    def test_cli_missing_openai_key_errors_only_the_openai_cells(self, tmp_path, stubbed_runner, monkeypatch, capsys):
        """The Anthropic half of a head-to-head still runs and scores when the OpenAI
        eval key is absent; the OpenAI cells are infrastructure errors, not fails."""
        monkeypatch.setenv("MODEL_EVAL_ANTHROPIC_API_KEY", "sk-ant-eval")
        monkeypatch.delenv("MODEL_EVAL_OPENAI_API_KEY", raising=False)

        code = eval_run.main(
            [
                "--output-dir",
                str(tmp_path),
                "--scenario",
                "signups_table_explanation",
                "--models",
                "anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra",
                "--no-fail-on-violation",
            ],
            sleep=lambda _s: None,
        )

        out = capsys.readouterr().out
        assert code == eval_run.EXIT_ERRORED
        assert "INFRASTRUCTURE ERROR: no openai client (RuntimeError: MODEL_EVAL_OPENAI_API_KEY is required)" in out
        assert "BEHAVIORAL FAILURE" not in out
        assert stubbed_runner["providers"] == ["anthropic"]  # only the Anthropic cell reached the model loop
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["scored"] == 1 and summary["errored"] == 1 and summary["failed"] == 0
        cells = {(cell["provider"], cell["model"]): cell for cell in summary["cells"]}
        assert cells[("anthropic", "claude-sonnet-4-6")]["passed"] == 1
        assert cells[("openai", "gpt-5.6-terra")]["infra_error_histogram"] == {"RuntimeError": 1}

    def test_cli_with_every_provider_unavailable_reports_could_not_start(self, tmp_path, stubbed_runner, monkeypatch, capsys):
        monkeypatch.delenv("MODEL_EVAL_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("MODEL_EVAL_OPENAI_API_KEY", raising=False)

        code = eval_run.main(
            ["--output-dir", str(tmp_path), "--models", "anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra"]
        )

        assert code == eval_run.EXIT_ERRORED
        assert "INFRASTRUCTURE ERROR: Evaluation could not start (anthropic: RuntimeError, openai: RuntimeError)." in capsys.readouterr().out
        assert stubbed_runner["calls"] == []
        assert not (tmp_path / "summary.json").exists()


def _openai_status_error(status: int, error: dict, request_id: str | None = "req_openai_1"):
    """A real ``openai`` SDK exception, built the way its client builds one from a response.

    The SDK passes the *inner* error object as ``body`` and reads ``code`` /
    ``type`` / ``param`` from it; ``request_id`` comes from ``x-request-id``.
    """
    import openai

    headers = {"x-request-id": request_id} if request_id else {}
    response = httpx2.Response(
        status, request=httpx2.Request("POST", "https://api.openai.com/v1/responses"), headers=headers
    )
    cls = {401: openai.AuthenticationError, 404: openai.NotFoundError, 429: openai.RateLimitError}.get(
        status, openai.APIStatusError
    )
    return cls(message=error["message"], response=response, body=error)


def _anthropic_status_error(status: int, envelope: dict, request_id: str | None = "req_anthropic_1"):
    """A real ``anthropic`` SDK exception: ``body`` is the full ``{"type": "error", "error": {...}}`` envelope."""
    headers = {"request-id": request_id} if request_id else {}
    response = httpx2.Response(status, request=_REQUEST, headers=headers)
    cls = {401: anthropic.AuthenticationError, 404: anthropic.NotFoundError}.get(status, anthropic.APIStatusError)
    return cls(message=f"Error code: {status} - {envelope}", response=response, body=envelope)


class TestErroredRunDiagnostics:
    """Errored records carry the provider's own diagnosis (status, API code/type, redacted message,
    request id) so a 401 is actionable from the artifact alone."""

    OPENAI_BAD_KEY = {
        "message": (
            "Incorrect API key provided: sk-proj-abc1************************************xyz9. "
            "You can find your API key at https://platform.openai.com/account/api-keys."
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }

    def test_redaction_collapses_keys_and_bearer_tokens_and_whitespace(self):
        message = (
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-proj-abc1***************xyz9. "
            "You can find your API key at https://platform.openai.com/account/api-keys.',\n 'type': 'invalid_request_error'}}"
        )
        scrubbed = eval_run.scrub_error_message(RuntimeError(message))
        assert "sk-proj" not in scrubbed and "xyz9" not in scrubbed
        assert "Incorrect API key provided: sk-…." in scrubbed
        assert "\n" not in scrubbed
        # Unmasked keys of either vendor's shape, and bearer credentials, collapse too.
        assert eval_run.redact_secrets("key sk-ant-api03-AbCdEfGhIjKlMnOp_QrSt-uv end") == "key sk-… end"
        assert eval_run.redact_secrets("Authorization: Bearer sk_live_ABCDEFGHIJ") == "Authorization: Bearer …"
        # Short ``sk-`` fragments that cannot be keys are left alone so prose survives.
        assert eval_run.redact_secrets("prefix sk-eval ok") == "prefix sk-eval ok"
        assert eval_run.scrub_error_message(None) == ""
        assert len(eval_run.scrub_error_message(RuntimeError("x" * 1000))) == eval_run.ERROR_MESSAGE_MAX_CHARS
        assert eval_run.ERROR_MESSAGE_MAX_CHARS == 300

    def test_describe_error_reads_openai_code_type_status_message_and_request_id(self):
        exc = _openai_status_error(401, self.OPENAI_BAD_KEY)
        error = eval_run.describe_error(exc)
        assert error == {
            "error_type": "AuthenticationError",
            "status_code": 401,
            "code": "invalid_api_key",
            "type": "invalid_request_error",
            "message": (
                "Incorrect API key provided: sk-…. "
                "You can find your API key at https://platform.openai.com/account/api-keys."
            ),
            "request_id": "req_openai_1",
        }
        assert "xyz9" not in json.dumps(error)  # the masked key never reaches the artifact

    @pytest.mark.parametrize(
        ("status", "error", "expected_code"),
        [
            (429, {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}, "insufficient_quota"),
            (404, {"message": "The model `gpt-5.6-terra` does not exist or you do not have access to it.", "type": "invalid_request_error", "code": "model_not_found"}, "model_not_found"),
            (401, {"message": "You have insufficient permissions for this operation. Missing scopes: model.request", "type": "invalid_request_error", "code": None}, None),
        ],
    )
    def test_describe_error_distinguishes_openai_failure_modes(self, status, error, expected_code):
        described = eval_run.describe_error(_openai_status_error(status, error))
        assert described["status_code"] == status
        assert described["code"] == expected_code
        assert described["type"] == error["type"]
        assert described["message"] == error["message"]

    def test_describe_error_reads_anthropic_type_from_the_envelope_and_request_id_header(self):
        exc = _anthropic_status_error(
            401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        )
        error = eval_run.describe_error(exc)
        assert error == {
            "error_type": "AuthenticationError",
            "status_code": 401,
            "code": None,  # Anthropic errors carry a ``type`` but no ``code``
            "type": "authentication_error",
            "message": "invalid x-api-key",  # the API message, not the SDK's "Error code: 401 - {...}" wrapper
            "request_id": "req_anthropic_1",
        }
        assert eval_run.describe_error(_anthropic_status_error(404, {"type": "error", "error": {"type": "not_found_error", "message": "model: nope"}}, request_id=None))["request_id"] is None

    def test_describe_error_falls_back_to_str_for_non_sdk_and_duck_typed_errors(self):
        class DuckStatusError(Exception):
            status_code = 503
            body = "<html>upstream</html>"  # non-dict body: nothing to extract

        assert eval_run.describe_error(DuckStatusError("gateway down")) == {
            "error_type": "DuckStatusError",
            "status_code": 503,
            "code": None,
            "type": None,
            "message": "gateway down",
            "request_id": None,
        }
        assert eval_run.describe_error(RuntimeError("MODEL_EVAL_OPENAI_API_KEY is required"))["message"] == (
            "MODEL_EVAL_OPENAI_API_KEY is required"
        )
        assert eval_run.describe_error(None) == {
            "error_type": "UnknownError", "status_code": None, "code": None, "type": None, "message": "", "request_id": None
        }

    def test_errored_record_and_summary_carry_the_structured_error(self):
        class FailingResponses:
            def __init__(self, exc):
                self.exc = exc

            def create(self, **kwargs):
                raise self.exc

        client = SimpleNamespace(responses=FailingResponses(_openai_status_error(401, self.OPENAI_BAD_KEY)))
        record = eval_run.run_one(
            scenario="signups_table_explanation",
            provider="openai",
            model="gpt-5.6-terra",
            effort="high",
            rep=1,
            client=client,
            sleep=lambda _s: None,
        )

        assert record["status"] == "errored"
        assert record["attempts"] == 1  # 401 is not retryable
        assert record["error_type"] == "AuthenticationError"  # kept flat for the histogram / older readers
        assert record["error"]["status_code"] == 401
        assert record["error"]["code"] == "invalid_api_key"
        assert record["error"]["request_id"] == "req_openai_1"
        assert "xyz9" not in json.dumps(record)

        markdown = render_summary_markdown(aggregate_runs([record]), [record])
        assert "- infrastructure error `AuthenticationError/invalid_api_key (401)`: 1" in markdown
        assert (
            "rep1: ERRORED (AuthenticationError/invalid_api_key (401): Incorrect API key provided: sk-…. "
            "You can find your API key at https://platform.openai.com/account/api-keys. "
            "[request_id=req_openai_1], 1 attempt(s))"
        ) in markdown

    def test_histogram_separates_same_class_different_code_and_uses_type_when_code_is_absent(self):
        def errored(exc, rep):
            return eval_run.run_one(
                scenario="signups_table_explanation", provider="openai", model="m", effort="high", rep=rep,
                client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(exc))),
                sleep=lambda _s: None,
            )

        runs = [
            errored(_openai_status_error(401, self.OPENAI_BAD_KEY), 1),
            errored(_openai_status_error(401, self.OPENAI_BAD_KEY), 2),
            errored(_openai_status_error(401, {"message": "Missing scopes: model.request", "type": "invalid_request_error", "code": None}), 3),
            errored(_anthropic_status_error(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}), 4),
        ]
        cell = aggregate_runs(runs)["cells"][0]
        assert cell["infra_error_histogram"] == {
            "AuthenticationError/invalid_api_key (401)": 2,
            "AuthenticationError/invalid_request_error (401)": 1,
            "AuthenticationError/authentication_error (401)": 1,
        }

    def test_unavailable_client_record_carries_the_cause_message(self):
        record = eval_run.run_one(
            scenario="signups_table_explanation",
            provider="openai",
            model="gpt-5.6-terra",
            effort="high",
            rep=1,
            client=eval_run.ClientUnavailable("openai", RuntimeError("MODEL_EVAL_OPENAI_API_KEY is required")),
        )
        assert record["error"] == {
            "error_type": "RuntimeError",
            "status_code": None,
            "code": None,
            "type": None,
            "message": "MODEL_EVAL_OPENAI_API_KEY is required",
            "request_id": None,
        }
        assert "ERRORED (RuntimeError: MODEL_EVAL_OPENAI_API_KEY is required, 0 attempt(s))" in render_summary_markdown(
            aggregate_runs([record]), [record]
        )

    def test_legacy_errored_records_without_error_object_render_as_before(self):
        run = _errored("m1", "high", 3)
        markdown = render_summary_markdown(aggregate_runs([run]), [run])
        assert "rep3: ERRORED (RateLimitError, 3 attempt(s))" in markdown
        assert "- infrastructure error `RateLimitError`: 1" in markdown


def test_make_provider_wraps_each_cell_client_in_its_own_provider_class():
    """A cross-provider matrix hands each cell's SDK client to the matching provider,
    and the OpenAI request carries the effort/budget/reasoning shape production sends."""
    from evals.conversation_runner import make_provider
    from utils.llm.anthropic_provider import AnthropicProvider
    from utils.llm.openai_provider import OpenAIProvider
    from utils.llm.types import ModelRequest

    anthropic_client, openai_client = object(), object()

    anthropic_provider = make_provider("anthropic", client=anthropic_client, model="claude-sonnet-4-6")
    assert isinstance(anthropic_provider, AnthropicProvider)
    assert anthropic_provider.model == "claude-sonnet-4-6"

    openai_provider = make_provider("openai", client=openai_client, model="gpt-5.6-terra")
    assert isinstance(openai_provider, OpenAIProvider)
    assert openai_provider._client() is openai_client  # injected, never built from a secret
    kwargs = openai_provider.build_kwargs(
        ModelRequest(
            system="s", messages=[{"role": "user", "content": "q"}], tools=[], effort="high", max_tokens=16_000, timeout_s=None
        )
    )
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["max_output_tokens"] >= 25_000
    assert kwargs["store"] is False
    assert kwargs["include"] == ["reasoning.encrypted_content"]

    with pytest.raises(ValueError, match="Unknown provider"):
        make_provider("gemini", client=object(), model="x")


def test_openai_client_factory_builds_a_real_client_when_key_and_package_exist(monkeypatch, capsys):
    """Enablement step 1 is done: with ``openai`` installed the eval factory needs only the key."""
    import openai

    monkeypatch.setenv("MODEL_EVAL_OPENAI_API_KEY", "sk-proj-evalkey0123456789")
    from evals.conversation_runner import build_openai_client

    client = build_openai_client()
    assert isinstance(client, openai.OpenAI)
    assert client.max_retries == 2
    out = capsys.readouterr().out
    assert "MODEL_EVAL_OPENAI_API_KEY shape prefix=sk-proj- length=25" in out
    assert "evalkey" not in out  # the log never carries key material


def test_describe_key_shape_reports_copy_mistakes_without_the_value():
    from evals.conversation_runner import describe_key_shape

    assert describe_key_shape("sk-proj-" + "a" * 150) == "prefix=sk-proj- length=158"
    assert describe_key_shape("sk-" + "b" * 48) == "prefix=sk- length=51"
    assert describe_key_shape('"sk-proj-abc"') == (
        "prefix=unexpected (does not start with sk-) length=13 (contains quote characters)"
    )
    assert describe_key_shape("Bearer sk-proj-abc") == (
        "prefix=unexpected (does not start with sk-) length=18 (contains internal whitespace; starts with 'Bearer ')"
    )
    assert "abc" not in describe_key_shape("sk-abcdefghijk")


def test_fixture_is_not_mutated_between_repeats():
    original = copy.deepcopy(FIXTURE)
    client = _FlakyClient()

    for rep in (1, 2):
        eval_run.run_one(
            scenario="signups_table_explanation",
            model="claude-sonnet-4-6",
            effort="high",
            rep=rep,
            client=client,
            sleep=lambda _s: None,
        )

    assert load_fixture("signups_table_explanation") == original
