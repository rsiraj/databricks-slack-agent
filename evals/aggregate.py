"""Aggregate per-run evaluation records into per-(provider, model, effort) comparison tables."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

# Run records written before provider-qualified matrices carry no ``provider``;
# they were all Anthropic runs.
LEGACY_PROVIDER = "anthropic"

USAGE_CLASSES = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile; ``None`` for an empty input."""
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_ms": _mean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def run_provider(run: dict[str, Any]) -> str:
    """The provider a run record belongs to (``anthropic`` for pre-provider records)."""
    return str(run.get("provider") or LEGACY_PROVIDER)


def infra_error_label(run: dict[str, Any], *, with_message: bool = False) -> str:
    """Human-readable identity of an errored run's failure.

    ``<ExceptionClass>/<api code or type> (<http status>)``, dropping whatever
    the record lacks, so ``AuthenticationError/invalid_api_key (401)`` and
    ``AuthenticationError/authentication_error (401)`` (Anthropic has no
    ``code``) are distinct histogram keys while a legacy record with only
    ``error_type`` still reads ``RateLimitError``. ``with_message`` appends the
    redacted API message and request id for the per-run detail line.
    """
    error = run.get("error") if isinstance(run.get("error"), dict) else {}
    label = str(error.get("error_type") or run.get("error_type") or "unknown")
    code = error.get("code") or error.get("type")
    if code:
        label += f"/{code}"
    status = error.get("status_code")
    if status is not None:
        label += f" ({status})"
    if with_message:
        if error.get("message"):
            label += f": {error['message']}"
        if error.get("request_id"):
            label += f" [request_id={error['request_id']}]"
    return label


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise run records (see ``evals.run``) per (provider, model, effort).

    A run record has ``provider``, ``model``, ``effort``, ``scenario``, ``rep``,
    ``status`` (``"scored"`` or ``"errored"``), and, when scored, ``trace`` and
    ``result``. Errored runs are reported separately and never counted as
    behavioral fails.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run_provider(run), str(run.get("model")), str(run.get("effort")))].append(run)

    cells: list[dict[str, Any]] = []
    for (provider, model, effort), members in sorted(grouped.items()):
        scored = [run for run in members if run.get("status") == "scored"]
        errored = [run for run in members if run.get("status") == "errored"]
        passed = [run for run in scored if (run.get("result") or {}).get("passed")]
        traces = [run.get("trace") or {} for run in scored]
        results = [run.get("result") or {} for run in scored]

        wall_clock = [
            float((trace.get("latency_ms") or {}).get("wall_clock_ms") or 0) for trace in traces
        ]
        api_call_latencies = [
            float(call.get("latency_ms") or 0)
            for trace in traces
            for call in trace.get("api_call_events") or []
        ]
        usage_means = {}
        for usage_class in USAGE_CLASSES:
            values = [float((trace.get("usage") or {}).get(usage_class) or 0) for trace in traces]
            usage_means[usage_class] = _mean(values)
        costs = [
            float(trace["estimated_cost_usd"])
            for trace in traces
            if trace.get("estimated_cost_usd") is not None
        ]
        all_violations = [
            violation for result in results for violation in result.get("violations") or []
        ]
        violation_histogram = Counter(
            violation.get("code", "unknown") for violation in all_violations
        )
        # Keyed "code [severity]" so a soft coverage miss is never mistaken for
        # a blocking failure in the job summary.
        violation_severity_histogram = Counter(
            f"{violation.get('code', 'unknown')} [{_severity(violation)}]"
            for violation in all_violations
        )
        hard_count = sum(1 for violation in all_violations if _severity(violation) == "hard")
        # Keyed by class/code (status) so three 401s with different API codes
        # are three different problems in the summary, not one count.
        error_histogram = Counter(infra_error_label(run) for run in errored)
        scores = [
            float(result["score"]) for result in results if result.get("score") is not None
        ]
        api_calls = [float(trace.get("api_calls") or 0) for trace in traces]
        budget_hits = sum(1 for trace in traces if trace.get("hit_api_call_budget"))

        cells.append(
            {
                "provider": provider,
                "model": model,
                "effort": effort,
                "runs": len(members),
                "scored": len(scored),
                "errored": len(errored),
                "passed": len(passed),
                "pass_rate": round(len(passed) / len(scored), 4) if scored else None,
                "mean_score": _mean(scores),
                "wall_clock": _latency_stats(wall_clock),
                "api_call_latency": _latency_stats(api_call_latencies),
                "mean_api_calls": _mean(api_calls),
                # How close each cell runs to production's iteration bound
                # (worker.MAX_ITERATIONS); a budget hit is a hard failure.
                "p95_api_calls": percentile(api_calls, 95),
                "api_call_budget_hits": budget_hits,
                "mean_tool_calls": _mean(
                    [float(len(trace.get("tool_events") or [])) for trace in traces]
                ),
                "mean_usage": usage_means,
                "mean_estimated_cost_usd": round(mean(costs), 6) if costs else None,
                "hard_violations": hard_count,
                "soft_violations": len(all_violations) - hard_count,
                "violation_histogram": dict(violation_histogram.most_common()),
                "violation_severity_histogram": dict(violation_severity_histogram.most_common()),
                "infra_error_histogram": dict(error_histogram.most_common()),
            }
        )

    return {
        "runs": len(runs),
        "scored": sum(cell["scored"] for cell in cells),
        "errored": sum(cell["errored"] for cell in cells),
        "failed": sum(cell["scored"] - cell["passed"] for cell in cells),
        "hard_violations": sum(cell["hard_violations"] for cell in cells),
        "soft_violations": sum(cell["soft_violations"] for cell in cells),
        "cells": cells,
    }


def _severity(violation: dict[str, Any]) -> str:
    return str(violation.get("severity") or "hard")


def _severity_counts(result: dict[str, Any]) -> tuple[int, int]:
    """``(hard, soft)`` counts for one scored result, derived from its violations."""
    violations = result.get("violations") or []
    hard = sum(1 for violation in violations if _severity(violation) == "hard")
    return hard, len(violations) - hard


def _fmt(value: Any, digits: int = 0, prefix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{prefix}{value:,.{digits}f}"
    return f"{prefix}{value}"


def render_summary_markdown(summary: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    """Render the aggregate table plus a per-run detail list."""
    status = "PASS"
    if summary["errored"]:
        status = "ERRORED"
    elif summary["failed"]:
        status = "FAIL"
    lines = [
        f"## Conversation evaluation: {status}",
        "",
        f"- Runs: {summary['runs']} (scored {summary['scored']}, "
        f"behavioral fails {summary['failed']}, infrastructure errors {summary['errored']})",
        f"- Violations: hard {summary.get('hard_violations', 0)} (block the gate), "
        f"soft {summary.get('soft_violations', 0)} (reported only; content coverage, "
        "lowers the score)",
        "",
        "| Provider | Model | Effort | Runs | Pass rate | Mean score | "
        "Wall-clock mean/p50/p95 (ms) | API-call p50/p95 (ms) | API calls mean/p95 | "
        "Budget hits | Tool calls | Input | Cache write | Cache read | Output | "
        "Est. cost (USD) | Infra errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in summary["cells"]:
        wall = cell["wall_clock"]
        api = cell["api_call_latency"]
        usage = cell["mean_usage"]
        pass_rate = cell["pass_rate"]
        lines.append(
            f"| {cell.get('provider') or LEGACY_PROVIDER} | `{cell['model']}` | {cell['effort']} | "
            f"{cell['scored']}/{cell['runs']} | "
            f"{_fmt(pass_rate * 100 if pass_rate is not None else None, 1)}% "
            f"({cell['passed']}/{cell['scored']}) | {_fmt(cell['mean_score'], 3)} | "
            f"{_fmt(wall['mean_ms'])}/{_fmt(wall['p50_ms'])}/{_fmt(wall['p95_ms'])} | "
            f"{_fmt(api['p50_ms'])}/{_fmt(api['p95_ms'])} | {_fmt(cell['mean_api_calls'], 1)}/"
            f"{_fmt(cell.get('p95_api_calls'))} | {cell.get('api_call_budget_hits', 0)} | "
            f"{_fmt(cell['mean_tool_calls'], 1)} | {_fmt(usage['input_tokens'])} | "
            f"{_fmt(usage['cache_creation_input_tokens'])} | "
            f"{_fmt(usage['cache_read_input_tokens'])} | {_fmt(usage['output_tokens'])} | "
            f"{_fmt(cell['mean_estimated_cost_usd'], 4, '$')} | {cell['errored']} |"
        )

    for cell in summary["cells"]:
        if not cell["violation_histogram"] and not cell["infra_error_histogram"]:
            continue
        lines.extend(
            ["", f"### {cell.get('provider') or LEGACY_PROVIDER} / `{cell['model']}` / {cell['effort']}", ""]
        )
        histogram = cell.get("violation_severity_histogram") or {
            f"{code} [hard]": count for code, count in cell["violation_histogram"].items()
        }
        for key, count in histogram.items():
            code, _space, severity = key.rpartition(" ")
            lines.append(f"- violation `{code}` {severity}: {count}")
        for error_label, count in cell["infra_error_histogram"].items():
            lines.append(f"- infrastructure error `{error_label}`: {count}")

    lines.extend(["", "### Runs", ""])
    for run in runs:
        label = (
            f"{run_provider(run)} / `{run.get('model')}` / {run.get('effort')} / "
            f"`{run.get('scenario')}` rep{run.get('rep')}"
        )
        if run.get("status") == "errored":
            lines.append(
                f"- {label}: ERRORED ({infra_error_label(run, with_message=True)}, "
                f"{run.get('attempts', 1)} attempt(s))"
            )
            continue
        result = run.get("result") or {}
        trace = run.get("trace") or {}
        metrics = result.get("metrics") or {}
        outcome = "PASS" if result.get("passed") else "FAIL"
        hard, soft = _severity_counts(result)
        lines.append(
            f"- {label}: {outcome} score={_fmt(result.get('score'), 3)} "
            f"hard={hard} soft={soft} "
            f"api_calls={metrics.get('api_calls', 0)}/{_fmt((trace.get('budget') or {}).get('max_api_calls'))} "
            f"hit_budget={'yes' if trace.get('hit_api_call_budget') else 'no'} "
            f"tool_calls={metrics.get('tool_calls', 0)} "
            f"preflights={metrics.get('metadata_preflights_injected', 0)} "
            f"synthetic_limits={metrics.get('synthetic_limitations', 0)} "
            f"wall_clock_ms={_fmt((trace.get('latency_ms') or {}).get('wall_clock_ms'))} "
            f"cost={_fmt(trace.get('estimated_cost_usd'), 4, '$')}"
        )
        for violation in result.get("violations") or []:
            lines.append(
                f"  - `{violation['code']}` [{violation.get('severity', 'hard')}]: "
                f"{violation['message']}"
            )
    return "\n".join(lines) + "\n"
