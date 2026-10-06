"""Baseline-relative regression rule for the conversation-evaluation PR gate.

The gate used to run each scenario once and fail on any hard violation. With a
nondeterministic model that is a coin flip: the current prompts fail the
percent-change scenario about one run in three for a real, known model
behavior, so unrelated PRs went red and were rerun until green. This module
replaces "any failure fails" with "a regression relative to a committed
baseline fails", measured over enough repeats to clear the noise floor, and
keeps infrastructure errors out of the behavioral verdict entirely.

Everything here is pure: it takes run records (see ``evals.run.run_one``) and a
baseline document, and returns a verdict. No model calls, no I/O beyond the
``load``/``write`` helpers.

The rule
--------

For each scenario, with ``N`` current repeats and ``R`` baseline repeats::

    margin = max(1 / N, 1 / sqrt(N * R))

and the scenario is a **regression** when either

1. ``current_pass_rate < baseline_pass_rate - margin`` (the pass-rate rule,
   the primary instrument), or
2. a hard violation code that has zero occurrences in **every scenario of the
   baseline entry** (entry-wide, not just this scenario) shows up in at least
   ``NEW_CODE_MIN_REPEATS`` (2) of the ``N`` current repeats of a scenario, or
   in at least ``NEW_CODE_MIN_FRACTION`` (50%) of them (the new-code rule).

Why the new-code rule is entry-wide and needs recurrence: a code already
recorded anywhere in the entry is a *known* failure mode of this model and
prompt, and its frequency is exactly what the pass-rate margin governs; only a
failure mode the baseline has never seen at all is a new kind of behavior.
And a small clean baseline sample cannot prove a code is absent: with
``R = 6`` a failure mode with a true rate of about 10% has a
``0.9 ** 6 ~= 53%`` chance of zero observations, so the first later occurrence
of it would read as a "new" regression about half the time on the prompt
under test doing nothing wrong. Requiring two of three repeats (or half of
them) makes a single such observation informational, not a verdict. Codes
absent from the *scenario's own* baseline are still listed per row
(``unseen_here``) so reviewers see them, but they never fail the gate on
their own.

Any run with ``status: "errored"`` (provider 5xx/429/timeout after retries,
missing key, missing package) makes the whole verdict **errored**: the pass
rate over the remaining samples is not trustworthy, and the failure is not the
PR's fault. Errored verdicts carry a distinct exit code (``EXIT_ERRORED``, 2)
so the workflow can label the check "failed for infrastructure", not
"behavioral regression". The errored row's reason names the failure from the
run record's structured ``error`` (exception class, API code/type, HTTP
status, redacted message, request id) so the cause is readable from the
verdict table alone.

Arithmetic, with the gate's ``N = 3`` and a baseline pooled from two runs
(``R = 6``)::

    margin = max(1/3, 1/sqrt(18)) = max(0.333, 0.236) = 0.333

so a scenario the baseline passed 6/6 (1.0) regresses only when the current
rate is below ``1.0 - 0.333 = 0.667``: 2/3 passes, 1/3 fails. A scenario the
baseline passed 3/6 (0.5) has threshold 0.167 and regresses at 0/3; it also
regresses through an entry-wide-new violation code in 2 of its 3 repeats at
any pass rate (which, against a 6/6 baseline, the pass-rate rule already
catches). The ``1/N`` term
guarantees the tolerance is never smaller than one conversation, so a single
flip can never fail the gate on its own; the ``1/sqrt(N*R)`` term is the usual
standard-error scale for a proportion estimated from ``N*R`` Bernoulli samples
and takes over when both sides have many repeats.

The baseline document
---------------------

``evals/baseline.json`` holds one entry per ``(provider, model, effort)`` cell,
keyed ``"<provider>/<model>/<effort>"``, each with its own per-scenario counts
and provenance (every source run URL that contributed samples). The gate
selects the entry matching the cell of the run being judged, so a baseline for
the next production model can be committed ahead of the switch; a run whose
cell has no entry cannot be judged and exits with ``EXIT_ERRORED``. Entries
may pool several result trees (``--from-results`` given more than once), which
is how ``R`` grows beyond a single comparison run.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evals.aggregate import infra_error_label, run_provider

BASELINE_SCHEMA_VERSION = 2

# The only places a baseline may be written without an explicit --output-dir:
# inside this repository (the committed evals/baseline.json).
REPO_ROOT = Path(__file__).resolve().parents[1]

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_ERRORED = 2

REQUIRED_ENTRY_FIELDS = (
    "provider",
    "model",
    "effort",
    "recorded_at",
    "source_run_urls",
    "scenarios",
)
REQUIRED_SCENARIO_FIELDS = (
    "repeats",
    "pass_rate",
    "mean_score",
    "hard_violation_codes",
)
# Optional per-entry provenance for *which prompt* the baseline measured:
# ``{"git_commit": "<sha>", "system_prompt_sha256": "<hex prefix>"}``. The
# writer fills it from the checkout ``--write-baseline`` runs on; the gate reads
# it for the verdict header and tolerates entries recorded before it existed.
PROMPT_VERSION_FIELDS = ("git_commit", "system_prompt_sha256")
PROMPT_SHA256_PREFIX_CHARS = 12
_HEX_RE = re.compile(r"^[0-9a-f]+$")

# Row verdicts. ``pass`` and ``regression`` are the behavioral outcomes;
# ``no_baseline`` (a scenario the entry lacks) and ``baseline_mismatch`` (a run
# spanning several cells) mean the baseline cannot judge this run and are
# treated as failures (exit 1) because the fix is a committed baseline, not a
# rerun; ``no_model_baseline`` (no entry at all for the run's cell) and
# ``errored`` exit 2 so the workflow labels them "cannot judge", never
# "behavioral regression".
VERDICT_PASS = "pass"
VERDICT_REGRESSION = "regression"
VERDICT_NO_BASELINE = "no_baseline"
VERDICT_NO_MODEL_BASELINE = "no_model_baseline"
VERDICT_MISMATCH = "baseline_mismatch"
VERDICT_ERRORED = "errored"

STATUS_PASS = "pass"
STATUS_REGRESSION = "regression"
STATUS_NO_MODEL_BASELINE = "no_model_baseline"
STATUS_ERRORED = "errored"

_RATE_EPSILON = 1e-9

# The new-code rule fires for a code unseen anywhere in the baseline entry only
# when it recurs: in at least this many of the current repeats, or in at least
# this fraction of them (so N = 2 fires at 1 of 2, N = 3 needs 2 of 3, N = 6
# needs 3 of 6). See the module docstring for why one observation is not enough.
NEW_CODE_MIN_REPEATS = 2
NEW_CODE_MIN_FRACTION = 0.5


class BaselineError(ValueError):
    """The baseline document is malformed, or its path is not acceptable."""


def cell_key(provider: str, model: str, effort: str) -> str:
    """The ``baselines`` key for a ``(provider, model, effort)`` cell."""
    return f"{provider}/{model}/{effort}"


# --- path safety --------------------------------------------------------------
#
# ``--baseline`` / ``--write-baseline`` / ``--from-results`` come straight from
# the command line. Every open happens through a resolved ``Path`` that has
# passed these checks, mirroring ``evals.run._path_under`` for output files.


def _reject_parent_components(path: Path, flag: str) -> None:
    if any(part == os.pardir for part in path.parts):
        raise BaselineError(f"{flag} path must not contain '..' components: {path}")


def resolve_baseline_read_path(path: str | Path) -> Path:
    """The resolved ``--baseline`` file: no ``..``, ``.json`` suffix, must exist."""
    raw = Path(path)
    _reject_parent_components(raw, "--baseline")
    resolved = raw.resolve()
    if resolved.suffix != ".json":
        raise BaselineError(f"--baseline must name a .json file: {raw}")
    if not resolved.is_file():
        raise BaselineError(f"--baseline file does not exist: {raw}")
    return resolved


def resolve_baseline_write_path(
    path: str | Path, *, allowed_roots: Iterable[str | Path] = (REPO_ROOT,)
) -> Path:
    """The resolved ``--write-baseline`` file: no ``..``, ``.json``, under an allowed root.

    ``allowed_roots`` is the repository root plus the run's resolved
    ``--output-dir``; anything else raises ``ValueError``.
    """
    raw = Path(path)
    if any(part == os.pardir for part in raw.parts):
        raise ValueError(f"--write-baseline path must not contain '..' components: {raw}")
    resolved = raw.resolve()
    if resolved.suffix != ".json":
        raise ValueError(f"--write-baseline must name a .json file: {raw}")
    roots = [Path(root).resolve() for root in allowed_roots]
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(
            f"Refusing to write baseline outside the repository or the output directory: "
            f"{resolved} (allowed under {', '.join(str(root) for root in roots)})"
        )
    if resolved.is_dir():
        raise ValueError(f"--write-baseline names a directory: {raw}")
    return resolved


def resolve_results_dir(path: str | Path) -> Path:
    """The resolved ``--from-results`` directory; must exist."""
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"--from-results is not a directory: {path}")
    return resolved


def tolerance_margin(current_repeats: int, baseline_repeats: int) -> float:
    """``max(1/N, 1/sqrt(N*R))`` — see the module docstring for the arithmetic."""
    n = int(current_repeats)
    r = int(baseline_repeats)
    if n < 1 or r < 1:
        raise ValueError("repeats must be positive integers")
    return max(1.0 / n, 1.0 / math.sqrt(n * r))


def _hard_codes(result: dict[str, Any] | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for violation in (result or {}).get("violations") or []:
        if str(violation.get("severity") or "hard") == "hard":
            counts[str(violation.get("code") or "unknown")] += 1
    return counts


def _infra_error_details(errored: Iterable[dict[str, Any]]) -> list[str]:
    """Distinct ``<label>: <redacted message> [request_id=…]`` lines; records with no message contribute nothing."""
    details = set()
    for run in errored:
        detail = infra_error_label(run, with_message=True)
        if detail != infra_error_label(run):
            details.add(detail)
    return sorted(details)


def scenario_stats(runs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-scenario counts over run records: repeats, scored, errored, passed, codes.

    ``pass_rate`` and ``mean_score`` are over *scored* runs only. ``cells`` is
    the set of ``(provider, model, effort)`` triples seen, so callers can
    refuse to compare a multi-cell matrix against a single-cell baseline.

    ``hard_violation_codes`` counts violations (a run may raise one code
    twice); ``hard_violation_code_runs`` counts the scored *runs* in which each
    code appears, which is what the new-code rule's recurrence test uses.

    ``infra_error_types`` is keyed by the run's structured ``error``
    (``<ExceptionClass>/<api code or type> (<http status>)``, falling back to
    the bare ``error_type`` for records without one) and ``infra_error_details``
    lists the distinct redacted API messages / request ids, so an errored
    verdict says *why* the provider failed, not just which exception class.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run.get("scenario")), []).append(run)

    stats: dict[str, dict[str, Any]] = {}
    for scenario, members in sorted(grouped.items()):
        scored = [run for run in members if run.get("status") == "scored"]
        errored = [run for run in members if run.get("status") == "errored"]
        passed = [run for run in scored if (run.get("result") or {}).get("passed")]
        scores = [
            float((run.get("result") or {}).get("score"))
            for run in scored
            if (run.get("result") or {}).get("score") is not None
        ]
        codes: Counter[str] = Counter()
        code_runs: Counter[str] = Counter()
        for run in scored:
            run_codes = _hard_codes(run.get("result"))
            codes.update(run_codes)
            code_runs.update(set(run_codes))
        stats[scenario] = {
            "repeats": len(members),
            "scored": len(scored),
            "errored": len(errored),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(scored), 4) if scored else None,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "hard_violation_codes": dict(sorted(codes.items())),
            "hard_violation_code_runs": dict(sorted(code_runs.items())),
            "infra_error_types": dict(
                sorted(Counter(infra_error_label(run) for run in errored).items())
            ),
            "infra_error_details": _infra_error_details(errored),
            "cells": sorted(
                {
                    (run_provider(run), str(run.get("model")), str(run.get("effort")))
                    for run in members
                }
            ),
        }
    return stats


def _single_cell(stats: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    cells = sorted({cell for entry in stats.values() for cell in entry["cells"]})
    if len(cells) != 1:
        raise ValueError(
            "A baseline describes exactly one (provider, model, effort) cell; "
            f"these runs span {len(cells)}: "
            + ", ".join(f"{p}:{m}/{e}" for p, m, e in cells)
        )
    return cells[0]


def _scenario_sources(
    runs: Iterable[dict[str, Any]], default_url: str | None
) -> dict[str, list[dict[str, Any]]]:
    """Per scenario, ``[{source_run_url, repeats, passed}]`` over scored runs, grouped by URL.

    A run record may carry ``source_run_url`` (set by ``evals.run`` when it
    loads a result tree with ``--from-results``); records without one fall back
    to ``default_url``. This is what makes a pooled baseline auditable: the
    reviewer can see that 3/6 is 1/3 from one run plus 2/3 from another.
    """
    grouped: dict[str, dict[str | None, list[int]]] = {}
    for run in runs:
        if run.get("status") != "scored":
            continue
        url = run.get("source_run_url") or default_url
        counts = grouped.setdefault(str(run.get("scenario")), {}).setdefault(url, [0, 0])
        counts[0] += 1
        counts[1] += 1 if (run.get("result") or {}).get("passed") else 0
    return {
        scenario: [
            {"source_run_url": url, "repeats": repeats, "passed": passed}
            for url, (repeats, passed) in sorted(by_url.items(), key=lambda item: item[0] or "")
        ]
        for scenario, by_url in grouped.items()
    }


def build_baseline(
    runs: Iterable[dict[str, Any]],
    *,
    recorded_at: str | None = None,
    source_run_url: str | None = None,
    prompt_version: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A baseline document with one entry: the cell these clean runs describe.

    ``runs`` may pool several result trees for the same cell (the repeats add
    up per scenario); each record's ``source_run_url`` (or the
    ``source_run_url`` argument for records without one) is kept per scenario
    in ``sources`` and collected at the entry level in ``source_run_urls``.

    ``prompt_version`` (``{"git_commit", "system_prompt_sha256"}``, see
    ``PROMPT_VERSION_FIELDS``) records which system prompt the samples
    measured; ``evals.run`` derives it from the checkout it runs on.

    Refuses runs with infrastructure errors: a baseline records ``repeats``
    clean samples per scenario, and an errored sample is neither a pass nor a
    fail. Re-run (or drop the errored provider from the matrix) instead.
    """
    runs = list(runs)
    stats = scenario_stats(runs)
    if not stats:
        raise ValueError("No runs to build a baseline from")
    errored = {name: entry["errored"] for name, entry in stats.items() if entry["errored"]}
    if errored:
        raise ValueError(
            "Refusing to write a baseline from runs with infrastructure errors: "
            + ", ".join(f"{name} ({count} errored)" for name, count in sorted(errored.items()))
        )
    provider, model, effort = _single_cell(stats)
    stamp = recorded_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sources = _scenario_sources(runs, source_run_url)
    scenarios = {
        name: {
            "repeats": entry["scored"],
            "passed": entry["passed"],
            "pass_rate": entry["pass_rate"],
            "mean_score": entry["mean_score"],
            "hard_violation_codes": entry["hard_violation_codes"],
            "sources": sources.get(name, []),
        }
        for name, entry in stats.items()
    }
    urls = sorted(
        {
            source["source_run_url"]
            for per_scenario in sources.values()
            for source in per_scenario
            if source["source_run_url"]
        }
    )
    entry: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "effort": effort,
        "recorded_at": stamp,
        "source_run_urls": urls,
        "scenarios": scenarios,
    }
    if prompt_version is not None:
        _validate_prompt_version(f"baseline.baselines[{cell_key(provider, model, effort)!r}]", prompt_version)
        entry["prompt_version"] = dict(prompt_version)
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baselines": {cell_key(provider, model, effort): entry},
    }


def merge_baselines(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    """``existing`` with every entry of ``incoming`` added or replaced by cell key.

    Re-recording one model's baseline never touches another model's entry, so
    a file can carry the production cell and the next candidate side by side.
    """
    validate_baseline(incoming)
    if existing is None:
        return incoming
    validate_baseline(existing)
    merged = {**existing["baselines"], **incoming["baselines"]}
    return {"schema_version": BASELINE_SCHEMA_VERSION, "baselines": dict(sorted(merged.items()))}


def _validate_scenario(where: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise BaselineError(f"{where} must be an object")
    missing = [field for field in REQUIRED_SCENARIO_FIELDS if field not in entry]
    if missing:
        raise BaselineError(f"{where} is missing {', '.join(missing)}")
    repeats = entry["repeats"]
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise BaselineError(f"{where}.repeats must be a positive integer")
    rate = entry["pass_rate"]
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 <= rate <= 1:
        raise BaselineError(f"{where}.pass_rate must be within [0, 1]")
    codes = entry["hard_violation_codes"]
    if not isinstance(codes, dict) or any(
        not isinstance(count, int) or count < 0 for count in codes.values()
    ):
        raise BaselineError(f"{where}.hard_violation_codes must map code -> count")
    passed = entry.get("passed")
    if passed is not None and (
        not isinstance(passed, int) or isinstance(passed, bool) or not 0 <= passed <= repeats
    ):
        raise BaselineError(f"{where}.passed must be an integer within [0, repeats]")
    sources = entry.get("sources")
    if sources is not None:
        if not isinstance(sources, list):
            raise BaselineError(f"{where}.sources must be a list")
        total = 0
        for source in sources:
            if (
                not isinstance(source, dict)
                or not isinstance(source.get("repeats"), int)
                or isinstance(source.get("repeats"), bool)
                or source["repeats"] < 1
                or not isinstance(source.get("passed"), int)
                or isinstance(source.get("passed"), bool)
                or not 0 <= source["passed"] <= source["repeats"]
                or not (
                    source.get("source_run_url") is None
                    or isinstance(source.get("source_run_url"), str)
                )
            ):
                raise BaselineError(
                    f"{where}.sources entries must be {{source_run_url, repeats, passed}}"
                )
            total += source["repeats"]
        if sources and total != repeats:
            raise BaselineError(f"{where}.sources repeats must add up to {where}.repeats")


def _validate_prompt_version(where: str, value: Any) -> None:
    """``prompt_version`` is optional; when present it must name a commit and a prompt hash."""
    if not isinstance(value, dict):
        raise BaselineError(f"{where}.prompt_version must be an object")
    missing = [field for field in PROMPT_VERSION_FIELDS if field not in value]
    if missing:
        raise BaselineError(f"{where}.prompt_version is missing {', '.join(missing)}")
    commit = value["git_commit"]
    if not isinstance(commit, str) or not 7 <= len(commit) <= 40 or not _HEX_RE.match(commit):
        raise BaselineError(f"{where}.prompt_version.git_commit must be a 7-40 character hex commit id")
    digest = value["system_prompt_sha256"]
    if (
        not isinstance(digest, str)
        or not PROMPT_SHA256_PREFIX_CHARS <= len(digest) <= 64
        or not _HEX_RE.match(digest)
    ):
        raise BaselineError(
            f"{where}.prompt_version.system_prompt_sha256 must be a hex sha256 "
            f"(at least {PROMPT_SHA256_PREFIX_CHARS} characters)"
        )


def prompt_version_matches(recorded: dict[str, Any] | None, current_sha256: str | None) -> bool | None:
    """Whether a recorded ``prompt_version`` hashes the prompt under test.

    ``None`` when either side is unknown (an entry recorded before the field
    existed, or a caller without a live prompt); the gate treats that as
    "cannot say" and never as a mismatch. Prefix comparison, so a full digest
    matches the stored prefix.
    """
    if not recorded or not current_sha256:
        return None
    stored = str(recorded.get("system_prompt_sha256") or "")
    if not stored:
        return None
    shorter = min(len(stored), len(current_sha256))
    return stored[:shorter] == current_sha256[:shorter]


def validate_baseline(document: Any) -> dict[str, Any]:
    """Raise ``BaselineError`` unless ``document`` has the baseline shape; return it."""
    if not isinstance(document, dict):
        raise BaselineError("baseline must be a JSON object")
    if document.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise BaselineError(
            f"baseline schema_version must be {BASELINE_SCHEMA_VERSION}, "
            f"got {document.get('schema_version')!r}; re-record the file with --write-baseline"
        )
    baselines = document.get("baselines")
    if not isinstance(baselines, dict) or not baselines:
        raise BaselineError(
            "baseline.baselines must be a non-empty object keyed by provider/model/effort"
        )
    for key, entry in baselines.items():
        where = f"baseline.baselines[{key!r}]"
        if not isinstance(entry, dict):
            raise BaselineError(f"{where} must be an object")
        missing = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
        if missing:
            raise BaselineError(f"{where} is missing {', '.join(missing)}")
        for field in ("provider", "model", "effort"):
            if not isinstance(entry[field], str) or not entry[field]:
                raise BaselineError(f"{where}.{field} must be a string")
        expected_key = cell_key(entry["provider"], entry["model"], entry["effort"])
        if key != expected_key:
            raise BaselineError(f"{where} describes cell {expected_key!r}; the key must match")
        urls = entry["source_run_urls"]
        if not isinstance(urls, list) or any(not isinstance(url, str) or not url for url in urls):
            raise BaselineError(f"{where}.source_run_urls must be a list of URLs")
        scenarios = entry["scenarios"]
        if not isinstance(scenarios, dict) or not scenarios:
            raise BaselineError(f"{where}.scenarios must be a non-empty object keyed by scenario")
        for name, scenario in scenarios.items():
            _validate_scenario(f"{where}.scenarios[{name!r}]", scenario)
        if "prompt_version" in entry:
            _validate_prompt_version(where, entry["prompt_version"])
    return document


def select_baseline_entry(
    baseline: dict[str, Any], cell: tuple[str, str, str]
) -> dict[str, Any] | None:
    """The entry for ``(provider, model, effort)``, or ``None`` when the file has none."""
    return baseline["baselines"].get(cell_key(*cell))


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Read and validate the baseline at ``path`` (see ``resolve_baseline_read_path``)."""
    resolved = resolve_baseline_read_path(path)
    try:
        document = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read baseline {path}: {exc}") from exc
    return validate_baseline(document)


def write_baseline(
    path: str | Path,
    document: dict[str, Any],
    *,
    allowed_roots: Iterable[str | Path] = (REPO_ROOT,),
    merge: bool = True,
) -> dict[str, Any]:
    """Write ``document`` to ``path`` and return what was written.

    With ``merge`` (the default) an existing file's other cell entries are kept
    and only the cells in ``document`` are replaced; a malformed existing file
    is an error rather than silently overwritten. The path must satisfy
    ``resolve_baseline_write_path``.
    """
    resolved = resolve_baseline_write_path(path, allowed_roots=allowed_roots)
    validate_baseline(document)
    existing = None
    if merge and resolved.is_file():
        try:
            existing = validate_baseline(json.loads(resolved.read_text()))
        except (OSError, json.JSONDecodeError, BaselineError) as exc:
            raise BaselineError(
                f"cannot merge into existing baseline {path}: {exc}; delete it to start over"
            ) from exc
    merged = merge_baselines(existing, document)
    resolved.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return merged


def _recorded_codes(scenario_entry: dict[str, Any]) -> set[str]:
    """Hard violation codes a baseline scenario recorded at least once."""
    return {
        str(code)
        for code, count in (scenario_entry.get("hard_violation_codes") or {}).items()
        if count
    }


def entry_wide_codes(entry: dict[str, Any] | None) -> set[str]:
    """Every hard violation code recorded in any scenario of a baseline entry."""
    codes: set[str] = set()
    for scenario_entry in ((entry or {}).get("scenarios") or {}).values():
        codes |= _recorded_codes(scenario_entry)
    return codes


def new_code_recurs(runs_with_code: int, scored: int) -> bool:
    """Whether an entry-wide-unseen code occurred often enough to be a verdict.

    ``runs_with_code`` is the number of scored current repeats in which the
    code appeared. True at ``>= NEW_CODE_MIN_REPEATS`` runs or at
    ``>= NEW_CODE_MIN_FRACTION`` of the scored runs (so 1 of 2 fires, 1 of 3
    does not, 1 of 1 does).
    """
    if scored < 1 or runs_with_code < 1:
        return False
    return (
        runs_with_code >= NEW_CODE_MIN_REPEATS
        or runs_with_code / scored >= NEW_CODE_MIN_FRACTION - _RATE_EPSILON
    )


def _baseline_rate(entry: dict[str, Any]) -> float:
    # Prefer the exact count when the file has it; the rounded float is a fallback
    # for hand-edited baselines.
    passed = entry.get("passed")
    if isinstance(passed, int) and not isinstance(passed, bool):
        return passed / int(entry["repeats"])
    return float(entry["pass_rate"])


def evaluate_regression(
    baseline: dict[str, Any], stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Apply the rule to ``stats`` (from ``scenario_stats``) against ``baseline``.

    Returns ``{"status", "exit_code", "cell", "rows"}`` where ``status`` is one
    of ``pass`` / ``regression`` / ``no_model_baseline`` / ``errored``, ``cell``
    is the ``provider/model/effort`` key the run was matched on (``None`` when
    the run spans several cells), and each row carries the numbers the verdict
    table prints plus a human-readable ``reason``.

    Per row, ``new_codes`` are the codes that failed the new-code rule (unseen
    in every scenario of the entry *and* recurring, see ``new_code_recurs``);
    ``novel_codes`` maps every entry-wide-unseen code to the number of repeats
    it occurred in (informational when below the bar); ``unseen_here`` lists
    the codes this scenario's own baseline never recorded (informational only).
    """
    validate_baseline(baseline)
    cells = sorted({cell for entry in stats.values() for cell in entry["cells"]})
    key: str | None = None
    entry: dict[str, Any] | None = None
    unjudgeable_verdict: str | None = None
    unjudgeable_reason = ""
    if len(cells) == 1:
        key = cell_key(*cells[0])
        entry = select_baseline_entry(baseline, cells[0])
        if entry is None:
            unjudgeable_verdict = VERDICT_NO_MODEL_BASELINE
            unjudgeable_reason = (
                f"No baseline entry for {key}; the file has "
                + ", ".join(f"`{k}`" for k in sorted(baseline["baselines"]))
                + ". Record one with `python -m evals.run --from-results <artifact> "
                f"--models {cells[0][0]}:{cells[0][1]} --efforts {cells[0][2]} "
                "--write-baseline evals/baseline.json` in the same PR as the model change."
            )
    elif cells:
        unjudgeable_verdict = VERDICT_MISMATCH
        unjudgeable_reason = (
            "The gate judges one (provider, model, effort) cell; this run used "
            + ", ".join(f"{p}:{m}/{e}" for p, m, e in cells)
            + ". Select one cell (--models/--efforts) and judge it on its own."
        )

    entries = (entry or {}).get("scenarios", {})
    known_codes = entry_wide_codes(entry)
    rows: list[dict[str, Any]] = []
    for scenario, current in sorted(stats.items()):
        scenario_entry = entries.get(scenario)
        row: dict[str, Any] = {
            "scenario": scenario,
            "baseline_passed": None,
            "baseline_repeats": None,
            "baseline_pass_rate": None,
            "current_passed": current["passed"],
            "current_scored": current["scored"],
            "current_repeats": current["repeats"],
            "current_pass_rate": current["pass_rate"],
            "errored": current["errored"],
            "infra_errors": list(current.get("infra_error_details") or []),
            "margin": None,
            "threshold": None,
            "new_codes": [],
            "novel_codes": {},
            "unseen_here": [],
            "code_runs": dict(current.get("hard_violation_code_runs") or {}),
            "verdict": VERDICT_PASS,
            "reason": "",
        }
        if current["errored"]:
            row["verdict"] = VERDICT_ERRORED
            types = ", ".join(f"{k} x{v}" for k, v in current["infra_error_types"].items())
            row["reason"] = (
                f"{current['errored']} of {current['repeats']} conversation(s) hit an "
                f"infrastructure error ({types}); not a behavioral verdict."
            )
            if row["infra_errors"]:
                # The provider's own diagnosis (redacted message, request id) from
                # the run record's structured ``error``: a 401 reads as "incorrect
                # key" vs "no access to model" without opening the artifact.
                row["reason"] += " Provider said: " + " | ".join(row["infra_errors"]) + "."
        if unjudgeable_verdict is not None:
            if row["verdict"] == VERDICT_PASS:
                row["verdict"] = unjudgeable_verdict
                row["reason"] = unjudgeable_reason
            rows.append(row)
            continue
        if scenario_entry is None:
            if row["verdict"] == VERDICT_PASS:
                row["verdict"] = VERDICT_NO_BASELINE
                row["reason"] = (
                    f"No baseline entry for this scenario under {key}; record one with "
                    "`python -m evals.run --write-baseline evals/baseline.json`."
                )
            rows.append(row)
            continue

        baseline_rate = _baseline_rate(scenario_entry)
        row["baseline_repeats"] = int(scenario_entry["repeats"])
        row["baseline_passed"] = (
            scenario_entry.get("passed")
            if isinstance(scenario_entry.get("passed"), int)
            else round(baseline_rate * int(scenario_entry["repeats"]))
        )
        row["baseline_pass_rate"] = round(baseline_rate, 4)

        if current["scored"]:
            margin = tolerance_margin(current["scored"], int(scenario_entry["repeats"]))
            threshold = baseline_rate - margin
            row["margin"] = round(margin, 4)
            row["threshold"] = round(max(0.0, threshold), 4)
            current_codes = set(current["hard_violation_codes"])
            # Informational: codes this scenario's own baseline never recorded.
            row["unseen_here"] = sorted(current_codes - _recorded_codes(scenario_entry))
            # The rule: codes unseen in *every* scenario of the entry, with the
            # number of current repeats each occurred in.
            code_runs = row["code_runs"] or current["hard_violation_codes"]
            novel = {
                code: int(code_runs.get(code, 0)) for code in sorted(current_codes - known_codes)
            }
            row["novel_codes"] = novel
            new_codes = sorted(
                code for code, runs in novel.items() if new_code_recurs(runs, current["scored"])
            )
            row["new_codes"] = new_codes
            reasons: list[str] = []
            notes: list[str] = []
            # Exact fraction, not the 4-decimal ``pass_rate``: 1/3 must not read as
            # below a 1/3 threshold because 0.3333 < 0.33333.
            current_rate = current["passed"] / current["scored"]
            if current_rate < threshold - _RATE_EPSILON:
                reasons.append(
                    f"pass rate {current['passed']}/{current['scored']} is below "
                    f"{row['threshold']:.3f} (baseline {row['baseline_passed']}/"
                    f"{scenario_entry['repeats']} minus margin {margin:.3f})"
                )
            if new_codes:
                reasons.append(
                    "hard violation code(s) never recorded in any scenario of the baseline "
                    "entry recurred: "
                    + ", ".join(f"`{code}` in {novel[code]}/{current['scored']} repeats" for code in new_codes)
                )
            below_bar = [code for code in novel if code not in new_codes]
            if below_bar:
                notes.append(
                    ", ".join(f"`{code}`" for code in below_bar)
                    + " is not in any scenario of the baseline entry but occurred in only "
                    + ", ".join(f"{novel[code]}/{current['scored']}" for code in below_bar)
                    + f" repeat(s), below the {NEW_CODE_MIN_REPEATS}-repeat / "
                    f"{NEW_CODE_MIN_FRACTION:.0%} bar; the pass-rate rule governs it"
                )
            known_unseen_here = [code for code in row["unseen_here"] if code in known_codes]
            if known_unseen_here:
                notes.append(
                    "unseen in this scenario's baseline but a known failure mode elsewhere in the "
                    "entry: " + ", ".join(f"`{code}`" for code in known_unseen_here)
                    + "; the pass-rate rule governs it"
                )
            if reasons and row["verdict"] == VERDICT_PASS:
                row["verdict"] = VERDICT_REGRESSION
                row["reason"] = "; ".join(reasons) + "."
            elif reasons:
                # Errored rows keep their infra reason but still show the numbers.
                row["reason"] += " Also: " + "; ".join(reasons) + "."
            if notes and row["verdict"] == VERDICT_PASS:
                row["reason"] = "Informational: " + "; ".join(notes) + "."
            elif notes:
                row["reason"] += " Informational: " + "; ".join(notes) + "."
        rows.append(row)

    if any(row["verdict"] == VERDICT_ERRORED for row in rows):
        status, exit_code = STATUS_ERRORED, EXIT_ERRORED
    elif any(row["verdict"] == VERDICT_NO_MODEL_BASELINE for row in rows):
        status, exit_code = STATUS_NO_MODEL_BASELINE, EXIT_ERRORED
    elif any(row["verdict"] != VERDICT_PASS for row in rows):
        status, exit_code = STATUS_REGRESSION, EXIT_REGRESSION
    else:
        status, exit_code = STATUS_PASS, EXIT_PASS
    return {
        "status": status,
        "exit_code": exit_code,
        "cell": key,
        # Which prompt the matched entry measured (None for entries recorded
        # before the field existed, or when no entry matched).
        "baseline_prompt_version": (entry or {}).get("prompt_version"),
        "rows": rows,
    }


def _prompt_version_line(recorded: dict[str, Any] | None, current_prompt_sha256: str | None) -> str:
    if not recorded:
        return "- Baseline prompt: not recorded (entry predates `prompt_version`; re-baseline to add it)"
    line = (
        f"- Baseline prompt: commit `{str(recorded['git_commit'])[:12]}`, system prompt sha256 "
        f"`{recorded['system_prompt_sha256']}`"
    )
    match = prompt_version_matches(recorded, current_prompt_sha256)
    if match is True:
        line += " (same prompt as this run)"
    elif match is False:
        line += (
            f" — **differs from the prompt under test** "
            f"(`{str(current_prompt_sha256)[:PROMPT_SHA256_PREFIX_CHARS]}`): this baseline measured "
            "another prompt; re-baseline in this PR if the prompt change is intended"
        )
    return line


def _rate_cell(passed: int | None, repeats: int | None) -> str:
    if passed is None or not repeats:
        return "n/a"
    return f"{passed}/{repeats} ({passed / repeats * 100:.0f}%)"


def _unseen_cell(row: dict[str, Any], code: str) -> str:
    """`` `code` k/N``: how many current repeats the scenario-unseen code occurred in."""
    runs = (row.get("code_runs") or {}).get(code)
    if runs is None or not row.get("current_scored"):
        return f"`{code}`"
    return f"`{code}` {runs}/{row['current_scored']}"


def render_verdict_markdown(
    verdict: dict[str, Any],
    *,
    baseline_path: str | None = None,
    current_prompt_sha256: str | None = None,
) -> str:
    """The verdict table: scenario | baseline pass | current pass | threshold | new codes | unseen here | verdict.

    ``New codes`` are the codes that failed the new-code rule; ``Unseen here``
    lists codes this scenario's own baseline never recorded, each with the
    number of current repeats it occurred in (informational, never a verdict).
    ``current_prompt_sha256`` (the hash of the prompt this run sent) lets the
    header say whether the matched entry measured the same prompt; it is
    informational and never changes the verdict.
    """
    status = verdict["status"]
    headline = {
        STATUS_PASS: "PASS",
        STATUS_REGRESSION: "REGRESSION",
        STATUS_NO_MODEL_BASELINE: "NO BASELINE FOR THIS MODEL (cannot judge)",
        STATUS_ERRORED: "ERRORED (infrastructure)",
    }[status]
    lines = [f"## Conversation evaluation gate: {headline}", ""]
    if baseline_path:
        lines.append(f"- Baseline: `{baseline_path}`")
    if verdict.get("cell"):
        lines.append(f"- Baseline entry: `{verdict['cell']}`")
        if status != STATUS_NO_MODEL_BASELINE:
            lines.append(_prompt_version_line(verdict.get("baseline_prompt_version"), current_prompt_sha256))
    lines.extend(
        [
            "- Rule: a scenario regresses when its pass rate drops below "
            "`baseline - max(1/N, 1/sqrt(N*R))`, or when a hard violation code that no "
            "scenario of the baseline entry ever recorded occurs in at least "
            f"{NEW_CODE_MIN_REPEATS} of its N repeats (or in at least "
            f"{NEW_CODE_MIN_FRACTION:.0%} of them). `Unseen here` lists codes this "
            "scenario's own baseline lacks (informational). Infrastructure errors are "
            "labelled and never counted as regressions (exit code 2).",
            "",
            "| Scenario | Baseline pass | Current pass | Threshold | New codes | Unseen here | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in verdict["rows"]:
        new_codes = ", ".join(f"`{code}`" for code in row["new_codes"]) or "-"
        unseen_here = ", ".join(_unseen_cell(row, code) for code in row.get("unseen_here") or []) or "-"
        threshold = "n/a" if row["threshold"] is None else f"{row['threshold']:.3f}"
        current = _rate_cell(row["current_passed"], row["current_scored"])
        if row["errored"]:
            current += f" +{row['errored']} errored"
        lines.append(
            f"| `{row['scenario']}` | {_rate_cell(row['baseline_passed'], row['baseline_repeats'])} | "
            f"{current} | {threshold} | {new_codes} | {unseen_here} | **{row['verdict']}** |"
        )
    reasons = [row for row in verdict["rows"] if row["reason"]]
    if reasons:
        lines.append("")
        for row in reasons:
            lines.append(f"- `{row['scenario']}`: {row['reason']}")
    return "\n".join(lines) + "\n"


def load_runs_from_dir(
    root: str | Path, *, source_run_url: str | None = None
) -> list[dict[str, Any]]:
    """Every ``rep<k>.json`` run record under ``root`` (an ``evals.run`` output tree).

    ``root`` must be an existing directory (``resolve_results_dir``); records
    are read through the resolved path. When ``source_run_url`` is given every
    record is tagged with it so a pooled baseline can attribute samples to the
    run that produced them.
    """
    resolved = resolve_results_dir(root)
    records: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("rep*.json")):
        try:
            record = json.loads(path.resolve().read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read run record {path}: {exc}") from exc
        if isinstance(record, dict) and "scenario" in record and "status" in record:
            if source_run_url:
                record["source_run_url"] = source_run_url
            records.append(record)
    if not records:
        raise ValueError(f"No run records (rep*.json) found under {root}")
    return records


def select_cell(
    runs: list[dict[str, Any]], *, provider: str | None, model: str | None, effort: str | None
) -> list[dict[str, Any]]:
    """Keep the runs matching whichever of provider/model/effort are given."""
    kept = []
    for run in runs:
        if provider is not None and run_provider(run) != provider:
            continue
        if model is not None and str(run.get("model")) != model:
            continue
        if effort is not None and str(run.get("effort")) != effort:
            continue
        kept.append(run)
    return kept
