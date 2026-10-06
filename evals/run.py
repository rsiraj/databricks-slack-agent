"""CLI for the model-in-the-loop PR evaluation and model/effort comparisons.

Default invocation (``python -m evals.run``) runs one model, one effort, every
fixture in ``evals/fixtures`` once and exits 1 on any behavioral violation. The
PR gate instead passes ``--repeats 3 --baseline evals/baseline.json``: the exit
code then comes from the baseline-relative regression rule in
``evals.regression`` (0 pass, 1 regression, 2 infrastructure error) rather
than from "any failed conversation". ``--write-baseline PATH`` records a run
(or, with ``--from-results DIR`` given once or several times, one or more
existing artifact trees pooled together) as the baseline entry for that run's
(provider, model, effort) cell, keeping any other cells already in the file.
The same command accepts a matrix of models, efforts and repeats and writes
per-run traces plus an aggregate comparison table.

Exit codes: 0 pass; 1 behavioral failure/regression (or a baseline that cannot
judge the run); 2 infrastructure error (``EXIT_ERRORED``), always distinct so a
workflow can label a red check "infra" rather than "regression".

``--models`` entries may be provider-qualified (``anthropic:claude-sonnet-4-6``,
``openai:gpt-5.6-terra``) so one matrix can compare vendors head-to-head; bare
ids fall back to ``--provider``. One SDK client is built per provider that the
matrix needs (``MODEL_EVAL_ANTHROPIC_API_KEY`` / ``MODEL_EVAL_OPENAI_API_KEY``);
a provider whose client cannot be built (missing key or package) has every one
of its cells recorded as ``errored``, never as a behavioral failure, while the
other providers' cells still run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import anthropic

from evals.aggregate import aggregate_runs, infra_error_label, render_summary_markdown, run_provider
from evals.regression import (
    EXIT_ERRORED,
    EXIT_REGRESSION,
    PROMPT_SHA256_PREFIX_CHARS,
    REPO_ROOT,
    STATUS_ERRORED,
    STATUS_NO_MODEL_BASELINE,
    STATUS_REGRESSION,
    BaselineError,
    build_baseline,
    evaluate_regression,
    load_baseline,
    load_runs_from_dir,
    render_verdict_markdown,
    resolve_baseline_write_path,
    scenario_stats,
    select_cell,
    write_baseline,
)
from evals.conversation_runner import (
    CLIENT_FACTORIES,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    DEFAULT_THINKING,
    MAX_API_CALLS,
    MAX_TOKENS_PER_CALL,
    build_client,
    default_model_for,
    run_conversation,
)
from evals.scoring import score_trace
from evals.synthetic_tools import FIXTURES_DIR, load_fixture
from utils.llm import DEFAULT_PROVIDER, SUPPORTED_PROVIDERS
from utils.system_prompt import load_system_prompt

MAX_CONVERSATION_ATTEMPTS = 3
_COMMIT_ID_RE = re.compile(r"^[0-9a-f]{7,40}$")
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_MAX_DELAY_SECONDS = 30.0
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_CONNECTION_ERROR_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})
_UNSAFE_COMPONENT_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")
_DOT_RUN_RE = re.compile(r"\.{2,}")
# Anything that looks like an API key or bearer token is redacted before an
# error message is written to an artifact: OpenAI's 401 text echoes a (partly
# masked) key, and the run records are uploaded as workflow artifacts. ``*`` is
# in the class so a masked ``sk-proj-****abcd`` collapses whole, not just its
# unmasked prefix.
_API_KEY_RE = re.compile(r"sk-[A-Za-z0-9_*\-]{8,}")
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9_.*\-]{8,}")
ERROR_MESSAGE_MAX_CHARS = 300


def system_prompt_sha256(provider: str = DEFAULT_PROVIDER) -> str:
    """Full hex sha256 of the system prompt this checkout sends for ``provider``.

    The baseline stores its first ``PROMPT_SHA256_PREFIX_CHARS`` characters so
    a reader can tell which prompt an entry measured; the gate compares the
    prefix against this value for the verdict header.
    """
    return hashlib.sha256(load_system_prompt(provider).encode("utf-8")).hexdigest()


def resolve_prompt_commit(explicit: str | None = None) -> str | None:
    """The commit whose prompts this checkout holds, for ``prompt_version.git_commit``.

    ``--prompt-commit`` wins (a 7-40 character hex id, ``ValueError`` otherwise),
    then ``GITHUB_SHA`` in Actions, then ``git rev-parse HEAD``; ``None`` when
    none of those is available (no git, not a checkout).
    """
    if explicit:
        candidate = explicit.strip().lower()
        if not _COMMIT_ID_RE.match(candidate):
            raise ValueError(f"--prompt-commit must be a 7-40 character hex commit id, got {explicit!r}")
        return candidate
    candidate = (os.environ.get("GITHUB_SHA") or "").strip().lower()
    if not candidate:
        try:
            candidate = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
        except (OSError, subprocess.CalledProcessError):
            return None
    return candidate if _COMMIT_ID_RE.match(candidate) else None


def prompt_version_for(provider: str, *, commit: str | None = None) -> dict[str, str] | None:
    """``{"git_commit", "system_prompt_sha256"}`` for a baseline entry, or ``None`` without a commit id."""
    resolved = resolve_prompt_commit(commit)
    if resolved is None:
        return None
    return {
        "git_commit": resolved,
        "system_prompt_sha256": system_prompt_sha256(provider)[:PROMPT_SHA256_PREFIX_CHARS],
    }


def redact_secrets(text: str) -> str:
    """Replace key-shaped tokens (``sk-…``) and bearer credentials in ``text``."""
    text = _API_KEY_RE.sub("sk-…", text)
    return _BEARER_RE.sub("Bearer …", text)


def scrub_error_message(exc: BaseException | str | None) -> str:
    """One-line, key-redacted, length-capped message for run records and summaries.

    Accepts the exception itself (``str(exc)``) or an already-extracted API
    error message string.
    """
    if exc is None:
        return ""
    text = " ".join(str(exc).split())
    text = redact_secrets(text)
    if len(text) > ERROR_MESSAGE_MAX_CHARS:
        text = text[: ERROR_MESSAGE_MAX_CHARS - 1] + "…"
    return text


def _error_body(exc: BaseException) -> dict[str, Any]:
    """The API error object inside an SDK exception's ``body``, whatever the nesting.

    OpenAI hands its exceptions the inner ``{"message", "type", "code", "param"}``
    object; Anthropic hands the full envelope ``{"type": "error", "error": {...}}``.
    Both are reduced to the inner object; anything else yields ``{}``.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}
    inner = body.get("error")
    return inner if isinstance(inner, dict) else body


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def describe_error(exc: BaseException | None) -> dict[str, Any]:
    """Structured, artifact-safe description of an infrastructure error.

    Works for both provider SDKs without importing either's error classes
    (``openai`` is optional): ``status_code`` and ``request_id`` are read off the
    exception, ``code``/``type`` from the exception attributes or the API error
    body (OpenAI: ``code`` such as ``invalid_api_key`` / ``insufficient_quota`` /
    ``model_not_found`` plus ``type``; Anthropic: ``type`` such as
    ``authentication_error``), and ``message`` is the API error message when the
    body has one, else ``str(exc)``, redacted and capped. A 401 is then
    diagnosable from the artifact alone: "incorrect key", "missing scope" and
    "no access to model" are the same exception class with different codes.
    """
    if exc is None:
        return {
            "error_type": "UnknownError",
            "status_code": None,
            "code": None,
            "type": None,
            "message": "",
            "request_id": None,
        }
    body = _error_body(exc)
    status = getattr(exc, "status_code", None)
    code = _str_or_none(getattr(exc, "code", None)) or _str_or_none(body.get("code"))
    error_type = _str_or_none(getattr(exc, "type", None)) or _str_or_none(body.get("type"))
    message = _str_or_none(body.get("message")) or str(exc)
    return {
        "error_type": type(exc).__name__,
        "status_code": status if isinstance(status, int) else None,
        "code": code,
        "type": error_type,
        "message": scrub_error_message(message),
        "request_id": _str_or_none(getattr(exc, "request_id", None)),
    }


def available_scenarios() -> list[str]:
    return sorted(path.stem for path in FIXTURES_DIR.glob("*.json"))


def validate_scenarios(names: list[str]) -> list[str]:
    """Return ``names`` only if every entry names an existing ``evals/fixtures/<name>.json``.

    CLI-provided scenario names are matched against the fixture files on disk
    before anything is opened, so an arbitrary string never reaches ``open()``
    or an output path.
    """
    known = set(available_scenarios())
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(
            f"Unknown scenario(s): {', '.join(repr(name) for name in unknown)}. "
            f"Available: {', '.join(sorted(known)) or '(none)'}"
        )
    return list(names)


def parse_model_spec(spec: str, default_provider: str = DEFAULT_PROVIDER) -> tuple[str, str]:
    """Split a ``--models`` entry into ``(provider, model)``.

    ``"openai:gpt-5.6-terra"`` -> ``("openai", "gpt-5.6-terra")``; a bare
    ``"claude-sonnet-4-6"`` -> ``(default_provider, "claude-sonnet-4-6")``. Only
    the first colon splits, and only when the prefix is a supported provider, so
    a typo such as ``opnai:gpt-5.6`` is rejected instead of being sent to the
    default provider as a model id.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Model spec must be a non-empty string")
    spec = spec.strip()
    if ":" not in spec:
        return default_provider, spec
    prefix, _colon, model = spec.partition(":")
    prefix = prefix.strip().lower()
    model = model.strip()
    if prefix not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider {prefix!r} in model spec {spec!r}; "
            f"expected one of {', '.join(SUPPORTED_PROVIDERS)}"
        )
    if not model:
        raise ValueError(f"Model spec {spec!r} names a provider but no model id")
    return prefix, model


def parse_model_specs(
    specs: list[str], default_provider: str = DEFAULT_PROVIDER
) -> list[tuple[str, str]]:
    """``parse_model_spec`` over a list, preserving order and dropping exact duplicates."""
    seen: set[tuple[str, str]] = set()
    parsed: list[tuple[str, str]] = []
    for spec in specs:
        pair = parse_model_spec(spec, default_provider)
        if pair not in seen:
            seen.add(pair)
            parsed.append(pair)
    return parsed


class ClientUnavailable(RuntimeError):
    """Placeholder client for a provider whose SDK client could not be built.

    Carries the original failure so every cell for that provider is recorded as
    ``errored`` with the underlying exception class (``RuntimeError`` for a
    missing key, ``ProviderNotConfigured`` for a missing package), while cells
    for other providers in the same matrix still run.
    """

    def __init__(self, provider: str, cause: BaseException) -> None:
        super().__init__(f"{provider} client unavailable: {type(cause).__name__}: {cause}")
        self.provider = provider
        self.cause = cause


def _safe_component(value: str) -> str:
    """Reduce a CLI-derived name to a single, traversal-free path component.

    Keeps ``[A-Za-z0-9._-]``, replaces everything else with ``_``, strips
    leading dots and collapses any remaining ``..`` run. Raises ``ValueError``
    when nothing usable is left.
    """
    if not isinstance(value, str):
        raise ValueError(f"Path component must be a string, got {type(value).__name__}")
    cleaned = _UNSAFE_COMPONENT_CHARS_RE.sub("_", value).lstrip(".")
    cleaned = _DOT_RUN_RE.sub("_", cleaned)
    if not cleaned:
        raise ValueError(f"Path component {value!r} is empty after sanitization")
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned or cleaned in (".", os.sep, os.pardir):
        raise ValueError(f"Path component {value!r} is not a safe single path segment")
    return cleaned


def _path_under(root: Path, *parts: str) -> Path:
    """Join ``parts`` onto the resolved ``root`` and refuse anything that escapes it."""
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError(f"Refusing to write outside output directory {root}: {candidate}")
    return candidate


def is_retryable_error(exc: BaseException) -> bool:
    """Infrastructure failures worth re-running the whole conversation for."""
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None) or 0
        return status == 429 or status >= 500
    # APITimeoutError subclasses APIConnectionError.
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    # Other SDKs (openai) use the same class names and ``status_code`` attribute
    # without sharing a base class; match by shape so ``--provider openai``
    # retries the same failure classes.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ in _CONNECTION_ERROR_NAMES


def _backoff_seconds(attempt: int, rng: Callable[[], float]) -> float:
    return min(RETRY_BASE_DELAY_SECONDS * (2**attempt) + rng(), RETRY_MAX_DELAY_SECONDS)


def run_one(
    *,
    scenario: str,
    model: str,
    effort: str | None,
    rep: int,
    client: Any,
    provider: str = DEFAULT_PROVIDER,
    thinking: str | None = DEFAULT_THINKING,
    max_tokens: int = MAX_TOKENS_PER_CALL,
    max_api_calls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    max_attempts: int = MAX_CONVERSATION_ATTEMPTS,
    fixture_loader: Callable[[str], dict[str, Any]] = load_fixture,
) -> dict[str, Any]:
    """Run and score one conversation, retrying only on infrastructure errors.

    Returns a run record with ``status`` ``"scored"`` or ``"errored"``. Errored
    runs carry the exception class name and are never behavioral failures.
    ``max_api_calls=None`` keeps the fixture's or production's budget.
    """
    record: dict[str, Any] = {
        "scenario": scenario,
        "provider": provider,
        "model": model,
        "effort": effort,
        "rep": rep,
        "attempts": 0,
    }
    last_error: BaseException | None = None
    if isinstance(client, ClientUnavailable):
        # No SDK client for this provider (missing key/package): an infrastructure
        # error, recorded without attempting the conversation.
        last_error = client.cause
        max_attempts = 0
    for attempt in range(max_attempts):
        record["attempts"] = attempt + 1
        try:
            fixture = fixture_loader(scenario)
            trace = run_conversation(
                copy.deepcopy(fixture),
                client=client,
                provider=provider,
                model=model,
                effort=effort,
                thinking=thinking,
                max_tokens=max_tokens,
                max_api_calls=max_api_calls,
            )
            result = score_trace(trace, fixture)
        except Exception as exc:  # noqa: BLE001 - classified below
            last_error = exc
            if is_retryable_error(exc) and attempt + 1 < max_attempts:
                sleep(_backoff_seconds(attempt, rng))
                continue
            break
        record.update({"status": "scored", "trace": trace, "result": result})
        return record

    error = describe_error(last_error)
    record.update(
        {
            "status": "errored",
            # ``error_type`` stays at the top level for the aggregate histogram and
            # older readers; ``error`` carries the provider's own diagnosis.
            "error_type": error["error_type"],
            "error": error,
            "retryable": bool(last_error is not None and is_retryable_error(last_error)),
            "trace": {
                "scenario": scenario,
                "provider": provider,
                "model": model,
                "effort": effort,
                "stop_reason": "infrastructure_error",
                "error_type": error["error_type"],
            },
            "result": None,
        }
    )
    return record


def run_matrix(
    *,
    models: list[str],
    efforts: list[str | None],
    scenarios: list[str],
    repeats: int,
    client: Any = None,
    clients: Mapping[str, Any] | None = None,
    concurrency: int = 2,
    provider: str = DEFAULT_PROVIDER,
    thinking: str | None = DEFAULT_THINKING,
    max_tokens: int = MAX_TOKENS_PER_CALL,
    max_api_calls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    run_fn: Callable[..., dict[str, Any]] = run_one,
) -> list[dict[str, Any]]:
    """Run every (provider, model, effort, scenario, rep) combination, bounded by ``concurrency``.

    ``models`` entries may be provider-qualified (see ``parse_model_spec``);
    bare ids use ``provider``. Each cell is run with ``clients[<provider>]``
    when ``clients`` is given (a ``ClientUnavailable`` value marks the whole
    provider as errored), otherwise with ``client``.
    """
    specs = parse_model_specs(models, provider)
    jobs = [
        {"provider": cell_provider, "model": model, "effort": effort, "scenario": scenario, "rep": rep}
        for cell_provider, model in specs
        for effort in efforts
        for scenario in scenarios
        for rep in range(1, repeats + 1)
    ]

    def client_for(cell_provider: str) -> Any:
        if clients is None:
            return client
        if cell_provider in clients:
            return clients[cell_provider]
        return ClientUnavailable(
            cell_provider, RuntimeError(f"No client was built for provider {cell_provider!r}")
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(
                run_fn,
                client=client_for(job["provider"]),
                thinking=thinking,
                max_tokens=max_tokens,
                max_api_calls=max_api_calls,
                sleep=sleep,
                **job,
            )
            for job in jobs
        ]
        return [future.result() for future in futures]


def run_output_path(output_root: Path, run: dict[str, Any]) -> Path:
    """Filesystem location for one run record: ``<root>/<model>/<effort>/<scenario>/rep<k>.json``.

    Directory names use the sanitized form of the model, effort and scenario;
    the JSON payload keeps the original values. The result is verified to stay
    under ``output_root``.
    """
    effort = run["effort"]
    rep = int(run["rep"])
    return _path_under(
        output_root,
        _safe_component(str(run["model"])),
        _safe_component("none" if effort is None else str(effort)),
        _safe_component(str(run["scenario"])),
        f"rep{rep}.json",
    )


def write_outputs(output_dir: Path, runs: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Write per-run JSON, summary.json and summary.md; return (summary, markdown)."""
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        rep_path = run_output_path(output_root, run)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(run, indent=2, default=str) + "\n")

    summary = aggregate_runs(runs)
    markdown = render_summary_markdown(summary, runs)
    _path_under(output_root, "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    _path_under(output_root, "summary.md").write_text(markdown)

    if len(runs) == 1:
        # Preserve the single-run artifact layout the CI gate has always produced.
        only = runs[0]
        _path_under(output_root, "trace.json").write_text(
            json.dumps(only.get("trace"), indent=2, default=str) + "\n"
        )
        result = only.get("result") or {
            "scenario": only["scenario"],
            "passed": False,
            "status": "errored",
            "violations": [
                {
                    "code": "evaluation_infrastructure",
                    "message": (
                        f"Evaluation could not run ({infra_error_label(only, with_message=True)})."
                    ),
                    "severity": "hard",
                }
            ],
        }
        _path_under(output_root, "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return summary, markdown


def _csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=DEFAULT_PROVIDER,
        help=(
            "Model provider. 'openai' needs the openai package and "
            "$MODEL_EVAL_OPENAI_API_KEY; models default to that provider's production default."
        ),
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated model ids, optionally provider-qualified as <provider>:<model> "
            "(e.g. anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra); bare ids use --provider "
            "(default: $MODEL_EVAL_MODEL or the provider's production default)."
        ),
    )
    parser.add_argument(
        "--efforts",
        help=(
            "Comma-separated effort levels from "
            f"{', '.join(EFFORT_LEVELS)} (default: $MODEL_EVAL_EFFORT or 'high')."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1, help="Runs per (model, effort, scenario).")
    parser.add_argument(
        "--scenarios",
        help="Comma-separated fixture names (default: every fixture in evals/fixtures).",
    )
    parser.add_argument("--scenario", help="Legacy single-scenario alias for --scenarios.")
    parser.add_argument("--output-dir", type=Path, default=Path("eval-results"))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--thinking",
        choices=["adaptive", "disabled", "none"],
        default=DEFAULT_THINKING,
        help="Thinking mode; 'none' omits the parameter entirely.",
    )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_CALL)
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=None,
        help=(
            "Per-conversation model-call budget for comparison runs. Overrides a "
            f"fixture's max_api_calls; default is production's worker.MAX_ITERATIONS ({MAX_API_CALLS})."
        ),
    )
    parser.add_argument(
        "--fail-on-violation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit 1 when any scored conversation fails (default; CI gate behavior).",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="Print fixtures and exit.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Judge this run against a committed baseline (evals/baseline.json) with the "
            "regression rule in evals.regression instead of failing on any violation. "
            "Exit 0 pass, 1 regression, 2 infrastructure error. Requires a single "
            "(provider, model, effort) cell."
        ),
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Record this run's per-scenario pass rates and hard violation codes as the "
            "baseline entry for its (provider, model, effort) cell at PATH, keeping other "
            "cells already in the file (implies --no-fail-on-violation; refuses runs with "
            "infrastructure errors). PATH must be a .json file under the repository or "
            "--output-dir. Commit the file in the same PR as the prompt/model change."
        ),
    )
    parser.add_argument(
        "--from-results",
        type=Path,
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "Do not call any model: load rep*.json run records from an existing evals.run "
            "output tree (for example a downloaded model-comparison artifact) and apply "
            "--baseline / --write-baseline to them. Repeat the flag to pool several trees "
            "of the same cell into one baseline entry (repeats add up per scenario). "
            "--models/--efforts select one cell when the trees hold several."
        ),
    )
    parser.add_argument(
        "--source-run-url",
        action="append",
        default=None,
        metavar="URL",
        help=(
            "Provenance recorded in the baseline (default: the GitHub Actions run URL when "
            "running in Actions, else none). With several --from-results, give one URL per "
            "directory in the same order, or a single URL for all of them."
        ),
    )
    parser.add_argument(
        "--prompt-commit",
        default=None,
        metavar="SHA",
        help=(
            "With --write-baseline: the commit whose prompts the recorded runs measured, stored "
            "as prompt_version.git_commit next to the sha256 of load_system_prompt() from this "
            "checkout (default: GITHUB_SHA in Actions, else git rev-parse HEAD). Run "
            "--write-baseline on the commit the comparison ran on so the hash matches."
        ),
    )
    return parser


def build_clients(
    providers: list[str],
    *,
    client_factory: Callable[[], Any] | None = None,
    client_factories: Mapping[str, Callable[[], Any]] = CLIENT_FACTORIES,
) -> dict[str, Any]:
    """One SDK client per provider; a failed build becomes a ``ClientUnavailable`` marker.

    ``client_factory`` (tests) is used for every provider when given; otherwise
    each provider's own factory from ``client_factories`` runs. Nothing raises:
    the caller decides whether *every* provider failing aborts the run.
    """
    clients: dict[str, Any] = {}
    for name in providers:
        factory = client_factory or client_factories.get(name, build_client)
        try:
            clients[name] = factory()
        except Exception as exc:  # noqa: BLE001 - surfaced per cell as an infra error
            clients[name] = ClientUnavailable(name, exc)
    return clients


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """CLI entry point. ``client_factory`` (tests) replaces every provider's SDK client builder."""
    args = build_parser().parse_args(argv)
    if args.list_scenarios:
        print("\n".join(available_scenarios()))
        return 0

    provider = args.provider
    models = _csv(
        args.models, [os.environ.get("MODEL_EVAL_MODEL") or default_model_for(provider)]
    )
    try:
        specs = parse_model_specs(models, provider)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    efforts = _csv(args.efforts, [os.environ.get("MODEL_EVAL_EFFORT") or DEFAULT_EFFORT])
    scenarios = _csv(args.scenarios, [args.scenario] if args.scenario else available_scenarios())
    try:
        scenarios = validate_scenarios(scenarios)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    invalid_efforts = sorted(set(efforts) - set(EFFORT_LEVELS))
    if invalid_efforts:
        raise SystemExit(f"Unknown effort level(s): {', '.join(invalid_efforts)}")
    thinking = None if args.thinking == "none" else args.thinking
    if args.max_api_calls is not None and args.max_api_calls < 1:
        raise SystemExit("--max-api-calls must be a positive integer")
    output_root = Path(args.output_dir).resolve()

    baseline: dict[str, Any] | None = None
    if args.baseline is not None:
        try:
            baseline = load_baseline(args.baseline)
        except BaselineError as exc:
            raise SystemExit(f"--baseline: {exc}") from None
    if args.baseline is not None and args.write_baseline is not None:
        raise SystemExit("--baseline and --write-baseline are mutually exclusive")
    # Validate the destination before spending anything on model calls.
    write_path: Path | None = None
    if args.write_baseline is not None:
        try:
            write_path = resolve_baseline_write_path(
                args.write_baseline, allowed_roots=(REPO_ROOT, output_root)
            )
        except ValueError as exc:
            raise SystemExit(f"--write-baseline: {exc}") from None
    source_run_urls: list[str] = list(args.source_run_url or [])
    if args.from_results is None and len(source_run_urls) > 1:
        raise SystemExit("--source-run-url may be given once for a live run")
    if args.prompt_commit is not None:
        if write_path is None:
            raise SystemExit("--prompt-commit only applies with --write-baseline")
        try:
            resolve_prompt_commit(args.prompt_commit)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None

    if args.from_results is not None:
        return _replay_results(
            args,
            baseline=baseline,
            output_root=output_root,
            write_path=write_path,
            source_run_urls=source_run_urls,
        )

    providers_needed = sorted({cell_provider for cell_provider, _model in specs})
    clients = build_clients(providers_needed, client_factory=client_factory)
    unavailable = {
        name: marker for name, marker in clients.items() if isinstance(marker, ClientUnavailable)
    }
    if unavailable and len(unavailable) == len(clients):
        # No provider can run at all: keep the historical "could not start" report
        # rather than a matrix of identical errored cells.
        output_root.mkdir(parents=True, exist_ok=True)
        details = ", ".join(
            f"{name}: {type(marker.cause).__name__}" for name, marker in unavailable.items()
        )
        message = f"Evaluation could not start ({details}).\n"
        _path_under(output_root, "summary.md").write_text(
            f"## Conversation evaluation: ERRORED\n\n- {message}"
        )
        print(f"INFRASTRUCTURE ERROR: {message}", end="")
        return EXIT_ERRORED
    for name, marker in unavailable.items():
        print(
            f"INFRASTRUCTURE ERROR: no {name} client ({type(marker.cause).__name__}: "
            f"{marker.cause}); its cells will be recorded as errored."
        )

    runs = run_matrix(
        models=models,
        efforts=efforts,
        scenarios=scenarios,
        repeats=max(1, args.repeats),
        clients=clients,
        concurrency=args.concurrency,
        provider=provider,
        thinking=thinking,
        max_tokens=args.max_tokens,
        max_api_calls=args.max_api_calls,
        sleep=sleep,
    )
    summary, markdown = write_outputs(output_root, runs)
    print(markdown, end="")
    _append_step_summary(markdown)

    if baseline is not None:
        return _judge_against_baseline(runs, baseline, args.baseline, output_root)
    if write_path is not None:
        return _record_baseline(
            runs,
            write_path,
            output_root=output_root,
            source_run_url=source_run_urls[0] if source_run_urls else default_source_run_url(),
            prompt_commit=args.prompt_commit,
        )

    if summary["errored"]:
        print(
            f"INFRASTRUCTURE ERROR: {summary['errored']} conversation(s) could not complete "
            "after retries; see each run's `error` (class, API code, status, message). "
            "Not counted as behavioral failures."
        )
        return EXIT_ERRORED
    if summary["failed"] and args.fail_on_violation:
        print(f"BEHAVIORAL FAILURE: {summary['failed']} conversation(s) violated scoring rules.")
        return EXIT_REGRESSION
    return 0


def _append_step_summary(markdown: str) -> None:
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with Path(github_summary).open("a") as handle:
            handle.write(markdown)


def default_source_run_url() -> str | None:
    """The GitHub Actions run URL when running in Actions, else ``None``."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def _judge_against_baseline(
    runs: list[dict[str, Any]], baseline: dict[str, Any], baseline_path: Path, output_root: Path
) -> int:
    """Apply the regression rule, write verdict.md/json, print the table, return its exit code."""
    verdict = evaluate_regression(baseline, scenario_stats(runs))
    # The header names the prompt the baseline measured and whether this checkout
    # sends the same one; informational, never part of the verdict.
    providers = {run_provider(run) for run in runs}
    current_prompt = system_prompt_sha256(providers.pop()) if len(providers) == 1 else None
    verdict_markdown = render_verdict_markdown(
        verdict, baseline_path=str(baseline_path), current_prompt_sha256=current_prompt
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _path_under(output_root, "verdict.md").write_text(verdict_markdown)
    _path_under(output_root, "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(verdict_markdown, end="")
    _append_step_summary(verdict_markdown)
    if verdict["status"] == STATUS_ERRORED:
        print(
            "INFRASTRUCTURE ERROR: at least one conversation could not complete after retries; "
            "the verdict is 'errored', not a behavioral regression."
        )
    elif verdict["status"] == STATUS_NO_MODEL_BASELINE:
        available = ", ".join(sorted(baseline["baselines"]))
        print(
            f"NO BASELINE: {baseline_path} has no entry for {verdict['cell']} (it has: "
            f"{available}). The gate cannot judge this run; record a baseline for this "
            "model/effort in the same PR (see evals/README.md, 'Re-baselining is deliberate'). "
            "Not a behavioral regression."
        )
    elif verdict["status"] == STATUS_REGRESSION:
        print(
            "BEHAVIORAL REGRESSION: at least one scenario regressed relative to "
            f"{baseline_path} (see the verdict table)."
        )
    return int(verdict["exit_code"])


def _record_baseline(
    runs: list[dict[str, Any]],
    write_path: Path,
    *,
    output_root: Path,
    source_run_url: str | None,
    prompt_commit: str | None = None,
) -> int:
    """``--write-baseline``: persist the runs' cell as a baseline entry; exit 2 if untrustworthy.

    ``source_run_url`` is the provenance for records that do not already carry
    one (``--from-results`` tags each tree's records with its own URL).
    ``prompt_commit`` (``--prompt-commit``) names the commit whose prompts the
    runs measured; the entry's ``prompt_version`` pairs it with the sha256 of
    the system prompt in this checkout.
    """
    try:
        document = build_baseline(runs, source_run_url=source_run_url)
    except ValueError as exc:
        print(f"INFRASTRUCTURE ERROR: {exc}")
        return EXIT_ERRORED
    ((key, entry),) = document["baselines"].items()
    version = prompt_version_for(entry["provider"], commit=prompt_commit)
    if version is not None:
        entry["prompt_version"] = version
    try:
        merged = write_baseline(write_path, document, allowed_roots=(REPO_ROOT, output_root))
    except ValueError as exc:
        raise SystemExit(f"--write-baseline: {exc}") from None
    rows = ", ".join(
        f"{name} {scenario['passed']}/{scenario['repeats']}"
        for name, scenario in sorted(entry["scenarios"].items())
    )
    others = sorted(set(merged["baselines"]) - {key})
    kept = f"; kept {', '.join(others)}" if others else ""
    prompt_note = (
        f"; prompt {version['git_commit'][:12]} sha256 {version['system_prompt_sha256']}"
        if version
        else "; prompt_version not recorded (no commit id: pass --prompt-commit)"
    )
    print(
        f"Wrote baseline entry {key} to {write_path}: {rows} "
        f"(sources: {', '.join(entry['source_run_urls']) or 'none'}){kept}{prompt_note}"
    )
    return 0


def rescore_runs(
    runs: list[dict[str, Any]],
    *,
    fixture_loader: Callable[[str], dict[str, Any]] = load_fixture,
    scenarios: Iterable[str] | None = None,
) -> int:
    """Re-score every scored record that carries a full trace; return how many.

    A downloaded artifact stores the ``result`` the scorer produced *when the
    run happened*. Judging or baselining it later must apply the rules in this
    checkout (a severity table change such as ``duplicate_sql`` becoming soft
    would otherwise never reach an existing artifact). Records without a real
    trace (no ``tool_events``), for scenarios with no fixture on disk, or with
    ``status != "scored"`` are left untouched.
    """
    available = set(scenarios if scenarios is not None else available_scenarios())
    fixtures: dict[str, dict[str, Any]] = {}
    rescored = 0
    for run in runs:
        if run.get("status") != "scored":
            continue
        trace = run.get("trace")
        scenario = str(run.get("scenario"))
        if not isinstance(trace, dict) or "tool_events" not in trace or scenario not in available:
            continue
        fixture = fixtures.get(scenario)
        if fixture is None:
            fixture = fixtures[scenario] = fixture_loader(scenario)
        run["result"] = score_trace(trace, fixture)
        rescored += 1
    return rescored


def _replay_results(
    args: argparse.Namespace,
    *,
    baseline: dict[str, Any] | None,
    output_root: Path,
    write_path: Path | None,
    source_run_urls: list[str],
) -> int:
    """``--from-results DIR [...]``: apply --baseline / --write-baseline to recorded runs.

    No model calls. Several directories are pooled; each one's records are
    tagged with its ``--source-run-url`` (one per directory in order, or one
    URL for all) so the baseline can attribute samples to their runs.
    """
    if baseline is None and write_path is None:
        raise SystemExit("--from-results needs --baseline or --write-baseline")
    dirs: list[Path] = list(args.from_results)
    if len(source_run_urls) <= 1:
        urls = [source_run_urls[0] if source_run_urls else default_source_run_url()] * len(dirs)
    elif len(source_run_urls) == len(dirs):
        urls = source_run_urls
    else:
        raise SystemExit(
            f"--source-run-url was given {len(source_run_urls)} time(s) for "
            f"{len(dirs)} --from-results directories; give one per directory or one in total"
        )
    runs: list[dict[str, Any]] = []
    try:
        for directory, url in zip(dirs, urls):
            runs.extend(load_runs_from_dir(directory, source_run_url=url))
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    rescored = rescore_runs(runs)
    if rescored:
        print(
            f"Re-scored {rescored} run record(s) with the current scoring rules "
            "(the stored `result` reflected the rules at recording time)."
        )
    # --models/--efforts narrow a multi-cell tree (a comparison artifact) to one cell.
    provider = model = effort = None
    if args.models:
        specs = parse_model_specs(_csv(args.models, []), args.provider)
        if len(specs) != 1:
            raise SystemExit("--from-results accepts exactly one --models entry")
        provider, model = specs[0]
    if args.efforts:
        efforts = _csv(args.efforts, [])
        if len(efforts) != 1:
            raise SystemExit("--from-results accepts exactly one --efforts entry")
        effort = efforts[0]
    runs = select_cell(runs, provider=provider, model=model, effort=effort)
    where = ", ".join(str(directory) for directory in dirs)
    if not runs:
        raise SystemExit(f"No run records under {where} match the requested cell")
    if args.scenarios or args.scenario:
        wanted = set(_csv(args.scenarios, [args.scenario] if args.scenario else []))
        runs = [run for run in runs if str(run.get("scenario")) in wanted]
    print(f"Loaded {len(runs)} run record(s) from {len(dirs)} tree(s): {where}")
    if baseline is not None:
        return _judge_against_baseline(runs, baseline, args.baseline, output_root)
    assert write_path is not None
    return _record_baseline(
        runs, write_path, output_root=output_root, source_run_url=None, prompt_commit=args.prompt_commit
    )


if __name__ == "__main__":
    raise SystemExit(main())
