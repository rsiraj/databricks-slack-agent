"""Run one production-shaped conversation against deterministic fake tools."""

from __future__ import annotations

import inspect
import json
import os
import time
from typing import Any, Callable

import anthropic

from evals.pricing import estimate_cost_from_usage
from evals.synthetic_tools import SyntheticTools
from tools import calc_tool, databricks_tool, github_tool
from tools.registry import all_tool_specs
from utils.job_run_evidence_guard import JobRunEvidenceGuard
from utils.llm import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    ModelRequest,
    ModelResponse,
    Provider,
    ProviderNotConfigured,
)
from utils.llm.anthropic_provider import THINKING_MODES, AnthropicProvider
from utils.llm.openai_provider import OpenAIProvider
from utils.numeric_guard import (
    NumericEvidenceGuard,
    numeric_guard_max_revisions,
    numeric_guard_mode,
)
from utils.slack_formatting import normalize_slack_mrkdwn
from utils.sql_run_guard import SqlRunGuard, auto_satisfy_metadata_precondition
from utils.system_prompt import load_system_prompt


# Keep these literals in sync with worker.py / utils.llm defaults; a test
# (tests/test_conversation_eval_scoring.py) compares them by AST so they cannot
# drift. They are literals rather than ``from worker import ...`` because
# importing ``worker`` constructs boto3/Slack clients at module load and needs
# APP_SECRET_ARN; the eval job runs without any of that.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "high"
DEFAULT_THINKING = "adaptive"
# Budgets mirror production. ``MAX_API_CALLS`` is ``worker.MAX_ITERATIONS``: the
# number of model calls after which the bot posts "Hit max iterations without
# completing" to the user, so exhausting it is a behavioral failure, not a
# harness limit. ``MAX_TOKENS_PER_CALL`` is what ``worker._resolve_max_tokens``
# yields for the default model and effort with ``ANTHROPIC_MAX_TOKENS`` unset,
# i.e. ``recommended_max_tokens(DEFAULT_MODEL, DEFAULT_EFFORT)``, and the value
# Terraform sets in production. Production has no separate per-conversation
# output-token cap; its effective bound is per-call ``MAX_TOKENS`` times the
# iteration budget.
MAX_API_CALLS = 25
MAX_TOKENS_PER_CALL = 24_000
MAX_TOTAL_OUTPUT_TOKENS = MAX_TOKENS_PER_CALL * MAX_API_CALLS
MAX_FINAL_REVISIONS = 1
ALL_TOOLS = (
    github_tool.GITHUB_TOOLS + databricks_tool.DATABRICKS_TOOLS + calc_tool.CALC_TOOLS
)
TOOL_SPECS = all_tool_specs()

USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    return {"role": "assistant", "content": list(response.content)}


def _final_text(response: ModelResponse) -> str:
    return "".join(
        str(block.get("text") or "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def supports_output_config(client: Any) -> bool:
    """True when ``client.messages.create`` accepts ``output_config`` as a kwarg.

    Older SDKs need ``extra_body``; fakes with ``**kwargs`` accept anything.
    """
    create = getattr(getattr(client, "messages", None), "create", None)
    if create is None:
        return True
    try:
        parameters = inspect.signature(create).parameters
    except (TypeError, ValueError):
        return True
    if "output_config" in parameters:
        return True
    return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def default_model_for(provider: str) -> str:
    return DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_MODEL


def make_provider(
    provider: str,
    *,
    client: Any,
    model: str,
    thinking: str | None = DEFAULT_THINKING,
) -> Provider:
    """Wrap an SDK client in the production provider for ``provider``.

    The Anthropic path builds the same request the worker sends (plus the
    eval-only ``thinking`` knob and the ``extra_body`` fallback for old SDKs).
    """
    if provider == "anthropic":
        if thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {THINKING_MODES}, got {thinking!r}")
        return AnthropicProvider(
            model,
            client=client,
            thinking=thinking,
            effort_via_extra_body=not supports_output_config(client),
        )
    if provider == "openai":
        return OpenAIProvider(model, client=client)
    raise ValueError(f"Unknown provider {provider!r}; expected one of {', '.join(SUPPORTED_PROVIDERS)}")


def _remove_private_revisions(messages: list[dict[str, Any]], *start_indices: int | None) -> None:
    """Drop each private draft + correction-request pair, highest index first.

    Same contract as ``worker._remove_private_revisions``: the recorded
    conversation keeps only what the model did in response (tool calls, the
    final answer), never the rejected draft or the guard's instruction.
    """
    for index in sorted((i for i in start_indices if i is not None), reverse=True):
        del messages[index : index + 2]


class _TimedDispatch:
    """Wrap a tool dispatcher so each call's latency can be attributed later."""

    def __init__(self, dispatch: Callable[[str, dict[str, Any]], dict[str, Any]]):
        self._dispatch = dispatch
        self.latencies_ms: list[float] = []

    def __call__(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return self._dispatch(name, tool_input)
        finally:
            self.latencies_ms.append(_elapsed_ms(started))

    def take(self) -> float | None:
        return self.latencies_ms.pop(0) if self.latencies_ms else None


def resolve_max_api_calls(
    fixture: dict[str, Any], override: int | None = None
) -> tuple[int, str]:
    """Effective per-conversation API-call budget and where it came from.

    Precedence: an explicit ``override`` (the ``--max-api-calls`` CLI flag, for
    comparison runs) beats the fixture's optional ``max_api_calls`` key, which
    beats the production default ``MAX_API_CALLS``. Returns ``(budget, source)``
    with ``source`` in ``{"override", "fixture", "default"}``.
    """
    if override is not None:
        value, source = override, "override"
    elif fixture.get("max_api_calls") is not None:
        value, source = fixture["max_api_calls"], "fixture"
    else:
        return MAX_API_CALLS, "default"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_api_calls ({source}) must be a positive integer, got {value!r}")
    return value, source


def run_conversation(
    fixture: dict[str, Any],
    *,
    client: Any,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    effort: str | None = DEFAULT_EFFORT,
    thinking: str | None = DEFAULT_THINKING,
    max_tokens: int = MAX_TOKENS_PER_CALL,
    max_api_calls: int | None = None,
    tool_runtime: Any | None = None,
) -> dict[str, Any]:
    """Run the model loop and return a JSON-safe, thinking-free trace.

    The trace records per-API-call latency, model, stop reason, and full usage;
    per-tool latency; conversation totals; wall-clock time; and an estimated
    cost from ``evals.pricing`` (``None`` for unknown model ids).

    ``max_api_calls`` overrides the fixture's ``max_api_calls`` and the
    production default (see ``resolve_max_api_calls``). The output-token budget
    is derived the same way production's effective bound is: per-call
    ``max_tokens`` times the API-call budget.
    """
    run_started = time.perf_counter()
    api_call_budget, api_call_budget_source = resolve_max_api_calls(fixture, max_api_calls)
    output_token_budget = max_tokens * api_call_budget
    model_provider = make_provider(provider, client=client, model=model, thinking=thinking)
    system_prompt = load_system_prompt(provider)
    messages: list[dict[str, Any]] = [{"role": "user", "content": fixture["user_prompt"]}]
    synthetic_tools = tool_runtime or SyntheticTools(fixture)
    dispatch = _TimedDispatch(synthetic_tools.dispatch)
    sql_guard = SqlRunGuard()
    evidence_guard = JobRunEvidenceGuard()
    tool_events: list[dict[str, Any]] = []
    api_call_events: list[dict[str, Any]] = []
    assistant_texts: list[str] = []
    usage_totals = {field: 0 for field in USAGE_FIELDS}
    stop_reason = "max_api_calls"
    final_text = ""
    api_calls = 0
    final_revision_count = 0
    revision_start_index: int | None = None
    output_token_budget_exceeded = False
    response_models: list[str] = []

    # Numeric grounding mirrors worker._run_agent_loop: same env-driven mode and
    # revision budget, same grounding sources (operational constants, user
    # message, every model-visible tool input/result; never the system prompt),
    # one private revision request, then a visible footnote if figures are
    # still unverified.
    numeric_mode = numeric_guard_mode()
    numeric_max_revisions = numeric_guard_max_revisions()
    numeric_guard = NumericEvidenceGuard()
    numeric_revision_count = 0
    numeric_revision_start_indices: list[int] = []
    numeric_unverified_figures: list[str] = []
    if numeric_mode != "off":
        numeric_guard.record_user_text(str(fixture["user_prompt"]))

    for iteration in range(api_call_budget):
        # The eval client carries its own client-level timeout, so no
        # per-request timeout is sent (matching the harness's historical request).
        request = ModelRequest(
            system=system_prompt,
            messages=messages,
            tools=TOOL_SPECS,
            effort=effort or "",
            max_tokens=max_tokens,
            timeout_s=None,
            model=model,
        )
        call_started = time.perf_counter()
        response = model_provider.create(request)
        call_latency_ms = _elapsed_ms(call_started)
        api_calls += 1
        usage = response.usage.as_dict()
        for field, value in usage.items():
            usage_totals[field] += value
        response_model = str(response.model or model)
        response_models.append(response_model)
        api_call_events.append(
            {
                "iteration": iteration,
                "latency_ms": call_latency_ms,
                "model": response_model,
                "stop_reason": str(response.raw_stop_reason or response.stop_reason),
                "usage": usage,
            }
        )
        messages.append(_assistant_message(response))

        visible_text = _final_text(response)
        if visible_text:
            assistant_texts.append(visible_text)

        # Record the budget breach without masking a completed answer: a final
        # end_turn is still captured (and scoring flags the budget separately).
        if usage_totals["output_tokens"] > output_token_budget:
            output_token_budget_exceeded = True

        if response.stop_reason == "end_turn":
            stop_reason = "end_turn"
            fallback_used = False
            violations = evidence_guard.validate(visible_text)
            if violations:
                revision_instruction = evidence_guard.revision_instruction(violations)
                if (
                    revision_instruction is not None
                    and final_revision_count < MAX_FINAL_REVISIONS
                    and not output_token_budget_exceeded
                ):
                    if revision_start_index is None:
                        revision_start_index = len(messages) - 1
                    messages.append({"role": "user", "content": revision_instruction})
                    final_revision_count += 1
                    continue
                visible_text = evidence_guard.safe_fallback()
                fallback_used = True

            # Only model-authored text is checked; the evidence guard's fallback
            # is computed from tool data already.
            if numeric_mode != "off" and not fallback_used:
                unverified = numeric_guard.validate(visible_text)
                if unverified:
                    figures = [item["text"] for item in unverified]
                    if (
                        numeric_mode == "revise"
                        and numeric_revision_count < numeric_max_revisions
                        and not output_token_budget_exceeded
                    ):
                        numeric_revision_start_indices.append(len(messages) - 1)
                        messages.append(
                            {
                                "role": "user",
                                "content": numeric_guard.revision_instruction(unverified),
                            }
                        )
                        numeric_revision_count += 1
                        continue
                    visible_text = numeric_guard.annotate(visible_text, unverified)
                    numeric_unverified_figures = figures

            _remove_private_revisions(
                messages, revision_start_index, *numeric_revision_start_indices
            )
            final_text = normalize_slack_mrkdwn(visible_text)
            break

        if output_token_budget_exceeded:
            stop_reason = "output_token_budget"
            break

        if response.stop_reason != "tool_use":
            stop_reason = str(response.raw_stop_reason or response.stop_reason)
            break

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_name = str(block.get("name") or "")
            tool_input = dict(block.get("input") or {})
            initial_rejection = sql_guard.before_call(tool_name, tool_input)
            result, preflight_attempts = auto_satisfy_metadata_precondition(
                sql_guard,
                tool_name,
                tool_input,
                initial_rejection,
                dispatch,
            )
            for attempt in preflight_attempts:
                metadata_result = attempt["result"]
                if metadata_result is None:
                    exception = attempt["exception"]
                    metadata_result = {
                        "error": "Automatic metadata preflight failed.",
                        "error_type": type(exception).__name__,
                    }
                metadata_is_error = bool(
                    "error" in metadata_result
                    or metadata_result.get("status") in {"FAILED", "CANCELED", "CLOSED"}
                )
                tool_events.append(
                    {
                        "iteration": iteration,
                        "name": "databricks_describe_object",
                        "input": attempt["input"],
                        "result": metadata_result,
                        # A successful zero-row metadata lookup did not satisfy the
                        # SQL precondition, but it is not itself a tool failure.
                        "is_error": metadata_is_error,
                        "source": "runtime_preflight",
                        "model_visible": False,
                        "latency_ms": dispatch.take(),
                    }
                )
            source = "guard" if result is not None else "fixture"
            tool_latency_ms: float | None = 0.0
            if result is None:
                result = dispatch(tool_name, tool_input)
                tool_latency_ms = dispatch.take()
                result = sql_guard.after_call(tool_name, tool_input, result)

            is_error = bool(
                isinstance(result, dict)
                and (
                    "error" in result
                    or result.get("status") in {"FAILED", "CANCELED", "CLOSED"}
                )
            )
            evidence_guard.record_tool_result(tool_name, tool_input, result)
            numeric_guard.record_tool_result(tool_name, tool_input, result)
            tool_events.append(
                {
                    "iteration": iteration,
                    "name": tool_name,
                    "input": tool_input,
                    "result": result,
                    "is_error": is_error,
                    "source": source,
                    "model_visible": True,
                    "latency_ms": tool_latency_ms,
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.get("id"),
                    "content": json.dumps(result, default=str),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    api_latency_ms = round(sum(call["latency_ms"] for call in api_call_events), 3)
    tool_latency_ms = round(
        sum(float(event.get("latency_ms") or 0) for event in tool_events), 3
    )
    cost_model = response_models[0] if response_models else model
    # True when the loop ran out of model calls without an end_turn: production
    # would have posted "Hit max iterations without completing" at this point.
    hit_api_call_budget = stop_reason == "max_api_calls"
    return {
        "scenario": fixture["id"],
        "provider": model_provider.name,
        "model": model,
        "response_models": sorted(set(response_models)),
        "effort": effort,
        "thinking": thinking,
        "max_tokens": max_tokens,
        "stop_reason": stop_reason,
        "final_text": final_text,
        "assistant_texts": assistant_texts,
        "tool_events": tool_events,
        "api_call_events": api_call_events,
        "api_calls": api_calls,
        "hit_api_call_budget": hit_api_call_budget,
        "tool_calls": len(tool_events),
        "output_tokens": usage_totals["output_tokens"],
        "output_token_budget_exceeded": output_token_budget_exceeded,
        "final_revisions": final_revision_count,
        "numeric_guard_mode": numeric_mode,
        "numeric_guard_revisions": numeric_revision_count,
        "numeric_guard_unverified_figures": numeric_unverified_figures,
        "usage": usage_totals,
        "latency_ms": {
            "api_total_ms": api_latency_ms,
            "tool_total_ms": tool_latency_ms,
            "wall_clock_ms": _elapsed_ms(run_started),
        },
        "estimated_cost_usd": estimate_cost_from_usage(cost_model, usage_totals),
        "budget": {
            "max_api_calls": api_call_budget,
            "max_api_calls_source": api_call_budget_source,
            "max_output_tokens": output_token_budget,
        },
    }


def build_client() -> anthropic.Anthropic:
    api_key = os.environ.get("MODEL_EVAL_ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MODEL_EVAL_ANTHROPIC_API_KEY is required")
    # SDK retries cover transport blips inside one call; evals.run retries whole
    # conversations on persistent 429/5xx/connection failures.
    return anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=2)


def describe_key_shape(api_key: str) -> str:
    """Non-secret description of an API key's *shape* for the job log.

    Never includes any of the key's characters beyond the public ``sk-`` /
    ``sk-proj-`` prefix class. Exists because a 401 ``invalid_api_key`` from a
    freshly pasted repository secret is almost always a copy problem (quotes,
    a trailing newline, ``Bearer`` prefix, truncated value), and the log is the
    only place the person fixing the secret can see which one it is.
    """
    if api_key.startswith("sk-proj-"):
        prefix = "sk-proj-"
    elif api_key.startswith("sk-"):
        prefix = "sk-"
    else:
        prefix = "unexpected (does not start with sk-)"
    quirks = []
    if any(ch in api_key for ch in "\"'`"):
        quirks.append("contains quote characters")
    if any(ch.isspace() for ch in api_key):
        quirks.append("contains internal whitespace")
    if api_key.lower().startswith("bearer "):
        quirks.append("starts with 'Bearer '")
    description = f"prefix={prefix} length={len(api_key)}"
    return f"{description} ({'; '.join(quirks)})" if quirks else description


def build_openai_client() -> Any:
    """OpenAI client for ``--provider openai``; requires the optional dependency.

    Uses a dedicated evaluation key (``MODEL_EVAL_OPENAI_API_KEY``), never the
    bot's app secret. Raises ``ProviderNotConfigured`` when the ``openai``
    package is not installed so the CLI reports a clean infrastructure error.
    """
    api_key = os.environ.get("MODEL_EVAL_OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MODEL_EVAL_OPENAI_API_KEY is required")
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProviderNotConfigured(
            "OpenAI evaluation requires the openai package (pip install openai)"
        ) from exc
    print(f"openai eval client: MODEL_EVAL_OPENAI_API_KEY shape {describe_key_shape(api_key)}")
    return openai.OpenAI(api_key=api_key, timeout=120.0, max_retries=2)


CLIENT_FACTORIES: dict[str, Callable[[], Any]] = {
    "anthropic": build_client,
    "openai": build_openai_client,
}
