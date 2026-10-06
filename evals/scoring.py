"""Deterministic scoring for synthetic conversation traces."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from utils.numeric_guard import NumericEvidenceGuard


# The gate blocks merges on *behavioral* failures only: tool misuse and
# avoidable tool errors, unsupported claims (``unsupported_answer_claim``),
# ungrounded or forbidden numbers, forbidden identifiers, missing required tool
# usage, metadata ordering, budgets, incomplete conversations and empty answers.
# Those codes default to ``hard`` (``passed`` is false when any is present).
# Content-coverage checks -- whether the answer happened to mention a specific
# fact -- are reported as ``soft``: they still appear in ``summary.md`` and
# reduce the graded ``score`` so model comparisons see coverage differences,
# but they never fail the gate. A fixture may move any code in either direction
# via ``violation_severity: {"<code>": "soft" | "hard"}``; a fixture whose
# required pattern *is* the behavioral point (for example "a percent figure is
# stated") promotes ``missing_answer_fact`` to ``hard``. ``synthetic_limitation``
# counts are informational metrics, never violations.
SEVERITY_WEIGHTS = {"hard": 0.25, "soft": 0.05}
DEFAULT_SEVERITY = "hard"
# Codes whose default differs from ``hard``.
# - ``ungrounded_number`` mirrors the production numeric guard, which annotates
#   rather than blocks; a fixture that is specifically about arithmetic promotes
#   it with ``violation_severity``.
# - ``missing_answer_fact`` (``required_final_patterns``) is content coverage,
#   not behavior: a correct, well-grounded answer can phrase a fact differently
#   or omit it without misusing a tool or inventing anything.
DEFAULT_CODE_SEVERITY = {"ungrounded_number": "soft", "missing_answer_fact": "soft"}

# ``avoidable_tool_failure`` covers every tool error, but the errors are not
# equally serious. The severity is decided by the *category* the guard or
# fixture attached (``validation_failure`` / ``fixture_failure`` /
# ``synthetic_failure``), so the gate blocks on wrong or unsafe actions and only
# reports wasted ones:
#
# - ``duplicate_sql`` is SOFT: the model re-issued a byte-identical query whose
#   result it already had (an earlier eval run re-ran a COUNT/MAX query to
#   "re-verify" a figure after a numeric-guard revision request). The
#   production impact is one round trip; nothing wrong or unsafe reached the
#   warehouse and the answer is built from the same data.
# - ``wrong_github_source`` is SOFT: the model read a repository file the
#   fixture does not model -- in the table-explanation scenario, the
#   deliberately stale ``Signups MA - Merge.sql`` notebook (comparison runs
#   two earlier runs each read it once, *after* the current
#   ``Signups - Merge.sql``). The read is a wasted call, not a wrong
#   answer: in production it simply succeeds and returns the file. What must
#   not happen is the answer *using* that file -- presenting the MA notebook as
#   the current build source -- and the fixture catches that as a hard
#   ``unsupported_answer_claim`` through ``forbidden_final_patterns``.
# - Everything else is HARD: a column that is not in the live schema, an
#   analytical query before the object's metadata was verified, JSON-style
#   struct access, SQL the parser rejects, a write statement, a denied
#   function. Each is either an invented fact, a skipped precondition, or a
#   blocked unsafe action.
#
# Categories absent from this table default to HARD. A fixture may override a
# category with ``violation_severity: {"avoidable_tool_failure:<category>":
# "hard" | "soft"}`` or the whole code with ``{"avoidable_tool_failure": ...}``;
# the category-qualified key wins over the bare code.
AVOIDABLE_FAILURE_CATEGORY_SEVERITY = {
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
DEFAULT_AVOIDABLE_FAILURE_SEVERITY = "hard"

# A standalone figure in the final answer: optional thousands separators and
# decimals, not glued to letters, underscores, other digit runs, a decimal
# point, or a hyphenated identifier (``col_36``, ``v36``, ``1236``, ``36th``,
# ``3.36`` and ``us-east-1`` never yield ``36`` or ``1``). Same lookarounds as
# ``utils.numeric_guard``.
_FINAL_NUMBER_TOKEN = re.compile(
    r"(?<![\w.])(?<![A-Za-z_]-)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)"
)

# A hedged figure: ``~12%``, ``approximately 12%``, ``about 1,500``. The figure
# itself is captured in the same shape ``NumericEvidenceGuard`` recognises so
# it can be re-validated on its own. Hedging a figure a tool returned (``~12%``
# for ``result_display: "12"``, ``~28.6%`` for ``28.587...``) is presentation;
# hedging a figure no tool returned is an in-head estimate.
_APPROXIMATION_HEDGE = re.compile(
    r"(?:~\s?|(?i:\b(?:approximately|roughly|about|around|nearly|almost|circa)\s+))"
    r"(?P<figure>[-+−]?[$€£]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[kMB]?"
    r"(?![A-Za-z0-9_])(?:\s?%|\s(?:percent|pct)\b)?)"
)


def _violation(
    code: str,
    message: str,
    event_index: int | None = None,
    *,
    category: str | None = None,
) -> dict[str, Any]:
    violation: dict[str, Any] = {"code": code, "message": message}
    if event_index is not None:
        violation["event_index"] = event_index
    if category:
        violation["category"] = category
    return violation


def _to_decimal(raw: Any) -> Decimal | None:
    try:
        return Decimal(str(raw).replace(",", "").strip())
    except InvalidOperation:
        return None


def final_number_tokens(text: str) -> list[Decimal]:
    """Every standalone numeric token in ``text`` as a ``Decimal``.

    ``"12,480"`` and ``"12480"`` both yield ``Decimal("12480")``; ``"35.0"``
    compares equal to ``"35"``. Digits embedded in identifiers or longer digit
    runs are not tokens.
    """
    values: list[Decimal] = []
    for match in _FINAL_NUMBER_TOKEN.finditer(text or ""):
        value = _to_decimal(match.group(0))
        if value is not None:
            values.append(value)
    return values


def _required_tool_calls(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise ``required_tool_calls`` entries to ``{tool, min, input_contains}``."""
    parsed: list[dict[str, Any]] = []
    for entry in fixture.get("required_tool_calls") or []:
        if isinstance(entry, str):
            entry = {"tool": entry}
        tool = str(entry.get("tool") or "").strip()
        if not tool:
            continue
        try:
            minimum = max(1, int(entry.get("min", 1)))
        except (TypeError, ValueError):
            minimum = 1
        contains = entry.get("input_contains")
        parsed.append(
            {
                "tool": tool,
                "min": minimum,
                "input_contains": str(contains).lower() if contains else None,
            }
        )
    return parsed


def _numeric_grounding_violations(
    trace: dict[str, Any], fixture: dict[str, Any], final_text: str
) -> list[dict[str, Any]]:
    """Re-run the production numeric guard over the recorded trace.

    Grounds the user prompt plus every tool input and result in the trace (the
    same evidence the model could quote), then reports the figures in the final
    answer that none of them support. The system prompt is deliberately not
    grounded here so the eval is at least as strict as production.

    Approximation markers (``~``, "approximately", "about", ...) are judged
    against the same evidence: a hedged figure that rounds from a tool value is
    a presentation choice and passes; a hedged figure no tool supports is an
    in-head estimate and is reported separately as ``ungrounded_approximation``
    (hard by default, unlike ``ungrounded_number``), because the hedge is how
    a model dresses up arithmetic it did not run.
    """
    guard = NumericEvidenceGuard()
    guard.record_user_text(str(fixture.get("user_prompt") or trace.get("user_prompt") or ""))
    for event in trace.get("tool_events", []):
        guard.record_tool_result(
            str(event.get("name", "")), event.get("input"), event.get("result")
        )
    violations: list[dict[str, Any]] = []
    unverified = guard.validate(final_text)
    if unverified:
        figures = ", ".join(item["text"] for item in unverified)
        violations.append(
            _violation(
                "ungrounded_number",
                (
                    "The final answer contains figures that no tool result, tool input, or "
                    f"the user's message supports: {figures}."
                ),
            )
        )
    approximated = ungrounded_approximations(guard, final_text)
    if approximated:
        figures = ", ".join(approximated)
        violations.append(
            _violation(
                "ungrounded_approximation",
                (
                    "The final answer approximates figures that no tool result supports: "
                    f"{figures}. Quote the tool's result_display exactly or omit the figure."
                ),
            )
        )
    return violations


def ungrounded_approximations(guard: NumericEvidenceGuard, text: str) -> list[str]:
    """Hedged figures in ``text`` (``~29%``, ``about 1,500``) that ``guard`` cannot ground.

    Each hedged figure is validated on its own so the surrounding sentence
    cannot vouch for it; the guard's own leniency (rounding to the displayed
    precision, k/M/B suffixes, percent rescaling) applies unchanged. Figures
    the guard never treats as claims (small integers, years) are not reported.
    """
    flagged: list[str] = []
    seen: set[str] = set()
    for match in _APPROXIMATION_HEDGE.finditer(text or ""):
        hedged = match.group(0).strip()
        if hedged in seen:
            continue
        if guard.validate(match.group("figure")):
            seen.add(hedged)
            flagged.append(hedged)
    return flagged


def _forbidden_identifier_scopes(fixture: dict[str, Any]) -> list[tuple[set[str], list[str]]]:
    """Return ``(scoped_tables, identifiers)`` pairs from the fixture.

    ``forbidden_identifier_scope`` may be one object or a list of objects shaped
    ``{"table": "cat.sch.tbl" | "tables": [...], "identifiers": [...]}``. The
    legacy top-level ``forbidden_sql_identifiers`` list applies to every SQL
    relation when no scope is given (and is ignored when a scope exists).
    """
    scopes = fixture.get("forbidden_identifier_scope")
    if scopes:
        if isinstance(scopes, dict):
            scopes = [scopes]
        parsed: list[tuple[set[str], list[str]]] = []
        for scope in scopes:
            tables = scope.get("tables") or (
                [scope["table"]] if scope.get("table") else []
            )
            parsed.append(
                (
                    {str(table).replace("`", "").lower() for table in tables},
                    [str(value).lower() for value in scope.get("identifiers", [])],
                )
            )
        return parsed
    legacy = fixture.get("forbidden_sql_identifiers") or []
    return [(set(), [str(value).lower() for value in legacy])] if legacy else []


def default_severity(code: str, category: str | None = None) -> str:
    """The gate's default severity for ``code`` (and, for tool failures, its category)."""
    if code == "avoidable_tool_failure" and category:
        return AVOIDABLE_FAILURE_CATEGORY_SEVERITY.get(category, DEFAULT_AVOIDABLE_FAILURE_SEVERITY)
    return DEFAULT_CODE_SEVERITY.get(code, DEFAULT_SEVERITY)


def _apply_severity(
    violations: list[dict[str, Any]], fixture: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach ``severity`` to each violation.

    Precedence: fixture override for ``<code>:<category>``, then fixture
    override for ``<code>``, then the category table for tool failures, then
    the per-code default (``hard`` unless listed in ``DEFAULT_CODE_SEVERITY``).
    An override that is not ``hard``/``soft`` is ignored.
    """
    overrides = {
        str(code): str(severity).lower()
        for code, severity in (fixture.get("violation_severity") or {}).items()
    }
    for violation in violations:
        code = violation["code"]
        category = violation.get("category")
        severity = None
        if category:
            severity = overrides.get(f"{code}:{category}")
        if severity is None:
            severity = overrides.get(code)
        if severity not in SEVERITY_WEIGHTS:
            severity = default_severity(code, category)
        violation["severity"] = severity
    return violations


def graded_score(violations: list[dict[str, Any]]) -> float:
    """``1 - weighted_violations`` clamped to [0, 1]; 1.0 for a clean trace."""
    penalty = sum(
        SEVERITY_WEIGHTS.get(violation.get("severity", DEFAULT_SEVERITY), SEVERITY_WEIGHTS["hard"])
        for violation in violations
    )
    return round(max(0.0, min(1.0, 1.0 - penalty)), 4)


def _is_schema_inspection_sql(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql.replace("`", "")).strip()
    return bool(
        re.match(r"^(DESCRIBE|DESC)\b", normalized, flags=re.IGNORECASE)
        or re.match(r"^SHOW\s+CREATE\s+TABLE\b", normalized, flags=re.IGNORECASE)
        or re.search(r"\binformation_schema\.(?:tables|columns)\b", normalized, re.IGNORECASE)
    )


def _resolved_relation_names(sql: str, catalog: str = "", schema: str = "") -> set[str]:
    """Extract enough physical relation names for deterministic fixture scoring."""
    normalized = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.DOTALL)
    relations = set()
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+"
        r"((?:`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][\w$]*)){0,2})",
        normalized,
        flags=re.IGNORECASE,
    ):
        relation = re.sub(r"\s*\.\s*", ".", match.group(1).replace("`", "")).lower()
        if relation.count(".") == 0 and catalog and schema:
            relation = f"{catalog}.{schema}.{relation}".lower()
        elif relation.count(".") == 1 and catalog:
            relation = f"{catalog}.{relation}".lower()
        relations.add(relation)
    return relations


_SQL_SNIPPET_LIMIT = 240


def avoidable_failure_category(result: dict[str, Any]) -> str:
    """The guard/fixture category of a tool error (``tool_error`` when it has none)."""
    return str(
        result.get("validation_failure")
        or result.get("fixture_failure")
        or result.get("synthetic_failure")
        or "tool_error"
    )


def _avoidable_failure_message(
    name: str, tool_input: dict[str, Any], result: dict[str, Any]
) -> str:
    """Describe an avoidable tool error so the CI log is self-explanatory.

    The category alone (for example ``column_not_in_live_schema``) does not
    say *which* column the model invented or what it ran; an earlier CI run
    needed the artifact download to learn the SQL used ``original_timestamp``.
    Include the guard's ``missing_columns`` and a whitespace-collapsed SQL
    snippet whenever they are present.
    """
    category = avoidable_failure_category(result)
    message = f"{name} returned an avoidable error ({category})."

    missing = result.get("missing_columns")
    if isinstance(missing, (list, tuple)) and missing:
        joined = ", ".join(str(column) for column in missing)
        message += f" missing_columns: {joined}."

    sql = tool_input.get("sql") if isinstance(tool_input, dict) else None
    if isinstance(sql, str) and sql.strip():
        snippet = " ".join(sql.split())
        if len(snippet) > _SQL_SNIPPET_LIMIT:
            snippet = snippet[: _SQL_SNIPPET_LIMIT - 3].rstrip() + "..."
        message += f" sql: {snippet}"

    return message


def score_trace(trace: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    """Return a stable pass/fail result without another model call."""
    violations: list[dict[str, Any]] = []
    described_at_event: dict[str, int] = {}
    analytical_objects: set[str] = set()
    successful_tools: set[str] = set()
    forbidden_scopes = _forbidden_identifier_scopes(fixture)
    required_descriptions = {
        value.lower() for value in fixture.get("required_descriptions_before_query", [])
    }
    required_tool_calls = _required_tool_calls(fixture)
    required_tool_call_counts = [0 for _requirement in required_tool_calls]

    for index, event in enumerate(trace.get("tool_events", [])):
        name = str(event.get("name", ""))
        tool_input = event.get("input") or {}
        result = event.get("result") or {}

        if not event.get("is_error"):
            serialized_input = json.dumps(tool_input, default=str, sort_keys=True).lower()
            for position, requirement in enumerate(required_tool_calls):
                if name != requirement["tool"]:
                    continue
                contains = requirement["input_contains"]
                if contains and contains not in serialized_input:
                    continue
                required_tool_call_counts[position] += 1

        if event.get("is_error"):
            violations.append(
                _violation(
                    "avoidable_tool_failure",
                    _avoidable_failure_message(name, tool_input, result),
                    index,
                    category=avoidable_failure_category(result),
                )
            )
        else:
            successful_tools.add(name)

        if name == "databricks_describe_object" and not event.get("is_error"):
            object_name = ".".join(
                str(tool_input.get(part, "")).strip().replace("`", "").lower()
                for part in ("catalog", "schema", "name")
            )
            described_at_event[object_name] = index

        if name != "databricks_execute_sql":
            continue

        sql = str(tool_input.get("sql", ""))
        lowered_sql = sql.replace("`", "").lower()
        if _is_schema_inspection_sql(sql):
            continue

        relations = _resolved_relation_names(
            sql,
            str(tool_input.get("catalog") or ""),
            str(tool_input.get("schema") or ""),
        )
        for scoped_tables, identifiers in forbidden_scopes:
            if scoped_tables and not (scoped_tables & relations):
                continue
            for identifier in identifiers:
                if re.search(rf"\b{re.escape(identifier)}\b", lowered_sql):
                    violations.append(
                        _violation(
                            "forbidden_sql_identifier",
                            (
                                "Persisted-table SQL referenced build-only identifier "
                                f"{identifier!r}."
                            ),
                            index,
                        )
                    )
        for object_name in sorted(relations & required_descriptions):
            analytical_objects.add(object_name)
            description_event = described_at_event.get(object_name)
            if description_event is None or description_event >= index:
                violations.append(
                    _violation(
                        "metadata_ordering",
                        (
                            f"{object_name} was queried before its exact typed metadata "
                            "was verified."
                        ),
                        index,
                    )
                )

    for object_name in fixture.get("required_analytical_objects", []):
        if object_name.lower() not in analytical_objects:
            violations.append(
                _violation(
                    "missing_analytical_evidence",
                    f"No analytical query used required object {object_name}.",
                )
            )

    for tool_name in fixture.get("required_successful_tools", []):
        if tool_name not in successful_tools:
            violations.append(
                _violation(
                    "missing_tool_evidence",
                    f"No successful {tool_name} call was observed.",
                )
            )

    for requirement, observed in zip(required_tool_calls, required_tool_call_counts):
        if observed >= requirement["min"]:
            continue
        qualifier = (
            f" whose input contains {requirement['input_contains']!r}"
            if requirement["input_contains"]
            else ""
        )
        violations.append(
            _violation(
                "required_tool_missing",
                (
                    f"Expected at least {requirement['min']} successful {requirement['tool']} "
                    f"call(s){qualifier}; observed {observed}."
                ),
            )
        )

    if trace.get("stop_reason") != "end_turn":
        violations.append(
            _violation(
                "incomplete_conversation",
                f"Conversation stopped with {trace.get('stop_reason')!r}, not 'end_turn'.",
            )
        )

    final_text = str(trace.get("final_text") or "").strip()
    if not final_text:
        violations.append(_violation("empty_answer", "The model did not produce a final answer."))
    else:
        for pattern in fixture.get("required_final_patterns", []):
            if not re.search(pattern, final_text):
                violations.append(
                    _violation(
                        "missing_answer_fact",
                        f"The final answer did not match required evidence pattern {pattern!r}.",
                    )
                )
        for pattern in fixture.get("forbidden_final_patterns", []):
            if re.search(pattern, final_text):
                violations.append(
                    _violation(
                        "unsupported_answer_claim",
                        f"The final answer matched forbidden claim pattern {pattern!r}.",
                    )
                )

        final_numbers = set(final_number_tokens(final_text))
        for raw in fixture.get("required_final_numbers", []):
            expected = _to_decimal(raw)
            if expected is None or expected in final_numbers:
                continue
            violations.append(
                _violation(
                    "required_number_missing",
                    f"The final answer did not state the required figure {str(raw)!r}.",
                )
            )
        for raw in fixture.get("forbidden_final_numbers", []):
            forbidden = _to_decimal(raw)
            if forbidden is None or forbidden not in final_numbers:
                continue
            violations.append(
                _violation(
                    "forbidden_number_present",
                    f"The final answer stated the forbidden figure {str(raw)!r}.",
                )
            )

        if fixture.get("numeric_grounding"):
            violations.extend(_numeric_grounding_violations(trace, fixture, final_text))

    budget = trace.get("budget") or {}
    api_calls = trace.get("api_calls", 0)
    max_api_calls = budget.get("max_api_calls", 0)
    # The budget mirrors worker.MAX_ITERATIONS, so exhausting it is exactly the
    # point where production posts "Hit max iterations without completing".
    if trace.get("hit_api_call_budget") or api_calls > max_api_calls:
        violations.append(
            _violation(
                "api_call_budget",
                f"The conversation used its API-call budget ({api_calls}/{max_api_calls} calls) "
                "without finishing; production would have posted 'Hit max iterations'.",
            )
        )
    if trace.get("output_tokens", 0) > budget.get("max_output_tokens", 0):
        violations.append(_violation("output_token_budget", "The output-token budget was exceeded."))

    violations = _apply_severity(violations, fixture)
    hard_violations = [v for v in violations if v["severity"] == "hard"]
    usage = trace.get("usage") or {}
    return {
        "scenario": fixture["id"],
        "passed": not hard_violations,
        "score": graded_score(violations),
        "hard_violations": len(hard_violations),
        "soft_violations": len(violations) - len(hard_violations),
        "avoidable_tool_failures": sum(
            1 for violation in violations if violation["code"] == "avoidable_tool_failure"
        ),
        "violations": violations,
        "metrics": {
            "api_calls": api_calls,
            "max_api_calls": max_api_calls,
            "hit_api_call_budget": bool(trace.get("hit_api_call_budget", False)),
            "tool_calls": len(trace.get("tool_events", [])),
            "output_tokens": trace.get("output_tokens", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "wall_clock_ms": (trace.get("latency_ms") or {}).get("wall_clock_ms"),
            "estimated_cost_usd": trace.get("estimated_cost_usd"),
            "metadata_preflights_injected": sum(
                1
                for event in trace.get("tool_events", [])
                if event.get("source") == "runtime_preflight"
            ),
            "synthetic_limitations": sum(
                1
                for event in trace.get("tool_events", [])
                if (event.get("result") or {}).get("synthetic_limitation")
            ),
            "numeric_guard_revisions": trace.get("numeric_guard_revisions"),
        },
    }
