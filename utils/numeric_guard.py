"""Deterministic check that every figure in an answer came from a tool.

The model is told never to do arithmetic in its head. This guard is the
backstop: it collects every number the model could legitimately quote (the
user's own message, prior-turn user text and tool results, every tool
input/result of the current run, and a short allowlist of operational constants
the bot quotes about itself) and, on ``end_turn``, extracts the quantity claims
from the draft and reports the ones that match none of them. The worker asks
for one private revision (mirroring ``JobRunEvidenceGuard``) and, if the draft
still contains unverified figures, appends a footnote instead of blocking the
answer.

Matching is deliberately lenient about presentation (thousands separators,
rounding to the displayed precision, ``k``/``M``/``B`` suffixes) and
deliberately conservative about what counts as a claim: dates, times, years,
versions, identifiers, Slack timestamps, long ids, and small integers are never
flagged.

What is grounded
----------------
* The system prompt is **not** grounded. It contains business narrative
  ("~-36%") that would otherwise vouch for coincidental figures ("36 columns").
  Only ``reference_constants()`` - the operational limits the bot legitimately
  quotes about itself (row cap, rate-limit ceilings, iteration budget, Slack
  chunk size, retention, GitHub read cap, image cap) - are grounded, and they
  are read from code/env, never from prompt text.
* Prior-turn user text and tool results are grounded; prior assistant text is
  not (a number the bot previously produced without a tool stays unverified).

Skipped tool-result fields
--------------------------
Positional and identifier fields ground coincidental figures
(``ordinal_position: 12`` must not vouch for "12%"), so ``_walk`` skips values
under these keys, at any depth:

* exact names: ``ordinal_position``, ``position``, ``index``, ``iteration``,
  ``line``, ``page``, ``offset``, ``page_size``, ``limit``, ``row_limit``,
  ``status_code``, ``ttl``, ``updated_at``, ``version``;
* any key ending in ``_id``, ``_ts``, ``_index`` or ``_count`` - except the
  legitimately quoted counts ``column_count``, ``row_count``,
  ``fetched_run_count`` and ``total_count``;
* opaque blobs: keys named ``payload``, ``encrypted_content`` or
  ``provider_reasoning``, and any block whose ``type`` is
  ``provider_reasoning``.

Percentages
-----------
Numbers that carried a percent marker in tool text (``"12%"``), numeric values
under keys containing ``pct``/``percent``/``rate``/``ratio``/``share`` (SQL
result columns included), the user's own percentages and ``calculate`` results
are tracked separately as *percent values*. A whole-number percent claim
("12%") is grounded only by a percent value or by x100 of a non-integer
grounded value (``0.12``); a plain integer ``12`` never grounds it. The
x100 / /100 rescaling is likewise only applied to percent values and to
non-integer grounded values.
"""

from __future__ import annotations

import json
import os
import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from utils.slack_formatting import SLACK_MESSAGE_MAX_CHARS

NUMERIC_GUARD_MODES = frozenset({"revise", "annotate", "off"})
DEFAULT_NUMERIC_GUARD_MODE = "revise"
DEFAULT_NUMERIC_GUARD_MAX_REVISIONS = 1

# Small integers are almost always words or ordinals ("top 5", "3 tables") and
# are cheap to say without a tool; anything larger is treated as a claim.
SMALL_INTEGER_MAX = 12
YEAR_MIN = 1900
YEAR_MAX = 2100
CONTEXT_CHARS = 40

# Operational constants the bot may quote about itself. Mirrors of values that
# live in modules this one must not import (worker imports us; ratelimit and
# slack_helpers pull in boto3 / slack_sdk); tests pin them to the originals.
MAX_ITERATIONS = 25  # worker.MAX_ITERATIONS
RETENTION_DAYS = 365  # utils.sessions.SESSION_RETENTION_DAYS, utils.feedback
GITHUB_READ_CAP_CHARS = 60_000  # tools.github_tool._truncate default
IMAGE_CAP_MB = 5  # utils.slack_helpers.MAX_IMAGE_BYTES
IMAGE_CAP_BYTES = IMAGE_CAP_MB * 1024 * 1024

# Values under these keys are positions or identifiers, never quantities.
SKIP_KEYS = frozenset(
    {
        "ordinal_position",
        "position",
        "index",
        "iteration",
        "line",
        "page",
        "offset",
        "page_size",
        "limit",
        "row_limit",
        "status_code",
        "ttl",
        "updated_at",
        "version",
        # Opaque blobs.
        "payload",
        "encrypted_content",
        "provider_reasoning",
    }
)
SKIP_KEY_SUFFIXES = ("_id", "_ts", "_index", "_count")
KEEP_COUNT_KEYS = frozenset({"column_count", "row_count", "fetched_run_count", "total_count"})
SKIP_BLOCK_TYPES = frozenset({"provider_reasoning"})
PERCENT_KEY_MARKERS = ("pct", "percent", "rate", "ratio", "share")

_SUFFIX_MULTIPLIERS = {
    "k": Decimal(1_000),
    "M": Decimal(1_000_000),
    "B": Decimal(1_000_000_000),
}

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_SLACK_TS = re.compile(r"\b\d{10}\.\d{6}\b")
_ISO_DATETIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
_US_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_MONTH_DATE = re.compile(
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?\b"
)
_CLOCK_TIME = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:\s?[AaPp]\.?[Mm]\.?)?\b"
)
_VERSION = re.compile(r"\bv?\d+(?:\.\d+){2,}\b")
_PART_OF = re.compile(r"\bPart\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"\d{10,}")

# Order matters: Slack timestamps and long ids before dates so their digit runs
# are removed whole; code is removed first so SQL literals never become claims.
_NON_CLAIM_PATTERNS = (
    _FENCED_CODE,
    _INLINE_CODE,
    _SLACK_TS,
    _LONG_DIGITS,
    _ISO_DATETIME,
    _US_DATE,
    _MONTH_DATE,
    _CLOCK_TIME,
    _VERSION,
    _PART_OF,
)

# A quantity claim: optional sign, optional currency symbol, digits with
# optional thousands separators and decimals, optional k/M/B multiplier, and an
# optional percent marker. The lookarounds reject digits that are glued to an
# identifier (``table_v2``, ``us-east-1``, ``abc123``, ``5th``, ``10x``).
_CANDIDATE = re.compile(
    r"(?<![\w.])(?<![A-Za-z_]-)"
    r"(?P<sign>[-+−])?"
    r"(?P<currency>[$€£])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<suffix>[kMB])?"
    r"(?![A-Za-z0-9_])"
    r"(?P<pct>\s?%|\s(?:percent|pct)\b)?"
)

# Numbers inside tool results and user text: permissive, no exclusions, since
# over-grounding only ever makes the guard more lenient. The percent marker is
# captured so "12%" in a tool string can vouch for a whole-number percent claim.
_TEXT_NUMBER = re.compile(
    r"(?<![\w.])-?(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<pct>\s?%|\s(?:percent|pct)\b)?"
)


def numeric_guard_mode() -> str:
    """Read ``NUMERIC_GUARD_MODE`` (``revise`` | ``annotate`` | ``off``)."""
    raw = (os.environ.get("NUMERIC_GUARD_MODE") or "").strip().lower()
    return raw if raw in NUMERIC_GUARD_MODES else DEFAULT_NUMERIC_GUARD_MODE


def numeric_guard_max_revisions() -> int:
    """Read ``NUMERIC_GUARD_MAX_REVISIONS`` (non-negative int, default 1)."""
    raw = os.environ.get("NUMERIC_GUARD_MAX_REVISIONS")
    if raw is None or raw.strip() == "":
        return DEFAULT_NUMERIC_GUARD_MAX_REVISIONS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_NUMERIC_GUARD_MAX_REVISIONS
    return value if value >= 0 else DEFAULT_NUMERIC_GUARD_MAX_REVISIONS


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def reference_constants() -> dict[str, int]:
    """Operational limits the bot may quote about itself without a tool call.

    Read from the same environment variables (with the same defaults) as the
    modules that enforce them, so a redeploy with a different row cap or rate
    limit is reflected here without touching prompt text.
    """
    return {
        "row_limit": _env_int("DATABRICKS_ROW_LIMIT", 1000),
        "ratelimit_user_per_hour": _env_int("RATELIMIT_USER_PER_HOUR", 60),
        "ratelimit_user_per_day": _env_int("RATELIMIT_USER_PER_DAY", 200),
        "ratelimit_channel_per_hour": _env_int("RATELIMIT_CHANNEL_PER_HOUR", 200),
        "ratelimit_sql_per_day": _env_int("RATELIMIT_SQL_PER_DAY", 500),
        "max_iterations": MAX_ITERATIONS,
        "slack_chunk_chars": SLACK_MESSAGE_MAX_CHARS,
        "retention_days": RETENTION_DAYS,
        "github_read_cap_chars": GITHUB_READ_CAP_CHARS,
        "image_cap_mb": IMAGE_CAP_MB,
        "image_cap_bytes": IMAGE_CAP_BYTES,
    }


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped[:1] in ("{", "[")


def _prose_without_non_claims(text: str) -> str:
    for pattern in _NON_CLAIM_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text if text != "-0" else "0"


def _is_skipped_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in SKIP_KEYS:
        return True
    if lowered in KEEP_COUNT_KEYS:
        return False
    return lowered.endswith(SKIP_KEY_SUFFIXES)


def _is_percent_key(key: str | None) -> bool:
    if not key:
        return False
    lowered = key.lower()
    return any(marker in lowered for marker in PERCENT_KEY_MARKERS)


def _is_integer(value: Decimal) -> bool:
    return value == value.to_integral_value()


class _Candidate:
    """One numeric token found in the draft, with its display metadata."""

    __slots__ = ("text", "mantissa", "places", "multiplier", "is_percent", "context")

    def __init__(
        self,
        text: str,
        mantissa: Decimal,
        places: int,
        multiplier: Decimal,
        is_percent: bool,
        context: str,
    ) -> None:
        self.text = text
        self.mantissa = mantissa
        self.places = places
        self.multiplier = multiplier
        self.is_percent = is_percent
        self.context = context

    @property
    def value(self) -> Decimal:
        return self.mantissa * self.multiplier


class NumericEvidenceGuard:
    """Track every number a tool (or the user) supplied and flag the rest.

    ``constants`` defaults to :func:`reference_constants`; pass ``{}`` to
    ground nothing up front.
    """

    def __init__(self, constants: Mapping[str, Any] | None = None) -> None:
        self._grounded: set[Decimal] = set()
        self._percent_values: set[Decimal] = set()
        self.record_reference_constants(
            reference_constants() if constants is None else constants
        )

    # ------------------------------------------------------------------ record

    def _add(self, value: Decimal, *, percent: bool = False) -> None:
        value = abs(value)
        self._grounded.add(value)
        if percent:
            self._percent_values.add(value)

    def _add_text(self, text: str, *, percent: bool = False) -> None:
        for match in _TEXT_NUMBER.finditer(text):
            value = _to_decimal(match.group("num"))
            if value is not None:
                self._add(value, percent=percent or bool(match.group("pct")))

    def _walk(self, node: Any, depth: int = 0, key: str | None = None) -> None:
        if depth > 50 or node is None or isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            try:
                self._add(Decimal(str(node)), percent=_is_percent_key(key))
            except InvalidOperation:
                pass
            return
        if isinstance(node, Decimal):
            self._add(node, percent=_is_percent_key(key))
            return
        if isinstance(node, str):
            if _looks_like_json(node):
                try:
                    decoded = json.loads(node)
                except ValueError:
                    decoded = None
                if isinstance(decoded, (dict, list)):
                    self._walk(decoded, depth + 1, key)
                    return
            self._add_text(node, percent=_is_percent_key(key))
            return
        if isinstance(node, dict):
            if node.get("type") in SKIP_BLOCK_TYPES:
                return
            self._walk_sql_rows(node, depth)
            for child_key, value in node.items():
                child = str(child_key)
                if _is_skipped_key(child):
                    continue
                if child == "sample_rows" and self._has_sql_shape(node):
                    continue  # handled by _walk_sql_rows with column context
                self._walk(value, depth + 1, child)
            return
        if isinstance(node, (list, tuple, set)):
            for item in node:
                self._walk(item, depth + 1, key)

    @staticmethod
    def _has_sql_shape(node: dict[str, Any]) -> bool:
        columns = node.get("columns")
        rows = node.get("sample_rows")
        return (
            isinstance(columns, list)
            and isinstance(rows, list)
            and all(isinstance(column, str) for column in columns)
        )

    def _walk_sql_rows(self, node: dict[str, Any], depth: int) -> None:
        """Give SQL result cells their column name so ``pct_change`` columns count."""
        if not self._has_sql_shape(node):
            return
        columns: list[str] = node["columns"]
        for row in node["sample_rows"]:
            if isinstance(row, (list, tuple)):
                for position, cell in enumerate(row):
                    column = columns[position] if position < len(columns) else None
                    self._walk(cell, depth + 2, column)
            else:
                self._walk(row, depth + 2)

    def record_reference_constants(self, constants: Mapping[str, Any]) -> None:
        """Ground the operational constants the bot may quote about itself."""
        for value in constants.values():
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float, Decimal)):
                self._add(Decimal(str(value)))

    def record_user_text(self, text: str) -> None:
        """Numbers the user typed are grounded."""
        if text:
            self._add_text(str(text))

    def record_tool_result(self, tool_name: str, tool_input: Any, result: Any) -> None:
        """Ground every number the model can see in a tool call and its result."""
        self._walk(tool_input)
        self._walk(result)
        if tool_name == "calculate" and isinstance(result, dict):
            # A deliberate computation may be a percentage (pct, pct_change, x*100).
            if result.get("result") is not None:
                self._walk(result.get("result"), key="pct")
            display = result.get("result_display")
            if display is not None:
                self._add_text(str(display), percent=True)

    def record_history(self, messages: Iterable[dict[str, Any]]) -> None:
        """Ground prior-turn user text and tool results (not prior assistant text).

        A follow-up may legitimately restate a number an earlier tool returned;
        a number the bot previously produced without a tool stays unverified.
        """
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role != "user":
                continue
            if isinstance(content, str):
                self.record_user_text(content)
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    self.record_user_text(str(block.get("text") or ""))
                elif block_type == "tool_result":
                    self._walk(block.get("content"))

    # ---------------------------------------------------------------- validate

    def _candidates(self, draft: str) -> list[_Candidate]:
        prose = _prose_without_non_claims(draft)
        found: list[_Candidate] = []
        for match in _CANDIDATE.finditer(prose):
            raw_num = match.group("num")
            suffix = match.group("suffix")
            is_percent = bool(match.group("pct"))
            has_currency = bool(match.group("currency"))
            has_sign = bool(match.group("sign"))
            mantissa = _to_decimal(raw_num)
            if mantissa is None:
                continue
            is_plain_integer = (
                "." not in raw_num and not suffix and not is_percent and not has_currency
            )
            if is_plain_integer and not has_sign:
                if mantissa <= SMALL_INTEGER_MAX:
                    continue
                if re.fullmatch(r"\d{4}", raw_num) and YEAR_MIN <= mantissa <= YEAR_MAX:
                    continue
            elif is_plain_integer and mantissa <= SMALL_INTEGER_MAX:
                continue
            places = len(raw_num.split(".", 1)[1]) if "." in raw_num else 0
            multiplier = _SUFFIX_MULTIPLIERS.get(suffix or "", Decimal(1))
            text = match.group(0).strip()
            start = max(0, match.start() - CONTEXT_CHARS // 2)
            end = min(len(prose), match.end() + CONTEXT_CHARS // 2)
            context = " ".join(prose[start:end].split())
            found.append(
                _Candidate(text, mantissa, places, multiplier, is_percent, context)
            )
        return found

    @staticmethod
    def _rounds_to(scaled: Decimal, candidate: _Candidate) -> bool:
        if scaled == abs(candidate.value):
            return True
        quantum = Decimal(1).scaleb(-candidate.places)
        mantissa = abs(candidate.mantissa)
        try:
            in_display_units = scaled / candidate.multiplier
            for rounding in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                if in_display_units.quantize(quantum, rounding=rounding) == mantissa:
                    return True
        except InvalidOperation:
            return False
        return False

    def _matches(self, candidate: _Candidate) -> bool:
        if not candidate.is_percent:
            return any(self._rounds_to(grounded, candidate) for grounded in self._grounded)

        whole_percent = candidate.places == 0 and candidate.multiplier == 1
        rescales = (Decimal(100), Decimal("0.01"))
        # Percent values vouch at any scale; a whole-number percent claim is
        # never grounded by a plain integer (positional fields are integers).
        for grounded in self._percent_values:
            if self._rounds_to(grounded, candidate):
                return True
            if any(self._rounds_to(grounded * scale, candidate) for scale in rescales):
                return True
        for grounded in self._grounded:
            if _is_integer(grounded):
                if not whole_percent and self._rounds_to(grounded, candidate):
                    return True
                continue
            if self._rounds_to(grounded, candidate) and not whole_percent:
                return True
            if self._rounds_to(grounded * Decimal(100), candidate):
                return True
            if not whole_percent and self._rounds_to(grounded * Decimal("0.01"), candidate):
                return True
        return False

    def validate(self, draft: str) -> list[dict[str, str]]:
        """Return the quantity claims in ``draft`` that no tool result supports."""
        unverified: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in self._candidates(draft or ""):
            if candidate.text in seen:
                continue
            if self._matches(candidate):
                continue
            seen.add(candidate.text)
            unverified.append(
                {
                    "text": candidate.text,
                    "value": _format_decimal(candidate.value),
                    "context": candidate.context,
                }
            )
        return unverified

    # ----------------------------------------------------------------- outputs

    @staticmethod
    def revision_instruction(unverified: list[dict[str, str]]) -> str:
        figures = "; ".join(
            f"\"{item['text']}\" (in: ...{item['context']}...)" for item in unverified
        )
        return (
            "Revise your previous draft before answering the user. These figures do "
            "not match any number returned by a tool in this conversation: "
            f"{figures}. For each one, either compute it with the `calculate` tool "
            "(or a SQL query) and quote the tool's `result_display` exactly, or remove "
            "the figure. Do not perform arithmetic yourself. Do not change any figure "
            "that IS grounded in a tool result, and keep every other statement. Do not "
            "mention this internal correction request."
        )

    @staticmethod
    def annotate(draft: str, unverified: list[dict[str, str]]) -> str:
        if not unverified:
            return draft
        figures = ", ".join(item["text"] for item in unverified)
        note = f"_Figures not verified against a tool result: {figures}._"
        return f"{draft.rstrip()}\n\n{note}" if draft.strip() else note
