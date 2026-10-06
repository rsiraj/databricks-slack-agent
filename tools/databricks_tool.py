"""Databricks SQL tool — execute with internal polling and a statement_id escape hatch.

The Service Principal has read-only data access at the source, but the tool
ALSO enforces a defense-in-depth guard layer outside the LLM (per security
review):

  - Statement-type allowlist — SELECT / WITH / SHOW / DESCRIBE / DESC / EXPLAIN
    only, checked on the leading keyword AND on the parsed statement tree
    (sqlglot, Databricks dialect). A write anywhere in the tree — including
    `WITH ... INSERT`, `EXPLAIN DELETE`, or a MERGE — is rejected. When sqlglot
    cannot parse a SELECT / WITH statement at all (Databricks syntax it does
    not model, e.g. `FOR SYSTEM_VERSION AS OF`), the guard falls back to a
    literal- and comment-aware token scan for write / admin keywords and the
    function deny-list instead of rejecting, and logs `sql_parse_fallback`;
    Databricks remains the authority on syntax.
  - Single-statement-only — the statement is scanned with a literal- and
    comment-aware tokenizer so a ';' in a string or comment is ignored while a
    real second statement is rejected; sqlglot's statement split is a second
    check on the same rule.
  - Side-effecting / egress function deny-list — `http_request`,
    `remote_query`, `read_files`, `ai_query`, ... are rejected whether they
    appear as a function call in the tree, as a quoted function name, or in a
    fixed IDENTIFIER literal. Unresolved dynamic function names and SQL that
    cannot be tokenized safely are rejected before the parser fallback.
  - Server-side row cap — `row_limit=DEFAULT_ROW_LIMIT` is set on every
    SQL Statement Execution API call. Caps results regardless of what
    the SQL says; sets `truncated=true` in the response when applied.

Auth: PAT stored in AWS Secrets Manager (rotated externally). We pull `host` and
`token` from the secret JSON on first use and cache for ~5 min within the warm
container. On 401 from Databricks, we invalidate the cache and retry once. HTTP
429/5xx responses and network timeouts are retried within a small fixed budget;
job triggers use a Databricks idempotency token so those retries cannot launch
duplicate runs. Retry sleeps, per-request timeouts, and the SQL poll loop all
respect the invocation deadline in ``utils.deadline`` so a slow warehouse cannot
push the worker past the Lambda timeout.
"""

import functools
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Callable
from datetime import date as date_type
from datetime import datetime, time as datetime_time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import boto3
import requests
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType

from utils import deadline as run_deadline
from utils.audit import audit
from utils.ratelimit import check_sql_global

logger = logging.getLogger(__name__)

# Set DATABRICKS_WAREHOUSE_ID on the worker Lambda to your team's SQL warehouse.
# There is deliberately no default: SQL calls fail with a configuration error until
# it is set. The Service Principal whose PAT is in DATABRICKS_SECRET_ARN must have
# CAN_USE on this warehouse.
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

# Internal poll budget per single execute call (keeps us comfortably under Lambda timeout).
INITIAL_WAIT_S = "10s"
MAX_POLL_SECONDS = 90
POLL_INTERVAL_SECONDS = 3
# Seconds of the invocation deadline always left untouched so the model can
# still compose and post an answer after a slow tool call.
RESPONSE_RESERVE_SECONDS = 30.0
# Planning figure for one status poll round-trip when deciding whether another
# poll fits in the remaining budget. The actual request timeout is bounded
# separately by the deadline.
EXPECTED_POLL_REQUEST_SECONDS = 5.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

# Defense-in-depth caps (security review). Defaults are deliberately permissive
# enough to never affect a normal analyst query but tight enough to neutralise
# data-exfiltration attempts that slip past the prompt-level guardrails.
DEFAULT_ROW_LIMIT = int(os.environ.get("DATABRICKS_ROW_LIMIT", "1000"))
ALLOWED_FIRST_KEYWORDS = frozenset({"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"})
WRITE_KEYWORDS = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP", "ALTER", "TRUNCATE",
        "REPLACE", "COPY", "RESTORE", "OPTIMIZE", "VACUUM", "REORG", "CLONE",
    }
)
# Functions that leave the warehouse (HTTP egress, remote federation, model
# endpoints, streaming sources) or execute arbitrary code. Matched
# case-insensitively as function-call names in the parsed tree and in the SQL
# text with literals/comments removed.
DENIED_SQL_FUNCTIONS = frozenset(
    {
        "http_request",
        "remote_query",
        "read_files",
        "ai_query",
        "ai_gen",
        "ai_forecast",
        "vector_search",
        "read_kafka",
        "read_kinesis",
        "read_pulsar",
        "read_state_metadata",
        "python_udf",
        "reflect",
        "java_method",
        "sql",
        "execute_immediate",
    }
)
STATEMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")

# Secret cache (warm-container scoped).
_secrets_client = boto3.client("secretsmanager")
_creds_cache: dict[str, Any] = {"host": None, "token": None, "expires_at": 0.0}
CREDS_TTL_SECONDS = 300  # 5 minutes

# HTTP retries are deliberately short and bounded. SQL submissions are read-only,
# and Jobs run-now requests carry an idempotency token so a lost response cannot
# launch the same requested run more than once.
MAX_HTTP_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 10.0
RETRYABLE_STATUS_CODES = frozenset({429, *range(500, 600)})


class DatabricksToolError(RuntimeError):
    """Carries a structured tool error dict to the tool boundary.

    Raised deep inside helpers (credential loading, HTTP status handling) and
    converted back into the dict by ``_structured_errors`` so every public tool
    function returns ``{"error": ...}`` instead of leaking an exception.
    """

    def __init__(self, error: dict[str, Any]):
        super().__init__(str(error.get("error", "Databricks tool error")))
        self.error = error


def _structured_errors(fn: Callable[..., dict]) -> Callable[..., dict]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except DatabricksToolError as exc:
            return dict(exc.error)

    return wrapper


# ---------------------------------------------------------------------------
# SQL validation (defense-in-depth, outside the LLM layer)
# ---------------------------------------------------------------------------

# sqlglot node types that mutate data or schema. Any occurrence anywhere in the
# tree (top-level, inside a CTE, inside EXPLAIN) is rejected.
_WRITE_NODE_TYPES: tuple[type, ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterIndex",
        "TruncateTable", "Comment", "Copy", "LoadData", "Clone", "Undrop",
    )
    if hasattr(exp, name)
)
# Session / control / administrative statements. Not writes, but not read-only
# queries either.
_CONTROL_NODE_TYPES: tuple[type, ...] = tuple(
    getattr(exp, name)
    for name in (
        "Command", "Set", "Use", "Grant", "Revoke", "Transaction", "Commit", "Rollback",
        "Cache", "Uncache", "Refresh", "Analyze", "Declare", "Execute", "Kill", "Pragma",
        "Attach", "Detach", "Install", "Export", "Put", "Get", "Lock", "Summarize",
        "AlterSession",
    )
    if hasattr(exp, name)
)
_QUERY_TOP_LEVEL_TYPES: tuple[type, ...] = (exp.Select, exp.SetOperation)
_METADATA_TOP_LEVEL_TYPES: tuple[type, ...] = (exp.Describe, exp.Show)

# Databricks DESCRIBE forms sqlglot does not model. They are allowed only when
# the whole statement (comments removed) matches one of these read-only shapes;
# the tail character class excludes ';' and anything that could open a nested
# statement.
_META_TAIL = r"[\w\s.`'\"*%(),=:/-]*"
_DESCRIBE_ALLOWLIST = re.compile(
    r"^(?:DESCRIBE|DESC)\s+(?:(?:TABLE|HISTORY|DETAIL|FUNCTION|SCHEMA|DATABASE|CATALOG|"
    r"VOLUME|CONNECTION|PROVIDER|RECIPIENT|SHARE)\s+)?(?:(?:EXTENDED|FORMATTED)\s+)?"
    rf"(?!QUERY\b)[A-Za-z_`\"]{_META_TAIL}$",
    flags=re.IGNORECASE,
)
_EXPLAIN_PREFIX = re.compile(
    r"^EXPLAIN(?:\s+(?:EXTENDED|CODEGEN|COST|FORMATTED))?\s+", flags=re.IGNORECASE
)
_DESCRIBE_PREFIX = re.compile(r"^(?:DESCRIBE|DESC)\s+(?:QUERY\s+)?", flags=re.IGNORECASE)
_DENIED_FUNCTION_CALL = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, DENIED_SQL_FUNCTIONS))) + r")\s*\(",
    flags=re.IGNORECASE,
)
_EXECUTE_IMMEDIATE = re.compile(r"\bEXECUTE\s+IMMEDIATE\b", flags=re.IGNORECASE)
_ANSI_ESCAPES = re.compile(r"\x1b\[[0-9;]*m")

# Token-level guard used only when sqlglot cannot parse a SELECT / WITH
# statement. Session / admin keywords that are not writes but must never run
# through this tool. Scanned on the SQL with comments removed and literals /
# quoted identifiers collapsed, so `WHERE operation = 'DELETE'` is fine.
_FALLBACK_CONTROL_KEYWORDS = frozenset(
    {
        "GRANT", "REVOKE", "DENY", "SET", "USE", "REFRESH", "CACHE", "UNCACHE", "ANALYZE",
        "MSCK", "CALL", "EXECUTE", "DECLARE", "BEGIN", "COMMIT", "ROLLBACK", "LOCK",
    }
)
# REPLACE is a write keyword only as `REPLACE TABLE` / `REPLACE INTO`; the bare
# token is the common replace() string function, so it is matched separately.
_FALLBACK_BLOCKED_TOKEN = re.compile(
    r"\b("
    + "|".join(sorted((WRITE_KEYWORDS - {"REPLACE"}) | _FALLBACK_CONTROL_KEYWORDS))
    + r")\b|\b(REPLACE)\s+(?:TABLE|INTO)\b",
    flags=re.IGNORECASE,
)
# Every Databricks SHOW statement is read-only, so SHOW is validated by the
# token scan rather than an allowlist of forms. SHOW CREATE TABLE is the one
# read-only form that contains a write keyword; only this exact leading phrase
# is exempted, so a second CREATE anywhere in the statement is still blocked.
_SHOW_CREATE_TABLE_PREFIX = re.compile(r"^SHOW\s+CREATE\s+TABLE\b", flags=re.IGNORECASE)


def _block_comment_end(sql: str, start: int) -> int:
    """Return the end of a possibly nested Databricks block comment."""
    depth = 1
    position = start + 2
    while position < len(sql):
        pair = sql[position:position + 2]
        if pair == "/*":
            depth += 1
            position += 2
        elif pair == "*/":
            depth -= 1
            position += 2
            if depth == 0:
                return position
        else:
            position += 1
    return len(sql)


def _tokenize_sql(sql: str) -> list[tuple[str, str]]:
    """Split SQL into ``(kind, text)`` chunks: ``code``, ``literal``, ``comment``.

    A single left-to-right scan so a comment marker inside a literal and a
    quote inside a comment are both handled correctly (regex stripping in
    either order gets one of those wrong). Handles single/double-quoted
    strings with backslash escapes and doubled quotes, ``r'raw'`` strings
    without escapes, backtick identifiers, ``--`` line comments, and ``/* */``
    nested block comments (an unterminated comment or literal runs to the end).
    """
    chunks: list[tuple[str, str]] = []
    code_start = 0
    i = 0
    n = len(sql)

    def flush_code(end: int) -> None:
        if end > code_start:
            chunks.append(("code", sql[code_start:end]))

    while i < n:
        ch = sql[i]
        two = sql[i:i + 2]
        if two == "--":
            flush_code(i)
            end = sql.find("\n", i)
            end = n if end == -1 else end
            chunks.append(("comment", sql[i:end]))
            i = code_start = end
            continue
        if two == "/*":
            flush_code(i)
            end = _block_comment_end(sql, i)
            chunks.append(("comment", sql[i:end]))
            i = code_start = end
            continue
        if ch in "'\"`":
            raw = (
                ch != "`"
                and i > 0
                and sql[i - 1] in "rR"
                and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_"))
            )
            flush_code(i - 1 if raw else i)
            j = i + 1
            while j < n:
                if ch != "`" and not raw and sql[j] == "\\":
                    j += 2
                    continue
                if sql[j] == ch:
                    break
                j += 1
            end = min(j + 1, n)
            chunks.append(("literal", sql[(i - 1 if raw else i):end]))
            i = code_start = end
            continue
        i += 1
    flush_code(n)
    return chunks


def _sql_without_comments(sql: str) -> str:
    """Return the SQL with comments replaced by a single space, literals intact."""
    return "".join(text if kind != "comment" else " " for kind, text in _tokenize_sql(sql))


def _sql_code_only(sql: str) -> str:
    """Return the SQL with comments removed and every literal collapsed to ``''``."""
    parts = []
    for kind, text in _tokenize_sql(sql):
        if kind == "code":
            parts.append(text)
        elif kind == "literal":
            parts.append("''")
        else:
            parts.append(" ")
    return "".join(parts)


def _strip_leading_sql_comments_and_whitespace(sql: str) -> str:
    """Return `sql` with leading whitespace + line/block comments removed."""
    cleaned = sql
    while True:
        stripped = cleaned.lstrip()
        if stripped.startswith("--"):
            newline_idx = stripped.find("\n")
            if newline_idx == -1:
                return ""
            cleaned = stripped[newline_idx + 1:]
        elif stripped.startswith("/*"):
            cleaned = stripped[_block_comment_end(stripped, 0):]
        else:
            return stripped


def _first_sql_keyword(sql: str) -> str:
    """Return the first SQL keyword (uppercased), ignoring leading comments."""
    cleaned = _strip_leading_sql_comments_and_whitespace(sql)
    match = re.match(r"^([A-Za-z]+)", cleaned)
    return match.group(1).upper() if match else ""


def _has_multiple_statements(sql: str) -> bool:
    """True if `sql` contains more than one non-trivial statement.

    Uses the literal- and comment-aware tokenizer so an embedded ';' (e.g.
    `SELECT 'a;b'`, `SELECT 'a -- b'; ...` is still caught, `-- ;` is not)
    is classified correctly. Anything after the first code-level ';' that
    is not whitespace or a comment counts as a second statement.
    """
    seen_terminator = False
    for kind, text in _tokenize_sql(sql):
        if kind == "comment":
            continue
        if seen_terminator:
            if kind == "literal" or text.strip():
                return True
            continue
        if kind == "code" and ";" in text:
            _, _, tail = text.partition(";")
            if tail.strip():
                return True
            seen_terminator = True
    return False


def _parse_statements(sql: str) -> tuple[list[exp.Expression] | None, str | None]:
    """Parse with sqlglot (Databricks, then Spark). Returns (statements, error)."""
    sqlglot_logger = logging.getLogger("sqlglot")
    previous_level = sqlglot_logger.level
    # sqlglot logs a warning whenever it falls back to a bare Command; those
    # statements are rejected below, so the noise adds nothing.
    sqlglot_logger.setLevel(logging.ERROR)
    last_error: str | None = None
    try:
        for dialect in ("databricks", "spark"):
            try:
                parsed = sqlglot.parse(sql, read=dialect)
            except SqlglotError as exc:
                # ParseError for grammar sqlglot does not model, TokenError for
                # an unterminated literal / comment. Both mean "cannot verify
                # structurally", never "safe".
                last_error = _ANSI_ESCAPES.sub("", str(exc)).splitlines()[0][:200]
                continue
            statements = [
                statement
                for statement in parsed
                if statement is not None and not isinstance(statement, exp.Semicolon)
            ]
            return statements, None
    finally:
        sqlglot_logger.setLevel(previous_level)
    return None, last_error or "unknown parse error"


def _function_name(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Anonymous):
        return (node.name or "").lower() or None
    if isinstance(node, exp.Func):
        try:
            return node.sql_name().lower()
        except Exception:  # pragma: no cover - defensive against exotic nodes
            return None
    return None


def _write_statement_error(kind: str) -> dict:
    return {
        "error": (
            f"Write statements are not allowed, even inside a CTE, subquery, or EXPLAIN. "
            f"Found: {kind}. Only a single read-only SELECT / WITH ... SELECT / SHOW / "
            "DESCRIBE / EXPLAIN <query> may be executed."
        ),
        "validation_failure": "write_statement",
        "first_keyword": kind,
    }


def _denied_function_error(name: str) -> dict:
    return {
        "error": (
            f"The function {name}() is not permitted through this tool because it can "
            "leave the warehouse, call external services, or execute arbitrary code. "
            "Query Unity Catalog tables and views directly instead."
        ),
        "validation_failure": "denied_function",
        "denied_function": name,
    }


def _function_validation_error(sql: str) -> dict | None:
    """Check function names before the parser's permissive syntax fallback.

    Tokenization preserves quoted identifiers and string values, unlike the
    keyword-only fallback. IDENTIFIER can invoke a function whose name is a
    string, so only a single literal naming a fixed function is verifiable.
    IDENTIFIER used for a table or column remains supported.
    """
    try:
        tokens = sqlglot.Dialect.get_or_raise("databricks").tokenize(sql)
    except SqlglotError:
        return {
            "error": "Could not tokenize SQL safely. Close any unfinished literals or comments and retry.",
            "validation_failure": "unparseable_sql",
        }

    literal_types = {TokenType.STRING, TokenType.RAW_STRING}
    for index, token in enumerate(tokens[:-1]):
        if (
            token.token_type in literal_types
            or tokens[index + 1].token_type != TokenType.L_PAREN
        ):
            continue
        name = token.text.lower()
        if name in DENIED_SQL_FUNCTIONS:
            return _denied_function_error(name)
        if name != "identifier":
            continue

        depth = 0
        closing = None
        for position in range(index + 1, len(tokens)):
            kind = tokens[position].token_type
            if kind == TokenType.L_PAREN:
                depth += 1
            elif kind == TokenType.R_PAREN:
                depth -= 1
                if depth == 0:
                    closing = position
                    break
        if (
            closing is None
            or closing + 1 >= len(tokens)
            or tokens[closing + 1].token_type != TokenType.L_PAREN
        ):
            continue

        arguments = tokens[index + 2:closing]
        fixed_name = (
            arguments[0].text
            if len(arguments) == 1 and arguments[0].token_type in literal_types
            else ""
        )
        # Allow ordinary qualified names, with optional backticks on each part.
        # Expressions, parameters, and ambiguous names fail closed.
        component = r"(?:[A-Za-z_][\w$]*|`[A-Za-z_][\w$]*`)"
        if not re.fullmatch(rf"{component}(?:\s*\.\s*{component})*", fixed_name):
            return {
                "error": "Dynamic function invocation requires a single literal function name; use an explicit function call instead.",
                "validation_failure": "unresolved_function",
            }
        resolved_name = fixed_name.rsplit(".", 1)[-1].strip().strip("`").lower()
        if resolved_name in DENIED_SQL_FUNCTIONS:
            return _denied_function_error(resolved_name)
        if resolved_name == "identifier":
            return {
                "error": "Nested dynamic function invocation cannot be verified; use an explicit function call instead.",
                "validation_failure": "unresolved_function",
            }
    return None


def _blocked_token_error(body: str, *, skip_prefix: re.Pattern[str] | None = None) -> dict | None:
    """Token-level read-only check shared by the parse fallback and SHOW.

    Rejects when ``body`` holds more than one statement, any write / admin
    keyword as a token outside literals and comments, or a deny-listed
    function call. ``skip_prefix`` names a leading phrase that is exempt from
    the keyword scan (``SHOW CREATE TABLE``); everything after it is still
    scanned. Returns None when the statement may proceed.
    """
    if _has_multiple_statements(body):
        return {
            "error": "Only one statement is allowed per call.",
            "validation_failure": "multiple_statements",
        }
    code_only = _sql_code_only(body)
    scanned = code_only
    if skip_prefix is not None:
        prefix = skip_prefix.match(code_only)
        if prefix:
            scanned = code_only[prefix.end():]
    blocked = _FALLBACK_BLOCKED_TOKEN.search(scanned)
    if blocked:
        keyword = (blocked.group(1) or blocked.group(2)).upper()
        if keyword in WRITE_KEYWORDS:
            return _write_statement_error(keyword)
        return {
            "error": (
                f"Only read-only statements are allowed (SELECT / WITH / SHOW / DESCRIBE / "
                f"EXPLAIN). Got: {keyword}."
            ),
            "validation_failure": "disallowed_statement_type",
            "first_keyword": keyword,
        }
    denied_call = _DENIED_FUNCTION_CALL.search(code_only)
    if denied_call:
        return _denied_function_error(denied_call.group(1).lower())
    return None


def _token_scan_error(body: str, first: str, parse_error: str | None) -> dict | None:
    """Fallback for a SELECT / WITH statement sqlglot cannot parse.

    sqlglot is the stronger check when it parses; when it cannot (Databricks
    syntax it does not model), rejecting would block valid read-only SQL, so
    the guard falls back to the pre-sqlglot token scan (``_blocked_token_error``):
    single statement, no write / admin keyword anywhere as a token outside
    literals and comments, and no deny-listed function call. Databricks stays
    the authority on syntax. Returns None when the statement may proceed; the
    skipped structural check is recorded as a ``sql_parse_fallback`` audit event.
    """
    error = _blocked_token_error(body)
    if error is not None:
        return error
    audit(
        "sql_parse_fallback",
        first_keyword=first,
        parse_error=parse_error,
        check="token_scan",
        sql=body,
    )
    return None


def _read_only_structure_error(sql: str, *, allow_metadata: bool = True) -> dict | None:
    """Validate the parsed shape of one statement. None means read-only.

    ``allow_metadata=False`` is used for the body of EXPLAIN / DESCRIBE QUERY,
    where only a query expression (SELECT / set operation / WITH ... SELECT)
    is acceptable.
    """
    function_error = _function_validation_error(sql)
    if function_error is not None:
        return function_error
    body = _strip_leading_sql_comments_and_whitespace(sql)
    first = _first_sql_keyword(body)

    if first == "EXPLAIN":
        if not allow_metadata:
            return {
                "error": "Nested EXPLAIN is not allowed.",
                "validation_failure": "disallowed_statement_type",
                "first_keyword": "EXPLAIN",
            }
        inner = _EXPLAIN_PREFIX.sub("", body, count=1)
        inner_first = _first_sql_keyword(inner)
        if inner == body or not inner.strip():
            return {
                "error": "EXPLAIN must be followed by a read-only query.",
                "validation_failure": "unparseable_sql",
                "first_keyword": "EXPLAIN",
            }
        if inner_first in WRITE_KEYWORDS:
            return _write_statement_error(inner_first)
        if inner_first not in {"SELECT", "WITH"} and not inner.lstrip().startswith("("):
            return {
                "error": (
                    "EXPLAIN is only allowed for a read-only query (SELECT / WITH ... SELECT). "
                    f"Got: EXPLAIN {inner_first or '(unrecognised)'}."
                ),
                "validation_failure": "disallowed_statement_type",
                "first_keyword": "EXPLAIN",
            }
        return _read_only_structure_error(inner, allow_metadata=False)

    if first == "SHOW":
        # Every Databricks SHOW statement is read-only, and sqlglot models them
        # as an opaque Command, so the structural rule is the token scan: one
        # statement, no write / admin keyword token (SHOW CREATE TABLE's leading
        # CREATE excepted), no deny-listed function.
        if not allow_metadata:
            return {
                "error": "SHOW is not allowed inside EXPLAIN or DESCRIBE QUERY.",
                "validation_failure": "disallowed_statement_type",
                "first_keyword": "SHOW",
            }
        return _blocked_token_error(body, skip_prefix=_SHOW_CREATE_TABLE_PREFIX)

    statements, parse_error = _parse_statements(body)
    if statements is None:
        if first in {"DESCRIBE", "DESC"} and allow_metadata:
            remainder = _DESCRIBE_PREFIX.sub("", body, count=1)
            remainder_first = _first_sql_keyword(remainder)
            if remainder_first in {"SELECT", "WITH"} or remainder.lstrip().startswith("("):
                return _read_only_structure_error(remainder, allow_metadata=False)
            if remainder_first in WRITE_KEYWORDS:
                return _write_statement_error(remainder_first)
            if _DESCRIBE_ALLOWLIST.match(_sql_without_comments(body).strip()):
                return None
        if first in WRITE_KEYWORDS:
            return _write_statement_error(first)
        if first in {"SELECT", "WITH"}:
            return _token_scan_error(body, first, parse_error)
        return {
            "error": (
                f"Could not parse this as a single Databricks SQL statement ({parse_error}). "
                "Fix the syntax and resubmit; the tool only executes statements it can verify "
                "as read-only."
            ),
            "validation_failure": "unparseable_sql",
            "first_keyword": first,
        }

    if len(statements) > 1:
        return {
            "error": "Only one statement is allowed per call.",
            "validation_failure": "multiple_statements",
        }
    if not statements:
        return {"error": "Empty SQL statement.", "validation_failure": "empty"}

    statement = statements[0]
    for node in statement.walk():
        if isinstance(node, _WRITE_NODE_TYPES):
            return _write_statement_error(node.key.upper())
        if isinstance(node, _CONTROL_NODE_TYPES):
            label = node.key.upper()
            if isinstance(node, exp.Command):
                label = (node.name or "COMMAND").upper()
            return {
                "error": (
                    f"Only read-only statements are allowed (SELECT / WITH / SHOW / DESCRIBE / "
                    f"EXPLAIN). Got: {label}."
                ),
                "validation_failure": "disallowed_statement_type",
                "first_keyword": label,
            }
        function_name = _function_name(node)
        if function_name in DENIED_SQL_FUNCTIONS:
            return _denied_function_error(function_name)

    allowed_types = _QUERY_TOP_LEVEL_TYPES + (_METADATA_TOP_LEVEL_TYPES if allow_metadata else ())
    if not isinstance(statement, allowed_types):
        return {
            "error": (
                "Only read-only statements are allowed (SELECT / WITH / SHOW / DESCRIBE / "
                f"EXPLAIN). Got: {statement.key.upper()}."
            ),
            "validation_failure": "disallowed_statement_type",
            "first_keyword": statement.key.upper(),
        }
    return None


def _validate_sql(sql: str) -> dict | None:
    """Return an error dict if `sql` should be rejected, else None.

    Errors mirror the shape of the existing tool error responses so the
    agent loop handles them naturally — `is_error` propagates, and the
    rejection message is human-readable for the user.

    Layers, in order: leading-keyword allowlist, tokenizer-based single
    statement check, metadata/shape lint, function deny-list on the SQL text,
    then sqlglot structural validation of the whole statement tree. A SELECT /
    WITH statement sqlglot cannot parse falls back to a token-level scan for
    write / admin keywords (see ``_token_scan_error``) rather than being
    rejected outright.
    """
    if not sql or not sql.strip():
        return {
            "error": "Empty SQL statement.",
            "validation_failure": "empty",
        }

    first = _first_sql_keyword(sql)
    if first not in ALLOWED_FIRST_KEYWORDS:
        return {
            "error": (
                f"Only read-only statements are allowed "
                f"(SELECT / WITH / SHOW / DESCRIBE / EXPLAIN). "
                f"Got: {first or '(unrecognised)'}."
            ),
            "validation_failure": "disallowed_statement_type",
            "first_keyword": first,
        }

    if _has_multiple_statements(sql):
        return {
            "error": "Only one statement is allowed per call.",
            "validation_failure": "multiple_statements",
        }

    syntax_only = _sql_code_only(sql)
    if first in {"SHOW", "DESCRIBE", "DESC"} and re.search(
        r"\bLIMIT\s+\d+\s*;?\s*$", syntax_only, flags=re.IGNORECASE
    ):
        return {
            "error": (
                "LIMIT is not valid on SHOW or DESCRIBE statements. Run the metadata statement "
                "without LIMIT; metadata responses are already compacted."
            ),
            "validation_failure": "invalid_metadata_limit",
        }

    direct_window_in_where = re.search(
        r"\bWHERE\s+(?:ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|NTILE)\s*\([^)]*\)"
        r"\s*OVER\s*\(",
        syntax_only,
        flags=re.IGNORECASE,
    )
    if direct_window_in_where:
        return {
            "error": (
                "Window functions cannot be evaluated directly in WHERE. Use QUALIFY or compute "
                "the window value in a CTE and filter it in the outer query."
            ),
            "validation_failure": "window_function_in_where",
        }

    denied_call = _DENIED_FUNCTION_CALL.search(syntax_only)
    if denied_call:
        return _denied_function_error(denied_call.group(1).lower())
    if _EXECUTE_IMMEDIATE.search(syntax_only):
        return _denied_function_error("execute_immediate")

    return _read_only_structure_error(sql)


def _repair_metadata_limit(sql: str) -> tuple[str, bool]:
    """Remove a trailing LIMIT that Databricks metadata statements reject."""
    if _first_sql_keyword(sql) not in {"SHOW", "DESCRIBE", "DESC"}:
        return sql, False
    repaired, replacements = re.subn(
        r"\s+LIMIT\s+\d+\s*(;?)\s*$",
        r"\1",
        sql,
        count=1,
        flags=re.IGNORECASE,
    )
    return repaired.strip(), replacements == 1


def _load_creds(force_refresh: bool = False) -> tuple[str, str]:
    """Return (host, token), pulled from Secrets Manager and cached for 5 min."""
    now = time.time()
    if not force_refresh and _creds_cache["token"] and _creds_cache["expires_at"] > now:
        return _creds_cache["host"], _creds_cache["token"]

    resp = _secrets_client.get_secret_value(SecretId=os.environ["DATABRICKS_SECRET_ARN"])
    try:
        blob = json.loads(resp["SecretString"])
    except (KeyError, TypeError, ValueError):
        blob = None
    if not isinstance(blob, dict):
        raise DatabricksToolError(
            {
                "error": (
                    "The Databricks credential secret is not a JSON object with host and "
                    "token keys. Ask in #data-assistant-help to check DATABRICKS_SECRET_ARN."
                ),
                "config_failure": "malformed_credential_secret",
            }
        )
    missing = [key for key in ("host", "token") if not str(blob.get(key) or "").strip()]
    if missing:
        # Key names only — never echo the secret's values.
        raise DatabricksToolError(
            {
                "error": (
                    "The Databricks credential secret is missing required key(s): "
                    f"{', '.join(missing)}. Ask in #data-assistant-help to check DATABRICKS_SECRET_ARN."
                ),
                "config_failure": "missing_credential_keys",
                "missing_keys": missing,
            }
        )

    # Normalize host to bare hostname (no scheme, no trailing slash).
    host = str(blob["host"]).strip().rstrip("/")
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    token = str(blob["token"]).strip()

    _creds_cache["host"] = host
    _creds_cache["token"] = token
    _creds_cache["expires_at"] = now + CREDS_TTL_SECONDS
    return host, token

DATABRICKS_TOOLS = [
    {
        "name": "databricks_execute_sql",
        "description": (
            "Execute a read-only SQL statement against the configured Databricks SQL warehouse. "
            "The tool polls internally for up to ~90 seconds. If the query is still running at that "
            "point, the tool returns the statement_id and a PENDING/RUNNING status — surface this to "
            "the user and ask them to reply with 'check on statement <id>', then call "
            "databricks_get_statement to retrieve results. "
            f"Results are capped server-side at {DEFAULT_ROW_LIMIT} rows; when this fires the "
            "response's `truncated` field is true. Every successful response includes `is_sample` "
            "and a `result_scope` instruction — follow it, disclose sampling/truncation, and offer "
            "to narrow the query rather than paginating raw rows through Slack. The tool also "
            "rejects pre-flight any statement that isn't a single SELECT / WITH / SHOW / "
            "DESCRIBE / EXPLAIN, any write anywhere in the statement (including inside a CTE "
            "or EXPLAIN), and functions that leave the warehouse such as http_request, "
            "remote_query, read_files, or ai_query. Before analytical SQL references a "
            "system.* object outside "
            "information_schema, call databricks_describe_object for that exact object in the "
            "current user turn. For an explicitly defined strict funnel, the outermost SELECT "
            "must preserve the exact funnel_numerator and funnel_denominator aliases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A single read-only SQL statement (SELECT / WITH / SHOW / DESCRIBE / "
                        "EXPLAIN). Multiple statements separated by ';' are rejected. Do not use "
                        "an analytical system.* query as a schema probe; describe the object first."
                    ),
                },
                "catalog": {
                    "type": "string",
                    "description": "Optional default catalog.",
                },
                "schema": {
                    "type": "string",
                    "description": "Optional default schema.",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "databricks_get_statement",
        "description": (
            "Retrieve the status/results of a previously submitted SQL statement by its statement_id. "
            "Use this when the user follows up on a long-running query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "statement_id": {
                    "type": "string",
                    "description": "The statement_id returned by databricks_execute_sql.",
                }
            },
            "required": ["statement_id"],
        },
    },
    {
        "name": "databricks_find_objects",
        "description": (
            "Find accessible Databricks tables and views by object name using typed "
            "information-schema discovery. Use exact mode first when the user names an object; "
            "use contains only when the name is partial. This avoids guessing catalog/schema paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Table or view name, without catalog/schema qualifiers.",
                },
                "match_mode": {
                    "type": "string",
                    "enum": ["exact", "contains"],
                    "description": "Name matching mode (default exact).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum objects to return (default 20, max 100).",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "databricks_describe_object",
        "description": (
            "Return structured metadata for one fully qualified Databricks table or view. "
            "Use catalog/schema/name values returned by databricks_find_objects. The result has "
            "`object` (the table itself: full_name, object_type, table_type, comment), "
            "`column_count` (the authoritative number of columns; never count anything else), "
            "and `columns` (the complete typed list, one entry per column with name, data_type, "
            "is_nullable, comment). This tool does not run user-authored SQL or sample object data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "schema": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["catalog", "schema", "name"],
        },
    },
    {
        "name": "databricks_run_job",
        "description": (
            "Request a shared Slack confirmation button for an existing Databricks job. This "
            "tool NEVER launches a job; only a verified channel member's button click can launch it. The Service "
            "Principal has CAN_MANAGE_RUN only on an approved subset of DAB and UI-managed jobs. "
            "A 403 means this bot is not authorized to run the requested job; it does not by "
            "itself prove that the schedule is paused. Cannot edit, create, pause, unpause, or "
            "delete jobs. Inspect the job and recent runs, then call this tool to display the "
            "exact job and parameters for confirmation. Do not ask for a typed yes. Returns "
            "confirmation_pending, not a run_id; end the turn after explaining the button."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "Databricks job_id (numeric). The history table stores it as a string; convert to int.",
                },
                "notebook_params": {
                    "type": "object",
                    "description": "Optional notebook parameters as key-value pairs of strings.",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "databricks_get_run_status",
        "description": (
            "Fetch the status of a triggered job run by run_id. Returns life_cycle_state "
            "(PENDING / RUNNING / TERMINATING / TERMINATED / SKIPPED / INTERNAL_ERROR), "
            "result_state when terminated (SUCCESS / FAILED / TIMEDOUT / CANCELED), and "
            "timing info. Use to follow up on a previously triggered run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "The run_id posted by the confirmed launch handler or retrieved from run history.",
                },
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "databricks_cancel_run",
        "description": (
            "Cancel an in-progress job run by run_id. Use as a corrective action when a "
            "run was triggered in error or needs to be stopped. Cancellation is async — "
            "the run transitions through TERMINATING before reaching TERMINATED with "
            "result_state=CANCELED. No confirmation needed (it's a corrective action)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "The run_id to cancel.",
                },
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "databricks_get_recent_runs",
        "description": (
            "Fetch the last N completed runs for a job_id, with their durations and a "
            "summary block (median / min / max duration of successful runs). Use this "
            "before triggering a job to give the user a duration estimate in the "
            "confirmation prompt, and after triggering to set expectations for when to "
            "come back. Also useful as a standalone answer to 'how long does this job "
            "typically take?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "Databricks job_id (numeric).",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many recent completed runs to fetch (default 10, max 25).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "databricks_list_job_runs",
        "description": (
            "List job runs visible to the bot across the Databricks workspace for one calendar "
            "date, using the Jobs REST API rather than SQL system tables. Use this for questions "
            "such as 'what jobs ran today?' or 'what is currently running?'. The date defaults "
            "to today in America/New_York. When active_only is true and date is omitted, all "
            "currently active visible runs are returned regardless of when they started. Returns "
            "up to 100 runs with job/run IDs, names, states, timings, and run-page URLs. These are "
            "observed executions, not schedule configuration; never infer or label a cadence from "
            "the timestamps alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": (
                        "Optional calendar date in YYYY-MM-DD format. Defaults to today in "
                        "America/New_York unless active_only is true; omit it with active_only "
                        "to list all currently active runs."
                    ),
                },
                "active_only": {
                    "type": "boolean",
                    "description": "If true, return only active runs for the selected date.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum runs to return (default 100, max 100).",
                },
            },
        },
    },
]


# Allowlisted API route prefixes — the URL is always built from this table plus
# the host pulled from Secrets Manager.  No URL string ever enters as a parameter.
_API_ROUTES: dict[str, str] = {
    "sql": "2.0/sql/statements",
    "jobs": "2.1/jobs",
}


def _headers() -> dict:
    _, token = _load_creds()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _build_url(api: str, path: str = "") -> str:
    """Construct a Databricks API URL from the allowlisted route table.

    `api`  — must be a key of ``_API_ROUTES`` (allowlist-validated here).
    `path` — an optional sub-path appended after the fixed prefix (e.g. a
             statement_id or a fixed verb like ``runs/get``).  The host is
             always sourced from Secrets Manager, never from caller input.
    """
    if api not in _API_ROUTES:
        raise ValueError(f"Unknown API type {api!r}. Allowed: {sorted(_API_ROUTES)}")
    host, _ = _load_creds()
    base = f"https://{host}/api/{_API_ROUTES[api]}"
    return f"{base}/{path}" if path else base


def _retry_delay_seconds(response: requests.Response | None, retry_number: int) -> float:
    """Return a capped Retry-After or exponential delay with small jitter."""
    retry_after = (getattr(response, "headers", None) or {}).get("Retry-After")
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), MAX_RETRY_DELAY_SECONDS)
        except (TypeError, ValueError):
            pass

    exponential = min(2 ** max(retry_number - 1, 0), MAX_RETRY_DELAY_SECONDS)
    jitter_window_ms = int(min(0.25, MAX_RETRY_DELAY_SECONDS - exponential) * 1000)
    jitter = secrets.randbelow(jitter_window_ms + 1) / 1000 if jitter_window_ms else 0.0
    return exponential + jitter


def _retry_budget_seconds() -> float | None:
    """Seconds we may still spend waiting on Databricks, or None when no deadline is set.

    Always leaves ``RESPONSE_RESERVE_SECONDS`` for the model to compose a reply.
    """
    remaining = run_deadline.remaining_seconds()
    if remaining is None:
        return None
    return max(0.0, remaining - RESPONSE_RESERVE_SECONDS)


def _can_wait(delay_seconds: float) -> bool:
    budget = _retry_budget_seconds()
    return budget is None or delay_seconds <= budget


def _request_timeout(requested: float = DEFAULT_REQUEST_TIMEOUT_SECONDS) -> float:
    """Per-request timeout that cannot outlive the invocation deadline minus the reserve."""
    budget = _retry_budget_seconds()
    if budget is None:
        return requested
    return run_deadline.bounded_timeout(min(requested, max(1.0, budget)), floor=1.0)


def _response_error_detail(response: Any) -> tuple[str | None, str | None]:
    """Return (error_code, message) from a Databricks error body, without headers."""
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        code = payload.get("error_code")
        message = payload.get("message")
        if message is None and "error" in payload:
            message = payload["error"]
        if isinstance(message, dict):
            message = message.get("message") or json.dumps(message)[:500]
        code_text = str(code)[:100] if code else None
        message_text = str(message).strip()[:500] if message else None
        return code_text, message_text
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return None, text.strip()[:300]
    return None, None


def _http_error(response: Any, operation: str) -> dict | None:
    """Structured error for a non-2xx response, else None.

    Includes the Databricks ``error_code`` and ``message`` so the model gets
    something actionable. Only the body is read — never request headers or
    the bearer token.
    """
    status = getattr(response, "status_code", None)
    if status is None or status < 400:
        return None
    code, message = _response_error_detail(response)
    if status in RETRYABLE_STATUS_CODES:
        text = (
            f"Databricks {operation} is temporarily unavailable (HTTP {status} after "
            f"{MAX_HTTP_ATTEMPTS} attempts)"
        )
    elif status in {401, 403}:
        text = f"Databricks {operation} was refused (HTTP {status})"
    else:
        text = f"Databricks {operation} request failed (HTTP {status})"
    detail = " ".join(part for part in ((f"[{code}]" if code else ""), message or "") if part)
    if detail:
        text = f"{text}: {detail}"
    error: dict[str, Any] = {"error": text, "status_code": status}
    if code:
        error["error_code"] = code
    return error


def _ensure_ok(response: Any, operation: str) -> None:
    error = _http_error(response, operation)
    if error is not None:
        raise DatabricksToolError(error)


def _safe_statement_id(statement_id: Any) -> str | None:
    """Return the statement_id if it is a plain identifier safe for a URL path segment."""
    if not isinstance(statement_id, str):
        return None
    return statement_id if STATEMENT_ID_PATTERN.match(statement_id) else None


def _invalid_statement_id_error(statement_id: Any) -> dict:
    return {
        "error": (
            "statement_id must be 8-128 characters of letters, digits, '-' or '_' exactly "
            "as returned by databricks_execute_sql."
        ),
        "validation_failure": "invalid_argument",
        "argument": "statement_id",
    }


def _request_operation(method: str, api: str, path: str) -> str:
    """Return a stable, non-sensitive label for request telemetry."""
    route = path or "execute"
    if api == "sql" and method.upper() == "GET":
        route = "statement"
    return f"{api}.{route.replace('/', '.')}"


def _request_with_retry(method: str, api: str, path: str = "", **kwargs) -> requests.Response:
    """Send a bounded request, refreshing auth once and retrying transient failures.

    The full URL is constructed internally from the ``_API_ROUTES`` allowlist
    and the host in Secrets Manager — no URL string is accepted as a parameter,
    eliminating the SSRF taint path flagged by static analysis.
    """
    auth_refreshed = False
    transient_failure_count = 0
    operation = _request_operation(method, api, path)
    requested_timeout = kwargs.pop("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        url = _build_url(api, path)
        # Re-bounded on every attempt so a retry never outlives the invocation deadline.
        kwargs["timeout"] = _request_timeout(requested_timeout)
        try:
            response = requests.request(method, url, headers=_headers(), **kwargs)
        except requests.exceptions.SSLError:
            # Certificate/configuration failures are deterministic and should not
            # consume the retry budget.
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            transient_failure_count += 1
            delay = _retry_delay_seconds(None, attempt)
            will_retry = attempt < MAX_HTTP_ATTEMPTS and _can_wait(delay)
            delay = delay if will_retry else None
            audit(
                "databricks_transient_failure",
                operation=operation,
                attempt=attempt,
                max_attempts=MAX_HTTP_ATTEMPTS,
                failure_type=type(exc).__name__,
                status_code=None,
                will_retry=will_retry,
                retry_delay_seconds=delay,
            )
            if not will_retry:
                raise
            logger.warning(
                "Transient Databricks network failure; retrying attempt %d/%d in %.2fs",
                attempt + 1,
                MAX_HTTP_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            continue

        if response.status_code == 401 and not auth_refreshed and attempt < MAX_HTTP_ATTEMPTS:
            logger.warning("Databricks returned 401 — refreshing cached secret and retrying")
            _load_creds(force_refresh=True)
            auth_refreshed = True
            continue

        if (
            response.status_code in RETRYABLE_STATUS_CODES
            and attempt < MAX_HTTP_ATTEMPTS
            and _can_wait(_retry_delay_seconds(response, attempt))
        ):
            delay = _retry_delay_seconds(response, attempt)
            transient_failure_count += 1
            audit(
                "databricks_transient_failure",
                operation=operation,
                attempt=attempt,
                max_attempts=MAX_HTTP_ATTEMPTS,
                failure_type="http",
                status_code=response.status_code,
                will_retry=True,
                retry_delay_seconds=delay,
            )
            logger.warning(
                "Databricks returned transient HTTP %d; retrying attempt %d/%d in %.2fs",
                response.status_code,
                attempt + 1,
                MAX_HTTP_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            transient_failure_count += 1
            audit(
                "databricks_transient_failure",
                operation=operation,
                attempt=attempt,
                max_attempts=MAX_HTTP_ATTEMPTS,
                failure_type="http",
                status_code=response.status_code,
                will_retry=False,
                retry_delay_seconds=None,
            )
        elif transient_failure_count and response.status_code < 400:
            audit(
                "databricks_recovery_succeeded",
                operation=operation,
                attempts=attempt,
                transient_failure_count=transient_failure_count,
            )

        return response

    raise RuntimeError("Databricks HTTP retry loop exhausted without a response")


def _metadata_hex_literal(value: str, field_name: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned or len(cleaned) > 255 or any(ord(char) < 32 for char in cleaned):
        raise ValueError(f"{field_name} must be a non-empty metadata name under 256 characters")
    # Hex contains no SQL metacharacters. Decoding it inside the generated query
    # preserves quotes, backslashes, and Unicode without interpolating raw input.
    return cleaned.encode("utf-8").hex()


def _metadata_string_expression(hex_literal: str) -> str:
    return f"CAST(unhex('{hex_literal}') AS STRING)"


@_structured_errors
def find_objects(name: str, match_mode: str = "exact", limit: int = 20) -> dict:
    """Find accessible tables/views without asking the model to write discovery SQL.

    Metadata lookups are not charged against the global SQL cap; only
    model-authored ``databricks_execute_sql`` calls count.
    """
    try:
        encoded_name = _metadata_hex_literal(name, "name")
        capped_limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError) as exc:
        return {"error": str(exc), "validation_failure": "invalid_metadata_input"}

    if match_mode not in {"exact", "contains"}:
        return {
            "error": "match_mode must be exact or contains",
            "validation_failure": "invalid_match_mode",
        }

    name_expression = _metadata_string_expression(encoded_name)
    if match_mode == "exact":
        predicate = f"LOWER(table_name) = LOWER({name_expression})"
    else:
        predicate = f"INSTR(LOWER(table_name), LOWER({name_expression})) > 0"

    sql = (
        "SELECT table_catalog, table_schema, table_name, table_type "
        "FROM system.information_schema.tables "
        f"WHERE {predicate} "
        "ORDER BY table_catalog, table_schema, table_name "
        f"LIMIT {capped_limit}"
    )
    result = execute_sql(sql, metadata_query=True)
    if result.get("status") == "SUCCEEDED":
        result["metadata_discovery"] = {
            "name": name,
            "match_mode": match_mode,
            "match_count": result.get("row_count", 0),
        }
    return result


DESCRIBE_OBJECT_NOTE = (
    "column_count is the authoritative number of columns. `object` describes the table "
    "itself and is not a column."
)
DESCRIBE_OBJECT_NOT_FOUND_WARNING = (
    "No accessible column metadata matched this exact object. Use "
    "databricks_find_objects before trying another qualified name."
)
# Internal result-set headers of the single information_schema UNION query that
# backs describe_object. They never reach the model; describe_object reshapes
# the flat rows into ``object`` / ``columns`` before returning.
_DESCRIBE_HEADERS = (
    "metadata_kind",
    "ordinal_position",
    "metadata_name",
    "data_type",
    "is_nullable",
    "comment",
    "data_source_format",
    "created",
    "last_altered",
)


def object_type_from_table_type(table_type: Any) -> str | None:
    """Map information_schema ``table_type`` to a coarse object kind.

    ``table_type`` carries storage detail (MANAGED, EXTERNAL, FOREIGN, shallow
    clones, ...). Consumers that only care whether something is a table or a
    view read ``object_type`` instead.
    """
    if table_type is None or str(table_type).strip() == "":
        return None
    normalized = re.sub(r"\s+", "_", str(table_type).strip().upper())
    if normalized in {"VIEW", "MATERIALIZED_VIEW", "STREAMING_TABLE"}:
        return normalized
    return "TABLE"


def _describe_nullable(value: Any) -> bool | None:
    """Convert information_schema ``is_nullable`` (``YES``/``NO``) to a boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"YES", "Y", "TRUE", "1"}:
        return True
    if text in {"NO", "N", "FALSE", "0"}:
        return False
    return None


def _describe_ordinal(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_describe_result(
    catalog: str,
    schema: str,
    name: str,
    headers: list[Any],
    rows: list[list[Any]],
    *,
    truncated: bool = False,
) -> dict:
    """Reshape flat ``metadata_kind`` rows into the structured describe contract.

    Shared with the synthetic evaluation runtime so both paths emit the exact
    same shape. ``rows`` may contain at most one ``table`` row plus N ``column``
    rows; ``column_count`` is always ``len(columns)``.
    """
    index = {str(header).lower(): position for position, header in enumerate(headers)}

    def field(row: list[Any], header: str) -> Any:
        position = index.get(header)
        if position is None or position >= len(row):
            return None
        return row[position]

    obj: dict[str, Any] = {
        "catalog": catalog,
        "schema": schema,
        "name": name,
        "full_name": f"{catalog}.{schema}.{name}",
        "object_type": None,
        "table_type": None,
        "comment": None,
        "data_source_format": None,
        "created": None,
        "last_altered": None,
    }
    columns: list[dict[str, Any]] = []
    for row in rows:
        kind = str(field(row, "metadata_kind") or "").lower()
        if kind == "table":
            table_type = field(row, "data_type")
            obj["table_type"] = str(table_type) if table_type is not None else None
            obj["object_type"] = object_type_from_table_type(table_type)
            obj["comment"] = field(row, "comment")
            obj["data_source_format"] = field(row, "data_source_format")
            obj["created"] = field(row, "created")
            obj["last_altered"] = field(row, "last_altered")
        elif kind == "column":
            columns.append(
                {
                    "ordinal_position": _describe_ordinal(field(row, "ordinal_position")),
                    "name": field(row, "metadata_name"),
                    "data_type": field(row, "data_type"),
                    "is_nullable": _describe_nullable(field(row, "is_nullable")),
                    "comment": field(row, "comment"),
                }
            )
    columns.sort(
        key=lambda column: (
            column["ordinal_position"] is None,
            column["ordinal_position"] if column["ordinal_position"] is not None else 0,
        )
    )
    result: dict[str, Any] = {
        "status": "SUCCEEDED",
        "object": obj,
        "column_count": len(columns),
        "columns": columns,
        "columns_truncated": bool(truncated),
        "note": DESCRIBE_OBJECT_NOTE,
    }
    if not columns and obj["object_type"] is None:
        result["warning"] = DESCRIBE_OBJECT_NOT_FOUND_WARNING
    return result


@_structured_errors
def describe_object(catalog: str, schema: str, name: str) -> dict:
    """Return the structured object + typed-column contract for an exact object.

    One information_schema UNION query feeds the result; the flat rows are
    reshaped so the model sees ``object`` (the table itself), ``column_count``
    (authoritative), and ``columns`` (complete typed list) rather than a mixed
    row set whose row_count is off by one.

    Also used by the worker's automatic preflight; never counted against the
    global SQL cap.
    """
    try:
        encoded_catalog = _metadata_hex_literal(catalog, "catalog")
        encoded_schema = _metadata_hex_literal(schema, "schema")
        encoded_name = _metadata_hex_literal(name, "name")
    except ValueError as exc:
        return {"error": str(exc), "validation_failure": "invalid_metadata_input"}

    catalog_expression = _metadata_string_expression(encoded_catalog)
    schema_expression = _metadata_string_expression(encoded_schema)
    name_expression = _metadata_string_expression(encoded_name)
    sql = (
        "SELECT 'table' AS metadata_kind, CAST(NULL AS INT) AS ordinal_position, "
        "table_name AS metadata_name, table_type AS data_type, CAST(NULL AS STRING) "
        "AS is_nullable, comment, data_source_format, CAST(created AS STRING) AS created, "
        "CAST(last_altered AS STRING) AS last_altered "
        "FROM system.information_schema.tables "
        f"WHERE table_catalog = {catalog_expression} "
        f"AND table_schema = {schema_expression} "
        f"AND table_name = {name_expression} "
        "UNION ALL "
        "SELECT 'column' AS metadata_kind, ordinal_position, column_name AS metadata_name, "
        "full_data_type AS data_type, is_nullable, comment, CAST(NULL AS STRING) AS data_source_format, "
        "CAST(NULL AS STRING) AS created, CAST(NULL AS STRING) AS last_altered "
        "FROM system.information_schema.columns "
        f"WHERE table_catalog = {catalog_expression} "
        f"AND table_schema = {schema_expression} "
        f"AND table_name = {name_expression} "
        "ORDER BY metadata_kind DESC, ordinal_position"
    )
    raw = execute_sql(sql, metadata_query=True)
    if raw.get("status") != "SUCCEEDED" or "error" in raw:
        # Errors, validation failures, and still-running statements keep their
        # own contract; only attach which object was requested.
        raw["object"] = {
            "catalog": catalog,
            "schema": schema,
            "name": name,
            "full_name": f"{catalog}.{schema}.{name}",
        }
        return raw

    rows = raw.get("sample_rows") or []
    row_count = raw.get("row_count")
    truncated = bool(raw.get("truncated")) or (
        isinstance(row_count, int) and row_count > len(rows)
    )
    result = build_describe_result(
        catalog,
        schema,
        name,
        raw.get("columns") or list(_DESCRIBE_HEADERS),
        rows,
        truncated=truncated,
    )
    for bookkeeping_key in ("statement_id", "warehouse", "automatic_repair"):
        if bookkeeping_key in raw:
            result[bookkeeping_key] = raw[bookkeeping_key]
    return result


def _funnel_semantic_check(columns: list[str], rows: list[list[Any]]) -> dict | None:
    """Check the opt-in strict-funnel output contract without inferring funnel semantics."""
    column_indexes = {name.lower(): index for index, name in enumerate(columns) if name}
    contract_markers = {
        "funnel_numerator",
        "funnel_denominator",
        "funnel_rate",
        "funnel_rate_pct",
    }
    present_markers = contract_markers & column_indexes.keys()
    if not present_markers:
        return None
    required_counts = {"funnel_numerator", "funnel_denominator"}
    missing_counts = sorted(required_counts - column_indexes.keys())
    if missing_counts:
        return {
            "contract": "strict_funnel_subset",
            "status": "WARNING",
            "violations": [
                "The final result omitted required strict-funnel aliases: "
                + ", ".join(missing_counts)
            ],
            "instruction": (
                "Rerun the query with funnel_numerator and funnel_denominator in the outermost "
                "SELECT. Keep human-friendly labels in the written answer, not the SQL aliases."
            ),
        }

    numerator_index = column_indexes["funnel_numerator"]
    denominator_index = column_indexes["funnel_denominator"]
    rate_index = column_indexes.get("funnel_rate")
    rate_pct_index = column_indexes.get("funnel_rate_pct")
    violations = []

    def as_number(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    for row_index, row in enumerate(rows):
        numerator = as_number(row[numerator_index])
        denominator = as_number(row[denominator_index])
        if numerator is not None and denominator is not None:
            if numerator < 0 or denominator < 0:
                violations.append(f"row {row_index + 1}: funnel counts cannot be negative")
            elif numerator > denominator:
                violations.append(
                    f"row {row_index + 1}: numerator {numerator:g} exceeds denominator "
                    f"{denominator:g}"
                )

        if rate_index is not None:
            rate = as_number(row[rate_index])
            if rate is not None and not 0 <= rate <= 1:
                violations.append(f"row {row_index + 1}: funnel_rate {rate:g} is outside 0..1")
        if rate_pct_index is not None:
            rate_pct = as_number(row[rate_pct_index])
            if rate_pct is not None and not 0 <= rate_pct <= 100:
                violations.append(
                    f"row {row_index + 1}: funnel_rate_pct {rate_pct:g} is outside 0..100"
                )

    return {
        "contract": "strict_funnel_subset",
        "status": "WARNING" if violations else "PASSED",
        "violations": violations,
        "instruction": (
            "Do not present this funnel result until the SQL grain, deduplication, and subset "
            "relationship are reconciled."
            if violations
            else "The explicit strict-funnel subset and rate bounds passed for returned rows."
        ),
    }


SAMPLE_ROW_LIMIT = 50
# Tool-generated metadata lookups (describe_object) need every information_schema
# row, not a 50-row sample: a wide table must not look truncated at 50 columns.
METADATA_SAMPLE_ROW_LIMIT = DEFAULT_ROW_LIMIT


def _summarize_result(result_json: dict, sample_limit: int = SAMPLE_ROW_LIMIT) -> dict:
    """Flatten a SUCCEEDED result into a compact summary for the model."""
    manifest = result_json.get("manifest", {}) or {}
    result = result_json.get("result", {}) or {}
    schema = manifest.get("schema", {}) or {}
    columns = [c.get("name") for c in schema.get("columns", [])]
    rows = result.get("data_array", []) or []
    row_count = manifest.get("total_row_count", len(rows))
    truncated = manifest.get("truncated", False)

    # Keep sample small to avoid blowing context.
    sample_rows = rows[:sample_limit]
    sample_row_count = len(sample_rows)
    is_sample = truncated or row_count > sample_row_count

    if truncated:
        result_scope = (
            f"The warehouse truncated this result at the server row cap, and sample_rows contains "
            f"only the first {sample_row_count} returned rows. Treat these rows as a sample, disclose "
            f"the truncation, and do not make full-population claims from them."
        )
    elif is_sample:
        result_scope = (
            f"sample_rows contains the first {sample_row_count} of {row_count} rows. Treat them as "
            f"a sample unless the SQL itself produced a complete aggregate."
        )
    else:
        result_scope = f"sample_rows contains all {row_count} returned rows."

    summary = {
        "status": "SUCCEEDED",
        "statement_id": result_json.get("statement_id"),
        "row_count": row_count,
        "truncated": truncated,
        "is_sample": is_sample,
        "result_scope": result_scope,
        "columns": columns,
        "sample_rows": sample_rows,
        "sample_row_count": sample_row_count,
    }
    semantic_check = _funnel_semantic_check(columns, sample_rows)
    if semantic_check is not None:
        summary["semantic_check"] = semantic_check
    return summary


def _format_status(payload: dict, sample_limit: int = SAMPLE_ROW_LIMIT) -> dict:
    status_obj = payload.get("status") or {}
    state = status_obj.get("state")
    out: dict[str, Any] = {
        "status": state,
        "statement_id": payload.get("statement_id"),
    }
    if state == "SUCCEEDED":
        return _summarize_result(payload, sample_limit)
    if state in {"FAILED", "CANCELED", "CLOSED"}:
        err = status_obj.get("error") or {}
        out["error"] = err.get("message") or err.get("error_code")
    return out


def _poll_budget_remaining(poll_cutoff: float) -> float:
    """Seconds left for internal polling: the smaller of our own cap and the deadline budget."""
    own = poll_cutoff - time.time()
    budget = _retry_budget_seconds()
    return own if budget is None else min(own, budget)


@_structured_errors
def execute_sql(
    sql: str,
    catalog: str | None = None,
    schema: str | None = None,
    *,
    metadata_query: bool = False,
) -> dict:
    """Validate and run one read-only statement.

    ``metadata_query`` marks tool-generated information_schema lookups
    (find_objects / describe_object, including the worker's auto-preflight).
    They skip the global SQL cap; only model-authored SQL is counted.
    """
    sql, metadata_limit_repaired = _repair_metadata_limit(sql)
    if metadata_limit_repaired:
        audit("sql_repaired", repair="removed_invalid_metadata_limit", sql=sql)

    def finalize(result: dict[str, Any]) -> dict[str, Any]:
        if metadata_limit_repaired:
            result["automatic_repair"] = "removed_invalid_metadata_limit"
        return result

    sample_limit = METADATA_SAMPLE_ROW_LIMIT if metadata_query else SAMPLE_ROW_LIMIT

    validation_error = _validate_sql(sql)
    if validation_error is not None:
        logger.warning(
            "SQL rejected pre-flight",
            extra={
                "validation_failure": validation_error.get("validation_failure"),
                "first_keyword": validation_error.get("first_keyword"),
                "sql": sql,
            },
        )
        audit(
            "sql_rejected",
            sql=sql,
            validation_failure=validation_error.get("validation_failure"),
            first_keyword=validation_error.get("first_keyword"),
        )
        return finalize(validation_error)

    if not metadata_query:
        sql_allowed, sql_reason = check_sql_global()
        if not sql_allowed:
            audit("rate_limit_blocked", scope="sql_global", reason=sql_reason, sql=sql)
            return finalize({
                "error": (
                    f"Global SQL rate limit reached ({sql_reason}). The bot is in "
                    "cool-down for the rest of the day — ask in #data-assistant-help if "
                    "this is urgent and the cap needs to be raised."
                ),
                "rate_limit_failure": sql_reason,
            })

    if not WAREHOUSE_ID:
        return finalize({
            "error": (
                "DATABRICKS_WAREHOUSE_ID is not configured for this deployment. "
                "Set it on the worker to the ID of a SQL warehouse the service "
                "principal can use."
            ),
            "configuration_failure": "missing_warehouse_id",
        })

    body: dict[str, Any] = {
        "warehouse_id": WAREHOUSE_ID,
        "statement": sql,
        "wait_timeout": INITIAL_WAIT_S,
        "on_wait_timeout": "CONTINUE",
        "format": "JSON_ARRAY",
        "disposition": "INLINE",
        # Server-side row cap — caps regardless of what the SQL says, sets
        # `truncated=true` in the response when applied (per Databricks SQL
        # Statement Execution API docs).
        "row_limit": DEFAULT_ROW_LIMIT,
    }
    if catalog:
        body["catalog"] = catalog
    if schema:
        body["schema"] = schema

    started = time.time()
    r = _request_with_retry("POST", "sql", json=body, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    _ensure_ok(r, "sql.execute")
    payload = r.json()

    state = (payload.get("status") or {}).get("state")
    statement_id = payload.get("statement_id")

    if state in {"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"}:
        return finalize(_format_status(payload, sample_limit))

    if _safe_statement_id(statement_id) is None:
        return finalize({
            "error": (
                "Databricks accepted the statement but returned a statement_id in an "
                "unexpected format, so the tool cannot poll for its result."
            ),
            "status": state or "PENDING",
            "statement_id": statement_id,
        })

    # Poll until done, until our own ~90s cap is used up, or until the invocation
    # deadline (minus the response reserve) can no longer fit another poll.
    poll_cutoff = started + MAX_POLL_SECONDS
    while _poll_budget_remaining(poll_cutoff) >= POLL_INTERVAL_SECONDS + EXPECTED_POLL_REQUEST_SECONDS:
        time.sleep(POLL_INTERVAL_SECONDS)
        payload = get_statement_raw(statement_id)
        state = (payload.get("status") or {}).get("state")
        if state in {"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"}:
            return finalize(_format_status(payload, sample_limit))

    polled_seconds = int(time.time() - started)
    return finalize({
        "status": state or "RUNNING",
        "statement_id": statement_id,
        "note": (
            "still running; call databricks_get_statement with this statement_id to retrieve "
            f"the result. Internal polling stopped after ~{polled_seconds}s to leave time to "
            "respond. If the user is waiting, ask them to reply 'check on statement "
            "<statement_id>'."
        ),
    })


def get_statement_raw(statement_id: str) -> dict:
    safe_id = _safe_statement_id(statement_id)
    if safe_id is None:
        raise DatabricksToolError(_invalid_statement_id_error(statement_id))
    r = _request_with_retry("GET", "sql", path=safe_id, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    _ensure_ok(r, "sql.statement")
    return r.json()


@_structured_errors
def get_statement(statement_id: str) -> dict:
    return _format_status(get_statement_raw(statement_id))


# ---------------------------------------------------------------------------
# Jobs API — run-now / get-run / cancel-run
# ---------------------------------------------------------------------------


def _summarize_run(payload: dict) -> dict:
    """Compact subset of the Databricks runs/get response."""
    state = payload.get("state") or {}
    return {
        "run_id": payload.get("run_id"),
        "job_id": payload.get("job_id"),
        "run_name": payload.get("run_name"),
        "life_cycle_state": state.get("life_cycle_state"),
        "result_state": state.get("result_state"),
        "state_message": state.get("state_message"),
        "start_time_ms": payload.get("start_time"),
        "setup_duration_ms": payload.get("setup_duration"),
        "execution_duration_ms": payload.get("execution_duration"),
        "cleanup_duration_ms": payload.get("cleanup_duration"),
        "end_time_ms": payload.get("end_time"),
        "run_page_url": payload.get("run_page_url"),
    }


@_structured_errors
def get_job_configuration(job_id: int) -> dict:
    """Read live configuration for the trusted confirmation handler (not an LLM tool)."""
    r = _request_with_retry("GET", "jobs", path="get", params={"job_id": int(job_id)}, timeout=30)
    _ensure_ok(r, "jobs.get")
    return r.json()


@_structured_errors
def run_job(job_id: int, notebook_params: dict | None = None, *, idempotency_token: str) -> dict:
    """Trusted button handler only; dispatch never exposes this execution function."""
    body: dict[str, Any] = {
        "job_id": int(job_id),
        # Reused by _request_with_retry for every HTTP attempt. Databricks uses
        # this token to guarantee that an ambiguous retry launches one run.
        "idempotency_token": idempotency_token,
    }
    if notebook_params:
        # Databricks expects all values as strings here.
        body["notebook_params"] = {k: str(v) for k, v in notebook_params.items()}

    r = _request_with_retry("POST", "jobs", path="run-now", json=body, timeout=30)
    if r.status_code == 403:
        return {
            "error": (
                "403 from Databricks. The bot's Service Principal is not authorized to "
                "run this job. This may mean the job is outside the approved "
                "CAN_MANAGE_RUN allowlist; a 403 alone does not prove the job is paused."
            ),
            "status_code": 403,
        }
    _ensure_ok(r, "jobs.run-now")
    payload = r.json()
    run_id = payload.get("run_id")
    audit(
        "job_triggered",
        job_id=int(job_id),
        run_id=run_id,
        number_in_job=payload.get("number_in_job"),
    )
    return {
        "run_id": run_id,
        "number_in_job": payload.get("number_in_job"),
    }


@_structured_errors
def get_run_status(run_id: int) -> dict:
    r = _request_with_retry(
        "GET",
        "jobs",
        path="runs/get",
        params={"run_id": int(run_id)},
        timeout=30,
    )
    _ensure_ok(r, "jobs.runs.get")
    return _summarize_run(r.json())


@_structured_errors
def cancel_run(run_id: int) -> dict:
    r = _request_with_retry(
        "POST",
        "jobs",
        path="runs/cancel",
        json={"run_id": int(run_id)},
        timeout=30,
    )
    _ensure_ok(r, "jobs.runs.cancel")
    audit("job_cancelled", run_id=int(run_id))
    return {"run_id": int(run_id), "status": "cancellation_requested"}


def _human_duration(ms: int | None) -> str | None:
    """Format milliseconds as e.g. '9m 14s' for human consumption."""
    if not ms or ms < 0:
        return None
    seconds = ms // 1000
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


@_structured_errors
def get_recent_runs(job_id: int, limit: int = 10) -> dict:
    """Recent completed runs for a job, with duration summary across successful runs."""
    capped_limit = max(1, min(int(limit), 25))
    r = _request_with_retry(
        "GET",
        "jobs",
        path="runs/list",
        params={
            "job_id": int(job_id),
            "completed_only": "true",
            "limit": capped_limit,
            "expand_tasks": "false",
        },
        timeout=30,
    )
    _ensure_ok(r, "jobs.runs.list")
    payload = r.json()
    raw_runs = payload.get("runs", []) or []

    summarized: list[dict] = []
    successful_durations: list[int] = []
    for run in raw_runs:
        state = run.get("state") or {}
        result_state = state.get("result_state")
        start_ms = run.get("start_time")
        end_ms = run.get("end_time")
        duration_ms = (
            (end_ms - start_ms) if (start_ms and end_ms and end_ms > start_ms) else None
        )
        summarized.append(
            {
                "run_id": run.get("run_id"),
                "result_state": result_state,
                "start_time_ms": start_ms,
                "duration_ms": duration_ms,
                "duration_human": _human_duration(duration_ms),
            }
        )
        if result_state == "SUCCESS" and duration_ms:
            successful_durations.append(duration_ms)

    summary: dict | None = None
    if successful_durations:
        sorted_durs = sorted(successful_durations)
        n = len(sorted_durs)
        if n % 2:
            median_ms = sorted_durs[n // 2]
        else:
            median_ms = (sorted_durs[n // 2 - 1] + sorted_durs[n // 2]) // 2
        summary = {
            "successful_run_count": n,
            "median_duration_ms": median_ms,
            "median_duration_human": _human_duration(median_ms),
            "min_duration_ms": sorted_durs[0],
            "min_duration_human": _human_duration(sorted_durs[0]),
            "max_duration_ms": sorted_durs[-1],
            "max_duration_human": _human_duration(sorted_durs[-1]),
        }

    return {
        "job_id": int(job_id),
        "fetched_run_count": len(summarized),
        "successful_run_summary": summary,
        "runs": summarized,
    }


@_structured_errors
def list_job_runs(
    run_date: str | None = None,
    active_only: bool = False,
    limit: int = 100,
) -> dict:
    """List workspace-visible runs for one America/New_York calendar date."""
    timezone = ZoneInfo("America/New_York")
    if run_date:
        try:
            selected_date = date_type.fromisoformat(run_date)
        except ValueError:
            return {
                "error": "date must use YYYY-MM-DD format",
                "validation_failure": "invalid_date",
            }
    else:
        selected_date = datetime.now(timezone).date()

    capped_limit = max(1, min(int(limit), 100))
    base_params: dict[str, Any] = {"expand_tasks": "false"}
    date_filter_applied = not (active_only and run_date is None)
    if date_filter_applied:
        start = datetime.combine(selected_date, datetime_time.min, tzinfo=timezone)
        end = start + timedelta(days=1)
        base_params.update(
            {
                "start_time_from": int(start.timestamp() * 1000),
                "start_time_to": int(end.timestamp() * 1000) - 1,
            }
        )
    if active_only:
        base_params["active_only"] = "true"

    raw_runs: list[dict[str, Any]] = []
    next_page_token: str | None = None
    while len(raw_runs) < capped_limit:
        params = dict(base_params)
        params["limit"] = min(25, capped_limit - len(raw_runs))
        if next_page_token:
            params["page_token"] = next_page_token

        r = _request_with_retry(
            "GET",
            "jobs",
            path="runs/list",
            params=params,
            timeout=30,
        )
        if r.status_code == 403:
            return {
                "error": (
                    "The bot is not authorized to list workspace job runs. "
                    "Do not fall back to guessing SQL system-table schemas."
                ),
                "status_code": 403,
            }
        _ensure_ok(r, "jobs.runs.list")
        payload = r.json()
        raw_runs.extend(payload.get("runs", []) or [])
        next_page_token = payload.get("next_page_token")
        if not next_page_token:
            break

    runs = []
    for raw_run in raw_runs[:capped_limit]:
        summarized = _summarize_run(raw_run)
        start_ms = summarized.get("start_time_ms")
        end_ms = summarized.get("end_time_ms")
        summarized.update(
            {
                "run_type": raw_run.get("run_type"),
                "trigger": raw_run.get("trigger"),
                "start_time": (
                    datetime.fromtimestamp(start_ms / 1000, timezone).isoformat()
                    if start_ms
                    else None
                ),
                "end_time": (
                    datetime.fromtimestamp(end_ms / 1000, timezone).isoformat()
                    if end_ms
                    else None
                ),
            }
        )
        runs.append(summarized)

    return {
        "date": selected_date.isoformat() if date_filter_applied else None,
        "scope": "started_on_date" if date_filter_applied else "currently_active",
        "timezone": str(timezone),
        "active_only": bool(active_only),
        "fetched_run_count": len(runs),
        "truncated": bool(next_page_token),
        "interpretation_note": (
            "These are observed executions only. Do not infer configured cadence, normal duration, "
            "processed data volume, root cause, or anomalies from this result alone. Read the job "
            "configuration for cadence and use per-job recent history for duration comparisons."
        ),
        "runs": runs,
    }


def dispatch(name: str, inp: dict[str, Any]) -> dict:
    try:
        return _dispatch(name, inp)
    except DatabricksToolError as exc:
        return dict(exc.error)


def _dispatch(name: str, inp: dict[str, Any]) -> dict:
    if name == "databricks_execute_sql":
        return execute_sql(inp["sql"], inp.get("catalog"), inp.get("schema"))
    if name == "databricks_get_statement":
        return get_statement(inp["statement_id"])
    if name == "databricks_find_objects":
        return find_objects(
            inp["name"],
            inp.get("match_mode", "exact"),
            inp.get("limit", 20),
        )
    if name == "databricks_describe_object":
        return describe_object(inp["catalog"], inp["schema"], inp["name"])
    if name == "databricks_run_job":
        return {"error": "Job runs require a verified Slack confirmation button click.",
                "validation_failure": "job_confirmation_required"}
    if name == "databricks_get_run_status":
        return get_run_status(inp["run_id"])
    if name == "databricks_cancel_run":
        return cancel_run(inp["run_id"])
    if name == "databricks_get_recent_runs":
        return get_recent_runs(inp["job_id"], inp.get("limit", 10))
    if name == "databricks_list_job_runs":
        return list_job_runs(
            inp.get("date"),
            inp.get("active_only", False),
            inp.get("limit", 100),
        )
    return {"error": f"Unknown databricks tool: {name}"}
