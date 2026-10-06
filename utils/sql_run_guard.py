"""Per-run safeguards for Databricks SQL tool calls."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from sqlglot import Dialect, exp, parse_one
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.tokens import TokenType

SQL_TOOL_NAME = "databricks_execute_sql"
FIND_OBJECTS_TOOL_NAME = "databricks_find_objects"
DESCRIBE_OBJECT_TOOL_NAME = "databricks_describe_object"
REPLAN_SQL_CALL_THRESHOLD = 10
REPLAN_FAILURE_THRESHOLD = 2
METADATA_PREFLIGHT_FAILURES = {
    "object_metadata_required",
    "system_object_metadata_required",
}


def auto_satisfy_metadata_precondition(
    sql_guard: "SqlRunGuard",
    tool_name: str,
    tool_input: dict[str, Any],
    rejection: dict[str, Any] | None,
    dispatch: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Describe exact first-use relations and re-check a guarded SQL call.

    The returned attempts are operational evidence for logging and evaluation;
    they are not model-visible tool results. A failed or inexact preflight keeps
    the original rejection in place.
    """
    if (
        not rejection
        or tool_name != SQL_TOOL_NAME
        or rejection.get("validation_failure") not in METADATA_PREFLIGHT_FAILURES
    ):
        return rejection, []

    metadata_inputs: list[dict[str, str]] = []
    for object_name in rejection.get("objects") or []:
        parts = str(object_name).split(".")
        if len(parts) != 3 or not all(parts):
            return rejection, []
        metadata_inputs.append(
            {"catalog": parts[0], "schema": parts[1], "name": parts[2]}
        )

    if not metadata_inputs:
        return rejection, []

    attempts: list[dict[str, Any]] = []
    for metadata_input in metadata_inputs:
        try:
            metadata_result = dispatch(DESCRIBE_OBJECT_TOOL_NAME, metadata_input)
        except Exception as exc:
            attempts.append(
                {
                    "input": metadata_input,
                    "result": None,
                    "exception": exc,
                    "succeeded": False,
                }
            )
            return rejection, attempts

        sql_guard.after_call(DESCRIBE_OBJECT_TOOL_NAME, metadata_input, metadata_result)
        succeeded = _describe_result_has_metadata(metadata_result)
        attempts.append(
            {
                "input": metadata_input,
                "result": metadata_result,
                "exception": None,
                "succeeded": succeeded,
            }
        )
        if not succeeded:
            return rejection, attempts

    return sql_guard.before_call(tool_name, tool_input), attempts


def _normalize_sql(sql: str) -> str:
    """Normalize whitespace between tokens, preserving literal/identifier bytes."""
    sql = (sql or "").strip()
    try:
        tokens = Dialect.get_or_raise("databricks").tokenize(sql)
    except SqlglotError:
        # Malformed SQL is never made equivalent to another query by guessing
        # where its literal or comment boundaries should have been.
        return sql
    parts = []
    offset = 0
    for token in tokens:
        gap = sql[offset:token.start]
        # A non-whitespace gap contains a comment; preserve it verbatim.
        parts.append(" " if gap and not gap.strip() else gap)
        if token is not tokens[-1] or token.token_type != TokenType.SEMICOLON:
            parts.append(sql[token.start:token.end + 1])
        offset = token.end + 1
    parts.append(sql[offset:])
    return "".join(parts).strip()


def _sql_fingerprint(tool_input: dict[str, Any]) -> str:
    """Catalog, schema, and whitespace-normalized SQL: the exact-repeat identity."""
    return "|".join(
        (
            _normalize_object_name(str(tool_input.get("catalog") or "")),
            _normalize_object_name(str(tool_input.get("schema") or "")),
            _normalize_sql(str(tool_input.get("sql", ""))),
        )
    )


def _is_schema_inspection_sql(sql: str) -> bool:
    """Return whether SQL can resolve an unknown object, field, or column."""
    normalized = _normalize_sql(sql).replace("`", "")
    return bool(
        re.match(r"^(DESCRIBE|DESC)\b", normalized, flags=re.IGNORECASE)
        or re.match(r"^SHOW\s+CREATE\s+TABLE\b", normalized, flags=re.IGNORECASE)
        or re.match(r"^SHOW\s+(?:CATALOGS|SCHEMAS|TABLES)\b", normalized, flags=re.IGNORECASE)
        or re.search(r"\binformation_schema\.columns\b", normalized, flags=re.IGNORECASE)
        or re.search(r"\binformation_schema\.tables\b", normalized, flags=re.IGNORECASE)
    )


def _requires_schema_inspection(result: dict[str, Any]) -> bool:
    serialized = json.dumps(result, default=str)
    return bool(
        re.search(
            r"UNRESOLVED_COLUMN|FIELD_NOT_FOUND|TABLE_OR_VIEW_NOT_FOUND|"
            r"unresolved column|cannot resolve|No such struct field",
            serialized,
            re.IGNORECASE,
        )
        or re.search(
            r"INSUFFICIENT_PERMISSIONS.*USE (?:CATALOG|SCHEMA)",
            serialized,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _normalize_object_name(name: str) -> str:
    return re.sub(r"\s*\.\s*", ".", name.replace("`", "")).lower()


def _relation_names(sql: str) -> set[str]:
    """Extract physical FROM/JOIN relations conservatively for per-object safeguards."""
    normalized = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql or "", flags=re.DOTALL)
    cte_names = {
        match.group(1).lower()
        for match in re.finditer(
            r"(?:\bWITH\b|,)\s*`?([A-Za-z_][\w$]*)`?"
            r"(?:\s*\([^)]*\))?\s+AS\s*\(",
            normalized,
            flags=re.IGNORECASE,
        )
    }
    relation_pattern = (
        r"\b(?:FROM|JOIN)\s+"
        r"((?:`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][\w$]*)){0,2})"
    )
    excluded = {"lateral", "range", "values", "unnest", "explode"}
    relations = set()
    for match in re.finditer(relation_pattern, normalized, flags=re.IGNORECASE):
        relation = _normalize_object_name(match.group(1))
        remaining = normalized[match.end():].lstrip()
        if remaining.startswith("(") or "://" in relation:
            continue
        if relation not in cte_names and relation not in excluded:
            relations.add(relation)
    return relations


def _resolved_relation_names(
    sql: str,
    catalog: str | None = None,
    schema: str | None = None,
) -> set[str]:
    """Apply SQL Statement API defaults to unqualified extracted relations."""
    normalized_catalog = _normalize_object_name(catalog or "")
    normalized_schema = _normalize_object_name(schema or "")
    resolved = set()
    for relation in _relation_names(sql):
        part_count = relation.count(".") + 1
        if part_count == 1 and normalized_catalog and normalized_schema:
            resolved.add(f"{normalized_catalog}.{normalized_schema}.{relation}")
        elif part_count == 2 and normalized_catalog:
            resolved.add(f"{normalized_catalog}.{relation}")
        else:
            resolved.add(relation)
    return resolved


def _simple_query_columns(sql: str) -> set[str] | None:
    """Return referenced columns for an unambiguous one-table query.

    Complex scopes deliberately fail open. This preflight is a narrow safety net
    for validation queries, not a replacement for Databricks analysis.
    """
    try:
        expression = parse_one(sql, read="databricks")
    except (ParseError, ValueError):
        return None

    if len(list(expression.find_all(exp.Select))) != 1:
        return None
    if expression.find(exp.Join) or expression.find(exp.With) or expression.find(exp.Subquery):
        return None

    tables = list(expression.find_all(exp.Table))
    if len(tables) != 1:
        return None

    table = tables[0]
    table_aliases = {part.lower() for part in (table.name, table.alias) if part}
    select = expression.find(exp.Select)
    select_aliases = {
        item.alias.lower()
        for item in select.expressions
        if getattr(item, "alias", "")
    }

    columns: set[str] = set()
    for column in expression.find_all(exp.Column):
        if column.is_star:
            continue
        qualifier = (column.table or "").lower()
        name = column.name.lower()
        if qualifier and qualifier not in table_aliases:
            # Spark parses `payload.status` like a qualified column. When the
            # qualifier is not this table's alias, validate the leftmost struct
            # root and leave nested-field validation to Databricks.
            parts = [part.name.lower() for part in column.parts]
            columns.add(parts[0] if len(parts) >= 3 else qualifier)
        elif not qualifier and name in select_aliases:
            continue
        else:
            columns.add(name)
    return columns


def _is_structured_describe_result(result: dict[str, Any]) -> bool:
    """True for the current describe_object contract (``object``/``column_count``/``columns``).

    The legacy contract (a flat SQL result set with ``metadata_kind`` rows) is
    still tolerated because replayed DynamoDB history may contain it.
    """
    return (
        "column_count" in result
        and isinstance(result.get("columns"), list)
        and all(isinstance(column, dict) for column in result.get("columns") or [])
    )


def _structured_column_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        column
        for column in result.get("columns") or []
        if isinstance(column, dict) and column.get("name")
    ]


def _describe_result_has_metadata(result: dict[str, Any]) -> bool:
    """Whether a describe_object result proves the exact object exists."""
    if not isinstance(result, dict) or "error" in result or result.get("status") != "SUCCEEDED":
        return False
    if _is_structured_describe_result(result):
        return bool(result.get("column_count")) or bool(
            (result.get("object") or {}).get("object_type")
            if isinstance(result.get("object"), dict)
            else None
        )
    return bool(result.get("row_count"))


def _described_columns(result: dict[str, Any]) -> tuple[set[str], bool]:
    """Extract a complete typed-column inventory from describe_object output."""
    if _is_structured_describe_result(result):
        entries = _structured_column_entries(result)
        columns = {str(column["name"]).lower() for column in entries}
        complete = bool(columns) and not result.get("columns_truncated") and (
            result.get("column_count") == len(entries)
        )
        return columns, complete

    headers = [str(value).lower() for value in result.get("columns") or []]
    try:
        kind_index = headers.index("metadata_kind")
        name_index = headers.index("metadata_name")
    except ValueError:
        return set(), False

    rows = result.get("sample_rows") or []
    columns = {
        str(row[name_index]).lower()
        for row in rows
        if len(row) > max(kind_index, name_index)
        and str(row[kind_index]).lower() == "column"
        and row[name_index]
    }
    complete = bool(columns) and not result.get("truncated") and (
        result.get("row_count") == result.get("sample_row_count") == len(rows)
    )
    return columns, complete


def _described_column_types(result: dict[str, Any]) -> dict[str, str]:
    """Extract typed columns from a describe_object result."""
    if _is_structured_describe_result(result):
        return {
            str(column["name"]).lower(): str(column["data_type"])
            for column in _structured_column_entries(result)
            if column.get("data_type")
        }

    headers = [str(value).lower() for value in result.get("columns") or []]
    try:
        kind_index = headers.index("metadata_kind")
        name_index = headers.index("metadata_name")
        type_index = headers.index("data_type")
    except ValueError:
        return {}

    return {
        str(row[name_index]).lower(): str(row[type_index])
        for row in result.get("sample_rows") or []
        if len(row) > max(kind_index, name_index, type_index)
        and str(row[kind_index]).lower() == "column"
        and row[name_index]
        and row[type_index]
    }


def _described_object_type(result: dict[str, Any]) -> str | None:
    """Return the object kind (TABLE, VIEW, MATERIALIZED_VIEW, ...) from describe_object output.

    The structured contract exposes ``object.object_type``; the legacy flat
    contract only carried information_schema ``table_type`` (MANAGED, EXTERNAL,
    VIEW, ...), which is returned verbatim in upper case.
    """
    if _is_structured_describe_result(result):
        obj = result.get("object")
        if not isinstance(obj, dict):
            return None
        object_type = obj.get("object_type") or obj.get("table_type")
        return str(object_type).upper() if object_type else None

    headers = [str(value).lower() for value in result.get("columns") or []]
    try:
        kind_index = headers.index("metadata_kind")
        type_index = headers.index("data_type")
    except ValueError:
        return None

    for row in result.get("sample_rows") or []:
        if (
            len(row) > max(kind_index, type_index)
            and str(row[kind_index]).lower() == "table"
            and row[type_index]
        ):
            return str(row[type_index]).upper()
    return None


def _history_object(
    sql: str,
    catalog: str | None = None,
    schema: str | None = None,
) -> str | None:
    """Resolve the target of a DESCRIBE HISTORY statement when it is a table name."""
    normalized = _normalize_sql(sql)
    match = re.match(
        r"^DESCRIBE\s+HISTORY\s+"
        r"((?:`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][\w$]*)){0,2})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    object_name = _normalize_object_name(match.group(1))
    part_count = object_name.count(".") + 1
    normalized_catalog = _normalize_object_name(catalog or "")
    normalized_schema = _normalize_object_name(schema or "")
    if part_count == 1 and normalized_catalog and normalized_schema:
        return f"{normalized_catalog}.{normalized_schema}.{object_name}"
    if part_count == 2 and normalized_catalog:
        return f"{normalized_catalog}.{object_name}"
    return object_name


def _described_object(sql: str) -> str | None:
    normalized = _normalize_sql(sql)
    match = re.match(
        r"^(?:DESCRIBE|DESC)(?:\s+TABLE)?\s+"
        r"((?:`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][\w$]*)){1,2})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.match(
            r"^SHOW\s+CREATE\s+TABLE\s+"
            r"((?:`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][\w$]*)){1,2})",
            normalized,
            flags=re.IGNORECASE,
        )
    if match:
        return _normalize_object_name(match.group(1))

    if re.search(r"\binformation_schema\.columns\b", normalized, flags=re.IGNORECASE):
        catalog_match = re.search(
            r"\btable_catalog\s*=\s*'([^']+)'", normalized, flags=re.IGNORECASE
        )
        if not catalog_match:
            catalog_match = re.search(
                r"\bFROM\s+`?([A-Za-z_][\w$]*)`?\.information_schema\.columns\b",
                normalized,
                flags=re.IGNORECASE,
            )
        schema_match = re.search(
            r"\btable_schema\s*=\s*'([^']+)'", normalized, flags=re.IGNORECASE
        )
        table_match = re.search(
            r"\btable_name\s*=\s*'([^']+)'", normalized, flags=re.IGNORECASE
        )
        if catalog_match and schema_match and table_match:
            return _normalize_object_name(
                f"{catalog_match.group(1)}.{schema_match.group(1)}.{table_match.group(1)}"
            )
    return None


def _object_failure(result: dict[str, Any]) -> bool:
    serialized = json.dumps(result, default=str)
    return _requires_schema_inspection(result) or bool(
        re.search(r"\bSQL of VIEW\b", serialized, flags=re.IGNORECASE)
    )


def _failed_objects(
    sql: str,
    result: dict[str, Any],
    catalog: str | None = None,
    schema: str | None = None,
) -> set[str]:
    """Return every physical relation that may explain a deterministic failure."""
    serialized = json.dumps(result, default=str)
    explicit = re.search(
        r"SQL of VIEW\s+[`'\"]?([A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*){1,2})",
        serialized,
        flags=re.IGNORECASE,
    )
    if explicit:
        return {_normalize_object_name(explicit.group(1))}
    return _resolved_relation_names(sql, catalog, schema)


class SqlRunGuard:
    """Track SQL calls and return actionable guardrail responses."""

    def __init__(self) -> None:
        # fingerprint -> where the earlier execution lives: the ordinal of that
        # SQL call in this run plus, once ``after_call`` sees it, the result's
        # ``statement_id`` and ``row_count``.
        self._seen_sql: dict[str, dict[str, Any]] = {}
        self.sql_call_count = 0
        self.sql_failure_count = 0
        self.schema_inspection_required = False
        self._blocked_objects: dict[str, str] = {}
        self._verified_system_objects: set[str] = set()
        self._verified_objects: set[str] = set()
        self._object_columns: dict[str, set[str]] = {}
        self._object_column_types: dict[str, dict[str, str]] = {}
        self._object_types: dict[str, str] = {}
        self._complete_column_objects: set[str] = set()

    def before_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
        """Return a rejection response, or None when the call may proceed."""
        if tool_name != SQL_TOOL_NAME:
            return None

        sql = str(tool_input.get("sql", ""))
        catalog = str(tool_input.get("catalog") or "")
        schema = str(tool_input.get("schema") or "")
        is_schema_inspection = _is_schema_inspection_sql(sql)
        history_object = _history_object(sql, catalog, schema)
        if history_object:
            if history_object not in self._verified_objects:
                return {
                    "error": (
                        "Exact live object metadata is required before requesting Delta history."
                    ),
                    "validation_failure": "object_metadata_required",
                    "objects": [history_object],
                    "required_next_action": (
                        "Call databricks_describe_object for the exact object, then use "
                        "DESCRIBE HISTORY only if its live object type is a table."
                    ),
                }
            if self._object_types.get(history_object) in {
                "VIEW",
                "MATERIALIZED VIEW",
                "MATERIALIZED_VIEW",
            }:
                return {
                    "error": "DESCRIBE HISTORY applies to Delta tables, not views.",
                    "validation_failure": "history_requires_delta_table",
                    "object": history_object,
                    "object_type": self._object_types[history_object],
                    "required_next_action": (
                        "For a view, inspect its live definition with SHOW CREATE TABLE and use "
                        "the configured view-history source for change tracking."
                    ),
                }
        if self.schema_inspection_required and not is_schema_inspection:
            return {
                "error": (
                    "The previous SQL failed because an object, field, or column was unavailable. "
                    "Inspect accessible live metadata before issuing another analytical query."
                ),
                "validation_failure": "schema_inspection_required",
                "required_next_action": (
                    "Run SHOW CATALOGS/SCHEMAS/TABLES, DESCRIBE TABLE, SHOW CREATE TABLE, "
                    "or query information_schema.tables/columns, then revise the analytical SQL."
                ),
            }

        relations = _resolved_relation_names(sql, catalog, schema)
        if not is_schema_inspection:
            blocked = sorted(relations & self._blocked_objects.keys())
            if blocked:
                return {
                    "error": (
                        "A prior deterministic failure involved this object, so another analytical "
                        "query against it is blocked until its exact live metadata is described."
                    ),
                    "validation_failure": "object_circuit_breaker",
                    "blocked_objects": blocked,
                    "required_next_action": (
                        "Call databricks_describe_object for the exact object. If the object is a "
                        "broken view, report the blocker instead of retrying it."
                    ),
                }

            unverified_objects = sorted(relations - self._verified_objects)
            unverified_non_system = [
                relation
                for relation in unverified_objects
                if not relation.startswith("system.")
            ]
            if unverified_non_system:
                return {
                    "error": (
                        "Exact live column metadata is required before the first analytical "
                        "query against an object in this run."
                    ),
                    "validation_failure": "object_metadata_required",
                    "objects": unverified_non_system,
                    "required_next_action": (
                        "Call databricks_describe_object for each exact catalog, schema, and name, "
                        "then build the SQL only from returned columns."
                    ),
                }

            unverified_system_objects = sorted(
                relation
                for relation in relations
                if relation.startswith("system.")
                and not relation.startswith("system.information_schema.")
                and relation not in self._verified_system_objects
            )
            if unverified_system_objects:
                return {
                    "error": (
                        "System-table schemas vary by workspace and must be verified before the "
                        "first analytical query in this run."
                    ),
                    "validation_failure": "system_object_metadata_required",
                    "objects": unverified_system_objects,
                    "required_next_action": (
                        "Call databricks_describe_object with the exact catalog, schema, and name."
                    ),
                }

            if len(relations) == 1:
                object_name = next(iter(relations))
                if object_name in self._complete_column_objects:
                    invalid_struct_roots = sorted(
                        root.lower()
                        for root in re.findall(
                            r"\b([A-Za-z_][\w$]*)\s*:",
                            sql,
                        )
                        if self._object_column_types.get(object_name, {})
                        .get(root.lower(), "")
                        .upper()
                        .startswith("STRUCT<")
                    )
                    if invalid_struct_roots:
                        return {
                            "error": (
                                "The SQL uses JSON-style colon access on a live STRUCT column."
                            ),
                            "validation_failure": "invalid_struct_access_syntax",
                            "object": object_name,
                            "columns": invalid_struct_roots,
                            "required_next_action": (
                                "Use dot notation for STRUCT fields, for example "
                                "entity_metadata.job_info.job_id."
                            ),
                        }
                    referenced_columns = _simple_query_columns(sql)
                    if referenced_columns is not None:
                        missing = sorted(
                            referenced_columns - self._object_columns.get(object_name, set())
                        )
                        if missing:
                            return {
                                "error": (
                                    "The analytical SQL references columns that are absent from "
                                    "the complete live metadata for its source object."
                                ),
                                "validation_failure": "column_not_in_live_schema",
                                "object": object_name,
                                "missing_columns": missing,
                                "required_next_action": (
                                    "Revise the SQL using only columns returned by "
                                    "databricks_describe_object. Build-time CTE aliases are not "
                                    "persisted target columns."
                                ),
                            }

        fingerprint = _sql_fingerprint(tool_input)
        earlier = self._seen_sql.get(fingerprint)
        if earlier is not None and not is_schema_inspection:
            return self._duplicate_rejection(earlier)

        self.sql_call_count += 1
        self._seen_sql[fingerprint] = {"sql_call": self.sql_call_count}
        return None

    @staticmethod
    def _duplicate_rejection(earlier: dict[str, Any]) -> dict[str, Any]:
        """Point the model at the earlier result instead of just refusing.

        An earlier eval run (``signups_table_explanation`` rep 2): after a
        numeric-guard revision request the model re-issued the byte-identical
        ``COUNT(*)/MAX(signup_date)`` query it had already run to "re-verify"
        a figure. "Already executed" alone did not say where that result was.
        """
        ordinal = earlier["sql_call"]
        where = f"SQL call {ordinal} earlier this turn"
        details = []
        if earlier.get("statement_id"):
            details.append(f"statement_id {earlier['statement_id']}")
        if earlier.get("row_count") is not None:
            details.append(f"{earlier['row_count']} row(s)")
        if details:
            where += f" ({', '.join(details)})"
        rejection: dict[str, Any] = {
            "error": (
                f"This exact SQL was already executed during the current run as {where}; "
                "its result is already in this conversation."
            ),
            "validation_failure": "duplicate_sql",
            "earlier_sql_call": ordinal,
            "required_next_action": (
                f"Reuse the result of {where} instead of re-running it. Re-running cannot "
                "verify a figure; quote the earlier result. If you need different data, "
                "change the query intentionally."
            ),
        }
        if earlier.get("statement_id"):
            rejection["earlier_statement_id"] = earlier["statement_id"]
        if earlier.get("row_count") is not None:
            rejection["earlier_row_count"] = earlier["row_count"]
        return rejection

    def after_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Observe a completed call and add a replan warning when needed."""
        if tool_name == FIND_OBJECTS_TOOL_NAME:
            if "error" not in result and result.get("status") == "SUCCEEDED" and result.get("row_count"):
                self.schema_inspection_required = False
            return result

        if tool_name == DESCRIBE_OBJECT_TOOL_NAME:
            if _describe_result_has_metadata(result):
                object_name = _normalize_object_name(
                    f"{tool_input.get('catalog', '')}."
                    f"{tool_input.get('schema', '')}."
                    f"{tool_input.get('name', '')}"
                )
                self._blocked_objects.pop(object_name, None)
                columns, complete = _described_columns(result)
                column_types = _described_column_types(result)
                object_type = _described_object_type(result)
                if column_types:
                    self._object_column_types[object_name] = column_types
                if object_type:
                    self._object_types[object_name] = object_type
                if columns:
                    self._verified_objects.add(object_name)
                    self._object_columns[object_name] = columns
                    if complete:
                        self._complete_column_objects.add(object_name)
                    else:
                        self._complete_column_objects.discard(object_name)
                    if object_name.startswith("system."):
                        self._verified_system_objects.add(object_name)
                self.schema_inspection_required = False
            return result

        if tool_name != SQL_TOOL_NAME:
            return result

        sql = str(tool_input.get("sql", ""))
        failed = "error" in result or result.get("status") in {"FAILED", "CANCELED", "CLOSED"}

        if failed:
            self.sql_failure_count += 1
        else:
            earlier = self._seen_sql.get(_sql_fingerprint(tool_input))
            if earlier is not None:
                # Remember where the result landed so a later duplicate can be
                # pointed at it (statement_id / row_count are optional fields).
                if result.get("statement_id"):
                    earlier["statement_id"] = str(result["statement_id"])
                if result.get("row_count") is not None:
                    earlier["row_count"] = result["row_count"]
        failed_objects: set[str] = set()
        if failed and _object_failure(result):
            failed_objects = _failed_objects(
                sql,
                result,
                str(tool_input.get("catalog") or ""),
                str(tool_input.get("schema") or ""),
            )
            for object_name in failed_objects:
                self._blocked_objects[object_name] = str(result.get("error", "query failed"))

        if _requires_schema_inspection(result):
            # Prefer an object-scoped circuit breaker whenever the failed SQL exposes
            # its physical relations. A failure on one table must not stall an
            # unrelated, already-valid analytical query. Fall back to the global
            # requirement only when no relation can be identified safely.
            self.schema_inspection_required = not failed_objects
        elif _is_schema_inspection_sql(sql) and not failed:
            self.schema_inspection_required = False
            object_name = _described_object(sql)
            if object_name:
                self._blocked_objects.pop(object_name, None)
                if object_name.startswith("system."):
                    self._verified_system_objects.add(object_name)

        if (
            self.sql_call_count >= REPLAN_SQL_CALL_THRESHOLD
            or self.sql_failure_count >= REPLAN_FAILURE_THRESHOLD
        ):
            result = dict(result)
            result["query_budget_warning"] = (
                f"This run has used {self.sql_call_count} SQL calls with "
                f"{self.sql_failure_count} failures. Consolidate the evidence and revise the plan "
                "before issuing more SQL; ask the user if the intended metric or source is ambiguous."
            )

        return result
