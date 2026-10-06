"""Deterministic tool implementations for model-in-the-loop evaluations.

Nothing in this module performs network I/O.  It intentionally accepts a
useful range of valid analytical SQL while returning errors for the production
mistakes the scenario is designed to catch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from tools import calc_tool
from tools.databricks_tool import (
    _DESCRIBE_HEADERS,
    _repair_metadata_limit,
    _validate_sql,
    build_describe_result,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Repository areas the system prompt names as reference material the model may
# read directly (``prompts/business/00_repo_context.md``: "consult docs/ first
# rather than searching for code"; ``resources/jobs/*.job.yml`` is the job
# source of truth). Reading a file there is prompt-instructed, not a guessed
# path, so a fixture that does not model the file answers with an honest
# limitation instead of the ``wrong_github_source`` error reserved for the
# stale-notebook mistake the scenario is designed to catch. an earlier CI run
# read ``docs/runbooks/all-jobs.md`` after ``resources/jobs/``
# searches came back empty, exactly as ``core/workflows/job-trigger.md`` says.
DOCUMENTED_REPO_PREFIXES = ("docs/", "resources/")


def load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown evaluation fixture: {name}")
    return json.loads(path.read_text())


def _success(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "status": "SUCCEEDED",
        "statement_id": "synthetic-statement",
        "row_count": len(rows),
        "truncated": False,
        "is_sample": False,
        "result_scope": f"sample_rows contains all {len(rows)} synthetic rows.",
        "columns": columns,
        "sample_rows": rows,
        "sample_row_count": len(rows),
    }


def _projection_not_modeled(
    columns: list[str],
    result_scope: str,
    unsupported_columns: list[str] | None = None,
) -> dict[str, Any]:
    """An honest zero-row limitation for valid SQL the fixture has no rows for.

    ``row_count`` is ``None`` and the scope text forbids inferring absence, so
    the model can neither read the shape as "no data" nor mistake rows meant
    for another query grain as its own.
    """
    result: dict[str, Any] = {
        "status": "SUCCEEDED",
        "statement_id": "synthetic-statement",
        "row_count": None,
        "truncated": False,
        "is_sample": False,
        "result_scope": (
            f"{result_scope} No production data absence or zero value may be inferred."
        ),
        "columns": columns,
        "sample_rows": [],
        "sample_row_count": 0,
        "synthetic_limitation": "projection_not_modeled",
    }
    if unsupported_columns is not None:
        result["unsupported_columns"] = unsupported_columns
    return result


def _file_not_modeled(path: str) -> dict[str, Any]:
    """An honest non-error for a documented repository file the fixture lacks.

    No ``error`` key, so the runner does not mark it ``is_error`` and the scorer
    counts a ``synthetic_limitation`` rather than an avoidable failure. The
    scope text forbids reading it as "file missing" or inventing its contents.
    """
    return {
        "path": path,
        "ref": "main",
        "content": None,
        "result_scope": (
            "This repository file is not modeled by the synthetic fixture. Do not infer "
            "that it is missing, and do not describe its contents."
        ),
        "synthetic_limitation": "file_not_modeled",
    }


def is_documented_repo_path(path: str) -> bool:
    return path.strip("/ ").lower().startswith(DOCUMENTED_REPO_PREFIXES)


def _object_name(tool_input: dict[str, Any]) -> str:
    return ".".join(
        str(tool_input.get(part, "")).strip().replace("`", "").lower()
        for part in ("catalog", "schema", "name")
    )


def _column_parts(column: list[Any]) -> tuple[str, str, Any, str]:
    """Split a fixture column into ``(name, data_type, comment, is_nullable)``.

    Fixture columns are ``[name, type, comment]`` or ``[name, type, comment,
    nullable]`` where ``nullable`` is a boolean or ``"YES"``/``"NO"``; omitted
    means nullable, matching the original fixtures.
    """
    name = str(column[0])
    data_type = str(column[1]) if len(column) > 1 else ""
    comment = column[2] if len(column) > 2 else None
    nullable: Any = column[3] if len(column) > 3 else True
    if isinstance(nullable, str):
        is_nullable = nullable.strip().upper() in {"YES", "Y", "TRUE", "1"}
    else:
        is_nullable = bool(nullable) if nullable is not None else True
    return name, data_type, comment, "YES" if is_nullable else "NO"


def _create_sql(object_name: str, metadata: dict[str, Any]) -> str:
    """The fixture's ``create_sql`` or a DDL statement derived from its columns."""
    if metadata.get("create_sql"):
        return str(metadata["create_sql"])
    definitions = []
    for column in metadata.get("columns", []):
        name, data_type, _comment, is_nullable = _column_parts(column)
        definitions.append(f"{name} {data_type}" + ("" if is_nullable == "YES" else " NOT NULL"))
    comment = str(metadata.get("comment") or "").replace("'", "''")
    return (
        f"CREATE TABLE {object_name} ({', '.join(definitions)}) USING DELTA "
        f"COMMENT '{comment}'"
    )


def _projection_spec(sql: str) -> list[tuple[str, str | None]]:
    """Return output names paired with fixture semantics derived from expressions."""
    try:
        expression = parse_one(sql, read="databricks")
    except (ParseError, ValueError):
        return []
    select = expression.find(exp.Select)
    if select is None:
        return []

    projections = []
    for index, item in enumerate(select.expressions):
        output_name = str(item.alias_or_name or f"_c{index}")
        value_expression = item.this if isinstance(item, exp.Alias) else item
        semantic: str | None = None
        if isinstance(value_expression, exp.Column):
            semantic = value_expression.name.lower()
        elif isinstance(value_expression, exp.Literal):
            semantic = f"literal:{value_expression.this}"
        else:
            rendered = re.sub(r"\s+", " ", value_expression.sql(dialect="databricks"))
            lowered = rendered.lower()
            distinct_match = re.search(
                r"\bcount\s*\(\s*distinct\s+`?([a-z_][\w$]*)`?\s*\)",
                lowered,
            )
            min_max_match = re.search(
                r"\b(min|max)\s*\(\s*`?([a-z_][\w$]*)`?\s*\)",
                lowered,
            )
            if distinct_match:
                semantic = f"distinct:{distinct_match.group(1)}"
            elif min_max_match:
                semantic = f"{min_max_match.group(1)}:{min_max_match.group(2)}"
            elif ("count" in lowered or "sum" in lowered) and "case when" in lowered:
                equality = value_expression.find(exp.EQ)
                equality_column = equality.left if equality is not None else None
                equality_value = equality.right if equality is not None else None
                if isinstance(equality_value, exp.Cast):
                    equality_value = equality_value.this
                equality_match = re.search(
                    r"`?([a-z_][\w$]*)`?\s*=\s*(?:date\s*)?'([^']+)'",
                    lowered,
                )
                null_match = re.search(
                    r"`?([a-z_][\w$]*)`?\s+is\s+null",
                    lowered,
                )
                if isinstance(equality_column, exp.Column) and isinstance(
                    equality_value, exp.Literal
                ):
                    semantic = (
                        f"count:eq:{equality_column.name.lower()}:"
                        f"{str(equality_value.this).lower()}"
                    )
                elif equality_match:
                    semantic = (
                        f"count:eq:{equality_match.group(1)}:{equality_match.group(2)}"
                    )
                elif null_match:
                    semantic = f"count:null:{null_match.group(1)}"
                elif "dob" in lowered and "1900" in lowered:
                    semantic = "count:bad_dob"
                elif "age" in lowered:
                    semantic = "count:suspicious_age"
            elif "/" in lowered and "count" in lowered:
                semantic = "percentage"
            elif re.search(r"\bcount\s*\(\s*\*\s*\)", lowered):
                semantic = "count:rows"
        projections.append((output_name, semantic))
    return projections


def _project_result(
    sql: str,
    semantic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    projections = _projection_spec(sql)
    unsupported = [name for name, semantic in projections if semantic is None]
    missing = [
        name
        for name, semantic in projections
        if semantic is not None
        and not semantic.startswith("literal:")
        and any(semantic not in row for row in semantic_rows)
    ]
    if not projections or unsupported or missing:
        if projections:
            return _projection_not_modeled(
                [name for name, _semantic in projections],
                "This valid projection is not modeled by the synthetic fixture.",
                sorted(set(unsupported + missing)),
            )
        return {
            "error": "The SQL projection is not implemented by this synthetic scenario.",
            "fixture_failure": "unsupported_projection",
            "unsupported_columns": sorted(set(unsupported + missing)),
        }

    columns = [name for name, _semantic in projections]
    rows = []
    for semantic_row in semantic_rows:
        row = []
        for _name, semantic in projections:
            if semantic.startswith("literal:"):
                literal = semantic.removeprefix("literal:")
                try:
                    value: Any = float(literal) if "." in literal else int(literal)
                except ValueError:
                    value = literal
            else:
                value = semantic_row[semantic]
            row.append(value)
        rows.append(row)
    return _success(columns, rows)


class SyntheticTools:
    """State-free fixture dispatcher with no path to production services."""

    def __init__(self, fixture: dict[str, Any]):
        self.fixture = fixture
        self.objects = {key.lower(): value for key, value in fixture["objects"].items()}
        # Optional: scenarios without a repository source answer GitHub searches
        # with zero matches and refuse file reads, mirroring an empty search hit.
        self.github_file = fixture.get("github_file")

    def _github_paths(self) -> list[str]:
        if not self.github_file:
            return []
        path = self.github_file["path"]
        return [path.replace("Signups - Merge.sql", "Signups MA - Merge.sql"), path]

    @staticmethod
    def _searchable_path(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if name.startswith("github_"):
            return self._github(name, tool_input)
        if name.startswith("databricks_"):
            return self._databricks(name, tool_input)
        if name == calc_tool.CALCULATE_TOOL_NAME:
            # Deterministic and side-effect free, so the eval runs the real tool.
            return calc_tool.dispatch(name, tool_input)
        return {
            "error": f"Synthetic fixture does not implement tool {name}",
            "fixture_failure": "unknown_tool",
        }

    def _github(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        path = self.github_file["path"] if self.github_file else ""
        mh_path = path.replace("Signups - Merge.sql", "Signups MA - Merge.sql")
        if not self.github_file and name in {"github_get_file_content", "github_get_commits"}:
            return {
                "error": "This scenario has no repository source; nothing matched a search.",
                "fixture_failure": "no_github_source",
                "requested_path": tool_input.get("path"),
            }
        if name == "github_search_code":
            query = str(tool_input.get("query", ""))
            path_match = re.search(r"(?:^|\s)path:([^\s]+)", query, re.IGNORECASE)
            path_prefix = path_match.group(1).strip("/ ").lower() if path_match else ""
            terms_query = re.sub(r"(?:^|\s)\w+:[^\s]+", " ", query)
            terms = self._searchable_path(terms_query).split()
            matches = []
            for candidate in self._github_paths():
                lowered_path = candidate.lower()
                searchable = self._searchable_path(candidate)
                if path_prefix and not lowered_path.startswith(path_prefix + "/"):
                    continue
                if terms and not all(term in searchable for term in terms):
                    continue
                matches.append(candidate)
            return {
                "total_count": len(matches),
                "items": [
                    {
                        "path": candidate,
                        "sha": "synthetic-mh" if candidate == mh_path else "synthetic-current",
                        "url": "https://example.invalid/mh"
                        if candidate == mh_path
                        else "https://example.invalid/current",
                    }
                    for candidate in matches
                ],
            }
        if name == "github_find_files":
            needle = self._searchable_path(str(tool_input.get("name", "")))
            prefix = str(tool_input.get("path_prefix", "")).strip("/ ").lower()
            try:
                limit = max(1, min(int(tool_input.get("limit", 20)), 100))
            except (TypeError, ValueError):
                limit = 20
            matches = [
                candidate
                for candidate in self._github_paths()
                if (not prefix or candidate.lower().startswith(prefix + "/"))
                and needle in self._searchable_path(candidate)
            ][:limit]
            return {
                "query": tool_input.get("name"),
                "match_count": len(matches),
                "matches": matches,
                "truncated": False,
                "source_tree_truncated": False,
            }
        if name == "github_get_file_content":
            requested = str(tool_input.get("path", ""))
            if requested == path:
                return {"path": path, "ref": "main", "content": self.github_file["content"]}
            if is_documented_repo_path(requested):
                return _file_not_modeled(requested)
            return {
                "error": "The requested file is outside this scenario's current-source fixture.",
                "fixture_failure": "wrong_github_source",
                "requested_path": requested,
            }
        if name == "github_get_commits":
            if tool_input.get("path") != path:
                return {
                    "error": "Commit history requested for an unsupported fixture path.",
                    "fixture_failure": "wrong_github_source",
                }
            return {
                "path": path,
                "commits": [
                    {
                        "sha": "synthetic-current",
                        "author": "Data Team",
                        "date": "2026-02-03T20:23:15Z",
                        "message": "Update signups merge",
                        "url": "https://example.invalid/commit",
                    }
                ],
            }
        return {
            "error": f"Tool {name} is unnecessary for this synthetic scenario.",
            "fixture_failure": "unsupported_github_operation",
        }

    def _databricks(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if name == "databricks_find_objects":
            needle = str(tool_input.get("name", "")).lower()
            rows = []
            for object_name, metadata in self.objects.items():
                if needle in object_name.rsplit(".", 1)[-1]:
                    catalog, schema, table = object_name.split(".", 2)
                    rows.append([catalog, schema, table, metadata["type"]])
            result = _success(
                ["table_catalog", "table_schema", "table_name", "table_type"], rows
            )
            result["metadata_discovery"] = {
                "name": tool_input.get("name"),
                "match_mode": tool_input.get("match_mode", "exact"),
                "match_count": len(rows),
            }
            return result

        if name == "databricks_describe_object":
            object_name = _object_name(tool_input)
            catalog, schema, table = (object_name.split(".", 2) + ["", ""])[:3]
            metadata = self.objects.get(object_name)
            rows: list[list[Any]] = []
            if metadata is not None:
                rows.append(
                    [
                        "table",
                        None,
                        table,
                        metadata["type"],
                        None,
                        metadata["comment"],
                        "DELTA",
                        "2024-01-01T00:00:00Z",
                        "2026-08-28T09:00:00Z",
                    ]
                )
                for position, column in enumerate(metadata["columns"], 1):
                    column_name, data_type, comment, is_nullable = _column_parts(column)
                    rows.append(
                        [
                            "column",
                            position,
                            column_name,
                            data_type,
                            is_nullable,
                            comment,
                            None,
                            None,
                            None,
                        ]
                    )
            # Same reshaping code as production, so the eval sees the exact contract.
            result = build_describe_result(
                catalog, schema, table, list(_DESCRIBE_HEADERS), rows
            )
            result["statement_id"] = "synthetic-statement"
            return result

        if name == "databricks_execute_sql":
            sql, metadata_limit_repaired = _repair_metadata_limit(
                str(tool_input.get("sql", ""))
            )
            result = self._execute_sql(sql)
            if metadata_limit_repaired:
                result["automatic_repair"] = "removed_invalid_metadata_limit"
            return result

        return {
            "error": f"Tool {name} is outside this read-only explanation scenario.",
            "fixture_failure": "unsupported_databricks_operation",
        }

    def _execute_sql(self, sql: str) -> dict[str, Any]:
        validation = _validate_sql(sql)
        if validation:
            return validation

        normalized = re.sub(r"\s+", " ", sql.replace("`", "")).strip()
        lowered = normalized.lower()

        fixture_result = self._fixture_sql_result(sql, lowered)
        if fixture_result is not None:
            return fixture_result

        if (
            re.search(r"\binternal_sum\b", lowered)
            and "acme_prod.reporting.signups" in lowered
        ):
            return {
                "error": "[UNRESOLVED_COLUMN.WITH_SUGGESTION] internal_sum cannot be resolved",
                "synthetic_failure": "build_only_column",
            }

        show_create = re.match(r"show create table ([\w.]+)", lowered)
        if show_create and show_create.group(1) in self.objects:
            object_name = show_create.group(1)
            return _success(
                ["createtab_stmt"],
                [[_create_sql(object_name, self.objects[object_name])]],
            )

        if lowered.startswith("describe history acme_prod.reporting.signups"):
            return _success(
                ["version", "timestamp", "operation"],
                [[112, "2026-08-28T09:00:00Z", "MERGE"]],
            )

        if "information_schema.tables" in lowered:
            return self._information_schema_tables(sql)

        if "information_schema.columns" in lowered:
            return self._information_schema_columns(sql)

        if "system.access.table_lineage" in lowered:
            if "group by" in lowered:
                return _project_result(
                    sql,
                    [
                        {
                            "entity_type": "NOTEBOOK",
                            "distinct:entity_id": 1,
                            # The prompt mandates COUNT(DISTINCT event_id) for
                            # read counts; every lineage row is its own event.
                            "distinct:event_id": 4,
                            # Prompt pattern 4 groups reads by created_by.
                            "created_by": "sp-signups-merge",
                            "count:rows": 4,
                        }
                    ],
                )
            lineage_columns = (
                "source_table_full_name",
                "target_table_full_name",
                "entity_type",
                "entity_id",
            )
            return _project_result(
                sql,
                [
                    dict(zip(lineage_columns, row))
                    for row in self.fixture.get("lineage_rows", [])
                ],
            )

        if "acme_prod.reporting.signups" in lowered:
            projection_semantics = {
                semantic for _name, semantic in _projection_spec(sql) if semantic
            }
            grouped_dimensions = {
                "signup_flow",
                "autofill_ind",
                "device_type",
            }
            if "group by" in lowered and projection_semantics & grouped_dimensions:
                return _project_result(
                    sql,
                    [
                        {
                            "signup_flow": "signup",
                            "autofill_ind": "false",
                            "device_type": "Mobile",
                            "count:rows": 642000,
                            "percentage": 66.9,
                        },
                        {
                            "signup_flow": "deals",
                            "autofill_ind": "true",
                            "device_type": "Desktop",
                            "count:rows": 318000,
                            "percentage": 33.1,
                        },
                    ],
                )
            if "dob" in lowered:
                return _project_result(
                    sql,
                    [
                        {
                            "count:rows": 1284000,
                            "distinct:_anonymousid": 1284000,
                            "distinct:signup_stitch_id": 1269000,
                            "min:signup_date": "2024-01-02",
                            "max:signup_date": "2026-08-27",
                            "count:null:signup_stitch_id": 15000,
                            "count:eq:dob:1492-02-16": 1284000,
                            "count:bad_dob": 214000,
                            "count:suspicious_age": 14500,
                        }
                    ],
                )
            if re.search(r"\b(?:count|min|max|sum|avg)\s*\(", lowered):
                return _project_result(
                    sql,
                    [
                        {
                            "count:rows": 1284000,
                            "distinct:_anonymousid": 1284000,
                            "distinct:signup_stitch_id": 1269000,
                            "min:signup_date": "2024-01-02",
                            "max:signup_date": "2026-08-27",
                            "count:eq:autofill_ind:true": 392000,
                            "count:eq:autofill_ind:false": 821000,
                            "count:null:autofill_ind": 71000,
                            "count:eq:device_type:desktop": 465000,
                            "count:eq:device_type:mobile": 731000,
                            "count:eq:device_type:tablet": 88000,
                            "distinct:device_type": 3,
                            "distinct:signup_flow": 7,
                            "count:suspicious_age": 14500,
                        }
                    ],
                )
            return _projection_not_modeled(
                [name for name, _semantic in _projection_spec(sql)],
                "This valid signups query shape is not modeled by the synthetic fixture.",
            )

        source_objects = [
            object_name
            for object_name in self.objects
            if object_name not in {
                "acme_prod.reporting.signups",
                "system.access.table_lineage",
            }
        ]
        if any(object_name in lowered for object_name in source_objects):
            return _project_result(
                sql,
                [
                    {
                        "count:rows": 1250000,
                        "min:event_date": "2024-01-01",
                        "max:event_date": "2026-08-27",
                        "min:_date": "2024-01-01",
                        "max:_date": "2026-08-27",
                    }
                ],
            )

        return {
            "error": "The SQL does not match an object supported by this synthetic scenario.",
            "fixture_failure": "unsupported_sql",
            "sql": sql,
        }

    def _fixture_sql_result(self, sql: str, lowered: str) -> dict[str, Any] | None:
        """Serve fixture-declared rows for SQL matching a ``sql_results`` pattern.

        ``lowered`` is the whitespace-collapsed, backtick-free, lower-cased SQL.
        Entries are tried in fixture order and the first ``re.search`` match
        wins. Output column names come from the model's own projection when its
        width matches the fixture rows, so aliases line up with what the model
        asked for; otherwise the entry's declared ``columns`` are used. An
        entry with ``"unmodeled": true`` serves the ``projection_not_modeled``
        limitation instead of rows, so a fixture can stop a query grain it does
        not model from being answered with rows meant for another grain.
        """
        for entry in self.fixture.get("sql_results") or []:
            pattern = str(entry.get("pattern") or "")
            if not pattern or not re.search(pattern, lowered, re.DOTALL):
                continue
            if entry.get("unmodeled"):
                return _projection_not_modeled(
                    [name for name, _semantic in _projection_spec(sql)],
                    str(
                        entry.get("result_scope")
                        or "This valid query shape is not modeled by the synthetic fixture."
                    ),
                )
            rows = [list(row) for row in entry.get("rows") or []]
            declared = [str(column) for column in entry.get("columns") or []]
            width = len(rows[0]) if rows else len(declared)
            projected = [name for name, _semantic in _projection_spec(sql)]
            if (
                projected
                and len(projected) == width
                and all(name and name != "*" for name in projected)
            ):
                columns = projected
            else:
                columns = (declared + [f"_c{i}" for i in range(len(declared), width)])[:width]
            return _success(columns, rows)
        return None

    @staticmethod
    def _metadata_filter(sql: str, column: str) -> str | None:
        match = re.search(
            rf"\b{re.escape(column)}\s*=\s*'([^']+)'",
            sql,
            re.IGNORECASE,
        )
        return match.group(1).lower() if match else None

    def _information_schema_matches(self, sql: str) -> list[tuple[str, dict[str, Any]]]:
        relation = re.search(
            r"\bfrom\s+([a-z_][\w$]*)\.information_schema\.(?:tables|columns)\b",
            sql.replace("`", ""),
            re.IGNORECASE,
        )
        relation_catalog = relation.group(1).lower() if relation else None
        catalog = self._metadata_filter(sql, "table_catalog") or relation_catalog
        schema = self._metadata_filter(sql, "table_schema")
        name = self._metadata_filter(sql, "table_name")

        matches = []
        for object_name, metadata in self.objects.items():
            object_catalog, object_schema, object_table = object_name.split(".", 2)
            if catalog and object_catalog != catalog:
                continue
            if schema and object_schema != schema:
                continue
            if name and object_table != name:
                continue
            matches.append((object_name, metadata))
        return matches

    def _information_schema_tables(self, sql: str) -> dict[str, Any]:
        rows = []
        for object_name, metadata in self._information_schema_matches(sql):
            catalog, schema, name = object_name.split(".", 2)
            rows.append([catalog, schema, name, metadata["type"], metadata["comment"]])
        return _success(
            ["table_catalog", "table_schema", "table_name", "table_type", "comment"],
            rows,
        )

    def _information_schema_columns(self, sql: str) -> dict[str, Any]:
        rows = []
        for _object_name, metadata in self._information_schema_matches(sql):
            for column in metadata["columns"]:
                column_name, data_type, comment, is_nullable = _column_parts(column)
                rows.append([column_name, data_type, comment, is_nullable])
        return _success(["column_name", "data_type", "comment", "is_nullable"], rows)
