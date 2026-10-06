"""Conversation-derived regression cases for per-run SQL safeguards."""

import pytest

from tools.databricks_tool import object_type_from_table_type
from utils.sql_run_guard import SqlRunGuard


def _sql(value: str) -> dict:
    return {"sql": value}


@pytest.mark.parametrize("first,second", [
    ("SELECT 'a  b'", "SELECT 'a b'"),
    ("SELECT 'a\nb'", "SELECT 'a b'"),
    ("SELECT r'a  b'", "SELECT r'a b'"),
    ("SELECT 'it''s  fine'", "SELECT 'it''s fine'"),
    ("SELECT `a  b` FROM t", "SELECT `a b` FROM t"),
])
def test_distinct_literal_and_identifier_whitespace_is_not_a_duplicate(first, second):
    guard = SqlRunGuard()
    _describe(guard, "acme_prod.reporting.t", ("a  b", "a b"))
    default = {"catalog": "acme_prod", "schema": "reporting"}
    assert guard.before_call("databricks_execute_sql", {"sql": first, **default}) is None
    guard.after_call("databricks_execute_sql", {"sql": first, **default}, {"status": "SUCCEEDED", "row_count": 1})
    assert guard.before_call("databricks_execute_sql", {"sql": second, **default}) is None


def _describe(
    guard: SqlRunGuard,
    object_name: str,
    columns: tuple[str, ...],
    *,
    complete: bool = True,
    object_type: str = "MANAGED",
    column_types: dict[str, str] | None = None,
) -> None:
    """Feed the structured describe_object contract to the guard.

    ``object_type`` here is the information_schema ``table_type`` (MANAGED,
    EXTERNAL, VIEW, ...); the structured shape carries it as ``table_type`` and
    derives the coarse ``object_type`` the same way the production tool does.
    """
    catalog, schema, name = object_name.split(".")
    typed_columns = [
        {
            "ordinal_position": index,
            "name": column,
            "data_type": (column_types or {}).get(column, "STRING"),
            "is_nullable": True,
            "comment": None,
        }
        for index, column in enumerate(columns, start=1)
    ]
    column_count = len(typed_columns) if complete else len(typed_columns) + 50
    guard.after_call(
        "databricks_describe_object",
        {"catalog": catalog, "schema": schema, "name": name},
        {
            "status": "SUCCEEDED",
            "object": {
                "catalog": catalog,
                "schema": schema,
                "name": name,
                "full_name": object_name,
                "object_type": object_type_from_table_type(object_type),
                "table_type": object_type,
                "comment": "test object",
                "data_source_format": "DELTA",
                "created": None,
                "last_altered": None,
            },
            "column_count": column_count,
            "columns": typed_columns,
            "columns_truncated": not complete,
            "note": "column_count is the authoritative number of columns.",
        },
    )


def _describe_legacy(
    guard: SqlRunGuard,
    object_name: str,
    columns: tuple[str, ...],
    *,
    complete: bool = True,
    object_type: str = "MANAGED",
    column_types: dict[str, str] | None = None,
) -> None:
    """Feed the pre-restructure flat describe_object shape (replayed history)."""
    catalog, schema, name = object_name.split(".")
    rows = [["table", None, name, object_type, None, "test object"]]
    rows.extend(
        [
            "column",
            index,
            column,
            (column_types or {}).get(column, "STRING"),
            "YES",
            None,
        ]
        for index, column in enumerate(columns, start=1)
    )
    row_count = len(rows) if complete else len(rows) + 50
    guard.after_call(
        "databricks_describe_object",
        {"catalog": catalog, "schema": schema, "name": name},
        {
            "status": "SUCCEEDED",
            "row_count": row_count,
            "sample_row_count": len(rows),
            "truncated": not complete,
            "columns": [
                "metadata_kind",
                "ordinal_position",
                "metadata_name",
                "data_type",
                "is_nullable",
                "comment",
            ],
            "sample_rows": rows,
        },
    )


def test_rejects_exact_duplicate_sql_with_whitespace_changes():
    guard = SqlRunGuard()

    assert guard.before_call("databricks_execute_sql", _sql("SELECT  1")) is None
    rejection = guard.before_call("databricks_execute_sql", _sql(" SELECT 1; "))

    assert rejection["validation_failure"] == "duplicate_sql"
    assert guard.sql_call_count == 1


def test_duplicate_sql_rejection_points_at_the_earlier_result():
    """An earlier eval run (table_explanation rep 2): after a numeric-guard
    revision request the model re-ran the byte-identical COUNT/MAX query to
    "re-verify" a figure. The rejection must say which earlier call holds the
    result, not just that the SQL was already executed."""
    guard = SqlRunGuard()
    first = _sql("SELECT 1 AS probe")
    repeated = _sql(
        "\nSELECT COUNT(*) AS total_rows, MAX(signup_date) AS latest_signup_date\n"
        "FROM acme_prod.reporting.signups\n"
    )
    _describe(guard, "acme_prod.reporting.signups", ("signup_date",))

    assert guard.before_call("databricks_execute_sql", first) is None
    guard.after_call("databricks_execute_sql", first, {"status": "SUCCEEDED", "row_count": 1})
    assert guard.before_call("databricks_execute_sql", repeated) is None
    guard.after_call(
        "databricks_execute_sql",
        repeated,
        {
            "status": "SUCCEEDED",
            "statement_id": "01ef-abc",
            "row_count": 1,
            "columns": ["total_rows", "latest_signup_date"],
            "sample_rows": [[1284000, "2026-08-27"]],
        },
    )

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql(
            "SELECT COUNT(*) AS total_rows, MAX(signup_date) AS latest_signup_date "
            "FROM acme_prod.reporting.signups;"
        ),
    )

    assert rejection["validation_failure"] == "duplicate_sql"
    assert rejection["earlier_sql_call"] == 2
    assert rejection["earlier_statement_id"] == "01ef-abc"
    assert rejection["earlier_row_count"] == 1
    assert "SQL call 2 earlier this turn (statement_id 01ef-abc, 1 row(s))" in rejection["error"]
    assert rejection["required_next_action"].startswith("Reuse the result of SQL call 2 earlier this turn")
    assert "Re-running cannot verify a figure" in rejection["required_next_action"]
    assert guard.sql_call_count == 2

    # Before the result is observed the pointer still names the call ordinal.
    fresh = SqlRunGuard()
    assert fresh.before_call("databricks_execute_sql", first) is None
    early = fresh.before_call("databricks_execute_sql", first)
    assert early["earlier_sql_call"] == 1
    assert "SQL call 1 earlier this turn;" in early["error"]
    assert "statement_id" not in early


def test_unresolved_column_blocks_only_the_failed_object_before_retry():
    guard = SqlRunGuard()
    first = _sql("SELECT guessed_column FROM acme_prod.reporting.signups")
    _describe(
        guard,
        "acme_prod.reporting.signups",
        ("_anonymousid",),
        complete=False,
    )

    assert guard.before_call("databricks_execute_sql", first) is None
    guard.after_call(
        "databricks_execute_sql",
        first,
        {"error": "[UNRESOLVED_COLUMN.WITH_SUGGESTION] guessed_column cannot be resolved"},
    )

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT another_guess FROM acme_prod.reporting.signups"),
    )
    assert rejection["validation_failure"] == "object_circuit_breaker"
    assert rejection["blocked_objects"] == ["acme_prod.reporting.signups"]

    _describe(guard, "another_catalog.reporting.audit_log", ("action_name",))
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT action_name FROM another_catalog.reporting.audit_log"),
    ) is None


def test_object_and_struct_field_failures_are_object_scoped():
    for error in (
        "[TABLE_OR_VIEW_NOT_FOUND] acme_prod.reporting.signup_unified",
        "[FIELD_NOT_FOUND] No such struct field statusCode",
        "[INSUFFICIENT_PERMISSIONS] User does not have USE SCHEMA on system.lakeflow",
    ):
        guard = SqlRunGuard()
        _describe(
            guard,
            "catalog_a.schema_a.guessed_object",
            ("known_value",),
            complete=False,
        )
        failed = {
            "sql": "SELECT guessed_value FROM guessed_object",
            "catalog": "catalog_a",
            "schema": "schema_a",
        }
        guard.before_call("databricks_execute_sql", failed)
        guard.after_call("databricks_execute_sql", failed, {"error": error})

        rejection = guard.before_call(
            "databricks_execute_sql",
            {
                "sql": "SELECT another_guess FROM guessed_object",
                "catalog": "catalog_a",
                "schema": "schema_a",
            },
        )

        assert rejection["validation_failure"] == "object_circuit_breaker"
        _describe(guard, "catalog_a.schema_a.another_object", ("another_guess",))
        assert guard.before_call(
            "databricks_execute_sql",
            {
                "sql": "SELECT another_guess FROM another_object",
                "catalog": "catalog_a",
                "schema": "schema_a",
            },
        ) is None


def test_relationless_schema_failure_uses_global_metadata_recovery():
    guard = SqlRunGuard()
    failed = _sql("SELECT guessed_column")
    guard.before_call("databricks_execute_sql", failed)
    guard.after_call(
        "databricks_execute_sql",
        failed,
        {"error": "[UNRESOLVED_COLUMN] guessed_column cannot be resolved"},
    )

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT another_guess FROM another_object"),
    )
    assert rejection["validation_failure"] == "schema_inspection_required"


def test_join_failure_blocks_each_possible_source_but_not_unrelated_objects():
    guard = SqlRunGuard()
    _describe(
        guard,
        "catalog_a.schema_a.left_table",
        ("id",),
        complete=False,
    )
    _describe(
        guard,
        "catalog_b.schema_b.right_table",
        ("id",),
        complete=False,
    )
    failed = _sql(
        "SELECT missing_column FROM catalog_a.schema_a.left_table l "
        "JOIN catalog_b.schema_b.right_table r ON l.id = r.id"
    )
    guard.before_call("databricks_execute_sql", failed)
    guard.after_call(
        "databricks_execute_sql",
        failed,
        {"error": "[UNRESOLVED_COLUMN] missing_column cannot be resolved"},
    )

    for object_name in (
        "catalog_a.schema_a.left_table",
        "catalog_b.schema_b.right_table",
    ):
        rejection = guard.before_call(
            "databricks_execute_sql",
            _sql(f"SELECT COUNT(*) FROM {object_name}"),
        )
        assert rejection["validation_failure"] == "object_circuit_breaker"

    _describe(guard, "catalog_c.schema_c.other_table", ("id",))
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT COUNT(*) FROM catalog_c.schema_c.other_table"),
    ) is None


def test_object_discovery_query_clears_metadata_requirement():
    guard = SqlRunGuard()
    failed = _sql("SHOW CREATE TABLE acme_prod.reporting.signup_unified")
    guard.before_call("databricks_execute_sql", failed)
    guard.after_call(
        "databricks_execute_sql",
        failed,
        {"error": "[TABLE_OR_VIEW_NOT_FOUND] signup_unified"},
    )

    metadata = _sql(
        "SELECT table_catalog, table_schema, table_name "
        "FROM system.information_schema.tables "
        "WHERE table_name = 'signup_unified'"
    )
    assert guard.before_call("databricks_execute_sql", metadata) is None
    guard.after_call("databricks_execute_sql", metadata, {"status": "SUCCEEDED"})

    assert guard.schema_inspection_required is False


def test_only_typed_exact_metadata_verifies_an_object_for_analysis():
    guard = SqlRunGuard()
    failed = _sql("SELECT guessed_column FROM acme_prod.reporting.signups")
    guard.before_call("databricks_execute_sql", failed)
    guard.after_call(
        "databricks_execute_sql",
        failed,
        {"error": "UNRESOLVED_COLUMN: guessed_column"},
    )

    metadata = _sql(
        "SELECT column_name, data_type, comment "
        "FROM acme_prod.information_schema.columns "
        "WHERE table_schema = 'reporting' AND table_name = 'signups'"
    )
    assert guard.before_call("databricks_execute_sql", metadata) is None
    guard.after_call("databricks_execute_sql", metadata, {"status": "SUCCEEDED"})

    assert guard.schema_inspection_required is False
    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT _anonymousId FROM acme_prod.reporting.signups"),
    )
    assert rejection["validation_failure"] == "object_metadata_required"

    _describe(guard, "acme_prod.reporting.signups", ("_anonymousid",))
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT _anonymousId FROM acme_prod.reporting.signups"),
    ) is None


def test_schema_inspection_can_be_repeated_to_recover_from_later_failure():
    guard = SqlRunGuard()
    metadata = _sql("DESCRIBE TABLE acme_prod.reporting.signups")
    failed = _sql("SELECT guessed_column FROM acme_prod.reporting.signups")

    guard.before_call("databricks_execute_sql", metadata)
    guard.after_call("databricks_execute_sql", metadata, {"status": "SUCCEEDED"})
    guard.before_call("databricks_execute_sql", failed)
    guard.after_call(
        "databricks_execute_sql",
        failed,
        {"error": "UNRESOLVED_COLUMN: guessed_column"},
    )

    assert guard.before_call("databricks_execute_sql", metadata) is None


def test_object_failure_blocks_retries_until_exact_metadata_is_described():
    guard = SqlRunGuard()
    _describe(
        guard,
        "acme_prod.reporting.partner_identifier",
        ("partner_ref_id",),
        complete=False,
    )
    failed = _sql("SELECT partner_ref_id FROM acme_prod.reporting.partner_identifier")
    assert guard.before_call("databricks_execute_sql", failed) is None
    guard.after_call(
        "databricks_execute_sql",
        failed,
        {"error": "[CAST_INVALID_INPUT] SQL of VIEW acme_prod.reporting.partner_identifier"},
    )

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT COUNT(*) FROM acme_prod.reporting.partner_identifier"),
    )
    assert rejection["validation_failure"] == "object_circuit_breaker"
    assert rejection["blocked_objects"] == ["acme_prod.reporting.partner_identifier"]

    _describe(guard, "acme_prod.reporting.partner_identifier", ("partner_ref_id",))
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT COUNT(*) FROM acme_prod.reporting.partner_identifier"),
    ) is None


def test_system_object_requires_typed_metadata_before_first_analytical_query():
    guard = SqlRunGuard()
    analytical = _sql("SELECT action_name FROM system.access.audit")

    rejection = guard.before_call("databricks_execute_sql", analytical)
    assert rejection["validation_failure"] == "system_object_metadata_required"
    assert rejection["objects"] == ["system.access.audit"]

    _describe(guard, "system.access.audit", ("action_name",))
    assert guard.before_call("databricks_execute_sql", analytical) is None


def test_system_object_defaults_are_applied_before_metadata_check():
    guard = SqlRunGuard()
    rejection = guard.before_call(
        "databricks_execute_sql",
        {
            "sql": "SELECT action_name FROM audit",
            "catalog": "system",
            "schema": "access",
        },
    )

    assert rejection["validation_failure"] == "system_object_metadata_required"
    assert rejection["objects"] == ["system.access.audit"]


def test_nested_struct_fields_validate_the_root_column():
    guard = SqlRunGuard()
    _describe(
        guard,
        "system.access.table_lineage",
        ("entity_metadata", "target_table_full_name"),
        column_types={
            "entity_metadata": "STRUCT<job_info: STRUCT<job_id: STRING>>",
        },
    )

    assert guard.before_call(
        "databricks_execute_sql",
        _sql(
            "SELECT entity_metadata.job_info.job_id AS job_id "
            "FROM system.access.table_lineage"
        ),
    ) is None


def test_json_style_access_is_rejected_for_live_struct_column():
    guard = SqlRunGuard()
    _describe(
        guard,
        "system.access.table_lineage",
        ("entity_metadata",),
        column_types={
            "entity_metadata": "STRUCT<job_info: STRUCT<job_id: STRING>>",
        },
    )

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql(
            "SELECT entity_metadata:job_info.job_id "
            "FROM system.access.table_lineage"
        ),
    )

    assert rejection["validation_failure"] == "invalid_struct_access_syntax"
    assert rejection["columns"] == ["entity_metadata"]


def test_describe_history_requires_metadata_and_rejects_views():
    guard = SqlRunGuard()
    sql = _sql("DESCRIBE HISTORY acme_prod.reporting.partner_a")

    metadata_required = guard.before_call("databricks_execute_sql", sql)
    assert metadata_required["validation_failure"] == "object_metadata_required"
    assert metadata_required["objects"] == ["acme_prod.reporting.partner_a"]

    _describe(
        guard,
        "acme_prod.reporting.partner_a",
        ("view_date",),
        object_type="VIEW",
    )
    rejection = guard.before_call("databricks_execute_sql", sql)

    assert rejection["validation_failure"] == "history_requires_delta_table"
    assert rejection["object_type"] == "VIEW"


def test_query_expression_error_does_not_block_the_source_object():
    guard = SqlRunGuard()
    _describe(
        guard,
        "acme_prod.reporting.signups",
        ("numerator", "denominator"),
    )
    failed = _sql("SELECT numerator / denominator FROM acme_prod.reporting.signups")
    guard.before_call("databricks_execute_sql", failed)
    guard.after_call(
        "databricks_execute_sql",
        failed,
        {"error": "[DIVIDE_BY_ZERO] Division by zero"},
    )

    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT COUNT(*) FROM acme_prod.reporting.signups"),
    ) is None


def test_first_analytical_query_requires_exact_live_metadata():
    guard = SqlRunGuard()

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT signup_date FROM acme_prod.reporting.signups"),
    )

    assert rejection["validation_failure"] == "object_metadata_required"
    assert rejection["objects"] == ["acme_prod.reporting.signups"]
    assert guard.sql_call_count == 0


def test_complete_schema_rejects_build_alias_before_databricks_execution():
    guard = SqlRunGuard()
    _describe(
        guard,
        "acme_prod.reporting.signups",
        ("_anonymousid", "signup_date", "autofill_ind", "dob"),
    )

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql(
            "SELECT COUNT(*) AS total, "
            "COUNT(CASE WHEN internal_sum > 0 THEN 1 END) AS internal_count, "
            "COUNT(CASE WHEN autofill_ind = 'autofill' THEN 1 END) AS autofill_count "
            "FROM acme_prod.reporting.signups "
            "WHERE signup_date >= CURRENT_DATE - INTERVAL 30 DAYS"
        ),
    )

    assert rejection["validation_failure"] == "column_not_in_live_schema"
    assert rejection["missing_columns"] == ["internal_sum"]
    assert guard.sql_call_count == 0


def test_complete_schema_allows_valid_case_and_function_columns():
    guard = SqlRunGuard()
    _describe(
        guard,
        "acme_prod.reporting.signups",
        ("signup_date", "autofill_ind", "dob"),
    )

    assert guard.before_call(
        "databricks_execute_sql",
        _sql(
            "SELECT COUNT(CASE WHEN autofill_ind = 'autofill' THEN 1 END) AS n "
            "FROM acme_prod.reporting.signups "
            "WHERE signup_date >= CURRENT_DATE - INTERVAL 30 DAYS "
            "AND YEAR(dob) >= 1900 ORDER BY n"
        ),
    ) is None


def test_incomplete_or_complex_metadata_checks_fail_open():
    incomplete = SqlRunGuard()
    _describe(
        incomplete,
        "acme_prod.reporting.signups",
        ("signup_date",),
        complete=False,
    )
    assert incomplete.before_call(
        "databricks_execute_sql",
        _sql("SELECT unknown_column FROM acme_prod.reporting.signups"),
    ) is None

    complex_query = SqlRunGuard()
    _describe(complex_query, "catalog_a.schema_a.left_table", ("id",))
    _describe(complex_query, "catalog_b.schema_b.right_table", ("id",))
    assert complex_query.before_call(
        "databricks_execute_sql",
        _sql(
            "SELECT l.unknown_column FROM catalog_a.schema_a.left_table l "
            "JOIN catalog_b.schema_b.right_table r ON l.id = r.id"
        ),
    ) is None


def test_metadata_enforcement_ignores_ctes_table_functions_and_paths():
    guard = SqlRunGuard()

    for sql in (
        "WITH values_cte(value) AS (SELECT 1) SELECT value FROM values_cte",
        "SELECT * FROM read_files('/tmp/example.csv')",
        "SELECT * FROM delta.`s3://example-bucket/path`",
    ):
        assert guard.before_call("databricks_execute_sql", _sql(sql)) is None


def test_default_catalog_and_schema_resolve_metadata_requirement():
    guard = SqlRunGuard()
    tool_input = {
        "sql": "SELECT signup_date FROM signups",
        "catalog": "acme_prod",
        "schema": "reporting",
    }

    rejection = guard.before_call("databricks_execute_sql", tool_input)

    assert rejection["validation_failure"] == "object_metadata_required"
    assert rejection["objects"] == ["acme_prod.reporting.signups"]


def test_adds_replan_warning_after_ten_sql_calls():
    guard = SqlRunGuard()
    result = {}

    for value in range(10):
        tool_input = _sql(f"SELECT {value}")
        assert guard.before_call("databricks_execute_sql", tool_input) is None
        result = guard.after_call(
            "databricks_execute_sql",
            tool_input,
            {"status": "SUCCEEDED"},
        )

    assert "query_budget_warning" in result
    assert "10 SQL calls" in result["query_budget_warning"]


def test_adds_replan_warning_after_two_failures():
    guard = SqlRunGuard()
    result = {}

    for value in range(2):
        tool_input = _sql(f"SELECT broken_{value}")
        guard.before_call("databricks_execute_sql", tool_input)
        result = guard.after_call(
            "databricks_execute_sql",
            tool_input,
            {"error": "PARSE_SYNTAX_ERROR"},
        )

    assert "query_budget_warning" in result
    assert "2 failures" in result["query_budget_warning"]


def test_non_sql_tools_are_not_guarded():
    guard = SqlRunGuard()

    assert guard.before_call("github_search_code", {"query": "signups"}) is None
    result = {"items": []}
    assert guard.after_call("github_search_code", {"query": "signups"}, result) is result


def test_structured_describe_reads_object_type_and_typed_columns():
    guard = SqlRunGuard()
    _describe(
        guard,
        "acme_prod.reporting.signups",
        ("_anonymousid", "payload"),
        object_type="EXTERNAL",
        column_types={"payload": "STRUCT<status: STRING>"},
    )

    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT payload:status FROM acme_prod.reporting.signups"),
    )["validation_failure"] == "invalid_struct_access_syntax"
    missing = guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT nope FROM acme_prod.reporting.signups"),
    )
    assert missing["validation_failure"] == "column_not_in_live_schema"
    assert missing["missing_columns"] == ["nope"]
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("DESCRIBE HISTORY acme_prod.reporting.signups"),
    ) is None


def test_structured_describe_truncated_columns_are_not_treated_as_complete():
    guard = SqlRunGuard()
    _describe(guard, "acme_prod.reporting.signups", ("_anonymousid",), complete=False)

    # Incomplete inventories release the object but must not reject unknown columns.
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT other_column FROM acme_prod.reporting.signups"),
    ) is None


def test_structured_describe_materialized_view_blocks_delta_history():
    guard = SqlRunGuard()
    _describe(
        guard,
        "acme_prod.reporting.signups_mv",
        ("_anonymousid",),
        object_type="MATERIALIZED_VIEW",
    )

    rejection = guard.before_call(
        "databricks_execute_sql",
        _sql("DESCRIBE HISTORY acme_prod.reporting.signups_mv"),
    )
    assert rejection["validation_failure"] == "history_requires_delta_table"
    assert rejection["object_type"] == "MATERIALIZED_VIEW"


def test_legacy_flat_describe_shape_is_still_understood():
    """Replayed DynamoDB history predates the structured contract; it must not crash."""
    guard = SqlRunGuard()
    _describe_legacy(
        guard,
        "acme_prod.reporting.signups",
        ("_anonymousid", "payload"),
        column_types={"payload": "STRUCT<status: STRING>"},
    )
    _describe_legacy(
        guard,
        "acme_prod.reporting.some_view",
        ("id",),
        object_type="VIEW",
    )

    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT payload:status FROM acme_prod.reporting.signups"),
    )["validation_failure"] == "invalid_struct_access_syntax"
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("SELECT nope FROM acme_prod.reporting.signups"),
    )["validation_failure"] == "column_not_in_live_schema"
    assert guard.before_call(
        "databricks_execute_sql",
        _sql("DESCRIBE HISTORY acme_prod.reporting.some_view"),
    )["validation_failure"] == "history_requires_delta_table"


def test_describe_without_metadata_in_either_shape_does_not_release_object():
    for empty in (
        {"status": "SUCCEEDED", "row_count": 0, "sample_rows": []},
        {
            "status": "SUCCEEDED",
            "object": {"full_name": "c.s.t", "object_type": None},
            "column_count": 0,
            "columns": [],
            "columns_truncated": False,
        },
        {"error": "boom"},
    ):
        guard = SqlRunGuard()
        guard.after_call(
            "databricks_describe_object",
            {"catalog": "c", "schema": "s", "name": "t"},
            empty,
        )
        rejection = guard.before_call("databricks_execute_sql", _sql("SELECT 1 FROM c.s.t"))
        assert rejection["validation_failure"] == "object_metadata_required"
