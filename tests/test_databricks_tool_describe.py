"""Structured ``databricks_describe_object`` contract.

A live thread showed the old flat result set (one ``metadata_kind="table"`` row
plus N ``column`` rows, ``row_count = N + 1``) being read as "N + 1 columns".
The tool now returns ``object`` / ``column_count`` / ``columns`` so the count
cannot be misread and nullability is a real boolean.
"""

import json

import pytest

from tools import databricks_tool as dt


FLAT_HEADERS = list(dt._DESCRIBE_HEADERS)


def _flat_rows(column_count: int, *, table_type: str = "MANAGED", nullable: str = "YES"):
    rows = [
        [
            "table",
            None,
            "signups",
            table_type,
            None,
            "One row per anonymous identity.",
            "DELTA",
            "2024-01-01T00:00:00Z",
            "2026-08-28T09:00:00Z",
        ]
    ]
    rows.extend(
        ["column", position, f"col_{position}", "STRING", nullable, None, None, None, None]
        for position in range(1, column_count + 1)
    )
    return rows


def _fake_execute(rows, *, truncated=False, row_count=None, extra=None):
    def fake(sql, **kwargs):
        assert kwargs.get("metadata_query") is True
        # The real information_schema DATA_TYPE only returns STRUCT/ARRAY;
        # FULL_DATA_TYPE supplies the nested definitions in these fixtures.
        assert "full_data_type AS data_type" in sql
        result = {
            "status": "SUCCEEDED",
            "statement_id": "stmt-describe",
            "row_count": len(rows) if row_count is None else row_count,
            "truncated": truncated,
            "is_sample": False,
            "result_scope": "x",
            "columns": FLAT_HEADERS,
            "sample_rows": rows,
            "sample_row_count": len(rows),
        }
        result.update(extra or {})
        return result

    return fake


def test_describe_returns_object_header_and_authoritative_column_count(monkeypatch):
    monkeypatch.setattr(dt, "execute_sql", _fake_execute(_flat_rows(35)))

    result = dt.describe_object("acme_prod", "reporting", "signups")

    assert set(result) == {
        "status",
        "object",
        "column_count",
        "columns",
        "columns_truncated",
        "note",
        "statement_id",
    }
    assert result["status"] == "SUCCEEDED"
    assert result["statement_id"] == "stmt-describe"
    # 35 columns + 1 table row must not surface as 36.
    assert result["column_count"] == 35 == len(result["columns"])
    assert result["columns_truncated"] is False
    assert result["object"] == {
        "catalog": "acme_prod",
        "schema": "reporting",
        "name": "signups",
        "full_name": "acme_prod.reporting.signups",
        "object_type": "TABLE",
        "table_type": "MANAGED",
        "comment": "One row per anonymous identity.",
        "data_source_format": "DELTA",
        "created": "2024-01-01T00:00:00Z",
        "last_altered": "2026-08-28T09:00:00Z",
    }
    assert "column_count is the authoritative number of columns" in result["note"]
    serialized = json.dumps(result)
    for legacy_key in ("row_count", "sample_row_count", "sample_rows", "metadata_kind"):
        assert legacy_key not in serialized


def test_describe_columns_are_typed_with_boolean_nullability(monkeypatch):
    rows = _flat_rows(2)
    rows[1][4] = "YES"
    rows[2][4] = "NO"
    rows[2][3] = "STRUCT<status: STRING>"
    rows[2][5] = "Nested payload"
    monkeypatch.setattr(dt, "execute_sql", _fake_execute(rows))

    result = dt.describe_object("acme_prod", "reporting", "signups")

    assert result["columns"] == [
        {
            "ordinal_position": 1,
            "name": "col_1",
            "data_type": "STRING",
            "is_nullable": True,
            "comment": None,
        },
        {
            "ordinal_position": 2,
            "name": "col_2",
            "data_type": "STRUCT<status: STRING>",
            "is_nullable": False,
            "comment": "Nested payload",
        },
    ]
    assert result["columns"][0]["is_nullable"] is True
    assert result["columns"][1]["is_nullable"] is False


def test_live_full_type_metadata_drives_struct_access_guard(monkeypatch):
    from utils.sql_run_guard import SqlRunGuard

    rows = _flat_rows(1)
    rows[1][2] = "entity_metadata"
    rows[1][3] = "STRUCT<job_info: STRUCT<job_id: STRING>>"
    monkeypatch.setattr(dt, "execute_sql", _fake_execute(rows))
    target = {"catalog": "system", "schema": "access", "name": "table_lineage"}
    metadata = dt.describe_object(**target)
    guard = SqlRunGuard()
    guard.after_call("databricks_describe_object", target, metadata)

    rejected = guard.before_call("databricks_execute_sql", {
        "sql": "SELECT entity_metadata:job_info.job_id FROM system.access.table_lineage",
    })
    assert rejected["validation_failure"] == "invalid_struct_access_syntax"
    assert guard.before_call("databricks_execute_sql", {
        "sql": "SELECT entity_metadata.job_info.job_id FROM system.access.table_lineage",
    }) is None


def test_describe_orders_columns_by_ordinal_position_regardless_of_row_order(monkeypatch):
    rows = _flat_rows(3)
    table_row, column_rows = rows[0], rows[1:]
    monkeypatch.setattr(
        dt, "execute_sql", _fake_execute(list(reversed(column_rows)) + [table_row])
    )

    result = dt.describe_object("acme_prod", "reporting", "signups")

    assert [column["ordinal_position"] for column in result["columns"]] == [1, 2, 3]
    assert result["object"]["object_type"] == "TABLE"


@pytest.mark.parametrize(
    ("table_type", "object_type"),
    [
        ("MANAGED", "TABLE"),
        ("EXTERNAL", "TABLE"),
        ("FOREIGN", "TABLE"),
        ("MANAGED_SHALLOW_CLONE", "TABLE"),
        ("VIEW", "VIEW"),
        ("MATERIALIZED_VIEW", "MATERIALIZED_VIEW"),
        ("MATERIALIZED VIEW", "MATERIALIZED_VIEW"),
        ("STREAMING_TABLE", "STREAMING_TABLE"),
    ],
)
def test_object_type_is_derived_from_information_schema_table_type(
    monkeypatch, table_type, object_type
):
    monkeypatch.setattr(dt, "execute_sql", _fake_execute(_flat_rows(1, table_type=table_type)))

    result = dt.describe_object("acme_prod", "reporting", "signups")

    assert result["object"]["table_type"] == table_type
    assert result["object"]["object_type"] == object_type


def test_describe_not_found_keeps_succeeded_warning_contract(monkeypatch):
    monkeypatch.setattr(dt, "execute_sql", _fake_execute([]))

    result = dt.describe_object("catalog_a", "schema_a", "missing")

    assert result["status"] == "SUCCEEDED"
    assert "error" not in result
    assert result["column_count"] == 0
    assert result["columns"] == []
    assert result["object"]["full_name"] == "catalog_a.schema_a.missing"
    assert result["object"]["object_type"] is None
    assert result["warning"] == dt.DESCRIBE_OBJECT_NOT_FOUND_WARNING


def test_describe_flags_truncated_column_inventory(monkeypatch):
    monkeypatch.setattr(
        dt, "execute_sql", _fake_execute(_flat_rows(10), truncated=True, row_count=200)
    )

    result = dt.describe_object("acme_prod", "reporting", "signups")

    assert result["columns_truncated"] is True
    assert result["column_count"] == 10 == len(result["columns"])


def test_describe_passes_through_failures_and_running_statements(monkeypatch):
    for raw in (
        {"error": "boom", "validation_failure": "x"},
        {"status": "FAILED", "statement_id": "s", "error": "bad"},
        {"status": "RUNNING", "statement_id": "s", "note": "still running"},
    ):
        monkeypatch.setattr(dt, "execute_sql", lambda sql, **kwargs: dict(raw))

        result = dt.describe_object("acme_prod", "reporting", "signups")

        for key, value in raw.items():
            assert result[key] == value
        assert result["object"]["full_name"] == "acme_prod.reporting.signups"
        assert "column_count" not in result


def test_describe_rejects_invalid_identifiers_before_running_sql(monkeypatch):
    monkeypatch.setattr(
        dt,
        "execute_sql",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run SQL")),
    )

    result = dt.describe_object("", "reporting", "signups")

    assert result["validation_failure"] == "invalid_metadata_input"


def test_describe_query_selects_table_and_column_metadata(monkeypatch):
    captured = {}

    def fake(sql, **kwargs):
        captured["sql"] = sql
        return _fake_execute(_flat_rows(1))(sql, **kwargs)

    monkeypatch.setattr(dt, "execute_sql", fake)

    dt.describe_object("acme_prod", "reporting", "signups")

    sql = captured["sql"]
    assert "system.information_schema.tables" in sql
    assert "system.information_schema.columns" in sql
    assert "data_source_format" in sql
    assert "last_altered" in sql
    assert "signups" not in sql  # hex-encoded literals only


def test_metadata_queries_keep_every_row_instead_of_a_fifty_row_sample():
    payload = {
        "statement_id": "s",
        "status": {"state": "SUCCEEDED"},
        "manifest": {
            "total_row_count": 120,
            "truncated": False,
            "schema": {"columns": [{"name": header} for header in FLAT_HEADERS]},
        },
        "result": {"data_array": _flat_rows(119)},
    }

    default = dt._format_status(payload)
    metadata = dt._format_status(payload, dt.METADATA_SAMPLE_ROW_LIMIT)

    assert default["sample_row_count"] == dt.SAMPLE_ROW_LIMIT == 50
    assert metadata["sample_row_count"] == 120
    assert metadata["is_sample"] is False


def test_tool_description_states_column_count_is_authoritative():
    tool = next(t for t in dt.DATABRICKS_TOOLS if t["name"] == "databricks_describe_object")

    assert "column_count" in tool["description"]
    assert "authoritative" in tool["description"]
    assert tool["input_schema"]["required"] == ["catalog", "schema", "name"]
    assert set(tool["input_schema"]["properties"]) == {"catalog", "schema", "name"}


def test_build_describe_result_is_shared_with_the_synthetic_runtime():
    result = dt.build_describe_result(
        "c", "s", "t", FLAT_HEADERS, _flat_rows(2, nullable="NO")
    )

    assert result["column_count"] == 2
    assert [column["is_nullable"] for column in result["columns"]] == [False, False]
    assert result["object"]["full_name"] == "c.s.t"
