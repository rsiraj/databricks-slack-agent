"""Worker-level automatic metadata safety checks."""

import worker
from utils.sql_run_guard import SqlRunGuard, auto_satisfy_metadata_precondition


def _metadata_result(object_name: str, columns: tuple[str, ...]) -> dict:
    """Structured describe_object contract: object header + authoritative column_count."""
    catalog, schema, name = object_name.split(".")
    typed_columns = [
        {
            "ordinal_position": index,
            "name": column,
            "data_type": "STRING",
            "is_nullable": True,
            "comment": None,
        }
        for index, column in enumerate(columns, start=1)
    ]
    return {
        "status": "SUCCEEDED",
        "object": {
            "catalog": catalog,
            "schema": schema,
            "name": name,
            "full_name": object_name,
            "object_type": "TABLE",
            "table_type": "MANAGED",
            "comment": "test object",
            "data_source_format": "DELTA",
            "created": None,
            "last_altered": None,
        },
        "column_count": len(typed_columns),
        "columns": typed_columns,
        "columns_truncated": False,
        "note": "column_count is the authoritative number of columns.",
    }


def _legacy_metadata_result(object_name: str, columns: tuple[str, ...]) -> dict:
    """Pre-restructure flat contract still present in replayed DynamoDB history."""
    name = object_name.rsplit(".", 1)[-1]
    rows = [["table", None, name, "MANAGED", None, "test object"]]
    rows.extend(
        ["column", index, column, "STRING", "YES", None]
        for index, column in enumerate(columns, start=1)
    )
    return {
        "status": "SUCCEEDED",
        "row_count": len(rows),
        "sample_row_count": len(rows),
        "truncated": False,
        "columns": [
            "metadata_kind",
            "ordinal_position",
            "metadata_name",
            "data_type",
            "is_nullable",
            "comment",
        ],
        "sample_rows": rows,
    }


def test_first_use_metadata_is_injected_then_cached(monkeypatch):
    guard = SqlRunGuard()
    tool_input = {
        "sql": "SELECT signup_date FROM acme_prod.reporting.signups"
    }
    calls = []

    def fake_dispatch(name, inp):
        calls.append((name, inp))
        return _metadata_result(
            "acme_prod.reporting.signups",
            ("signup_date",),
        )

    monkeypatch.setattr(worker, "_dispatch_tool", fake_dispatch)
    rejection = guard.before_call("databricks_execute_sql", tool_input)

    assert worker._auto_satisfy_metadata_precondition(
        guard,
        "databricks_execute_sql",
        tool_input,
        rejection,
    ) is None
    assert calls == [
        (
            "databricks_describe_object",
            {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
        )
    ]

    second = {"sql": "SELECT COUNT(*) FROM acme_prod.reporting.signups"}
    assert guard.before_call("databricks_execute_sql", second) is None
    assert len(calls) == 1


def test_injected_schema_blocks_build_only_column(monkeypatch):
    guard = SqlRunGuard()
    tool_input = {
        "sql": "SELECT internal_sum FROM acme_prod.reporting.signups"
    }
    monkeypatch.setattr(
        worker,
        "_dispatch_tool",
        lambda *_: _metadata_result(
            "acme_prod.reporting.signups",
            ("_anonymousid", "signup_date"),
        ),
    )

    result = worker._auto_satisfy_metadata_precondition(
        guard,
        "databricks_execute_sql",
        tool_input,
        guard.before_call("databricks_execute_sql", tool_input),
    )

    assert result["validation_failure"] == "column_not_in_live_schema"
    assert result["missing_columns"] == ["internal_sum"]
    assert guard.sql_call_count == 0


def test_empty_injected_metadata_does_not_release_query(monkeypatch):
    guard = SqlRunGuard()
    tool_input = {"sql": "SELECT value FROM catalog_a.schema_a.unknown_table"}
    monkeypatch.setattr(
        worker,
        "_dispatch_tool",
        lambda *_: {
            "status": "SUCCEEDED",
            "object": {"full_name": "catalog_a.schema_a.unknown_table", "object_type": None},
            "column_count": 0,
            "columns": [],
            "columns_truncated": False,
            "warning": "No accessible column metadata matched this exact object.",
        },
    )

    original = guard.before_call("databricks_execute_sql", tool_input)
    result = worker._auto_satisfy_metadata_precondition(
        guard,
        "databricks_execute_sql",
        tool_input,
        original,
    )

    assert result == original
    assert guard.sql_call_count == 0


def test_legacy_empty_injected_metadata_does_not_release_query(monkeypatch):
    guard = SqlRunGuard()
    tool_input = {"sql": "SELECT value FROM catalog_a.schema_a.unknown_table"}
    monkeypatch.setattr(
        worker,
        "_dispatch_tool",
        lambda *_: {"status": "SUCCEEDED", "row_count": 0, "sample_rows": []},
    )

    original = guard.before_call("databricks_execute_sql", tool_input)
    result = worker._auto_satisfy_metadata_precondition(
        guard,
        "databricks_execute_sql",
        tool_input,
        original,
    )

    assert result == original
    assert guard.sql_call_count == 0


def test_legacy_flat_metadata_shape_still_releases_and_guards_columns(monkeypatch):
    """Replayed history may carry the old flat describe shape; it must not crash."""
    guard = SqlRunGuard()
    tool_input = {"sql": "SELECT internal_sum FROM acme_prod.reporting.signups"}
    monkeypatch.setattr(
        worker,
        "_dispatch_tool",
        lambda *_: _legacy_metadata_result(
            "acme_prod.reporting.signups",
            ("_anonymousid", "signup_date"),
        ),
    )

    result = worker._auto_satisfy_metadata_precondition(
        guard,
        "databricks_execute_sql",
        tool_input,
        guard.before_call("databricks_execute_sql", tool_input),
    )

    assert result["validation_failure"] == "column_not_in_live_schema"
    assert result["missing_columns"] == ["internal_sum"]
    assert guard.before_call(
        "databricks_execute_sql",
        {"sql": "SELECT signup_date FROM acme_prod.reporting.signups"},
    ) is None


def test_shared_preflight_describes_every_exact_object_before_release():
    guard = SqlRunGuard()
    tool_input = {
        "sql": (
            "SELECT r.signup_date "
            "FROM acme_prod.reporting.signups r "
            "JOIN acme_prod.app_server.signup_unified u "
            "ON u.anonymous_id = r._anonymousId"
        )
    }
    columns_by_object = {
        "acme_prod.reporting.signups": ("signup_date", "_anonymousid"),
        "acme_prod.app_server.signup_unified": ("anonymous_id",),
    }

    def dispatch(_name, metadata_input):
        object_name = ".".join(
            metadata_input[part] for part in ("catalog", "schema", "name")
        )
        return _metadata_result(object_name, columns_by_object[object_name])

    rejection = guard.before_call("databricks_execute_sql", tool_input)
    result, attempts = auto_satisfy_metadata_precondition(
        guard,
        "databricks_execute_sql",
        tool_input,
        rejection,
        dispatch,
    )

    assert result is None
    assert [attempt["input"] for attempt in attempts] == [
        {"catalog": "acme_prod", "schema": "app_server", "name": "signup_unified"},
        {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
    ]
    assert all(attempt["succeeded"] for attempt in attempts)


def test_shared_preflight_does_not_guess_an_inexact_object():
    guard = SqlRunGuard()
    rejection = {
        "validation_failure": "object_metadata_required",
        "objects": ["signups"],
    }
    dispatches = []

    result, attempts = auto_satisfy_metadata_precondition(
        guard,
        "databricks_execute_sql",
        {"sql": "SELECT signup_date FROM signups"},
        rejection,
        lambda name, tool_input: dispatches.append((name, tool_input)),
    )

    assert result is rejection
    assert attempts == []
    assert dispatches == []


def test_explicit_description_needs_no_automatic_preflight():
    guard = SqlRunGuard()
    result, attempts = auto_satisfy_metadata_precondition(
        guard,
        "databricks_describe_object",
        {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
        None,
        lambda *_: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )

    assert result is None
    assert attempts == []
