"""sqlglot-backed SQL guard — tools/databricks_tool.py._validate_sql.

Each case in ``BYPASSES`` was verified to slip past the previous keyword-only
guard. These tests pin the structural (parsed-tree) checks, the literal- and
comment-aware single-statement scanner, and the side-effecting function
deny-list.
"""

import pytest

from tools import databricks_tool as dt
from tools.databricks_tool import _repair_metadata_limit, _validate_sql

# (sql, expected validation_failure) — every one of these previously returned None.
BYPASSES = [
    (
        "WITH x AS (SELECT 1 AS id) INSERT INTO acme_prod.reporting.signups SELECT * FROM x",
        "write_statement",
    ),
    ("EXPLAIN DELETE FROM acme_prod.reporting.signups", "write_statement"),
    ("SELECT 'a -- b'; DROP TABLE t", "multiple_statements"),
    ("SELECT http_request(conn=>'c', method=>'POST', path=>'/x')", "denied_function"),
]

ALLOWED = [
    "WITH x AS (SELECT 1) SELECT * FROM x",
    # Delta time travel; parsed by sqlglot >= 30.17 (normalised to TIMESTAMP AS OF).
    "SELECT * FROM t FOR SYSTEM_TIME AS OF '2026-01-01'",
    "SELECT 1 UNION SELECT 2",
    "SELECT a FROM t INTERSECT SELECT a FROM u",
    "SELECT a FROM t EXCEPT SELECT a FROM u",
    "SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 ORDER BY 1",
    "EXPLAIN SELECT 1",
    "EXPLAIN EXTENDED WITH a AS (SELECT 1) SELECT * FROM a",
    "DESCRIBE HISTORY acme_prod.reporting.signups",
    "DESCRIBE DETAIL acme_prod.reporting.signups",
    "DESC TABLE EXTENDED acme_prod.reporting.signups signup_date",
    "DESCRIBE QUERY SELECT 1",
    "SHOW TABLES IN x",
    "SHOW TABLES IN acme_prod.reporting LIKE 'reg*'",
    "SHOW TBLPROPERTIES acme_prod.reporting.signups",
    "SHOW CREATE TABLE acme_prod.reporting.signups",
    "SHOW PARTITIONS acme_prod.reporting.signups",
    "SHOW COLUMNS IN acme_prod.reporting.signups",
    "SELECT 'a;b'",
    'SELECT "x -- y" FROM t',
    "SELECT 1 /* trailing ; comment */",
    "SELECT 1;",
    "SELECT 1; -- done",
    "SELECT * FROM t WHERE x = 1 -- comment ; here",
    "SELECT * FROM t WHERE name = 'http_request('",
    "SELECT sql_text, ai_query_count FROM t",
    "SELECT * FROM t TIMESTAMP AS OF '2026-01-01'",
    "SELECT * FROM delta.`s3://example-bucket/path`",
    "SELECT entity_metadata:job_info.job_id FROM system.access.table_lineage",
    "SELECT * FROM t WHERE x = ${var}",
    "SELECT :param",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts) AS rn FROM source QUALIFY rn = 1",
    (
        "SELECT * FROM system.access.audit WHERE event_time >= current_timestamp() - "
        "INTERVAL 7 DAYS AND action_name IN ('commandSubmit')"
    ),
    (
        "WITH Signups AS (\n"
        "  SELECT rc.anonymous_id, rc.access_data_dob::date AS dob,\n"
        "         ROW_NUMBER() OVER (PARTITION BY rc.anonymous_id ORDER BY rc.original_timestamp"
        " NULLS LAST) AS row_num\n"
        "  FROM acme_prod.app_server.signup_created rc\n"
        "  WHERE rc.event_date >= current_date() - INTERVAL '30 days'\n"
        "), Referral AS (\n"
        "  SELECT anonymous_id, flow FROM acme_prod.app_server.referral_created\n"
        "  WHERE context_source IS DISTINCT FROM 'mobile-app'\n"
        "  QUALIFY ROW_NUMBER() OVER (PARTITION BY anonymous_id ORDER BY original_timestamp DESC) = 1\n"
        ")\n"
        "SELECT r.anonymous_id, get_json_object(r.payload, '$.a') AS j, h.flow\n"
        "FROM Signups r LEFT JOIN Referral h ON h.anonymous_id = r.anonymous_id\n"
        "LATERAL VIEW explode(r.arr) e AS x\n"
        "WHERE r.row_num = 1 AND EXISTS (SELECT 1 FROM q WHERE q.id = r.anonymous_id)\n"
        "GROUP BY ALL HAVING COUNT(*) > 1 ORDER BY 1 DESC NULLS LAST LIMIT 100"
    ),
]

# Every Databricks SHOW statement is read-only. The guard accepts any single
# SHOW statement with no write / admin keyword token (SHOW CREATE TABLE's
# leading CREATE excepted) and no deny-listed function, instead of an
# allowlist of forms that rejected valid ones as disallowed_statement_type.
SHOW_ALLOWED = [
    "SHOW TAGS",
    "SHOW EXTERNAL LOCATIONS",
    "SHOW MATERIALIZED VIEWS IN x",
    "SHOW STREAMING TABLES",
    "SHOW SYSTEM FUNCTIONS",
    "SHOW USER FUNCTIONS LIKE 'a*'",
    "SHOW ALL FUNCTIONS",
    "SHOW VOLUMES IN c.s",
    "SHOW CONNECTIONS",
    "SHOW CREDENTIALS",
    "SHOW SHARES",
    "SHOW RECIPIENTS",
    "SHOW PROVIDERS",
    "SHOW CATALOGS LIKE 'm*'",
    "SHOW GRANTS ON TABLE t",
    "SHOW GRANTS TO user",
    "SHOW TBLPROPERTIES t",
    "SHOW PARTITIONS t",
    "SHOW COLUMNS IN t",
    "SHOW CREATE TABLE t",
    "show create table t",
    "SHOW VIEWS IN s",
    "SHOW TABLES IN s LIKE 'x*'",
    "SHOW TABLE EXTENDED IN s LIKE 'x*'",
    "SHOW SCHEMAS",
    "SHOW DATABASES",
    "SHOW FUNCTIONS",
    "SHOW GROUPS",
    "SHOW USERS",
    "SHOW TABLES; -- DROP TABLE t",
    "SHOW TABLES IN s LIKE 'drop*'",
    "/* CREATE */ SHOW TABLES IN `create_schema`",
]

SHOW_BLOCKED = [
    ("SHOW TABLES /* outer /* inner */ ' */; DROP TABLE t", "multiple_statements"),
    ("SHOW TABLES /* outer /* inner */ ` */; SELECT 2", "multiple_statements"),
    ("SHOW CREATE TABLE t; CREATE TABLE x", "multiple_statements"),
    ("SHOW TABLES IN s; DROP TABLE t", "multiple_statements"),
    ("SHOW CREATE TABLE t CREATE TABLE x", "write_statement"),
    ("SHOW TABLES IN s DROP TABLE t", "write_statement"),
    ("SHOW CREATE VIEW v", "write_statement"),
    ("SHOW GRANTS ON t GRANT SELECT ON t TO u", "disallowed_statement_type"),
    ("SHOW TABLES IN (SELECT http_request(conn => 'c', method => 'GET', path => '/x'))", "denied_function"),
    ("SHOW TABLES IN (SELECT REMOTE_QUERY('conn', query => 'select 1'))", "denied_function"),
]

BLOCKED = [
    ("SELECT 'abc", "unparseable_sql"),
    ("SELECT 1; DROP TABLE t", "multiple_statements"),
    ("SHOW TABLES; DROP TABLE t", "multiple_statements"),
    ("SELECT 1 -- it's\n; DROP TABLE t", "multiple_statements"),
    (r"SELECT r'\'; DROP TABLE t", "multiple_statements"),
    ("SELECT 'it''s'; DROP TABLE t", "multiple_statements"),
    ("WITH x AS (INSERT INTO t VALUES (1)) SELECT 1", "write_statement"),
    ("WITH x AS (SELECT 1) DELETE FROM t WHERE id IN (SELECT * FROM x)", "write_statement"),
    ("WITH x AS (SELECT 1) UPDATE t SET a = 1", "write_statement"),
    ("WITH x AS (SELECT 1) MERGE INTO t USING x ON t.id = x.id WHEN MATCHED THEN DELETE", "write_statement"),
    ("EXPLAIN INSERT INTO t VALUES (1)", "write_statement"),
    ("EXPLAIN MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE", "write_statement"),
    ("DESCRIBE QUERY DELETE FROM t", "write_statement"),
    ("EXPLAIN SHOW TABLES", "disallowed_statement_type"),
    ("EXPLAIN EXPLAIN SELECT 1", "disallowed_statement_type"),
    ("EXPLAIN SET x = 1", "disallowed_statement_type"),
    ("SHOW USERS; SELECT 1", "multiple_statements"),
    ("SELECT * FROM remote_query('conn', query => 'select 1')", "denied_function"),
    ("SELECT * FROM READ_FILES('/Volumes/x/y')", "denied_function"),
    ("SELECT ai_query('endpoint', prompt)", "denied_function"),
    ("SELECT ai_gen('x')", "denied_function"),
    ("SELECT ai_forecast(TABLE(t), horizon => 1)", "denied_function"),
    ("SELECT * FROM vector_search(index => 'i', query => 'q')", "denied_function"),
    ("SELECT * FROM read_kafka(bootstrapServers => 'x')", "denied_function"),
    ("SELECT * FROM read_kinesis(streamName => 'x')", "denied_function"),
    ("SELECT * FROM read_pulsar(serviceUrl => 'x')", "denied_function"),
    ("SELECT * FROM read_state_metadata('/p')", "denied_function"),
    ("SELECT reflect('java.util.UUID', 'randomUUID')", "denied_function"),
    ("SELECT java_method('java.lang.Math', 'max', 1, 2)", "denied_function"),
    ("SELECT sql('DROP TABLE t')", "denied_function"),
    ("SELECT * FROM t WHERE x = (SELECT http_request(conn => 'c'))", "denied_function"),
    ("SELECT   HTTP_REQUEST  (conn => 'c')", "denied_function"),
    ("SELECT 1；DROP TABLE t", "write_statement"),
    ("EXPLAIN", "unparseable_sql"),
    ("DESCRIBE 123", "unparseable_sql"),
    ("SHOW GRANT SELECT ON t TO u; GRANT SELECT ON t TO u", "multiple_statements"),
]

# Valid Databricks read-only SQL that the pinned sqlglot (30.14.0; still true on
# 30.18.0) cannot parse, plus syntactically broken SELECTs. None of these may
# be rejected by the guard: sqlglot failing to parse means "cannot verify
# structurally", not "unsafe", so the guard falls back to the token scan and
# lets Databricks be the authority on syntax.
PARSE_FALLBACK_ALLOWED = [
    # Exact statement rejected as unparseable_sql in an earlier CI run
    # ("Conversation evaluations"), scored as avoidable_tool_failure. It is not
    # valid Databricks SQL, but the guard's job is read-only enforcement, not
    # syntax linting; the warehouse returns the syntax error.
    "\nSELECT DESCRIBE HISTORY acme_prod.reporting.signups LIMIT 5\n",
    "SELECT * FROM acme_prod.reporting.signups FOR SYSTEM_VERSION AS OF 3",
    # `FOR SYSTEM_TIME AS OF` moved to ALLOWED: sqlglot >= 30.17 parses it (as
    # TIMESTAMP AS OF), so the tree walk allows it and no fallback fires.
    "SELECT 5 MOD 2",
    "EXPLAIN SELECT * FROM t FOR SYSTEM_VERSION AS OF 3",
    "SELECT replace(name, 'a', 'b') FROM t FOR SYSTEM_VERSION AS OF 3",
    "SELECT * FROM t FOR SYSTEM_VERSION AS OF 3 WHERE operation = 'DELETE' -- DROP",
    "SELECT * FROM t FOR SYSTEM_VERSION AS OF 3 WHERE `update` = 1",
    "SELECT * FROM",
    "SELECT 1；SELECT 2",
]


@pytest.mark.parametrize("sql", [
    "/* outer /* inner */ ' DROP TABLE */ SELECT 1",
    "SELECT 1 /* outer /* inner */ ' ; DROP TABLE t */",
    "SHOW TABLES /* outer /* inner */ ' ; DROP TABLE t */",
    "/* outer /* middle /* inner */ ; */ ' */ SHOW TABLES",
])
def test_valid_nested_comments_preserve_read_only_queries(sql):
    assert _validate_sql(sql) is None


def test_nested_comment_boundaries_keep_following_statements_visible():
    sql = "SHOW TABLES /* outer /* inner */ ' */; DROP TABLE t"
    chunks = dt._tokenize_sql(sql)
    assert chunks == [
        ("code", "SHOW TABLES "),
        ("comment", "/* outer /* inner */ ' */"),
        ("code", "; DROP TABLE t"),
    ]
    assert dt._has_multiple_statements(sql)
    assert dt._read_only_structure_error(sql)["validation_failure"] == "multiple_statements"


@pytest.mark.parametrize("sql,name", [
    ("SELECT IDENTIFIER('http_request')(conn=>'c',method=>'POST',path=>'/x')", "http_request"),
    ("SELECT IDENTIFIER('reflect')('java.lang.System','getenv','PATH')", "reflect"),
    ("SELECT `http_request`(conn=>'c',method=>'POST',path=>'/x') FROM t FOR SYSTEM_VERSION AS OF 1", "http_request"),
    ("SELECT `reflect`('java.lang.System','getenv','PATH') FROM t FOR SYSTEM_VERSION AS OF 1", "reflect"),
    ("SELECT IDENTIFIER('system.builtin.`http_request`')(conn=>'c')", "http_request"),
    ("SELECT IDENTIFIER /* comment */ ('http_request') /* comment */ (conn=>'c')", "http_request"),
])
def test_quoted_and_identifier_function_bypasses_are_rejected(sql, name):
    for validate in (_validate_sql, dt._read_only_structure_error):
        error = validate(sql)
        assert error["validation_failure"] == "denied_function"
        assert error["denied_function"] == name


@pytest.mark.parametrize("sql", [
    "SELECT IDENTIFIER(:function_name)(1)",
    "SELECT IDENTIFIER('http_' || 'request')(conn=>'c')",
    "SELECT IDENTIFIER(concat('http_', 'request'))(conn=>'c')",
    "SELECT IDENTIFIER('identifier')('http_request')(conn=>'c')",
])
def test_unresolved_dynamic_function_names_fail_closed(sql):
    assert _validate_sql(sql)["validation_failure"] == "unresolved_function"


@pytest.mark.parametrize("sql", [
    "SELECT IDENTIFIER('max')(id) FROM t",
    "SELECT IDENTIFIER('`max`')(id) FROM t FOR SYSTEM_VERSION AS OF 1",
    "SELECT * FROM IDENTIFIER('acme_prod.reporting.signups')",
    "SELECT IDENTIFIER(:column_name) FROM t",
    "SELECT `replace`(name, 'http_request(', '') FROM t FOR SYSTEM_VERSION AS OF 1",
    "SELECT 'IDENTIFIER(:function_name)(1)'",
])
def test_verifiable_functions_and_nonfunction_identifiers_remain_allowed(sql):
    assert _validate_sql(sql) is None


@pytest.mark.parametrize("sql,expected", BYPASSES)
def test_previous_bypasses_are_now_rejected(sql, expected):
    err = _validate_sql(sql)

    assert err is not None, sql
    assert err["validation_failure"] == expected
    assert err["error"]


@pytest.mark.parametrize("sql", ALLOWED)
def test_read_only_forms_are_allowed(sql):
    assert _validate_sql(sql) is None, sql


@pytest.mark.parametrize("sql,expected", BLOCKED)
def test_dangerous_forms_are_blocked(sql, expected):
    err = _validate_sql(sql)

    assert err is not None, sql
    assert err["validation_failure"] == expected, (sql, err)


@pytest.mark.parametrize("sql", SHOW_ALLOWED)
def test_every_read_only_show_form_is_allowed(sql, audits):
    assert _validate_sql(sql) is None, sql
    # SHOW never went through sqlglot as a structural check, so it is not a
    # parse fallback and must not be audited as one.
    assert audits == []


@pytest.mark.parametrize("sql,expected", SHOW_BLOCKED)
def test_show_with_write_admin_or_denied_tokens_is_blocked(sql, expected):
    err = _validate_sql(sql)

    assert err is not None, sql
    assert err["validation_failure"] == expected, (sql, err)


def test_show_structural_layer_rechecks_independently():
    # The structural layer applies the SHOW rule itself, independent of the
    # text-level checks _validate_sql runs first.
    assert dt._read_only_structure_error("SHOW CREATE TABLE t") is None
    assert dt._read_only_structure_error("SHOW TAGS") is None
    assert dt._read_only_structure_error("SHOW TABLES IN s; DROP TABLE t")["validation_failure"] == (
        "multiple_statements"
    )
    err = dt._read_only_structure_error("SHOW CREATE TABLE t CREATE TABLE x")
    assert err["validation_failure"] == "write_statement"
    assert err["first_keyword"] == "CREATE"
    err = dt._read_only_structure_error("SHOW TABLES IN (SELECT http_request(conn => 'c'))")
    assert err["validation_failure"] == "denied_function"
    err = dt._read_only_structure_error("SHOW TABLES", allow_metadata=False)
    assert err["validation_failure"] == "disallowed_statement_type"
    assert err["first_keyword"] == "SHOW"


def test_write_statement_error_names_the_offending_statement():
    err = _validate_sql("WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x")

    assert err["validation_failure"] == "write_statement"
    assert err["first_keyword"] == "INSERT"
    assert "INSERT" in err["error"]
    assert "CTE" in err["error"]


def test_denied_function_error_names_the_function():
    err = _validate_sql("SELECT * FROM t WHERE Remote_Query('c', query => 'x') IS NOT NULL")

    assert err["validation_failure"] == "denied_function"
    assert err["denied_function"] == "remote_query"
    assert "remote_query()" in err["error"]


def test_merge_keeps_legacy_leading_keyword_code():
    # Leading-keyword rejections keep the code and field existing tests and the
    # system prompt rely on.
    err = _validate_sql("MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE")

    assert err["validation_failure"] == "disallowed_statement_type"
    assert err["first_keyword"] == "MERGE"


def test_fullwidth_semicolon_is_not_a_statement_separator():
    # U+FF1B is not SQL syntax, so it can neither split statements nor be executed.
    # sqlglot cannot parse it, so the token-scan fallback is what catches DROP.
    assert dt._has_multiple_statements("SELECT 1；SELECT 2") is False
    err = _validate_sql("SELECT 1；DROP TABLE t")
    assert err is not None
    assert err["validation_failure"] == "write_statement"
    assert err["first_keyword"] == "DROP"


def test_unparseable_error_is_actionable_and_free_of_terminal_escapes():
    err = _validate_sql("DESCRIBE 123")

    assert err["validation_failure"] == "unparseable_sql"
    assert "\x1b" not in err["error"]
    assert "Fix the syntax" in err["error"]


@pytest.fixture
def audits(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(dt, "audit", lambda event, **fields: events.append((event, fields)))
    return events


@pytest.mark.parametrize("sql", PARSE_FALLBACK_ALLOWED)
def test_sql_sqlglot_cannot_parse_falls_back_to_token_scan(sql, audits):
    assert _validate_sql(sql) is None, sql

    # Exactly one fallback event proves sqlglot really failed on this case (for
    # EXPLAIN it fails on the inner query; the outer statement parses as an
    # opaque Command) and that the token scan, not the tree walk, allowed it.
    fallback = [fields for event, fields in audits if event == "sql_parse_fallback"]
    assert len(fallback) == 1
    assert fallback[0]["first_keyword"] in {"SELECT", "WITH"}
    assert fallback[0]["parse_error"]
    assert "\x1b" not in fallback[0]["parse_error"]
    assert fallback[0]["check"] == "token_scan"


def test_no_fallback_audit_when_sqlglot_parses(audits):
    assert _validate_sql("WITH x AS (SELECT 1) SELECT * FROM x") is None
    assert audits == []


class TestTokenScanFallback:
    """Force sqlglot to fail so every case exercises the fallback path."""

    @pytest.fixture(autouse=True)
    def _sqlglot_always_fails(self, monkeypatch):
        def raise_parse_error(sql, read=None, **kwargs):
            raise dt.SqlglotError("forced parse failure")

        monkeypatch.setattr(dt.sqlglot, "parse", raise_parse_error)

    def test_write_inside_cte_is_still_blocked(self, audits):
        err = _validate_sql("WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x")

        assert err["validation_failure"] == "write_statement"
        assert err["first_keyword"] == "INSERT"
        assert audits == []

    def test_denied_function_is_still_blocked(self, audits):
        err = _validate_sql("SELECT http_request(conn=>'c', method=>'POST', path=>'/x')")
        assert err["validation_failure"] == "denied_function"
        assert err["denied_function"] == "http_request"

        # The structural layer re-checks the deny-list itself, independent of
        # the text scan _validate_sql runs first.
        err = dt._read_only_structure_error("SELECT * FROM read_kafka(bootstrapServers => 'x')")
        assert err["validation_failure"] == "denied_function"
        assert audits == []

    @pytest.mark.parametrize(
        "sql,expected,keyword",
        [
            ("WITH x AS (SELECT 1) DELETE FROM t WHERE id IN (SELECT * FROM x)", "write_statement", "DELETE"),
            ("WITH x AS (SELECT 1) UPDATE t SET a = 1", "write_statement", "UPDATE"),
            ("WITH x AS (SELECT 1) MERGE INTO t USING x ON t.id = x.id", "write_statement", "MERGE"),
            ("SELECT 1 UNION ALL SELECT 1 FROM t CREATE TABLE u AS SELECT 1", "write_statement", "CREATE"),
            ("WITH x AS (SELECT 1) REPLACE TABLE t AS SELECT * FROM x", "write_statement", "REPLACE"),
            ("SELECT * FROM t OPTIMIZE t", "write_statement", "OPTIMIZE"),
            ("WITH x AS (SELECT 1) SET spark.sql.x = 1", "disallowed_statement_type", "SET"),
            ("WITH x AS (SELECT 1) GRANT SELECT ON t TO u", "disallowed_statement_type", "GRANT"),
            ("WITH x AS (SELECT 1) CALL proc()", "disallowed_statement_type", "CALL"),
            ("SELECT 1 REFRESH TABLE t", "disallowed_statement_type", "REFRESH"),
            ("SELECT 1 EXECUTE IMMEDIATE 'DROP TABLE t'", "denied_function", None),
            ("EXPLAIN WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x", "write_statement", "INSERT"),
        ],
    )
    def test_write_and_admin_tokens_are_blocked(self, sql, expected, keyword, audits):
        err = _validate_sql(sql)

        assert err is not None, sql
        assert err["validation_failure"] == expected, (sql, err)
        if keyword:
            assert err["first_keyword"] == keyword
        assert audits == []

    @pytest.mark.parametrize(
        "sql",
        [
            "WITH x AS (SELECT 1) SELECT * FROM x",
            "SELECT * FROM t WHERE operation IN ('DELETE', 'MERGE', 'CREATE') -- DROP TABLE t",
            "SELECT * FROM t /* INSERT INTO t */ WHERE `update` = 1 AND \"set\" = 2",
            "SELECT replace(name, 'a', 'b'), updated_at, is_deleted, call_id FROM t",
            "SELECT 1 UNION SELECT 2",
            "EXPLAIN SELECT * FROM t WHERE a = 1",
        ],
    )
    def test_read_only_text_is_allowed_and_audited(self, sql, audits):
        assert _validate_sql(sql) is None, sql

        assert [event for event, _ in audits] == ["sql_parse_fallback"]
        assert audits[0][1]["parse_error"] == "forced parse failure"
        assert audits[0][1]["sql"].strip()

    def test_second_statement_is_still_blocked(self):
        assert _validate_sql("SELECT 1; DROP TABLE t")["validation_failure"] == "multiple_statements"
        assert dt._read_only_structure_error("SELECT 1; SELECT 2")["validation_failure"] == (
            "multiple_statements"
        )

    def test_fallback_is_limited_to_select_and_with(self, audits):
        # DESCRIBE keeps the explicit allowlist and SHOW uses the token scan
        # directly (never a sqlglot fallback); neither emits a fallback audit.
        assert _validate_sql("DESCRIBE HISTORY t") is None
        assert _validate_sql("SHOW TAGS") is None
        err = _validate_sql("DESCRIBE 123")
        assert err["validation_failure"] == "unparseable_sql"
        assert audits == []


def test_metadata_limit_and_window_lints_still_run_before_parsing():
    assert (
        _validate_sql("DESCRIBE TABLE system.access.audit LIMIT 10")["validation_failure"]
        == "invalid_metadata_limit"
    )
    assert (
        _validate_sql(
            "SELECT id FROM source WHERE ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts) = 1"
        )["validation_failure"]
        == "window_function_in_where"
    )


class TestTokenizer:
    def test_literal_containing_comment_marker_stays_a_literal(self):
        kinds = [kind for kind, _ in dt._tokenize_sql("SELECT 'a -- b'; DROP TABLE t")]
        assert kinds == ["code", "literal", "code"]

    def test_comment_containing_quote_stays_a_comment(self):
        chunks = dt._tokenize_sql("SELECT 1 -- it's\n; DROP TABLE t")
        assert ("comment", "-- it's") in chunks
        assert dt._sql_code_only("SELECT 1 -- it's\n; DROP TABLE t") == "SELECT 1  \n; DROP TABLE t"

    def test_raw_string_has_no_escapes(self):
        chunks = dt._tokenize_sql(r"SELECT r'\' AS x")
        assert ("literal", r"r'\'") in chunks

    def test_backtick_identifier_and_double_quotes(self):
        code = dt._sql_code_only('SELECT `a;b`, "x -- y" FROM t')
        assert code == "SELECT '', '' FROM t"

    def test_unterminated_block_comment_runs_to_end(self):
        assert dt._sql_without_comments("SELECT 1 /* never ends ; DROP") == "SELECT 1  "


class TestHasMultipleStatementsScanner:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 'a -- b'",
            "SELECT 'a;b'; ",
            "SELECT 1; /* trailing */",
            "SELECT 1; -- trailing",
            "SELECT \"a;b\"",
            "SELECT `a;b`",
        ],
    )
    def test_single(self, sql):
        assert dt._has_multiple_statements(sql) is False

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 'a -- b'; DROP TABLE t",
            "SELECT 1; 'literal'",
            "SELECT 1 -- it's\n; DROP TABLE t",
            "SELECT 1 /* x */; SELECT 2",
        ],
    )
    def test_multiple(self, sql):
        assert dt._has_multiple_statements(sql) is True


def test_synthetic_tools_import_contract():
    # evals/synthetic_tools.py imports these two names and relies on their shapes.
    assert _validate_sql("SELECT 1") is None
    rejected = _validate_sql("DROP TABLE t")
    assert set(rejected) >= {"error", "validation_failure"}
    assert _repair_metadata_limit("DESCRIBE t LIMIT 5") == ("DESCRIBE t", True)
