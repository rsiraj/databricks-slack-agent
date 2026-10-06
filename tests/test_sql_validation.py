"""Defense-in-depth SQL guardrails — tools/databricks_tool.py.

These checks run *outside* the LLM and are the last line of defense if a
prompt-level jailbreak tries to issue a destructive or multi-statement query.
A regression here is security-relevant and would pass the Docker build check
silently, so it is exactly what this suite exists to catch.
"""

import pytest

from tools import databricks_tool as dt


class TestFirstSqlKeyword:
    @pytest.mark.parametrize(
        "sql,expected",
        [
            ("SELECT 1", "SELECT"),
            ("  select 1", "SELECT"),
            ("\n\t WITH x AS (SELECT 1) SELECT * FROM x", "WITH"),
            ("-- a comment\nSELECT 1", "SELECT"),
            ("/* block */ SELECT 1", "SELECT"),
            ("/* multi\nline */\nDESCRIBE t", "DESCRIBE"),
            ("", ""),
            ("   ", ""),
            ("123 SELECT", ""),  # leading non-alpha => no keyword
        ],
    )
    def test_first_keyword(self, sql, expected):
        assert dt._first_sql_keyword(sql) == expected


class TestStripLeadingComments:
    def test_unterminated_block_comment_returns_empty(self):
        assert dt._strip_leading_sql_comments_and_whitespace("/* never ends") == ""

    def test_line_comment_without_newline_returns_empty(self):
        assert dt._strip_leading_sql_comments_and_whitespace("-- just a comment") == ""

    def test_mixed_leading_comments_stripped(self):
        assert (
            dt._strip_leading_sql_comments_and_whitespace("-- a\n  /* b */ SELECT 1")
            == "SELECT 1"
        )

    def test_plain_whitespace_stripped(self):
        assert dt._strip_leading_sql_comments_and_whitespace("  \n SELECT 1") == "SELECT 1"


class TestHasMultipleStatements:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "SELECT 1;",  # trailing ; only
            "SELECT 1;   ",  # trailing ; + whitespace
            "SELECT 'a;b' FROM t",  # ; inside string literal
            "SELECT 1 -- ; not a statement",  # ; inside line comment
            "SELECT 1 /* ; */ FROM t",  # ; inside block comment
            "SELECT `a;b` FROM t",  # ; inside backtick identifier
        ],
    )
    def test_single_statement(self, sql):
        assert dt._has_multiple_statements(sql) is False

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DROP TABLE t",
            "SELECT 1; SELECT 2",
            "WITH x AS (SELECT 1) SELECT * FROM x; DELETE FROM y",
        ],
    )
    def test_multiple_statements(self, sql):
        assert dt._has_multiple_statements(sql) is True


class TestValidateSql:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM t",
            "select * from t",
            "WITH x AS (SELECT 1) SELECT * FROM x",
            "SHOW TABLES",
            "DESCRIBE t",
            "DESC t",
            "EXPLAIN SELECT 1",
            "  -- header comment\n  SELECT 1",
        ],
    )
    def test_allows_read_only(self, sql):
        assert dt._validate_sql(sql) is None

    @pytest.mark.parametrize(
        "sql,keyword",
        [
            ("DELETE FROM t", "DELETE"),
            ("DROP TABLE t", "DROP"),
            ("INSERT INTO t VALUES (1)", "INSERT"),
            ("UPDATE t SET a = 1", "UPDATE"),
            ("MERGE INTO t USING s ON t.id = s.id", "MERGE"),
            ("ALTER TABLE t ADD COLUMN c INT", "ALTER"),
            ("CREATE TABLE t (a INT)", "CREATE"),
            ("TRUNCATE TABLE t", "TRUNCATE"),
            ("GRANT SELECT ON t TO u", "GRANT"),
            ("USE catalog", "USE"),
            ("SET x = 1", "SET"),
        ],
    )
    def test_rejects_disallowed_statement_types(self, sql, keyword):
        err = dt._validate_sql(sql)
        assert err is not None
        assert err["validation_failure"] == "disallowed_statement_type"
        assert err["first_keyword"] == keyword

    @pytest.mark.parametrize("sql", ["", "   ", "\n\t"])
    def test_rejects_empty(self, sql):
        err = dt._validate_sql(sql)
        assert err is not None
        assert err["validation_failure"] == "empty"

    def test_rejects_multiple_statements(self):
        err = dt._validate_sql("SELECT 1; DROP TABLE t")
        assert err is not None
        assert err["validation_failure"] == "multiple_statements"

    def test_statement_type_checked_before_multiple(self):
        # A disallowed leading keyword is rejected on type, not on the trailing ;.
        err = dt._validate_sql("DROP TABLE t; SELECT 1")
        assert err["validation_failure"] == "disallowed_statement_type"


class TestSummarizeResult:
    @staticmethod
    def _payload(row_count, rows, truncated=False):
        return {
            "statement_id": "stmt-1",
            "manifest": {
                "total_row_count": row_count,
                "truncated": truncated,
                "schema": {"columns": [{"name": "value"}]},
            },
            "result": {"data_array": rows},
        }

    def test_marks_first_fifty_rows_as_sample(self):
        summary = dt._summarize_result(
            self._payload(75, [[value] for value in range(75)])
        )

        assert summary["is_sample"] is True
        assert summary["sample_row_count"] == 50
        assert "first 50 of 75 rows" in summary["result_scope"]

    def test_truncation_warning_for_server_capped_result(self):
        summary = dt._summarize_result(
            self._payload(1000, [[value] for value in range(1000)], truncated=True)
        )

        assert summary["truncated"] is True
        assert summary["is_sample"] is True
        assert "disclose the truncation" in summary["result_scope"]

    def test_complete_small_result_is_not_sampled(self):
        summary = dt._summarize_result(self._payload(2, [[1], [2]]))

        assert summary["is_sample"] is False
        assert summary["result_scope"] == "sample_rows contains all 2 returned rows."
