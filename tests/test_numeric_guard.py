"""Unit tests for the numeric grounding guard."""

import json

import pytest

from utils import numeric_guard as ng
from utils.numeric_guard import NumericEvidenceGuard


def _texts(unverified):
    return [item["text"] for item in unverified]


def _guard_with(result, tool_name="databricks_execute_sql", tool_input=None):
    guard = NumericEvidenceGuard()
    guard.record_tool_result(tool_name, tool_input or {}, result)
    return guard


class TestGrounding:
    def test_numbers_from_tool_result_fields_are_grounded(self):
        guard = _guard_with(
            {"row_count": 1, "columns": ["n"], "sample_rows": [[1234, 56.78]]}
        )
        assert guard.validate("There are 1,234 rows and the value is 56.78.") == []

    def test_ungrounded_number_is_reported_with_context(self):
        guard = _guard_with({"row_count": 1, "sample_rows": [[1000]]})
        unverified = guard.validate("Revenue was 1,000 last month and grew by 234 units.")
        assert len(unverified) == 1
        item = unverified[0]
        assert item["text"] == "234"
        assert item["value"] == "234"
        assert "grew by 234 units" in item["context"]
        assert len(item["context"]) <= ng.CONTEXT_CHARS + 6

    def test_user_text_numbers_are_grounded(self):
        guard = NumericEvidenceGuard()
        guard.record_user_text("Compare 1500 signups against the 1,200 from last week")
        assert guard.validate("You mentioned 1,500 and 1200.") == []

    def test_operational_constants_are_grounded_by_default(self):
        guard = NumericEvidenceGuard()
        assert guard.validate("The cap is 1,000 rows; memory lasts 365 days.") == []
        assert guard.validate("Replies are chunked at 3,500 characters; 60,000-char file cap.") == []
        assert guard.validate("Limits: 60/hour, 200/day per user, 500 SQL/day, 25 iterations.") == []

    def test_constants_can_be_disabled_or_overridden(self, monkeypatch):
        assert _texts(NumericEvidenceGuard(constants={}).validate("Capped at 1,000 rows.")) == [
            "1,000"
        ]
        monkeypatch.setenv("DATABRICKS_ROW_LIMIT", "250")
        assert ng.reference_constants()["row_limit"] == 250
        guard = NumericEvidenceGuard()
        assert guard.validate("Capped at 250 rows.") == []
        assert _texts(guard.validate("Capped at 1,000 rows.")) == ["1,000"]

    def test_reference_constants_mirror_their_source_modules(self, monkeypatch):
        import worker
        from tools.databricks_tool import DEFAULT_ROW_LIMIT
        from utils.feedback import FEEDBACK_RETENTION_DAYS
        from utils.sessions import SESSION_RETENTION_DAYS
        from utils.slack_formatting import SLACK_MESSAGE_MAX_CHARS
        from utils.slack_helpers import MAX_IMAGE_BYTES

        for name in (
            "DATABRICKS_ROW_LIMIT",
            "RATELIMIT_USER_PER_HOUR",
            "RATELIMIT_USER_PER_DAY",
            "RATELIMIT_CHANNEL_PER_HOUR",
            "RATELIMIT_SQL_PER_DAY",
        ):
            monkeypatch.delenv(name, raising=False)
        constants = ng.reference_constants()
        assert constants["row_limit"] == DEFAULT_ROW_LIMIT == 1000
        assert constants["max_iterations"] == worker.MAX_ITERATIONS
        assert constants["slack_chunk_chars"] == SLACK_MESSAGE_MAX_CHARS
        assert constants["retention_days"] == SESSION_RETENTION_DAYS == FEEDBACK_RETENTION_DAYS
        assert constants["image_cap_bytes"] == MAX_IMAGE_BYTES
        assert constants["github_read_cap_chars"] == 60_000
        assert (
            constants["ratelimit_user_per_hour"],
            constants["ratelimit_user_per_day"],
            constants["ratelimit_channel_per_hour"],
            constants["ratelimit_sql_per_day"],
        ) == (60, 200, 200, 500)

    def test_system_prompt_is_not_grounded(self):
        from utils.system_prompt import load_system_prompt

        # The live bug: the known-data-events prompt says "~-36%", which used to
        # vouch for a coincidental "36 columns".
        assert "-36%" in load_system_prompt()
        guard = NumericEvidenceGuard()
        guard.record_user_text("How many columns does the table have?")
        assert _texts(guard.validate("The table has 36 columns.")) == ["36"]
        assert not hasattr(guard, "record_reference_text")

    def test_tool_input_numbers_are_grounded(self):
        guard = _guard_with(
            {"row_count": 0, "sample_rows": []},
            tool_input={"sql": "SELECT * FROM t WHERE d >= CURRENT_DATE - INTERVAL 30 DAYS LIMIT 250"},
        )
        assert guard.validate("I looked at the last 30 days and capped at 250 rows.") == []

    def test_json_string_tool_results_are_decoded(self):
        payload = json.dumps({"runs": [{"duration_ms": 3960000, "attempt": 2}]})
        guard = _guard_with({"raw": payload})
        assert guard.validate("The duration was 3,960,000 ms.") == []

    def test_free_text_tool_results_are_scanned(self):
        guard = _guard_with({"content": "median duration 4,512 seconds across 37 runs"})
        assert guard.validate("Median 4512 s over 37 runs.") == []

    def test_calculate_result_and_display_are_grounded(self):
        guard = _guard_with(
            {
                "expression": "pct_change(old, new)",
                "variables": {"old": 1000, "new": 1234},
                "result": 23.4,
                "result_display": "23.40",
                "steps": ["pct_change(1000, 1234)"],
            },
            tool_name="calculate",
        )
        assert guard.validate("That is a 23.4% increase (via calculate), 23.40 exactly.") == []

    def test_failed_calculate_result_does_not_ground_anything_new(self):
        guard = _guard_with(
            {"error": "Division by zero", "validation_failure": "division_by_zero"},
            tool_name="calculate",
            tool_input={"expression": "1/0"},
        )
        assert _texts(guard.validate("So the ratio is 57.")) == ["57"]

    def test_record_history_grounds_prior_tool_results_and_user_text_only(self):
        guard = NumericEvidenceGuard()
        guard.record_history(
            [
                {"role": "user", "content": "What about 4,400 signups?"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Roughly 9,999 in total."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": json.dumps({"row_count": 1, "sample_rows": [[8765]]}),
                        }
                    ],
                },
                {"role": "assistant", "content": "8,765 rows."},
            ]
        )
        assert guard.validate("Earlier: 4,400 requested, 8,765 found.") == []
        assert _texts(guard.validate("As I said, 9,999 in total.")) == ["9,999"]

    def test_record_history_never_grounds_numbers_from_provider_reasoning_payloads(self):
        guard = NumericEvidenceGuard()
        guard.record_history(
            [
                {"role": "user", "content": "How many?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "provider_reasoning",
                            "provider": "openai",
                            "payload": {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [{"type": "summary_text", "text": "Probably 31,337 rows."}],
                                "encrypted_content": "ZZZ 424242",
                            },
                        },
                        {"type": "text", "text": "About 31,337."},
                    ],
                },
            ]
        )
        assert _texts(guard.validate("There are 31,337 rows and 424,242 events.")) == [
            "31,337",
            "424,242",
        ]


class TestMatchingRules:
    def test_rounding_to_displayed_precision(self):
        guard = _guard_with({"avg": 12.3456})
        assert guard.validate("The average is 12.35.") == []
        assert guard.validate("The average is 12.3.") == []
        assert guard.validate("The average is about 12.") == []  # small int rule aside
        assert _texts(guard.validate("The average is 12.36.")) == ["12.36"]

    def test_integer_rounding_of_a_decimal_tool_value(self):
        guard = _guard_with({"minutes": 66.4})
        assert guard.validate("It took 66 minutes.") == []

    def test_percentage_matches_fraction_times_100(self):
        guard = _guard_with({"share": 0.4512})
        assert guard.validate("That is 45.12% of traffic, roughly 45%.") == []

    def test_percentage_divided_by_100_needs_percent_context(self):
        # A plain integer is never rescaled into a percentage ...
        assert _texts(_guard_with({"bps": 4512}).validate("That is 45.12%.")) == ["45.12%"]
        # ... unless its key says it is one, or a non-integer tool value backs it
        # (4512.0 is integer-valued and still does not count).
        assert _guard_with({"rate_bps": 4512}).validate("That is 45.12%.") == []
        assert _guard_with({"bps": 4511.7}).validate("That is 45.12%.") == []
        assert _texts(_guard_with({"bps": 4512.0}).validate("That is 45.12%.")) == ["45.12%"]

    def test_percent_word_counts_as_percentage(self):
        guard = _guard_with({"share": 0.25})
        assert guard.validate("About 25 percent of users.") == []

    def test_ungrounded_percentage_is_flagged(self):
        guard = _guard_with({"old": 1000, "new": 1234})
        assert _texts(guard.validate("A 23.4% increase.")) == ["23.4%"]

    def test_k_and_m_suffixes(self):
        guard = _guard_with({"users": 1234, "events": 2_500_000, "spend": 3_000_000_000})
        assert guard.validate("~1.2k users, 2.5M events, 3B spend.") == []
        assert _texts(guard.validate("1.5M events")) == ["1.5M"]

    def test_suffix_is_case_sensitive(self):
        guard = _guard_with({"n": 5000})
        assert guard.validate("5k rows") == []
        # 5K is treated as an identifier-like token rather than a claim.
        assert guard.validate("5K rows") == []
        assert _texts(guard.validate("5m rows")) == []

    def test_currency_and_thousands_separators(self):
        guard = _guard_with({"spend": 1234.6})
        assert guard.validate("Spend was $1,234.60, about $1,235.") == []
        assert _texts(guard.validate("Spend was $1,234.")) == ["$1,234"]

    def test_half_even_and_half_up_rounding_are_both_accepted(self):
        guard = _guard_with({"spend": 1234.5})
        assert guard.validate("About 1,234 or 1,235.") == []

    def test_sign_is_ignored_for_matching(self):
        guard = _guard_with({"delta": -12.5})
        assert guard.validate("It dropped 12.5%, i.e. -12.5.") == []

    def test_repeated_unverified_figure_reported_once(self):
        guard = NumericEvidenceGuard()
        unverified = guard.validate("It was 456 then, and 456 again, plus 789.")
        assert _texts(unverified) == ["456", "789"]


class TestPositionalFieldsAndPercentContext:
    def test_ordinal_position_does_not_ground_a_whole_percent(self):
        guard = _guard_with(
            {
                "column_count": 2,
                "columns": [
                    {"name": "a", "type": "string", "ordinal_position": 12},
                    {"name": "b", "type": "int", "ordinal_position": 13},
                ],
            },
            tool_name="describe_object",
        )
        assert _texts(guard.validate("About 12% of rows, and 13 columns.")) == ["12%", "13"]

    @pytest.mark.parametrize(
        "result",
        [
            {"pct_change": 12.0},
            {"pct_change": 12},
            {"percent_of_total": 12},
            {"conversion_rate": 12.0},
            {"ratio": 12},
            {"share": 12},
            {"summary": "grew 12% week over week"},
            {"summary": "grew 12 percent week over week"},
            {"columns": ["day", "pct_change"], "sample_rows": [["2026-08-01", 12.0]]},
            {"fraction": 0.12},
            {"fraction": 0.1234},
        ],
    )
    def test_whole_percent_grounded_by_percent_context_or_decimal(self, result):
        assert _guard_with(result).validate("About 12% of rows.") == []

    @pytest.mark.parametrize(
        "result",
        [
            {"ordinal_position": 12},
            {"n": 12},
            {"columns": ["day", "n"], "sample_rows": [["2026-08-01", 12]]},
            {"delta": 12.4},  # non-integer but not x100 and not percent-keyed
            {"delta": 1200},  # integers are never divided by 100
        ],
    )
    def test_whole_percent_not_grounded_by_plain_numbers(self, result):
        assert _texts(_guard_with(result).validate("About 12% of rows.")) == ["12%"]

    def test_user_percentages_and_calculate_results_are_percent_context(self):
        guard = NumericEvidenceGuard()
        guard.record_user_text("Is it still around 12%?")
        assert guard.validate("Yes, about 12%.") == []

        guard = _guard_with(
            {"expression": "1200 / 100", "result": 12, "result_display": "12"},
            tool_name="calculate",
        )
        assert guard.validate("That is 12% (via calculate).") == []

    def test_decimal_percent_still_matches_same_scale_value(self):
        # A percent claim with decimals is specific enough to match any value at
        # the same scale, integer or not; only whole percents need context.
        assert _guard_with({"n": 45.12}).validate("45.12% of rows.") == []
        assert _guard_with({"n": 45.1234}).validate("45.12% of rows.") == []
        assert _guard_with({"n": 45}).validate("45.0% of rows.") == []
        assert _texts(_guard_with({"n": 45}).validate("45% of rows.")) == ["45%"]

    @pytest.mark.parametrize(
        "key",
        [
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
            "job_id",
            "run_ID",
            "event_ts",
            "row_index",
            "sample_row_count",
            "payload",
            "encrypted_content",
            "provider_reasoning",
        ],
    )
    def test_positional_identifier_and_opaque_keys_are_skipped(self, key):
        guard = _guard_with({key: 4567, "nested": [{key: "value 8910"}]})
        assert _texts(guard.validate("Found 4,567 rows and 8,910 events.")) == ["4,567", "8,910"]

    @pytest.mark.parametrize(
        "key", ["column_count", "row_count", "fetched_run_count", "total_count"]
    )
    def test_legitimate_count_keys_are_kept(self, key):
        assert _guard_with({key: 4567}).validate("Found 4,567.") == []

    def test_describe_object_column_count_is_grounded_but_positions_are_not(self):
        guard = _guard_with(
            {
                "column_count": 35,
                "columns": [
                    {"name": f"c{i}", "type": "string", "ordinal_position": i + 1}
                    for i in range(35)
                ],
            },
            tool_name="describe_object",
        )
        assert guard.validate("The table has 35 columns.") == []
        assert _texts(guard.validate("The table has 36 columns.")) == ["36"]

    def test_provider_reasoning_blocks_are_skipped(self):
        guard = _guard_with(
            {
                "content": [
                    {"type": "provider_reasoning", "signature": "abc", "text": "maybe 4567"},
                    {"type": "text", "text": "returned 8910 rows"},
                ]
            }
        )
        assert _texts(guard.validate("4,567 vs 8,910.")) == ["4,567"]

    def test_skipping_applies_inside_json_strings_and_history(self):
        guard = NumericEvidenceGuard()
        guard.record_history(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": json.dumps({"job_id": 4567, "row_count": 8910}),
                        }
                    ],
                }
            ]
        )
        assert _texts(guard.validate("4,567 vs 8,910.")) == ["4,567"]


class TestExclusions:
    @pytest.mark.parametrize(
        "draft",
        [
            "Data goes back to 2024 and refreshes in 2026.",
            "The window is 2026-08-12 to 2026-08-19.",
            "Started at 2026-08-12T10:45:00Z and 2026-08-12 10:45:30.",
            "It ran at 10:45 and again at 23:59:59.",
            "Runs at 9:30 AM and 4:15 p.m.",
            "Released as v1.2.3 and 10.20.30.",
            "This is Part 2 of 3.",
            "See table_v2, us-east-1, abc123, run_99, col_45_v3.",
            "The message ts is 1723456789.123456.",
            "Job id 1234567890 and run 98765432101234.",
            "Top 5 tables, 3 tables, 12 steps, 0 failures, 7th row, 10x faster.",
            "Between Aug 25, 2026 and September 30 2026, also 25 Aug 2026.",
            "Dated 08/25/2026 and 8/25.",
            "```sql\nSELECT 999 FROM t LIMIT 4567\n```",
            "Inline `LIMIT 777` and `row_45` are code.",
            "The 2,000-row cap is 2000 rows in the year 2000.",
        ],
    )
    def test_non_claims_are_never_flagged(self, draft):
        guard = NumericEvidenceGuard()
        flagged = _texts(guard.validate(draft))
        # The trailing case checks that a four-digit *year* is skipped even when
        # the same digits also appear as a real claim with a thousands separator.
        if draft.startswith("The 2,000-row"):
            assert flagged == ["2,000"]
        else:
            assert flagged == []

    def test_prose_next_to_excluded_tokens_is_still_checked(self):
        guard = NumericEvidenceGuard()
        flagged = _texts(
            guard.validate(
                "On 2026-08-12 at 10:45 the job table_v2 processed 4,567 rows in Part 1 of 2."
            )
        )
        assert flagged == ["4,567"]

    def test_small_integers_with_decimals_or_percent_are_claims(self):
        guard = NumericEvidenceGuard()
        assert _texts(guard.validate("Only 5% converted, 3.5 on average, $7 each.")) == [
            "5%",
            "3.5",
            "$7",
        ]

    def test_year_like_numbers_with_sign_or_currency_are_claims(self):
        guard = NumericEvidenceGuard()
        assert _texts(guard.validate("Cost $2024, change -2025.")) == ["$2024", "-2025"]


class TestOutputs:
    def test_revision_instruction_lists_figures_and_forbids_own_arithmetic(self):
        guard = NumericEvidenceGuard()
        unverified = guard.validate("Up 23.4% to 1,234 units.")
        text = guard.revision_instruction(unverified)
        assert '"23.4%"' in text
        assert '"1,234"' in text
        assert "`calculate`" in text
        assert "result_display" in text
        assert "Do not perform arithmetic yourself" in text
        assert "Do not change any figure that IS grounded" in text
        assert "Do not mention this internal correction request" in text

    def test_annotate_appends_italic_footnote(self):
        guard = NumericEvidenceGuard()
        draft = "Up 23.4% to 1,234 units."
        unverified = guard.validate(draft)
        annotated = guard.annotate(draft, unverified)
        assert annotated.startswith(draft)
        assert annotated.endswith("_Figures not verified against a tool result: 23.4%, 1,234._")
        assert "\n\n_" in annotated

    def test_annotate_without_unverified_returns_draft_unchanged(self):
        assert NumericEvidenceGuard.annotate("Fine.", []) == "Fine."


class TestConfig:
    def test_mode_defaults_and_validation(self, monkeypatch):
        monkeypatch.delenv("NUMERIC_GUARD_MODE", raising=False)
        assert ng.numeric_guard_mode() == "revise"
        monkeypatch.setenv("NUMERIC_GUARD_MODE", "ANNOTATE")
        assert ng.numeric_guard_mode() == "annotate"
        monkeypatch.setenv("NUMERIC_GUARD_MODE", "off")
        assert ng.numeric_guard_mode() == "off"
        monkeypatch.setenv("NUMERIC_GUARD_MODE", "bogus")
        assert ng.numeric_guard_mode() == "revise"

    def test_max_revisions_defaults_and_validation(self, monkeypatch):
        monkeypatch.delenv("NUMERIC_GUARD_MAX_REVISIONS", raising=False)
        assert ng.numeric_guard_max_revisions() == 1
        monkeypatch.setenv("NUMERIC_GUARD_MAX_REVISIONS", "3")
        assert ng.numeric_guard_max_revisions() == 3
        monkeypatch.setenv("NUMERIC_GUARD_MAX_REVISIONS", "0")
        assert ng.numeric_guard_max_revisions() == 0
        monkeypatch.setenv("NUMERIC_GUARD_MAX_REVISIONS", "-2")
        assert ng.numeric_guard_max_revisions() == 1
        monkeypatch.setenv("NUMERIC_GUARD_MAX_REVISIONS", "many")
        assert ng.numeric_guard_max_revisions() == 1
