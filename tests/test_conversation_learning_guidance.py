"""Regression checks derived from reviewed conversations (fictional example data)."""

from utils.system_prompt import load_system_prompt


def test_missing_year_and_relative_date_defaults():
    prompt = load_system_prompt()

    assert "Default a missing year; do not ask for it" in prompt
    assert '"Last week" means the previous completed Monday–Sunday calendar week' in prompt
    assert "the only missing detail is a year" in prompt


def test_dated_launch_is_a_known_breakpoint():
    prompt = load_system_prompt()

    assert "2026-08-03 — `identification_method = 'anonymous'` begins" in prompt
    assert "**not** equivalent to non-autofill signups" in prompt
    assert "year-to-date, or month-over-month" in prompt


def test_schema_and_reporting_parity_are_required_before_analysis():
    prompt = load_system_prompt()

    assert "Inspect schema before the first analytical query" in prompt
    assert "After `UNRESOLVED_COLUMN`, `FIELD_NOT_FOUND`, `TABLE_OR_VIEW_NOT_FOUND`" in prompt
    assert "Preserve published metric semantics" in prompt
    assert "source table, grain, dedup key, date column, and filters exactly" in prompt


def test_table_explanations_separate_build_aliases_from_persisted_columns():
    prompt = load_system_prompt()

    assert "Build a persisted-column inventory from the live metadata" in prompt
    assert "build-time alias, not a queryable target column" in prompt
    assert "Never reference such a name against the final table" in prompt
    assert "fields used only in `WHERE`, `CASE`, `JOIN`, `GROUP BY`, and `ORDER BY`" in prompt


def test_signup_grain_and_masking_guidance():
    prompt = load_system_prompt()

    assert "not necessarily one real person" in prompt
    assert "proves only that the queried protected column" in prompt
    assert "separately persisted or derived columns in the same table" in prompt


def test_learned_identifier_and_metric_gotchas_are_preserved():
    prompt = load_system_prompt()

    assert "can fail with `CAST_INVALID_INPUT`" in prompt
    assert "Do not assume `partner_ref_id` is an explodable string" in prompt
    assert "`pre_order_ind = 1` when `order_date_differential < 0`" in prompt
    assert "Do not reverse the sign from the column name" in prompt
