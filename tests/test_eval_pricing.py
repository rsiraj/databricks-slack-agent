"""Pricing table math used by the evaluation harness and the usage report."""

from evals.pricing import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICING_USD_PER_MTOK,
    estimate_cost_from_usage,
    estimate_cost_usd,
    resolve_pricing,
)


def test_seeded_models_have_list_prices():
    assert PRICING_USD_PER_MTOK["claude-sonnet-4-6"] == {"input": 3.0, "output": 15.0}
    assert PRICING_USD_PER_MTOK["claude-sonnet-5"] == {"input": 2.0, "output": 10.0}
    assert PRICING_USD_PER_MTOK["claude-opus-5"] == {"input": 5.0, "output": 25.0}
    assert PRICING_USD_PER_MTOK["claude-opus-4-8"] == {"input": 5.0, "output": 25.0}
    assert PRICING_USD_PER_MTOK["claude-haiku-4-5"] == {"input": 1.0, "output": 5.0}
    assert CACHE_WRITE_MULTIPLIER == 1.25
    assert CACHE_READ_MULTIPLIER == 0.10


def test_cost_applies_cache_multipliers():
    # 1M uncached input at $3, 1M cache write at $3.75, 1M cache read at $0.30,
    # 1M output at $15 -> $22.05 total.
    cost = estimate_cost_usd(
        "claude-sonnet-4-6",
        input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == 22.05


def test_unknown_model_returns_none_without_raising():
    assert estimate_cost_usd("claude-experimental-9", input_tokens=10) is None
    assert estimate_cost_usd(None, input_tokens=10) is None
    assert estimate_cost_usd("", input_tokens=10) is None
    assert resolve_pricing("nope") is None


def test_dated_snapshot_ids_resolve_to_base_pricing():
    assert resolve_pricing("claude-sonnet-4-6-20260115") == PRICING_USD_PER_MTOK["claude-sonnet-4-6"]
    assert resolve_pricing("CLAUDE-OPUS-5") == PRICING_USD_PER_MTOK["claude-opus-5"]


def test_openai_row_uses_explicit_cached_price_and_no_cache_write_premium():
    row = PRICING_USD_PER_MTOK["gpt-5.6-terra"]
    assert row["input"] == 2.0
    assert row["cached_input"] == 0.20
    assert row["output"] == 12.0
    assert row["cache_write_multiplier"] == 1.0
    # 1M uncached input at $2, 1M "cache write" at plain input price $2 (no
    # premium), 1M cached input at $0.20, 1M output at $12 -> $16.20.
    cost = estimate_cost_usd(
        "gpt-5.6-terra",
        input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == 16.20
    assert resolve_pricing("gpt-5.6-terra-2026-08-01") is None  # only Anthropic-style -YYYYMMDD suffixes strip
    assert resolve_pricing("GPT-5.6-TERRA") == row


def test_anthropic_rows_keep_multiplier_pricing_when_no_override():
    # Regression guard: adding optional keys for OpenAI must not change Claude math.
    assert estimate_cost_usd("claude-sonnet-4-6", cache_read_input_tokens=1_000_000) == 0.30
    assert estimate_cost_usd("claude-sonnet-4-6", cache_creation_input_tokens=1_000_000) == 3.75


def test_usage_dict_wrapper_tolerates_missing_and_none_fields():
    assert estimate_cost_from_usage("claude-haiku-4-5", {"output_tokens": 1_000_000}) == 5.0
    assert estimate_cost_from_usage(
        "claude-haiku-4-5", {"input_tokens": None, "output_tokens": 2_000_000}
    ) == 10.0
    assert estimate_cost_from_usage("claude-haiku-4-5", None) == 0.0
