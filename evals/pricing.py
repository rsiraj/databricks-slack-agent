"""List pricing used to estimate evaluation and production spend.

IMPORTANT: these numbers are a convenience snapshot for *relative* model and
effort comparisons. Re-verify every row against the official pricing page
(Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing; OpenAI:
https://openai.com/api/pricing) before quoting a dollar figure to anyone, and
add a row whenever a new model id is trialled. Unknown model ids deliberately
produce ``None`` instead of a guess.

Prices are USD per million tokens (MTok). Anthropic cache pricing follows the
standard multipliers: cache writes cost 1.25x the input price, cache reads
cost 0.1x. Rows may override either with an explicit ``cached_input`` price
(cache reads) or ``cache_write_multiplier`` — OpenAI's automatic caching has a
flat cached-input price and no write charge, so its row sets both.
"""

from __future__ import annotations

import re
from typing import Any

# Snapshot taken 2026-09-04. Re-verify before use (see module docstring).
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # OpenAI Responses API; cached input is priced explicitly, no cache-write charge.
    "gpt-5.6-terra": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "cache_write_multiplier": 1.0,
    },
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def resolve_pricing(model: str | None) -> dict[str, float] | None:
    """Return the pricing row for ``model`` (tolerating dated snapshot suffixes)."""
    if not model:
        return None
    candidate = str(model).strip().lower()
    if candidate in PRICING_USD_PER_MTOK:
        return PRICING_USD_PER_MTOK[candidate]
    undated = _DATE_SUFFIX_RE.sub("", candidate)
    return PRICING_USD_PER_MTOK.get(undated)


def estimate_cost_usd(
    model: str | None,
    *,
    input_tokens: int | None = 0,
    cache_creation_input_tokens: int | None = 0,
    cache_read_input_tokens: int | None = 0,
    output_tokens: int | None = 0,
) -> float | None:
    """Estimate USD cost for one call or one aggregate; ``None`` for unknown models."""
    pricing = resolve_pricing(model)
    if pricing is None:
        return None
    input_price = pricing["input"] / 1_000_000
    output_price = pricing["output"] / 1_000_000
    cache_write_price = input_price * pricing.get("cache_write_multiplier", CACHE_WRITE_MULTIPLIER)
    if "cached_input" in pricing:
        cache_read_price = pricing["cached_input"] / 1_000_000
    else:
        cache_read_price = input_price * CACHE_READ_MULTIPLIER
    cost = (
        (input_tokens or 0) * input_price
        + (cache_creation_input_tokens or 0) * cache_write_price
        + (cache_read_input_tokens or 0) * cache_read_price
        + (output_tokens or 0) * output_price
    )
    return round(cost, 6)


def estimate_cost_from_usage(model: str | None, usage: dict[str, Any] | None) -> float | None:
    """Convenience wrapper for a usage dict shaped like the Messages API ``usage``."""
    usage = usage or {}
    return estimate_cost_usd(
        model,
        input_tokens=int(usage.get("input_tokens") or 0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )
