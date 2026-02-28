"""
Spread score calculator — computes a real-time 0-100 quality score from
live L2 order book data.

The score incorporates both raw BBO spread and depth-weighted spread
across the top N levels, giving a more accurate picture of actual
execution cost than raw spread alone.

Used by:
  - ``L2OrderBook``  — recomputed on every BBO change
  - ``market_scorer`` — as ``live_spread_score`` override
  - ``MarketLifecycleManager`` — for real-time promotion/demotion decisions
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class SpreadScore:
    """Point-in-time spread quality assessment."""

    raw_spread_cents: float = 0.0
    depth_weighted_spread_cents: float = 0.0
    score: float = 0.0           # 0-100 composite
    timestamp: float = 0.0


def compute_spread_score(
    best_bid: float,
    best_ask: float,
    bid_levels: list[tuple[float, float]],   # [(price, size), ...]
    ask_levels: list[tuple[float, float]],
    top_n: int = 3,
    timestamp: float | None = None,
) -> SpreadScore:
    """Compute a spread score from L2 book state.

    Parameters
    ----------
    best_bid, best_ask:
        Top-of-book prices.
    bid_levels, ask_levels:
        Up to *top_n* levels as ``(price, size)`` tuples, sorted
        best-to-worst (bids descending, asks ascending).
    top_n:
        Number of levels to include in the depth-weighted calculation.

    Returns
    -------
    SpreadScore
        Contains raw spread, depth-weighted spread and a 0-100 score.
    """
    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return SpreadScore(timestamp=(timestamp or time.time()))

    raw_spread = best_ask - best_bid
    raw_spread_cents = round(raw_spread * 100, 4)

    # ── Depth-weighted spread across top N levels ──────────────────────
    # For each level pair (bid_i, ask_i) compute the spread weighted by
    # the *minimum* of available size at that depth (the size you could
    # actually trade through).
    dw_numer = 0.0
    dw_denom = 0.0

    bids = bid_levels[:top_n]
    asks = ask_levels[:top_n]
    n_pairs = min(len(bids), len(asks))

    if n_pairs > 0:
        for i in range(n_pairs):
            bid_price, bid_size = bids[i]
            ask_price, ask_size = asks[i]
            if bid_price <= 0 or ask_price <= 0:
                continue
            spread_i = ask_price - bid_price
            weight = min(bid_size, ask_size)
            if weight > 0:
                dw_numer += spread_i * weight
                dw_denom += weight

    if dw_denom > 0:
        dw_spread_cents = round((dw_numer / dw_denom) * 100, 4)
    else:
        dw_spread_cents = raw_spread_cents

    # ── Score: combine raw + depth-weighted (60/40 blend) ──────────────
    # Use the blended spread for scoring.
    blended_cents = 0.6 * raw_spread_cents + 0.4 * dw_spread_cents
    score = _spread_cents_to_score(blended_cents)

    return SpreadScore(
        raw_spread_cents=raw_spread_cents,
        depth_weighted_spread_cents=dw_spread_cents,
        score=round(score, 2),
        timestamp=(timestamp or time.time()),
    )


def _spread_cents_to_score(spread_cents: float) -> float:
    """Map spread in cents to a 0-100 score.

    Identical curve to ``market_scorer.score_spread`` so scores are
    comparable across the old and new pipelines::

        ≤1¢  → 100
        2¢   →  ~89
        5¢   →  ~56
        ≥10¢ →    0
    """
    if spread_cents <= 0:
        return 50.0  # no data
    if spread_cents <= 1.0:
        return 100.0
    if spread_cents >= 10.0:
        return 0.0
    return max(0.0, 100.0 - (spread_cents - 1.0) * (100.0 / 9.0))
