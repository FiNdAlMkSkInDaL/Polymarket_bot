"""
Market scoring engine — computes a 0-100 composite quality score for each
market based on volume, liquidity, spread, time-to-resolution, price range,
trade frequency, and whale interest.

Higher-scoring markets offer better trading opportunities (tighter spreads,
deeper books, more activity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


# ── Score weights (must sum to 1.0) ────────────────────────────────────────
_WEIGHTS = {
    "volume":          0.20,
    "liquidity":       0.20,
    "spread":          0.20,
    "time_to_resolve": 0.15,
    "price_range":     0.10,
    "trade_freq":      0.10,
    "whale_interest":  0.05,
}


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a market's composite score."""

    volume: float = 0.0
    liquidity: float = 0.0
    spread: float = 0.0
    time_to_resolve: float = 0.0
    price_range: float = 0.0
    trade_freq: float = 0.0
    whale_interest: float = 0.0
    mti_penalty: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        return {
            "vol": round(self.volume, 1),
            "liq": round(self.liquidity, 1),
            "sprd": round(self.spread, 1),
            "ttr": round(self.time_to_resolve, 1),
            "prng": round(self.price_range, 1),
            "freq": round(self.trade_freq, 1),
            "whale": round(self.whale_interest, 1),
            "mti": round(self.mti_penalty, 1),
            "total": round(self.total, 1),
        }


def score_volume(daily_volume_usd: float) -> float:
    """Log-scaled volume score.  $500→20,  $50k→50,  $500k→80,  $1M+→100."""
    if daily_volume_usd <= 0:
        return 0.0
    log_vol = math.log10(max(daily_volume_usd, 1))
    # Map log10(500)=2.7 → 20,  log10(1_000_000)=6 → 100
    score = (log_vol - 2.0) * 25.0
    return max(0.0, min(100.0, score))


def score_liquidity(liquidity_usd: float) -> float:
    """Log-scaled liquidity depth.  $1k→10,  $10k→30,  $100k→70,  $500k+→100."""
    if liquidity_usd <= 0:
        return 0.0
    log_liq = math.log10(max(liquidity_usd, 1))
    score = (log_liq - 2.0) * 25.0
    return max(0.0, min(100.0, score))


def score_spread(spread_cents: float) -> float:
    """Tighter spread → higher score.  ≤1¢→100,  2¢→90,  5¢→60,  ≥10¢→0."""
    if spread_cents <= 0:
        # No book data yet — neutral score (will be re-scored once live data arrives)
        return 50.0
    if spread_cents <= 1.0:
        return 100.0
    if spread_cents >= 10.0:
        return 0.0
    return max(0.0, 100.0 - (spread_cents - 1.0) * (100.0 / 9.0))


def score_time_to_resolution(end_date: datetime | None) -> float:
    """Optimal: 7-60 days.  Too close (<3d) or too far (>180d) → penalised."""
    if end_date is None:
        return 60.0  # perpetual / unknown — decent but not peak

    days = (end_date - datetime.now(timezone.utc)).days
    if days < 0:
        return 0.0  # already expired
    if days < 3:
        return 10.0
    if days < 7:
        return 30.0 + (days - 3) * 10.0  # 30–70
    if days <= 60:
        return 100.0
    if days <= 180:
        return 100.0 - (days - 60) * (50.0 / 120.0)  # 100→50
    return 40.0  # very far out


def score_price_range(mid_price: float) -> float:
    """YES mid-price ∈ [0.15, 0.85] → good edge.  Near 0 or 1 → no edge."""
    if mid_price <= 0 or mid_price >= 1.0:
        return 50.0  # no data yet
    if 0.15 <= mid_price <= 0.85:
        return 100.0
    if mid_price < 0.15:
        return max(0.0, mid_price / 0.15 * 100.0)
    # mid_price > 0.85
    return max(0.0, (1.0 - mid_price) / 0.15 * 100.0)


def score_trade_frequency(trades_per_minute: float) -> float:
    """More trades → higher score.  0→0,  1→40,  5→80,  10+→100."""
    if trades_per_minute <= 0:
        return 0.0
    score = trades_per_minute * 10.0
    return min(100.0, score)


def score_whale_interest(has_whale_activity: bool) -> float:
    """Binary bonus for whale presence."""
    return 100.0 if has_whale_activity else 0.0


def compute_mti_penalty(taker_count: int, total_count: int) -> float:
    """Compute the Maker/Taker Imbalance penalty.

    If > threshold% of trades are taker-initiated (aggressive), the market
    is penalised — high taker ratios indicate toxic/informed flow.

    Returns the penalty in points (0 or ``mti_penalty_points``).
    """
    if total_count <= 0:
        return 0.0
    mti = taker_count / total_count
    threshold = settings.strategy.mti_threshold
    if mti > threshold:
        return settings.strategy.mti_penalty_points
    return 0.0


def compute_score(
    *,
    daily_volume_usd: float = 0.0,
    liquidity_usd: float = 0.0,
    spread_cents: float = 0.0,
    end_date: datetime | None = None,
    mid_price: float = 0.5,
    trades_per_minute: float = 0.0,
    has_whale_activity: bool = False,
    taker_count: int = 0,
    total_count: int = 0,
) -> ScoreBreakdown:
    """Compute the composite 0-100 quality score for a market.

    Returns a ScoreBreakdown with individual component scores and total.
    The MTI (Maker/Taker Imbalance) penalty is subtracted from the
    weighted sum when taker flow exceeds the configured threshold.
    """
    penalty = compute_mti_penalty(taker_count, total_count)

    bd = ScoreBreakdown(
        volume=score_volume(daily_volume_usd),
        liquidity=score_liquidity(liquidity_usd),
        spread=score_spread(spread_cents),
        time_to_resolve=score_time_to_resolution(end_date),
        price_range=score_price_range(mid_price),
        trade_freq=score_trade_frequency(trades_per_minute),
        whale_interest=score_whale_interest(has_whale_activity),
        mti_penalty=penalty,
    )

    weighted_sum = (
        bd.volume * _WEIGHTS["volume"]
        + bd.liquidity * _WEIGHTS["liquidity"]
        + bd.spread * _WEIGHTS["spread"]
        + bd.time_to_resolve * _WEIGHTS["time_to_resolve"]
        + bd.price_range * _WEIGHTS["price_range"]
        + bd.trade_freq * _WEIGHTS["trade_freq"]
        + bd.whale_interest * _WEIGHTS["whale_interest"]
    )

    bd.total = max(0.0, min(100.0, weighted_sum - penalty))

    return bd
