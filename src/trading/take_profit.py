"""
Dynamic take-profit calculator.

Computes the exit price target using the adaptive α mean-reversion model:

    P_target = P_entry + α · (VWAP_no - P_entry)

where α ∈ [α_min, α_max] is adjusted based on volatility, book depth,
whale confluence, and time to market resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class TakeProfitResult:
    """Computed exit parameters."""

    entry_price: float
    target_price: float
    alpha: float
    spread_cents: float     # target_price - entry_price  (in cents)
    viable: bool            # False if spread < min_spread_cents


def compute_take_profit(
    entry_price: float,
    no_vwap: float,
    *,
    realised_vol: float = 0.0,
    book_depth_ratio: float = 1.0,
    whale_confluence: bool = False,
    days_to_resolution: int = 30,
) -> TakeProfitResult:
    """Calculate the dynamic take-profit target.

    Parameters
    ----------
    entry_price:
        The fill price of the NO buy order (e.g., 0.47).
    no_vwap:
        The rolling 60-min VWAP of the NO token (e.g., 0.65).
    realised_vol:
        Rolling σ of 1-min log returns.  Higher → lower α.
    book_depth_ratio:
        Ratio of resting bid liquidity at the target zone vs average.
        > 1.0 means deeper book → higher α.
    whale_confluence:
        If True, whale confirmation pushes α higher.
    days_to_resolution:
        Days until market resolves.  Closer → lower α.

    Returns
    -------
    TakeProfitResult with the computed target price and metadata.
    """
    strat = settings.strategy

    # Start with default alpha
    alpha = strat.alpha_default

    # ── Adjustments ─────────────────────────────────────────────────────────

    # 1. Volatility adjustment: high vol → capture less (exit sooner)
    #    Benchmark σ ≈ 0.02 for "normal" prediction market 1-min bars.
    if realised_vol > 0:
        vol_factor = min(realised_vol / 0.02, 3.0)  # cap at 3×
        alpha -= 0.05 * (vol_factor - 1.0)

    # 2. Book depth: deeper resting liquidity → can afford to wait
    if book_depth_ratio > 1.0:
        alpha += 0.03 * min(book_depth_ratio - 1.0, 3.0)

    # 3. Whale confluence → conviction bump
    if whale_confluence:
        alpha += 0.08

    # 4. Time decay: closer to resolution → less room for mean reversion
    if days_to_resolution < 14:
        alpha -= 0.05 * (1.0 - days_to_resolution / 14.0)

    # Clamp alpha
    alpha = max(strat.alpha_min, min(strat.alpha_max, alpha))

    # ── Target price ────────────────────────────────────────────────────────
    if no_vwap <= entry_price:
        # Edge case: VWAP has already moved below our entry (rare).
        # Fall back to a minimal fixed spread.
        target = entry_price + strat.min_spread_cents / 100.0
        alpha = strat.alpha_min
    else:
        target = entry_price + alpha * (no_vwap - entry_price)

    spread_cents = (target - entry_price) * 100.0

    viable = spread_cents >= strat.min_spread_cents

    result = TakeProfitResult(
        entry_price=round(entry_price, 4),
        target_price=round(target, 4),
        alpha=round(alpha, 4),
        spread_cents=round(spread_cents, 2),
        viable=viable,
    )

    log.info(
        "take_profit_computed",
        entry=result.entry_price,
        target=result.target_price,
        alpha=result.alpha,
        spread_cents=result.spread_cents,
        viable=result.viable,
    )

    return result
