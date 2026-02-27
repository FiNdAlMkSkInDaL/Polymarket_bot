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
    viable: bool            # False if spread < fee-adjusted minimum
    fee_floor_cents: float = 0.0  # entry_fee + exit_fee + margin (cents)
    entry_fee_bps: int = 0
    exit_fee_bps: int = 0


def compute_take_profit(
    entry_price: float,
    no_vwap: float,
    *,
    realised_vol: float = 0.0,
    book_depth_ratio: float = 1.0,
    whale_confluence: bool = False,
    days_to_resolution: int = 30,
    entry_fee_bps: int = 0,
    exit_fee_bps: int = 0,
    desired_margin_cents: float | None = None,
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
    entry_fee_bps:
        Taker fee on entry leg in basis points (e.g. 156).
    exit_fee_bps:
        Taker fee on exit leg in basis points.  Use 0 for maker exit.
    desired_margin_cents:
        Minimum profit margin in cents above fee costs.  Defaults to
        ``settings.strategy.desired_margin_cents``.

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

    # ── Fee-floor enforcement ──────────────────────────────────────────────
    margin = (
        desired_margin_cents
        if desired_margin_cents is not None
        else strat.desired_margin_cents
    )
    entry_fee_cents = entry_price * entry_fee_bps / 10_000 * 100
    exit_fee_cents = target * exit_fee_bps / 10_000 * 100
    fee_floor_cents = round(entry_fee_cents + exit_fee_cents + margin, 4)

    # Widen the target if the vol-scaled spread is below the fee floor
    if spread_cents < fee_floor_cents:
        # Solve: target' = entry + fee_floor/100, then re-check exit fee
        # One Newton step is sufficient for convergence.
        target_adj = entry_price + fee_floor_cents / 100.0
        exit_fee_adj = target_adj * exit_fee_bps / 10_000 * 100
        fee_floor_cents = round(entry_fee_cents + exit_fee_adj + margin, 4)
        target = entry_price + fee_floor_cents / 100.0
        spread_cents = fee_floor_cents

    viable = spread_cents >= max(strat.min_spread_cents, fee_floor_cents)

    result = TakeProfitResult(
        entry_price=round(entry_price, 4),
        target_price=round(target, 4),
        alpha=round(alpha, 4),
        spread_cents=round(spread_cents, 2),
        viable=viable,
        fee_floor_cents=round(fee_floor_cents, 2),
        entry_fee_bps=entry_fee_bps,
        exit_fee_bps=exit_fee_bps,
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


# ═══════════════════════════════════════════════════════════════════════════
#  Pillar 3 — Adaptive volatility-based TP rescaling
# ═══════════════════════════════════════════════════════════════════════════

def compute_dynamic_spread(
    sigma_30: float,
    base_spread_cents: float | None = None,
    *,
    sigma_ref: float = 0.02,
    sensitivity: float | None = None,
    min_mult: float | None = None,
    max_mult: float | None = None,
) -> float:
    """Scale the minimum take-profit spread based on 30-min rolling σ.

    Formula::

        spread = base × (1 + k × (σ₃₀ - σ_ref) / σ_ref)

    Clamped to ``[base × min_mult,  base × max_mult]``.

    When σ₃₀ > σ_ref (panic), spread widens → hold for bigger reversion.
    When σ₃₀ < σ_ref (calm), spread tightens → scalp quickly.

    Parameters
    ----------
    sigma_30:
        30-minute rolling standard deviation of 1-min log returns.
    base_spread_cents:
        Baseline spread floor in cents.  Defaults to
        ``settings.strategy.min_spread_cents``.
    sigma_ref:
        Reference σ for "normal" conditions (default 0.02).
    sensitivity:
        The *k* multiplier.  Defaults to
        ``settings.strategy.tp_vol_sensitivity``.
    min_mult / max_mult:
        Clamp bounds as multiples of *base_spread_cents*.

    Returns
    -------
    float — the dynamically-scaled minimum spread in cents.
    """
    strat = settings.strategy
    base = base_spread_cents if base_spread_cents is not None else strat.min_spread_cents
    k = sensitivity if sensitivity is not None else strat.tp_vol_sensitivity
    lo = (min_mult if min_mult is not None else strat.tp_spread_min_mult) * base
    hi = (max_mult if max_mult is not None else strat.tp_spread_max_mult) * base

    if sigma_ref <= 0 or base <= 0:
        return base

    if sigma_30 <= 0:
        # No volatility data yet — use base unchanged.
        return base

    ratio = (sigma_30 - sigma_ref) / sigma_ref
    scaled = base * (1.0 + k * ratio)
    result = max(lo, min(hi, scaled))

    return round(result, 2)
