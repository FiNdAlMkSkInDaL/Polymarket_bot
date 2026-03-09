"""
Fee-curve utilities for the 2026 Polymarket Dynamic Fee regime.

The fee curve for crypto/sports markets is:

    Fee(p) = f_max · 4 · p · (1 - p)

where *p* is the mid-price (probability) and *f_max* = 2.00%.
Peak fee is at p = 0.50 (2.00%), tapering to 0 at p ∈ {0, 1}.

Political / non-fee markets remain at 0%.
"""

from __future__ import annotations

from src.core.config import settings


def get_fee_rate(price: float, *, fee_enabled: bool = True, f_max: float | None = None) -> float:
    """Return the one-way fee as a fraction for a given mid-price.

    Parameters
    ----------
    price:
        Market probability / mid-price in [0, 1].
    fee_enabled:
        Whether this market category charges dynamic fees.
    f_max:
        Maximum fee rate (default from config: 1.56% = 0.0156).

    Returns
    -------
    float
        Fee fraction in [0, f_max].  0.0 if fee_enabled is False.
    """
    if not fee_enabled:
        return 0.0
    if f_max is None:
        f_max = settings.strategy.fee_max_pct / 100.0
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return f_max * 4.0 * price * (1.0 - price)


def compute_roundtrip_fee_cents(
    entry_price: float,
    exit_price: float,
    *,
    fee_enabled: bool = True,
    f_max: float | None = None,
) -> float:
    """Compute total round-trip fee drag in cents.

    Returns
    -------
    float
        Total fee in cents (entry_fee + exit_fee) * 100.
    """
    entry_fee = get_fee_rate(entry_price, fee_enabled=fee_enabled, f_max=f_max)
    exit_fee = get_fee_rate(exit_price, fee_enabled=fee_enabled, f_max=f_max)
    return (entry_fee + exit_fee) * 100.0


def compute_adaptive_stop_loss_cents(
    sl_base_cents: float,
    entry_price: float,
    *,
    fee_enabled: bool = True,
    f_max: float | None = None,
    ewma_vol: float | None = None,
    ref_vol: float = 0.70,
    is_adaptive: bool = True,
    max_multiplier: float = 1.5,
) -> float:
    """Compute the volatility- and fee-adaptive stop-loss trigger in cents.

    Formula
    -------
    1. **Vol multiplier** (stretch only, never shrink)::

           vol_ratio  = ewma_vol / ref_vol
           multiplier = clamp(vol_ratio, 1.0, max_multiplier)

       Falls back to 1.0 when *ewma_vol* is unavailable, ≤ 0, or
       *is_adaptive* is False.

    2. **Stretched baseline**::

           sl_stretched = sl_base_cents × multiplier

    3. **Fee deduction** (invariant preserved)::

           SL_trigger = sl_stretched - roundtrip_fee_cents

    4. **Absolute floor**::

           SL_trigger = max(1.0, SL_trigger)

    Parameters
    ----------
    sl_base_cents:
        Raw stop-loss threshold in cents (e.g. 4.0).
    entry_price:
        Entry price (probability) in [0, 1].
    fee_enabled:
        Whether this market category charges dynamic fees.
    f_max:
        Maximum fee rate fraction (default from config).
    ewma_vol:
        Current EWMA volatility from OHLCVAggregator.  ``None``
        during cold-start.
    ref_vol:
        Reference ("normal") EWMA volatility.  Used as denominator.
    is_adaptive:
        Master switch for vol-adaptive stretching.
    max_multiplier:
        Upper bound on the vol multiplier (e.g. 1.5 = +50%).

    Returns
    -------
    float
        Adjusted stop-loss in cents.  Always ≥ 1.0 (floor).
    """
    # ── Step 1: vol multiplier (stretch-only) ──────────────────────────
    if is_adaptive and ewma_vol is not None and ewma_vol > 0 and ref_vol > 0:
        vol_ratio = ewma_vol / ref_vol
        multiplier = max(1.0, min(vol_ratio, max_multiplier))
    else:
        multiplier = 1.0

    # ── Step 2: apply multiplier to base BEFORE fee deduction ──────────
    sl_stretched = sl_base_cents * multiplier

    if not fee_enabled:
        return max(1.0, round(sl_stretched, 2))

    # ── Step 3: estimate exit price & deduct fees ──────────────────────
    estimated_exit = max(0.01, entry_price - sl_stretched / 100.0)

    fee_drag_cents = compute_roundtrip_fee_cents(
        entry_price, estimated_exit, fee_enabled=fee_enabled, f_max=f_max
    )

    trigger = sl_stretched - fee_drag_cents

    # ── Step 4: absolute floor ─────────────────────────────────────────
    return max(1.0, round(trigger, 2))


def compute_adaptive_trailing_offset_cents(
    base_offset_cents: float,
    *,
    ewma_downside_vol: float | None = None,
    ref_vol: float = 0.70,
    is_adaptive: bool = True,
    max_multiplier: float = 1.5,
) -> float:
    """Compute the volatility-adaptive trailing stop offset in cents.

    Uses the same anti-shrinkage clamp as the baseline stop-loss:

        multiplier = clamp(σ_downside / σ_ref, 1.0, max_multiplier)
        offset     = base_offset_cents × multiplier

    No fee deduction is applied because roundtrip fees are already
    accounted for in the breakeven-activation threshold.

    Parameters
    ----------
    base_offset_cents:
        Raw trailing stop offset in cents (from config).
    ewma_downside_vol:
        Current downside semi-variance EWMA σ.  ``None`` during cold-start.
    ref_vol:
        Reference ("normal") EWMA volatility.
    is_adaptive:
        Master switch for vol-adaptive stretching.
    max_multiplier:
        Upper bound on the vol multiplier.

    Returns
    -------
    float
        Trailing stop offset in cents.  Always ≥ base_offset_cents.
    """
    if is_adaptive and ewma_downside_vol is not None and ewma_downside_vol > 0 and ref_vol > 0:
        vol_ratio = ewma_downside_vol / ref_vol
        multiplier = max(1.0, min(vol_ratio, max_multiplier))
    else:
        multiplier = 1.0

    return round(base_offset_cents * multiplier, 2)


def compute_net_pnl_cents(
    entry_price: float,
    exit_price: float,
    size: float,
    *,
    fee_enabled: bool = True,
    f_max: float | None = None,
) -> float:
    """Compute net PnL in cents after deducting round-trip fees.

    PnL = [(exit - entry) - Fee_entry - Fee_exit] × size × 100
    """
    entry_fee = get_fee_rate(entry_price, fee_enabled=fee_enabled, f_max=f_max)
    exit_fee = get_fee_rate(exit_price, fee_enabled=fee_enabled, f_max=f_max)
    gross = exit_price - entry_price
    net = gross - entry_fee - exit_fee
    return round(net * size * 100.0, 2)
