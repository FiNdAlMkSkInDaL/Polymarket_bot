"""
Fee-curve utilities for the 2026 Polymarket Dynamic Fee regime.

The fee curve for crypto/sports markets is:

    Fee(p) = f_max · 4 · p · (1 - p)

where *p* is the mid-price (probability) and *f_max* = 1.56%.
Peak fee is at p = 0.50 (1.56%), tapering to 0 at p ∈ {0, 1}.

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
) -> float:
    """Compute the fee-adaptive stop-loss trigger in cents.

    The stop-loss must absorb the round-trip fee cost:

        SL_trigger = SL_base - (Fee_entry + Fee_exit) * 100

    where Fee_exit is estimated at the expected stop-loss exit price:

        p_exit = entry_price - SL_base / 100

    Parameters
    ----------
    sl_base_cents:
        Raw stop-loss threshold in cents (e.g. 8.0).
    entry_price:
        Entry price (probability) in [0, 1].
    fee_enabled:
        Whether this market category charges dynamic fees.
    f_max:
        Maximum fee rate fraction (default from config).

    Returns
    -------
    float
        Tightened stop-loss in cents.  Always ≥ 1.0 (floor).
    """
    if not fee_enabled:
        return sl_base_cents

    # Estimate exit price (where the stop would fire)
    estimated_exit = max(0.01, entry_price - sl_base_cents / 100.0)

    fee_drag_cents = compute_roundtrip_fee_cents(
        entry_price, estimated_exit, fee_enabled=fee_enabled, f_max=f_max
    )

    trigger = sl_base_cents - fee_drag_cents
    return max(1.0, round(trigger, 2))  # floor at 1 cent


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
