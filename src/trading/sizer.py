"""
Liquidity-sensing position sizer — ``compute_depth_aware_size``.

Analyses the first *N* levels of the L2 order book and caps order size
so that it never consumes more than ``max_impact_pct`` percent of the
available near-touch liquidity.  This prevents the bot from moving the
market against itself on thin books.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import settings
from src.core.logger import get_logger
from src.data.orderbook import OrderbookSnapshot, OrderbookTracker

log = get_logger(__name__)


@dataclass
class SizingResult:
    """Output of the depth-aware sizer."""

    size_usd: float           # dollar amount to trade
    size_shares: float        # shares at entry_price
    available_liq_usd: float  # liquidity within impact window
    method: str               # "depth_aware" | "fallback"
    capped: bool              # True if impact cap reduced the size


def compute_depth_aware_size(
    book: OrderbookTracker,
    entry_price: float,
    max_trade_usd: float,
    *,
    max_impact_pct: float | None = None,
    impact_depth_cents: float | None = None,
    side: str = "BUY",
) -> SizingResult:
    """Compute an order size that respects near-touch liquidity.

    Parameters
    ----------
    book:
        The ``OrderbookTracker`` for the asset being traded.
    entry_price:
        Expected entry price (used to convert USD → shares).
    max_trade_usd:
        Hard capital ceiling for this trade.
    max_impact_pct:
        Max percentage of near-touch liquidity the order may consume.
        Defaults to ``settings.strategy.max_impact_pct``.
    impact_depth_cents:
        How many cents from mid-price to scan for available liquidity.
        Defaults to ``settings.strategy.impact_depth_cents``.
    side:
        ``"BUY"`` → scan ask side, ``"SELL"`` → scan bid side.

    Returns
    -------
    SizingResult with the recommended size and diagnostics.
    """
    strat = settings.strategy
    impact_pct = max_impact_pct if max_impact_pct is not None else strat.max_impact_pct
    depth_cents = impact_depth_cents if impact_depth_cents is not None else strat.impact_depth_cents

    if entry_price <= 0:
        return SizingResult(
            size_usd=0.0,
            size_shares=0.0,
            available_liq_usd=0.0,
            method="rejected",
            capped=False,
        )

    # ── Fallback: no orderbook data → 50% haircut ──────────────────────────
    if not book.has_data:
        fallback_usd = max_trade_usd * 0.50
        shares = round(fallback_usd / entry_price, 2)
        if shares < 1:
            shares = 0.0
            fallback_usd = 0.0
        log.info(
            "sizer_fallback",
            reason="no_book_data",
            size_usd=fallback_usd,
            shares=shares,
        )
        return SizingResult(
            size_usd=round(fallback_usd, 4),
            size_shares=shares,
            available_liq_usd=0.0,
            method="fallback",
            capped=False,
        )

    # ── Depth-aware sizing ──────────────────────────────────────────────────
    snap = book.snapshot()
    mid = snap.mid_price
    if mid <= 0:
        mid = entry_price  # best-effort

    depth_boundary = depth_cents / 100.0

    # Collect levels within the impact window
    if side.upper() == "BUY":
        levels = book.levels("ask", 5)
        available_liq_usd = sum(
            lv.price * lv.size
            for lv in levels
            if lv.price <= mid + depth_boundary
        )
    else:
        levels = book.levels("bid", 5)
        available_liq_usd = sum(
            lv.price * lv.size
            for lv in levels
            if lv.price >= mid - depth_boundary
        )

    max_impact_usd = available_liq_usd * impact_pct / 100.0
    capped = max_impact_usd < max_trade_usd and available_liq_usd > 0

    final_usd = min(max_trade_usd, max_impact_usd) if available_liq_usd > 0 else max_trade_usd * 0.50
    shares = round(final_usd / entry_price, 2)

    if shares < 1:
        log.info(
            "sizer_insufficient_liquidity",
            available_usd=round(available_liq_usd, 2),
            impact_cap_usd=round(max_impact_usd, 2),
            shares=shares,
        )
        return SizingResult(
            size_usd=0.0,
            size_shares=0.0,
            available_liq_usd=round(available_liq_usd, 2),
            method="depth_aware",
            capped=capped,
        )

    log.info(
        "sizer_depth_aware",
        size_usd=round(final_usd, 4),
        shares=shares,
        available_liq=round(available_liq_usd, 2),
        impact_cap=round(max_impact_usd, 2),
        capped=capped,
    )

    return SizingResult(
        size_usd=round(final_usd, 4),
        size_shares=shares,
        available_liq_usd=round(available_liq_usd, 2),
        method="depth_aware",
        capped=capped,
    )
