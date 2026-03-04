"""
Liquidity-sensing position sizer — ``compute_depth_aware_size`` and
``compute_kelly_size``.

The depth-aware sizer analyses L2 order-book liquidity to prevent
self-impact.  The Kelly sizer overlays a fractional-Kelly capital-
allocation formula using signal strength and historical win-rate.

Together they enforce:
  1. Never consume > ``max_impact_pct`` of near-touch liquidity.
  2. Never bet > optimal Kelly fraction of bankroll.
  3. Never exceed the hard ``max_trade_usd`` cap.
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


# ═══════════════════════════════════════════════════════════════════════════
#  Fractional Kelly Sizing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class KellyResult:
    """Output of the Kelly sizer."""

    kelly_fraction: float      # full Kelly f*
    adj_fraction: float        # after applying fractional multiplier
    size_usd: float            # final dollar size
    size_shares: float         # shares at entry_price
    method: str                # "kelly" | "kelly_capped" | "kelly_no_edge"
    edge: float                # estimated edge (p*b - q) / b
    win_prob: float            # estimated win probability (after discounting)
    estimated_p: float = 0.0   # raw estimated win probability (before discounting)
    adjusted_p: float = 0.0    # discounted win probability used in Kelly formula
    uncertainty_penalty: float = 0.0  # applied uncertainty discount


def compute_kelly_size(
    *,
    signal_score: float,
    win_rate: float,
    avg_win_cents: float,
    avg_loss_cents: float,
    bankroll_usd: float,
    entry_price: float,
    max_trade_usd: float,
    book: OrderbookTracker | None = None,
    kelly_fraction_mult: float | None = None,
    max_kelly_pct: float | None = None,
    signal_metadata: dict | None = None,
    total_trades: int = 0,
    _precomputed_depth_result: SizingResult | None = None,
) -> KellyResult:
    """Compute position size using fractional Kelly criterion with
    **edge discounting** and **probability capping**.

    Guardrails vs. the naive Kelly formula:

    1. **Probability cap** — raw *p* is clamped to ``kelly_p_cap``
       (default 0.85).  We never assume near-certainty.
    2. **Edge discounting** — the estimated *p* is shrunk toward 0.5
       by an ``uncertainty_penalty`` (0.0–1.0) sourced from the signal
       framework's market-structure analysis:

       .. math::
           p_{adj} = 0.5 + (p - 0.5) \\times (1 - \\text{uncertainty\\_penalty})

    3. **Calibration logging** — every invocation logs ``estimated_p``,
       ``adjusted_p``, and ``uncertainty_penalty`` so that the
       relationship between predicted and realised edge can be audited
       offline.

    Parameters
    ----------
    signal_score:
        Normalised signal strength 0.0–1.0 from the signal framework.
    win_rate:
        Historical win rate (0.0–1.0). Falls back to 0.55 if ≤ 0.
    avg_win_cents / avg_loss_cents:
        Average PnL for winning/losing trades. Used for payoff ratio.
    bankroll_usd:
        Current available capital.
    entry_price:
        Expected entry price (for USD → shares conversion).
    max_trade_usd:
        Hard per-trade capital ceiling.
    book:
        Optional orderbook tracker for depth capping.
    kelly_fraction_mult:
        Fraction of full Kelly to use (default from config, typically 0.25).
    max_kelly_pct:
        Max % of bankroll per position (default from config, typically 10%).
    signal_metadata:
        Metadata dict from the signal framework.  Expected to contain
        ``"uncertainty_penalty"`` (float 0.0–1.0).  Falls back to
        ``kelly_default_uncertainty`` from config if absent.
    """
    strat = settings.strategy
    k_mult = kelly_fraction_mult if kelly_fraction_mult is not None else strat.kelly_fraction
    max_pct = max_kelly_pct if max_kelly_pct is not None else strat.kelly_max_pct
    p_cap = strat.kelly_p_cap

    # ── Cold-start bypass ───────────────────────────────────────────────
    # When total_trades < MIN_KELLY_TRADES, the Kelly formula has
    # insufficient data to compute a meaningful edge.  Use an adaptive
    # fraction of max_trade_usd that decays toward Kelly-optimal as
    # trades accumulate, and halts if rolling expectancy is negative.
    min_kelly_trades = strat.min_kelly_trades
    if total_trades < min_kelly_trades:
        # ── Adaptive cold-start fraction ───────────────────────────────
        # Blend between cold_start_frac (initial) and Kelly-optimal (0)
        # as trade count grows.  This replaces the fixed 50% that kept
        # placing max-sized trades regardless of accumulating losses.
        blend_weight = max(0.0, (min_kelly_trades - total_trades) / min_kelly_trades)
        base_cold_frac = strat.cold_start_frac * blend_weight

        # ── Negative expectancy halt ───────────────────────────────────
        # If rolling N-trade expectancy is negative, throttle sizing.
        meta = signal_metadata or {}
        rolling_expectancy = float(meta.get("rolling_expectancy_cents", 0.0))
        halt_window = strat.cold_start_halt_window
        if (strat.cold_start_negative_ev_halt
                and total_trades >= halt_window
                and rolling_expectancy < 0):
            # Reduce fraction by 75% when expectancy is negative
            base_cold_frac *= 0.25
            log.warning(
                "kelly_cold_start_negative_ev_throttle",
                total_trades=total_trades,
                rolling_expectancy=round(rolling_expectancy, 2),
                throttled_frac=round(base_cold_frac, 4),
            )

        cold_usd = max_trade_usd * base_cold_frac

        # Depth cap if book available
        if book is not None and book.has_data:
            depth_result = compute_depth_aware_size(
                book=book,
                entry_price=entry_price,
                max_trade_usd=cold_usd,
                side="BUY",
            )
            if depth_result.size_usd > 0:
                cold_usd = depth_result.size_usd

        if entry_price <= 0:
            shares = 0.0
        else:
            shares = round(cold_usd / entry_price, 2)

        if shares < 1:
            cold_usd = 0.0
            shares = 0.0

        log.info(
            "kelly_cold_start",
            total_trades=total_trades,
            min_required=min_kelly_trades,
            size_usd=round(cold_usd, 2),
            shares=shares,
        )

        return KellyResult(
            kelly_fraction=0.0,
            adj_fraction=0.0,
            size_usd=round(cold_usd, 4),
            size_shares=shares,
            method="kelly_cold_start",
            edge=0.0,
            win_prob=0.0,
            estimated_p=0.0,
            adjusted_p=0.0,
            uncertainty_penalty=0.0,
        )

    # ── Estimate win probability ────────────────────────────────────────
    # Use exponentially-decayed win rate if available in metadata,
    # else fall back to aggregate win_rate, else prior from config.
    meta = signal_metadata or {}
    decayed_wr = float(meta.get("decayed_win_rate", 0.0))
    if decayed_wr > 0:
        base_wr = decayed_wr
    elif win_rate > 0:
        base_wr = win_rate
    else:
        base_wr = strat.kelly_prior_win_rate
    # Signal adds up to 15 percentage points
    raw_p = min(p_cap, max(0.01, base_wr + 0.15 * signal_score))

    # ── Edge discounting via uncertainty penalty ────────────────────────
    # Extract uncertainty from signal metadata; fall back to conservative default
    uncertainty_penalty = float(meta.get("uncertainty_penalty", strat.kelly_default_uncertainty))
    uncertainty_penalty = max(0.0, min(1.0, uncertainty_penalty))

    # Shrink estimated p toward 0.5 based on uncertainty
    p_adj = 0.5 + (raw_p - 0.5) * (1.0 - uncertainty_penalty)
    p_adj = max(0.01, min(p_cap, p_adj))

    q = 1.0 - p_adj

    # ── Payoff ratio ────────────────────────────────────────────────────
    avg_win = abs(avg_win_cents) if avg_win_cents != 0 else 5.0
    avg_loss = abs(avg_loss_cents) if avg_loss_cents != 0 else 5.0
    b = avg_win / avg_loss  # odds ratio

    # ── Kelly fraction ──────────────────────────────────────────────────
    edge = p_adj * b - q
    if edge <= 0:
        # Negative edge → don't bet
        log.info(
            "kelly_negative_edge",
            estimated_p=round(raw_p, 4),
            adjusted_p=round(p_adj, 4),
            uncertainty_penalty=round(uncertainty_penalty, 4),
            b=round(b, 3),
            edge=round(edge, 4),
        )
        return KellyResult(
            kelly_fraction=0.0,
            adj_fraction=0.0,
            size_usd=0.0,
            size_shares=0.0,
            method="kelly_no_edge",
            edge=round(edge, 4),
            win_prob=round(p_adj, 4),
            estimated_p=round(raw_p, 4),
            adjusted_p=round(p_adj, 4),
            uncertainty_penalty=round(uncertainty_penalty, 4),
        )

    full_kelly = edge / b  # f*
    adj_kelly = full_kelly * k_mult

    # ── Capital caps ────────────────────────────────────────────────────
    kelly_usd = adj_kelly * bankroll_usd
    cap_usd = bankroll_usd * max_pct / 100.0
    size_usd = min(kelly_usd, cap_usd, max_trade_usd)

    # ── Depth cap (if book available) ───────────────────────────────────
    method = "kelly"
    if book is not None and book.has_data:
        if _precomputed_depth_result is not None:
            depth_result = _precomputed_depth_result
        else:
            depth_result = compute_depth_aware_size(
                book=book,
                entry_price=entry_price,
                max_trade_usd=size_usd,
                side="BUY",
            )
        if depth_result.size_usd < size_usd and depth_result.size_usd > 0:
            size_usd = depth_result.size_usd
            method = "kelly_depth_capped"

    if entry_price <= 0:
        shares = 0.0
    else:
        shares = round(size_usd / entry_price, 2)

    if shares < 1:
        size_usd = 0.0
        shares = 0.0

    # ── Calibration logging ─────────────────────────────────────────────
    log.info(
        "kelly_sizing",
        estimated_p=round(raw_p, 4),
        adjusted_p=round(p_adj, 4),
        uncertainty_penalty=round(uncertainty_penalty, 4),
        full_kelly=round(full_kelly, 4),
        adj_kelly=round(adj_kelly, 4),
        b=round(b, 3),
        edge=round(edge, 4),
        size_usd=round(size_usd, 2),
        shares=shares,
        method=method,
    )

    return KellyResult(
        kelly_fraction=round(full_kelly, 6),
        adj_fraction=round(adj_kelly, 6),
        size_usd=round(size_usd, 4),
        size_shares=shares,
        method=method,
        edge=round(edge, 4),
        win_prob=round(p_adj, 4),
        estimated_p=round(raw_p, 4),
        adjusted_p=round(p_adj, 4),
        uncertainty_penalty=round(uncertainty_penalty, 4),
    )
