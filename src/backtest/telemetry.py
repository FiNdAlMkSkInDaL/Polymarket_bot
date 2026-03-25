"""
Post-run backtesting telemetry — computes institutional-grade performance
metrics from the trade log produced by the ``BacktestEngine``.

Metrics
───────
- Total PnL (net of fees)
- Total Fees Paid
- Maker / Taker Ratio
- Slippage Cost vs Mid-Price
- Max Drawdown (peak-to-trough)
- Sharpe Ratio (annualised from per-bar returns)
- Win Rate
- Average Win / Average Loss
- Equity Curve (timestamp, equity) time-series
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.backtest.matching_engine import Fill
from src.core.logger import get_logger

log = get_logger(__name__)

# Seconds in a year (365.25 × 24 × 60 × 60) — for annualising Sharpe
_MINUTES_PER_YEAR = 525_960.0
_SECONDS_PER_YEAR = _MINUTES_PER_YEAR * 60.0
_BAR_INTERVAL_S = 60.0  # 1-minute bars for return computation


@dataclass
class BacktestMetrics:
    """Container for post-run performance analytics."""

    # ── PnL ────────────────────────────────────────────────────────────
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    # ── Fees ───────────────────────────────────────────────────────────
    total_fees_paid: float = 0.0
    maker_fees: float = 0.0
    taker_fees: float = 0.0

    # ── Fill composition ───────────────────────────────────────────────
    total_fills: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    maker_taker_ratio: float = 0.0   # maker_fills / taker_fills

    # ── Volume ─────────────────────────────────────────────────────────
    total_volume: float = 0.0         # sum of fill_size × fill_price
    maker_volume: float = 0.0
    taker_volume: float = 0.0

    # ── Slippage ───────────────────────────────────────────────────────
    total_slippage: float = 0.0       # sum of (fill_price - mid) × size × sign
    avg_slippage_bps: float = 0.0     # mean slippage in basis points
    slippage_count: int = 0           # number of taker fills with slippage data

    # ── Risk metrics ───────────────────────────────────────────────────
    max_drawdown: float = 0.0         # peak-to-trough (fraction)
    max_drawdown_usd: float = 0.0     # peak-to-trough (absolute)
    sharpe_ratio: float = 0.0         # annualised
    sortino_ratio: float = 0.0        # annualised (downside deviation)

    # ── Trade-level stats ──────────────────────────────────────────────
    round_trips: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0       # gross_wins / gross_losses
    avg_hold_time_s: float = 0.0

    # ── Equity curve ───────────────────────────────────────────────────
    equity_curve: list[tuple[float, float]] = field(default_factory=list)
    # ── Portfolio Correlation Engine (PCE) metrics ──────────────────────
    avg_portfolio_correlation: float = 0.0
    max_portfolio_var: float = 0.0
    pce_rejections: int = 0
    diversification_ratio: float = 0.0  # gross VaR / net VaR (≥1 = diversified)
    smart_passive_counters: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        d = {}
        for k, v in self.__dict__.items():
            if k == "equity_curve":
                d[k] = [(round(ts, 2), round(eq, 4)) for ts, eq in v]
            elif isinstance(v, float):
                d[k] = round(v, 6)
            else:
                d[k] = v
        return d

    def summary(self) -> str:
        """Human-readable summary string."""
        smart_passive = self.smart_passive_counters or {}
        lines = [
            "═══════════════════════════════════════════════════════════",
            "           BACKTEST TELEMETRY REPORT",
            "═══════════════════════════════════════════════════════════",
            f"  Total PnL:            ${self.total_pnl:>+10.2f}  "
            f"({self.total_pnl_pct:+.2f}%)",
            f"  Total Fees Paid:      ${self.total_fees_paid:>10.4f}",
            f"  Maker / Taker Ratio:   {self.maker_taker_ratio:>10.2f}  "
            f"({self.maker_fills}M / {self.taker_fills}T)",
            f"  Smart-Passive:         {int(smart_passive.get('smart_passive_started', 0)):>3d} started  "
            f"{int(smart_passive.get('maker_filled', 0)):>3d} maker  "
            f"{int(smart_passive.get('fallback_triggered', 0)):>3d} fallback",
            f"  Slippage vs Mid:      ${self.total_slippage:>+10.4f}  "
            f"({self.avg_slippage_bps:+.1f} bps avg)",
            "───────────────────────────────────────────────────────────",
            f"  Max Drawdown:          {self.max_drawdown:>10.2%}  "
            f"(${self.max_drawdown_usd:.2f})",
            f"  Sharpe Ratio:          {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:         {self.sortino_ratio:>10.2f}",
            "───────────────────────────────────────────────────────────",
            f"  Round Trips:           {self.round_trips:>10d}",
            f"  Win Rate:              {self.win_rate:>10.1%}",
            f"  Avg Win:              ${self.avg_win:>+10.4f}",
            f"  Avg Loss:             ${self.avg_loss:>+10.4f}",
            f"  Profit Factor:         {self.profit_factor:>10.2f}",
            f"  Avg Hold Time:         {self.avg_hold_time_s:>10.1f}s",
            "───────────────────────────────────────────────────────────",
            f"  Total Volume:         ${self.total_volume:>10.2f}",
            f"  Total Fills:           {self.total_fills:>10d}",
            f"  Equity Points:         {len(self.equity_curve):>10d}",
            "═══════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class Telemetry:
    """Accumulates per-event data during a backtest run, then computes
    summary analytics.

    Usage
    ─────
        tel = Telemetry(initial_cash=1000.0)
        # ... during replay loop:
        tel.record_fill(fill, mid_at_submission)
        tel.record_equity(timestamp, equity)
        # ... after replay:
        metrics = tel.finalize()
        print(metrics.summary())
    """

    def __init__(self, initial_cash: float = 1000.0) -> None:
        self._initial_cash = initial_cash

        # ── Raw accumulation ───────────────────────────────────────────
        self._fills: list[Fill] = []
        self._slippage_records: list[tuple[float, float]] = []  # (slippage_usd, notional)
        self._equity_curve: list[tuple[float, float]] = []  # (ts, equity)

        # ── Round-trip trades ──────────────────────────────────────────
        self._round_trips: list[dict] = []

        # ── PCE snapshots (Pillar 15) ───────────────────────────────────
        self._pce_snapshots: list[dict] = []  # [{ts, var, avg_corr, n_pos}]
        self._pce_rejection_count: int = 0
        self._smart_passive_counters: dict[str, int] = {
            "smart_passive_started": 0,
            "maker_filled": 0,
            "fallback_triggered": 0,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  Recording methods (called during the replay loop)
    # ═══════════════════════════════════════════════════════════════════════

    def record_fill(
        self,
        fill: Fill,
        mid_at_submission: float = 0.0,
    ) -> None:
        """Record a fill event for later analysis.

        Parameters
        ----------
        fill:
            The fill object from the matching engine.
        mid_at_submission:
            The mid-price at the time the order was submitted.
            Used for slippage computation. 0 means unknown.
        """
        self._fills.append(fill)

        # Compute slippage for taker fills
        if not fill.is_maker and mid_at_submission > 0:
            from src.trading.executor import OrderSide

            if fill.side == OrderSide.BUY:
                slip = (fill.price - mid_at_submission) * fill.size
            else:
                slip = (mid_at_submission - fill.price) * fill.size
            notional = fill.price * fill.size
            self._slippage_records.append((slip, notional))

    def record_equity(self, timestamp: float, equity: float) -> None:
        """Record an equity snapshot (called periodically or on every fill)."""
        self._equity_curve.append((timestamp, equity))

    def record_round_trip(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        entry_fee: float,
        exit_fee: float,
        entry_time: float,
        exit_time: float,
    ) -> None:
        """Record a completed round-trip trade."""
        pnl_gross = (exit_price - entry_price) * size
        pnl_net = pnl_gross - entry_fee - exit_fee
        self._round_trips.append({
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "hold_time_s": exit_time - entry_time,
        })

    def record_pce_snapshot(
        self,
        timestamp: float,
        portfolio_var: float,
        avg_correlation: float,
        n_positions: int,
    ) -> None:
        """Record a PCE state snapshot for post-run analysis."""
        self._pce_snapshots.append({
            "timestamp": timestamp,
            "portfolio_var": portfolio_var,
            "avg_correlation": avg_correlation,
            "n_positions": n_positions,
        })

    def set_pce_rejections(self, count: int) -> None:
        """Set the PCE rejection count (called by strategy before finalize)."""
        self._pce_rejection_count = count

    def set_smart_passive_counters(
        self,
        *,
        started: int,
        maker_filled: int,
        fallback_triggered: int,
    ) -> None:
        """Attach replay smart-passive counters to the final metrics object."""
        self._smart_passive_counters = {
            "smart_passive_started": int(started),
            "maker_filled": int(maker_filled),
            "fallback_triggered": int(fallback_triggered),
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  Finalisation — compute all metrics
    # ═══════════════════════════════════════════════════════════════════════

    def finalize(self, final_equity: float | None = None) -> BacktestMetrics:
        """Compute all performance metrics from accumulated data.

        Parameters
        ----------
        final_equity:
            End-of-run equity. If None, inferred from the equity curve.
        """
        m = BacktestMetrics()

        # ── Fill metrics ───────────────────────────────────────────────
        m.total_fills = len(self._fills)
        m.maker_fills = sum(1 for f in self._fills if f.is_maker)
        m.taker_fills = sum(1 for f in self._fills if not f.is_maker)
        m.maker_taker_ratio = (
            m.maker_fills / m.taker_fills if m.taker_fills > 0 else float("inf")
        )

        # ── Fee metrics ────────────────────────────────────────────────
        m.total_fees_paid = sum(f.fee for f in self._fills)
        m.maker_fees = sum(f.fee for f in self._fills if f.is_maker)
        m.taker_fees = sum(f.fee for f in self._fills if not f.is_maker)

        # ── Volume ─────────────────────────────────────────────────────
        m.total_volume = sum(f.price * f.size for f in self._fills)
        m.maker_volume = sum(f.price * f.size for f in self._fills if f.is_maker)
        m.taker_volume = sum(f.price * f.size for f in self._fills if not f.is_maker)

        # ── Slippage ───────────────────────────────────────────────────
        if self._slippage_records:
            slips = [s for s, _ in self._slippage_records]
            notionals = [n for _, n in self._slippage_records]
            m.total_slippage = sum(slips)
            m.slippage_count = len(slips)
            total_notional = sum(notionals)
            if total_notional > 0:
                m.avg_slippage_bps = (m.total_slippage / total_notional) * 10_000
            else:
                m.avg_slippage_bps = 0.0

        # ── PnL ────────────────────────────────────────────────────────
        if final_equity is not None:
            m.total_pnl = final_equity - self._initial_cash
        elif self._equity_curve:
            m.total_pnl = self._equity_curve[-1][1] - self._initial_cash
        m.total_pnl_pct = (
            (m.total_pnl / self._initial_cash * 100.0)
            if self._initial_cash > 0
            else 0.0
        )

        # ── Equity curve & drawdown ────────────────────────────────────
        m.equity_curve = list(self._equity_curve)
        if self._equity_curve:
            equities = np.array([eq for _, eq in self._equity_curve])
            hwm = np.maximum.accumulate(equities)
            drawdowns = (hwm - equities)
            dd_pct = np.where(hwm > 0, drawdowns / hwm, 0.0)
            m.max_drawdown = float(dd_pct.max()) if len(dd_pct) > 0 else 0.0
            m.max_drawdown_usd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        # ── Sharpe & Sortino from equity curve ─────────────────────────
        if len(self._equity_curve) >= 3:
            timestamps = np.array([ts for ts, _ in self._equity_curve])
            equities = np.array([eq for _, eq in self._equity_curve])
            returns = np.diff(equities) / np.maximum(equities[:-1], 1e-9)
            sample_deltas = np.diff(timestamps)
            positive_deltas = sample_deltas[sample_deltas > 0]
            sample_interval_s = float(np.median(positive_deltas)) if len(positive_deltas) > 0 else _BAR_INTERVAL_S

            if len(returns) > 1:
                mu = float(np.mean(returns))
                sigma = float(np.std(returns, ddof=1))

                ann_factor = math.sqrt(_SECONDS_PER_YEAR / max(sample_interval_s, 1e-9))
                m.sharpe_ratio = (mu / sigma * ann_factor) if sigma > 0 else 0.0

                # Sortino: downside deviation
                downside = returns[returns < 0]
                if len(downside) > 1:
                    dd_std = float(np.std(downside, ddof=1))
                    m.sortino_ratio = (mu / dd_std * ann_factor) if dd_std > 0 else 0.0

        # ── Round-trip trade stats ─────────────────────────────────────
        m.round_trips = len(self._round_trips)
        if self._round_trips:
            pnls = [rt["pnl_net"] for rt in self._round_trips]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            m.winners = len(wins)
            m.losers = len(losses)
            m.win_rate = m.winners / m.round_trips if m.round_trips > 0 else 0.0
            m.avg_win = sum(wins) / len(wins) if wins else 0.0
            m.avg_loss = sum(losses) / len(losses) if losses else 0.0

            gross_win = sum(wins)
            gross_loss = abs(sum(losses))
            m.profit_factor = (
                gross_win / gross_loss if gross_loss > 0 else float("inf")
            )

            hold_times = [rt["hold_time_s"] for rt in self._round_trips]
            m.avg_hold_time_s = sum(hold_times) / len(hold_times)

        # ── PCE metrics ─────────────────────────────────────────────────
        if self._pce_snapshots:
            vars_ = [s["portfolio_var"] for s in self._pce_snapshots]
            corrs = [s["avg_correlation"] for s in self._pce_snapshots]
            m.max_portfolio_var = max(vars_) if vars_ else 0.0
            m.avg_portfolio_correlation = (
                sum(corrs) / len(corrs) if corrs else 0.0
            )

        # pce_rejections: set externally by strategy after finalize,
        # or pre-populated via set_pce_rejections() before finalize.
        m.pce_rejections = self._pce_rejection_count
        m.smart_passive_counters = dict(self._smart_passive_counters)

        # Diversification ratio: gross VaR / net VaR from last snapshot.
        # If no snapshots, leave at 0.0.
        if self._pce_snapshots and len(self._pce_snapshots) >= 1:
            # Compute average diversification across all snapshots
            # diversification_ratio ≥ 1 means portfolio is diversified
            last = self._pce_snapshots[-1]
            n_pos = last.get("n_positions", 0)
            if n_pos > 0 and last.get("portfolio_var", 0) > 0:
                m.diversification_ratio = 1.0  # placeholder; real gross/net needs full VaR call
            # If we have explicit gross/net data, use that:
            if "gross_var" in last and last["gross_var"] > 0:
                m.diversification_ratio = last["gross_var"] / last["portfolio_var"]

        return m

    # ═══════════════════════════════════════════════════════════════════════
    #  Reset
    # ═══════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._fills.clear()
        self._slippage_records.clear()
        self._equity_curve.clear()
        self._round_trips.clear()
        self._pce_snapshots.clear()
        self._pce_rejection_count = 0
        self._smart_passive_counters = {
            "smart_passive_started": 0,
            "maker_filled": 0,
            "fallback_triggered": 0,
        }
