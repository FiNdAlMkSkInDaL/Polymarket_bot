"""
Tests for the Telemetry module — post-run performance analytics.

Covers:
- Sharpe ratio computation
- Max drawdown calculation
- Maker/taker ratio
- PnL and fee aggregation
- Slippage tracking
- Round-trip trade stats
- Equity curve recording
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.backtest.matching_engine import Fill
from src.backtest.telemetry import BacktestMetrics, Telemetry, _MINUTES_PER_YEAR
from src.trading.executor import OrderSide


def _make_fill(
    *,
    price: float = 0.50,
    size: float = 10.0,
    fee: float = 0.0,
    ts: float = 0.0,
    is_maker: bool = False,
    side: OrderSide = OrderSide.BUY,
    order_id: str = "SIM-1",
) -> Fill:
    return Fill(
        order_id=order_id,
        price=price,
        size=size,
        fee=fee,
        timestamp=ts,
        is_maker=is_maker,
        side=side,
    )


class TestFillMetrics:
    """Fill counting and categorisation."""

    def test_empty_telemetry(self):
        tel = Telemetry(initial_cash=1000.0)
        m = tel.finalize(final_equity=1000.0)
        assert m.total_fills == 0
        assert m.total_pnl == 0.0
        assert m.sharpe_ratio == 0.0

    def test_fill_counts(self):
        tel = Telemetry(initial_cash=1000.0)

        # 3 maker fills, 2 taker fills
        for i in range(3):
            tel.record_fill(_make_fill(is_maker=True, ts=float(i)))
        for i in range(2):
            tel.record_fill(_make_fill(is_maker=False, fee=0.05, ts=float(i + 3)))

        m = tel.finalize(final_equity=1000.0)
        assert m.total_fills == 5
        assert m.maker_fills == 3
        assert m.taker_fills == 2

    def test_maker_taker_ratio(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_fill(_make_fill(is_maker=True))
        tel.record_fill(_make_fill(is_maker=True))
        tel.record_fill(_make_fill(is_maker=False, fee=0.01))

        m = tel.finalize(final_equity=1000.0)
        assert m.maker_taker_ratio == 2.0

    def test_maker_taker_ratio_no_takers(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_fill(_make_fill(is_maker=True))

        m = tel.finalize(final_equity=1000.0)
        assert m.maker_taker_ratio == float("inf")


class TestFeeAggregation:
    """Fee totals: maker vs taker."""

    def test_total_fees(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_fill(_make_fill(is_maker=True, fee=0.0))
        tel.record_fill(_make_fill(is_maker=False, fee=0.50))
        tel.record_fill(_make_fill(is_maker=False, fee=0.30))

        m = tel.finalize(final_equity=1000.0)
        assert abs(m.total_fees_paid - 0.80) < 1e-9
        assert m.maker_fees == 0.0
        assert abs(m.taker_fees - 0.80) < 1e-9


class TestVolumeMetrics:
    """Volume computation = price × size."""

    def test_total_volume(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_fill(_make_fill(price=0.50, size=100.0, is_maker=True))
        tel.record_fill(_make_fill(price=0.60, size=50.0, is_maker=False, fee=0.1))

        m = tel.finalize(final_equity=1000.0)
        expected = 0.50 * 100.0 + 0.60 * 50.0
        assert abs(m.total_volume - expected) < 1e-9
        assert abs(m.maker_volume - 50.0) < 1e-9
        assert abs(m.taker_volume - 30.0) < 1e-9


class TestSlippage:
    """Slippage vs mid-price calculation."""

    def test_buy_slippage(self):
        """Buy at 0.52 with mid at 0.50 → slippage = +0.02 × size."""
        tel = Telemetry(initial_cash=1000.0)
        fill = _make_fill(price=0.52, size=100.0, is_maker=False, fee=0.1, side=OrderSide.BUY)
        tel.record_fill(fill, mid_at_submission=0.50)

        m = tel.finalize(final_equity=1000.0)
        expected_slip = (0.52 - 0.50) * 100.0  # = 2.0
        assert abs(m.total_slippage - expected_slip) < 1e-9
        assert m.slippage_count == 1

    def test_sell_slippage(self):
        """Sell at 0.48 with mid at 0.50 → slippage = +0.02 × size."""
        tel = Telemetry(initial_cash=1000.0)
        fill = _make_fill(price=0.48, size=100.0, is_maker=False, fee=0.1, side=OrderSide.SELL)
        tel.record_fill(fill, mid_at_submission=0.50)

        m = tel.finalize(final_equity=1000.0)
        expected_slip = (0.50 - 0.48) * 100.0  # = 2.0
        assert abs(m.total_slippage - expected_slip) < 1e-9

    def test_maker_fill_no_slippage(self):
        """Maker fills don't contribute to slippage."""
        tel = Telemetry(initial_cash=1000.0)
        fill = _make_fill(is_maker=True)
        tel.record_fill(fill, mid_at_submission=0.50)

        m = tel.finalize(final_equity=1000.0)
        assert m.total_slippage == 0.0
        assert m.slippage_count == 0

    def test_zero_mid_no_slippage(self):
        """If mid_at_submission is 0 (unknown), no slippage recorded."""
        tel = Telemetry(initial_cash=1000.0)
        fill = _make_fill(is_maker=False, fee=0.01)
        tel.record_fill(fill, mid_at_submission=0.0)

        m = tel.finalize(final_equity=1000.0)
        assert m.slippage_count == 0

    def test_avg_slippage_bps(self):
        """Average slippage in basis points."""
        tel = Telemetry(initial_cash=1000.0)
        # Buy at 0.51 with mid 0.50, size 100: slippage = 1.0, notional = 51.0
        fill = _make_fill(price=0.51, size=100.0, is_maker=False, fee=0.01, side=OrderSide.BUY)
        tel.record_fill(fill, mid_at_submission=0.50)

        m = tel.finalize(final_equity=1000.0)
        # bps = (total_slip / total_notional) * 10_000
        expected_bps = (1.0 / 51.0) * 10_000
        assert abs(m.avg_slippage_bps - expected_bps) < 0.1


class TestPnL:
    """Total PnL from equity."""

    def test_positive_pnl(self):
        tel = Telemetry(initial_cash=1000.0)
        m = tel.finalize(final_equity=1200.0)
        assert abs(m.total_pnl - 200.0) < 1e-9
        assert abs(m.total_pnl_pct - 20.0) < 1e-9

    def test_negative_pnl(self):
        tel = Telemetry(initial_cash=1000.0)
        m = tel.finalize(final_equity=800.0)
        assert abs(m.total_pnl - (-200.0)) < 1e-9
        assert abs(m.total_pnl_pct - (-20.0)) < 1e-9

    def test_pnl_from_equity_curve(self):
        """If final_equity not provided, infer from last equity point."""
        tel = Telemetry(initial_cash=1000.0)
        tel.record_equity(1.0, 1010.0)
        tel.record_equity(2.0, 1050.0)

        m = tel.finalize()
        assert abs(m.total_pnl - 50.0) < 1e-9


class TestMaxDrawdown:
    """Peak-to-trough drawdown computation."""

    def test_no_drawdown(self):
        """Monotonically increasing equity → 0% drawdown."""
        tel = Telemetry(initial_cash=1000.0)
        for i in range(10):
            tel.record_equity(float(i), 1000.0 + i * 10.0)
        m = tel.finalize(final_equity=1090.0)
        assert m.max_drawdown == 0.0

    def test_simple_drawdown(self):
        """Peak at 1200, trough at 900 → 25% drawdown."""
        tel = Telemetry(initial_cash=1000.0)
        equity_points = [1000, 1100, 1200, 1000, 900, 1050]
        for i, eq in enumerate(equity_points):
            tel.record_equity(float(i), float(eq))

        m = tel.finalize(final_equity=1050.0)
        # max_dd = (1200 - 900) / 1200 = 0.25
        assert abs(m.max_drawdown - 0.25) < 1e-6
        assert abs(m.max_drawdown_usd - 300.0) < 1e-6

    def test_multiple_drawdowns(self):
        """Multiple peaks — picks the worst."""
        tel = Telemetry(initial_cash=1000.0)
        equity_points = [1000, 1500, 1200, 1500, 1000]
        for i, eq in enumerate(equity_points):
            tel.record_equity(float(i), float(eq))

        m = tel.finalize(final_equity=1000.0)
        # Worst: 1500→1000 = 33.3% DD
        expected_dd = (1500 - 1000) / 1500
        assert abs(m.max_drawdown - expected_dd) < 1e-6


class TestSharpeRatio:
    """Annualised Sharpe ratio from equity curve returns."""

    def test_zero_returns(self):
        """Constant equity → Sharpe = 0 (zero std dev)."""
        tel = Telemetry(initial_cash=1000.0)
        for i in range(10):
            tel.record_equity(float(i), 1000.0)
        m = tel.finalize(final_equity=1000.0)
        assert m.sharpe_ratio == 0.0

    def test_positive_sharpe(self):
        """Steadily increasing equity → positive Sharpe."""
        tel = Telemetry(initial_cash=1000.0)
        for i in range(20):
            tel.record_equity(float(i), 1000.0 + float(i) * 5.0)
        m = tel.finalize(final_equity=1095.0)
        assert m.sharpe_ratio > 0

    def test_sharpe_annualisation_factor(self):
        """Stable positive returns still annualise to a large positive Sharpe."""
        # We can verify by constructing known returns
        tel = Telemetry(initial_cash=1000.0)
        # Create equity: 1000, 1010, 1020.1, 1030.301, ...
        equity = 1000.0
        points = [equity]
        for i in range(50):
            equity *= 1.001  # 0.1% per bar
            points.append(equity)

        for i, eq in enumerate(points):
            tel.record_equity(float(i), eq)

        m = tel.finalize(final_equity=points[-1])

        # Expected: mean return ≈ 0.001, std ≈ very small
        # Sharpe should be very high (consistent returns)
        assert m.sharpe_ratio > 50  # very stable returns → high Sharpe

    def test_sharpe_respects_equity_sample_spacing(self):
        tel_fast = Telemetry(initial_cash=1000.0)
        tel_slow = Telemetry(initial_cash=1000.0)

        fast_points = [(0.0, 1000.0), (60.0, 1010.0), (120.0, 1005.0), (180.0, 1015.0)]
        slow_points = [(0.0, 1000.0), (600.0, 1010.0), (1200.0, 1005.0), (1800.0, 1015.0)]

        for ts, eq in fast_points:
            tel_fast.record_equity(ts, eq)
        for ts, eq in slow_points:
            tel_slow.record_equity(ts, eq)

        fast_metrics = tel_fast.finalize(final_equity=fast_points[-1][1])
        slow_metrics = tel_slow.finalize(final_equity=slow_points[-1][1])

        assert fast_metrics.sharpe_ratio > slow_metrics.sharpe_ratio

    def test_insufficient_data(self):
        """<3 equity samples → Sharpe = 0."""
        tel = Telemetry(initial_cash=1000.0)
        tel.record_equity(0.0, 1000.0)
        tel.record_equity(1.0, 1010.0)
        m = tel.finalize(final_equity=1010.0)
        assert m.sharpe_ratio == 0.0


class TestRoundTrips:
    """Round-trip trade statistics."""

    def test_single_winner(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_round_trip(
            entry_price=0.50, exit_price=0.55, size=100.0,
            entry_fee=0.5, exit_fee=0.5,
            entry_time=0.0, exit_time=10.0,
        )
        m = tel.finalize(final_equity=1004.0)

        assert m.round_trips == 1
        assert m.winners == 1
        assert m.losers == 0
        assert m.win_rate == 1.0
        # pnl_net = (0.55 - 0.50) × 100 - 0.5 - 0.5 = 4.0
        assert abs(m.avg_win - 4.0) < 1e-9
        assert m.avg_hold_time_s == 10.0

    def test_mixed_round_trips(self):
        tel = Telemetry(initial_cash=1000.0)
        # Winner: +5 net
        tel.record_round_trip(
            entry_price=0.50, exit_price=0.60, size=100.0,
            entry_fee=2.0, exit_fee=3.0,
            entry_time=0.0, exit_time=60.0,
        )
        # Loser: -3 net
        tel.record_round_trip(
            entry_price=0.55, exit_price=0.50, size=100.0,
            entry_fee=1.0, exit_fee=1.0,
            entry_time=10.0, exit_time=20.0,
        )

        m = tel.finalize(final_equity=1002.0)
        assert m.round_trips == 2
        assert m.winners == 1
        assert m.losers == 1
        assert m.win_rate == 0.5
        # Winner pnl: (0.6-0.5)*100 - 5 = 5.0
        # Loser pnl: (0.5-0.55)*100 - 2 = -7.0
        assert abs(m.avg_win - 5.0) < 1e-9
        assert abs(m.avg_loss - (-7.0)) < 1e-9

    def test_profit_factor(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_round_trip(
            entry_price=0.40, exit_price=0.50, size=100.0,
            entry_fee=0, exit_fee=0,
            entry_time=0, exit_time=1,
        )
        tel.record_round_trip(
            entry_price=0.50, exit_price=0.45, size=100.0,
            entry_fee=0, exit_fee=0,
            entry_time=2, exit_time=3,
        )

        m = tel.finalize(final_equity=1005.0)
        # gross win = 10.0, gross loss = 5.0
        assert abs(m.profit_factor - 2.0) < 1e-9


class TestEquityCurve:
    """Equity curve recording and serialisation."""

    def test_equity_curve_stored(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_equity(1.0, 1000.0)
        tel.record_equity(2.0, 1010.0)
        tel.record_equity(3.0, 1005.0)

        m = tel.finalize(final_equity=1005.0)
        assert len(m.equity_curve) == 3
        assert m.equity_curve[0] == (1.0, 1000.0)

    def test_summary_string(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_equity(0.0, 1000.0)
        m = tel.finalize(final_equity=1000.0)
        s = m.summary()
        assert "BACKTEST TELEMETRY REPORT" in s


class TestReset:
    """Telemetry reset clears all data."""

    def test_reset(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_fill(_make_fill())
        tel.record_equity(0.0, 1000.0)
        tel.record_round_trip(
            entry_price=0.5, exit_price=0.6, size=10,
            entry_fee=0, exit_fee=0, entry_time=0, exit_time=1,
        )

        tel.reset()
        m = tel.finalize(final_equity=1000.0)
        assert m.total_fills == 0
        assert m.round_trips == 0
        assert len(m.equity_curve) == 0


class TestSerialization:
    """BacktestMetrics.to_dict()"""

    def test_to_dict_roundtrip(self):
        tel = Telemetry(initial_cash=1000.0)
        tel.record_fill(_make_fill(fee=0.5, is_maker=False))
        tel.record_equity(0.0, 1000.0)
        tel.record_equity(1.0, 1005.0)

        m = tel.finalize(final_equity=1005.0)
        d = m.to_dict()

        assert isinstance(d, dict)
        assert "total_pnl" in d
        assert "equity_curve" in d
        assert isinstance(d["equity_curve"], list)
