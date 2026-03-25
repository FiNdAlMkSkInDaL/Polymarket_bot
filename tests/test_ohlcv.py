"""
Tests for the OHLCV aggregator — bar construction and rolling stats.
"""

import time

import numpy as np
import pytest

from src.data.ohlcv import OHLCVAggregator, BAR_INTERVAL
from src.data.websocket_client import TradeEvent


def _make_trade(
    price: float,
    size: float,
    ts: float,
    asset_id: str = "NO_TOKEN",
    *,
    side: str = "buy",
    is_taker: bool = False,
) -> TradeEvent:
    return TradeEvent(
        timestamp=ts,
        market_id="MARKET_1",
        asset_id=asset_id,
        side=side,
        price=price,
        size=size,
        is_yes=False,
        is_taker=is_taker,
    )


class TestOHLCVAggregator:
    def test_single_bar_construction(self):
        """Trades within one interval should not produce a bar until the next interval."""
        agg = OHLCVAggregator("NO_TOKEN", lookback_minutes=10)
        t0 = 1000.0

        # 5 trades within the same 60s window
        assert agg.on_trade(_make_trade(0.50, 10, t0)) is None
        assert agg.on_trade(_make_trade(0.52, 5, t0 + 10)) is None
        assert agg.on_trade(_make_trade(0.48, 8, t0 + 30)) is None
        assert agg.on_trade(_make_trade(0.55, 3, t0 + 50)) is None

        # First trade of next interval closes the bar
        bar = agg.on_trade(_make_trade(0.53, 2, t0 + 61))
        assert bar is not None
        assert bar.open == 0.50
        assert bar.high == 0.55
        assert bar.low == 0.48
        assert bar.close == 0.55
        assert bar.trade_count == 4
        assert bar.volume == pytest.approx(26.0)

    def test_vwap_calculation(self):
        """VWAP should weight price by volume."""
        agg = OHLCVAggregator("NO_TOKEN", lookback_minutes=10)
        t0 = 1000.0

        # Bar 1: single trade
        agg.on_trade(_make_trade(0.50, 100, t0))
        bar1 = agg.on_trade(_make_trade(0.60, 10, t0 + 61))  # closes bar 1
        assert bar1 is not None
        assert bar1.vwap == pytest.approx(0.50)  # only one trade in bar 1

    def test_rolling_volatility_after_multiple_bars(self):
        """After enough bars, rolling_volatility should be > 0 if prices move."""
        agg = OHLCVAggregator("NO_TOKEN", lookback_minutes=60)
        t0 = 1000.0

        prices = [0.50, 0.52, 0.48, 0.55, 0.60, 0.45, 0.50, 0.53, 0.51, 0.49]
        for i, p in enumerate(prices):
            agg.on_trade(_make_trade(p, 10, t0 + i * BAR_INTERVAL))
            # Trigger bar close with next tick
            agg.on_trade(_make_trade(p + 0.01, 1, t0 + (i + 1) * BAR_INTERVAL + 0.1))

        assert len(agg.bars) >= 5
        assert agg.rolling_volatility > 0
        assert agg.rolling_vwap > 0
        assert agg.avg_bar_volume > 0

    def test_current_price_tracks_last_trade(self):
        agg = OHLCVAggregator("NO_TOKEN")
        t0 = 1000.0
        agg.on_trade(_make_trade(0.65, 5, t0))
        assert agg.current_price == 0.65
        agg.on_trade(_make_trade(0.70, 3, t0 + 10))
        assert agg.current_price == 0.70

    def test_empty_aggregator(self):
        agg = OHLCVAggregator("NO_TOKEN")
        assert agg.current_price == 0.0
        assert agg.rolling_vwap == 0.0
        assert agg.rolling_volatility == 0.0

    def test_trade_flow_moments_track_taker_buy_and_sell_volume(self):
        agg = OHLCVAggregator("NO_TOKEN")
        t0 = 1000.0

        agg.on_trade(_make_trade(0.50, 5, t0, side="buy", is_taker=True))
        agg.on_trade(_make_trade(0.49, 2, t0 + 0.5, side="sell", is_taker=True))

        buy_volume, sell_volume = agg.trade_flow_moments(2000, current_time_ms=int((t0 + 1.0) * 1000))
        assert buy_volume == pytest.approx(5.0)
        assert sell_volume == pytest.approx(2.0)
        assert agg.trade_flow_imbalance(2000, current_time_ms=int((t0 + 1.0) * 1000)) == pytest.approx(3.0 / 7.0)

        agg.on_trade(_make_trade(0.48, 7, t0 + 3.0, side="sell", is_taker=True))
        late_buy_volume, late_sell_volume = agg.trade_flow_moments(2000, current_time_ms=int((t0 + 3.1) * 1000))
        assert late_buy_volume == pytest.approx(0.0)
        assert late_sell_volume == pytest.approx(7.0)
        assert agg.trade_flow_imbalance(2000, current_time_ms=int((t0 + 3.1) * 1000)) == pytest.approx(-1.0)


# ── Pillar 11.2: Downside Semi-Variance EWMA ────────────────────────────


class TestDownsideSemiVariance:
    """Prove that rolling_downside_vol_ewma only reacts to adverse moves."""

    def test_uptrend_downside_vol_near_zero(self):
        """Winning Trade Invariant: purely positive returns → total EWMA vol
        increases significantly, but downside vol remains practically 0."""
        agg = OHLCVAggregator("NO_TOKEN", lookback_minutes=60)
        t0 = 1000.0

        # Monotonically increasing prices: every log-return is positive
        prices = [0.50 + 0.01 * i for i in range(15)]  # 0.50 → 0.64
        for i, p in enumerate(prices):
            agg.on_trade(_make_trade(p, 10, t0 + i * BAR_INTERVAL))
            agg.on_trade(_make_trade(p + 0.001, 1, t0 + (i + 1) * BAR_INTERVAL + 0.1))

        assert len(agg.bars) >= 5
        # Total vol increases significantly (prices are moving)
        assert agg.rolling_volatility_ewma > 0.005
        # Downside vol remains practically zero (no negative returns)
        # All downside_r = min(r, 0) = 0 → EWMA decays seed toward 0
        assert agg.rolling_downside_vol_ewma < 0.001
        # Ratio must be negligible: downside is < 10% of total
        assert agg.rolling_downside_vol_ewma < agg.rolling_volatility_ewma * 0.1

    def test_downtrend_downside_vol_tracks_total(self):
        """If the market only goes DOWN, downside vol ≈ total vol."""
        agg = OHLCVAggregator("NO_TOKEN", lookback_minutes=60)
        t0 = 1000.0

        # Monotonically decreasing prices: every log-return is negative
        prices = [0.70 - 0.01 * i for i in range(15)]  # 0.70 → 0.56
        for i, p in enumerate(prices):
            agg.on_trade(_make_trade(p, 10, t0 + i * BAR_INTERVAL))
            agg.on_trade(_make_trade(p - 0.001, 1, t0 + (i + 1) * BAR_INTERVAL + 0.1))

        assert len(agg.bars) >= 5
        assert agg.rolling_volatility_ewma > 0
        # In a pure downtrend, downside vol should closely track total vol
        assert agg.rolling_downside_vol_ewma > agg.rolling_volatility_ewma * 0.5

    def test_cold_start_zero(self):
        """Before any bars close, downside vol is 0."""
        agg = OHLCVAggregator("NO_TOKEN")
        assert agg.rolling_downside_vol_ewma == 0.0

    def test_downside_vol_initialised_on_first_bar(self):
        """After the first bar close, downside vol state is seeded."""
        agg = OHLCVAggregator("NO_TOKEN", lookback_minutes=60)
        t0 = 1000.0

        # Create 3 bars to get enough log returns
        for i, p in enumerate([0.50, 0.48, 0.46]):
            agg.on_trade(_make_trade(p, 10, t0 + i * BAR_INTERVAL))
            agg.on_trade(_make_trade(p - 0.005, 1, t0 + (i + 1) * BAR_INTERVAL + 0.1))

        assert agg._downside_ewma_initialised is True
        assert agg.rolling_downside_vol_ewma > 0
