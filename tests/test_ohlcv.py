"""
Tests for the OHLCV aggregator — bar construction and rolling stats.
"""

import time

import numpy as np
import pytest

from src.data.ohlcv import OHLCVAggregator, BAR_INTERVAL
from src.data.websocket_client import TradeEvent


def _make_trade(price: float, size: float, ts: float, asset_id: str = "NO_TOKEN") -> TradeEvent:
    return TradeEvent(
        timestamp=ts,
        market_id="MARKET_1",
        asset_id=asset_id,
        side="buy",
        price=price,
        size=size,
        is_yes=False,
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
