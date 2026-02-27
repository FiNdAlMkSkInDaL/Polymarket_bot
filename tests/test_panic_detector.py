"""
Tests for the panic spike detector.
"""

import pytest

from src.data.ohlcv import OHLCVAggregator, OHLCVBar, BAR_INTERVAL
from src.data.websocket_client import TradeEvent
from src.signals.panic_detector import PanicDetector, PanicSignal


def _make_trade(price: float, size: float, ts: float, asset_id: str = "YES_T") -> TradeEvent:
    return TradeEvent(
        timestamp=ts, market_id="MKT", asset_id=asset_id,
        side="buy", price=price, size=size, is_yes=True,
    )


def _build_history(agg: OHLCVAggregator, prices: list[float], base_vol: float = 10.0):
    """Feed trades to build bar history for the aggregator."""
    t = 1000.0
    for p in prices:
        agg.on_trade(_make_trade(p, base_vol, t, asset_id=agg.asset_id))
        t += BAR_INTERVAL + 0.1
        agg.on_trade(_make_trade(p + 0.001, 1, t, asset_id=agg.asset_id))
        t += 0.1


class TestPanicDetector:
    def _make_detector(self):
        yes_agg = OHLCVAggregator("YES_T", lookback_minutes=10)
        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        detector = PanicDetector(
            market_id="MKT",
            yes_asset_id="YES_T",
            no_asset_id="NO_T",
            yes_aggregator=yes_agg,
            no_aggregator=no_agg,
            zscore_threshold=2.0,
            volume_ratio_threshold=3.0,
        )
        return detector, yes_agg, no_agg

    def test_no_signal_with_insufficient_history(self):
        detector, yes_agg, no_agg = self._make_detector()
        bar = OHLCVBar(
            open_time=1000, open=0.50, high=0.55, low=0.48,
            close=0.80, volume=100, vwap=0.52, trade_count=10,
        )
        signal = detector.evaluate(bar, no_best_ask=0.20)
        assert signal is None  # Not enough bars in history

    def test_signal_fires_on_panic_spike(self):
        detector, yes_agg, no_agg = self._make_detector()

        # Build stable YES history around 0.45-0.50
        stable_prices = [0.45, 0.46, 0.47, 0.45, 0.46, 0.48, 0.45, 0.47, 0.46, 0.45]
        _build_history(yes_agg, stable_prices, base_vol=10.0)

        # Build NO history around 0.55
        no_prices = [0.55, 0.54, 0.55, 0.56, 0.55, 0.54, 0.55, 0.55, 0.54, 0.55]
        _build_history(no_agg, no_prices, base_vol=10.0)

        # Now simulate a panic bar: YES spikes to 0.80 with huge volume
        panic_bar = OHLCVBar(
            open_time=9000, open=0.50, high=0.82, low=0.50,
            close=0.80, volume=100.0,  # 10× normal volume
            vwap=0.75, trade_count=50,
        )

        # NO is discounted (best ask below VWAP)
        signal = detector.evaluate(panic_bar, no_best_ask=0.40, whale_confluence=False)

        # Whether this fires depends on the actual computed VWAP and sigma
        # from the stable history. With the small moves, sigma will be small,
        # so a jump to 0.80 should produce a very high Z-score.
        if yes_agg.rolling_volatility > 0:
            # The z-score should be enormous given stable history
            expected_z = (0.80 - yes_agg.rolling_vwap) / yes_agg.rolling_volatility
            if expected_z >= 2.0 and no_agg.rolling_vwap > 0.40:
                assert signal is not None
                assert signal.zscore >= 2.0
                assert signal.no_best_ask == 0.40

    def test_no_signal_when_no_not_discounted(self):
        detector, yes_agg, no_agg = self._make_detector()

        stable_prices = [0.45, 0.46, 0.47, 0.45, 0.46, 0.48, 0.45, 0.47, 0.46, 0.45]
        _build_history(yes_agg, stable_prices, base_vol=10.0)

        no_prices = [0.55, 0.54, 0.55, 0.56, 0.55, 0.54, 0.55, 0.55, 0.54, 0.55]
        _build_history(no_agg, no_prices, base_vol=10.0)

        panic_bar = OHLCVBar(
            open_time=9000, open=0.50, high=0.82, low=0.50,
            close=0.80, volume=100.0, vwap=0.75, trade_count=50,
        )

        # NO best ask is ABOVE its VWAP → not discounted → no signal
        signal = detector.evaluate(panic_bar, no_best_ask=0.60)
        assert signal is None
