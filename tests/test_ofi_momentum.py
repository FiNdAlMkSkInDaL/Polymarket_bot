from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.signals.ofi_momentum import OFIMomentumDetector


@dataclass
class FakeLevel:
    price: float
    size: float


@dataclass
class FakeSnapshot:
    best_bid: float
    best_ask: float
    timestamp: float


class FakeBook:
    def __init__(self, bid_price: float, bid_size: float, ask_price: float, ask_size: float, timestamp_ms: int):
        self._bid = FakeLevel(bid_price, bid_size)
        self._ask = FakeLevel(ask_price, ask_size)
        self._snapshot = FakeSnapshot(
            best_bid=bid_price,
            best_ask=ask_price,
            timestamp=timestamp_ms / 1000.0,
        )

    def levels(self, side: str, n: int = 1):
        if side.lower() in ("bid", "buy"):
            return [self._bid][:n]
        return [self._ask][:n]

    def snapshot(self):
        return self._snapshot


class ToxicFakeBook(FakeBook):
    def toxicity_metrics(self, side: str = "BUY"):
        return {
            "toxicity_index": 0.9 if side.upper() == "BUY" else 0.1,
            "toxicity_depth_evaporation": 0.35,
            "toxicity_sweep_ratio": 0.4,
        }


class FakeTradeAggregator:
    def __init__(self, trade_flow_imbalance: float):
        self._trade_flow_imbalance = trade_flow_imbalance

    def trade_flow_imbalance(self, window_ms: int, *, current_time_ms: int | None = None) -> float:
        return self._trade_flow_imbalance


class TestOFIMomentumDetector:
    def test_vi_formula_and_buy_trigger(self):
        detector = OFIMomentumDetector(
            market_id="MKT",
            no_asset_id="NO_TOKEN",
            window_ms=2000,
            threshold=0.85,
        )

        book1 = FakeBook(0.49, 95.0, 0.51, 5.0, 1_000)
        book2 = FakeBook(0.49, 96.0, 0.51, 4.0, 1_800)

        assert detector.evaluate(no_book=book1, timestamp_ms=1_000) is not None
        result = detector.evaluate(no_book=book2, timestamp_ms=1_800)

        assert result is not None
        assert result.metadata["direction"] == "BUY"
        assert result.metadata["current_vi"] == pytest.approx(0.92, abs=1e-6)
        assert result.metadata["rolling_vi"] == pytest.approx(0.91, abs=1e-6)

        signal = detector.generate_signal(no_book=book2, timestamp_ms=1_800)
        assert signal is not None
        assert signal.direction == "BUY"
        assert signal.rolling_vi == pytest.approx(0.913333, abs=1e-6)
        assert signal.signal_source == "ofi_momentum"

    def test_sell_trigger_on_negative_rolling_vi(self):
        detector = OFIMomentumDetector(market_id="MKT", window_ms=2000, threshold=0.85)

        detector.record_top_of_book(5.0, 95.0, timestamp_ms=1_000)
        result = detector.evaluate(bid_size=4.0, ask_size=96.0, best_ask=0.52, timestamp_ms=1_500)

        assert result is not None
        assert result.metadata["direction"] == "SELL"
        assert result.metadata["rolling_vi"] == pytest.approx(-0.91, abs=1e-6)

    def test_window_prunes_expired_samples(self):
        detector = OFIMomentumDetector(market_id="MKT", window_ms=1000, threshold=0.3)

        detector.record_top_of_book(90.0, 10.0, timestamp_ms=1_000)   # 0.8
        detector.record_top_of_book(80.0, 20.0, timestamp_ms=1_400)   # 0.6

        result = detector.evaluate(bid_size=70.0, ask_size=30.0, best_ask=0.55, timestamp_ms=2_100)

        assert result is not None
        assert result.metadata["rolling_vi"] == pytest.approx(0.5, abs=1e-6)
        assert detector.current_vi == pytest.approx(0.4, abs=1e-6)
        assert detector.rolling_vi == pytest.approx(0.5, abs=1e-6)

    def test_returns_none_without_valid_top_of_book(self):
        detector = OFIMomentumDetector(market_id="MKT")

        assert detector.evaluate() is None
        assert detector.evaluate(bid_size=0.0, ask_size=0.0, timestamp_ms=1_000) is None

    def test_signal_carries_toxicity_metrics_from_book(self):
        detector = OFIMomentumDetector(market_id="MKT", window_ms=2000, threshold=0.75)

        book = ToxicFakeBook(0.49, 95.0, 0.51, 5.0, 1_000)
        signal = detector.generate_signal(no_book=book, timestamp_ms=1_000)

        assert signal is not None
        assert signal.toxicity_index == pytest.approx(0.9, abs=1e-6)
        assert signal.toxicity_depth_evaporation == pytest.approx(0.35, abs=1e-6)
        assert signal.toxicity_sweep_ratio == pytest.approx(0.4, abs=1e-6)

    def test_trade_verified_imbalance_suppresses_unconfirmed_l2_pressure(self):
        detector = OFIMomentumDetector(
            market_id="MKT",
            window_ms=2000,
            threshold=0.85,
            tvi_kappa=1.0,
        )
        book = FakeBook(0.49, 95.0, 0.51, 5.0, 1_000)

        signal = detector.generate_signal(
            no_book=book,
            trade_aggregator=FakeTradeAggregator(0.0),
            timestamp_ms=1_000,
        )

        assert signal is None

    def test_trade_verified_imbalance_preserves_confirmed_flow(self):
        detector = OFIMomentumDetector(
            market_id="MKT",
            window_ms=2000,
            threshold=0.85,
            tvi_kappa=1.0,
        )
        book = FakeBook(0.49, 96.0, 0.51, 4.0, 1_000)

        signal = detector.generate_signal(
            no_book=book,
            trade_aggregator=FakeTradeAggregator(0.92),
            timestamp_ms=1_000,
        )

        assert signal is not None
        assert signal.direction == "BUY"
        assert signal.raw_rolling_vi == pytest.approx(0.92, abs=1e-6)
        assert signal.trade_flow_imbalance == pytest.approx(0.92, abs=1e-6)
        assert signal.tvi_multiplier == pytest.approx(1.0, abs=1e-6)
        assert signal.rolling_vi == pytest.approx(0.92, abs=1e-6)