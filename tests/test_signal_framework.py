"""
Tests for the multi-signal framework — imbalance, spread-compression,
and composite evaluator.
"""

import time

import pytest

from src.signals.signal_framework import (
    CompositeSignalEvaluator,
    OrderbookImbalanceSignal,
    SignalGenerator,
    SignalResult,
    SpreadCompressionSignal,
)


# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeSnap:
    def __init__(self, bids=None, asks=None, best_bid=0.0, best_ask=0.0):
        self.bids = bids or []
        self.asks = asks or []
        self.best_bid = best_bid
        self.best_ask = best_ask


class FakeBook:
    def __init__(self, snap, has_data=True):
        self.has_data = has_data
        self._snap = snap

    def snapshot(self):
        return self._snap


class AlwaysFireSignal(SignalGenerator):
    """Test signal that always fires with a fixed score."""

    def __init__(self, score: float = 0.8):
        self._score = score

    @property
    def name(self) -> str:
        return "always"

    def evaluate(self, **kwargs):
        return SignalResult(
            name=self.name,
            market_id="TEST",
            score=self._score,
            metadata={},
            timestamp=time.time(),
        )


class NeverFireSignal(SignalGenerator):
    @property
    def name(self) -> str:
        return "never"

    def evaluate(self, **kwargs):
        return None


# ── OrderbookImbalanceSignal ────────────────────────────────────────────────


class TestOrderbookImbalance:
    def test_fires_when_bid_depth_exceeds_threshold(self):
        sig = OrderbookImbalanceSignal("MKT", imbalance_threshold=2.0, min_depth_usd=10.0)

        snap = FakeSnap(
            bids=[(0.44, 50), (0.43, 50)],  # total bid = 100
            asks=[(0.46, 20), (0.47, 10)],   # total ask = 30
            best_bid=0.44,
            best_ask=0.46,
        )
        book = FakeBook(snap)

        result = sig.evaluate(no_book=book)
        assert result is not None
        assert result.name == "imbalance"
        assert result.score > 0
        assert result.metadata["imbalance_ratio"] > 2.0

    def test_no_fire_when_balanced(self):
        sig = OrderbookImbalanceSignal("MKT", imbalance_threshold=2.0, min_depth_usd=10.0)

        snap = FakeSnap(
            bids=[(0.44, 30)],
            asks=[(0.46, 30)],
            best_bid=0.44,
            best_ask=0.46,
        )
        book = FakeBook(snap)

        result = sig.evaluate(no_book=book)
        assert result is None

    def test_no_fire_when_low_depth(self):
        sig = OrderbookImbalanceSignal("MKT", imbalance_threshold=2.0, min_depth_usd=100.0)

        snap = FakeSnap(
            bids=[(0.44, 20)],
            asks=[(0.46, 5)],
            best_bid=0.44,
            best_ask=0.46,
        )
        book = FakeBook(snap)

        result = sig.evaluate(no_book=book)
        assert result is None

    def test_no_fire_without_book(self):
        sig = OrderbookImbalanceSignal("MKT")
        assert sig.evaluate() is None
        assert sig.evaluate(no_book=None) is None

    def test_score_normalisation(self):
        sig = OrderbookImbalanceSignal("MKT", imbalance_threshold=2.0, min_depth_usd=0.0)

        # 4:1 ratio → at 2x threshold → score = 1.0
        snap = FakeSnap(
            bids=[(0.44, 80)],
            asks=[(0.46, 20)],
        )
        book = FakeBook(snap)

        result = sig.evaluate(no_book=book)
        assert result is not None
        assert result.score == 1.0


# ── SpreadCompressionSignal ─────────────────────────────────────────────────


class TestSpreadCompression:
    def test_fires_when_spread_compresses(self):
        sig = SpreadCompressionSignal("MKT", compression_pct=0.5, min_history=5)

        # Build history of wide spreads
        for _ in range(10):
            sig.record_spread(0.04)

        # Now a tight spread
        snap = FakeSnap(best_bid=0.44, best_ask=0.45)  # spread = 0.01
        book = FakeBook(snap)

        result = sig.evaluate(no_book=book)
        assert result is not None
        assert result.name == "spread_compression"
        assert result.score > 0

    def test_no_fire_when_spread_is_normal(self):
        sig = SpreadCompressionSignal("MKT", compression_pct=0.5, min_history=5)

        for _ in range(10):
            sig.record_spread(0.04)

        # Spread is same as average
        snap = FakeSnap(best_bid=0.44, best_ask=0.48)  # 0.04
        book = FakeBook(snap)

        result = sig.evaluate(no_book=book)
        assert result is None

    def test_needs_minimum_history(self):
        sig = SpreadCompressionSignal("MKT", compression_pct=0.5, min_history=10)

        sig.record_spread(0.04)  # only 1 sample

        snap = FakeSnap(best_bid=0.449, best_ask=0.451)
        book = FakeBook(snap)

        result = sig.evaluate(no_book=book)
        assert result is None

    def test_no_fire_without_book(self):
        sig = SpreadCompressionSignal("MKT")
        assert sig.evaluate() is None


# ── CompositeSignalEvaluator ────────────────────────────────────────────────


class TestCompositeSignalEvaluator:
    def test_single_signal_fires(self):
        evaluator = CompositeSignalEvaluator(
            [(AlwaysFireSignal(0.8), 1.0)],
            min_composite_score=0.3,
        )
        actionable, score, fired = evaluator.is_actionable()
        assert actionable is True
        assert score == pytest.approx(0.8, abs=0.01)
        assert len(fired) == 1

    def test_no_signals_fire(self):
        evaluator = CompositeSignalEvaluator(
            [(NeverFireSignal(), 1.0)],
            min_composite_score=0.3,
        )
        actionable, score, fired = evaluator.is_actionable()
        assert actionable is False
        assert score == 0.0
        assert len(fired) == 0

    def test_weighted_average(self):
        evaluator = CompositeSignalEvaluator(
            [
                (AlwaysFireSignal(1.0), 2.0),
                (AlwaysFireSignal(0.5), 1.0),
            ],
            min_composite_score=0.0,
        )
        score, fired = evaluator.evaluate()
        # (1.0 * 2/3) + (0.5 * 1/3) = 0.833
        assert score == pytest.approx(0.833, abs=0.01)
        assert len(fired) == 2

    def test_partial_fire_only_weights_active(self):
        evaluator = CompositeSignalEvaluator(
            [
                (AlwaysFireSignal(0.6), 1.0),
                (NeverFireSignal(), 1.0),
            ],
            min_composite_score=0.0,
        )
        score, fired = evaluator.evaluate()
        # Only AlwaysFire contributes: 0.6 * (0.5 / 0.5) = 0.6
        assert score == pytest.approx(0.6, abs=0.01)
        assert len(fired) == 1

    def test_below_threshold(self):
        evaluator = CompositeSignalEvaluator(
            [(AlwaysFireSignal(0.2), 1.0)],
            min_composite_score=0.5,
        )
        actionable, score, fired = evaluator.is_actionable()
        assert actionable is False
        assert len(fired) == 1  # signal fired but composite too low
