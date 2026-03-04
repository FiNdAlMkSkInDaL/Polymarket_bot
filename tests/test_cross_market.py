"""Tests for the CrossMarketSignalGenerator (SI-3)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.signals.cross_market import CrossMarketSignal, CrossMarketSignalGenerator


@dataclass
class FakeEstimate:
    blended: float
    empirical_corr: float = 0.0
    structural_corr: float = 0.0
    overlap_bars: int = 100


class FakeCorrelationMatrix:
    def __init__(self, pairs: dict):
        self._pairs = pairs

    def all_pairs(self):
        return self._pairs


class FakePCE:
    def __init__(self, pairs: dict | None = None):
        self.corr_matrix = FakeCorrelationMatrix(pairs or {})


class TestCrossMarketSignalGenerator:
    def test_no_signals_when_empty(self):
        gen = CrossMarketSignalGenerator(FakePCE(), shadow_mode=True)
        assert gen.scan() == []

    def test_no_signal_below_z_threshold(self):
        """Returns below z_entry should not produce signals."""
        pce = FakePCE({
            ("MKT_A", "MKT_B"): FakeEstimate(blended=0.90),
        })
        gen = CrossMarketSignalGenerator(pce, z_entry=2.0, min_correlation=0.50)
        gen.record_return("MKT_A", 0.001)
        gen.record_return("MKT_B", 0.001)  # correlated → spread ≈ 0
        signals = gen.scan()
        assert signals == []

    def test_signal_on_divergence(self):
        """A large return on A with none on B should produce a signal."""
        pce = FakePCE({
            ("MKT_A", "MKT_B"): FakeEstimate(blended=0.90),
        })
        gen = CrossMarketSignalGenerator(
            pce, z_entry=1.0, min_correlation=0.50, spread_ewma_lambda=0.50,
        )
        # First call: seeds the spread variance
        gen.record_return("MKT_A", 0.001)
        gen.record_return("MKT_B", 0.001)
        gen.scan()

        # Second call: large divergence
        gen.record_return("MKT_A", 0.20)   # big move on A
        gen.record_return("MKT_B", 0.001)  # B didn't follow
        signals = gen.scan()
        assert len(signals) > 0
        sig = signals[0]
        assert isinstance(sig, CrossMarketSignal)
        assert abs(sig.z_score) >= 1.0
        assert sig.correlation == pytest.approx(0.90, abs=0.01)

    def test_low_correlation_pair_ignored(self):
        """Pairs with |ρ| below min_correlation should be skipped."""
        pce = FakePCE({
            ("MKT_A", "MKT_B"): FakeEstimate(blended=0.10),  # low corr
        })
        gen = CrossMarketSignalGenerator(pce, min_correlation=0.50)
        gen.record_return("MKT_A", 0.10)
        gen.record_return("MKT_B", 0.001)
        signals = gen.scan()
        assert signals == []

    def test_missing_return_data_skipped(self):
        """Pairs where one market has no return data should be skipped."""
        pce = FakePCE({
            ("MKT_A", "MKT_B"): FakeEstimate(blended=0.90),
        })
        gen = CrossMarketSignalGenerator(pce, min_correlation=0.50)
        gen.record_return("MKT_A", 0.10)
        # MKT_B return not recorded
        signals = gen.scan()
        assert signals == []

    def test_returns_cleared_after_scan(self):
        """Returns should be cleared after each scan cycle."""
        pce = FakePCE({
            ("MKT_A", "MKT_B"): FakeEstimate(blended=0.90),
        })
        gen = CrossMarketSignalGenerator(pce, min_correlation=0.50)
        gen.record_return("MKT_A", 0.10)
        gen.record_return("MKT_B", 0.001)
        gen.scan()
        # After scan, returns should be cleared
        assert gen._last_returns == {}

    def test_shadow_mode(self):
        gen = CrossMarketSignalGenerator(FakePCE(), shadow_mode=True)
        assert gen.is_shadow is True

    def test_reset(self):
        gen = CrossMarketSignalGenerator(FakePCE())
        gen.record_return("MKT_A", 0.01)
        gen._spread_var[("A", "B")] = 0.001
        gen.reset()
        assert gen._last_returns == {}
        assert gen._spread_var == {}
