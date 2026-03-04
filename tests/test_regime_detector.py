"""Tests for the RegimeDetector (SI-1)."""

from __future__ import annotations

import math

import pytest

from src.signals.regime_detector import RegimeDetector, RegimeState, _sigmoid


class TestSigmoid:
    def test_mid(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_positive(self):
        assert _sigmoid(5.0) > 0.99

    def test_negative(self):
        assert _sigmoid(-5.0) < 0.01

    def test_clamp(self):
        """Values beyond ±20 should not overflow."""
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-8)
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-8)


class TestRegimeDetector:
    @pytest.fixture
    def detector(self):
        return RegimeDetector("MARKET_A", threshold=0.40)

    def test_initial_state(self, detector):
        assert detector.regime_score == 0.50
        assert detector.is_mean_revert is True

    def test_state_snapshot(self, detector):
        s = detector.state()
        assert isinstance(s, RegimeState)
        assert s.bar_count == 0

    def test_update_returns_score(self, detector):
        score = detector.update(log_return=0.001, ewma_vol=0.01, ew_vol=0.01)
        assert 0.0 <= score <= 1.0

    def test_trending_regime(self, detector):
        """Feed sustained positive returns → should push toward trending."""
        for _ in range(30):
            detector.update(log_return=0.02, ewma_vol=0.03, ew_vol=0.01)
        # persistent + high returns + vol expansion = trending
        assert detector.regime_score < 0.50

    def test_mean_reverting_regime(self, detector):
        """Feed alternating returns → should push toward mean-reversion."""
        for i in range(30):
            sign = 1 if i % 2 == 0 else -1
            detector.update(log_return=sign * 0.01, ewma_vol=0.01, ew_vol=0.01)
        # alternating = negative autocorrelation = MR
        assert detector.regime_score > 0.45

    def test_state_snapshot_after_updates(self, detector):
        for i in range(10):
            detector.update(log_return=0.001, ewma_vol=0.01, ew_vol=0.01)
        s = detector.state()
        assert s.bar_count == 10
        assert 0.0 <= s.autocorr_factor <= 1.0
        assert 0.0 <= s.vol_ratio_factor <= 1.0
        assert 0.0 <= s.persistence_factor <= 1.0


class TestRegimeDetectorVolRatio:
    def test_stable_vol_high_score(self):
        det = RegimeDetector("MKT")
        # ewma == ew → ratio = 1.0 → favorable
        for _ in range(10):
            det.update(log_return=0.001, ewma_vol=0.01, ew_vol=0.01)
        s = det.state()
        assert s.vol_ratio_factor > 0.5  # ratio=1 maps to ~0.73

    def test_expanding_vol_low_score(self):
        det = RegimeDetector("MKT")
        for _ in range(10):
            det.update(log_return=0.001, ewma_vol=0.03, ew_vol=0.01)
        s = det.state()
        assert s.vol_ratio_factor < 0.5  # ratio=3 → unfavorable
