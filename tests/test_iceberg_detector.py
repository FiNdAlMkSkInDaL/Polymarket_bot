"""Tests for the IcebergDetector (SI-2)."""

from __future__ import annotations

import time

import pytest

from src.signals.iceberg_detector import IcebergDetector, IcebergSignal


class TestIcebergDetector:
    @pytest.fixture
    def detector(self):
        return IcebergDetector(
            "ASSET_A",
            refill_window_s=5.0,
            min_refills=3,
            size_tolerance_pct=0.30,
        )

    def test_initial_state(self, detector):
        assert len(detector.active_icebergs) == 0
        assert detector.has_iceberg("BUY") is False
        assert detector.has_iceberg("SELL") is False
        assert detector.strongest_iceberg("BUY") is None

    def test_no_signal_on_normal_level_changes(self, detector):
        """Random level additions and removals should not trigger."""
        result = detector.on_level_change("BUY", 0.50, 0.0, 100.0)
        assert result is None
        result = detector.on_level_change("BUY", 0.50, 100.0, 200.0)
        assert result is None

    def test_single_refill_not_enough(self, detector):
        """A single consume-and-replenish should NOT trigger (need 3+)."""
        # Level consumed
        detector.on_level_change("BUY", 0.50, 100.0, 0.0)
        # Same level reappears within window
        result = detector.on_level_change("BUY", 0.50, 0.0, 100.0)
        assert result is None
        assert detector.has_iceberg("BUY") is False

    def test_iceberg_detected_after_min_refills(self, detector):
        """3 refill cycles at the same price should trigger an iceberg signal."""
        for _ in range(3):
            detector.on_level_change("BUY", 0.50, 100.0, 0.0)  # consumed
            result = detector.on_level_change("BUY", 0.50, 0.0, 100.0)  # refilled

        assert result is not None
        assert isinstance(result, IcebergSignal)
        assert result.side == "BUY"
        assert result.price == 0.50
        assert result.refill_count >= 3
        assert result.avg_slice_size == pytest.approx(100.0)
        assert result.confidence > 0

    def test_iceberg_on_sell_side(self, detector):
        """Icebergs should work on the SELL side too."""
        for _ in range(3):
            detector.on_level_change("SELL", 0.55, 50.0, 0.0)
            result = detector.on_level_change("SELL", 0.55, 0.0, 50.0)

        assert result is not None
        assert result.side == "SELL"
        assert detector.has_iceberg("SELL") is True

    def test_size_tolerance_rejects_different_sizes(self, detector):
        """Refills with very different sizes should be rejected."""
        detector.on_level_change("BUY", 0.50, 100.0, 0.0)
        # Refill with 200 (100% different) — should NOT count
        result = detector.on_level_change("BUY", 0.50, 0.0, 200.0)
        assert result is None

    def test_size_tolerance_accepts_similar_sizes(self, detector):
        """Refills within 30% tolerance should count."""
        for _ in range(3):
            detector.on_level_change("BUY", 0.50, 100.0, 0.0)
            result = detector.on_level_change("BUY", 0.50, 0.0, 110.0)  # 10% diff

        assert result is not None  # should still trigger

    def test_reset_clears_state(self, detector):
        for _ in range(3):
            detector.on_level_change("BUY", 0.50, 100.0, 0.0)
            detector.on_level_change("BUY", 0.50, 0.0, 100.0)

        assert detector.has_iceberg("BUY") is True
        detector.reset()
        assert detector.has_iceberg("BUY") is False
        assert len(detector.active_icebergs) == 0

    def test_strongest_iceberg(self, detector):
        """strongest_iceberg should return the one with highest confidence."""
        # Create one iceberg at 0.50 with 3 refills
        for _ in range(3):
            detector.on_level_change("BUY", 0.50, 100.0, 0.0)
            detector.on_level_change("BUY", 0.50, 0.0, 100.0)

        strongest = detector.strongest_iceberg("BUY")
        assert strongest is not None
        assert strongest.price == 0.50

    def test_different_prices_tracked_independently(self, detector):
        """Refills at different prices should be tracked separately."""
        # 2 refills at 0.50 — not enough
        for _ in range(2):
            detector.on_level_change("BUY", 0.50, 100.0, 0.0)
            detector.on_level_change("BUY", 0.50, 0.0, 100.0)

        # 3 refills at 0.45 — should trigger
        for _ in range(3):
            detector.on_level_change("BUY", 0.45, 80.0, 0.0)
            result = detector.on_level_change("BUY", 0.45, 0.0, 80.0)

        assert result is not None
        assert result.price == 0.45
        # The 0.50 level should not be an iceberg (only 2 refills)
        icebergs = detector.active_icebergs
        assert (("BUY", 0.50) not in icebergs) or icebergs[("BUY", 0.50)].refill_count < 3
