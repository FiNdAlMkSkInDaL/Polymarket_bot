"""Tests for src.data.market_scorer — composite quality scoring engine."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.data.market_scorer import (
    ScoreBreakdown,
    compute_score,
    score_liquidity,
    score_price_range,
    score_spread,
    score_time_to_resolution,
    score_trade_frequency,
    score_volume,
    score_whale_interest,
)


# ── Individual scorers ────────────────────────────────────────────────────

class TestScoreVolume:
    def test_zero_volume(self):
        assert score_volume(0) == 0.0

    def test_negative_volume(self):
        assert score_volume(-100) == 0.0

    def test_low_volume(self):
        # $500 → log10(500) ≈ 2.7 → (2.7-2)*25 = 17.5
        s = score_volume(500)
        assert 15.0 <= s <= 20.0

    def test_medium_volume(self):
        s = score_volume(50_000)
        assert 40.0 <= s <= 70.0

    def test_high_volume(self):
        s = score_volume(1_000_000)
        assert s == 100.0

    def test_monotonic_increasing(self):
        """Higher volume → higher score."""
        assert score_volume(100) < score_volume(10_000) < score_volume(500_000)


class TestScoreLiquidity:
    def test_zero(self):
        assert score_liquidity(0) == 0.0

    def test_high(self):
        assert score_liquidity(1_000_000) == 100.0

    def test_monotonic(self):
        assert score_liquidity(1_000) < score_liquidity(100_000)


class TestScoreSpread:
    def test_no_data(self):
        # spread_cents=0 → neutral score
        assert score_spread(0) == 50.0

    def test_tight_spread(self):
        assert score_spread(1.0) == 100.0

    def test_wide_spread(self):
        assert score_spread(10.0) == 0.0

    def test_medium_spread(self):
        s = score_spread(5.0)
        assert 40.0 <= s <= 70.0


class TestScoreTimeToResolution:
    def test_expired(self):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        assert score_time_to_resolution(past) == 0.0

    def test_optimal_range(self):
        optimal = datetime.now(timezone.utc) + timedelta(days=30)
        assert score_time_to_resolution(optimal) == 100.0

    def test_very_close(self):
        close = datetime.now(timezone.utc) + timedelta(days=1)
        assert score_time_to_resolution(close) == 10.0

    def test_none_perpetual(self):
        assert score_time_to_resolution(None) == 60.0

    def test_far_out(self):
        far = datetime.now(timezone.utc) + timedelta(days=365)
        s = score_time_to_resolution(far)
        assert s <= 50.0


class TestScorePriceRange:
    def test_mid_price_optimal(self):
        assert score_price_range(0.50) == 100.0

    def test_edges(self):
        assert score_price_range(0.15) == 100.0
        assert score_price_range(0.85) == 100.0

    def test_extreme_low(self):
        s = score_price_range(0.05)
        assert s < 50.0

    def test_no_data(self):
        assert score_price_range(0) == 50.0


class TestScoreTradeFrequency:
    def test_zero(self):
        assert score_trade_frequency(0) == 0.0

    def test_low(self):
        assert 30.0 <= score_trade_frequency(3) <= 40.0

    def test_capped(self):
        assert score_trade_frequency(20) == 100.0


class TestScoreWhale:
    def test_no_whale(self):
        assert score_whale_interest(False) == 0.0

    def test_whale(self):
        assert score_whale_interest(True) == 100.0


# ── Composite score ───────────────────────────────────────────────────────

class TestComputeScore:
    def test_default_returns_breakdown(self):
        bd = compute_score()
        assert isinstance(bd, ScoreBreakdown)
        assert 0 <= bd.total <= 100

    def test_high_quality_market(self):
        bd = compute_score(
            daily_volume_usd=200_000,
            liquidity_usd=100_000,
            spread_cents=1.0,
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
            mid_price=0.50,
            trades_per_minute=5,
            has_whale_activity=True,
        )
        assert bd.total >= 80

    def test_low_quality_market(self):
        bd = compute_score(
            daily_volume_usd=10,
            liquidity_usd=10,
            spread_cents=9.0,
            end_date=datetime.now(timezone.utc) + timedelta(days=365),
            mid_price=0.02,
            trades_per_minute=0,
            has_whale_activity=False,
        )
        assert bd.total <= 25

    def test_as_dict(self):
        bd = compute_score(daily_volume_usd=10_000, liquidity_usd=5_000)
        d = bd.as_dict()
        assert "total" in d
        assert "vol" in d
        assert "liq" in d
        assert all(isinstance(v, float) for v in d.values())

    def test_weights_sum_to_one(self):
        from src.data.market_scorer import _WEIGHTS
        assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-6


# ── MTI penalty (Pillar 9) ───────────────────────────────────────────────

class TestMTIPenalty:
    """Market Taker Intensity penalty tests."""

    def test_no_penalty_below_threshold(self):
        """MTI below threshold → no penalty → mti_penalty == 0."""
        bd = compute_score(
            daily_volume_usd=100_000,
            liquidity_usd=50_000,
            taker_count=50,
            total_count=100,  # MTI = 0.50, below default 0.80
        )
        assert bd.mti_penalty == 0.0

    def test_penalty_above_threshold(self):
        """MTI above threshold → penalty applied → lower score."""
        bd_clean = compute_score(
            daily_volume_usd=100_000,
            liquidity_usd=50_000,
            taker_count=0,
            total_count=100,
        )
        bd_toxic = compute_score(
            daily_volume_usd=100_000,
            liquidity_usd=50_000,
            taker_count=90,
            total_count=100,  # MTI = 0.90, above 0.80
        )
        assert bd_toxic.mti_penalty > 0
        assert bd_toxic.total < bd_clean.total

    def test_no_trades_no_penalty(self):
        """Zero total counts → no penalty (avoid division by zero)."""
        bd = compute_score(
            daily_volume_usd=100_000,
            liquidity_usd=50_000,
            taker_count=0,
            total_count=0,
        )
        assert bd.mti_penalty == 0.0

    def test_mti_in_as_dict(self):
        """MTI field appears in the score breakdown dict."""
        bd = compute_score(daily_volume_usd=10_000, liquidity_usd=5_000)
        d = bd.as_dict()
        assert "mti" in d

    def test_score_clamped_to_zero(self):
        """Even with extreme penalty, score floor is 0."""
        bd = compute_score(
            daily_volume_usd=10,
            liquidity_usd=10,
            taker_count=100,
            total_count=100,  # MTI = 1.0
        )
        assert bd.total >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Tail-market hard veto (Deliverable E)
# ═══════════════════════════════════════════════════════════════════════════

class TestTailMarketVeto:
    """Markets near 0 or 1 must receive a hard zero score."""

    def test_score_price_range_low_tail(self):
        """mid_price below rpe_tail_veto_threshold → 0."""
        assert score_price_range(0.05) == 0.0

    def test_score_price_range_high_tail(self):
        """mid_price above (1 - rpe_tail_veto_threshold) → 0."""
        assert score_price_range(0.95) == 0.0

    def test_score_price_range_at_threshold(self):
        """mid_price exactly at threshold boundary → 0.0 (strict <)."""
        from src.core.config import settings
        threshold = settings.strategy.rpe_tail_veto_threshold
        # Slightly below threshold → vetoed
        assert score_price_range(threshold - 0.001) == 0.0
        # Slightly above (1 - threshold) → vetoed
        assert score_price_range(1.0 - threshold + 0.001) == 0.0

    def test_score_price_range_mid_range_unaffected(self):
        """Mid-range prices are NOT vetoed."""
        assert score_price_range(0.50) == 100.0
        assert score_price_range(0.30) == 100.0

    def test_compute_score_low_tail_hard_zero(self):
        """compute_score with tail-market mid_price → total = 0."""
        bd = compute_score(
            daily_volume_usd=100_000,
            liquidity_usd=50_000,
            mid_price=0.03,
        )
        assert bd.total == 0.0
        assert bd.price_range == 0.0

    def test_compute_score_high_tail_hard_zero(self):
        """compute_score with high-tail mid_price → total = 0."""
        bd = compute_score(
            daily_volume_usd=100_000,
            liquidity_usd=50_000,
            mid_price=0.97,
        )
        assert bd.total == 0.0
        assert bd.price_range == 0.0

    def test_compute_score_mid_range_not_vetoed(self):
        """compute_score with mid-range price retains a positive score."""
        bd = compute_score(
            daily_volume_usd=100_000,
            liquidity_usd=50_000,
            mid_price=0.50,
        )
        assert bd.total > 0.0
        assert bd.price_range == 100.0
