"""
Tests for Pillar 15.1: Covariance-Adjusted Kelly (MCPV penalty).

Verifies that correlated positions receive smaller Kelly sizes via the
compute_mcpv_penalty mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.trading.sizer import compute_kelly_size, compute_mcpv_penalty


# ── Helpers ──────────────────────────────────────────────────────────────


@dataclass
class _FakePosition:
    market_id: str
    entry_price: float
    entry_size: float


def _make_pce(corr_value: float = 0.0) -> MagicMock:
    """Create a mock PortfolioCorrelationEngine with fixed pairwise corr."""
    pce = MagicMock()
    pce.corr_matrix.get.return_value = corr_value
    return pce


# ── Test compute_mcpv_penalty ────────────────────────────────────────────


class TestMCPVPenalty:
    """Direct unit tests for the penalty multiplier function."""

    def test_no_positions_returns_one(self):
        pce = _make_pce(0.5)
        assert compute_mcpv_penalty(pce, "mkt_a", []) == 1.0

    def test_self_only_position_returns_one(self):
        """If the only open position is in the same market, no penalty."""
        pce = _make_pce(1.0)
        pos = _FakePosition(market_id="mkt_a", entry_price=0.5, entry_size=10)
        result = compute_mcpv_penalty(pce, "mkt_a", [pos])
        assert result == 1.0

    def test_uncorrelated_returns_one(self):
        pce = _make_pce(0.0)
        pos = _FakePosition(market_id="mkt_b", entry_price=0.5, entry_size=10)
        result = compute_mcpv_penalty(pce, "mkt_a", [pos])
        assert result == 1.0

    def test_below_threshold_returns_one(self):
        """ρ = 0.5 is below the 0.65 default threshold → no penalty."""
        pce = _make_pce(0.5)
        pos = _FakePosition(market_id="mkt_b", entry_price=0.5, entry_size=10)
        result = compute_mcpv_penalty(pce, "mkt_a", [pos])
        assert result == 1.0

    def test_high_correlation_reduces_penalty(self):
        pce = _make_pce(0.9)
        pos = _FakePosition(market_id="mkt_b", entry_price=0.5, entry_size=20)
        result = compute_mcpv_penalty(pce, "mkt_a", [pos])
        # penalty_mult = max(0.05, 1.0 - 0.9) = 0.1
        assert result == pytest.approx(0.1, abs=0.01)

    def test_max_correlation_hits_floor(self):
        pce = _make_pce(1.0)
        pos = _FakePosition(market_id="mkt_b", entry_price=0.5, entry_size=20)
        result = compute_mcpv_penalty(pce, "mkt_a", [pos])
        assert result == pytest.approx(0.05, abs=0.001)

    def test_moderate_correlation(self):
        """ρ = 0.7 exceeds the 0.65 threshold → penalty applies."""
        pce = _make_pce(0.7)
        pos = _FakePosition(market_id="mkt_b", entry_price=0.4, entry_size=10)
        result = compute_mcpv_penalty(pce, "mkt_a", [pos])
        # penalty_mult = max(0.05, 1.0 - 0.7) = 0.3
        assert result == pytest.approx(0.3, abs=0.01)


# ── Test Kelly integration with MCPV ─────────────────────────────────────


class TestKellyWithMCPV:
    """Prove that a correlated second position gets a smaller Kelly size."""

    _KELLY_KWARGS = dict(
        signal_score=0.8,
        win_rate=0.60,
        avg_win_cents=6.0,
        avg_loss_cents=4.0,
        bankroll_usd=1000.0,
        entry_price=0.45,
        max_trade_usd=50.0,
        total_trades=100,
    )

    def test_correlated_position_gets_smaller_size(self):
        """Core assertion: high-ρ second position → smaller adj_fraction."""
        # Baseline: no PCE → full size
        baseline = compute_kelly_size(**self._KELLY_KWARGS)
        assert baseline.size_usd > 0
        assert baseline.mcpv_penalty == 0.0

        # With high correlation (ρ=0.9) → should shrink
        pce = _make_pce(0.9)
        peer = _FakePosition(market_id="mkt_b", entry_price=0.5, entry_size=20)
        correlated = compute_kelly_size(
            **self._KELLY_KWARGS,
            pce=pce,
            proposed_market_id="mkt_a",
            open_positions=[peer],
        )
        assert correlated.size_usd > 0  # still trades
        assert correlated.size_usd < baseline.size_usd
        assert correlated.mcpv_penalty > 0.0
        assert correlated.adj_fraction < baseline.adj_fraction

    def test_uncorrelated_no_penalty(self):
        pce = _make_pce(0.0)
        peer = _FakePosition(market_id="mkt_b", entry_price=0.5, entry_size=20)
        result = compute_kelly_size(
            **self._KELLY_KWARGS,
            pce=pce,
            proposed_market_id="mkt_a",
            open_positions=[peer],
        )
        baseline = compute_kelly_size(**self._KELLY_KWARGS)
        assert result.size_usd == baseline.size_usd
        assert result.mcpv_penalty == 0.0

    def test_no_pce_no_penalty(self):
        """When PCE is None, no penalty is applied."""
        result = compute_kelly_size(**self._KELLY_KWARGS)
        assert result.mcpv_penalty == 0.0

    def test_correlated_eth_trade_smaller_with_btc_open(self):
        """Prove: a correlated ETH trade is sized smaller when a BTC position
        is already open (ρ = 0.80 > threshold 0.65)."""
        baseline = compute_kelly_size(**self._KELLY_KWARGS)

        pce = _make_pce(0.80)
        btc_position = _FakePosition(
            market_id="BTC_ABOVE_100K",
            entry_price=0.55,
            entry_size=30,
        )
        eth_result = compute_kelly_size(
            **self._KELLY_KWARGS,
            pce=pce,
            proposed_market_id="ETH_ABOVE_5K",
            open_positions=[btc_position],
        )
        assert eth_result.size_usd > 0
        assert eth_result.size_usd < baseline.size_usd
        assert eth_result.mcpv_penalty > 0.0
        assert eth_result.adj_fraction < baseline.adj_fraction
