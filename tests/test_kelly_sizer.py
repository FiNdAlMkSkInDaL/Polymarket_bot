"""
Tests for Kelly criterion position sizing.
"""

import pytest

from src.trading.sizer import KellyResult, compute_kelly_size


class TestKellySizing:
    """Core Kelly formula and edge-case tests."""

    def test_positive_edge_returns_nonzero_size(self):
        result = compute_kelly_size(
            signal_score=0.8,
            win_rate=0.60,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=50.0,
            total_trades=100,
        )
        assert result.size_usd > 0
        assert result.size_shares > 0
        assert result.method == "kelly"
        assert result.edge > 0
        assert result.win_prob > 0.5

    def test_negative_edge_returns_zero(self):
        result = compute_kelly_size(
            signal_score=0.0,
            win_rate=0.30,  # low win rate
            avg_win_cents=3.0,
            avg_loss_cents=7.0,  # bad payoff
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=50.0,
            total_trades=100,
        )
        assert result.size_usd == 0.0
        assert result.size_shares == 0.0
        assert result.method == "kelly_no_edge"
        assert result.edge <= 0

    def test_size_capped_by_max_trade(self):
        result = compute_kelly_size(
            signal_score=1.0,
            win_rate=0.80,
            avg_win_cents=10.0,
            avg_loss_cents=2.0,
            bankroll_usd=100_000.0,  # large bankroll
            entry_price=0.45,
            max_trade_usd=20.0,  # small cap
            total_trades=100,
        )
        assert result.size_usd <= 20.0

    def test_size_capped_by_kelly_max_pct(self):
        result = compute_kelly_size(
            signal_score=1.0,
            win_rate=0.90,
            avg_win_cents=20.0,
            avg_loss_cents=2.0,
            bankroll_usd=100.0,
            entry_price=0.45,
            max_trade_usd=500.0,
            max_kelly_pct=5.0,  # 5% of 100 = $5
            total_trades=100,
        )
        assert result.size_usd <= 5.0

    def test_fractional_kelly_reduces_size(self):
        full = compute_kelly_size(
            signal_score=0.5,
            win_rate=0.60,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=200.0,
            kelly_fraction_mult=1.0,
            max_kelly_pct=100.0,
            total_trades=100,
        )
        quarter = compute_kelly_size(
            signal_score=0.5,
            win_rate=0.60,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=200.0,
            kelly_fraction_mult=0.25,
            max_kelly_pct=100.0,
            total_trades=100,
        )
        assert quarter.adj_fraction < full.adj_fraction
        # Quarter-Kelly should produce smaller or equal size
        assert quarter.size_usd <= full.size_usd + 0.01

    def test_higher_signal_gives_larger_size(self):
        low = compute_kelly_size(
            signal_score=0.2,
            win_rate=0.55,
            avg_win_cents=5.0,
            avg_loss_cents=5.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            kelly_fraction_mult=0.5,
            max_kelly_pct=50.0,
            total_trades=100,
        )
        high = compute_kelly_size(
            signal_score=0.9,
            win_rate=0.55,
            avg_win_cents=5.0,
            avg_loss_cents=5.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            kelly_fraction_mult=0.5,
            max_kelly_pct=50.0,
            total_trades=100,
        )
        # Higher signal → higher win_prob → higher edge → larger size
        assert high.edge >= low.edge
        assert high.win_prob > low.win_prob

    def test_zero_entry_price_returns_zero(self):
        result = compute_kelly_size(
            signal_score=0.8,
            win_rate=0.60,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.0,
            max_trade_usd=50.0,
        )
        assert result.size_shares == 0.0

    def test_defaults_when_no_history(self):
        """When win_rate=0 and avg values=0, should use fallback defaults."""
        result = compute_kelly_size(
            signal_score=0.5,
            win_rate=0.0,
            avg_win_cents=0.0,
            avg_loss_cents=0.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=50.0,
            total_trades=100,
        )
        # Should not crash; defaults to 0.55 WR, 5/5 payoff
        assert result.win_prob > 0
        assert result.method in ("kelly", "kelly_no_edge")

    def test_cold_start_bypasses_kelly(self):
        """When total_trades < MIN_KELLY_TRADES, use cold-start sizing."""
        result = compute_kelly_size(
            signal_score=0.5,
            win_rate=0.0,
            avg_win_cents=0.0,
            avg_loss_cents=-12.58,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=50.0,
            total_trades=1,  # below threshold
        )
        assert result.method == "kelly_cold_start"
        assert result.size_usd > 0  # should NOT be blocked

    def test_kelly_fraction_field(self):
        result = compute_kelly_size(
            signal_score=0.7,
            win_rate=0.60,
            avg_win_cents=8.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            kelly_fraction_mult=0.5,
            max_kelly_pct=100.0,
            total_trades=100,
        )
        # adj_fraction should be ~0.5 * full_kelly
        assert result.adj_fraction == pytest.approx(
            result.kelly_fraction * 0.5, abs=0.001
        )


class TestKellyEdgeDiscounting:
    """Tests for p-cap, edge discounting, and uncertainty penalty."""

    def test_p_capped_at_kelly_p_cap(self):
        """Even with perfect signal, estimated_p should not exceed p_cap."""
        result = compute_kelly_size(
            signal_score=1.0,
            win_rate=0.99,
            avg_win_cents=10.0,
            avg_loss_cents=2.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            total_trades=100,
        )
        # Default p_cap is 0.85 — estimated_p should be clamped
        assert result.estimated_p <= 0.85 + 1e-9

    def test_uncertainty_penalty_shrinks_p_toward_half(self):
        """High uncertainty should pull adjusted_p closer to 0.5."""
        no_penalty = compute_kelly_size(
            signal_score=0.8,
            win_rate=0.65,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            signal_metadata={"uncertainty_penalty": 0.0},
            total_trades=100,
        )
        high_penalty = compute_kelly_size(
            signal_score=0.8,
            win_rate=0.65,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            signal_metadata={"uncertainty_penalty": 0.8},
            total_trades=100,
        )
        # Higher uncertainty → adjusted_p closer to 0.5 → less edge → smaller size
        assert high_penalty.adjusted_p < no_penalty.adjusted_p
        assert abs(high_penalty.adjusted_p - 0.5) < abs(no_penalty.adjusted_p - 0.5)
        assert high_penalty.size_usd <= no_penalty.size_usd

    def test_full_uncertainty_collapses_edge(self):
        """Uncertainty=1.0 should set adjusted_p = 0.5 → zero edge."""
        result = compute_kelly_size(
            signal_score=0.8,
            win_rate=0.65,
            avg_win_cents=5.0,
            avg_loss_cents=5.0,  # symmetric payoff
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            signal_metadata={"uncertainty_penalty": 1.0},
            total_trades=100,
        )
        assert result.adjusted_p == pytest.approx(0.5, abs=1e-9)
        assert result.method == "kelly_no_edge"

    def test_no_signal_metadata_uses_default_uncertainty(self):
        """When signal_metadata is None, default uncertainty (0.5) is used."""
        result = compute_kelly_size(
            signal_score=0.7,
            win_rate=0.60,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            signal_metadata=None,
            total_trades=100,
        )
        # Default uncertainty = 0.5 → adjusted_p should be between raw_p and 0.5
        assert result.uncertainty_penalty == pytest.approx(0.5, abs=1e-9)
        assert result.adjusted_p < result.estimated_p
        assert result.adjusted_p > 0.5

    def test_kelly_result_fields_populated(self):
        """All new KellyResult fields should be populated."""
        result = compute_kelly_size(
            signal_score=0.7,
            win_rate=0.60,
            avg_win_cents=6.0,
            avg_loss_cents=4.0,
            bankroll_usd=1000.0,
            entry_price=0.45,
            max_trade_usd=100.0,
            signal_metadata={"uncertainty_penalty": 0.3},
            total_trades=100,
        )
        assert result.estimated_p > 0
        assert result.adjusted_p > 0
        assert result.uncertainty_penalty == pytest.approx(0.3, abs=1e-9)
        assert result.adjusted_p <= result.estimated_p
