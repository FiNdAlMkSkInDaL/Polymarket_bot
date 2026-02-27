"""
Tests for the dynamic take-profit calculator.
"""

import pytest

from src.trading.take_profit import compute_take_profit, TakeProfitResult


class TestTakeProfit:
    def test_basic_target_calculation(self):
        """Default α = 0.5 → target halfway between entry and VWAP."""
        result = compute_take_profit(entry_price=0.47, no_vwap=0.65)
        assert result.viable is True
        # target = 0.47 + 0.5 * (0.65 - 0.47) = 0.47 + 0.09 = 0.56
        assert result.target_price == pytest.approx(0.56, abs=0.02)
        assert result.alpha == pytest.approx(0.50, abs=0.05)
        assert result.spread_cents > 4.0

    def test_high_volatility_reduces_alpha(self):
        """High σ should lower α (exit sooner)."""
        baseline = compute_take_profit(entry_price=0.47, no_vwap=0.65, realised_vol=0.02)
        high_vol = compute_take_profit(entry_price=0.47, no_vwap=0.65, realised_vol=0.06)
        assert high_vol.alpha < baseline.alpha
        assert high_vol.target_price < baseline.target_price

    def test_whale_confluence_increases_alpha(self):
        """Whale confirmation should push α higher (more aggressive target)."""
        no_whale = compute_take_profit(entry_price=0.47, no_vwap=0.65, whale_confluence=False)
        whale = compute_take_profit(entry_price=0.47, no_vwap=0.65, whale_confluence=True)
        assert whale.alpha > no_whale.alpha
        assert whale.target_price > no_whale.target_price

    def test_near_resolution_reduces_alpha(self):
        """Close to resolution → lower α (less mean reversion expected)."""
        far = compute_take_profit(entry_price=0.47, no_vwap=0.65, days_to_resolution=30)
        near = compute_take_profit(entry_price=0.47, no_vwap=0.65, days_to_resolution=3)
        assert near.alpha < far.alpha

    def test_not_viable_when_spread_too_small(self):
        """If the VWAP is very close to entry, spread < 4¢ → not viable."""
        result = compute_take_profit(entry_price=0.63, no_vwap=0.65)
        # target ≈ 0.63 + 0.5 * 0.02 = 0.64, spread = 1¢ < 4¢
        assert result.viable is False

    def test_alpha_clamped_to_bounds(self):
        """α should never exceed [0.3, 0.7] regardless of adjustments."""
        # Stack all upward adjustments
        result = compute_take_profit(
            entry_price=0.30,
            no_vwap=0.80,
            realised_vol=0.001,  # very low vol → pushes α up
            book_depth_ratio=5.0,  # deep book → pushes α up
            whale_confluence=True,  # whale → pushes α up
            days_to_resolution=60,  # far out → no reduction
        )
        assert result.alpha <= 0.70

        # Stack all downward adjustments
        result2 = compute_take_profit(
            entry_price=0.30,
            no_vwap=0.80,
            realised_vol=0.10,  # very high vol → pushes α down
            book_depth_ratio=0.1,
            whale_confluence=False,
            days_to_resolution=1,  # very close → pushes α down
        )
        assert result2.alpha >= 0.30

    def test_edge_case_vwap_below_entry(self):
        """If VWAP < entry (rare), should still produce a viable minimum spread."""
        result = compute_take_profit(entry_price=0.70, no_vwap=0.60)
        assert result.target_price > result.entry_price
        assert result.alpha == 0.30  # falls to alpha_min
