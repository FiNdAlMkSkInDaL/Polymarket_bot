"""
Tests for the dynamic fee model — verifying f = f_max × 4 × p × (1-p).

Ensures the backtest matching engine's fee computation is consistent
with the production ``src.trading.fees.get_fee_rate`` at all price
boundaries and interior points.
"""

from __future__ import annotations

import pytest

from src.trading.fees import get_fee_rate


class TestFeeFormula:
    """Verify the Polymarket dynamic fee curve at key points."""

    def test_zero_at_boundary_0(self):
        assert get_fee_rate(0.0, fee_enabled=True, f_max=0.0156) == 0.0

    def test_zero_at_boundary_1(self):
        assert get_fee_rate(1.0, fee_enabled=True, f_max=0.0156) == 0.0

    def test_max_at_midprice(self):
        """At p=0.50, f = f_max × 4 × 0.25 = f_max = 1.56%."""
        rate = get_fee_rate(0.50, fee_enabled=True, f_max=0.0156)
        assert abs(rate - 0.0156) < 1e-10

    def test_symmetry(self):
        """Fee curve is symmetric: fee(p) == fee(1-p)."""
        for p in [0.1, 0.2, 0.3, 0.4, 0.45]:
            r1 = get_fee_rate(p, fee_enabled=True, f_max=0.0156)
            r2 = get_fee_rate(1.0 - p, fee_enabled=True, f_max=0.0156)
            assert abs(r1 - r2) < 1e-12, f"Asymmetry at p={p}"

    def test_monotone_increase_to_midpoint(self):
        """Fee should increase from 0 to 0.50."""
        prev = 0.0
        for p in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
            r = get_fee_rate(p, fee_enabled=True, f_max=0.0156)
            assert r >= prev, f"Not monotone at p={p}"
            prev = r

    def test_monotone_decrease_from_midpoint(self):
        """Fee should decrease from 0.50 to 1.0."""
        prev = get_fee_rate(0.50, fee_enabled=True, f_max=0.0156)
        for p in [0.60, 0.70, 0.80, 0.90, 0.99]:
            r = get_fee_rate(p, fee_enabled=True, f_max=0.0156)
            assert r <= prev, f"Not monotone at p={p}"
            prev = r

    def test_specific_values(self):
        """Spot-check specific price→fee mappings."""
        f_max = 0.0156
        test_cases = [
            (0.10, f_max * 4 * 0.10 * 0.90),   # 0.005616
            (0.25, f_max * 4 * 0.25 * 0.75),   # 0.0117
            (0.33, f_max * 4 * 0.33 * 0.67),   # ~0.01382
            (0.50, f_max),                       # 0.0156
            (0.75, f_max * 4 * 0.75 * 0.25),   # 0.0117
        ]
        for price, expected in test_cases:
            rate = get_fee_rate(price, fee_enabled=True, f_max=f_max)
            assert abs(rate - expected) < 1e-10, (
                f"At p={price}: got {rate}, expected {expected}"
            )

    def test_disabled_returns_zero(self):
        """When fee_enabled=False, fee rate is always 0."""
        for p in [0.0, 0.25, 0.50, 0.75, 1.0]:
            assert get_fee_rate(p, fee_enabled=False) == 0.0

    def test_custom_fmax(self):
        """Custom f_max is respected."""
        custom = 0.02
        rate = get_fee_rate(0.50, fee_enabled=True, f_max=custom)
        assert abs(rate - custom) < 1e-10

    def test_negative_price_returns_zero(self):
        """Edge case: negative price → 0."""
        assert get_fee_rate(-0.5, fee_enabled=True, f_max=0.0156) == 0.0

    def test_rate_never_exceeds_fmax(self):
        """Fee rate should never exceed f_max for any valid price."""
        f_max = 0.0156
        for i in range(1, 100):
            p = i / 100.0
            rate = get_fee_rate(p, fee_enabled=True, f_max=f_max)
            assert rate <= f_max + 1e-12, f"Exceeded f_max at p={p}"
