"""Tests for src.trading.fees — dynamic fee curve utilities."""

from __future__ import annotations

import pytest

from src.trading.fees import (
    compute_adaptive_stop_loss_cents,
    compute_net_pnl_cents,
    compute_roundtrip_fee_cents,
    get_fee_rate,
)


# ── get_fee_rate ──────────────────────────────────────────────────────────


class TestGetFeeRate:
    """Test the dynamic fee curve Fee(p) = f_max × 4 × p × (1 − p)."""

    def test_peak_at_midpoint(self):
        """Maximum fee at p = 0.50."""
        rate = get_fee_rate(0.50, f_max=0.0156)
        assert abs(rate - 0.0156) < 1e-6

    def test_zero_at_boundaries(self):
        assert get_fee_rate(0.0, f_max=0.0156) == 0.0
        assert get_fee_rate(1.0, f_max=0.0156) == 0.0

    def test_symmetric(self):
        """Fee(0.3) == Fee(0.7)."""
        assert abs(get_fee_rate(0.3, f_max=0.0156) - get_fee_rate(0.7, f_max=0.0156)) < 1e-10

    def test_fee_disabled(self):
        assert get_fee_rate(0.50, fee_enabled=False, f_max=0.0156) == 0.0

    def test_near_zero_price(self):
        rate = get_fee_rate(0.01, f_max=0.0156)
        assert rate > 0
        assert rate < 0.001  # very low fee near extremes

    def test_typical_values(self):
        """Spot-check known values: Fee(0.25) = 0.0156 × 4 × 0.25 × 0.75 = 0.0117."""
        rate = get_fee_rate(0.25, f_max=0.0156)
        assert abs(rate - 0.0117) < 1e-4

    def test_monotonic_to_peak(self):
        """Fee increases from 0 to 0.5 then decreases from 0.5 to 1."""
        lo = get_fee_rate(0.1, f_max=0.0156)
        mid = get_fee_rate(0.3, f_max=0.0156)
        peak = get_fee_rate(0.5, f_max=0.0156)
        assert lo < mid < peak


# ── compute_roundtrip_fee_cents ───────────────────────────────────────────


class TestRoundtripFeeCents:
    def test_symmetric_roundtrip(self):
        """Entry at 0.50, exit at 0.50 → 2 × 1.56% × 100 = 3.12¢."""
        rt = compute_roundtrip_fee_cents(0.50, 0.50, fee_enabled=True, f_max=0.0156)
        assert abs(rt - 3.12) < 0.01

    def test_disabled(self):
        rt = compute_roundtrip_fee_cents(0.50, 0.50, fee_enabled=False)
        assert rt == 0.0

    def test_asymmetric(self):
        """Different entry/exit prices → sum of individual fee rates × 100."""
        rt = compute_roundtrip_fee_cents(0.30, 0.60, fee_enabled=True, f_max=0.0156)
        expected = (get_fee_rate(0.30, f_max=0.0156) + get_fee_rate(0.60, f_max=0.0156)) * 100
        assert abs(rt - expected) < 1e-6

    def test_edge_entry_zero(self):
        rt = compute_roundtrip_fee_cents(0.0, 0.50, fee_enabled=True, f_max=0.0156)
        # entry fee is 0 (price=0), exit fee is 1.56
        assert abs(rt - 1.56) < 0.01


# ── compute_adaptive_stop_loss_cents ──────────────────────────────────────


class TestAdaptiveStopLoss:
    def test_basic_tightening(self):
        """SL at midpoint should be tighter than the raw SL_base."""
        raw = 8.0
        adaptive = compute_adaptive_stop_loss_cents(raw, 0.50, fee_enabled=True, f_max=0.0156)
        assert adaptive < raw

    def test_floor_at_1_cent(self):
        """Even with huge fee drag, SL never goes below 1.0¢."""
        adaptive = compute_adaptive_stop_loss_cents(2.0, 0.50, fee_enabled=True, f_max=0.0156)
        assert adaptive >= 1.0

    def test_no_tightening_when_disabled(self):
        raw = 8.0
        adaptive = compute_adaptive_stop_loss_cents(raw, 0.50, fee_enabled=False)
        assert adaptive == raw

    def test_less_tightening_at_extremes(self):
        """Fees are lower at extreme prices → less SL tightening."""
        mid_sl = compute_adaptive_stop_loss_cents(8.0, 0.50, fee_enabled=True, f_max=0.0156)
        edge_sl = compute_adaptive_stop_loss_cents(8.0, 0.10, fee_enabled=True, f_max=0.0156)
        assert edge_sl > mid_sl  # less tightening at p=0.10

    def test_exact_calculation(self):
        """Verify formula: SL_trigger = SL_base - roundtrip_fee."""
        sl_base = 8.0
        entry = 0.50
        exit_est = max(0.01, entry - sl_base / 100.0)
        fee_drag = compute_roundtrip_fee_cents(entry, exit_est, fee_enabled=True, f_max=0.0156)
        expected = max(1.0, round(sl_base - fee_drag, 2))
        actual = compute_adaptive_stop_loss_cents(sl_base, entry, fee_enabled=True, f_max=0.0156)
        assert abs(actual - expected) < 0.01


# ── compute_net_pnl_cents ────────────────────────────────────────────────


class TestNetPnlCents:
    def test_breakeven_minus_fees(self):
        """Entry=exit → PnL = -(entry_fee + exit_fee) × size × 100."""
        pnl = compute_net_pnl_cents(0.50, 0.50, size=100, fee_enabled=True, f_max=0.0156)
        assert pnl < 0  # fees eat into PnL

    def test_profitable_after_fees(self):
        """Enough spread to overcome round-trip fee."""
        pnl = compute_net_pnl_cents(0.40, 0.60, size=100, fee_enabled=True, f_max=0.0156)
        assert pnl > 0

    def test_no_fees(self):
        """Without fees, PnL is pure price difference."""
        pnl = compute_net_pnl_cents(0.40, 0.50, size=100, fee_enabled=False)
        expected = (0.50 - 0.40) * 100 * 100.0  # = 1000.0 cents
        assert abs(pnl - expected) < 0.01

    def test_loss(self):
        """Exit below entry → negative PnL (even more negative with fees)."""
        pnl_no_fee = compute_net_pnl_cents(0.50, 0.45, size=100, fee_enabled=False)
        pnl_fee = compute_net_pnl_cents(0.50, 0.45, size=100, fee_enabled=True, f_max=0.0156)
        assert pnl_no_fee < 0
        assert pnl_fee < pnl_no_fee  # fees make the loss worse

    def test_zero_size(self):
        pnl = compute_net_pnl_cents(0.40, 0.50, size=0, fee_enabled=True, f_max=0.0156)
        assert pnl == 0.0
