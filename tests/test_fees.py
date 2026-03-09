"""Tests for src.trading.fees — dynamic fee curve utilities."""

from __future__ import annotations

import pytest

from src.trading.fees import (
    compute_adaptive_stop_loss_cents,
    compute_adaptive_trailing_offset_cents,
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


# ── Pillar 11.1: Volatility-Adaptive Stop-Loss Invariants ─────────────


class TestVolAdaptiveStopLossInvariants:
    """Mathematical proofs for the volatility-adaptive stop-loss engine."""

    # ── Anti-Shrinkage Invariant ──────────────────────────────────────
    def test_low_vol_never_shrinks_below_base(self):
        """When ewma_vol < ref_vol, multiplier must be exactly 1.0."""
        base = 4.0
        # Low vol: 0.3 < ref_vol 0.7 → multiplier clamped at 1.0
        with_low_vol = compute_adaptive_stop_loss_cents(
            base, 0.50, fee_enabled=False,
            ewma_vol=0.30, ref_vol=0.70, is_adaptive=True, max_multiplier=1.5,
        )
        without_vol = compute_adaptive_stop_loss_cents(
            base, 0.50, fee_enabled=False,
            ewma_vol=None, ref_vol=0.70, is_adaptive=True, max_multiplier=1.5,
        )
        assert with_low_vol == without_vol == base

    def test_anti_shrinkage_at_half_ref(self):
        """Even at 50% of ref vol, the SL never goes below the base."""
        base = 6.0
        result = compute_adaptive_stop_loss_cents(
            base, 0.40, fee_enabled=False,
            ewma_vol=0.35, ref_vol=0.70, is_adaptive=True, max_multiplier=2.0,
        )
        assert result == base

    # ── Max Clamp Invariant ───────────────────────────────────────────
    def test_extreme_vol_clamps_at_max_multiplier(self):
        """When ewma_vol >> ref_vol, multiplier clamps at max_multiplier."""
        base = 4.0
        max_mult = 1.5
        # 5× ref_vol → vol_ratio=5.0, should clamp to 1.5
        result = compute_adaptive_stop_loss_cents(
            base, 0.50, fee_enabled=False,
            ewma_vol=3.50, ref_vol=0.70, is_adaptive=True, max_multiplier=max_mult,
        )
        assert result == base * max_mult  # 6.0

    def test_clamp_at_custom_max(self):
        """Max multiplier of 2.0 means the stop can at most double."""
        base = 5.0
        result = compute_adaptive_stop_loss_cents(
            base, 0.50, fee_enabled=False,
            ewma_vol=10.0, ref_vol=0.70, is_adaptive=True, max_multiplier=2.0,
        )
        assert result == base * 2.0  # 10.0

    # ── Missing Data Invariant ────────────────────────────────────────
    def test_none_vol_defaults_to_unit_multiplier(self):
        """Cold-start (ewma_vol=None) → multiplier=1.0."""
        base = 4.0
        result = compute_adaptive_stop_loss_cents(
            base, 0.50, fee_enabled=False,
            ewma_vol=None, ref_vol=0.70, is_adaptive=True, max_multiplier=1.5,
        )
        assert result == base

    def test_zero_vol_defaults_to_unit_multiplier(self):
        """Zero volatility → multiplier=1.0."""
        base = 4.0
        result = compute_adaptive_stop_loss_cents(
            base, 0.50, fee_enabled=False,
            ewma_vol=0.0, ref_vol=0.70, is_adaptive=True, max_multiplier=1.5,
        )
        assert result == base

    def test_adaptive_disabled_defaults_to_unit_multiplier(self):
        """is_adaptive=False ignores high vol, uses multiplier=1.0."""
        base = 4.0
        result = compute_adaptive_stop_loss_cents(
            base, 0.50, fee_enabled=False,
            ewma_vol=2.0, ref_vol=0.70, is_adaptive=False, max_multiplier=1.5,
        )
        assert result == base

    # ── Order of Operations Invariant ─────────────────────────────────
    def test_multiplier_applied_before_fee_deduction(self):
        """Prove: SL_trigger = (base × multiplier) - fees, not base - fees then ×."""
        base = 8.0
        entry = 0.50
        ewma_vol = 1.05  # 1.5× ref → multiplier = 1.5
        ref_vol = 0.70
        max_mult = 1.5

        # Expected: stretch first, then deduct fees
        stretched = base * max_mult  # 12.0
        exit_est = max(0.01, entry - stretched / 100.0)
        fee_drag = compute_roundtrip_fee_cents(entry, exit_est, fee_enabled=True, f_max=0.0156)
        expected = max(1.0, round(stretched - fee_drag, 2))

        actual = compute_adaptive_stop_loss_cents(
            base, entry, fee_enabled=True, f_max=0.0156,
            ewma_vol=ewma_vol, ref_vol=ref_vol, is_adaptive=True, max_multiplier=max_mult,
        )
        assert abs(actual - expected) < 0.01

        # Prove the WRONG order (fees first, then multiply) gives different result
        no_vol_sl = compute_adaptive_stop_loss_cents(
            base, entry, fee_enabled=True, f_max=0.0156,
            ewma_vol=None, ref_vol=ref_vol, is_adaptive=True, max_multiplier=max_mult,
        )
        wrong_order = no_vol_sl * max_mult  # multiply AFTER fee deduction
        assert abs(actual - wrong_order) > 0.01  # they must differ

    def test_stretch_then_fee_with_fees_enabled(self):
        """With fees, stretched SL is reduced by fee drag; verify exact chain."""
        base = 4.0
        entry = 0.40
        ewma_vol = 0.84  # 1.2× ref
        ref_vol = 0.70
        max_mult = 1.5

        multiplier = max(1.0, min(ewma_vol / ref_vol, max_mult))  # 1.2
        stretched = base * multiplier  # 4.8
        exit_est = max(0.01, entry - stretched / 100.0)
        fee_drag = compute_roundtrip_fee_cents(entry, exit_est, fee_enabled=True, f_max=0.0156)
        expected = max(1.0, round(stretched - fee_drag, 2))

        actual = compute_adaptive_stop_loss_cents(
            base, entry, fee_enabled=True, f_max=0.0156,
            ewma_vol=ewma_vol, ref_vol=ref_vol, is_adaptive=True, max_multiplier=max_mult,
        )
        assert abs(actual - expected) < 0.01
        # The stretched value must be > base (proves stretch happened)
        assert stretched > base


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


# ── Pillar 11.2: Adaptive Trailing Offset Invariants ─────────────────────


class TestAdaptiveTrailingOffsetInvariants:
    """Proofs for the volatility-adaptive trailing stop offset."""

    def test_low_vol_never_shrinks(self):
        """When downside vol < ref_vol, offset stays at base."""
        base = 3.0
        result = compute_adaptive_trailing_offset_cents(
            base, ewma_downside_vol=0.30, ref_vol=0.70,
            is_adaptive=True, max_multiplier=1.5,
        )
        assert result == base

    def test_high_vol_stretches(self):
        """When downside vol = 1.4 (2× ref_vol 0.7), offset doubles."""
        base = 3.0
        result = compute_adaptive_trailing_offset_cents(
            base, ewma_downside_vol=1.40, ref_vol=0.70,
            is_adaptive=True, max_multiplier=2.0,
        )
        assert result == base * 2.0

    def test_clamp_at_max_multiplier(self):
        """Extreme vol clamps at max_multiplier."""
        base = 3.0
        result = compute_adaptive_trailing_offset_cents(
            base, ewma_downside_vol=5.0, ref_vol=0.70,
            is_adaptive=True, max_multiplier=1.5,
        )
        assert result == base * 1.5

    def test_none_vol_defaults_to_base(self):
        """Cold-start: None vol → multiplier=1.0."""
        base = 3.0
        result = compute_adaptive_trailing_offset_cents(
            base, ewma_downside_vol=None, ref_vol=0.70,
            is_adaptive=True, max_multiplier=1.5,
        )
        assert result == base

    def test_zero_vol_defaults_to_base(self):
        """Zero downside vol → multiplier=1.0."""
        base = 3.0
        result = compute_adaptive_trailing_offset_cents(
            base, ewma_downside_vol=0.0, ref_vol=0.70,
            is_adaptive=True, max_multiplier=1.5,
        )
        assert result == base

    def test_adaptive_disabled(self):
        """is_adaptive=False ignores vol, returns base."""
        base = 3.0
        result = compute_adaptive_trailing_offset_cents(
            base, ewma_downside_vol=2.0, ref_vol=0.70,
            is_adaptive=False, max_multiplier=1.5,
        )
        assert result == base

    def test_no_fee_deduction(self):
        """Trailing offset does NOT subtract fees (unlike baseline SL).

        Structural proof: at the same vol multiplier, baseline SL applies
        fee deduction (result < base * mult), but trailing offset returns
        exactly base * multiplier with no reduction.
        """
        base = 3.0
        entry = 0.50
        ewma = 1.05  # 1.05/0.70 = 1.5 → clamped at 1.5
        ref = 0.70
        max_mult = 1.5

        trailing = compute_adaptive_trailing_offset_cents(
            base, ewma_downside_vol=ewma, ref_vol=ref,
            is_adaptive=True, max_multiplier=max_mult,
        )
        # Trailing offset is exactly base * multiplier — no fee deduction
        assert trailing == base * max_mult  # 4.5

        # Contrast: baseline SL at same multiplier is LESS due to fee drag
        baseline_sl = compute_adaptive_stop_loss_cents(
            base, entry, fee_enabled=True, f_max=0.0156,
            ewma_vol=ewma, ref_vol=ref,
            is_adaptive=True, max_multiplier=max_mult,
        )
        assert baseline_sl < trailing  # fees make SL strictly smaller
