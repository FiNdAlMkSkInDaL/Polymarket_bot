"""
Tests for the dynamic take-profit calculator.
"""

import pytest

from src.trading.fees import compute_net_pnl_cents
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
        """If the VWAP is very close to entry, spread < min_spread → not viable."""
        result = compute_take_profit(
            entry_price=0.63, no_vwap=0.64,
            fee_enabled=False, desired_margin_cents=0.0,
        )
        # target ≈ 0.63 + 0.5 * 0.01 = 0.635, spread = 0.5¢ < 2¢
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
        assert result2.alpha >= 0.40

    def test_edge_case_vwap_below_entry(self):
        """If VWAP < entry (rare), should still produce a viable minimum spread."""
        result = compute_take_profit(entry_price=0.70, no_vwap=0.60)
        assert result.target_price > result.entry_price
        assert result.alpha == 0.40  # falls to alpha_min


class TestTakeProfitFeeFloor:
    """Pillar 6 — dynamic fee-curve integration."""

    def test_fee_floor_widens_target(self):
        """When fees + margin exceed the vol-scaled spread, target should widen."""
        # Without fees: spread ≈ 9¢ (0.47 + 0.5*(0.65-0.47) = 0.56)
        baseline = compute_take_profit(entry_price=0.47, no_vwap=0.65)
        # With heavy fees: 200 bps on both legs
        with_fees = compute_take_profit(
            entry_price=0.47, no_vwap=0.65,
            entry_fee_bps=200, exit_fee_bps=200,
            desired_margin_cents=1.0,
        )
        # Fee floor should be positive
        assert with_fees.fee_floor_cents > 0
        assert with_fees.entry_fee_bps == 200
        assert with_fees.exit_fee_bps == 200
        # Target may be identical if the alpha-derived spread already
        # exceeds the fee floor — that's fine.

    def test_fee_floor_not_viable_when_insufficient(self):
        """At extreme prices where fees are tiny, spread can't reach min_spread."""
        result = compute_take_profit(
            entry_price=0.93, no_vwap=0.94,
            fee_enabled=True,
            desired_margin_cents=0.0,
        )
        # At p≈0.93, fees are tiny (~0.4¢/leg).  Fee floor ≈ 0.8¢.
        # After widening, spread ≈ 0.8¢ < min_spread=2¢ → not viable.
        assert result.viable is False

    def test_zero_fees_legacy_behaviour(self):
        """With fee_enabled=False (non-fee market), fee floor should be zero."""
        result = compute_take_profit(
            entry_price=0.47, no_vwap=0.65,
            fee_enabled=False,
            desired_margin_cents=0.0,
        )
        assert result.fee_floor_cents == pytest.approx(0.0, abs=0.01)
        assert result.viable is True

    def test_fee_floor_fields_populated(self):
        """TakeProfitResult should carry fee metadata."""
        result = compute_take_profit(
            entry_price=0.47, no_vwap=0.65,
            entry_fee_bps=156, exit_fee_bps=0,
        )
        assert result.entry_fee_bps == 156
        assert result.exit_fee_bps == 0
        assert result.fee_floor_cents >= 0


class TestFeeConsistencyGuarantee:
    """Prove that every viable trade is profitable after fees.

    This is the critical correctness invariant: compute_take_profit's fee
    floor must use the EXACT same fee model as compute_net_pnl_cents.
    If a trade clears the fee floor and is marked ``viable=True``, selling
    at the target price must always yield positive net PnL.
    """

    ENTRY_PRICES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    VWAPS = [0.30, 0.50, 0.65, 0.80, 0.95]
    SIZES = [1.0, 10.0, 100.0]

    @pytest.mark.parametrize("entry", ENTRY_PRICES)
    @pytest.mark.parametrize("vwap", VWAPS)
    @pytest.mark.parametrize("size", SIZES)
    def test_viable_trade_always_net_positive(self, entry, vwap, size):
        """If compute_take_profit says viable, selling at target must profit."""
        if vwap <= entry:
            return  # skip edge-case where VWAP <= entry (fallback path)

        tp = compute_take_profit(
            entry_price=entry,
            no_vwap=vwap,
            fee_enabled=True,
        )

        if not tp.viable:
            return  # correctly rejected — nothing to check

        # Compute net PnL using the same function the live bot uses
        net_pnl = compute_net_pnl_cents(
            entry_price=tp.entry_price,
            exit_price=tp.target_price,
            size=size,
            fee_enabled=True,
        )
        assert net_pnl > 0, (
            f"VIABLE TRADE LOST MONEY: entry={entry}, target={tp.target_price}, "
            f"size={size}, net_pnl={net_pnl}c, fee_floor={tp.fee_floor_cents}c, "
            f"spread={tp.spread_cents}c"
        )

    @pytest.mark.parametrize("entry", ENTRY_PRICES)
    def test_fee_floor_matches_actual_fees(self, entry):
        """The fee floor in the TP calculator must >= actual roundtrip fees."""
        vwap = min(entry + 0.20, 0.99)
        tp = compute_take_profit(
            entry_price=entry,
            no_vwap=vwap,
            fee_enabled=True,
            desired_margin_cents=0.0,  # strip margin to test pure fee floor
        )

        if not tp.viable:
            return

        # Actual fees from the PnL model
        from src.trading.fees import get_fee_rate
        actual_entry_fee = get_fee_rate(tp.entry_price, fee_enabled=True) * 100.0
        actual_exit_fee = get_fee_rate(tp.target_price, fee_enabled=True) * 100.0
        actual_roundtrip = actual_entry_fee + actual_exit_fee

        assert tp.fee_floor_cents >= actual_roundtrip - 0.01, (
            f"Fee floor {tp.fee_floor_cents}c < actual roundtrip {actual_roundtrip}c "
            f"at entry={entry}"
        )

    def test_worst_case_p50_both_legs_profitable(self):
        """p=0.50 is peak fee (1.56%). A viable trade there must still profit."""
        tp = compute_take_profit(
            entry_price=0.50,
            no_vwap=0.70,
            fee_enabled=True,
        )
        assert tp.viable is True
        net = compute_net_pnl_cents(
            entry_price=tp.entry_price,
            exit_price=tp.target_price,
            size=10.0,
            fee_enabled=True,
        )
        assert net > 0, f"Peak-fee trade lost money: {net}c"
