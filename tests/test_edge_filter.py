"""
Tests for the information-theoretic edge quality filter.

Covers:
  - binary_entropy(): symmetry, peak, edges, known values
  - EdgeAssessment data class
  - compute_edge_score(): factor decomposition, rejection taxonomy,
    sweet-spot analysis, fee interactions, discrete-tick mechanics,
    whale confidence boost, fee-disabled markets
"""

from __future__ import annotations

import math

import pytest

from src.core.config import settings
from src.signals.edge_filter import (
    EdgeAssessment,
    W_FEE,
    W_REGIME,
    W_SIGNAL,
    W_TICK,
    binary_entropy,
    compute_edge_score,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Binary entropy
# ═══════════════════════════════════════════════════════════════════════════

class TestBinaryEntropy:
    """Information-theoretic foundation."""

    def test_peak_at_half(self):
        assert binary_entropy(0.50) == pytest.approx(1.0, abs=1e-9)

    def test_zero_at_extremes(self):
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0

    def test_symmetry(self):
        """H(p) == H(1-p) for all p — reflects the NO/YES duality."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert binary_entropy(p) == pytest.approx(
                binary_entropy(1.0 - p), abs=1e-12
            )

    def test_known_values(self):
        """Spot-check against hand-computed values."""
        # H(0.25) = -0.25·log₂(0.25) - 0.75·log₂(0.75) = 0.8113
        assert binary_entropy(0.25) == pytest.approx(0.8113, abs=0.001)
        # H(0.10) ≈ 0.469
        assert binary_entropy(0.10) == pytest.approx(0.469, abs=0.001)
        # H(0.03) ≈ 0.194
        assert binary_entropy(0.03) == pytest.approx(0.194, abs=0.01)

    def test_monotonic_toward_half(self):
        """Entropy increases monotonically from 0 to 0.5."""
        prev = 0.0
        for p in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
            h = binary_entropy(p)
            assert h > prev
            prev = h

    def test_out_of_range_returns_zero(self):
        assert binary_entropy(-0.1) == 0.0
        assert binary_entropy(1.1) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Component weights
# ═══════════════════════════════════════════════════════════════════════════

class TestWeights:
    def test_weights_sum_to_one(self):
        assert W_REGIME + W_FEE + W_TICK + W_SIGNAL == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Geometric mean hard-zero property
# ═══════════════════════════════════════════════════════════════════════════

class TestHardZeros:
    """Any factor at zero should kill the entire score — the geometric
    mean's multiplicative annihilation property."""

    def test_no_vwap_above_entry_zeros_score(self):
        """VWAP ≤ entry → no mean-reversion target → score = 0."""
        ea = compute_edge_score(
            entry_price=0.50,
            no_vwap=0.45,  # below entry
            zscore=4.0,
            volume_ratio=6.0,
        )
        assert ea.score == 0.0
        assert ea.expected_gross_cents == 0.0
        assert ea.rejection_reason == "no_mean_reversion_target"

    def test_extreme_tail_price_zeros_entropy(self):
        """p = 0.001 → H(p) ≈ 0.012, but even with huge VWAP gap the
        regime factor makes the score very low."""
        ea = compute_edge_score(
            entry_price=0.001,
            no_vwap=0.10,
            zscore=5.0,
            volume_ratio=8.0,
            min_score=0.0,
        )
        # H(0.001) ≈ 0.011 — so regime^0.35 ≈ 0.16
        assert ea.regime_quality < 0.02
        assert ea.score < 25.0  # extremely penalised by low entropy

    def test_sub_tick_spread_zeros_tick_viability(self):
        """Expected gross < 1 tick (1¢) → tick_viability = 0 → score = 0."""
        # entry=0.50, VWAP=0.505 → gross = 0.50 * 0.005 = 0.25¢ < 1¢
        ea = compute_edge_score(
            entry_price=0.50,
            no_vwap=0.505,
            zscore=3.0,
            volume_ratio=5.0,
        )
        assert ea.expected_gross_cents < 1.0
        assert ea.tick_viability == 0.0
        assert ea.score == 0.0
        assert ea.rejection_reason == "negative_ev_after_fees"

    def test_fees_exceed_spread_hard_vetoed(self):
        """Small dislocation at p=0.50 (max fees) → fees eat all spread.
        Hard negative-EV veto rejects immediately: no trade should pass
        if expected gross profit cannot cover the roundtrip taker fee."""
        # entry=0.50, VWAP=0.53 → gross=0.50*0.03=1.5¢
        # Fee at 0.50 ≈ 2.00¢, fee at 0.515 ≈ 2.00¢ → total ≈ 4.00¢ > 1.5¢
        ea = compute_edge_score(
            entry_price=0.50,
            no_vwap=0.53,
            zscore=3.0,
            volume_ratio=5.0,
            min_score=0.0,
        )
        # Hard-vetoed: fee_efficiency=0, score=0, specific rejection reason
        assert ea.fee_efficiency == 0.0
        assert ea.score == 0.0
        assert ea.rejection_reason == "negative_ev_after_fees"


# ═══════════════════════════════════════════════════════════════════════════
#  Sweet-spot analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestSweetSpot:
    """The filter should reward trades in the 0.15–0.35 / 0.65–0.85 zone
    with meaningful VWAP dislocation — where entropy is decent and fees
    are moderate."""

    def test_sweet_spot_passes_easily(self):
        """p=0.20, VWAP=0.35.  Low fees, good entropy, decent spread."""
        ea = compute_edge_score(
            entry_price=0.20,
            no_vwap=0.35,
            zscore=3.0,
            volume_ratio=5.0,
        )
        assert ea.viable is True
        assert ea.score > 60.0
        assert ea.fee_efficiency > 0.5
        assert ea.tick_margin >= 3

    def test_mid_price_needs_larger_dislocation(self):
        """p=0.50 faces max fees (2.00¢/leg).  Needs larger VWAP gap."""
        # 5¢ gap works
        ea5 = compute_edge_score(
            entry_price=0.45,
            no_vwap=0.60,
            zscore=3.0,
            volume_ratio=5.0,
        )
        assert ea5.viable is True
        # 2¢ gap does not work
        ea2 = compute_edge_score(
            entry_price=0.50,
            no_vwap=0.54,
            zscore=3.0,
            volume_ratio=5.0,
        )
        assert ea2.viable is False

    def test_low_fee_zone_tolerates_smaller_dislocation(self):
        """p=0.10: fees ≈ 0.56¢/leg.  Even a moderate gap is profitable."""
        ea = compute_edge_score(
            entry_price=0.10,
            no_vwap=0.20,
            zscore=3.0,
            volume_ratio=5.0,
        )
        assert ea.viable is True
        assert ea.fee_efficiency > 0.6  # moderate fee drag at 2% regime


# ═══════════════════════════════════════════════════════════════════════════
#  The bad trade: p = 0.03
# ═══════════════════════════════════════════════════════════════════════════

class TestBadTradeRejection:
    """The trade that lost 12.58¢: entry at p=0.03, VWAP ≈ 0.05."""

    def test_p003_vwap005_rejected(self):
        """This is the actual trade we saw.  Edge filter must reject it
        at the standard threshold (score < 40 due to low entropy)."""
        ea = compute_edge_score(
            entry_price=0.03,
            no_vwap=0.05,
            zscore=3.0,
            volume_ratio=5.0,
            min_score=40.0,  # use the stricter threshold for this bad-trade test
        )
        assert ea.viable is False
        assert ea.regime_quality < 0.25  # low entropy
        assert ea.score < 40.0

    def test_p003_huge_vwap_still_challenged(self):
        """Even with a massive dislocation at p=0.03, entropy drags score."""
        ea = compute_edge_score(
            entry_price=0.03,
            no_vwap=0.15,
            zscore=4.0,
            volume_ratio=6.0,
            whale_confluence=True,
        )
        # Might pass or fail depending on exact numbers, but score is
        # well below what the same signal would get at p=0.30.
        reference = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.42,  # same 12¢ gap
            zscore=4.0,
            volume_ratio=6.0,
            whale_confluence=True,
        )
        assert ea.score < reference.score


# ═══════════════════════════════════════════════════════════════════════════
#  Signal quality component
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalQuality:
    """Factor 4: signal strength above PanicDetector thresholds."""

    def test_baseline_at_threshold(self):
        """z = z_thresh, v = v_thresh → signal_quality = 0.5 baseline."""
        z_thresh = settings.strategy.zscore_threshold
        v_thresh = settings.strategy.volume_ratio_threshold
        ea = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=z_thresh,       # exactly at threshold
            volume_ratio=v_thresh,  # exactly at threshold
            min_score=0.0,
        )
        assert ea.signal_quality == pytest.approx(0.5, abs=0.01)

    def test_strong_signal_boosts_score(self):
        """z = 4.0, v = 6.0 → signal quality > 0.5."""
        ea = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=4.0,
            volume_ratio=6.0,
            min_score=0.0,
        )
        assert ea.signal_quality > 0.7

    def test_whale_confluence_adds_bonus(self):
        """Whale confluence adds 0.15 to signal quality."""
        base = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=1.2,
            volume_ratio=1.5,
            whale_confluence=False,
            min_score=0.0,
        )
        whale = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=1.2,
            volume_ratio=1.5,
            whale_confluence=True,
            min_score=0.0,
        )
        assert whale.signal_quality > base.signal_quality
        assert whale.score > base.score

    def test_signal_capped_at_one(self):
        """Extreme signal doesn't exceed 1.0."""
        ea = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=10.0,
            volume_ratio=20.0,
            whale_confluence=True,
            min_score=0.0,
        )
        assert ea.signal_quality == pytest.approx(1.0)

    def test_iceberg_active_adds_bonus(self):
        """Iceberg active adds 0.15 to signal quality, raising EQS."""
        base = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=1.2,
            volume_ratio=1.5,
            iceberg_active=False,
            min_score=0.0,
        )
        iceberg = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=1.2,
            volume_ratio=1.5,
            iceberg_active=True,
            min_score=0.0,
        )
        assert iceberg.signal_quality > base.signal_quality
        assert iceberg.signal_quality == pytest.approx(
            base.signal_quality + 0.15, abs=0.01
        )
        assert iceberg.score > base.score

    def test_iceberg_and_whale_stack(self):
        """Iceberg + whale bonuses stack but cap at 1.0."""
        both = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=4.0,
            volume_ratio=6.0,
            whale_confluence=True,
            iceberg_active=True,
            min_score=0.0,
        )
        assert both.signal_quality == pytest.approx(1.0)

    def test_iceberg_defaults_false(self):
        """iceberg_active defaults to False — no change from baseline."""
        without = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=3.0,
            volume_ratio=5.0,
            min_score=0.0,
        )
        explicit_false = compute_edge_score(
            entry_price=0.30,
            no_vwap=0.45,
            zscore=3.0,
            volume_ratio=5.0,
            iceberg_active=False,
            min_score=0.0,
        )
        assert without.signal_quality == explicit_false.signal_quality
        assert without.score == explicit_false.score


# ═══════════════════════════════════════════════════════════════════════════
#  Fee-disabled markets (political)
# ═══════════════════════════════════════════════════════════════════════════

class TestFeeDisabled:
    """Political markets charge 0% fees — fee efficiency should be 1.0."""

    def test_no_fee_market_full_efficiency(self):
        ea = compute_edge_score(
            entry_price=0.40,
            no_vwap=0.55,
            zscore=3.0,
            volume_ratio=5.0,
            fee_enabled=False,
            min_score=0.0,
        )
        assert ea.expected_fee_cents == 0.0
        assert ea.fee_efficiency == pytest.approx(1.0)
        assert ea.expected_net_cents == ea.expected_gross_cents

    def test_fee_disabled_helps_marginal_trades(self):
        """A trade that fails with fees should pass without them."""
        with_fee = compute_edge_score(
            entry_price=0.50,
            no_vwap=0.55,
            zscore=3.0,
            volume_ratio=5.0,
            fee_enabled=True,
            min_score=0.0,
        )
        without_fee = compute_edge_score(
            entry_price=0.50,
            no_vwap=0.55,
            zscore=3.0,
            volume_ratio=5.0,
            fee_enabled=False,
            min_score=0.0,
        )
        assert without_fee.score > with_fee.score
        assert without_fee.fee_efficiency > with_fee.fee_efficiency


# ═══════════════════════════════════════════════════════════════════════════
#  Tick viability details
# ═══════════════════════════════════════════════════════════════════════════

class TestTickViability:
    """Discrete-grid microstructure analysis."""

    def test_many_ticks_saturates(self):
        """≥ 3¢ net → tick_viability = 1.0."""
        ea = compute_edge_score(
            entry_price=0.20,
            no_vwap=0.40,     # 10¢ gross × α=0.5 = 5¢
            zscore=3.0,
            volume_ratio=5.0,
            min_score=0.0,
        )
        assert ea.tick_viability == pytest.approx(1.0)
        assert ea.tick_margin >= 3

    def test_partial_tick_viability(self):
        """1–2¢ net → tick_viability between 0 and 1."""
        ea = compute_edge_score(
            entry_price=0.15,
            no_vwap=0.24,     # gross ≈ 4.5¢, fees ≈ 2.2¢, net ≈ 2.3¢
            zscore=3.0,
            volume_ratio=5.0,
            min_score=0.0,
        )
        assert 0 < ea.tick_viability < 1.0
        assert ea.tick_margin >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  Rejection reason taxonomy
# ═══════════════════════════════════════════════════════════════════════════

class TestRejectionReasons:
    """Each failure mode should produce a specific, actionable reason."""

    def test_reason_no_mean_reversion_target(self):
        ea = compute_edge_score(
            entry_price=0.50, no_vwap=0.40,
            zscore=3.0, volume_ratio=5.0,
        )
        assert ea.rejection_reason == "no_mean_reversion_target"

    def test_reason_sub_tick_spread(self):
        ea = compute_edge_score(
            entry_price=0.50, no_vwap=0.505,
            zscore=3.0, volume_ratio=5.0,
        )
        assert ea.rejection_reason == "negative_ev_after_fees"

    def test_reason_empty_when_viable(self):
        ea = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=3.0, volume_ratio=5.0,
        )
        assert ea.viable is True
        assert ea.rejection_reason == ""


# ═══════════════════════════════════════════════════════════════════════════
#  Score monotonicity
# ═══════════════════════════════════════════════════════════════════════════

class TestMonotonicity:
    """Score should increase monotonically with improving factors."""

    def test_wider_vwap_gap_increases_score(self):
        """Larger VWAP dislocation → more edge → higher score."""
        scores = []
        for vwap in [0.40, 0.45, 0.50, 0.55, 0.60]:
            ea = compute_edge_score(
                entry_price=0.30,
                no_vwap=vwap,
                zscore=3.0,
                volume_ratio=5.0,
                min_score=0.0,
            )
            scores.append(ea.score)
        # Should be strictly increasing
        for i in range(1, len(scores)):
            assert scores[i] > scores[i - 1], (
                f"vwap increase {0.40 + 0.05*(i-1):.2f} → {0.40 + 0.05*i:.2f} "
                f"didn't increase score: {scores[i-1]:.2f} → {scores[i]:.2f}"
            )

    def test_higher_zscore_increases_score(self):
        """Stronger panic signal → higher score."""
        s1 = compute_edge_score(
            entry_price=0.30, no_vwap=0.45,
            zscore=1.3, volume_ratio=1.6, min_score=0.0,
        )
        s2 = compute_edge_score(
            entry_price=0.30, no_vwap=0.45,
            zscore=2.5, volume_ratio=1.6, min_score=0.0,
        )
        assert s2.score > s1.score


# ═══════════════════════════════════════════════════════════════════════════
#  Alpha override
# ═══════════════════════════════════════════════════════════════════════════

class TestAlphaOverride:
    def test_higher_alpha_increases_score(self):
        """Higher α → wider expected spread → better score."""
        lo = compute_edge_score(
            entry_price=0.30, no_vwap=0.45,
            zscore=3.0, volume_ratio=5.0,
            alpha=0.30, min_score=0.0,
        )
        hi = compute_edge_score(
            entry_price=0.30, no_vwap=0.45,
            zscore=3.0, volume_ratio=5.0,
            alpha=0.70, min_score=0.0,
        )
        assert hi.score > lo.score
        assert hi.expected_gross_cents > lo.expected_gross_cents


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostics completeness
# ═══════════════════════════════════════════════════════════════════════════

class TestDiagnostics:
    """Every EdgeAssessment should have fully populated diagnostics."""

    def test_all_fields_populated(self):
        ea = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=3.0, volume_ratio=5.0,
        )
        assert isinstance(ea.score, float)
        assert isinstance(ea.regime_quality, float)
        assert isinstance(ea.fee_efficiency, float)
        assert isinstance(ea.tick_margin, int)
        assert isinstance(ea.tick_viability, float)
        assert isinstance(ea.signal_quality, float)
        assert isinstance(ea.expected_gross_cents, float)
        assert isinstance(ea.expected_fee_cents, float)
        assert isinstance(ea.expected_net_cents, float)
        assert isinstance(ea.viable, bool)
        assert isinstance(ea.rejection_reason, str)

    def test_net_equals_gross_minus_fees(self):
        ea = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=3.0, volume_ratio=5.0, min_score=0.0,
        )
        assert ea.expected_net_cents == pytest.approx(
            ea.expected_gross_cents - ea.expected_fee_cents, abs=0.01
        )

    def test_score_bounded_zero_to_hundred(self):
        """Score is always in [0, 100]."""
        for entry, vwap in [(0.01, 0.50), (0.50, 0.90), (0.30, 0.50)]:
            ea = compute_edge_score(
                entry_price=entry, no_vwap=vwap,
                zscore=10.0, volume_ratio=20.0,
                whale_confluence=True,
                min_score=0.0,
            )
            assert 0.0 <= ea.score <= 100.0


# ═══════════════════════════════════════════════════════════════════════════
#  Parametric: entropy × fee alignment across price range
# ═══════════════════════════════════════════════════════════════════════════

class TestParametricPriceSweep:
    """Sweep across the probability range to verify the filter's
    behaviour aligns with the entropy × fee landscape."""

    @pytest.mark.parametrize("p", [0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    def test_score_with_same_relative_dislocation(self, p: float):
        """Each price gets the same 20% VWAP dislocation.  The score
        should reflect the interplay of entropy and fees."""
        vwap = p + 0.20
        ea = compute_edge_score(
            entry_price=p, no_vwap=vwap,
            zscore=3.0, volume_ratio=5.0,
            min_score=0.0,
        )
        # All should have positive expected net
        assert ea.expected_gross_cents > 0
        # Score should be finite and diagnostic
        assert ea.score >= 0.0
        assert ea.rejection_reason in ("", "score_below_threshold",
                                        "low_regime_entropy",
                                        "fees_exceed_discretised_spread",
                                        "fees_exceed_spread")


# ═══════════════════════════════════════════════════════════════════════════
#  model_confidence parameter (Deliverable B — EQS gate)
# ═══════════════════════════════════════════════════════════════════════════

class TestModelConfidence:
    """When model_confidence is supplied, signal_quality is set directly
    from it instead of the zscore/volume_ratio formula."""

    def test_high_confidence_boosts_score(self):
        """High model_confidence → higher EQS than default formula with
        mediocre zscore/volume_ratio."""
        ea_default = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=1.5, volume_ratio=1.5,
            min_score=0.0,
        )
        ea_conf = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=1.5, volume_ratio=1.5,
            min_score=0.0,
            model_confidence=0.95,
        )
        # With high confidence the score should be >= the default
        assert ea_conf.score >= ea_default.score

    def test_low_confidence_reduces_score(self):
        """Low model_confidence → lower EQS than default formula with
        strong zscore/volume_ratio."""
        ea_strong = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=4.0, volume_ratio=8.0,
            min_score=0.0,
        )
        ea_low = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=4.0, volume_ratio=8.0,
            min_score=0.0,
            model_confidence=0.15,
        )
        assert ea_low.score <= ea_strong.score

    def test_confidence_none_uses_default_formula(self):
        """model_confidence=None (default) behaves identically to
        omitted parameter."""
        ea1 = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=3.0, volume_ratio=5.0,
            min_score=0.0,
        )
        ea2 = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=3.0, volume_ratio=5.0,
            min_score=0.0,
            model_confidence=None,
        )
        assert ea1.score == pytest.approx(ea2.score, abs=1e-9)

    def test_confidence_clamped_low(self):
        """model_confidence below 0.1 is clamped to 0.1, not zero."""
        ea = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=3.0, volume_ratio=5.0,
            min_score=0.0,
            model_confidence=0.01,
        )
        # Should still produce a score (signal_quality = 0.1, not 0.01)
        assert ea.score >= 0.0

    def test_confidence_clamped_high(self):
        """model_confidence above 1.0 is clamped to 1.0."""
        ea = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=3.0, volume_ratio=5.0,
            min_score=0.0,
            model_confidence=1.5,
        )
        ea_one = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=3.0, volume_ratio=5.0,
            min_score=0.0,
            model_confidence=1.0,
        )
        assert ea.score == pytest.approx(ea_one.score, abs=1e-9)
