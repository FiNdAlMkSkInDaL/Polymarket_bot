"""
Tests for the Dynamic Prior Generation Engine.

Covers:
  - TagPriorRegistry: tag normalisation, matching, priority, fallback
  - GenericBayesianModel DPGE upgrade: tag-based priors, L2 imbalance,
    time-decay kernel, backward compatibility
  - RPECalibrationTracker: per-tag Brier scoring
  - Model-probe routing integration
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from src.signals.tag_prior_registry import TagPrior, TagPriorRegistry
from src.signals.resolution_probability import (
    GenericBayesianModel,
    ModelEstimate,
    RPECalibrationTracker,
    ResolutionProbabilityEngine,
)

# Ensure paper mode
os.environ.setdefault("PAPER_MODE", "true")
os.environ.setdefault("DEPLOYMENT_ENV", "PAPER")


# ── Helpers ──────────────────────────────────────────────────────────────

@dataclass
class FakeMarketInfo:
    """Minimal MarketInfo stand-in for tests."""

    condition_id: str = "TEST_MKT"
    question: str = "Will X happen?"
    yes_token_id: str = "YES_TOKEN"
    no_token_id: str = "NO_TOKEN"
    daily_volume_usd: float = 100_000.0
    end_date: datetime | None = None
    active: bool = True
    event_id: str = "EVT_1"
    liquidity_usd: float = 50_000.0
    score: float = 80.0
    accepting_orders: bool = True
    tags: str = ""

    def __post_init__(self) -> None:
        if self.end_date is None:
            self.end_date = datetime.now(timezone.utc) + timedelta(days=30)


def _make_model(
    obs_weight: float = 5.0,
    prior_k: float = 0.0,
    tag_registry: TagPriorRegistry | None = None,
) -> GenericBayesianModel:
    """Create a GenericBayesianModel with dynamic priors enabled."""
    return GenericBayesianModel(
        obs_weight=obs_weight,
        prior_k=prior_k,
        tag_registry=tag_registry,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TagPriorRegistry Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTagPriorRegistry:
    """Tag routing and prior resolution."""

    def test_politics_tag(self) -> None:
        reg = TagPriorRegistry()
        alpha, beta, label = reg.get_prior("Politics")
        assert label == "politics"
        assert alpha == 3.0
        assert beta == 2.0

    def test_elections_tag(self) -> None:
        reg = TagPriorRegistry()
        alpha, beta, label = reg.get_prior("Elections")
        assert label == "politics"

    def test_sports_tag(self) -> None:
        reg = TagPriorRegistry()
        alpha, beta, label = reg.get_prior("Sports, NBA")
        assert label == "sports"
        assert alpha == 2.5
        assert beta == 2.5

    def test_legal_tag(self) -> None:
        reg = TagPriorRegistry()
        _, _, label = reg.get_prior("Legal, Supreme Court")
        assert label == "legal"

    def test_economy_tag(self) -> None:
        reg = TagPriorRegistry()
        alpha, beta, label = reg.get_prior("Economy, Financial")
        assert label == "economy"
        assert alpha == 2.5
        assert beta == 2.0

    def test_geopolitics_tag(self) -> None:
        reg = TagPriorRegistry()
        _, _, label = reg.get_prior("Geopolitics, Conflict")
        assert label == "geopolitics"

    def test_pop_culture_tag(self) -> None:
        reg = TagPriorRegistry()
        _, _, label = reg.get_prior("Pop Culture, Entertainment")
        assert label == "pop_culture"

    def test_science_tech_tag(self) -> None:
        reg = TagPriorRegistry()
        _, _, label = reg.get_prior("Science, Technology, AI")
        assert label == "science_tech"

    def test_weather_tag(self) -> None:
        reg = TagPriorRegistry()
        alpha, _, label = reg.get_prior("Weather, Hurricane")
        assert label == "weather"
        assert alpha == 2.2

    def test_crypto_fallback(self) -> None:
        """Crypto tag should match but give symmetric prior."""
        reg = TagPriorRegistry()
        alpha, beta, label = reg.get_prior("Crypto")
        assert label == "crypto_fallback"
        assert alpha == 2.0
        assert beta == 2.0

    def test_empty_tags_returns_default(self) -> None:
        reg = TagPriorRegistry()
        alpha, beta, label = reg.get_prior("")
        assert label == "default_fallback"
        assert alpha == 2.0
        assert beta == 3.0

    def test_none_tags_returns_default(self) -> None:
        reg = TagPriorRegistry()
        alpha, beta, label = reg.get_prior(None)  # type: ignore
        assert label == "default_fallback"

    def test_unknown_tag_returns_default(self) -> None:
        reg = TagPriorRegistry()
        _, _, label = reg.get_prior("completely_unknown_category_xyz")
        assert label == "default_fallback"

    def test_case_insensitive(self) -> None:
        reg = TagPriorRegistry()
        _, _, label1 = reg.get_prior("POLITICS")
        _, _, label2 = reg.get_prior("politics")
        _, _, label3 = reg.get_prior("PoLiTiCs")
        assert label1 == label2 == label3 == "politics"

    def test_first_match_wins(self) -> None:
        """When tags contain multiple matches, first in priority wins."""
        reg = TagPriorRegistry()
        # "crypto" is first in priority table
        _, _, label = reg.get_prior("crypto, politics")
        assert label == "crypto_fallback"

    def test_comma_separated_second_tag_matches(self) -> None:
        """Second tag in comma-separated list should also match."""
        reg = TagPriorRegistry()
        _, _, label = reg.get_prior("misc, sports")
        assert label == "sports"

    def test_list_priors_returns_all(self) -> None:
        reg = TagPriorRegistry()
        priors = reg.list_priors()
        assert len(priors) > 5
        labels = [p["label"] for p in priors]
        assert "politics" in labels
        assert "sports" in labels
        assert "default_fallback" in labels

    def test_custom_table_override(self) -> None:
        """Custom prior table for testing."""
        import re
        custom_prior = TagPrior(pattern=r"\btest\b", alpha=10.0, beta=1.0, label="test_custom")
        compiled = [(re.compile(custom_prior.pattern, re.IGNORECASE), custom_prior)]
        reg = TagPriorRegistry(prior_table=compiled)
        alpha, beta, label = reg.get_prior("test")
        assert label == "test_custom"
        assert alpha == 10.0


# ═══════════════════════════════════════════════════════════════════════════
#  GenericBayesianModel — Dynamic Prior Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGenericBayesianModelDynamicPriors:
    """Test the upgraded GenericBayesianModel with DPGE features."""

    def test_tag_prior_shifts_estimate(self) -> None:
        """Politics prior Beta(3,2) should bias YES vs default Beta(2,3)."""
        model = _make_model()
        market_politics = FakeMarketInfo(tags="politics")
        market_default = FakeMarketInfo(tags="unknown_xyz")

        est_pol = model.estimate(
            market_politics, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
        )
        est_def = model.estimate(
            market_default, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
        )
        assert est_pol is not None and est_def is not None
        # Politics prior favours YES → higher posterior
        assert est_pol.probability > est_def.probability

    def test_sports_prior_symmetric(self) -> None:
        """Sports Beta(2.5, 2.5) should produce near-symmetric estimate."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")
        est = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
        )
        assert est is not None
        assert abs(est.probability - 0.50) < 0.05

    def test_metadata_contains_prior_source(self) -> None:
        """Estimate metadata should include prior_source and DPGE fields."""
        model = _make_model()
        market = FakeMarketInfo(tags="politics")
        est = model.estimate(
            market, market_price=0.60,
            days_to_resolution=15, total_duration_days=60.0,
        )
        assert est is not None
        assert est.metadata["prior_source"] == "politics"
        assert est.metadata["dynamic_prior_enabled"] is True
        assert "w_prior" in est.metadata
        assert "n_eff" in est.metadata
        assert "kappa_eff" in est.metadata

    def test_invalid_price_returns_none(self) -> None:
        model = _make_model()
        market = FakeMarketInfo()
        assert model.estimate(market, market_price=0.0) is None
        assert model.estimate(market, market_price=1.0) is None
        assert model.estimate(market, market_price=-0.5) is None

    def test_model_name(self) -> None:
        model = _make_model()
        assert model.name == "generic_bayesian"

    def test_always_can_handle(self) -> None:
        model = _make_model()
        for tags in ("crypto", "politics", "sports", "", "random"):
            market = FakeMarketInfo(tags=tags)
            assert model.can_handle(market) is True


# ═══════════════════════════════════════════════════════════════════════════
#  L2 Order Book Imbalance Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestL2ImbalanceUpdate:
    """Test L2 book_depth_ratio continuous observation stream."""

    def test_l2_ratio_above_1_shifts_yes(self) -> None:
        """book_depth_ratio > 1 (more bids) → YES probability increases."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")

        est_neutral = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=1.0, l2_reliable=True,
        )
        est_bullish = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=3.0, l2_reliable=True,
        )
        assert est_neutral is not None and est_bullish is not None
        assert est_bullish.probability > est_neutral.probability

    def test_l2_ratio_below_1_shifts_no(self) -> None:
        """book_depth_ratio < 1 (more asks) → YES probability decreases."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")

        est_neutral = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=1.0, l2_reliable=True,
        )
        est_bearish = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=0.3, l2_reliable=True,
        )
        assert est_neutral is not None and est_bearish is not None
        assert est_bearish.probability < est_neutral.probability

    def test_l2_unreliable_ignored(self) -> None:
        """When L2 is unreliable, book_depth_ratio should have no effect."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")

        est_no_l2 = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=None, l2_reliable=False,
        )
        est_unreliable = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=5.0, l2_reliable=False,
        )
        assert est_no_l2 is not None and est_unreliable is not None
        assert abs(est_no_l2.probability - est_unreliable.probability) < 1e-10

    def test_l2_metadata_tracked(self) -> None:
        """Metadata should reflect L2 activity status."""
        model = _make_model()
        market = FakeMarketInfo(tags="economy")

        est_active = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=2.0, l2_reliable=True,
        )
        est_inactive = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
        )
        assert est_active is not None and est_inactive is not None
        assert est_active.metadata["l2_active"] is True
        assert est_inactive.metadata["l2_active"] is False

    def test_l2_extreme_ratio_clamped(self) -> None:
        """Extreme book_depth_ratio should be clamped via ln(r) ∈ [-2, 2]."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")

        # ratio=100 → ln(100) ≈ 4.6 but clamped to 2.0
        est_extreme = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=100.0, l2_reliable=True,
        )
        est_moderate = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
            book_depth_ratio=math.exp(2.0), l2_reliable=True,  # e^2 ≈ 7.39
        )
        assert est_extreme is not None and est_moderate is not None
        # Due to clamping, extreme and e^2 should produce same result
        assert abs(est_extreme.probability - est_moderate.probability) < 0.01

    def test_l2_pseudo_counts_symmetric(self) -> None:
        """ln(r) and ln(1/r) should produce symmetric pseudo-counts."""
        alpha_up, beta_up = GenericBayesianModel._l2_pseudo_counts(2.0, kappa_eff=1.5)
        alpha_dn, beta_dn = GenericBayesianModel._l2_pseudo_counts(0.5, kappa_eff=1.5)
        # ratio=2 → α_delta = κ·ln(2), β_delta = 0
        # ratio=0.5 → α_delta = 0, β_delta = κ·ln(2)
        assert abs(alpha_up - beta_dn) < 1e-10
        assert abs(beta_up - alpha_dn) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
#  Time-Decay (Theta) Kernel Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTimaDecayKernel:
    """Sigmoid time-decay prior/observation weight tests."""

    def test_near_expiry_market_price_dominates(self) -> None:
        """At t → 0 (near expiry), w_prior is low and posterior ≈ market price."""
        model = _make_model()
        market = FakeMarketInfo(tags="politics")

        # Use a very short total_duration so t = 1/30 is very small
        est = model.estimate(
            market, market_price=0.85,
            days_to_resolution=1, total_duration_days=300.0,
        )
        assert est is not None
        # Near expiry with large total_duration, n_eff is very large
        assert abs(est.probability - 0.85) < 0.10
        # w_prior should be small (prior influence drops near expiry)
        assert est.metadata["w_prior"] < 0.3

    def test_early_market_prior_dominates(self) -> None:
        """At t → 1 (early market), w_prior → 1 and prior has more weight."""
        model = _make_model()
        market = FakeMarketInfo(tags="politics")  # Beta(3,2), mean=0.6

        est = model.estimate(
            market, market_price=0.30,
            days_to_resolution=89, total_duration_days=90.0,
        )
        assert est is not None
        # Prior pulls toward 0.6, so estimate should be between 0.30 and 0.60
        assert est.probability > 0.30
        assert est.metadata["w_prior"] > 0.8

    def test_sigmoid_weight_at_halflife(self) -> None:
        """At t = t_half, w_prior should be exactly 0.5."""
        gamma = 8.0
        t_half = 0.15
        total = 100.0
        days = total * t_half  # 15 days left when total=100 and t_half=0.15

        w = GenericBayesianModel._sigmoid_time_weight(days, total, gamma, t_half)
        assert abs(w - 0.5) < 0.01

    def test_sigmoid_weight_monotonic(self) -> None:
        """w_prior should increase monotonically as days_to_resolution increases."""
        gamma, t_half, total = 8.0, 0.15, 90.0
        weights = []
        for d in range(0, 91, 5):
            w = GenericBayesianModel._sigmoid_time_weight(float(d), total, gamma, t_half)
            weights.append(w)
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1] - 1e-10

    def test_expired_market_zero_weight(self) -> None:
        """Total duration ≤ 0 → w_prior = 0."""
        w = GenericBayesianModel._sigmoid_time_weight(0.0, 0.0, 8.0, 0.15)
        assert w == 0.0

    def test_l2_kappa_suppressed_near_expiry(self) -> None:
        """Near expiry, kappa_eff should be much lower than at market open."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")

        est_near = model.estimate(
            market, market_price=0.50,
            days_to_resolution=1, total_duration_days=300.0,
            book_depth_ratio=5.0, l2_reliable=True,
        )
        est_far = model.estimate(
            market, market_price=0.50,
            days_to_resolution=250, total_duration_days=300.0,
            book_depth_ratio=5.0, l2_reliable=True,
        )
        assert est_near is not None and est_far is not None
        # kappa_eff should be much lower near expiry than early
        assert est_near.metadata["kappa_eff"] < est_far.metadata["kappa_eff"]
        # And significantly suppressed
        assert est_near.metadata["kappa_eff"] < 0.5

    def test_time_decay_confidence_drops_near_expiry(self) -> None:
        """Confidence should drop near expiry to prevent terminal noise signals."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")

        est_far = model.estimate(
            market, market_price=0.60,
            days_to_resolution=60, total_duration_days=90.0,
        )
        est_near = model.estimate(
            market, market_price=0.60,
            days_to_resolution=2, total_duration_days=90.0,
        )
        assert est_far is not None and est_near is not None
        # Near expiry, confidence should be lower (w_prior is low)
        assert est_near.confidence < est_far.confidence


# ═══════════════════════════════════════════════════════════════════════════
#  RPECalibrationTracker — Per-Tag Brier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRPECalibrationPerTag:
    """Test per-tag Brier score breakdown."""

    def test_per_tag_brier_empty(self) -> None:
        tracker = RPECalibrationTracker()
        assert tracker.compute_per_tag_brier() == {}

    def test_per_tag_brier_single_tag(self) -> None:
        tracker = RPECalibrationTracker()
        tracker.record_signal(
            "MKT1", 0.7, 0.5, "buy_yes", 1000.0,
            prior_source="politics",
        )
        tracker.on_market_resolved("MKT1", 1.0)  # YES won

        result = tracker.compute_per_tag_brier()
        assert "politics" in result
        assert result["politics"]["count"] == 1
        # Brier = (0.7 - 1.0)^2 = 0.09
        assert abs(result["politics"]["brier"] - 0.09) < 0.01
        assert result["politics"]["direction_accuracy"] == 1.0

    def test_per_tag_brier_multiple_tags(self) -> None:
        tracker = RPECalibrationTracker()

        # Politics signals
        tracker.record_signal("MKT1", 0.8, 0.5, "buy_yes", 1000.0, prior_source="politics")
        tracker.record_signal("MKT2", 0.3, 0.5, "buy_no", 1001.0, prior_source="politics")

        # Sports signals
        tracker.record_signal("MKT3", 0.6, 0.5, "buy_yes", 1002.0, prior_source="sports")

        # Resolve
        tracker.on_market_resolved("MKT1", 1.0)  # correct buy_yes
        tracker.on_market_resolved("MKT2", 0.0)  # correct buy_no
        tracker.on_market_resolved("MKT3", 0.0)  # wrong buy_yes

        result = tracker.compute_per_tag_brier()
        assert "politics" in result
        assert "sports" in result
        assert result["politics"]["count"] == 2
        assert result["politics"]["direction_accuracy"] == 1.0
        assert result["sports"]["count"] == 1
        assert result["sports"]["direction_accuracy"] == 0.0

    def test_calibration_summary_includes_per_tag(self) -> None:
        tracker = RPECalibrationTracker()
        tracker.record_signal("MKT1", 0.7, 0.5, "buy_yes", 1000.0, prior_source="politics")
        tracker.on_market_resolved("MKT1", 1.0)

        summary = tracker.calibration_summary()
        assert "per_tag_brier" in summary
        assert "politics" in summary["per_tag_brier"]

    def test_l2_active_tracking(self) -> None:
        tracker = RPECalibrationTracker()
        tracker.record_signal(
            "MKT1", 0.6, 0.5, "buy_yes", 1000.0,
            l2_active=True, theta_w_prior=0.7,
        )
        # Verify the entry has the right fields
        entries = tracker._by_market["MKT1"]
        assert len(entries) == 1
        assert entries[0].l2_active is True
        assert entries[0].theta_w_prior == 0.7


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: RPE with Dynamic Priors
# ═══════════════════════════════════════════════════════════════════════════


class TestRPEWithDynamicPriors:
    """Integration tests for RPE signal firing with dynamic priors."""

    def test_generic_enabled_with_dynamic_prior_produces_divergence(self) -> None:
        """With a strong tag prior and different market price, divergence exists."""
        model = _make_model(obs_weight=2.0)
        rpe = ResolutionProbabilityEngine(
            models=[model],
            confidence_threshold=0.02,
            shadow_mode=False,
            min_confidence=0.01,
            generic_enabled=True,
        )

        # Politics prior Beta(3,2), mean=0.6.  Market at 0.20.
        # With low obs_weight=2, prior should pull estimate toward 0.6,
        # creating divergence against market at 0.20.
        market = FakeMarketInfo(tags="politics", condition_id="MKT_POL")

        signal = rpe.evaluate(
            market=market,
            market_price=0.20,
            days_to_resolution=45,
            total_duration_days=90.0,
        )
        # The model estimate should have probability > 0.20 due to politics prior
        estimate = rpe.get_estimate("MKT_POL")
        assert estimate is not None
        assert estimate.probability > 0.30  # prior pulls toward 0.6
        assert estimate.metadata["prior_source"] == "politics"

    def test_generic_disabled_blocks_signal_but_caches(self) -> None:
        """When generic is disabled, signal is blocked but estimate is cached."""
        model = _make_model(obs_weight=2.0)
        rpe = ResolutionProbabilityEngine(
            models=[model],
            confidence_threshold=0.02,
            shadow_mode=False,
            min_confidence=0.01,
            generic_enabled=False,
        )

        market = FakeMarketInfo(tags="politics", condition_id="MKT_BLOCKED")
        signal = rpe.evaluate(
            market=market,
            market_price=0.20,
            days_to_resolution=45,
            total_duration_days=90.0,
        )
        assert signal is None
        # But estimate should still be cached
        assert rpe.get_estimate("MKT_BLOCKED") is not None

    def test_near_expiry_no_signal_due_to_convergence(self) -> None:
        """Near expiry, posterior converges to market price → no divergence."""
        model = _make_model(obs_weight=5.0)
        rpe = ResolutionProbabilityEngine(
            models=[model],
            confidence_threshold=0.04,
            shadow_mode=False,
            min_confidence=0.01,
            generic_enabled=True,
        )

        market = FakeMarketInfo(tags="politics", condition_id="MKT_EXPIRY")
        signal = rpe.evaluate(
            market=market,
            market_price=0.85,
            days_to_resolution=1,
            total_duration_days=90.0,
        )
        # Near expiry, n_eff is huge → posterior ≈ 0.85 → no divergence
        assert signal is None


# ═══════════════════════════════════════════════════════════════════════════
#  Backward Compatibility Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Ensure existing behaviour is preserved when DPGE is disabled."""

    def test_legacy_market_anchored_prior(self) -> None:
        """When dynamic_prior is disabled and k > 0, use market-anchored prior."""
        from src.core.config import settings
        original = settings.strategy.rpe_dynamic_prior_enabled
        try:
            object.__setattr__(settings.strategy, "rpe_dynamic_prior_enabled", False)

            model = _make_model(prior_k=4.0)
            market = FakeMarketInfo(tags="politics")
            est = model.estimate(
                market, market_price=0.80,
                days_to_resolution=30, total_duration_days=90.0,
            )
            assert est is not None
            assert est.metadata["prior_source"] == "market_anchored"
        finally:
            object.__setattr__(settings.strategy, "rpe_dynamic_prior_enabled", original)

    def test_legacy_static_prior(self) -> None:
        """When dynamic_prior disabled and k=0, use static alpha0/beta0."""
        from src.core.config import settings
        original = settings.strategy.rpe_dynamic_prior_enabled
        try:
            object.__setattr__(settings.strategy, "rpe_dynamic_prior_enabled", False)

            model = GenericBayesianModel(
                alpha0=3.0, beta0=1.0,
                obs_weight=5.0, prior_k=0.0,
            )
            market = FakeMarketInfo(tags="politics")
            est = model.estimate(
                market, market_price=0.50,
                days_to_resolution=30, total_duration_days=90.0,
            )
            assert est is not None
            assert est.metadata["prior_source"] == "static_legacy"
        finally:
            object.__setattr__(settings.strategy, "rpe_dynamic_prior_enabled", original)

    def test_probability_always_clamped(self) -> None:
        """Probability should always be in [0.01, 0.99]."""
        model = _make_model(obs_weight=0.1)
        market = FakeMarketInfo(tags="politics")

        est = model.estimate(
            market, market_price=0.99,
            days_to_resolution=1, total_duration_days=90.0,
        )
        assert est is not None
        assert 0.01 <= est.probability <= 0.99

    def test_confidence_always_clamped(self) -> None:
        """Confidence should always be in [0.05, 0.95]."""
        model = _make_model()
        market = FakeMarketInfo(tags="sports")
        est = model.estimate(
            market, market_price=0.50,
            days_to_resolution=30, total_duration_days=90.0,
        )
        assert est is not None
        assert 0.05 <= est.confidence <= 0.95
