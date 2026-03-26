from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from src.execution.ctf_execution_manifest import build_ctf_execution_manifest
from src.execution.ctf_unwind_manifest import CtfUnwindLeg, CtfUnwindManifest
from src.execution.escalation_policy_interface import EscalationPolicyInterface
from src.execution.live_escalation_policy import LiveEscalationPolicy
from src.execution.live_unwind_cost_estimator import LiveUnwindCostEstimator
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindLeg, Si9UnwindManifest


class _StubTracker:
    def __init__(self, *, asset_id: str, best_bid: float, timestamp: float = 1712345.678) -> None:
        self.asset_id = asset_id
        self.best_bid = best_bid
        self._timestamp = timestamp

    def snapshot(self):
        class _Snapshot:
            def __init__(self, timestamp: float) -> None:
                self.timestamp = timestamp

        return _Snapshot(self._timestamp)


def _config() -> Si9UnwindConfig:
    return Si9UnwindConfig(
        market_sell_threshold=Decimal("0.050000"),
        passive_unwind_threshold=Decimal("0.010000"),
        max_hold_recovery_ms=100,
        min_best_bid=Decimal("0.010000"),
    )


def _providers(best_bids: dict[str, float]) -> dict[str, OrderbookBestBidProvider]:
    return {
        market_id: OrderbookBestBidProvider(_StubTracker(asset_id=market_id, best_bid=best_bid))
        for market_id, best_bid in best_bids.items()
    }


def _si9_execution_manifest() -> Si9ExecutionManifest:
    return Si9ExecutionManifest(
        cluster_id="cluster-1",
        legs=(
            Si9LegManifest("mkt-a", "YES", Decimal("0.33"), Decimal("2"), True, 0),
            Si9LegManifest("mkt-b", "YES", Decimal("0.31"), Decimal("2"), False, 1),
        ),
        net_edge=Decimal("0.025"),
        required_share_counts=Decimal("2"),
        bottleneck_market_id="mkt-a",
        manifest_timestamp_ms=100,
        max_leg_fill_wait_ms=250,
        cancel_on_stale_ms=500,
    )


def _si9_unwind_manifest(*, recommended_action: str = "HOLD_FOR_RECOVERY") -> Si9UnwindManifest:
    return Si9UnwindManifest(
        cluster_id="cluster-1",
        hanging_legs=(
            Si9UnwindLeg("mkt-a", "YES", Decimal("2.0"), Decimal("0.40"), Decimal("0.39"), Decimal("0.02"), 0),
            Si9UnwindLeg("mkt-b", "YES", Decimal("2.0"), Decimal("0.35"), Decimal("0.34"), Decimal("0.02"), 1),
        ),
        unwind_reason="MANUAL_ABORT",
        original_manifest=_si9_execution_manifest(),
        unwind_timestamp_ms=200,
        total_estimated_unwind_cost=Decimal("0.04"),
        recommended_action=recommended_action,  # type: ignore[arg-type]
    )


def _ctf_unwind_manifest(*, recommended_action: str = "HOLD_FOR_RECOVERY") -> CtfUnwindManifest:
    original_manifest = build_ctf_execution_manifest(
        market_id="ctf-1",
        yes_price=Decimal("0.38"),
        no_price=Decimal("0.40"),
        net_edge=Decimal("0.18"),
        gas_estimate=Decimal("0.01"),
        default_anchor_volume=Decimal("10"),
        max_capital_per_signal=Decimal("25"),
        max_size_per_leg=Decimal("8"),
        taker_fee_yes=Decimal("0.01"),
        taker_fee_no=Decimal("0.01"),
        manifest_timestamp_ms=1000,
        cancel_on_stale_ms=100,
    )
    return CtfUnwindManifest(
        cluster_id="ctf-1",
        hanging_legs=(
            CtfUnwindLeg("ctf-1", "NO", Decimal("3.0"), Decimal("0.39"), Decimal("0.385"), Decimal("0.015"), 0),
        ),
        unwind_reason="SECOND_LEG_REJECTED",
        original_manifest=original_manifest,
        unwind_timestamp_ms=500,
        total_estimated_unwind_cost=Decimal("0.015"),
        recommended_action=recommended_action,  # type: ignore[arg-type]
    )


def test_live_escalation_policy_satisfies_interface() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.39, "mkt-b": 0.34}))

    assert isinstance(LiveEscalationPolicy(estimator, _config()), EscalationPolicyInterface)


def test_live_unwind_cost_estimator_single_leg_cost_uses_decimal_formula() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"ctf-1": 0.385}))

    total = estimator.estimate_total_cost(_ctf_unwind_manifest())

    assert total == Decimal("0.015")


def test_live_unwind_cost_estimator_multi_leg_total_sums_all_legs() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.39, "mkt-b": 0.34}))

    total = estimator.estimate_total_cost(_si9_unwind_manifest())

    assert total == Decimal("0.040")


def test_estimate_manifest_updates_si9_leg_bids_and_costs() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.38, "mkt-b": 0.33}))

    refreshed = estimator.estimate_manifest(_si9_unwind_manifest())

    assert isinstance(refreshed, Si9UnwindManifest)
    assert refreshed.hanging_legs[0].current_best_bid == Decimal("0.38")
    assert refreshed.hanging_legs[0].estimated_unwind_cost == Decimal("0.04")
    assert refreshed.total_estimated_unwind_cost == Decimal("0.080")


def test_estimate_manifest_updates_ctf_leg_bids_and_costs() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"ctf-1": 0.380}))

    refreshed = estimator.estimate_manifest(_ctf_unwind_manifest())

    assert isinstance(refreshed, CtfUnwindManifest)
    assert refreshed.hanging_legs[0].current_best_bid == Decimal("0.38")
    assert refreshed.hanging_legs[0].estimated_unwind_cost == Decimal("0.03")


def test_estimator_raises_on_missing_provider() -> None:
    estimator = LiveUnwindCostEstimator({})

    with pytest.raises(ValueError, match="OrderbookBestBidProvider"):
        estimator.estimate_manifest(_ctf_unwind_manifest())


def test_estimator_raises_on_missing_live_best_bid() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"ctf-1": 0.0}))

    with pytest.raises(ValueError, match="live best bid"):
        estimator.estimate_manifest(_ctf_unwind_manifest())


def test_estimator_preserves_original_manifest_unchanged() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.38, "mkt-b": 0.33}))
    manifest = _si9_unwind_manifest()

    refreshed = estimator.estimate_manifest(manifest)

    assert manifest.hanging_legs[0].current_best_bid == Decimal("0.39")
    assert refreshed is not manifest


def test_hold_for_recovery_escalates_to_market_sell_only_after_timeout_expires() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.395, "mkt-b": 0.345}))
    policy = LiveEscalationPolicy(estimator, _config())

    escalated = policy.escalate_manifest(_si9_unwind_manifest(), 301)

    assert escalated.recommended_action == "MARKET_SELL"


def test_hold_for_recovery_does_not_time_escalate_at_exact_boundary() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.395, "mkt-b": 0.345}))
    policy = LiveEscalationPolicy(estimator, _config())

    escalated = policy.escalate_manifest(_si9_unwind_manifest(), 300)

    assert escalated.recommended_action == "HOLD_FOR_RECOVERY"


def test_threshold_below_passive_boundary_recommends_passive_unwind() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.399, "mkt-b": 0.349}))
    policy = LiveEscalationPolicy(estimator, _config())

    escalated = policy.escalate_manifest(_si9_unwind_manifest(), 250)

    assert escalated.recommended_action == "PASSIVE_UNWIND"


def test_threshold_between_boundaries_recommends_hold_for_recovery() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.39, "mkt-b": 0.34}))
    policy = LiveEscalationPolicy(estimator, _config())

    escalated = policy.escalate_manifest(_si9_unwind_manifest(), 250)

    assert escalated.recommended_action == "HOLD_FOR_RECOVERY"


def test_threshold_above_market_boundary_recommends_market_sell() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.37, "mkt-b": 0.32}))
    policy = LiveEscalationPolicy(estimator, _config())

    escalated = policy.escalate_manifest(_si9_unwind_manifest(), 250)

    assert escalated.recommended_action == "MARKET_SELL"


def test_policy_leaves_original_manifest_completely_unchanged() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.37, "mkt-b": 0.32}))
    policy = LiveEscalationPolicy(estimator, _config())
    manifest = _si9_unwind_manifest()

    escalated = policy.escalate_manifest(manifest, 250)

    assert manifest.recommended_action == "HOLD_FOR_RECOVERY"
    assert manifest.total_estimated_unwind_cost == Decimal("0.04")
    assert escalated is not manifest


def test_should_escalate_returns_true_when_recommendation_changes() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.37, "mkt-b": 0.32}))
    policy = LiveEscalationPolicy(estimator, _config())

    assert policy.should_escalate(_si9_unwind_manifest(), 250) is True


def test_should_escalate_returns_false_when_manifest_already_matches_live_state() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.399, "mkt-b": 0.349}))
    policy = LiveEscalationPolicy(estimator, _config())
    manifest = replace(
        _si9_unwind_manifest(recommended_action="PASSIVE_UNWIND"),
        hanging_legs=(
            Si9UnwindLeg("mkt-a", "YES", Decimal("2.0"), Decimal("0.40"), Decimal("0.399"), Decimal("0.002"), 0),
            Si9UnwindLeg("mkt-b", "YES", Decimal("2.0"), Decimal("0.35"), Decimal("0.349"), Decimal("0.002"), 1),
        ),
        total_estimated_unwind_cost=Decimal("0.004"),
    )

    assert policy.should_escalate(manifest, 250) is False


def test_should_surrender_remains_false_in_live_policy() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"mkt-a": 0.39, "mkt-b": 0.34}))
    policy = LiveEscalationPolicy(estimator, _config())

    assert policy.should_surrender(_si9_unwind_manifest(), 999) is False


def test_ctf_manifest_uses_live_cost_estimator_and_policy_thresholds() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"ctf-1": 0.387}))
    policy = LiveEscalationPolicy(estimator, _config())

    escalated = policy.escalate_manifest(_ctf_unwind_manifest(), 550)

    assert isinstance(escalated, CtfUnwindManifest)
    assert escalated.total_estimated_unwind_cost == Decimal("0.009")
    assert escalated.recommended_action == "PASSIVE_UNWIND"


def test_ctf_hold_manifest_time_escalates_to_market_sell_after_timeout() -> None:
    estimator = LiveUnwindCostEstimator(_providers({"ctf-1": 0.387}))
    policy = LiveEscalationPolicy(estimator, _config())

    escalated = policy.escalate_manifest(_ctf_unwind_manifest(), 650)

    assert escalated.recommended_action == "MARKET_SELL"


def test_incoming_manifest_is_frozen_against_mutation_attempts() -> None:
    manifest = _si9_unwind_manifest()

    with pytest.raises(FrozenInstanceError):
        manifest.recommended_action = "MARKET_SELL"  # type: ignore[misc]