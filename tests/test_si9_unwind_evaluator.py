from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_unwind_evaluator import Si9UnwindEvaluator
from src.execution.si9_unwind_manifest import Si9UnwindConfig


def _config(**overrides: object) -> Si9UnwindConfig:
    values = {
        "market_sell_threshold": Decimal("0.050"),
        "passive_unwind_threshold": Decimal("0.010"),
        "max_hold_recovery_ms": 100,
        "min_best_bid": Decimal("0.050"),
    }
    values.update(overrides)
    return Si9UnwindConfig(**values)


def _manifest() -> Si9ExecutionManifest:
    return Si9ExecutionManifest(
        cluster_id="cluster-1",
        legs=(
            Si9LegManifest("mkt-a", "YES", Decimal("0.33"), Decimal("2"), True, 0),
            Si9LegManifest("mkt-b", "YES", Decimal("0.31"), Decimal("2"), False, 1),
            Si9LegManifest("mkt-c", "YES", Decimal("0.31"), Decimal("2"), False, 2),
        ),
        net_edge=Decimal("0.025"),
        required_share_counts=Decimal("2"),
        bottleneck_market_id="mkt-a",
        manifest_timestamp_ms=100,
        max_leg_fill_wait_ms=250,
        cancel_on_stale_ms=500,
    )


def test_valid_si9_unwind_config_construction_passes() -> None:
    config = _config()

    assert config.market_sell_threshold == Decimal("0.050")
    assert config.passive_unwind_threshold == Decimal("0.010")
    assert config.max_hold_recovery_ms == 100


def test_market_sell_threshold_not_above_passive_threshold_raises_value_error() -> None:
    with pytest.raises(ValueError, match="market_sell_threshold"):
        _config(market_sell_threshold=Decimal("0.010"), passive_unwind_threshold=Decimal("0.010"))


@pytest.mark.parametrize("min_best_bid", [Decimal("0"), Decimal("1")])
def test_min_best_bid_outside_interval_raises_value_error(min_best_bid: Decimal) -> None:
    with pytest.raises(ValueError, match="min_best_bid"):
        _config(min_best_bid=min_best_bid)


def test_single_hanging_leg_cost_matches_manual_calculation() -> None:
    evaluator = Si9UnwindEvaluator(_config())

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("2"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.35")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.hanging_legs[0].estimated_unwind_cost.quantize(Decimal("0.000001")) == Decimal("0.100000")


def test_multiple_hanging_legs_total_cost_is_exact_sum() -> None:
    evaluator = Si9UnwindEvaluator(_config())

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[
            ("mkt-a", Decimal("2"), Decimal("0.40")),
            ("mkt-b", Decimal("1.5"), Decimal("0.38")),
        ],
        current_bids={"mkt-a": Decimal("0.35"), "mkt-b": Decimal("0.36")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.total_estimated_unwind_cost == Decimal("0.13")


def test_recommended_action_market_sell_when_guard_circuit_open_regardless_of_cost() -> None:
    evaluator = Si9UnwindEvaluator(_config(market_sell_threshold=Decimal("1.0")))

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.399")},
        unwind_reason="GUARD_CIRCUIT_OPEN",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.recommended_action == "MARKET_SELL"


def test_recommended_action_market_sell_when_cost_exceeds_market_threshold() -> None:
    evaluator = Si9UnwindEvaluator(_config(market_sell_threshold=Decimal("0.05")))

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("2"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.35")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.recommended_action == "MARKET_SELL"


def test_recommended_action_passive_unwind_when_cost_below_passive_threshold() -> None:
    evaluator = Si9UnwindEvaluator(_config(passive_unwind_threshold=Decimal("0.010")))

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.395")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.recommended_action == "PASSIVE_UNWIND"


def test_recommended_action_hold_for_recovery_for_intermediate_cost() -> None:
    evaluator = Si9UnwindEvaluator(_config())

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.38")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.recommended_action == "HOLD_FOR_RECOVERY"


def test_passive_threshold_boundary_selects_passive_unwind() -> None:
    evaluator = Si9UnwindEvaluator(_config(passive_unwind_threshold=Decimal("0.020"), market_sell_threshold=Decimal("0.050")))

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.38")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.recommended_action == "PASSIVE_UNWIND"


def test_market_threshold_boundary_selects_market_sell() -> None:
    evaluator = Si9UnwindEvaluator(_config(market_sell_threshold=Decimal("0.020"), passive_unwind_threshold=Decimal("0.010")))

    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.38")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert manifest.recommended_action == "MARKET_SELL"


def test_escalate_promotes_hold_for_recovery_after_timeout() -> None:
    evaluator = Si9UnwindEvaluator(_config(max_hold_recovery_ms=100))
    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.38")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    escalated = evaluator.escalate(manifest, current_timestamp_ms=301)

    assert manifest.recommended_action == "HOLD_FOR_RECOVERY"
    assert escalated.recommended_action == "MARKET_SELL"
    assert escalated is not manifest


def test_escalate_returns_unchanged_manifest_before_timeout_expires() -> None:
    evaluator = Si9UnwindEvaluator(_config(max_hold_recovery_ms=100))
    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.38")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    assert evaluator.escalate(manifest, current_timestamp_ms=300) is manifest


@pytest.mark.parametrize(
    ("current_bid", "expected_action"),
    [
        (Decimal("0.395"), "PASSIVE_UNWIND"),
        (Decimal("0.34"), "MARKET_SELL"),
    ],
)
def test_escalate_does_not_modify_non_hold_recommendations(current_bid: Decimal, expected_action: str) -> None:
    evaluator = Si9UnwindEvaluator(_config())
    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": current_bid},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    result = evaluator.escalate(manifest, current_timestamp_ms=999)

    assert manifest.recommended_action == expected_action
    assert result is manifest


def test_si9_unwind_manifest_is_frozen() -> None:
    evaluator = Si9UnwindEvaluator(_config())
    manifest = evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.38")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )

    with pytest.raises(FrozenInstanceError):
        manifest.recommended_action = "MARKET_SELL"  # type: ignore[misc]


def test_current_best_bid_below_min_best_bid_raises_value_error() -> None:
    evaluator = Si9UnwindEvaluator(_config(min_best_bid=Decimal("0.20")))

    with pytest.raises(ValueError, match="min_best_bid"):
        evaluator.evaluate(
            cluster_id="cluster-1",
            hanging_legs=[("mkt-a", Decimal("1"), Decimal("0.40"))],
            current_bids={"mkt-a": Decimal("0.19")},
            unwind_reason="MANUAL_ABORT",
            original_manifest=_manifest(),
            unwind_timestamp_ms=200,
        )


def test_zero_hanging_legs_raises_value_error() -> None:
    evaluator = Si9UnwindEvaluator(_config())

    with pytest.raises(ValueError, match="hanging_legs"):
        evaluator.evaluate(
            cluster_id="cluster-1",
            hanging_legs=[],
            current_bids={},
            unwind_reason="MANUAL_ABORT",
            original_manifest=_manifest(),
            unwind_timestamp_ms=200,
        )