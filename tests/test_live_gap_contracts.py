from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from src.execution.escalation_policy_interface import EscalationPolicyInterface, PaperEscalationPolicy
from src.execution.live_book_interface import LiveBestBidProvider, PaperBestBidProvider
from src.execution.multi_signal_orchestrator import OrchestratorEvent
from src.execution.position_lifecycle_interface import PaperPositionLifecycle, PositionLifecycleInterface
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.execution.position_manager_lifecycle import PositionManagerLifecycle
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_unwind_evaluator import Si9UnwindEvaluator
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.unwind_executor_interface import PaperUnwindExecutor, UnwindExecutionReceipt, UnwindExecutor


def _config() -> Si9UnwindConfig:
    return Si9UnwindConfig(
        market_sell_threshold=Decimal("0.050000"),
        passive_unwind_threshold=Decimal("0.010000"),
        max_hold_recovery_ms=100,
        min_best_bid=Decimal("0.010000"),
    )


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


def _unwind_manifest():
    evaluator = Si9UnwindEvaluator(_config())
    return evaluator.evaluate(
        cluster_id="cluster-1",
        hanging_legs=[("mkt-a", Decimal("2"), Decimal("0.40"))],
        current_bids={"mkt-a": Decimal("0.39")},
        unwind_reason="MANUAL_ABORT",
        original_manifest=_manifest(),
        unwind_timestamp_ms=200,
    )


def test_paper_best_bid_provider_satisfies_live_best_bid_provider_abc() -> None:
    assert isinstance(PaperBestBidProvider({}), LiveBestBidProvider)


def test_orderbook_best_bid_provider_satisfies_live_best_bid_provider_abc() -> None:
    class _StubTracker:
        asset_id = "mkt-a"

        @property
        def best_bid(self) -> float:
            return 0.19

        def snapshot(self):
            class _Snapshot:
                timestamp = 1.0

            return _Snapshot()

    assert isinstance(OrderbookBestBidProvider(_StubTracker()), LiveBestBidProvider)


def test_paper_unwind_executor_satisfies_unwind_executor_abc() -> None:
    assert isinstance(PaperUnwindExecutor(_config()), UnwindExecutor)


def test_paper_position_lifecycle_satisfies_position_lifecycle_interface_abc() -> None:
    assert isinstance(PaperPositionLifecycle(2), PositionLifecycleInterface)


def test_position_manager_lifecycle_satisfies_position_lifecycle_interface_abc() -> None:
    class _StubPositionManager:
        max_open = 2

        def get_open_positions(self):
            return []

        def cleanup_closed(self):
            return []

    assert isinstance(PositionManagerLifecycle(_StubPositionManager()), PositionLifecycleInterface)


def test_paper_escalation_policy_satisfies_escalation_policy_interface_abc() -> None:
    assert isinstance(PaperEscalationPolicy(Si9UnwindEvaluator(_config()), 500), EscalationPolicyInterface)


def test_all_paper_implementations_construct_with_valid_config() -> None:
    provider = PaperBestBidProvider({"mkt-a": Decimal("0.19")})
    executor = PaperUnwindExecutor(_config())
    lifecycle = PaperPositionLifecycle(3)
    policy = PaperEscalationPolicy(Si9UnwindEvaluator(_config()), 500)

    assert provider.get_best_bid("mkt-a") == Decimal("0.19")
    assert isinstance(executor, PaperUnwindExecutor)
    assert lifecycle.active_position_count == 0
    assert isinstance(policy, PaperEscalationPolicy)


def test_unwind_execution_receipt_is_frozen() -> None:
    receipt = PaperUnwindExecutor(_config()).execute_unwind(_unwind_manifest(), 250)

    with pytest.raises(FrozenInstanceError):
        receipt.action_taken = "MARKET_SELL"  # type: ignore[misc]


@pytest.mark.parametrize(
    "event_type",
    [
        "CTF_DISPATCHED",
        "CTF_REJECTED",
        "OFI_DISPATCHED",
        "OFI_REJECTED",
        "SI9_DISPATCHED",
        "SI9_REJECTED",
        "UNWIND_INITIATED",
        "UNWIND_ESCALATED",
        "UNWIND_COMPLETE",
        "POSITION_RESERVED",
        "POSITION_RELEASED",
        "TICK_PROCESSED",
    ],
)
def test_orchestrator_event_payload_serializes_for_all_event_types(event_type: str) -> None:
    event = OrchestratorEvent(
        event_type=event_type,  # type: ignore[arg-type]
        timestamp_ms=1000,
        source="SYSTEM",
        payload={"event_type": event_type, "cost": "0.01"},
        orchestrator_health="GREEN",
    )

    json.dumps(event.payload)


def test_paper_escalation_policy_should_surrender_always_false() -> None:
    policy = PaperEscalationPolicy(Si9UnwindEvaluator(_config()), 500)

    assert policy.should_surrender(_unwind_manifest(), 1000) is False


@pytest.mark.parametrize(
    "recommended_action",
    ["MARKET_SELL", "PASSIVE_UNWIND", "HOLD_FOR_RECOVERY"],
)
def test_paper_unwind_executor_records_action_without_raising_on_valid_manifest(recommended_action: str) -> None:
    manifest = replace(_unwind_manifest(), recommended_action=recommended_action)  # type: ignore[arg-type]
    executor = PaperUnwindExecutor(_config())

    receipt = executor.execute_unwind(manifest, 1000)

    assert isinstance(receipt, UnwindExecutionReceipt)
    assert receipt.execution_timestamp_ms == 1000