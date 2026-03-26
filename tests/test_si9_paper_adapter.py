from __future__ import annotations

from decimal import Decimal

from src.execution.dispatch_guard import GuardDecision
from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.mev_router import MevExecutionRouter
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.si9_paper_adapter import Si9PaperAdapter, Si9PaperAdapterConfig
from src.execution.si9_paper_ledger import Si9PaperLedger
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus
from src.detectors.si9_cluster_config import Si9ClusterConfig
from src.signals.si9_matrix_detector import Si9MatrixDetector
from src.signals.si9_matrix_detector import Si9MatrixSignal


class RejectSpecificMarketGuard(DispatchGuard):
    def __init__(self, config: DispatchGuardConfig, market_id: str):
        super().__init__(config)
        self._market_id = market_id

    def check(self, context, current_timestamp_ms: int):  # type: ignore[override]
        if context.market_id == self._market_id:
            return GuardDecision(allowed=False, reason="RATE_EXCEEDED")
        return super().check(context, current_timestamp_ms)


def _signal() -> Si9MatrixSignal:
    return Si9MatrixSignal(
        cluster_id="cluster-1",
        market_ids=("mkt-a", "mkt-b", "mkt-c"),
        best_yes_asks={
            "mkt-a": Decimal("0.180000"),
            "mkt-b": Decimal("0.190000"),
            "mkt-c": Decimal("0.200000"),
        },
        ask_sizes={
            "mkt-a": Decimal("5.000000"),
            "mkt-b": Decimal("5.000000"),
            "mkt-c": Decimal("5.000000"),
        },
        total_yes_ask=Decimal("0.570000"),
        gross_edge=Decimal("0.430000"),
        net_edge=Decimal("0.030000"),
        target_yield=Decimal("0.020000"),
        bottleneck_market_id="mkt-b",
        required_share_counts=Decimal("2.500000"),
    )


def _guard_config(**overrides: int) -> DispatchGuardConfig:
    values = {
        "dedup_window_ms": 100,
        "max_dispatches_per_source_per_window": 10,
        "rate_window_ms": 200,
        "circuit_breaker_threshold": 2,
        "circuit_breaker_reset_ms": 300,
        "max_open_positions_per_market": 10,
    }
    values.update(overrides)
    return DispatchGuardConfig(**values)


def _bus_config(**overrides) -> CoordinationBusConfig:
    values = {
        "slot_lease_ms": 100,
        "max_slots_per_source": 10,
        "max_total_slots": 10,
        "allow_same_source_reentry": True,
    }
    values.update(overrides)
    return CoordinationBusConfig(**values)


def _snapshot_provider(market_id: str) -> dict[str, float]:
    _ = market_id
    return {
        "yes_bid": 0.45,
        "yes_ask": 0.55,
        "no_bid": 0.45,
        "no_ask": 0.55,
    }


def _make_adapter(
    *,
    guard_config: DispatchGuardConfig | None = None,
    bus: SignalCoordinationBus | None = None,
) -> tuple[Si9PaperAdapter, SignalCoordinationBus]:
    shared_bus = bus or SignalCoordinationBus(_bus_config())
    dispatcher = PriorityDispatcher(MevExecutionRouter(_snapshot_provider), "paper")
    adapter = Si9PaperAdapter(
        dispatcher,
        DispatchGuard(guard_config or _guard_config()),
        Si9PaperLedger(),
        Si9PaperAdapterConfig(
            max_expected_net_edge=Decimal("0.050000"),
            max_capital_per_cluster=Decimal("20.000000"),
            max_leg_fill_wait_ms=100,
            cancel_on_stale_ms=50,
            mode="paper",
            unwind_config=Si9UnwindConfig(
                market_sell_threshold=Decimal("1.000000"),
                passive_unwind_threshold=Decimal("0.050000"),
                max_hold_recovery_ms=100,
                min_best_bid=Decimal("0.010000"),
            ),
            bus=shared_bus,
        ),
    )
    return adapter, shared_bus


def test_bus_grants_bottleneck_leg_slot() -> None:
    adapter, shared_bus = _make_adapter()

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.cluster_outcome == "FULL_FILL"
    assert receipt.slot_decisions
    assert receipt.slot_decisions[0].granted is True
    assert receipt.slot_decisions[0].market_id == "mkt-b"
    assert shared_bus.bus_snapshot(1000).total_active_slots == 0


def test_bus_releases_on_hanging_leg_resolution() -> None:
    adapter, shared_bus = _make_adapter(
        guard_config=_guard_config(max_dispatches_per_source_per_window=1, rate_window_ms=200),
    )

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.cluster_outcome == "HANGING_LEG"
    assert shared_bus.bus_snapshot(1000).total_active_slots == 0


def test_hanging_leg_produces_unwind_manifest_on_receipt() -> None:
    adapter, _ = _make_adapter(
        guard_config=_guard_config(max_dispatches_per_source_per_window=1, rate_window_ms=200),
    )

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.unwind_manifest is not None


def test_unwind_manifest_recommended_action_is_valid_literal() -> None:
    adapter, _ = _make_adapter(
        guard_config=_guard_config(max_dispatches_per_source_per_window=1, rate_window_ms=200),
    )

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.unwind_manifest is not None
    assert receipt.unwind_manifest.recommended_action in {"MARKET_SELL", "PASSIVE_UNWIND", "HOLD_FOR_RECOVERY"}


def test_unwind_manifest_reason_matches_second_leg_hanging_case() -> None:
    shared_bus = SignalCoordinationBus(_bus_config())
    dispatcher = PriorityDispatcher(MevExecutionRouter(_snapshot_provider), "paper")
    adapter = Si9PaperAdapter(
        dispatcher,
        RejectSpecificMarketGuard(_guard_config(), "mkt-a"),
        Si9PaperLedger(),
        Si9PaperAdapterConfig(
            max_expected_net_edge=Decimal("0.050000"),
            max_capital_per_cluster=Decimal("20.000000"),
            max_leg_fill_wait_ms=100,
            cancel_on_stale_ms=50,
            mode="paper",
            unwind_config=Si9UnwindConfig(
                market_sell_threshold=Decimal("1.000000"),
                passive_unwind_threshold=Decimal("0.050000"),
                max_hold_recovery_ms=100,
                min_best_bid=Decimal("0.010000"),
            ),
            bus=shared_bus,
        ),
    )

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.cluster_outcome == "HANGING_LEG"
    assert receipt.unwind_manifest is not None
    assert receipt.unwind_manifest.unwind_reason == "SECOND_LEG_REJECTED"


def test_ledger_total_unwind_manifests_increments_on_hanging_leg() -> None:
    adapter, _ = _make_adapter(
        guard_config=_guard_config(max_dispatches_per_source_per_window=1, rate_window_ms=200),
    )

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.unwind_manifest is not None
    assert adapter._ledger.snapshot().total_unwind_manifests == 1


def test_ledger_gross_estimated_unwind_cost_accumulates_correctly() -> None:
    adapter, _ = _make_adapter(
        guard_config=_guard_config(max_dispatches_per_source_per_window=1, rate_window_ms=200),
    )

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.unwind_manifest is not None
    assert adapter._ledger.snapshot().gross_estimated_unwind_cost == receipt.unwind_manifest.total_estimated_unwind_cost


def test_cluster_snapshot_with_explicit_eval_timestamp_produces_correct_tradeable_state() -> None:
    detector = Si9MatrixDetector(
        Si9ClusterConfig(
            target_yield=Decimal("0.02"),
            taker_fee_per_leg=Decimal("0.002"),
            slippage_budget=Decimal("0.001"),
            ghost_town_floor=Decimal("0.85"),
            implausible_edge_ceil=Decimal("0.15"),
            max_ask_age_ms=100,
            min_cluster_size=3,
        )
    )
    detector.register_cluster("cluster-1", ["mkt-a", "mkt-b", "mkt-c"])
    detector.update_best_yes_ask("mkt-a", Decimal("0.31"), Decimal("2"), 0)
    detector.update_best_yes_ask("mkt-b", Decimal("0.32"), Decimal("2"), 101)
    detector.update_best_yes_ask("mkt-c", Decimal("0.33"), Decimal("2"), 101)

    snapshot = detector.cluster_snapshot("cluster-1", eval_timestamp_ms=100)

    assert snapshot.eval_timestamp_ms == 100
    assert snapshot.would_be_tradeable is True
    assert snapshot.suppression_reason is None


def test_cluster_snapshot_with_none_eval_timestamp_preserves_existing_behavior() -> None:
    detector = Si9MatrixDetector(
        Si9ClusterConfig(
            target_yield=Decimal("0.02"),
            taker_fee_per_leg=Decimal("0.002"),
            slippage_budget=Decimal("0.001"),
            ghost_town_floor=Decimal("0.85"),
            implausible_edge_ceil=Decimal("0.15"),
            max_ask_age_ms=100,
            min_cluster_size=3,
        )
    )
    detector.register_cluster("cluster-1", ["mkt-a", "mkt-b", "mkt-c"])
    detector.update_best_yes_ask("mkt-a", Decimal("0.31"), Decimal("2"), 0)
    detector.update_best_yes_ask("mkt-b", Decimal("0.32"), Decimal("2"), 101)
    detector.update_best_yes_ask("mkt-c", Decimal("0.33"), Decimal("2"), 101)

    snapshot = detector.cluster_snapshot("cluster-1")

    assert snapshot.eval_timestamp_ms is None
    assert snapshot.would_be_tradeable is False
    assert snapshot.suppression_reason == "STALE_LEG"
