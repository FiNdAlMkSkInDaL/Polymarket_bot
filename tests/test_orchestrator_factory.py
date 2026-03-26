from __future__ import annotations

from decimal import Decimal

from src.execution.escalation_policy_interface import EscalationPolicyInterface
from src.execution.live_execution_boundary import LiveExecutionBoundary
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.detectors.ctf_peg_config import CtfPegConfig
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_signal_bridge import OfiSignalBridgeConfig
from src.execution.orchestrator_factory import build_live_orchestrator, build_paper_orchestrator
from src.execution.si9_paper_adapter import Si9PaperAdapterConfig
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.signal_coordination_bus import CoordinationBusConfig
from src.execution.unwind_executor_interface import PaperUnwindExecutor
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


class _StubOrderbookTracker:
    def __init__(self) -> None:
        self.asset_id = "mkt-a"
        self.best_bid = 0.47
        self.best_ask = 0.48

    def snapshot(self):
        class _Snapshot:
            best_bid = 0.47
            best_ask = 0.48
            timestamp = 1712345.678

        return _Snapshot()


class _StubPositionManager:
    max_open = 4

    def get_open_positions(self) -> list[object]:
        return []

    def cleanup_closed(self) -> list[object]:
        return []


class _StubVenueAdapter(VenueAdapter):
    def submit_order(
        self,
        market_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        order_type: str,
        client_order_id: str,
    ) -> VenueOrderResponse:
        _ = (market_id, side, price, size, order_type)
        return VenueOrderResponse(
            client_order_id=client_order_id,
            venue_order_id=f"venue-{client_order_id}",
            status="ACCEPTED",
            rejection_reason=None,
            venue_timestamp_ms=1000,
            latency_ms=1,
        )

    def cancel_order(self, client_order_id: str, market_id: str) -> VenueCancelResponse:
        _ = market_id
        return VenueCancelResponse(
            client_order_id=client_order_id,
            cancelled=True,
            rejection_reason=None,
            venue_timestamp_ms=1001,
        )

    def get_order_status(self, client_order_id: str) -> VenueOrderStatus:
        return VenueOrderStatus(
            client_order_id=client_order_id,
            venue_order_id=f"venue-{client_order_id}",
            fill_status="OPEN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("1"),
            average_fill_price=None,
        )

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        _ = asset_symbol
        return Decimal("100.000000")


class _StubEscalationPolicy(EscalationPolicyInterface):
    def should_escalate(self, manifest, current_timestamp_ms: int) -> bool:
        _ = (manifest, current_timestamp_ms)
        return False

    def should_surrender(self, manifest, current_timestamp_ms: int) -> bool:
        _ = (manifest, current_timestamp_ms)
        return False


def _ctf_config() -> CtfPegConfig:
    return CtfPegConfig(
        min_yield=Decimal("0.050000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        slippage_budget=Decimal("0.005000"),
        gas_ewma_alpha=Decimal("0.500000"),
        max_desync_ms=400,
    )


def _ctf_adapter_config() -> CtfPaperAdapterConfig:
    return CtfPaperAdapterConfig(
        max_expected_net_edge=Decimal("0.250000"),
        max_capital_per_signal=Decimal("25.000000"),
        default_anchor_volume=Decimal("10.000000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        cancel_on_stale_ms=250,
        max_size_per_leg=Decimal("8.000000"),
        mode="paper",
        bus=None,
    )


def _si9_adapter_config() -> Si9PaperAdapterConfig:
    return Si9PaperAdapterConfig(
        max_expected_net_edge=Decimal("0.050000"),
        max_capital_per_cluster=Decimal("20.000000"),
        max_leg_fill_wait_ms=100,
        cancel_on_stale_ms=50,
        mode="paper",
        unwind_config=Si9UnwindConfig(
            market_sell_threshold=Decimal("0.040000"),
            passive_unwind_threshold=Decimal("0.010000"),
            max_hold_recovery_ms=100,
            min_best_bid=Decimal("0.010000"),
        ),
        bus=None,
    )


def _bus_config() -> CoordinationBusConfig:
    return CoordinationBusConfig(
        slot_lease_ms=500,
        max_slots_per_source=10,
        max_total_slots=10,
        allow_same_source_reentry=False,
    )


def _guard_config() -> DispatchGuardConfig:
    return DispatchGuardConfig(
        dedup_window_ms=100,
        max_dispatches_per_source_per_window=10,
        rate_window_ms=200,
        circuit_breaker_threshold=2,
        circuit_breaker_reset_ms=300,
        max_open_positions_per_market=10,
    )


def _ofi_bridge_config() -> OfiSignalBridgeConfig:
    return OfiSignalBridgeConfig(
        max_capital_per_signal=Decimal("15.000000"),
        mode="paper",
        slot_side_lock=True,
        source_enabled=True,
    )


def _orchestrator_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        tick_interval_ms=50,
        max_pending_unwinds=4,
        max_concurrent_clusters=4,
        signal_sources_enabled=frozenset({"CTF", "SI9", "OFI"}),
    )


def _build_factory_orchestrator(
    *,
    ask_proxy: dict[str, Decimal] | None = None,
    clusters: list[tuple[str, list[str]]] | None = None,
) -> MultiSignalOrchestrator:
    return build_paper_orchestrator(
        ctf_config=_ctf_config(),
        ctf_adapter_config=_ctf_adapter_config(),
        si9_cluster_configs=clusters or [("cluster-1", ["mkt-a", "mkt-b", "mkt-c"])],
        si9_adapter_config=_si9_adapter_config(),
        ofi_bridge_config=_ofi_bridge_config(),
        bus_config=_bus_config(),
        guard_config=_guard_config(),
        orchestrator_config=_orchestrator_config(),
        ask_proxy={} if ask_proxy is None else dict(ask_proxy),
    )


def _build_live_factory_orchestrator() -> MultiSignalOrchestrator:
    return build_live_orchestrator(
        config=LiveOrchestratorConfig(
            orchestrator_config=_orchestrator_config(),
            bus_config=_bus_config(),
            guard_config=_guard_config(),
            ctf_adapter_config=_ctf_adapter_config(),
            si9_adapter_config=_si9_adapter_config(),
            ofi_bridge_config=_ofi_bridge_config(),
            ctf_peg_config=_ctf_config(),
            si9_cluster_configs=(("cluster-1", ("mkt-a", "mkt-b", "mkt-c")),),
            unwind_config=_si9_adapter_config().unwind_config,
            deployment_phase="PAPER",
            session_id="live-session-1",
            max_position_release_failures=2,
            heartbeat_interval_ms=500,
        ),
        orderbook_tracker=_StubOrderbookTracker(),
        position_manager=_StubPositionManager(),
        execution_boundary=LiveExecutionBoundary(
            venue_adapter=_StubVenueAdapter(),
            wallet_balance_provider=None,
            ofi_exit_router=None,
        ),
        unwind_executor=PaperUnwindExecutor(_si9_adapter_config().unwind_config),
        escalation_policy=_StubEscalationPolicy(),
    )


def test_factory_produces_multi_signal_orchestrator_instance() -> None:
    orchestrator = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})

    assert isinstance(orchestrator, MultiSignalOrchestrator)


def test_factory_with_empty_ask_proxy_constructs_without_error() -> None:
    orchestrator = _build_factory_orchestrator(ask_proxy={})

    assert orchestrator.best_bid_provider.get_best_bid("missing") is None


def test_factory_with_multiple_si9_cluster_configs_registers_all_clusters() -> None:
    orchestrator = _build_factory_orchestrator(
        clusters=[
            ("cluster-1", ["mkt-a", "mkt-b", "mkt-c"]),
            ("cluster-2", ["mkt-d", "mkt-e", "mkt-f"]),
        ],
        ask_proxy={"mkt-a": Decimal("0.19")},
    )

    assert orchestrator.si9_detector is not None
    assert set(orchestrator.si9_detector.cluster_members) == {"cluster-1", "cluster-2"}


def test_two_factory_calls_produce_independent_instances_without_shared_state() -> None:
    first = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})
    second = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})

    assert first is not second
    assert first.bus is not second.bus
    assert first.guard is not second.guard
    assert first.dispatcher is not second.dispatcher
    assert first.ctf_adapter is not second.ctf_adapter
    assert first.ofi_bridge is not second.ofi_bridge
    assert first.si9_adapter is not second.si9_adapter


def test_factory_built_orchestrator_snapshot_runs_immediately_after_construction() -> None:
    orchestrator = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})

    snapshot = orchestrator.orchestrator_snapshot(1000)

    assert snapshot.pending_unwind_count == 0
    assert snapshot.active_position_count == 0
    assert snapshot.health == "GREEN"


def test_factory_shared_bus_identity() -> None:
    orchestrator = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})

    assert orchestrator.ctf_adapter._bus is orchestrator.si9_adapter._bus
    assert orchestrator.ctf_adapter._bus is orchestrator.ofi_bridge.bus
    assert orchestrator.ctf_adapter._bus is orchestrator.bus


def test_factory_shared_guard_identity() -> None:
    orchestrator = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})

    assert orchestrator.ctf_adapter._guard is orchestrator.si9_adapter._guard
    assert orchestrator.ctf_adapter._guard is orchestrator.ofi_bridge.guard
    assert orchestrator.ctf_adapter._guard is orchestrator.guard
    assert orchestrator.dispatcher.guard is orchestrator.guard


def test_factory_shared_dispatcher_identity() -> None:
    orchestrator = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})

    assert orchestrator.ctf_adapter._dispatcher is orchestrator.si9_adapter._dispatcher
    assert orchestrator.ctf_adapter._dispatcher is orchestrator.ofi_bridge.dispatcher
    assert orchestrator.ctf_adapter._dispatcher is orchestrator.dispatcher


def test_paper_and_live_factories_produce_different_instances_without_shared_state() -> None:
    paper_orchestrator = _build_factory_orchestrator(ask_proxy={"mkt-a": Decimal("0.19")})
    live_orchestrator = _build_live_factory_orchestrator()

    assert paper_orchestrator is not live_orchestrator
    assert paper_orchestrator.bus is not live_orchestrator.bus
    assert paper_orchestrator.guard is not live_orchestrator.guard
    assert paper_orchestrator.dispatcher is not live_orchestrator.dispatcher
    assert paper_orchestrator.ctf_adapter is not live_orchestrator.ctf_adapter
    assert paper_orchestrator.ofi_bridge is not live_orchestrator.ofi_bridge
    assert paper_orchestrator.si9_adapter is not live_orchestrator.si9_adapter
