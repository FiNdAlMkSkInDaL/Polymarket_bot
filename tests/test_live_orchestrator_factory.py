from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.detectors.ctf_peg_config import CtfPegConfig
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.escalation_policy_interface import EscalationPolicyInterface
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.orchestrator_factory import build_live_orchestrator
from src.execution.orchestrator_health_monitor import HealthMonitorConfig, HealthReport, OrchestratorHealthMonitor
from src.execution.ofi_signal_bridge import OfiSignalBridgeConfig
from src.execution.si9_paper_adapter import Si9PaperAdapterConfig
from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindManifest
from src.execution.signal_coordination_bus import CoordinationBusConfig
from src.execution.unwind_executor_interface import PaperUnwindExecutor, UnwindExecutionReceipt, UnwindExecutor
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


class _StubOrderbookTracker:
    def __init__(self, *, asset_id: str = "mkt-a", best_bid: float = 0.47, best_ask: float = 0.48, timestamp: float = 1712345.678) -> None:
        self.asset_id = asset_id
        self.best_bid = best_bid
        self.best_ask = best_ask
        self._timestamp = timestamp

    def snapshot(self):
        return SimpleNamespace(
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            timestamp=self._timestamp,
        )


class _StubPositionManager:
    def __init__(self, *, max_open: int = 4, open_positions: list[object] | None = None) -> None:
        self.max_open = max_open
        self._open_positions = list(open_positions or [])
        self.cleanup_calls = 0

    def get_open_positions(self) -> list[object]:
        return list(self._open_positions)

    def cleanup_closed(self) -> list[object]:
        self.cleanup_calls += 1
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
            latency_ms=2,
        )

    def cancel_order(
        self,
        client_order_id: str,
        market_id: str,
    ) -> VenueCancelResponse:
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
    def should_escalate(
        self,
        manifest: Si9UnwindManifest,
        current_timestamp_ms: int,
    ) -> bool:
        _ = (manifest, current_timestamp_ms)
        return False

    def should_surrender(
        self,
        manifest: Si9UnwindManifest,
        current_timestamp_ms: int,
    ) -> bool:
        _ = (manifest, current_timestamp_ms)
        return False


class _SnapshotStubOrchestrator:
    def __init__(self, snapshot) -> None:
        self._snapshot = snapshot

    def orchestrator_snapshot(self, current_timestamp_ms: int):
        _ = current_timestamp_ms
        return self._snapshot


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


def _si9_adapter_config(*, mode: str = "paper") -> Si9PaperAdapterConfig:
    return Si9PaperAdapterConfig(
        max_expected_net_edge=Decimal("0.050000"),
        max_capital_per_cluster=Decimal("20.000000"),
        max_leg_fill_wait_ms=100,
        cancel_on_stale_ms=50,
        mode=mode,
        unwind_config=_unwind_config(),
        bus=None,
    )


def _ofi_bridge_config(*, mode: str = "paper") -> OfiSignalBridgeConfig:
    return OfiSignalBridgeConfig(
        max_capital_per_signal=Decimal("15.000000"),
        mode=mode,
        slot_side_lock=True,
        source_enabled=True,
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


def _orchestrator_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        tick_interval_ms=50,
        max_pending_unwinds=4,
        max_concurrent_clusters=4,
        signal_sources_enabled=frozenset({"CTF", "SI9", "OFI"}),
    )


def _unwind_config() -> Si9UnwindConfig:
    return Si9UnwindConfig(
        market_sell_threshold=Decimal("0.040000"),
        passive_unwind_threshold=Decimal("0.010000"),
        max_hold_recovery_ms=100,
        min_best_bid=Decimal("0.010000"),
    )


def _live_config(
    *,
    deployment_phase: str = "PAPER",
    si9_mode: str = "paper",
    ofi_mode: str = "paper",
    session_id: str = "live-session-1",
    max_position_release_failures: int = 2,
    heartbeat_interval_ms: int = 500,
) -> LiveOrchestratorConfig:
    return LiveOrchestratorConfig(
        orchestrator_config=_orchestrator_config(),
        bus_config=_bus_config(),
        guard_config=_guard_config(),
        ctf_adapter_config=_ctf_adapter_config(),
        si9_adapter_config=_si9_adapter_config(mode=si9_mode),
        ofi_bridge_config=_ofi_bridge_config(mode=ofi_mode),
        ctf_peg_config=_ctf_config(),
        si9_cluster_configs=(("cluster-1", ("mkt-a", "mkt-b", "mkt-c")),),
        unwind_config=_unwind_config(),
        deployment_phase=deployment_phase,
        session_id=session_id,
        max_position_release_failures=max_position_release_failures,
        heartbeat_interval_ms=heartbeat_interval_ms,
    )


def _build_live_factory_orchestrator(*, deployment_phase: str = "PAPER") -> MultiSignalOrchestrator:
    return build_live_orchestrator(
        config=_live_config(deployment_phase=deployment_phase),
        orderbook_tracker=_StubOrderbookTracker(),
        position_manager=_StubPositionManager(),
        venue_adapter=_StubVenueAdapter(),
        unwind_executor=PaperUnwindExecutor(_unwind_config()),
        escalation_policy=_StubEscalationPolicy(),
    )


def _health_config() -> HealthMonitorConfig:
    return HealthMonitorConfig(
        max_release_failures_before_halt=2,
        stale_snapshot_threshold_ms=500,
        min_heartbeat_interval_ms=100,
    )


def test_live_orchestrator_config_valid_construction_passes() -> None:
    config = _live_config()

    assert config.deployment_phase == "PAPER"
    assert config.session_id == "live-session-1"


def test_live_orchestrator_config_rejects_empty_session_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        _live_config(session_id="   ")


def test_live_orchestrator_config_live_with_dry_run_si9_adapter_raises() -> None:
    with pytest.raises(ValueError, match="si9_adapter_config.mode"):
        _live_config(deployment_phase="LIVE", si9_mode="dry_run")


def test_live_orchestrator_config_live_with_dry_run_ofi_adapter_raises() -> None:
    with pytest.raises(ValueError, match="ofi_bridge_config.mode"):
        _live_config(deployment_phase="LIVE", ofi_mode="dry_run")


def test_live_orchestrator_config_paper_with_dry_run_adapter_config_passes() -> None:
    config = _live_config(deployment_phase="PAPER", si9_mode="dry_run", ofi_mode="dry_run")

    assert config.si9_adapter_config.mode == "dry_run"
    assert config.ofi_bridge_config.mode == "dry_run"


def test_live_orchestrator_config_rejects_invalid_release_failure_threshold() -> None:
    with pytest.raises(ValueError, match="max_position_release_failures"):
        _live_config(max_position_release_failures=0)


def test_live_orchestrator_config_rejects_non_positive_heartbeat_interval() -> None:
    with pytest.raises(ValueError, match="heartbeat_interval_ms"):
        _live_config(heartbeat_interval_ms=0)


def test_build_live_orchestrator_raises_runtime_error_when_venue_adapter_interface_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.execution.venue_adapter_interface":
            raise ImportError("forced missing interface")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    with pytest.raises(RuntimeError, match="VenueAdapter interface not yet available"):
        build_live_orchestrator(
            config=_live_config(),
            orderbook_tracker=_StubOrderbookTracker(),
            position_manager=_StubPositionManager(),
            venue_adapter=_StubVenueAdapter(),
            unwind_executor=PaperUnwindExecutor(_unwind_config()),
            escalation_policy=_StubEscalationPolicy(),
        )


def test_build_live_orchestrator_raises_type_error_for_wrong_venue_adapter_type() -> None:
    with pytest.raises(TypeError, match="venue_adapter must implement VenueAdapter"):
        build_live_orchestrator(
            config=_live_config(),
            orderbook_tracker=_StubOrderbookTracker(),
            position_manager=_StubPositionManager(),
            venue_adapter=object(),
            unwind_executor=PaperUnwindExecutor(_unwind_config()),
            escalation_policy=_StubEscalationPolicy(),
        )


def test_build_live_orchestrator_returns_multi_signal_orchestrator() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert isinstance(orchestrator, MultiSignalOrchestrator)


def test_build_live_orchestrator_live_phase_constructs_dispatcher_with_client_order_id_generator() -> None:
    orchestrator = _build_live_factory_orchestrator(deployment_phase="LIVE")

    assert isinstance(orchestrator, MultiSignalOrchestrator)
    assert orchestrator.dispatcher._client_order_id_generator is not None


def test_two_live_factory_calls_produce_independent_instances_without_shared_state() -> None:
    first = _build_live_factory_orchestrator()
    second = _build_live_factory_orchestrator()

    assert first is not second
    assert first.bus is not second.bus
    assert first.guard is not second.guard
    assert first.dispatcher is not second.dispatcher
    assert first.ctf_adapter is not second.ctf_adapter
    assert first.ofi_bridge is not second.ofi_bridge
    assert first.si9_adapter is not second.si9_adapter


def test_live_factory_shared_bus_identity() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert orchestrator.ctf_adapter._bus is orchestrator.si9_adapter._bus
    assert orchestrator.ctf_adapter._bus is orchestrator.ofi_bridge.bus
    assert orchestrator.ctf_adapter._bus is orchestrator.bus


def test_live_factory_shared_guard_identity() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert orchestrator.ctf_adapter._guard is orchestrator.si9_adapter._guard
    assert orchestrator.ctf_adapter._guard is orchestrator.ofi_bridge.guard
    assert orchestrator.ctf_adapter._guard is orchestrator.guard
    assert orchestrator.dispatcher.guard is orchestrator.guard


def test_live_factory_shared_dispatcher_identity() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert orchestrator.ctf_adapter._dispatcher is orchestrator.si9_adapter._dispatcher
    assert orchestrator.ctf_adapter._dispatcher is orchestrator.ofi_bridge.dispatcher
    assert orchestrator.ctf_adapter._dispatcher is orchestrator.dispatcher


def test_live_factory_registers_all_si9_clusters() -> None:
    config = replace(
        _live_config(),
        si9_cluster_configs=(
            ("cluster-1", ("mkt-a", "mkt-b", "mkt-c")),
            ("cluster-2", ("mkt-d", "mkt-e", "mkt-f")),
        ),
    )
    orchestrator = build_live_orchestrator(
        config=config,
        orderbook_tracker=_StubOrderbookTracker(),
        position_manager=_StubPositionManager(),
        venue_adapter=_StubVenueAdapter(),
        unwind_executor=PaperUnwindExecutor(_unwind_config()),
        escalation_policy=_StubEscalationPolicy(),
    )

    assert orchestrator.si9_detector is not None
    assert set(orchestrator.si9_detector.cluster_members) == {"cluster-1", "cluster-2"}


def test_live_factory_uses_orderbook_best_bid_provider() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert orchestrator.best_bid_provider.get_best_bid("mkt-a") == Decimal("0.47")


def test_live_factory_uses_position_manager_lifecycle_bridge() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert orchestrator.position_lifecycle.__class__.__name__ == "PositionManagerLifecycle"


def test_orchestrator_health_monitor_construction_passes_with_valid_config() -> None:
    monitor = OrchestratorHealthMonitor(_build_live_factory_orchestrator(), _health_config())

    assert isinstance(monitor, OrchestratorHealthMonitor)


def test_is_safe_to_trade_returns_true_on_clean_green_snapshot() -> None:
    monitor = OrchestratorHealthMonitor(_build_live_factory_orchestrator(), _health_config())

    assert monitor.is_safe_to_trade(1000) is True


def test_is_safe_to_trade_returns_false_when_health_is_red() -> None:
    base_snapshot = _build_live_factory_orchestrator().orchestrator_snapshot(1000)
    red_snapshot = replace(base_snapshot, health="RED")
    monitor = OrchestratorHealthMonitor(_SnapshotStubOrchestrator(red_snapshot), _health_config())

    assert monitor.is_safe_to_trade(1000) is False


def test_is_safe_to_trade_returns_false_when_health_is_yellow() -> None:
    base_snapshot = _build_live_factory_orchestrator().orchestrator_snapshot(1000)
    yellow_snapshot = replace(base_snapshot, health="YELLOW")
    monitor = OrchestratorHealthMonitor(_SnapshotStubOrchestrator(yellow_snapshot), _health_config())

    assert monitor.is_safe_to_trade(1000) is False


def test_is_safe_to_trade_returns_false_after_max_release_failures() -> None:
    monitor = OrchestratorHealthMonitor(_build_live_factory_orchestrator(), _health_config())

    monitor.record_position_release_failure()
    monitor.record_position_release_failure()

    assert monitor.is_safe_to_trade(1000) is False


def test_is_safe_to_trade_returns_false_when_snapshot_is_stale() -> None:
    base_snapshot = _build_live_factory_orchestrator().orchestrator_snapshot(1000)
    stale_snapshot = replace(base_snapshot, timestamp_ms=100)
    monitor = OrchestratorHealthMonitor(_SnapshotStubOrchestrator(stale_snapshot), _health_config())

    assert monitor.is_safe_to_trade(1000) is False


def test_is_safe_to_trade_returns_false_when_heartbeat_gap_exceeds_threshold() -> None:
    orchestrator = _SnapshotStubOrchestrator(_build_live_factory_orchestrator().orchestrator_snapshot(1000))
    monitor = OrchestratorHealthMonitor(orchestrator, _health_config())

    assert monitor.is_safe_to_trade(1000) is True
    assert monitor.is_safe_to_trade(1401) is False


def test_record_position_release_failure_increments_counter_correctly() -> None:
    monitor = OrchestratorHealthMonitor(_build_live_factory_orchestrator(), _health_config())

    monitor.record_position_release_failure()
    report = monitor.check(1000)

    assert report.consecutive_release_failures == 1


def test_reset_release_failure_count_returns_counter_to_zero_and_reenables_trading() -> None:
    monitor = OrchestratorHealthMonitor(_build_live_factory_orchestrator(), _health_config())

    monitor.record_position_release_failure()
    monitor.record_position_release_failure()
    assert monitor.is_safe_to_trade(1000) is False

    monitor.reset_release_failure_count()

    assert monitor.is_safe_to_trade(1001) is True


def test_health_report_is_frozen() -> None:
    report = HealthReport(
        timestamp_ms=1000,
        orchestrator_health="GREEN",
        is_safe_to_trade=True,
        consecutive_release_failures=0,
        last_snapshot_age_ms=0,
        heartbeat_ok=True,
        halt_reason=None,
    )

    with pytest.raises(FrozenInstanceError):
        report.halt_reason = "STOP"  # type: ignore[misc]


def test_halt_reason_is_none_when_safe_to_trade_is_true() -> None:
    monitor = OrchestratorHealthMonitor(_build_live_factory_orchestrator(), _health_config())

    report = monitor.check(1000)

    assert report.is_safe_to_trade is True
    assert report.halt_reason is None


def test_halt_reason_is_populated_with_specific_reason_when_not_safe_to_trade() -> None:
    base_snapshot = _build_live_factory_orchestrator().orchestrator_snapshot(1000)
    red_snapshot = replace(base_snapshot, health="RED")
    monitor = OrchestratorHealthMonitor(_SnapshotStubOrchestrator(red_snapshot), _health_config())

    report = monitor.check(1000)

    assert report.is_safe_to_trade is False
    assert report.halt_reason == "ORCHESTRATOR_HEALTH_RED"