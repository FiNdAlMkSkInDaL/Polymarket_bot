from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.detectors.ctf_peg_config import CtfPegConfig
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.escalation_policy_interface import EscalationPolicyInterface
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.orchestrator_factory import build_live_orchestrator
from src.execution.orchestrator_health_monitor import HealthMonitorConfig, OrchestratorHealthMonitor
from src.execution.ofi_signal_bridge import OfiSignalBridgeConfig
from src.execution.priority_context import PriorityOrderContext
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_paper_adapter import Si9PaperAdapterConfig
from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindLeg, Si9UnwindManifest
from src.execution.signal_coordination_bus import CoordinationBusConfig
from src.execution.unwind_executor_interface import PaperUnwindExecutor
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus
from src.monitoring.orchestrator_telemetry_adapter import OrchestratorTelemetryAdapter
from src.execution.multi_signal_orchestrator import OrchestratorConfig


class _StubTracker:
    def __init__(self) -> None:
        self.asset_id = "mkt-a"
        self.best_bid = 0.47
        self.best_ask = 0.48
        self._timestamp_ms = 1_700_000_000_000

    def snapshot(self):
        return SimpleNamespace(best_bid=self.best_bid, best_ask=self.best_ask, timestamp=float(self._timestamp_ms))


class _StubPositionManager:
    def __init__(self) -> None:
        self.max_open = 4

    def get_open_positions(self) -> list[object]:
        return []

    def cleanup_closed(self) -> list[object]:
        return []


class _StubVenueAdapter(VenueAdapter):
    def submit_order(self, market_id: str, side: str, price: Decimal, size: Decimal, order_type: str, client_order_id: str) -> VenueOrderResponse:
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
        return VenueCancelResponse(client_order_id=client_order_id, cancelled=True, rejection_reason=None, venue_timestamp_ms=1001)

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
    def should_escalate(self, manifest: Si9UnwindManifest, current_timestamp_ms: int) -> bool:
        _ = (manifest, current_timestamp_ms)
        return False

    def should_surrender(self, manifest: Si9UnwindManifest, current_timestamp_ms: int) -> bool:
        _ = (manifest, current_timestamp_ms)
        return False


def _config() -> LiveOrchestratorConfig:
    return LiveOrchestratorConfig(
        orchestrator_config=OrchestratorConfig(
            tick_interval_ms=50,
            max_pending_unwinds=4,
            max_concurrent_clusters=4,
            signal_sources_enabled=frozenset({"CTF", "SI9", "OFI"}),
        ),
        bus_config=CoordinationBusConfig(
            slot_lease_ms=500,
            max_slots_per_source=10,
            max_total_slots=10,
            allow_same_source_reentry=False,
        ),
        guard_config=DispatchGuardConfig(
            dedup_window_ms=100,
            max_dispatches_per_source_per_window=10,
            rate_window_ms=200,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_ms=300,
            max_open_positions_per_market=10,
        ),
        ctf_adapter_config=CtfPaperAdapterConfig(
            max_expected_net_edge=Decimal("0.25"),
            max_capital_per_signal=Decimal("25"),
            default_anchor_volume=Decimal("10"),
            taker_fee_yes=Decimal("0.01"),
            taker_fee_no=Decimal("0.01"),
            cancel_on_stale_ms=250,
            max_size_per_leg=Decimal("8"),
            mode="paper",
            bus=None,
        ),
        si9_adapter_config=Si9PaperAdapterConfig(
            max_expected_net_edge=Decimal("0.05"),
            max_capital_per_cluster=Decimal("20"),
            max_leg_fill_wait_ms=100,
            cancel_on_stale_ms=50,
            mode="paper",
            unwind_config=_unwind_config(),
            bus=None,
        ),
        ofi_bridge_config=OfiSignalBridgeConfig(
            max_capital_per_signal=Decimal("15"),
            mode="paper",
            slot_side_lock=True,
            source_enabled=True,
        ),
        ctf_peg_config=CtfPegConfig(
            min_yield=Decimal("0.05"),
            taker_fee_yes=Decimal("0.01"),
            taker_fee_no=Decimal("0.01"),
            slippage_budget=Decimal("0.005"),
            gas_ewma_alpha=Decimal("0.5"),
            max_desync_ms=400,
        ),
        si9_cluster_configs=(("cluster-1", ("mkt-a", "mkt-b", "mkt-c")),),
        unwind_config=_unwind_config(),
        deployment_phase="PAPER",
        session_id="telemetry-session",
        max_position_release_failures=2,
        heartbeat_interval_ms=100,
    )


def _unwind_config() -> Si9UnwindConfig:
    return Si9UnwindConfig(
        market_sell_threshold=Decimal("0.040000"),
        passive_unwind_threshold=Decimal("0.010000"),
        max_hold_recovery_ms=100,
        min_best_bid=Decimal("0.010000"),
    )


def _build_adapter() -> tuple[OrchestratorTelemetryAdapter, OrchestratorHealthMonitor]:
    orchestrator = build_live_orchestrator(
        config=_config(),
        orderbook_tracker=_StubTracker(),
        position_manager=_StubPositionManager(),
        venue_adapter=_StubVenueAdapter(),
        unwind_executor=PaperUnwindExecutor(_unwind_config()),
        escalation_policy=_StubEscalationPolicy(),
    )
    health_monitor = OrchestratorHealthMonitor(
        orchestrator,
        HealthMonitorConfig(
            max_release_failures_before_halt=2,
            stale_snapshot_threshold_ms=500,
            min_heartbeat_interval_ms=100,
        ),
    )
    adapter = OrchestratorTelemetryAdapter(orchestrator, health_monitor)
    return adapter, health_monitor


def _seed_live_state(adapter: OrchestratorTelemetryAdapter, timestamp_ms: int) -> None:
    orchestrator = adapter._orchestrator
    guard = orchestrator.guard
    bus = orchestrator.bus
    context = PriorityOrderContext(
        market_id="mkt-a",
        side="YES",
        signal_source="OFI",
        target_price=Decimal("0.41"),
        anchor_volume=Decimal("4"),
        max_capital=Decimal("10"),
        conviction_scalar=Decimal("0.8"),
    )
    bus.request_slot("mkt-a", "YES", "OFI", timestamp_ms)
    bus.request_slot("mkt-a", "NO", "OFI", timestamp_ms)
    guard.record_dispatch(context, timestamp_ms)
    guard.record_suppression("OFI")
    manifest = Si9ExecutionManifest(
        cluster_id="cluster-1",
        legs=(
            Si9LegManifest("mkt-a", "YES", Decimal("0.31"), Decimal("2"), True, 0),
            Si9LegManifest("mkt-b", "YES", Decimal("0.32"), Decimal("2"), False, 1),
        ),
        net_edge=Decimal("0.02"),
        required_share_counts=Decimal("2"),
        bottleneck_market_id="mkt-a",
        manifest_timestamp_ms=timestamp_ms,
        max_leg_fill_wait_ms=200,
        cancel_on_stale_ms=300,
    )
    unwind_manifest = Si9UnwindManifest(
        cluster_id="cluster-1",
        hanging_legs=(
            Si9UnwindLeg(
                market_id="mkt-a",
                side="YES",
                filled_size=Decimal("2"),
                filled_price=Decimal("0.31"),
                current_best_bid=Decimal("0.29"),
                estimated_unwind_cost=Decimal("0.04"),
                leg_index=0,
            ),
        ),
        unwind_reason="BUS_EVICTED",
        original_manifest=manifest,
        unwind_timestamp_ms=timestamp_ms,
        total_estimated_unwind_cost=Decimal("0.04"),
        recommended_action="PASSIVE_UNWIND",
    )
    orchestrator._pending_unwinds["cluster-1"] = unwind_manifest


def test_constructor_requires_orchestrator() -> None:
    with pytest.raises(ValueError, match="orchestrator"):
        OrchestratorTelemetryAdapter(None, object())  # type: ignore[arg-type]


def test_constructor_requires_health_monitor() -> None:
    adapter, _ = _build_adapter()

    with pytest.raises(ValueError, match="health_monitor"):
        OrchestratorTelemetryAdapter(adapter._orchestrator, None)  # type: ignore[arg-type]


def test_export_health_snapshot_includes_health_monitor_fields() -> None:
    adapter, _ = _build_adapter()

    snapshot = adapter.export_health_snapshot(1_700_000_000_000)

    assert snapshot["health_monitor"]["is_safe_to_trade"] is True
    assert snapshot["health_monitor"]["consecutive_release_failures"] == 0
    assert snapshot["health_monitor"]["halt_reason"] is None


def test_export_health_snapshot_includes_dispatch_guard_state() -> None:
    adapter, _ = _build_adapter()
    _seed_live_state(adapter, 1_700_000_000_000)

    snapshot = adapter.export_health_snapshot(1_700_000_000_050)

    assert snapshot["dispatch_guard"]["circuit_breaker_status"]["state"] == "CLOSED"
    assert snapshot["dispatch_guard"]["dispatch_rate_counters"]["OFI"] == 1
    assert snapshot["dispatch_guard"]["active_open_positions_by_market"]["mkt-a"] == 1


def test_export_health_snapshot_includes_coordination_bus_leases_and_horizons() -> None:
    adapter, _ = _build_adapter()
    _seed_live_state(adapter, 1_700_000_000_000)

    snapshot = adapter.export_health_snapshot(1_700_000_000_100)
    leases = snapshot["coordination_bus"]["active_slot_leases"]

    assert len(leases) == 2
    assert leases[0]["lease_expires_ms"] == 1_700_000_000_500
    assert leases[0]["expiration_horizon_ms"] == 400


def test_export_health_snapshot_includes_active_unwinds() -> None:
    adapter, _ = _build_adapter()
    _seed_live_state(adapter, 1_700_000_000_000)

    snapshot = adapter.export_health_snapshot(1_700_000_000_100)
    active_unwinds = snapshot["unwind_ledger"]["active_unwinds"]

    assert len(active_unwinds) == 1
    assert active_unwinds[0]["cluster_id"] == "cluster-1"
    assert active_unwinds[0]["total_estimated_unwind_cost"] == "0.04"
    assert active_unwinds[0]["hanging_legs"][0]["filled_price"] == "0.31"


def test_export_health_snapshot_converts_release_failures_into_serializable_health_state() -> None:
    adapter, health_monitor = _build_adapter()
    health_monitor.record_position_release_failure()
    health_monitor.record_position_release_failure()

    snapshot = adapter.export_health_snapshot(1_700_000_000_000)

    assert snapshot["health_monitor"]["consecutive_release_failures"] == 2
    assert snapshot["health_monitor"]["is_safe_to_trade"] is False
    assert snapshot["health_monitor"]["halt_reason"] == "RELEASE_FAILURE_THRESHOLD_REACHED"


def test_export_health_snapshot_json_dumps_without_type_error() -> None:
    adapter, _ = _build_adapter()
    _seed_live_state(adapter, 1_700_000_000_000)

    payload = adapter.export_health_snapshot(1_700_000_000_100)

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)


def test_export_health_snapshot_evicts_expired_leases_using_injected_timestamp() -> None:
    adapter, _ = _build_adapter()
    _seed_live_state(adapter, 1_700_000_000_000)

    snapshot = adapter.export_health_snapshot(1_700_000_000_600)

    assert snapshot["coordination_bus"]["total_active_slots"] == 0
    assert snapshot["coordination_bus"]["active_slot_leases"] == []


def test_export_health_snapshot_returns_empty_unwind_ledger_when_no_pending_unwinds() -> None:
    adapter, _ = _build_adapter()

    snapshot = adapter.export_health_snapshot(1_700_000_000_000)

    assert snapshot["unwind_ledger"]["active_unwinds"] == []