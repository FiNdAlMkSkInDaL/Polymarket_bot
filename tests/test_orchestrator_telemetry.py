from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.live_execution_boundary import LiveExecutionBoundary
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.multi_signal_orchestrator import OrchestratorConfig
from src.execution.orchestrator_factory import build_live_orchestrator
from src.execution.orchestrator_health_monitor import HealthMonitorConfig, OrchestratorHealthMonitor
from src.execution.priority_context import PriorityOrderContext
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus
from src.monitoring.orchestrator_telemetry_adapter import OrchestratorTelemetryAdapter


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


def _config() -> LiveOrchestratorConfig:
    return LiveOrchestratorConfig(
        orchestrator_config=OrchestratorConfig(
            tick_interval_ms=50,
            max_pending_unwinds=0,
            max_concurrent_clusters=1,
            signal_sources_enabled=frozenset({"OFI", "CONTAGION", "REWARD"}),
        ),
        guard_config=DispatchGuardConfig(
            dedup_window_ms=100,
            max_dispatches_per_source_per_window=10,
            rate_window_ms=200,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_ms=300,
            max_open_positions_per_market=10,
        ),
        deployment_phase="PAPER",
        session_id="telemetry-session",
        max_position_release_failures=2,
        heartbeat_interval_ms=100,
    )


def _build_adapter() -> tuple[OrchestratorTelemetryAdapter, OrchestratorHealthMonitor]:
    orchestrator = build_live_orchestrator(
        config=_config(),
        orderbook_tracker=_StubTracker(),
        position_manager=_StubPositionManager(),
        execution_boundary=LiveExecutionBoundary(
            venue_adapter=_StubVenueAdapter(),
            wallet_balance_provider=None,
            ofi_exit_router=None,
        ),
    )
    health_monitor = OrchestratorHealthMonitor(
        orchestrator,
        HealthMonitorConfig(
            max_release_failures_before_halt=2,
            stale_snapshot_threshold_ms=500,
            min_heartbeat_interval_ms=100,
        ),
    )
    return OrchestratorTelemetryAdapter(orchestrator, health_monitor), health_monitor


def _record_guard_activity(adapter: OrchestratorTelemetryAdapter, timestamp_ms: int) -> None:
    context = PriorityOrderContext(
        market_id="mkt-a",
        side="YES",
        signal_source="OFI",
        target_price=Decimal("0.41"),
        anchor_volume=Decimal("4"),
        max_capital=Decimal("10"),
        conviction_scalar=Decimal("0.8"),
    )
    adapter._orchestrator.guard.record_dispatch(context, timestamp_ms)
    adapter._orchestrator.guard.record_suppression("OFI")


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
    _record_guard_activity(adapter, 1_700_000_000_000)

    snapshot = adapter.export_health_snapshot(1_700_000_000_050)

    assert snapshot["dispatch_guard"]["circuit_breaker_status"]["state"] == "CLOSED"
    assert snapshot["dispatch_guard"]["dispatch_rate_counters"]["OFI"] == 1
    assert snapshot["dispatch_guard"]["active_open_positions_by_market"]["mkt-a"] == 1


def test_export_health_snapshot_reports_empty_coordination_bus_for_lean_kernel() -> None:
    adapter, _ = _build_adapter()

    snapshot = adapter.export_health_snapshot(1_700_000_000_100)

    assert snapshot["coordination_bus"]["total_active_slots"] == 0
    assert snapshot["coordination_bus"]["slots_by_source"] == {}
    assert snapshot["coordination_bus"]["active_slot_leases"] == []


def test_export_health_snapshot_reports_empty_unwind_ledger_for_lean_kernel() -> None:
    adapter, _ = _build_adapter()

    snapshot = adapter.export_health_snapshot(1_700_000_000_100)

    assert snapshot["unwind_ledger"]["active_unwinds"] == []


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
    _record_guard_activity(adapter, 1_700_000_000_000)

    payload = adapter.export_health_snapshot(1_700_000_000_100)

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
