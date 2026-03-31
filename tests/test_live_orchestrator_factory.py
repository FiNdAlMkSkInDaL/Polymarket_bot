from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.live_execution_boundary import LiveExecutionBoundary
from src.execution.live_wallet_balance import LiveWalletBalanceProvider
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig, OrchestratorSnapshot
from src.execution.orchestrator_factory import build_live_orchestrator
from src.execution.orchestrator_health_monitor import HealthMonitorConfig, HealthReport, OrchestratorHealthMonitor
from src.execution.priority_context import PriorityOrderContext, RewardExecutionHints
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

    def get_open_positions(self) -> list[object]:
        return list(self._open_positions)


class _StubVenueAdapter(VenueAdapter):
    def submit_order(self, market_id: str, side: str, price: Decimal, size: Decimal, order_type: str, client_order_id: str) -> VenueOrderResponse:
        _ = (market_id, side, price, size, order_type)
        return VenueOrderResponse(
            client_order_id=client_order_id,
            venue_order_id=f"venue-{client_order_id}",
            status="ACCEPTED",
            rejection_reason=None,
            venue_timestamp_ms=1000,
            latency_ms=2,
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


class _SnapshotStubOrchestrator:
    def __init__(self, snapshot) -> None:
        self._snapshot = snapshot

    def orchestrator_snapshot(self, current_timestamp_ms: int):
        _ = current_timestamp_ms
        return self._snapshot


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
        max_pending_unwinds=0,
        max_concurrent_clusters=1,
        signal_sources_enabled=frozenset({"OFI", "CONTAGION", "REWARD"}),
    )


def _live_config(
    *,
    deployment_phase: str = "PAPER",
    session_id: str = "live-session-1",
    max_position_release_failures: int = 2,
    heartbeat_interval_ms: int = 500,
) -> LiveOrchestratorConfig:
    return LiveOrchestratorConfig(
        orchestrator_config=_orchestrator_config(),
        guard_config=_guard_config(),
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
        execution_boundary=_execution_boundary(deployment_phase=deployment_phase),
    )


def _execution_boundary(*, deployment_phase: str = "PAPER") -> LiveExecutionBoundary:
    adapter = _StubVenueAdapter()
    wallet_provider = None
    if deployment_phase == "LIVE":
        wallet_provider = LiveWalletBalanceProvider(
            adapter,
            tracked_assets=["USDC"],
            initial_balances={"USDC": Decimal("100.000000")},
        )
    return LiveExecutionBoundary(
        venue_adapter=adapter,
        wallet_balance_provider=wallet_provider,
        ofi_exit_router=None,
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


def test_live_orchestrator_config_rejects_invalid_release_failure_threshold() -> None:
    with pytest.raises(ValueError, match="max_position_release_failures"):
        _live_config(max_position_release_failures=0)


def test_live_orchestrator_config_rejects_non_positive_heartbeat_interval() -> None:
    with pytest.raises(ValueError, match="heartbeat_interval_ms"):
        _live_config(heartbeat_interval_ms=0)


def test_build_live_orchestrator_live_requires_wallet_balance_provider() -> None:
    with pytest.raises(ValueError, match="wallet balance provider"):
        build_live_orchestrator(
            config=_live_config(deployment_phase="LIVE"),
            orderbook_tracker=_StubOrderbookTracker(),
            position_manager=_StubPositionManager(),
            execution_boundary=LiveExecutionBoundary(
                venue_adapter=_StubVenueAdapter(),
                wallet_balance_provider=None,
                ofi_exit_router=None,
            ),
        )


def test_build_live_orchestrator_live_requires_venue_adapter() -> None:
    with pytest.raises(ValueError, match="venue adapter"):
        build_live_orchestrator(
            config=_live_config(deployment_phase="LIVE"),
            orderbook_tracker=_StubOrderbookTracker(),
            position_manager=_StubPositionManager(),
            execution_boundary=LiveExecutionBoundary(
                venue_adapter=None,
                wallet_balance_provider=LiveWalletBalanceProvider(
                    _StubVenueAdapter(),
                    tracked_assets=["USDC"],
                    initial_balances={"USDC": Decimal("100.000000")},
                ),
                ofi_exit_router=None,
            ),
        )


def test_build_live_orchestrator_returns_multi_signal_orchestrator() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert isinstance(orchestrator, MultiSignalOrchestrator)


def test_build_live_orchestrator_live_phase_constructs_dispatcher_with_client_order_id_generator() -> None:
    orchestrator = _build_live_factory_orchestrator(deployment_phase="LIVE")

    assert isinstance(orchestrator, MultiSignalOrchestrator)
    assert isinstance(orchestrator.dispatcher._client_order_id_generator, ClientOrderIdGenerator)


def test_live_factory_threads_wallet_provider_into_graph() -> None:
    orchestrator = _build_live_factory_orchestrator(deployment_phase="LIVE")

    assert orchestrator.wallet_balance_provider is not None


def test_two_live_factory_calls_produce_independent_instances_without_shared_state() -> None:
    first = _build_live_factory_orchestrator()
    second = _build_live_factory_orchestrator()

    assert first is not second
    assert first.guard is not second.guard
    assert first.dispatcher is not second.dispatcher


def test_live_factory_shared_guard_identity() -> None:
    orchestrator = _build_live_factory_orchestrator()

    assert orchestrator.dispatcher.guard is orchestrator.guard


def test_orchestrator_health_monitor_construction_passes_with_valid_config() -> None:
    monitor = OrchestratorHealthMonitor(_build_live_factory_orchestrator(), _health_config())

    assert isinstance(monitor, OrchestratorHealthMonitor)


def test_is_safe_to_trade_returns_false_when_health_is_red() -> None:
    base_snapshot = _build_live_factory_orchestrator().orchestrator_snapshot(1000)
    red_snapshot = replace(base_snapshot, health="RED")
    monitor = OrchestratorHealthMonitor(_SnapshotStubOrchestrator(red_snapshot), _health_config())

    report = monitor.check(1000)

    assert report.is_safe_to_trade is False
    assert report.orchestrator_health == "RED"


def test_health_monitor_blocks_reward_entries_on_yellow() -> None:
    base_snapshot = _build_live_factory_orchestrator().orchestrator_snapshot(1000)
    yellow_snapshot = replace(base_snapshot, health="YELLOW")
    monitor = OrchestratorHealthMonitor(_SnapshotStubOrchestrator(yellow_snapshot), _health_config())
    context = PriorityOrderContext(
        market_id="mkt-a",
        side="YES",
        signal_source="REWARD",
        conviction_scalar=Decimal("1"),
        target_price=Decimal("0.48"),
        anchor_volume=Decimal("5"),
        max_capital=Decimal("2.4000"),
        execution_hints=RewardExecutionHints(
            post_only=True,
            time_in_force="GTC",
            liquidity_intent="MAKER_REWARD",
            allow_taker_escalation=False,
            quote_id="reward-1",
            tick_size=Decimal("0.01"),
            cancel_on_stale_ms=15_000,
            replace_only_if_price_moves_ticks=1,
            metadata={},
        ),
        signal_metadata={},
    )

    reason = monitor.dispatch_guard_reason(context, 1000)

    assert reason == "ORCHESTRATOR_HEALTH_YELLOW"


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
        report.halt_reason = "x"  # type: ignore[misc]


def test_health_monitor_blocks_ofi_and_contagion_during_yellow() -> None:
    base_snapshot = _build_live_factory_orchestrator().orchestrator_snapshot(1000)
    yellow_snapshot = replace(base_snapshot, health="YELLOW")
    monitor = OrchestratorHealthMonitor(_SnapshotStubOrchestrator(yellow_snapshot), _health_config())

    ofi_context = PriorityOrderContext(
        market_id="mkt-a",
        side="YES",
        signal_source="OFI",
        target_price=Decimal("0.41"),
        anchor_volume=Decimal("4"),
        max_capital=Decimal("10"),
        conviction_scalar=Decimal("0.8"),
    )
    contagion_context = replace(ofi_context, signal_source="CONTAGION")

    assert monitor.dispatch_guard_reason(ofi_context, 1000) == "ORCHESTRATOR_HEALTH_YELLOW"
    assert monitor.dispatch_guard_reason(contagion_context, 1000) == "ORCHESTRATOR_HEALTH_YELLOW"
