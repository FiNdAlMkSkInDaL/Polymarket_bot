from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.live_execution_boundary import LiveExecutionBoundary
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_signal_bridge import OfiEntrySignal
from src.execution.orchestrator_factory import build_live_orchestrator
from src.execution.orchestrator_health_monitor import HealthMonitorConfig, OrchestratorHealthMonitor
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


class _TrackedOrderbook:
    def __init__(self) -> None:
        self.asset_id = "mkt-a"
        self.best_bid = 0.47
        self.best_ask = 0.48
        self._timestamp = 1_700_000_000_000.0

    def apply_bbo(self, *, best_bid: float, best_ask: float, timestamp_ms: int) -> None:
        self.best_bid = best_bid
        self.best_ask = best_ask
        self._timestamp = float(timestamp_ms)

    def snapshot(self):
        return SimpleNamespace(best_bid=self.best_bid, best_ask=self.best_ask, timestamp=self._timestamp)


class _StubPositionManager:
    def __init__(self, *, should_raise_cleanup: bool = False) -> None:
        self.max_open = 4
        self._should_raise_cleanup = should_raise_cleanup

    def get_open_positions(self) -> list[object]:
        return []

    def cleanup_closed(self) -> list[object]:
        if self._should_raise_cleanup:
            raise RuntimeError("cleanup failed")
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


def _health_config(*, max_release_failures_before_halt: int = 2) -> HealthMonitorConfig:
    return HealthMonitorConfig(
        max_release_failures_before_halt=max_release_failures_before_halt,
        stale_snapshot_threshold_ms=500,
        min_heartbeat_interval_ms=100,
    )


def _ofi_signal(timestamp_ms: int) -> OfiEntrySignal:
    return OfiEntrySignal(
        market_id="mkt-a",
        side="YES",
        target_price=Decimal("0.41"),
        anchor_volume=Decimal("4"),
        conviction_scalar=Decimal("0.8"),
        signal_timestamp_ms=timestamp_ms,
        tvi_kappa=Decimal("0.2"),
        ofi_window_ms=250,
    )


class MockBot:
    def __init__(self, *, position_manager: _StubPositionManager | None = None) -> None:
        self._book_trackers = {"mkt-a": _TrackedOrderbook()}
        self.positions = _StubPositionManager() if position_manager is None else position_manager
        self._live_orchestrator: MultiSignalOrchestrator | None = None
        self._orchestrator_health_monitor: OrchestratorHealthMonitor | None = None
        self._startup_config: LiveOrchestratorConfig | None = None

    async def _on_clob_fill(self, order) -> None:
        _ = order

    def startup(self, *, session_id: str, timestamp_ms: int = 1_700_000_000_000) -> None:
        _ = timestamp_ms
        self._startup_config = LiveOrchestratorConfig(
            orchestrator_config=_orchestrator_config(),
            guard_config=_guard_config(),
            deployment_phase="PAPER",
            session_id=session_id,
            max_position_release_failures=2,
            heartbeat_interval_ms=100,
        )
        self._live_orchestrator = build_live_orchestrator(
            config=self._startup_config,
            orderbook_tracker=self._book_trackers["mkt-a"],
            position_manager=self.positions,
            execution_boundary=LiveExecutionBoundary(
                venue_adapter=_StubVenueAdapter(),
                wallet_balance_provider=None,
                ofi_exit_router=None,
            ),
        )
        self._orchestrator_health_monitor = OrchestratorHealthMonitor(
            self._live_orchestrator,
            _health_config(max_release_failures_before_halt=self._startup_config.max_position_release_failures),
        )

    def on_bbo_update(self, *, asset_id: str, best_bid: Decimal, best_ask: Decimal, timestamp_ms: int) -> None:
        tracker = self._book_trackers[asset_id]
        tracker.apply_bbo(best_bid=float(best_bid), best_ask=float(best_ask), timestamp_ms=timestamp_ms)

    def route_ofi_signal(self, signal: OfiEntrySignal, *, max_capital: Decimal, timestamp_ms: int):
        assert self._live_orchestrator is not None
        assert self._orchestrator_health_monitor is not None
        if not self._orchestrator_health_monitor.is_safe_to_trade(timestamp_ms):
            return None
        return self._live_orchestrator.on_ofi_signal(signal, max_capital, timestamp_ms)

    def cleanup_once(self, *, timestamp_ms: int) -> list[object]:
        _ = timestamp_ms
        assert self._orchestrator_health_monitor is not None
        try:
            removed_positions = self.positions.cleanup_closed()
        except Exception:
            self._orchestrator_health_monitor.record_position_release_failure()
            return []
        self._orchestrator_health_monitor.reset_release_failure_count()
        return removed_positions


def test_integration_simulator_startup_sequence_constructs_live_orchestrator_and_health_monitor() -> None:
    bot = MockBot()

    bot.startup(session_id="session-blueprint-1")

    assert bot._startup_config is not None
    assert bot._startup_config.session_id == "session-blueprint-1"
    assert isinstance(bot._live_orchestrator, MultiSignalOrchestrator)
    assert isinstance(bot._orchestrator_health_monitor, OrchestratorHealthMonitor)


def test_integration_simulator_shared_identity_matches_factory_contract() -> None:
    bot = MockBot()
    bot.startup(session_id="session-blueprint-2")

    assert bot._live_orchestrator is not None
    orchestrator = bot._live_orchestrator

    assert orchestrator.dispatcher.guard is orchestrator.guard
    assert orchestrator.load_shedder is not None
    assert orchestrator.wallet_balance_provider is None


def test_integration_simulator_routes_ofi_only_after_health_check() -> None:
    bot = MockBot()
    bot.startup(session_id="session-blueprint-3")

    assert bot._live_orchestrator is not None
    assert bot._orchestrator_health_monitor is not None
    tracker = bot._book_trackers["mkt-a"]
    sequence: list[str] = []

    original_is_safe_to_trade = bot._orchestrator_health_monitor.is_safe_to_trade

    def _recording_is_safe_to_trade(current_timestamp_ms: int) -> bool:
        sequence.append(f"health:{current_timestamp_ms}")
        return original_is_safe_to_trade(current_timestamp_ms)

    bot._orchestrator_health_monitor.is_safe_to_trade = _recording_is_safe_to_trade  # type: ignore[method-assign]

    original_on_ofi_signal = bot._live_orchestrator.on_ofi_signal

    def _recording_on_ofi_signal(signal: OfiEntrySignal, max_capital: Decimal, timestamp_ms: int):
        sequence.append(f"orchestrator:{timestamp_ms}")
        return original_on_ofi_signal(signal, max_capital, timestamp_ms)

    bot._live_orchestrator.on_ofi_signal = _recording_on_ofi_signal  # type: ignore[method-assign]

    bot.on_bbo_update(
        asset_id="mkt-a",
        best_bid=Decimal("0.63"),
        best_ask=Decimal("0.64"),
        timestamp_ms=1_700_000_000_100,
    )

    event = bot.route_ofi_signal(_ofi_signal(1_700_000_000_100), max_capital=Decimal("12"), timestamp_ms=1_700_000_000_100)

    assert tracker.best_bid == 0.63
    assert tracker.best_ask == 0.64
    assert sequence == ["health:1700000000100", "orchestrator:1700000000100"]
    assert event is not None
    assert event.event_type == "OFI_DISPATCHED"


def test_integration_simulator_cleanup_failure_increments_monitor_and_halts_trading_at_threshold() -> None:
    bot = MockBot(position_manager=_StubPositionManager(should_raise_cleanup=True))
    bot.startup(session_id="session-blueprint-4")

    assert bot._orchestrator_health_monitor is not None

    assert bot._orchestrator_health_monitor.is_safe_to_trade(1_700_000_000_000) is True

    bot.cleanup_once(timestamp_ms=1_700_000_000_010)
    report_after_first_failure = bot._orchestrator_health_monitor.check(1_700_000_000_020)
    bot.cleanup_once(timestamp_ms=1_700_000_000_030)
    report_after_second_failure = bot._orchestrator_health_monitor.check(1_700_000_000_040)

    assert report_after_first_failure.consecutive_release_failures == 1
    assert report_after_first_failure.is_safe_to_trade is True
    assert report_after_second_failure.consecutive_release_failures == 2
    assert report_after_second_failure.is_safe_to_trade is False
    assert report_after_second_failure.halt_reason == "RELEASE_FAILURE_THRESHOLD_REACHED"
