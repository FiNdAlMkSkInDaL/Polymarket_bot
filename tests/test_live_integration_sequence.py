from __future__ import annotations

import builtins
from decimal import Decimal
from types import SimpleNamespace

from src.detectors.ctf_peg_config import CtfPegConfig
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.escalation_policy_interface import EscalationPolicyInterface
from src.execution.live_execution_boundary import LiveExecutionBoundary
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.orchestrator_factory import build_live_orchestrator
from src.execution.orchestrator_health_monitor import HealthMonitorConfig, OrchestratorHealthMonitor
from src.execution.ofi_signal_bridge import OfiEntrySignal, OfiSignalBridgeConfig
from src.execution.si9_paper_adapter import Si9PaperAdapterConfig
from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindManifest
from src.execution.signal_coordination_bus import CoordinationBusConfig
from src.execution.unwind_executor_interface import PaperUnwindExecutor
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


class _TrackedOrderbook:
    def __init__(
        self,
        *,
        asset_id: str = "mkt-a",
        best_bid: float = 0.41,
        best_ask: float = 0.42,
        timestamp_ms: int = 1_700_000_000_000,
    ) -> None:
        self.asset_id = asset_id
        self._best_bid = float(best_bid)
        self._best_ask = float(best_ask)
        self._timestamp_ms = int(timestamp_ms)
        self.best_bid_reads = 0
        self.snapshot_calls = 0
        self._bids = [self._best_bid]
        self._asks = [self._best_ask]

    @property
    def best_bid(self) -> float:
        self.best_bid_reads += 1
        return self._best_bid

    @property
    def best_ask(self) -> float:
        return self._best_ask

    def snapshot(self):
        self.snapshot_calls += 1
        return SimpleNamespace(
            best_bid=self._best_bid,
            best_ask=self._best_ask,
            timestamp=float(self._timestamp_ms),
        )

    def apply_bbo(self, *, best_bid: float, best_ask: float, timestamp_ms: int) -> None:
        self._best_bid = float(best_bid)
        self._best_ask = float(best_ask)
        self._timestamp_ms = int(timestamp_ms)
        self._bids[0] = self._best_bid
        self._asks[0] = self._best_ask


class _StubPositionManager:
    def __init__(self, *, max_open: int = 4, should_raise_cleanup: bool = False) -> None:
        self.max_open = max_open
        self.should_raise_cleanup = should_raise_cleanup
        self.cleanup_calls = 0
        self._open_positions: list[object] = []

    def get_open_positions(self) -> list[object]:
        return list(self._open_positions)

    def cleanup_closed(self) -> list[object]:
        self.cleanup_calls += 1
        if self.should_raise_cleanup:
            raise RuntimeError("cleanup failed")
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
            venue_timestamp_ms=1_000,
            latency_ms=1,
        )

    def cancel_order(self, client_order_id: str, market_id: str) -> VenueCancelResponse:
        _ = market_id
        return VenueCancelResponse(
            client_order_id=client_order_id,
            cancelled=True,
            rejection_reason=None,
            venue_timestamp_ms=1_001,
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
    def should_escalate(self, manifest: Si9UnwindManifest, current_timestamp_ms: int) -> bool:
        _ = (manifest, current_timestamp_ms)
        return False

    def should_surrender(self, manifest: Si9UnwindManifest, current_timestamp_ms: int) -> bool:
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
        unwind_config=_unwind_config(),
        bus=None,
    )


def _ofi_bridge_config() -> OfiSignalBridgeConfig:
    return OfiSignalBridgeConfig(
        max_capital_per_signal=Decimal("15.000000"),
        mode="paper",
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
            bus_config=_bus_config(),
            guard_config=_guard_config(),
            ctf_adapter_config=_ctf_adapter_config(),
            si9_adapter_config=_si9_adapter_config(),
            ofi_bridge_config=_ofi_bridge_config(),
            ctf_peg_config=_ctf_config(),
            si9_cluster_configs=(("cluster-1", ("mkt-a", "mkt-b", "mkt-c")),),
            unwind_config=_unwind_config(),
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
            unwind_executor=PaperUnwindExecutor(_unwind_config()),
            escalation_policy=_StubEscalationPolicy(),
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

    assert orchestrator.ctf_adapter._bus is orchestrator.si9_adapter._bus
    assert orchestrator.ctf_adapter._bus is orchestrator.ofi_bridge.bus
    assert orchestrator.ctf_adapter._bus is orchestrator.bus
    assert orchestrator.ctf_adapter._guard is orchestrator.si9_adapter._guard
    assert orchestrator.ctf_adapter._guard is orchestrator.ofi_bridge.guard
    assert orchestrator.ctf_adapter._guard is orchestrator.guard
    assert orchestrator.ctf_adapter._dispatcher is orchestrator.si9_adapter._dispatcher
    assert orchestrator.ctf_adapter._dispatcher is orchestrator.ofi_bridge.dispatcher
    assert orchestrator.ctf_adapter._dispatcher is orchestrator.dispatcher


def test_integration_simulator_bbo_ingress_updates_provider_and_routes_ofi_only_after_health_check() -> None:
    bot = MockBot()
    bot.startup(session_id="session-blueprint-3")

    assert bot._live_orchestrator is not None
    assert bot._orchestrator_health_monitor is not None
    tracker = bot._book_trackers["mkt-a"]
    provider = bot._live_orchestrator.best_bid_provider
    sequence: list[str] = []

    original_is_safe_to_trade = bot._orchestrator_health_monitor.is_safe_to_trade

    def _recording_is_safe_to_trade(current_timestamp_ms: int) -> bool:
        sequence.append(f"health:{current_timestamp_ms}")
        return original_is_safe_to_trade(current_timestamp_ms)

    bot._orchestrator_health_monitor.is_safe_to_trade = _recording_is_safe_to_trade  # type: ignore[method-assign]

    original_bridge_on_signal = bot._live_orchestrator.ofi_bridge.on_signal

    def _recording_bridge_on_signal(signal: OfiEntrySignal, max_capital: Decimal, timestamp_ms: int):
        sequence.append(f"bridge:{timestamp_ms}")
        return original_bridge_on_signal(signal, max_capital, timestamp_ms)

    bot._live_orchestrator.ofi_bridge.on_signal = _recording_bridge_on_signal  # type: ignore[method-assign]

    bot.on_bbo_update(
        asset_id="mkt-a",
        best_bid=Decimal("0.63"),
        best_ask=Decimal("0.64"),
        timestamp_ms=1_700_000_000_100,
    )

    original_list = builtins.list

    def _forbidden_list(*args, **kwargs):
        raise AssertionError("list allocation is not allowed in OrderbookBestBidProvider O(1) read path")

    try:
        builtins.list = _forbidden_list
        best_bid = provider.get_best_bid("mkt-a")
        best_bid_timestamp_ms = provider.get_best_bid_timestamp_ms("mkt-a")
    finally:
        builtins.list = original_list

    event = bot.route_ofi_signal(_ofi_signal(1_700_000_000_100), max_capital=Decimal("12"), timestamp_ms=1_700_000_000_100)

    assert best_bid == Decimal("0.63")
    assert best_bid_timestamp_ms == 1_700_000_000_100
    assert tracker.best_bid_reads == 1
    assert tracker.snapshot_calls == 1
    assert sequence == ["health:1700000000100", "bridge:1700000000100"]
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