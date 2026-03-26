from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from src.detectors.ctf_peg_config import CtfPegConfig
from src.detectors.si9_cluster_config import Si9ClusterConfig
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.ctf_paper_adapter import CtfPaperAdapter, CtfPaperAdapterConfig
from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.escalation_policy_interface import EscalationPolicyInterface, PaperEscalationPolicy
from src.execution.live_book_interface import PaperBestBidProvider
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.orchestrator_load_shedder import OrchestratorLoadShedder
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.execution.mev_router import MevExecutionRouter, MevMarketSnapshot
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_paper_ledger import OfiPaperLedger
from src.execution.ofi_signal_bridge import OfiSignalBridge, OfiSignalBridgeConfig
from src.execution.position_lifecycle_interface import PaperPositionLifecycle
from src.execution.position_manager_lifecycle import PositionManagerLifecycle
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.si9_paper_adapter import Si9PaperAdapter, Si9PaperAdapterConfig
from src.execution.si9_paper_ledger import Si9PaperLedger
from src.execution.si9_unwind_evaluator import Si9UnwindEvaluator
from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus
from src.execution.unwind_executor_interface import PaperUnwindExecutor, UnwindExecutor
from src.signals.si9_matrix_detector import Si9MatrixDetector

if TYPE_CHECKING:
    from src.data.orderbook import OrderbookTracker
    from src.execution.venue_adapter_interface import VenueAdapter
    from src.trading.position_manager import PositionManager


def _paper_snapshot_provider(market_id: str) -> MevMarketSnapshot:
    _ = market_id
    return MevMarketSnapshot(
        yes_bid=0.45,
        yes_ask=0.55,
        no_bid=0.45,
        no_ask=0.55,
    )


def _default_si9_cluster_config(si9_cluster_configs: list[tuple[str, list[str]]]) -> Si9ClusterConfig:
    minimum_cluster_size = 2
    if si9_cluster_configs:
        minimum_cluster_size = max(2, min(len(market_ids) for _, market_ids in si9_cluster_configs))
    return Si9ClusterConfig(
        target_yield=Decimal("0.02"),
        taker_fee_per_leg=Decimal("0.002"),
        slippage_budget=Decimal("0.001"),
        ghost_town_floor=Decimal("0.85"),
        implausible_edge_ceil=Decimal("0.15"),
        max_ask_age_ms=1000,
        min_cluster_size=minimum_cluster_size,
        tiebreak_policy="lowest_market_id",
    )


def _deployment_phase_dispatch_mode(
    deployment_phase: Literal["PAPER", "DRY_RUN", "LIVE"],
) -> Literal["paper", "dry_run", "live"]:
    return {
        "PAPER": "paper",
        "DRY_RUN": "dry_run",
        "LIVE": "live",
    }[deployment_phase]


def _ranked_unique_market_ids(*market_groups: list[str] | tuple[str, ...]) -> list[str]:
    ranked_market_ids: list[str] = []
    seen: set[str] = set()
    for group in market_groups:
        for market_id in group:
            market_key = str(market_id or "").strip()
            if not market_key or market_key in seen:
                continue
            seen.add(market_key)
            ranked_market_ids.append(market_key)
    return ranked_market_ids


def _live_snapshot_provider(orderbook_tracker: OrderbookTracker):
    def _snapshot_provider(market_id: str) -> MevMarketSnapshot:
        if str(market_id).strip() != str(getattr(orderbook_tracker, "asset_id", "")).strip():
            return _paper_snapshot_provider(market_id)
        try:
            snapshot = orderbook_tracker.snapshot()
            best_bid = float(getattr(snapshot, "best_bid", orderbook_tracker.best_bid))
            best_ask = float(getattr(snapshot, "best_ask", orderbook_tracker.best_ask))
        except Exception:
            return _paper_snapshot_provider(market_id)
        if best_bid <= 0.0 or best_ask <= 0.0:
            return _paper_snapshot_provider(market_id)
        return MevMarketSnapshot(
            yes_bid=best_bid,
            yes_ask=best_ask,
            no_bid=best_bid,
            no_ask=best_ask,
        )

    return _snapshot_provider


def build_paper_orchestrator(
    ctf_config: CtfPegConfig,
    ctf_adapter_config: CtfPaperAdapterConfig,
    si9_cluster_configs: list[tuple[str, list[str]]],
    si9_adapter_config: Si9PaperAdapterConfig,
    ofi_bridge_config: OfiSignalBridgeConfig,
    bus_config: CoordinationBusConfig,
    guard_config: DispatchGuardConfig,
    orchestrator_config: OrchestratorConfig,
    ask_proxy: dict[str, Decimal],
) -> MultiSignalOrchestrator:
    shared_bus = SignalCoordinationBus(bus_config)
    shared_guard = DispatchGuard(guard_config)
    shared_dispatcher = PriorityDispatcher(
        MevExecutionRouter(_paper_snapshot_provider),
        si9_adapter_config.mode,
        guard=shared_guard,
        guard_enabled=False,
    )

    ctf_adapter = CtfPaperAdapter(
        shared_dispatcher,
        shared_guard,
        CtfPaperAdapterConfig(
            max_expected_net_edge=ctf_adapter_config.max_expected_net_edge,
            max_capital_per_signal=ctf_adapter_config.max_capital_per_signal,
            default_anchor_volume=ctf_adapter_config.default_anchor_volume,
            taker_fee_yes=ctf_adapter_config.taker_fee_yes,
            taker_fee_no=ctf_adapter_config.taker_fee_no,
            cancel_on_stale_ms=ctf_adapter_config.cancel_on_stale_ms,
            max_size_per_leg=ctf_adapter_config.max_size_per_leg,
            mode=ctf_adapter_config.mode,
            bus=shared_bus,
        ),
    )
    si9_adapter = Si9PaperAdapter(
        shared_dispatcher,
        shared_guard,
        Si9PaperLedger(),
        Si9PaperAdapterConfig(
            max_expected_net_edge=si9_adapter_config.max_expected_net_edge,
            max_capital_per_cluster=si9_adapter_config.max_capital_per_cluster,
            max_leg_fill_wait_ms=si9_adapter_config.max_leg_fill_wait_ms,
            cancel_on_stale_ms=si9_adapter_config.cancel_on_stale_ms,
            mode=si9_adapter_config.mode,
            unwind_config=si9_adapter_config.unwind_config,
            bus=shared_bus,
        ),
    )
    ofi_bridge = OfiSignalBridge(
        shared_dispatcher,
        shared_guard,
        shared_bus,
        OfiPaperLedger(),
        OfiSignalBridgeConfig(
            max_capital_per_signal=ofi_bridge_config.max_capital_per_signal,
            mode=ofi_bridge_config.mode,
            slot_side_lock=ofi_bridge_config.slot_side_lock,
            source_enabled=ofi_bridge_config.source_enabled,
        ),
    )
    evaluator = Si9UnwindEvaluator(si9_adapter_config.unwind_config)
    si9_cluster_config = _default_si9_cluster_config(si9_cluster_configs)
    si9_detector = Si9MatrixDetector(si9_cluster_config)
    for cluster_id, market_ids in si9_cluster_configs:
        si9_detector.register_cluster(cluster_id, list(market_ids))
    initial_ranked_market_ids = _ranked_unique_market_ids(
        tuple(ask_proxy.keys()),
        tuple(market_id for _, market_ids in si9_cluster_configs for market_id in market_ids),
    )
    load_shedder = OrchestratorLoadShedder(
        max_active_l2_markets=max(1, len(initial_ranked_market_ids)),
        ranked_market_ids=initial_ranked_market_ids,
        deployment_phase="PAPER",
    )

    orchestrator = MultiSignalOrchestrator(
        bus=shared_bus,
        guard=shared_guard,
        dispatcher=shared_dispatcher,
        ctf_adapter=ctf_adapter,
        ofi_bridge=ofi_bridge,
        si9_adapter=si9_adapter,
        best_bid_provider=PaperBestBidProvider(dict(ask_proxy)),
        unwind_executor=PaperUnwindExecutor(si9_adapter_config.unwind_config),
        position_lifecycle=PaperPositionLifecycle(orchestrator_config.max_concurrent_clusters),
        escalation_policy=PaperEscalationPolicy(evaluator, surrender_after_ms=max(si9_adapter_config.unwind_config.max_hold_recovery_ms, 1)),
        config=orchestrator_config,
        load_shedder=load_shedder,
    )
    orchestrator.bind_detector_context(
        ctf_config=ctf_config,
        si9_detector=si9_detector,
        si9_cluster_config=si9_cluster_config,
    )
    return orchestrator


def build_live_orchestrator(
    config: LiveOrchestratorConfig,
    orderbook_tracker: OrderbookTracker,
    position_manager: PositionManager,
    venue_adapter: VenueAdapter,
    unwind_executor: UnwindExecutor,
    escalation_policy: EscalationPolicyInterface,
) -> MultiSignalOrchestrator:
    """
    Constructs a fully wired live-capable orchestrator.
    Uses real OrderbookTracker and PositionManager backends.
    Requires a VenueAdapter implementation - paper or live.
    All shared instances (bus, guard, dispatcher) constructed once internally.
    """
    try:
        from src.execution.venue_adapter_interface import VenueAdapter as _VenueAdapter
    except ImportError as exc:
        raise RuntimeError(
            "VenueAdapter interface not yet available. "
            "Ensure Agent 1 has delivered src/execution/venue_adapter_interface.py"
        ) from exc

    if not isinstance(venue_adapter, _VenueAdapter):
        raise TypeError(f"venue_adapter must implement VenueAdapter, got {type(venue_adapter)}")

    shared_bus = SignalCoordinationBus(config.bus_config)
    shared_guard = DispatchGuard(config.guard_config)
    dispatch_mode = _deployment_phase_dispatch_mode(config.deployment_phase)
    client_order_id_generator = None
    if dispatch_mode == "live":
        client_order_id_generator = ClientOrderIdGenerator("MANUAL", config.session_id)
    shared_dispatcher = PriorityDispatcher(
        MevExecutionRouter(_live_snapshot_provider(orderbook_tracker)),
        dispatch_mode,
        guard=shared_guard,
        guard_enabled=False,
        venue_adapter=venue_adapter if dispatch_mode == "live" else None,
        client_order_id_generator=client_order_id_generator,
    )

    ctf_adapter = CtfPaperAdapter(
        shared_dispatcher,
        shared_guard,
        CtfPaperAdapterConfig(
            max_expected_net_edge=config.ctf_adapter_config.max_expected_net_edge,
            max_capital_per_signal=config.ctf_adapter_config.max_capital_per_signal,
            default_anchor_volume=config.ctf_adapter_config.default_anchor_volume,
            taker_fee_yes=config.ctf_adapter_config.taker_fee_yes,
            taker_fee_no=config.ctf_adapter_config.taker_fee_no,
            cancel_on_stale_ms=config.ctf_adapter_config.cancel_on_stale_ms,
            max_size_per_leg=config.ctf_adapter_config.max_size_per_leg,
            mode=config.ctf_adapter_config.mode,
            bus=shared_bus,
        ),
    )
    si9_adapter = Si9PaperAdapter(
        shared_dispatcher,
        shared_guard,
        Si9PaperLedger(),
        Si9PaperAdapterConfig(
            max_expected_net_edge=config.si9_adapter_config.max_expected_net_edge,
            max_capital_per_cluster=config.si9_adapter_config.max_capital_per_cluster,
            max_leg_fill_wait_ms=config.si9_adapter_config.max_leg_fill_wait_ms,
            cancel_on_stale_ms=config.si9_adapter_config.cancel_on_stale_ms,
            mode=config.si9_adapter_config.mode,
            unwind_config=config.unwind_config,
            bus=shared_bus,
        ),
    )
    ofi_bridge = OfiSignalBridge(
        shared_dispatcher,
        shared_guard,
        shared_bus,
        OfiPaperLedger(),
        OfiSignalBridgeConfig(
            max_capital_per_signal=config.ofi_bridge_config.max_capital_per_signal,
            mode=config.ofi_bridge_config.mode,
            slot_side_lock=config.ofi_bridge_config.slot_side_lock,
            source_enabled=config.ofi_bridge_config.source_enabled,
        ),
    )

    cluster_config_rows = [
        (cluster_id, list(market_ids))
        for cluster_id, market_ids in config.si9_cluster_configs
    ]
    initial_ranked_market_ids = _ranked_unique_market_ids(
        (str(getattr(orderbook_tracker, "asset_id", "")),),
        tuple(market_id for _, market_ids in config.si9_cluster_configs for market_id in market_ids),
    )
    load_shedder = OrchestratorLoadShedder(
        max_active_l2_markets=25,
        ranked_market_ids=initial_ranked_market_ids,
        position_manager=position_manager,
        deployment_phase=config.deployment_phase,
    )
    si9_cluster_config = _default_si9_cluster_config(cluster_config_rows)
    si9_detector = Si9MatrixDetector(si9_cluster_config)
    for cluster_id, market_ids in config.si9_cluster_configs:
        si9_detector.register_cluster(cluster_id, list(market_ids))

    orchestrator = MultiSignalOrchestrator(
        bus=shared_bus,
        guard=shared_guard,
        dispatcher=shared_dispatcher,
        ctf_adapter=ctf_adapter,
        ofi_bridge=ofi_bridge,
        si9_adapter=si9_adapter,
        best_bid_provider=OrderbookBestBidProvider(orderbook_tracker),
        unwind_executor=unwind_executor,
        position_lifecycle=PositionManagerLifecycle(position_manager),
        escalation_policy=escalation_policy,
        config=config.orchestrator_config,
        load_shedder=load_shedder,
    )
    orchestrator.bind_detector_context(
        ctf_config=config.ctf_peg_config,
        si9_detector=si9_detector,
        si9_cluster_config=si9_cluster_config,
    )
    return orchestrator