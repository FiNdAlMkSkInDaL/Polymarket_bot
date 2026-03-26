from __future__ import annotations

from decimal import Decimal

from src.detectors.ctf_peg_config import CtfPegConfig
from src.detectors.si9_cluster_config import Si9ClusterConfig
from src.execution.ctf_paper_adapter import CtfPaperAdapter, CtfPaperAdapterConfig
from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.escalation_policy_interface import PaperEscalationPolicy
from src.execution.live_book_interface import PaperBestBidProvider
from src.execution.mev_router import MevExecutionRouter, MevMarketSnapshot
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_paper_ledger import OfiPaperLedger
from src.execution.ofi_signal_bridge import OfiSignalBridge, OfiSignalBridgeConfig
from src.execution.position_lifecycle_interface import PaperPositionLifecycle
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.si9_paper_adapter import Si9PaperAdapter, Si9PaperAdapterConfig
from src.execution.si9_paper_ledger import Si9PaperLedger
from src.execution.si9_unwind_evaluator import Si9UnwindEvaluator
from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus
from src.execution.unwind_executor_interface import PaperUnwindExecutor
from src.signals.si9_matrix_detector import Si9MatrixDetector


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
    )
    orchestrator.bind_detector_context(
        ctf_config=ctf_config,
        si9_detector=si9_detector,
        si9_cluster_config=si9_cluster_config,
    )
    return orchestrator