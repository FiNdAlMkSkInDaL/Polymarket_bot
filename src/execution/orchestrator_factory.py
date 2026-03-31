from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.dispatch_guard import DispatchGuard
from src.execution.live_execution_boundary import LiveExecutionBoundary
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.mev_router import MevExecutionRouter, MevMarketSnapshot
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.orchestrator_load_shedder import OrchestratorLoadShedder
from src.execution.priority_dispatcher import PriorityDispatcher

if TYPE_CHECKING:
    from src.data.orderbook import OrderbookTracker
    from src.execution.dispatch_guard_config import DispatchGuardConfig
    from src.trading.position_manager import PositionManager


def _paper_snapshot_provider(market_id: str) -> MevMarketSnapshot:
    _ = market_id
    return MevMarketSnapshot(
        yes_bid=0.45,
        yes_ask=0.55,
        no_bid=0.45,
        no_ask=0.55,
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


def _live_snapshot_provider(orderbook_tracker: OrderbookTracker | Mapping[str, OrderbookTracker | tuple[OrderbookTracker, OrderbookTracker]]):
    def _resolve_trackers(market_id: str) -> tuple[OrderbookTracker | None, OrderbookTracker | None]:
        market_key = str(market_id).strip()
        if isinstance(orderbook_tracker, Mapping):
            tracker_value = orderbook_tracker.get(market_key)
            if isinstance(tracker_value, tuple):
                return tracker_value[0], tracker_value[1]
            return tracker_value, tracker_value
        tracker_key = str(getattr(orderbook_tracker, "asset_id", "")).strip()
        if market_key != tracker_key:
            return None, None
        return orderbook_tracker, orderbook_tracker

    def _snapshot_provider(market_id: str) -> MevMarketSnapshot:
        yes_tracker, no_tracker = _resolve_trackers(market_id)
        if yes_tracker is None or no_tracker is None:
            return _paper_snapshot_provider(market_id)
        try:
            yes_snapshot = yes_tracker.snapshot()
            no_snapshot = no_tracker.snapshot()
            yes_bid = float(getattr(yes_snapshot, "best_bid", yes_tracker.best_bid))
            yes_ask = float(getattr(yes_snapshot, "best_ask", yes_tracker.best_ask))
            no_bid = float(getattr(no_snapshot, "best_bid", no_tracker.best_bid))
            no_ask = float(getattr(no_snapshot, "best_ask", no_tracker.best_ask))
        except Exception:
            return _paper_snapshot_provider(market_id)
        if yes_bid <= 0.0 or yes_ask <= 0.0 or no_bid <= 0.0 or no_ask <= 0.0:
            return _paper_snapshot_provider(market_id)
        return MevMarketSnapshot(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
        )

    return _snapshot_provider


def build_paper_orchestrator(
    *,
    guard_config: DispatchGuardConfig,
    orchestrator_config: OrchestratorConfig,
    ask_proxy: dict[str, Decimal],
) -> MultiSignalOrchestrator:
    shared_guard = DispatchGuard(guard_config)
    shared_dispatcher = PriorityDispatcher(
        MevExecutionRouter(_paper_snapshot_provider),
        "paper",
        guard=shared_guard,
        guard_enabled=False,
    )
    initial_ranked_market_ids = _ranked_unique_market_ids(tuple(ask_proxy.keys()))
    load_shedder = OrchestratorLoadShedder(
        max_active_l2_markets=max(1, len(initial_ranked_market_ids) or 1),
        ranked_market_ids=initial_ranked_market_ids,
        deployment_phase="PAPER",
    )
    return MultiSignalOrchestrator(
        guard=shared_guard,
        dispatcher=shared_dispatcher,
        config=orchestrator_config,
        load_shedder=load_shedder,
    )


def build_live_orchestrator(
    config: LiveOrchestratorConfig,
    orderbook_tracker: OrderbookTracker | Mapping[str, OrderbookTracker | tuple[OrderbookTracker, OrderbookTracker]],
    position_manager: PositionManager,
    execution_boundary: LiveExecutionBoundary,
    ofi_exit_trackers: Mapping[str, OrderbookTracker] | None = None,
) -> MultiSignalOrchestrator:
    """
    Construct the lean live orchestrator runtime around a shared dispatcher.
    """
    shared_guard = DispatchGuard(config.guard_config)
    dispatch_mode = _deployment_phase_dispatch_mode(config.deployment_phase)
    venue_adapter = None if execution_boundary is None else execution_boundary.venue_adapter
    wallet_balance_provider = None if execution_boundary is None else execution_boundary.wallet_balance_provider
    client_order_id_generator = None
    if dispatch_mode == "live":
        if venue_adapter is None or wallet_balance_provider is None:
            raise ValueError("live orchestrator requires a venue adapter and wallet balance provider")
        client_order_id_generator = ClientOrderIdGenerator("MANUAL", config.session_id)

    shared_dispatcher = PriorityDispatcher(
        MevExecutionRouter(_live_snapshot_provider(orderbook_tracker)),
        dispatch_mode,
        guard=shared_guard,
        guard_enabled=False,
        venue_adapter=venue_adapter if dispatch_mode == "live" else None,
        client_order_id_generator=client_order_id_generator,
        wallet_balance_provider=wallet_balance_provider if dispatch_mode == "live" else None,
    )
    initial_ranked_market_ids = _ranked_unique_market_ids(
        tuple(orderbook_tracker.keys()) if isinstance(orderbook_tracker, Mapping) else (str(getattr(orderbook_tracker, "asset_id", "")),),
    )
    load_shedder = OrchestratorLoadShedder(
        max_active_l2_markets=25,
        ranked_market_ids=initial_ranked_market_ids,
        position_manager=position_manager,
        deployment_phase=config.deployment_phase,
    )

    return MultiSignalOrchestrator(
        guard=shared_guard,
        dispatcher=shared_dispatcher,
        config=config.orchestrator_config,
        load_shedder=load_shedder,
        wallet_balance_provider=wallet_balance_provider,
        ofi_exit_router=None if execution_boundary is None else execution_boundary.ofi_exit_router,
        ofi_exit_trackers=ofi_exit_trackers,
        active_position_count_provider=lambda: len(position_manager.get_open_positions()),
    )