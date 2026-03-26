"""Thin event-to-router dispatcher for isolated MEV execution playbooks."""

from __future__ import annotations

from src.events.mev_events import (
    DisputeArbitrageSignal,
    MMPredationSignal,
    ShadowSweepSignal,
)
from src.execution.mev_router import MevExecutionBatch, MevExecutionRouter


class MevDispatcher:
    """Central event router for isolated MEV execution research.

    The dispatcher stays intentionally small. It accepts strict signal
    dataclasses and forwards them to MevExecutionRouter.
    """

    def __init__(self, router: MevExecutionRouter) -> None:
        self._router = router

    def on_mempool_whale_detected(self, event: ShadowSweepSignal) -> MevExecutionBatch:
        return self._router.execute_shadow_sweep(
            market_id=event.target_market_id,
            direction=event.direction,
            max_capital=event.max_capital,
            premium_pct=event.premium_pct,
        )

    def on_mm_vulnerability_detected(self, event: MMPredationSignal) -> MevExecutionBatch:
        return self._router.execute_mm_trap(
            target_market_id=event.target_market_id,
            correlated_market_id=event.correlated_market_id,
            v_attack=event.v_attack,
            trap_direction=event.trap_direction,
        )

    def on_uma_dispute_panic(self, event: DisputeArbitrageSignal) -> MevExecutionBatch:
        return self._router.execute_d3_panic_absorption(
            market_id=event.market_id,
            panic_direction=event.panic_direction,
            limit_price=event.limit_price,
            max_capital=event.max_capital,
        )
