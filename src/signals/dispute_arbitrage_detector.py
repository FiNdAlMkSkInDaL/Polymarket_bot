"""Strict D3 dispute-arbitrage detector wrapping the UMA dispute tracker."""

from __future__ import annotations

from typing import Callable

from src.core.config import settings
from src.events.mev_events import DisputeArbitrageSignal
from src.signals.uma_dispute_tracker import UmaDisputeTracker


class DisputeArbitrageDetector:
    """Translate UMA dispute tracker signals into strict MEV event contracts.

    The wrapped tracker remains the source of truth for market-state transitions
    and EWMA-based panic gating. This detector adds a configurable minimum panic
    discount requirement before emitting Agent 1's strict event dataclass.
    """

    def __init__(
        self,
        tracker: UmaDisputeTracker,
        *,
        max_capital: float | None = None,
        min_panic_discount: float | None = None,
        panic_direction: str = "YES",
    ) -> None:
        self._tracker = tracker
        self._max_capital = float(
            settings.strategy.max_trade_size_usd if max_capital is None else max_capital
        )
        self._min_panic_discount = float(
            settings.mev_d3_panic_threshold if min_panic_discount is None else min_panic_discount
        )
        self._panic_direction = str(panic_direction or "YES").strip().upper()
        self._callbacks: list[Callable[[DisputeArbitrageSignal], object]] = []

    @property
    def tracker(self) -> UmaDisputeTracker:
        return self._tracker

    @property
    def min_panic_price(self) -> float:
        return max(0.0, min(1.0, 1.0 - self._min_panic_discount))

    def register_callback(self, callback: Callable[[DisputeArbitrageSignal], object]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    async def poll_once(self, current_prices: dict[str, float]) -> list[DisputeArbitrageSignal]:
        tracker_signals = await self._tracker.poll_once(current_prices)
        events = self.translate_signals(tracker_signals)
        for event in events:
            for callback in self._callbacks:
                callback(event)
        return events

    def translate_signals(self, tracker_signals: list[object]) -> list[DisputeArbitrageSignal]:
        events: list[DisputeArbitrageSignal] = []
        configured_limit = self.min_panic_price

        for tracker_signal in tracker_signals:
            market_id = str(getattr(tracker_signal, "market_id", "") or "").strip()
            current_price = float(getattr(tracker_signal, "current_price", 0.0) or 0.0)

            if not market_id:
                continue
            if current_price <= 0.0:
                continue
            if current_price > configured_limit:
                continue

            events.append(
                DisputeArbitrageSignal(
                    market_id=market_id,
                    panic_direction=self._panic_direction,
                    limit_price=current_price,
                    max_capital=self._max_capital,
                )
            )

        return events
