"""Paper-only event bus wiring isolated MEV detectors into the dispatcher."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config import settings
from src.events.mev_events import DisputeArbitrageSignal, ShadowSweepSignal
from src.execution.mev_dispatcher import MevDispatcher

if TYPE_CHECKING:
    from src.signals.dispute_arbitrage_detector import DisputeArbitrageDetector
    from src.signals.shadow_sweep_detector import ShadowSweepDetector


class MevPaperAdapter:
    """Wire isolated MEV detectors into the strict dispatcher in paper mode.

    The adapter is intentionally small. It validates the MEV configuration,
    registers detector callbacks, and immediately forwards strict events into
    the corresponding dispatcher handlers.
    """

    def __init__(self, dispatcher: MevDispatcher) -> None:
        self._validate_mev_config()
        self._dispatcher = dispatcher
        self._shadow_detector: ShadowSweepDetector | None = None
        self._d3_detector: DisputeArbitrageDetector | None = None
        self.shadow_dispatch_results: list[object] = []
        self.d3_dispatch_results: list[object] = []

    def register_detectors(
        self,
        shadow_detector: ShadowSweepDetector,
        d3_detector: DisputeArbitrageDetector,
    ) -> None:
        self._shadow_detector = shadow_detector
        self._d3_detector = d3_detector
        shadow_detector.register_callback(self._on_shadow_signal)
        d3_detector.register_callback(self._on_d3_signal)

    def _on_shadow_signal(self, signal: ShadowSweepSignal) -> object:
        result = self._dispatcher.on_mempool_whale_detected(signal)
        self.shadow_dispatch_results.append(result)
        return result

    def _on_d3_signal(self, signal: DisputeArbitrageSignal) -> object:
        result = self._dispatcher.on_uma_dispute_panic(signal)
        self.d3_dispatch_results.append(result)
        return result

    @staticmethod
    def _validate_mev_config() -> None:
        premium = float(settings.mev_shadow_sweep_premium_pct)
        if not 0.0 <= premium < 1.0:
            raise ValueError(
                f"MEV_SHADOW_SWEEP_PREMIUM_PCT must be >= 0 and < 1; got {premium!r}"
            )

        panic_threshold = float(settings.mev_d3_panic_threshold)
        if not 0.0 < panic_threshold < 1.0:
            raise ValueError(
                f"MEV_D3_PANIC_THRESHOLD must be > 0 and < 1; got {panic_threshold!r}"
            )
