from __future__ import annotations

from dataclasses import dataclass

from src.risk.entropy_circuit_breaker import EntropyCircuitBreaker
from src.signals.mm_predation_detector import (
    CorrelatedOrderBookState,
    MMPredationDetector,
    MMPredationSignal,
)
from src.signals.toxic_wake_detector import ToxicWakeDetector


@dataclass(frozen=True, slots=True)
class AdvancedTradeTickResult:
    market_id: str
    wake_intensity: float
    wake_toxic: bool
    mm_predation_signal: MMPredationSignal | None = None


@dataclass(frozen=True, slots=True)
class AdvancedMarketState:
    market_id: str
    is_toxic_wake: bool
    is_safe_entropy: bool
    active_mm_traps: list[MMPredationSignal]


class AdvancedSignalEngine:
    """Unified facade over advanced defensive and offensive signal detectors."""

    __slots__ = (
        "mm_predation_detector",
        "entropy_circuit_breaker",
        "toxic_wake_detector",
        "_correlated_market_by_target",
        "_active_mm_traps_by_market",
        "_last_make_safety_by_market",
    )

    def __init__(
        self,
        *,
        mm_predation_detector: MMPredationDetector | None = None,
        entropy_circuit_breaker: EntropyCircuitBreaker | None = None,
        toxic_wake_detector: ToxicWakeDetector | None = None,
    ) -> None:
        self.mm_predation_detector = mm_predation_detector or MMPredationDetector()
        self.entropy_circuit_breaker = entropy_circuit_breaker or EntropyCircuitBreaker()
        self.toxic_wake_detector = toxic_wake_detector or ToxicWakeDetector()
        self._correlated_market_by_target: dict[str, str] = {}
        self._active_mm_traps_by_market: dict[str, dict[str, MMPredationSignal]] = {}
        self._last_make_safety_by_market: dict[str, bool] = {}

    @property
    def last_make_safety_by_market(self) -> dict[str, bool]:
        return self._last_make_safety_by_market

    def register_correlated_market(self, target_market_id: str, correlated_market_id: str) -> None:
        target_key = str(target_market_id).strip()
        correlated_key = str(correlated_market_id).strip()
        self._correlated_market_by_target[target_key] = correlated_key
        self.mm_predation_detector.register_correlation(target_key, correlated_key)

    def set_correlated_order_book(
        self,
        market_id: str,
        order_book: CorrelatedOrderBookState,
    ) -> None:
        self.mm_predation_detector.set_order_book(market_id, order_book)

    def on_trade_tick(
        self,
        market_id: str,
        maker_address: str,
        price_delta: float,
        trade_volume: float,
        timestamp: float,
    ) -> AdvancedTradeTickResult:
        market_key = str(market_id).strip()
        maker_key = str(maker_address or "").strip()

        wake_intensity = self.toxic_wake_detector.process_trade_tick(
            market_key,
            timestamp,
            trade_volume,
        )
        wake_toxic = self.toxic_wake_detector.is_wake_toxic(market_key, timestamp)

        mm_signal = None
        if maker_key:
            correlated_market_id = self._correlated_market_by_target.get(market_key)
            if correlated_market_id is None:
                self.mm_predation_detector.tracker.process_fill_event(
                    maker_key,
                    price_delta,
                    trade_volume,
                )
                self._remove_active_trap(market_key, maker_key)
            else:
                mm_signal = self.mm_predation_detector.evaluate_market_tick(
                    target_market_id=market_key,
                    maker_address=maker_key,
                    price_delta=price_delta,
                    taker_volume=trade_volume,
                    correlated_market_id=correlated_market_id,
                )
                if mm_signal is None:
                    self._remove_active_trap(market_key, maker_key)
                else:
                    self._active_mm_traps_by_market.setdefault(market_key, {})[maker_key] = mm_signal

        return AdvancedTradeTickResult(
            market_id=market_key,
            wake_intensity=wake_intensity,
            wake_toxic=wake_toxic,
            mm_predation_signal=mm_signal,
        )

    def on_orderbook_update(
        self,
        market_id: str,
        top_bid_vol: float,
        top_ask_vol: float,
    ) -> bool:
        market_key = str(market_id).strip()
        is_safe = self.entropy_circuit_breaker.is_safe_to_make(
            market_key,
            top_bid_vol,
            top_ask_vol,
        )
        self._last_make_safety_by_market[market_key] = is_safe
        return is_safe

    def get_market_state(self, market_id: str, current_timestamp: float) -> AdvancedMarketState:
        market_key = str(market_id).strip()
        is_toxic_wake = self.toxic_wake_detector.is_wake_toxic(market_key, current_timestamp)
        is_safe_entropy = self._last_make_safety_by_market.get(market_key, True)
        active_mm_traps = list(self._active_mm_traps_by_market.get(market_key, {}).values())
        return AdvancedMarketState(
            market_id=market_key,
            is_toxic_wake=is_toxic_wake,
            is_safe_entropy=is_safe_entropy,
            active_mm_traps=active_mm_traps,
        )

    def _remove_active_trap(self, market_id: str, maker_address: str) -> None:
        traps_by_maker = self._active_mm_traps_by_market.get(market_id)
        if traps_by_maker is None:
            return
        traps_by_maker.pop(maker_address, None)
        if not traps_by_maker:
            self._active_mm_traps_by_market.pop(market_id, None)
