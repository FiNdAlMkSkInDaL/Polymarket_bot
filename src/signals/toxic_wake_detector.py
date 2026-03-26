from __future__ import annotations

from src.signals.advanced_math import HawkesToxicWakeState


class ToxicWakeDetector:
    """Regime-classification wrapper over per-market Hawkes toxic wake state."""

    __slots__ = (
        "decay_rate",
        "impulse_scale",
        "contagion_intensity_threshold",
        "_wake_states",
    )

    def __init__(
        self,
        *,
        decay_rate: float = 1.0,
        impulse_scale: float = 1.0,
        contagion_intensity_threshold: float = 100.0,
    ) -> None:
        self.decay_rate = float(decay_rate)
        self.impulse_scale = float(impulse_scale)
        self.contagion_intensity_threshold = float(contagion_intensity_threshold)
        self._wake_states: dict[str, HawkesToxicWakeState] = {}

    @property
    def wake_states(self) -> dict[str, HawkesToxicWakeState]:
        return self._wake_states

    def get_or_create_state(self, market_id: str) -> HawkesToxicWakeState:
        market_key = str(market_id).strip()
        state = self._wake_states.get(market_key)
        if state is None:
            state = HawkesToxicWakeState(
                decay_rate=self.decay_rate,
                impulse_scale=self.impulse_scale,
            )
            self._wake_states[market_key] = state
        return state

    def process_trade_tick(
        self,
        market_id: str,
        timestamp: float,
        trade_volume: float,
    ) -> float:
        state = self.get_or_create_state(market_id)
        return state.update(timestamp, trade_volume)

    def is_wake_toxic(self, market_id: str, current_timestamp: float) -> bool:
        state = self.get_or_create_state(market_id)
        intensity = state.get_intensity(current_timestamp)
        return intensity >= self.contagion_intensity_threshold