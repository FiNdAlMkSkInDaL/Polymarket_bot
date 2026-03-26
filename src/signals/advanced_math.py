"""Advanced $O(1)$ microstructure math utilities for binary markets."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class HawkesToxicWakeState:
    """Single-factor Hawkes-style excitation tracker with recursive decay.

    The state update is:

    excitation_t = excitation_{t-1} * exp(-decay_rate * dt) + impulse_scale * volume

    This preserves $O(1)$ updates because the previous state is the only
    quantity carried forward.
    """

    decay_rate: float = 1.0
    impulse_scale: float = 1.0
    last_update_time: float = 0.0
    excitation_state: float = 0.0

    def update(self, timestamp: float, trade_volume: float) -> float:
        """Apply time decay through *timestamp* and add the new volume impulse."""
        event_time = float(timestamp)
        dt = self._elapsed(event_time)
        decayed_state = self.excitation_state * math.exp(-self.decay_rate * dt)
        impulse = self.impulse_scale * max(0.0, float(trade_volume))
        self.excitation_state = decayed_state + impulse
        self.last_update_time = max(self.last_update_time, event_time)
        return self.excitation_state

    def get_intensity(self, current_timestamp: float) -> float:
        """Return the decayed excitation intensity without mutating state."""
        dt = self._elapsed(float(current_timestamp))
        return self.excitation_state * math.exp(-self.decay_rate * dt)

    def _elapsed(self, timestamp: float) -> float:
        if self.last_update_time <= 0.0:
            return 0.0
        return max(0.0, timestamp - self.last_update_time)


class BinaryTouchEntropyMethod:
    """Binary Shannon entropy over bid/ask touch volume share."""

    @staticmethod
    def calculate_entropy(bid_volume: float, ask_volume: float) -> float:
        bid = max(0.0, float(bid_volume))
        ask = max(0.0, float(ask_volume))
        total = bid + ask
        if total <= 0.0:
            return 0.0

        entropy = 0.0
        for probability in (bid / total, ask / total):
            if probability > 0.0:
                entropy -= probability * math.log2(probability)
        return entropy


class BoundaryAvellanedaStoikovMethod:
    """Binary-boundary reservation price under bounded variance."""

    @staticmethod
    def get_reservation_price(
        mid_price: float,
        inventory_qty: float,
        risk_aversion_gamma: float,
        raw_variance: float,
        time_to_resolution: float,
    ) -> float:
        bounded_mid = min(1.0, max(0.0, float(mid_price)))
        bounded_variance = max(0.0, float(raw_variance)) * bounded_mid * (1.0 - bounded_mid)
        horizon = max(0.0, float(time_to_resolution))
        inventory = float(inventory_qty)
        gamma = max(0.0, float(risk_aversion_gamma))
        reservation_price = bounded_mid - inventory * gamma * bounded_variance * horizon
        return min(1.0, max(0.0, reservation_price))