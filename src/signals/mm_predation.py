from __future__ import annotations


class MarketMakerFingerprint:
    """Track a maker's quote sensitivity with O(1) recursive updates."""

    __slots__ = (
        "maker_address",
        "_alpha",
        "_kappa_ewma",
        "_kappa_current",
        "_observations",
    )

    def __init__(
        self,
        maker_address: str,
        *,
        alpha: float = 0.2,
        initial_sensitivity: float = 0.0,
    ) -> None:
        alpha_value = float(alpha)
        if not 0.0 < alpha_value <= 1.0:
            raise ValueError("alpha must be in the interval (0.0, 1.0]")

        self.maker_address = str(maker_address).strip()
        self._alpha = alpha_value
        self._kappa_ewma = float(initial_sensitivity)
        self._kappa_current = 0.0
        self._observations = 0

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def kappa_current(self) -> float:
        return self._kappa_current

    @property
    def kappa_ewma(self) -> float:
        return self._kappa_ewma

    @property
    def observations(self) -> int:
        return self._observations

    def update_sensitivity(self, price_delta: float, taker_volume: float) -> float:
        volume = float(taker_volume)
        if volume <= 0.0:
            raise ValueError("taker_volume must be strictly greater than 0")

        self._kappa_current = float(price_delta) / volume
        self._kappa_ewma = (
            self._alpha * self._kappa_current
            + (1.0 - self._alpha) * self._kappa_ewma
        )
        self._observations += 1
        return self._kappa_ewma

    def calculate_attack_volume(self, target_spread_delta: float) -> float:
        spread_delta = float(target_spread_delta)
        if spread_delta == 0.0:
            return 0.0
        if abs(self._kappa_ewma) <= 1e-12:
            return float("inf")
        return spread_delta / self._kappa_ewma