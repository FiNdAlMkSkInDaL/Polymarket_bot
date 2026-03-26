from __future__ import annotations

import random


class SimulatedMarketMaker:
    """Deterministic institutional-style market maker with fixed quote sensitivity."""

    __slots__ = (
        "maker_address",
        "true_kappa",
        "mid_price",
        "inventory",
        "_noise_std",
        "_rng",
    )

    def __init__(
        self,
        maker_address: str,
        *,
        true_kappa: float = 0.00005,
        initial_mid_price: float = 0.5,
        noise_std: float = 0.00001,
        random_seed: int | None = 7,
    ) -> None:
        kappa_value = float(true_kappa)
        if kappa_value <= 0.0:
            raise ValueError("true_kappa must be strictly greater than 0")

        self.maker_address = str(maker_address).strip()
        self.true_kappa = kappa_value
        self.mid_price = float(initial_mid_price)
        self.inventory = 0.0
        self._noise_std = max(0.0, float(noise_std))
        self._rng = random.Random(random_seed)

    def receive_taker_flow(self, volume: float) -> float:
        taker_volume = float(volume)
        if taker_volume <= 0.0:
            raise ValueError("volume must be strictly greater than 0")

        self.inventory += taker_volume
        noise = self._rng.gauss(0.0, self._noise_std) if self._noise_std > 0.0 else 0.0
        self.mid_price += self.true_kappa * taker_volume + noise
        return self.mid_price