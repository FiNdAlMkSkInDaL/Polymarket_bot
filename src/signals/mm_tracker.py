from __future__ import annotations

from src.signals.mm_predation import MarketMakerFingerprint


class MarketMakerTracker:
    """O(1) state manager for per-maker quote sensitivity fingerprints."""

    __slots__ = ("_alpha", "_initial_sensitivity", "_fingerprints")

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        initial_sensitivity: float = 0.0,
    ) -> None:
        self._alpha = float(alpha)
        self._initial_sensitivity = float(initial_sensitivity)
        self._fingerprints: dict[str, MarketMakerFingerprint] = {}

    @property
    def fingerprints(self) -> dict[str, MarketMakerFingerprint]:
        return self._fingerprints

    def get_or_create_fingerprint(self, maker_address: str) -> MarketMakerFingerprint:
        maker_key = str(maker_address).strip()
        fingerprint = self._fingerprints.get(maker_key)
        if fingerprint is None:
            fingerprint = MarketMakerFingerprint(
                maker_key,
                alpha=self._alpha,
                initial_sensitivity=self._initial_sensitivity,
            )
            self._fingerprints[maker_key] = fingerprint
        return fingerprint

    def process_fill_event(
        self,
        maker_address: str,
        price_delta: float,
        taker_volume: float,
    ) -> float:
        fingerprint = self.get_or_create_fingerprint(maker_address)
        return fingerprint.update_sensitivity(price_delta, taker_volume)

    def get_vulnerable_makers(
        self,
        target_spread_delta: float,
        max_capital: float,
    ) -> list[tuple[str, float]]:
        capital_limit = float(max_capital)
        vulnerable: list[tuple[str, float]] = []
        for maker_address, fingerprint in self._fingerprints.items():
            required_attack_volume = fingerprint.calculate_attack_volume(target_spread_delta)
            if required_attack_volume == float("inf"):
                continue
            if required_attack_volume <= capital_limit:
                vulnerable.append((maker_address, required_attack_volume))
        return sorted(vulnerable, key=lambda item: item[1])