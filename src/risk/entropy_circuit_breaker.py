from __future__ import annotations

from src.signals.advanced_math import BinaryTouchEntropyMethod


class EntropyCircuitBreaker:
    """Defensive gatekeeper that blocks passive quoting in maximum-confusion books."""

    __slots__ = (
        "_entropy_calculator",
        "critical_entropy_threshold",
        "_last_entropy_by_market",
    )

    def __init__(self, *, critical_entropy_threshold: float = 0.95) -> None:
        self._entropy_calculator = BinaryTouchEntropyMethod()
        self.critical_entropy_threshold = float(critical_entropy_threshold)
        self._last_entropy_by_market: dict[str, float] = {}

    @property
    def last_entropy_by_market(self) -> dict[str, float]:
        return self._last_entropy_by_market

    def is_safe_to_make(
        self,
        market_id: str,
        top_bid_vol: float,
        top_ask_vol: float,
    ) -> bool:
        market_key = str(market_id).strip()
        entropy = self._entropy_calculator.calculate_entropy(top_bid_vol, top_ask_vol)
        self._last_entropy_by_market[market_key] = entropy
        return entropy <= self.critical_entropy_threshold