from __future__ import annotations

from dataclasses import dataclass, field


def _normalize_direction(direction: str) -> str:
    value = (direction or "").strip().upper()
    if value not in {"YES", "NO"}:
        raise ValueError(f"Unsupported exposure direction: {direction!r}")
    return value


def _normalize_strategy(strategy_source: str | None) -> str:
    value = (strategy_source or "").strip()
    return value or "unknown_strategy"


@dataclass(slots=True)
class MarketDirectionalExposure:
    yes_by_strategy: dict[str, int] = field(default_factory=dict)
    no_by_strategy: dict[str, int] = field(default_factory=dict)

    def strategy_counts(self, direction: str) -> dict[str, int]:
        return self.yes_by_strategy if direction == "YES" else self.no_by_strategy

    def total(self, direction: str) -> int:
        return sum(self.strategy_counts(direction).values())


@dataclass(frozen=True, slots=True)
class ExposureRegistration:
    market_id: str
    strategy_source: str
    direction: str


class EnsembleRiskManager:
    """O(1) directional exposure gate across live strategies.

    A scalar net exposure is insufficient once hedges are allowed because
    YES and NO inventory can coexist on the same market across strategies.
    This manager therefore stores per-market directional ownership in a
    single hash map while keeping all updates and queries O(1).
    """

    def __init__(self) -> None:
        self._market_exposures: dict[str, MarketDirectionalExposure] = {}
        self._position_index: dict[str, ExposureRegistration] = {}

    def can_enter(
        self,
        *,
        market_id: str,
        strategy_source: str | None,
        direction: str,
    ) -> tuple[bool, dict[str, str] | None]:
        strategy = _normalize_strategy(strategy_source)
        normalized_direction = _normalize_direction(direction)
        exposure = self._market_exposures.get(market_id)
        if exposure is None:
            return True, None

        same_direction = exposure.strategy_counts(normalized_direction)
        blockers = [name for name, count in same_direction.items() if count > 0 and name != strategy]
        if not blockers:
            return True, None

        return False, {
            "market_id": market_id,
            "direction": normalized_direction,
            "strategy_source": strategy,
            "blocking_strategy": sorted(blockers)[0],
        }

    def can_enter_batch(
        self,
        *,
        strategy_source: str | None,
        exposures: list[tuple[str, str]],
    ) -> tuple[bool, dict[str, str] | None]:
        for market_id, direction in exposures:
            allowed, reason = self.can_enter(
                market_id=market_id,
                strategy_source=strategy_source,
                direction=direction,
            )
            if not allowed:
                return False, reason
        return True, None

    def register_position(
        self,
        *,
        position_id: str,
        market_id: str,
        strategy_source: str | None,
        direction: str,
    ) -> None:
        normalized_direction = _normalize_direction(direction)
        normalized_strategy = _normalize_strategy(strategy_source)
        new_registration = ExposureRegistration(
            market_id=market_id,
            strategy_source=normalized_strategy,
            direction=normalized_direction,
        )
        existing = self._position_index.get(position_id)
        if existing == new_registration:
            return
        if existing is not None:
            self.release_position(position_id)

        exposure = self._market_exposures.setdefault(market_id, MarketDirectionalExposure())
        strategy_counts = exposure.strategy_counts(normalized_direction)
        strategy_counts[normalized_strategy] = strategy_counts.get(normalized_strategy, 0) + 1
        self._position_index[position_id] = new_registration

    def release_position(self, position_id: str) -> None:
        existing = self._position_index.pop(position_id, None)
        if existing is None:
            return

        exposure = self._market_exposures.get(existing.market_id)
        if exposure is None:
            return

        strategy_counts = exposure.strategy_counts(existing.direction)
        remaining = strategy_counts.get(existing.strategy_source, 0) - 1
        if remaining > 0:
            strategy_counts[existing.strategy_source] = remaining
        else:
            strategy_counts.pop(existing.strategy_source, None)

        if not exposure.yes_by_strategy and not exposure.no_by_strategy:
            self._market_exposures.pop(existing.market_id, None)

    def exposure_snapshot(self, market_id: str) -> dict[str, dict[str, int]]:
        exposure = self._market_exposures.get(market_id)
        if exposure is None:
            return {"YES": {}, "NO": {}}
        return {
            "YES": dict(exposure.yes_by_strategy),
            "NO": dict(exposure.no_by_strategy),
        }