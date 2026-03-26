from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.execution.si9_execution_manifest import Si9ExecutionManifest
from src.execution.unwind_manifest import RecommendedAction


UnwindReason = Literal[
    "SECOND_LEG_REJECTED",
    "CLUSTER_STALE",
    "BUS_EVICTED",
    "GUARD_CIRCUIT_OPEN",
    "MANUAL_ABORT",
]


@dataclass(frozen=True, slots=True)
class Si9UnwindLeg:
    market_id: str
    side: Literal["YES"]
    filled_size: Decimal
    filled_price: Decimal
    current_best_bid: Decimal
    estimated_unwind_cost: Decimal
    leg_index: int


@dataclass(frozen=True, slots=True)
class Si9UnwindManifest:
    cluster_id: str
    hanging_legs: tuple[Si9UnwindLeg, ...]
    unwind_reason: UnwindReason
    original_manifest: Si9ExecutionManifest
    unwind_timestamp_ms: int
    total_estimated_unwind_cost: Decimal
    recommended_action: RecommendedAction


@dataclass(frozen=True, slots=True)
class Si9UnwindConfig:
    market_sell_threshold: Decimal
    passive_unwind_threshold: Decimal
    max_hold_recovery_ms: int
    min_best_bid: Decimal

    def __post_init__(self) -> None:
        if self.market_sell_threshold <= Decimal("0"):
            raise ValueError("market_sell_threshold must be strictly positive")
        if self.passive_unwind_threshold <= Decimal("0"):
            raise ValueError("passive_unwind_threshold must be strictly positive")
        if self.market_sell_threshold <= self.passive_unwind_threshold:
            raise ValueError("market_sell_threshold must be strictly greater than passive_unwind_threshold")
        if not isinstance(self.max_hold_recovery_ms, int) or self.max_hold_recovery_ms <= 0:
            raise ValueError("max_hold_recovery_ms must be a strictly positive int")
        if self.min_best_bid <= Decimal("0") or self.min_best_bid >= Decimal("1"):
            raise ValueError("min_best_bid must be in the interval (0, 1)")