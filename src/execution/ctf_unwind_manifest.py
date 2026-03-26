from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.execution.ctf_execution_manifest import CtfExecutionManifest
from src.execution.unwind_manifest import RecommendedAction


CtfUnwindReason = Literal["SECOND_LEG_REJECTED"]


@dataclass(frozen=True, slots=True)
class CtfUnwindLeg:
    market_id: str
    side: Literal["YES", "NO"]
    filled_size: Decimal
    filled_price: Decimal
    current_best_bid: Decimal
    estimated_unwind_cost: Decimal
    leg_index: int


@dataclass(frozen=True, slots=True)
class CtfUnwindManifest:
    cluster_id: str
    hanging_legs: tuple[CtfUnwindLeg, ...]
    unwind_reason: CtfUnwindReason
    original_manifest: CtfExecutionManifest
    unwind_timestamp_ms: int
    total_estimated_unwind_cost: Decimal
    recommended_action: RecommendedAction