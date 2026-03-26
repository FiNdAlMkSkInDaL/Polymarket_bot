from __future__ import annotations

from decimal import Decimal
from typing import Literal, Protocol, runtime_checkable


RecommendedAction = Literal["MARKET_SELL", "PASSIVE_UNWIND", "HOLD_FOR_RECOVERY"]
UnwindActionTaken = Literal["MARKET_SELL", "PASSIVE_UNWIND", "HOLD", "SKIPPED"]


@runtime_checkable
class UnwindLeg(Protocol):
    market_id: str
    side: Literal["YES", "NO"]
    filled_size: Decimal
    filled_price: Decimal
    current_best_bid: Decimal
    estimated_unwind_cost: Decimal
    leg_index: int


@runtime_checkable
class UnwindManifest(Protocol):
    cluster_id: str
    hanging_legs: tuple[UnwindLeg, ...]
    unwind_reason: str
    original_manifest: object
    unwind_timestamp_ms: int
    total_estimated_unwind_cost: Decimal
    recommended_action: RecommendedAction