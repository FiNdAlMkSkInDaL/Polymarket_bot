from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class Si9ClusterConfig:
    target_yield: Decimal
    taker_fee_per_leg: Decimal
    slippage_budget: Decimal
    ghost_town_floor: Decimal = Decimal("0.85")
    implausible_edge_ceil: Decimal = Decimal("0.15")
    max_ask_age_ms: int = 1000
    min_cluster_size: int = 2
    tiebreak_policy: Literal["stable_index", "lowest_market_id"] = "lowest_market_id"

    def __post_init__(self) -> None:
        if self.target_yield < Decimal("0") or self.target_yield >= Decimal("1"):
            raise ValueError("target_yield must be in the interval [0, 1)")
        if self.taker_fee_per_leg <= Decimal("0"):
            raise ValueError("taker_fee_per_leg must be strictly positive")
        if self.slippage_budget < Decimal("0"):
            raise ValueError("slippage_budget cannot be negative")
        if self.ghost_town_floor <= Decimal("0") or self.ghost_town_floor >= Decimal("1"):
            raise ValueError("ghost_town_floor must be in the interval (0, 1)")
        if self.implausible_edge_ceil <= Decimal("0") or self.implausible_edge_ceil >= Decimal("1"):
            raise ValueError("implausible_edge_ceil must be in the interval (0, 1)")
        if self.max_ask_age_ms < 0:
            raise ValueError("max_ask_age_ms cannot be negative")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.tiebreak_policy not in {"stable_index", "lowest_market_id"}:
            raise ValueError("tiebreak_policy must be 'stable_index' or 'lowest_market_id'")