from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True, slots=True)
class Si9LegManifest:
    market_id: str
    side: Literal["YES"]
    target_price: Decimal
    target_size: Decimal
    is_bottleneck: bool
    leg_index: int


@dataclass(frozen=True, slots=True)
class Si9ExecutionManifest:
    cluster_id: str
    # FLAG-1-RESOLVED: the manifest fixes all legs to the same target size and
    # orders the bottleneck leg first, which is the execution-side contract the
    # hanging-leg unwind path can reconcile against.
    legs: tuple[Si9LegManifest, ...]
    net_edge: Decimal
    required_share_counts: Decimal
    bottleneck_market_id: str
    manifest_timestamp_ms: int
    # FLAG-2-RESOLVED: these stale/cancel timing controls travel with the frozen
    # manifest so execution never re-queries detector state after planning.
    max_leg_fill_wait_ms: int
    cancel_on_stale_ms: int