from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from src.execution.si9_execution_manifest import Si9ExecutionManifest
from src.execution.si9_unwind_manifest import (
    RecommendedAction,
    Si9UnwindConfig,
    Si9UnwindLeg,
    Si9UnwindManifest,
)


class Si9UnwindEvaluator:
    def __init__(self, config: Si9UnwindConfig):
        self._config = config

    def evaluate(
        self,
        cluster_id: str,
        hanging_legs: list[tuple[str, Decimal, Decimal]],
        current_bids: dict[str, Decimal],
        unwind_reason: str,
        original_manifest: Si9ExecutionManifest,
        unwind_timestamp_ms: int,
    ) -> Si9UnwindManifest:
        if cluster_id != original_manifest.cluster_id:
            raise ValueError("cluster_id must match original_manifest.cluster_id")
        if not hanging_legs:
            raise ValueError("hanging_legs must contain at least one filled leg")
        if unwind_reason not in {
            "SECOND_LEG_REJECTED",
            "CLUSTER_STALE",
            "BUS_EVICTED",
            "GUARD_CIRCUIT_OPEN",
            "MANUAL_ABORT",
        }:
            raise ValueError("unsupported unwind_reason")

        leg_index_by_market = {leg.market_id: leg.leg_index for leg in original_manifest.legs}
        unwind_legs: list[Si9UnwindLeg] = []
        total_estimated_unwind_cost = Decimal("0")
        for market_id, filled_size, filled_price in hanging_legs:
            if market_id not in leg_index_by_market:
                raise ValueError(f"Unknown hanging leg market_id: {market_id!r}")
            if filled_size <= Decimal("0"):
                raise ValueError("filled_size must be strictly positive")
            if filled_price <= Decimal("0") or filled_price >= Decimal("1"):
                raise ValueError("filled_price must be in the interval (0, 1)")

            current_best_bid = current_bids.get(market_id)
            if current_best_bid is None:
                raise ValueError(f"Missing current best bid for market_id: {market_id!r}")
            if current_best_bid < self._config.min_best_bid:
                raise ValueError("current_best_bid is below config.min_best_bid")

            estimated_unwind_cost = filled_size * (filled_price - current_best_bid)
            total_estimated_unwind_cost += estimated_unwind_cost
            unwind_legs.append(
                Si9UnwindLeg(
                    market_id=market_id,
                    side="YES",
                    filled_size=filled_size,
                    filled_price=filled_price,
                    current_best_bid=current_best_bid,
                    estimated_unwind_cost=estimated_unwind_cost,
                    leg_index=leg_index_by_market[market_id],
                )
            )

        return Si9UnwindManifest(
            cluster_id=cluster_id,
            hanging_legs=tuple(sorted(unwind_legs, key=lambda leg: leg.leg_index)),
            unwind_reason=unwind_reason,
            original_manifest=original_manifest,
            unwind_timestamp_ms=int(unwind_timestamp_ms),
            total_estimated_unwind_cost=total_estimated_unwind_cost,
            recommended_action=self._recommended_action(unwind_reason, total_estimated_unwind_cost),
        )

    def escalate(
        self,
        manifest: Si9UnwindManifest,
        current_timestamp_ms: int,
    ) -> Si9UnwindManifest:
        elapsed_ms = int(current_timestamp_ms) - manifest.unwind_timestamp_ms
        if manifest.recommended_action != "HOLD_FOR_RECOVERY" or elapsed_ms <= self._config.max_hold_recovery_ms:
            return manifest
        return replace(
            manifest,
            unwind_timestamp_ms=int(current_timestamp_ms),
            recommended_action="MARKET_SELL",
        )

    def _recommended_action(
        self,
        unwind_reason: str,
        total_estimated_unwind_cost: Decimal,
    ) -> RecommendedAction:
        if unwind_reason == "GUARD_CIRCUIT_OPEN" or total_estimated_unwind_cost >= self._config.market_sell_threshold:
            return "MARKET_SELL"
        if total_estimated_unwind_cost <= self._config.passive_unwind_threshold:
            return "PASSIVE_UNWIND"
        return "HOLD_FOR_RECOVERY"