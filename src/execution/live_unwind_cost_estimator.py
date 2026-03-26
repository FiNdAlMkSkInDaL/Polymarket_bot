from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.execution.unwind_manifest import UnwindLeg, UnwindManifest


class LiveUnwindCostEstimator:
    def __init__(self, best_bid_providers: dict[str, OrderbookBestBidProvider]):
        self._best_bid_providers = dict(best_bid_providers)

    def estimate_manifest(self, manifest: UnwindManifest):
        updated_legs = []
        total_estimated_unwind_cost = Decimal("0")
        for leg in manifest.hanging_legs:
            current_best_bid = self._get_best_bid(leg.market_id)
            estimated_unwind_cost = self._estimate_leg_cost(
                filled_size=leg.filled_size,
                filled_price=leg.filled_price,
                current_best_bid=current_best_bid,
            )
            total_estimated_unwind_cost += estimated_unwind_cost
            updated_legs.append(
                replace(
                    leg,
                    current_best_bid=current_best_bid,
                    estimated_unwind_cost=estimated_unwind_cost,
                )
            )
        return replace(
            manifest,
            hanging_legs=tuple(updated_legs),
            total_estimated_unwind_cost=total_estimated_unwind_cost,
        )

    def estimate_total_cost(self, manifest: UnwindManifest) -> Decimal:
        refreshed_manifest = self.estimate_manifest(manifest)
        return refreshed_manifest.total_estimated_unwind_cost

    @staticmethod
    def _estimate_leg_cost(
        *,
        filled_size: Decimal,
        filled_price: Decimal,
        current_best_bid: Decimal,
    ) -> Decimal:
        if not isinstance(filled_size, Decimal) or not filled_size.is_finite():
            raise ValueError("filled_size must be a finite Decimal")
        if not isinstance(filled_price, Decimal) or not filled_price.is_finite():
            raise ValueError("filled_price must be a finite Decimal")
        if not isinstance(current_best_bid, Decimal) or not current_best_bid.is_finite():
            raise ValueError("current_best_bid must be a finite Decimal")
        return filled_size * (filled_price - current_best_bid)

    def _get_best_bid(self, market_id: str) -> Decimal:
        provider = self._best_bid_providers.get(str(market_id).strip())
        if provider is None:
            raise ValueError(f"Missing OrderbookBestBidProvider for market_id: {market_id!r}")
        best_bid = provider.get_best_bid(market_id)
        if best_bid is None:
            raise ValueError(f"Missing live best bid for market_id: {market_id!r}")
        if not isinstance(best_bid, Decimal) or not best_bid.is_finite():
            raise ValueError(f"Live best bid must be a finite Decimal for market_id: {market_id!r}")
        return best_bid