from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.si9_execution_manifest import Si9ExecutionManifest
from src.execution.si9_unwind_manifest import Si9UnwindManifest


@dataclass(frozen=True, slots=True)
class Si9LedgerSnapshot:
    total_clusters_attempted: int
    total_full_fills: int
    total_hanging_leg_events: int
    total_cluster_suppressed: int
    total_guard_rejected: int
    total_unwind_manifests: int
    total_market_sell_recommendations: int
    total_passive_unwind_recommendations: int
    total_hold_recommendations: int
    gross_edge_captured: Decimal
    gross_estimated_unwind_cost: Decimal
    total_capital_deployed: Decimal
    mean_estimated_unwind_cost: Decimal
    mean_net_edge: Decimal
    hanging_leg_rate: Decimal
    first_dispatch_ms: int | None
    last_dispatch_ms: int | None


class Si9PaperLedger:
    """O(1) accumulator ledger for SI-9 paper execution outcomes."""

    __slots__ = (
        "_total_clusters_attempted",
        "_total_full_fills",
        "_total_hanging_leg_events",
        "_total_cluster_suppressed",
        "_total_guard_rejected",
        "_total_unwind_manifests",
        "_total_market_sell_recommendations",
        "_total_passive_unwind_recommendations",
        "_total_hold_recommendations",
        "_gross_edge_captured",
        "_gross_estimated_unwind_cost",
        "_total_capital_deployed",
        "_net_edge_sum",
        "_first_dispatch_ms",
        "_last_dispatch_ms",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Si9LedgerSnapshot:
        self._total_clusters_attempted = 0
        self._total_full_fills = 0
        self._total_hanging_leg_events = 0
        self._total_cluster_suppressed = 0
        self._total_guard_rejected = 0
        self._total_unwind_manifests = 0
        self._total_market_sell_recommendations = 0
        self._total_passive_unwind_recommendations = 0
        self._total_hold_recommendations = 0
        self._gross_edge_captured = Decimal("0")
        self._gross_estimated_unwind_cost = Decimal("0")
        self._total_capital_deployed = Decimal("0")
        self._net_edge_sum = Decimal("0")
        self._first_dispatch_ms = None
        self._last_dispatch_ms = None
        return self.snapshot()

    def record_full_fill(
        self,
        manifest: Si9ExecutionManifest,
        receipts: tuple[DispatchReceipt, ...],
        current_timestamp_ms: int,
    ) -> None:
        self._record_attempt(manifest, receipts, current_timestamp_ms)
        self._total_full_fills += 1
        self._gross_edge_captured += manifest.net_edge * manifest.required_share_counts

    def record_hanging_leg(
        self,
        manifest: Si9ExecutionManifest,
        receipts: tuple[DispatchReceipt, ...],
        current_timestamp_ms: int,
    ) -> None:
        self._record_attempt(manifest, receipts, current_timestamp_ms)
        self._total_hanging_leg_events += 1

    def record_cluster_suppressed(
        self,
        manifest: Si9ExecutionManifest,
        current_timestamp_ms: int,
    ) -> None:
        self._record_attempt(manifest, tuple(), current_timestamp_ms)
        self._total_cluster_suppressed += 1

    def record_guard_rejected(
        self,
        manifest: Si9ExecutionManifest,
        receipts: tuple[DispatchReceipt, ...],
        current_timestamp_ms: int,
    ) -> None:
        self._record_attempt(manifest, receipts, current_timestamp_ms)
        self._total_guard_rejected += 1

    def record_unwind_manifest(self, unwind_manifest: Si9UnwindManifest) -> None:
        self._total_unwind_manifests += 1
        self._gross_estimated_unwind_cost += unwind_manifest.total_estimated_unwind_cost
        if unwind_manifest.recommended_action == "MARKET_SELL":
            self._total_market_sell_recommendations += 1
        elif unwind_manifest.recommended_action == "PASSIVE_UNWIND":
            self._total_passive_unwind_recommendations += 1
        else:
            self._total_hold_recommendations += 1

    def snapshot(self) -> Si9LedgerSnapshot:
        mean_net_edge = Decimal("0")
        mean_estimated_unwind_cost = Decimal("0")
        hanging_leg_rate = Decimal("0")
        if self._total_clusters_attempted > 0:
            attempts = Decimal(self._total_clusters_attempted)
            mean_net_edge = self._net_edge_sum / attempts
            hanging_leg_rate = Decimal(self._total_hanging_leg_events) / attempts
        if self._total_unwind_manifests > 0:
            mean_estimated_unwind_cost = self._gross_estimated_unwind_cost / Decimal(self._total_unwind_manifests)
        return Si9LedgerSnapshot(
            total_clusters_attempted=self._total_clusters_attempted,
            total_full_fills=self._total_full_fills,
            total_hanging_leg_events=self._total_hanging_leg_events,
            total_cluster_suppressed=self._total_cluster_suppressed,
            total_guard_rejected=self._total_guard_rejected,
            total_unwind_manifests=self._total_unwind_manifests,
            total_market_sell_recommendations=self._total_market_sell_recommendations,
            total_passive_unwind_recommendations=self._total_passive_unwind_recommendations,
            total_hold_recommendations=self._total_hold_recommendations,
            gross_edge_captured=self._gross_edge_captured,
            gross_estimated_unwind_cost=self._gross_estimated_unwind_cost,
            total_capital_deployed=self._total_capital_deployed,
            mean_estimated_unwind_cost=mean_estimated_unwind_cost,
            mean_net_edge=mean_net_edge,
            hanging_leg_rate=hanging_leg_rate,
            first_dispatch_ms=self._first_dispatch_ms,
            last_dispatch_ms=self._last_dispatch_ms,
        )

    def _record_attempt(
        self,
        manifest: Si9ExecutionManifest,
        receipts: tuple[DispatchReceipt, ...],
        current_timestamp_ms: int,
    ) -> None:
        timestamp_ms = int(current_timestamp_ms)
        self._total_clusters_attempted += 1
        self._net_edge_sum += manifest.net_edge
        if self._first_dispatch_ms is None:
            self._first_dispatch_ms = timestamp_ms
        self._last_dispatch_ms = timestamp_ms
        self._total_capital_deployed += sum(
            (
                receipt.fill_price * receipt.fill_size
                for receipt in receipts
                if receipt.executed and receipt.fill_price is not None and receipt.fill_size is not None
            ),
            start=Decimal("0"),
        )