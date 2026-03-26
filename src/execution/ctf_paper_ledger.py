from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.execution.ctf_execution_manifest import CtfExecutionReceipt


@dataclass(frozen=True, slots=True)
class CtfLedgerSnapshot:
    total_clusters_attempted: int
    total_dispatched: int
    total_executed: int
    total_suppressed: int
    total_anchor_rejected: int
    total_second_leg_rejected: int
    total_partial_fills: int
    total_bus_rejected: int
    gross_edge_captured: Decimal
    gross_realized_pnl: Decimal
    total_capital_deployed: Decimal
    mean_net_edge: Decimal
    mean_conviction_scalar: Decimal
    second_leg_rejection_rate: Decimal
    first_dispatch_ms: int | None
    last_dispatch_ms: int | None


class CtfPaperLedger:
    __slots__ = (
        "_total_clusters_attempted",
        "_total_dispatched",
        "_total_executed",
        "_total_suppressed",
        "_total_anchor_rejected",
        "_total_second_leg_rejected",
        "_total_partial_fills",
        "_total_bus_rejected",
        "_gross_edge_captured",
        "_gross_realized_pnl",
        "_total_capital_deployed",
        "_total_conviction_scalar",
        "_first_dispatch_ms",
        "_last_dispatch_ms",
    )

    def __init__(self) -> None:
        self.reset()

    def record(self, receipt: CtfExecutionReceipt, conviction_scalar: Decimal) -> None:
        timestamp_ms = int(receipt.execution_timestamp_ms)
        self._total_clusters_attempted += 1
        if self._first_dispatch_ms is None:
            self._first_dispatch_ms = timestamp_ms
        self._last_dispatch_ms = timestamp_ms

        outcome = receipt.execution_outcome
        if outcome == "BUS_REJECTED":
            self._total_bus_rejected += 1
            return
        if outcome == "GUARD_REJECTED":
            self._total_suppressed += 1
            return

        self._total_dispatched += 1
        self._total_capital_deployed += receipt.total_capital_deployed

        if outcome == "ANCHOR_REJECTED":
            self._total_anchor_rejected += 1
            return
        if outcome == "SECOND_LEG_REJECTED":
            self._total_second_leg_rejected += 1
            return
        if outcome == "PARTIAL_FILL":
            self._total_partial_fills += 1
            self._total_executed += 1
            self._gross_edge_captured += receipt.realized_net_edge * self._filled_cluster_size(receipt)
            self._total_conviction_scalar += conviction_scalar
            return

        self._total_executed += 1
        self._gross_edge_captured += receipt.realized_net_edge * self._filled_cluster_size(receipt)
        self._gross_realized_pnl += receipt.realized_pnl
        self._total_conviction_scalar += conviction_scalar

    def ledger_snapshot(self) -> CtfLedgerSnapshot:
        mean_net_edge = Decimal("0")
        if self._total_capital_deployed > Decimal("0"):
            mean_net_edge = self._gross_edge_captured / self._total_capital_deployed

        mean_conviction_scalar = Decimal("0")
        if self._total_executed > 0:
            mean_conviction_scalar = self._total_conviction_scalar / Decimal(self._total_executed)

        second_leg_rejection_rate = Decimal("0")
        if self._total_clusters_attempted > 0:
            second_leg_rejection_rate = Decimal(self._total_second_leg_rejected) / Decimal(self._total_clusters_attempted)

        return CtfLedgerSnapshot(
            total_clusters_attempted=self._total_clusters_attempted,
            total_dispatched=self._total_dispatched,
            total_executed=self._total_executed,
            total_suppressed=self._total_suppressed,
            total_anchor_rejected=self._total_anchor_rejected,
            total_second_leg_rejected=self._total_second_leg_rejected,
            total_partial_fills=self._total_partial_fills,
            total_bus_rejected=self._total_bus_rejected,
            gross_edge_captured=self._gross_edge_captured,
            gross_realized_pnl=self._gross_realized_pnl,
            total_capital_deployed=self._total_capital_deployed,
            mean_net_edge=mean_net_edge,
            mean_conviction_scalar=mean_conviction_scalar,
            second_leg_rejection_rate=second_leg_rejection_rate,
            first_dispatch_ms=self._first_dispatch_ms,
            last_dispatch_ms=self._last_dispatch_ms,
        )

    def reset(self) -> None:
        self._total_clusters_attempted = 0
        self._total_dispatched = 0
        self._total_executed = 0
        self._total_suppressed = 0
        self._total_anchor_rejected = 0
        self._total_second_leg_rejected = 0
        self._total_partial_fills = 0
        self._total_bus_rejected = 0
        self._gross_edge_captured = Decimal("0")
        self._gross_realized_pnl = Decimal("0")
        self._total_capital_deployed = Decimal("0")
        self._total_conviction_scalar = Decimal("0")
        self._first_dispatch_ms = None
        self._last_dispatch_ms = None

    @staticmethod
    def _filled_cluster_size(receipt: CtfExecutionReceipt) -> Decimal:
        return min(receipt.yes_receipt.filled_size, receipt.no_receipt.filled_size)