from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class OfiLedgerSnapshot:
    total_signals: int
    total_dispatched: int
    total_guard_rejected: int
    total_bus_rejected: int
    total_source_disabled: int
    gross_capital_deployed: Decimal
    mean_conviction_scalar: Decimal
    dispatch_rate: Decimal
    first_signal_ms: int | None
    last_signal_ms: int | None


class OfiPaperLedger:
    __slots__ = (
        "_total_signals",
        "_total_dispatched",
        "_total_guard_rejected",
        "_total_bus_rejected",
        "_total_source_disabled",
        "_gross_capital_deployed",
        "_conviction_scalar_sum",
        "_first_signal_ms",
        "_last_signal_ms",
    )

    def __init__(self) -> None:
        self.reset()

    def record_signal(
        self,
        *,
        outcome: str,
        conviction_scalar: Decimal,
        timestamp_ms: int,
        deployed_capital: Decimal = Decimal("0"),
    ) -> None:
        event_timestamp_ms = int(timestamp_ms)
        self._total_signals += 1
        if self._first_signal_ms is None:
            self._first_signal_ms = event_timestamp_ms
        self._last_signal_ms = event_timestamp_ms

        if outcome == "DISPATCHED":
            self._total_dispatched += 1
            self._gross_capital_deployed += deployed_capital
            self._conviction_scalar_sum += conviction_scalar
            return
        if outcome == "GUARD_REJECTED":
            self._total_guard_rejected += 1
            return
        if outcome == "BUS_REJECTED":
            self._total_bus_rejected += 1
            return
        if outcome == "SOURCE_DISABLED":
            self._total_source_disabled += 1

    def snapshot(self) -> OfiLedgerSnapshot:
        mean_conviction_scalar = Decimal("0")
        if self._total_dispatched > 0:
            mean_conviction_scalar = self._conviction_scalar_sum / Decimal(self._total_dispatched)

        dispatch_rate = Decimal("0")
        if self._total_signals > 0:
            dispatch_rate = Decimal(self._total_dispatched) / Decimal(self._total_signals)

        return OfiLedgerSnapshot(
            total_signals=self._total_signals,
            total_dispatched=self._total_dispatched,
            total_guard_rejected=self._total_guard_rejected,
            total_bus_rejected=self._total_bus_rejected,
            total_source_disabled=self._total_source_disabled,
            gross_capital_deployed=self._gross_capital_deployed,
            mean_conviction_scalar=mean_conviction_scalar,
            dispatch_rate=dispatch_rate,
            first_signal_ms=self._first_signal_ms,
            last_signal_ms=self._last_signal_ms,
        )

    def reset(self) -> OfiLedgerSnapshot:
        self._total_signals = 0
        self._total_dispatched = 0
        self._total_guard_rejected = 0
        self._total_bus_rejected = 0
        self._total_source_disabled = 0
        self._gross_capital_deployed = Decimal("0")
        self._conviction_scalar_sum = Decimal("0")
        self._first_signal_ms = None
        self._last_signal_ms = None
        return self.snapshot()