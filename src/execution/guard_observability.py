from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.execution.dispatch_guard import DispatchGuard
from src.execution.signal_coordination_bus import CoordinationBusSnapshot, SignalCoordinationBus


@dataclass(frozen=True, slots=True)
class SuppressionEntry:
    signal_source: str
    circuit_state: str
    consecutive_suppressions: int
    active_dedup_keys: int
    per_source_dispatch_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ObservabilitySnapshot:
    snapshot_timestamp_ms: int
    per_source: dict[str, dict]
    bus_snapshot: CoordinationBusSnapshot
    total_circuit_opens: int
    total_active_slots: int
    highest_suppression_source: str | None
    system_health: Literal["GREEN", "YELLOW", "RED"]


class GuardObservabilityPanel:
    def __init__(self, guards: dict[str, DispatchGuard], bus: SignalCoordinationBus):
        self._guards = guards
        self._bus = bus

    def full_snapshot(self, current_timestamp_ms: int) -> ObservabilitySnapshot:
        snapshot_timestamp_ms = int(current_timestamp_ms)
        per_source = {
            signal_source: guard.guard_snapshot(snapshot_timestamp_ms)
            for signal_source, guard in self._guards.items()
        }
        bus_snapshot = self._bus.bus_snapshot(snapshot_timestamp_ms)
        total_circuit_opens = sum(
            1 for snapshot in per_source.values() if snapshot["circuit_state"] == "OPEN"
        )
        ranked_sources = sorted(
            per_source.items(),
            key=lambda item: (-item[1]["consecutive_suppressions"], item[0]),
        )
        highest_suppression_source = None
        if ranked_sources and ranked_sources[0][1]["consecutive_suppressions"] > 0:
            highest_suppression_source = ranked_sources[0][0]

        if total_circuit_opens > 0:
            system_health: Literal["GREEN", "YELLOW", "RED"] = "RED"
        elif any(
            snapshot["consecutive_suppressions"] >= (self._guards[source].config.circuit_breaker_threshold / 2)
            for source, snapshot in per_source.items()
        ):
            system_health = "YELLOW"
        else:
            system_health = "GREEN"

        return ObservabilitySnapshot(
            snapshot_timestamp_ms=snapshot_timestamp_ms,
            per_source=per_source,
            bus_snapshot=bus_snapshot,
            total_circuit_opens=total_circuit_opens,
            total_active_slots=bus_snapshot.total_active_slots,
            highest_suppression_source=highest_suppression_source,
            system_health=system_health,
        )

    def suppression_report(self, current_timestamp_ms: int) -> list[SuppressionEntry]:
        snapshot_timestamp_ms = int(current_timestamp_ms)
        report = [
            SuppressionEntry(
                signal_source=signal_source,
                circuit_state=snapshot["circuit_state"],
                consecutive_suppressions=snapshot["consecutive_suppressions"],
                active_dedup_keys=snapshot["active_dedup_keys"],
                per_source_dispatch_counts=snapshot["per_source_dispatch_counts"],
            )
            for signal_source, snapshot in (
                (signal_source, guard.guard_snapshot(snapshot_timestamp_ms))
                for signal_source, guard in self._guards.items()
            )
        ]
        return sorted(report, key=lambda entry: (-entry.consecutive_suppressions, entry.signal_source))