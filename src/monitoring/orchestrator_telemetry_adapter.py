from __future__ import annotations

from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator
from src.execution.orchestrator_health_monitor import OrchestratorHealthMonitor


class OrchestratorTelemetryAdapter:
    def __init__(
        self,
        orchestrator: MultiSignalOrchestrator,
        health_monitor: OrchestratorHealthMonitor,
    ) -> None:
        if orchestrator is None:
            raise ValueError("orchestrator is required")
        if health_monitor is None:
            raise ValueError("health_monitor is required")
        self._orchestrator = orchestrator
        self._health_monitor = health_monitor

    def export_health_snapshot(self, current_timestamp_ms: int) -> dict[str, Any]:
        timestamp_ms = int(current_timestamp_ms)
        health_report = self._health_monitor.check(timestamp_ms)
        orchestrator_snapshot = self._orchestrator.orchestrator_snapshot(timestamp_ms)
        guard_snapshot = self._orchestrator.guard.guard_snapshot(timestamp_ms)
        bus_snapshot = orchestrator_snapshot.observability.bus_snapshot

        return {
            "timestamp_ms": timestamp_ms,
            "orchestrator": self._json_safe(
                {
                    "health": orchestrator_snapshot.health,
                    "pending_unwind_count": orchestrator_snapshot.pending_unwind_count,
                    "active_position_count": orchestrator_snapshot.active_position_count,
                }
            ),
            "health_monitor": self._json_safe(
                {
                    "is_safe_to_trade": health_report.is_safe_to_trade,
                    "consecutive_release_failures": health_report.consecutive_release_failures,
                    "halt_reason": health_report.halt_reason,
                }
            ),
            "dispatch_guard": self._json_safe(
                {
                    "circuit_breaker_status": {
                        "state": guard_snapshot["circuit_state"],
                        "consecutive_suppressions": guard_snapshot["consecutive_suppressions"],
                        "opened_at_ms": guard_snapshot["circuit_opened_at_ms"],
                    },
                    "dispatch_rate_counters": guard_snapshot["per_source_dispatch_counts"],
                    "active_open_positions_by_market": guard_snapshot["open_position_counts"],
                    "active_dedup_keys": guard_snapshot["active_dedup_keys"],
                }
            ),
            "coordination_bus": self._json_safe(
                {
                    "total_active_slots": bus_snapshot.total_active_slots,
                    "slots_by_source": bus_snapshot.slots_by_source,
                    "slots_by_market": bus_snapshot.slots_by_market,
                    "expired_reclaimed_count": bus_snapshot.expired_reclaimed_count,
                    "active_slot_leases": [],
                }
            ),
            "unwind_ledger": self._json_safe(
                {
                    "active_unwinds": [],
                }
            ),
        }

    def _json_safe(self, value: Any):
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, Enum):
            return str(value.value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if is_dataclass(value):
            return self._json_safe(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._json_safe(item) for item in value]
        return str(value)