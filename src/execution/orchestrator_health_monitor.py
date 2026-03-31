from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator
from src.execution.priority_context import PriorityOrderContext


@dataclass(frozen=True, slots=True)
class HealthMonitorConfig:
    max_release_failures_before_halt: int
    stale_snapshot_threshold_ms: int
    min_heartbeat_interval_ms: int

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.max_release_failures_before_halt, int) or self.max_release_failures_before_halt < 1:
            errors.append("max_release_failures_before_halt must be >= 1")
        if not isinstance(self.stale_snapshot_threshold_ms, int) or self.stale_snapshot_threshold_ms <= 0:
            errors.append("stale_snapshot_threshold_ms must be a strictly positive int")
        if not isinstance(self.min_heartbeat_interval_ms, int) or self.min_heartbeat_interval_ms <= 0:
            errors.append("min_heartbeat_interval_ms must be a strictly positive int")
        if errors:
            raise ValueError("; ".join(errors))


@dataclass(frozen=True, slots=True)
class HealthReport:
    timestamp_ms: int
    orchestrator_health: Literal["GREEN", "YELLOW", "RED"]
    is_safe_to_trade: bool
    consecutive_release_failures: int
    last_snapshot_age_ms: int
    heartbeat_ok: bool
    halt_reason: str | None
    allows_position_management: bool = True
    allows_new_panic_entries: bool = True


class OrchestratorHealthMonitor:
    def __init__(
        self,
        orchestrator: MultiSignalOrchestrator,
        config: HealthMonitorConfig,
    ):
        self._orchestrator = orchestrator
        self._config = config
        self._consecutive_release_failures = 0
        self._last_check_timestamp_ms: int | None = None
        dispatcher = getattr(self._orchestrator, "dispatcher", None)
        bind_gate = getattr(dispatcher, "bind_pre_dispatch_gate", None)
        if callable(bind_gate):
            bind_gate(self)

    def check(self, current_timestamp_ms: int) -> HealthReport:
        return self._build_report(int(current_timestamp_ms), advance_heartbeat=True)

    def dispatch_guard_reason(
        self,
        context: PriorityOrderContext,
        dispatch_timestamp_ms: int,
    ) -> str | None:
        report = self._build_report(int(dispatch_timestamp_ms), advance_heartbeat=False)
        if report.orchestrator_health == "RED":
            return report.halt_reason or "ORCHESTRATOR_HEALTH_RED"
        if report.orchestrator_health == "YELLOW" and context.signal_source in {"OFI", "CONTAGION", "REWARD"}:
            return report.halt_reason or "DEGRADED_RISK_ENTRY_BLOCKED"
        return None

    def _build_report(self, timestamp_ms: int, *, advance_heartbeat: bool) -> HealthReport:
        snapshot = self._orchestrator.orchestrator_snapshot(timestamp_ms)
        last_snapshot_age_ms = max(0, timestamp_ms - int(snapshot.timestamp_ms))

        heartbeat_gap_ms = 0
        if self._last_check_timestamp_ms is not None:
            heartbeat_gap_ms = max(0, timestamp_ms - self._last_check_timestamp_ms)
        heartbeat_ok = self._last_check_timestamp_ms is None or (
            heartbeat_gap_ms <= (self._config.min_heartbeat_interval_ms * 3)
        )

        effective_health, halt_reason = self._resolve_health_state(
            snapshot_health=snapshot.health,
            last_snapshot_age_ms=last_snapshot_age_ms,
            heartbeat_ok=heartbeat_ok,
        )
        report = HealthReport(
            timestamp_ms=timestamp_ms,
            orchestrator_health=effective_health,
            is_safe_to_trade=(effective_health == "GREEN"),
            consecutive_release_failures=self._consecutive_release_failures,
            last_snapshot_age_ms=last_snapshot_age_ms,
            heartbeat_ok=heartbeat_ok,
            halt_reason=halt_reason,
            allows_position_management=(effective_health != "RED"),
            allows_new_panic_entries=(effective_health == "GREEN"),
        )
        if advance_heartbeat:
            self._last_check_timestamp_ms = timestamp_ms
        return report

    def is_safe_to_trade(self, current_timestamp_ms: int) -> bool:
        """
        Conservative gate. Returns True only when the effective health is GREEN.
        Callers without entry-vs-exit intent should default to no new entries on YELLOW.
        """
        return self.check(current_timestamp_ms).is_safe_to_trade

    def record_position_release_failure(self) -> None:
        self._consecutive_release_failures += 1

    def reset_release_failure_count(self) -> None:
        self._consecutive_release_failures = 0

    def _resolve_health_state(
        self,
        *,
        snapshot_health: Literal["GREEN", "YELLOW", "RED"],
        last_snapshot_age_ms: int,
        heartbeat_ok: bool,
    ) -> tuple[Literal["GREEN", "YELLOW", "RED"], str | None]:
        if snapshot_health == "RED":
            return "RED", "ORCHESTRATOR_HEALTH_RED"
        if self._consecutive_release_failures >= self._config.max_release_failures_before_halt:
            return "YELLOW", "RELEASE_FAILURE_THRESHOLD_REACHED"
        if last_snapshot_age_ms > self._config.stale_snapshot_threshold_ms:
            return "YELLOW", "STALE_SNAPSHOT"
        if not heartbeat_ok:
            return "YELLOW", "HEARTBEAT_GAP_EXCEEDED"
        if snapshot_health == "YELLOW":
            return "YELLOW", "ORCHESTRATOR_HEALTH_YELLOW"
        return "GREEN", None