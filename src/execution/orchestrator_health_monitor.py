from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator


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

    def check(self, current_timestamp_ms: int) -> HealthReport:
        timestamp_ms = int(current_timestamp_ms)
        snapshot = self._orchestrator.orchestrator_snapshot(timestamp_ms)
        last_snapshot_age_ms = max(0, timestamp_ms - int(snapshot.timestamp_ms))

        heartbeat_gap_ms = 0
        if self._last_check_timestamp_ms is not None:
            heartbeat_gap_ms = max(0, timestamp_ms - self._last_check_timestamp_ms)
        heartbeat_ok = self._last_check_timestamp_ms is None or (
            heartbeat_gap_ms <= (self._config.min_heartbeat_interval_ms * 3)
        )

        halt_reason = self._resolve_halt_reason(
            orchestrator_health=snapshot.health,
            last_snapshot_age_ms=last_snapshot_age_ms,
            heartbeat_ok=heartbeat_ok,
        )
        report = HealthReport(
            timestamp_ms=timestamp_ms,
            orchestrator_health=snapshot.health,
            is_safe_to_trade=(halt_reason is None),
            consecutive_release_failures=self._consecutive_release_failures,
            last_snapshot_age_ms=last_snapshot_age_ms,
            heartbeat_ok=heartbeat_ok,
            halt_reason=halt_reason,
        )
        self._last_check_timestamp_ms = timestamp_ms
        return report

    def is_safe_to_trade(self, current_timestamp_ms: int) -> bool:
        """
        Conservative gate. Returns True only when health is GREEN
        and no position release failures have accumulated beyond threshold.
        bot.py main loop calls this before processing any signal.
        """
        return self.check(current_timestamp_ms).is_safe_to_trade

    def record_position_release_failure(self) -> None:
        self._consecutive_release_failures += 1

    def reset_release_failure_count(self) -> None:
        self._consecutive_release_failures = 0

    def _resolve_halt_reason(
        self,
        *,
        orchestrator_health: Literal["GREEN", "YELLOW", "RED"],
        last_snapshot_age_ms: int,
        heartbeat_ok: bool,
    ) -> str | None:
        if orchestrator_health == "RED":
            return "ORCHESTRATOR_HEALTH_RED"
        if self._consecutive_release_failures >= self._config.max_release_failures_before_halt:
            return "RELEASE_FAILURE_THRESHOLD_REACHED"
        if last_snapshot_age_ms > self._config.stale_snapshot_threshold_ms:
            return "STALE_SNAPSHOT"
        if not heartbeat_ok:
            return "HEARTBEAT_GAP_EXCEEDED"
        if orchestrator_health != "GREEN":
            return f"ORCHESTRATOR_HEALTH_{orchestrator_health}"
        return None