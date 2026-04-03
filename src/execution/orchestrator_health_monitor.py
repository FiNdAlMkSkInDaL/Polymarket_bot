from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.core.logger import get_logger
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator
from src.execution.priority_context import PriorityOrderContext
from src.execution.volatility_monitor import RollingMidPriceVolatilityMonitor, VolatilityStatus


log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HealthMonitorConfig:
    max_release_failures_before_halt: int
    stale_snapshot_threshold_ms: int
    min_heartbeat_interval_ms: int
    volatility_window_ms: int = 300_000
    max_safe_volatility_cents: float = 0.0

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.max_release_failures_before_halt, int) or self.max_release_failures_before_halt < 1:
            errors.append("max_release_failures_before_halt must be >= 1")
        if not isinstance(self.stale_snapshot_threshold_ms, int) or self.stale_snapshot_threshold_ms <= 0:
            errors.append("stale_snapshot_threshold_ms must be a strictly positive int")
        if not isinstance(self.min_heartbeat_interval_ms, int) or self.min_heartbeat_interval_ms <= 0:
            errors.append("min_heartbeat_interval_ms must be a strictly positive int")
        if not isinstance(self.volatility_window_ms, int) or self.volatility_window_ms <= 0:
            errors.append("volatility_window_ms must be a strictly positive int")
        if self.max_safe_volatility_cents < 0:
            errors.append("max_safe_volatility_cents must be >= 0")
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
    volatility_asset_id: str | None = None
    rolling_mid_price_volatility_cents: float | None = None
    volatility_sample_count: int = 0
    volatility_threshold_breached: bool = False


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
        self._last_reported_health: tuple[str, str | None] | None = None
        self._volatility_monitor = RollingMidPriceVolatilityMonitor(
            window_ms=self._config.volatility_window_ms,
            max_safe_volatility_cents=self._config.max_safe_volatility_cents,
        )
        dispatcher = getattr(self._orchestrator, "dispatcher", None)
        bind_gate = getattr(dispatcher, "bind_pre_dispatch_gate", None)
        if callable(bind_gate):
            bind_gate(self)

    def record_mid_price(self, asset_id: str, mid_price: float, timestamp_ms: int) -> None:
        self._volatility_monitor.record_mid_price(asset_id, mid_price, int(timestamp_ms))

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
        volatility_status = self._volatility_monitor.current_status(timestamp_ms)

        effective_health, halt_reason = self._resolve_health_state(
            snapshot_health=snapshot.health,
            last_snapshot_age_ms=last_snapshot_age_ms,
            heartbeat_ok=heartbeat_ok,
            volatility_status=volatility_status,
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
            volatility_asset_id=volatility_status.asset_id,
            rolling_mid_price_volatility_cents=volatility_status.sigma_cents,
            volatility_sample_count=volatility_status.sample_count,
            volatility_threshold_breached=volatility_status.is_breached,
        )
        health_signature = (report.orchestrator_health, report.halt_reason)
        if health_signature != self._last_reported_health:
            log.info(
                "orchestrator_health_status",
                health=report.orchestrator_health,
                is_safe_to_trade=report.is_safe_to_trade,
                halt_reason=report.halt_reason,
                last_snapshot_age_ms=report.last_snapshot_age_ms,
                heartbeat_ok=report.heartbeat_ok,
                consecutive_release_failures=report.consecutive_release_failures,
                volatility_asset_id=report.volatility_asset_id,
                rolling_mid_price_volatility_cents=report.rolling_mid_price_volatility_cents,
                volatility_threshold_breached=report.volatility_threshold_breached,
            )
            self._last_reported_health = health_signature
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
        volatility_status: VolatilityStatus,
    ) -> tuple[Literal["GREEN", "YELLOW", "RED"], str | None]:
        if snapshot_health == "RED":
            return "RED", "ORCHESTRATOR_HEALTH_RED"
        if self._consecutive_release_failures >= self._config.max_release_failures_before_halt:
            return "YELLOW", "RELEASE_FAILURE_THRESHOLD_REACHED"
        if last_snapshot_age_ms > self._config.stale_snapshot_threshold_ms:
            return "YELLOW", "STALE_SNAPSHOT"
        if not heartbeat_ok:
            return "YELLOW", "HEARTBEAT_GAP_EXCEEDED"
        if volatility_status.is_breached:
            return "YELLOW", "VOLATILITY_THRESHOLD_EXCEEDED"
        if snapshot_health == "YELLOW":
            return "YELLOW", "ORCHESTRATOR_HEALTH_YELLOW"
        return "GREEN", None