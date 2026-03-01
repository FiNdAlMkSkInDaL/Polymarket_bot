"""
Stale-data kill-switch — gates all execution on WebSocket freshness.

Compares each incoming ``server_timestamp`` against the local VPS clock
(adjusted for a one-time calibration offset).  If lag exceeds the
configured threshold, all order placement is blocked until the
connection stabilises for *N* consecutive healthy checks.
"""

from __future__ import annotations

import time
from enum import Enum

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


class LatencyState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


class LatencyGuard:
    """Pre-flight latency gatekeeper for the execution loop.

    Usage::

        guard = LatencyGuard()
        guard.calibrate(first_server_ts)       # one-time on WS connect

        state = guard.check(event.timestamp)
        if state == LatencyState.BLOCKED:
            continue                           # skip all execution
    """

    def __init__(
        self,
        *,
        block_ms: int | None = None,
        warn_ms: int | None = None,
        recovery_count: int | None = None,
    ):
        strat = settings.strategy
        self._block_ms: int = block_ms if block_ms is not None else strat.latency_block_ms
        self._warn_ms: int = warn_ms if warn_ms is not None else strat.latency_warn_ms
        self._recovery_n: int = recovery_count if recovery_count is not None else strat.latency_recovery_count

        # Clock-skew offset (seconds): server_time - local_time
        self._clock_offset: float = 0.0
        self._calibrated: bool = False

        # State machine
        self._state: LatencyState = LatencyState.HEALTHY
        self._consecutive_healthy: int = 0
        self._last_delta_ms: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def state(self) -> LatencyState:
        return self._state

    @property
    def last_delta_ms(self) -> float:
        """Most recent measured latency in milliseconds."""
        return self._last_delta_ms

    def calibrate(self, server_ts: float) -> None:
        """One-time clock-skew calibration on first WS message.

        Parameters
        ----------
        server_ts:
            Unix epoch seconds from the first server message.
        """
        if self._calibrated:
            return
        self._clock_offset = server_ts - time.time()
        self._calibrated = True
        log.info(
            "latency_guard_calibrated",
            offset_ms=round(self._clock_offset * 1000, 1),
        )

    def check(self, server_ts: float) -> LatencyState:
        """Evaluate latency for an incoming message.

        Parameters
        ----------
        server_ts:
            The ``timestamp`` field from the server message (epoch seconds).

        Returns
        -------
        LatencyState indicating whether execution should proceed.
        """
        if not self._calibrated:
            # Auto-calibrate on first check if not done explicitly.
            self.calibrate(server_ts)

        adjusted_local = time.time() + self._clock_offset
        delta_ms = abs(adjusted_local - server_ts) * 1000.0
        self._last_delta_ms = delta_ms

        prev_state = self._state

        if delta_ms >= self._block_ms:
            self._consecutive_healthy = 0
            self._state = LatencyState.BLOCKED
            if prev_state != LatencyState.BLOCKED:
                log.warning(
                    "latency_BLOCKED",
                    delta_ms=round(delta_ms, 1),
                    threshold_ms=self._block_ms,
                )
        elif delta_ms >= self._warn_ms:
            self._consecutive_healthy = 0
            if self._state == LatencyState.BLOCKED:
                # Still in recovery zone — stay blocked until fully healthy.
                pass
            else:
                self._state = LatencyState.DEGRADED
                if prev_state == LatencyState.HEALTHY:
                    log.info(
                        "latency_degraded",
                        delta_ms=round(delta_ms, 1),
                        threshold_ms=self._warn_ms,
                    )
        else:
            # Healthy tick
            self._consecutive_healthy += 1
            if self._state == LatencyState.BLOCKED:
                if self._consecutive_healthy >= self._recovery_n:
                    self._state = LatencyState.HEALTHY
                    log.info(
                        "latency_recovered",
                        consecutive=self._consecutive_healthy,
                        delta_ms=round(delta_ms, 1),
                    )
                    self._consecutive_healthy = 0
                # else: stay blocked, waiting for more healthy ticks
            else:
                self._state = LatencyState.HEALTHY

        return self._state

    def is_blocked(self) -> bool:
        """Convenience: True when execution must be halted."""
        return self._state == LatencyState.BLOCKED

    def force_block(self, reason: str = "external") -> None:
        """Force the guard into BLOCKED state from an external source.

        Used by the ``BookHeartbeat`` and ``AdverseSelectionGuard`` to
        halt execution without waiting for a latency check.
        """
        if self._state != LatencyState.BLOCKED:
            log.warning("latency_force_blocked", reason=reason)
        self._state = LatencyState.BLOCKED
        self._consecutive_healthy = 0

    def reset(self) -> None:
        """Reset to initial state (e.g. on WS reconnect)."""
        self._state = LatencyState.HEALTHY
        self._consecutive_healthy = 0
        self._calibrated = False
        self._clock_offset = 0.0
