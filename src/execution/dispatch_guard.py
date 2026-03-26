from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.priority_context import PriorityOrderContext


_CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]
_GuardReason = Literal["OK", "CIRCUIT_OPEN", "DUPLICATE", "RATE_EXCEEDED", "POSITION_CAP"]


@dataclass(frozen=True, slots=True)
class GuardDecision:
    allowed: bool
    reason: _GuardReason


class DispatchGuard:
    def __init__(self, config: DispatchGuardConfig):
        self._config = config
        self._circuit_state: _CircuitState = "CLOSED"
        self._consecutive_suppressions = 0
        self._circuit_opened_at_ms: int | None = None
        self._half_open_probe_in_flight = False
        self._last_checked_timestamp_ms: int | None = None
        self._dedup_last_seen: dict[tuple[str, str, str], int] = {}
        self._dedup_eviction_queue: deque[tuple[int, tuple[str, str, str]]] = deque()
        self._per_source_dispatch_timestamps: dict[str, deque[int]] = {}
        self._open_position_counts: dict[str, int] = {}

    @property
    def config(self) -> DispatchGuardConfig:
        return self._config

    def check(
        self,
        context: PriorityOrderContext,
        current_timestamp_ms: int,
    ) -> GuardDecision:
        self._last_checked_timestamp_ms = int(current_timestamp_ms)
        self._evict_expired_entries(self._last_checked_timestamp_ms)

        if self._circuit_state == "OPEN":
            assert self._circuit_opened_at_ms is not None
            elapsed = self._last_checked_timestamp_ms - self._circuit_opened_at_ms
            if elapsed < self._config.circuit_breaker_reset_ms:
                return GuardDecision(allowed=False, reason="CIRCUIT_OPEN")
            self._circuit_state = "HALF_OPEN"
            self._half_open_probe_in_flight = False

        if self._circuit_state == "HALF_OPEN":
            if self._half_open_probe_in_flight:
                return GuardDecision(allowed=False, reason="CIRCUIT_OPEN")
            self._half_open_probe_in_flight = True
            return GuardDecision(allowed=True, reason="OK")

        dedup_key = self._dedup_key(context)
        last_seen_ms = self._dedup_last_seen.get(dedup_key)
        if (
            last_seen_ms is not None
            and self._last_checked_timestamp_ms - last_seen_ms < self._config.dedup_window_ms
        ):
            return GuardDecision(allowed=False, reason="DUPLICATE")

        source_timestamps = self._per_source_dispatch_timestamps.get(context.signal_source)
        if source_timestamps is not None:
            if len(source_timestamps) >= self._config.max_dispatches_per_source_per_window:
                return GuardDecision(allowed=False, reason="RATE_EXCEEDED")

        if self._open_position_counts.get(context.market_id, 0) >= self._config.max_open_positions_per_market:
            return GuardDecision(allowed=False, reason="POSITION_CAP")

        return GuardDecision(allowed=True, reason="OK")

    def record_dispatch(
        self,
        context: PriorityOrderContext,
        current_timestamp_ms: int,
    ) -> None:
        timestamp_ms = int(current_timestamp_ms)
        self._last_checked_timestamp_ms = timestamp_ms
        self._evict_expired_entries(timestamp_ms)

        dedup_key = self._dedup_key(context)
        self._dedup_last_seen[dedup_key] = timestamp_ms
        self._dedup_eviction_queue.append((timestamp_ms, dedup_key))

        source_timestamps = self._per_source_dispatch_timestamps.setdefault(
            context.signal_source,
            deque(),
        )
        source_timestamps.append(timestamp_ms)
        self._open_position_counts[context.market_id] = self._open_position_counts.get(context.market_id, 0) + 1

        self._consecutive_suppressions = 0
        if self._circuit_state == "HALF_OPEN":
            self._circuit_state = "CLOSED"
        self._circuit_opened_at_ms = None
        self._half_open_probe_in_flight = False

    def record_suppression(self, signal_source: str) -> None:
        _ = signal_source
        self._consecutive_suppressions += 1
        if self._circuit_state == "HALF_OPEN":
            self._trip_circuit(self._last_checked_timestamp_ms)
            return
        if self._circuit_state == "CLOSED" and self._consecutive_suppressions >= self._config.circuit_breaker_threshold:
            self._trip_circuit(self._last_checked_timestamp_ms)

    def reset_circuit(self) -> None:
        self._circuit_state = "CLOSED"
        self._consecutive_suppressions = 0
        self._circuit_opened_at_ms = None
        self._half_open_probe_in_flight = False

    def guard_snapshot(self, current_timestamp_ms: int | None = None) -> dict:
        if current_timestamp_ms is not None:
            self._last_checked_timestamp_ms = int(current_timestamp_ms)
        if self._last_checked_timestamp_ms is not None:
            self._evict_expired_entries(self._last_checked_timestamp_ms)
        return {
            "circuit_state": self._circuit_state,
            "consecutive_suppressions": self._consecutive_suppressions,
            "circuit_opened_at_ms": self._circuit_opened_at_ms,
            "active_dedup_keys": len(self._dedup_last_seen),
            "per_source_dispatch_counts": {
                source: len(timestamps)
                for source, timestamps in self._per_source_dispatch_timestamps.items()
            },
            "open_position_counts": dict(self._open_position_counts),
        }

    def _dedup_key(self, context: PriorityOrderContext) -> tuple[str, str, str]:
        return (context.market_id, context.side, context.signal_source)

    def _evict_expired_entries(self, current_timestamp_ms: int) -> None:
        dedup_cutoff = current_timestamp_ms - self._config.dedup_window_ms
        while self._dedup_eviction_queue and self._dedup_eviction_queue[0][0] <= dedup_cutoff:
            recorded_at_ms, dedup_key = self._dedup_eviction_queue.popleft()
            if self._dedup_last_seen.get(dedup_key) == recorded_at_ms:
                self._dedup_last_seen.pop(dedup_key, None)

        rate_cutoff = current_timestamp_ms - self._config.rate_window_ms
        for source in tuple(self._per_source_dispatch_timestamps.keys()):
            timestamps = self._per_source_dispatch_timestamps[source]
            while timestamps and timestamps[0] <= rate_cutoff:
                timestamps.popleft()
            if not timestamps:
                self._per_source_dispatch_timestamps.pop(source, None)

    def _trip_circuit(self, current_timestamp_ms: int | None) -> None:
        self._circuit_state = "OPEN"
        self._circuit_opened_at_ms = 0 if current_timestamp_ms is None else int(current_timestamp_ms)
        self._half_open_probe_in_flight = False