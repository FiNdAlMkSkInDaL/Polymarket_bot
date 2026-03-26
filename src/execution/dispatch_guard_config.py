from __future__ import annotations

from dataclasses import dataclass


def _require_positive_int(field_name: str, value: int) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int")
    if value <= 0:
        raise ValueError(f"{field_name} must be strictly positive; got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class DispatchGuardConfig:
    dedup_window_ms: int
    max_dispatches_per_source_per_window: int
    rate_window_ms: int
    circuit_breaker_threshold: int
    circuit_breaker_reset_ms: int
    max_open_positions_per_market: int

    def __post_init__(self) -> None:
        for field_name in (
            "dedup_window_ms",
            "max_dispatches_per_source_per_window",
            "rate_window_ms",
            "circuit_breaker_threshold",
            "circuit_breaker_reset_ms",
            "max_open_positions_per_market",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        if self.circuit_breaker_threshold < 2:
            raise ValueError(
                "circuit_breaker_threshold must be >= 2; a single suppression must not open the circuit"
            )
        if self.max_open_positions_per_market < 1:
            raise ValueError("max_open_positions_per_market must be >= 1")