from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.core.config import settings


def snapshot_timestamp(snapshot: Any) -> float:
    """Return the best available timestamp for a book-like snapshot."""
    local_time = float(getattr(snapshot, "timestamp", 0.0) or 0.0)
    if local_time > 0.0:
        return local_time
    return float(getattr(snapshot, "server_time", 0.0) or 0.0)


@dataclass(frozen=True, slots=True)
class CausalLagConfig:
    max_leader_age_ms: float
    max_lagger_age_ms: float
    max_causal_lag_ms: float
    allow_negative_lag: bool = False


@dataclass(frozen=True, slots=True)
class CausalLagAssessment:
    is_valid: bool
    leader_timestamp: float
    lagger_timestamp: float
    reference_timestamp: float
    leader_age_ms: float
    lagger_age_ms: float
    causal_lag_ms: float
    gate_result: str


@dataclass(frozen=True, slots=True)
class CrossBookSyncAssessment:
    is_synchronized: bool
    latest_timestamp: float
    delta_ms: float
    book_count: int


class CrossBookSyncGate:
    """O(1) max-min timestamp divergence check across related books."""

    def __init__(self, max_desync_ms: float | None = None) -> None:
        threshold = (
            settings.strategy.max_cross_book_desync_ms
            if max_desync_ms is None
            else max_desync_ms
        )
        self._max_desync_ms = max(0.0, float(threshold))

    @property
    def max_desync_ms(self) -> float:
        return self._max_desync_ms

    def assess(self, snapshots: Iterable[Any]) -> CrossBookSyncAssessment:
        timestamps: list[float] = []
        for snapshot in snapshots:
            timestamp = snapshot_timestamp(snapshot)
            if timestamp <= 0.0:
                return CrossBookSyncAssessment(
                    is_synchronized=False,
                    latest_timestamp=0.0,
                    delta_ms=float("inf"),
                    book_count=len(timestamps) + 1,
                )
            timestamps.append(timestamp)

        if not timestamps:
            return CrossBookSyncAssessment(
                is_synchronized=False,
                latest_timestamp=0.0,
                delta_ms=float("inf"),
                book_count=0,
            )

        latest_timestamp = max(timestamps)
        earliest_timestamp = min(timestamps)
        delta_ms = (latest_timestamp - earliest_timestamp) * 1000.0
        return CrossBookSyncAssessment(
            is_synchronized=delta_ms <= (self._max_desync_ms + 1e-9),
            latest_timestamp=latest_timestamp,
            delta_ms=delta_ms,
            book_count=len(timestamps),
        )


class CausalLagGate:
    """Asymmetric freshness gate for leader -> lagger propagation checks."""

    _EPSILON_MS = 1e-9

    def __init__(
        self,
        config: CausalLagConfig | None = None,
        *,
        max_leader_age_ms: float | None = None,
        max_lagger_age_ms: float | None = None,
        max_causal_lag_ms: float | None = None,
        allow_negative_lag: bool | None = None,
    ) -> None:
        if config is None:
            strat = settings.strategy
            config = CausalLagConfig(
                max_leader_age_ms=(
                    strat.contagion_arb_max_leader_age_ms
                    if max_leader_age_ms is None
                    else max_leader_age_ms
                ),
                max_lagger_age_ms=(
                    strat.contagion_arb_max_lagger_age_ms
                    if max_lagger_age_ms is None
                    else max_lagger_age_ms
                ),
                max_causal_lag_ms=(
                    strat.contagion_arb_max_causal_lag_ms
                    if max_causal_lag_ms is None
                    else max_causal_lag_ms
                ),
                allow_negative_lag=(
                    strat.contagion_arb_allow_negative_lag
                    if allow_negative_lag is None
                    else allow_negative_lag
                ),
            )
        self._config = CausalLagConfig(
            max_leader_age_ms=max(0.0, float(config.max_leader_age_ms)),
            max_lagger_age_ms=max(0.0, float(config.max_lagger_age_ms)),
            max_causal_lag_ms=max(0.0, float(config.max_causal_lag_ms)),
            allow_negative_lag=bool(config.allow_negative_lag),
        )

    @property
    def config(self) -> CausalLagConfig:
        return self._config

    def assess(
        self,
        leader_snapshot: Any,
        lagger_snapshot: Any,
        *,
        reference_timestamp: float | None = None,
    ) -> CausalLagAssessment:
        leader_timestamp = snapshot_timestamp(leader_snapshot)
        lagger_timestamp = snapshot_timestamp(lagger_snapshot)
        if leader_timestamp <= 0.0 or lagger_timestamp <= 0.0:
            return CausalLagAssessment(
                is_valid=False,
                leader_timestamp=leader_timestamp,
                lagger_timestamp=lagger_timestamp,
                reference_timestamp=0.0,
                leader_age_ms=float("inf"),
                lagger_age_ms=float("inf"),
                causal_lag_ms=float("inf"),
                gate_result="missing_timestamp",
            )

        effective_reference = max(
            float(reference_timestamp or 0.0),
            leader_timestamp,
            lagger_timestamp,
        )
        leader_age_ms = max(0.0, (effective_reference - leader_timestamp) * 1000.0)
        lagger_age_ms = max(0.0, (effective_reference - lagger_timestamp) * 1000.0)
        causal_lag_ms = (leader_timestamp - lagger_timestamp) * 1000.0

        if leader_age_ms > self._config.max_leader_age_ms + self._EPSILON_MS:
            gate_result = "leader_stale"
        elif lagger_age_ms > self._config.max_lagger_age_ms + self._EPSILON_MS:
            gate_result = "lagger_stale"
        elif not self._config.allow_negative_lag and causal_lag_ms < -self._EPSILON_MS:
            gate_result = "lagger_newer_than_leader"
        else:
            effective_lag_ms = (
                abs(causal_lag_ms)
                if self._config.allow_negative_lag
                else max(0.0, causal_lag_ms)
            )
            if effective_lag_ms > self._config.max_causal_lag_ms + self._EPSILON_MS:
                gate_result = "causal_lag_too_large"
            else:
                gate_result = "accepted"

        return CausalLagAssessment(
            is_valid=gate_result == "accepted",
            leader_timestamp=leader_timestamp,
            lagger_timestamp=lagger_timestamp,
            reference_timestamp=effective_reference,
            leader_age_ms=leader_age_ms,
            lagger_age_ms=lagger_age_ms,
            causal_lag_ms=causal_lag_ms,
            gate_result=gate_result,
        )