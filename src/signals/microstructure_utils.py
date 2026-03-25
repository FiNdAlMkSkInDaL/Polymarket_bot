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