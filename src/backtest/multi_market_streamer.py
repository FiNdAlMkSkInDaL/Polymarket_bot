from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class TimestampedMarketRecord:
    timestamp_ms: int
    market_id: str
    record: dict[str, Any]


def iter_multiplexed_market_records(
    primary_path: Path,
    secondary_path: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    heap: list[tuple[int, int, int, TimestampedMarketRecord, Iterator[TimestampedMarketRecord]]] = []
    for source_index, path in enumerate((primary_path, secondary_path)):
        iterator = _iter_market_records(path)
        first_record = next(iterator, None)
        if first_record is not None:
            heapq.heappush(heap, (first_record.timestamp_ms, source_index, 0, first_record, iterator))

    emission_index = 0
    while heap:
        _, source_index, _, record, iterator = heapq.heappop(heap)
        yield record.market_id, record.record
        emission_index += 1
        next_record = next(iterator, None)
        if next_record is not None:
            heapq.heappush(heap, (next_record.timestamp_ms, source_index, emission_index, next_record, iterator))


def _iter_market_records(path: Path) -> Iterator[TimestampedMarketRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            payload = raw.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            timestamp_ms = _timestamp_ms_from_raw_record(raw, payload)
            if timestamp_ms <= 0:
                continue
            market_id = str(payload.get("market") or raw.get("asset_id") or "").strip()
            if not market_id:
                continue
            yield TimestampedMarketRecord(timestamp_ms=timestamp_ms, market_id=market_id, record=raw)


def _timestamp_ms_from_raw_record(raw: dict[str, Any], payload: dict[str, Any]) -> int:
    for value in (payload.get("timestamp"), raw.get("local_ts")):
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 1e12:
            return int(numeric)
        return int(numeric * 1000)
    return 0