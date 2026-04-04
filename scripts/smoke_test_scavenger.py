#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

try:
    import psutil
except ImportError:  # pragma: no cover - fallback path only
    psutil = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_scavenger_protocol_batch import render_markdown, run_scavenger_batch
from src.backtest.real_scavenger_smoke import prepare_real_scavenger_smoke_lake
from src.backtest.scavenger_protocol import ScavengerConfig


PARQUET_LAKE_ROOT = PROJECT_ROOT / "data" / "parquet_lake"
SMOKE_START_DATE = date(2026, 4, 1)
SMOKE_END_DATE = date(2026, 4, 3)
LOOKBACK_DAYS = 1
LOOKAHEAD_DAYS = 1
EXPECTED_MEMORY_LIMIT_MB = 750.0


@dataclass(frozen=True, slots=True)
class SmokeReport:
    data_source: str
    input_root: str
    raw_source_root: str | None
    start_date: str
    end_date: str
    candidate_market_count: int | None
    available_market_count: int | None
    prepared_row_count: int | None
    peak_memory_mb: float
    expected_limit_mb: float
    measurement_method: str
    tearsheet_summary: dict[str, object]


class PeakMemoryMonitor:
    def __init__(self, sample_interval_seconds: float = 0.05) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def start(self) -> None:
        if psutil is None:
            return
        self._thread = threading.Thread(target=self._run, name="scavenger-smoke-rss", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def peak_memory_mb(self) -> float | None:
        if psutil is None:
            return None
        return self._peak_rss_bytes / (1024.0 * 1024.0)

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop_event.is_set():
            self._peak_rss_bytes = max(self._peak_rss_bytes, process.memory_info().rss)
            time.sleep(self.sample_interval_seconds)
        self._peak_rss_bytes = max(self._peak_rss_bytes, process.memory_info().rss)


def _row(
    *,
    timestamp: datetime,
    resolution_timestamp: datetime,
    market_id: str,
    event_id: str,
    best_bid: float,
    best_ask: float,
    final_resolution_value: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "event_id": event_id,
        "token_id": "YES",
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": 100.0,
        "ask_depth": 100.0,
        "resolution_timestamp": resolution_timestamp,
        "final_resolution_value": final_resolution_value,
    }


def _lake_has_any_parquet(input_root: Path) -> bool:
    return any(input_root.rglob("*.parquet"))


def _bootstrap_synthetic_lake(input_root: Path) -> None:
    if _lake_has_any_parquet(input_root):
        return

    input_root.mkdir(parents=True, exist_ok=True)
    day1 = datetime(2026, 4, 1, 0, 0, 0)
    day2 = day1 + timedelta(days=1)
    day3 = day2 + timedelta(days=1)
    day4 = day3 + timedelta(days=1)

    fixture_rows: dict[str, list[dict[str, object]]] = {
        "2026-04-01": [
            _row(
                timestamp=datetime(2026, 4, 1, 9, 0, 0),
                resolution_timestamp=datetime(2026, 4, 1, 18, 0, 0),
                market_id="smoke-short",
                event_id="smoke-short-signal",
                best_bid=0.96,
                best_ask=0.99,
                final_resolution_value=1.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 1, 9, 15, 0),
                resolution_timestamp=datetime(2026, 4, 1, 18, 0, 0),
                market_id="smoke-short",
                event_id="smoke-short-fill",
                best_bid=0.94,
                best_ask=0.96,
                final_resolution_value=1.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 1, 10, 0, 0),
                resolution_timestamp=datetime(2026, 4, 2, 12, 0, 0),
                market_id="smoke-lockup-a",
                event_id="smoke-lockup-a-signal",
                best_bid=0.96,
                best_ask=0.99,
                final_resolution_value=0.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 1, 10, 10, 0),
                resolution_timestamp=datetime(2026, 4, 2, 12, 0, 0),
                market_id="smoke-lockup-a",
                event_id="smoke-lockup-a-fill",
                best_bid=0.94,
                best_ask=0.96,
                final_resolution_value=0.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 1, 10, 0, 0),
                resolution_timestamp=datetime(2026, 4, 3, 12, 0, 0),
                market_id="smoke-lockup-b",
                event_id="smoke-lockup-b-signal",
                best_bid=0.96,
                best_ask=0.99,
                final_resolution_value=1.0,
            ),
        ],
        "2026-04-02": [
            _row(
                timestamp=datetime(2026, 4, 2, 9, 0, 0),
                resolution_timestamp=datetime(2026, 4, 2, 20, 0, 0),
                market_id="smoke-day2",
                event_id="smoke-day2-signal",
                best_bid=0.96,
                best_ask=0.99,
                final_resolution_value=1.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 2, 9, 5, 0),
                resolution_timestamp=datetime(2026, 4, 2, 20, 0, 0),
                market_id="smoke-day2",
                event_id="smoke-day2-fill",
                best_bid=0.95,
                best_ask=0.95,
                final_resolution_value=1.0,
            ),
        ],
        "2026-04-03": [
            _row(
                timestamp=datetime(2026, 4, 3, 11, 0, 0),
                resolution_timestamp=datetime(2026, 4, 4, 12, 0, 0),
                market_id="smoke-day3",
                event_id="smoke-day3-signal",
                best_bid=0.96,
                best_ask=0.99,
                final_resolution_value=1.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 3, 11, 30, 0),
                resolution_timestamp=datetime(2026, 4, 4, 12, 0, 0),
                market_id="smoke-day3",
                event_id="smoke-day3-fill",
                best_bid=0.94,
                best_ask=0.96,
                final_resolution_value=1.0,
            ),
        ],
        "2026-04-04": [
            _row(
                timestamp=datetime(2026, 4, 4, 9, 0, 0),
                resolution_timestamp=datetime(2026, 4, 4, 12, 0, 0),
                market_id="smoke-lookahead",
                event_id="smoke-lookahead-marker",
                best_bid=0.99,
                best_ask=1.0,
                final_resolution_value=1.0,
            )
        ],
    }

    for day_str, rows in fixture_rows.items():
        day_dir = input_root / day_str
        day_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(day_dir / "smoke.parquet")


def _resolve_smoke_input() -> tuple[
    Path,
    date,
    date,
    int,
    int,
    str,
    str | None,
    int | None,
    int | None,
    int | None,
]:
    real_slice = prepare_real_scavenger_smoke_lake(PROJECT_ROOT)
    if real_slice is not None:
        return (
            real_slice.input_root,
            real_slice.start_date,
            real_slice.end_date,
            0,
            0,
            "real_snapshot_smoke",
            str(real_slice.raw_root),
            real_slice.candidate_market_count,
            real_slice.available_market_count,
            real_slice.row_count,
        )

    _bootstrap_synthetic_lake(PARQUET_LAKE_ROOT)
    return (
        PARQUET_LAKE_ROOT,
        SMOKE_START_DATE,
        SMOKE_END_DATE,
        LOOKBACK_DAYS,
        LOOKAHEAD_DAYS,
        "synthetic_fallback",
        None,
        None,
        None,
        None,
    )


def run_smoke_test() -> SmokeReport:
    (
        input_root,
        start_date,
        end_date,
        lookback_days,
        lookahead_days,
        data_source,
        raw_source_root,
        candidate_market_count,
        available_market_count,
        prepared_row_count,
    ) = _resolve_smoke_input()
    config = ScavengerConfig()
    monitor = PeakMemoryMonitor()

    tracemalloc.start()
    monitor.start()
    try:
        result = run_scavenger_batch(
            input_root,
            config,
            start_date=start_date,
            end_date=end_date,
            window_lookback_days=lookback_days,
            window_lookahead_days=lookahead_days,
        )
        tracemalloc_peak_mb = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
    finally:
        monitor.stop()
        tracemalloc.stop()

    rss_peak_mb = monitor.peak_memory_mb()
    if rss_peak_mb is not None:
        peak_memory_mb = rss_peak_mb
        measurement_method = "psutil_rss"
    else:
        peak_memory_mb = tracemalloc_peak_mb
        measurement_method = "tracemalloc_peak"

    print(f"Data Source: {data_source}")
    print(f"Input Root: {input_root}")
    if raw_source_root is not None:
        print(f"Raw Source Root: {raw_source_root}")
        print(
            "Real Slice Coverage: "
            f"{candidate_market_count} candidate markets within the 72-hour window out of "
            f"{available_market_count} raw markets; prepared rows={prepared_row_count}."
        )
    print()
    print(render_markdown(result.tearsheet, result.summary, input_root))
    print()
    print(
        f"Smoke Test Complete. Peak Memory Usage: {peak_memory_mb:.2f} MB. "
        f"Expected limit: {EXPECTED_MEMORY_LIMIT_MB:.0f} MB."
    )

    return SmokeReport(
        data_source=data_source,
        input_root=str(input_root),
        raw_source_root=raw_source_root,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        candidate_market_count=candidate_market_count,
        available_market_count=available_market_count,
        prepared_row_count=prepared_row_count,
        peak_memory_mb=peak_memory_mb,
        expected_limit_mb=EXPECTED_MEMORY_LIMIT_MB,
        measurement_method=measurement_method,
        tearsheet_summary=result.summary,
    )


def main() -> None:
    report = run_smoke_test()
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()