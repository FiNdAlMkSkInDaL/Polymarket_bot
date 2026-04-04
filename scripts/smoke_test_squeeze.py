#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

try:
    import psutil
except ImportError:  # pragma: no cover - fallback path only
    psutil = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_conditional_probability_squeeze_batch import (
    SqueezeBatchResult,
    render_markdown,
    run_squeeze_batch,
    write_batch_outputs,
)
from src.backtest.conditional_probability_squeeze import ConditionalProbabilitySqueezeConfig


PARQUET_LAKE_ROOT = PROJECT_ROOT / "data" / "parquet_lake" / "squeeze_smoke"
PAIRS_CONFIG_PATH = PROJECT_ROOT / "config" / "squeeze_pairs.smoke.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "squeeze_diagnostics" / "smoke_test"
SMOKE_START_DATE = date(2026, 4, 1)
SMOKE_END_DATE = date(2026, 4, 3)
EXPECTED_MEMORY_LIMIT_MB = 750.0


@dataclass(frozen=True, slots=True)
class SmokeReport:
    input_root: str
    pairs_config: str
    output_dir: str
    start_date: str
    end_date: str
    peak_memory_mb: float
    expected_limit_mb: float
    measurement_method: str
    batch_summary: dict[str, object]
    ranking_preview: list[dict[str, object]]


class PeakMemoryMonitor:
    def __init__(self, sample_interval_seconds: float = 0.05) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def start(self) -> None:
        if psutil is None:
            return
        self._thread = threading.Thread(target=self._run, name="squeeze-smoke-rss", daemon=True)
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


def _timestamp(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    millisecond: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, millisecond * 1000, tzinfo=timezone.utc)


def _row(
    timestamp: datetime,
    market_id: str,
    *,
    best_bid: float,
    best_ask: float,
    bid_depth: float,
    ask_depth: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "event_id": f"event-{market_id}",
        "token_id": f"token-{market_id}",
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
    }


def _lake_has_any_parquet(input_root: Path) -> bool:
    return any(input_root.rglob("*.parquet"))


def _bootstrap_synthetic_lake(input_root: Path) -> None:
    if _lake_has_any_parquet(input_root):
        return

    fixture_rows: dict[str, list[dict[str, object]]] = {
        "2026-04-01": [
            _row(_timestamp(2026, 4, 1, 8, 0, 0), "smoke-parent-alpha", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 0), "smoke-child-alpha", best_bid=0.60, best_ask=0.61, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 0, 100), "smoke-parent-alpha", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 0, 100), "smoke-child-alpha", best_bid=0.60, best_ask=0.61, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 1), "smoke-parent-alpha", best_bid=0.65, best_ask=0.66, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 1), "smoke-child-alpha", best_bid=0.58, best_ask=0.59, bid_depth=500.0, ask_depth=500.0),
        ],
        "2026-04-02": [
            _row(_timestamp(2026, 4, 2, 9, 0, 0), "smoke-parent-beta", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 0), "smoke-child-beta", best_bid=0.60, best_ask=0.61, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 0, 100), "smoke-parent-beta", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 0, 100), "smoke-child-beta", best_bid=0.60, best_ask=0.61, bid_depth=10.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 1), "smoke-parent-beta", best_bid=0.55, best_ask=0.56, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 1), "smoke-child-beta", best_bid=0.40, best_ask=0.41, bid_depth=500.0, ask_depth=500.0),
        ],
        "2026-04-03": [
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "smoke-parent-alpha", best_bid=0.70, best_ask=0.71, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "smoke-child-alpha", best_bid=0.20, best_ask=0.21, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "smoke-parent-beta", best_bid=0.70, best_ask=0.71, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "smoke-child-beta", best_bid=0.20, best_ask=0.21, bid_depth=500.0, ask_depth=500.0),
        ],
    }

    for day_str, rows in fixture_rows.items():
        day_dir = input_root / day_str
        day_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(day_dir / "smoke.parquet")


def _bootstrap_pairs_config(config_path: Path) -> None:
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "smoke-alpha",
                        "parent_market_id": "smoke-parent-alpha",
                        "child_market_id": "smoke-child-alpha",
                        "notes": "Profitable dummy squeeze pair for smoke validation.",
                    },
                    {
                        "pair_id": "smoke-beta",
                        "parent_market_id": "smoke-parent-beta",
                        "child_market_id": "smoke-child-beta",
                        "notes": "Toxic dummy squeeze pair with a forced orphan flatten.",
                    },
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _smoke_config() -> ConditionalProbabilitySqueezeConfig:
    return ConditionalProbabilitySqueezeConfig(
        order_size=100.0,
        entry_gap_threshold=0.025,
        entry_zscore_threshold=10.0,
        minimum_theoretical_edge_dollars=0.0,
        exit_gap_threshold=0.05,
        exit_zscore_threshold=10.0,
        z_window_events=2,
        timestamp_unit="auto",
        route_latency_ms=100,
        max_quote_age_ms=1_000,
        max_hold_ms=5_000,
        process_by_day=True,
        chunk_days=1,
        warmup_days=1,
        collect_engine="streaming",
    )


def _preview_ranking(result: SqueezeBatchResult) -> pl.DataFrame:
    return result.ranking.select(
        [
            "pair_id",
            "total_valid_signals_generated",
            "fok_survival_rate_at_route_latency",
            "successful_fok_net_pnl",
            "flattened_basket_net_loss",
            "ranking_net_pnl",
            "status",
        ]
    ).head(10)


def run_smoke_test(
    *,
    input_root: Path = PARQUET_LAKE_ROOT,
    pairs_config_path: Path = PAIRS_CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    start_date: date = SMOKE_START_DATE,
    end_date: date = SMOKE_END_DATE,
) -> SmokeReport:
    _bootstrap_synthetic_lake(input_root)
    _bootstrap_pairs_config(pairs_config_path)

    monitor = PeakMemoryMonitor()
    config = _smoke_config()
    tracemalloc.start()
    monitor.start()
    try:
        result = run_squeeze_batch(
            input_root,
            pairs_config_path,
            config,
            start_date=start_date,
            end_date=end_date,
        )
        write_batch_outputs(result, output_dir, input_root=input_root, pairs_config_path=pairs_config_path)
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

    preview = _preview_ranking(result)
    print(render_markdown(result, input_root, pairs_config_path))
    print()
    print("Aggregated Diagnostics Preview:")
    print(preview)
    print()
    print(
        f"Smoke Test Complete. Peak Memory Usage: {peak_memory_mb:.2f} MB. "
        f"Expected limit: {EXPECTED_MEMORY_LIMIT_MB:.0f} MB."
    )

    return SmokeReport(
        input_root=str(input_root),
        pairs_config=str(pairs_config_path),
        output_dir=str(output_dir),
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        peak_memory_mb=peak_memory_mb,
        expected_limit_mb=EXPECTED_MEMORY_LIMIT_MB,
        measurement_method=measurement_method,
        batch_summary=result.summary,
        ranking_preview=preview.to_dicts(),
    )


def main() -> None:
    report = run_smoke_test()
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()