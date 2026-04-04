#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
import tracemalloc
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.reduce_mid_tier_probability_compression_sweep import (
    render_pareto_frontier_markdown,
    run_reducer,
)
from scripts.sweep_mid_tier_probability_compression import build_threshold_grid, discover_daily_partitions, run_sweep


PARQUET_LAKE_ROOT = PROJECT_ROOT / "data" / "parquet_lake"
EXPECTED_MEMORY_LIMIT_MB = 750.0
SMOKE_DAY_COUNT = 3
SMOKE_THRESHOLDS = build_threshold_grid(0.85, 0.98, 0.01)
SMOKE_NOTIONALS = (10.0, 25.0, 50.0)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_smoke"


@dataclass(slots=True)
class SmokeTestArtifacts:
    parquet_root: str
    selected_dates: tuple[str, ...]
    daily_output: str
    reduced_output: str
    summary_output: str
    markdown_output: str
    tracemalloc_peak_mb: float
    resource_peak_mb: float | None
    rss_peak_mb: float | None
    peak_memory_mb: float
    expected_limit_mb: float
    within_expected_memory_limit: bool
    sweep_summary: dict[str, Any]
    reducer_summary: dict[str, Any]
    pareto_frontier_markdown: str


def _resource_peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


class _PeakRssSampler:
    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_mb = 0.0
        try:
            import psutil
        except ImportError:
            self._process = None
        else:
            self._process = psutil.Process()

    def _sample_once(self) -> None:
        if self._process is None:
            return
        try:
            rss_mb = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return
        self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample_once()

    def start(self) -> None:
        if self._process is None:
            return
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._process is None:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 4.0))
        self._sample_once()
        return self._peak_rss_mb


def _measure_peak_memory_mb(rss_peak_mb: float | None) -> tuple[float, float | None, float | None, float]:
    _, traced_peak_bytes = tracemalloc.get_traced_memory()
    traced_peak_mb = float(traced_peak_bytes) / (1024.0 * 1024.0)
    resource_peak_mb = _resource_peak_memory_mb()
    candidates = [traced_peak_mb]
    if resource_peak_mb is not None:
        candidates.append(resource_peak_mb)
    if rss_peak_mb is not None:
        candidates.append(rss_peak_mb)
    peak_memory_mb = max(candidates)
    return traced_peak_mb, resource_peak_mb, rss_peak_mb, peak_memory_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Mid-Tier Probability Compression smoke test over a three-day parquet root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PARQUET_LAKE_ROOT,
        help="Date-partitioned Parquet root to smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for daily, reducer, summary, and Pareto artifacts.",
    )
    parser.add_argument(
        "--day-count",
        type=int,
        default=SMOKE_DAY_COUNT,
        help="Number of earliest daily partitions to include in the smoke window.",
    )
    return parser.parse_args()


def _select_smoke_dates(parquet_root: Path, day_count: int) -> tuple[str, ...]:
    partitions = discover_daily_partitions(parquet_root)
    if len(partitions) < day_count:
        raise ValueError(f"Smoke test requires at least {day_count} daily partitions under {parquet_root}")
    return tuple(trade_date for trade_date, _ in partitions[:day_count])


def run_smoke_test(
    parquet_root: Path = PARQUET_LAKE_ROOT,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    day_count: int = SMOKE_DAY_COUNT,
    thresholds: tuple[float, ...] = SMOKE_THRESHOLDS,
    notionals: tuple[float, ...] = SMOKE_NOTIONALS,
) -> SmokeTestArtifacts:
    selected_dates = _select_smoke_dates(parquet_root, day_count)
    start_date = selected_dates[0]
    end_date = selected_dates[-1]

    daily_output = output_dir / "daily_panel.parquet"
    reduced_output = output_dir / "reducer_rankings.parquet"
    summary_output = output_dir / "smoke_test_summary.json"
    markdown_output = output_dir / "pareto_frontier.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    rss_sampler = _PeakRssSampler()
    tracemalloc.start()
    rss_sampler.start()
    try:
        sweep_artifacts = run_sweep(
            parquet_root,
            thresholds=thresholds,
            notionals=notionals,
            start_date=start_date,
            end_date=end_date,
        )
        sweep_artifacts.daily_rows.write_parquet(daily_output, compression="zstd")

        reducer_artifacts = run_reducer(daily_output)
        reducer_artifacts.rankings.write_parquet(reduced_output, compression="zstd")
    finally:
        rss_peak_mb = rss_sampler.stop()
        tracemalloc_peak_mb, resource_peak_mb, rss_peak_mb, peak_memory_mb = _measure_peak_memory_mb(
            rss_peak_mb
        )
        tracemalloc.stop()

    pareto_markdown = render_pareto_frontier_markdown(
        reducer_artifacts.pareto_frontier,
        title=f"Pareto Frontier ({start_date} to {end_date})",
    )
    memory_bits = [f"tracemalloc: {tracemalloc_peak_mb:.2f} MB"]
    if rss_peak_mb is not None:
        memory_bits.append(f"rss: {rss_peak_mb:.2f} MB")
    if resource_peak_mb is not None:
        memory_bits.append(f"resource: {resource_peak_mb:.2f} MB")
    report = (
        f"Smoke Test Complete. Peak Memory Usage: {peak_memory_mb:.2f} MB "
        f"({'; '.join(memory_bits)}). Expected limit: 750 MB."
    )

    artifacts = SmokeTestArtifacts(
        parquet_root=str(parquet_root),
        selected_dates=selected_dates,
        daily_output=str(daily_output),
        reduced_output=str(reduced_output),
        summary_output=str(summary_output),
        markdown_output=str(markdown_output),
        tracemalloc_peak_mb=tracemalloc_peak_mb,
        resource_peak_mb=resource_peak_mb,
        rss_peak_mb=rss_peak_mb,
        peak_memory_mb=peak_memory_mb,
        expected_limit_mb=EXPECTED_MEMORY_LIMIT_MB,
        within_expected_memory_limit=peak_memory_mb <= EXPECTED_MEMORY_LIMIT_MB,
        sweep_summary=sweep_artifacts.summary,
        reducer_summary=reducer_artifacts.summary,
        pareto_frontier_markdown=pareto_markdown,
    )

    markdown_output.write_text(report + "\n\n" + pareto_markdown, encoding="utf-8")
    summary_output.write_text(json.dumps(asdict(artifacts), indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def main() -> int:
    args = parse_args()
    artifacts = run_smoke_test(
        args.input_root,
        output_dir=args.output_dir,
        day_count=args.day_count,
    )
    print(
        "Smoke Test Complete. Peak Memory Usage: "
        f"{artifacts.peak_memory_mb:.2f} MB "
        f"(tracemalloc: {artifacts.tracemalloc_peak_mb:.2f} MB"
        + (
            f"; rss: {artifacts.rss_peak_mb:.2f} MB"
            if artifacts.rss_peak_mb is not None
            else ""
        )
        + (
            f"; resource: {artifacts.resource_peak_mb:.2f} MB)"
            if artifacts.resource_peak_mb is not None
            else ")"
        )
        + ". Expected limit: 750 MB."
    )
    print()
    print(artifacts.pareto_frontier_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())