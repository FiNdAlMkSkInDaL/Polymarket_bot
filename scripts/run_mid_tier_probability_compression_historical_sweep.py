#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import sys
import threading
import tracemalloc
from typing import Any, Callable, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.reduce_mid_tier_probability_compression_sweep import (
    render_pareto_frontier_markdown,
    run_reducer,
)
from scripts.sweep_mid_tier_probability_compression import (
    build_threshold_grid,
    parse_notionals,
    render_markdown as render_sweep_markdown,
    run_sweep,
)


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "artifacts" / "l2_parquet_lake_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_historical_full396_run"
DEFAULT_THRESHOLDS = build_threshold_grid(0.80, 0.95, 0.01)
DEFAULT_NOTIONALS = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)
EXPECTED_MEMORY_LIMIT_MB = 750.0


T = TypeVar("T")


@dataclass(slots=True)
class PhaseMemory:
    tracemalloc_peak_mb: float
    resource_peak_mb: float | None
    rss_peak_mb: float | None
    peak_memory_mb: float
    within_expected_memory_limit: bool


@dataclass(slots=True)
class HistoricalSweepArtifacts:
    input_root: str
    output_dir: str
    daily_output: str
    rankings_output: str
    sweep_summary_output: str
    sweep_markdown_output: str
    reducer_summary_output: str
    pareto_markdown_output: str
    execution_summary_output: str
    sweep_memory: PhaseMemory
    reducer_memory: PhaseMemory
    sweep_summary: dict[str, Any]
    reducer_summary: dict[str, Any]
    execution_summary: dict[str, Any]


def _resource_peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _resource_module_available() -> bool:
    try:
        import resource  # noqa: F401
    except ImportError:
        return False
    return True


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


def _measure_phase(callback: Callable[[], T]) -> tuple[T, PhaseMemory]:
    rss_sampler = _PeakRssSampler()
    tracemalloc.start()
    rss_sampler.start()
    try:
        result = callback()
    finally:
        rss_peak_mb = rss_sampler.stop()
        tracemalloc_peak_mb, resource_peak_mb, rss_peak_mb, peak_memory_mb = _measure_peak_memory_mb(rss_peak_mb)
        tracemalloc.stop()

    return result, PhaseMemory(
        tracemalloc_peak_mb=tracemalloc_peak_mb,
        resource_peak_mb=resource_peak_mb,
        rss_peak_mb=rss_peak_mb,
        peak_memory_mb=peak_memory_mb,
        within_expected_memory_limit=peak_memory_mb <= EXPECTED_MEMORY_LIMIT_MB,
    )


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)


def _max_optional(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return max(present)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Mid-Tier Probability Compression historical sweep with memory tracking.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top2-threshold-start", type=float, default=0.80)
    parser.add_argument("--top2-threshold-end", type=float, default=0.95)
    parser.add_argument("--top2-threshold-step", type=float, default=0.01)
    parser.add_argument("--notionals", default="10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--midtier-yes-threshold", type=float, default=0.15)
    parser.add_argument("--max-concurrent-names", type=int, default=100)
    parser.add_argument("--quote-side", choices=("yes", "no"), default="yes")
    parser.add_argument("--timestamp-unit", choices=("ns", "us", "ms", "s"), default="ms")
    parser.add_argument(
        "--resolution-timestamp-unit",
        choices=("ns", "us", "ms", "s"),
        default="ms",
    )
    return parser.parse_args()


def _build_execution_summary(
    *,
    input_root: Path,
    output_dir: Path,
    daily_output: Path,
    rankings_output: Path,
    sweep_summary_output: Path,
    sweep_markdown_output: Path,
    reducer_summary_output: Path,
    pareto_markdown_output: Path,
    thresholds: tuple[float, ...],
    notionals: tuple[float, ...],
    sweep_memory: PhaseMemory,
    reducer_memory: PhaseMemory,
    sweep_summary: dict[str, Any],
    reducer_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifacts": {
            "daily_panel": str(daily_output),
            "pareto_frontier_markdown": str(pareto_markdown_output),
            "rankings_csv": str(rankings_output),
            "reducer_summary_json": str(reducer_summary_output),
            "sweep_summary_json": str(sweep_summary_output),
            "sweep_summary_markdown": str(sweep_markdown_output),
        },
        "combination_count": int(reducer_summary["combination_count"]),
        "days_processed": int(sweep_summary["days_processed"]),
        "input_root": str(input_root),
        "memory": {
            "expected_limit_mb": EXPECTED_MEMORY_LIMIT_MB,
            "reducer_peak_memory_mb": round(reducer_memory.peak_memory_mb, 2),
            "reducer_peak_rss_mb": _round_optional(reducer_memory.rss_peak_mb),
            "reducer_within_750_mb": reducer_memory.within_expected_memory_limit,
            "resource_module_available": _resource_module_available(),
            "resource_peak_mb": _round_optional(
                _max_optional(sweep_memory.resource_peak_mb, reducer_memory.resource_peak_mb)
            ),
            "rss_sampler_available": sweep_memory.rss_peak_mb is not None or reducer_memory.rss_peak_mb is not None,
            "sweep_peak_memory_mb": round(sweep_memory.peak_memory_mb, 2),
            "sweep_peak_rss_mb": _round_optional(sweep_memory.rss_peak_mb),
            "sweep_within_750_mb": sweep_memory.within_expected_memory_limit,
        },
        "notionals": [float(value) for value in notionals],
        "output_dir": str(output_dir),
        "pareto_frontier_count": int(reducer_summary["pareto_frontier_count"]),
        "thresholds": [float(value) for value in thresholds],
        "top3_pareto_coordinates": list(reducer_summary.get("pareto_frontier", []))[:3],
    }


def run_historical_sweep(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    notionals: tuple[float, ...] = DEFAULT_NOTIONALS,
    start_date: str | None = None,
    end_date: str | None = None,
    quote_side: str = "yes",
    timestamp_unit: str = "ms",
    resolution_timestamp_unit: str = "ms",
    midtier_yes_threshold: float = 0.15,
    max_concurrent_names: int = 100,
) -> HistoricalSweepArtifacts:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_output = output_dir / "daily_panel.parquet"
    rankings_output = output_dir / "rankings.csv"
    sweep_summary_output = output_dir / "sweep_summary.json"
    sweep_markdown_output = output_dir / "sweep_summary.md"
    reducer_summary_output = output_dir / "reducer_summary.json"
    pareto_markdown_output = output_dir / "pareto_frontier.md"
    execution_summary_output = output_dir / "execution_summary.json"

    sweep_artifacts, sweep_memory = _measure_phase(
        lambda: run_sweep(
            input_root,
            thresholds=thresholds,
            notionals=notionals,
            start_date=start_date,
            end_date=end_date,
            quote_side=quote_side,
            timestamp_unit=timestamp_unit,
            resolution_timestamp_unit=resolution_timestamp_unit,
            midtier_yes_threshold=midtier_yes_threshold,
            max_concurrent_names=max_concurrent_names,
        )
    )

    sweep_artifacts.daily_rows.write_parquet(daily_output, compression="zstd")
    sweep_summary_output.write_text(json.dumps(sweep_artifacts.summary, indent=2, sort_keys=True), encoding="utf-8")
    sweep_markdown_output.write_text(
        render_sweep_markdown(sweep_artifacts, input_root=input_root, daily_output=daily_output),
        encoding="utf-8",
    )
    sweep_summary = sweep_artifacts.summary

    del sweep_artifacts
    gc.collect()

    reducer_artifacts, reducer_memory = _measure_phase(lambda: run_reducer(daily_output))
    reducer_artifacts.rankings.write_csv(rankings_output)
    reducer_summary_output.write_text(json.dumps(reducer_artifacts.summary, indent=2, sort_keys=True), encoding="utf-8")

    pareto_markdown = render_pareto_frontier_markdown(reducer_artifacts.pareto_frontier)
    pareto_markdown_output.write_text(pareto_markdown, encoding="utf-8")

    execution_summary = _build_execution_summary(
        input_root=input_root,
        output_dir=output_dir,
        daily_output=daily_output,
        rankings_output=rankings_output,
        sweep_summary_output=sweep_summary_output,
        sweep_markdown_output=sweep_markdown_output,
        reducer_summary_output=reducer_summary_output,
        pareto_markdown_output=pareto_markdown_output,
        thresholds=thresholds,
        notionals=notionals,
        sweep_memory=sweep_memory,
        reducer_memory=reducer_memory,
        sweep_summary=sweep_summary,
        reducer_summary=reducer_artifacts.summary,
    )
    execution_summary_output.write_text(json.dumps(execution_summary, indent=2, sort_keys=True), encoding="utf-8")

    return HistoricalSweepArtifacts(
        input_root=str(input_root),
        output_dir=str(output_dir),
        daily_output=str(daily_output),
        rankings_output=str(rankings_output),
        sweep_summary_output=str(sweep_summary_output),
        sweep_markdown_output=str(sweep_markdown_output),
        reducer_summary_output=str(reducer_summary_output),
        pareto_markdown_output=str(pareto_markdown_output),
        execution_summary_output=str(execution_summary_output),
        sweep_memory=sweep_memory,
        reducer_memory=reducer_memory,
        sweep_summary=sweep_summary,
        reducer_summary=reducer_artifacts.summary,
        execution_summary=execution_summary,
    )


def main() -> int:
    args = parse_args()
    artifacts = run_historical_sweep(
        args.input_root,
        output_dir=args.output_dir,
        thresholds=build_threshold_grid(
            args.top2_threshold_start,
            args.top2_threshold_end,
            args.top2_threshold_step,
        ),
        notionals=parse_notionals(args.notionals),
        start_date=args.start_date,
        end_date=args.end_date,
        quote_side=args.quote_side,
        timestamp_unit=args.timestamp_unit,
        resolution_timestamp_unit=args.resolution_timestamp_unit,
        midtier_yes_threshold=args.midtier_yes_threshold,
        max_concurrent_names=args.max_concurrent_names,
    )

    print(json.dumps(artifacts.execution_summary, indent=2, sort_keys=True))
    print("\n---PARETO---\n")
    print(Path(artifacts.pareto_markdown_output).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())