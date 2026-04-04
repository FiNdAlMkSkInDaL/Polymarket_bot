#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.build_l2_parquet_lake import (
    DEFAULT_BATCH_LINES,
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_FLUSH_ROWS,
    DEPTH_LEVELS,
    MANIFEST_NAME,
    MarketMetadata,
    RunStats,
    _clean_text,
    _discover_days,
    load_metadata,
    process_market_day,
)


STATE_DIR_NAME = "_lake_builder_state"
COMPLETED_DIR_NAME = "completed"
IN_PROGRESS_DIR_NAME = "in_progress"
LAST_RUN_NAME = "last_run.json"
LOG_FILE_NAME = "run_lake_builder.log"
DEFAULT_MEMORY_CAP_MB = 750.0
BYTES_PER_MB = 1024.0 * 1024.0
IGNORED_AGGREGATE_DELTA_KEYS = {
    "metadata_rows_loaded",
    "metadata_rows_rejected",
    "days_processed",
    "markets_considered",
}


@dataclass(frozen=True, slots=True)
class WorkUnit:
    day: str
    day_dir: Path
    metadata: MarketMetadata

    @property
    def job_id(self) -> str:
        return f"{self.day}:{self.metadata.market_id}"

    @property
    def yes_path(self) -> Path:
        return self.day_dir / f"{self.metadata.yes_asset_id}.jsonl"

    @property
    def no_path(self) -> Path:
        return self.day_dir / f"{self.metadata.no_asset_id}.jsonl"

    @property
    def delta_path(self) -> Path:
        return self.day_dir / f"{self.metadata.market_id}.jsonl"

    def source_paths(self) -> dict[str, Path]:
        return {
            "yes_snapshot": self.yes_path,
            "no_snapshot": self.no_path,
            "market_delta": self.delta_path,
        }


@dataclass(slots=True)
class WrapperRunStats:
    discovered_units: int = 0
    processed_units: int = 0
    skipped_completed_units: int = 0
    resumed_partial_units: int = 0
    rebuilt_stale_units: int = 0
    completed_days: int = 0
    peak_tracemalloc_bytes: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "discovered_units": self.discovered_units,
            "processed_units": self.processed_units,
            "skipped_completed_units": self.skipped_completed_units,
            "resumed_partial_units": self.resumed_partial_units,
            "rebuilt_stale_units": self.rebuilt_stale_units,
            "completed_days": self.completed_days,
            "peak_tracemalloc_mb": round(self.peak_tracemalloc_bytes / BYTES_PER_MB, 3),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable execution wrapper around scripts/build_l2_parquet_lake.py.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw_ticks"),
        help="Root directory containing raw_ticks/YYYY-MM-DD/*.jsonl partitions.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        action="append",
        required=True,
        help="One or more metadata JSON files that resolve market_id, event_id, YES token, and NO token.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for the final Parquet lake and resumable state.",
    )
    parser.add_argument(
        "--day",
        action="append",
        dest="days",
        default=[],
        help="Optional YYYY-MM-DD partition to process. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--market-id",
        action="append",
        default=[],
        help="Optional market_id filter. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=DEFAULT_BATCH_LINES,
        help="Maximum raw JSONL lines parsed into Polars per batch.",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=DEFAULT_FLUSH_ROWS,
        help="Maximum buffered final rows before a Parquet flush.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help="Zstd compression level for output Parquet files.",
    )
    parser.add_argument(
        "--memory-cap-mb",
        type=float,
        default=DEFAULT_MEMORY_CAP_MB,
        help="Fail the run if tracemalloc peak exceeds this threshold.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Emit an info log every N processed market-day units.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Optional explicit directory for resumable state. Defaults to <output-root>/_lake_builder_state.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path. Defaults to <state-dir>/run_lake_builder.log.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore completed markers for the selected work units and rebuild them from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which market-day units would run without mutating outputs.",
    )
    return parser.parse_args(argv)


def _setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime

    logger = logging.getLogger("run_lake_builder")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _strict_schema() -> dict[str, str]:
    return {
        "timestamp": "Datetime(ms, UTC)",
        "market_id": "Utf8",
        "event_id": "Utf8",
        "token_id": "Utf8[YES|NO]",
        "best_bid": "Float64",
        "best_ask": "Float64",
        "bid_depth": "Float64 top-5 notional",
        "ask_depth": "Float64 top-5 notional",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.tmp"
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for _ in range(10):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_marker_path(state_root: Path, unit: WorkUnit) -> Path:
    return state_root / COMPLETED_DIR_NAME / unit.day / f"{unit.metadata.market_id}.json"


def _in_progress_marker_path(state_root: Path, unit: WorkUnit) -> Path:
    return state_root / IN_PROGRESS_DIR_NAME / unit.day / f"{unit.metadata.market_id}.json"


def _source_signature_for_path(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve(strict=False)
    if not path.exists():
        return {
            "path": _canonicalize_source_path(resolved_path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
        }
    stat = path.stat()
    return {
        "path": _canonicalize_source_path(resolved_path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _canonicalize_source_path(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        normalized = str(PureWindowsPath(text))
        return normalized[0].lower() + normalized[1:]
    return os.path.normpath(text)


def _normalize_source_signature_entry(signature: Mapping[str, Any] | None) -> dict[str, Any]:
    signature = signature or {}
    size_bytes = signature.get("size_bytes")
    mtime_ns = signature.get("mtime_ns")
    return {
        "path": _canonicalize_source_path(signature.get("path")),
        "exists": bool(signature.get("exists")),
        "size_bytes": None if size_bytes is None else int(size_bytes),
        "mtime_ns": None if mtime_ns is None else int(mtime_ns),
    }


def _source_signature_matches(
    current: Mapping[str, Mapping[str, Any]] | None,
    stored: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    current = current or {}
    stored = stored or {}
    if set(current) != set(stored):
        return False
    for key in current:
        if _normalize_source_signature_entry(current[key]) != _normalize_source_signature_entry(stored[key]):
            return False
    return True


def _source_signature(unit: WorkUnit) -> dict[str, dict[str, Any]]:
    return {
        label: _source_signature_for_path(path)
        for label, path in unit.source_paths().items()
    }


def _marker_outputs_exist(output_root: Path, marker: Mapping[str, Any]) -> bool:
    for relative_path in marker.get("written_files", []):
        if not (output_root / relative_path).exists():
            return False
    return True


def _cleanup_written_files(output_root: Path, marker: Mapping[str, Any], logger: logging.Logger) -> int:
    removed = 0
    for relative_path in marker.get("written_files", []):
        path = output_root / relative_path
        if path.exists():
            path.unlink()
            removed += 1
            _prune_empty_parents(path.parent, stop_at=output_root)
    if removed:
        logger.info("Removed %s stale parquet shard(s) for %s", removed, marker.get("job_id", "unknown"))
    return removed


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    while current != stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _build_work_units(
    *,
    raw_root: Path,
    days: Sequence[str],
    metadata_by_market: Mapping[str, MarketMetadata],
    market_filter: set[str],
) -> tuple[list[WorkUnit], list[str]]:
    units: list[WorkUnit] = []
    missing_days: list[str] = []

    for day in days:
        day_dir = raw_root / day
        if not day_dir.exists():
            missing_days.append(day)
            continue

        available = {path.stem for path in day_dir.glob("*.jsonl")}
        if not available:
            continue

        for metadata in metadata_by_market.values():
            if market_filter and metadata.market_id not in market_filter:
                continue
            related_stems = {
                metadata.market_id,
                metadata.yes_asset_id,
                metadata.no_asset_id,
            }
            if not available.intersection(related_stems):
                continue
            units.append(WorkUnit(day=day, day_dir=day_dir, metadata=metadata))

    units.sort(key=lambda unit: (unit.day, unit.metadata.market_id))
    return units, missing_days


def _diff_stats(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0)
        if isinstance(after_value, dict):
            delta_map: dict[str, int] = {}
            before_dict = before_value if isinstance(before_value, dict) else {}
            for nested_key in set(before_dict) | set(after_value):
                nested_delta = int(after_value.get(nested_key, 0)) - int(before_dict.get(nested_key, 0))
                if nested_delta:
                    delta_map[nested_key] = nested_delta
            diff[key] = delta_map
            continue
        diff[key] = int(after_value) - int(before_value)
    return diff


def _accumulate_counter(target: Counter[str], delta: Mapping[str, Any]) -> None:
    for key, value in delta.items():
        target[str(key)] += int(value)


def _counter_total(delta: Mapping[str, Any] | None) -> int:
    if not isinstance(delta, Mapping):
        return 0
    return sum(int(value) for value in delta.values())


def _accumulate_stats_delta(stats: RunStats, delta: Mapping[str, Any]) -> None:
    for key, value in delta.items():
        if key in IGNORED_AGGREGATE_DELTA_KEYS:
            continue
        if key == "markets_completed":
            stats.markets_completed += int(value)
            continue
        if key == "output_rows":
            stats.output_rows += int(value)
            continue
        if key == "rejected_rows":
            stats.rejected_rows += int(value)
            continue
        if key == "markets_skipped" and isinstance(value, Mapping):
            _accumulate_counter(stats.markets_skipped, value)
            continue
        if key == "raw_records_read" and isinstance(value, Mapping):
            _accumulate_counter(stats.raw_records_read, value)
            continue
        if key == "raw_records_parsed" and isinstance(value, Mapping):
            _accumulate_counter(stats.raw_records_parsed, value)
            continue
        if key == "raw_records_malformed" and isinstance(value, Mapping):
            _accumulate_counter(stats.raw_records_malformed, value)
            continue
        if key == "raw_batches_salvaged" and isinstance(value, Mapping):
            _accumulate_counter(stats.raw_batches_salvaged, value)


def _aggregate_selected_stats(
    *,
    units: Sequence[WorkUnit],
    state_root: Path,
    metadata_stats: RunStats,
) -> RunStats:
    aggregate = RunStats()
    aggregate.metadata_rows_loaded = metadata_stats.metadata_rows_loaded
    aggregate.metadata_rows_rejected.update(metadata_stats.metadata_rows_rejected)
    aggregate.days_processed = len({unit.day for unit in units})
    aggregate.markets_considered = len(units)

    for unit in units:
        marker = _read_json(_completed_marker_path(state_root, unit))
        if marker is None:
            continue
        _accumulate_stats_delta(aggregate, marker.get("run_stats_delta", {}))
    return aggregate


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _write_marker(
    *,
    path: Path,
    unit: WorkUnit,
    source_files: Mapping[str, Any],
    written_files: Sequence[str],
    started_at: str,
    updated_at: str,
    run_stats_delta: Mapping[str, Any] | None = None,
    status: str,
    duration_seconds: float | None = None,
    peak_tracemalloc_mb: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": unit.job_id,
        "status": status,
        "day": unit.day,
        "market_id": unit.metadata.market_id,
        "event_id": unit.metadata.event_id,
        "started_at": started_at,
        "updated_at": updated_at,
        "source_files": dict(source_files),
        "written_files": list(written_files),
    }
    if run_stats_delta is not None:
        payload["run_stats_delta"] = dict(run_stats_delta)
    if duration_seconds is not None:
        payload["duration_seconds"] = round(duration_seconds, 3)
    if peak_tracemalloc_mb is not None:
        payload["peak_tracemalloc_mb"] = round(peak_tracemalloc_mb, 3)
    if error is not None:
        payload["error"] = error
    _write_json(path, payload)
    return payload


def _log_progress(
    *,
    logger: logging.Logger,
    processed_index: int,
    total_units: int,
    unit: WorkUnit,
    duration_seconds: float,
    peak_tracemalloc_mb: float,
    current_run_stats: RunStats,
    run_stats_delta: Mapping[str, Any],
) -> None:
    logger.info(
        "Completed market-day %s/%s | %s | duration=%.2fs | peak_tracemalloc=%.2f MB | output_rows=%s | rejected_rows=%s | malformed_raw_lines=%s | salvaged_batches=%s",
        processed_index,
        total_units,
        unit.job_id,
        duration_seconds,
        peak_tracemalloc_mb,
        current_run_stats.output_rows,
        current_run_stats.rejected_rows,
        _counter_total(run_stats_delta.get("raw_records_malformed")),
        _counter_total(run_stats_delta.get("raw_batches_salvaged")),
    )


def run_builder(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_root = args.state_dir.resolve() if args.state_dir is not None else output_root / STATE_DIR_NAME
    state_root.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file.resolve() if args.log_file is not None else state_root / LOG_FILE_NAME
    logger = _setup_logger(log_file)

    metadata_paths = [path.resolve() for path in args.metadata]
    metadata_stats = RunStats()
    metadata_by_market = load_metadata(metadata_paths, metadata_stats)
    if not metadata_by_market:
        raise ValueError(
            "No market metadata rows were loaded. Use metadata files that include market_id, event_id, and YES/NO token ids."
        )
    market_filter = {value.lower() for value in args.market_id if _clean_text(value)}
    days = _discover_days(raw_root, args.days)
    units, missing_days = _build_work_units(
        raw_root=raw_root,
        days=days,
        metadata_by_market=metadata_by_market,
        market_filter=market_filter,
    )
    if not units:
        raise ValueError(
            "No market-day work units were discovered. Check the selected day filters, metadata coverage, and raw JSONL partitions."
        )
    units_by_day: dict[str, list[WorkUnit]] = defaultdict(list)
    for unit in units:
        units_by_day[unit.day].append(unit)

    wrapper_stats = WrapperRunStats(discovered_units=len(units))
    current_run_stats = RunStats()
    current_run_stats.metadata_rows_loaded = metadata_stats.metadata_rows_loaded
    current_run_stats.metadata_rows_rejected.update(metadata_stats.metadata_rows_rejected)

    logger.info(
        "Discovered %s market-day unit(s) across %s day partition(s).",
        len(units),
        len(units_by_day),
    )
    if missing_days:
        logger.warning("Selected day partition(s) not found under raw root: %s", ", ".join(missing_days))

    if args.dry_run:
        for index, unit in enumerate(units, start=1):
            logger.info("Dry run %s/%s | %s", index, len(units), unit.job_id)
        manifest = {
            "generated_at": _now_iso(),
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "state_root": str(state_root),
            "days": days,
            "depth_levels": DEPTH_LEVELS,
            "strict_schema": _strict_schema(),
            "stats": _aggregate_selected_stats(units=units, state_root=state_root, metadata_stats=metadata_stats).to_json(),
            "current_run": {
                "mode": "dry_run",
                "memory_cap_mb": float(args.memory_cap_mb),
                "missing_selected_days": missing_days,
                **wrapper_stats.to_json(),
            },
        }
        _write_json(output_root / MANIFEST_NAME, manifest)
        _write_json(state_root / LAST_RUN_NAME, manifest)
        return manifest

    started_at = _now_iso()
    start_clock = perf_counter()
    tracemalloc.start()
    try:
        processed_index = 0
        for day in sorted(units_by_day):
            day_units = units_by_day[day]
            processed_in_day = 0
            skipped_in_day = 0
            day_counted_in_stats = False

            for unit in day_units:
                source_files = _source_signature(unit)
                completed_path = _completed_marker_path(state_root, unit)
                in_progress_path = _in_progress_marker_path(state_root, unit)
                completed_marker = _read_json(completed_path)
                in_progress_marker = _read_json(in_progress_path)

                if args.rebuild:
                    if completed_marker is not None:
                        _cleanup_written_files(output_root, completed_marker, logger)
                        completed_path.unlink(missing_ok=True)
                        completed_marker = None
                        wrapper_stats.rebuilt_stale_units += 1
                    if in_progress_marker is not None:
                        _cleanup_written_files(output_root, in_progress_marker, logger)
                        in_progress_path.unlink(missing_ok=True)
                        in_progress_marker = None
                        wrapper_stats.resumed_partial_units += 1

                if completed_marker is not None:
                    marker_is_fresh = (
                        _source_signature_matches(source_files, completed_marker.get("source_files"))
                        and _marker_outputs_exist(output_root, completed_marker)
                    )
                    if marker_is_fresh:
                        if in_progress_path.exists():
                            in_progress_path.unlink(missing_ok=True)
                        wrapper_stats.skipped_completed_units += 1
                        skipped_in_day += 1
                        logger.info("Skipping completed market-day %s", unit.job_id)
                        continue

                    _cleanup_written_files(output_root, completed_marker, logger)
                    completed_path.unlink(missing_ok=True)
                    completed_marker = None
                    wrapper_stats.rebuilt_stale_units += 1
                    logger.info("Rebuilding stale market-day %s", unit.job_id)

                if in_progress_marker is not None:
                    _cleanup_written_files(output_root, in_progress_marker, logger)
                    in_progress_path.unlink(missing_ok=True)
                    wrapper_stats.resumed_partial_units += 1
                    logger.info("Resuming interrupted market-day %s", unit.job_id)

                if not day_counted_in_stats:
                    current_run_stats.days_processed += 1
                    day_counted_in_stats = True

                current_run_stats.markets_considered += 1
                processed_in_day += 1
                processed_index += 1
                unit_started_at = _now_iso()
                written_files: list[str] = []
                before_stats = current_run_stats.to_json()
                tracemalloc.reset_peak()

                def record_output_file(path: Path) -> None:
                    relative_path = path.relative_to(output_root).as_posix()
                    if relative_path in written_files:
                        return
                    written_files.append(relative_path)
                    _write_marker(
                        path=in_progress_path,
                        unit=unit,
                        source_files=source_files,
                        written_files=written_files,
                        started_at=unit_started_at,
                        updated_at=_now_iso(),
                        status="in_progress",
                    )

                _write_marker(
                    path=in_progress_path,
                    unit=unit,
                    source_files=source_files,
                    written_files=written_files,
                    started_at=unit_started_at,
                    updated_at=unit_started_at,
                    status="in_progress",
                )

                unit_clock = perf_counter()
                try:
                    process_market_day(
                        day=unit.day,
                        day_dir=unit.day_dir,
                        metadata=unit.metadata,
                        output_root=output_root,
                        batch_lines=args.batch_lines,
                        flush_rows=args.flush_rows,
                        compression_level=args.compression_level,
                        stats=current_run_stats,
                        on_file_written=record_output_file,
                    )
                except Exception as exc:
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    wrapper_stats.peak_tracemalloc_bytes = max(wrapper_stats.peak_tracemalloc_bytes, peak_bytes)
                    _write_marker(
                        path=in_progress_path,
                        unit=unit,
                        source_files=source_files,
                        written_files=written_files,
                        started_at=unit_started_at,
                        updated_at=_now_iso(),
                        status="failed",
                        error=repr(exc),
                        peak_tracemalloc_mb=peak_bytes / BYTES_PER_MB,
                    )
                    logger.exception("Market-day failed: %s", unit.job_id)
                    raise

                _, peak_bytes = tracemalloc.get_traced_memory()
                wrapper_stats.peak_tracemalloc_bytes = max(wrapper_stats.peak_tracemalloc_bytes, peak_bytes)
                duration_seconds = perf_counter() - unit_clock
                run_stats_delta = _diff_stats(before_stats, current_run_stats.to_json())
                _write_marker(
                    path=completed_path,
                    unit=unit,
                    source_files=source_files,
                    written_files=written_files,
                    started_at=unit_started_at,
                    updated_at=_now_iso(),
                    run_stats_delta=run_stats_delta,
                    status="completed",
                    duration_seconds=duration_seconds,
                    peak_tracemalloc_mb=peak_bytes / BYTES_PER_MB,
                )
                in_progress_path.unlink(missing_ok=True)
                wrapper_stats.processed_units += 1

                if processed_index % max(args.progress_every, 1) == 0:
                    _log_progress(
                        logger=logger,
                        processed_index=processed_index,
                        total_units=len(units),
                        unit=unit,
                        duration_seconds=duration_seconds,
                        peak_tracemalloc_mb=peak_bytes / BYTES_PER_MB,
                        current_run_stats=current_run_stats,
                        run_stats_delta=run_stats_delta,
                    )

                if peak_bytes > int(args.memory_cap_mb * BYTES_PER_MB):
                    raise MemoryError(
                        f"tracemalloc peak {peak_bytes / BYTES_PER_MB:.2f} MB exceeded cap {args.memory_cap_mb:.2f} MB"
                    )

            wrapper_stats.completed_days += 1
            logger.info(
                "Completed date partition %s | processed=%s | skipped_completed=%s | total=%s",
                day,
                processed_in_day,
                skipped_in_day,
                len(day_units),
            )
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    aggregate_stats = _aggregate_selected_stats(
        units=units,
        state_root=state_root,
        metadata_stats=metadata_stats,
    )
    duration_seconds = perf_counter() - start_clock
    manifest = {
        "generated_at": _now_iso(),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "days": days,
        "depth_levels": DEPTH_LEVELS,
        "strict_schema": _strict_schema(),
        "stats": aggregate_stats.to_json(),
        "current_run": {
            "started_at": started_at,
            "completed_at": _now_iso(),
            "duration_seconds": round(duration_seconds, 3),
            "memory_cap_mb": float(args.memory_cap_mb),
            "missing_selected_days": missing_days,
            "current_run_stats": current_run_stats.to_json(),
            **wrapper_stats.to_json(),
        },
    }
    _write_json(output_root / MANIFEST_NAME, manifest)
    _write_json(state_root / LAST_RUN_NAME, manifest)
    logger.info(
        "Run finished | manifest=%s | aggregate_output_rows=%s | aggregate_rejected_rows=%s | malformed_raw_lines=%s | salvaged_batches=%s | peak_tracemalloc=%.2f MB",
        output_root / MANIFEST_NAME,
        aggregate_stats.output_rows,
        aggregate_stats.rejected_rows,
        sum(aggregate_stats.raw_records_malformed.values()),
        sum(aggregate_stats.raw_batches_salvaged.values()),
        wrapper_stats.peak_tracemalloc_bytes / BYTES_PER_MB,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_builder(args)
    except Exception as exc:
        logger = logging.getLogger("run_lake_builder")
        if logger.handlers:
            logger.error("run_lake_builder failed: %s", exc)
        else:
            print(f"run_lake_builder failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "manifest": str(args.output_root.resolve() / MANIFEST_NAME),
                "stats": manifest["stats"],
                "current_run": manifest["current_run"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())