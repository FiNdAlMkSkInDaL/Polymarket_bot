#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.build_l2_parquet_lake import MarketMetadata, RunStats, _clean_text, _discover_days, load_metadata
from scripts.run_lake_builder import (
    STATE_DIR_NAME,
    WorkUnit,
    _build_work_units,
    _completed_marker_path,
    _in_progress_marker_path,
    _marker_outputs_exist,
    _read_json,
    _source_signature,
    _source_signature_matches,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit run_lake_builder markers and classify data-backed versus validation-only market-day units.",
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
        help="Parquet lake root containing _lake_builder_state.",
    )
    parser.add_argument(
        "--day",
        action="append",
        dest="days",
        default=[],
        help="Optional YYYY-MM-DD partition filter. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--market-id",
        action="append",
        default=[],
        help="Optional market_id filter. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional destination for the JSON audit report. Defaults to <output-root>/_lake_builder_state/missing_data_audit.json.",
    )
    return parser.parse_args(argv)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _unit_payload(unit: WorkUnit) -> dict[str, str]:
    return {
        "job_id": unit.job_id,
        "day": unit.day,
        "market_id": unit.metadata.market_id,
        "event_id": unit.metadata.event_id,
    }


def _relative_paths(paths: list[str]) -> list[str]:
    return [str(path).replace("\\", "/") for path in paths]


def _classify_unit(
    *,
    unit: WorkUnit,
    output_root: Path,
    state_root: Path,
) -> dict[str, Any]:
    completed_path = _completed_marker_path(state_root, unit)
    in_progress_path = _in_progress_marker_path(state_root, unit)
    completed_marker = _read_json(completed_path)
    in_progress_marker = _read_json(in_progress_path)
    source_files = _source_signature(unit)

    payload = {
        **_unit_payload(unit),
        "completed_marker": str(completed_path),
        "in_progress_marker_present": bool(in_progress_marker),
    }
    if completed_marker is None:
        missing = [
            name
            for name, signature in source_files.items()
            if (not signature.get("exists")) or not int(signature.get("size_bytes") or 0) > 0
        ]
        payload.update(
            {
                "category": "missing_marker",
                "missing_source_files": missing,
                "source_files": source_files,
            }
        )
        return payload

    written_files = [str(path) for path in completed_marker.get("written_files", [])]
    has_l2_book = any(
        relative_path.startswith("l2_book/") or relative_path.startswith("l2_book\\")
        for relative_path in written_files
    )
    missing_required_file = int(
        completed_marker.get("run_stats_delta", {}).get("markets_skipped", {}).get("missing_required_file", 0) or 0
    ) > 0
    output_rows = int(completed_marker.get("run_stats_delta", {}).get("output_rows", 0) or 0)
    normalized_source_match = _source_signature_matches(source_files, completed_marker.get("source_files"))
    raw_missing = [
        name
        for name, signature in source_files.items()
        if (not signature.get("exists")) or not int(signature.get("size_bytes") or 0) > 0
    ]
    missing_written_files = [
        relative_path
        for relative_path in written_files
        if not (output_root / relative_path).exists()
    ]

    if has_l2_book:
        category = "data_backed"
    elif missing_required_file:
        category = "validation_only_missing_required_file"
    elif output_rows == 0:
        category = "validation_only_zero_output"
    else:
        category = "validation_only_other"

    payload.update(
        {
            "category": category,
            "normalized_source_match": normalized_source_match,
            "direct_source_match": completed_marker.get("source_files") == source_files,
            "outputs_exist": _marker_outputs_exist(output_root, completed_marker),
            "missing_written_files": _relative_paths(missing_written_files),
            "written_files": _relative_paths(written_files),
            "output_rows": output_rows,
            "missing_source_files": raw_missing,
            "run_stats_delta": completed_marker.get("run_stats_delta", {}),
        }
    )
    return payload


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    state_root = output_root / STATE_DIR_NAME
    report_path = args.report_path.resolve() if args.report_path is not None else state_root / "missing_data_audit.json"

    metadata_stats = RunStats()
    metadata_paths = [path.resolve() for path in args.metadata]
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

    summary: Counter[str] = Counter()
    missing_patterns: Counter[str] = Counter()
    case_only_mismatches = 0
    recoverable_missing_required_units: list[dict[str, Any]] = []
    still_missing_required_units: list[dict[str, Any]] = []
    zero_output_units: list[dict[str, Any]] = []
    missing_markers: list[dict[str, Any]] = []
    marker_output_issues: list[dict[str, Any]] = []

    for unit in units:
        entry = _classify_unit(unit=unit, output_root=output_root, state_root=state_root)
        category = str(entry["category"])
        summary[category] += 1
        if not entry.get("direct_source_match") and entry.get("normalized_source_match"):
            case_only_mismatches += 1
        if entry.get("missing_written_files"):
            marker_output_issues.append(entry)
        if category == "validation_only_missing_required_file":
            missing = list(entry.get("missing_source_files", []))
            if missing:
                missing_patterns[",".join(missing)] += 1
                still_missing_required_units.append(entry)
            else:
                recoverable_missing_required_units.append(entry)
        elif category == "validation_only_zero_output":
            zero_output_units.append(entry)
        elif category == "missing_marker":
            missing_markers.append(entry)

    rerun_candidates = [
        _unit_payload(
            WorkUnit(
                day=entry["day"],
                day_dir=raw_root / entry["day"],
                metadata=MarketMetadata(
                    market_id=entry["market_id"],
                    event_id=entry["event_id"],
                    yes_asset_id="",
                    no_asset_id="",
                ),
            )
        )
        for entry in recoverable_missing_required_units
    ]
    rerun_command: list[str] | None = None
    if recoverable_missing_required_units:
        rerun_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_lake_builder.py"),
            "--raw-root",
            str(raw_root),
        ]
        for metadata_path in metadata_paths:
            rerun_command.extend(["--metadata", str(metadata_path)])
        rerun_command.extend(["--output-root", str(output_root), "--rebuild"])
        for entry in recoverable_missing_required_units:
            rerun_command.extend(["--day", entry["day"], "--market-id", entry["market_id"]])

    report = {
        "generated_at": _now_iso(),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "report_path": str(report_path),
        "days": days,
        "missing_selected_days": missing_days,
        "metadata_paths": [str(path) for path in metadata_paths],
        "metadata_rows_loaded": metadata_stats.metadata_rows_loaded,
        "units_analyzed": len(units),
        "summary": dict(summary),
        "missing_required_file_patterns": dict(missing_patterns),
        "normalized_source_matches_with_direct_case_mismatch": case_only_mismatches,
        "recoverable_missing_required_file_units": recoverable_missing_required_units,
        "still_missing_required_file_units": still_missing_required_units,
        "zero_output_validation_only_units": zero_output_units,
        "missing_marker_units": missing_markers,
        "marker_output_issues": marker_output_issues,
        "rerun_candidates": rerun_candidates,
        "rerun_command": rerun_command,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())