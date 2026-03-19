#!/usr/bin/env python3
"""Audit recorded forward-test raw ticks for L2 replay readiness."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SNAPSHOT_EVENT_TYPES = {"book", "snapshot", "book_snapshot"}
DELTA_EVENT_TYPES = {"price_change"}


@dataclass
class AuditSummary:
    dates: list[str]
    files_scanned: int
    records_scanned: int
    source_counts: Counter[str]
    event_type_counts: Counter[str]
    asset_counts: Counter[str]
    snapshot_records: int
    delta_records: int
    trade_records: int
    malformed_lines: int
    files_with_snapshots: int
    files_with_deltas: int
    files_with_trades: int
    max_levels_bid: int
    max_levels_ask: int
    snapshot_triggers: Counter[str]

    def to_dict(self) -> dict[str, Any]:
        assets = self.asset_counts.most_common(10)
        return {
            "dates": self.dates,
            "files_scanned": self.files_scanned,
            "records_scanned": self.records_scanned,
            "source_counts": dict(self.source_counts),
            "event_type_counts": dict(self.event_type_counts),
            "top_assets": [{"asset_id": asset_id, "records": count} for asset_id, count in assets],
            "snapshot_records": self.snapshot_records,
            "delta_records": self.delta_records,
            "trade_records": self.trade_records,
            "malformed_lines": self.malformed_lines,
            "files_with_snapshots": self.files_with_snapshots,
            "files_with_deltas": self.files_with_deltas,
            "files_with_trades": self.files_with_trades,
            "max_levels_bid": self.max_levels_bid,
            "max_levels_ask": self.max_levels_ask,
            "snapshot_triggers": dict(self.snapshot_triggers),
        }


def _iter_date_dirs(data_dir: Path, dates: list[str] | None) -> list[Path]:
    raw_root = data_dir / "raw_ticks"
    if not raw_root.exists():
        raise FileNotFoundError(f"raw tick directory not found: {raw_root}")

    if dates:
        selected = [raw_root / date for date in dates]
        missing = [path.name for path in selected if not path.exists()]
        if missing:
            raise FileNotFoundError(f"requested date directories missing: {', '.join(missing)}")
        return selected

    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def _classify_record(record: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    source = str(record.get("source") or "missing")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    event_type = str(payload.get("event_type") or "missing")
    return source, event_type, payload


def audit_data(data_dir: Path, dates: list[str] | None, sample_limit: int | None) -> AuditSummary:
    source_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    asset_counts: Counter[str] = Counter()
    snapshot_triggers: Counter[str] = Counter()

    files_scanned = 0
    records_scanned = 0
    snapshot_records = 0
    delta_records = 0
    trade_records = 0
    malformed_lines = 0
    files_with_snapshots = 0
    files_with_deltas = 0
    files_with_trades = 0
    max_levels_bid = 0
    max_levels_ask = 0
    per_file_tail_limit = 100

    selected_dirs = _iter_date_dirs(data_dir, dates)
    date_dirs = list(reversed(selected_dirs)) if sample_limit is not None else selected_dirs

    for date_dir in date_dirs:
        jsonl_files = sorted(date_dir.glob("*.jsonl"))
        if sample_limit is not None:
            jsonl_files = sorted(
                jsonl_files,
                key=lambda path: (path.stat().st_mtime, path.name),
                reverse=True,
            )

        for jsonl_path in jsonl_files:
            files_scanned += 1
            file_has_snapshot = False
            file_has_delta = False
            file_has_trade = False

            with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
                raw_lines = handle.readlines()

            if sample_limit is not None:
                raw_lines = raw_lines[-per_file_tail_limit:]
                line_iter = reversed(raw_lines)
            else:
                line_iter = raw_lines

            for raw_line in line_iter:
                if sample_limit is not None and records_scanned >= sample_limit:
                    break

                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue

                records_scanned += 1
                source, event_type, payload = _classify_record(record)
                asset_id = str(record.get("asset_id") or payload.get("asset_id") or "unknown")

                source_counts[source] += 1
                event_type_counts[event_type] += 1
                asset_counts[asset_id] += 1

                if source == "trade" or event_type == "last_trade_price":
                    trade_records += 1
                    file_has_trade = True

                if event_type in DELTA_EVENT_TYPES:
                    delta_records += 1
                    file_has_delta = True

                if event_type in SNAPSHOT_EVENT_TYPES:
                    snapshot_records += 1
                    file_has_snapshot = True
                    bids = payload.get("bids") or []
                    asks = payload.get("asks") or []
                    if isinstance(bids, list):
                        max_levels_bid = max(max_levels_bid, len(bids))
                    if isinstance(asks, list):
                        max_levels_ask = max(max_levels_ask, len(asks))
                    trigger = str(payload.get("snapshot_trigger") or "unknown")
                    snapshot_triggers[trigger] += 1

            if file_has_snapshot:
                files_with_snapshots += 1
            if file_has_delta:
                files_with_deltas += 1
            if file_has_trade:
                files_with_trades += 1

            if sample_limit is not None and records_scanned >= sample_limit:
                    break

        if sample_limit is not None and records_scanned >= sample_limit:
            break

    return AuditSummary(
        dates=[path.name for path in selected_dirs],
        files_scanned=files_scanned,
        records_scanned=records_scanned,
        source_counts=source_counts,
        event_type_counts=event_type_counts,
        asset_counts=asset_counts,
        snapshot_records=snapshot_records,
        delta_records=delta_records,
        trade_records=trade_records,
        malformed_lines=malformed_lines,
        files_with_snapshots=files_with_snapshots,
        files_with_deltas=files_with_deltas,
        files_with_trades=files_with_trades,
        max_levels_bid=max_levels_bid,
        max_levels_ask=max_levels_ask,
        snapshot_triggers=snapshot_triggers,
    )


def _print_human(summary: AuditSummary) -> None:
    payload = summary.to_dict()
    print("Forward data audit")
    print(f"Dates: {', '.join(payload['dates']) if payload['dates'] else 'none'}")
    print(f"Files scanned: {payload['files_scanned']}")
    print(f"Records scanned: {payload['records_scanned']}")
    print(f"Sources: {payload['source_counts']}")
    print(f"Event types: {payload['event_type_counts']}")
    print(f"Snapshot records: {payload['snapshot_records']}")
    print(f"Delta records: {payload['delta_records']}")
    print(f"Trade records: {payload['trade_records']}")
    print(f"Files with snapshots: {payload['files_with_snapshots']}")
    print(f"Files with deltas: {payload['files_with_deltas']}")
    print(f"Files with trades: {payload['files_with_trades']}")
    print(f"Max bid levels observed: {payload['max_levels_bid']}")
    print(f"Max ask levels observed: {payload['max_levels_ask']}")
    print(f"Snapshot triggers: {payload['snapshot_triggers']}")
    print(f"Malformed lines: {payload['malformed_lines']}")
    print(f"Top assets: {payload['top_assets']}")


def _validate_acceptance(summary: AuditSummary) -> list[str]:
    failures: list[str] = []
    if summary.files_scanned == 0:
        failures.append("no raw tick files found")
    if summary.records_scanned == 0:
        failures.append("no records scanned")
    if summary.delta_records == 0:
        failures.append("no L2 delta records found")
    if summary.snapshot_records == 0:
        failures.append("no L2 snapshot records found")
    if summary.max_levels_bid == 0 or summary.max_levels_ask == 0:
        failures.append("snapshot payloads do not contain bid/ask depth arrays")
    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Recorded data root directory")
    parser.add_argument("--date", action="append", dest="dates", help="Specific YYYY-MM-DD directory to audit; repeatable")
    parser.add_argument("--sample-limit", type=int, default=None, help="Stop after scanning this many records")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--require-wfo-ready", action="store_true", help="Exit non-zero unless both deltas and snapshots with depth arrays are present")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary = audit_data(Path(args.data_dir), args.dates, args.sample_limit)

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        _print_human(summary)

    if args.require_wfo_ready:
        failures = _validate_acceptance(summary)
        if failures:
            print("WFO readiness check failed:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())