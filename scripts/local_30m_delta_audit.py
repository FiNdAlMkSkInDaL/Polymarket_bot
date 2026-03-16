#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LogRow:
    t: datetime
    event: str


def _parse_iso_dt(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_rows(log_path: Path) -> list[LogRow]:
    rows: list[LogRow] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts = obj.get("timestamp") or obj.get("ts")
            event = obj.get("event")
            if not ts or not event:
                continue
            try:
                t = _parse_iso_dt(str(ts))
            except ValueError:
                continue
            rows.append(LogRow(t=t, event=str(event)))
    return rows


def _window(rows: Iterable[LogRow], start: datetime, end: datetime) -> list[LogRow]:
    return [r for r in rows if start <= r.t < end]


def _count_any(rows: Iterable[LogRow], names: set[str]) -> int:
    return sum(1 for r in rows if r.event in names)


def _count_contains(rows: Iterable[LogRow], needle: str) -> int:
    n = needle.lower()
    return sum(1 for r in rows if n in r.event.lower())


def _count_fill_like(rows: Iterable[LogRow]) -> int:
    return sum(
        1
        for r in rows
        if ("fill" in r.event.lower()) or ("execution" in r.event.lower())
    )


def _infer_restart_ts(rows: list[LogRow]) -> datetime | None:
    markers = [r.t for r in rows if r.event in {"bot_starting", "bot_running"}]
    return max(markers) if markers else None


def _resolve_log_path(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Log file not found: {p}")

    candidates = [
        Path("logs/bot_fresh.log"),
        Path("logs/bot_console.log"),
        Path("logs/bot.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No log file found at logs/bot_fresh.log, logs/bot_console.log, or logs/bot.jsonl")


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a 1h pre-restart strict window vs 30m post-restart aggressive window "
            "for local PAPER execution funnel metrics."
        )
    )
    parser.add_argument(
        "--restart-ts",
        default=None,
        help="Restart timestamp in ISO-8601 (example: 2026-03-16T15:20:00Z). If omitted, latest bot_starting/bot_running is used.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Log file path. Defaults: logs/bot_fresh.log -> logs/bot_console.log -> logs/bot.jsonl",
    )
    args = parser.parse_args()

    log_path = _resolve_log_path(args.log)
    rows = _load_rows(log_path)
    if not rows:
        raise RuntimeError(f"No parseable JSON log rows found in {log_path}")

    restart_ts = _parse_iso_dt(args.restart_ts) if args.restart_ts else _infer_restart_ts(rows)
    if restart_ts is None:
        raise RuntimeError(
            "Could not infer restart timestamp from bot_starting/bot_running. Provide --restart-ts explicitly."
        )

    strict_start = restart_ts - timedelta(hours=1)
    strict_end = restart_ts
    aggr_start = restart_ts
    aggr_end = restart_ts + timedelta(minutes=30)

    strict_rows = _window(rows, strict_start, strict_end)
    aggr_rows = _window(rows, aggr_start, aggr_end)

    latest_log_ts = max(r.t for r in rows)
    aggr_complete = latest_log_ts >= aggr_end

    panic_names = {"panic_signal_fired", "panic_signal", "PanicSignal"}
    drift_names = {"drift_signal_fired", "drift_signal", "DriftSignal"}

    metrics = [
        ("panic_signals", lambda rr: _count_any(rr, panic_names)),
        ("drift_signals", lambda rr: _count_any(rr, drift_names)),
        ("order_placed_paper", lambda rr: _count_any(rr, {"order_placed_paper"})),
        ("entry_chaser_cancelled", lambda rr: _count_any(rr, {"entry_chaser_cancelled"})),
        ("chaser_escalating", lambda rr: _count_contains(rr, "chaser_escalat")),
        ("fills_like_events", _count_fill_like),
        ("position_opened", lambda rr: _count_any(rr, {"position_opened"})),
    ]

    strict_counts: dict[str, int] = {k: fn(strict_rows) for k, fn in metrics}
    aggr_counts: dict[str, int] = {k: fn(aggr_rows) for k, fn in metrics}

    _print_section("Local Delta Audit")
    print(f"log_path: {log_path}")
    print(f"restart_ts_utc: {restart_ts.isoformat()}")
    print(f"strict_window_utc: {strict_start.isoformat()} -> {strict_end.isoformat()}")
    print(f"aggressive_window_utc: {aggr_start.isoformat()} -> {aggr_end.isoformat()}")
    print(f"aggressive_window_complete: {aggr_complete}")
    if not aggr_complete:
        print(f"latest_log_ts_utc: {latest_log_ts.isoformat()}")
        print("note: aggressive 30m window is not fully available yet.")

    _print_section("Counts")
    print(f"{'metric':<24} {'strict_1h':>10} {'aggr_30m':>10} {'delta':>10} {'aggr_x2_vs_strict':>18}")
    for key, _ in metrics:
        s = strict_counts[key]
        a = aggr_counts[key]
        delta = a - s
        aggr_x2_vs_strict = (a * 2) - s
        print(f"{key:<24} {s:>10} {a:>10} {delta:>10} {aggr_x2_vs_strict:>18}")

    _print_section("Primary Fill KPI")
    strict_fill_kpi = strict_counts["fills_like_events"] + strict_counts["position_opened"]
    aggr_fill_kpi = aggr_counts["fills_like_events"] + aggr_counts["position_opened"]
    print(f"strict_fill_or_position_opened: {strict_fill_kpi}")
    print(f"aggr_fill_or_position_opened: {aggr_fill_kpi}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
