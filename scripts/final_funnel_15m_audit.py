#!/usr/bin/env python3
"""Strict 15-minute bounded funnel baseline audit for active PAPER run."""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("logs/trades.db")
RAW_TICKS_DIR = Path("data/raw_ticks")
OUT_PATH = Path("/tmp/final_funnel_15m_summary.json")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(ts: object) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def db_counts() -> dict[str, int]:
    out = {"trades": 0, "live_positions": 0, "live_orders": 0}
    if not DB_PATH.exists():
        return out
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        for table in out:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                out[table] = int(cur.fetchone()[0] or 0)
            except Exception:
                out[table] = 0
    finally:
        conn.close()
    return out


def count_trade_events_from_raw_ticks(start_dt: datetime, end_dt: datetime) -> int:
    if not RAW_TICKS_DIR.exists():
        return 0
    total = 0
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    ts_keys = ("timestamp", "ts", "time", "t")

    for fp in RAW_TICKS_DIR.rglob("*"):
        if not fp.is_file():
            continue
        try:
            mt = fp.stat().st_mtime
        except Exception:
            continue
        if mt < start_ts - 60 or mt > end_ts + 60:
            continue

        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    event_dt = None
                    for k in ts_keys:
                        if k not in row:
                            continue
                        v = row.get(k)
                        if isinstance(v, (int, float)):
                            try:
                                event_dt = datetime.fromtimestamp(float(v), tz=timezone.utc)
                            except Exception:
                                event_dt = None
                        else:
                            event_dt = parse_iso(v)
                        if event_dt is not None:
                            break
                    if event_dt is None:
                        continue
                    if start_dt <= event_dt <= end_dt:
                        total += 1
        except Exception:
            continue
    return total


def main() -> None:
    start_dt = now_utc()
    start_db = db_counts()

    # Strict bounded observation window: 15 minutes.
    time.sleep(15 * 60)

    end_dt = now_utc()
    end_db = db_counts()

    infra = {
        "stale_bar_flush_tick_total": 0,
        "yes_bars_flushed_total": 0,
        "no_bars_flushed_total": 0,
    }
    ingestion = {
        "l2_synced_total": 0,
        "trade_events_processed_total": 0,
    }
    signals = {
        "panic_signal_evaluations_total": 0,
        "drift_signal_evaluations_total": 0,
    }
    rejections: dict[str, int] = {}

    for fp in sorted(Path("logs").glob("bot.jsonl*")):
        if not fp.is_file():
            continue
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue

                    row_dt = parse_iso(row.get("timestamp"))
                    if row_dt is None or row_dt < start_dt or row_dt > end_dt:
                        continue

                    ev = str(row.get("event", ""))

                    if ev == "stale_bar_flush_tick":
                        infra["stale_bar_flush_tick_total"] += 1
                        infra["yes_bars_flushed_total"] += int(row.get("yes_bars_flushed", 0) or 0)
                        infra["no_bars_flushed_total"] += int(row.get("no_bars_flushed", 0) or 0)

                    if ev == "l2_synced":
                        ingestion["l2_synced_total"] += 1

                    if ev.startswith("panic_signal_"):
                        signals["panic_signal_evaluations_total"] += 1
                    if ev.startswith("drift_signal_"):
                        signals["drift_signal_evaluations_total"] += 1

                    if ev.endswith("_rejected") or "gate" in ev:
                        rejections[ev] = rejections.get(ev, 0) + 1
        except Exception:
            continue

    ingestion["trade_events_processed_total"] = count_trade_events_from_raw_ticks(start_dt, end_dt)

    execution = {
        "trades_delta": int(end_db["trades"] - start_db["trades"]),
        "live_positions_delta": int(end_db["live_positions"] - start_db["live_positions"]),
        "live_orders_delta": int(end_db["live_orders"] - start_db["live_orders"]),
    }

    summary: dict[str, object] = {
        "window_start_utc": start_dt.isoformat().replace("+00:00", "Z"),
        "window_end_utc": end_dt.isoformat().replace("+00:00", "Z"),
        "infrastructure": infra,
        "ingestion": ingestion,
        "signal_generation": signals,
        "gate_rejections_breakdown": dict(sorted(rejections.items(), key=lambda kv: kv[1], reverse=True)),
        "execution_deltas": execution,
    }

    if execution["trades_delta"] == 0 or execution["live_positions_delta"] == 0:
        if rejections:
            top_name, top_count = sorted(rejections.items(), key=lambda kv: kv[1], reverse=True)[0]
            summary["largest_execution_bottleneck"] = {
                "largest_bottleneck_gate": top_name,
                "count": top_count,
            }
        else:
            summary["largest_execution_bottleneck"] = {
                "largest_bottleneck_gate": "no_signal_or_gate_events_observed",
                "count": 0,
            }

    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
