#!/usr/bin/env python3
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# External orchestration performs the 4-minute wait.
wait_seconds = 0
time.sleep(wait_seconds)

root = Path.home() / "polymarket-bot"
log_files = [root / "logs" / "bot.jsonl", root / "logs" / "bot_fresh.log", root / "logs" / "bot.jsonl.1"]
cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)

bar_total = 0
bar_vol_gt0 = 0
bar_examples = []
panic_count = 0
drift_count = 0


def parse_ts(v):
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


for fp in log_files:
    if not fp.exists():
        continue
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue
            ts = parse_ts(d.get("timestamp"))
            if ts is None or ts < cutoff:
                continue

            ev = str(d.get("event", ""))
            if ev == "bar_closed":
                bar_total += 1
                vol = float(d.get("volume", 0) or 0)
                tc = float(d.get("trade_count", 0) or 0)
                if vol > 0 and tc > 0:
                    bar_vol_gt0 += 1
                    if len(bar_examples) < 5:
                        bar_examples.append(
                            {
                                "timestamp": d.get("timestamp"),
                                "asset_id": d.get("asset_id"),
                                "volume": vol,
                                "trade_count": tc,
                            }
                        )
            elif ev == "panic_signal_fired":
                panic_count += 1
            elif ev == "drift_signal_fired":
                drift_count += 1

# active DB
logs_db = root / "logs" / "trades.db"
data_db = root / "data" / "trades.db"
db = logs_db
if data_db.exists() and (not logs_db.exists() or data_db.stat().st_mtime > logs_db.stat().st_mtime):
    db = data_db

live_positions = 0
if db.exists():
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM live_positions")
        live_positions = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()

print(json.dumps({
    "bar_closed_total_last5m": bar_total,
    "bar_closed_with_volume_tradecount_gt0_last5m": bar_vol_gt0,
    "bar_examples": bar_examples,
    "panic_signal_fired_last5m": panic_count,
    "drift_signal_fired_last5m": drift_count,
    "live_positions": live_positions,
    "db_path": str(db),
}, indent=2))
