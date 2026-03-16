#!/usr/bin/env python3
import sqlite3
import time
from pathlib import Path

# Wait 3.5 minutes to allow fresh bars/execution attempts.
time.sleep(210)

root = Path.home() / "polymarket-bot"
logs_db = root / "logs" / "trades.db"
data_db = root / "data" / "trades.db"

db = logs_db
if data_db.exists() and (not logs_db.exists() or data_db.stat().st_mtime > logs_db.stat().st_mtime):
    db = data_db

live_positions = 0
trades = 0
if db.exists():
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM live_positions")
        live_positions = int(cur.fetchone()[0] or 0)
        cur.execute("SELECT count(*) FROM trades")
        trades = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()

open_hits = 0
for lf in [root / "logs" / "bot.jsonl", root / "logs" / "bot_fresh.log"]:
    if not lf.exists():
        continue
    try:
        with lf.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "open_position" in line or "order_place_success" in line:
                    open_hits += 1
    except Exception:
        pass

print(f"live_positions={live_positions}")
print(f"trades={trades}")
print(f"open_position_logs={'yes' if open_hits > 0 else 'no'}")
