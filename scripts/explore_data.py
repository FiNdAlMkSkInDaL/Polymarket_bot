"""Quick exploration of the pulled VPS data."""
import sqlite3
import json
from pathlib import Path

DATA_DIR = Path(r"C:\vps_dump")

# ── 1. Trade database ───────────────────────────────────────────────────
db_path = DATA_DIR / "trades.db"
print(f"=== trades.db ({db_path.stat().st_size / 1024:.1f} KB) ===\n")

db = sqlite3.connect(str(db_path))
db.row_factory = sqlite3.Row
cur = db.cursor()

# Tables
tables = cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
for t in tables:
    print(f"TABLE: {t['name']}")
    print(t["sql"])
    count = cur.execute(f"SELECT COUNT(*) FROM [{t['name']}]").fetchone()[0]
    print(f"  → {count} rows\n")

# Completed trades sample
try:
    print("=== Recent completed trades ===")
    rows = cur.execute(
        "SELECT * FROM trades WHERE exit_price IS NOT NULL "
        "ORDER BY exit_time DESC LIMIT 10"
    ).fetchall()
    for r in rows:
        d = dict(r)
        print(json.dumps(d, indent=2, default=str))
        print()
except Exception as e:
    print(f"No completed trades table or error: {e}")

# Summary stats
try:
    stats = cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN pnl_cents < 0 THEN 1 ELSE 0 END) as losers,
            SUM(CASE WHEN pnl_cents = 0 THEN 1 ELSE 0 END) as breakeven,
            ROUND(AVG(pnl_cents), 4) as avg_pnl,
            ROUND(SUM(pnl_cents), 4) as total_pnl,
            ROUND(AVG(hold_seconds), 1) as avg_hold_s,
            ROUND(AVG(alpha), 4) as avg_alpha
        FROM trades
        WHERE exit_price IS NOT NULL
    """).fetchone()
    print("=== Aggregate Stats (completed trades) ===")
    print(json.dumps(dict(stats), indent=2))
except Exception as e:
    print(f"Stats query error: {e}")

db.close()

# ── 2. Structured log sample ────────────────────────────────────────────
print("\n=== bot.jsonl (last 5 lines) ===")
log_path = DATA_DIR / "bot.jsonl"
if log_path.exists():
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    print(f"Total log lines: {len(lines)}")
    for line in lines[-5:]:
        try:
            print(json.dumps(json.loads(line), indent=2))
        except json.JSONDecodeError:
            print(line)
else:
    print("bot.jsonl not found")

# ── 3. Raw tick data summary ────────────────────────────────────────────
print("\n=== Raw tick data summary ===")
for date_dir in sorted((DATA_DIR / "raw_ticks").rglob("*")):
    if date_dir.is_dir():
        jsonl_files = list(date_dir.glob("*.jsonl"))
        if jsonl_files:
            total_size = sum(f.stat().st_size for f in jsonl_files)
            print(f"  {date_dir.name}: {len(jsonl_files)} assets, {total_size / 1024 / 1024:.1f} MB")

# Sample one tick file
tick_dirs = sorted((DATA_DIR / "raw_ticks").rglob("*.jsonl"))
if tick_dirs:
    sample = tick_dirs[0]
    print(f"\nSample tick file: {sample.name}")
    with open(sample, "r") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            try:
                print(json.dumps(json.loads(line), indent=2))
            except json.JSONDecodeError:
                print(line)
