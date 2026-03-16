#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

db = Path("logs/trades.db")
conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row
cur = conn.cursor()

out = {}

# Discover all tables.
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
out["tables"] = tables

for t in tables:
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
    if "id" in cols:
        rows = cur.execute(f"SELECT * FROM {t} WHERE id=?", ("POS-1",)).fetchall()
        if rows:
            out[t] = [dict(r) for r in rows]
    # Some tables use pos_id instead of id.
    if "pos_id" in cols:
        rows = cur.execute(f"SELECT * FROM {t} WHERE pos_id=?", ("POS-1",)).fetchall()
        if rows:
            out[t] = [dict(r) for r in rows]

print(json.dumps(out, default=str))
conn.close()
