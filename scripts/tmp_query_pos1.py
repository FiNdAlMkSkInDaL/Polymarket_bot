#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

db = Path("logs/trades.db")
conn = sqlite3.connect(str(db))
try:
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(live_positions)").fetchall()]
    wanted = [
        "id",
        "state",
        "unrealised_pnl_cents",
        "entry_price",
        "target_price",
        "stop_loss_trigger",
        "exit_order_id",
        "created_at",
        "updated_at",
    ]
    selected = [c for c in wanted if c in cols]
    if not selected:
        print(json.dumps({"found": False, "id": "POS-1", "error": "no_expected_columns", "columns": cols}))
        raise SystemExit(0)

    sql = f"SELECT {', '.join(selected)} FROM live_positions WHERE id = ?"
    row = cur.execute(sql, ("POS-1",)).fetchone()
    if row is None:
        print(json.dumps({"found": False, "id": "POS-1", "columns": cols}))
    else:
        print(json.dumps({"found": True, "columns": selected, **dict(zip(selected, row))}, default=str))
finally:
    conn.close()
