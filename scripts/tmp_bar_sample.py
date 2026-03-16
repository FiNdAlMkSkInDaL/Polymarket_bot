#!/usr/bin/env python3
import json
from pathlib import Path

root = Path.home() / "polymarket-bot" / "logs"
files = [root / "bot.jsonl", root / "bot_fresh.log", root / "bot.jsonl.1", root / "bot.jsonl.2"]
rows = []
for fp in files:
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
            if d.get("event") == "bar_closed":
                rows.append((d.get("timestamp"), d.get("asset_id"), d.get("volume"), d.get("trade_count"), fp.name))

print("bar_closed_count", len(rows))
for r in rows[-5:]:
    print(r)
