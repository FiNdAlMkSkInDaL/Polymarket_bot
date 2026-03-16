#!/usr/bin/env python3
import json
from pathlib import Path

files = [
    Path("logs/bot.jsonl"),
    Path("logs/bot_fresh.log"),
    Path("logs/bot.jsonl.1"),
    Path("logs/bot.jsonl.2"),
    Path("logs/bot.jsonl.3"),
]
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
            if d.get("event") == "RAW_WS_DUMP":
                rows.append((fp.name, d))

print("count", len(rows))
for name, d in rows[-20:]:
    out = {
        "file": name,
        "timestamp": d.get("timestamp"),
        "msg": d.get("msg"),
    }
    print(json.dumps(out, ensure_ascii=True))
