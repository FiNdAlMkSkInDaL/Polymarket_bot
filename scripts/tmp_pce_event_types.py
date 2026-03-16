#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

files = [
    Path("logs/bot.jsonl"),
    Path("logs/bot_fresh.log"),
    Path("logs/bot.jsonl.1"),
    Path("logs/bot.jsonl.2"),
    Path("logs/bot.jsonl.3"),
]

counts = Counter()
samples = {}
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
            ev = str(d.get("event", ""))
            if "pce_var_gate" in ev:
                counts[ev] += 1
                samples.setdefault(ev, d)

print(dict(counts))
for ev, d in samples.items():
    out = {
        "timestamp": d.get("timestamp"),
        "portfolio_var": d.get("portfolio_var"),
        "var_usd": d.get("var_usd"),
        "projected_var_usd": d.get("projected_var_usd"),
        "max_var_usd": d.get("max_var_usd"),
        "allowed": d.get("allowed"),
        "reason": d.get("reason"),
        "exceeds_threshold": d.get("exceeds_threshold"),
    }
    print(ev, json.dumps(out))
