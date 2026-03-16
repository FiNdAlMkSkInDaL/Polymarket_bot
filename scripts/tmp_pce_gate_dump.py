#!/usr/bin/env python3
import json
from pathlib import Path

root = Path.home() / "polymarket-bot" / "logs"
files = [root / "bot.jsonl", root / "bot_fresh.log", root / "bot.jsonl.1", root / "bot.jsonl.2"]

hits = []
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
            if ev in {"pce_var_gate", "pce_var_gate_blocked", "pce_var_gate_exceeded"}:
                hits.append((fp.name, d))

print("total_hits", len(hits))
for name, d in hits[-10:]:
    out = {
        "file": name,
        "timestamp": d.get("timestamp"),
        "event": d.get("event"),
        "portfolio_var": d.get("portfolio_var"),
        "var_usd": d.get("var_usd"),
        "projected_var_usd": d.get("projected_var_usd"),
        "max_var_usd": d.get("max_var_usd"),
        "allowed": d.get("allowed"),
        "reason": d.get("reason"),
        "raw": d,
    }
    print(json.dumps(out, ensure_ascii=True))
