import json, collections, sys
from datetime import datetime

# Count events only from the live session (after 22:20 UTC on March 10)
cutoff = "2026-03-10T22:20:00"
c = collections.Counter()
rejection_reasons = collections.Counter()
total = 0
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            ts = d.get("timestamp", "")
            if ts < cutoff:
                continue
            total += 1
            ev = d.get("event", "?")
            c[ev] += 1
            if ev == "edge_assessment":
                viable = d.get("viable", False)
                reason = d.get("reason", "none")
                rejection_reasons[f"edge: viable={viable} reason={reason}"] += 1
            elif ev == "eqs_rejected":
                reason = d.get("reason", "?")
                rejection_reasons[f"eqs_rejected: {reason}"] += 1
            elif ev == "trend_guard_suppressed":
                rejection_reasons["trend_guard_suppressed"] += 1
            elif ev == "zscore_near_miss":
                rejection_reasons["zscore_near_miss"] += 1
            elif ev == "pce_var_gate":
                blocked = d.get("exceeds_threshold", False)
                rejection_reasons[f"pce_var_gate: blocked={blocked}"] += 1
        except:
            pass

print(f"=== LIVE SESSION EVENTS (after {cutoff}) ===")
print(f"Total events: {total}\n")
print("--- Event Frequency ---")
for ev, cnt in c.most_common(40):
    print(f"{cnt:>6}  {ev}")

print("\n--- Signal Rejection Breakdown ---")
for key, cnt in rejection_reasons.most_common(20):
    print(f"{cnt:>6}  {key}")
