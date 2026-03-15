import json, collections

# Analyze the active signal window: 18:48 to 22:20
start = "2026-03-10T18:48:00"
end   = "2026-03-10T22:21:00"

c = collections.Counter()
rejection_detail = collections.Counter()
panic_markets = collections.Counter()
total_panics = 0

with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            ts = d.get("timestamp", "")
            if ts < start or ts > end:
                continue
            ev = d.get("event", "?")
            c[ev] += 1

            if ev == "panic_signal_fired":
                total_panics += 1
                mid = d.get("market_id", d.get("condition_id", "?"))
                panic_markets[mid] += 1

            elif ev == "edge_assessment":
                viable = d.get("viable", False)
                reason = d.get("reason", "none")
                rejection_detail[f"edge: viable={viable} reason={reason}"] += 1

            elif ev == "eqs_rejected":
                reason = d.get("reason", "?")
                rejection_detail[f"eqs_rejected: {reason}"] += 1

            elif ev == "pce_var_gate":
                blocked = d.get("exceeds_threshold", False)
                rejection_detail[f"pce_var_gate: blocked={blocked}"] += 1

            elif ev in ("trend_guard_suppressed", "zscore_near_miss",
                        "market_rejected", "market_evicted"):
                rejection_detail[ev] += 1

        except:
            pass

print(f"=== SIGNAL WINDOW ({start} -> {end}) ===")
print(f"Total log events: {sum(c.values())}")
print(f"Total panic_signal_fired: {total_panics}\n")

print("--- All Event Types ---")
for ev, cnt in c.most_common(50):
    print(f"{cnt:>6}  {ev}")

print("\n--- Signal Rejection/Filter Breakdown ---")
for key, cnt in rejection_detail.most_common(20):
    print(f"{cnt:>6}  {key}")

print(f"\n--- Unique Markets with Panic Signals: {len(panic_markets)} ---")
for mid, cnt in panic_markets.most_common(10):
    print(f"{cnt:>6}  {mid}")
