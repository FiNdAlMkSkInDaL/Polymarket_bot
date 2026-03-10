import json, collections

# Only live sessions: 18:50 onwards
cutoff = "2026-03-10T18:50:00"
c = collections.Counter()

with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            ts = d.get("timestamp", "")
            if ts < cutoff:
                continue
            ev = d.get("event", "?")
            if ev in ("panic_signal_fired", "edge_assessment", "eqs_rejected",
                      "pce_var_gate", "trend_guard_suppressed", "zscore_near_miss",
                      "market_scored", "market_health", "market_promoted",
                      "market_evicted", "market_rejected", "market_demoted",
                      "periodic_stats", "trade_attempted", "order_placed",
                      "trade_executed", "bot_starting", "bot_stopped",
                      "no_eligible_markets_debug", "stale_trade_eviction"):
                c[ev] += 1
        except:
            pass

print("=== LIVE SESSION SIGNAL PIPELINE (after 18:50 UTC) ===")
for ev, cnt in c.most_common(30):
    print(f"{cnt:>6}  {ev}")

# Show last periodic_stats
print("\n=== Last periodic_stats ===")
last_stats = None
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get("event") == "periodic_stats":
                last_stats = d
        except:
            pass
if last_stats:
    for k, v in sorted(last_stats.items()):
        print(f"  {k}: {v}")

# Show the trend_guard_suppressed events in live
print("\n=== Live trend_guard_suppressed ===")
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            ts = d.get("timestamp", "")
            if ts >= cutoff and d.get("event") == "trend_guard_suppressed":
                print(f"  {ts} market={d.get('market','?')[:20]}... reason={d.get('reason','?')}")
        except:
            pass

# Show no_eligible_markets_debug
print("\n=== no_eligible_markets_debug ===")
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            ts = d.get("timestamp", "")
            if ts >= cutoff and d.get("event") == "no_eligible_markets_debug":
                cands = d.get("candidate_count", d.get("candidates", "?"))
                reason = d.get("reason", "?")
                print(f"  {ts} candidates={cands} reason={reason}")
        except:
            pass
