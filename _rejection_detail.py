import json, collections
c = collections.Counter()
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get("event") == "edge_assessment":
                reason = d.get("reason", "unknown")
                viable = d.get("viable", False)
                c[f"viable={viable} reason={reason}"] += 1
            elif d.get("event") == "eqs_rejected":
                reason = d.get("reason", d.get("gate", "unknown"))
                c[f"eqs_rejected: {reason}"] += 1
            elif d.get("event") == "pce_var_gate":
                action = d.get("action", d.get("result", "unknown"))
                c[f"pce_var_gate: {action}"] += 1
            elif d.get("event") == "trend_guard_suppressed":
                c["trend_guard_suppressed"] += 1
            elif d.get("event") == "zscore_near_miss":
                c["zscore_near_miss"] += 1
        except:
            pass
for key, cnt in c.most_common(30):
    print(f"{cnt:>6}  {key}")
