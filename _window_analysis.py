import json

# Check if edge_assessment events are from backtests or live sessions
# Backtests run during 18:35-18:42 pipeline phase
# Live session starts after bot_starting events

print("=== Edge Assessment Timestamps ===")
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    first_ea = None
    last_ea = None
    ea_count = 0
    for line in f:
        try:
            d = json.loads(line)
            if d.get("event") == "edge_assessment":
                ts = d.get("timestamp", "")
                ea_count += 1
                if first_ea is None:
                    first_ea = ts
                last_ea = ts
        except:
            pass
print(f"Total edge_assessments: {ea_count}")
print(f"First: {first_ea}")
print(f"Last:  {last_ea}")

# Now count per-session
print("\n=== Edge Assessments by time window ===")
windows = {
    "backtest_phase (18:35-18:50)": ("2026-03-10T18:35:00", "2026-03-10T18:50:00"),
    "live_session_1 (18:50-20:13)": ("2026-03-10T18:50:00", "2026-03-10T20:13:00"),
    "live_session_2 (20:13-20:26)": ("2026-03-10T20:13:00", "2026-03-10T20:26:00"),
    "live_session_3 (20:26-22:20)": ("2026-03-10T20:26:00", "2026-03-10T22:20:00"),
    "live_session_4 (22:20+)":      ("2026-03-10T22:20:00", "2026-03-10T23:59:00"),
}
for name, (start, end) in windows.items():
    count = 0
    viable_count = 0
    with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
        for line in f:
            try:
                d = json.loads(line)
                ts = d.get("timestamp", "")
                if d.get("event") == "edge_assessment" and start <= ts <= end:
                    count += 1
                    if d.get("viable"):
                        viable_count += 1
            except:
                pass
    print(f"  {name}: {count} total, {viable_count} viable")

# Count panic signals per window
print("\n=== Panic Signals by time window ===")
for name, (start, end) in windows.items():
    count = 0
    with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
        for line in f:
            try:
                d = json.loads(line)
                ts = d.get("timestamp", "")
                if d.get("event") == "panic_signal_fired" and start <= ts <= end:
                    count += 1
            except:
                pass
    print(f"  {name}: {count}")

# Count eqs_rejected per window
print("\n=== EQS Rejected by time window ===")
for name, (start, end) in windows.items():
    count = 0
    with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
        for line in f:
            try:
                d = json.loads(line)
                ts = d.get("timestamp", "")
                if d.get("event") == "eqs_rejected" and start <= ts <= end:
                    count += 1
            except:
                pass
    print(f"  {name}: {count}")
