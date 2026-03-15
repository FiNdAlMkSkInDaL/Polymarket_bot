import json

# Find all bot_starting and bot_stopped events to map sessions
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            ev = d.get("event", "")
            if ev in ("bot_starting", "bot_stopped", "bot_stopping", "final_stats", "panic_signal_fired"):
                if ev == "panic_signal_fired":
                    # just count a summary
                    pass
                else:
                    ts = d.get("timestamp", "?")
                    extra = ""
                    if ev == "final_stats":
                        extra = f" signals={d.get('total_signals',0)} trades={d.get('total_trades',0)}"
                    print(f"{ts}  {ev}{extra}")
        except:
            pass

# Now count panic signals per session window
print("\n--- Panic signal timestamp range ---")
first_panic = None
last_panic = None
count = 0
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get("event") == "panic_signal_fired":
                ts = d.get("timestamp", "")
                if first_panic is None:
                    first_panic = ts
                last_panic = ts
                count += 1
        except:
            pass
print(f"Count: {count}")
print(f"First: {first_panic}")
print(f"Last:  {last_panic}")
