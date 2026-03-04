"""Scan bot.jsonl for all unique event types and count them."""
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(r"C:\vps_dump")

event_counts = Counter()
signal_events = []  # Collect signal-relevant events

SIGNAL_KEYWORDS = {
    "panic", "signal", "entry", "exit", "position", "take_profit",
    "stop_loss", "trade", "edge", "zscore", "alpha", "order",
    "fill", "chaser", "tp_rescale", "open_position", "close_position",
    "rpe", "whale", "sizer", "kelly",
}

for log_file in sorted(DATA_DIR.glob("bot.jsonl*")):
    print(f"Scanning {log_file.name}...")
    with open(log_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                event = obj.get("event", "UNKNOWN")
                event_counts[event] += 1
                # Collect signal-relevant events
                if any(kw in event.lower() for kw in SIGNAL_KEYWORDS):
                    signal_events.append(obj)
            except json.JSONDecodeError:
                event_counts["_PARSE_ERROR"] += 1

print(f"\n=== Event type distribution ({sum(event_counts.values())} total lines) ===")
for event, count in event_counts.most_common(60):
    print(f"  {count:>6}  {event}")

print(f"\n=== Signal-relevant events: {len(signal_events)} ===")

# Show examples of each signal event type
signal_types = Counter(e["event"] for e in signal_events)
print("\nSignal event breakdown:")
for event, count in signal_types.most_common():
    print(f"  {count:>6}  {event}")

# Show a few examples of the most interesting events
for event_name in ["panic_signal_fired", "position_opened", "position_closed",
                   "take_profit_computed", "stop_loss_triggered", "entry_signal",
                   "signal_evaluated", "edge_quality_check", "trade_executed",
                   "chaser_fill", "tp_rescale"]:
    examples = [e for e in signal_events if e["event"] == event_name]
    if examples:
        print(f"\n--- Example: {event_name} ({len(examples)} total) ---")
        for ex in examples[:2]:
            print(json.dumps(ex, indent=2, default=str))
