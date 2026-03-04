"""Extract ALL signal-relevant events with full detail for analysis."""
import json
from pathlib import Path

DATA_DIR = Path(r"C:\vps_dump")

RELEVANT = {
    "panic_signal_fired", "take_profit_computed", "edge_assessment",
    "zscore_near_miss", "spread_signal_fired", "position_opened",
    "position_closed", "order_placed_paper", "paper_fill",
    "sizer_depth_aware", "kelly_cold_start", "tp_rescaled",
    "skip_entry_low_spread", "exit_order_placed", "chaser_filled",
    "chaser_abandoned", "exit_chaser_abandoned", "pce_var_sizing_cap",
}

events = []
for log_file in sorted(DATA_DIR.glob("bot.jsonl*")):
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("event") in RELEVANT:
                    events.append(obj)
            except json.JSONDecodeError:
                pass

# Sort by timestamp
events.sort(key=lambda e: e.get("timestamp", ""))

print(f"Total relevant events: {len(events)}\n")

# Dump all by type
for etype in sorted(set(e["event"] for e in events)):
    subset = [e for e in events if e["event"] == etype]
    print(f"\n{'='*60}")
    print(f"EVENT: {etype} ({len(subset)} occurrences)")
    print(f"{'='*60}")
    for e in subset:
        print(json.dumps(e, indent=2, default=str))
