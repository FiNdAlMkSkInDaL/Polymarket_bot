#!/usr/bin/env python3
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

root = Path.home() / "polymarket-bot"
log_files = [root / "logs" / "bot.jsonl", root / "logs" / "bot_fresh.log"]

cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)

bar_total = 0
bar_vol_gt0 = 0
bar_tc_gt0 = 0

alpha = Counter(
    {
        "zscore_low": 0,
        "volume_ratio_low": 0,
        "trend_guard_suppressed": 0,
        "no_discount": 0,
        "low_displacement": 0,
        "high_vol_veto": 0,
        "regime_trend_veto": 0,
    }
)

exec_reason = Counter()
pce_blocks = 0


def parse_ts(v):
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


for fp in log_files:
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

            ts = parse_ts(d.get("timestamp"))
            if ts is None or ts < cutoff:
                continue

            ev = str(d.get("event", ""))

            if ev == "bar_closed":
                bar_total += 1
                try:
                    if float(d.get("volume", 0) or 0) > 0:
                        bar_vol_gt0 += 1
                except Exception:
                    pass
                try:
                    if float(d.get("trade_count", 0) or 0) > 0:
                        bar_tc_gt0 += 1
                except Exception:
                    pass

            if ev == "spike_check_fail_zscore":
                alpha["zscore_low"] += 1
            elif ev == "spike_check_fail_volume":
                alpha["volume_ratio_low"] += 1
            elif ev == "trend_guard_suppressed":
                alpha["trend_guard_suppressed"] += 1
            elif ev == "spike_check_fail_no_not_discounted":
                alpha["no_discount"] += 1
            elif ev == "drift_eval":
                gate = str(d.get("gate", ""))
                passed = bool(d.get("passed", False))
                if not passed and gate == "displacement_below_threshold":
                    alpha["low_displacement"] += 1
                if not passed and gate in {"high_volume_bar", "ewma_vol_bounds"}:
                    alpha["high_vol_veto"] += 1
            elif ev == "meta_controller_veto" and str(d.get("reason", "")) == "regime_trend_veto":
                alpha["regime_trend_veto"] += 1

            if ev == "eqs_rejected":
                reason = str(d.get("reason", "unknown"))
                exec_reason[reason] += 1

            if ev in {"pce_var_gate", "pce_var_gate_blocked", "pce_var_gate_exceeded"}:
                pce_blocks += 1

# Add pce gate as execution bottleneck category
exec_reason["pce_var_gate_block"] += pce_blocks

alpha_top = max(alpha.items(), key=lambda kv: kv[1]) if alpha else ("none", 0)
exec_top = max(exec_reason.items(), key=lambda kv: kv[1]) if exec_reason else ("none", 0)

print(json.dumps(
    {
        "bar_quality": {
            "bar_closed_total": bar_total,
            "bar_closed_volume_gt0": bar_vol_gt0,
            "bar_closed_trade_count_gt0": bar_tc_gt0,
            "bars_have_real_volume": bar_vol_gt0 > 0 and bar_tc_gt0 > 0,
        },
        "alpha_counts": dict(alpha),
        "alpha_top": {"name": alpha_top[0], "count": alpha_top[1]},
        "execution_counts": dict(exec_reason),
        "execution_top": {"name": exec_top[0], "count": exec_top[1]},
    },
    indent=2,
))
