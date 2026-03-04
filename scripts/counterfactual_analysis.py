#!/usr/bin/env python3
"""
Counterfactual analysis: OLD logic vs NEW logic on real VPS data.

For each alpha-relevant event from the bot logs, this script computes
what the OLD code would have produced vs what the NEW code produces,
then summarises the differences and their PnL impact.

Data sources:
  - C:\vps_dump\bot.jsonl*      → structured log events
  - C:\vps_dump\raw_ticks\...   → raw tick data for forward-price lookup
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(r"C:\vps_dump")

# ══════════════════════════════════════════════════════════════════════════
#  Config defaults (mirroring src/core/config.py)
# ══════════════════════════════════════════════════════════════════════════
ALPHA_DEFAULT = 0.50
ALPHA_MIN = 0.40
ALPHA_MAX = 0.70
ZSCORE_THRESHOLD = 2.0       # config had 1.5 for detector, but logs show threshold=2.0
VOLUME_RATIO_THRESHOLD = 1.2
MIN_SPREAD_CENTS = 4.0
DESIRED_MARGIN_CENTS = 2.5
FEE_MAX = 0.0156


def get_fee_rate(p: float) -> float:
    """Quadratic fee curve: Fee(p) = f_max * 4 * p * (1-p)"""
    return FEE_MAX * 4.0 * p * (1.0 - p)


# ══════════════════════════════════════════════════════════════════════════
#  Load all relevant log events
# ══════════════════════════════════════════════════════════════════════════
def load_events() -> list[dict]:
    events = []
    for log_file in sorted(DATA_DIR.glob("bot.jsonl*")):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    events.sort(key=lambda e: e.get("timestamp", ""))
    return events


# ══════════════════════════════════════════════════════════════════════════
#  Load raw tick data for forward-price simulation
# ══════════════════════════════════════════════════════════════════════════
def load_trades_for_asset(asset_id: str) -> list[dict]:
    """Load trade ticks for a specific asset across all dates."""
    trades = []
    for date_dir in sorted(DATA_DIR.glob("raw_ticks/*/*")):
        for tick_file in date_dir.glob("*.jsonl"):
            # Match by asset_id prefix in filename or content
            if not (asset_id[:10] in str(tick_file)):
                continue
            with open(tick_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("source") == "trade":
                            for pc in obj.get("payload", {}).get("price_changes", []):
                                trades.append({
                                    "ts": obj["local_ts"],
                                    "price": float(pc["price"]),
                                    "size": float(pc["size"]),
                                    "asset_id": pc["asset_id"],
                                    "side": pc["side"],
                                })
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
    trades.sort(key=lambda t: t["ts"])
    return trades


def find_forward_price(trades: list[dict], entry_ts: float, horizon_min: float = 10.0) -> float | None:
    """Find the mid-price `horizon_min` minutes after entry_ts."""
    target_ts = entry_ts + horizon_min * 60
    best = None
    for t in trades:
        if t["ts"] >= target_ts:
            best = t["price"]
            break
    return best


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: Take-Profit α — OLD vs NEW
# ══════════════════════════════════════════════════════════════════════════
def compute_alpha_old(entry: float, vwap: float, realised_vol: float = 0.0,
                      whale: bool = False, days_to_res: float = 365.0,
                      book_depth_ratio: float = 1.0) -> float:
    """OLD take-profit alpha: symmetric vol penalty."""
    alpha = ALPHA_DEFAULT

    # Old: symmetric vol adjustment (always decreases alpha for high vol)
    if realised_vol > 0:
        vol_factor = min(realised_vol / 0.02, 3.0)
        if vol_factor > 1.0:
            alpha -= 0.05 * (vol_factor - 1.0)  # OLD: decreased α
        else:
            alpha -= 0.08 * (1.0 - vol_factor)

    if book_depth_ratio > 1.0:
        alpha += 0.03 * min(book_depth_ratio - 1.0, 3.0)

    if whale:
        alpha += 0.08

    if days_to_res < 14:
        alpha -= 0.05 * (1.0 - days_to_res / 14.0)

    # OLD: no VWAP proximity factor

    return max(ALPHA_MIN, min(ALPHA_MAX, alpha))


def compute_alpha_new(entry: float, vwap: float, realised_vol: float = 0.0,
                      whale: bool = False, days_to_res: float = 365.0,
                      book_depth_ratio: float = 1.0) -> float:
    """NEW take-profit alpha: asymmetric vol + VWAP proximity."""
    alpha = ALPHA_DEFAULT

    # NEW: asymmetric vol
    if realised_vol > 0:
        vol_factor = min(realised_vol / 0.02, 3.0)
        if vol_factor > 1.0:
            alpha += 0.04 * (vol_factor - 1.0)  # NEW: increases α
        else:
            alpha -= 0.08 * (1.0 - vol_factor)

    if book_depth_ratio > 1.0:
        alpha += 0.03 * min(book_depth_ratio - 1.0, 3.0)

    if whale:
        alpha += 0.08

    if days_to_res < 14:
        alpha -= 0.05 * (1.0 - days_to_res / 14.0)

    # NEW: VWAP proximity factor
    if vwap > entry and entry > 0:
        discount_pct = (vwap - entry) / vwap
        if discount_pct > 0.10:
            alpha += 0.04 * min(discount_pct / 0.20, 1.0)
        elif discount_pct < 0.02:
            alpha -= 0.05

    return max(ALPHA_MIN, min(ALPHA_MAX, alpha))


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: Edge Filter signal_quality — OLD vs NEW
# ══════════════════════════════════════════════════════════════════════════
def signal_quality_old(zscore: float, v_ratio: float,
                       z_thresh: float = ZSCORE_THRESHOLD,
                       v_thresh: float = VOLUME_RATIO_THRESHOLD,
                       whale: bool = False) -> float:
    """OLD signal quality: linear, unbounded."""
    z_excess = max(0.0, (zscore - z_thresh) / z_thresh) if z_thresh > 0 else 0.0
    v_excess = max(0.0, (v_ratio - v_thresh) / v_thresh) if v_thresh > 0 else 0.0
    q = min(1.0, 0.5 + 0.25 * z_excess + 0.25 * v_excess)
    if whale:
        q = min(1.0, q + 0.15)
    return q


def signal_quality_new(zscore: float, v_ratio: float,
                       z_thresh: float = ZSCORE_THRESHOLD,
                       v_thresh: float = VOLUME_RATIO_THRESHOLD,
                       whale: bool = False) -> float:
    """NEW signal quality: diminishing returns on extreme z."""
    z_excess = max(0.0, (zscore - z_thresh) / z_thresh) if z_thresh > 0 else 0.0
    v_excess = max(0.0, (v_ratio - v_thresh) / v_thresh) if v_thresh > 0 else 0.0

    # Diminishing returns
    if z_excess <= 2.0:
        z_contribution = min(0.35, 0.25 * z_excess)
    else:
        z_contribution = 0.35 + 0.05 * min(z_excess - 2.0, 2.0)

    q = min(1.0, 0.5 + z_contribution + 0.20 * min(v_excess, 2.0))
    if whale:
        q = min(1.0, q + 0.15)
    return q


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: Panic detector intra-bar momentum discount
# ══════════════════════════════════════════════════════════════════════════
def effective_zscore_old(raw_zscore: float) -> float:
    """OLD: no intra-bar discount."""
    return raw_zscore


def effective_zscore_new(raw_zscore: float, close_position_in_bar: float = 0.5) -> float:
    """NEW: discount if close is in bottom half of bar range.
    close_position_in_bar: (close - low) / (high - low), in [0, 1].
    """
    if close_position_in_bar < 0.5 and raw_zscore > 0:
        factor = 0.5 + close_position_in_bar  # [0.5, 1.0)
        return raw_zscore * factor
    return raw_zscore


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4: TP anti-ratchet guard
# ══════════════════════════════════════════════════════════════════════════
def tp_rescale_old(old_target: float, new_target: float) -> float:
    """OLD: no guard, new target used as-is."""
    return new_target


def tp_rescale_new(old_target: float, new_target: float, original_target: float) -> float:
    """NEW: cap at original target — can only tighten, never widen."""
    return min(new_target, original_target)


# ══════════════════════════════════════════════════════════════════════════
#  Main analysis
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  COUNTERFACTUAL ANALYSIS: OLD LOGIC vs NEW LOGIC")
    print("  Data: VPS logs from March 2-4, 2026")
    print("=" * 80)

    all_events = load_events()
    print(f"\nLoaded {len(all_events)} total log events\n")

    # ── 1. Take-Profit α analysis ──────────────────────────────────────
    tp_events = [e for e in all_events if e["event"] == "take_profit_computed"]
    print("=" * 80)
    print(f"  1. TAKE-PROFIT ALPHA ANALYSIS ({len(tp_events)} events)")
    print("=" * 80)
    print(f"\n{'Timestamp':<28} {'Entry':>6} {'Target':>7} {'Logged α':>8} "
          f"{'OLD α':>6} {'NEW α':>6} {'Δα':>6} {'OLD tgt':>8} {'NEW tgt':>8}")
    print("-" * 110)

    tp_alpha_diffs = []
    for e in tp_events:
        entry = e["entry"]
        target = e["target"]
        logged_alpha = e["alpha"]
        spread = e["spread_cents"]

        # We don't have realised_vol or VWAP in the log, but we can derive VWAP
        # from entry + alpha: target = entry + alpha * (vwap - entry)
        # → vwap = (target - entry) / alpha + entry  [if alpha > 0]
        if logged_alpha > 0:
            implied_vwap = entry + (target - entry) / logged_alpha
        else:
            implied_vwap = entry + 0.10  # fallback

        # Without realised_vol in logs, test at vol=0 (no vol adjustment)
        # and also at reasonable vol levels
        old_alpha = compute_alpha_old(entry, implied_vwap)
        new_alpha = compute_alpha_new(entry, implied_vwap)

        old_target = entry + old_alpha * (implied_vwap - entry)
        new_target = entry + new_alpha * (implied_vwap - entry)

        diff = new_alpha - old_alpha
        tp_alpha_diffs.append(diff)

        ts = e.get("timestamp", "")[:27]
        print(f"{ts:<28} {entry:>6.3f} {target:>7.4f} {logged_alpha:>8.4f} "
              f"{old_alpha:>6.3f} {new_alpha:>6.3f} {diff:>+6.3f} "
              f"{old_target:>8.4f} {new_target:>8.4f}")

    if tp_alpha_diffs:
        avg_diff = sum(tp_alpha_diffs) / len(tp_alpha_diffs)
        print(f"\n  Average α change (NEW - OLD): {avg_diff:+.4f}")
        print(f"  NEW α is {'HIGHER' if avg_diff > 0 else 'LOWER'} → "
              f"{'holds longer, targets further' if avg_diff > 0 else 'exits sooner, targets closer'}")

    # ── 2. Edge Filter Signal Quality analysis ─────────────────────────
    edge_events = [e for e in all_events if e["event"] == "edge_assessment"]
    print(f"\n{'=' * 80}")
    print(f"  2. EDGE FILTER SIGNAL QUALITY ({len(edge_events)} events)")
    print("=" * 80)

    panic_events = [e for e in all_events if e["event"] == "panic_signal_fired"]
    # Build a lookup: market → list of panic z-scores
    panic_by_market = defaultdict(list)
    for p in panic_events:
        panic_by_market[p["market"]].append(p)

    print(f"\n{'Timestamp':<28} {'Entry':>6} {'VWAP':>6} {'z':>5} {'v_ratio':>7} "
          f"{'OLD sq':>7} {'NEW sq':>7} {'Δsq':>6} {'Viable':>6} {'Reason'}")
    print("-" * 120)

    sq_diffs = []
    viable_changes = []
    for e in edge_events:
        entry = e["entry"]
        vwap = e["vwap"]
        logged_sq = e["signal_q"]
        viable = e["viable"]
        reason = e.get("reason", "")

        # Find the corresponding panic signal to get z-score and v_ratio
        # Use the closest panic event by timestamp for the same market
        # (edge_assessment doesn't log market, but we can match by timestamp proximity)
        ts = e.get("timestamp", "")

        # The edge_assessment doesn't include z-score/v_ratio directly,
        # but the signal_quality factor is computed from them.
        # We can infer: if logged_sq > 0.5, there was excess z/v.
        # For the comparison, we need the raw z and v values.
        # Let's find the nearest panic_signal_fired before this edge event.
        nearest_panic = None
        for p in panic_events:
            if p["timestamp"] <= ts:
                nearest_panic = p

        if nearest_panic:
            z = nearest_panic["zscore"]
            v = nearest_panic["v_ratio"]
            whale = nearest_panic.get("whale", False)
        else:
            # No panic signal before this edge — might be a spread signal
            z = 0
            v = 0
            whale = False

        old_sq = signal_quality_old(z, v, whale=whale)
        new_sq = signal_quality_new(z, v, whale=whale)
        diff = new_sq - old_sq
        sq_diffs.append(diff)

        # Check if viability would change with the EQS
        # (signal_quality affects the geometric mean EQS score)
        viable_old = logged_sq  # This is what was logged
        viable_changes.append({
            "entry": entry, "old_sq": old_sq, "new_sq": new_sq,
            "diff": diff, "viable": viable, "reason": reason,
        })

        print(f"{ts:<28} {entry:>6.3f} {vwap:>6.3f} {z:>5.2f} {v:>7.2f} "
              f"{old_sq:>7.4f} {new_sq:>7.4f} {diff:>+6.4f} {'✓' if viable else '✗':>6} {reason}")

    if sq_diffs:
        avg_sq_diff = sum(sq_diffs) / len(sq_diffs)
        print(f"\n  Average signal_quality change: {avg_sq_diff:+.4f}")
        decreases = sum(1 for d in sq_diffs if d < -0.001)
        increases = sum(1 for d in sq_diffs if d > 0.001)
        neutral = len(sq_diffs) - decreases - increases
        print(f"  Decreased: {decreases}, Increased: {increases}, Neutral: {neutral}")

    # ── 3. Panic Detector — z-score and near-misses ────────────────────
    near_misses = [e for e in all_events if e["event"] == "zscore_near_miss"]
    print(f"\n{'=' * 80}")
    print(f"  3. PANIC DETECTOR ANALYSIS ({len(panic_events)} fires, {len(near_misses)} near-misses)")
    print("=" * 80)

    print(f"\n  3a. Fired signals — intra-bar momentum discount effect:")
    print(f"  (Without bar OHLC data from raw ticks, showing sensitivity analysis)")
    print(f"\n{'Timestamp':<28} {'Market':<12} {'Raw Z':>6} {'Z@0.5cp':>7} {'Z@0.3cp':>7} "
          f"{'Z@0.1cp':>7} {'Would filter?'}")
    print("-" * 100)

    filtered_at = {"0.5": 0, "0.3": 0, "0.1": 0}
    for p in panic_events:
        z = p["zscore"]
        ts = p.get("timestamp", "")[:27]
        mkt = p["market"][-8:]

        z_05 = effective_zscore_new(z, 0.5)
        z_03 = effective_zscore_new(z, 0.3)
        z_01 = effective_zscore_new(z, 0.1)

        filter_03 = z_03 < ZSCORE_THRESHOLD
        filter_01 = z_01 < ZSCORE_THRESHOLD

        if z_05 < ZSCORE_THRESHOLD: filtered_at["0.5"] += 1
        if filter_03: filtered_at["0.3"] += 1
        if filter_01: filtered_at["0.1"] += 1

        filter_label = ""
        if filter_01:
            filter_label = "FILTERED @0.1"
        elif filter_03:
            filter_label = "FILTERED @0.3"

        print(f"{ts:<28} ...{mkt:<12} {z:>6.3f} {z_05:>7.3f} {z_03:>7.3f} "
              f"{z_01:>7.3f} {filter_label}")

    print(f"\n  Would be filtered at close_pos=0.5: {filtered_at['0.5']}/{len(panic_events)}")
    print(f"  Would be filtered at close_pos=0.3: {filtered_at['0.3']}/{len(panic_events)}")
    print(f"  Would be filtered at close_pos=0.1: {filtered_at['0.1']}/{len(panic_events)}")

    print(f"\n  3b. Near-misses — would ANY become fires with different threshold?")
    z_1_5_fires = sum(1 for e in near_misses if e["zscore"] >= 1.5)
    v_pass = sum(1 for e in near_misses
                 if e["zscore"] >= ZSCORE_THRESHOLD and e["v_ratio"] >= VOLUME_RATIO_THRESHOLD)
    print(f"  Near-misses that would fire at z_thresh=1.5: {z_1_5_fires}/{len(near_misses)}")
    print(f"  Near-misses that pass BOTH z≥{ZSCORE_THRESHOLD} AND v≥{VOLUME_RATIO_THRESHOLD}: {v_pass}/{len(near_misses)}")

    # ── 4. TP Rescale anti-ratchet ─────────────────────────────────────
    tp_rescale_events = [e for e in all_events if e["event"] == "tp_rescaled"]
    print(f"\n{'=' * 80}")
    print(f"  4. TP RESCALE ANTI-RATCHET ({len(tp_rescale_events)} events)")
    print("=" * 80)

    # Find the position_opened for POS-15 to get the original target
    pos_events = [e for e in all_events if e["event"] == "position_opened"]
    original_targets = {}
    for pe in pos_events:
        original_targets[pe["pos_id"]] = pe["target"]

    for e in tp_rescale_events:
        pos_id = e["pos_id"]
        old_tgt = e["old_target"]
        new_tgt = e["new_target"]
        orig_tgt = original_targets.get(pos_id, old_tgt)

        # What old logic would do (no cap)
        old_logic_tgt = new_tgt  # old logic just uses the new target as-is

        # What new logic does (cap at original)
        new_logic_tgt = min(new_tgt, orig_tgt)

        print(f"\n  Position: {pos_id}")
        print(f"    Original target at open: {orig_tgt:.4f}")
        print(f"    Pre-rescale target:      {old_tgt:.4f}")
        print(f"    Rescaled to:             {new_tgt:.4f}")
        print(f"    OLD logic would set:     {old_logic_tgt:.4f} {'(would have widened!)' if old_logic_tgt > orig_tgt else '(tightened ✓)'}")
        print(f"    NEW logic caps at:       {new_logic_tgt:.4f} {'(capped ✓)' if new_logic_tgt < old_logic_tgt else '(same)'}")

    # ── 5. The actual trade — PnL trace ────────────────────────────────
    pos_closed = [e for e in all_events if e["event"] == "position_closed"]
    print(f"\n{'=' * 80}")
    print(f"  5. ACTUAL TRADE PnL ANALYSIS ({len(pos_closed)} closed positions)")
    print("=" * 80)

    for e in pos_closed:
        pos_id = e["pos_id"]
        entry = e["entry"]
        exit_p = e["exit"]
        pnl = e["pnl_cents"]
        reason = e["reason"]
        hold_s = e["hold_seconds"]

        orig_data = next((p for p in pos_events if p["pos_id"] == pos_id), None)

        print(f"\n  {pos_id}: Entry={entry:.4f}, Exit={exit_p:.4f}, PnL={pnl:+.2f}¢, "
              f"Reason={reason}, Hold={hold_s:.0f}s")
        if orig_data:
            print(f"    Original target: {orig_data['target']:.4f}")
            print(f"    Alpha used: {orig_data['alpha']:.4f}")
            print(f"    Entry fee: {get_fee_rate(entry)*100:.2f}¢, "
                  f"Exit fee: {get_fee_rate(exit_p)*100:.2f}¢, "
                  f"Total fees: {(get_fee_rate(entry)+get_fee_rate(exit_p))*100:.2f}¢")

        # Simulate what different α values would have done
        if orig_data:
            implied_vwap = entry + (orig_data['target'] - entry) / orig_data['alpha']
            print(f"    Implied VWAP: {implied_vwap:.4f}")

            for label, alpha_fn in [("OLD", compute_alpha_old), ("NEW", compute_alpha_new)]:
                a = alpha_fn(entry, implied_vwap)
                tgt = entry + a * (implied_vwap - entry)
                # Check: would the exit have been different?
                would_tp = exit_p >= tgt
                print(f"    {label} α={a:.4f} → target={tgt:.4f}, "
                      f"exit{'≥' if would_tp else '<'}target → "
                      f"{'TP hit' if would_tp else 'TP NOT hit'}")

    # ── 6. Forward price analysis from raw ticks ───────────────────────
    print(f"\n{'=' * 80}")
    print(f"  6. FORWARD PRICE ANALYSIS (what happened AFTER signals)")
    print("=" * 80)

    # For each panic signal, look at what the NO token price did afterward
    # This tells us if the mean-reversion bet would have been profitable
    market_ids = list(set(p["market"] for p in panic_events))

    for market_id in market_ids[:3]:  # Limit to first 3 markets
        market_panics = [p for p in panic_events if p["market"] == market_id]
        print(f"\n  Market: ...{market_id[-12:]}")
        print(f"  Panic fires: {len(market_panics)}")

        # Show the NO ask price trajectory during panic signals
        print(f"\n  {'Timestamp':<28} {'Z':>5} {'V':>5} {'YES':>5} {'NO ask':>6}")
        print(f"  {'-'*60}")
        for p in market_panics:
            ts = p.get("timestamp", "")[:27]
            print(f"  {ts:<28} {p['zscore']:>5.2f} {p['v_ratio']:>5.2f} "
                  f"{p['yes_price']:>5.2f} {p['no_ask']:>6.2f}")

        # Check if mean-reversion actually happened
        if len(market_panics) >= 2:
            first_no = market_panics[0]["no_ask"]
            last_no = market_panics[-1]["no_ask"]
            first_yes = market_panics[0]["yes_price"]
            last_yes = market_panics[-1]["yes_price"]
            print(f"\n  YES price drift: {first_yes:.2f} → {last_yes:.2f} ({last_yes - first_yes:+.2f})")
            print(f"  NO ask drift:    {first_no:.2f} → {last_no:.2f} ({last_no - first_no:+.2f})")
            print(f"  → The NO token {'CHEAPENED' if last_no < first_no else 'RECOVERED'} "
                  f"(our entry side {'got better' if last_no < first_no else 'got worse'})")

    # ── 7. Edge assessment viability analysis ──────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  7. EDGE ASSESSMENT VIABILITY BREAKDOWN")
    print("=" * 80)

    reason_counts = defaultdict(int)
    for e in edge_events:
        if not e["viable"]:
            reason_counts[e.get("reason", "unknown")] += 1

    viable_count = sum(1 for e in edge_events if e["viable"])
    print(f"\n  Viable: {viable_count}/{len(edge_events)}")
    print(f"  Rejected reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:>4}  {reason}")

    # How close were rejected trades to viability?
    close_calls = []
    for e in edge_events:
        if not e["viable"] and e["net_cents"] > -1.0:
            close_calls.append(e)

    if close_calls:
        print(f"\n  Close calls (net > -1.0¢, nearly viable): {len(close_calls)}")
        for e in close_calls:
            print(f"    entry={e['entry']:.2f}, gross={e['gross_cents']:.2f}¢, "
                  f"fees={e['fee_cents']:.2f}¢, net={e['net_cents']:+.2f}¢, "
                  f"signal_q={e['signal_q']:.3f}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    print("""
  Key findings from the real data:

  1. TAKE-PROFIT α:
     - Most entries have entry far below VWAP (large discounts)
     - The VWAP proximity factor will INCREASE α for these trades
     - Without vol data in logs, can't fully assess asymmetric vol impact
     - The logged α values (0.40–0.49) suggest vol was reducing α under the old logic

  2. SIGNAL QUALITY:
     - The diminishing returns change primarily affects z>4.0 (extreme z)
     - In your data, several signals have z=3.5-4.1 → moderate effect
     - The change DECREASES signal_q for extreme z-scores → more conservative

  3. PANIC DETECTOR:
     - Without bar OHLC in the logs, sensitivity analysis shows:
     - At worst case (close at bar low, cp=0.1), z-score discounted by ~60%
     - Many moderate z-scores (2.0-2.5) would be filtered at cp<0.3
     - This is the RISKIEST change: could filter out valid entries

  4. TP ANTI-RATCHET:
     - In the one rescale observed, target moved DOWN (tightened)
     - The anti-ratchet guard wouldn't have changed this specific case
     - But it's a pure safety fix — prevents pathological widening

  5. ACTUAL TRADING:
     - Only 2 positions opened, 1 closed
     - POS-15: -18.96¢ loss despite exiting at "target"
     - Most edges are eaten by fees (fee_cents > gross_cents)
     - THIS IS THE DOMINANT ISSUE: fee drag > alpha capture
""")


if __name__ == "__main__":
    main()
