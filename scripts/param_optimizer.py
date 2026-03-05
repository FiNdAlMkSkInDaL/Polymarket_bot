#!/usr/bin/env python3
"""
Parameter optimiser — replay raw tick data, build 1-min OHLCV bars,
compute the exact statistics that each gate evaluates, and output
data-driven parameter recommendations.

Analyses:
  1. Per-market price distribution → optimal tradeable band
  2. 1-min bar z-scores → zscore_threshold calibration
  3. Volume ratio distribution → volume_ratio_threshold calibration
  4. Mean-reversion P&L simulation → actual edge at each threshold
  5. NO discount from VWAP → no_discount_factor calibration
  6. EWMA volatility distribution → drift_vol_ceiling calibration
  7. Displacement distribution → drift_z_threshold calibration
  8. Fee-efficiency / EQS viability at various entry points
  9. Trade frequency analysis → how often signals would fire
  10. Mean-reversion half-life → optimal alpha / TP target
"""
from __future__ import annotations

import json
import math
import os
import sys
import statistics

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ── Data structures ─────────────────────────────────────────────────────
@dataclass
class Bar:
    open_time: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int

@dataclass
class BarBuilder:
    open_time: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0
    vwap_num: float = 0.0
    vwap_den: float = 0.0

    def add(self, price: float, size: float, ts: float):
        if self.tick_count == 0:
            self.open_time = ts
            self.open = price
            self.high = price
            self.low = price
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
        self.close = price
        self.volume += price * size
        self.tick_count += 1
        self.vwap_num += price * size
        self.vwap_den += size

    @property
    def vwap(self) -> float:
        return self.vwap_num / self.vwap_den if self.vwap_den > 0 else 0.0

    def to_bar(self) -> Bar:
        return Bar(self.open_time, self.open, self.high, self.low,
                   self.close, self.volume, self.tick_count)


BAR_DURATION = 60.0  # 1-min bars
LOOKBACK = 60        # 60-bar rolling window for VWAP/vol
EWMA_LAMBDA = 0.94   # RiskMetrics


def build_bars(trades: list[tuple[float, float, float]]) -> list[Bar]:
    """Build 1-min bars from (ts, price, size) tuples."""
    if not trades:
        return []
    trades.sort(key=lambda t: t[0])
    bars = []
    builder = BarBuilder()
    bar_start = trades[0][0]

    for ts, price, size in trades:
        if ts - bar_start >= BAR_DURATION and builder.tick_count > 0:
            bars.append(builder.to_bar())
            builder = BarBuilder()
            bar_start = ts
        builder.add(price, size, ts)

    if builder.tick_count > 0:
        bars.append(builder.to_bar())
    return bars


def rolling_vwap(bars: list[Bar], idx: int, lookback: int = LOOKBACK) -> float:
    start = max(0, idx - lookback + 1)
    total_vol = sum(b.volume for b in bars[start:idx+1])
    if total_vol == 0:
        return 0.0
    # Weighted by bar volume
    return sum(b.close * b.volume for b in bars[start:idx+1]) / total_vol


def rolling_std(bars: list[Bar], idx: int, lookback: int = LOOKBACK) -> float:
    start = max(0, idx - lookback + 1)
    prices = [b.close for b in bars[start:idx+1]]
    if len(prices) < 3:
        return 0.0
    # Log returns std
    returns = [math.log(prices[i]/prices[i-1]) for i in range(1, len(prices))
               if prices[i-1] > 0 and prices[i] > 0]
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns)


def ewma_vol(bars: list[Bar], idx: int) -> float:
    """EWMA volatility (RiskMetrics λ=0.94) up to idx."""
    lam = EWMA_LAMBDA
    var = 0.0
    started = False
    for i in range(1, idx + 1):
        if bars[i-1].close <= 0 or bars[i].close <= 0:
            continue
        ret = math.log(bars[i].close / bars[i-1].close)
        if not started:
            var = ret * ret
            started = True
        else:
            var = lam * var + (1 - lam) * ret * ret
    return math.sqrt(var) if var > 0 else 0.0


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def fee_rate(p: float, f_max: float = 0.0156) -> float:
    """Polymarket parabolic fee: f(p) = f_max * 4 * p * (1-p)."""
    return f_max * 4.0 * p * (1.0 - p)


# ── Load data ───────────────────────────────────────────────────────────
def load_trades_from_dir(data_dir: Path) -> dict[str, list[tuple[float, float, float]]]:
    """Load all trade events, keyed by asset_id → [(ts, price, size)]."""
    trades: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    
    for day_dir in sorted(data_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        for jsonl_file in day_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Only process trade events (not L2)
                    source = rec.get("source", "")
                    if source != "trade":
                        continue
                    
                    payload = rec.get("payload", {})
                    if payload.get("event_type") != "price_change":
                        continue
                    
                    local_ts = rec.get("local_ts", 0)
                    
                    for pc in payload.get("price_changes", []):
                        try:
                            price = float(pc.get("price", 0))
                            size = float(pc.get("size", 0))
                            asset_id = pc.get("asset_id", "")
                            if price > 0 and size > 0 and asset_id:
                                trades[asset_id].append((local_ts, price, size))
                        except (ValueError, TypeError):
                            continue
    return trades


# ── Main analysis ───────────────────────────────────────────────────────
def main():
    data_dir = Path(__file__).parent.parent / "data" / "vps_march2026" / "ticks"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 80)
    print("  PARAMETER OPTIMISER — Raw Tick Data Analysis")
    print("=" * 80)
    print(f"\nLoading trades from {data_dir}...")
    
    all_trades = load_trades_from_dir(data_dir)
    print(f"Loaded {len(all_trades)} unique asset_ids")
    
    # Filter to assets with enough data
    active_assets = {aid: t for aid, t in all_trades.items() if len(t) >= 50}
    print(f"Assets with ≥50 trades: {len(active_assets)}")
    
    total_trades = sum(len(t) for t in active_assets.values())
    print(f"Total trade ticks: {total_trades:,}")

    # ────────────────────────────────────────────────────────────────────
    # 1. Price distribution — where do these markets trade?
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  1. PRICE DISTRIBUTION")
    print("=" * 80)
    
    all_prices = []
    asset_median_prices = {}
    for aid, tlist in active_assets.items():
        prices = [t[1] for t in tlist]
        med = statistics.median(prices)
        asset_median_prices[aid] = med
        all_prices.extend(prices)
    
    # Bin into price ranges
    bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30),
            (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70),
            (0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.0)]
    
    print(f"\n  {'Price Range':>15}  {'Ticks':>10}  {'% of Total':>10}  {'H(p)':>8}")
    for lo, hi in bins:
        count = sum(1 for p in all_prices if lo <= p < hi)
        pct = 100.0 * count / len(all_prices) if all_prices else 0
        mid = (lo + hi) / 2
        h = binary_entropy(mid)
        print(f"  {lo:.2f} – {hi:.2f}       {count:>10,}  {pct:>9.1f}%  {h:>7.3f}")
    
    # How many assets at the tails?
    tail_assets = sum(1 for m in asset_median_prices.values() if m < 0.05 or m > 0.95)
    mid_assets = sum(1 for m in asset_median_prices.values() if 0.15 <= m <= 0.85)
    print(f"\n  Assets at tails (<0.05 or >0.95): {tail_assets}")
    print(f"  Assets in tradeable mid-range (0.15-0.85): {mid_assets}")

    # ────────────────────────────────────────────────────────────────────
    # 2. Build bars and compute z-scores / vol / displacement
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  2. BUILDING 1-MIN BARS + Z-SCORES")
    print("=" * 80)
    
    all_zscores = []
    all_vol_ratios = []
    all_ewma_vols = []
    all_displacements = []
    all_no_discounts = []
    all_spreads = []
    
    # Mean-reversion P&L simulation
    # For each z-score event, simulate: buy at close, sell when reverts to VWAP
    reversion_events = []  # (zscore, entry_price, vwap, reverted_to, max_adverse, bars_to_revert)
    
    markets_with_signal = 0
    bar_count_per_asset = {}
    
    for aid, tlist in sorted(active_assets.items(), key=lambda x: -len(x[1])):
        bars = build_bars(tlist)
        bar_count_per_asset[aid] = len(bars)
        
        if len(bars) < 10:
            continue
        
        # Compute bar volumes for volume ratio
        bar_volumes = [b.volume for b in bars]
        
        had_signal = False
        
        for i in range(5, len(bars)):
            # Rolling stats
            vwap = rolling_vwap(bars, i)
            sigma = rolling_std(bars, i)
            ev = ewma_vol(bars, i)
            
            if sigma <= 0 or vwap <= 0:
                continue
            
            # Z-score
            z = (bars[i].close - vwap) / sigma
            all_zscores.append(z)
            
            # Volume ratio
            avg_vol = statistics.mean(bar_volumes[max(0,i-LOOKBACK):i]) if i > 0 else 1.0
            if avg_vol > 0:
                v_ratio = bars[i].volume / avg_vol
                all_vol_ratios.append(v_ratio)
            
            # EWMA vol
            if ev > 0:
                all_ewma_vols.append(ev)
            
            # Displacement (drift calculation)
            disp = (bars[i].close - vwap) / sigma
            all_displacements.append(disp)
            
            # NO discount check: is current close < vwap * factor?
            if vwap > 0:
                discount = bars[i].close / vwap
                all_no_discounts.append(discount)
            
            # Mean-reversion simulation: when z exceeds threshold,
            # track forward bars to see if price reverts
            abs_z = abs(z)
            if abs_z >= 0.8:  # low threshold to collect lots of events
                had_signal = True
                entry = bars[i].close
                target = vwap  # full reversion target
                
                # Track forward
                max_adverse = 0.0
                bars_to_revert = None
                reverted_to = entry  # worst case: no reversion
                
                for j in range(i + 1, min(i + 61, len(bars))):  # up to 60 bars forward
                    # Reversion: price moves back toward VWAP
                    if z > 0:
                        # Price was above VWAP, we'd short / buy NO
                        move = entry - bars[j].close  # positive = price fell back (good)
                        adverse = bars[j].high - entry  # price went higher (bad)
                    else:
                        # Price was below VWAP, we'd buy
                        move = bars[j].close - entry  # positive = price rose back (good)
                        adverse = entry - bars[j].low  # price went lower (bad)
                    
                    max_adverse = max(max_adverse, adverse)
                    
                    if move >= abs(entry - vwap) * 0.5:  # 50% reversion = our alpha=0.50
                        bars_to_revert = j - i
                        reverted_to = bars[j].close
                        break
                
                reversion_events.append({
                    'zscore': abs_z,
                    'entry': entry,
                    'vwap': vwap,
                    'sigma': sigma,
                    'ewma_vol': ev,
                    'reverted': bars_to_revert is not None,
                    'bars_to_revert': bars_to_revert,
                    'max_adverse': max_adverse,
                    'pnl_cents': (reverted_to - entry) * 100 if z < 0 else (entry - reverted_to) * 100,
                    'vol_ratio': v_ratio if avg_vol > 0 else 1.0,
                    'discount': discount if vwap > 0 else 1.0,
                })
        
        if had_signal:
            markets_with_signal += 1
    
    print(f"\n  Total 1-min bars built: {sum(bar_count_per_asset.values()):,}")
    print(f"  Markets with ≥1 signal event: {markets_with_signal}")
    print(f"  Total z-score observations: {len(all_zscores):,}")
    print(f"  Total reversion events tracked: {len(reversion_events):,}")

    # ────────────────────────────────────────────────────────────────────
    # 3. Z-SCORE DISTRIBUTION
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  3. Z-SCORE DISTRIBUTION (for zscore_threshold)")
    print("=" * 80)
    
    if all_zscores:
        abs_z = [abs(z) for z in all_zscores]
        thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        print(f"\n  {'Threshold':>10}  {'Events':>10}  {'% of bars':>10}  {'Per hour':>10}")
        total_bars = len(all_zscores)
        # Assume ~48 hours of data across 2 days
        hours = 48.0
        for t in thresholds:
            count = sum(1 for z in abs_z if z >= t)
            pct = 100.0 * count / total_bars
            per_hour = count / hours
            print(f"  {t:>10.1f}  {count:>10,}  {pct:>9.2f}%  {per_hour:>9.1f}")

    # ────────────────────────────────────────────────────────────────────
    # 4. VOLUME RATIO DISTRIBUTION
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  4. VOLUME RATIO DISTRIBUTION (for volume_ratio_threshold)")
    print("=" * 80)
    
    if all_vol_ratios:
        vr_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        print(f"\n  {'Threshold':>10}  {'Events ≥':>10}  {'% of bars':>10}")
        for t in vr_thresholds:
            count = sum(1 for v in all_vol_ratios if v >= t)
            pct = 100.0 * count / len(all_vol_ratios)
            print(f"  {t:>10.1f}  {count:>10,}  {pct:>9.2f}%")

    # ────────────────────────────────────────────────────────────────────
    # 5. MEAN-REVERSION P&L BY Z-SCORE THRESHOLD
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  5. MEAN-REVERSION P&L SIMULATION (α=0.50, 60-bar window)")
    print("=" * 80)
    
    if reversion_events:
        thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        print(f"\n  {'z-thresh':>8}  {'Signals':>8}  {'Reverted':>8}  {'Rev%':>6}  "
              f"{'Avg PnL¢':>9}  {'Med PnL¢':>9}  {'AvgAdv¢':>8}  {'Sharpe':>7}")
        
        for t in thresholds:
            events = [e for e in reversion_events if e['zscore'] >= t]
            if not events:
                continue
            rev_count = sum(1 for e in events if e['reverted'])
            rev_pct = 100.0 * rev_count / len(events)
            pnls = [e['pnl_cents'] for e in events]
            avg_pnl = statistics.mean(pnls)
            med_pnl = statistics.median(pnls)
            avg_adv = statistics.mean([e['max_adverse'] * 100 for e in events])
            std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
            print(f"  {t:>8.1f}  {len(events):>8}  {rev_count:>8}  {rev_pct:>5.1f}%  "
                  f"{avg_pnl:>9.2f}  {med_pnl:>9.2f}  {avg_adv:>8.2f}  {sharpe:>7.3f}")

    # ────────────────────────────────────────────────────────────────────
    # 5b. P&L CONDITIONED ON VOLUME RATIO
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  5b. P&L CONDITIONED ON VOLUME RATIO (z >= 1.0)")
    print("=" * 80)
    
    if reversion_events:
        z1_events = [e for e in reversion_events if e['zscore'] >= 1.0]
        vr_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        print(f"\n  {'VR-thresh':>10}  {'Signals':>8}  {'Rev%':>6}  {'Avg PnL¢':>9}  {'Sharpe':>7}")
        
        for vt in vr_thresholds:
            events = [e for e in z1_events if e['vol_ratio'] >= vt]
            if not events:
                continue
            rev_count = sum(1 for e in events if e['reverted'])
            rev_pct = 100.0 * rev_count / len(events)
            pnls = [e['pnl_cents'] for e in events]
            avg_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
            print(f"  {vt:>10.1f}  {len(events):>8}  {rev_pct:>5.1f}%  {avg_pnl:>9.2f}  {sharpe:>7.3f}")

    # ────────────────────────────────────────────────────────────────────
    # 6. NO DISCOUNT DISTRIBUTION (for no_discount_factor)
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  6. NO DISCOUNT DISTRIBUTION (close / VWAP ratio)")
    print("=" * 80)
    
    if all_no_discounts:
        disc_thresholds = [0.96, 0.97, 0.98, 0.99, 0.995, 1.0]
        print(f"\n  {'Factor':>10}  {'Bars below':>12}  {'% of bars':>10}")
        for dt in disc_thresholds:
            count = sum(1 for d in all_no_discounts if d < dt)
            pct = 100.0 * count / len(all_no_discounts)
            print(f"  {dt:>10.3f}  {count:>12,}  {pct:>9.2f}%")

    # ────────────────────────────────────────────────────────────────────
    # 7. EWMA VOLATILITY DISTRIBUTION (for drift_vol_ceiling)
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  7. EWMA VOLATILITY DISTRIBUTION (for drift_vol_ceiling)")
    print("=" * 80)
    
    if all_ewma_vols:
        sorted_vols = sorted(all_ewma_vols)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n  {'Percentile':>12}  {'EWMA σ':>12}")
        for p in percentiles:
            idx = int(len(sorted_vols) * p / 100)
            idx = min(idx, len(sorted_vols) - 1)
            print(f"  {p:>11}%  {sorted_vols[idx]:>12.6f}")
        
        vol_ceilings = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
        print(f"\n  {'Ceiling':>10}  {'Bars below':>12}  {'% of bars':>10}")
        for vc in vol_ceilings:
            count = sum(1 for v in all_ewma_vols if v < vc)
            pct = 100.0 * count / len(all_ewma_vols)
            print(f"  {vc:>10.3f}  {count:>12,}  {pct:>9.2f}%")

    # ────────────────────────────────────────────────────────────────────
    # 8. FEE EFFICIENCY ANALYSIS
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  8. FEE EFFICIENCY ANALYSIS (maker = 0 bps)")
    print("=" * 80)
    
    print(f"\n  {'Entry Price':>12}  {'Fee(entry)':>10}  {'Fee(exit)':>10}  "
          f"{'RT Fee¢':>8}  {'H(p)':>6}  {'Need TP¢':>8}")
    for p in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        # Maker entry = 0 fee, taker exit
        fee_entry = 0.0  # maker
        fee_exit = fee_rate(p + 0.02) * (p + 0.02)  # taker exit ~2¢ above
        rt_fee = (fee_entry + fee_exit) * 100  # in cents
        h = binary_entropy(p)
        need_tp = rt_fee + 1.0  # need to clear fees + 1¢ profit
        print(f"  {p:>12.2f}  {fee_entry*100:>9.2f}¢  {fee_exit*100:>9.2f}¢  "
              f"{rt_fee:>7.2f}¢  {h:>5.3f}  {need_tp:>7.2f}¢")

    # ────────────────────────────────────────────────────────────────────
    # 9. MEAN-REVERSION HALF-LIFE
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  9. MEAN-REVERSION HALF-LIFE (bars to 50% reversion)")
    print("=" * 80)
    
    if reversion_events:
        reverted = [e for e in reversion_events if e['reverted'] and e['zscore'] >= 1.0]
        if reverted:
            half_lives = [e['bars_to_revert'] for e in reverted]
            print(f"\n  Events z≥1.0 that reverted: {len(reverted)}")
            print(f"  Median half-life: {statistics.median(half_lives):.0f} bars ({statistics.median(half_lives):.0f} min)")
            print(f"  Mean half-life:   {statistics.mean(half_lives):.1f} bars")
            print(f"  P25 half-life:    {sorted(half_lives)[len(half_lives)//4]:.0f} bars")
            print(f"  P75 half-life:    {sorted(half_lives)[3*len(half_lives)//4]:.0f} bars")
            
            # By z-score bucket
            print(f"\n  {'Z-bucket':>10}  {'Count':>7}  {'Med HL':>7}  {'Mean HL':>8}")
            for zlo, zhi in [(0.8, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]:
                bucket = [e for e in reverted if zlo <= e['zscore'] < zhi]
                if len(bucket) >= 3:
                    hls = [e['bars_to_revert'] for e in bucket]
                    print(f"  {zlo:.1f}–{zhi:.1f}     {len(bucket):>7}  "
                          f"{statistics.median(hls):>6.0f}m  {statistics.mean(hls):>7.1f}m")

    # ────────────────────────────────────────────────────────────────────
    # 10. EXPECTED VALUE ANALYSIS: signal frequency × avg PnL
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  10. EXPECTED VALUE = signal_freq × avg_pnl (per hour)")
    print("=" * 80)
    
    if reversion_events:
        hours = 48.0  # ~2 days
        thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        print(f"\n  {'z-thresh':>8}  {'Sig/hr':>7}  {'Avg PnL¢':>9}  "
              f"{'EV¢/hr':>8}  {'EV$/day':>8}")

        best_ev = -999
        best_thresh = 0
        for t in thresholds:
            events = [e for e in reversion_events if e['zscore'] >= t]
            if not events:
                continue
            sig_per_hour = len(events) / hours
            pnls = [e['pnl_cents'] for e in events]
            avg_pnl = statistics.mean(pnls)
            ev_per_hour = sig_per_hour * avg_pnl
            ev_per_day = ev_per_hour * 24 / 100  # convert cents to dollars
            marker = " ◀ BEST" if ev_per_hour > best_ev else ""
            if ev_per_hour > best_ev:
                best_ev = ev_per_hour
                best_thresh = t
            print(f"  {t:>8.1f}  {sig_per_hour:>7.1f}  {avg_pnl:>9.2f}  "
                  f"{ev_per_hour:>8.2f}  {ev_per_day:>8.4f}{marker}")
        
        print(f"\n  ★ EV-maximising z-score threshold: {best_thresh}")
    
    # ────────────────────────────────────────────────────────────────────
    # 10b. JOINT OPTIMISATION: z-score × volume ratio
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  10b. JOINT EV OPTIMISATION (z-score × volume_ratio)")
    print("=" * 80)
    
    if reversion_events:
        print(f"\n  {'z':>5}  {'VR':>5}  {'Signals':>8}  {'Rev%':>6}  "
              f"{'Avg PnL¢':>9}  {'EV¢/hr':>8}  {'Sharpe':>7}")
        
        best_joint_ev = -999
        best_joint = (0, 0)
        
        for zt in [0.8, 1.0, 1.2, 1.5]:
            for vt in [0.5, 0.8, 1.0, 1.2, 1.5]:
                events = [e for e in reversion_events
                          if e['zscore'] >= zt and e['vol_ratio'] >= vt]
                if len(events) < 5:
                    continue
                rev_count = sum(1 for e in events if e['reverted'])
                rev_pct = 100.0 * rev_count / len(events)
                pnls = [e['pnl_cents'] for e in events]
                avg_pnl = statistics.mean(pnls)
                std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
                sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
                sig_per_hour = len(events) / hours
                ev_per_hour = sig_per_hour * avg_pnl
                marker = ""
                if ev_per_hour > best_joint_ev:
                    best_joint_ev = ev_per_hour
                    best_joint = (zt, vt)
                    marker = " ★"
                print(f"  {zt:>5.1f}  {vt:>5.1f}  {len(events):>8}  {rev_pct:>5.1f}%  "
                      f"{avg_pnl:>9.2f}  {ev_per_hour:>8.2f}  {sharpe:>7.3f}{marker}")
        
        print(f"\n  ★ EV-maximising joint: z≥{best_joint[0]}, VR≥{best_joint[1]}")

    # ────────────────────────────────────────────────────────────────────
    # 11. EQS VIABILITY AT DIFFERENT THRESHOLDS
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  11. EQS VIABILITY SIMULATION")
    print("=" * 80)
    
    if reversion_events:
        # Simulate EQS for each event
        eqs_pass_counts = defaultdict(int)
        eqs_total_counts = defaultdict(int)
        eqs_pnl_sums = defaultdict(float)
        
        for e in reversion_events:
            if e['zscore'] < 1.0:
                continue
            
            p = e['entry']
            if p <= 0 or p >= 1:
                continue
                
            # Compute EQS components
            r = binary_entropy(p)
            
            # Fee efficiency (maker entry = 0 fee)
            vwap_val = e['vwap']
            alpha = 0.50
            gross = abs(vwap_val - p) * alpha * 100  # cents
            rt_fee = fee_rate(p) * p * 100  # exit fee only (maker entry)
            f_eff = max(0.10, 1.0 - rt_fee / gross) if gross > 0 else 0.0
            
            # Tick viability
            net = gross - rt_fee
            tick_v = min(1.0, net / 3.0) if net > 0 else 0.0
            
            # Signal quality (simplified)
            z_excess = max(0, e['zscore'] - 1.5)
            v_excess = max(0, e['vol_ratio'] - 1.2)
            s_q = 0.5 + 0.35 * (1 - math.exp(-0.5 * z_excess)) + 0.20 * min(v_excess, 2.0)
            
            # Geometric mean EQS
            if r > 0 and f_eff > 0 and tick_v > 0 and s_q > 0:
                eqs = 100 * (r ** 0.35) * (f_eff ** 0.30) * (tick_v ** 0.20) * (s_q ** 0.15)
            else:
                eqs = 0.0
            
            for threshold in [30, 35, 40, 45, 50, 55, 60]:
                eqs_total_counts[threshold] += 1
                if eqs >= threshold:
                    eqs_pass_counts[threshold] += 1
                    eqs_pnl_sums[threshold] += e['pnl_cents']
        
        print(f"\n  {'EQS Thresh':>10}  {'Pass':>8}  {'Total':>8}  {'Pass%':>7}  {'Avg PnL¢':>9}")
        for threshold in [30, 35, 40, 45, 50, 55, 60]:
            total = eqs_total_counts[threshold]
            passed = eqs_pass_counts[threshold]
            pct = 100.0 * passed / total if total > 0 else 0
            avg_pnl = eqs_pnl_sums[threshold] / passed if passed > 0 else 0
            print(f"  {threshold:>10}  {passed:>8}  {total:>8}  {pct:>6.1f}%  {avg_pnl:>9.2f}")

    # ────────────────────────────────────────────────────────────────────
    # 12. TRADE INTER-ARRIVAL TIME (market activity periods)
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  12. TRADE INTER-ARRIVAL TIMES (seconds)")
    print("=" * 80)
    
    all_gaps = []
    for aid, tlist in active_assets.items():
        sorted_t = sorted(t[0] for t in tlist)
        for i in range(1, len(sorted_t)):
            gap = sorted_t[i] - sorted_t[i-1]
            if 0 < gap < 3600:  # cap at 1 hour
                all_gaps.append(gap)
    
    if all_gaps:
        sorted_gaps = sorted(all_gaps)
        print(f"\n  Total inter-arrival observations: {len(sorted_gaps):,}")
        for p in [25, 50, 75, 90, 95, 99]:
            idx = min(int(len(sorted_gaps) * p / 100), len(sorted_gaps) - 1)
            print(f"  P{p}: {sorted_gaps[idx]:.1f}s")
        
        # How many minutes per hour have ≥1 trade?
        # Group by 1-min buckets
        from collections import Counter
        all_ts = []
        for aid, tlist in active_assets.items():
            all_ts.extend(t[0] for t in tlist)
        all_ts.sort()
        if all_ts:
            min_buckets = set()
            for ts in all_ts:
                min_buckets.add(int(ts // 60))
            total_minutes = (all_ts[-1] - all_ts[0]) / 60
            active_pct = 100.0 * len(min_buckets) / total_minutes if total_minutes > 0 else 0
            print(f"\n  Time span: {total_minutes/60:.1f} hours")
            print(f"  Minutes with ≥1 trade: {len(min_buckets):,} / {int(total_minutes):,} ({active_pct:.1f}%)")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
