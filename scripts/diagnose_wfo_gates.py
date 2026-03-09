#!/usr/bin/env python3
"""
WFO Zero-Trades Gate Diagnostic — instruments the BotReplayAdapter's
10-step signal funnel to identify which gate is rejecting 100% of events.

Usage:
    python scripts/diagnose_wfo_gates.py --data-dir data/vps_march2026

    # With explicit market from market_map.json (first market by default):
    python scripts/diagnose_wfo_gates.py --data-dir data/vps_march2026 --market-index 0

    # With specific dates:
    python scripts/diagnose_wfo_gates.py --data-dir data/vps_march2026 \
        --start-date 2026-01-01 --end-date 2026-01-07
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.data_loader import DataLoader
from src.backtest.data_recorder import MarketDataRecorder
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.strategy import BotReplayAdapter
from src.core.config import StrategyParams, settings
from src.data.ohlcv import OHLCVAggregator, OHLCVBar
from src.signals.edge_filter import compute_edge_score
from src.signals.panic_detector import PanicDetector


def load_market_configs(data_dir: str) -> list[dict]:
    """Load market_map.json from the data directory hierarchy."""
    base = Path(data_dir)
    candidates = [
        base / "market_map.json",
        base.parent / "market_map.json",
        Path("data") / "market_map.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as fh:
                raw = json.load(fh)
            configs = []
            for entry in raw:
                mid = entry.get("market_id", "")
                yid = str(entry.get("yes_id", ""))
                nid = str(entry.get("no_id", ""))
                if mid and yid and nid:
                    configs.append({
                        "market_id": mid,
                        "yes_asset_id": yid,
                        "no_asset_id": nid,
                        "yes_price": entry.get("yes_price", 0),
                        "no_price": entry.get("no_price", 0),
                    })
            return configs
    return []


def collect_files_for_dates(data_dir: str, dates: list[str]) -> list[Path]:
    """Gather tick files for the given dates."""
    files: list[Path] = []
    base = Path(data_dir)
    for d in dates:
        files.extend(MarketDataRecorder.data_files_for_date(data_dir, d))
        parquet_dir = base / d
        if parquet_dir.exists():
            files.extend(sorted(parquet_dir.glob("*.parquet")))
    return files


def diagnose_single_market(
    data_dir: str,
    dates: list[str],
    market_id: str,
    yes_asset_id: str,
    no_asset_id: str,
) -> dict:
    """Run a single-market backtest with full gate instrumentation."""

    files = collect_files_for_dates(data_dir, dates)
    if not files:
        return {"error": "No data files found"}

    asset_ids = {yes_asset_id, no_asset_id}
    loader = DataLoader(files, asset_ids=asset_ids)

    # Count events from the loader (consume it once for stats)
    events_list = list(loader)
    total_events = len(events_list)

    if total_events == 0:
        return {
            "error": "DataLoader produced 0 events for these asset IDs",
            "yes_asset_id": yes_asset_id,
            "no_asset_id": no_asset_id,
            "files_checked": len(files),
        }

    # Count events by type and asset
    type_counts: Counter = Counter()
    asset_counts: Counter = Counter()
    for ev in events_list:
        type_counts[ev.event_type] += 1
        asset_counts[ev.asset_id] += 1

    print(f"\n{'='*70}")
    print(f"  MARKET: {market_id[:20]}...")
    print(f"  YES ID: {yes_asset_id[:20]}...")
    print(f"  NO  ID: {no_asset_id[:20]}...")
    print(f"{'='*70}")
    print(f"\n  Total events from DataLoader: {total_events:,}")
    print(f"  Event types: {dict(type_counts)}")
    print(f"  Events per asset:")
    for aid, cnt in sorted(asset_counts.items(), key=lambda x: -x[1]):
        label = "YES" if aid == yes_asset_id else "NO" if aid == no_asset_id else "???"
        print(f"    {label} ({aid[:20]}...): {cnt:,}")

    yes_trade_count = sum(
        1 for ev in events_list
        if ev.event_type == "trade" and ev.asset_id == yes_asset_id
    )
    no_trade_count = sum(
        1 for ev in events_list
        if ev.event_type == "trade" and ev.asset_id == no_asset_id
    )
    print(f"\n  YES trades: {yes_trade_count:,}")
    print(f"  NO  trades: {no_trade_count:,}")

    # ── Now instrument the gate funnel ──────────────────────────────────
    params = StrategyParams()  # defaults
    yes_agg = OHLCVAggregator(yes_asset_id)
    no_agg = OHLCVAggregator(no_asset_id)

    detector = PanicDetector(
        market_id=market_id,
        yes_asset_id=yes_asset_id,
        no_asset_id=no_asset_id,
        yes_aggregator=yes_agg,
        no_aggregator=no_agg,
        zscore_threshold=params.zscore_threshold,
        volume_ratio_threshold=params.volume_ratio_threshold,
        trend_guard_pct=params.trend_guard_pct,
        trend_guard_bars=params.trend_guard_bars,
    )

    # Simulate the BBO tracking (same as BacktestEngine)
    _SYNTH_HALF_SPREAD = 0.02
    bbo_per_asset: dict[str, tuple[float, float]] = {}

    # Gate counters
    gates = {
        "total_yes_bars": 0,
        "G01_near_resolved": 0,
        "G02_price_band": 0,
        "G03_cooldown": 0,
        "G04_max_positions": 0,
        "G05_no_best_ask_zero": 0,
        "G06_panic_detector": 0,
        "G06a_history_lt5": 0,
        "G06b_sigma_zero": 0,
        "G06c_zscore_low": 0,
        "G06d_volume_ratio_low": 0,
        "G06e_no_vwap_zero": 0,
        "G06f_no_not_discounted": 0,
        "G06g_trend_guard": 0,
        "G07_no_best_bid_zero": 0,
        "G08_spread_too_narrow": 0,
        "G09_eqs_not_viable": 0,
        "G09a_no_mean_reversion": 0,
        "G09b_sub_tick": 0,
        "G09c_fees_exceed": 0,
        "G09d_low_entropy": 0,
        "G09e_below_threshold": 0,
        "G10_size_too_small": 0,
        "PASSED_ALL_GATES": 0,
    }

    # Diagnostic accumulators
    zscore_samples: list[float] = []
    no_vwap_samples: list[float] = []
    yes_price_samples: list[float] = []
    eqs_score_samples: list[float] = []
    gross_cents_samples: list[float] = []

    last_signal_time = 0.0
    cooldown_seconds = params.signal_cooldown_minutes * 60.0

    from src.data.websocket_client import TradeEvent

    for ev in events_list:
        if ev.event_type != "trade":
            continue

        data = ev.data
        trade_price = float(data.get("price", 0))
        trade_size = float(data.get("size", 0))
        trade_side = str(data.get("side", "buy")).lower()
        if trade_price <= 0 or trade_size <= 0:
            continue

        # Update synthetic BBO
        half = _SYNTH_HALF_SPREAD
        synth_bid = max(0.01, trade_price - half)
        synth_ask = min(0.99, trade_price + half)
        bbo_per_asset[ev.asset_id] = (synth_bid, synth_ask)

        trade_event = TradeEvent(
            timestamp=ev.timestamp,
            market_id=data.get("market_id", data.get("market", "")),
            asset_id=ev.asset_id,
            side=trade_side,
            price=trade_price,
            size=trade_size,
            is_yes=data.get("is_yes", True),
            is_taker=data.get("is_taker", False),
        )

        # Feed aggregators
        if ev.asset_id == yes_asset_id:
            bar = yes_agg.on_trade(trade_event)
            if bar is not None:
                # ── Evaluate the full gate funnel ──────────────────────
                gates["total_yes_bars"] += 1
                yes_price = bar.close
                bar_time = bar.open_time

                # G01: Near-resolved
                if yes_price >= 0.97 or yes_price <= 0.03:
                    gates["G01_near_resolved"] += 1
                    continue

                # G02: Price band
                if not (params.min_tradeable_price < yes_price < params.max_tradeable_price):
                    gates["G02_price_band"] += 1
                    continue

                # G03: Cooldown
                if bar_time - last_signal_time < cooldown_seconds:
                    gates["G03_cooldown"] += 1
                    continue

                # G04: Max positions (always 0 in this diagnostic)
                # (skip — no positions in diagnostic)

                # G05: NO best ask
                no_bbo = bbo_per_asset.get(no_asset_id)
                if no_bbo:
                    best_ask = no_bbo[1]
                else:
                    best_ask = 0.0
                if best_ask <= 0:
                    gates["G05_no_best_ask_zero"] += 1
                    continue

                # G06: PanicDetector (decomposed)
                # Check sub-gates manually for diagnosis
                if len(yes_agg.bars) < 5:
                    gates["G06_panic_detector"] += 1
                    gates["G06a_history_lt5"] += 1
                    continue

                vwap = yes_agg.rolling_vwap
                sigma = yes_agg.rolling_volatility
                if sigma <= 0 or vwap <= 0:
                    gates["G06_panic_detector"] += 1
                    gates["G06b_sigma_zero"] += 1
                    continue

                delta_p = bar.close - vwap
                zscore = delta_p / sigma

                # Intra-bar correction (same as PanicDetector)
                bar_range = bar.high - bar.low
                if bar_range > 0 and delta_p > 0:
                    close_position = (bar.close - bar.low) / bar_range
                    if close_position < 0.5:
                        retracement_factor = 0.7 + 0.3 * close_position
                        zscore *= retracement_factor

                zscore_samples.append(zscore)
                yes_price_samples.append(yes_price)

                v_ratio = (bar.volume / yes_agg.avg_bar_volume) if yes_agg.avg_bar_volume > 0 else 0.0

                if zscore < params.zscore_threshold:
                    gates["G06_panic_detector"] += 1
                    gates["G06c_zscore_low"] += 1
                    continue

                if v_ratio < params.volume_ratio_threshold:
                    gates["G06_panic_detector"] += 1
                    gates["G06d_volume_ratio_low"] += 1
                    continue

                no_vwap = no_agg.rolling_vwap
                no_vwap_samples.append(no_vwap)
                if no_vwap <= 0:
                    gates["G06_panic_detector"] += 1
                    gates["G06e_no_vwap_zero"] += 1
                    continue

                if best_ask >= no_vwap * settings.strategy.no_discount_factor:
                    gates["G06_panic_detector"] += 1
                    gates["G06f_no_not_discounted"] += 1
                    continue

                # Trend guard
                trend_bars = params.trend_guard_bars
                trend_pct = params.trend_guard_pct
                if len(yes_agg.bars) >= trend_bars:
                    bars_list = yes_agg.bars
                    recent_close = bars_list[-1].close
                    anchor_close = bars_list[-trend_bars].close
                    if anchor_close > 0:
                        move = (recent_close - anchor_close) / anchor_close
                        if move >= trend_pct:
                            gates["G06_panic_detector"] += 1
                            gates["G06g_trend_guard"] += 1
                            continue

                # Signal passed PanicDetector! Record for cooldown
                last_signal_time = bar_time

                # G07: NO best bid
                no_bbo_bid = bbo_per_asset.get(no_asset_id)
                best_bid = no_bbo_bid[0] if no_bbo_bid else 0.0
                if best_bid <= 0:
                    gates["G07_no_best_bid_zero"] += 1
                    continue

                # G08: Spread
                spread = best_ask - best_bid
                if spread * 100.0 < params.min_spread_cents:
                    gates["G08_spread_too_narrow"] += 1
                    continue

                # G09: EQS
                exec_mode = "maker" if params.maker_routing_enabled else "taker"
                eqs = compute_edge_score(
                    entry_price=best_ask,
                    no_vwap=no_vwap if no_vwap > 0 else best_ask,
                    zscore=zscore,
                    volume_ratio=v_ratio,
                    whale_confluence=False,
                    fee_enabled=True,
                    alpha=params.alpha_default,
                    zscore_threshold=params.zscore_threshold,
                    min_score=params.min_edge_score,
                    current_ewma_vol=no_agg.rolling_volatility_ewma or None,
                    execution_mode=exec_mode,
                )
                eqs_score_samples.append(eqs.score)
                gross_cents_samples.append(eqs.expected_gross_cents)

                if not eqs.viable:
                    gates["G09_eqs_not_viable"] += 1
                    reason = eqs.rejection_reason
                    if reason == "no_mean_reversion_target":
                        gates["G09a_no_mean_reversion"] += 1
                    elif reason == "sub_tick_spread":
                        gates["G09b_sub_tick"] += 1
                    elif reason in ("fees_exceed_discretised_spread", "fees_exceed_spread"):
                        gates["G09c_fees_exceed"] += 1
                    elif reason == "low_regime_entropy":
                        gates["G09d_low_entropy"] += 1
                    elif reason == "score_below_threshold":
                        gates["G09e_below_threshold"] += 1
                    continue

                # G10: Size
                trade_usd = min(
                    1000.0 * params.kelly_fraction,
                    params.max_trade_size_usd,
                )
                size = trade_usd / best_ask if best_ask > 0 else 0
                if size < 1:
                    gates["G10_size_too_small"] += 1
                    continue

                gates["PASSED_ALL_GATES"] += 1

        elif ev.asset_id == no_asset_id:
            no_agg.on_trade(trade_event)

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n  YES aggregator bars closed: {len(yes_agg.bars)}")
    print(f"  NO  aggregator bars closed: {len(no_agg.bars)}")
    print(f"  YES rolling VWAP:  {yes_agg.rolling_vwap:.4f}")
    print(f"  YES rolling σ:     {yes_agg.rolling_volatility:.6f}")
    print(f"  NO  rolling VWAP:  {no_agg.rolling_vwap:.4f}")
    print(f"  NO  rolling σ:     {no_agg.rolling_volatility:.6f}")
    print(f"  NO  EWMA σ:        {no_agg.rolling_volatility_ewma:.6f}")

    print(f"\n{'─'*70}")
    print(f"  GATE FUNNEL (bars entering each gate)")
    print(f"{'─'*70}")
    total_bars = gates["total_yes_bars"]
    remaining = total_bars
    for gate_name in [
        "G01_near_resolved", "G02_price_band", "G03_cooldown",
        "G04_max_positions", "G05_no_best_ask_zero",
        "G06_panic_detector", "G07_no_best_bid_zero",
        "G08_spread_too_narrow", "G09_eqs_not_viable",
        "G10_size_too_small",
    ]:
        rejected = gates[gate_name]
        pct = (rejected / total_bars * 100) if total_bars > 0 else 0
        remaining -= rejected
        bar_chart = "█" * int(pct / 2)
        print(f"  {gate_name:<30s}  rejected {rejected:>6,}  ({pct:>5.1f}%)  {bar_chart}")

    passed = gates["PASSED_ALL_GATES"]
    pct = (passed / total_bars * 100) if total_bars > 0 else 0
    print(f"\n  {'PASSED ALL GATES':<30s}  {passed:>6,}  ({pct:>5.1f}%)")
    print(f"  Total YES bars evaluated:     {total_bars:>6,}")

    # PanicDetector sub-gate breakdown
    g6_total = gates["G06_panic_detector"]
    if g6_total > 0:
        print(f"\n{'─'*70}")
        print(f"  PANIC DETECTOR SUB-GATE BREAKDOWN ({g6_total} rejections)")
        print(f"{'─'*70}")
        for sub_gate in [
            "G06a_history_lt5", "G06b_sigma_zero", "G06c_zscore_low",
            "G06d_volume_ratio_low", "G06e_no_vwap_zero",
            "G06f_no_not_discounted", "G06g_trend_guard",
        ]:
            cnt = gates[sub_gate]
            pct = (cnt / g6_total * 100) if g6_total > 0 else 0
            bar_chart = "█" * int(pct / 2)
            print(f"  {sub_gate:<30s}  {cnt:>6,}  ({pct:>5.1f}%)  {bar_chart}")

    # EQS sub-gate breakdown
    g9_total = gates["G09_eqs_not_viable"]
    if g9_total > 0:
        print(f"\n{'─'*70}")
        print(f"  EQS REJECTION BREAKDOWN ({g9_total} rejections)")
        print(f"{'─'*70}")
        for sub_gate in [
            "G09a_no_mean_reversion", "G09b_sub_tick",
            "G09c_fees_exceed", "G09d_low_entropy",
            "G09e_below_threshold",
        ]:
            cnt = gates[sub_gate]
            pct = (cnt / g9_total * 100) if g9_total > 0 else 0
            print(f"  {sub_gate:<30s}  {cnt:>6,}  ({pct:>5.1f}%)")

    # Z-score distribution
    if zscore_samples:
        import numpy as np
        zs = np.array(zscore_samples)
        print(f"\n{'─'*70}")
        print(f"  Z-SCORE DISTRIBUTION (post price-band, pre-detector)")
        print(f"{'─'*70}")
        print(f"  N samples:  {len(zs):,}")
        print(f"  Mean:       {zs.mean():.3f}")
        print(f"  Std:        {zs.std():.3f}")
        print(f"  Min:        {zs.min():.3f}")
        print(f"  P25:        {np.percentile(zs, 25):.3f}")
        print(f"  Median:     {np.percentile(zs, 50):.3f}")
        print(f"  P75:        {np.percentile(zs, 75):.3f}")
        print(f"  P90:        {np.percentile(zs, 90):.3f}")
        print(f"  P95:        {np.percentile(zs, 95):.3f}")
        print(f"  P99:        {np.percentile(zs, 99):.3f}")
        print(f"  Max:        {zs.max():.3f}")
        print(f"  Threshold:  {params.zscore_threshold}")
        pct_above = (zs >= params.zscore_threshold).mean() * 100
        print(f"  % above threshold: {pct_above:.1f}%")

    # EQS score distribution
    if eqs_score_samples:
        import numpy as np
        es = np.array(eqs_score_samples)
        print(f"\n{'─'*70}")
        print(f"  EQS SCORE DISTRIBUTION (signals that passed PanicDetector)")
        print(f"{'─'*70}")
        print(f"  N samples:  {len(es):,}")
        print(f"  Mean:       {es.mean():.1f}")
        print(f"  Min:        {es.min():.1f}")
        print(f"  Max:        {es.max():.1f}")
        print(f"  Threshold:  {params.min_edge_score}")

    if gross_cents_samples:
        import numpy as np
        gc = np.array(gross_cents_samples)
        print(f"\n  GROSS CENTS (expected alpha spread):")
        print(f"  Mean:       {gc.mean():.3f}")
        print(f"  Median:     {np.median(gc):.3f}")
        print(f"  Max:        {gc.max():.3f}")
        print(f"  % positive: {(gc > 0).mean()*100:.1f}%")

    print(f"\n{'='*70}\n")

    return gates


def main():
    parser = argparse.ArgumentParser(description="Diagnose WFO zero-trades gate blockage")
    parser.add_argument("--data-dir", required=True, help="Path to recorded tick data directory")
    parser.add_argument("--market-index", type=int, default=0, help="Index in market_map.json (default: 0)")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (default: first available)")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: 7 days from start)")
    parser.add_argument("--max-markets", type=int, default=3, help="Max markets to diagnose (default: 3)")
    args = parser.parse_args()

    # Discover available dates
    available = MarketDataRecorder.available_dates(args.data_dir)
    if not available:
        print(f"ERROR: No recorded dates found in {args.data_dir}")
        print(f"  Searched: {args.data_dir}/raw_ticks/ and {args.data_dir}/ticks/")
        sys.exit(1)

    print(f"Available dates: {available[0]} to {available[-1]} ({len(available)} days)")

    # Select date range
    if args.start_date:
        dates = [d for d in available if d >= args.start_date]
    else:
        dates = available

    if args.end_date:
        dates = [d for d in dates if d <= args.end_date]
    elif len(dates) > 7:
        dates = dates[:7]  # default: first 7 days

    print(f"Diagnosing dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Load market configs
    markets = load_market_configs(args.data_dir)
    if not markets:
        print("ERROR: No market_map.json found. Cannot determine asset IDs.")
        sys.exit(1)

    print(f"Markets found in map: {len(markets)}")

    # Diagnose each market
    n_markets = min(args.max_markets, len(markets))
    for i in range(n_markets):
        mc = markets[i]
        diagnose_single_market(
            data_dir=args.data_dir,
            dates=dates,
            market_id=mc["market_id"],
            yes_asset_id=mc["yes_asset_id"],
            no_asset_id=mc["no_asset_id"],
        )


if __name__ == "__main__":
    main()
