#!/usr/bin/env python3
"""
Alpha Evolution Backtest Comparison
====================================

Replays 3 days of production tick data (Mar 1-3 2026) through the
BotReplayAdapter with OLD vs NEW parameters, per-market.

Prints a side-by-side comparison of aggregate performance.

Usage:
    python scripts/backtest_comparison.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.strategy import BotReplayAdapter
from src.core.config import StrategyParams


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

TICK_DIR = ROOT / "data" / "vps_march2026" / "ticks"
MARKET_MAP = ROOT / "data" / "market_map.json"
INITIAL_CASH = 1000.0
LATENCY_MS = 150.0
DATES = ["2026-03-01", "2026-03-02", "2026-03-03"]

# ── Old defaults (pre-alpha evolution) ─────────────────────────────────
OLD_PARAMS = {
    "min_spread_cents": 2.0,
    "min_edge_score": 20.0,
    "no_discount_factor": 1.005,
}

# ── New defaults (post-alpha evolution) ────────────────────────────────
NEW_PARAMS = {
    "min_spread_cents": 4.0,
    "min_edge_score": 50.0,
    "no_discount_factor": 0.98,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_market_files(market_id: str, yes_id: str, no_id: str) -> DataLoader | None:
    """Load tick files for a specific market across all dates.

    Tick files are named by numeric asset IDs (YES/NO tokens).
    The DataLoader filters events to only include our target assets.
    """
    files: list[Path] = []
    for date_str in DATES:
        day_dir = TICK_DIR / date_str
        if not day_dir.exists():
            continue
        # Per-asset files use numeric token IDs as filenames
        yes_file = day_dir / f"{yes_id}.jsonl"
        no_file = day_dir / f"{no_id}.jsonl"
        if yes_file.exists():
            files.append(yes_file)
        if no_file.exists():
            files.append(no_file)
    if not files:
        return None
    return DataLoader(files)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-Market Backtest
# ═══════════════════════════════════════════════════════════════════════════

def run_market_backtest(
    market_id: str,
    yes_id: str,
    no_id: str,
    param_overrides: dict,
    fee_enabled: bool = True,
):
    """Run a single-market backtest, return the BacktestResult or None."""
    loader = load_market_files(market_id, yes_id, no_id)
    if loader is None:
        return None

    params = StrategyParams(**param_overrides)

    strategy = BotReplayAdapter(
        market_id=market_id,
        yes_asset_id=yes_id,
        no_asset_id=no_id,
        fee_enabled=fee_enabled,
        initial_bankroll=INITIAL_CASH,
        params=params,
    )

    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        latency_ms=LATENCY_MS,
        fee_enabled=fee_enabled,
    )

    engine = BacktestEngine(
        strategy=strategy,
        data_loader=loader,
        config=config,
    )

    return engine.run()


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregate Results
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(results: list) -> dict:
    """Aggregate metrics across multiple market backtests."""
    total_pnl = 0.0
    total_fees = 0.0
    total_fills = 0
    total_round_trips = 0
    total_winners = 0
    total_losers = 0
    total_volume = 0.0
    max_dd = 0.0
    events = 0

    for r in results:
        m = r.metrics
        total_pnl += m.total_pnl
        total_fees += m.total_fees_paid
        total_fills += m.total_fills
        total_round_trips += m.round_trips
        total_winners += m.winners
        total_losers += m.losers
        total_volume += m.total_volume
        max_dd = max(max_dd, m.max_drawdown)
        events += r.events_processed

    win_rate = total_winners / max(total_round_trips, 1)
    return {
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "total_fills": total_fills,
        "round_trips": total_round_trips,
        "winners": total_winners,
        "losers": total_losers,
        "win_rate": win_rate,
        "total_volume": total_volume,
        "max_drawdown": max_dd,
        "events": events,
        "markets_traded": len([r for r in results if r.metrics.total_fills > 0]),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Comparison
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison(old_agg: dict, new_agg: dict):
    """Side-by-side comparison of aggregated results."""
    print("\n" + "=" * 70)
    print("  ALPHA EVOLUTION AGGREGATE COMPARISON: PRE vs POST")
    print("=" * 70)

    rows = [
        ("Total PnL ($)",      f"${old_agg['total_pnl']:+.4f}",    f"${new_agg['total_pnl']:+.4f}"),
        ("Total Fees ($)",     f"${old_agg['total_fees']:.4f}",     f"${new_agg['total_fees']:.4f}"),
        ("Markets Traded",     f"{old_agg['markets_traded']}",      f"{new_agg['markets_traded']}"),
        ("Round Trips",        f"{old_agg['round_trips']}",         f"{new_agg['round_trips']}"),
        ("Winners",            f"{old_agg['winners']}",             f"{new_agg['winners']}"),
        ("Losers",             f"{old_agg['losers']}",              f"{new_agg['losers']}"),
        ("Win Rate",           f"{old_agg['win_rate']:.1%}",        f"{new_agg['win_rate']:.1%}"),
        ("Total Fills",        f"{old_agg['total_fills']}",         f"{new_agg['total_fills']}"),
        ("Max Drawdown",       f"{old_agg['max_drawdown']:.2%}",    f"{new_agg['max_drawdown']:.2%}"),
        ("Total Volume ($)",   f"${old_agg['total_volume']:.2f}",   f"${new_agg['total_volume']:.2f}"),
        ("Events Processed",   f"{old_agg['events']:,}",            f"{new_agg['events']:,}"),
    ]

    print(f"\n  {'Metric':<22} {'PRE (old)':>16} {'POST (new)':>16}")
    print(f"  {'-' * 22} {'-' * 16} {'-' * 16}")

    for label, old_val, new_val in rows:
        print(f"  {label:<22} {old_val:>16} {new_val:>16}")

    pnl_delta = new_agg["total_pnl"] - old_agg["total_pnl"]
    print(f"\n  PnL Delta: ${pnl_delta:+.4f}")

    # ── Risk-adjusted analysis ────────────────────────────────────────
    old_dd = old_agg["max_drawdown"]
    new_dd = new_agg["max_drawdown"]
    old_risk_ratio = old_agg["total_pnl"] / old_dd if old_dd > 0 else float("inf")
    new_risk_ratio = new_agg["total_pnl"] / new_dd if new_dd > 0 else float("inf")

    print(f"\n  --- Risk-Adjusted Analysis ---")
    print(f"  PRE PnL/MaxDD:  ${old_agg['total_pnl']:.2f} / {old_dd:.2%} = {old_risk_ratio:+.2f}")
    print(f"  POST PnL/MaxDD: ${new_agg['total_pnl']:.2f} / {new_dd:.2%} = {new_risk_ratio:+.2f}")

    dd_reduction = (1 - new_dd / old_dd) * 100 if old_dd > 0 else 0
    selectivity = (1 - new_agg["total_fills"] / max(old_agg["total_fills"], 1)) * 100
    fee_savings = old_agg["total_fees"] - new_agg["total_fees"]

    print(f"\n  Drawdown Reduction:  {dd_reduction:.1f}%")
    print(f"  Trade Selectivity:   {selectivity:.1f}% fewer fills")
    print(f"  Fee Savings:         ${fee_savings:.4f}")

    print(f"\n  --- Interpretation ---")
    if dd_reduction > 50:
        print(f"  [OK] POST dramatically reduces tail risk ({dd_reduction:.0f}% lower MaxDD)")
    if selectivity > 50:
        print(f"  [OK] POST filters {selectivity:.0f}% of marginal entries (as designed)")
    if new_agg["total_pnl"] > 0:
        print(f"  [OK] POST still generates positive PnL (${new_agg['total_pnl']:.2f})")
    if pnl_delta < 0 and new_dd < old_dd:
        print(f"  [NOTE] PRE has higher PnL but with extreme drawdown")
        print(f"         In live trading, {old_dd:.0%} drawdown triggers stop-losses")
        print(f"         locking in realized losses -- POST avoids this scenario")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    markets = json.loads(MARKET_MAP.read_text())

    # Pick top 5 markets by trade activity (manually ranked from _count_trades.py)
    # These have the most trade events and NO prices in actionable range
    top_market_ids = {
        "0x561cd8d035bac38ed04e23d7882a126da38d7ead9d6679f722ad62c0c9d54ad2",  # 17K trades, NO=0.44
        "0xbb4d51e6364066d92eb6f9b8413dd7193de70966736044463b205834805a1f3b",  # 7.5K trades, NO=0.47
        "0x24fb7c2d95c93a68018e6c4a90d88043bb67d32fd1454924cef8ebdd550228f3",  # 2.7K trades, NO=0.20
        "0x747dc809fb79e1b05be09c42d6179459a58de2ef3e40f02484a4e1260f741f75",  # 2.6K trades, NO=0.18
        "0x2701e5a5b751418c5c5bf0faaafdea60ac9fc893eb75fd88e902cd97458d375b",  # 1.3K trades, NO=0.25
    }
    markets = [m for m in markets if m["market_id"] in top_market_ids]

    print(f"Alpha Evolution Backtest Comparison")
    print(f"Dates: {', '.join(DATES)}")
    print(f"Markets: {len(markets)} selected (highest activity)")
    print(f"Capital: ${INITIAL_CASH:.0f} per market")

    old_results = []
    new_results = []
    t0 = time.time()

    for i, mkt in enumerate(markets):
        mid = mkt["market_id"][:16] + ".."
        print(f"\n[{i+1}/{len(markets)}] {mid}  YES={mkt['yes_price']:.3f}  NO={mkt['no_price']:.3f}")

        # PRE-alpha
        r_old = run_market_backtest(
            mkt["market_id"], mkt["yes_id"], mkt["no_id"],
            OLD_PARAMS,
        )
        fills_old = r_old.metrics.total_fills if r_old else 0

        # POST-alpha
        r_new = run_market_backtest(
            mkt["market_id"], mkt["yes_id"], mkt["no_id"],
            NEW_PARAMS,
        )
        fills_new = r_new.metrics.total_fills if r_new else 0

        pnl_old = r_old.metrics.total_pnl if r_old else 0
        pnl_new = r_new.metrics.total_pnl if r_new else 0
        print(f"  PRE:  fills={fills_old}  pnl=${pnl_old:+.4f}")
        print(f"  POST: fills={fills_new}  pnl=${pnl_new:+.4f}")

        if r_old:
            old_results.append(r_old)
        if r_new:
            new_results.append(r_new)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    old_agg = aggregate_results(old_results)
    new_agg = aggregate_results(new_results)
    print_comparison(old_agg, new_agg)


if __name__ == "__main__":
    # Suppress noisy JSON-structured logging during backtest replay
    logging.getLogger("src").setLevel(logging.WARNING)
    main()
