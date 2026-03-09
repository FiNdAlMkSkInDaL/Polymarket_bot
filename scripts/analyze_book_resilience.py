#!/usr/bin/env python3
"""
analyze_book_resilience.py — Liquidity Heatmap & OBI Threshold Optimizer

Reads backfilled L2 JSONL files produced by ``backfill_data.py`` and derives
microstructure statistics for tuning the StopLossMonitor's preemptive
liquidity-drain threshold (``sl_preemptive_obi_threshold``).

Metrics
───────
  1. **Book Depth Ratio (BDR)** leading up to price moves ≥ 2 ticks
     (1 tick = $0.01).  Rolling BDR = Σ(bid_depth) / Σ(ask_depth) over
     the last 5 delta updates.

  2. **Depth-Near-Mid Variance** — variance of bid-side depth in a
     rolling window, identifying "hollow book" regimes where resting
     liquidity evaporates before adverse price moves.

Output
──────
  Prints a per-market summary table and a portfolio-wide recommendation
  for ``sl_preemptive_obi_threshold`` based on the 10th percentile of
  the BDR observed in the 500 ms window before ≥ 2-tick adverse moves.

Usage
─────
  python scripts/analyze_book_resilience.py \\
      --data-dir data/vps_march2026/ticks \\
      --market-map data/market_map.json \\
      --output-json logs/book_resilience_report.json

  # Single date
  python scripts/analyze_book_resilience.py --date 2026-03-03
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

TICK_SIZE = 0.01  # $0.01 per tick
ADVERSE_MOVE_TICKS = 2  # ≥ 2 ticks = adverse move threshold
BDR_LOOKBACK = 5  # number of recent deltas for rolling BDR
DEPTH_VARIANCE_WINDOW = 20  # rolling window for depth-near-mid variance
DEFAULT_DATA_DIR = Path("data/vps_march2026/ticks")
DEFAULT_MARKET_MAP = Path("data/market_map.json")


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketStats:
    """Accumulated statistics for a single market."""
    market_id: str
    total_deltas: int = 0
    total_snapshots: int = 0
    adverse_moves: int = 0
    bdr_before_adverse: list[float] = field(default_factory=list)
    depth_variance_samples: list[float] = field(default_factory=list)
    hollow_book_count: int = 0  # depth variance > 2σ episodes


@dataclass
class BookState:
    """Lightweight running book state reconstructed from deltas."""
    bids: dict[float, float] = field(default_factory=dict)  # price → size
    asks: dict[float, float] = field(default_factory=dict)  # price → size
    last_mid: float = 0.0
    mid_history: list[float] = field(default_factory=list)  # recent mid prices


# ═══════════════════════════════════════════════════════════════════════════
#  JSONL record iterator
# ═══════════════════════════════════════════════════════════════════════════

def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed dicts from a JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ═══════════════════════════════════════════════════════════════════════════
#  Book reconstruction helpers
# ═══════════════════════════════════════════════════════════════════════════

def apply_snapshot(book: BookState, payload: dict) -> None:
    """Reset book state from a snapshot payload."""
    book.bids.clear()
    book.asks.clear()
    for b in payload.get("bids", []):
        try:
            price = float(b["price"])
            size = float(b["size"])
            if price > 0 and size > 0:
                book.bids[price] = size
        except (KeyError, TypeError, ValueError):
            continue
    for a in payload.get("asks", []):
        try:
            price = float(a["price"])
            size = float(a["size"])
            if price > 0 and size > 0:
                book.asks[price] = size
        except (KeyError, TypeError, ValueError):
            continue
    _update_mid(book)


def apply_delta(book: BookState, payload: dict) -> None:
    """Apply delta changes to the book."""
    for change in payload.get("changes", []):
        try:
            side = change.get("side", "").upper()
            price = float(change["price"])
            size = float(change["size"])
        except (KeyError, TypeError, ValueError):
            continue

        target = book.bids if side == "BUY" else book.asks
        if size <= 0:
            target.pop(price, None)
        else:
            target[price] = size
    _update_mid(book)


def _update_mid(book: BookState) -> None:
    """Recompute mid price from current BBO."""
    best_bid = max(book.bids.keys()) if book.bids else 0.0
    best_ask = min(book.asks.keys()) if book.asks else 0.0
    if best_bid > 0 and best_ask > 0:
        book.last_mid = (best_bid + best_ask) / 2.0
    elif best_bid > 0:
        book.last_mid = best_bid
    elif best_ask > 0:
        book.last_mid = best_ask
    book.mid_history.append(book.last_mid)


def compute_bdr(book: BookState, levels: int = 5) -> float:
    """Compute bid/ask depth ratio over the top *levels* price levels."""
    bid_prices = sorted(book.bids.keys(), reverse=True)[:levels]
    ask_prices = sorted(book.asks.keys())[:levels]

    bid_depth = sum(p * book.bids[p] for p in bid_prices)
    ask_depth = sum(p * book.asks[p] for p in ask_prices)

    if ask_depth <= 0:
        return 1.0
    return bid_depth / ask_depth


def compute_depth_near_mid(book: BookState, band_cents: float = 2.0) -> float:
    """Sum bid-side depth within *band_cents* of mid (in dollars)."""
    if book.last_mid <= 0:
        return 0.0
    band = band_cents / 100.0
    lo = book.last_mid - band
    return sum(
        book.bids[p] for p in book.bids if p >= lo
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Per-market analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_market_file(path: Path) -> MarketStats:
    """Process one JSONL file and extract liquidity statistics."""
    market_id = path.stem
    stats = MarketStats(market_id=market_id)
    book = BookState()

    # Rolling buffers
    recent_bdrs: deque[float] = deque(maxlen=BDR_LOOKBACK)
    depth_window: deque[float] = deque(maxlen=DEPTH_VARIANCE_WINDOW)
    prev_mid: float = 0.0

    for rec in iter_jsonl(path):
        payload = rec.get("payload", {})
        if not isinstance(payload, dict):
            continue

        etype = payload.get("event_type", "")

        if etype in ("book", "snapshot", "book_snapshot"):
            apply_snapshot(book, payload)
            stats.total_snapshots += 1
            prev_mid = book.last_mid
            continue

        if etype != "price_change":
            continue

        # L2 delta
        apply_delta(book, payload)
        stats.total_deltas += 1

        # Update rolling BDR
        bdr = compute_bdr(book)
        recent_bdrs.append(bdr)

        # Update depth-near-mid variance window
        dnm = compute_depth_near_mid(book)
        depth_window.append(dnm)

        if len(depth_window) >= DEPTH_VARIANCE_WINDOW:
            arr = np.array(depth_window)
            var = float(np.var(arr))
            stats.depth_variance_samples.append(var)

            # Hollow book = variance exceeds 2× the running mean variance
            if len(stats.depth_variance_samples) > 10:
                mean_var = float(np.mean(stats.depth_variance_samples[-50:]))
                if mean_var > 0 and var > 2.0 * mean_var:
                    stats.hollow_book_count += 1

        # Detect adverse move (≥ 2 ticks from previous mid)
        current_mid = book.last_mid
        if prev_mid > 0 and current_mid > 0:
            move_ticks = abs(current_mid - prev_mid) / TICK_SIZE
            if move_ticks >= ADVERSE_MOVE_TICKS:
                stats.adverse_moves += 1
                if recent_bdrs:
                    avg_bdr = float(np.mean(list(recent_bdrs)))
                    stats.bdr_before_adverse.append(avg_bdr)

        prev_mid = current_mid

    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  Portfolio-wide aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_report(
    all_stats: list[MarketStats],
) -> dict:
    """Compute portfolio-level OBI threshold recommendation."""
    all_bdr_adverse: list[float] = []
    all_depth_var: list[float] = []

    per_market: list[dict] = []
    for s in all_stats:
        all_bdr_adverse.extend(s.bdr_before_adverse)
        all_depth_var.extend(s.depth_variance_samples)

        p10 = float(np.percentile(s.bdr_before_adverse, 10)) if s.bdr_before_adverse else 0.0
        median_bdr = float(np.median(s.bdr_before_adverse)) if s.bdr_before_adverse else 0.0
        mean_depth_var = float(np.mean(s.depth_variance_samples)) if s.depth_variance_samples else 0.0

        per_market.append({
            "market_id": s.market_id[:20] + "..." if len(s.market_id) > 20 else s.market_id,
            "total_deltas": s.total_deltas,
            "total_snapshots": s.total_snapshots,
            "adverse_moves": s.adverse_moves,
            "bdr_p10": round(p10, 4),
            "bdr_median": round(median_bdr, 4),
            "depth_near_mid_variance_mean": round(mean_depth_var, 6),
            "hollow_book_episodes": s.hollow_book_count,
        })

    # Portfolio-wide threshold recommendation
    if all_bdr_adverse:
        portfolio_p10 = float(np.percentile(all_bdr_adverse, 10))
        portfolio_p25 = float(np.percentile(all_bdr_adverse, 25))
        portfolio_median = float(np.median(all_bdr_adverse))
    else:
        portfolio_p10 = 0.20  # fallback to current default
        portfolio_p25 = 0.30
        portfolio_median = 0.50

    if all_depth_var:
        depth_var_mean = float(np.mean(all_depth_var))
        depth_var_std = math.sqrt(float(np.var(all_depth_var)))
    else:
        depth_var_mean = 0.0
        depth_var_std = 0.0

    # Recommended threshold: 10th percentile of BDR before adverse moves,
    # floored at 0.05 and capped at 0.50.
    recommended_threshold = max(0.05, min(0.50, round(portfolio_p10, 2)))

    return {
        "portfolio_summary": {
            "total_markets_analyzed": len(all_stats),
            "total_adverse_moves": sum(s.adverse_moves for s in all_stats),
            "total_hollow_book_episodes": sum(s.hollow_book_count for s in all_stats),
            "bdr_before_adverse_p10": round(portfolio_p10, 4),
            "bdr_before_adverse_p25": round(portfolio_p25, 4),
            "bdr_before_adverse_median": round(portfolio_median, 4),
            "depth_near_mid_variance_mean": round(depth_var_mean, 6),
            "depth_near_mid_variance_std": round(depth_var_std, 6),
        },
        "recommended_config": {
            "sl_preemptive_obi_threshold": recommended_threshold,
            "rationale": (
                f"10th percentile of book_depth_ratio observed in the "
                f"{BDR_LOOKBACK}-update window before ≥{ADVERSE_MOVE_TICKS}-tick "
                f"adverse moves across {len(all_stats)} markets. "
                f"BDR P10={portfolio_p10:.4f}, P25={portfolio_p25:.4f}, "
                f"Median={portfolio_median:.4f}. "
                f"Hollow-book episodes={sum(s.hollow_book_count for s in all_stats)}."
            ),
        },
        "per_market": sorted(per_market, key=lambda m: m["adverse_moves"], reverse=True),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def find_jsonl_files(
    data_dir: Path,
    dates: list[date] | None = None,
) -> list[Path]:
    """Find all JSONL files under data_dir, optionally filtered by dates."""
    files: list[Path] = []
    if dates:
        for d in dates:
            day_dir = data_dir / d.isoformat()
            if day_dir.is_dir():
                files.extend(sorted(day_dir.glob("*.jsonl")))
    else:
        for day_dir in sorted(data_dir.iterdir()):
            if day_dir.is_dir():
                files.extend(sorted(day_dir.glob("*.jsonl")))
    return files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze historical L2 book resilience for OBI threshold tuning.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing backfilled JSONL files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--market-map", type=str, default=str(DEFAULT_MARKET_MAP),
        help=f"Path to market_map.json (default: {DEFAULT_MARKET_MAP}).",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Analyze a single date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start of date range (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End of date range (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Write full report to JSON file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    # Resolve date filter
    dates: list[date] | None = None
    if args.date:
        dates = [date.fromisoformat(args.date)]
    elif args.start_date:
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date) if args.end_date else date.today()
        dates = []
        cur = start
        while cur <= end:
            dates.append(cur)
            cur += timedelta(days=1)

    files = find_jsonl_files(data_dir, dates)
    if not files:
        print("No JSONL files found.", file=sys.stderr)
        return 1

    print(f"Analyzing {len(files)} JSONL files...")

    all_stats: list[MarketStats] = []
    for i, fpath in enumerate(files):
        stats = analyze_market_file(fpath)
        all_stats.append(stats)
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  Progress: {i + 1}/{len(files)} files processed")

    report = aggregate_report(all_stats)

    # Print summary
    print("\n" + "=" * 72)
    print("BOOK RESILIENCE REPORT — OBI Threshold Optimization")
    print("=" * 72)

    ps = report["portfolio_summary"]
    print(f"  Markets analyzed:       {ps['total_markets_analyzed']}")
    print(f"  Total adverse moves:    {ps['total_adverse_moves']}")
    print(f"  Hollow-book episodes:   {ps['total_hollow_book_episodes']}")
    print(f"  BDR before adverse P10: {ps['bdr_before_adverse_p10']:.4f}")
    print(f"  BDR before adverse P25: {ps['bdr_before_adverse_p25']:.4f}")
    print(f"  BDR before adverse Med: {ps['bdr_before_adverse_median']:.4f}")
    print(f"  Depth variance mean:    {ps['depth_near_mid_variance_mean']:.6f}")
    print(f"  Depth variance σ:       {ps['depth_near_mid_variance_std']:.6f}")

    rc = report["recommended_config"]
    print(f"\n  ▸ Recommended sl_preemptive_obi_threshold: {rc['sl_preemptive_obi_threshold']}")
    print(f"    Rationale: {rc['rationale']}")

    # Top-5 markets by adverse moves
    print(f"\n  Top-5 markets by adverse moves:")
    for m in report["per_market"][:5]:
        print(
            f"    {m['market_id']:23s}  "
            f"adverse={m['adverse_moves']:5d}  "
            f"BDR_P10={m['bdr_p10']:.4f}  "
            f"hollow={m['hollow_book_episodes']}"
        )
    print("=" * 72)

    print("\nTo apply this threshold, set in StrategyParams / .env:")
    print(f"  SL_PREEMPTIVE_OBI_THRESHOLD={rc['sl_preemptive_obi_threshold']}")

    # Write JSON report
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
