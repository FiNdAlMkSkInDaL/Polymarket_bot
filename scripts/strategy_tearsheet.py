#!/usr/bin/env python3
"""
Pillar 16 — Alpha-Source Attribution Tearsheet

Parses trades.db and generates a performance breakdown by signal type
(panic, drift, rpe, stink_bid).  Outputs a Markdown table to the console
and writes strategy_performance.json for automated review.

Usage:
    python scripts/strategy_tearsheet.py [--db PATH_TO_TRADES_DB]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ── Resolve default DB path ────────────────────────────────────────────────
DEFAULT_DB = Path("logs/trades.db")

SIGNAL_TYPES = ("panic", "drift", "rpe", "stink_bid")


def _query_trades(db_path: Path) -> list[dict]:
    """Load closed trades from SQLite."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT id, market_id, signal_type, meta_weight, "
            "entry_price, entry_size, exit_price, pnl_cents, "
            "hold_seconds, exit_reason, is_probe, created_at "
            "FROM trades WHERE state = 'CLOSED' "
            "ORDER BY exit_time ASC"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def _infer_signal_type(trade: dict) -> str:
    """Infer signal_type for legacy rows that lack the column.

    Heuristic:
      - id starts with 'RPE-'    → rpe
      - id starts with 'STINK-'  → stink_bid
      - otherwise                → panic (conservative default)
    """
    st = trade.get("signal_type") or ""
    if st:
        return st
    pos_id = trade.get("id", "")
    if pos_id.startswith("RPE-"):
        return "rpe"
    if pos_id.startswith("STINK-"):
        return "stink_bid"
    return "panic"


def _compute_metrics(trades: list[dict]) -> dict:
    """Compute performance metrics for a list of trades."""
    if not trades:
        return {
            "total_trades": 0,
            "total_pnl_cents": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_hold_seconds": 0.0,
            "max_drawdown_cents": 0.0,
            "capital_used_usd": 0.0,
            "strategy_efficiency": 0.0,
            "avg_meta_weight": 0.0,
        }

    pnls = [t["pnl_cents"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t["hold_seconds"] for t in trades if t.get("hold_seconds")]
    meta_weights = [t.get("meta_weight") or 1.0 for t in trades]

    total = len(pnls)
    total_pnl = sum(pnls)
    win_rate = len(wins) / total if total else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Max drawdown (cumulative)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    # Capital used = sum(entry_price * entry_size)
    capital_used = sum(
        (t.get("entry_price") or 0.0) * (t.get("entry_size") or 0.0)
        for t in trades
    )

    # Strategy Efficiency = PnL / Capital Used (per-dollar return)
    efficiency = (total_pnl / 100.0) / capital_used if capital_used > 0 else 0.0

    return {
        "total_trades": total,
        "total_pnl_cents": round(total_pnl, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "avg_hold_seconds": round(sum(holds) / len(holds), 1) if holds else 0.0,
        "max_drawdown_cents": round(max_dd, 2),
        "capital_used_usd": round(capital_used, 2),
        "strategy_efficiency": round(efficiency, 6),
        "avg_meta_weight": round(sum(meta_weights) / len(meta_weights), 3),
    }


def _build_tearsheet(db_path: Path) -> dict:
    """Build the full tearsheet: per-signal-type + portfolio aggregate."""
    trades = _query_trades(db_path)

    # Tag legacy rows
    for t in trades:
        t["signal_type"] = _infer_signal_type(t)

    # Group by signal type
    by_type: dict[str, list[dict]] = {st: [] for st in SIGNAL_TYPES}
    for t in trades:
        st = t["signal_type"]
        if st not in by_type:
            by_type[st] = []
        by_type[st].append(t)

    result: dict = {}
    for st in SIGNAL_TYPES:
        result[st] = _compute_metrics(by_type[st])

    result["portfolio"] = _compute_metrics(trades)

    return result


def _format_markdown(tearsheet: dict) -> str:
    """Format the tearsheet as a Markdown table."""
    lines: list[str] = []
    lines.append("# Pillar 16 — Alpha-Source Attribution Tearsheet")
    lines.append("")
    lines.append(
        "| Signal Type | Trades | Total PnL (¢) | Win Rate | Profit Factor | "
        "Avg Hold (s) | Max DD (¢) | Capital ($) | Efficiency | Meta Weight |"
    )
    lines.append(
        "|-------------|--------|---------------|----------|---------------|"
        "-------------|------------|-------------|------------|-------------|"
    )

    for st in list(SIGNAL_TYPES) + ["portfolio"]:
        m = tearsheet[st]
        pf = m["profit_factor"] if isinstance(m["profit_factor"], str) else f"{m['profit_factor']:.2f}"
        label = st.upper() if st != "portfolio" else "**PORTFOLIO**"
        lines.append(
            f"| {label} | {m['total_trades']} | {m['total_pnl_cents']:.2f} | "
            f"{m['win_rate']:.1%} | {pf} | "
            f"{m['avg_hold_seconds']:.0f} | {m['max_drawdown_cents']:.2f} | "
            f"{m['capital_used_usd']:.2f} | {m['strategy_efficiency']:.4%} | "
            f"{m['avg_meta_weight']:.2f} |"
        )

    lines.append("")
    lines.append("*Efficiency = PnL / Capital Used (return per dollar deployed)*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pillar 16 — Alpha-Source Attribution Tearsheet"
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help="Path to trades.db (default: logs/trades.db)",
    )
    parser.add_argument(
        "--json-out", type=Path, default=Path("strategy_performance.json"),
        help="Output path for JSON report (default: strategy_performance.json)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: database not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    tearsheet = _build_tearsheet(args.db)

    # Console output
    print(_format_markdown(tearsheet))

    # JSON output
    args.json_out.write_text(json.dumps(tearsheet, indent=2, default=str))
    print(f"\nJSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
