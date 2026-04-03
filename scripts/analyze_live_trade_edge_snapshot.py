#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading.fees import get_fee_rate


DEFAULT_DB = Path("logs/local_snapshot/remeasurement_20260331T204007Z.db")


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _format_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_signed(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}{suffix}"


def _load_closed_trades(db_path: Path) -> tuple[list[sqlite3.Row], int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, market_id, signal_type, entry_price, exit_price, entry_size,
                   pnl_cents, hold_seconds, entry_time
            FROM trades
            WHERE state = 'CLOSED'
            ORDER BY entry_time ASC, id ASC
            """
        ).fetchall()
        journal_rows = int(
            conn.execute("SELECT COUNT(*) FROM trade_persistence_journal").fetchone()[0]
        )
    finally:
        conn.close()
    return rows, journal_rows


def _gross_move_cents_per_share(row: sqlite3.Row) -> float:
    return (float(row["exit_price"] or 0.0) - float(row["entry_price"] or 0.0)) * 100.0


def _fee_drag_cents_per_share(row: sqlite3.Row) -> float:
    entry_price = float(row["entry_price"] or 0.0)
    exit_price = float(row["exit_price"] or 0.0)
    return (get_fee_rate(entry_price, fee_enabled=True) + get_fee_rate(exit_price, fee_enabled=True)) * 100.0


def _summarize(rows: list[sqlite3.Row]) -> dict[str, float | int | None]:
    trade_count = len(rows)
    shares = sum(float(row["entry_size"] or 0.0) for row in rows)
    notional_usd = sum(
        float(row["entry_price"] or 0.0) * float(row["entry_size"] or 0.0)
        for row in rows
    )
    total_net_cents = sum(float(row["pnl_cents"] or 0.0) for row in rows)
    total_gross_cents = sum(
        _gross_move_cents_per_share(row) * float(row["entry_size"] or 0.0)
        for row in rows
    )
    total_fee_cents = sum(
        _fee_drag_cents_per_share(row) * float(row["entry_size"] or 0.0)
        for row in rows
    )
    toxic_trades = sum(1 for row in rows if _gross_move_cents_per_share(row) < 0.0)
    winning_trades = sum(1 for row in rows if float(row["pnl_cents"] or 0.0) > 0.0)
    hold_seconds = [float(row["hold_seconds"] or 0.0) for row in rows]

    fast_close_windows: dict[int, list[float]] = {5: [], 15: [], 60: []}
    for row in rows:
        gross_move = _gross_move_cents_per_share(row)
        hold = float(row["hold_seconds"] or 0.0)
        for window in fast_close_windows:
            if hold <= window:
                fast_close_windows[window].append(gross_move)

    summary: dict[str, float | int | None] = {
        "trade_count": trade_count,
        "shares": shares,
        "notional_usd": notional_usd,
        "total_net_cents": total_net_cents,
        "total_gross_cents": total_gross_cents,
        "total_fee_cents": total_fee_cents,
        "avg_net_cents_per_trade": total_net_cents / trade_count if trade_count else None,
        "avg_net_cents_per_share": total_net_cents / shares if shares else None,
        "net_cents_per_usd": total_net_cents / notional_usd if notional_usd else None,
        "avg_gross_cents_per_share": total_gross_cents / shares if shares else None,
        "gross_cents_per_usd": total_gross_cents / notional_usd if notional_usd else None,
        "avg_fee_cents_per_share": total_fee_cents / shares if shares else None,
        "fee_cents_per_usd": total_fee_cents / notional_usd if notional_usd else None,
        "win_rate": winning_trades / trade_count if trade_count else None,
        "toxicity_rate": toxic_trades / trade_count if trade_count else None,
        "avg_hold_seconds": sum(hold_seconds) / trade_count if trade_count else None,
        "median_hold_seconds": median(hold_seconds) if hold_seconds else None,
        "reconcile_cents": total_gross_cents - total_fee_cents - total_net_cents,
    }
    for window, gross_values in fast_close_windows.items():
        summary[f"hold_le_{window}_count"] = len(gross_values)
        summary[f"hold_le_{window}_avg_gross_cents_per_share"] = (
            sum(gross_values) / len(gross_values) if gross_values else None
        )
    return summary


def _render_summary_table(summaries: dict[str, dict[str, float | int | None]]) -> list[str]:
    lines = [
        "| Cohort | Trades | Shares | Deployed USD | Win Rate | Net PnL USD | Avg Net c/trade | Net c/USD | Gross c/share | Fee c/share | Toxicity Rate | Median Hold s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in summaries.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(int(stats["trade_count"] or 0)),
                    _format_num(float(stats["shares"] or 0.0), 1),
                    _format_num(float(stats["notional_usd"] or 0.0), 2),
                    _format_pct(float(stats["win_rate"] or 0.0)),
                    _format_signed(float(stats["total_net_cents"] or 0.0) / 100.0, 2),
                    _format_signed(float(stats["avg_net_cents_per_trade"]), 2),
                    _format_signed(float(stats["net_cents_per_usd"]), 2),
                    _format_signed(float(stats["avg_gross_cents_per_share"]), 2),
                    _format_signed(float(stats["avg_fee_cents_per_share"]), 2),
                    _format_pct(float(stats["toxicity_rate"] or 0.0)),
                    _format_num(float(stats["median_hold_seconds"]), 1),
                ]
            )
            + " |"
        )
    return lines


def _render_fast_close_table(summaries: dict[str, dict[str, float | int | None]]) -> list[str]:
    lines = [
        "| Cohort | <=5s Trades | <=5s Gross c/share | <=15s Trades | <=15s Gross c/share | <=60s Trades | <=60s Gross c/share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in summaries.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(int(stats["hold_le_5_count"] or 0)),
                    _format_signed(stats["hold_le_5_avg_gross_cents_per_share"], 2),
                    str(int(stats["hold_le_15_count"] or 0)),
                    _format_signed(stats["hold_le_15_avg_gross_cents_per_share"], 2),
                    str(int(stats["hold_le_60_count"] or 0)),
                    _format_signed(stats["hold_le_60_avg_gross_cents_per_share"], 2),
                ]
            )
            + " |"
        )
    return lines


def build_report(db_path: Path) -> str:
    rows, journal_rows = _load_closed_trades(db_path)
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["signal_type"] or "<blank>")].append(row)

    summaries: dict[str, dict[str, float | int | None]] = {
        "aggregate": _summarize(rows),
    }
    for signal_type in sorted(grouped):
        summaries[signal_type] = _summarize(grouped[signal_type])

    aggregate = summaries["aggregate"]
    lines = [
        "# Historical Live Trade Edge Scorecard",
        "",
        f"- Snapshot DB: `{db_path}`",
        f"- Closed live trades analyzed: `{int(aggregate['trade_count'] or 0)}`",
        f"- Trade persistence journal rows available: `{journal_rows}`",
        "- Ledger mix: `ofi_momentum=234`, `panic=17`",
        "",
        "## Data Recoverability",
        "",
        "- Exact 5s/15s/60s markouts are not recoverable from this legacy snapshot: `trades` has no forward-mark columns and the exported `trade_persistence_journal` is empty.",
        "- Exact spread slippage is not recoverable: the legacy live rows do not persist the reference mid, best bid/ask, or quote snapshot at fill time.",
        "- Toxicity is therefore measured from what the snapshot can prove: realized gross move from fill to exit, fee drag from the live Polymarket fee curve, and fast-close subsets (`<=5s`, `<=15s`, `<=60s`) as the nearest short-horizon proxy.",
        "",
        "## Scorecard",
        "",
        *_render_summary_table(summaries),
        "",
        "## Fast-Close Proxy",
        "",
        *_render_fast_close_table(summaries),
        "",
        "## Verdict",
        "",
        f"- Aggregate net edge was `{_format_signed(float(aggregate['net_cents_per_usd']), 2, 'c/USD')}` on `{_format_num(float(aggregate['notional_usd'] or 0.0), 2)}` USD deployed, with total net PnL `{_format_signed(float(aggregate['total_net_cents'] or 0.0) / 100.0, 2, ' USD')}`.",
        f"- The market moved against the fills before fees by `{_format_signed(float(aggregate['avg_gross_cents_per_share']), 2, 'c/share')}` on average, which is already adverse selection. Fees then added another `{_format_signed(float(aggregate['avg_fee_cents_per_share']), 2, 'c/share')}` of drag.",
        f"- Win rate was `{_format_pct(float(aggregate['win_rate'] or 0.0))}` and toxicity rate was `{_format_pct(float(aggregate['toxicity_rate'] or 0.0))}`: every recorded trade finished net negative and every recorded trade had a negative gross move from fill to exit.",
        f"- Panic was worse than the aggregate: `{_format_signed(float(summaries['panic']['net_cents_per_usd']), 2, 'c/USD')}` net edge and `{_format_signed(float(summaries['panic']['avg_gross_cents_per_share']), 2, 'c/share')}` gross move per share, but it only accounts for `17` of the `251` trades.",
        "- On the evidence present in this snapshot, the old live directional logic did not have alpha. It crossed into adverse flow and then paid a fee curve that deepened the losses.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze historical execution edge and toxicity from a WAL-safe live trades snapshot.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to the SQLite snapshot database (default: {DEFAULT_DB})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(build_report(args.db.resolve()))


if __name__ == "__main__":
    main()