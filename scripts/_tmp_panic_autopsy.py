from __future__ import annotations

import argparse
import sqlite3
import statistics
from collections import Counter
from pathlib import Path


def _load_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM trades ORDER BY entry_time"))
    finally:
        conn.close()


def _summarize(rows: list[sqlite3.Row], *, label: str) -> list[str]:
    pnl = [float(row["pnl_cents"] or 0.0) for row in rows]
    holds = [float(row["hold_seconds"] or 0.0) for row in rows]
    price_move_cents = [
        (float(row["exit_price"] or 0.0) - float(row["entry_price"] or 0.0)) * 100.0
        for row in rows
    ]
    entry_fees = [float(row["entry_fee_bps"] or 0.0) for row in rows]

    lines = [f"## {label}"]
    lines.append(f"- Trades: {len(rows)}")
    lines.append(f"- Exit reasons: {dict(Counter(str(row['exit_reason']) for row in rows))}")
    lines.append(f"- Wins: {sum(1 for value in pnl if value > 0)}")
    lines.append(f"- Average pnl per trade: {sum(pnl) / len(pnl):.2f} cents")
    lines.append(f"- Median pnl per trade: {statistics.median(pnl):.2f} cents")
    lines.append(f"- Average hold: {sum(holds) / len(holds):.1f}s")
    lines.append(f"- Median hold: {statistics.median(holds):.1f}s")
    lines.append(f"- Average entry->exit move: {sum(price_move_cents) / len(price_move_cents):.2f} cents")
    lines.append(f"- Median entry->exit move: {statistics.median(price_move_cents):.2f} cents")
    lines.append(f"- Average entry fee: {sum(entry_fees) / len(entry_fees):.1f} bps")

    stop_rows = [row for row in rows if str(row["exit_reason"]) == "stop_loss"]
    if stop_rows:
        stop_holds = [float(row["hold_seconds"] or 0.0) for row in stop_rows]
        lines.append(f"- Stop-loss trades: {len(stop_rows)}")
        lines.append(f"- Median stop-loss hold: {statistics.median(stop_holds):.1f}s")
        lines.append(f"- Stop-loss hold <= 1s: {sum(1 for value in stop_holds if value <= 1)}")
        lines.append(f"- Stop-loss hold <= 10s: {sum(1 for value in stop_holds if value <= 10)}")
        lines.append(f"- Stop-loss hold <= 60s: {sum(1 for value in stop_holds if value <= 60)}")

    worst = sorted(rows, key=lambda row: float(row["pnl_cents"] or 0.0))[:5]
    lines.append("- Worst trades:")
    for row in worst:
        lines.append(
            "  "
            f"{row['id']} | {row['signal_type']} | {row['exit_reason']} | "
            f"entry={float(row['entry_price']):.4f} exit={float(row['exit_price']):.4f} | "
            f"hold={float(row['hold_seconds']):.1f}s | pnl={float(row['pnl_cents']):.2f}c"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize autopsy_db.sqlite trade outcomes")
    parser.add_argument(
        "--db",
        default="data/autopsy_db.sqlite",
        help="Path to the extracted SQLite autopsy database",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    rows = _load_rows(db_path)
    if not rows:
        raise SystemExit("No trades found in database")

    panic_rows = [row for row in rows if str(row["signal_type"]) == "panic"]
    ofi_rows = [row for row in rows if str(row["signal_type"]) == "ofi_momentum"]

    output: list[str] = []
    output.extend(_summarize(rows, label="All Trades"))
    output.append("")
    if panic_rows:
        output.extend(_summarize(panic_rows, label="Panic Trades"))
        output.append("")
    if ofi_rows:
        output.extend(_summarize(ofi_rows, label="OFI Momentum Trades"))

    print("\n".join(output))


if __name__ == "__main__":
    main()