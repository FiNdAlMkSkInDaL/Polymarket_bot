#!/usr/bin/env python3
"""CLI wrapper for SQLite-backed OFI toxicity TCA summaries.

This script no longer reads flat log files. It imports the public SQLite
aggregation helper from ``src.monitoring.trade_store`` and formats the returned
bucket summaries for terminal or JSON output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitoring.trade_store import get_ofi_toxicity_pnl_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print SQLite-backed OFI momentum toxicity PnL buckets.",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=10,
        help="Number of fixed-width toxicity buckets over [0, 1] (default: 10).",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum trade count required to print a bucket (default: 1).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional SQLite path. Defaults to the trade store runtime path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the bucket summary as JSON.",
    )
    return parser.parse_args()


def render_summary(rows: list[dict[str, Any]], min_trades: int) -> str:
    lines: list[str] = []
    lines.append("OFI Momentum Toxicity TCA")
    lines.append("=" * 72)
    total_trades = sum(int(row.get("trade_count", 0) or 0) for row in rows)
    total_net_pnl = sum(float(row.get("total_net_pnl_cents", 0.0) or 0.0) for row in rows)
    total_fee_drag = sum(float(row.get("total_taker_fee_drag_cents", 0.0) or 0.0) for row in rows)
    lines.append(f"Trades aggregated: {total_trades}")

    if not rows:
        lines.append("No closed OFI momentum trades were found in the SQLite store.")
        return "\n".join(lines)

    lines.append(
        "Overall averages: "
        f"avg_net_pnl={total_net_pnl / max(1, total_trades):.3f}c "
        f"avg_fee_drag={total_fee_drag / max(1, total_trades):.3f}c"
    )
    lines.append("")
    lines.append(
        "Bucket  Tox Range        Trades  Win%   Avg Net(c)   Total Net(c)  Total Fee(c)"
    )
    lines.append("-" * 72)
    for row in rows:
        trade_count = int(row.get("trade_count", 0) or 0)
        if trade_count < min_trades:
            continue
        win_rate_pct = float(row.get("win_rate", 0.0) or 0.0) * 100.0
        lines.append(
            f"B{int(row.get('bucket_index', 0)) + 1:<5}"
            f"{float(row.get('toxicity_min', 0.0) or 0.0):>6.3f}-{float(row.get('toxicity_max', 0.0) or 0.0):<6.3f}"
            f"{trade_count:>8}"
            f"{win_rate_pct:>7.1f}"
            f"{float(row.get('avg_net_pnl_cents', 0.0) or 0.0):>13.3f}"
            f"{float(row.get('total_net_pnl_cents', 0.0) or 0.0):>15.3f}"
            f"{float(row.get('total_taker_fee_drag_cents', 0.0) or 0.0):>14.3f}"
        )
    return "\n".join(lines)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trade_count": sum(int(row.get("trade_count", 0) or 0) for row in rows),
        "bucket_summary": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def _load_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    return await get_ofi_toxicity_pnl_summary(
        buckets=args.buckets,
        db_path=args.db_path,
    )


def main() -> int:
    args = parse_args()
    if args.buckets <= 0:
        raise SystemExit("--buckets must be >= 1")
    rows = asyncio.run(_load_summary(args))
    print(render_summary(rows, args.min_trades))

    if args.output_json is not None:
        write_json(args.output_json, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())