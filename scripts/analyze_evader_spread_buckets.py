from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_obi_evader_backtest import load_maker_fills, reconstruct_markouts
from src.trading.fees import get_fee_rate


BUCKETS = (
    ("<= 0.5c", None, 0.5),
    ("0.5c - 1.0c", 0.5, 1.0),
    ("1.0c - 2.0c", 1.0, 2.0),
    ("> 2.0c", 2.0, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket ObiEvader maker fills by spread width at fill.")
    parser.add_argument("--db", default="logs/universal_backtest.db")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", default="docs/evader_spread_analysis.md")
    return parser.parse_args()


def _bucket_for_spread(spread_cents: float) -> str:
    for label, lower, upper in BUCKETS:
        if lower is not None and spread_cents <= lower:
            continue
        if upper is not None and spread_cents > upper:
            continue
        return label
    return "> 2.0c"


def summarize_buckets(fills: list[object]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {
        label: {
            "fills": 0,
            "coverage_60s": 0,
            "wins_60s": 0,
            "net_edge_60s_cents_total": 0.0,
            "net_pnl_60s_cents_total": 0.0,
        }
        for label, _, _ in BUCKETS
    }

    for fill in fills:
        spread_cents = getattr(fill, "spread_cents", None)
        future_mid = getattr(fill, "future_mid_yes", {}).get(60)
        if spread_cents is None:
            continue
        bucket = summary[_bucket_for_spread(float(spread_cents))]
        bucket["fills"] += 1

        gross_edge = fill.edge_vs_yes_mid(future_mid)
        if gross_edge is None:
            continue
        exit_fee_cents = get_fee_rate(future_mid or 0.0) * 100.0
        net_edge = gross_edge - exit_fee_cents
        bucket["coverage_60s"] += 1
        bucket["net_edge_60s_cents_total"] += net_edge
        bucket["net_pnl_60s_cents_total"] += net_edge * float(getattr(fill, "entry_size"))
        if net_edge > 0:
            bucket["wins_60s"] += 1

    for bucket in summary.values():
        coverage = int(bucket["coverage_60s"])
        bucket["win_rate_60s"] = (int(bucket["wins_60s"]) / coverage) if coverage else 0.0
        bucket["avg_net_edge_60s_cents"] = (float(bucket["net_edge_60s_cents_total"]) / coverage) if coverage else 0.0
        bucket["total_pnl_60s_usd"] = float(bucket["net_pnl_60s_cents_total"]) / 100.0
    return summary


def render_markdown(summary: dict[str, dict[str, float | int]], *, start_date: str, end_date: str, total_fills: int) -> str:
    profitable = [label for label, row in summary.items() if float(row["avg_net_edge_60s_cents"]) > 0]
    verdict = (
        "Only the widest spread buckets were profitable at 60s." if profitable else "No spread bucket was profitable at 60s."
    )
    lines = [
        f"# ObiEvader Spread-Width Analysis ({start_date} to {end_date})",
        "",
        "## Scope",
        "",
        f"- Existing maker fills analyzed from `logs/universal_backtest.db`: `{total_fills:,}`",
        "- No new backtests were run; fill-time spread and 60s markouts were reconstructed from the archived L2 tape.",
        "- Buckets are based on total quoted spread at the moment the maker fill was observed.",
        "",
        "## 60s Performance by Fill-Time Spread",
        "",
        "| Spread Bucket | Total Fills | 60s Coverage | 60s Win Rate | Avg 60s Conservative Net Edge (c/share) | Total 60s PnL (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _, _ in BUCKETS:
        row = summary[label]
        lines.append(
            "| {label} | {fills:,} | {coverage:,} | {win_rate:.2%} | {edge:.3f} | {pnl:.2f} |".format(
                label=label,
                fills=int(row["fills"]),
                coverage=int(row["coverage_60s"]),
                win_rate=float(row["win_rate_60s"]),
                edge=float(row["avg_net_edge_60s_cents"]),
                pnl=float(row["total_pnl_60s_usd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict,
        ]
    )
    if profitable:
        lines.append("Profitable buckets: " + ", ".join(f"`{label}`" for label in profitable) + ".")
    else:
        best_label, best_row = max(summary.items(), key=lambda item: float(item[1]["avg_net_edge_60s_cents"]))
        lines.append(
            "Best bucket was `{label}` at `{edge:.3f}` c/share, which remains below zero.".format(
                label=best_label,
                edge=float(best_row["avg_net_edge_60s_cents"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    fills = load_maker_fills(Path(args.db), start_date=args.start_date, end_date=args.end_date)
    reconstruct_markouts(
        fills,
        input_dir=Path(args.input_dir),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    summary = summarize_buckets(fills)
    markdown = render_markdown(
        summary,
        start_date=args.start_date,
        end_date=args.end_date,
        total_fills=len(fills),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n---MARKDOWN---\n")
    print(markdown)


if __name__ == "__main__":
    main()