from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sys
from statistics import mean, median
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.fees import get_fee_rate

from scripts.run_universal_backtest import iter_replay_events, resolve_tick_root
from src.data.orderbook import OrderbookTracker


HORIZONS = (5, 15, 60)


@dataclass(slots=True)
class FillRecord:
    fill_id: str
    asset_id: str
    direction: str
    entry_price: float
    entry_size: float
    entry_time_ms: int
    signal_source: str
    fill_mid_yes: float | None = None
    spread_cents: float | None = None
    future_mid_yes: dict[int, float | None] = field(default_factory=dict)

    def instrument_mid(self, yes_mid: float | None) -> float | None:
        if yes_mid is None:
            return None
        if self.direction.upper() == "YES":
            return yes_mid
        return 1.0 - yes_mid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OBI universal backtest fills and markouts.")
    parser.add_argument("--db", default="logs/universal_backtest.db")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", default=None, help="Optional path to write markdown report.")
    return parser.parse_args()


def load_fills(db_path: Path) -> list[FillRecord]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            select id, asset_id, direction, entry_price, entry_size, entry_time, signal_source
            from shadow_trades
            where exit_reason = 'TAKER_FILL'
            order by asset_id, entry_time, id
            """
        ).fetchall()
    finally:
        conn.close()

    fills: list[FillRecord] = []
    for row in rows:
        fills.append(
            FillRecord(
                fill_id=str(row[0]),
                asset_id=str(row[1]),
                direction=str(row[2]),
                entry_price=float(row[3]),
                entry_size=float(row[4]),
                entry_time_ms=int(round(float(row[5]) * 1000)),
                signal_source=str(row[6]),
                future_mid_yes={h: None for h in HORIZONS},
            )
        )
    return fills


def reconstruct_markouts(
    fills: list[FillRecord],
    *,
    input_dir: Path,
    start_date: str,
    end_date: str,
) -> None:
    fills_by_asset: dict[str, list[FillRecord]] = defaultdict(list)
    for fill in fills:
        fills_by_asset[fill.asset_id].append(fill)

    fill_idx = {asset_id: 0 for asset_id in fills_by_asset}
    horizon_idx = {(asset_id, horizon): 0 for asset_id in fills_by_asset for horizon in HORIZONS}
    trackers: dict[str, OrderbookTracker] = {}

    for event in iter_replay_events(resolve_tick_root(input_dir), start_date=start_date, end_date=end_date):
        if event.event_type not in {"BOOK", "PRICE_CHANGE"}:
            continue
        tracker = trackers.get(event.asset_id)
        if tracker is None:
            tracker = OrderbookTracker(event.asset_id)
            trackers[event.asset_id] = tracker

        if event.event_type == "BOOK":
            tracker.on_book_snapshot(event.payload)
        else:
            tracker.on_price_change(event.payload)

        snapshot = tracker.snapshot()
        if snapshot.best_bid <= 0 or snapshot.best_ask <= 0:
            continue
        yes_mid = snapshot.mid_price
        asset_fills = fills_by_asset.get(event.asset_id)
        if not asset_fills:
            continue

        idx = fill_idx[event.asset_id]
        while idx < len(asset_fills) and asset_fills[idx].entry_time_ms <= event.timestamp_ms:
            fill = asset_fills[idx]
            if fill.fill_mid_yes is None:
                fill.fill_mid_yes = yes_mid
                fill.spread_cents = snapshot.spread * 100.0
            idx += 1
        fill_idx[event.asset_id] = idx

        for horizon in HORIZONS:
            key = (event.asset_id, horizon)
            idx = horizon_idx[key]
            while idx < len(asset_fills):
                fill = asset_fills[idx]
                target_time_ms = fill.entry_time_ms + horizon * 1000
                if target_time_ms > event.timestamp_ms:
                    break
                if fill.future_mid_yes[horizon] is None:
                    fill.future_mid_yes[horizon] = yes_mid
                idx += 1
            horizon_idx[key] = idx


def _avg(values: Iterable[float]) -> float:
    vals = [value for value in values]
    return mean(vals) if vals else 0.0


def summarize(fills: list[FillRecord]) -> dict[str, object]:
    side_counts = Counter(fill.direction.upper() for fill in fills)
    valid_spreads = [fill.spread_cents for fill in fills if fill.spread_cents is not None]
    half_spread_paid = []
    fill_mid_gaps = []
    for fill in fills:
        instrument_mid = fill.instrument_mid(fill.fill_mid_yes)
        if instrument_mid is None:
            continue
        fill_mid_gaps.append((fill.entry_price - instrument_mid) * 100.0)
        if fill.spread_cents is not None:
            half_spread_paid.append(fill.spread_cents / 2.0)

    horizons: dict[int, dict[str, float | int]] = {}
    for horizon in HORIZONS:
        gross_edges: list[float] = []
        net_edges: list[float] = []
        pnl_cents: list[float] = []
        wins = 0
        coverage = 0
        for fill in fills:
            instrument_mid = fill.instrument_mid(fill.future_mid_yes[horizon])
            if instrument_mid is None:
                continue
            coverage += 1
            gross_edge = (instrument_mid - fill.entry_price) * 100.0
            fee_cents = (get_fee_rate(fill.entry_price) + get_fee_rate(instrument_mid)) * 100.0
            net_edge = gross_edge - fee_cents
            gross_edges.append(gross_edge)
            net_edges.append(net_edge)
            pnl_cents.append(net_edge * fill.entry_size)
            if net_edge > 0:
                wins += 1

        horizons[horizon] = {
            "coverage": coverage,
            "win_rate": (wins / coverage) if coverage else 0.0,
            "gross_edge_cents_mean": _avg(gross_edges),
            "gross_edge_cents_median": median(gross_edges) if gross_edges else 0.0,
            "net_edge_cents_mean": _avg(net_edges),
            "net_edge_cents_median": median(net_edges) if net_edges else 0.0,
            "net_pnl_cents_mean": _avg(pnl_cents),
            "net_pnl_cents_total": sum(pnl_cents),
        }

    return {
        "total_fills": len(fills),
        "side_counts": dict(side_counts),
        "avg_full_spread_cents": _avg(value for value in valid_spreads if value is not None),
        "avg_half_spread_paid_cents": _avg(value for value in half_spread_paid if value is not None),
        "avg_fill_vs_mid_cents": _avg(value for value in fill_mid_gaps if value is not None),
        "horizons": horizons,
    }


def render_markdown(summary: dict[str, object], *, date_label: str, db_path: Path, input_dir: Path) -> str:
    side_counts = summary["side_counts"]
    horizons = summary["horizons"]
    lines = [
        f"# OBI Scalper Backtest Report ({date_label})",
        "",
        "## Setup",
        "",
        f"- Strategy: `src.signals.obi_scalper.ObiScalper`",
        f"- Replay source: `{input_dir}`",
        f"- Persisted fills DB: `{db_path}`",
        f"- Simulated fills: `{summary['total_fills']:,}`",
        f"- YES fills: `{int(side_counts.get('YES', 0)):,}`",
        f"- NO fills: `{int(side_counts.get('NO', 0)):,}`",
        "",
        "## Execution Cost",
        "",
        f"- Average observed full spread: `{summary['avg_full_spread_cents']:.3f}` cents",
        f"- Average taker crossing cost (half-spread): `{summary['avg_half_spread_paid_cents']:.3f}` cents/share",
        f"- Average fill minus prevailing instrument mid: `{summary['avg_fill_vs_mid_cents']:.3f}` cents/share",
        "",
        "## Forward Markouts",
        "",
        "| Horizon | Coverage | Win Rate | Gross Edge (mean, c/share) | Net Edge After Fees (mean, c/share) | Net Edge Median (c/share) | Mean Net PnL (c/fill) | Total Net PnL (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in HORIZONS:
        row = horizons[horizon]
        lines.append(
            "| {h}s | {coverage:,} | {win_rate:.2%} | {gross:.3f} | {net:.3f} | {net_med:.3f} | {net_fill:.3f} | {net_total:.2f} |".format(
                h=horizon,
                coverage=int(row["coverage"]),
                win_rate=float(row["win_rate"]),
                gross=float(row["gross_edge_cents_mean"]),
                net=float(row["net_edge_cents_mean"]),
                net_med=float(row["net_edge_cents_median"]),
                net_fill=float(row["net_pnl_cents_mean"]),
                net_total=float(row["net_pnl_cents_total"]) / 100.0,
            )
        )

    sixty = horizons[60]
    verdict = "profitable" if float(sixty["net_edge_cents_mean"]) > 0 else "toxic"
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "The 60-second post-fill markout is **{verdict}** on a fee-adjusted basis: "
                "mean net edge is `{edge:.3f}` cents/share with a `{win_rate:.2%}` win rate."
            ).format(
                verdict=verdict,
                edge=float(sixty["net_edge_cents_mean"]),
                win_rate=float(sixty["win_rate"]),
            ),
            (
                "If the 60-second horizon is the decision horizon, the signal is {decision}."
            ).format(
                decision="worth deeper refinement" if verdict == "profitable" else "not viable as a taker strategy at the current threshold",
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    input_dir = Path(args.input_dir)
    fills = load_fills(db_path)
    reconstruct_markouts(fills, input_dir=input_dir, start_date=args.start_date, end_date=args.end_date)
    summary = summarize(fills)
    markdown = render_markdown(
        summary,
        date_label=f"{args.start_date} to {args.end_date}",
        db_path=db_path,
        input_dir=input_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n---MARKDOWN---\n")
    print(markdown)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()