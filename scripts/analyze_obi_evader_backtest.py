from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from statistics import mean, median
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_universal_backtest import iter_replay_events, resolve_tick_root
from src.data.orderbook import OrderbookTracker
from src.trading.fees import get_fee_rate


HORIZONS = (5, 15, 60)
TAKER_BASELINE = {
    5: {"net_edge_cents_mean": -16.256, "net_pnl_usd_total": -88101.37, "win_rate": 0.0648},
    15: {"net_edge_cents_mean": -16.138, "net_pnl_usd_total": -87440.86, "win_rate": 0.0815},
    60: {"net_edge_cents_mean": -16.040, "net_pnl_usd_total": -86795.44, "win_rate": 0.1189},
}


@dataclass(slots=True)
class MakerFillRecord:
    fill_id: str
    asset_id: str
    quote_side: str
    entry_price: float
    entry_size: float
    entry_time_ms: int
    entry_day: str
    fill_mid_yes: float | None = None
    spread_cents: float | None = None
    future_mid_yes: dict[int, float | None] = field(default_factory=dict)

    def edge_vs_yes_mid(self, yes_mid: float | None) -> float | None:
        if yes_mid is None:
            return None
        if self.quote_side == "ASK":
            return (self.entry_price - yes_mid) * 100.0
        return (yes_mid - self.entry_price) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze maker fills from the ObiEvader universal replay.")
    parser.add_argument("--db", default="logs/universal_backtest.db")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--strategy-label", default="src.signals.obi_evader.ObiEvader")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _epoch_window(start_date: str, end_date: str) -> tuple[float, float]:
    lower = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    upper = (datetime.fromisoformat(end_date) + timedelta(days=1)).replace(tzinfo=timezone.utc)
    return lower.timestamp(), upper.timestamp()


def load_maker_fills(db_path: Path, *, start_date: str, end_date: str) -> list[MakerFillRecord]:
    start_ts, end_ts = _epoch_window(start_date, end_date)
    conn = sqlite3.connect(db_path)
    try:
        try:
            rows = conn.execute(
                """
                select id, asset_id, reference_price_band, entry_price, entry_size, entry_time
                from shadow_trades
                where exit_reason = 'MAKER_FILL' and reference_price_band like 'MAKER:%'
                  and entry_time >= ? and entry_time < ?
                order by asset_id, entry_time, id
                            """,
                (start_ts, end_ts),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table: shadow_trades" not in str(exc).lower():
                raise
            rows = []
    finally:
        conn.close()

    fills: list[MakerFillRecord] = []
    for row in rows:
        band = str(row[2])
        quote_side = band.split(":", 1)[1].upper() if ":" in band else "BID"
        fills.append(
            MakerFillRecord(
                fill_id=str(row[0]),
                asset_id=str(row[1]),
                quote_side=quote_side,
                entry_price=float(row[3]),
                entry_size=float(row[4]),
                entry_time_ms=int(round(float(row[5]) * 1000)),
                entry_day=datetime.fromtimestamp(float(row[5]), tz=timezone.utc).date().isoformat(),
                future_mid_yes={h: None for h in HORIZONS},
            )
        )
    return fills


def reconstruct_markouts(
    fills: list[MakerFillRecord],
    *,
    input_dir: Path,
    start_date: str,
    end_date: str,
) -> None:
    fills_by_asset: dict[str, list[MakerFillRecord]] = defaultdict(list)
    for fill in fills:
        fills_by_asset[fill.asset_id].append(fill)

    fill_idx = {asset_id: 0 for asset_id in fills_by_asset}
    horizon_idx = {(asset_id, horizon): 0 for asset_id in fills_by_asset for horizon in HORIZONS}
    trackers: dict[str, OrderbookTracker] = {}

    for event in iter_replay_events(resolve_tick_root(input_dir), start_date=start_date, end_date=end_date):
        if event.event_type not in {"BOOK", "PRICE_CHANGE"}:
            continue
        asset_fills = fills_by_asset.get(event.asset_id)
        if not asset_fills:
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


def summarize(fills: list[MakerFillRecord]) -> dict[str, object]:
    quote_counts = Counter(fill.quote_side for fill in fills)
    immediate_edges = [edge for fill in fills if (edge := fill.edge_vs_yes_mid(fill.fill_mid_yes)) is not None]
    spreads = [fill.spread_cents for fill in fills if fill.spread_cents is not None]
    horizons: dict[int, dict[str, float | int]] = {}

    for horizon in HORIZONS:
        gross_edges: list[float] = []
        net_edges: list[float] = []
        adverse_selection: list[float] = []
        pnl_cents: list[float] = []
        wins = 0
        coverage = 0
        for fill in fills:
            gross_edge = fill.edge_vs_yes_mid(fill.future_mid_yes[horizon])
            immediate_edge = fill.edge_vs_yes_mid(fill.fill_mid_yes)
            if gross_edge is None or immediate_edge is None:
                continue
            coverage += 1
            exit_fee_cents = get_fee_rate(fill.future_mid_yes[horizon] or 0.0) * 100.0
            conservative_net_edge = gross_edge - exit_fee_cents
            gross_edges.append(gross_edge)
            net_edges.append(conservative_net_edge)
            adverse_selection.append(immediate_edge - gross_edge)
            pnl_cents.append(conservative_net_edge * fill.entry_size)
            if conservative_net_edge > 0:
                wins += 1

        horizons[horizon] = {
            "coverage": coverage,
            "win_rate": (wins / coverage) if coverage else 0.0,
            "gross_edge_cents_mean": _avg(gross_edges),
            "gross_edge_cents_median": median(gross_edges) if gross_edges else 0.0,
            "net_edge_cents_mean": _avg(net_edges),
            "net_edge_cents_median": median(net_edges) if net_edges else 0.0,
            "adverse_selection_cents_mean": _avg(adverse_selection),
            "net_pnl_cents_total": sum(pnl_cents),
        }

    return {
        "total_fills": len(fills),
        "quote_counts": dict(quote_counts),
        "avg_full_spread_cents": _avg(value for value in spreads if value is not None),
        "avg_capture_at_fill_cents": _avg(value for value in immediate_edges if value is not None),
        "horizons": horizons,
        "daily": summarize_daily(fills),
    }


def summarize_daily(fills: list[MakerFillRecord]) -> dict[str, dict[str, float | int]]:
    by_day: dict[str, list[MakerFillRecord]] = defaultdict(list)
    for fill in fills:
        by_day[fill.entry_day].append(fill)

    daily: dict[str, dict[str, float | int]] = {}
    for day, day_fills in sorted(by_day.items()):
        total_pnl_cents = 0.0
        coverage = 0
        wins = 0
        net_edges = []
        for fill in day_fills:
            gross_edge = fill.edge_vs_yes_mid(fill.future_mid_yes[60])
            if gross_edge is None:
                continue
            coverage += 1
            exit_fee_cents = get_fee_rate(fill.future_mid_yes[60] or 0.0) * 100.0
            net_edge = gross_edge - exit_fee_cents
            net_edges.append(net_edge)
            total_pnl_cents += net_edge * fill.entry_size
            if net_edge > 0:
                wins += 1
        daily[day] = {
            "fills": len(day_fills),
            "coverage_60s": coverage,
            "win_rate_60s": (wins / coverage) if coverage else 0.0,
            "net_edge_60s_cents_mean": _avg(net_edges),
            "net_pnl_usd_total": total_pnl_cents / 100.0,
        }
    return daily


def render_markdown(
    summary: dict[str, object],
    *,
    date_label: str,
    db_path: Path,
    input_dir: Path,
    strategy_label: str,
) -> str:
    quote_counts = summary["quote_counts"]
    horizons = summary["horizons"]
    daily = summary["daily"]
    lines = [
        f"# OBI Evader Maker Backtest Report ({date_label})",
        "",
        "## Setup",
        "",
        f"- Strategy: `{strategy_label}`",
        f"- Replay source: `{input_dir}`",
        f"- Fill database: `{db_path}`",
        f"- Maker fills analyzed: `{summary['total_fills']:,}`",
        f"- Passive bid fills: `{int(quote_counts.get('BID', 0)):,}`",
        f"- Passive ask fills: `{int(quote_counts.get('ASK', 0)):,}`",
        f"- Average observed full spread at fill: `{summary['avg_full_spread_cents']:.3f}` cents",
        f"- Average captured edge at fill vs mid: `{summary['avg_capture_at_fill_cents']:.3f}` cents/share",
        "",
        "## Maker Markouts",
        "",
        "| Horizon | Coverage | Win Rate | Gross Maker Edge (c/share) | Conservative Net Edge (c/share) | Mean Adverse Selection (c/share) | Total Conservative PnL (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in HORIZONS:
        row = horizons[horizon]
        lines.append(
            "| {h}s | {coverage:,} | {win_rate:.2%} | {gross:.3f} | {net:.3f} | {adverse:.3f} | {pnl:.2f} |".format(
                h=horizon,
                coverage=int(row["coverage"]),
                win_rate=float(row["win_rate"]),
                gross=float(row["gross_edge_cents_mean"]),
                net=float(row["net_edge_cents_mean"]),
                adverse=float(row["adverse_selection_cents_mean"]),
                pnl=float(row["net_pnl_cents_total"]) / 100.0,
            )
        )

    lines.extend(
        [
            "",
            "## Daily 60s Breakdown",
            "",
            "| Day | Fills | 60s Coverage | 60s Win Rate | 60s Net Edge (c/share) | 60s Net PnL (USD) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if daily:
        for day, row in daily.items():
            lines.append(
                "| {day} | {fills:,} | {coverage:,} | {win_rate:.2%} | {edge:.3f} | {pnl:.2f} |".format(
                    day=day,
                    fills=int(row["fills"]),
                    coverage=int(row["coverage_60s"]),
                    win_rate=float(row["win_rate_60s"]),
                    edge=float(row["net_edge_60s_cents_mean"]),
                    pnl=float(row["net_pnl_usd_total"]),
                )
            )
    else:
        lines.append("| none | 0 | 0 | 0.00% | 0.000 | 0.00 |")

    lines.extend(
        [
            "",
            "## Comparison vs Prior Taker OBI",
            "",
            "| Horizon | Taker Net Edge (c/share) | Maker Net Edge (c/share) | Improvement (c/share) | Taker Total PnL (USD) | Maker Total PnL (USD) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon in HORIZONS:
        maker_row = horizons[horizon]
        taker_row = TAKER_BASELINE[horizon]
        maker_net = float(maker_row["net_edge_cents_mean"])
        taker_net = float(taker_row["net_edge_cents_mean"])
        lines.append(
            "| {h}s | {taker:.3f} | {maker:.3f} | {delta:.3f} | {taker_pnl:.2f} | {maker_pnl:.2f} |".format(
                h=horizon,
                taker=taker_net,
                maker=maker_net,
                delta=maker_net - taker_net,
                taker_pnl=float(taker_row["net_pnl_usd_total"]),
                maker_pnl=float(maker_row["net_pnl_cents_total"]) / 100.0,
            )
        )

    sixty = horizons[60]
    lines.extend(["", "## Verdict", ""])
    if int(summary["total_fills"]) == 0:
        lines.extend(
            [
                "The spread-gated replay produced **no maker fills** over this window, so there is no realized 60s edge to score.",
                "On this baseline day the `> 2.0c` gate filtered the strategy down to zero executed fills, which means the offline wide-spread result did not translate into realized activity on this run.",
            ]
        )
    else:
        verdict = "positive maker edge" if float(sixty["net_edge_cents_mean"]) > 0 else "still toxic"
        lines.extend(
            [
                (
                    "The OBI evasion layer produces **{verdict}** at 60 seconds: conservative net edge is "
                    "`{edge:.3f}` cents/share versus `{taker_edge:.3f}` cents/share for the prior taker variant."
                ).format(
                    verdict=verdict,
                    edge=float(sixty["net_edge_cents_mean"]),
                    taker_edge=float(TAKER_BASELINE[60]["net_edge_cents_mean"]),
                ),
                (
                    "Average adverse selection at 60 seconds is `{adverse:.3f}` cents/share, which is {assessment}."
                ).format(
                    adverse=float(sixty["adverse_selection_cents_mean"]),
                    assessment="below the captured spread and consistent with viable passive quoting" if float(sixty["net_edge_cents_mean"]) > 0 else "still overwhelming the captured spread",
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    input_dir = Path(args.input_dir)
    fills = load_maker_fills(db_path, start_date=args.start_date, end_date=args.end_date)
    reconstruct_markouts(fills, input_dir=input_dir, start_date=args.start_date, end_date=args.end_date)
    summary = summarize(fills)
    markdown = render_markdown(
        summary,
        date_label=f"{args.start_date} to {args.end_date}",
        db_path=db_path,
        input_dir=input_dir,
        strategy_label=args.strategy_label,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n---MARKDOWN---\n")
    print(markdown)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()