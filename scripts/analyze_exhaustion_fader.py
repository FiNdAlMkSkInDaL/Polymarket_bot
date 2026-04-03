from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from statistics import mean, median
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_universal_backtest import iter_file_events, resolve_tick_root
from src.data.orderbook import OrderbookTracker


HORIZON_SECONDS = 60


@dataclass(slots=True)
class FadeFillRecord:
    fill_id: str
    market_id: str
    asset_id: str
    quote_side: str
    entry_price: float
    entry_size: float
    entry_time_ms: int
    fill_mid_yes: float | None = None
    spread_cents: float | None = None
    future_mid_yes: float | None = None

    def edge_vs_yes_mid(self, yes_mid: float | None) -> float | None:
        if yes_mid is None:
            return None
        if self.quote_side == "ASK":
            return (self.entry_price - yes_mid) * 100.0
        return (yes_mid - self.entry_price) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ExhaustionFader maker fills and 60s markouts.")
    parser.add_argument("--db", default="logs/universal_backtest_exhaustion_2026-03-25.db")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", default=None, help="Optional path to write the markdown scorecard.")
    parser.add_argument("--progress-every-events", type=int, default=None, help="Optional heartbeat interval for long markout reconstruction runs.")
    return parser.parse_args()


def _epoch_window(start_date: str, end_date: str) -> tuple[float, float]:
    lower = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    upper = (datetime.fromisoformat(end_date) + timedelta(days=1)).replace(tzinfo=timezone.utc)
    return lower.timestamp(), upper.timestamp()


def load_fade_fills(db_path: Path, *, start_date: str, end_date: str) -> list[FadeFillRecord]:
    start_ts, end_ts = _epoch_window(start_date, end_date)
    conn = sqlite3.connect(db_path)
    try:
        table_row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'shadow_trades'"
        ).fetchone()
        if table_row is None:
            return []
        rows = conn.execute(
            """
            select id, market_id, asset_id, reference_price_band, entry_price, entry_size, entry_time
            from shadow_trades
            where exit_reason = 'MAKER_FILL'
              and reference_price_band like 'MAKER:%'
              and entry_time >= ? and entry_time < ?
            order by asset_id, entry_time, id
            """,
            (start_ts, end_ts),
        ).fetchall()
    finally:
        conn.close()

    fills: list[FadeFillRecord] = []
    for row in rows:
        reference_price_band = str(row[3])
        quote_side = reference_price_band.split(":", 1)[1].upper() if ":" in reference_price_band else "BID"
        fills.append(
            FadeFillRecord(
                fill_id=str(row[0]),
                market_id=str(row[1]),
                asset_id=str(row[2]),
                quote_side=quote_side,
                entry_price=float(row[4]),
                entry_size=float(row[5]),
                entry_time_ms=int(round(float(row[6]) * 1000)),
            )
        )
    return fills


def reconstruct_markouts(
    fills: list[FadeFillRecord],
    *,
    input_dir: Path,
    start_date: str,
    end_date: str,
    progress_every_events: int | None = None,
    progress_label: str | None = None,
) -> None:
    if progress_every_events is not None and progress_every_events <= 0:
        raise ValueError("progress_every_events must be strictly positive when provided")
    if not fills:
        if progress_every_events is not None:
            normalized_progress_label = str(progress_label or f"{start_date}:{end_date}").strip() or f"{start_date}:{end_date}"
            print(
                "analysis_complete label={label} events=0 fill_snapshots_captured=0 markouts_captured=0".format(
                    label=normalized_progress_label,
                ),
                flush=True,
            )
        return
    fills_by_asset: dict[str, list[FadeFillRecord]] = defaultdict(list)
    for fill in fills:
        fills_by_asset[fill.asset_id].append(fill)

    fill_index = {asset_id: 0 for asset_id in fills_by_asset}
    future_index = {asset_id: 0 for asset_id in fills_by_asset}
    trackers: dict[str, OrderbookTracker] = {}
    processed_events = 0
    next_progress_event = progress_every_events
    normalized_progress_label = str(progress_label or f"{start_date}:{end_date}").strip() or f"{start_date}:{end_date}"

    if next_progress_event is not None:
        print(
            "analysis_start label={label} fills={fills} input_dir={input_dir}".format(
                label=normalized_progress_label,
                fills=len(fills),
                input_dir=input_dir,
            ),
            flush=True,
        )

    tick_root = resolve_tick_root(input_dir)
    relevant_asset_ids = sorted(fills_by_asset)
    for event in _iter_relevant_asset_events(
        tick_root,
        asset_ids=relevant_asset_ids,
        start_date=start_date,
        end_date=end_date,
    ):
        if event.event_type not in {"BOOK", "PRICE_CHANGE"}:
            continue
        processed_events += 1
        asset_fills = fills_by_asset.get(event.asset_id)
        if not asset_fills:
            if next_progress_event is not None and processed_events >= next_progress_event:
                print(
                    "analysis_progress label={label} events={events} fill_snapshots_captured={fills_captured} markouts_captured={markouts_captured}".format(
                        label=normalized_progress_label,
                        events=processed_events,
                        fills_captured=sum(fill_index.values()),
                        markouts_captured=sum(future_index.values()),
                    ),
                    flush=True,
                )
                next_progress_event += progress_every_events
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

        idx = fill_index[event.asset_id]
        while idx < len(asset_fills) and asset_fills[idx].entry_time_ms <= event.timestamp_ms:
            fill = asset_fills[idx]
            if fill.fill_mid_yes is None:
                fill.fill_mid_yes = yes_mid
                fill.spread_cents = snapshot.spread * 100.0
            idx += 1
        fill_index[event.asset_id] = idx

        idx = future_index[event.asset_id]
        while idx < len(asset_fills):
            fill = asset_fills[idx]
            target_time_ms = fill.entry_time_ms + (HORIZON_SECONDS * 1000)
            if target_time_ms > event.timestamp_ms:
                break
            if fill.future_mid_yes is None:
                fill.future_mid_yes = yes_mid
            idx += 1
        future_index[event.asset_id] = idx

        if next_progress_event is not None and processed_events >= next_progress_event:
            print(
                "analysis_progress label={label} events={events} fill_snapshots_captured={fills_captured} markouts_captured={markouts_captured}".format(
                    label=normalized_progress_label,
                    events=processed_events,
                    fills_captured=sum(fill_index.values()),
                    markouts_captured=sum(future_index.values()),
                ),
                flush=True,
            )
            next_progress_event += progress_every_events

    if next_progress_event is not None:
        print(
            "analysis_complete label={label} events={events} fill_snapshots_captured={fills_captured} markouts_captured={markouts_captured}".format(
                label=normalized_progress_label,
                events=processed_events,
                fills_captured=sum(fill_index.values()),
                markouts_captured=sum(future_index.values()),
            ),
            flush=True,
        )


def _iter_relevant_asset_events(
    tick_root: Path,
    *,
    asset_ids: list[str],
    start_date: str,
    end_date: str,
):
    for date_dir in sorted(candidate for candidate in tick_root.iterdir() if candidate.is_dir()):
        if date_dir.name < start_date or date_dir.name > end_date:
            continue
        for asset_id in asset_ids:
            file_path = date_dir / f"{asset_id}.jsonl"
            if not file_path.is_file():
                continue
            yield from iter_file_events(file_path)


def _avg(values: Iterable[float]) -> float:
    collected = [value for value in values]
    return mean(collected) if collected else 0.0


def summarize(fills: list[FadeFillRecord]) -> dict[str, object]:
    quote_counts = Counter(fill.quote_side for fill in fills)
    spread_capture: list[float] = []
    full_spreads: list[float] = []
    signed_markouts: list[float] = []
    adverse_selection_losses: list[float] = []
    net_edges: list[float] = []
    pnl_cents: list[float] = []
    filled_sizes: list[float] = [fill.entry_size for fill in fills]
    wins = 0
    mean_reversion_wins = 0
    coverage = 0

    for fill in fills:
        immediate_edge = fill.edge_vs_yes_mid(fill.fill_mid_yes)
        future_edge = fill.edge_vs_yes_mid(fill.future_mid_yes)
        if immediate_edge is not None:
            spread_capture.append(immediate_edge)
        if fill.spread_cents is not None:
            full_spreads.append(fill.spread_cents)
        if immediate_edge is None or future_edge is None:
            continue

        coverage += 1
        signed_markout = future_edge - immediate_edge
        adverse_selection = max(0.0, immediate_edge - future_edge)
        net_edge = future_edge
        signed_markouts.append(signed_markout)
        adverse_selection_losses.append(adverse_selection)
        net_edges.append(net_edge)
        pnl_cents.append(net_edge * fill.entry_size)
        if net_edge > 0:
            wins += 1
        if signed_markout > 0:
            mean_reversion_wins += 1

    return {
        "total_fades": len(fills),
        "coverage_60s": coverage,
        "total_filled_shares": sum(filled_sizes),
        "average_fill_size": _avg(filled_sizes),
        "quote_counts": dict(quote_counts),
        "avg_full_spread_cents": _avg(full_spreads),
        "avg_spread_capture_cents": _avg(spread_capture),
        "median_spread_capture_cents": median(spread_capture) if spread_capture else 0.0,
        "avg_markout_60s_cents": _avg(signed_markouts),
        "median_markout_60s_cents": median(signed_markouts) if signed_markouts else 0.0,
        "mean_reversion_win_rate": (mean_reversion_wins / coverage) if coverage else 0.0,
        "avg_adverse_selection_60s_cents": _avg(adverse_selection_losses),
        "avg_net_edge_60s_cents": _avg(net_edges),
        "median_net_edge_60s_cents": median(net_edges) if net_edges else 0.0,
        "win_rate": (wins / coverage) if coverage else 0.0,
        "total_pnl_cents": sum(pnl_cents),
    }


def render_markdown(summary: dict[str, object], *, date_label: str, db_path: Path, input_dir: Path) -> str:
    quote_counts = summary["quote_counts"]
    avg_net_edge = float(summary["avg_net_edge_60s_cents"])
    total_pnl_usd = float(summary["total_pnl_cents"]) / 100.0
    total_fades = int(summary["total_fades"])
    if total_fades == 0:
        verdict = "no demonstrated edge"
    elif avg_net_edge > 0 and total_pnl_usd > 0:
        verdict = "positive net edge"
    else:
        verdict = "negative net edge"

    lines = [
        f"# ExhaustionFader Toxic-Day Scorecard ({date_label})",
        "",
        "## Run",
        "",
        "- Strategy: `src.signals.exhaustion_fader.ExhaustionFader`",
        f"- Replay source: `{input_dir}`",
        f"- Replay DB: `{db_path}`",
        f"- Markout horizon: `{HORIZON_SECONDS}s`",
        "",
        "## Method",
        "",
        "- Maker fills were loaded from the universal replay database.",
        "- Fill-time spread capture and 60-second forward markouts were reconstructed from the same raw tick stream used by the replay engine.",
        "- Signed 60-second markout is defined as `future edge - fill-time edge`; positive means the post-fill move mean-reverted in the fade's favor.",
        "- Total simulated PnL is the 60-second mark-to-mid edge times filled size; no additional exit-fee adjustment is applied in this scorecard.",
        "",
        "## Performance",
        "",
        f"- Total fades executed: `{total_fades:,}`",
        f"- Fades with full 60s horizon: `{int(summary['coverage_60s']):,}`",
        f"- Total filled shares: `{float(summary['total_filled_shares']):.1f}`",
        f"- Average fill size: `{float(summary['average_fill_size']):.4f}`",
        f"- Bid fills: `{int(quote_counts.get('BID', 0)):,}`",
        f"- Ask fills: `{int(quote_counts.get('ASK', 0)):,}`",
        f"- Win rate of fades: `{float(summary['win_rate']):.2%}`",
        f"- Mean-reversion success rate: `{float(summary['mean_reversion_win_rate']):.2%}`",
        f"- Average observed full spread at fill: `{float(summary['avg_full_spread_cents']):.5f}c/share`",
        f"- Average spread captured at fill: `{float(summary['avg_spread_capture_cents']):.5f}c/share`",
        f"- Median spread captured at fill: `{float(summary['median_spread_capture_cents']):.5f}c/share`",
        f"- Average 60s signed markout: `{float(summary['avg_markout_60s_cents']):.5f}c/share`",
        f"- Median 60s signed markout: `{float(summary['median_markout_60s_cents']):.5f}c/share`",
        f"- Average 60s adverse-selection loss: `{float(summary['avg_adverse_selection_60s_cents']):.5f}c/share`",
        f"- Average 60s net edge: `{avg_net_edge:.5f}c/share`",
        f"- Median 60s net edge: `{float(summary['median_net_edge_60s_cents']):.5f}c/share`",
        f"- Total simulated PnL: `${total_pnl_usd:.2f}`",
        "",
        "## Verdict",
        "",
        (
            "Flat-OBI retail-spike fading produced **{verdict}** on `{date}`: the 60-second mark-to-mid result was "
            "`{edge:.5f}c/share` on average with a `{win_rate:.2%}` win rate across `{fills:,}` completed fades."
        ).format(
            verdict=verdict,
            date=date_label,
            edge=avg_net_edge,
            win_rate=float(summary["win_rate"]),
            fills=int(summary["coverage_60s"]),
        ),
    ]
    if total_fades == 0:
        lines.append(
            "The replay produced zero dispatches and zero fills, so this partition does not provide evidence of a positive mean-reversion edge for the current trigger thresholds."
        )
    else:
        lines.append(
            (
                "The pure mean-reversion component contributed `{markout:.5f}c/share` on average, which is {assessment}."
            ).format(
                markout=float(summary["avg_markout_60s_cents"]),
                assessment="favorable" if float(summary["avg_markout_60s_cents"]) > 0 else "not favorable",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    input_dir = Path(args.input_dir)
    fills = load_fade_fills(db_path, start_date=args.start_date, end_date=args.end_date)
    reconstruct_markouts(
        fills,
        input_dir=input_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        progress_every_events=args.progress_every_events,
        progress_label=f"exhaustion-analysis:{args.start_date}",
    )
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