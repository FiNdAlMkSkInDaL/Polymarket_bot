from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_universal_backtest import iter_replay_events, resolve_tick_root
from src.data.orderbook import OrderbookTracker


SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(slots=True)
class DayStats:
    trade_count: int = 0
    volume_usd: float = 0.0
    spread_time_weighted_sum: float = 0.0
    spread_time_seconds: float = 0.0


@dataclass(slots=True)
class MarketSummary:
    market_id: str
    avg_daily_trade_count: float
    avg_daily_volume_usd: float
    avg_time_weighted_spread_cents: float
    active_days: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan historical L2 archives for long-tail markets with wide spreads and sufficient flow.")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data")
    parser.add_argument("--start-date", default="2026-03-15")
    parser.add_argument("--end-date", default="2026-03-19")
    parser.add_argument("--output", default="docs/long_tail_universe_report.md")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-daily-volume-usd", type=float, default=50_000.0)
    parser.add_argument("--min-daily-trade-count", type=float, default=50.0)
    parser.add_argument("--min-time-weighted-spread-cents", type=float, default=3.0)
    parser.add_argument("--require-spread-filter", action="store_true")
    parser.add_argument("--report-title", default="Long Tail Universe Report")
    return parser.parse_args()


def date_from_timestamp_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).date().isoformat()


def iter_days(start_date: str, end_date: str) -> list[str]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    total_days = (end - start).days + 1
    return [(start.fromordinal(start.toordinal() + offset)).isoformat() for offset in range(total_days)]


def finalize_span(
    day_stats: dict[str, dict[str, DayStats]],
    trackers: dict[tuple[str, str, str], OrderbookTracker],
    last_timestamp_ms: dict[tuple[str, str, str], int],
    final_timestamp_ms: int,
) -> None:
    for key, tracker in trackers.items():
        day, market_id, _asset_id = key
        start_ms = last_timestamp_ms.get(key, 0)
        if start_ms <= 0 or final_timestamp_ms <= start_ms:
            continue
        snapshot = tracker.snapshot()
        if snapshot.best_bid <= 0 or snapshot.best_ask <= 0:
            continue
        duration_seconds = (final_timestamp_ms - start_ms) / 1000.0
        if duration_seconds <= 0:
            continue
        stats = day_stats[day][market_id]
        stats.spread_time_weighted_sum += snapshot.spread * 100.0 * duration_seconds
        stats.spread_time_seconds += duration_seconds


def scan_markets(input_dir: Path, *, start_date: str, end_date: str) -> list[MarketSummary]:
    days = iter_days(start_date, end_date)
    day_count = len(days)
    day_stats: dict[str, dict[str, DayStats]] = {day: defaultdict(DayStats) for day in days}
    trackers: dict[tuple[str, str, str], OrderbookTracker] = {}
    last_timestamp_ms: dict[tuple[str, str, str], int] = {}
    last_day: str | None = None
    last_event_timestamp_ms = 0

    for event in iter_replay_events(resolve_tick_root(input_dir), start_date=start_date, end_date=end_date):
        day = date_from_timestamp_ms(event.timestamp_ms)
        if day not in day_stats:
            continue

        if last_day is not None and day != last_day:
            day_end = int(datetime.fromisoformat(last_day).replace(tzinfo=timezone.utc).timestamp() * 1000) + SECONDS_PER_DAY * 1000
            day_trackers = {key: tracker for key, tracker in trackers.items() if key[0] == last_day}
            day_last_timestamps = {key: value for key, value in last_timestamp_ms.items() if key[0] == last_day}
            finalize_span(day_stats, day_trackers, day_last_timestamps, day_end)

        key = (day, event.market_id, event.asset_id)
        tracker = trackers.get(key)
        if tracker is None:
            tracker = OrderbookTracker(event.asset_id)
            trackers[key] = tracker

        previous_timestamp_ms = last_timestamp_ms.get(key)
        if previous_timestamp_ms is not None:
            snapshot = tracker.snapshot()
            if snapshot.best_bid > 0 and snapshot.best_ask > 0 and event.timestamp_ms > previous_timestamp_ms:
                duration_seconds = (event.timestamp_ms - previous_timestamp_ms) / 1000.0
                stats = day_stats[day][event.market_id]
                stats.spread_time_weighted_sum += snapshot.spread * 100.0 * duration_seconds
                stats.spread_time_seconds += duration_seconds

        if event.event_type == "BOOK":
            tracker.on_book_snapshot(event.payload)
        elif event.event_type == "PRICE_CHANGE":
            tracker.on_price_change(event.payload)
        elif event.event_type == "TRADE":
            stats = day_stats[day][event.market_id]
            stats.trade_count += 1
            stats.volume_usd += float(event.trade_price or 0) * float(event.trade_size or 0)

        last_timestamp_ms[key] = event.timestamp_ms
        last_day = day
        last_event_timestamp_ms = max(last_event_timestamp_ms, event.timestamp_ms)

    if last_day is not None:
        day_end = int(datetime.fromisoformat(last_day).replace(tzinfo=timezone.utc).timestamp() * 1000) + SECONDS_PER_DAY * 1000
        day_trackers = {key: tracker for key, tracker in trackers.items() if key[0] == last_day}
        day_last_timestamps = {key: value for key, value in last_timestamp_ms.items() if key[0] == last_day}
        finalize_span(day_stats, day_trackers, day_last_timestamps, day_end)

    market_ids = sorted({market_id for stats_by_market in day_stats.values() for market_id in stats_by_market})
    summaries: list[MarketSummary] = []
    for market_id in market_ids:
        trade_count_total = 0
        volume_total = 0.0
        spread_daily_values: list[float] = []
        active_days = 0
        for day in days:
            stats = day_stats[day].get(market_id)
            if stats is None:
                spread_daily_values.append(0.0)
                continue
            trade_count_total += stats.trade_count
            volume_total += stats.volume_usd
            if stats.spread_time_seconds > 0:
                spread_daily_values.append(stats.spread_time_weighted_sum / stats.spread_time_seconds)
                active_days += 1
            else:
                spread_daily_values.append(0.0)

        summaries.append(
            MarketSummary(
                market_id=market_id,
                avg_daily_trade_count=trade_count_total / day_count,
                avg_daily_volume_usd=volume_total / day_count,
                avg_time_weighted_spread_cents=sum(spread_daily_values) / day_count,
                active_days=active_days,
            )
        )
    return summaries


def apply_goldilocks_filter(
    summaries: Iterable[MarketSummary],
    *,
    max_daily_volume_usd: float,
    min_daily_trade_count: float,
    min_time_weighted_spread_cents: float,
    require_spread_filter: bool,
) -> list[MarketSummary]:
    filtered = [
        summary
        for summary in summaries
        if summary.avg_daily_volume_usd < max_daily_volume_usd
        and summary.avg_daily_trade_count > min_daily_trade_count
        and (
            not require_spread_filter
            or summary.avg_time_weighted_spread_cents > min_time_weighted_spread_cents
        )
    ]
    return sorted(filtered, key=lambda item: (item.avg_time_weighted_spread_cents, item.avg_daily_trade_count), reverse=True)


def render_markdown(
    shortlisted: list[MarketSummary],
    *,
    start_date: str,
    end_date: str,
    total_markets: int,
    limit: int,
    report_title: str,
    max_daily_volume_usd: float,
    min_daily_trade_count: float,
    min_time_weighted_spread_cents: float,
    require_spread_filter: bool,
) -> str:
    lines = [
        f"# {report_title} ({start_date} to {end_date})",
        "",
        "## Screen",
        "",
        f"- Average Daily Volume < `${max_daily_volume_usd:,.0f}`",
        f"- Average Daily Trade Count > `{min_daily_trade_count:g}`",
        (
            f"- Average Time-Weighted Spread > `{min_time_weighted_spread_cents:g}` cents"
            if require_spread_filter
            else "- Average Time-Weighted Spread is not hard-filtered; qualifying markets are ranked by widest spread."
        ),
        f"- Unique markets scanned: `{total_markets:,}`",
        f"- Markets passing filter: `{len(shortlisted):,}`",
        "",
        f"## Top {limit} Markets",
        "",
        "| Rank | Market ID | Avg Daily Trade Count | Avg Daily Volume (USD) | Avg Time-Weighted Spread (cents) | Active Days |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    if shortlisted:
        for index, summary in enumerate(shortlisted[:limit], start=1):
            lines.append(
                "| {rank} | {market_id} | {trades:.1f} | {volume:,.2f} | {spread:.3f} | {active_days} |".format(
                    rank=index,
                    market_id=summary.market_id,
                    trades=summary.avg_daily_trade_count,
                    volume=summary.avg_daily_volume_usd,
                    spread=summary.avg_time_weighted_spread_cents,
                    active_days=summary.active_days,
                )
            )
    else:
        lines.append("| 1 | none | 0.0 | 0.00 | 0.000 | 0 |")

    lines.extend(["", "## Verdict", ""])
    if shortlisted:
        widest = shortlisted[0]
        if require_spread_filter:
            lines.append(
                "A long-tail universe does exist in this sample. The widest qualifying market was `{market_id}` with `{spread:.3f}` cents average time-weighted spread, `{trades:.1f}` trades/day, and `${volume:,.2f}` daily volume.".format(
                    market_id=widest.market_id,
                    spread=widest.avg_time_weighted_spread_cents,
                    trades=widest.avg_daily_trade_count,
                    volume=widest.avg_daily_volume_usd,
                )
            )
        else:
            lines.append(
                "This relaxed frontier exposes the real long-tail trade-off: `{market_id}` sits at the widest end of the viable sub-`${cap:,.0f}` / `>{trades:g}` trades-day universe with `{spread:.3f}` cents average spread, `{trade_count:.1f}` trades/day, and `${volume:,.2f}` daily volume.".format(
                    market_id=widest.market_id,
                    cap=max_daily_volume_usd,
                    trades=min_daily_trade_count,
                    spread=widest.avg_time_weighted_spread_cents,
                    trade_count=widest.avg_daily_trade_count,
                    volume=widest.avg_daily_volume_usd,
                )
            )
    else:
        if require_spread_filter:
            lines.append("No market met all configured constraints across the five-day baseline, so this long-tail ecosystem was not confirmed by the archive sample.")
        else:
            lines.append("No market met the relaxed volume and trade-count constraints across the five-day baseline, so even the looser long-tail frontier was absent in this archive sample.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    summaries = scan_markets(input_dir, start_date=args.start_date, end_date=args.end_date)
    shortlisted = apply_goldilocks_filter(
        summaries,
        max_daily_volume_usd=args.max_daily_volume_usd,
        min_daily_trade_count=args.min_daily_trade_count,
        min_time_weighted_spread_cents=args.min_time_weighted_spread_cents,
        require_spread_filter=args.require_spread_filter,
    )
    markdown = render_markdown(
        shortlisted,
        start_date=args.start_date,
        end_date=args.end_date,
        total_markets=len(summaries),
        limit=args.limit,
        report_title=args.report_title,
        max_daily_volume_usd=args.max_daily_volume_usd,
        min_daily_trade_count=args.min_daily_trade_count,
        min_time_weighted_spread_cents=args.min_time_weighted_spread_cents,
        require_spread_filter=args.require_spread_filter,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "markets_scanned": len(summaries),
                "markets_passing": len(shortlisted),
                "max_daily_volume_usd": args.max_daily_volume_usd,
                "min_daily_trade_count": args.min_daily_trade_count,
                "min_time_weighted_spread_cents": args.min_time_weighted_spread_cents,
                "require_spread_filter": args.require_spread_filter,
                "top_markets": [
                    {
                        "market_id": summary.market_id,
                        "avg_daily_trade_count": summary.avg_daily_trade_count,
                        "avg_daily_volume_usd": summary.avg_daily_volume_usd,
                        "avg_time_weighted_spread_cents": summary.avg_time_weighted_spread_cents,
                        "active_days": summary.active_days,
                    }
                    for summary in shortlisted[: args.limit]
                ],
            },
            indent=2,
        )
    )
    print("\n---MARKDOWN---\n")
    print(markdown)


if __name__ == "__main__":
    main()