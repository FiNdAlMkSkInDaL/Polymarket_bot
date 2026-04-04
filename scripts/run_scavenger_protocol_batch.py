#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.scavenger_protocol import (
    ScavengerConfig,
    ScavengerPortfolioState,
    build_scavenger_diagnostic_frames,
    simulate_scavenger_portfolio,
    summarize_near_misses,
)


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ScavengerBatchResult:
    tearsheet: pl.DataFrame
    near_misses: pl.DataFrame
    near_miss_top_markets: pl.DataFrame
    portfolio: pl.DataFrame
    summary: dict[str, object]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-safe day-window batch runner for the Scavenger Protocol parquet lake.",
    )
    parser.add_argument("input_root", type=Path, help="Root parquet-lake directory with YYYY-MM-DD partitions.")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional CSV tearsheet output path.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional Markdown tearsheet output path.",
    )
    parser.add_argument(
        "--portfolio-output",
        type=Path,
        default=None,
        help="Optional CSV or Parquet output path for the capital-aware trade ledger.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive YYYY-MM-DD start date.")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive YYYY-MM-DD end date.")
    parser.add_argument("--window-lookback-days", type=int, default=3)
    parser.add_argument("--window-lookahead-days", type=int, default=3)
    parser.add_argument("--resolution-window-hours", type=int, default=72)
    parser.add_argument("--signal-best-ask-min", type=float, default=0.99)
    parser.add_argument("--signal-best-bid-max", type=float, default=0.96)
    parser.add_argument("--maker-bid-price", type=float, default=0.95)
    parser.add_argument("--starting-bankroll-usdc", type=float, default=5000.0)
    parser.add_argument("--max-notional-per-market-usdc", type=float, default=250.0)
    return parser.parse_args()


def _parse_date(raw_value: str | None) -> date | None:
    if raw_value is None:
        return None
    return date.fromisoformat(raw_value)


def _discover_partition_files(input_root: Path) -> dict[date, list[Path]]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    partitions: dict[date, list[Path]] = {}
    for parquet_path in sorted(input_root.rglob("*.parquet")):
        partition_date: date | None = None
        for parent in (parquet_path.parent, *parquet_path.parents):
            if parent == input_root.parent:
                break
            if DATE_DIR_RE.match(parent.name):
                partition_date = date.fromisoformat(parent.name)
                break
        if partition_date is None:
            continue
        partitions.setdefault(partition_date, []).append(parquet_path)

    if not partitions:
        raise ValueError(f"No date-partitioned parquet files found under {input_root}")
    return partitions


def _iter_days(start_day: date, end_day: date) -> list[date]:
    total_days = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(total_days + 1)]


def _empty_portfolio_like() -> pl.DataFrame:
    return pl.DataFrame()


def _write_frame(frame: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        frame.write_csv(output_path)
        return
    if suffix == ".parquet":
        frame.write_parquet(output_path)
        return
    raise ValueError(f"Unsupported output format: {output_path}")


def _serialize_near_miss_daily_counts(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        rows.append(
            {
                "date": row["date"],
                "near_miss_count": int(row["near_miss_count"]),
                "near_miss_market_count": int(row["near_miss_market_count"]),
            }
        )
    return rows


def _serialize_near_miss_top_markets(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        rows.append(
            {
                "market_id": row["market_id"],
                "near_miss_count": int(row["near_miss_count"]),
                "token_ids": list(row["token_ids"] or []),
                "closest_near_miss_price_gap": float(row["closest_near_miss_price_gap"]),
                "first_near_miss_date": row["first_near_miss_date"].isoformat(),
                "last_near_miss_date": row["last_near_miss_date"].isoformat(),
            }
        )
    return rows


def render_markdown(tearsheet: pl.DataFrame, summary: dict[str, object], input_root: Path) -> str:
    lines = [
        "# Scavenger Protocol Daily Tearsheet",
        "",
        f"- Input root: {input_root}",
        f"- Starting bankroll: {summary['starting_bankroll_usdc']:.2f} USDC",
        f"- Ending capital: {summary['ending_capital_usdc']:.2f} USDC",
        f"- Processed signal days: {summary['processed_signal_days']}",
        f"- Incomplete trailing signal days skipped: {summary['skipped_incomplete_signal_days']}",
        f"- Near-miss tolerance: {summary['near_miss_price_tolerance_pct']:.2f}%",
        f"- Total near-misses: {summary['near_misses']}",
        "",
        "| Date | Starting Capital | Ending Capital | Number of Fills | Number of Signals Rejected | Catastrophic Losses | Near-Misses |",
        "|------|------------------|----------------|-----------------|----------------------------|---------------------|-------------|",
    ]
    for row in tearsheet.iter_rows(named=True):
        lines.append(
            f"| {row['date']} | {row['starting_capital_usdc']:.2f} | {row['ending_capital_usdc']:.2f} | "
            f"{row['fills']} | {row['rejected_signals_capital_lockup']} | {row['catastrophic_losses']} | {row['near_misses']} |"
        )

    lines.extend(["", "## Near-Miss Diagnostics", ""])
    near_miss_top_markets = summary.get("near_miss_top_markets", [])
    if near_miss_top_markets:
        lines.extend(
            [
                "| Market ID | Near-Misses | Tokens | Closest Price Gap | First Seen | Last Seen |",
                "|-----------|-------------|--------|-------------------|------------|-----------|",
            ]
        )
        for row in near_miss_top_markets:
            token_ids = ", ".join(row["token_ids"])
            lines.append(
                f"| {row['market_id']} | {row['near_miss_count']} | {token_ids} | "
                f"{row['closest_near_miss_price_gap'] * 100.0:.2f}% | {row['first_near_miss_date']} | {row['last_near_miss_date']} |"
            )
    else:
        lines.append("- No near-misses observed.")
    return "\n".join(lines)


def run_scavenger_batch(
    input_root: Path,
    config: ScavengerConfig,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    window_lookback_days: int = 3,
    window_lookahead_days: int = 3,
) -> ScavengerBatchResult:
    partition_files = _discover_partition_files(input_root)
    available_days = sorted(partition_files)
    first_day = start_date or available_days[0]
    last_day = end_date or available_days[-1]
    evaluation_cutoff_day = min(last_day, available_days[-1] - timedelta(days=window_lookahead_days))

    portfolio_state = ScavengerPortfolioState.from_config(config)
    scheduled_fills: defaultdict[date, int] = defaultdict(int)
    daily_rows: list[dict[str, object]] = []
    portfolio_frames: list[pl.DataFrame] = []
    near_miss_frames: list[pl.DataFrame] = []

    for day in _iter_days(first_day, last_day):
        day_start = datetime.combine(day, time.min)
        day_end = day_start + timedelta(days=1)
        portfolio_state.release_until(day_start)
        starting_capital_usdc = float(portfolio_state.available_cash_usdc)
        rejected_signals_capital_lockup = 0
        near_miss_count = 0
        near_miss_market_count = 0

        window_start = day - timedelta(days=window_lookback_days)
        window_end = day + timedelta(days=window_lookahead_days)
        window_files = [
            path
            for partition_day, paths in partition_files.items()
            if window_start <= partition_day <= window_end
            for path in paths
        ]
        if window_files:
            _, candidates_lf, _, near_misses_lf = build_scavenger_diagnostic_frames(
                window_files,
                config=config,
                lightweight=True,
            )
            daily_near_misses = near_misses_lf.filter(
                pl.col("near_miss_date") == pl.lit(day, dtype=pl.Date)
            ).collect(engine="streaming")
            near_miss_count = daily_near_misses.height
            near_miss_market_count = (
                int(daily_near_misses.select(pl.col("market_id").n_unique()).item())
                if daily_near_misses.height
                else 0
            )
            if daily_near_misses.height:
                near_miss_frames.append(daily_near_misses)
        else:
            daily_near_misses = pl.DataFrame()

        if day <= evaluation_cutoff_day:
            if window_files:
                daily_candidates = (
                    candidates_lf.filter(pl.col("order_date") == pl.lit(day, dtype=pl.Date)).collect(
                        engine="streaming"
                    )
                )
            else:
                daily_candidates = pl.DataFrame()

            daily_portfolio, portfolio_state = simulate_scavenger_portfolio(
                daily_candidates,
                config=config,
                state=portfolio_state,
                finalize=False,
            )
            if daily_portfolio.height:
                portfolio_frames.append(daily_portfolio)
                rejected_signals_capital_lockup = daily_portfolio.filter(
                    pl.col("rejection_reason") == "capital_lockup"
                ).height
                for row in daily_portfolio.filter(pl.col("accepted") & pl.col("filled")).iter_rows(named=True):
                    fill_date = row["fill_date"]
                    if fill_date is not None:
                        scheduled_fills[fill_date] += 1

        released_today = portfolio_state.release_until(day_end)
        catastrophic_losses = sum(1 for release in released_today if release.catastrophic_loss)
        daily_rows.append(
            {
                "date": day.isoformat(),
                "starting_capital_usdc": round(starting_capital_usdc, 6),
                "ending_capital_usdc": round(float(portfolio_state.available_cash_usdc), 6),
                "fills": scheduled_fills.pop(day, 0),
                "rejected_signals_capital_lockup": rejected_signals_capital_lockup,
                "catastrophic_losses": catastrophic_losses,
                "near_misses": near_miss_count,
                "near_miss_market_count": near_miss_market_count,
            }
        )

    tearsheet = pl.DataFrame(daily_rows)
    near_misses = pl.concat(near_miss_frames, how="diagonal") if near_miss_frames else pl.DataFrame()
    _, near_miss_top_markets = summarize_near_misses(near_misses, top_n=10)
    portfolio = pl.concat(portfolio_frames, how="diagonal") if portfolio_frames else _empty_portfolio_like()
    summary = {
        "starting_bankroll_usdc": float(config.starting_bankroll_usdc),
        "ending_capital_usdc": float(tearsheet["ending_capital_usdc"][-1]) if tearsheet.height else float(config.starting_bankroll_usdc),
        "processed_signal_days": max((evaluation_cutoff_day - first_day).days + 1, 0) if evaluation_cutoff_day >= first_day else 0,
        "skipped_incomplete_signal_days": max((last_day - evaluation_cutoff_day).days, 0) if evaluation_cutoff_day < last_day else 0,
        "near_miss_price_tolerance_pct": float(config.near_miss_price_tolerance) * 100.0,
        "near_misses": near_misses.height,
        "near_miss_daily_counts": _serialize_near_miss_daily_counts(
            tearsheet.select(
                "date",
                pl.col("near_misses").alias("near_miss_count"),
                "near_miss_market_count",
            )
        ),
        "near_miss_top_markets": _serialize_near_miss_top_markets(near_miss_top_markets),
    }
    return ScavengerBatchResult(
        tearsheet=tearsheet,
        near_misses=near_misses,
        near_miss_top_markets=near_miss_top_markets,
        portfolio=portfolio,
        summary=summary,
    )


def main() -> None:
    args = _parse_args()
    config = ScavengerConfig(
        resolution_window_hours=args.resolution_window_hours,
        signal_best_ask_min=args.signal_best_ask_min,
        signal_best_bid_max=args.signal_best_bid_max,
        maker_bid_price=args.maker_bid_price,
        starting_bankroll_usdc=args.starting_bankroll_usdc,
        max_notional_per_market_usdc=args.max_notional_per_market_usdc,
    )
    result = run_scavenger_batch(
        args.input_root,
        config,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        window_lookback_days=args.window_lookback_days,
        window_lookahead_days=args.window_lookahead_days,
    )

    print(result.tearsheet)
    if args.csv_output is not None:
        _write_frame(result.tearsheet, args.csv_output)
    if args.portfolio_output is not None:
        _write_frame(result.portfolio, args.portfolio_output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(result.tearsheet, result.summary, args.input_root),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()