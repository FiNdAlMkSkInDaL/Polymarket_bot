#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

try:
    import psutil
except ImportError:  # pragma: no cover - fallback path only
    psutil = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.scavenger_protocol import (
    FLOAT_EPSILON,
    ScavengerConfig,
    ScavengerPortfolioState,
    build_scavenger_candidate_frame,
    collect_scavenger_price_distribution,
    load_scavenger_metadata_frame,
    simulate_scavenger_portfolio,
    summarize_scavenger_price_distribution,
)


DEFAULT_LAKE_ROOT = PROJECT_ROOT / "artifacts" / "l2_parquet_lake_full"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "clob_arb_baseline_metadata.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "scavenger_historical_sweep"

METADATA_SCHEMA: dict[str, pl.DataType] = {
    "market_id": pl.Utf8,
    "token_id": pl.Utf8,
    "metadata_event_id": pl.Utf8,
    "resolution_timestamp": pl.Datetime("us", "UTC"),
    "final_resolution_value": pl.Float64,
}

FILL_EXPORT_SCHEMA: dict[str, pl.DataType] = {
    "market_id": pl.Utf8,
    "token_id": pl.Utf8,
    "signal_event_id": pl.Utf8,
    "fill_event_id": pl.Utf8,
    "order_posted_at": pl.Datetime("us"),
    "fill_timestamp": pl.Datetime("us"),
    "resolution_timestamp": pl.Datetime("us"),
    "fill_reason": pl.Utf8,
    "order_price": pl.Float64,
    "signal_best_bid": pl.Float64,
    "signal_best_ask": pl.Float64,
    "fill_best_bid": pl.Float64,
    "fill_best_ask": pl.Float64,
    "ticket_notional_usdc": pl.Float64,
    "contracts_if_filled": pl.Float64,
    "final_resolution_value": pl.Float64,
    "raw_roi": pl.Float64,
    "actual_raw_pnl": pl.Float64,
    "actual_raw_roi": pl.Float64,
    "actual_pnl_usdc_if_filled": pl.Float64,
    "actual_apr": pl.Float64,
    "settled_yes": pl.Boolean,
    "catastrophic_loss": pl.Boolean,
    "tail_to_zero": pl.Boolean,
    "capital_before_signal_usdc": pl.Float64,
    "capital_after_signal_usdc": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class SweepReport:
    lake_root: str
    l2_book_root: str
    metadata_path: str
    output_root: str
    source_date_partitions: list[str]
    source_discovered_units: int
    source_completed_units: int
    source_hourly_parquet_files: int
    source_rows: int
    source_unique_markets: int
    source_unique_tokens: int
    peak_process_rss_mb: float
    measurement_method: str
    average_daily_peak_locked_capital_usdc: float
    average_daily_capital_utilization_pct: float
    max_daily_peak_locked_capital_usdc: float
    fills_csv: str
    price_distribution_csv: str
    markdown_tearsheet: str
    summary_json: str
    tearsheet_summary: dict[str, object]


class PeakMemoryMonitor:
    def __init__(self, sample_interval_seconds: float = 0.05) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def start(self) -> None:
        if psutil is None:
            return
        self._thread = threading.Thread(target=self._run, name="scavenger-historical-rss", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def peak_memory_mb(self) -> float | None:
        if psutil is None:
            return None
        return self._peak_rss_bytes / (1024.0 * 1024.0)

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop_event.is_set():
            self._peak_rss_bytes = max(self._peak_rss_bytes, process.memory_info().rss)
            time.sleep(self.sample_interval_seconds)
        self._peak_rss_bytes = max(self._peak_rss_bytes, process.memory_info().rss)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Scavenger Protocol historical sweep against the centralized l2_book parquet lake.",
    )
    parser.add_argument("--lake-root", type=Path, default=DEFAULT_LAKE_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-lookback-days", type=int, default=1)
    parser.add_argument("--window-lookahead-days", type=int, default=1)
    parser.add_argument("--resolution-window-hours", type=int, default=72)
    parser.add_argument("--signal-best-ask-min", type=float, default=0.99)
    parser.add_argument("--signal-best-bid-max", type=float, default=0.96)
    parser.add_argument("--maker-bid-price", type=float, default=0.95)
    parser.add_argument("--near-miss-price-tolerance", type=float, default=0.02)
    parser.add_argument("--starting-bankroll-usdc", type=float, default=5000.0)
    parser.add_argument("--max-notional-per-market-usdc", type=float, default=250.0)
    parser.add_argument("--price-distribution-chunk-size", type=int, default=64)
    return parser.parse_args()


def _parse_listish(raw_value: Any) -> list[Any]:
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str) and raw_value:
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_datetime(raw_value: Any) -> datetime | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _load_scavenger_metadata_frame(metadata_path: Path) -> pl.DataFrame:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    markets_by_token = payload.get("markets_by_token")
    if not isinstance(markets_by_token, dict):
        raise ValueError(f"Expected markets_by_token mapping in {metadata_path}")

    rows: list[dict[str, Any]] = []
    for row in markets_by_token.values():
        if not isinstance(row, dict):
            continue
        market_id = str(row.get("conditionId") or "").strip().lower()
        if not market_id:
            continue
        events = row.get("events")
        if not isinstance(events, list) or not events or not isinstance(events[0], dict):
            continue
        event = events[0]
        event_id = str(event.get("id") or row.get("eventId") or "").strip()
        resolution_timestamp = _parse_datetime(row.get("endDate") or event.get("endDate"))
        token_ids = _parse_listish(row.get("clobTokenIds"))
        outcome_prices = _parse_listish(row.get("outcomePrices"))
        if not event_id or resolution_timestamp is None or len(token_ids) != 2 or len(outcome_prices) != 2:
            continue
        for token_id, side, final_value in (
            (str(token_ids[0]).strip(), "YES", outcome_prices[0]),
            (str(token_ids[1]).strip(), "NO", outcome_prices[1]),
        ):
            if not token_id:
                continue
            try:
                parsed_final_value = float(final_value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "market_id": market_id,
                    "token_id": side,
                    "metadata_event_id": event_id,
                    "resolution_timestamp": resolution_timestamp,
                    "final_resolution_value": parsed_final_value,
                }
            )

    if not rows:
        raise ValueError(f"No scavenger metadata rows could be loaded from {metadata_path}")

    return pl.DataFrame(rows, schema=METADATA_SCHEMA).unique(subset=["market_id", "token_id"], keep="first")


def _discover_l2_book_partitions(lake_root: Path) -> tuple[Path, dict[date, list[Path]]]:
    l2_book_root = lake_root / "l2_book"
    if not l2_book_root.exists():
        raise FileNotFoundError(f"Centralized l2_book root not found under {lake_root}")

    partitions: dict[date, list[Path]] = {}
    for parquet_path in sorted(l2_book_root.glob("date=*/hour=*/*.parquet")):
        partition_name = parquet_path.parent.parent.name
        if not partition_name.startswith("date="):
            continue
        partition_day = date.fromisoformat(partition_name.removeprefix("date="))
        partitions.setdefault(partition_day, []).append(parquet_path)

    if not partitions:
        raise ValueError(f"No l2_book parquet partitions found under {l2_book_root}")

    return l2_book_root, partitions


def _load_lake_coverage_stats(lake_root: Path) -> tuple[int, int]:
    manifest_path = lake_root / "manifest.json"
    if not manifest_path.exists():
        return 0, 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = payload.get("stats") if isinstance(payload, dict) else None
    current_run = payload.get("current_run") if isinstance(payload, dict) else None
    discovered_units = 0
    completed_units = 0
    if isinstance(stats, dict):
        discovered_units = int(stats.get("markets_considered") or 0)
        completed_units = int(stats.get("markets_completed") or 0)
    if not discovered_units and isinstance(current_run, dict):
        discovered_units = int(current_run.get("discovered_units") or 0)
    if not completed_units and isinstance(current_run, dict):
        completed_units = int(current_run.get("processed_units") or 0)
    return discovered_units, completed_units


def _iter_days(start_day: date, end_day: date) -> list[date]:
    return [start_day + timedelta(days=offset) for offset in range((end_day - start_day).days + 1)]


def _scan_enriched_scavenger_frame(
    parquet_paths: list[Path],
    metadata_frame: pl.DataFrame,
) -> pl.LazyFrame:
    metadata_lf = metadata_frame.lazy()
    return (
        pl.scan_parquet([str(path) for path in parquet_paths], low_memory=True, cache=False)
        .select(
            pl.col("timestamp"),
            pl.col("market_id").cast(pl.Utf8).str.to_lowercase().alias("market_id"),
            pl.col("event_id").cast(pl.Utf8).alias("event_id"),
            pl.col("token_id").cast(pl.Utf8).alias("token_id"),
            pl.col("best_bid"),
            pl.col("best_ask"),
            pl.col("bid_depth"),
            pl.col("ask_depth"),
        )
        .join(metadata_lf, on=["market_id", "token_id"], how="inner")
        .with_columns(pl.coalesce([pl.col("event_id"), pl.col("metadata_event_id")]).alias("event_id"))
        .select(
            "timestamp",
            "market_id",
            "event_id",
            "token_id",
            "best_bid",
            "best_ask",
            "bid_depth",
            "ask_depth",
            "resolution_timestamp",
            "final_resolution_value",
        )
    )


def _safe_collect(frame: pl.LazyFrame) -> pl.DataFrame:
    try:
        return frame.collect(engine="streaming")
    except Exception:
        return frame.collect()


def _collect_source_frame_stats(parquet_paths: list[Path]) -> dict[str, int]:
    if not parquet_paths:
        return {
            "source_hourly_parquet_files": 0,
            "source_rows": 0,
            "source_unique_markets": 0,
            "source_unique_tokens": 0,
        }

    summary = _safe_collect(
        pl.scan_parquet([str(path) for path in parquet_paths], low_memory=True, cache=False)
        .select(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("unique_markets"),
            pl.col("token_id").n_unique().alias("unique_tokens"),
        )
    )
    row = summary.row(0, named=True)
    return {
        "source_hourly_parquet_files": len(parquet_paths),
        "source_rows": int(row["rows"] or 0),
        "source_unique_markets": int(row["unique_markets"] or 0),
        "source_unique_tokens": int(row["unique_tokens"] or 0),
    }


def _serialize_near_miss_daily_counts(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        rows.append(
            {
                "date": row["date"],
                "near_miss_count": int(row["near_misses"]),
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


def _build_fill_export(portfolio: pl.DataFrame) -> pl.DataFrame:
    if portfolio.is_empty():
        return pl.DataFrame(schema=FILL_EXPORT_SCHEMA)

    return (
        portfolio.filter(pl.col("accepted") & pl.col("filled"))
        .with_columns((pl.col("final_resolution_value") <= FLOAT_EPSILON).alias("tail_to_zero"))
        .select(list(FILL_EXPORT_SCHEMA))
        .sort(["fill_timestamp", "resolution_timestamp", "market_id", "token_id"])
        .with_columns(
            pl.col("order_price").cast(pl.Float64),
            pl.col("signal_best_bid").cast(pl.Float64),
            pl.col("signal_best_ask").cast(pl.Float64),
            pl.col("fill_best_bid").cast(pl.Float64),
            pl.col("fill_best_ask").cast(pl.Float64),
            pl.col("ticket_notional_usdc").cast(pl.Float64),
            pl.col("contracts_if_filled").cast(pl.Float64),
            pl.col("final_resolution_value").cast(pl.Float64),
            pl.col("raw_roi").cast(pl.Float64),
            pl.col("actual_raw_pnl").cast(pl.Float64),
            pl.col("actual_raw_roi").cast(pl.Float64),
            pl.col("actual_pnl_usdc_if_filled").cast(pl.Float64),
            pl.col("actual_apr").cast(pl.Float64),
            pl.col("capital_before_signal_usdc").cast(pl.Float64),
            pl.col("capital_after_signal_usdc").cast(pl.Float64),
        )
    )


def _render_markdown(
    tearsheet: pl.DataFrame,
    summary: dict[str, object],
    fills_df: pl.DataFrame,
    price_distribution_df: pl.DataFrame,
) -> str:
    distribution_summary = summary["price_distribution_summary"]
    winner_side_summary = distribution_summary["winner_side_summary"]
    loser_side_summary = distribution_summary["loser_side_summary"]
    lines = [
        "# Scavenger Protocol Price Distribution Diagnostic",
        "",
        f"- Lake root: {summary['lake_root']}",
        f"- L2 book root: {summary['l2_book_root']}",
        f"- Metadata path: {summary['metadata_path']}",
        f"- Source discovered units: {summary['source_discovered_units']}",
        f"- Source completed units: {summary['source_completed_units']}",
        f"- Source hourly parquet files: {summary['source_hourly_parquet_files']}",
        f"- Source date partitions: {', '.join(summary['source_date_partitions'])}",
        f"- Source rows: {summary['source_rows']}",
        f"- Source unique markets: {summary['source_unique_markets']}",
        f"- Source unique tokens: {summary['source_unique_tokens']}",
        f"- Distribution units analysed: {distribution_summary['unit_count']}",
        f"- Starting bankroll: {summary['starting_bankroll_usdc']:.2f} USDC",
        f"- Ending capital: {summary['ending_capital_usdc']:.2f} USDC",
        f"- Net return: {summary['net_return_pct'] * 100.0:.2f}%",
        f"- Filled trades: {summary['portfolio_fills']}",
        f"- Accepted orders: {summary['portfolio_orders_accepted']}",
        f"- Signals rejected for capital lockup: {summary['signals_rejected_capital_lockup']}",
        f"- Catastrophic losses: {summary['catastrophic_losses']}",
        f"- Tail-to-zero fills: {summary['tail_to_zero_fills']}",
        f"- Total actual PnL: {summary['portfolio_realized_pnl_usdc']:.2f} USDC",
        f"- Mean actual raw ROI: {summary['mean_actual_raw_roi_pct']:.2f}%",
        f"- Average Daily Capital Utilization: {summary['average_daily_capital_utilization_pct']:.2f}%",
        f"- Average Daily Peak Locked Capital: {summary['average_daily_peak_locked_capital_usdc']:.2f} USDC",
        f"- Max Daily Peak Locked Capital: {summary['max_daily_peak_locked_capital_usdc']:.2f} USDC",
        f"- Median deepest dip: {distribution_summary['median_deepest_dip']:.4f}" if distribution_summary['median_deepest_dip'] is not None else "- Median deepest dip: n/a",
        f"- Deepest dip interquartile range: {distribution_summary['p25_deepest_dip']:.4f} .. {distribution_summary['p75_deepest_dip']:.4f}" if distribution_summary['p25_deepest_dip'] is not None else "- Deepest dip interquartile range: n/a",
        f"- Modal deepest dip bucket: {distribution_summary['modal_deepest_dip_bucket']:.2f}" if distribution_summary['modal_deepest_dip_bucket'] is not None else "- Modal deepest dip bucket: n/a",
        f"- Median highest spike: {distribution_summary['median_highest_spike']:.4f}" if distribution_summary['median_highest_spike'] is not None else "- Median highest spike: n/a",
        f"- Winner-side median deepest dip: {winner_side_summary['median_deepest_dip']:.4f}" if winner_side_summary['median_deepest_dip'] is not None else "- Winner-side median deepest dip: n/a",
        f"- Winner-side deepest dip interquartile range: {winner_side_summary['p25_deepest_dip']:.4f} .. {winner_side_summary['p75_deepest_dip']:.4f}" if winner_side_summary['p25_deepest_dip'] is not None else "- Winner-side deepest dip interquartile range: n/a",
        f"- Loser-side median deepest dip: {loser_side_summary['median_deepest_dip']:.4f}" if loser_side_summary['median_deepest_dip'] is not None else "- Loser-side median deepest dip: n/a",
        f"- Current maker bid: {distribution_summary['current_bid_price']:.2f}",
        f"- Current maker bid touch rate: {distribution_summary['current_bid_touch_rate_pct']:.2f}% ({distribution_summary['current_bid_touch_count']}/{distribution_summary['unit_count']})",
        f"- Recommended realistic scavenge bid: {distribution_summary['recommended_realistic_scavenge_bid']:.2f}" if distribution_summary['recommended_realistic_scavenge_bid'] is not None else "- Recommended realistic scavenge bid: n/a",
        f"- Recommended bid touch rate: {distribution_summary['recommended_bid_touch_rate_pct']:.2f}% ({distribution_summary['recommended_bid_touch_count']}/{distribution_summary['unit_count']})",
        f"- Winner-side touch rate at current bid: {winner_side_summary['bid_touch_rate_pct']:.2f}% ({winner_side_summary['bid_touch_count']}/{winner_side_summary['unit_count']})" if winner_side_summary['unit_count'] else "- Winner-side touch rate at current bid: n/a",
        f"- Winner-side touch rate at recommended bid: {winner_side_summary['recommended_bid_touch_rate_pct']:.2f}% ({winner_side_summary['recommended_bid_touch_count']}/{winner_side_summary['unit_count']})" if winner_side_summary['unit_count'] else "- Winner-side touch rate at recommended bid: n/a",
        f"- Peak process RSS: {summary['peak_process_rss_mb']:.2f} MB ({summary['measurement_method']})",
        f"- Processed signal days: {summary['processed_signal_days']}",
        f"- Incomplete trailing signal days skipped: {summary['skipped_incomplete_signal_days']}",
        "",
        "| Date | Starting Capital | Ending Capital | Fills | Signals Rejected | Catastrophic Losses | Peak Locked Capital | Peak Utilization |",
        "|------|------------------|----------------|-------|------------------|---------------------|---------------------|------------------|",
    ]

    for row in tearsheet.iter_rows(named=True):
        lines.append(
            f"| {row['date']} | {row['starting_capital_usdc']:.2f} | {row['ending_capital_usdc']:.2f} | "
            f"{row['fills']} | {row['rejected_signals_capital_lockup']} | {row['catastrophic_losses']} | "
            f"{row['peak_locked_capital_usdc']:.2f} | {row['peak_capital_utilization_pct'] * 100.0:.2f}% |"
        )

    lines.extend(["", "## Threshold Relaxation Table", ""])
    touch_curve = distribution_summary["touch_curve"]
    if touch_curve:
        lines.extend(
            [
                "| Candidate Bid | Units Touched | Touch Rate |",
                "|---------------|---------------|------------|",
            ]
        )
        for row in touch_curve:
            lines.append(
                f"| {row['bid_level']:.2f} | {row['touched_units']} | {row['touch_rate_pct']:.2f}% |"
            )
    else:
        lines.append("- No price-distribution touch curve available.")

    lines.extend(["", "## Distribution Table", "", "- Market IDs are reported as `market_id:token_id` to disambiguate YES and NO sides.", "", "| Market ID | Final Result (0/1) | Deepest Dip (Lowest Ask observed) | Highest Spike (Highest Bid observed) |", "|-----------|---------------------|-----------------------------------|--------------------------------------|"])
    for row in price_distribution_df.iter_rows(named=True):
        lines.append(
            f"| {row['market_id']}:{row['token_id']} | {int(row['final_result'])} | {float(row['deepest_dip']):.4f} | {float(row['highest_spike']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Fills Export",
            "",
            f"- Output path: {summary['fills_csv']}",
            f"- Filled trade rows written: {fills_df.height}",
            f"- Price distribution CSV: {summary['price_distribution_csv']}",
        ]
    )
    return "\n".join(lines)


def run_historical_sweep(args: argparse.Namespace) -> SweepReport:
    config = ScavengerConfig(
        resolution_window_hours=args.resolution_window_hours,
        signal_best_ask_min=args.signal_best_ask_min,
        signal_best_bid_max=args.signal_best_bid_max,
        maker_bid_price=args.maker_bid_price,
        near_miss_price_tolerance=args.near_miss_price_tolerance,
        starting_bankroll_usdc=args.starting_bankroll_usdc,
        max_notional_per_market_usdc=args.max_notional_per_market_usdc,
    )
    l2_book_root, partition_files = _discover_l2_book_partitions(args.lake_root)
    metadata_frame = load_scavenger_metadata_frame(args.metadata)
    source_discovered_units, source_completed_units = _load_lake_coverage_stats(args.lake_root)
    available_days = sorted(partition_files)
    all_l2_book_files = [path for paths in partition_files.values() for path in paths]
    source_stats = _collect_source_frame_stats(all_l2_book_files)
    first_day = available_days[0]
    last_day = available_days[-1]
    evaluation_cutoff_day = min(last_day, available_days[-1] - timedelta(days=args.window_lookahead_days))
    dataset_end = datetime.combine(last_day + timedelta(days=1), dt_time.min)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    fills_csv_path = output_root / "fills.csv"
    price_distribution_csv_path = output_root / "price_distribution.csv"
    markdown_path = output_root / "historical_sweep_tearsheet.md"
    summary_json_path = output_root / "summary.json"

    portfolio_state = ScavengerPortfolioState.from_config(config)
    scheduled_fills: defaultdict[date, int] = defaultdict(int)
    daily_rows: list[dict[str, object]] = []
    portfolio_frames: list[pl.DataFrame] = []
    price_distribution_df = pl.DataFrame()

    monitor = PeakMemoryMonitor()
    tracemalloc.start()
    monitor.start()
    try:
        price_distribution_df = collect_scavenger_price_distribution(
            all_l2_book_files,
            config=config,
            metadata_frame=metadata_frame,
            chunk_size=args.price_distribution_chunk_size,
        )
        for day in _iter_days(first_day, last_day):
            day_start = datetime.combine(day, dt_time.min)
            day_end = day_start + timedelta(days=1)
            portfolio_state.release_until(day_start)
            starting_capital_usdc = float(portfolio_state.available_cash_usdc)
            rejected_signals_capital_lockup = 0

            window_start = max(first_day, day - timedelta(days=args.window_lookback_days))
            window_end = min(last_day, day + timedelta(days=args.window_lookahead_days))
            window_files = [
                path
                for partition_day, paths in partition_files.items()
                if window_start <= partition_day <= window_end
                for path in paths
            ]

            if day <= evaluation_cutoff_day:
                if window_files:
                    _, candidates_lf, _ = build_scavenger_candidate_frame(
                        window_files,
                        config=config,
                        lightweight=True,
                        metadata_frame=metadata_frame,
                    )
                    daily_candidates = _safe_collect(
                        candidates_lf.filter(
                            (pl.col("order_date") == pl.lit(day, dtype=pl.Date))
                            & (pl.col("resolution_timestamp") <= pl.lit(dataset_end, dtype=pl.Datetime("us")))
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
                daily_cash_points = [starting_capital_usdc]
                if daily_portfolio.height:
                    daily_cash_points.extend(
                        float(value)
                        for value in daily_portfolio["capital_before_signal_usdc"].drop_nulls().to_list()
                    )
                    daily_cash_points.extend(
                        float(value)
                        for value in daily_portfolio["capital_after_signal_usdc"].drop_nulls().to_list()
                    )
            else:
                daily_portfolio = pl.DataFrame()
                daily_cash_points = [starting_capital_usdc]

            released_today = portfolio_state.release_until(day_end)
            catastrophic_losses = sum(1 for release in released_today if release.catastrophic_loss)
            ending_capital_usdc = float(portfolio_state.available_cash_usdc)
            daily_cash_points.append(ending_capital_usdc)
            min_available_cash = min(daily_cash_points)
            peak_locked_capital_usdc = max(float(config.starting_bankroll_usdc) - min_available_cash, 0.0)
            peak_capital_utilization_pct = peak_locked_capital_usdc / float(config.starting_bankroll_usdc)

            daily_rows.append(
                {
                    "date": day.isoformat(),
                    "starting_capital_usdc": round(starting_capital_usdc, 6),
                    "ending_capital_usdc": round(ending_capital_usdc, 6),
                    "fills": scheduled_fills.pop(day, 0),
                    "rejected_signals_capital_lockup": rejected_signals_capital_lockup,
                    "catastrophic_losses": catastrophic_losses,
                    "peak_locked_capital_usdc": round(peak_locked_capital_usdc, 6),
                    "peak_capital_utilization_pct": round(peak_capital_utilization_pct, 9),
                }
            )

        tracemalloc_peak_mb = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
    finally:
        monitor.stop()
        tracemalloc.stop()

    rss_peak_mb = monitor.peak_memory_mb()
    if rss_peak_mb is not None:
        peak_process_rss_mb = rss_peak_mb
        measurement_method = "psutil_rss"
    else:
        peak_process_rss_mb = tracemalloc_peak_mb
        measurement_method = "tracemalloc_peak"

    tearsheet = pl.DataFrame(daily_rows).sort("date")
    portfolio = pl.concat(portfolio_frames, how="diagonal") if portfolio_frames else pl.DataFrame()
    price_distribution_df.write_csv(price_distribution_csv_path)
    fills_df = _build_fill_export(portfolio)
    fills_df.write_csv(fills_csv_path)

    average_daily_peak_locked_capital_usdc = (
        float(tearsheet["peak_locked_capital_usdc"].mean()) if tearsheet.height else 0.0
    )
    average_daily_capital_utilization_pct = (
        float(tearsheet["peak_capital_utilization_pct"].mean()) * 100.0 if tearsheet.height else 0.0
    )
    max_daily_peak_locked_capital_usdc = (
        float(tearsheet["peak_locked_capital_usdc"].max()) if tearsheet.height else 0.0
    )
    mean_actual_raw_roi_pct = (
        float(fills_df["actual_raw_roi"].mean()) * 100.0 if fills_df.height else 0.0
    )
    ending_capital_usdc = (
        float(tearsheet["ending_capital_usdc"][-1]) if tearsheet.height else float(config.starting_bankroll_usdc)
    )
    price_distribution_summary = summarize_scavenger_price_distribution(
        price_distribution_df,
        current_bid_price=float(config.maker_bid_price),
    )

    summary = {
        "lake_root": str(args.lake_root),
        "l2_book_root": str(l2_book_root),
        "metadata_path": str(args.metadata),
        "source_date_partitions": [day.isoformat() for day in available_days],
        "source_discovered_units": source_discovered_units,
        "source_completed_units": source_completed_units,
        **source_stats,
        "starting_bankroll_usdc": float(config.starting_bankroll_usdc),
        "ending_capital_usdc": ending_capital_usdc,
        "net_return_pct": (ending_capital_usdc - float(config.starting_bankroll_usdc))
        / float(config.starting_bankroll_usdc),
        "portfolio_orders_accepted": portfolio.filter(pl.col("accepted")).height if portfolio.height else 0,
        "portfolio_fills": fills_df.height,
        "signals_rejected_capital_lockup": portfolio.filter(pl.col("rejection_reason") == "capital_lockup").height
        if portfolio.height
        else 0,
        "catastrophic_losses": fills_df.filter(pl.col("catastrophic_loss")).height if fills_df.height else 0,
        "tail_to_zero_fills": fills_df.filter(pl.col("tail_to_zero")).height if fills_df.height else 0,
        "portfolio_realized_pnl_usdc": float(fills_df["actual_pnl_usdc_if_filled"].sum()) if fills_df.height else 0.0,
        "mean_actual_raw_roi_pct": mean_actual_raw_roi_pct,
        "average_daily_peak_locked_capital_usdc": average_daily_peak_locked_capital_usdc,
        "average_daily_capital_utilization_pct": average_daily_capital_utilization_pct,
        "max_daily_peak_locked_capital_usdc": max_daily_peak_locked_capital_usdc,
        "peak_process_rss_mb": float(peak_process_rss_mb),
        "measurement_method": measurement_method,
        "price_distribution_chunk_size": int(args.price_distribution_chunk_size),
        "processed_signal_days": max((evaluation_cutoff_day - first_day).days + 1, 0) if evaluation_cutoff_day >= first_day else 0,
        "skipped_incomplete_signal_days": max((last_day - evaluation_cutoff_day).days, 0) if evaluation_cutoff_day < last_day else 0,
        "price_distribution_summary": price_distribution_summary,
        "fills_csv": str(fills_csv_path),
        "price_distribution_csv": str(price_distribution_csv_path),
        "markdown_tearsheet": str(markdown_path),
        "summary_json": str(summary_json_path),
    }

    markdown_path.write_text(
        _render_markdown(tearsheet, summary, fills_df, price_distribution_df),
        encoding="utf-8",
    )
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return SweepReport(
        lake_root=str(args.lake_root),
        l2_book_root=str(l2_book_root),
        metadata_path=str(args.metadata),
        output_root=str(output_root),
        source_date_partitions=[day.isoformat() for day in available_days],
        source_discovered_units=source_discovered_units,
        source_completed_units=source_completed_units,
        source_hourly_parquet_files=source_stats["source_hourly_parquet_files"],
        source_rows=source_stats["source_rows"],
        source_unique_markets=source_stats["source_unique_markets"],
        source_unique_tokens=source_stats["source_unique_tokens"],
        peak_process_rss_mb=float(peak_process_rss_mb),
        measurement_method=measurement_method,
        average_daily_peak_locked_capital_usdc=average_daily_peak_locked_capital_usdc,
        average_daily_capital_utilization_pct=average_daily_capital_utilization_pct,
        max_daily_peak_locked_capital_usdc=max_daily_peak_locked_capital_usdc,
        fills_csv=str(fills_csv_path),
        price_distribution_csv=str(price_distribution_csv_path),
        markdown_tearsheet=str(markdown_path),
        summary_json=str(summary_json_path),
        tearsheet_summary=summary,
    )


def main() -> None:
    report = run_historical_sweep(_parse_args())
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()