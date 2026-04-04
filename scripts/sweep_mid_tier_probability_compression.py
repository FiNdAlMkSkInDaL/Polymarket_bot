#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import sys
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.backtest_mid_tier_probability_compression import (
    MidTierCompressionConfig,
    PreparedBacktest,
    materialize_prepared_backtest,
    prepare_backtest_grid,
)


DATE_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HIVE_DATE_DIR_PATTERN = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
DEFAULT_DAILY_OUTPUT = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_sweep_daily.parquet"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_sweep.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_sweep.md"
DAILY_ROW_KEYS = (
    "quote_side",
    "quote_schema",
    "no_bid_formula",
    "no_ask_formula",
    "maker_fill_reference",
    "maker_fill_partition_key",
    "concurrent_name_cap_mode",
    "concurrent_name_cap_release_key",
    "concurrent_name_cap_release_fallback_key",
    "unresolved_exit_mode",
    "top2_yes_threshold",
    "midtier_yes_threshold",
    "max_leg_notional_usd",
    "max_concurrent_names",
    "signal_leg_rows",
    "signal_snapshots",
    "candidate_orders",
    "accepted_orders",
    "rejected_orders",
    "filled_orders",
    "settlement_exit_fills",
    "simulated_exit_candidate_orders",
    "simulated_exit_accepted_orders",
    "simulated_exit_fills",
    "winning_fills",
    "losing_fills",
    "flat_fills",
    "fill_win_rate",
    "filled_entry_spread_sum",
    "avg_filled_entry_spread",
    "reserved_notional_usd",
    "deployed_notional_usd",
    "total_realized_pnl_usd",
    "single_fill_events",
    "favorite_collapse_hits",
    "total_legging_loss_usd",
    "max_legging_drawdown_usd",
    "worst_single_fill_event_pnl_usd",
)


@dataclass(slots=True)
class SweepArtifacts:
    summary: dict[str, Any]
    daily_rows: pl.DataFrame
    combo_summary: pl.DataFrame
    daily_leaders: pl.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a day-by-day parameter sweep for Mid-Tier Probability Compression.",
    )
    parser.add_argument("--input-root", required=True, help="Date-partitioned Parquet root with YYYY-MM-DD subdirectories.")
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD start date filter.")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD end date filter.")
    parser.add_argument("--top2-threshold-start", type=float, default=0.85)
    parser.add_argument("--top2-threshold-end", type=float, default=0.98)
    parser.add_argument("--top2-threshold-step", type=float, default=0.01)
    parser.add_argument("--notionals", default="10,25,50", help="Comma-separated max leg notionals in USD.")
    parser.add_argument("--midtier-yes-threshold", type=float, default=0.15)
    parser.add_argument("--max-concurrent-names", type=int, default=100)
    parser.add_argument("--quote-side", choices=("yes", "no"), default="yes")
    parser.add_argument("--timestamp-unit", choices=("ns", "us", "ms", "s"), default="ms")
    parser.add_argument(
        "--resolution-timestamp-unit",
        choices=("ns", "us", "ms", "s"),
        default="ms",
    )
    parser.add_argument("--daily-output", type=Path, default=DEFAULT_DAILY_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def build_threshold_grid(start: float, end: float, step: float) -> tuple[float, ...]:
    if step <= 0:
        raise ValueError("top2 threshold step must be positive")
    if end < start:
        raise ValueError("top2 threshold end must be greater than or equal to start")
    count = int(round((end - start) / step)) + 1
    return tuple(round(start + index * step, 4) for index in range(count))


def parse_notionals(value: str) -> tuple[float, ...]:
    notionals = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not notionals:
        raise ValueError("At least one notional must be provided")
    return notionals


def _partition_trade_date(path: Path) -> str | None:
    if DATE_DIR_PATTERN.match(path.name):
        return path.name
    match = HIVE_DATE_DIR_PATTERN.match(path.name)
    if match is not None:
        return match.group(1)
    return None


def _iter_candidate_partition_roots(root: Path) -> tuple[Path, ...]:
    candidates = [root]
    lake_child = root / "l2_book"
    if lake_child.is_dir():
        candidates.append(lake_child)
    return tuple(candidates)


def discover_daily_partitions(
    input_root: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[tuple[str, Path]]:
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not root.is_dir():
        raise ValueError("Sweep harness expects a directory root, not a single file")

    partitions_map: dict[str, Path] = {}
    for candidate_root in _iter_candidate_partition_roots(root):
        trade_date = _partition_trade_date(candidate_root)
        if trade_date is not None and any(candidate_root.rglob("*.parquet")):
            partitions_map.setdefault(trade_date, candidate_root)
            continue

        for child in candidate_root.iterdir():
            if not child.is_dir():
                continue
            trade_date = _partition_trade_date(child)
            if trade_date is None or not any(child.rglob("*.parquet")):
                continue
            partitions_map.setdefault(trade_date, child)

    partitions = sorted(partitions_map.items(), key=lambda item: item[0])

    if not partitions:
        raise ValueError(
            "No date partitions with parquet files were found. Expected either YYYY-MM-DD/*.parquet or l2_book/date=YYYY-MM-DD/hour=*/part-*.parquet"
        )

    filtered: list[tuple[str, Path]] = []
    for trade_date, partition_path in partitions:
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        filtered.append((trade_date, partition_path))

    if not filtered:
        raise ValueError("Date filters removed every partition from the sweep")
    return filtered


def _daily_row(*, trade_date: str, partition_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "trade_date": trade_date,
        "partition_path": str(partition_path),
    }
    for key in DAILY_ROW_KEYS:
        row[key] = summary[key]
    return row


def _build_combo_summary(daily_rows: pl.DataFrame) -> pl.DataFrame:
    if daily_rows.is_empty():
        return pl.DataFrame()
    return (
        daily_rows.group_by(["top2_yes_threshold", "max_leg_notional_usd"])
        .agg(
            pl.len().alias("days"),
            pl.when(pl.col("total_realized_pnl_usd") > 0.0).then(1).otherwise(0).sum().alias("positive_days"),
            pl.col("signal_snapshots").sum().alias("signal_snapshots"),
            pl.col("filled_orders").sum().alias("filled_orders"),
            pl.col("settlement_exit_fills").sum().alias("settlement_exit_fills"),
            pl.col("simulated_exit_fills").sum().alias("simulated_exit_fills"),
            pl.col("winning_fills").sum().alias("winning_fills"),
            pl.col("filled_entry_spread_sum").sum().alias("filled_entry_spread_sum"),
            pl.col("favorite_collapse_hits").sum().alias("favorite_collapse_hits"),
            pl.col("total_legging_loss_usd").sum().alias("total_legging_loss_usd"),
            pl.col("total_realized_pnl_usd").sum().alias("total_realized_pnl_usd"),
            pl.col("total_realized_pnl_usd").mean().alias("avg_daily_pnl_usd"),
            pl.col("max_legging_drawdown_usd").max().alias("worst_daily_legging_drawdown_usd"),
        )
        .with_columns(
            pl.when(pl.col("filled_orders") > 0)
            .then(pl.col("winning_fills") / pl.col("filled_orders"))
            .otherwise(0.0)
            .alias("fill_win_rate"),
            pl.when(pl.col("filled_orders") > 0)
            .then(pl.col("filled_entry_spread_sum") / pl.col("filled_orders"))
            .otherwise(0.0)
            .alias("avg_filled_entry_spread"),
        )
        .sort(["total_realized_pnl_usd", "top2_yes_threshold", "max_leg_notional_usd"], descending=[True, False, False])
    )


def _build_daily_leaders(daily_rows: pl.DataFrame) -> pl.DataFrame:
    if daily_rows.is_empty():
        return pl.DataFrame()
    return (
        daily_rows.sort(
            ["trade_date", "total_realized_pnl_usd", "max_legging_drawdown_usd", "top2_yes_threshold", "max_leg_notional_usd"],
            descending=[False, True, False, False, False],
        )
        .group_by("trade_date", maintain_order=True)
        .first()
        .select(
            "trade_date",
            "top2_yes_threshold",
            "max_leg_notional_usd",
            "filled_orders",
            "simulated_exit_fills",
            "avg_filled_entry_spread",
            "favorite_collapse_hits",
            "total_realized_pnl_usd",
            "max_legging_drawdown_usd",
        )
    )


def _write_table(frame: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        frame.write_csv(output_path)
        return
    if suffix in {".parquet", ".pq"}:
        frame.write_parquet(output_path, compression="zstd")
        return
    raise ValueError(f"Unsupported output suffix for {output_path}: expected .csv or .parquet")


def render_markdown(
    artifacts: SweepArtifacts,
    *,
    input_root: Path,
    daily_output: Path,
) -> str:
    lines = [
        "# Mid-Tier Probability Compression Sweep",
        "",
        "## Setup",
        "",
        f"- Input root: `{input_root}`",
        f"- Days processed: `{artifacts.summary['days_processed']}`",
        f"- Threshold grid: `{artifacts.summary['thresholds']}`",
        f"- Notional grid: `{artifacts.summary['notionals']}`",
        "- Baseline schema: YES best bid / ask inverted into the NO book via `no_bid = 1 - yes_ask` and `no_ask = 1 - yes_bid`.",
        "- Concurrent 100-name control is rolling: admission capacity is released on `resolution_timestamp`.",
        "",
        "## Aggregate Combo Summary",
        "",
        "| Top-2 Threshold | Max Leg Notional | Days | Positive Days | Filled Orders | Simulated Exit Fills | Avg Filled Entry Spread | Collapse Hits | Total Legging Loss (USD) | Worst Daily Legging DD (USD) | Total PnL (USD) | Avg Daily PnL (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in artifacts.combo_summary.to_dicts():
        lines.append(
            "| {threshold:.2f} | {notional:.2f} | {days} | {positive_days} | {filled_orders} | {simulated_exit_fills} | {avg_spread:.4f} | {collapse_hits} | {legging_loss:.2f} | {drawdown:.2f} | {pnl:.2f} | {avg_pnl:.2f} |".format(
                threshold=float(row["top2_yes_threshold"]),
                notional=float(row["max_leg_notional_usd"]),
                days=int(row["days"]),
                positive_days=int(row["positive_days"]),
                filled_orders=int(row["filled_orders"]),
                simulated_exit_fills=int(row["simulated_exit_fills"]),
                avg_spread=float(row["avg_filled_entry_spread"]),
                collapse_hits=int(row["favorite_collapse_hits"]),
                legging_loss=float(row["total_legging_loss_usd"]),
                drawdown=float(row["worst_daily_legging_drawdown_usd"]),
                pnl=float(row["total_realized_pnl_usd"]),
                avg_pnl=float(row["avg_daily_pnl_usd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Daily Leaders",
            "",
            "| Date | Best Threshold | Best Notional | Filled Orders | Simulated Exit Fills | Avg Filled Entry Spread | Collapse Hits | Daily PnL (USD) | Daily Legging DD (USD) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in artifacts.daily_leaders.to_dicts():
        lines.append(
            "| {trade_date} | {threshold:.2f} | {notional:.2f} | {filled_orders} | {simulated_exit_fills} | {avg_spread:.4f} | {collapse_hits} | {pnl:.2f} | {drawdown:.2f} |".format(
                trade_date=str(row["trade_date"]),
                threshold=float(row["top2_yes_threshold"]),
                notional=float(row["max_leg_notional_usd"]),
                filled_orders=int(row["filled_orders"]),
                simulated_exit_fills=int(row["simulated_exit_fills"]),
                avg_spread=float(row["avg_filled_entry_spread"]),
                collapse_hits=int(row["favorite_collapse_hits"]),
                pnl=float(row["total_realized_pnl_usd"]),
                drawdown=float(row["max_legging_drawdown_usd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Detail",
            "",
            f"- Full day-by-day panel written to `{daily_output}`.",
            "- The markdown stays compact on purpose; use the detailed panel for per-day parameter inspection.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_sweep(
    input_root: str | Path,
    *,
    thresholds: tuple[float, ...],
    notionals: tuple[float, ...],
    start_date: str | None = None,
    end_date: str | None = None,
    quote_side: str = "yes",
    timestamp_unit: str = "ms",
    resolution_timestamp_unit: str = "ms",
    midtier_yes_threshold: float = 0.15,
    max_concurrent_names: int = 100,
) -> SweepArtifacts:
    partitions = discover_daily_partitions(input_root, start_date=start_date, end_date=end_date)
    rows: list[dict[str, Any]] = []

    for trade_date, partition_path in partitions:
        base_config = MidTierCompressionConfig(
            quote_side=quote_side,
            timestamp_unit=timestamp_unit,
            resolution_timestamp_unit=resolution_timestamp_unit,
            top2_yes_threshold=min(thresholds),
            midtier_yes_threshold=midtier_yes_threshold,
            max_leg_notional_usd=1.0,
            max_concurrent_names=max_concurrent_names,
        )
        prepared_grid = prepare_backtest_grid(partition_path, thresholds, base_config)
        for threshold in thresholds:
            prepared: PreparedBacktest = prepared_grid[float(threshold)]
            threshold_config = replace(base_config, top2_yes_threshold=float(threshold))
            for notional in notionals:
                artifacts = materialize_prepared_backtest(
                    prepared,
                    replace(threshold_config, max_leg_notional_usd=float(notional)),
                )
                rows.append(_daily_row(trade_date=trade_date, partition_path=partition_path, summary=artifacts.summary))

    daily_rows = pl.DataFrame(rows).sort(["trade_date", "top2_yes_threshold", "max_leg_notional_usd"])
    combo_summary = _build_combo_summary(daily_rows)
    daily_leaders = _build_daily_leaders(daily_rows)
    summary = {
        "input_root": str(Path(input_root)),
        "days_processed": len(partitions),
        "thresholds": [float(value) for value in thresholds],
        "notionals": [float(value) for value in notionals],
        "quote_side": quote_side,
        "midtier_yes_threshold": midtier_yes_threshold,
        "max_concurrent_names": max_concurrent_names,
        "combo_summary": combo_summary.to_dicts(),
        "daily_leaders": daily_leaders.to_dicts(),
    }
    return SweepArtifacts(summary=summary, daily_rows=daily_rows, combo_summary=combo_summary, daily_leaders=daily_leaders)


def main() -> int:
    args = parse_args()
    thresholds = build_threshold_grid(args.top2_threshold_start, args.top2_threshold_end, args.top2_threshold_step)
    notionals = parse_notionals(args.notionals)
    artifacts = run_sweep(
        args.input_root,
        thresholds=thresholds,
        notionals=notionals,
        start_date=args.start_date,
        end_date=args.end_date,
        quote_side=args.quote_side,
        timestamp_unit=args.timestamp_unit,
        resolution_timestamp_unit=args.resolution_timestamp_unit,
        midtier_yes_threshold=args.midtier_yes_threshold,
        max_concurrent_names=args.max_concurrent_names,
    )

    _write_table(artifacts.daily_rows, args.daily_output)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(artifacts.summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_markdown(artifacts, input_root=Path(args.input_root), daily_output=args.daily_output)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8")

    print(json.dumps(artifacts.summary, indent=2, sort_keys=True))
    print("\n---MARKDOWN---\n")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())