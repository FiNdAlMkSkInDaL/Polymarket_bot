#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.sweep_mid_tier_probability_compression import DEFAULT_DAILY_OUTPUT


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_sweep_reduced.parquet"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_sweep_reduced.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_sweep_reduced.md"
REQUIRED_COLUMNS = {
    "trade_date",
    "top2_yes_threshold",
    "max_leg_notional_usd",
    "filled_orders",
    "total_realized_pnl_usd",
    "max_legging_drawdown_usd",
}


@dataclass(slots=True)
class ReducerArtifacts:
    summary: dict[str, Any]
    rankings: pl.DataFrame
    pareto_frontier: pl.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce the Mid-Tier Probability Compression day-by-day sweep panel into ranked parameter combinations.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DAILY_OUTPUT, help="Daily sweep panel (.parquet or .csv).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ranked reducer table output (.parquet or .csv).")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT, help="Reducer summary JSON output path.")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT, help="Reducer markdown output path.")
    parser.add_argument("--top", type=int, default=50, help="Maximum rows to render into markdown and stdout.")
    return parser.parse_args()


def _scan_panel(input_path: str | Path) -> pl.LazyFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(str(path), glob=True)
    if suffix == ".csv":
        return pl.scan_csv(str(path), try_parse_dates=True)
    raise ValueError(f"Unsupported panel suffix for {input_path}: expected .parquet or .csv")


def _validate_panel_schema(schema: pl.Schema) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(schema.names()))
    if missing:
        raise ValueError(f"Sweep panel is missing required columns: {', '.join(missing)}")
    if "winning_fills" not in schema.names() and "fill_win_rate" not in schema.names():
        raise ValueError(
            "Sweep panel must include winning_fills or fill_win_rate. Re-run the sweep with the current harness if needed."
        )


def _winning_fill_sum_expr(schema: pl.Schema) -> pl.Expr:
    if "winning_fills" in schema.names():
        return pl.col("winning_fills").sum().alias("winning_fills")
    return (pl.col("fill_win_rate") * pl.col("filled_orders")).sum().round(0).alias("winning_fills")


def _spread_sum_expr(schema: pl.Schema) -> pl.Expr:
    if "filled_entry_spread_sum" in schema.names():
        return pl.col("filled_entry_spread_sum").sum().alias("filled_entry_spread_sum")
    if "avg_filled_entry_spread" in schema.names():
        return (pl.col("avg_filled_entry_spread") * pl.col("filled_orders")).sum().alias("filled_entry_spread_sum")
    return pl.lit(0.0).alias("filled_entry_spread_sum")


def _simulated_exit_fill_sum_expr(schema: pl.Schema) -> pl.Expr:
    if "simulated_exit_fills" in schema.names():
        return pl.col("simulated_exit_fills").sum().alias("simulated_exit_fills")
    return pl.lit(0).alias("simulated_exit_fills")


def _flag_pareto_frontier(rankings: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    if rankings.is_empty():
        flagged = rankings.with_columns(pl.lit(False).alias("pareto_frontier"))
        return flagged, flagged

    frontier_ordered = (
        rankings.sort(
            [
                "absolute_worst_case_daily_legging_drawdown_usd",
                "aggregate_net_pnl_usd",
                "top2_yes_threshold",
                "max_leg_notional_usd",
            ],
            descending=[False, True, False, False],
        )
        .with_columns(
            pl.col("aggregate_net_pnl_usd")
            .max()
            .over("absolute_worst_case_daily_legging_drawdown_usd")
            .alias("_drawdown_bucket_max_pnl")
        )
        .with_columns(pl.col("_drawdown_bucket_max_pnl").cum_max().alias("_running_frontier_pnl"))
        .with_columns(
            (
                (pl.col("aggregate_net_pnl_usd") == pl.col("_drawdown_bucket_max_pnl"))
                & (pl.col("aggregate_net_pnl_usd") == pl.col("_running_frontier_pnl"))
            ).alias("pareto_frontier")
        )
    )

    frontier = (
        frontier_ordered.filter(pl.col("pareto_frontier"))
        .with_row_index("pareto_frontier_rank", offset=1)
        .select(
            "pareto_frontier_rank",
            "top2_yes_threshold",
            "max_leg_notional_usd",
            "aggregate_net_pnl_usd",
            "total_fills",
            "simulated_exit_fills",
            "winning_fills",
            "win_rate",
            "avg_filled_entry_spread",
            "absolute_worst_case_daily_legging_drawdown_usd",
            "risk_adjusted_score",
            "risk_adjusted_score_display",
        )
    )

    flagged = rankings.join(
        frontier.select("top2_yes_threshold", "max_leg_notional_usd", pl.lit(True).alias("pareto_frontier")),
        on=["top2_yes_threshold", "max_leg_notional_usd"],
        how="left",
    ).with_columns(pl.col("pareto_frontier").fill_null(False))
    return flagged, frontier


def _build_rankings(panel: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    schema = panel.collect_schema()
    _validate_panel_schema(schema)

    aggregated = (
        panel.group_by(["top2_yes_threshold", "max_leg_notional_usd"])
        .agg(
            pl.len().alias("days"),
            pl.col("trade_date").n_unique().alias("distinct_days"),
            pl.col("filled_orders").sum().alias("total_fills"),
            _winning_fill_sum_expr(schema),
            _simulated_exit_fill_sum_expr(schema),
            _spread_sum_expr(schema),
            pl.col("total_realized_pnl_usd").sum().alias("aggregate_net_pnl_usd"),
            pl.col("max_legging_drawdown_usd").abs().max().alias("absolute_worst_case_daily_legging_drawdown_usd"),
        )
        .with_columns(
            pl.when(pl.col("total_fills") > 0)
            .then(pl.col("winning_fills") / pl.col("total_fills"))
            .otherwise(0.0)
            .alias("win_rate"),
            pl.when(pl.col("total_fills") > 0)
            .then(pl.col("filled_entry_spread_sum") / pl.col("total_fills"))
            .otherwise(0.0)
            .alias("avg_filled_entry_spread"),
        )
        .with_columns(
            pl.when(pl.col("absolute_worst_case_daily_legging_drawdown_usd") > 0.0)
            .then(pl.col("aggregate_net_pnl_usd") / pl.col("absolute_worst_case_daily_legging_drawdown_usd"))
            .otherwise(None)
            .alias("risk_adjusted_score"),
            pl.when(pl.col("absolute_worst_case_daily_legging_drawdown_usd") > 0.0)
            .then(pl.col("aggregate_net_pnl_usd") / pl.col("absolute_worst_case_daily_legging_drawdown_usd"))
            .when(pl.col("aggregate_net_pnl_usd") > 0.0)
            .then(pl.lit(1.0e308))
            .when(pl.col("aggregate_net_pnl_usd") < 0.0)
            .then(pl.lit(-1.0e308))
            .otherwise(0.0)
            .alias("_risk_adjusted_sort_score"),
            pl.when(pl.col("absolute_worst_case_daily_legging_drawdown_usd") > 0.0)
            .then((pl.col("aggregate_net_pnl_usd") / pl.col("absolute_worst_case_daily_legging_drawdown_usd")).round(6).cast(pl.String))
            .when(pl.col("aggregate_net_pnl_usd") > 0.0)
            .then(pl.lit("inf"))
            .when(pl.col("aggregate_net_pnl_usd") < 0.0)
            .then(pl.lit("-inf"))
            .otherwise(pl.lit("0.0"))
            .alias("risk_adjusted_score_display"),
        )
        .collect()
    )

    flagged, frontier = _flag_pareto_frontier(aggregated)
    rankings = flagged.sort(
        ["_risk_adjusted_sort_score", "aggregate_net_pnl_usd", "top2_yes_threshold", "max_leg_notional_usd"],
        descending=[True, True, False, False],
    )
    rankings = rankings.select(
        "top2_yes_threshold",
        "max_leg_notional_usd",
        "days",
        "distinct_days",
        "aggregate_net_pnl_usd",
        "total_fills",
        "simulated_exit_fills",
        "winning_fills",
        "win_rate",
        "avg_filled_entry_spread",
        "absolute_worst_case_daily_legging_drawdown_usd",
        "risk_adjusted_score",
        "risk_adjusted_score_display",
        "pareto_frontier",
    )
    return rankings, frontier


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


def render_pareto_frontier_markdown(frontier: pl.DataFrame, *, title: str = "Pareto Frontier") -> str:
    lines = [
        f"## {title}",
        "",
        "| Frontier Rank | Top-2 Threshold | Max Leg Notional | Aggregate Net PnL (USD) | Total Fills | Simulated Exit Fills | Avg Filled Entry Spread | Win Rate | Abs Worst Daily Legging DD (USD) | Risk-Adjusted Score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if frontier.is_empty():
        lines.append("| 1 | n/a | n/a | 0.00 | 0 | 0 | 0.0000 | 0.00% | 0.00 | 0.0 |")
        return "\n".join(lines) + "\n"

    for row in frontier.to_dicts():
        lines.append(
            "| {rank} | {threshold:.2f} | {notional:.2f} | {pnl:.2f} | {fills} | {simulated_exit_fills} | {avg_spread:.4f} | {win_rate:.2%} | {drawdown:.2f} | {score} |".format(
                rank=int(row["pareto_frontier_rank"]),
                threshold=float(row["top2_yes_threshold"]),
                notional=float(row["max_leg_notional_usd"]),
                pnl=float(row["aggregate_net_pnl_usd"]),
                fills=int(row["total_fills"]),
                simulated_exit_fills=int(row["simulated_exit_fills"]),
                avg_spread=float(row["avg_filled_entry_spread"]),
                win_rate=float(row["win_rate"]),
                drawdown=float(row["absolute_worst_case_daily_legging_drawdown_usd"]),
                score=str(row["risk_adjusted_score_display"]),
            )
        )
    return "\n".join(lines) + "\n"


def render_markdown(artifacts: ReducerArtifacts, *, input_path: Path, top_n: int) -> str:
    top_rows = artifacts.rankings.head(max(1, top_n))
    lines = [
        "# Mid-Tier Probability Compression Reducer",
        "",
        "## Source",
        "",
        f"- Daily panel: `{input_path}`",
        f"- Parameter combinations ranked: `{artifacts.summary['combination_count']}`",
        f"- Pareto frontier rows: `{artifacts.summary['pareto_frontier_count']}`",
        "- Risk-adjusted ranking sorts by Aggregate Net PnL / ABS(Worst Daily Legging Drawdown).",
        "- Zero-drawdown handling: positive PnL ranks as `inf`, negative PnL ranks as `-inf`, and flat PnL ranks as `0.0`.",
        "",
    ]
    lines.append(render_pareto_frontier_markdown(artifacts.pareto_frontier).rstrip())
    lines.extend(
        [
            "",
        "## Ranked Table",
        "",
            "| Rank | Pareto | Top-2 Threshold | Max Leg Notional | Aggregate Net PnL (USD) | Total Fills | Simulated Exit Fills | Avg Filled Entry Spread | Win Rate | Abs Worst Daily Legging DD (USD) | Risk-Adjusted Score |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for index, row in enumerate(top_rows.to_dicts(), start=1):
        lines.append(
            "| {rank} | {pareto} | {threshold:.2f} | {notional:.2f} | {pnl:.2f} | {fills} | {simulated_exit_fills} | {avg_spread:.4f} | {win_rate:.2%} | {drawdown:.2f} | {score} |".format(
                rank=index,
                pareto="yes" if bool(row["pareto_frontier"]) else "no",
                threshold=float(row["top2_yes_threshold"]),
                notional=float(row["max_leg_notional_usd"]),
                pnl=float(row["aggregate_net_pnl_usd"]),
                fills=int(row["total_fills"]),
                simulated_exit_fills=int(row["simulated_exit_fills"]),
                avg_spread=float(row["avg_filled_entry_spread"]),
                win_rate=float(row["win_rate"]),
                drawdown=float(row["absolute_worst_case_daily_legging_drawdown_usd"]),
                score=str(row["risk_adjusted_score_display"]),
            )
        )
    return "\n".join(lines) + "\n"


def run_reducer(input_path: str | Path) -> ReducerArtifacts:
    rankings, frontier = _build_rankings(_scan_panel(input_path))
    summary = {
        "input_path": str(Path(input_path)),
        "combination_count": int(rankings.height),
        "pareto_frontier_count": int(frontier.height),
        "best_combination": rankings.row(0, named=True) if rankings.height else None,
        "pareto_frontier": frontier.to_dicts(),
    }
    return ReducerArtifacts(summary=summary, rankings=rankings, pareto_frontier=frontier)


def main() -> int:
    args = parse_args()
    artifacts = run_reducer(args.input)
    _write_table(artifacts.rankings, args.output)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(artifacts.summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown = render_markdown(artifacts, input_path=args.input, top_n=args.top)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8")

    print(json.dumps(artifacts.summary, indent=2, sort_keys=True, default=str))
    print("\n---MARKDOWN---\n")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())