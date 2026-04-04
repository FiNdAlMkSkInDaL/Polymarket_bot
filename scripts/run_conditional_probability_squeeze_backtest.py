from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.conditional_probability_squeeze import (
    ConditionalProbabilitySqueezeConfig,
    MarketSlice,
    default_minimum_theoretical_edge_dollars,
    run_conditional_probability_squeeze_backtest,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Conditional Probability Squeeze Polars backtest over a Parquet lake.",
    )
    parser.add_argument("source", nargs="+", help="Parquet file(s) or glob-expanded paths.")
    parser.add_argument("--market-a-id", required=True, help="Market ID for the long-buy leg.")
    parser.add_argument("--market-b-id", required=True, help="Market ID for the short-sell leg.")
    parser.add_argument("--market-a-token", help="Optional token ID for market A.")
    parser.add_argument("--market-b-token", help="Optional token ID for market B.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "squeeze_diagnostics",
        help="Directory for summary, report, and diagnostic parquet outputs.",
    )
    parser.add_argument("--order-size", type=float, default=100.0)
    parser.add_argument("--entry-gap-threshold", type=float, default=0.03)
    parser.add_argument("--entry-zscore-threshold", type=float, default=2.0)
    parser.add_argument(
        "--minimum-edge-over-combined-spread-ratio",
        type=float,
        default=0.03,
        help="Require bid_b - ask_a to exceed combined spread by this ratio before firing a signal.",
    )
    parser.add_argument(
        "--minimum-theoretical-edge-dollars",
        type=float,
        default=None,
        help=(
            "Require signal-time theoretical edge dollars ((bid_b - ask_a) * order_size) to exceed this floor. "
            "Defaults to 2%% of order size, equivalent to $10 on a 500-contract basket."
        ),
    )
    parser.add_argument("--exit-gap-threshold", type=float, default=0.05)
    parser.add_argument("--exit-zscore-threshold", type=float, default=0.0)
    parser.add_argument("--z-window-events", type=int, default=250)
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="Explicit source timestamp unit. Use auto to infer from magnitude.",
    )
    parser.add_argument(
        "--route-latency-ms",
        type=int,
        default=100,
        help="Decision-to-arrival latency in milliseconds. Defaults to 100ms (100000us).",
    )
    parser.add_argument("--max-quote-age-ms", type=int, default=5_000)
    parser.add_argument("--max-hold-ms", type=int, default=60_000)
    parser.add_argument("--taker-fee-bps", type=float, default=0.0)
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument("--warmup-days", type=int, default=1)
    parser.add_argument(
        "--chunk-lookahead-ms",
        type=int,
        help="Optional explicit lookahead window for time stops and flatten exits.",
    )
    parser.add_argument(
        "--collect-engine",
        choices=["auto", "streaming"],
        default="streaming",
        help="Polars collect engine to use inside each chunk.",
    )
    parser.add_argument(
        "--no-process-by-day",
        action="store_true",
        help="Disable day chunking and run as one full lazy query.",
    )
    return parser.parse_args(argv)


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")


def _write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "n/a"
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def _build_report(
    summary: dict[str, object],
    *,
    source: Sequence[str],
    market_a: MarketSlice,
    market_b: MarketSlice,
    output_dir: Path,
) -> str:
    latency_ms = int(summary["route_latency_ms"])
    latency_us = int(summary["route_latency_us"])
    lines = [
        "# Conditional Probability Squeeze Diagnostics",
        "",
        "## Scope",
        f"- Sources: {', '.join(source)}",
        f"- Market A: {market_a.market_id}" + (f" token={market_a.token_id}" if market_a.token_id else ""),
        f"- Market B: {market_b.market_id}" + (f" token={market_b.token_id}" if market_b.token_id else ""),
        f"- Output directory: {output_dir}",
        "",
        "## Execution Model",
        f"- Route-arrival latency: {latency_ms}ms ({latency_us}us)",
        f"- Minimum edge over combined spread: {float(summary.get('minimum_edge_over_combined_spread_ratio', 0.0)):.2%}",
        f"- Minimum theoretical edge dollars: {float(summary.get('minimum_theoretical_edge_dollars', 0.0)):.2f}",
        f"- Timestamp unit mode: {summary['timestamp_unit']}",
        f"- Process by day: {summary['process_by_day']}",
        f"- Chunk days: {summary['chunk_days']}",
        f"- Warmup days: {summary['warmup_days']}",
        f"- Chunk lookahead: {summary['chunk_lookahead_ms']}ms",
        f"- Chunks processed: {summary['chunks_processed']}",
        "",
        f"## FOK Survival Rate at {latency_ms}ms",
        f"- Total valid signals generated: {summary['total_valid_signals_generated']}",
        f"- Decision-time FOK passes: {summary['decision_time_fok_passes']}",
        f"- Route-arrival full rejections: {summary['route_arrival_full_rejections']}",
        f"- Partial fills requiring flatten: {summary['partial_fills_requiring_flatten']}",
        f"- Successful FOK baskets: {summary['successful_fok_baskets']}",
        f"- FOK survival rate at {latency_ms}ms: {float(summary['fok_survival_rate_at_route_latency']):.2%}",
        "",
        "## PnL Split",
        f"- Net PnL: {float(summary['net_pnl']):.6f}",
        f"- Successful FOK basket net PnL: {float(summary['successful_fok_net_pnl']):.6f}",
        f"- Flattened basket net PnL: {float(summary['flattened_basket_net_pnl']):.6f}",
        f"- Flattened basket net loss: {float(summary['flattened_basket_net_loss']):.6f}",
        "",
        "## Output Files",
        "- summary.json",
        "- report.md",
        "- signals.parquet",
        "- trades.parquet",
        "- successful_fok_baskets.parquet",
        "- route_arrival_rejections.parquet",
        "- flatten_baskets.parquet",
        "- chunk_stats.parquet",
    ]
    return "\n".join(lines) + "\n"


def build_config_from_args(args: argparse.Namespace) -> ConditionalProbabilitySqueezeConfig:
    minimum_theoretical_edge_dollars = (
        float(args.minimum_theoretical_edge_dollars)
        if args.minimum_theoretical_edge_dollars is not None
        else default_minimum_theoretical_edge_dollars(args.order_size)
    )

    return ConditionalProbabilitySqueezeConfig(
        order_size=args.order_size,
        entry_gap_threshold=args.entry_gap_threshold,
        entry_zscore_threshold=args.entry_zscore_threshold,
        minimum_edge_over_combined_spread_ratio=args.minimum_edge_over_combined_spread_ratio,
        minimum_theoretical_edge_dollars=minimum_theoretical_edge_dollars,
        exit_gap_threshold=args.exit_gap_threshold,
        exit_zscore_threshold=args.exit_zscore_threshold,
        z_window_events=args.z_window_events,
        timestamp_unit=args.timestamp_unit,
        route_latency_ms=args.route_latency_ms,
        max_quote_age_ms=args.max_quote_age_ms,
        max_hold_ms=args.max_hold_ms,
        process_by_day=not args.no_process_by_day,
        chunk_days=args.chunk_days,
        warmup_days=args.warmup_days,
        chunk_lookahead_ms=args.chunk_lookahead_ms,
        collect_engine=args.collect_engine,
        taker_fee_bps=args.taker_fee_bps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config_from_args(args)
    market_a = MarketSlice(market_id=args.market_a_id, token_id=args.market_a_token)
    market_b = MarketSlice(market_id=args.market_b_id, token_id=args.market_b_token)

    result = run_conditional_probability_squeeze_backtest(
        [Path(path) for path in args.source],
        market_a=market_a,
        market_b=market_b,
        config=config,
    )

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    signals_path = output_dir / "signals.parquet"
    trades_path = output_dir / "trades.parquet"
    success_path = output_dir / "successful_fok_baskets.parquet"
    rejection_path = output_dir / "route_arrival_rejections.parquet"
    flatten_path = output_dir / "flatten_baskets.parquet"
    chunk_stats_path = output_dir / "chunk_stats.parquet"

    successful_fok_baskets = result.trades.filter(pl.col("trade_state").is_in(["basket_closed", "basket_open"]))
    route_arrival_rejections = result.trades.filter(pl.col("trade_state") == "expired_before_fill")
    flatten_baskets = result.trades.filter(
        pl.col("trade_state").is_in(["flattened_stage1", "flattened_stage2", "partial_unresolved"])
    )

    _write_json(result.summary, summary_path)
    report_path.write_text(
        _build_report(
            result.summary,
            source=args.source,
            market_a=market_a,
            market_b=market_b,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
    _write_frame(result.signals, signals_path)
    _write_frame(result.trades, trades_path)
    _write_frame(successful_fok_baskets, success_path)
    _write_frame(route_arrival_rejections, rejection_path)
    _write_frame(flatten_baskets, flatten_path)
    _write_frame(result.chunk_stats if result.chunk_stats is not None else pl.DataFrame(), chunk_stats_path)

    latency_ms = int(result.summary["route_latency_ms"])
    print(f"Wrote squeeze diagnostics to {output_dir}")
    print(
        f"Signals={result.summary['total_valid_signals_generated']} | "
        f"FOK survival at {latency_ms}ms={float(result.summary['fok_survival_rate_at_route_latency']):.2%} | "
        f"route-arrival rejects={result.summary['route_arrival_full_rejections']} | "
        f"partial-fills={result.summary['partial_fills_requiring_flatten']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())