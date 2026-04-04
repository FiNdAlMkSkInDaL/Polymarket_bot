#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.conditional_probability_squeeze import (
    ConditionalProbabilitySqueezeConfig,
    MarketSlice,
    TimestampUnit,
    default_minimum_theoretical_edge_dollars,
    run_conditional_probability_squeeze_backtest,
)


DATE_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_PAIRS_CONFIG = PROJECT_ROOT / "config" / "squeeze_pairs.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "squeeze_diagnostics" / "batch"

RANKING_SCHEMA: dict[str, pl.DataType] = {
    "pair_id": pl.String,
    "parent_market_id": pl.String,
    "child_market_id": pl.String,
    "parent_token_id": pl.String,
    "child_token_id": pl.String,
    "notes": pl.String,
    "relationship_type": pl.String,
    "status": pl.String,
    "error": pl.String,
    "total_valid_signals_generated": pl.Int64,
    "decision_time_fok_passes": pl.Int64,
    "decision_time_fok_rejections": pl.Int64,
    "route_arrival_full_rejections": pl.Int64,
    "partial_fills_requiring_flatten": pl.Int64,
    "successful_fok_baskets": pl.Int64,
    "fok_survival_rate_at_route_latency": pl.Float64,
    "successful_fok_net_pnl": pl.Float64,
    "flattened_basket_net_pnl": pl.Float64,
    "flattened_basket_net_loss": pl.Float64,
    "ranking_net_pnl": pl.Float64,
    "net_pnl": pl.Float64,
    "gross_pnl": pl.Float64,
    "chunks_processed": pl.Int64,
    "route_latency_ms": pl.Int64,
    "route_latency_us": pl.Int64,
}


@dataclass(frozen=True, slots=True)
class SqueezePair:
    pair_id: str
    parent_market_id: str
    child_market_id: str
    parent_token_id: str | None = None
    child_token_id: str | None = None
    notes: str | None = None
    relationship_type: str | None = None


@dataclass(slots=True)
class SqueezeBatchResult:
    ranking: pl.DataFrame
    summary: dict[str, object]
    scan_root: Path
    source_files: tuple[Path, ...]
    pairs: tuple[SqueezePair, ...]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep configured parent/child pairs through the Conditional Probability Squeeze backtest.",
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Parquet lake root, l2_book root, or parquet file with YYYY-MM-DD partitions.",
    )
    parser.add_argument(
        "--pairs-config",
        type=Path,
        default=DEFAULT_PAIRS_CONFIG,
        help="JSON config mapping parent_market_id to child_market_id pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregated ranking artifacts.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive YYYY-MM-DD start date.")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive YYYY-MM-DD end date.")
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
    parser.add_argument("--route-latency-ms", type=int, default=100)
    parser.add_argument("--max-quote-age-ms", type=int, default=5_000)
    parser.add_argument("--max-hold-ms", type=int, default=60_000)
    parser.add_argument("--taker-fee-bps", type=float, default=0.0)
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument("--warmup-days", type=int, default=1)
    parser.add_argument("--chunk-lookahead-ms", type=int, default=None)
    parser.add_argument(
        "--collect-engine",
        choices=["auto", "streaming"],
        default="streaming",
    )
    parser.add_argument(
        "--no-process-by-day",
        action="store_true",
        help="Disable day chunking and run every pair as one full lazy query.",
    )
    return parser.parse_args(argv)


def _parse_date(raw_value: str | None) -> date | None:
    if raw_value is None:
        return None
    return date.fromisoformat(raw_value)


def _pair_id(parent_market_id: str, child_market_id: str) -> str:
    return f"{parent_market_id}__{child_market_id}"


def resolve_input_scan_root(input_root: Path) -> Path:
    if input_root.is_file():
        return input_root
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    book_root = input_root / "l2_book"
    if book_root.is_dir():
        return book_root
    return input_root


def _discover_partition_files(input_root: Path) -> dict[date, list[Path]]:
    scan_root = resolve_input_scan_root(input_root)
    if scan_root.is_file():
        return {date.min: [scan_root]}

    partitions: dict[date, list[Path]] = {}
    for parquet_path in sorted(scan_root.rglob("*.parquet")):
        partition_date: date | None = None
        for parent in (parquet_path.parent, *parquet_path.parents):
            if parent == scan_root.parent:
                break
            if DATE_DIR_PATTERN.match(parent.name):
                partition_date = date.fromisoformat(parent.name)
                break
        if partition_date is None:
            partition_date = date.min
        partitions.setdefault(partition_date, []).append(parquet_path)

    if not partitions:
        raise ValueError(f"No parquet files found under {input_root}")
    return partitions


def discover_source_files(
    input_root: Path,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[Path, ...]:
    partition_files = _discover_partition_files(input_root)
    source_files: list[Path] = []
    for partition_date in sorted(partition_files):
        if partition_date != date.min:
            if start_date is not None and partition_date < start_date:
                continue
            if end_date is not None and partition_date > end_date:
                continue
        source_files.extend(sorted(partition_files[partition_date]))
    if not source_files:
        raise ValueError("Date filters removed every parquet partition from the squeeze sweep")
    return tuple(source_files)


def load_squeeze_pairs(config_path: Path) -> tuple[SqueezePair, ...]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "pairs" in raw:
        raw_pairs = raw["pairs"]
    elif isinstance(raw, list):
        raw_pairs = raw
    elif isinstance(raw, dict):
        raw_pairs = []
        for parent_market_id, child_value in raw.items():
            if isinstance(child_value, str):
                raw_pairs.append(
                    {
                        "parent_market_id": parent_market_id,
                        "child_market_id": child_value,
                    }
                )
                continue
            if isinstance(child_value, dict):
                raw_pairs.append({"parent_market_id": parent_market_id, **child_value})
                continue
            raise ValueError("Unsupported squeeze pair mapping value in config JSON")
    else:
        raise ValueError("Pairs config must be a list, a {'pairs': [...]} object, or a parent->child mapping")

    pairs: list[SqueezePair] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError("Each squeeze pair entry must be a JSON object")
        parent_market_id = str(raw_pair["parent_market_id"])
        child_market_id = str(raw_pair["child_market_id"])
        pairs.append(
            SqueezePair(
                pair_id=str(raw_pair.get("pair_id") or _pair_id(parent_market_id, child_market_id)),
                parent_market_id=parent_market_id,
                child_market_id=child_market_id,
                parent_token_id=(
                    str(raw_pair["parent_token_id"]) if raw_pair.get("parent_token_id") is not None else None
                ),
                child_token_id=(
                    str(raw_pair["child_token_id"]) if raw_pair.get("child_token_id") is not None else None
                ),
                notes=str(raw_pair["notes"]) if raw_pair.get("notes") is not None else None,
                relationship_type=(
                    str(raw_pair["relationship_type"]) if raw_pair.get("relationship_type") is not None else None
                ),
            )
        )
    if not pairs:
        raise ValueError("Pairs config did not contain any parent/child mappings")
    return tuple(pairs)


def _ranking_defaults(pair: SqueezePair) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "parent_market_id": pair.parent_market_id,
        "child_market_id": pair.child_market_id,
        "parent_token_id": pair.parent_token_id,
        "child_token_id": pair.child_token_id,
        "notes": pair.notes,
        "relationship_type": pair.relationship_type,
        "status": "ok",
        "error": None,
        "total_valid_signals_generated": 0,
        "decision_time_fok_passes": 0,
        "decision_time_fok_rejections": 0,
        "route_arrival_full_rejections": 0,
        "partial_fills_requiring_flatten": 0,
        "successful_fok_baskets": 0,
        "fok_survival_rate_at_route_latency": 0.0,
        "successful_fok_net_pnl": 0.0,
        "flattened_basket_net_pnl": 0.0,
        "flattened_basket_net_loss": 0.0,
        "ranking_net_pnl": 0.0,
        "net_pnl": 0.0,
        "gross_pnl": 0.0,
        "chunks_processed": 0,
        "route_latency_ms": 0,
        "route_latency_us": 0,
    }


def _build_ranking_row(pair: SqueezePair, summary: dict[str, Any]) -> dict[str, Any]:
    successful_fok_net_pnl = float(summary["successful_fok_net_pnl"])
    flattened_basket_net_loss = float(summary["flattened_basket_net_loss"])
    row = _ranking_defaults(pair)
    row.update(
        {
            "total_valid_signals_generated": int(summary["total_valid_signals_generated"]),
            "decision_time_fok_passes": int(summary["decision_time_fok_passes"]),
            "decision_time_fok_rejections": int(summary["decision_time_fok_rejections"]),
            "route_arrival_full_rejections": int(summary["route_arrival_full_rejections"]),
            "partial_fills_requiring_flatten": int(summary["partial_fills_requiring_flatten"]),
            "successful_fok_baskets": int(summary["successful_fok_baskets"]),
            "fok_survival_rate_at_route_latency": float(summary["fok_survival_rate_at_route_latency"]),
            "successful_fok_net_pnl": successful_fok_net_pnl,
            "flattened_basket_net_pnl": float(summary["flattened_basket_net_pnl"]),
            "flattened_basket_net_loss": flattened_basket_net_loss,
            "ranking_net_pnl": successful_fok_net_pnl - flattened_basket_net_loss,
            "net_pnl": float(summary["net_pnl"]),
            "gross_pnl": float(summary["gross_pnl"]),
            "chunks_processed": int(summary["chunks_processed"]),
            "route_latency_ms": int(summary["route_latency_ms"]),
            "route_latency_us": int(summary["route_latency_us"]),
        }
    )
    return row


def _error_ranking_row(pair: SqueezePair, error: Exception, config: ConditionalProbabilitySqueezeConfig) -> dict[str, Any]:
    row = _ranking_defaults(pair)
    row.update(
        {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "route_latency_ms": int(config.route_latency_ms),
            "route_latency_us": int(config.route_latency_us),
        }
    )
    return row


def _empty_ranking_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=RANKING_SCHEMA)


def _ensure_ranking_schema(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in RANKING_SCHEMA.items()
        if name not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(missing)
    return frame.select(list(RANKING_SCHEMA))


def _sort_ranking(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return (
        frame.with_columns(
            [
                (pl.col("status") == "ok").alias("_status_ok"),
                pl.col("ranking_net_pnl").fill_null(0.0).alias("_ranking_net_pnl"),
            ]
        )
        .sort(["_status_ok", "_ranking_net_pnl", "pair_id"], descending=[True, True, False])
        .drop(["_status_ok", "_ranking_net_pnl"])
    )


def run_squeeze_batch(
    input_root: Path,
    pairs_config_path: Path,
    config: ConditionalProbabilitySqueezeConfig,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> SqueezeBatchResult:
    scan_root = resolve_input_scan_root(input_root)
    source_files = discover_source_files(input_root, start_date=start_date, end_date=end_date)
    pairs = load_squeeze_pairs(pairs_config_path)
    ranking_rows: list[dict[str, Any]] = []

    for pair in pairs:
        try:
            result = run_conditional_probability_squeeze_backtest(
                source_files,
                market_a=MarketSlice(market_id=pair.parent_market_id, token_id=pair.parent_token_id),
                market_b=MarketSlice(market_id=pair.child_market_id, token_id=pair.child_token_id),
                config=config,
            )
        except Exception as error:
            ranking_rows.append(_error_ranking_row(pair, error, config))
            continue
        ranking_rows.append(_build_ranking_row(pair, result.summary))

    ranking = _sort_ranking(_ensure_ranking_schema(pl.DataFrame(ranking_rows, schema=RANKING_SCHEMA))) if ranking_rows else _empty_ranking_frame()
    completed_pairs = int(ranking.filter(pl.col("status") == "ok").height) if ranking.height else 0
    failed_pairs = int(ranking.filter(pl.col("status") == "error").height) if ranking.height else 0
    top_pair_id = None
    top_pair_net_pnl = 0.0
    if completed_pairs:
        top_row = ranking.filter(pl.col("status") == "ok").row(0, named=True)
        top_pair_id = str(top_row["pair_id"])
        top_pair_net_pnl = float(top_row["ranking_net_pnl"])

    summary: dict[str, object] = {
        "pairs_requested": len(pairs),
        "pairs_completed": completed_pairs,
        "pairs_failed": failed_pairs,
        "source_file_count": len(source_files),
        "input_root": str(input_root),
        "scan_root": str(scan_root),
        "pairs_config": str(pairs_config_path),
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat() if end_date is not None else None,
        "route_latency_ms": int(config.route_latency_ms),
        "route_latency_us": int(config.route_latency_us),
        "minimum_edge_over_combined_spread_ratio": float(config.minimum_edge_over_combined_spread_ratio),
        "minimum_theoretical_edge_dollars": float(config.minimum_theoretical_edge_dollars),
        "process_by_day": bool(config.process_by_day),
        "chunk_days": int(config.chunk_days),
        "warmup_days": int(config.warmup_days),
        "chunk_lookahead_ms": int(config.effective_chunk_lookahead_ms),
        "top_pair_id": top_pair_id,
        "top_pair_ranking_net_pnl": top_pair_net_pnl,
    }
    return SqueezeBatchResult(
        ranking=ranking,
        summary=summary,
        scan_root=scan_root,
        source_files=source_files,
        pairs=pairs,
    )


def _write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def render_markdown(result: SqueezeBatchResult, input_root: Path, pairs_config_path: Path) -> str:
    lines = [
        "# Conditional Probability Squeeze Pair Sweep",
        "",
        "## Setup",
        f"- Input root: {input_root}",
        f"- Scan root: {result.scan_root}",
        f"- Pairs config: {pairs_config_path}",
        f"- Source files scanned: {result.summary['source_file_count']}",
        f"- Pairs requested: {result.summary['pairs_requested']}",
        f"- Pairs completed: {result.summary['pairs_completed']}",
        f"- Pairs failed: {result.summary['pairs_failed']}",
        f"- Route-arrival latency: {result.summary['route_latency_ms']}ms ({result.summary['route_latency_us']}us)",
        f"- Minimum edge over combined spread: {float(result.summary.get('minimum_edge_over_combined_spread_ratio', 0.0)):.2%}",
        f"- Minimum theoretical edge dollars: {float(result.summary.get('minimum_theoretical_edge_dollars', 0.0)):.2f}",
        f"- Process by day: {result.summary['process_by_day']}",
        f"- Chunk days: {result.summary['chunk_days']}",
        f"- Warmup days: {result.summary['warmup_days']}",
        f"- Chunk lookahead: {result.summary['chunk_lookahead_ms']}ms",
        "",
        "## Ranking",
        "| Rank | Pair | Relation | Parent | Child | Signals | FOK Survival | FOK Net PnL | Flatten Loss | Ranking Net PnL | Status |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    if result.ranking.is_empty():
        lines.append("| n/a | n/a | n/a | n/a | n/a | 0 | 0.00% | 0.000000 | 0.000000 | 0.000000 | empty |")
    else:
        for index, row in enumerate(result.ranking.iter_rows(named=True), start=1):
            lines.append(
                "| {rank} | {pair_id} | {relationship_type} | {parent} | {child} | {signals} | {survival:.2%} | {fok_pnl:.6f} | {flatten_loss:.6f} | {ranking_pnl:.6f} | {status} |".format(
                    rank=index,
                    pair_id=row["pair_id"],
                    relationship_type=row["relationship_type"] or "conditional",
                    parent=row["parent_market_id"],
                    child=row["child_market_id"],
                    signals=int(row["total_valid_signals_generated"] or 0),
                    survival=float(row["fok_survival_rate_at_route_latency"] or 0.0),
                    fok_pnl=float(row["successful_fok_net_pnl"] or 0.0),
                    flatten_loss=float(row["flattened_basket_net_loss"] or 0.0),
                    ranking_pnl=float(row["ranking_net_pnl"] or 0.0),
                    status=row["status"],
                )
            )
            if row["status"] == "error" and row["error"]:
                lines.append(f"Error: {row['error']}")

    return "\n".join(lines) + "\n"


def write_batch_outputs(result: SqueezeBatchResult, output_dir: Path, *, input_root: Path, pairs_config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(result.ranking, output_dir / "ranking.csv")
    _write_table(result.ranking, output_dir / "ranking.parquet")
    _write_json(result.summary, output_dir / "batch_summary.json")
    (output_dir / "ranking.md").write_text(
        render_markdown(result, input_root, pairs_config_path),
        encoding="utf-8",
    )


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
    config = build_config_from_args(args)
    result = run_squeeze_batch(
        args.input_root,
        args.pairs_config,
        config,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
    )
    write_batch_outputs(result, args.output_dir.resolve(), input_root=args.input_root, pairs_config_path=args.pairs_config)
    print(result.ranking)
    print(
        f"Wrote squeeze batch ranking to {args.output_dir.resolve()} | "
        f"pairs={result.summary['pairs_completed']}/{result.summary['pairs_requested']} ok | "
        f"top_pair={result.summary['top_pair_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())