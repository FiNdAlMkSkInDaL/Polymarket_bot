#!/usr/bin/env python3
from __future__ import annotations

"""Vectorized Polars backtest for Strategy 3: Mid-Tier Probability Compression.

The baseline lake is assumed to store YES-token best bid / best ask quotes. The
strategy trades the NO side passively, so the script derives the passive NO
book from the YES book with the full implicit spread preserved:

    no_bid = 1 - yes_ask
    no_ask = 1 - yes_bid

If the lake already stores NO-side quotes directly, use ``--quote-side no``.
"""

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl


QuoteSide = Literal["yes", "no"]

BASE_REQUIRED_COLUMNS = {
    "timestamp",
    "market_id",
    "event_id",
    "token_id",
    "best_bid",
    "best_ask",
    "bid_depth",
    "ask_depth",
}
RESOLUTION_COLUMNS = {
    "resolution_timestamp",
    "final_resolution_value",
}
REQUIRED_COLUMNS = BASE_REQUIRED_COLUMNS | RESOLUTION_COLUMNS
ENRICHED_MANIFEST_NAME = "enriched_manifest.json"
SIMULATED_EXIT_MODE = "simulated_last_no_bid"
SETTLEMENT_EXIT_MODE = "settlement"

INTEGER_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
FLOAT_DTYPES = {pl.Float32, pl.Float64}

CANDIDATE_ORDER_SCHEMA = {
    "event_id": pl.String,
    "market_id": pl.String,
    "token_id": pl.String,
    "order_timestamp": pl.Datetime,
    "resolution_timestamp": pl.Datetime,
    "entry_yes_bid": pl.Float64,
    "entry_yes_ask": pl.Float64,
    "entry_no_bid": pl.Float64,
    "entry_no_ask": pl.Float64,
    "entry_spread": pl.Float64,
    "future_min_no_ask_exclusive": pl.Float64,
    "bid_depth": pl.Float64,
    "ask_depth": pl.Float64,
    "legs_in_snapshot": pl.UInt32,
    "top2_yes_sum": pl.Float64,
    "midtier_yes_sum": pl.Float64,
    "math_remainder": pl.Float64,
    "midtier_dislocation": pl.Float64,
    "filled": pl.Boolean,
    "market_last_timestamp": pl.Datetime,
    "market_last_no_bid": pl.Float64,
    "market_last_no_ask": pl.Float64,
    "effective_exit_timestamp": pl.Datetime,
    "effective_exit_no_price": pl.Float64,
    "exit_mode": pl.String,
    "gamma_market_status": pl.String,
    "gamma_closed": pl.Boolean,
    "final_resolution_value": pl.Float64,
}


@dataclass(slots=True)
class MidTierCompressionConfig:
    quote_side: QuoteSide = "yes"
    timestamp_unit: str = "ms"
    resolution_timestamp_unit: str = "ms"
    top2_yes_threshold: float = 0.95
    midtier_yes_threshold: float = 0.15
    max_leg_notional_usd: float = 50.0
    max_concurrent_names: int = 100


@dataclass(slots=True)
class BacktestArtifacts:
    summary: dict[str, Any]
    orders: pl.DataFrame
    event_tail: pl.DataFrame


@dataclass(slots=True)
class PreparedBacktest:
    signal_leg_rows: int
    signal_snapshots: int
    scheduled_orders: pl.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the Mid-Tier Probability Compression maker strategy on a Parquet lake.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Parquet file, directory, or glob pointing at the normalized L2 lake.",
    )
    parser.add_argument(
        "--quote-side",
        choices=("yes", "no"),
        default="yes",
        help="Interpret best_bid / best_ask as YES quotes (default) or NO quotes.",
    )
    parser.add_argument("--timestamp-unit", choices=("ns", "us", "ms", "s"), default="ms")
    parser.add_argument(
        "--resolution-timestamp-unit",
        choices=("ns", "us", "ms", "s"),
        default="ms",
    )
    parser.add_argument("--top2-yes-threshold", type=float, default=0.95)
    parser.add_argument("--midtier-yes-threshold", type=float, default=0.15)
    parser.add_argument("--max-leg-notional-usd", type=float, default=50.0)
    parser.add_argument("--max-concurrent-names", type=int, default=100)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON output path for the summary metrics.",
    )
    parser.add_argument(
        "--orders-output",
        type=Path,
        default=None,
        help="Optional CSV or Parquet output path for the candidate order table.",
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=None,
        help="Optional CSV or Parquet output path for the event-level tail-risk table.",
    )
    return parser.parse_args()


def _resolve_scan_target(input_path: str | Path) -> str:
    input_str = str(input_path)
    if any(token in input_str for token in "*?[]"):
        return input_str

    path = Path(input_str)
    if path.is_dir():
        return str(path / "**" / "*.parquet")
    if path.is_file():
        return str(path)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _validate_base_schema(schema: pl.Schema) -> None:
    missing = sorted(BASE_REQUIRED_COLUMNS - set(schema.names()))
    if missing:
        raise ValueError(f"Parquet lake is missing required columns: {', '.join(missing)}")


def _find_enriched_manifest(input_path: str | Path) -> Path | None:
    input_str = str(input_path)
    if any(token in input_str for token in "*?[]"):
        return None

    path = Path(input_str)
    search_root = path if path.is_dir() else path.parent
    for candidate in (search_root, *search_root.parents):
        manifest_path = candidate / ENRICHED_MANIFEST_NAME
        if manifest_path.is_file():
            return manifest_path
    return None


def _load_enriched_manifest_table(manifest_path: Path) -> pl.DataFrame:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_markets = payload.get("markets") or []
    if not isinstance(raw_markets, list):
        raise ValueError(f"Malformed enriched manifest at {manifest_path}: expected a 'markets' list")

    rows: list[dict[str, Any]] = []
    for item in raw_markets:
        if not isinstance(item, dict):
            continue
        market_id = str(item.get("market_id") or "").strip().lower()
        if not market_id:
            continue
        rows.append(
            {
                "market_id": market_id,
                "gamma_market_status": str(item.get("gamma_market_status") or "missing_gamma").strip() or "missing_gamma",
                "gamma_closed": bool(item.get("gamma_closed")),
                "resolution_timestamp": item.get("resolution_timestamp"),
                "final_resolution_value": item.get("final_resolution_value"),
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "market_id": pl.String,
                "gamma_market_status": pl.String,
                "gamma_closed": pl.Boolean,
                "resolution_timestamp": pl.String,
                "final_resolution_value": pl.Float64,
            }
        )

    return pl.DataFrame(rows).select(
        pl.col("market_id").cast(pl.String).str.to_lowercase().alias("market_id"),
        pl.col("gamma_market_status").cast(pl.String).fill_null("missing_gamma").alias("gamma_market_status"),
        pl.col("gamma_closed").cast(pl.Boolean).fill_null(False).alias("gamma_closed"),
        pl.col("resolution_timestamp").cast(pl.String).alias("resolution_timestamp"),
        pl.col("final_resolution_value").cast(pl.Float64).alias("final_resolution_value"),
    )


def _datetime_expr(column_name: str, schema: pl.Schema, time_unit: str) -> pl.Expr:
    dtype = schema[column_name]
    base_type = dtype.base_type() if hasattr(dtype, "base_type") else dtype

    if base_type is pl.Datetime:
        return pl.col(column_name).cast(pl.Datetime(time_unit=time_unit))
    if dtype == pl.Date:
        return pl.col(column_name).cast(pl.Datetime(time_unit=time_unit))
    if dtype in INTEGER_DTYPES or dtype in FLOAT_DTYPES:
        return pl.from_epoch(pl.col(column_name).cast(pl.Int64), time_unit=time_unit)
    if dtype == pl.String:
        return (
            pl.col(column_name)
            .str.to_datetime(time_unit=time_unit, time_zone="UTC", strict=False)
            .dt.replace_time_zone(None)
        )
    raise TypeError(f"Unsupported dtype for {column_name}: {dtype}")


def _scan_preflight_quotes(input_path: str | Path, config: MidTierCompressionConfig) -> pl.LazyFrame:
    scan_target = _resolve_scan_target(input_path)
    lazy_frame = pl.scan_parquet(scan_target, glob=True).with_columns(
        pl.col("market_id").cast(pl.String).str.to_lowercase().alias("market_id")
    )
    schema = lazy_frame.collect_schema()
    _validate_base_schema(schema)

    return (
        lazy_frame.select(
            _datetime_expr("timestamp", schema, config.timestamp_unit).alias("timestamp"),
            pl.col("market_id").cast(pl.String),
            pl.col("event_id").cast(pl.String),
            pl.col("best_bid").cast(pl.Float64),
            pl.col("best_ask").cast(pl.Float64),
        )
        .filter(pl.col("timestamp").is_not_null())
        .filter(pl.col("best_bid").is_between(0.0, 1.0, closed="both"))
        .filter(pl.col("best_ask").is_between(0.0, 1.0, closed="both"))
        .filter(pl.col("best_ask") >= pl.col("best_bid"))
        .unique(subset=["event_id", "market_id", "timestamp"], keep="last")
    )


def _scan_quotes(
    input_path: str | Path,
    config: MidTierCompressionConfig,
    *,
    event_ids: tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    scan_target = _resolve_scan_target(input_path)
    lazy_frame = pl.scan_parquet(scan_target, glob=True).with_columns(
        pl.col("market_id").cast(pl.String).str.to_lowercase().alias("market_id")
    )
    if event_ids is not None:
        lazy_frame = lazy_frame.filter(pl.col("event_id").cast(pl.String).is_in(list(event_ids)))
    schema = lazy_frame.collect_schema()
    _validate_base_schema(schema)

    manifest_path = _find_enriched_manifest(input_path)
    if manifest_path is not None:
        metadata = _load_enriched_manifest_table(manifest_path)
        metadata_columns: list[str] = []
        schema_names = set(schema.names())
        for column_name in ("gamma_market_status", "gamma_closed", "resolution_timestamp", "final_resolution_value"):
            if column_name not in schema_names:
                metadata_columns.append(column_name)
        if metadata_columns:
            lazy_frame = lazy_frame.join(
                metadata.select("market_id", *metadata_columns).lazy(),
                on="market_id",
                how="left",
            )
            schema = lazy_frame.collect_schema()

    missing_resolution = sorted(RESOLUTION_COLUMNS - set(schema.names()))
    if missing_resolution:
        if manifest_path is None:
            raise ValueError(
                "Parquet lake is missing required columns: "
                f"{', '.join(missing_resolution)}. Run scripts/enrich_lake_metadata.py on the lake root first."
            )
        raise ValueError(
            f"Enriched manifest {manifest_path} did not supply required columns: {', '.join(missing_resolution)}"
        )

    if "gamma_market_status" in schema.names():
        status_expr = pl.col("gamma_market_status").cast(pl.String).fill_null("open_or_unresolved")
    else:
        status_expr = pl.when(pl.col("final_resolution_value").is_not_null()).then(pl.lit("resolved")).otherwise(
            pl.lit("open_or_unresolved")
        )

    if "gamma_closed" in schema.names():
        closed_expr = pl.col("gamma_closed").cast(pl.Boolean).fill_null(False)
    else:
        closed_expr = pl.col("final_resolution_value").is_not_null()

    return (
        lazy_frame.select(
            _datetime_expr("timestamp", schema, config.timestamp_unit).alias("timestamp"),
            pl.col("market_id").cast(pl.String),
            pl.col("event_id").cast(pl.String),
            pl.col("token_id").cast(pl.String),
            pl.col("best_bid").cast(pl.Float64),
            pl.col("best_ask").cast(pl.Float64),
            pl.col("bid_depth").cast(pl.Float64),
            pl.col("ask_depth").cast(pl.Float64),
            _datetime_expr("resolution_timestamp", schema, config.resolution_timestamp_unit).alias(
                "resolution_timestamp"
            ),
            pl.col("final_resolution_value").cast(pl.Float64),
            status_expr.alias("gamma_market_status"),
            closed_expr.alias("gamma_closed"),
        )
        .filter(pl.col("timestamp").is_not_null())
        .filter(
            pl.when(pl.col("resolution_timestamp").is_not_null() & pl.col("final_resolution_value").is_not_null())
            .then(pl.col("timestamp") <= pl.col("resolution_timestamp"))
            .otherwise(True)
        )
        .filter(pl.col("best_bid").is_between(0.0, 1.0, closed="both"))
        .filter(pl.col("best_ask").is_between(0.0, 1.0, closed="both"))
        .filter(pl.col("best_ask") >= pl.col("best_bid"))
        .unique(subset=["event_id", "market_id", "token_id", "timestamp"], keep="last")
    )


def _with_binary_books(quotes: pl.LazyFrame, quote_side: QuoteSide) -> pl.LazyFrame:
    if quote_side == "yes":
        return quotes.with_columns(
            pl.col("best_bid").alias("yes_bid"),
            pl.col("best_ask").alias("yes_ask"),
            # Passive NO bids rest against the inverted YES offer.
            (1.0 - pl.col("best_ask")).clip(0.0, 1.0).alias("no_bid"),
            # Future maker fills are tested against the inverted YES bid.
            (1.0 - pl.col("best_bid")).clip(0.0, 1.0).alias("no_ask"),
        )

    return quotes.with_columns(
        (1.0 - pl.col("best_ask")).clip(0.0, 1.0).alias("yes_bid"),
        (1.0 - pl.col("best_bid")).clip(0.0, 1.0).alias("yes_ask"),
        pl.col("best_bid").alias("no_bid"),
        pl.col("best_ask").alias("no_ask"),
    )


def _preflight_eligible_events(
    input_path: str | Path,
    config: MidTierCompressionConfig,
    *,
    min_top2_yes_threshold: float,
) -> tuple[str, ...]:
    preflight_quotes = _with_binary_books(_scan_preflight_quotes(input_path, config), config.quote_side)
    snapshot_rollup = (
        preflight_quotes.group_by(["event_id", "timestamp"])
        .agg(
            pl.len().alias("legs_in_snapshot"),
            pl.col("yes_bid").sort(descending=True).head(2).sum().alias("top2_yes_sum"),
            pl.col("yes_bid").sum().alias("snapshot_yes_sum"),
        )
        .with_columns((pl.col("snapshot_yes_sum") - pl.col("top2_yes_sum")).alias("midtier_yes_sum"))
        .with_columns(
            (
                (pl.col("legs_in_snapshot") >= 3)
                & (pl.col("top2_yes_sum") > min_top2_yes_threshold)
                & (pl.col("midtier_yes_sum") > config.midtier_yes_threshold)
            ).alias("eligible_snapshot")
        )
        .group_by("event_id")
        .agg(pl.col("eligible_snapshot").any().alias("eligible_event"))
        .filter(pl.col("eligible_event"))
        .select("event_id")
        .collect()
    )

    if snapshot_rollup.is_empty():
        return ()
    return tuple(str(value) for value in snapshot_rollup.get_column("event_id").to_list())


def _build_signal_base_rows(quotes: pl.LazyFrame, config: MidTierCompressionConfig) -> pl.LazyFrame:
    future_state = (
        quotes.sort(["market_id", "timestamp"], descending=[False, True])
        .with_columns(
            # The long-running fill window is partitioned strictly by market_id.
            # For YES-schema input this is the inverted future NO ask.
            pl.col("no_ask").cum_min().shift(1).over("market_id").alias("future_min_no_ask_exclusive")
        )
        .select("market_id", "timestamp", "future_min_no_ask_exclusive")
    )
    terminal_state = (
        quotes.sort(["market_id", "timestamp"])
        .group_by("market_id")
        .agg(
            pl.col("timestamp").last().alias("market_last_timestamp"),
            pl.col("no_bid").last().alias("market_last_no_bid"),
            pl.col("no_ask").last().alias("market_last_no_ask"),
        )
    )

    ranked = (
        quotes.sort(["event_id", "timestamp", "yes_bid", "market_id"], descending=[False, False, True, False])
        .with_columns(
            pl.len().over(["event_id", "timestamp"]).alias("legs_in_snapshot"),
            pl.col("yes_bid").rank(method="ordinal", descending=True).over(["event_id", "timestamp"]).alias(
                "leg_rank"
            ),
        )
        .with_columns(
            pl.when(pl.col("leg_rank") <= 2).then(pl.col("yes_bid")).otherwise(0.0).sum().over(
                ["event_id", "timestamp"]
            ).alias("top2_yes_sum"),
            pl.when(pl.col("leg_rank") > 2).then(pl.col("yes_bid")).otherwise(0.0).sum().over(
                ["event_id", "timestamp"]
            ).alias("midtier_yes_sum"),
        )
        .with_columns(
            pl.when(pl.col("top2_yes_sum") < 1.0)
            .then(1.0 - pl.col("top2_yes_sum"))
            .otherwise(0.0)
            .alias("math_remainder"),
        )
        .with_columns((pl.col("midtier_yes_sum") - pl.col("math_remainder")).alias("midtier_dislocation"))
        .filter(pl.col("legs_in_snapshot") >= 3)
        .filter(pl.col("leg_rank") > 2)
        .filter(pl.col("midtier_yes_sum") > config.midtier_yes_threshold)
        .filter(pl.col("no_bid") > 0.0)
        .join(future_state, on=["market_id", "timestamp"], how="left")
        .join(terminal_state, on="market_id", how="left")
        .with_columns(
            pl.col("timestamp").alias("order_timestamp"),
            pl.col("no_bid").alias("entry_no_bid"),
            pl.col("no_ask").alias("entry_no_ask"),
            pl.col("yes_bid").alias("entry_yes_bid"),
            pl.col("yes_ask").alias("entry_yes_ask"),
        )
        .with_columns(
            (pl.col("entry_no_ask") - pl.col("entry_no_bid")).alias("entry_spread"),
            (pl.col("future_min_no_ask_exclusive") <= pl.col("entry_no_bid")).fill_null(False).alias("filled")
        )
        .with_columns(
            pl.when(pl.col("final_resolution_value").is_not_null())
            .then(pl.lit(SETTLEMENT_EXIT_MODE))
            .otherwise(pl.lit(SIMULATED_EXIT_MODE))
            .alias("exit_mode"),
            pl.when(pl.col("final_resolution_value").is_not_null())
            .then((1.0 - pl.col("final_resolution_value")).clip(0.0, 1.0))
            .otherwise(pl.col("market_last_no_bid"))
            .alias("effective_exit_no_price"),
            pl.when(pl.col("final_resolution_value").is_not_null())
            .then(pl.coalesce(pl.col("resolution_timestamp"), pl.col("market_last_timestamp")))
            .otherwise(pl.col("market_last_timestamp"))
            .alias("effective_exit_timestamp"),
        )
    )

    return ranked.select(
        "order_timestamp",
        "resolution_timestamp",
        "market_id",
        "event_id",
        "token_id",
        "entry_yes_bid",
        "entry_yes_ask",
        "entry_no_bid",
        "entry_no_ask",
        "entry_spread",
        "future_min_no_ask_exclusive",
        "bid_depth",
        "ask_depth",
        "legs_in_snapshot",
        "top2_yes_sum",
        "midtier_yes_sum",
        "math_remainder",
        "midtier_dislocation",
        "filled",
        "market_last_timestamp",
        "market_last_no_bid",
        "market_last_no_ask",
        "effective_exit_timestamp",
        "effective_exit_no_price",
        "exit_mode",
        "gamma_market_status",
        "gamma_closed",
        "final_resolution_value",
    )


def _build_signal_rows(quotes: pl.LazyFrame, config: MidTierCompressionConfig) -> pl.LazyFrame:
    return _build_signal_base_rows(quotes, config).filter(pl.col("top2_yes_sum") > config.top2_yes_threshold)


def _build_candidate_orders(signal_rows: pl.LazyFrame) -> pl.LazyFrame:
    return (
        signal_rows.sort(["market_id", "order_timestamp"])
        .group_by("market_id", maintain_order=True)
        .agg(
            pl.col("event_id").first(),
            pl.col("token_id").first(),
            pl.col("order_timestamp").first(),
            pl.col("resolution_timestamp").first(),
            pl.col("entry_yes_bid").first(),
            pl.col("entry_yes_ask").first(),
            pl.col("entry_no_bid").first(),
            pl.col("entry_no_ask").first(),
            pl.col("entry_spread").first(),
            pl.col("future_min_no_ask_exclusive").first(),
            pl.col("bid_depth").first(),
            pl.col("ask_depth").first(),
            pl.col("legs_in_snapshot").first(),
            pl.col("top2_yes_sum").first(),
            pl.col("midtier_yes_sum").first(),
            pl.col("math_remainder").first(),
            pl.col("midtier_dislocation").first(),
            pl.col("filled").first(),
            pl.col("market_last_timestamp").first(),
            pl.col("market_last_no_bid").first(),
            pl.col("market_last_no_ask").first(),
            pl.col("effective_exit_timestamp").first(),
            pl.col("effective_exit_no_price").first(),
            pl.col("exit_mode").first(),
            pl.col("gamma_market_status").first(),
            pl.col("gamma_closed").first(),
            pl.col("final_resolution_value").first(),
        )
    )


def _empty_candidate_orders() -> pl.DataFrame:
    return pl.DataFrame(schema=CANDIDATE_ORDER_SCHEMA)


def _apply_concurrent_cap(candidates: pl.DataFrame, max_concurrent_names: int) -> pl.DataFrame:
    if candidates.is_empty():
        return candidates.with_columns(pl.lit(False).alias("accepted"))

    # Exact overlap admission is recursive: rejected names must not consume
    # future capacity, so we run the cap on the compressed first-signal table.
    # Capacity is released on resolution_timestamp, making this a rolling cap.
    ordered = candidates.sort(
        ["order_timestamp", "midtier_dislocation", "event_id", "market_id"],
        descending=[False, True, False, False],
    )
    active_resolutions: list[tuple[Any, int]] = []
    accepted_flags: list[bool] = []

    for row_index, row in enumerate(ordered.iter_rows(named=True)):
        order_timestamp = row["order_timestamp"]
        while active_resolutions and active_resolutions[0][0] <= order_timestamp:
            heapq.heappop(active_resolutions)

        accepted = len(active_resolutions) < max_concurrent_names
        accepted_flags.append(accepted)
        if accepted:
            heapq.heappush(active_resolutions, (row["effective_exit_timestamp"], row_index))

    return ordered.with_columns(pl.Series("accepted", accepted_flags))


def _attach_position_pnl(orders: pl.DataFrame, config: MidTierCompressionConfig) -> pl.DataFrame:
    notional = pl.lit(config.max_leg_notional_usd)
    return orders.with_columns(
        (pl.col("accepted") & pl.col("filled")).alias("effective_filled"),
        pl.when(pl.col("accepted")).then(notional).otherwise(0.0).alias("reserved_notional_usd"),
        pl.when(pl.col("accepted")).then(notional / pl.col("entry_no_bid")).otherwise(0.0).alias("target_contracts"),
        pl.when(pl.col("accepted") & pl.col("filled")).then(notional).otherwise(0.0).alias("filled_notional_usd"),
        pl.when(pl.col("accepted") & pl.col("filled"))
        .then((notional / pl.col("entry_no_bid")) * pl.col("effective_exit_no_price") - notional)
        .otherwise(0.0)
        .alias("realized_pnl_usd"),
    )


def _event_tail_rollup(orders: pl.DataFrame) -> pl.DataFrame:
    accepted_orders = orders.filter(pl.col("accepted"))
    if accepted_orders.is_empty():
        return pl.DataFrame(
            schema={
                "event_id": pl.String,
                "accepted_legs": pl.UInt32,
                "filled_legs": pl.UInt32,
                "winning_filled_legs": pl.UInt32,
                "event_deployed_notional_usd": pl.Float64,
                "event_realized_pnl_usd": pl.Float64,
                "resolution_timestamp": pl.Datetime,
                "max_top2_yes_sum": pl.Float64,
                "max_midtier_yes_sum": pl.Float64,
                "max_midtier_dislocation": pl.Float64,
                "single_fill_event": pl.Boolean,
                "favorite_collapse_hit": pl.Boolean,
                "legging_loss_usd": pl.Float64,
                "naked_equity_usd": pl.Float64,
                "naked_equity_peak_usd": pl.Float64,
                "naked_drawdown_usd": pl.Float64,
            }
        )

    event_rollup = (
        accepted_orders.group_by("event_id")
        .agg(
            pl.len().alias("accepted_legs"),
            pl.col("effective_filled").cast(pl.UInt32).sum().alias("filled_legs"),
            pl.when(pl.col("effective_filled") & (pl.col("final_resolution_value") == 1.0))
            .then(1)
            .otherwise(0)
            .cast(pl.UInt32)
            .sum()
            .alias("winning_filled_legs"),
            pl.col("filled_notional_usd").sum().alias("event_deployed_notional_usd"),
            pl.col("realized_pnl_usd").sum().alias("event_realized_pnl_usd"),
            pl.col("resolution_timestamp").max().alias("resolution_timestamp"),
            pl.col("top2_yes_sum").max().alias("max_top2_yes_sum"),
            pl.col("midtier_yes_sum").max().alias("max_midtier_yes_sum"),
            pl.col("midtier_dislocation").max().alias("max_midtier_dislocation"),
        )
        .with_columns(
            (pl.col("filled_legs") == 1).alias("single_fill_event"),
            ((pl.col("filled_legs") == 1) & (pl.col("winning_filled_legs") == 1)).alias("favorite_collapse_hit"),
        )
        .with_columns(
            pl.when(pl.col("favorite_collapse_hit"))
            .then(pl.col("event_deployed_notional_usd"))
            .otherwise(0.0)
            .alias("legging_loss_usd")
        )
        .sort("resolution_timestamp")
    )

    naked_events = event_rollup.filter(pl.col("single_fill_event")).sort("resolution_timestamp")
    if naked_events.is_empty():
        return event_rollup.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("naked_equity_usd"),
            pl.lit(None, dtype=pl.Float64).alias("naked_equity_peak_usd"),
            pl.lit(None, dtype=pl.Float64).alias("naked_drawdown_usd"),
        )

    naked_events = naked_events.with_columns(pl.col("event_realized_pnl_usd").cum_sum().alias("naked_equity_usd"))
    naked_events = naked_events.with_columns(pl.col("naked_equity_usd").cum_max().alias("naked_running_peak_usd"))
    naked_events = naked_events.with_columns(
        pl.when(pl.col("naked_running_peak_usd") > 0.0)
        .then(pl.col("naked_running_peak_usd"))
        .otherwise(0.0)
        .alias("naked_equity_peak_usd")
    )
    naked_events = naked_events.with_columns(
        (pl.col("naked_equity_usd") - pl.col("naked_equity_peak_usd")).alias("naked_drawdown_usd")
    )

    return event_rollup.join(
        naked_events.select("event_id", "naked_equity_usd", "naked_equity_peak_usd", "naked_drawdown_usd"),
        on="event_id",
        how="left",
    )


def _summary_from_tables(
    orders: pl.DataFrame,
    event_tail: pl.DataFrame,
    signal_leg_rows: int,
    signal_snapshots: int,
    config: MidTierCompressionConfig,
) -> dict[str, Any]:
    accepted_orders = orders.filter(pl.col("accepted"))
    filled_orders = int(orders.select(pl.col("effective_filled").cast(pl.UInt32).sum()).item() or 0)
    winning_fills = int(
        orders.select(
            pl.when(pl.col("effective_filled") & (pl.col("realized_pnl_usd") > 0.0))
            .then(1)
            .otherwise(0)
            .sum()
        ).item()
        or 0
    )
    losing_fills = int(
        orders.select(
            pl.when(pl.col("effective_filled") & (pl.col("realized_pnl_usd") < 0.0))
            .then(1)
            .otherwise(0)
            .sum()
        ).item()
        or 0
    )
    total_realized_pnl = float(orders.select(pl.col("realized_pnl_usd").sum()).item() or 0.0)
    total_legging_loss = float(event_tail.select(pl.col("legging_loss_usd").sum()).item() or 0.0)
    simulated_exit_candidate_orders = int(orders.filter(pl.col("exit_mode") == SIMULATED_EXIT_MODE).height)
    simulated_exit_accepted_orders = int(
        orders.filter(pl.col("accepted") & (pl.col("exit_mode") == SIMULATED_EXIT_MODE)).height
    )
    simulated_exit_fills = int(
        orders.select(
            pl.when(pl.col("effective_filled") & (pl.col("exit_mode") == SIMULATED_EXIT_MODE))
            .then(1)
            .otherwise(0)
            .sum()
        ).item()
        or 0
    )
    settlement_exit_fills = int(
        orders.select(
            pl.when(pl.col("effective_filled") & (pl.col("exit_mode") == SETTLEMENT_EXIT_MODE))
            .then(1)
            .otherwise(0)
            .sum()
        ).item()
        or 0
    )
    filled_entry_spread_sum = float(
        orders.select(
            pl.when(pl.col("effective_filled")).then(pl.col("entry_spread")).otherwise(0.0).sum()
        ).item()
        or 0.0
    )
    avg_filled_entry_spread = filled_entry_spread_sum / filled_orders if filled_orders else 0.0
    max_legging_drawdown = 0.0
    worst_single_fill_event_pnl = 0.0

    naked_events = event_tail.filter(pl.col("single_fill_event")) if not event_tail.is_empty() else event_tail
    if not naked_events.is_empty():
        max_legging_drawdown = abs(float(naked_events.select(pl.col("naked_drawdown_usd").min()).item() or 0.0))
        worst_single_fill_event_pnl = float(
            naked_events.select(pl.col("event_realized_pnl_usd").min()).item() or 0.0
        )

    return {
        "quote_side": config.quote_side,
        "quote_schema": "yes_bbo_inverted_to_no" if config.quote_side == "yes" else "no_bbo_direct",
        "no_bid_formula": "1.0 - yes_best_ask" if config.quote_side == "yes" else "input best_bid",
        "no_ask_formula": "1.0 - yes_best_bid" if config.quote_side == "yes" else "input best_ask",
        "maker_fill_reference": "future_min_no_ask_exclusive",
        "maker_fill_partition_key": "market_id",
        "signal_snapshot_partition": ["event_id", "timestamp"],
        "concurrent_name_cap_mode": "rolling",
        "concurrent_name_cap_release_key": "resolution_timestamp",
        "concurrent_name_cap_release_fallback_key": "last_quote_timestamp_for_open_markets",
        "unresolved_exit_mode": "last_no_bid_mark_to_market",
        "top2_yes_threshold": config.top2_yes_threshold,
        "midtier_yes_threshold": config.midtier_yes_threshold,
        "max_leg_notional_usd": config.max_leg_notional_usd,
        "max_concurrent_names": config.max_concurrent_names,
        "signal_leg_rows": signal_leg_rows,
        "signal_snapshots": signal_snapshots,
        "candidate_orders": int(orders.height),
        "accepted_orders": int(accepted_orders.height),
        "rejected_orders": int(orders.height - accepted_orders.height),
        "filled_orders": filled_orders,
        "settlement_exit_fills": settlement_exit_fills,
        "simulated_exit_candidate_orders": simulated_exit_candidate_orders,
        "simulated_exit_accepted_orders": simulated_exit_accepted_orders,
        "simulated_exit_fills": simulated_exit_fills,
        "winning_fills": winning_fills,
        "losing_fills": losing_fills,
        "flat_fills": int(max(0, filled_orders - winning_fills - losing_fills)),
        "fill_win_rate": round((winning_fills / filled_orders) if filled_orders else 0.0, 6),
        "filled_entry_spread_sum": round(filled_entry_spread_sum, 6),
        "avg_filled_entry_spread": round(avg_filled_entry_spread, 6),
        "reserved_notional_usd": round(float(accepted_orders.height * config.max_leg_notional_usd), 2),
        "deployed_notional_usd": round(float(orders.select(pl.col("filled_notional_usd").sum()).item() or 0.0), 2),
        "total_realized_pnl_usd": round(total_realized_pnl, 2),
        "single_fill_events": int(naked_events.height),
        "favorite_collapse_hits": int(
            event_tail.filter(pl.col("favorite_collapse_hit")).height if not event_tail.is_empty() else 0
        ),
        "total_legging_loss_usd": round(total_legging_loss, 2),
        "max_legging_drawdown_usd": round(max_legging_drawdown, 2),
        "worst_single_fill_event_pnl_usd": round(worst_single_fill_event_pnl, 2),
    }


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


def prepare_backtest_grid(
    input_path: str | Path,
    thresholds: tuple[float, ...],
    config: MidTierCompressionConfig | None = None,
) -> dict[float, PreparedBacktest]:
    if config is None:
        config = MidTierCompressionConfig()

    threshold_values: list[float] = []
    for value in thresholds:
        threshold = float(value)
        if threshold not in threshold_values:
            threshold_values.append(threshold)
    if not threshold_values:
        raise ValueError("At least one threshold is required")

    signal_leg_rows_by_threshold = {threshold: 0 for threshold in threshold_values}
    signal_snapshots_by_threshold = {threshold: 0 for threshold in threshold_values}
    candidate_frames_by_threshold = {threshold: [] for threshold in threshold_values}

    eligible_event_ids = _preflight_eligible_events(
        input_path,
        config,
        min_top2_yes_threshold=min(threshold_values),
    )

    if eligible_event_ids:
        quotes = _with_binary_books(
            _scan_quotes(input_path, config, event_ids=eligible_event_ids),
            config.quote_side,
        ).collect()

        event_ids = quotes.select(pl.col("event_id").unique(maintain_order=True)).get_column("event_id").to_list()
        for event_id in event_ids:
            event_quotes = quotes.filter(pl.col("event_id") == event_id)
            if event_quotes.is_empty():
                continue

            signal_base_rows = _build_signal_base_rows(event_quotes.lazy(), config).collect()
            if signal_base_rows.is_empty():
                continue

            for threshold in threshold_values:
                threshold_signal_rows = signal_base_rows.filter(pl.col("top2_yes_sum") > threshold)
                if threshold_signal_rows.is_empty():
                    continue

                signal_leg_rows_by_threshold[threshold] += int(threshold_signal_rows.height)
                signal_snapshots_by_threshold[threshold] += int(
                    threshold_signal_rows.select(pl.col("order_timestamp").n_unique()).item() or 0
                )

                candidates = _build_candidate_orders(threshold_signal_rows.lazy()).collect()
                if not candidates.is_empty():
                    candidate_frames_by_threshold[threshold].append(candidates)

    prepared: dict[float, PreparedBacktest] = {}
    for threshold in threshold_values:
        if candidate_frames_by_threshold[threshold]:
            candidates = pl.concat(candidate_frames_by_threshold[threshold], rechunk=True)
        else:
            candidates = _empty_candidate_orders()
        prepared[threshold] = PreparedBacktest(
            signal_leg_rows=signal_leg_rows_by_threshold[threshold],
            signal_snapshots=signal_snapshots_by_threshold[threshold],
            scheduled_orders=_apply_concurrent_cap(candidates, config.max_concurrent_names),
        )
    return prepared


def prepare_backtest(input_path: str | Path, config: MidTierCompressionConfig | None = None) -> PreparedBacktest:
    if config is None:
        config = MidTierCompressionConfig()

    prepared = prepare_backtest_grid(input_path, (config.top2_yes_threshold,), config)
    return prepared[config.top2_yes_threshold]


def materialize_prepared_backtest(
    prepared: PreparedBacktest,
    config: MidTierCompressionConfig,
) -> BacktestArtifacts:
    orders = _attach_position_pnl(prepared.scheduled_orders, config)
    event_tail = _event_tail_rollup(orders)
    summary = _summary_from_tables(
        orders,
        event_tail,
        signal_leg_rows=prepared.signal_leg_rows,
        signal_snapshots=prepared.signal_snapshots,
        config=config,
    )
    return BacktestArtifacts(summary=summary, orders=orders, event_tail=event_tail)


def run_backtest(input_path: str | Path, config: MidTierCompressionConfig | None = None) -> BacktestArtifacts:
    if config is None:
        config = MidTierCompressionConfig()

    prepared = prepare_backtest(input_path, config)
    return materialize_prepared_backtest(prepared, config)


def main() -> int:
    args = parse_args()
    artifacts = run_backtest(
        args.input,
        MidTierCompressionConfig(
            quote_side=args.quote_side,
            timestamp_unit=args.timestamp_unit,
            resolution_timestamp_unit=args.resolution_timestamp_unit,
            top2_yes_threshold=args.top2_yes_threshold,
            midtier_yes_threshold=args.midtier_yes_threshold,
            max_leg_notional_usd=args.max_leg_notional_usd,
            max_concurrent_names=args.max_concurrent_names,
        ),
    )

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(artifacts.summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.orders_output is not None:
        _write_table(artifacts.orders, args.orders_output)
    if args.events_output is not None:
        _write_table(artifacts.event_tail, args.events_output)

    print(json.dumps(artifacts.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())