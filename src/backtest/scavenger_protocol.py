from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl


REQUIRED_COLUMNS = {
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
}
GROUP_COLUMNS = ["market_id", "token_id"]
DATETIME_DTYPE = pl.Datetime("us")
MICROS_PER_SECOND = 1_000_000.0
SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class ScavengerConfig:
    resolution_window_hours: int = 72
    signal_best_ask_min: float = 0.99
    signal_best_bid_max: float = 0.96
    maker_bid_price: float = 0.95


@dataclass(frozen=True, slots=True)
class ScavengerBacktestResult:
    orders: pl.DataFrame
    fills: pl.DataFrame
    summary: pl.DataFrame


def _source_to_lazy_frame(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
) -> pl.LazyFrame:
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, (str, Path)):
        return pl.scan_parquet(str(source))

    paths = [str(path) for path in source]
    if not paths:
        raise ValueError("Expected at least one Parquet path.")
    return pl.scan_parquet(paths)


def _datetime_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    base = pl.col(column)
    is_temporal = getattr(dtype, "is_temporal", lambda: False)
    is_integer = getattr(dtype, "is_integer", lambda: False)
    is_float = getattr(dtype, "is_float", lambda: False)
    string_dtype = getattr(pl, "String", pl.Utf8)

    if is_temporal():
        return base.cast(DATETIME_DTYPE)

    if is_integer():
        abs_value = base.abs()
        return (
            pl.when(base.is_null())
            .then(pl.lit(None, dtype=DATETIME_DTYPE))
            .when(abs_value >= 1_000_000_000_000_000)
            .then(pl.from_epoch(base.cast(pl.Int64), time_unit="us"))
            .when(abs_value >= 1_000_000_000_000)
            .then(pl.from_epoch(base.cast(pl.Int64), time_unit="ms"))
            .otherwise(pl.from_epoch(base.cast(pl.Int64), time_unit="s"))
            .cast(DATETIME_DTYPE)
        )

    if is_float():
        return (
            (base.cast(pl.Float64) * MICROS_PER_SECOND)
            .round(0)
            .cast(pl.Int64)
            .cast(DATETIME_DTYPE)
        )

    if dtype in {pl.Utf8, string_dtype}:
        return base.str.to_datetime(strict=False).cast(DATETIME_DTYPE)

    raise TypeError(f"Unsupported timestamp dtype for {column}: {dtype!r}")


def _normalize_frame(frame: pl.LazyFrame) -> pl.LazyFrame:
    schema = frame.collect_schema()
    missing = sorted(REQUIRED_COLUMNS.difference(schema.names()))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return (
        frame.select(
            _datetime_expr("timestamp", schema["timestamp"]).alias("timestamp"),
            pl.col("market_id").cast(pl.Utf8),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("token_id").cast(pl.Utf8),
            pl.col("best_bid").cast(pl.Float64),
            pl.col("best_ask").cast(pl.Float64),
            pl.col("bid_depth").cast(pl.Float64),
            pl.col("ask_depth").cast(pl.Float64),
            _datetime_expr("resolution_timestamp", schema["resolution_timestamp"]).alias(
                "resolution_timestamp"
            ),
            pl.col("final_resolution_value").cast(pl.Float64),
        )
        .sort([*GROUP_COLUMNS, "timestamp"])
    )


def _build_orders(frame: pl.LazyFrame, config: ScavengerConfig) -> pl.LazyFrame:
    signal_window_seconds = float(config.resolution_window_hours) * SECONDS_PER_HOUR
    maker_bid_price = float(config.maker_bid_price)
    time_to_resolution_seconds = (
        (pl.col("resolution_timestamp").cast(pl.Int64) - pl.col("timestamp").cast(pl.Int64))
        / MICROS_PER_SECOND
    )
    signal_expr = (
        (pl.col("best_ask") >= float(config.signal_best_ask_min))
        & (pl.col("best_bid") <= float(config.signal_best_bid_max))
    ).fill_null(False)
    fill_trigger_expr = (
        (pl.col("best_ask") <= maker_bid_price) | (pl.col("best_bid") < maker_bid_price)
    ).fill_null(False)

    annotated = (
        frame.with_columns(
            time_to_resolution_seconds.alias("time_to_resolution_seconds"),
        )
        .filter(
            (pl.col("time_to_resolution_seconds") >= 0.0)
            & (pl.col("time_to_resolution_seconds") <= signal_window_seconds)
        )
        .with_columns(
            signal_expr.alias("entry_signal"),
            fill_trigger_expr.alias("fill_trigger"),
        )
        .with_columns(
            pl.col("fill_trigger")
            .cast(pl.Int64)
            .cum_sum()
            .shift(1)
            .fill_null(0)
            .over(GROUP_COLUMNS)
            .alias("cycle_id")
        )
    )

    orders = (
        annotated.filter(pl.col("entry_signal"))
        .group_by([*GROUP_COLUMNS, "cycle_id"])
        .agg(
            pl.len().alias("signal_row_count"),
            pl.col("timestamp").min().alias("order_posted_at"),
            pl.col("event_id").sort_by("timestamp").first().alias("signal_event_id"),
            pl.col("best_bid").sort_by("timestamp").first().alias("signal_best_bid"),
            pl.col("best_ask").sort_by("timestamp").first().alias("signal_best_ask"),
            pl.col("bid_depth").sort_by("timestamp").first().alias("signal_bid_depth"),
            pl.col("ask_depth").sort_by("timestamp").first().alias("signal_ask_depth"),
            pl.col("resolution_timestamp")
            .sort_by("timestamp")
            .first()
            .alias("signal_resolution_timestamp"),
            pl.col("final_resolution_value")
            .sort_by("timestamp")
            .first()
            .alias("signal_final_resolution_value"),
        )
        .with_columns(pl.lit(maker_bid_price).alias("order_price"))
        .sort([*GROUP_COLUMNS, "order_posted_at"])
    )

    fills = (
        annotated.filter(pl.col("fill_trigger"))
        .select(
            *GROUP_COLUMNS,
            "cycle_id",
            pl.col("timestamp").alias("fill_timestamp"),
            pl.col("event_id").alias("fill_event_id"),
            pl.col("best_bid").alias("fill_best_bid"),
            pl.col("best_ask").alias("fill_best_ask"),
            pl.col("bid_depth").alias("fill_bid_depth"),
            pl.col("ask_depth").alias("fill_ask_depth"),
            pl.when(pl.col("best_ask") <= maker_bid_price)
            .then(pl.lit("best_ask_le_0p95"))
            .otherwise(pl.lit("best_bid_lt_0p95"))
            .alias("fill_reason"),
            pl.col("resolution_timestamp").alias("fill_resolution_timestamp"),
            pl.col("final_resolution_value").alias("fill_final_resolution_value"),
        )
        .sort([*GROUP_COLUMNS, "fill_timestamp"])
    )

    settlement_price = pl.lit(1.0)
    actual_final_value = pl.coalesce(
        [pl.col("fill_final_resolution_value"), pl.col("signal_final_resolution_value")]
    )
    fill_age_seconds = (
        (pl.col("fill_timestamp").cast(pl.Int64) - pl.col("order_posted_at").cast(pl.Int64))
        / MICROS_PER_SECOND
    )
    lockup_seconds = (
        (
            pl.coalesce(
                [pl.col("fill_resolution_timestamp"), pl.col("signal_resolution_timestamp")]
            ).cast(pl.Int64)
            - pl.col("fill_timestamp").cast(pl.Int64)
        )
        / MICROS_PER_SECOND
    )
    assumed_raw_pnl = settlement_price - pl.col("order_price")
    actual_raw_pnl = actual_final_value - pl.col("order_price")

    return (
        orders.join(fills, on=[*GROUP_COLUMNS, "cycle_id"], how="inner")
        .filter(pl.col("fill_timestamp") > pl.col("order_posted_at"))
        .with_columns(
            fill_age_seconds.alias("time_to_fill_seconds"),
            lockup_seconds.alias("lockup_seconds"),
            pl.coalesce(
                [pl.col("fill_resolution_timestamp"), pl.col("signal_resolution_timestamp")]
            ).alias("resolution_timestamp"),
            actual_final_value.alias("final_resolution_value"),
            (actual_final_value == 1.0).alias("settled_yes"),
            assumed_raw_pnl.alias("raw_pnl"),
            (assumed_raw_pnl / pl.col("order_price")).alias("raw_roi"),
            actual_raw_pnl.alias("actual_raw_pnl"),
            (actual_raw_pnl / pl.col("order_price")).alias("actual_raw_roi"),
        )
        .with_columns(
            (pl.col("lockup_seconds") / SECONDS_PER_HOUR).alias("lockup_hours"),
            (pl.col("lockup_seconds") / SECONDS_PER_DAY).alias("lockup_days"),
            pl.when(pl.col("lockup_seconds") > 0.0)
            .then(pl.col("raw_roi") * SECONDS_PER_YEAR / pl.col("lockup_seconds"))
            .otherwise(None)
            .alias("apr"),
            pl.when(pl.col("lockup_seconds") > 0.0)
            .then(pl.col("actual_raw_roi") * SECONDS_PER_YEAR / pl.col("lockup_seconds"))
            .otherwise(None)
            .alias("actual_apr"),
        )
        .select(
            *GROUP_COLUMNS,
            "cycle_id",
            "signal_event_id",
            "fill_event_id",
            "order_posted_at",
            "fill_timestamp",
            "resolution_timestamp",
            "signal_best_bid",
            "signal_best_ask",
            "signal_bid_depth",
            "signal_ask_depth",
            "fill_best_bid",
            "fill_best_ask",
            "fill_bid_depth",
            "fill_ask_depth",
            "fill_reason",
            "signal_row_count",
            "order_price",
            "time_to_fill_seconds",
            "lockup_seconds",
            "lockup_hours",
            "lockup_days",
            "final_resolution_value",
            "settled_yes",
            "raw_pnl",
            "raw_roi",
            "actual_raw_pnl",
            "actual_raw_roi",
            "apr",
            "actual_apr",
        )
        .sort([*GROUP_COLUMNS, "order_posted_at"])
    )


def _build_summary(orders: pl.LazyFrame, fills: pl.LazyFrame, config: ScavengerConfig) -> pl.LazyFrame:
    order_stats = orders.select(
        pl.len().alias("orders_posted"),
        pl.col("signal_row_count").sum().fill_null(0).alias("signal_rows"),
        pl.col("market_id").n_unique().alias("markets_signaled"),
        pl.col("token_id").n_unique().alias("tokens_signaled"),
    )
    fill_stats = fills.select(
        pl.len().alias("fills"),
        pl.col("market_id").n_unique().alias("markets_filled"),
        pl.col("token_id").n_unique().alias("tokens_filled"),
        pl.col("order_price").sum().fill_null(0.0).alias("capital_committed"),
        pl.col("raw_pnl").sum().fill_null(0.0).alias("gross_pnl"),
        pl.col("raw_roi").mean().alias("mean_raw_roi"),
        pl.col("raw_roi").median().alias("median_raw_roi"),
        pl.col("actual_raw_roi").mean().alias("mean_actual_raw_roi"),
        pl.col("apr").mean().alias("mean_apr"),
        pl.col("apr").median().alias("median_apr"),
        pl.col("actual_apr").mean().alias("mean_actual_apr"),
        pl.col("lockup_hours").mean().alias("avg_lockup_hours"),
        pl.col("time_to_fill_seconds").mean().alias("avg_time_to_fill_seconds"),
        pl.col("settled_yes").mean().alias("yes_resolution_rate"),
    )
    return (
        order_stats.join(fill_stats, how="cross")
        .with_columns(
            pl.when(pl.col("orders_posted") > 0)
            .then(pl.col("fills") / pl.col("orders_posted"))
            .otherwise(0.0)
            .alias("fill_rate"),
            pl.lit(float(config.maker_bid_price)).alias("maker_bid_price"),
            pl.lit(float(config.signal_best_ask_min)).alias("signal_best_ask_min"),
            pl.lit(float(config.signal_best_bid_max)).alias("signal_best_bid_max"),
            pl.lit(int(config.resolution_window_hours)).alias("resolution_window_hours"),
        )
        .select(
            "resolution_window_hours",
            "maker_bid_price",
            "signal_best_ask_min",
            "signal_best_bid_max",
            "signal_rows",
            "orders_posted",
            "fills",
            "fill_rate",
            "markets_signaled",
            "tokens_signaled",
            "markets_filled",
            "tokens_filled",
            "capital_committed",
            "gross_pnl",
            "mean_raw_roi",
            "median_raw_roi",
            "mean_actual_raw_roi",
            "mean_apr",
            "median_apr",
            "mean_actual_apr",
            "avg_lockup_hours",
            "avg_time_to_fill_seconds",
            "yes_resolution_rate",
        )
    )


def build_scavenger_backtest(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    normalized = _normalize_frame(_source_to_lazy_frame(source))
    orders = _build_orders(normalized, config)
    summary = _build_summary(orders, orders, config)
    return orders, orders, summary


def run_scavenger_backtest(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
) -> ScavengerBacktestResult:
    orders_lf, fills_lf, summary_lf = build_scavenger_backtest(source, config=config)
    orders_df, fills_df, summary_df = pl.collect_all([orders_lf, fills_lf, summary_lf])
    return ScavengerBacktestResult(orders=orders_df, fills=fills_df, summary=summary_df)