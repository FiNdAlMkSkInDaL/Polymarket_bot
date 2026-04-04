from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import polars as pl


TimestampUnit = Literal["auto", "s", "ms", "us", "ns"]

REQUIRED_COLUMNS = {
    "timestamp",
    "market_id",
    "event_id",
    "token_id",
    "best_bid",
    "best_ask",
    "bid_depth",
    "ask_depth",
}

SIGNAL_SCHEMA: dict[str, pl.DataType] = {
    "trade_id": pl.Int64,
    "signal_ts": pl.Int64,
    "signal_entry_gap": pl.Float64,
    "signal_entry_gap_z": pl.Float64,
    "implied_residual_probability": pl.Float64,
    "decision_fok_ok": pl.Boolean,
    "entry_condition": pl.Boolean,
    "exit_condition": pl.Boolean,
}

TRADE_SCHEMA: dict[str, pl.DataType] = {
    "trade_id": pl.Int64,
    "trade_state": pl.String,
    "reason": pl.String,
    "signal_ts": pl.Int64,
    "arrival_ts": pl.Int64,
    "entry_exec_ts": pl.Int64,
    "exit_ts": pl.Int64,
    "exit_type": pl.String,
    "orphan_leg": pl.String,
    "flatten_stage": pl.Int64,
    "order_size": pl.Float64,
    "signal_entry_gap": pl.Float64,
    "signal_entry_gap_z": pl.Float64,
    "implied_residual_probability": pl.Float64,
    "exit_gap": pl.Float64,
    "entry_price_a": pl.Float64,
    "entry_price_b": pl.Float64,
    "exit_price_a": pl.Float64,
    "exit_price_b": pl.Float64,
    "gross_pnl": pl.Float64,
    "fees": pl.Float64,
    "net_pnl": pl.Float64,
}

CHUNK_SCHEMA: dict[str, pl.DataType] = {
    "chunk_id": pl.Int64,
    "signal_window_start_ts": pl.Int64,
    "signal_window_end_ts": pl.Int64,
    "scan_window_start_ts": pl.Int64,
    "scan_window_end_ts": pl.Int64,
    "signals": pl.Int64,
    "pretrade_rejections": pl.Int64,
    "route_arrival_full_rejections": pl.Int64,
    "partial_fills": pl.Int64,
    "surviving_fok_baskets": pl.Int64,
}

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

DAY_MS = 86_400_000
EPSILON = 1e-9
FLATTEN_STAGE_COUNT = 2
DEFAULT_MINIMUM_THEORETICAL_EDGE_RATE = 0.02


def default_minimum_theoretical_edge_dollars(order_size: float) -> float:
    if order_size <= 0:
        raise ValueError("order_size must be strictly positive")
    return float(order_size * DEFAULT_MINIMUM_THEORETICAL_EDGE_RATE)


@dataclass(frozen=True, slots=True)
class MarketSlice:
    market_id: str
    token_id: str | None = "YES"


@dataclass(frozen=True, slots=True)
class ConditionalProbabilitySqueezeConfig:
    order_size: float = 100.0
    entry_gap_threshold: float = 0.03
    entry_zscore_threshold: float = 2.0
    minimum_edge_over_combined_spread_ratio: float = 0.03
    minimum_theoretical_edge_dollars: float = 0.0
    exit_gap_threshold: float = 0.05
    exit_zscore_threshold: float = 0.0
    z_window_events: int = 250
    timestamp_unit: TimestampUnit = "auto"
    route_latency_ms: int = 100
    max_quote_age_ms: int = 5_000
    max_hold_ms: int = 60_000
    process_by_day: bool = True
    chunk_days: int = 1
    warmup_days: int = 1
    chunk_lookahead_ms: int | None = None
    collect_engine: Literal["auto", "streaming"] = "streaming"
    taker_fee_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.order_size <= 0:
            raise ValueError("order_size must be strictly positive")
        if self.entry_gap_threshold < 0:
            raise ValueError("entry_gap_threshold must be non-negative")
        if self.entry_zscore_threshold < 0:
            raise ValueError("entry_zscore_threshold must be non-negative")
        if self.minimum_edge_over_combined_spread_ratio < 0:
            raise ValueError("minimum_edge_over_combined_spread_ratio must be non-negative")
        if self.minimum_theoretical_edge_dollars < 0:
            raise ValueError("minimum_theoretical_edge_dollars must be non-negative")
        if self.exit_gap_threshold < 0:
            raise ValueError("exit_gap_threshold must be non-negative")
        if self.z_window_events <= 1:
            raise ValueError("z_window_events must be greater than one")
        if self.route_latency_ms < 0:
            raise ValueError("route_latency_ms must be non-negative")
        if self.max_quote_age_ms <= 0:
            raise ValueError("max_quote_age_ms must be strictly positive")
        if self.max_hold_ms <= 0:
            raise ValueError("max_hold_ms must be strictly positive")
        if self.chunk_days <= 0:
            raise ValueError("chunk_days must be strictly positive")
        if self.warmup_days < 0:
            raise ValueError("warmup_days must be non-negative")
        if self.chunk_lookahead_ms is not None and self.chunk_lookahead_ms < 0:
            raise ValueError("chunk_lookahead_ms must be non-negative when provided")
        if self.collect_engine not in {"auto", "streaming"}:
            raise ValueError("collect_engine must be either 'auto' or 'streaming'")
        if self.taker_fee_bps < 0:
            raise ValueError("taker_fee_bps must be non-negative")

    @property
    def route_latency_us(self) -> int:
        return int(self.route_latency_ms * 1_000)

    @property
    def effective_chunk_lookahead_ms(self) -> int:
        if self.chunk_lookahead_ms is not None:
            return int(self.chunk_lookahead_ms)
        return int(
            self.max_hold_ms
            + self.route_latency_ms
            + self.max_quote_age_ms * (FLATTEN_STAGE_COUNT + 1)
        )


@dataclass(slots=True)
class ConditionalProbabilitySqueezeResult:
    signals: pl.DataFrame
    trades: pl.DataFrame
    summary: dict[str, Any]
    aligned: pl.DataFrame | None = None
    chunk_stats: pl.DataFrame | None = None


def align_nested_market_books(
    market_a: pl.DataFrame | pl.LazyFrame,
    market_b: pl.DataFrame | pl.LazyFrame,
    *,
    config: ConditionalProbabilitySqueezeConfig | None = None,
) -> pl.LazyFrame:
    cfg = config or ConditionalProbabilitySqueezeConfig(process_by_day=False)
    prepared_a = _prepare_market_frame(market_a, timestamp_unit=cfg.timestamp_unit)
    prepared_b = _prepare_market_frame(market_b, timestamp_unit=cfg.timestamp_unit)
    return _build_aligned_frame(prepared_a, prepared_b, cfg)


def run_conditional_probability_squeeze_backtest(
    source: str | Path | Sequence[str | Path],
    *,
    market_a: MarketSlice,
    market_b: MarketSlice,
    config: ConditionalProbabilitySqueezeConfig | None = None,
    return_aligned: bool = False,
) -> ConditionalProbabilitySqueezeResult:
    cfg = config or ConditionalProbabilitySqueezeConfig()
    raw = _scan_parquet_source(source)
    prepared_a = _prepare_market_frame(
        _filter_market_slice(raw, market_a),
        timestamp_unit=cfg.timestamp_unit,
    )
    prepared_b = _prepare_market_frame(
        _filter_market_slice(raw, market_b),
        timestamp_unit=cfg.timestamp_unit,
    )
    if return_aligned and cfg.process_by_day:
        raise ValueError("return_aligned is not supported when process_by_day=True")
    if cfg.process_by_day:
        return _run_chunked_backtest(prepared_a, prepared_b, cfg)
    return _run_backtest_from_prepared_frames(prepared_a, prepared_b, cfg, return_aligned=return_aligned)


def run_conditional_probability_squeeze_backtest_from_frames(
    market_a: pl.DataFrame | pl.LazyFrame,
    market_b: pl.DataFrame | pl.LazyFrame,
    *,
    config: ConditionalProbabilitySqueezeConfig | None = None,
    return_aligned: bool = False,
) -> ConditionalProbabilitySqueezeResult:
    cfg = config or ConditionalProbabilitySqueezeConfig()
    prepared_a = _prepare_market_frame(market_a, timestamp_unit=cfg.timestamp_unit)
    prepared_b = _prepare_market_frame(market_b, timestamp_unit=cfg.timestamp_unit)
    if return_aligned and cfg.process_by_day:
        raise ValueError("return_aligned is not supported when process_by_day=True")
    if cfg.process_by_day:
        return _run_chunked_backtest(prepared_a, prepared_b, cfg)
    return _run_backtest_from_prepared_frames(prepared_a, prepared_b, cfg, return_aligned=return_aligned)


def _run_backtest_from_prepared_frames(
    prepared_a: pl.LazyFrame,
    prepared_b: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
    *,
    return_aligned: bool,
) -> ConditionalProbabilitySqueezeResult:
    aligned = _build_aligned_frame(prepared_a, prepared_b, config)
    surface = _build_execution_surface(aligned, config)
    signal_rows = surface.filter(pl.col("entry_signal")).with_row_index("trade_id")
    signal_rows = signal_rows.with_columns(pl.col("trade_id").cast(pl.Int64))

    pretrade_rejects = (
        signal_rows
        .filter(~pl.col("decision_fok_ok"))
        .select(
            [
                "trade_id",
                pl.lit("rejected_pretrade_depth").alias("trade_state"),
                pl.lit("decision_depth").alias("reason"),
                pl.col("timestamp").alias("signal_ts"),
                pl.lit(None, dtype=pl.Int64).alias("arrival_ts"),
                pl.lit(None, dtype=pl.Int64).alias("entry_exec_ts"),
                pl.lit(None, dtype=pl.Int64).alias("exit_ts"),
                pl.lit(None, dtype=pl.String).alias("exit_type"),
                pl.lit(None, dtype=pl.String).alias("orphan_leg"),
                pl.lit(None, dtype=pl.Int64).alias("flatten_stage"),
                pl.lit(config.order_size).alias("order_size"),
                pl.col("entry_gap").alias("signal_entry_gap"),
                pl.col("entry_gap_z").alias("signal_entry_gap_z"),
                "implied_residual_probability",
                pl.lit(None, dtype=pl.Float64).alias("exit_gap"),
                pl.lit(None, dtype=pl.Float64).alias("entry_price_a"),
                pl.lit(None, dtype=pl.Float64).alias("entry_price_b"),
                pl.lit(None, dtype=pl.Float64).alias("exit_price_a"),
                pl.lit(None, dtype=pl.Float64).alias("exit_price_b"),
                pl.lit(None, dtype=pl.Float64).alias("gross_pnl"),
                pl.lit(None, dtype=pl.Float64).alias("fees"),
                pl.lit(None, dtype=pl.Float64).alias("net_pnl"),
            ]
        )
    )

    entry_outcomes = _simulate_entry_outcomes(signal_rows, prepared_a, prepared_b, config)

    arrival_expired = (
        entry_outcomes
        .filter(pl.col("entry_state") == "expired_before_fill")
        .select(
            [
                "trade_id",
                pl.col("entry_state").alias("trade_state"),
                pl.lit("arrival_depth_evaporated").alias("reason"),
                pl.col("timestamp").alias("signal_ts"),
                "arrival_ts",
                pl.lit(None, dtype=pl.Int64).alias("entry_exec_ts"),
                pl.lit(None, dtype=pl.Int64).alias("exit_ts"),
                pl.lit(None, dtype=pl.String).alias("exit_type"),
                pl.lit(None, dtype=pl.String).alias("orphan_leg"),
                pl.lit(None, dtype=pl.Int64).alias("flatten_stage"),
                pl.lit(config.order_size).alias("order_size"),
                pl.col("entry_gap").alias("signal_entry_gap"),
                pl.col("entry_gap_z").alias("signal_entry_gap_z"),
                "implied_residual_probability",
                pl.lit(None, dtype=pl.Float64).alias("exit_gap"),
                pl.lit(None, dtype=pl.Float64).alias("entry_price_a"),
                pl.lit(None, dtype=pl.Float64).alias("entry_price_b"),
                pl.lit(None, dtype=pl.Float64).alias("exit_price_a"),
                pl.lit(None, dtype=pl.Float64).alias("exit_price_b"),
                pl.lit(None, dtype=pl.Float64).alias("gross_pnl"),
                pl.lit(None, dtype=pl.Float64).alias("fees"),
                pl.lit(None, dtype=pl.Float64).alias("net_pnl"),
            ]
        )
    )

    trade_frames = [
        pretrade_rejects,
        arrival_expired,
        _simulate_basket_closures(entry_outcomes, surface, config),
        _simulate_long_flatten(entry_outcomes.filter(pl.col("entry_state") == "partial_a"), prepared_a, config),
        _simulate_short_flatten(entry_outcomes.filter(pl.col("entry_state") == "partial_b"), prepared_b, config),
    ]
    trades = pl.concat(
        [_ensure_trade_schema(_collect_lazy(frame, config.collect_engine)) for frame in trade_frames],
        how="vertical_relaxed",
    ).sort("trade_id")

    signals = _ensure_signal_schema(
        _collect_lazy(
            signal_rows.select(
                [
                    "trade_id",
                    pl.col("timestamp").alias("signal_ts"),
                    pl.col("entry_gap").alias("signal_entry_gap"),
                    pl.col("entry_gap_z").alias("signal_entry_gap_z"),
                    "implied_residual_probability",
                    "decision_fok_ok",
                    "entry_condition",
                    "exit_condition",
                ]
            ),
            config.collect_engine,
        )
    ).sort("trade_id")

    aligned_df = _collect_lazy(aligned, config.collect_engine) if return_aligned else None
    chunk_stats = _empty_chunk_stats_frame()
    summary = _summarize_trades(trades, signals, config, chunk_stats=chunk_stats)
    return ConditionalProbabilitySqueezeResult(
        signals=signals,
        trades=trades,
        summary=summary,
        aligned=aligned_df,
        chunk_stats=chunk_stats,
    )


def _run_chunked_backtest(
    prepared_a: pl.LazyFrame,
    prepared_b: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> ConditionalProbabilitySqueezeResult:
    chunk_start_days = _chunk_start_days(prepared_a, prepared_b, config)
    if not chunk_start_days:
        empty_signals = _empty_signal_frame()
        empty_trades = _empty_trade_frame()
        empty_chunk_stats = _empty_chunk_stats_frame()
        return ConditionalProbabilitySqueezeResult(
            signals=empty_signals,
            trades=empty_trades,
            summary=_summarize_trades(empty_trades, empty_signals, config, chunk_stats=empty_chunk_stats),
            aligned=None,
            chunk_stats=empty_chunk_stats,
        )

    signal_frames: list[pl.DataFrame] = []
    trade_frames: list[pl.DataFrame] = []
    chunk_rows: list[dict[str, int]] = []
    trade_id_offset = 0

    warmup_ms = config.warmup_days * DAY_MS
    lookahead_ms = config.effective_chunk_lookahead_ms

    for chunk_id, chunk_start_day in enumerate(chunk_start_days, start=1):
        signal_window_start_ms = chunk_start_day * DAY_MS
        signal_window_end_ms = signal_window_start_ms + config.chunk_days * DAY_MS
        scan_window_start_ms = signal_window_start_ms - warmup_ms
        scan_window_end_ms = signal_window_end_ms + lookahead_ms

        chunk_result = _run_backtest_from_prepared_frames(
            _slice_prepared_frame(prepared_a, scan_window_start_ms, scan_window_end_ms),
            _slice_prepared_frame(prepared_b, scan_window_start_ms, scan_window_end_ms),
            config,
            return_aligned=False,
        )
        chunk_signals = chunk_result.signals.filter(
            (pl.col("signal_ts") >= signal_window_start_ms)
            & (pl.col("signal_ts") < signal_window_end_ms)
        )
        chunk_trades = chunk_result.trades.filter(
            (pl.col("signal_ts") >= signal_window_start_ms)
            & (pl.col("signal_ts") < signal_window_end_ms)
        )

        if chunk_signals.height != chunk_trades.height:
            raise ValueError("signal/trade row count mismatch while chunking conditional squeeze backtest")

        if chunk_signals.height:
            chunk_signals = chunk_signals.with_columns(
                (pl.col("trade_id") + trade_id_offset).cast(pl.Int64).alias("trade_id")
            )
            chunk_trades = chunk_trades.with_columns(
                (pl.col("trade_id") + trade_id_offset).cast(pl.Int64).alias("trade_id")
            )
            trade_id_offset += chunk_signals.height
            signal_frames.append(chunk_signals)
            trade_frames.append(chunk_trades)

        chunk_rows.append(
            _chunk_diagnostic_row(
                chunk_id,
                signal_window_start_ms,
                signal_window_end_ms,
                scan_window_start_ms,
                scan_window_end_ms,
                chunk_trades,
            )
        )

    signals = pl.concat(signal_frames, how="vertical_relaxed") if signal_frames else _empty_signal_frame()
    trades = pl.concat(trade_frames, how="vertical_relaxed") if trade_frames else _empty_trade_frame()
    chunk_stats = pl.DataFrame(chunk_rows, schema=CHUNK_SCHEMA).sort("chunk_id")
    summary = _summarize_trades(trades, signals, config, chunk_stats=chunk_stats)
    return ConditionalProbabilitySqueezeResult(
        signals=_ensure_signal_schema(signals).sort("trade_id"),
        trades=_ensure_trade_schema(trades).sort("trade_id"),
        summary=summary,
        aligned=None,
        chunk_stats=chunk_stats,
    )


def _scan_parquet_source(source: str | Path | Sequence[str | Path]) -> pl.LazyFrame:
    if isinstance(source, (str, Path)):
        return pl.scan_parquet(str(source))
    paths = [str(path) for path in source]
    if not paths:
        raise ValueError("source must contain at least one parquet path")
    return pl.scan_parquet(paths)


def _filter_market_slice(frame: pl.LazyFrame, market: MarketSlice) -> pl.LazyFrame:
    filtered = frame.filter(pl.col("market_id") == market.market_id)
    token_id = market.token_id
    if token_id is None:
        token_id = "YES"
    if token_id is not None:
        filtered = filtered.filter(pl.col("token_id") == token_id)
    return filtered


def _prepare_market_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    timestamp_unit: TimestampUnit,
) -> pl.LazyFrame:
    lazy_frame = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema = lazy_frame.collect_schema()
    missing = REQUIRED_COLUMNS.difference(set(schema.names()))
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"market frame is missing required columns: {missing_str}")

    return (
        lazy_frame
        .select(
            [
                _timestamp_ms_expr("timestamp", schema["timestamp"], timestamp_unit).alias("timestamp"),
                pl.col("market_id").cast(pl.String),
                pl.col("event_id").cast(pl.String),
                pl.col("token_id").cast(pl.String),
                pl.col("best_bid").cast(pl.Float64),
                pl.col("best_ask").cast(pl.Float64),
                pl.col("bid_depth").cast(pl.Float64),
                pl.col("ask_depth").cast(pl.Float64),
            ]
        )
        .filter(pl.col("timestamp").is_not_null())
        .with_columns(pl.col("timestamp").alias("quote_ts"))
        .sort("timestamp")
    )


def _build_aligned_frame(
    market_a: pl.LazyFrame,
    market_b: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> pl.LazyFrame:
    clock = (
        pl.concat(
            [
                market_a.select("timestamp"),
                market_b.select("timestamp"),
            ],
            how="vertical",
        )
        .unique()
        .sort("timestamp")
    )

    aligned_a = _rename_market_frame(market_a, prefix="a", join_key_name="timestamp")
    aligned_b = _rename_market_frame(market_b, prefix="b", join_key_name="timestamp")

    return (
        clock
        .join_asof(aligned_a, on="timestamp", strategy="backward")
        .join_asof(aligned_b, on="timestamp", strategy="backward")
        .filter(pl.col("quote_ts_a").is_not_null() & pl.col("quote_ts_b").is_not_null())
        .with_columns(
            [
                (pl.col("timestamp") - pl.col("quote_ts_a")).alias("quote_age_ms_a"),
                (pl.col("timestamp") - pl.col("quote_ts_b")).alias("quote_age_ms_b"),
            ]
        )
        .filter(
            (pl.col("quote_age_ms_a") <= config.max_quote_age_ms)
            & (pl.col("quote_age_ms_b") <= config.max_quote_age_ms)
        )
        .filter(
            (pl.col("best_bid_a") > 0)
            & (pl.col("best_ask_a") > 0)
            & (pl.col("best_bid_a") <= pl.col("best_ask_a"))
            & (pl.col("best_bid_b") > 0)
            & (pl.col("best_ask_b") > 0)
            & (pl.col("best_bid_b") <= pl.col("best_ask_b"))
        )
        .with_columns(
            [
                (pl.col("best_ask_a") - pl.col("best_bid_b")).alias("entry_gap"),
                (pl.col("best_bid_a") - pl.col("best_ask_b")).alias("exit_gap"),
                (pl.col("best_bid_b") - pl.col("best_ask_a")).alias("entry_edge"),
                ((pl.col("best_ask_a") - pl.col("best_bid_a")) + (pl.col("best_ask_b") - pl.col("best_bid_b")))
                .alias("combined_spread"),
                pl.when((1.0 - pl.col("best_bid_b")) > EPSILON)
                .then((pl.col("best_ask_a") - pl.col("best_bid_b")) / (1.0 - pl.col("best_bid_b")))
                .otherwise(None)
                .alias("implied_residual_probability"),
            ]
        )
        .with_columns(
            [
                pl.col("entry_gap").rolling_mean(window_size=config.z_window_events).alias("entry_gap_mean"),
                pl.col("entry_gap").rolling_std(window_size=config.z_window_events).alias("entry_gap_std"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("entry_gap_std") > EPSILON)
                .then((pl.col("entry_gap") - pl.col("entry_gap_mean")) / pl.col("entry_gap_std"))
                .otherwise(None)
                .alias("entry_gap_z")
            ]
        )
    )


def _build_execution_surface(
    aligned: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> pl.LazyFrame:
    order_size = config.order_size
    signal_theoretical_edge_dollars = pl.col("entry_edge") * order_size
    entry_condition_base = (
        (pl.col("entry_gap") <= config.entry_gap_threshold)
        | (pl.col("entry_gap_z") <= -config.entry_zscore_threshold).fill_null(False)
    )
    if config.minimum_edge_over_combined_spread_ratio <= 0:
        minimum_edge_condition = pl.lit(True)
    else:
        minimum_edge_condition = pl.col("entry_edge") >= (
            pl.col("combined_spread") * config.minimum_edge_over_combined_spread_ratio
        )
    if config.minimum_theoretical_edge_dollars <= 0:
        minimum_theoretical_edge_condition = pl.lit(True)
    else:
        minimum_theoretical_edge_condition = (
            signal_theoretical_edge_dollars >= config.minimum_theoretical_edge_dollars
        )
    entry_condition = entry_condition_base & minimum_edge_condition & minimum_theoretical_edge_condition
    exit_condition = (
        (pl.col("exit_gap") >= config.exit_gap_threshold)
        | (pl.col("entry_gap_z") >= config.exit_zscore_threshold).fill_null(False)
    )

    return (
        aligned
        .with_columns(
            [
                entry_condition_base.alias("entry_condition_base"),
                minimum_edge_condition.alias("minimum_entry_edge_ok"),
                signal_theoretical_edge_dollars.alias("signal_theoretical_edge_dollars"),
                minimum_theoretical_edge_condition.alias("minimum_theoretical_edge_ok"),
                entry_condition.alias("entry_condition"),
                exit_condition.alias("exit_condition"),
                (pl.col("best_ask_a") * order_size).alias("entry_depth_needed_a"),
                (pl.col("best_bid_b") * order_size).alias("entry_depth_needed_b"),
                (pl.col("best_bid_a") * order_size).alias("exit_depth_needed_a"),
                (pl.col("best_ask_b") * order_size).alias("exit_depth_needed_b"),
            ]
        )
        .with_columns(
            [
                (pl.col("entry_depth_needed_a") <= pl.col("ask_depth_a")).alias("decision_leg_a_ok"),
                (pl.col("entry_depth_needed_b") <= pl.col("bid_depth_b")).alias("decision_leg_b_ok"),
                (pl.col("exit_depth_needed_a") <= pl.col("bid_depth_a")).alias("exit_leg_a_ok"),
                (pl.col("exit_depth_needed_b") <= pl.col("ask_depth_b")).alias("exit_leg_b_ok"),
            ]
        )
        .with_columns(
            [
                (pl.col("decision_leg_a_ok") & pl.col("decision_leg_b_ok")).alias("decision_fok_ok"),
                (pl.col("exit_leg_a_ok") & pl.col("exit_leg_b_ok")).alias("exit_fok_ok"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("entry_condition")
                    & (
                        (~pl.col("entry_condition").shift(1).fill_null(False))
                        | (
                            (~pl.col("decision_fok_ok").shift(1).fill_null(False))
                            & pl.col("decision_fok_ok")
                        )
                    )
                ).alias("entry_signal")
            ]
        )
    )


def _simulate_entry_outcomes(
    signal_rows: pl.LazyFrame,
    market_a: pl.LazyFrame,
    market_b: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> pl.LazyFrame:
    order_size = config.order_size
    fill_a = _rename_market_frame(market_a, prefix="a_fill", join_key_name="timestamp_a_fill")
    fill_b = _rename_market_frame(market_b, prefix="b_fill", join_key_name="timestamp_b_fill")

    return (
        signal_rows
        .filter(pl.col("decision_fok_ok"))
        .with_columns((pl.col("timestamp") + config.route_latency_ms).alias("arrival_ts"))
        .join_asof(fill_a, left_on="arrival_ts", right_on="timestamp_a_fill", strategy="forward")
        .join_asof(fill_b, left_on="arrival_ts", right_on="timestamp_b_fill", strategy="forward")
        .with_columns(
            [
                (
                    (pl.col("best_ask_a_fill") > 0)
                    & ((pl.col("best_ask_a_fill") * order_size) <= pl.col("ask_depth_a_fill"))
                ).fill_null(False).alias("a_fill_ok"),
                (
                    (pl.col("best_bid_b_fill") > 0)
                    & ((pl.col("best_bid_b_fill") * order_size) <= pl.col("bid_depth_b_fill"))
                ).fill_null(False).alias("b_fill_ok"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("a_fill_ok") & pl.col("b_fill_ok"))
                .then(pl.lit("basket_filled"))
                .when(pl.col("a_fill_ok"))
                .then(pl.lit("partial_a"))
                .when(pl.col("b_fill_ok"))
                .then(pl.lit("partial_b"))
                .otherwise(pl.lit("expired_before_fill"))
                .alias("entry_state"),
                pl.max_horizontal("timestamp_a_fill", "timestamp_b_fill").alias("entry_exec_ts"),
            ]
        )
    )


def _simulate_basket_closures(
    entry_outcomes: pl.LazyFrame,
    surface: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> pl.LazyFrame:
    order_size = config.order_size
    fee_rate = config.taker_fee_bps / 10_000.0

    signal_exit_candidates = (
        surface
        .filter(pl.col("exit_condition") & pl.col("exit_fok_ok"))
        .select(
            [
                pl.col("timestamp").alias("signal_exit_ts"),
                pl.col("best_bid_a").alias("signal_exit_price_a"),
                pl.col("best_ask_b").alias("signal_exit_price_b"),
                pl.col("exit_gap").alias("signal_exit_gap"),
            ]
        )
    )
    time_stop_candidates = (
        surface
        .filter(pl.col("exit_fok_ok"))
        .select(
            [
                pl.col("timestamp").alias("time_stop_exit_ts"),
                pl.col("best_bid_a").alias("time_stop_exit_price_a"),
                pl.col("best_ask_b").alias("time_stop_exit_price_b"),
                pl.col("exit_gap").alias("time_stop_exit_gap"),
            ]
        )
    )

    basket_entries = (
        entry_outcomes
        .filter(pl.col("entry_state") == "basket_filled")
        .with_columns(
            [
                pl.col("entry_exec_ts").alias("entry_exec_ts"),
                (pl.col("entry_exec_ts") + 1).alias("signal_exit_lookup_ts"),
                (pl.col("entry_exec_ts") + config.max_hold_ms).alias("time_stop_lookup_ts"),
            ]
        )
    )

    signal_joined = basket_entries.join_asof(
        signal_exit_candidates,
        left_on="signal_exit_lookup_ts",
        right_on="signal_exit_ts",
        strategy="forward",
    )
    time_joined = (
        basket_entries
        .join_asof(
            time_stop_candidates,
            left_on="time_stop_lookup_ts",
            right_on="time_stop_exit_ts",
            strategy="forward",
        )
        .select(
            [
                "trade_id",
                "time_stop_exit_ts",
                "time_stop_exit_price_a",
                "time_stop_exit_price_b",
                "time_stop_exit_gap",
            ]
        )
    )

    return (
        signal_joined
        .join(time_joined, on="trade_id", how="left")
        .with_columns(
            [
                pl.when(
                    pl.col("signal_exit_ts").is_not_null()
                    & (pl.col("signal_exit_ts") <= pl.col("time_stop_lookup_ts"))
                )
                .then(pl.col("signal_exit_ts"))
                .otherwise(None)
                .alias("valid_signal_exit_ts")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("valid_signal_exit_ts").is_not_null())
                .then(pl.lit("signal"))
                .when(pl.col("time_stop_exit_ts").is_not_null())
                .then(pl.lit("time_stop"))
                .otherwise(pl.lit("open"))
                .alias("exit_type"),
                pl.when(pl.col("valid_signal_exit_ts").is_not_null())
                .then(pl.col("valid_signal_exit_ts"))
                .when(pl.col("time_stop_exit_ts").is_not_null())
                .then(pl.col("time_stop_exit_ts"))
                .otherwise(None)
                .alias("exit_ts"),
                pl.when(pl.col("valid_signal_exit_ts").is_not_null())
                .then(pl.col("signal_exit_price_a"))
                .when(pl.col("time_stop_exit_ts").is_not_null())
                .then(pl.col("time_stop_exit_price_a"))
                .otherwise(None)
                .alias("exit_price_a"),
                pl.when(pl.col("valid_signal_exit_ts").is_not_null())
                .then(pl.col("signal_exit_price_b"))
                .when(pl.col("time_stop_exit_ts").is_not_null())
                .then(pl.col("time_stop_exit_price_b"))
                .otherwise(None)
                .alias("exit_price_b"),
                pl.when(pl.col("valid_signal_exit_ts").is_not_null())
                .then(pl.col("signal_exit_gap"))
                .when(pl.col("time_stop_exit_ts").is_not_null())
                .then(pl.col("time_stop_exit_gap"))
                .otherwise(None)
                .alias("exit_gap"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("exit_ts").is_not_null())
                .then(pl.lit("basket_closed"))
                .otherwise(pl.lit("basket_open"))
                .alias("trade_state"),
                pl.when(pl.col("exit_type") == "open")
                .then(pl.lit("no_liquid_exit"))
                .otherwise(pl.col("exit_type"))
                .alias("reason"),
                pl.col("timestamp").alias("signal_ts"),
                pl.col("best_ask_a_fill").alias("entry_price_a"),
                pl.col("best_bid_b_fill").alias("entry_price_b"),
                pl.lit(config.order_size).alias("order_size"),
                pl.col("entry_gap").alias("signal_entry_gap"),
                pl.col("entry_gap_z").alias("signal_entry_gap_z"),
                pl.lit(None, dtype=pl.String).alias("orphan_leg"),
                pl.lit(None, dtype=pl.Int64).alias("flatten_stage"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("exit_ts").is_not_null())
                .then(
                    (pl.col("exit_price_a") - pl.col("entry_price_a")) * order_size
                    + (pl.col("entry_price_b") - pl.col("exit_price_b")) * order_size
                )
                .otherwise(None)
                .alias("gross_pnl")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("exit_ts").is_not_null())
                .then(
                    (
                        pl.col("entry_price_a")
                        + pl.col("entry_price_b")
                        + pl.col("exit_price_a")
                        + pl.col("exit_price_b")
                    )
                    * order_size
                    * fee_rate
                )
                .otherwise(None)
                .alias("fees")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("gross_pnl").is_not_null())
                .then(pl.col("gross_pnl") - pl.col("fees"))
                .otherwise(None)
                .alias("net_pnl")
            ]
        )
        .select(list(TRADE_SCHEMA))
    )


def _simulate_long_flatten(
    partial_rows: pl.LazyFrame,
    market_frame: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> pl.LazyFrame:
    order_size = config.order_size
    fee_rate = config.taker_fee_bps / 10_000.0
    stage1 = _rename_market_frame(market_frame, prefix="a_stage1", join_key_name="timestamp_a_stage1")
    stage2 = _rename_market_frame(market_frame, prefix="a_stage2", join_key_name="timestamp_a_stage2")

    return (
        partial_rows
        .with_columns(
            [
                pl.col("best_ask_a_fill").alias("entry_price_a"),
                pl.col("timestamp_a_fill").alias("entry_exec_ts"),
                (pl.col("timestamp_a_fill") + 1).alias("stage1_lookup_ts"),
            ]
        )
        .join_asof(stage1, left_on="stage1_lookup_ts", right_on="timestamp_a_stage1", strategy="forward")
        .with_columns(
            [
                (
                    (pl.col("best_bid_a_stage1") > 0)
                    & ((pl.col("best_bid_a_stage1") * order_size) <= pl.col("bid_depth_a_stage1"))
                ).fill_null(False).alias("stage1_ok"),
                pl.when(pl.col("timestamp_a_stage1").is_not_null())
                .then(pl.col("timestamp_a_stage1") + 1)
                .otherwise(None)
                .alias("stage2_lookup_ts"),
            ]
        )
        .join_asof(stage2, left_on="stage2_lookup_ts", right_on="timestamp_a_stage2", strategy="forward")
        .with_columns(
            [
                (
                    (pl.col("best_bid_a_stage2") > 0)
                    & ((pl.col("best_bid_a_stage2") * order_size) <= pl.col("bid_depth_a_stage2"))
                ).fill_null(False).alias("stage2_ok")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("stage1_ok"))
                .then(pl.lit("flattened_stage1"))
                .when(pl.col("stage2_ok"))
                .then(pl.lit("flattened_stage2"))
                .otherwise(pl.lit("partial_unresolved"))
                .alias("trade_state"),
                pl.when(pl.col("stage1_ok"))
                .then(pl.lit(1))
                .when(pl.col("stage2_ok"))
                .then(pl.lit(2))
                .otherwise(None)
                .alias("flatten_stage"),
                pl.when(pl.col("stage1_ok"))
                .then(pl.col("timestamp_a_stage1"))
                .when(pl.col("stage2_ok"))
                .then(pl.col("timestamp_a_stage2"))
                .otherwise(None)
                .alias("exit_ts"),
                pl.when(pl.col("stage1_ok"))
                .then(pl.col("best_bid_a_stage1"))
                .when(pl.col("stage2_ok"))
                .then(pl.col("best_bid_a_stage2"))
                .otherwise(None)
                .alias("exit_price_a"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("flatten_stage").is_not_null())
                .then(pl.lit("flatten"))
                .otherwise(pl.lit("no_liquid_flatten"))
                .alias("reason"),
                pl.col("timestamp").alias("signal_ts"),
                pl.col("entry_gap").alias("signal_entry_gap"),
                pl.col("entry_gap_z").alias("signal_entry_gap_z"),
                pl.lit(config.order_size).alias("order_size"),
                pl.lit("a").alias("orphan_leg"),
                pl.lit("flatten").alias("exit_type"),
                pl.lit(None, dtype=pl.Float64).alias("entry_price_b"),
                pl.lit(None, dtype=pl.Float64).alias("exit_price_b"),
                pl.lit(None, dtype=pl.Float64).alias("exit_gap"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("exit_ts").is_not_null())
                .then((pl.col("exit_price_a") - pl.col("entry_price_a")) * order_size)
                .otherwise(None)
                .alias("gross_pnl")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("exit_ts").is_not_null())
                .then((pl.col("entry_price_a") + pl.col("exit_price_a")) * order_size * fee_rate)
                .otherwise(None)
                .alias("fees")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("gross_pnl").is_not_null())
                .then(pl.col("gross_pnl") - pl.col("fees"))
                .otherwise(None)
                .alias("net_pnl")
            ]
        )
        .select(list(TRADE_SCHEMA))
    )


def _simulate_short_flatten(
    partial_rows: pl.LazyFrame,
    market_frame: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> pl.LazyFrame:
    order_size = config.order_size
    fee_rate = config.taker_fee_bps / 10_000.0
    stage1 = _rename_market_frame(market_frame, prefix="b_stage1", join_key_name="timestamp_b_stage1")
    stage2 = _rename_market_frame(market_frame, prefix="b_stage2", join_key_name="timestamp_b_stage2")

    return (
        partial_rows
        .with_columns(
            [
                pl.col("best_bid_b_fill").alias("entry_price_b"),
                pl.col("timestamp_b_fill").alias("entry_exec_ts"),
                (pl.col("timestamp_b_fill") + 1).alias("stage1_lookup_ts"),
            ]
        )
        .join_asof(stage1, left_on="stage1_lookup_ts", right_on="timestamp_b_stage1", strategy="forward")
        .with_columns(
            [
                (
                    (pl.col("best_ask_b_stage1") > 0)
                    & ((pl.col("best_ask_b_stage1") * order_size) <= pl.col("ask_depth_b_stage1"))
                ).fill_null(False).alias("stage1_ok"),
                pl.when(pl.col("timestamp_b_stage1").is_not_null())
                .then(pl.col("timestamp_b_stage1") + 1)
                .otherwise(None)
                .alias("stage2_lookup_ts"),
            ]
        )
        .join_asof(stage2, left_on="stage2_lookup_ts", right_on="timestamp_b_stage2", strategy="forward")
        .with_columns(
            [
                (
                    (pl.col("best_ask_b_stage2") > 0)
                    & ((pl.col("best_ask_b_stage2") * order_size) <= pl.col("ask_depth_b_stage2"))
                ).fill_null(False).alias("stage2_ok")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("stage1_ok"))
                .then(pl.lit("flattened_stage1"))
                .when(pl.col("stage2_ok"))
                .then(pl.lit("flattened_stage2"))
                .otherwise(pl.lit("partial_unresolved"))
                .alias("trade_state"),
                pl.when(pl.col("stage1_ok"))
                .then(pl.lit(1))
                .when(pl.col("stage2_ok"))
                .then(pl.lit(2))
                .otherwise(None)
                .alias("flatten_stage"),
                pl.when(pl.col("stage1_ok"))
                .then(pl.col("timestamp_b_stage1"))
                .when(pl.col("stage2_ok"))
                .then(pl.col("timestamp_b_stage2"))
                .otherwise(None)
                .alias("exit_ts"),
                pl.when(pl.col("stage1_ok"))
                .then(pl.col("best_ask_b_stage1"))
                .when(pl.col("stage2_ok"))
                .then(pl.col("best_ask_b_stage2"))
                .otherwise(None)
                .alias("exit_price_b"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("flatten_stage").is_not_null())
                .then(pl.lit("flatten"))
                .otherwise(pl.lit("no_liquid_flatten"))
                .alias("reason"),
                pl.col("timestamp").alias("signal_ts"),
                pl.col("entry_gap").alias("signal_entry_gap"),
                pl.col("entry_gap_z").alias("signal_entry_gap_z"),
                pl.lit(config.order_size).alias("order_size"),
                pl.lit("b").alias("orphan_leg"),
                pl.lit("flatten").alias("exit_type"),
                pl.lit(None, dtype=pl.Float64).alias("entry_price_a"),
                pl.lit(None, dtype=pl.Float64).alias("exit_price_a"),
                pl.lit(None, dtype=pl.Float64).alias("exit_gap"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("exit_ts").is_not_null())
                .then((pl.col("entry_price_b") - pl.col("exit_price_b")) * order_size)
                .otherwise(None)
                .alias("gross_pnl")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("exit_ts").is_not_null())
                .then((pl.col("entry_price_b") + pl.col("exit_price_b")) * order_size * fee_rate)
                .otherwise(None)
                .alias("fees")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("gross_pnl").is_not_null())
                .then(pl.col("gross_pnl") - pl.col("fees"))
                .otherwise(None)
                .alias("net_pnl")
            ]
        )
        .select(list(TRADE_SCHEMA))
    )


def _chunk_start_days(
    prepared_a: pl.LazyFrame,
    prepared_b: pl.LazyFrame,
    config: ConditionalProbabilitySqueezeConfig,
) -> list[int]:
    day_indices = _collect_lazy(
        pl.concat(
            [
                prepared_a.select(pl.col("timestamp").floordiv(DAY_MS).alias("day_index")),
                prepared_b.select(pl.col("timestamp").floordiv(DAY_MS).alias("day_index")),
            ],
            how="vertical",
        )
        .unique()
        .sort("day_index"),
        config.collect_engine,
    )
    if day_indices.is_empty():
        return []

    starts: list[int] = []
    next_window_start: int | None = None
    for day_index in day_indices.get_column("day_index").to_list():
        normalized_day = int(day_index)
        if next_window_start is None or normalized_day >= next_window_start:
            starts.append(normalized_day)
            next_window_start = normalized_day + config.chunk_days
    return starts


def _slice_prepared_frame(frame: pl.LazyFrame, start_ms: int, end_ms: int) -> pl.LazyFrame:
    return frame.filter((pl.col("timestamp") >= start_ms) & (pl.col("timestamp") < end_ms))


def _chunk_diagnostic_row(
    chunk_id: int,
    signal_window_start_ms: int,
    signal_window_end_ms: int,
    scan_window_start_ms: int,
    scan_window_end_ms: int,
    chunk_trades: pl.DataFrame,
) -> dict[str, int]:
    return {
        "chunk_id": int(chunk_id),
        "signal_window_start_ts": int(signal_window_start_ms),
        "signal_window_end_ts": int(signal_window_end_ms),
        "scan_window_start_ts": int(scan_window_start_ms),
        "scan_window_end_ts": int(scan_window_end_ms),
        "signals": int(chunk_trades.height),
        "pretrade_rejections": _trade_state_count(chunk_trades, "rejected_pretrade_depth"),
        "route_arrival_full_rejections": _trade_state_count(chunk_trades, "expired_before_fill"),
        "partial_fills": _partial_fill_count(chunk_trades),
        "surviving_fok_baskets": _trade_state_count(chunk_trades, "basket_closed")
        + _trade_state_count(chunk_trades, "basket_open"),
    }


def _rename_market_frame(frame: pl.LazyFrame, *, prefix: str, join_key_name: str) -> pl.LazyFrame:
    rename_map: dict[str, str] = {}
    for name in frame.collect_schema().names():
        if name == "timestamp":
            rename_map[name] = join_key_name
        else:
            rename_map[name] = f"{name}_{prefix}"
    return frame.rename(rename_map)


def _timestamp_ms_expr(column_name: str, dtype: pl.DataType, timestamp_unit: TimestampUnit) -> pl.Expr:
    expr = pl.col(column_name)
    base_type = dtype.base_type() if hasattr(dtype, "base_type") else dtype

    if base_type is pl.Datetime:
        return expr.dt.epoch(time_unit="ms")
    if dtype == pl.Date:
        return expr.cast(pl.Datetime(time_unit="ms")).dt.epoch(time_unit="ms")
    if dtype == pl.String:
        return expr.str.to_datetime(time_unit="us", strict=False).dt.epoch(time_unit="ms")
    if dtype not in INTEGER_DTYPES and dtype not in FLOAT_DTYPES:
        raise TypeError(f"Unsupported timestamp dtype for {column_name}: {dtype}")

    numeric = expr.cast(pl.Float64)
    abs_numeric = numeric.abs()
    if timestamp_unit == "s":
        return (numeric * 1_000.0).round(0).cast(pl.Int64)
    if timestamp_unit == "ms":
        return numeric.round(0).cast(pl.Int64)
    if timestamp_unit == "us":
        return (numeric / 1_000.0).round(0).cast(pl.Int64)
    if timestamp_unit == "ns":
        return (numeric / 1_000_000.0).round(0).cast(pl.Int64)

    return (
        pl.when(abs_numeric >= 100_000_000_000_000_000.0)
        .then((numeric / 1_000_000.0).round(0))
        .when(abs_numeric >= 100_000_000_000_000.0)
        .then((numeric / 1_000.0).round(0))
        .when(abs_numeric >= 100_000_000_000.0)
        .then(numeric.round(0))
        .otherwise((numeric * 1_000.0).round(0))
        .cast(pl.Int64)
    )


def _collect_lazy(frame: pl.LazyFrame, engine: Literal["auto", "streaming"]) -> pl.DataFrame:
    if engine == "streaming":
        return frame.collect(engine="streaming")
    return frame.collect()


def _ensure_signal_schema(signals: pl.DataFrame) -> pl.DataFrame:
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in SIGNAL_SCHEMA.items()
        if name not in signals.columns
    ]
    if missing:
        signals = signals.with_columns(missing)
    return signals.select(list(SIGNAL_SCHEMA))


def _ensure_trade_schema(trades: pl.DataFrame) -> pl.DataFrame:
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in TRADE_SCHEMA.items()
        if name not in trades.columns
    ]
    if missing:
        trades = trades.with_columns(missing)
    return trades.select(list(TRADE_SCHEMA))


def _empty_signal_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SIGNAL_SCHEMA)


def _empty_trade_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TRADE_SCHEMA)


def _empty_chunk_stats_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=CHUNK_SCHEMA)


def _trade_state_count(trades: pl.DataFrame, state: str) -> int:
    if trades.is_empty():
        return 0
    return int(trades.filter(pl.col("trade_state") == state).height)


def _partial_fill_count(trades: pl.DataFrame) -> int:
    if trades.is_empty():
        return 0
    return int(
        trades.filter(pl.col("trade_state").is_in(["flattened_stage1", "flattened_stage2", "partial_unresolved"]))
        .height
    )


def _safe_sum(frame: pl.DataFrame, expr: pl.Expr) -> float:
    if frame.is_empty():
        return 0.0
    value = frame.select(expr).item()
    return float(value or 0.0)


def _summarize_trades(
    trades: pl.DataFrame,
    signals: pl.DataFrame,
    config: ConditionalProbabilitySqueezeConfig,
    *,
    chunk_stats: pl.DataFrame | None,
) -> dict[str, Any]:
    total_valid_signals = int(signals.height)
    pretrade_rejections = _trade_state_count(trades, "rejected_pretrade_depth")
    route_arrival_full_rejections = _trade_state_count(trades, "expired_before_fill")
    basket_closed = _trade_state_count(trades, "basket_closed")
    basket_open = _trade_state_count(trades, "basket_open")
    flattened_stage1 = _trade_state_count(trades, "flattened_stage1")
    flattened_stage2 = _trade_state_count(trades, "flattened_stage2")
    unresolved_partials = _trade_state_count(trades, "partial_unresolved")
    realized_trades = int(trades.filter(pl.col("net_pnl").is_not_null()).height) if not trades.is_empty() else 0
    winning_trades = int(trades.filter(pl.col("net_pnl") > 0).height) if not trades.is_empty() else 0

    partial_fill_flatten_count = flattened_stage1 + flattened_stage2 + unresolved_partials
    decision_time_fok_passes = max(0, total_valid_signals - pretrade_rejections)
    successful_fok_baskets = basket_closed + basket_open
    route_arrival_failed_total = route_arrival_full_rejections + partial_fill_flatten_count

    successful_fok_trades = trades.filter(pl.col("trade_state").is_in(["basket_closed", "basket_open"]))
    flattened_trades = trades.filter(pl.col("trade_state").is_in(["flattened_stage1", "flattened_stage2"]))

    gross_pnl = _safe_sum(trades, pl.col("gross_pnl").fill_null(0.0).sum())
    net_pnl = _safe_sum(trades, pl.col("net_pnl").fill_null(0.0).sum())
    successful_fok_net_pnl = _safe_sum(successful_fok_trades, pl.col("net_pnl").fill_null(0.0).sum())
    flattened_basket_net_pnl = _safe_sum(flattened_trades, pl.col("net_pnl").fill_null(0.0).sum())
    flattened_basket_net_loss = _safe_sum(
        flattened_trades,
        pl.when(pl.col("net_pnl") < 0.0).then(-pl.col("net_pnl")).otherwise(0.0).sum(),
    )

    win_rate = (winning_trades / realized_trades) if realized_trades else 0.0
    avg_net_pnl = (net_pnl / realized_trades) if realized_trades else 0.0
    fok_survival_rate_at_route_latency = (
        successful_fok_baskets / decision_time_fok_passes if decision_time_fok_passes else 0.0
    )

    return {
        "route_latency_ms": int(config.route_latency_ms),
        "route_latency_us": int(config.route_latency_us),
        "timestamp_unit": config.timestamp_unit,
        "minimum_edge_over_combined_spread_ratio": float(config.minimum_edge_over_combined_spread_ratio),
        "minimum_theoretical_edge_dollars": float(config.minimum_theoretical_edge_dollars),
        "process_by_day": bool(config.process_by_day),
        "chunk_days": int(config.chunk_days),
        "warmup_days": int(config.warmup_days),
        "chunk_lookahead_ms": int(config.effective_chunk_lookahead_ms),
        "collect_engine": config.collect_engine,
        "total_valid_signals_generated": total_valid_signals,
        "decision_time_fok_rejections": pretrade_rejections,
        "decision_time_fok_passes": decision_time_fok_passes,
        "route_arrival_full_rejections": route_arrival_full_rejections,
        "partial_fills_requiring_flatten": partial_fill_flatten_count,
        "flattened_stage1": flattened_stage1,
        "flattened_stage2": flattened_stage2,
        "flatten_unresolved": unresolved_partials,
        "successful_fok_baskets": successful_fok_baskets,
        "route_arrival_failed_total": route_arrival_failed_total,
        "fok_survival_rate_at_route_latency": fok_survival_rate_at_route_latency,
        "basket_closed": basket_closed,
        "basket_open": basket_open,
        "realized_trades": realized_trades,
        "winning_trades": winning_trades,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "successful_fok_net_pnl": successful_fok_net_pnl,
        "flattened_basket_net_pnl": flattened_basket_net_pnl,
        "flattened_basket_net_loss": flattened_basket_net_loss,
        "win_rate": win_rate,
        "avg_net_pnl": avg_net_pnl,
        "chunks_processed": int(chunk_stats.height) if chunk_stats is not None else 0,
    }