from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

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
RAW_BOOK_COLUMNS = {
    "timestamp",
    "market_id",
    "event_id",
    "token_id",
    "best_bid",
    "best_ask",
    "bid_depth",
    "ask_depth",
}
GROUP_COLUMNS = ["market_id", "token_id"]
DATETIME_DTYPE = pl.Datetime("us")
MICROS_PER_SECOND = 1_000_000.0
SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY
FLOAT_EPSILON = 1e-9
ID_DTYPE = pl.Categorical
FLOAT_DTYPE = pl.Float32

NEAR_MISS_COLUMNS = [
    "timestamp",
    "near_miss_date",
    *GROUP_COLUMNS,
    "event_id",
    "resolution_timestamp",
    "best_bid",
    "best_ask",
    "near_miss_ask_shortfall",
    "near_miss_bid_overshoot",
    "near_miss_price_gap",
    "near_miss_reason",
    "time_to_resolution_hours",
]

PRICE_DISTRIBUTION_COLUMNS = [
    "market_id",
    "token_id",
    "resolution_timestamp",
    "final_resolution_value",
    "final_result",
    "deepest_dip",
    "highest_spike",
    "observation_count",
    "window_start",
    "window_end",
]

PRICE_DISTRIBUTION_PARTIAL_COLUMNS = [
    *GROUP_COLUMNS,
    "first_observation_timestamp",
    "resolution_timestamp",
    "final_resolution_value",
    "deepest_dip",
    "highest_spike",
    "observation_count",
    "window_start",
    "window_end",
]

FULL_CANDIDATE_COLUMNS = [
    *GROUP_COLUMNS,
    "cycle_id",
    "signal_event_id",
    "fill_event_id",
    "order_posted_at",
    "order_date",
    "fill_timestamp",
    "fill_date",
    "resolution_timestamp",
    "resolution_date",
    "signal_best_bid",
    "signal_best_ask",
    "fill_best_bid",
    "fill_best_ask",
    "fill_reason",
    "signal_row_count",
    "order_price",
    "filled",
    "ticket_notional_usdc",
    "contracts_if_filled",
    "order_to_resolution_seconds",
    "order_to_resolution_hours",
    "order_to_resolution_days",
    "time_to_fill_seconds",
    "lockup_seconds",
    "lockup_hours",
    "lockup_days",
    "final_resolution_value",
    "settled_yes",
    "catastrophic_loss",
    "raw_pnl",
    "raw_roi",
    "actual_raw_pnl",
    "actual_raw_roi",
    "assumed_yes_pnl_usdc",
    "resolution_cash_return_usdc_if_filled",
    "actual_pnl_usdc_if_filled",
    "apr",
    "actual_apr",
]

PORTFOLIO_CANDIDATE_COLUMNS = [
    *GROUP_COLUMNS,
    "cycle_id",
    "signal_event_id",
    "fill_event_id",
    "order_posted_at",
    "order_date",
    "fill_timestamp",
    "fill_date",
    "resolution_timestamp",
    "resolution_date",
    "signal_best_bid",
    "signal_best_ask",
    "fill_best_bid",
    "fill_best_ask",
    "fill_reason",
    "signal_row_count",
    "order_price",
    "filled",
    "ticket_notional_usdc",
    "contracts_if_filled",
    "order_to_resolution_seconds",
    "order_to_resolution_hours",
    "time_to_fill_seconds",
    "lockup_seconds",
    "lockup_hours",
    "final_resolution_value",
    "settled_yes",
    "catastrophic_loss",
    "raw_pnl",
    "raw_roi",
    "actual_raw_pnl",
    "actual_raw_roi",
    "assumed_yes_pnl_usdc",
    "resolution_cash_return_usdc_if_filled",
    "actual_pnl_usdc_if_filled",
    "actual_apr",
]


@dataclass(frozen=True, slots=True)
class ScavengerConfig:
    resolution_window_hours: int = 72
    signal_best_ask_min: float = 0.99
    signal_best_bid_max: float = 0.96
    maker_bid_price: float = 0.95
    near_miss_price_tolerance: float = 0.02
    starting_bankroll_usdc: float = 5_000.0
    max_notional_per_market_usdc: float = 250.0

    def __post_init__(self) -> None:
        if self.resolution_window_hours <= 0:
            raise ValueError("resolution_window_hours must be strictly positive")
        if self.maker_bid_price <= 0.0:
            raise ValueError("maker_bid_price must be strictly positive")
        if self.near_miss_price_tolerance <= 0.0:
            raise ValueError("near_miss_price_tolerance must be strictly positive")
        if self.starting_bankroll_usdc <= 0.0:
            raise ValueError("starting_bankroll_usdc must be strictly positive")
        if self.max_notional_per_market_usdc <= 0.0:
            raise ValueError("max_notional_per_market_usdc must be strictly positive")


@dataclass(frozen=True, slots=True)
class ScheduledRelease:
    resolution_timestamp: datetime
    cash_return_usdc: float
    reserved_notional_usdc: float
    catastrophic_loss: bool


@dataclass(frozen=True, slots=True)
class ScavengerBacktestResult:
    orders: pl.DataFrame
    candidates: pl.DataFrame
    fills: pl.DataFrame
    near_misses: pl.DataFrame
    price_distribution: pl.DataFrame
    portfolio: pl.DataFrame
    summary: pl.DataFrame


@dataclass(slots=True)
class ScavengerPortfolioState:
    available_cash_usdc: float
    locked_capital_usdc: float = 0.0
    peak_locked_capital_usdc: float = 0.0
    pending_releases: list[tuple[datetime, int, ScheduledRelease]] = field(default_factory=list)
    release_sequence: int = 0

    @classmethod
    def from_config(cls, config: ScavengerConfig) -> "ScavengerPortfolioState":
        return cls(available_cash_usdc=float(config.starting_bankroll_usdc))

    def can_allocate(self, notional_usdc: float) -> bool:
        return self.available_cash_usdc + FLOAT_EPSILON >= notional_usdc

    def reserve(self, notional_usdc: float) -> None:
        self.available_cash_usdc -= notional_usdc
        self.locked_capital_usdc += notional_usdc
        self.peak_locked_capital_usdc = max(self.peak_locked_capital_usdc, self.locked_capital_usdc)

    def schedule_release(self, release: ScheduledRelease) -> None:
        self.release_sequence += 1
        heapq.heappush(
            self.pending_releases,
            (release.resolution_timestamp, self.release_sequence, release),
        )

    def release_until(self, cutoff: datetime) -> list[ScheduledRelease]:
        released: list[ScheduledRelease] = []
        while self.pending_releases and self.pending_releases[0][0] <= cutoff:
            _, _, release = heapq.heappop(self.pending_releases)
            self.available_cash_usdc += release.cash_return_usdc
            self.locked_capital_usdc -= release.reserved_notional_usdc
            if abs(self.locked_capital_usdc) <= FLOAT_EPSILON:
                self.locked_capital_usdc = 0.0
            released.append(release)
        return released

    def release_all(self) -> list[ScheduledRelease]:
        released: list[ScheduledRelease] = []
        while self.pending_releases:
            _, _, release = heapq.heappop(self.pending_releases)
            self.available_cash_usdc += release.cash_return_usdc
            self.locked_capital_usdc -= release.reserved_notional_usdc
            if abs(self.locked_capital_usdc) <= FLOAT_EPSILON:
                self.locked_capital_usdc = 0.0
            released.append(release)
        return released


def _source_to_lazy_frame(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
) -> pl.LazyFrame:
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, (str, Path)):
        return pl.scan_parquet(str(source), low_memory=True, cache=False)

    paths = [str(path) for path in source]
    if not paths:
        raise ValueError("Expected at least one Parquet path.")
    return pl.scan_parquet(paths, low_memory=True, cache=False)


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


def load_scavenger_metadata_frame(metadata_path: str | Path) -> pl.DataFrame:
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
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

    return pl.DataFrame(
        rows,
        schema={
            "market_id": pl.Utf8,
            "token_id": pl.Utf8,
            "metadata_event_id": pl.Utf8,
            "resolution_timestamp": pl.Datetime("us", "UTC"),
            "final_resolution_value": pl.Float64,
        },
    ).unique(subset=["market_id", "token_id"], keep="first")


def scan_scavenger_source_frame(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    *,
    metadata_frame: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    frame = _source_to_lazy_frame(source)
    schema = frame.collect_schema()
    source_columns = set(schema.names())

    if REQUIRED_COLUMNS.issubset(source_columns):
        return frame

    if not RAW_BOOK_COLUMNS.issubset(source_columns):
        missing = sorted(REQUIRED_COLUMNS.difference(source_columns))
        raise ValueError(
            "Scavenger source is missing required columns and does not match the raw l2_book schema: "
            f"{', '.join(missing)}"
        )

    if metadata_frame is None:
        raise ValueError(
            "Raw 8-column scavenger sources require metadata_frame to add resolution_timestamp and "
            "final_resolution_value."
        )

    return (
        frame.select(
            pl.col("timestamp"),
            pl.col("market_id").cast(pl.Utf8).str.to_lowercase().alias("market_id"),
            pl.col("event_id").cast(pl.Utf8).alias("event_id"),
            pl.col("token_id").cast(pl.Utf8).alias("token_id"),
            pl.col("best_bid"),
            pl.col("best_ask"),
            pl.col("bid_depth"),
            pl.col("ask_depth"),
        )
        .join(metadata_frame.lazy(), on=["market_id", "token_id"], how="inner")
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

    return frame.select(
        _datetime_expr("timestamp", schema["timestamp"]).alias("timestamp"),
        pl.col("market_id").cast(ID_DTYPE),
        pl.col("event_id").cast(ID_DTYPE),
        pl.col("token_id").cast(ID_DTYPE),
        pl.col("best_bid").cast(FLOAT_DTYPE),
        pl.col("best_ask").cast(FLOAT_DTYPE),
        _datetime_expr("resolution_timestamp", schema["resolution_timestamp"]).alias(
            "resolution_timestamp"
        ),
        pl.col("final_resolution_value").cast(FLOAT_DTYPE),
    )


def _strict_resolution_window_expr(
    config: ScavengerConfig,
    *,
    column: str = "time_to_resolution_seconds",
) -> pl.Expr:
    signal_window_seconds = float(config.resolution_window_hours) * SECONDS_PER_HOUR
    return (pl.col(column) > 0.0) & (pl.col(column) < signal_window_seconds)


def _build_candidate_frames(
    frame: pl.LazyFrame,
    config: ScavengerConfig,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    maker_bid_price = float(config.maker_bid_price)
    ticket_notional_usdc = float(config.max_notional_per_market_usdc)
    near_miss_price_tolerance = float(config.near_miss_price_tolerance)
    null_float = pl.lit(None, dtype=FLOAT_DTYPE)
    time_to_resolution_seconds = (
        (pl.col("resolution_timestamp").cast(pl.Int64) - pl.col("timestamp").cast(pl.Int64))
        / MICROS_PER_SECOND
    )
    signal_expr = (
        (pl.col("best_ask") >= float(config.signal_best_ask_min))
        & (pl.col("best_bid") <= float(config.signal_best_bid_max))
    ).fill_null(False)
    near_miss_ask_shortfall = (
        (pl.lit(float(config.signal_best_ask_min)) - pl.col("best_ask"))
        .clip(lower_bound=0.0)
        .cast(FLOAT_DTYPE)
    )
    near_miss_bid_overshoot = (
        (pl.col("best_bid") - pl.lit(float(config.signal_best_bid_max)))
        .clip(lower_bound=0.0)
        .cast(FLOAT_DTYPE)
    )
    near_miss_expr = (
        (~signal_expr)
        & (near_miss_ask_shortfall < near_miss_price_tolerance)
        & (near_miss_bid_overshoot < near_miss_price_tolerance)
        & ((near_miss_ask_shortfall > 0.0) | (near_miss_bid_overshoot > 0.0))
    ).fill_null(False)
    fill_trigger_expr = (
        (pl.col("best_ask") <= maker_bid_price) | (pl.col("best_bid") < maker_bid_price)
    ).fill_null(False)

    annotated = (
        frame.with_columns(
            time_to_resolution_seconds.alias("time_to_resolution_seconds"),
        )
        .filter(_strict_resolution_window_expr(config))
        .with_columns(
            signal_expr.alias("entry_signal"),
            fill_trigger_expr.alias("fill_trigger"),
            near_miss_ask_shortfall.alias("near_miss_ask_shortfall"),
            near_miss_bid_overshoot.alias("near_miss_bid_overshoot"),
            near_miss_expr.alias("near_miss"),
        )
        .with_columns(
            pl.max_horizontal(["near_miss_ask_shortfall", "near_miss_bid_overshoot"])
            .cast(FLOAT_DTYPE)
            .alias("near_miss_price_gap")
        )
        .with_columns(
            pl.col("fill_trigger")
            .cast(pl.Int32)
            .cum_sum()
            .shift(1)
            .fill_null(0)
            .over(GROUP_COLUMNS, order_by="timestamp")
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
            pl.col("resolution_timestamp")
            .sort_by("timestamp")
            .first()
            .alias("signal_resolution_timestamp"),
            pl.col("final_resolution_value")
            .sort_by("timestamp")
            .first()
            .alias("signal_final_resolution_value"),
        )
        .with_columns(pl.lit(maker_bid_price, dtype=FLOAT_DTYPE).alias("order_price"))
    )

    near_misses = (
        annotated.filter(pl.col("near_miss"))
        .select(
            pl.col("timestamp"),
            pl.col("timestamp").dt.date().alias("near_miss_date"),
            *GROUP_COLUMNS,
            pl.col("event_id"),
            pl.col("resolution_timestamp"),
            pl.col("best_bid").cast(FLOAT_DTYPE).alias("best_bid"),
            pl.col("best_ask").cast(FLOAT_DTYPE).alias("best_ask"),
            pl.col("near_miss_ask_shortfall"),
            pl.col("near_miss_bid_overshoot"),
            pl.col("near_miss_price_gap"),
            pl.when(
                (pl.col("near_miss_ask_shortfall") > 0.0)
                & (pl.col("near_miss_bid_overshoot") > 0.0)
            )
            .then(pl.lit("both_price_thresholds"))
            .when(pl.col("near_miss_ask_shortfall") > 0.0)
            .then(pl.lit("best_ask_below_threshold"))
            .otherwise(pl.lit("best_bid_above_threshold"))
            .alias("near_miss_reason"),
            (pl.col("time_to_resolution_seconds") / SECONDS_PER_HOUR)
            .cast(FLOAT_DTYPE)
            .alias("time_to_resolution_hours"),
        )
        .select(NEAR_MISS_COLUMNS)
        .sort(["timestamp", "resolution_timestamp", *GROUP_COLUMNS])
    )

    fill_events = (
        annotated.filter(pl.col("fill_trigger"))
        .select(
            *GROUP_COLUMNS,
            "cycle_id",
            pl.col("timestamp").alias("fill_timestamp"),
            pl.col("event_id").alias("fill_event_id"),
            pl.col("best_bid").alias("fill_best_bid"),
            pl.col("best_ask").alias("fill_best_ask"),
            pl.when(pl.col("best_ask") <= maker_bid_price)
            .then(pl.lit("best_ask_le_0p95"))
            .otherwise(pl.lit("best_bid_lt_0p95"))
            .alias("fill_reason"),
            pl.col("resolution_timestamp").alias("fill_resolution_timestamp"),
            pl.col("final_resolution_value").alias("fill_final_resolution_value"),
        )
    )

    settlement_price = pl.lit(1.0)
    actual_final_value = pl.coalesce(
        [pl.col("fill_final_resolution_value"), pl.col("signal_final_resolution_value")]
    )
    filled_expr = pl.col("fill_timestamp").is_not_null()
    order_to_resolution_seconds = (
        (pl.col("resolution_timestamp").cast(pl.Int64) - pl.col("order_posted_at").cast(pl.Int64))
        / MICROS_PER_SECOND
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
    contracts_expr = pl.lit(ticket_notional_usdc) / pl.col("order_price")

    base_candidates = (
        orders.join(fill_events, on=[*GROUP_COLUMNS, "cycle_id"], how="left")
        .filter(pl.col("fill_timestamp").is_null() | (pl.col("fill_timestamp") > pl.col("order_posted_at")))
        .with_columns(
            pl.coalesce(
                [pl.col("fill_resolution_timestamp"), pl.col("signal_resolution_timestamp")]
            ).alias("resolution_timestamp"),
            actual_final_value.alias("final_resolution_value"),
            filled_expr.alias("filled"),
            pl.col("order_posted_at").dt.date().alias("order_date"),
            pl.col("fill_timestamp").dt.date().alias("fill_date"),
        )
        .with_columns(
            pl.col("resolution_timestamp").dt.date().alias("resolution_date"),
            order_to_resolution_seconds.cast(FLOAT_DTYPE).alias("order_to_resolution_seconds"),
            pl.lit(ticket_notional_usdc, dtype=FLOAT_DTYPE).alias("ticket_notional_usdc"),
            pl.when(pl.col("filled"))
            .then((contracts_expr - pl.lit(ticket_notional_usdc)).cast(FLOAT_DTYPE))
            .otherwise(null_float)
            .alias("assumed_yes_pnl_usdc"),
            pl.when(pl.col("filled"))
            .then((contracts_expr * actual_final_value).cast(FLOAT_DTYPE))
            .otherwise(null_float)
            .alias("resolution_cash_return_usdc_if_filled"),
            pl.when(pl.col("filled") & (actual_final_value <= FLOAT_EPSILON))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("catastrophic_loss"),
        )
    )

    metrics_candidates = (
        base_candidates.with_columns(
            pl.when(pl.col("filled")).then(fill_age_seconds.cast(FLOAT_DTYPE)).otherwise(null_float).alias("time_to_fill_seconds"),
            pl.when(pl.col("filled")).then(lockup_seconds.cast(FLOAT_DTYPE)).otherwise(null_float).alias("lockup_seconds"),
            (actual_final_value == 1.0).alias("settled_yes"),
            pl.when(pl.col("filled")).then(contracts_expr.cast(FLOAT_DTYPE)).otherwise(null_float).alias("contracts_if_filled"),
            pl.when(pl.col("filled")).then(assumed_raw_pnl.cast(FLOAT_DTYPE)).otherwise(null_float).alias("raw_pnl"),
            pl.when(pl.col("filled")).then((assumed_raw_pnl / pl.col("order_price")).cast(FLOAT_DTYPE)).otherwise(null_float).alias("raw_roi"),
            pl.when(pl.col("filled")).then(actual_raw_pnl.cast(FLOAT_DTYPE)).otherwise(null_float).alias("actual_raw_pnl"),
            pl.when(pl.col("filled")).then((actual_raw_pnl / pl.col("order_price")).cast(FLOAT_DTYPE)).otherwise(null_float).alias("actual_raw_roi"),
        )
        .with_columns(
            (pl.col("lockup_seconds") / SECONDS_PER_HOUR).cast(FLOAT_DTYPE).alias("lockup_hours"),
            (pl.col("lockup_seconds") / SECONDS_PER_DAY).cast(FLOAT_DTYPE).alias("lockup_days"),
            (pl.col("order_to_resolution_seconds") / SECONDS_PER_HOUR).cast(FLOAT_DTYPE).alias("order_to_resolution_hours"),
            (pl.col("order_to_resolution_seconds") / SECONDS_PER_DAY).cast(FLOAT_DTYPE).alias("order_to_resolution_days"),
            pl.when(pl.col("lockup_seconds") > 0.0)
            .then((pl.col("raw_roi") * SECONDS_PER_YEAR / pl.col("lockup_seconds")).cast(FLOAT_DTYPE))
            .otherwise(null_float)
            .alias("apr"),
            pl.when(pl.col("lockup_seconds") > 0.0)
            .then((pl.col("actual_raw_roi") * SECONDS_PER_YEAR / pl.col("lockup_seconds")).cast(FLOAT_DTYPE))
            .otherwise(null_float)
            .alias("actual_apr"),
            pl.when(pl.col("filled"))
            .then((pl.col("resolution_cash_return_usdc_if_filled") - pl.col("ticket_notional_usdc")).cast(FLOAT_DTYPE))
            .otherwise(null_float)
            .alias("actual_pnl_usdc_if_filled"),
        )
    )

    candidates = metrics_candidates.select(FULL_CANDIDATE_COLUMNS).sort(
        ["order_posted_at", "resolution_timestamp", *GROUP_COLUMNS]
    )
    lightweight_candidates = metrics_candidates.select(PORTFOLIO_CANDIDATE_COLUMNS).sort(
        ["order_posted_at", "resolution_timestamp", *GROUP_COLUMNS]
    )

    return orders, candidates, lightweight_candidates, near_misses


def _build_raw_summary(
    orders: pl.LazyFrame,
    fills: pl.LazyFrame,
    near_misses: pl.LazyFrame,
    config: ScavengerConfig,
) -> pl.LazyFrame:
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
        (pl.len() * float(config.max_notional_per_market_usdc)).alias("capital_committed_usdc"),
        pl.col("assumed_yes_pnl_usdc").sum().fill_null(0.0).alias("gross_pnl_usdc"),
        pl.col("actual_pnl_usdc_if_filled").sum().fill_null(0.0).alias("actual_gross_pnl_usdc"),
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
    near_miss_stats = near_misses.select(
        pl.len().alias("near_misses"),
        pl.col("market_id").n_unique().alias("near_miss_markets"),
        pl.col("token_id").n_unique().alias("near_miss_tokens"),
        pl.col("near_miss_price_gap").mean().alias("mean_near_miss_price_gap"),
        pl.col("near_miss_price_gap").min().alias("closest_near_miss_price_gap"),
    )
    return (
        order_stats.join(fill_stats, how="cross")
        .join(near_miss_stats, how="cross")
        .with_columns(
            pl.when(pl.col("orders_posted") > 0)
            .then(pl.col("fills") / pl.col("orders_posted"))
            .otherwise(0.0)
            .alias("fill_rate"),
            pl.lit(float(config.maker_bid_price)).alias("maker_bid_price"),
            pl.lit(float(config.near_miss_price_tolerance)).alias("near_miss_price_tolerance"),
            pl.lit(float(config.signal_best_ask_min)).alias("signal_best_ask_min"),
            pl.lit(float(config.signal_best_bid_max)).alias("signal_best_bid_max"),
            pl.lit(int(config.resolution_window_hours)).alias("resolution_window_hours"),
        )
        .select(
            "resolution_window_hours",
            "maker_bid_price",
            "near_miss_price_tolerance",
            "signal_best_ask_min",
            "signal_best_bid_max",
            "signal_rows",
            "orders_posted",
            "fills",
            "fill_rate",
            "near_misses",
            "near_miss_markets",
            "near_miss_tokens",
            "mean_near_miss_price_gap",
            "closest_near_miss_price_gap",
            "markets_signaled",
            "tokens_signaled",
            "markets_filled",
            "tokens_filled",
            "capital_committed_usdc",
            "gross_pnl_usdc",
            "actual_gross_pnl_usdc",
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


def _build_portfolio_summary(
    raw_summary: pl.DataFrame,
    portfolio: pl.DataFrame,
    state: ScavengerPortfolioState,
    config: ScavengerConfig,
) -> pl.DataFrame:
    raw_row = raw_summary.row(0, named=True)
    accepted = portfolio.filter(pl.col("accepted"))
    accepted_filled = accepted.filter(pl.col("filled"))
    rejected = portfolio.filter(pl.col("rejection_reason") == "capital_lockup")
    total_realized_pnl_usdc = float(
        accepted.select(pl.col("realized_pnl_usdc").fill_null(0.0).sum()).item() or 0.0
    )
    total_assumed_yes_pnl_usdc = float(
        accepted.select(pl.col("assumed_yes_pnl_usdc").fill_null(0.0).sum()).item() or 0.0
    )
    accepted_count = accepted.height
    filled_count = accepted_filled.height
    summary_row = dict(raw_row)
    summary_row.update(
        {
            "starting_bankroll_usdc": float(config.starting_bankroll_usdc),
            "ending_bankroll_usdc": float(state.available_cash_usdc),
            "net_return_pct": (
                (float(state.available_cash_usdc) - float(config.starting_bankroll_usdc))
                / float(config.starting_bankroll_usdc)
            ),
            "max_notional_per_market_usdc": float(config.max_notional_per_market_usdc),
            "portfolio_orders_accepted": accepted_count,
            "portfolio_fills": filled_count,
            "portfolio_fill_rate": (filled_count / accepted_count) if accepted_count else 0.0,
            "signals_rejected_capital_lockup": rejected.height,
            "catastrophic_losses": accepted_filled.filter(pl.col("catastrophic_loss")).height,
            "portfolio_realized_pnl_usdc": total_realized_pnl_usdc,
            "portfolio_assumed_yes_pnl_usdc": total_assumed_yes_pnl_usdc,
            "peak_locked_capital_usdc": float(state.peak_locked_capital_usdc),
        }
    )
    return pl.DataFrame([summary_row])


def _empty_portfolio_frame(candidates: pl.DataFrame) -> pl.DataFrame:
    if not candidates.columns:
        return pl.DataFrame(
            schema={
                "accepted": pl.Boolean,
                "filled": pl.Boolean,
                "rejection_reason": pl.String,
                "position_status": pl.String,
                "capital_before_signal_usdc": FLOAT_DTYPE,
                "capital_after_signal_usdc": FLOAT_DTYPE,
                "realized_pnl_usdc": FLOAT_DTYPE,
            }
        )
    return candidates.with_columns(
        pl.lit(False).alias("accepted"),
        pl.lit(None, dtype=pl.String).alias("rejection_reason"),
        pl.lit("rejected_capital_lockup").alias("position_status"),
        pl.lit(None, dtype=FLOAT_DTYPE).alias("capital_before_signal_usdc"),
        pl.lit(None, dtype=FLOAT_DTYPE).alias("capital_after_signal_usdc"),
        pl.lit(None, dtype=FLOAT_DTYPE).alias("realized_pnl_usdc"),
    )


def _sort_portfolio_candidates(candidates: pl.DataFrame) -> pl.DataFrame:
    return candidates.sort(
        ["order_posted_at", "order_to_resolution_seconds", "resolution_timestamp", *GROUP_COLUMNS],
        descending=[False, False, False, False, False],
    )


def simulate_scavenger_portfolio(
    candidates: pl.DataFrame,
    config: ScavengerConfig = ScavengerConfig(),
    *,
    state: ScavengerPortfolioState | None = None,
    finalize: bool = True,
) -> tuple[pl.DataFrame, ScavengerPortfolioState]:
    portfolio_state = state or ScavengerPortfolioState.from_config(config)
    if candidates.is_empty():
        if finalize:
            portfolio_state.release_all()
        return _empty_portfolio_frame(candidates), portfolio_state

    ordered_candidates = _sort_portfolio_candidates(candidates)
    records: list[dict[str, object]] = []
    for row in ordered_candidates.iter_rows(named=True):
        order_posted_at = row["order_posted_at"]
        resolution_timestamp = row["resolution_timestamp"]
        if not isinstance(order_posted_at, datetime) or not isinstance(resolution_timestamp, datetime):
            raise TypeError("Expected Python datetime objects in candidate rows")

        portfolio_state.release_until(order_posted_at)
        ticket_notional_usdc = float(row["ticket_notional_usdc"])
        capital_before_signal_usdc = float(portfolio_state.available_cash_usdc)
        accepted = portfolio_state.can_allocate(ticket_notional_usdc)

        record = dict(row)
        record["capital_before_signal_usdc"] = capital_before_signal_usdc
        if accepted:
            portfolio_state.reserve(ticket_notional_usdc)
            resolution_cash_return_usdc = (
                float(row["resolution_cash_return_usdc_if_filled"])
                if row["filled"]
                else ticket_notional_usdc
            )
            realized_pnl_usdc = (
                float(row["actual_pnl_usdc_if_filled"])
                if row["filled"]
                else 0.0
            )
            portfolio_state.schedule_release(
                ScheduledRelease(
                    resolution_timestamp=resolution_timestamp,
                    cash_return_usdc=resolution_cash_return_usdc,
                    reserved_notional_usdc=ticket_notional_usdc,
                    catastrophic_loss=bool(row["catastrophic_loss"]),
                )
            )
            record["accepted"] = True
            record["rejection_reason"] = None
            record["position_status"] = "accepted_filled" if row["filled"] else "accepted_unfilled"
            record["capital_after_signal_usdc"] = float(portfolio_state.available_cash_usdc)
            record["realized_pnl_usdc"] = realized_pnl_usdc
        else:
            record["accepted"] = False
            record["rejection_reason"] = "capital_lockup"
            record["position_status"] = "rejected_capital_lockup"
            record["capital_after_signal_usdc"] = capital_before_signal_usdc
            record["realized_pnl_usdc"] = None
        records.append(record)

    if finalize:
        portfolio_state.release_all()
    return pl.DataFrame(records), portfolio_state


def build_scavenger_candidate_frame(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
    *,
    lightweight: bool = False,
    metadata_frame: pl.DataFrame | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    orders, selected_candidates, fills, _ = build_scavenger_diagnostic_frames(
        source,
        config=config,
        lightweight=lightweight,
        metadata_frame=metadata_frame,
    )
    return orders, selected_candidates, fills


def _collect_lazy_frame(frame: pl.LazyFrame) -> pl.DataFrame:
    try:
        return frame.collect(engine="streaming")
    except Exception:
        return frame.collect()


def _build_scavenger_price_distribution_partial_frame(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
    *,
    metadata_frame: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    normalized = _normalize_frame(scan_scavenger_source_frame(source, metadata_frame=metadata_frame))
    time_to_resolution_seconds = (
        (pl.col("resolution_timestamp").cast(pl.Int64) - pl.col("timestamp").cast(pl.Int64))
        / MICROS_PER_SECOND
    )
    return (
        normalized.with_columns(time_to_resolution_seconds.alias("time_to_resolution_seconds"))
        .filter(_strict_resolution_window_expr(config))
        .group_by(GROUP_COLUMNS)
        .agg(
            pl.col("timestamp").min().alias("first_observation_timestamp"),
            pl.col("resolution_timestamp").sort_by("timestamp").first().alias("resolution_timestamp"),
            pl.col("final_resolution_value").sort_by("timestamp").first().alias("final_resolution_value"),
            pl.col("best_ask").min().cast(FLOAT_DTYPE).alias("deepest_dip"),
            pl.col("best_bid").max().cast(FLOAT_DTYPE).alias("highest_spike"),
            pl.len().alias("observation_count"),
            pl.col("timestamp").min().alias("window_start"),
            pl.col("timestamp").max().alias("window_end"),
        )
        .select(PRICE_DISTRIBUTION_PARTIAL_COLUMNS)
    )


def _finalize_scavenger_price_distribution_frame(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        frame.group_by(GROUP_COLUMNS)
        .agg(
            pl.col("resolution_timestamp")
            .sort_by("first_observation_timestamp")
            .first()
            .alias("resolution_timestamp"),
            pl.col("final_resolution_value")
            .sort_by("first_observation_timestamp")
            .first()
            .alias("final_resolution_value"),
            pl.col("deepest_dip").min().cast(FLOAT_DTYPE).alias("deepest_dip"),
            pl.col("highest_spike").max().cast(FLOAT_DTYPE).alias("highest_spike"),
            pl.col("observation_count").sum().alias("observation_count"),
            pl.col("window_start").min().alias("window_start"),
            pl.col("window_end").max().alias("window_end"),
        )
        .with_columns(
            pl.col("final_resolution_value").round(0).cast(pl.Int8).alias("final_result")
        )
        .select(PRICE_DISTRIBUTION_COLUMNS)
        .sort(["deepest_dip", "highest_spike", "market_id", "token_id"])
    )


def build_scavenger_price_distribution_frame(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
    *,
    metadata_frame: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    return _finalize_scavenger_price_distribution_frame(
        _build_scavenger_price_distribution_partial_frame(
            source,
            config=config,
            metadata_frame=metadata_frame,
        )
    )


def collect_scavenger_price_distribution(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
    *,
    metadata_frame: pl.DataFrame | None = None,
    chunk_size: int = 64,
) -> pl.DataFrame:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be strictly positive")

    if isinstance(source, (pl.DataFrame, pl.LazyFrame, str, Path)):
        return _collect_lazy_frame(
            build_scavenger_price_distribution_frame(
                source,
                config=config,
                metadata_frame=metadata_frame,
            )
        )

    paths = [Path(path) for path in source]
    if not paths:
        raise ValueError("Expected at least one Parquet path.")

    partial_frames: list[pl.DataFrame] = []
    for start in range(0, len(paths), chunk_size):
        partial_frames.append(
            _collect_lazy_frame(
                _build_scavenger_price_distribution_partial_frame(
                    paths[start : start + chunk_size],
                    config=config,
                    metadata_frame=metadata_frame,
                )
            )
        )

    combined = pl.concat(partial_frames, how="vertical")
    return _collect_lazy_frame(
        _finalize_scavenger_price_distribution_frame(combined.lazy())
    )


def build_scavenger_diagnostic_frames(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
    *,
    lightweight: bool = False,
    metadata_frame: pl.DataFrame | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    normalized = _normalize_frame(scan_scavenger_source_frame(source, metadata_frame=metadata_frame))
    orders, candidates, lightweight_candidates, near_misses = _build_candidate_frames(normalized, config)
    selected_candidates = lightweight_candidates if lightweight else candidates
    fills = selected_candidates.filter(pl.col("filled"))
    return orders, selected_candidates, fills, near_misses


def build_scavenger_backtest(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
    *,
    metadata_frame: pl.DataFrame | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    orders, _, fills, near_misses = build_scavenger_diagnostic_frames(
        source,
        config=config,
        metadata_frame=metadata_frame,
    )
    summary = _build_raw_summary(orders, fills, near_misses, config)
    return orders, fills, summary


def run_scavenger_backtest(
    source: pl.DataFrame | pl.LazyFrame | str | Path | Sequence[str | Path],
    config: ScavengerConfig = ScavengerConfig(),
    *,
    metadata_frame: pl.DataFrame | None = None,
) -> ScavengerBacktestResult:
    orders_lf, candidates_lf, fills_lf, near_misses_lf = build_scavenger_diagnostic_frames(
        source,
        config=config,
        metadata_frame=metadata_frame,
    )
    price_distribution_df = collect_scavenger_price_distribution(
        source,
        config=config,
        metadata_frame=metadata_frame,
    )
    raw_summary_lf = _build_raw_summary(orders_lf, fills_lf, near_misses_lf, config)
    orders_df, candidates_df, fills_df, near_misses_df, raw_summary_df = pl.collect_all(
        [orders_lf, candidates_lf, fills_lf, near_misses_lf, raw_summary_lf]
    )
    portfolio_df, portfolio_state = simulate_scavenger_portfolio(candidates_df, config=config)
    summary_df = _build_portfolio_summary(raw_summary_df, portfolio_df, portfolio_state, config)
    return ScavengerBacktestResult(
        orders=orders_df,
        candidates=candidates_df,
        fills=fills_df,
        near_misses=near_misses_df,
        price_distribution=price_distribution_df,
        portfolio=portfolio_df,
        summary=summary_df,
    )


def summarize_near_misses(
    near_misses: pl.DataFrame,
    *,
    top_n: int = 10,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if top_n <= 0:
        raise ValueError("top_n must be strictly positive")

    if near_misses.is_empty():
        return (
            pl.DataFrame(
                schema={
                    "near_miss_date": pl.Date,
                    "near_miss_count": pl.Int64,
                    "near_miss_market_count": pl.Int64,
                }
            ),
            pl.DataFrame(
                schema={
                    "market_id": pl.Utf8,
                    "near_miss_count": pl.Int64,
                    "token_ids": pl.List(pl.Utf8),
                    "closest_near_miss_price_gap": pl.Float64,
                    "first_near_miss_date": pl.Date,
                    "last_near_miss_date": pl.Date,
                }
            ),
        )

    normalized = near_misses.with_columns(
        pl.col("market_id").cast(pl.Utf8),
        pl.col("token_id").cast(pl.Utf8),
    )
    daily = normalized.group_by("near_miss_date").agg(
        pl.len().alias("near_miss_count"),
        pl.col("market_id").n_unique().alias("near_miss_market_count"),
    ).sort("near_miss_date")
    top_markets = (
        normalized.group_by("market_id")
        .agg(
            pl.len().alias("near_miss_count"),
            pl.col("token_id").sort().unique().alias("token_ids"),
            pl.col("near_miss_price_gap").min().alias("closest_near_miss_price_gap"),
            pl.col("near_miss_date").min().alias("first_near_miss_date"),
            pl.col("near_miss_date").max().alias("last_near_miss_date"),
        )
        .sort(
            ["near_miss_count", "closest_near_miss_price_gap", "market_id"],
            descending=[True, False, False],
        )
        .head(top_n)
    )
    return daily, top_markets


def summarize_scavenger_price_distribution(
    price_distribution: pl.DataFrame,
    *,
    current_bid_price: float,
) -> dict[str, object]:
    def _segment_summary(segment: pl.DataFrame, bid_price: float) -> dict[str, object]:
        if segment.is_empty():
            return {
                "unit_count": 0,
                "median_deepest_dip": None,
                "p25_deepest_dip": None,
                "p75_deepest_dip": None,
                "median_highest_spike": None,
                "bid_touch_count": 0,
                "bid_touch_rate_pct": 0.0,
            }

        deepest = segment["deepest_dip"]
        highest = segment["highest_spike"]
        touched = segment.filter(pl.col("deepest_dip") <= bid_price).height
        return {
            "unit_count": segment.height,
            "median_deepest_dip": float(deepest.median()),
            "p25_deepest_dip": float(deepest.quantile(0.25)),
            "p75_deepest_dip": float(deepest.quantile(0.75)),
            "median_highest_spike": float(highest.median()),
            "bid_touch_count": touched,
            "bid_touch_rate_pct": (touched / segment.height) * 100.0,
        }

    if price_distribution.is_empty():
        return {
            "unit_count": 0,
            "median_deepest_dip": None,
            "p25_deepest_dip": None,
            "p75_deepest_dip": None,
            "modal_deepest_dip_bucket": None,
            "median_highest_spike": None,
            "current_bid_price": float(current_bid_price),
            "current_bid_touch_count": 0,
            "current_bid_touch_rate_pct": 0.0,
            "recommended_realistic_scavenge_bid": None,
            "recommended_bid_touch_count": 0,
            "recommended_bid_touch_rate_pct": 0.0,
            "winner_side_summary": _segment_summary(pl.DataFrame(), float(current_bid_price)),
            "loser_side_summary": _segment_summary(pl.DataFrame(), float(current_bid_price)),
            "touch_curve": [],
        }

    distribution = price_distribution.with_columns(
        pl.col("deepest_dip").cast(pl.Float64),
        pl.col("highest_spike").cast(pl.Float64),
        ((pl.col("deepest_dip") * 100.0).floor() / 100.0).alias("deepest_dip_bucket"),
    )
    unit_count = distribution.height
    deepest_dip_series = distribution["deepest_dip"]
    highest_spike_series = distribution["highest_spike"]
    median_deepest_dip = float(deepest_dip_series.median())
    p25_deepest_dip = float(deepest_dip_series.quantile(0.25))
    p75_deepest_dip = float(deepest_dip_series.quantile(0.75))
    median_highest_spike = float(highest_spike_series.median())
    modal_bucket_row = (
        distribution.group_by("deepest_dip_bucket")
        .agg(pl.len().alias("bucket_count"))
        .sort(["bucket_count", "deepest_dip_bucket"], descending=[True, False])
        .row(0, named=True)
    )
    modal_deepest_dip_bucket = round(float(modal_bucket_row["deepest_dip_bucket"]), 2)
    winners = distribution.filter(pl.col("final_result") == 1)
    losers = distribution.filter(pl.col("final_result") == 0)
    winner_side_summary = _segment_summary(winners, float(current_bid_price))
    loser_side_summary = _segment_summary(losers, float(current_bid_price))
    recommendation_basis_median = (
        winner_side_summary["median_deepest_dip"]
        if winner_side_summary["median_deepest_dip"] is not None
        else median_deepest_dip
    )
    floored_recommendation_basis = math.floor((recommendation_basis_median + FLOAT_EPSILON) * 100.0) / 100.0
    recommended_bid = max(0.01, round(floored_recommendation_basis - 0.01, 2))
    current_bid_touch_count = distribution.filter(pl.col("deepest_dip") <= float(current_bid_price)).height
    recommended_bid_touch_count = distribution.filter(pl.col("deepest_dip") <= recommended_bid).height
    winner_recommended_bid_touch_count = winners.filter(pl.col("deepest_dip") <= recommended_bid).height
    min_bucket = max(1, int(math.floor(float(deepest_dip_series.min()) * 100.0)))
    max_bucket = min(99, int(math.ceil(float(deepest_dip_series.max()) * 100.0)))
    touch_curve: list[dict[str, object]] = []
    for cent in range(max_bucket, min_bucket - 1, -1):
        price_level = cent / 100.0
        touched_units = distribution.filter(pl.col("deepest_dip") <= price_level).height
        touch_curve.append(
            {
                "bid_level": round(price_level, 2),
                "touched_units": touched_units,
                "touch_rate_pct": (touched_units / unit_count) * 100.0,
            }
        )

    return {
        "unit_count": unit_count,
        "median_deepest_dip": median_deepest_dip,
        "p25_deepest_dip": p25_deepest_dip,
        "p75_deepest_dip": p75_deepest_dip,
        "modal_deepest_dip_bucket": modal_deepest_dip_bucket,
        "median_highest_spike": median_highest_spike,
        "current_bid_price": float(current_bid_price),
        "current_bid_touch_count": current_bid_touch_count,
        "current_bid_touch_rate_pct": (current_bid_touch_count / unit_count) * 100.0,
        "recommended_realistic_scavenge_bid": recommended_bid,
        "recommended_bid_touch_count": recommended_bid_touch_count,
        "recommended_bid_touch_rate_pct": (recommended_bid_touch_count / unit_count) * 100.0,
        "winner_side_summary": {
            **winner_side_summary,
            "recommended_bid_touch_count": winner_recommended_bid_touch_count,
            "recommended_bid_touch_rate_pct": (
                (winner_recommended_bid_touch_count / winners.height) * 100.0 if winners.height else 0.0
            ),
        },
        "loser_side_summary": loser_side_summary,
        "touch_curve": touch_curve,
    }