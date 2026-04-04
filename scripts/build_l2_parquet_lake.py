#!/usr/bin/env python3
"""Build a strict Polymarket L2 Parquet lake from raw JSONL snapshots and deltas.

The raw recorder stores two complementary feeds per market/day:

1. Per-token decimal JSONL files containing full ``book`` snapshots.
2. Per-market hex JSONL files containing ``price_change`` deltas for both tokens.

This pipeline reconstructs a synchronized YES/NO book state for each market,
drops incomplete rows under the "no Swiss-cheese" rule, and emits a strict
eight-column Parquet lake for quant research.

Final Parquet schema
--------------------
``timestamp``  : ``Datetime(ms, UTC)``
``market_id``  : ``Utf8``
``event_id``   : ``Utf8``
``token_id``   : ``Utf8`` with values ``YES`` or ``NO``
``best_bid``   : ``Float64``
``best_ask``   : ``Float64``
``bid_depth``  : ``Float64`` top-5 bid notional, ``sum(price * size)``
``ask_depth``  : ``Float64`` top-5 ask notional, ``sum(price * size)``

Notes
-----
- The converter is intentionally strict. A market-day is skipped if either
  YES/NO snapshot file or the market delta file is missing.
- We only emit rows after both YES and NO books are seeded and both pass the
  BBO/depth validity gates at the event timestamp.
- Malformed NDJSON lines are dropped during ingestion salvage rather than
    aborting the entire market-day build; the drop counts are surfaced in the
    emitted run statistics.
- We set ``POLARS_SKIP_CPU_CHECK=1`` before importing Polars because the local
  Windows environment can reject the default runtime wheel during CPU feature
  probing. This keeps the script deterministic on the current host.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from heapq import heappop, heappush
from io import BytesIO
from itertools import count, islice
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4

from sortedcontainers import SortedDict

os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

import polars as pl


DEPTH_LEVELS = 5
DEFAULT_BATCH_LINES = 2_000
DEFAULT_FLUSH_ROWS = 200_000
DEFAULT_COMPRESSION_LEVEL = 11
FINAL_DATASET_NAME = "l2_book"
VALIDATION_DATASET_NAME = "validation"
MANIFEST_NAME = "manifest.json"

TOKEN_RAW_SCHEMA: dict[str, pl.DataType] = {
    "local_ts": pl.Float64,
    "source": pl.Utf8,
    "asset_id": pl.Utf8,
    "payload": pl.Struct(
        {
            "market": pl.Utf8,
            "asset_id": pl.Utf8,
            "timestamp": pl.Utf8,
            "hash": pl.Utf8,
            "bids": pl.List(pl.Struct({"price": pl.Utf8, "size": pl.Utf8})),
            "asks": pl.List(pl.Struct({"price": pl.Utf8, "size": pl.Utf8})),
            "tick_size": pl.Utf8,
            "event_type": pl.Utf8,
            "last_trade_price": pl.Utf8,
        }
    ),
}

DELTA_RAW_SCHEMA: dict[str, pl.DataType] = {
    "local_ts": pl.Float64,
    "source": pl.Utf8,
    "asset_id": pl.Utf8,
    "payload": pl.Struct(
        {
            "market": pl.Utf8,
            "price_changes": pl.List(
                pl.Struct(
                    {
                        "asset_id": pl.Utf8,
                        "price": pl.Utf8,
                        "size": pl.Utf8,
                        "side": pl.Utf8,
                        "hash": pl.Utf8,
                        "best_bid": pl.Utf8,
                        "best_ask": pl.Utf8,
                    }
                )
            ),
            "timestamp": pl.Utf8,
            "event_type": pl.Utf8,
        }
    ),
}

FINAL_BUFFER_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "market_id": pl.Utf8,
    "event_id": pl.Utf8,
    "token_id": pl.Utf8,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "bid_depth": pl.Float64,
    "ask_depth": pl.Float64,
}

FINAL_PARQUET_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ms", "UTC"),
    "market_id": pl.Utf8,
    "event_id": pl.Utf8,
    "token_id": pl.Utf8,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "bid_depth": pl.Float64,
    "ask_depth": pl.Float64,
}

VALIDATION_BUFFER_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "market_id": pl.Utf8,
    "event_id": pl.Utf8,
    "token_id": pl.Utf8,
    "reason": pl.Utf8,
    "event_source": pl.Utf8,
    "details": pl.Utf8,
}

VALIDATION_PARQUET_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ms", "UTC"),
    "market_id": pl.Utf8,
    "event_id": pl.Utf8,
    "token_id": pl.Utf8,
    "reason": pl.Utf8,
    "event_source": pl.Utf8,
    "details": pl.Utf8,
}


@dataclass(frozen=True, slots=True)
class MarketMetadata:
    market_id: str
    event_id: str
    yes_asset_id: str
    no_asset_id: str

    def token_side(self, asset_id: str) -> str | None:
        if asset_id == self.yes_asset_id:
            return "YES"
        if asset_id == self.no_asset_id:
            return "NO"
        return None


@dataclass(frozen=True, slots=True)
class DeltaChange:
    asset_id: str
    token_id: str
    side: str
    price: float
    size: float


@dataclass(order=True, slots=True)
class StreamEvent:
    timestamp_ms: int
    source_rank: int
    ordinal: int
    event_source: str = field(compare=False)
    market_id: str = field(compare=False)
    event_id: str = field(compare=False)
    asset_id: str | None = field(compare=False, default=None)
    token_id: str | None = field(compare=False, default=None)
    bids: tuple[tuple[float, float], ...] = field(compare=False, default=())
    asks: tuple[tuple[float, float], ...] = field(compare=False, default=())
    delta_changes: tuple[DeltaChange, ...] = field(compare=False, default=())


@dataclass(slots=True)
class BookSummary:
    best_bid: float | None
    best_ask: float | None
    bid_depth: float
    ask_depth: float


@dataclass(slots=True)
class BookState:
    bids: SortedDict = field(default_factory=SortedDict)
    asks: SortedDict = field(default_factory=SortedDict)
    seeded: bool = False

    def apply_snapshot(
        self,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
    ) -> None:
        self.bids.clear()
        self.asks.clear()
        for price, size in bids:
            if price > 0 and size > 0:
                self.bids[-price] = size
        for price, size in asks:
            if price > 0 and size > 0:
                self.asks[price] = size
        self.seeded = True

    def apply_delta(self, *, side: str, price: float, size: float) -> None:
        if side not in {"BUY", "SELL"} or price <= 0:
            return
        if side == "BUY":
            key = -price
            target = self.bids
        else:
            key = price
            target = self.asks
        if size <= 0:
            target.pop(key, None)
        else:
            target[key] = size

    def summary(self, *, depth_levels: int = DEPTH_LEVELS) -> BookSummary:
        best_bid = -self.bids.peekitem(0)[0] if self.bids else None
        best_ask = self.asks.peekitem(0)[0] if self.asks else None

        bid_depth = 0.0
        for offset, (key, size) in enumerate(self.bids.items()):
            if offset >= depth_levels:
                break
            bid_depth += (-key) * size

        ask_depth = 0.0
        for offset, (price, size) in enumerate(self.asks.items()):
            if offset >= depth_levels:
                break
            ask_depth += price * size

        return BookSummary(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=round(bid_depth, 8),
            ask_depth=round(ask_depth, 8),
        )


@dataclass(slots=True)
class RunStats:
    metadata_rows_loaded: int = 0
    metadata_rows_rejected: Counter[str] = field(default_factory=Counter)
    days_processed: int = 0
    markets_considered: int = 0
    markets_completed: int = 0
    markets_skipped: Counter[str] = field(default_factory=Counter)
    raw_records_read: Counter[str] = field(default_factory=Counter)
    raw_records_parsed: Counter[str] = field(default_factory=Counter)
    raw_records_malformed: Counter[str] = field(default_factory=Counter)
    raw_batches_salvaged: Counter[str] = field(default_factory=Counter)
    output_rows: int = 0
    rejected_rows: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "metadata_rows_loaded": self.metadata_rows_loaded,
            "metadata_rows_rejected": dict(self.metadata_rows_rejected),
            "days_processed": self.days_processed,
            "markets_considered": self.markets_considered,
            "markets_completed": self.markets_completed,
            "markets_skipped": dict(self.markets_skipped),
            "raw_records_read": dict(self.raw_records_read),
            "raw_records_parsed": dict(self.raw_records_parsed),
            "raw_records_malformed": dict(self.raw_records_malformed),
            "raw_batches_salvaged": dict(self.raw_batches_salvaged),
            "output_rows": self.output_rows,
            "rejected_rows": self.rejected_rows,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Polymarket L2 JSONL into a strict Zstd-compressed Parquet lake.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw_ticks"),
        help="Root directory containing raw_ticks/YYYY-MM-DD/*.jsonl partitions.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        action="append",
        required=True,
        help="One or more metadata JSON files that resolve market_id, event_id, YES token, and NO token.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for the final Parquet lake and validation outputs.",
    )
    parser.add_argument(
        "--day",
        action="append",
        dest="days",
        default=[],
        help="Optional YYYY-MM-DD partition to process. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--market-id",
        action="append",
        default=[],
        help="Optional market_id filter. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=DEFAULT_BATCH_LINES,
        help="Maximum raw JSONL lines parsed into Polars per batch.",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=DEFAULT_FLUSH_ROWS,
        help="Maximum buffered final rows before a Parquet flush.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help="Zstd compression level for output Parquet files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty output root.",
    )
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _timestamp_ms(exchange_ts: Any, local_ts: Any) -> int | None:
    for value in (exchange_ts, local_ts):
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 1e15:
            numeric /= 1_000.0
        elif numeric <= 1e12:
            numeric *= 1_000.0
        return int(numeric)
    return None


def _parse_levels(raw_levels: Any) -> tuple[tuple[float, float], ...]:
    levels: list[tuple[float, float]] = []
    if not isinstance(raw_levels, list):
        return ()
    for level in raw_levels:
        if not isinstance(level, Mapping):
            continue
        try:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
        except (TypeError, ValueError):
            continue
        if price > 0 and size > 0:
            levels.append((price, size))
    return tuple(levels)


def _iter_metadata_candidate_rows(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_metadata_candidate_rows(item)
        return
    if not isinstance(payload, dict):
        return

    marker_keys = {
        "market_id",
        "market",
        "conditionId",
        "condition_id",
        "yes_id",
        "yes_token_id",
        "no_id",
        "no_token_id",
        "clobTokenIds",
        "eventId",
        "event_id",
    }
    if marker_keys.intersection(payload):
        yield payload

    for key in (
        "data",
        "rows",
        "results",
        "items",
        "entries",
        "markets",
        "selected_markets",
        "targets",
        "entry",
    ):
        nested = payload.get(key)
        if isinstance(nested, (list, dict)):
            yield from _iter_metadata_candidate_rows(nested)


def _metadata_from_row(row: Mapping[str, Any]) -> MarketMetadata | None:
    market_id = _clean_text(
        row.get("market_id")
        or row.get("market")
        or row.get("conditionId")
        or row.get("condition_id")
    ).lower()
    if not market_id.startswith("0x"):
        candidate_id = _clean_text(row.get("id")).lower()
        market_id = candidate_id if candidate_id.startswith("0x") else market_id
    if not market_id:
        return None

    event_id = _clean_text(row.get("event_id") or row.get("eventId"))
    if not event_id:
        events = row.get("events") or []
        if isinstance(events, list) and events:
            first = events[0] if isinstance(events[0], Mapping) else {}
            event_id = _clean_text(first.get("id"))
    if not event_id:
        return None

    yes_asset_id = _clean_text(
        row.get("yes_id") or row.get("yes_token_id") or row.get("yesTokenId")
    )
    no_asset_id = _clean_text(
        row.get("no_id") or row.get("no_token_id") or row.get("noTokenId")
    )
    if not yes_asset_id or not no_asset_id:
        token_ids = [
            _clean_text(value)
            for value in _parse_listish(row.get("clobTokenIds"))
            if _clean_text(value)
        ]
        if len(token_ids) >= 2:
            yes_asset_id = yes_asset_id or token_ids[0]
            no_asset_id = no_asset_id or token_ids[1]
    if not yes_asset_id or not no_asset_id:
        return None

    return MarketMetadata(
        market_id=market_id,
        event_id=event_id,
        yes_asset_id=yes_asset_id,
        no_asset_id=no_asset_id,
    )


def load_metadata(paths: Sequence[Path], stats: RunStats) -> dict[str, MarketMetadata]:
    by_market: dict[str, MarketMetadata] = {}

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _iter_metadata_candidate_rows(payload):
            meta = _metadata_from_row(row)
            if meta is None:
                stats.metadata_rows_rejected["missing_required_fields"] += 1
                continue
            current = by_market.get(meta.market_id)
            if current is not None and current != meta:
                raise ValueError(
                    f"Conflicting metadata for market {meta.market_id}: {current} vs {meta}"
                )
            by_market[meta.market_id] = meta

    token_owner: dict[str, str] = {}
    for meta in by_market.values():
        for asset_id in (meta.yes_asset_id, meta.no_asset_id):
            owner = token_owner.get(asset_id)
            if owner is not None and owner != meta.market_id:
                raise ValueError(
                    f"Token {asset_id} is assigned to multiple markets: {owner} and {meta.market_id}"
                )
            token_owner[asset_id] = meta.market_id

    stats.metadata_rows_loaded = len(by_market)
    return by_market


def _ensure_output_root(output_root: Path, *, force: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if force:
        return
    if any(output_root.iterdir()):
        raise SystemExit(
            f"Output root {output_root} is not empty. Use --force or choose a new directory."
        )


def _discover_days(raw_root: Path, selected_days: Sequence[str]) -> list[str]:
    if selected_days:
        return sorted({_clean_text(day) for day in selected_days if _clean_text(day)})
    return sorted(child.name for child in raw_root.iterdir() if child.is_dir())


def _iter_ndjson_batches(
    path: Path,
    *,
    schema: Mapping[str, pl.DataType],
    batch_lines: int,
    stats: RunStats,
    stream_key: str,
) -> Iterator[tuple[pl.DataFrame, int]]:
    def _empty_frame() -> pl.DataFrame:
        return pl.DataFrame(schema=schema)

    def _read_ndjson(raw_lines: list[bytes]) -> pl.DataFrame:
        return pl.read_ndjson(
            BytesIO(b"".join(raw_lines)),
            schema=schema,
            ignore_errors=True,
            low_memory=True,
            rechunk=False,
        )

    def _frame_from_decoded_records(decoded_records: list[dict[str, Any]]) -> tuple[pl.DataFrame, int]:
        if not decoded_records:
            return _empty_frame(), 0
        try:
            return pl.from_dicts(decoded_records, schema=schema, strict=False), 0
        except Exception:
            frames: list[pl.DataFrame] = []
            dropped_records = 0
            for record in decoded_records:
                try:
                    frames.append(pl.from_dicts([record], schema=schema, strict=False))
                except Exception:
                    dropped_records += 1
            if not frames:
                return _empty_frame(), dropped_records
            return pl.concat(frames, how="vertical_relaxed"), dropped_records

    with path.open("rb") as handle:
        while True:
            lines = list(islice(handle, batch_lines))
            if not lines:
                return

            try:
                frame = _read_ndjson(lines)
            except pl.exceptions.ComputeError:
                decoded_records: list[dict[str, Any]] = []
                malformed_lines = 0
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = json.loads(stripped)
                    except Exception:
                        malformed_lines += 1
                        continue
                    if not isinstance(parsed, dict):
                        malformed_lines += 1
                        continue
                    decoded_records.append(parsed)

                frame, dropped_records = _frame_from_decoded_records(decoded_records)
                malformed_lines += dropped_records
                stats.raw_batches_salvaged[stream_key] += 1
                if malformed_lines:
                    stats.raw_records_malformed[stream_key] += malformed_lines
            yield frame, len(lines)


def iter_snapshot_events(
    path: Path,
    *,
    metadata: MarketMetadata,
    token_id: str,
    batch_lines: int,
    stats: RunStats,
) -> Iterator[StreamEvent]:
    ordinal = count()
    stream_key = f"snapshot:{token_id.lower()}"

    for frame, raw_count in _iter_ndjson_batches(
        path,
        schema=TOKEN_RAW_SCHEMA,
        batch_lines=batch_lines,
        stats=stats,
        stream_key=stream_key,
    ):
        stats.raw_records_read[stream_key] += raw_count
        stats.raw_records_parsed[stream_key] += frame.height

        projected = frame.select(
            pl.col("local_ts"),
            pl.col("payload").struct.field("market").alias("market"),
            pl.col("payload").struct.field("asset_id").alias("payload_asset_id"),
            pl.col("payload").struct.field("timestamp").alias("exchange_ts"),
            pl.col("payload").struct.field("event_type").alias("event_type"),
            pl.col("payload").struct.field("bids").alias("bids"),
            pl.col("payload").struct.field("asks").alias("asks"),
        )

        for row in projected.iter_rows(named=True):
            if _clean_text(row["event_type"]).lower() != "book":
                continue
            timestamp_ms = _timestamp_ms(row["exchange_ts"], row["local_ts"])
            if timestamp_ms is None:
                continue
            asset_id = _clean_text(row["payload_asset_id"])
            yield StreamEvent(
                timestamp_ms=timestamp_ms,
                source_rank=0,
                ordinal=next(ordinal),
                event_source="snapshot",
                market_id=_clean_text(row["market"]).lower() or metadata.market_id,
                event_id=metadata.event_id,
                asset_id=asset_id,
                token_id=token_id,
                bids=_parse_levels(row["bids"]),
                asks=_parse_levels(row["asks"]),
            )


def iter_delta_events(
    path: Path,
    *,
    metadata: MarketMetadata,
    batch_lines: int,
    stats: RunStats,
) -> Iterator[StreamEvent]:
    ordinal = count()
    stream_key = "delta"

    for frame, raw_count in _iter_ndjson_batches(
        path,
        schema=DELTA_RAW_SCHEMA,
        batch_lines=batch_lines,
        stats=stats,
        stream_key=stream_key,
    ):
        stats.raw_records_read[stream_key] += raw_count
        stats.raw_records_parsed[stream_key] += frame.height

        projected = frame.select(
            pl.col("local_ts"),
            pl.col("payload").struct.field("market").alias("market"),
            pl.col("payload").struct.field("timestamp").alias("exchange_ts"),
            pl.col("payload").struct.field("event_type").alias("event_type"),
            pl.col("payload").struct.field("price_changes").alias("price_changes"),
        )

        for row in projected.iter_rows(named=True):
            if _clean_text(row["event_type"]).lower() != "price_change":
                continue
            timestamp_ms = _timestamp_ms(row["exchange_ts"], row["local_ts"])
            if timestamp_ms is None:
                continue

            changes: list[DeltaChange] = []
            for raw_change in row["price_changes"] or []:
                if not isinstance(raw_change, Mapping):
                    continue
                asset_id = _clean_text(raw_change.get("asset_id"))
                token_id = metadata.token_side(asset_id)
                if token_id is None:
                    continue
                try:
                    price = float(raw_change.get("price", 0))
                    size = float(raw_change.get("size", 0))
                except (TypeError, ValueError):
                    continue
                side = _clean_text(raw_change.get("side")).upper()
                changes.append(
                    DeltaChange(
                        asset_id=asset_id,
                        token_id=token_id,
                        side=side,
                        price=price,
                        size=size,
                    )
                )

            if not changes:
                continue

            yield StreamEvent(
                timestamp_ms=timestamp_ms,
                source_rank=1,
                ordinal=next(ordinal),
                event_source="delta",
                market_id=_clean_text(row["market"]).lower() or metadata.market_id,
                event_id=metadata.event_id,
                delta_changes=tuple(changes),
            )


def merge_event_streams(iterables: Sequence[Iterator[StreamEvent]]) -> Iterator[StreamEvent]:
    heap: list[tuple[StreamEvent, int, Iterator[StreamEvent]]] = []
    for index, iterable in enumerate(iterables):
        iterator = iter(iterable)
        try:
            first = next(iterator)
        except StopIteration:
            continue
        heappush(heap, (first, index, iterator))

    while heap:
        event, index, iterator = heappop(heap)
        yield event
        try:
            nxt = next(iterator)
        except StopIteration:
            continue
        heappush(heap, (nxt, index, iterator))


def _summary_reason(summary: BookSummary) -> str | None:
    if summary.best_bid is None or summary.best_ask is None:
        return "missing_bbo"
    if summary.best_bid <= 0 or summary.best_ask <= 0:
        return "non_positive_bbo"
    if summary.best_bid >= summary.best_ask:
        return "crossed_book"
    if summary.best_bid > 1.0 or summary.best_ask > 1.0:
        return "price_out_of_bounds"
    if summary.bid_depth <= 0 or summary.ask_depth <= 0:
        return "missing_depth"
    return None


def _datetime_hour_parts(timestamp_ms: int) -> tuple[str, str]:
    instant = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
    return instant.strftime("%Y-%m-%d"), instant.strftime("%H")


def _final_rows_to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=FINAL_PARQUET_SCHEMA)
    frame = pl.from_dicts(rows, schema=FINAL_BUFFER_SCHEMA)
    return frame.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
    ).select(list(FINAL_PARQUET_SCHEMA))


def _validation_rows_to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=VALIDATION_PARQUET_SCHEMA)
    frame = pl.from_dicts(rows, schema=VALIDATION_BUFFER_SCHEMA)
    return frame.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
    ).select(list(VALIDATION_PARQUET_SCHEMA))


def _write_buffer(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    dataset_name: str,
    market_id: str,
    day: str,
    hour: str,
    compression_level: int,
    kind: str,
    on_file_written: Callable[[Path], None] | None = None,
) -> None:
    if not rows:
        return

    target_dir = output_root / dataset_name / f"date={day}" / f"hour={hour}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{kind}-{market_id}-{uuid4().hex}.parquet"

    if dataset_name == FINAL_DATASET_NAME:
        frame = _final_rows_to_frame(rows)
    else:
        frame = _validation_rows_to_frame(rows)

    frame.write_parquet(
        target_path,
        compression="zstd",
        compression_level=compression_level,
        mkdir=True,
        statistics=True,
        use_pyarrow=False,
    )
    if on_file_written is not None:
        on_file_written(target_path)
    rows.clear()


def _append_reject(
    buffer: list[dict[str, Any]],
    *,
    timestamp_ms: int,
    market_id: str,
    event_id: str,
    token_id: str,
    reason: str,
    event_source: str,
    details: str,
) -> None:
    buffer.append(
        {
            "timestamp": timestamp_ms,
            "market_id": market_id,
            "event_id": event_id,
            "token_id": token_id,
            "reason": reason,
            "event_source": event_source,
            "details": details,
        }
    )


def process_market_day(
    *,
    day: str,
    day_dir: Path,
    metadata: MarketMetadata,
    output_root: Path,
    batch_lines: int,
    flush_rows: int,
    compression_level: int,
    stats: RunStats,
    on_file_written: Callable[[Path], None] | None = None,
) -> None:
    yes_path = day_dir / f"{metadata.yes_asset_id}.jsonl"
    no_path = day_dir / f"{metadata.no_asset_id}.jsonl"
    delta_path = day_dir / f"{metadata.market_id}.jsonl"

    missing: list[str] = [
        name
        for name, path in (("yes_snapshot", yes_path), ("no_snapshot", no_path), ("market_delta", delta_path))
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        stats.markets_skipped["missing_required_file"] += 1

        validation_rows: list[dict[str, Any]] = []
        _append_reject(
            validation_rows,
            timestamp_ms=int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000),
            market_id=metadata.market_id,
            event_id=metadata.event_id,
            token_id="MARKET",
            reason="missing_required_file",
            event_source="preflight",
            details=",".join(missing),
        )
        _write_buffer(
            rows=validation_rows,
            output_root=output_root,
            dataset_name=VALIDATION_DATASET_NAME,
            market_id=metadata.market_id,
            day=day,
            hour="00",
            compression_level=compression_level,
            kind="rejects",
            on_file_written=on_file_written,
        )
        stats.rejected_rows += 1
        return

    books = {
        metadata.yes_asset_id: BookState(),
        metadata.no_asset_id: BookState(),
    }

    final_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    current_partition: tuple[str, str] | None = None

    def flush_buffers(force: bool = False) -> None:
        nonlocal current_partition
        if current_partition is None:
            return
        if not force and not final_rows and not validation_rows:
            return
        day_part, hour_part = current_partition
        _write_buffer(
            rows=final_rows,
            output_root=output_root,
            dataset_name=FINAL_DATASET_NAME,
            market_id=metadata.market_id,
            day=day_part,
            hour=hour_part,
            compression_level=compression_level,
            kind="part",
            on_file_written=on_file_written,
        )
        _write_buffer(
            rows=validation_rows,
            output_root=output_root,
            dataset_name=VALIDATION_DATASET_NAME,
            market_id=metadata.market_id,
            day=day_part,
            hour=hour_part,
            compression_level=compression_level,
            kind="rejects",
            on_file_written=on_file_written,
        )

    streams = (
        iter_snapshot_events(
            yes_path,
            metadata=metadata,
            token_id="YES",
            batch_lines=batch_lines,
            stats=stats,
        ),
        iter_snapshot_events(
            no_path,
            metadata=metadata,
            token_id="NO",
            batch_lines=batch_lines,
            stats=stats,
        ),
        iter_delta_events(
            delta_path,
            metadata=metadata,
            batch_lines=batch_lines,
            stats=stats,
        ),
    )

    for event in merge_event_streams(streams):
        partition = _datetime_hour_parts(event.timestamp_ms)
        if current_partition is None:
            current_partition = partition
        elif partition != current_partition:
            flush_buffers(force=True)
            current_partition = partition

        if event.event_source == "snapshot":
            assert event.asset_id is not None
            target_book = books[event.asset_id]
            target_book.apply_snapshot(event.bids, event.asks)
            if event.market_id != metadata.market_id:
                _append_reject(
                    validation_rows,
                    timestamp_ms=event.timestamp_ms,
                    market_id=metadata.market_id,
                    event_id=metadata.event_id,
                    token_id=event.token_id or "MARKET",
                    reason="market_id_mismatch",
                    event_source=event.event_source,
                    details=f"observed={event.market_id}",
                )
                stats.rejected_rows += 1
                continue
        else:
            if event.market_id != metadata.market_id:
                _append_reject(
                    validation_rows,
                    timestamp_ms=event.timestamp_ms,
                    market_id=metadata.market_id,
                    event_id=metadata.event_id,
                    token_id="MARKET",
                    reason="market_id_mismatch",
                    event_source=event.event_source,
                    details=f"observed={event.market_id}",
                )
                stats.rejected_rows += 1
                continue

            if any(not books[change.asset_id].seeded for change in event.delta_changes):
                _append_reject(
                    validation_rows,
                    timestamp_ms=event.timestamp_ms,
                    market_id=metadata.market_id,
                    event_id=metadata.event_id,
                    token_id="MARKET",
                    reason="delta_before_snapshot",
                    event_source=event.event_source,
                    details="delta arrived before both token books were seeded",
                )
                stats.rejected_rows += 1
                continue

            for change in event.delta_changes:
                books[change.asset_id].apply_delta(
                    side=change.side,
                    price=change.price,
                    size=change.size,
                )

        yes_book = books[metadata.yes_asset_id]
        no_book = books[metadata.no_asset_id]
        if not yes_book.seeded or not no_book.seeded:
            _append_reject(
                validation_rows,
                timestamp_ms=event.timestamp_ms,
                market_id=metadata.market_id,
                event_id=metadata.event_id,
                token_id="MARKET",
                reason="pair_not_seeded",
                event_source=event.event_source,
                details="both YES and NO books must be seeded before emission",
            )
            stats.rejected_rows += 1
            continue

        yes_summary = yes_book.summary(depth_levels=DEPTH_LEVELS)
        no_summary = no_book.summary(depth_levels=DEPTH_LEVELS)
        yes_reason = _summary_reason(yes_summary)
        no_reason = _summary_reason(no_summary)
        if yes_reason or no_reason:
            reasons = []
            if yes_reason:
                reasons.append(f"YES:{yes_reason}")
            if no_reason:
                reasons.append(f"NO:{no_reason}")
            _append_reject(
                validation_rows,
                timestamp_ms=event.timestamp_ms,
                market_id=metadata.market_id,
                event_id=metadata.event_id,
                token_id="MARKET",
                reason="pair_incomplete",
                event_source=event.event_source,
                details=";".join(reasons),
            )
            stats.rejected_rows += 1
            continue

        final_rows.append(
            {
                "timestamp": event.timestamp_ms,
                "market_id": metadata.market_id,
                "event_id": metadata.event_id,
                "token_id": "YES",
                "best_bid": yes_summary.best_bid,
                "best_ask": yes_summary.best_ask,
                "bid_depth": yes_summary.bid_depth,
                "ask_depth": yes_summary.ask_depth,
            }
        )
        final_rows.append(
            {
                "timestamp": event.timestamp_ms,
                "market_id": metadata.market_id,
                "event_id": metadata.event_id,
                "token_id": "NO",
                "best_bid": no_summary.best_bid,
                "best_ask": no_summary.best_ask,
                "bid_depth": no_summary.bid_depth,
                "ask_depth": no_summary.ask_depth,
            }
        )
        stats.output_rows += 2

        if len(final_rows) >= flush_rows or len(validation_rows) >= flush_rows:
            flush_buffers(force=True)

    flush_buffers(force=True)
    stats.markets_completed += 1


def main() -> int:
    args = parse_args()
    _ensure_output_root(args.output_root, force=args.force)

    stats = RunStats()
    metadata_by_market = load_metadata(args.metadata, stats)
    market_filter = {value.lower() for value in args.market_id if _clean_text(value)}
    days = _discover_days(args.raw_root, args.days)

    for day in days:
        day_dir = args.raw_root / day
        if not day_dir.exists():
            stats.markets_skipped["missing_day_partition"] += 1
            continue

        stats.days_processed += 1
        available = {path.stem for path in day_dir.glob("*.jsonl")}
        for metadata in metadata_by_market.values():
            if market_filter and metadata.market_id not in market_filter:
                continue
            if metadata.market_id not in available:
                continue
            stats.markets_considered += 1
            process_market_day(
                day=day,
                day_dir=day_dir,
                metadata=metadata,
                output_root=args.output_root,
                batch_lines=args.batch_lines,
                flush_rows=args.flush_rows,
                compression_level=args.compression_level,
                stats=stats,
            )

    manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "days": days,
        "depth_levels": DEPTH_LEVELS,
        "strict_schema": {
            "timestamp": "Datetime(ms, UTC)",
            "market_id": "Utf8",
            "event_id": "Utf8",
            "token_id": "Utf8[YES|NO]",
            "best_bid": "Float64",
            "best_ask": "Float64",
            "bid_depth": "Float64 top-5 notional",
            "ask_depth": "Float64 top-5 notional",
        },
        "stats": stats.to_json(),
    }
    manifest_path = args.output_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"manifest": str(manifest_path), **stats.to_json()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())