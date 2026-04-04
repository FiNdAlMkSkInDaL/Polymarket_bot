#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import websockets
import websockets.exceptions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.logger import get_logger, setup_logging
from src.data.orderbook import OrderbookTracker
from src.data.market_discovery import MarketInfo, fetch_active_markets


log = get_logger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "l2_book_live"
LEGACY_OUTPUT_DIR_NAME = "l2_archive"
DEFAULT_FLUSH_ROWS = 10_000
DEFAULT_FLUSH_SECONDS = 300.0
DEFAULT_QUEUE_SIZE = 100_000
DEFAULT_MARKET_LIMIT = 200
DEFAULT_MAX_ASSETS_PER_SOCKET = 50
DEFAULT_CONNECT_STAGGER_SECONDS = 0.25
DEFAULT_MIN_FREE_GB = 10.0
DEFAULT_UNIVERSE_REFRESH_SECONDS = 7_200.0
DEFAULT_HEARTBEAT_SECONDS = 900.0
WRITER_STATE_DIR_NAME = "_state"
HANDOFF_FILE_NAME = "writer_handoff.json"

DELTA_EVENTS = frozenset(("price_change", "book_delta", "delta"))
SNAPSHOT_EVENTS = frozenset(("book", "snapshot", "book_snapshot"))
DICTIONARY_COLUMNS = ["market_id", "event_id", "token_id"]
PARQUET_COLUMNS = [
    "timestamp",
    "market_id",
    "event_id",
    "token_id",
    "best_bid",
    "best_ask",
    "bid_depth",
    "ask_depth",
]
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("token_id", pa.string(), nullable=False),
        pa.field("best_bid", pa.float64(), nullable=False),
        pa.field("best_ask", pa.float64(), nullable=False),
        pa.field("bid_depth", pa.float64(), nullable=False),
        pa.field("ask_depth", pa.float64(), nullable=False),
    ]
)


class LowDiskSpaceError(RuntimeError):
    """Raised when the compressor hits the configured free-space floor."""


@dataclass(frozen=True, slots=True)
class AssetSubscription:
    asset_id: str
    market_id: str
    event_id: str
    token_id: str
    yes_asset_id: str
    no_asset_id: str
    question: str
    daily_volume_usd: float

    @property
    def outcome(self) -> str:
        return self.token_id


@dataclass(frozen=True, slots=True)
class RotationBucket:
    bucket_id: str
    date_str: str
    hour_str: str | None
    file_stem: str


@dataclass(frozen=True, slots=True)
class BookSummary:
    best_bid: float | None
    best_ask: float | None
    bid_depth: float
    ask_depth: float


class MarketBookState:
    def __init__(
        self,
        *,
        market_id: str,
        event_id: str,
        yes_asset_id: str,
        no_asset_id: str,
    ) -> None:
        self.market_id = market_id
        self.event_id = event_id
        self.yes_asset_id = yes_asset_id
        self.no_asset_id = no_asset_id
        self.yes_book = OrderbookTracker(yes_asset_id)
        self.no_book = OrderbookTracker(no_asset_id)
        self.seeded_assets: set[str] = set()

    @property
    def is_seeded(self) -> bool:
        return (
            self.yes_asset_id in self.seeded_assets
            and self.no_asset_id in self.seeded_assets
        )

    def tracker_for(self, asset_id: str) -> OrderbookTracker | None:
        if asset_id == self.yes_asset_id:
            return self.yes_book
        if asset_id == self.no_asset_id:
            return self.no_book
        return None


class ShutdownSignalMonitor:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None
        self._loop_handlers: list[signal.Signals] = []
        self._previous_handlers: dict[signal.Signals, Any] = {}
        self.reason: str | None = None

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()
        for current_signal in self._iter_signals():
            try:
                loop.add_signal_handler(
                    current_signal,
                    self.request_shutdown,
                    f"signal:{current_signal.name}",
                )
                self._loop_handlers.append(current_signal)
            except (NotImplementedError, RuntimeError):
                try:
                    self._previous_handlers[current_signal] = signal.getsignal(current_signal)
                    signal.signal(current_signal, self._fallback_handler)
                except (OSError, RuntimeError, ValueError):
                    continue

    def restore(self) -> None:
        if self._loop is not None:
            for current_signal in self._loop_handlers:
                self._loop.remove_signal_handler(current_signal)
        for current_signal, previous_handler in self._previous_handlers.items():
            try:
                signal.signal(current_signal, previous_handler)
            except (OSError, RuntimeError, ValueError):
                continue
        self._loop_handlers.clear()
        self._previous_handlers.clear()

    async def wait(self) -> None:
        if self._event is None:
            raise RuntimeError("Shutdown monitor must be installed before waiting")
        await self._event.wait()

    def request_shutdown(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
            log.info("tick_compressor_shutdown_requested", reason=reason)
        if self._event is not None and not self._event.is_set():
            self._event.set()

    def _fallback_handler(self, signum: int, _frame: Any) -> None:
        if self._loop is None:
            return
        try:
            current_signal = signal.Signals(signum)
            reason = f"signal:{current_signal.name}"
        except ValueError:
            reason = f"signal:{signum}"
        self._loop.call_soon_threadsafe(self.request_shutdown, reason)

    @staticmethod
    def _iter_signals() -> tuple[signal.Signals, ...]:
        candidates: list[signal.Signals] = [signal.SIGINT]
        sigterm = getattr(signal, "SIGTERM", None)
        if sigterm is not None:
            candidates.append(sigterm)
        return tuple(candidates)


def _log_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    log.error(
        "tick_background_task_failed",
        task_name=task.get_name(),
        error=repr(exc),
        exc_info=exc,
    )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.tmp"
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for _ in range(10):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    temp_path.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture live Polymarket L2 updates into compressed Parquet chunks.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root directory for compressed Parquet chunks.")
    parser.add_argument("--market-limit", type=int, default=DEFAULT_MARKET_LIMIT, help="Number of active tradeable markets to subscribe to on each universe refresh.")
    parser.add_argument("--min-volume", type=float, default=None, help="Optional minimum daily volume override for tradeable-universe discovery.")
    parser.add_argument("--min-days-to-resolution", type=int, default=None, help="Optional minimum days-to-resolution override for tradeable-universe discovery.")
    parser.add_argument("--flush-rows", type=int, default=DEFAULT_FLUSH_ROWS, help="Maximum buffered rows before a flush to disk.")
    parser.add_argument("--flush-seconds", type=float, default=DEFAULT_FLUSH_SECONDS, help="Maximum wall-clock seconds between disk flushes.")
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE, help="Maximum in-memory row queue depth before websocket readers backpressure.")
    parser.add_argument("--rotation", choices=("hourly", "daily"), default="hourly", help="Chunk filename rotation cadence.")
    parser.add_argument("--compression", choices=("zstd", "snappy"), default="zstd", help="Parquet compression codec.")
    parser.add_argument("--max-assets-per-socket", type=int, default=DEFAULT_MAX_ASSETS_PER_SOCKET, help="Maximum subscribed token ids per websocket connection.")
    parser.add_argument("--connect-stagger-seconds", type=float, default=DEFAULT_CONNECT_STAGGER_SECONDS, help="Delay between launching websocket shards.")
    parser.add_argument("--ws-url", default=settings.clob_l2_ws_url, help="Polymarket L2 websocket URL.")
    parser.add_argument("--silence-timeout-seconds", type=float, default=settings.strategy.l2_silence_timeout_s, help="Close and reconnect a silent websocket after this many seconds.")
    parser.add_argument("--universe-refresh-seconds", type=float, default=DEFAULT_UNIVERSE_REFRESH_SECONDS, help="Seconds between active-universe refreshes. Set to 0 to disable periodic rebinding.")
    parser.add_argument("--heartbeat-seconds", type=float, default=DEFAULT_HEARTBEAT_SECONDS, help="Seconds between INFO-level heartbeat telemetry logs. Set to 0 to disable.")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB, help="Graceful low-disk stop threshold. Set to 0 to disable.")
    parser.add_argument("--log-dir", default="logs", help="Structured log directory.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def _normalize_exchange_ts(msg: dict[str, Any]) -> float | None:
    raw_value = msg.get("timestamp") or msg.get("server_timestamp") or msg.get("ts")
    if raw_value in (None, ""):
        return None
    try:
        timestamp = float(raw_value)
    except (TypeError, ValueError):
        return None
    if timestamp > 1e15:
        return timestamp / 1_000_000.0
    if timestamp > 1e12:
        return timestamp / 1_000.0
    return timestamp


def _event_type(msg: dict[str, Any]) -> str:
    return str(msg.get("event_type") or msg.get("type") or "").strip().lower()


def _timestamp_ms(msg: dict[str, Any], *, received_at: float) -> int:
    exchange_ts = _normalize_exchange_ts(msg)
    base_ts = exchange_ts if exchange_ts is not None else float(received_at)
    return int(round(base_ts * 1000.0))


def _normalize_delta_message(msg: dict[str, Any]) -> dict[str, Any]:
    if msg.get("changes") is not None or msg.get("data") is not None or msg.get("price") is not None:
        return msg
    if msg.get("price_changes") is not None:
        normalized = dict(msg)
        normalized["changes"] = msg.get("price_changes")
        return normalized
    return msg


def _book_summary(book: OrderbookTracker) -> BookSummary:
    bid_levels = book.levels("bid", 5)
    ask_levels = book.levels("ask", 5)
    best_bid = bid_levels[0].price if bid_levels else None
    best_ask = ask_levels[0].price if ask_levels else None
    bid_depth = sum(level.price * level.size for level in bid_levels)
    ask_depth = sum(level.price * level.size for level in ask_levels)
    return BookSummary(
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth=round(bid_depth, 8),
        ask_depth=round(ask_depth, 8),
    )


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


class LiveBookAssembler:
    def __init__(self, subscriptions_by_asset: dict[str, AssetSubscription]) -> None:
        self._subscriptions_by_asset = subscriptions_by_asset
        self._market_states: dict[str, MarketBookState] = {}
        self.processed_messages = 0
        self.ignored_messages = 0
        self.preseed_deltas = 0
        self.unpaired_events = 0
        self.invalid_pairs = 0
        self.emitted_rows = 0

    @property
    def market_count(self) -> int:
        return len(self._market_states)

    def process_message(self, msg: dict[str, Any], *, received_at: float) -> list[dict[str, Any]]:
        self.processed_messages += 1
        event_type = _event_type(msg)
        if event_type not in SNAPSHOT_EVENTS and event_type not in DELTA_EVENTS:
            self.ignored_messages += 1
            return []

        asset_id = str(msg.get("asset_id") or "").strip()
        if not asset_id:
            self.ignored_messages += 1
            return []

        subscription = self._subscriptions_by_asset.get(asset_id)
        if subscription is None:
            self.ignored_messages += 1
            return []

        state = self._market_state(subscription)
        tracker = state.tracker_for(asset_id)
        if tracker is None:
            self.ignored_messages += 1
            return []

        if event_type in SNAPSHOT_EVENTS:
            tracker.on_book_snapshot(msg)
            state.seeded_assets.add(asset_id)
        else:
            if asset_id not in state.seeded_assets:
                self.preseed_deltas += 1
                return []
            tracker.on_price_change(_normalize_delta_message(msg))

        if not state.is_seeded:
            self.unpaired_events += 1
            return []

        yes_summary = _book_summary(state.yes_book)
        no_summary = _book_summary(state.no_book)
        if _summary_reason(yes_summary) or _summary_reason(no_summary):
            self.invalid_pairs += 1
            return []

        timestamp_ms = _timestamp_ms(msg, received_at=received_at)
        rows = [
            {
                "timestamp": timestamp_ms,
                "market_id": state.market_id,
                "event_id": state.event_id,
                "token_id": "YES",
                "best_bid": yes_summary.best_bid,
                "best_ask": yes_summary.best_ask,
                "bid_depth": yes_summary.bid_depth,
                "ask_depth": yes_summary.ask_depth,
            },
            {
                "timestamp": timestamp_ms,
                "market_id": state.market_id,
                "event_id": state.event_id,
                "token_id": "NO",
                "best_bid": no_summary.best_bid,
                "best_ask": no_summary.best_ask,
                "bid_depth": no_summary.bid_depth,
                "ask_depth": no_summary.ask_depth,
            },
        ]
        self.emitted_rows += len(rows)
        return rows

    def _market_state(self, subscription: AssetSubscription) -> MarketBookState:
        current = self._market_states.get(subscription.market_id)
        if current is None or current.yes_asset_id != subscription.yes_asset_id or current.no_asset_id != subscription.no_asset_id:
            current = MarketBookState(
                market_id=subscription.market_id,
                event_id=subscription.event_id,
                yes_asset_id=subscription.yes_asset_id,
                no_asset_id=subscription.no_asset_id,
            )
            self._market_states[subscription.market_id] = current
        return current


def _chunk_assets(asset_ids: list[str], max_assets_per_socket: int) -> list[list[str]]:
    if max_assets_per_socket <= 0:
        raise ValueError("max_assets_per_socket must be a strictly positive int")
    if not asset_ids:
        return []
    return [
        asset_ids[index : index + max_assets_per_socket]
        for index in range(0, len(asset_ids), max_assets_per_socket)
    ]


def _effective_output_dir(output_dir: Path) -> Path:
    if output_dir.name != LEGACY_OUTPUT_DIR_NAME:
        return output_dir
    redirected = output_dir.with_name("l2_book_live")
    log.info(
        "tick_output_dir_redirected",
        requested_output_dir=str(output_dir),
        effective_output_dir=str(redirected),
    )
    return redirected


class ParquetTickWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        flush_rows: int,
        flush_seconds: float,
        queue_size: int,
        rotation: str,
        compression: str,
        min_free_gb: float,
    ) -> None:
        if flush_rows <= 0:
            raise ValueError("flush_rows must be a strictly positive int")
        if flush_seconds <= 0:
            raise ValueError("flush_seconds must be a strictly positive float")
        if queue_size <= 0:
            raise ValueError("queue_size must be a strictly positive int")
        if rotation not in {"hourly", "daily"}:
            raise ValueError("rotation must be 'hourly' or 'daily'")
        if compression not in {"zstd", "snappy"}:
            raise ValueError("compression must be 'zstd' or 'snappy'")
        if min_free_gb < 0:
            raise ValueError("min_free_gb must be >= 0")

        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._flush_rows = flush_rows
        self._flush_seconds = flush_seconds
        self._rotation = rotation
        self._compression = compression
        self._min_free_gb = min_free_gb
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self._buffer: list[dict[str, Any]] = []
        self._running = False
        self._last_flush_monotonic = time.monotonic()
        self._rows_written = 0
        self._files_written = 0
        self._bytes_written = 0
        self._high_watermark = 0
        self._part_counters: dict[str, int] = {}
        self._session_id = uuid4().hex
        self._started_at_iso = _now_iso()
        self._state_dir = self._output_dir / WRITER_STATE_DIR_NAME
        self._handoff_path = self._state_dir / HANDOFF_FILE_NAME
        self._loaded_handoff = self._load_handoff_state()
        self._last_flush_min_ts_ms: int | None = None
        self._last_flush_max_ts_ms: int | None = None
        self._last_written_files: list[str] = []
        self._stop_reason: str | None = None

    async def enqueue(self, row: dict[str, Any]) -> None:
        await self._queue.put(row)
        self._high_watermark = max(self._high_watermark, self._queue.qsize())

    async def run(self) -> None:
        self._running = True
        if self._loaded_handoff is not None:
            previous_ts = self._loaded_handoff.get("last_flush_max_ts_ms")
            last_flush_age_s = None
            if previous_ts is not None:
                try:
                    last_flush_age_s = round(max(0.0, time.time() - (float(previous_ts) / 1000.0)), 3)
                except (TypeError, ValueError):
                    last_flush_age_s = None
            log_fn = log.warning if not bool(self._loaded_handoff.get("clean_shutdown")) else log.info
            log_fn(
                "tick_handoff_loaded",
                previous_session_id=self._loaded_handoff.get("session_id"),
                previous_status=self._loaded_handoff.get("status"),
                previous_clean_shutdown=bool(self._loaded_handoff.get("clean_shutdown")),
                previous_last_flush_max_ts_ms=previous_ts,
                previous_last_flush_age_s=last_flush_age_s,
                previous_rows_written=self._loaded_handoff.get("rows_written"),
                previous_files_written=self._loaded_handoff.get("files_written"),
            )
        self._write_handoff_state(status="running", clean_shutdown=False)
        log.info(
            "tick_writer_started",
            output_dir=str(self._output_dir),
            flush_rows=self._flush_rows,
            flush_seconds=self._flush_seconds,
            rotation=self._rotation,
            compression=self._compression,
            min_free_gb=self._min_free_gb,
            dataset_schema="l2_book_live",
        )
        try:
            while self._running:
                await self._consume_and_flush_once()
        except LowDiskSpaceError:
            self._stop_reason = self._stop_reason or "low_disk_space"
            raise
        except asyncio.CancelledError:
            self._stop_reason = self._stop_reason or "cancelled"
            raise
        except Exception:
            self._stop_reason = self._stop_reason or "writer_failure"
            raise
        finally:
            while True:
                try:
                    self._buffer.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if self._buffer:
                pending_rows = self._buffer
                self._buffer = []
                await asyncio.to_thread(self._flush_rows_sync, pending_rows)
            clean_shutdown = self._stop_reason not in {None, "writer_failure", "cancelled"}
            self._write_handoff_state(status="stopped", clean_shutdown=clean_shutdown)
            log.info(
                "tick_writer_stopped",
                rows_written=self._rows_written,
                files_written=self._files_written,
                bytes_written=self._bytes_written,
                queue_high_watermark=self._high_watermark,
                stop_reason=self._stop_reason,
            )

    def stop(self, *, reason: str | None = None) -> None:
        if reason is not None and self._stop_reason is None:
            self._stop_reason = reason
        self._running = False

    async def _consume_and_flush_once(self) -> None:
        elapsed = time.monotonic() - self._last_flush_monotonic
        timeout = max(0.0, self._flush_seconds - elapsed)

        if timeout > 0:
            try:
                row = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                self._buffer.append(row)
            except asyncio.TimeoutError:
                pass
        elif not self._buffer:
            try:
                row = self._queue.get_nowait()
                self._buffer.append(row)
            except asyncio.QueueEmpty:
                pass

        while len(self._buffer) < self._flush_rows:
            try:
                row = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._buffer.append(row)

        should_flush = self._buffer and (
            len(self._buffer) >= self._flush_rows
            or (time.monotonic() - self._last_flush_monotonic) >= self._flush_seconds
        )
        if not should_flush:
            return

        rows = self._buffer
        self._buffer = []
        await asyncio.to_thread(self._flush_rows_sync, rows)
        self._last_flush_monotonic = time.monotonic()

    def _flush_rows_sync(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._ensure_free_space()

        grouped_rows: dict[RotationBucket, list[dict[str, Any]]] = {}
        min_ts_ms: int | None = None
        max_ts_ms: int | None = None
        for row in rows:
            timestamp_ms = int(row["timestamp"])
            min_ts_ms = timestamp_ms if min_ts_ms is None else min(min_ts_ms, timestamp_ms)
            max_ts_ms = timestamp_ms if max_ts_ms is None else max(max_ts_ms, timestamp_ms)
            bucket = self._rotation_bucket(timestamp_ms)
            grouped_rows.setdefault(bucket, []).append(row)

        written_files: list[str] = []
        for bucket, bucket_rows in sorted(grouped_rows.items(), key=lambda item: item[0].bucket_id):
            file_path = self._next_file_path(bucket)
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            table = self._rows_to_arrow(bucket_rows)
            pq.write_table(
                table,
                temp_path,
                compression=self._compression,
                write_statistics=True,
                use_dictionary=DICTIONARY_COLUMNS,
            )
            temp_path.replace(file_path)
            file_size = file_path.stat().st_size
            self._rows_written += len(bucket_rows)
            self._files_written += 1
            self._bytes_written += file_size
            written_files.append(file_path.relative_to(self._output_dir).as_posix())
            log.info(
                "tick_parquet_chunk_written",
                file_path=str(file_path),
                rows=len(bucket_rows),
                bytes=file_size,
                compression=self._compression,
                rotation_bucket=bucket.bucket_id,
                dataset_schema="l2_book_live",
            )
        self._last_flush_min_ts_ms = min_ts_ms
        self._last_flush_max_ts_ms = max_ts_ms
        self._last_written_files = written_files
        self._write_handoff_state(status="running", clean_shutdown=False)

    def _load_handoff_state(self) -> dict[str, Any] | None:
        if not self._handoff_path.exists():
            return None
        try:
            return json.loads(self._handoff_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning(
                "tick_handoff_load_failed",
                handoff_path=str(self._handoff_path),
                error=str(exc),
            )
            return None

    def _write_handoff_state(self, *, status: str, clean_shutdown: bool) -> None:
        payload = {
            "schema": "tick_writer_handoff_v1",
            "status": status,
            "clean_shutdown": clean_shutdown,
            "session_id": self._session_id,
            "started_at": self._started_at_iso,
            "updated_at": _now_iso(),
            "output_dir": str(self._output_dir),
            "rotation": self._rotation,
            "compression": self._compression,
            "rows_written": self._rows_written,
            "files_written": self._files_written,
            "bytes_written": self._bytes_written,
            "queue_high_watermark": self._high_watermark,
            "last_flush_min_ts_ms": self._last_flush_min_ts_ms,
            "last_flush_max_ts_ms": self._last_flush_max_ts_ms,
            "last_written_files": list(self._last_written_files),
            "stop_reason": self._stop_reason,
        }
        if self._loaded_handoff is not None:
            payload["resume_from_session_id"] = self._loaded_handoff.get("session_id")
            payload["resume_from_last_flush_max_ts_ms"] = self._loaded_handoff.get("last_flush_max_ts_ms")
        _write_json_atomic(self._handoff_path, payload)

    def _rows_to_arrow(self, rows: list[dict[str, Any]]) -> pa.Table:
        frame = pd.DataFrame(rows)
        for column in PARQUET_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[PARQUET_COLUMNS].copy()
        frame.sort_values(["timestamp", "market_id", "token_id"], inplace=True, kind="mergesort")
        frame["timestamp"] = pd.to_datetime(
            pd.to_numeric(frame["timestamp"], errors="coerce"),
            unit="ms",
            utc=True,
        )
        for column in ("best_bid", "best_ask", "bid_depth", "ask_depth"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        for column in ("market_id", "event_id", "token_id"):
            frame[column] = frame[column].astype("string")
        return pa.Table.from_pandas(frame, schema=PARQUET_SCHEMA, preserve_index=False)

    def _rotation_bucket(self, timestamp_ms: int) -> RotationBucket:
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
        date_str = timestamp.strftime("%Y-%m-%d")
        if self._rotation == "hourly":
            bucket_id = timestamp.strftime("%Y-%m-%d-%H")
            hour_str = timestamp.strftime("%H")
            file_stem = timestamp.strftime("l2_book_%Y-%m-%d_%H")
        else:
            bucket_id = date_str
            hour_str = None
            file_stem = timestamp.strftime("l2_book_%Y-%m-%d")
        return RotationBucket(bucket_id=bucket_id, date_str=date_str, hour_str=hour_str, file_stem=file_stem)

    def _next_file_path(self, bucket: RotationBucket) -> Path:
        date_dir = self._output_dir / f"date={bucket.date_str}"
        if bucket.hour_str is not None:
            date_dir = date_dir / f"hour={bucket.hour_str}"
        date_dir.mkdir(parents=True, exist_ok=True)
        next_part = self._part_counters.get(bucket.bucket_id)
        if next_part is None:
            next_part = self._discover_next_part(date_dir, bucket.file_stem)
        self._part_counters[bucket.bucket_id] = next_part + 1
        return date_dir / f"{bucket.file_stem}_{next_part:06d}.parquet"

    @staticmethod
    def _discover_next_part(date_dir: Path, file_stem: str) -> int:
        highest = 0
        for path in date_dir.glob(f"{file_stem}_*.parquet"):
            stem = path.stem
            suffix = stem.rsplit("_", 1)[-1]
            try:
                highest = max(highest, int(suffix))
            except ValueError:
                continue
        return highest + 1

    def _ensure_free_space(self) -> None:
        if self._min_free_gb <= 0:
            return
        usage = shutil.disk_usage(self._output_dir)
        free_bytes_floor = int(self._min_free_gb * (1024**3))
        if usage.free >= free_bytes_floor:
            return
        raise LowDiskSpaceError(
            f"Free space below threshold for {self._output_dir}: "
            f"{usage.free / (1024**3):.2f} GiB remaining < {self._min_free_gb:.2f} GiB floor"
        )


class L2RecordingSocket:
    _BACKOFF_BASE = 1.0
    _BACKOFF_MAX = 60.0

    def __init__(
        self,
        *,
        socket_id: int,
        asset_ids: list[str],
        writer: ParquetTickWriter,
        message_processor: LiveBookAssembler,
        subscriptions_by_asset: dict[str, AssetSubscription],
        ws_url: str,
        silence_timeout_seconds: float,
    ) -> None:
        self.socket_id = socket_id
        self.asset_ids = list(asset_ids)
        self._writer = writer
        self._message_processor = message_processor
        self._subscriptions_by_asset = subscriptions_by_asset
        self._ws_url = ws_url
        self._silence_timeout_seconds = silence_timeout_seconds
        self._ws: Any = None
        self._running = False
        self._last_message_time = 0.0
        self.reconnect_count = 0

    async def start(self) -> None:
        self._running = True
        attempt = 0
        while self._running:
            try:
                await self._connect_and_consume()
                attempt = 0
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidStatus,
                ConnectionError,
                OSError,
            ) as exc:
                attempt += 1
                self.reconnect_count += 1
                sleep_seconds = min(self._BACKOFF_MAX, self._BACKOFF_BASE * (2**attempt))
                log.warning(
                    "tick_socket_disconnected",
                    socket_id=self.socket_id,
                    asset_count=len(self.asset_ids),
                    attempt=attempt,
                    retry_in=round(sleep_seconds, 2),
                    reconnect_count=self.reconnect_count,
                    error=str(exc),
                )
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                self._running = False
                break

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()

    async def add_assets(self, new_ids: list[str]) -> None:
        added = [asset_id for asset_id in new_ids if asset_id not in self.asset_ids]
        if not added:
            return
        self.asset_ids.extend(added)
        if self._ws is None:
            return
        for asset_id in added:
            try:
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "book",
                    "assets_ids": [asset_id],
                }
                await self._ws.send(json.dumps(subscribe_msg))
                log.info(
                    "tick_socket_subscribed_dynamic",
                    socket_id=self.socket_id,
                    asset_id=asset_id,
                )
            except Exception as exc:
                log.warning(
                    "tick_socket_subscribe_failed",
                    socket_id=self.socket_id,
                    asset_id=asset_id,
                    error=str(exc),
                )

    async def remove_assets(self, ids_to_remove: list[str]) -> None:
        removing = [asset_id for asset_id in ids_to_remove if asset_id in self.asset_ids]
        if not removing:
            return
        for asset_id in removing:
            self.asset_ids.remove(asset_id)
        if self._ws is None:
            return
        for asset_id in removing:
            try:
                unsubscribe_msg = {
                    "type": "unsubscribe",
                    "channel": "book",
                    "assets_ids": [asset_id],
                }
                await self._ws.send(json.dumps(unsubscribe_msg))
                log.info(
                    "tick_socket_unsubscribed_dynamic",
                    socket_id=self.socket_id,
                    asset_id=asset_id,
                )
            except Exception as exc:
                log.warning(
                    "tick_socket_unsubscribe_failed",
                    socket_id=self.socket_id,
                    asset_id=asset_id,
                    error=str(exc),
                )

    async def _connect_and_consume(self) -> None:
        if not self.asset_ids:
            await asyncio.sleep(1.0)
            return
        async with websockets.connect(self._ws_url, ping_interval=20, max_size=2**24) as ws:
            self._ws = ws
            self._last_message_time = time.time()
            subscribe_msg = {
                "type": "subscribe",
                "channel": "book",
                "assets_ids": self.asset_ids,
            }
            await ws.send(json.dumps(subscribe_msg))
            log.info(
                "tick_socket_connected",
                socket_id=self.socket_id,
                asset_count=len(self.asset_ids),
                ws_url=self._ws_url,
            )
            silence_task = asyncio.create_task(
                self._silence_watchdog(ws),
                name=f"tick_socket_silence_{self.socket_id}",
            )
            try:
                async for raw in ws:
                    if not self._running:
                        break
                    self._last_message_time = time.time()
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("tick_socket_bad_json", socket_id=self.socket_id, sample=raw[:200])
                        continue
                    await self._handle_message(message)
            finally:
                silence_task.cancel()
                self._ws = None

    async def _silence_watchdog(self, ws: Any) -> None:
        while self._running:
            await asyncio.sleep(1.0)
            if not self.asset_ids:
                continue
            silence_seconds = time.time() - self._last_message_time
            if silence_seconds <= self._silence_timeout_seconds:
                continue
            log.warning(
                "tick_socket_silence_timeout",
                socket_id=self.socket_id,
                asset_count=len(self.asset_ids),
                silence_seconds=round(silence_seconds, 1),
                threshold_seconds=self._silence_timeout_seconds,
            )
            await ws.close()
            return

    async def _handle_message(self, message: dict[str, Any] | list[Any]) -> None:
        if isinstance(message, list):
            for item in message:
                if isinstance(item, (dict, list)):
                    await self._handle_message(item)
            return
        if not isinstance(message, dict):
            return

        rows = self._message_processor.process_message(message, received_at=time.time())
        for row in rows:
            await self._writer.enqueue(row)


class L2RecordingPool:
    def __init__(
        self,
        *,
        asset_ids: list[str],
        writer: ParquetTickWriter,
        message_processor: LiveBookAssembler,
        subscriptions_by_asset: dict[str, AssetSubscription],
        max_assets_per_socket: int,
        ws_url: str,
        silence_timeout_seconds: float,
        connect_stagger_seconds: float,
        socket_factory: Any | None = None,
    ) -> None:
        self._writer = writer
        self._message_processor = message_processor
        self._subscriptions_by_asset = subscriptions_by_asset
        self._max_assets_per_socket = max_assets_per_socket
        self._ws_url = ws_url
        self._silence_timeout_seconds = silence_timeout_seconds
        self._connect_stagger_seconds = max(0.0, connect_stagger_seconds)
        self._socket_factory = socket_factory or L2RecordingSocket
        self._sockets: list[L2RecordingSocket] = []
        self._socket_tasks: dict[L2RecordingSocket, asyncio.Task[None]] = {}
        self._asset_to_socket: dict[str, L2RecordingSocket] = {}
        self._lock = asyncio.Lock()
        self._started = False
        self._socket_failure: asyncio.Future[None] | None = None
        self._next_socket_id = 1

        for chunk in _chunk_assets(sorted(asset_ids), max_assets_per_socket):
            self._register_socket(self._make_socket(chunk))

    @property
    def asset_ids(self) -> list[str]:
        return sorted(self._asset_to_socket)

    @property
    def asset_count(self) -> int:
        return len(self._asset_to_socket)

    @property
    def socket_count(self) -> int:
        return len(self._sockets)

    @property
    def reconnect_count(self) -> int:
        return sum(socket.reconnect_count for socket in self._sockets)

    @property
    def failure_future(self) -> asyncio.Future[None]:
        if self._socket_failure is None:
            raise RuntimeError("L2 recording pool must be started before waiting on failures")
        return self._socket_failure

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self._started = True
            self._socket_failure = asyncio.get_running_loop().create_future()
            sockets_to_start = list(self._sockets)
        for socket in sockets_to_start:
            await self._start_socket_task(socket)
        log.info(
            "tick_pool_started",
            socket_count=self.socket_count,
            asset_count=self.asset_count,
        )

    async def stop(self) -> None:
        async with self._lock:
            self._started = False
            sockets = list(self._sockets)
            tasks = list(self._socket_tasks.values())
            self._sockets.clear()
            self._socket_tasks.clear()
            self._asset_to_socket.clear()
            failure_future = self._socket_failure
            self._socket_failure = None
        for socket in sockets:
            await socket.stop()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if failure_future is not None and not failure_future.done():
            failure_future.cancel()

    async def apply_universe(
        self,
        subscriptions_by_asset: dict[str, AssetSubscription],
    ) -> tuple[list[str], list[str], list[str]]:
        async with self._lock:
            current_assets = set(self._asset_to_socket)
            target_assets = set(subscriptions_by_asset)
            additions = sorted(target_assets - current_assets)
            removals = sorted(current_assets - target_assets)
            updated = sorted(
                asset_id
                for asset_id in current_assets & target_assets
                if self._subscriptions_by_asset.get(asset_id) != subscriptions_by_asset[asset_id]
            )

            for asset_id, subscription in subscriptions_by_asset.items():
                self._subscriptions_by_asset[asset_id] = subscription

            if additions:
                await self._add_assets_locked(additions)
            if removals:
                await self._remove_assets_locked(removals)
                for asset_id in removals:
                    self._subscriptions_by_asset.pop(asset_id, None)

            return additions, removals, updated

    def _make_socket(self, asset_ids: list[str]) -> L2RecordingSocket:
        socket = self._socket_factory(
            socket_id=self._next_socket_id,
            asset_ids=list(asset_ids),
            writer=self._writer,
            message_processor=self._message_processor,
            subscriptions_by_asset=self._subscriptions_by_asset,
            ws_url=self._ws_url,
            silence_timeout_seconds=self._silence_timeout_seconds,
        )
        self._next_socket_id += 1
        return socket

    def _register_socket(self, socket: L2RecordingSocket) -> None:
        self._sockets.append(socket)
        for asset_id in socket.asset_ids:
            self._asset_to_socket[asset_id] = socket

    async def _start_socket_task(self, socket: L2RecordingSocket) -> None:
        if socket in self._socket_tasks:
            return
        task = asyncio.create_task(socket.start(), name=f"l2_tick_socket_{socket.socket_id}")
        self._socket_tasks[socket] = task
        task.add_done_callback(lambda done_task, current_socket=socket: self._on_socket_task_done(current_socket, done_task))
        if self._connect_stagger_seconds > 0:
            await asyncio.sleep(self._connect_stagger_seconds)

    def _on_socket_task_done(self, socket: L2RecordingSocket, task: asyncio.Task[None]) -> None:
        self._socket_tasks.pop(socket, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            if not self._started or socket not in self._sockets:
                return
            exc = RuntimeError(f"L2 recording socket {socket.socket_id} exited unexpectedly")
        log.error(
            "tick_socket_task_failed",
            socket_id=socket.socket_id,
            asset_count=len(socket.asset_ids),
            error=repr(exc),
            exc_info=exc,
        )
        if self._socket_failure is not None and not self._socket_failure.done():
            self._socket_failure.set_exception(exc)

    async def _add_assets_locked(self, asset_ids: list[str]) -> None:
        remaining = [asset_id for asset_id in asset_ids if asset_id not in self._asset_to_socket]
        if not remaining:
            return

        for socket in self._sockets:
            if not remaining:
                break
            capacity = self._max_assets_per_socket - len(socket.asset_ids)
            if capacity <= 0:
                continue
            chunk = remaining[:capacity]
            remaining = remaining[capacity:]
            await socket.add_assets(chunk)
            for asset_id in chunk:
                self._asset_to_socket[asset_id] = socket

        while remaining:
            chunk = remaining[: self._max_assets_per_socket]
            remaining = remaining[self._max_assets_per_socket :]
            socket = self._make_socket(chunk)
            self._register_socket(socket)
            if self._started:
                await self._start_socket_task(socket)
            log.info(
                "tick_pool_socket_added",
                socket_id=socket.socket_id,
                asset_count=len(socket.asset_ids),
                total_sockets=self.socket_count,
            )

    async def _remove_assets_locked(self, asset_ids: list[str]) -> None:
        removals_by_socket: dict[L2RecordingSocket, list[str]] = {}
        for asset_id in asset_ids:
            socket = self._asset_to_socket.pop(asset_id, None)
            if socket is None:
                continue
            removals_by_socket.setdefault(socket, []).append(asset_id)

        for socket, socket_asset_ids in removals_by_socket.items():
            await socket.remove_assets(socket_asset_ids)

        empty_sockets = [socket for socket in self._sockets if not socket.asset_ids]
        for socket in empty_sockets:
            self._sockets.remove(socket)
            await socket.stop()
            task = self._socket_tasks.pop(socket, None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            log.info(
                "tick_pool_socket_retired",
                socket_id=socket.socket_id,
                total_sockets=self.socket_count,
            )


class UniverseRefreshLoop:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        pool: L2RecordingPool,
        refresh_seconds: float,
        resolve_subscriptions: Any = None,
    ) -> None:
        if refresh_seconds < 0:
            raise ValueError("refresh_seconds must be >= 0")
        self._args = args
        self._pool = pool
        self._refresh_seconds = float(refresh_seconds)
        self._resolve_subscriptions = resolve_subscriptions or _resolve_active_subscriptions
        self._running = False
        self._refresh_count = 0
        self._last_refresh_monotonic: float | None = None
        self._last_refresh_reason: str | None = None

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    @property
    def last_refresh_reason(self) -> str | None:
        return self._last_refresh_reason

    @property
    def last_refresh_age_seconds(self) -> float | None:
        if self._last_refresh_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self._last_refresh_monotonic)

    def note_refresh(self, *, reason: str) -> None:
        self._last_refresh_monotonic = time.monotonic()
        self._last_refresh_reason = reason
        self._refresh_count += 1

    async def run(self) -> None:
        self._running = True
        log.info(
            "tick_universe_refresh_started",
            refresh_seconds=self._refresh_seconds,
        )
        try:
            while self._running:
                await asyncio.sleep(self._refresh_seconds)
                if not self._running:
                    break
                started_at = time.monotonic()
                try:
                    target_subscriptions = await self._resolve_subscriptions(
                        self._args,
                        reason="periodic_refresh",
                    )
                    additions, removals, updated = await self._pool.apply_universe(target_subscriptions)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.error(
                        "tick_universe_refresh_failed",
                        refresh_seconds=self._refresh_seconds,
                        error=str(exc),
                        exc_info=True,
                    )
                    continue

                self.note_refresh(reason="periodic_refresh")
                log.info(
                    "tick_universe_refresh_complete",
                    refresh_count=self.refresh_count,
                    added=len(additions),
                    removed=len(removals),
                    updated=len(updated),
                    asset_count=self._pool.asset_count,
                    socket_count=self._pool.socket_count,
                    elapsed_s=round(time.monotonic() - started_at, 2),
                )
        finally:
            log.info(
                "tick_universe_refresh_stopped",
                refresh_count=self.refresh_count,
            )

    def stop(self) -> None:
        self._running = False


class CompressorHeartbeatLoop:
    def __init__(
        self,
        *,
        pool: L2RecordingPool,
        refresh_loop: UniverseRefreshLoop,
        message_processor: LiveBookAssembler,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._pool = pool
        self._refresh_loop = refresh_loop
        self._message_processor = message_processor
        self._interval_seconds = float(interval_seconds)
        self._running = False

    async def run(self) -> None:
        self._running = True
        log.info(
            "tick_heartbeat_started",
            interval_seconds=self._interval_seconds,
        )
        try:
            while self._running:
                await asyncio.sleep(self._interval_seconds)
                if not self._running:
                    break
                last_refresh_age_seconds = self._refresh_loop.last_refresh_age_seconds
                log.info(
                    "tick_compressor_heartbeat",
                    asset_count=self._pool.asset_count,
                    socket_count=self._pool.socket_count,
                    reconnect_count=self._pool.reconnect_count,
                    refresh_count=self._refresh_loop.refresh_count,
                    last_refresh_reason=self._refresh_loop.last_refresh_reason,
                    last_refresh_age_s=None if last_refresh_age_seconds is None else round(last_refresh_age_seconds, 1),
                    tracked_market_count=self._message_processor.market_count,
                    emitted_rows=self._message_processor.emitted_rows,
                    preseed_deltas=self._message_processor.preseed_deltas,
                    invalid_pairs=self._message_processor.invalid_pairs,
                )
        finally:
            log.info("tick_heartbeat_stopped")

    def stop(self) -> None:
        self._running = False


def _build_subscriptions(markets: list[MarketInfo]) -> dict[str, AssetSubscription]:
    subscriptions: dict[str, AssetSubscription] = {}
    for market in markets:
        market_id = str(market.condition_id or "").strip().lower()
        event_id = str(market.event_id or market.condition_id or "").strip()
        yes_asset_id = str(market.yes_token_id or "").strip()
        no_asset_id = str(market.no_token_id or "").strip()
        if market.yes_token_id:
            subscriptions[market.yes_token_id] = AssetSubscription(
                asset_id=yes_asset_id,
                market_id=market_id,
                event_id=event_id,
                token_id="YES",
                yes_asset_id=yes_asset_id,
                no_asset_id=no_asset_id,
                question=market.question,
                daily_volume_usd=float(market.daily_volume_usd),
            )
        if market.no_token_id:
            subscriptions[market.no_token_id] = AssetSubscription(
                asset_id=no_asset_id,
                market_id=market_id,
                event_id=event_id,
                token_id="NO",
                yes_asset_id=yes_asset_id,
                no_asset_id=no_asset_id,
                question=market.question,
                daily_volume_usd=float(market.daily_volume_usd),
            )
    return subscriptions


async def _resolve_active_subscriptions(
    args: argparse.Namespace,
    *,
    reason: str = "startup",
) -> dict[str, AssetSubscription]:
    markets = await fetch_active_markets(
        min_volume=args.min_volume,
        min_days_to_resolution=args.min_days_to_resolution,
        limit=args.market_limit,
    )
    ordered_markets = sorted(
        markets,
        key=lambda market: (-float(market.daily_volume_usd), market.condition_id),
    )
    subscriptions = _build_subscriptions(ordered_markets)
    if not subscriptions:
        raise RuntimeError("Active market discovery returned zero YES/NO token subscriptions")
    log.info(
        "tick_universe_resolved",
        reason=reason,
        market_count=len(markets),
        asset_count=len(subscriptions),
        market_limit=args.market_limit,
        min_volume=args.min_volume,
        min_days_to_resolution=args.min_days_to_resolution,
    )
    return subscriptions


async def _run(args: argparse.Namespace) -> int:
    loop = asyncio.get_running_loop()
    shutdown_monitor = ShutdownSignalMonitor()
    shutdown_monitor.install(loop)
    subscriptions_by_asset = await _resolve_active_subscriptions(args, reason="startup")
    message_processor = LiveBookAssembler(subscriptions_by_asset)
    effective_output_dir = _effective_output_dir(args.output_dir)
    writer = ParquetTickWriter(
        output_dir=effective_output_dir,
        flush_rows=args.flush_rows,
        flush_seconds=float(args.flush_seconds),
        queue_size=args.queue_size,
        rotation=args.rotation,
        compression=args.compression,
        min_free_gb=float(args.min_free_gb),
    )
    pool = L2RecordingPool(
        asset_ids=sorted(subscriptions_by_asset),
        writer=writer,
        message_processor=message_processor,
        subscriptions_by_asset=subscriptions_by_asset,
        max_assets_per_socket=args.max_assets_per_socket,
        ws_url=args.ws_url,
        silence_timeout_seconds=float(args.silence_timeout_seconds),
        connect_stagger_seconds=float(args.connect_stagger_seconds),
    )
    refresh_loop = UniverseRefreshLoop(
        args=args,
        pool=pool,
        refresh_seconds=float(args.universe_refresh_seconds),
    )
    refresh_loop.note_refresh(reason="startup")

    writer_task = asyncio.create_task(writer.run(), name="tick_writer")
    writer_task.add_done_callback(_log_task_exception)
    await pool.start()
    shutdown_task = asyncio.create_task(shutdown_monitor.wait(), name="tick_shutdown_wait")
    refresh_task: asyncio.Task[None] | None = None
    heartbeat_loop: CompressorHeartbeatLoop | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    wait_targets: list[asyncio.Future[Any] | asyncio.Task[Any]] = [writer_task, pool.failure_future, shutdown_task]
    if float(args.universe_refresh_seconds) > 0:
        refresh_task = asyncio.create_task(refresh_loop.run(), name="tick_universe_refresh")
        refresh_task.add_done_callback(_log_task_exception)
        wait_targets.append(refresh_task)
    else:
        log.info("tick_universe_refresh_disabled")
    if float(args.heartbeat_seconds) > 0:
        heartbeat_loop = CompressorHeartbeatLoop(
            pool=pool,
            refresh_loop=refresh_loop,
            message_processor=message_processor,
            interval_seconds=float(args.heartbeat_seconds),
        )
        heartbeat_task = asyncio.create_task(heartbeat_loop.run(), name="tick_heartbeat")
        heartbeat_task.add_done_callback(_log_task_exception)
        wait_targets.append(heartbeat_task)
    else:
        log.info("tick_heartbeat_disabled")
    exit_reason = "shutdown"
    try:
        done, _pending = await asyncio.wait(wait_targets, return_when=asyncio.FIRST_COMPLETED)
        for completed in done:
            if completed.cancelled():
                continue
            if completed is shutdown_task:
                exit_reason = shutdown_monitor.reason or "shutdown_requested"
                log.info("tick_compressor_shutdown_received", reason=exit_reason)
                return 0
            exc = completed.exception()
            if exc is not None:
                if completed is writer_task and isinstance(exc, LowDiskSpaceError):
                    exit_reason = "low_disk_space"
                elif completed is pool.failure_future:
                    exit_reason = "socket_failure"
                elif completed is refresh_task:
                    exit_reason = "universe_refresh_failure"
                elif completed is heartbeat_task:
                    exit_reason = "heartbeat_failure"
                else:
                    exit_reason = "failure"
                raise exc
            if completed is writer_task:
                exit_reason = "writer_unexpected_exit"
                raise RuntimeError("tick_writer exited unexpectedly")
            if completed is refresh_task:
                exit_reason = "universe_refresh_unexpected_exit"
                raise RuntimeError("tick_universe_refresh exited unexpectedly")
            if completed is heartbeat_task:
                exit_reason = "heartbeat_unexpected_exit"
                raise RuntimeError("tick_heartbeat exited unexpectedly")
    finally:
        shutdown_monitor.restore()
        if heartbeat_loop is not None:
            heartbeat_loop.stop()
        refresh_loop.stop()
        await pool.stop()
        writer.stop(reason=shutdown_monitor.reason or exit_reason)
        tasks_to_join = [writer_task, shutdown_task]
        if refresh_task is not None:
            tasks_to_join.append(refresh_task)
        if heartbeat_task is not None:
            tasks_to_join.append(heartbeat_task)
        for task in tasks_to_join:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks_to_join, return_exceptions=True)
    return 0


def main() -> int:
    args = _parse_args()
    setup_logging(
        log_dir=args.log_dir,
        level=getattr(logging, args.log_level.upper()),
        log_file="live_tick_compressor.jsonl",
    )
    try:
        return asyncio.run(_run(args))
    except LowDiskSpaceError as exc:
        log.error("tick_compressor_low_disk_stop", error=str(exc))
        return 0
    except KeyboardInterrupt:
        log.info("tick_compressor_stopped_by_signal", reason="keyboard_interrupt")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())