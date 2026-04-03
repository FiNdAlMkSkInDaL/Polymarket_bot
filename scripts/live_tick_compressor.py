#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from src.data.market_discovery import MarketInfo, fetch_active_markets


log = get_logger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "l2_archive"
DEFAULT_FLUSH_ROWS = 10_000
DEFAULT_FLUSH_SECONDS = 300.0
DEFAULT_QUEUE_SIZE = 100_000
DEFAULT_MARKET_LIMIT = 200
DEFAULT_MAX_ASSETS_PER_SOCKET = 50
DEFAULT_CONNECT_STAGGER_SECONDS = 0.25
DEFAULT_MIN_FREE_GB = 10.0
DEFAULT_UNIVERSE_REFRESH_SECONDS = 7_200.0
DEFAULT_HEARTBEAT_SECONDS = 900.0

DELTA_EVENTS = frozenset(("price_change", "book_delta", "delta"))
SNAPSHOT_EVENTS = frozenset(("book", "snapshot", "book_snapshot"))
DICTIONARY_COLUMNS = ["msg_type", "asset_id", "market_id", "outcome", "side"]
PARQUET_COLUMNS = [
    "local_ts",
    "exchange_ts",
    "msg_type",
    "asset_id",
    "market_id",
    "outcome",
    "price",
    "size",
    "sequence_id",
    "side",
    "payload",
]
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("local_ts", pa.float64(), nullable=False),
        pa.field("exchange_ts", pa.float64(), nullable=True),
        pa.field("msg_type", pa.string(), nullable=False),
        pa.field("asset_id", pa.string(), nullable=False),
        pa.field("market_id", pa.string(), nullable=True),
        pa.field("outcome", pa.string(), nullable=True),
        pa.field("price", pa.float64(), nullable=True),
        pa.field("size", pa.float64(), nullable=True),
        pa.field("sequence_id", pa.int64(), nullable=True),
        pa.field("side", pa.string(), nullable=True),
        pa.field("payload", pa.string(), nullable=False),
    ]
)


class LowDiskSpaceError(RuntimeError):
    """Raised when the compressor hits the configured free-space floor."""


@dataclass(frozen=True, slots=True)
class AssetSubscription:
    asset_id: str
    market_id: str
    outcome: str
    question: str
    daily_volume_usd: float


@dataclass(frozen=True, slots=True)
class RotationBucket:
    bucket_id: str
    date_str: str
    file_stem: str


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


def _normalize_sequence_id(msg: dict[str, Any]) -> int | None:
    raw_value = msg.get("seq") or msg.get("sequence") or msg.get("seq_num")
    if raw_value in (None, ""):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def build_tick_row(
    msg: dict[str, Any],
    *,
    received_at: float,
    subscriptions_by_asset: dict[str, AssetSubscription],
) -> dict[str, Any] | None:
    event_type = str(msg.get("event_type") or msg.get("type") or "").strip().lower()
    if event_type in SNAPSHOT_EVENTS:
        msg_type = "snapshot"
    elif event_type in DELTA_EVENTS:
        msg_type = "delta"
    else:
        return None

    asset_id = str(msg.get("asset_id") or "").strip()
    if not asset_id:
        return None

    subscription = subscriptions_by_asset.get(asset_id)
    market_id = subscription.market_id if subscription is not None else str(msg.get("market") or msg.get("condition_id") or "").strip() or None
    outcome = subscription.outcome if subscription is not None else None

    price = None
    size = None
    side = None
    price_changes = msg.get("price_changes")
    if msg_type == "delta" and isinstance(price_changes, list) and len(price_changes) == 1 and isinstance(price_changes[0], dict):
        price = _safe_float(price_changes[0].get("price"))
        size = _safe_float(price_changes[0].get("size"))
        side = _safe_text(price_changes[0].get("side"))

    return {
        "local_ts": float(received_at),
        "exchange_ts": _normalize_exchange_ts(msg),
        "msg_type": msg_type,
        "asset_id": asset_id,
        "market_id": market_id,
        "outcome": outcome,
        "price": price,
        "size": size,
        "sequence_id": _normalize_sequence_id(msg),
        "side": side,
        "payload": json.dumps(msg, separators=(",", ":"), default=str),
    }


def _chunk_assets(asset_ids: list[str], max_assets_per_socket: int) -> list[list[str]]:
    if max_assets_per_socket <= 0:
        raise ValueError("max_assets_per_socket must be a strictly positive int")
    if not asset_ids:
        return []
    return [
        asset_ids[index : index + max_assets_per_socket]
        for index in range(0, len(asset_ids), max_assets_per_socket)
    ]


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

    async def enqueue(self, row: dict[str, Any]) -> None:
        await self._queue.put(row)
        self._high_watermark = max(self._high_watermark, self._queue.qsize())

    async def run(self) -> None:
        self._running = True
        log.info(
            "tick_writer_started",
            output_dir=str(self._output_dir),
            flush_rows=self._flush_rows,
            flush_seconds=self._flush_seconds,
            rotation=self._rotation,
            compression=self._compression,
            min_free_gb=self._min_free_gb,
        )
        try:
            while self._running:
                await self._consume_and_flush_once()
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
            log.info(
                "tick_writer_stopped",
                rows_written=self._rows_written,
                files_written=self._files_written,
                bytes_written=self._bytes_written,
                queue_high_watermark=self._high_watermark,
            )

    def stop(self) -> None:
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
        for row in rows:
            bucket = self._rotation_bucket(row["local_ts"])
            grouped_rows.setdefault(bucket, []).append(row)

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
            log.info(
                "tick_parquet_chunk_written",
                file_path=str(file_path),
                rows=len(bucket_rows),
                bytes=file_size,
                compression=self._compression,
                rotation_bucket=bucket.bucket_id,
            )

    def _rows_to_arrow(self, rows: list[dict[str, Any]]) -> pa.Table:
        frame = pd.DataFrame(rows)
        for column in PARQUET_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[PARQUET_COLUMNS].copy()
        frame.sort_values("local_ts", inplace=True, kind="mergesort")
        frame["local_ts"] = pd.to_numeric(frame["local_ts"], errors="coerce").astype("float64")
        frame["exchange_ts"] = pd.to_numeric(frame["exchange_ts"], errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
        frame["sequence_id"] = pd.to_numeric(frame["sequence_id"], errors="coerce").astype("Int64")
        for column in ("msg_type", "asset_id", "market_id", "outcome", "side", "payload"):
            frame[column] = frame[column].astype("string")
        return pa.Table.from_pandas(frame, schema=PARQUET_SCHEMA, preserve_index=False)

    def _rotation_bucket(self, local_ts: float) -> RotationBucket:
        timestamp = datetime.fromtimestamp(local_ts, tz=UTC)
        date_str = timestamp.strftime("%Y-%m-%d")
        if self._rotation == "hourly":
            bucket_id = timestamp.strftime("%Y-%m-%d-%H")
            file_stem = timestamp.strftime("ticks_%Y-%m-%d_%H")
        else:
            bucket_id = date_str
            file_stem = timestamp.strftime("ticks_%Y-%m-%d")
        return RotationBucket(bucket_id=bucket_id, date_str=date_str, file_stem=file_stem)

    def _next_file_path(self, bucket: RotationBucket) -> Path:
        date_dir = self._output_dir / bucket.date_str
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
        subscriptions_by_asset: dict[str, AssetSubscription],
        ws_url: str,
        silence_timeout_seconds: float,
    ) -> None:
        self.socket_id = socket_id
        self.asset_ids = list(asset_ids)
        self._writer = writer
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

        row = build_tick_row(
            message,
            received_at=time.time(),
            subscriptions_by_asset=self._subscriptions_by_asset,
        )
        if row is None:
            return
        await self._writer.enqueue(row)


class L2RecordingPool:
    def __init__(
        self,
        *,
        asset_ids: list[str],
        writer: ParquetTickWriter,
        subscriptions_by_asset: dict[str, AssetSubscription],
        max_assets_per_socket: int,
        ws_url: str,
        silence_timeout_seconds: float,
        connect_stagger_seconds: float,
        socket_factory: Any | None = None,
    ) -> None:
        self._writer = writer
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
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._pool = pool
        self._refresh_loop = refresh_loop
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
                )
        finally:
            log.info("tick_heartbeat_stopped")

    def stop(self) -> None:
        self._running = False


def _build_subscriptions(markets: list[MarketInfo]) -> dict[str, AssetSubscription]:
    subscriptions: dict[str, AssetSubscription] = {}
    for market in markets:
        if market.yes_token_id:
            subscriptions[market.yes_token_id] = AssetSubscription(
                asset_id=market.yes_token_id,
                market_id=market.condition_id,
                outcome="YES",
                question=market.question,
                daily_volume_usd=float(market.daily_volume_usd),
            )
        if market.no_token_id:
            subscriptions[market.no_token_id] = AssetSubscription(
                asset_id=market.no_token_id,
                market_id=market.condition_id,
                outcome="NO",
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
    subscriptions_by_asset = await _resolve_active_subscriptions(args, reason="startup")
    writer = ParquetTickWriter(
        output_dir=args.output_dir,
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
    refresh_task: asyncio.Task[None] | None = None
    heartbeat_loop: CompressorHeartbeatLoop | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    wait_targets: list[asyncio.Future[Any] | asyncio.Task[Any]] = [writer_task, pool.failure_future]
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
            interval_seconds=float(args.heartbeat_seconds),
        )
        heartbeat_task = asyncio.create_task(heartbeat_loop.run(), name="tick_heartbeat")
        heartbeat_task.add_done_callback(_log_task_exception)
        wait_targets.append(heartbeat_task)
    else:
        log.info("tick_heartbeat_disabled")
    try:
        done, _pending = await asyncio.wait(wait_targets, return_when=asyncio.FIRST_COMPLETED)
        for completed in done:
            if completed.cancelled():
                continue
            exc = completed.exception()
            if exc is not None:
                raise exc
            if completed is writer_task:
                raise RuntimeError("tick_writer exited unexpectedly")
            if completed is refresh_task:
                raise RuntimeError("tick_universe_refresh exited unexpectedly")
            if completed is heartbeat_task:
                raise RuntimeError("tick_heartbeat exited unexpectedly")
    finally:
        if heartbeat_loop is not None:
            heartbeat_loop.stop()
        refresh_loop.stop()
        await pool.stop()
        writer.stop()
        tasks_to_join = [writer_task]
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
        log.info("tick_compressor_stopped_by_signal")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())