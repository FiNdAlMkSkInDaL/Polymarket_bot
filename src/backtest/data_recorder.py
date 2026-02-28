"""
Live market data recorder — captures raw WebSocket events to disk for
future backtesting replay.

Architecture
────────────
    WebSocket handler ──push──▶ asyncio.Queue  ──consume──▶ background flush worker
                                 (bounded, non-blocking)     (batched writes via to_thread)

File layout
───────────
    <data_dir>/raw_ticks/YYYY-MM-DD/<asset_id>.jsonl

Each line is a JSON object:

    {"local_ts": 1700000001.234, "source": "l2", "asset_id": "0xabc...", "payload": { ... }}

The ``payload`` is the **raw, unmutated** dict as received from the exchange.
``local_ts`` is the system receive-clock — the delta between ``local_ts``
and ``payload.timestamp`` captures realistic network latency for simulation.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.logger import get_logger

log = get_logger(__name__)


class MarketDataRecorder:
    """Non-blocking recorder that persists raw WS events to JSONL files.

    Design philosophy: **zero cached file handles**.  Each flush batch
    opens the target files, appends grouped lines, and closes them
    immediately.  This makes the recorder immune to fd-limit exhaustion
    regardless of the number of tracked markets, eliminates stale-handle
    bugs on day rollover, and costs < 1 ms per flush on local storage
    (negligible vs the 5-second flush interval).

    Parameters
    ----------
    data_dir:
        Root directory for recorded data (e.g. ``"data"``).
    queue_size:
        Max capacity of the internal async queue.
    flush_interval_s:
        Maximum seconds between disk flushes.
    flush_batch_size:
        Maximum records per flush batch.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        queue_size: int = 50_000,
        flush_interval_s: float = 5.0,
        flush_batch_size: int = 1_000,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=queue_size)
        self._flush_interval = flush_interval_s
        self._flush_batch = flush_batch_size
        self._running = False
        self._buffer: list[dict] = []
        self._records_written: int = 0
        self._records_dropped: int = 0
        self._write_errors: int = 0

        # Directory cache: avoid repeated mkdir syscalls within a flush
        self._known_dirs: set[Path] = set()

    # ═══════════════════════════════════════════════════════════════════════
    #  Producer side (called from WS handler — must not block)
    # ═══════════════════════════════════════════════════════════════════════

    def enqueue(self, source: str, msg: dict) -> None:
        """Push a raw event onto the recording queue.

        Parameters
        ----------
        source:
            Event type label: ``"l2"``, ``"trade"``, ``"book_snapshot"``.
        msg:
            The raw dict payload as received from the exchange.
        """
        asset_id = (
            msg.get("asset_id")
            or msg.get("market")
            or msg.get("token_id")
            or "unknown"
        )

        record = {
            "local_ts": time.time(),
            "source": source,
            "asset_id": asset_id,
            "payload": msg,
        }

        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._records_dropped += 1

    # ═══════════════════════════════════════════════════════════════════════
    #  Consumer side (background asyncio task)
    # ═══════════════════════════════════════════════════════════════════════

    async def run(self) -> None:
        """Long-running consumer coroutine. Launch as ``create_task(recorder.run())``."""
        self._running = True
        log.info("recorder_started", data_dir=str(self._data_dir))

        try:
            while self._running:
                await self._consume_batch()
        except asyncio.CancelledError:
            # Drain remaining items before exit
            while not self._queue.empty():
                try:
                    record = self._queue.get_nowait()
                    self._buffer.append(record)
                except asyncio.QueueEmpty:
                    break
            if self._buffer:
                await asyncio.to_thread(self._flush_sync)
        finally:
            log.info(
                "recorder_stopped",
                records_written=self._records_written,
                records_dropped=self._records_dropped,
                write_errors=self._write_errors,
            )

    async def _consume_batch(self) -> None:
        """Wait for events and flush in batches."""
        # Wait for at least one event (with timeout for periodic flush)
        try:
            record = await asyncio.wait_for(
                self._queue.get(), timeout=self._flush_interval
            )
            self._buffer.append(record)
        except asyncio.TimeoutError:
            pass

        # Drain the rest of the queue up to batch size
        drained = 0
        while drained < self._flush_batch and not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                self._buffer.append(record)
                drained += 1
            except asyncio.QueueEmpty:
                break

        # Flush if we have data
        if self._buffer:
            await asyncio.to_thread(self._flush_sync)

    def _flush_sync(self) -> None:
        """Synchronous disk write — called via ``to_thread`` to avoid
        blocking the event loop.

        Opens each target file, appends all grouped lines, and closes it
        immediately.  No file handles are cached across flush cycles.
        Each (date, asset) group is isolated: a write failure for one
        asset does not prevent other assets from being persisted.
        """
        # Group by (date, asset_id) for efficient file writes
        groups: dict[tuple[str, str], list[str]] = {}

        for record in self._buffer:
            ts = record["local_ts"]
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            asset_id = record["asset_id"]
            key = (date_str, asset_id)

            line = json.dumps(record, separators=(",", ":"), default=str)
            groups.setdefault(key, []).append(line)

        for (date_str, asset_id), lines in groups.items():
            try:
                file_path = self._resolve_path(date_str, asset_id)
                with open(file_path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
                self._records_written += len(lines)
            except OSError:
                self._write_errors += len(lines)
                log.exception(
                    "recorder_write_failed",
                    date=date_str,
                    asset_id=asset_id,
                    lines_lost=len(lines),
                )

        self._buffer.clear()

    def _resolve_path(self, date_str: str, asset_id: str) -> Path:
        """Build the target file path, creating the directory on first
        encounter (cached for the process lifetime to skip redundant
        ``mkdir`` syscalls)."""
        dir_path = self._data_dir / "raw_ticks" / date_str
        if dir_path not in self._known_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            self._known_dirs.add(dir_path)

        safe_name = asset_id.replace("/", "_").replace("\\", "_")
        return dir_path / f"{safe_name}.jsonl"

    def stop(self) -> None:
        """Signal the consumer loop to stop after the current batch."""
        self._running = False

    @property
    def stats(self) -> dict[str, int]:
        return {
            "records_written": self._records_written,
            "records_dropped": self._records_dropped,
            "write_errors": self._write_errors,
            "queue_depth": self._queue.qsize(),
            "buffer_size": len(self._buffer),
        }

    # ── Convenience: wire into existing data directory ─────────────────

    @staticmethod
    def data_files_for_date(
        data_dir: str | Path, date_str: str
    ) -> list[Path]:
        """List all JSONL tick files for a given date."""
        tick_dir = Path(data_dir) / "raw_ticks" / date_str
        if not tick_dir.exists():
            return []
        return sorted(tick_dir.glob("*.jsonl"))

    @staticmethod
    def available_dates(data_dir: str | Path) -> list[str]:
        """List all recorded dates in YYYY-MM-DD order.

        Scans both the ``raw_ticks/`` sub-directory (JSONL layout) and
        the top-level data directory (processed Parquet layout) for
        date-named folders.
        """
        base = Path(data_dir)
        date_names: set[str] = set()

        # Raw JSONL layout: <data_dir>/raw_ticks/YYYY-MM-DD/
        tick_dir = base / "raw_ticks"
        if tick_dir.exists():
            for d in tick_dir.iterdir():
                if d.is_dir() and len(d.name) == 10 and d.name[4] == "-":
                    date_names.add(d.name)

        # Processed Parquet layout: <data_dir>/YYYY-MM-DD/
        for d in base.iterdir():
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-":
                # Must contain at least one .parquet file
                if any(d.glob("*.parquet")):
                    date_names.add(d.name)

        return sorted(date_names)
