"""
L2 WebSocket Client — dedicated connection for real-time Level 2 order
book data from the Polymarket CLOB.

Separated from the existing ``MarketWebSocket`` (which handles trades)
to prevent book-delta storms from starving trade processing.

Lifecycle:
  1. Connect to the L2 WS endpoint.
  2. Subscribe to the ``book`` channel for each tracked asset.
  3. For each asset, transition its ``L2OrderBook`` to BUFFERING and
     fetch a REST snapshot.
  4. Apply the snapshot and replay buffered deltas.
  5. Forward subsequent deltas directly to the ``L2OrderBook``
     (no queue — synchronous O(log n) dispatch for minimal latency).
  6. On disconnect: reconnect with exponential backoff + jitter and
     re-subscribe + re-snapshot all active assets.

Desync handling:
  When an ``L2OrderBook`` detects a sequence gap, it fires the
  ``on_desync`` callback registered here.  We then re-fetch the REST
  snapshot and bring the book back to SYNCED.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import websockets
import websockets.exceptions

from src.core.config import settings
from src.core.logger import get_logger
from src.data.l2_book import BookState, L2OrderBook, fetch_l2_snapshot

log = get_logger(__name__)


def _log_task_exception(task: asyncio.Task) -> None:
    """Done-callback for fire-and-forget tasks to surface exceptions."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error(
            "l2ws_task_error",
            task_name=task.get_name(),
            error=repr(exc),
            exc_info=exc,
        )


# Pre-allocated event-type sets used in the hot message dispatch path.
_DELTA_EVENTS = frozenset(("price_change", "book_delta", "delta"))
_SNAPSHOT_EVENTS = frozenset(("book", "snapshot", "book_snapshot"))


class L2WebSocket:
    """Persistent WebSocket connection for Polymarket L2 order book data.

    Parameters
    ----------
    l2_books:
        Registry mapping ``asset_id → L2OrderBook``.  The client routes
        incoming deltas to the correct book instance.
    ws_url:
        WebSocket URL (defaults to ``settings.clob_l2_ws_url``).
    """

    _BACKOFF_BASE: float = 1.0
    _BACKOFF_MAX: float = 60.0

    def __init__(
        self,
        l2_books: dict[str, L2OrderBook],
        *,
        ws_url: str | None = None,
        recorder: Any | None = None,
    ):
        self._books = l2_books
        self._ws_url = ws_url or settings.clob_l2_ws_url
        self._recorder = recorder  # Optional MarketDataRecorder
        self._ws: Any = None
        self._running = False
        self._last_message_time: float = 0.0
        self._silence_timeout = settings.strategy.l2_silence_timeout_s
        self._snapshot_tasks: dict[str, asyncio.Task] = {}

        # Cumulative reconnect counter (never reset — used by health reporter)
        self.reconnect_count: int = 0

    # ═══════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════════════
    async def start(self) -> None:
        """Connect, subscribe, and consume messages with auto-reconnect."""
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
                sleep_time = min(
                    self._BACKOFF_MAX,
                    self._BACKOFF_BASE * (2 ** attempt),
                ) + random.uniform(0, 1)

                # Classify the disconnection reason for diagnostics
                if isinstance(exc, websockets.exceptions.ConnectionClosed):
                    reconnect_reason = "ws_close"
                elif isinstance(exc, websockets.exceptions.InvalidStatus):
                    reconnect_reason = "invalid_status"
                elif isinstance(exc, ConnectionError):
                    reconnect_reason = "connection_error"
                elif isinstance(exc, OSError):
                    reconnect_reason = "os_error"
                else:
                    reconnect_reason = "unknown"

                log.warning(
                    "l2_ws_disconnected",
                    error=str(exc),
                    reconnect_reason=reconnect_reason,
                    retry_in=round(sleep_time, 2),
                    attempt=attempt,
                    reconnect_count=self.reconnect_count,
                )
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                self._running = False
                break

    async def stop(self) -> None:
        self._running = False
        # Cancel any in-flight snapshot fetches
        for task in self._snapshot_tasks.values():
            task.cancel()
        self._snapshot_tasks.clear()
        if self._ws:
            await self._ws.close()

    # ── Dynamic asset management ──────────────────────────────────────
    async def add_assets(self, new_books: dict[str, L2OrderBook]) -> None:
        """Subscribe to additional assets on the live connection."""
        for asset_id, book in new_books.items():
            if asset_id in self._books:
                continue
            self._books[asset_id] = book
            if self._ws:
                try:
                    msg = {
                        "type": "subscribe",
                        "channel": "book",
                        "assets_ids": [asset_id],
                    }
                    await self._ws.send(json.dumps(msg))
                    log.info("l2_ws_subscribed_dynamic", asset_id=asset_id)
                    # Trigger snapshot fetch
                    self._schedule_snapshot(asset_id)
                except Exception as exc:
                    log.warning(
                        "l2_ws_subscribe_failed",
                        asset_id=asset_id,
                        error=str(exc),
                    )

    async def remove_assets(self, asset_ids: list[str]) -> None:
        """Unsubscribe from assets and remove from registry."""
        for aid in asset_ids:
            self._books.pop(aid, None)
            task = self._snapshot_tasks.pop(aid, None)
            if task:
                task.cancel()
            if self._ws:
                try:
                    msg = {
                        "type": "unsubscribe",
                        "channel": "book",
                        "assets_ids": [aid],
                    }
                    await self._ws.send(json.dumps(msg))
                    log.info("l2_ws_unsubscribed", asset_id=aid)
                except Exception as exc:
                    log.warning(
                        "l2_ws_unsubscribe_failed",
                        asset_id=aid,
                        error=str(exc),
                    )

    # ═══════════════════════════════════════════════════════════════════════
    #  Internal connection
    # ═══════════════════════════════════════════════════════════════════════
    async def _connect_and_consume(self) -> None:
        async with websockets.connect(self._ws_url, ping_interval=20) as ws:
            self._ws = ws
            self._last_message_time = time.time()
            log.info("l2_ws_connected", url=self._ws_url)

            # Subscribe to book channel for all tracked assets
            asset_ids = list(self._books.keys())
            if asset_ids:
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "book",
                    "assets_ids": asset_ids,
                }
                await ws.send(json.dumps(subscribe_msg))
                for aid in asset_ids:
                    log.info("l2_ws_subscribed", asset_id=aid, channel="book")

            # Only snapshot assets that aren't already SYNCED.
            # On first connect everything is EMPTY so all get snapshots.
            # On reconnect, SYNCED books stay SYNCED and just receive
            # new deltas — only DESYNCED / EMPTY / BUFFERING books need
            # a fresh snapshot.
            for aid in asset_ids:
                book = self._books.get(aid)
                if book is None or book.state != BookState.SYNCED:
                    self._schedule_snapshot(aid)

            # Silence watchdog
            silence_task = asyncio.create_task(
                self._silence_watchdog(ws), name="l2_silence_watchdog"
            )
            try:
                async for raw in ws:
                    if not self._running:
                        break
                    self._last_message_time = time.time()
                    try:
                        msg = json.loads(raw)
                        self._handle_message(msg)
                    except json.JSONDecodeError:
                        if raw.strip() == "INVALID OPERATION":
                            log.debug("l2_ws_invalid_op")
                        else:
                            log.warning("l2_ws_bad_json", raw=raw[:200])
            finally:
                silence_task.cancel()

    async def _silence_watchdog(self, ws: Any) -> None:
        """Close the WS if no messages for ``_silence_timeout`` seconds."""
        while self._running:
            await asyncio.sleep(1.0)
            elapsed = time.time() - self._last_message_time
            if elapsed > self._silence_timeout:
                log.warning(
                    "l2_ws_silence_timeout",
                    silence_s=round(elapsed, 1),
                    threshold_s=self._silence_timeout,
                )
                await ws.close()
                return

    # ═══════════════════════════════════════════════════════════════════════
    #  Message dispatch (synchronous — no queue)
    # ═══════════════════════════════════════════════════════════════════════
    def _handle_message(self, msg: dict | list) -> None:
        """Route incoming L2 message to the correct book."""
        if self._recorder and isinstance(msg, dict):
            self._recorder.enqueue("l2", msg)

        if isinstance(msg, list):
            for sub in msg:
                if isinstance(sub, dict):
                    self._handle_message(sub)
            return

        event_type = msg.get("event_type") or msg.get("type", "")
        asset_id = msg.get("asset_id") or ""

        book = self._books.get(asset_id)
        if not book:
            return

        if event_type in _DELTA_EVENTS:
            book.on_delta(msg)
        elif event_type in _SNAPSHOT_EVENTS:
            # Some WS implementations push full snapshots inline
            # Treat as snapshot data
            try:
                result = book.load_snapshot(
                    msg, trigger="periodic_server_snapshot",
                )
                if asyncio.iscoroutine(result):
                    task = asyncio.ensure_future(result)
                    task.add_done_callback(_log_task_exception)
            except Exception:
                log.error(
                    "l2ws_snapshot_dispatch_error",
                    asset_id=asset_id,
                    exc_info=True,
                )

    # ═══════════════════════════════════════════════════════════════════════
    #  Snapshot orchestration
    # ═══════════════════════════════════════════════════════════════════════
    def _schedule_snapshot(
        self, asset_id: str, *, trigger: str = "initial",
    ) -> None:
        """Schedule an async REST snapshot fetch for an asset."""
        # Cancel any in-flight fetch for this asset
        existing = self._snapshot_tasks.pop(asset_id, None)
        if existing and not existing.done():
            existing.cancel()

        task = asyncio.ensure_future(
            self._fetch_and_apply_snapshot(asset_id, trigger=trigger)
        )
        task.add_done_callback(_log_task_exception)
        self._snapshot_tasks[asset_id] = task

    async def _fetch_and_apply_snapshot(
        self, asset_id: str, *, trigger: str = "initial",
    ) -> None:
        """Fetch a REST snapshot and apply it to the book."""
        book = self._books.get(asset_id)
        if not book:
            return

        max_retries = settings.strategy.l2_seq_gap_max_retries

        for attempt in range(1, max_retries + 1):
            book.begin_buffering()

            data = await fetch_l2_snapshot(asset_id)
            if data is None:
                log.warning(
                    "l2_snapshot_retry",
                    asset_id=asset_id,
                    attempt=attempt,
                    max_retries=max_retries,
                )
                await asyncio.sleep(1.0 * attempt)
                continue

            success = await book.load_snapshot(data, trigger=trigger)
            if success:
                return

            log.warning(
                "l2_snapshot_apply_failed",
                asset_id=asset_id,
                attempt=attempt,
            )
            await asyncio.sleep(1.0 * attempt)

        log.error(
            "l2_snapshot_exhausted_retries",
            asset_id=asset_id,
            max_retries=max_retries,
        )
        # Transition out of BUFFERING so the next delta or reconnect
        # can trigger a fresh recovery instead of staying zombie.
        book.reset()

    async def _on_book_desync(self, asset_id: str) -> None:
        """Callback invoked by L2OrderBook when a sequence gap is detected."""
        log.warning("l2_desync_recovery_triggered", asset_id=asset_id)
        self._schedule_snapshot(asset_id, trigger="desync_recovery")
