"""
WebSocket client for the Polymarket CLOB.

Subscribes to real-time order book and trade data for monitored markets.
Emits normalised trade events to the OHLCV aggregator via an asyncio Queue.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import websockets
import websockets.exceptions

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


# ── Normalised trade event ─────────────────────────────────────────────────
@dataclass
class TradeEvent:
    """A single trade tick from the CLOB WebSocket."""

    timestamp: float          # unix epoch seconds
    market_id: str            # condition_id / token_id
    asset_id: str             # specific outcome token
    side: str                 # "buy" or "sell"
    price: float              # in [0, 1] dollars
    size: float               # number of shares
    is_yes: bool              # True if this is a YES outcome token
    is_taker: bool = False    # True if trade crossed the spread (aggressor)


# ── WebSocket subscriber ───────────────────────────────────────────────────
class MarketWebSocket:
    """Persistent WebSocket connection to Polymarket CLOB market stream."""

    # ── Exponential backoff constants ──────────────────────────────────────
    _BACKOFF_BASE: float = 1.0     # initial retry delay in seconds
    _BACKOFF_MAX: float = 60.0     # ceiling for retry delay

    def __init__(
        self,
        asset_ids: list[str],
        trade_queue: asyncio.Queue[TradeEvent],
        *,
        book_queue: asyncio.Queue | None = None,
        ws_url: str | None = None,
        queue_maxsize: int = 1000,
        recorder: Any | None = None,
    ):
        self.asset_ids = asset_ids
        self._recorder = recorder  # Optional MarketDataRecorder
        # ── Fix 2: Bounded queue — drop oldest when full ───────────────────
        self.trade_queue = trade_queue
        self.book_queue: asyncio.Queue = book_queue or asyncio.Queue(maxsize=2000)
        self._queue_maxsize = queue_maxsize
        self.ws_url = ws_url or settings.clob_ws_url
        self._ws: Any = None
        self._running = False
        self._last_message_time: float = 0.0
        self._silence_timeout = settings.strategy.ws_silence_timeout_s

        # Cumulative reconnect counter (never reset — used by health reporter)
        self.reconnect_count: int = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Connect and begin consuming messages.  Auto-reconnects with
        exponential backoff + jitter to avoid hammering the server."""
        self._running = True
        attempt = 0
        while self._running:
            try:
                await self._connect_and_consume()
                attempt = 0           # reset on a clean session
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
                log.warning(
                    "ws_disconnected",
                    error=str(exc),
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
        if self._ws:
            await self._ws.close()

    async def add_assets(self, new_ids: list[str]) -> None:
        """Subscribe to additional asset IDs on the live connection."""
        added = [aid for aid in new_ids if aid not in self.asset_ids]
        if not added:
            return
        self.asset_ids.extend(added)
        if self._ws:
            for aid in added:
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "market",
                    "assets_ids": [aid],
                }
                try:
                    await self._ws.send(json.dumps(subscribe_msg))
                    log.info("ws_subscribed_dynamic", asset_id=aid)
                except Exception as exc:
                    log.warning("ws_subscribe_failed", asset_id=aid, error=str(exc))

    async def remove_assets(self, ids_to_remove: list[str]) -> None:
        """Unsubscribe from asset IDs and remove from tracking."""
        removing = [aid for aid in ids_to_remove if aid in self.asset_ids]
        if not removing:
            return
        for aid in removing:
            self.asset_ids.remove(aid)
        if self._ws:
            for aid in removing:
                try:
                    unsub_msg = {"type": "unsubscribe", "channel": "market", "assets_ids": [aid]}
                    await self._ws.send(json.dumps(unsub_msg))
                    log.info("ws_unsubscribed", asset_id=aid)
                except Exception as exc:
                    log.warning("ws_unsubscribe_failed", asset_id=aid, error=str(exc))

    # ── internal ───────────────────────────────────────────────────────────
    async def _connect_and_consume(self) -> None:
        async with websockets.connect(self.ws_url, ping_interval=20) as ws:
            self._ws = ws
            self._last_message_time = time.time()
            log.info("ws_connected", url=self.ws_url)

            # Subscribe to market channel (trades + price_change events)
            subscribe_msg = {
                "type": "subscribe",
                "channel": "market",
                "assets_ids": self.asset_ids,
            }
            await ws.send(json.dumps(subscribe_msg))
            for asset_id in self.asset_ids:
                log.info("ws_subscribed", asset_id=asset_id, channels="market")

            # Launch silence watchdog alongside message consumer
            silence_task = asyncio.create_task(
                self._silence_watchdog(ws), name="ws_silence_watchdog"
            )
            try:
                async for raw in ws:
                    if not self._running:
                        break
                    self._last_message_time = time.time()
                    try:
                        msg = json.loads(raw)
                        await self._handle_message(msg)
                    except json.JSONDecodeError:
                        log.warning("ws_bad_json", raw=raw[:200])
            finally:
                silence_task.cancel()

    async def _silence_watchdog(self, ws: Any) -> None:
        """Close the WebSocket if no messages arrive for ``_silence_timeout``
        seconds, forcing a reconnect via the outer backoff loop."""
        while self._running:
            await asyncio.sleep(1.0)
            elapsed = time.time() - self._last_message_time
            if elapsed > self._silence_timeout:
                log.warning(
                    "ws_silence_timeout",
                    silence_s=round(elapsed, 1),
                    threshold_s=self._silence_timeout,
                )
                await ws.close()
                return

    async def _handle_message(self, msg: dict | list) -> None:
        """Parse incoming WS messages and emit TradeEvents."""
        if self._recorder and isinstance(msg, dict):
            self._recorder.enqueue("trade", msg)

        # Polymarket may send batch messages as a JSON array
        if isinstance(msg, list):
            for sub in msg:
                if isinstance(sub, dict):
                    await self._handle_message(sub)
            return

        event_type = msg.get("event_type") or msg.get("type", "")

        # Polymarket sends trade messages under different schemas depending
        # on the endpoint version.  We handle the common ones.
        if event_type in ("last_trade_price", "trade", "tick"):
            trades = msg.get("data", [msg])
            if isinstance(trades, dict):
                trades = [trades]
            for trade_data in trades:
                event = self._parse_trade(trade_data, msg)
                if event:
                    await self._enqueue(event)

        elif event_type in ("price_change", "book"):
            # Route orderbook events to the book queue
            try:
                self.book_queue.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    self.book_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.book_queue.put_nowait(msg)

    async def _enqueue(self, event: TradeEvent) -> None:
        """Put *event* on the queue. If the queue is full, drop the oldest
        message so the bot always processes the most recent price state."""
        try:
            self.trade_queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.trade_queue.get_nowait()   # discard oldest
            except asyncio.QueueEmpty:
                pass
            self.trade_queue.put_nowait(event)

    def _parse_trade(self, data: dict, parent: dict) -> TradeEvent | None:
        """Best-effort parse of a raw trade dict into a TradeEvent."""
        try:
            price = float(data.get("price", 0))
            size = float(data.get("size") or data.get("amount", 0))
            if price <= 0 or size <= 0:
                return None
            # Guard against NaN values which would corrupt OHLCV aggregation
            if price != price or size != size:  # NaN check
                return None

            asset_id = data.get("asset_id") or parent.get("asset_id", "")
            market_id = data.get("market") or data.get("condition_id") or parent.get("market", "")

            # Determine whether this is a YES or NO token.
            # Convention: the first asset in a two-outcome market is YES.
            outcome = (data.get("outcome") or data.get("side") or "").upper()
            is_yes = outcome != "NO"

            raw_ts = float(data.get("timestamp") or data.get("ts") or 0)
            # Polymarket sends µs-epoch; normalise to seconds so the
            # latency guard can compare against time.time().
            if raw_ts > 1e15:          # microseconds
                raw_ts /= 1_000_000
            elif raw_ts > 1e12:        # milliseconds
                raw_ts /= 1_000
            elif raw_ts == 0:
                raw_ts = time.time()
            # else: already seconds

            # Determine taker status: if the trade data includes a
            # 'taker_side' or 'aggressor' field use it; otherwise infer
            # from the 'side' field — market sells hitting bids are taker.
            is_taker = False
            taker_side = (data.get("taker_side") or data.get("aggressor") or "").lower()
            if taker_side in ("sell", "buy"):
                is_taker = True
            elif data.get("is_taker") is True:
                is_taker = True
            elif str(data.get("trade_type", "")).lower() == "taker":
                is_taker = True

            return TradeEvent(
                timestamp=raw_ts,
                market_id=str(market_id),
                asset_id=str(asset_id),
                side=data.get("side", "buy"),
                price=price,
                size=size,
                is_yes=is_yes,
                is_taker=is_taker,
            )
        except (ValueError, TypeError) as exc:
            log.debug("trade_parse_error", error=str(exc), data=data)
            return None
