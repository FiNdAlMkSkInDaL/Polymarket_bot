"""
WebSocket client for the Polymarket CLOB.

Subscribes to real-time order book and trade data for monitored markets.
Emits normalised trade events to the OHLCV aggregator via an asyncio Queue.
"""

from __future__ import annotations

import asyncio
import json
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


# ── WebSocket subscriber ───────────────────────────────────────────────────
class MarketWebSocket:
    """Persistent WebSocket connection to Polymarket CLOB market stream."""

    def __init__(
        self,
        asset_ids: list[str],
        trade_queue: asyncio.Queue[TradeEvent],
        *,
        ws_url: str | None = None,
    ):
        self.asset_ids = asset_ids
        self.trade_queue = trade_queue
        self.ws_url = ws_url or settings.clob_ws_url
        self._ws: Any = None
        self._running = False

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Connect and begin consuming messages.  Auto-reconnects."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_consume()
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidStatusCode,
                ConnectionError,
                OSError,
            ) as exc:
                log.warning("ws_disconnected", error=str(exc))
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self._running = False
                break

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    # ── internal ───────────────────────────────────────────────────────────
    async def _connect_and_consume(self) -> None:
        async with websockets.connect(self.ws_url, ping_interval=20) as ws:
            self._ws = ws
            log.info("ws_connected", url=self.ws_url)

            # Subscribe to each asset (both YES and NO outcome tokens)
            for asset_id in self.asset_ids:
                subscribe_msg = {
                    "type": "market",
                    "assets_ids": [asset_id],
                }
                await ws.send(json.dumps(subscribe_msg))
                log.info("ws_subscribed", asset_id=asset_id)

            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                    await self._handle_message(msg)
                except json.JSONDecodeError:
                    log.warning("ws_bad_json", raw=raw[:200])

    async def _handle_message(self, msg: dict) -> None:
        """Parse incoming WS messages and emit TradeEvents."""
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
                    await self.trade_queue.put(event)

        elif event_type in ("price_change", "book"):
            # Order book snapshots may be useful later; skip for now.
            pass

    def _parse_trade(self, data: dict, parent: dict) -> TradeEvent | None:
        """Best-effort parse of a raw trade dict into a TradeEvent."""
        try:
            price = float(data.get("price", 0))
            size = float(data.get("size") or data.get("amount", 0))
            if price <= 0 or size <= 0:
                return None

            asset_id = data.get("asset_id") or parent.get("asset_id", "")
            market_id = data.get("market") or data.get("condition_id") or parent.get("market", "")

            # Determine whether this is a YES or NO token.
            # Convention: the first asset in a two-outcome market is YES.
            outcome = (data.get("outcome") or data.get("side") or "").upper()
            is_yes = outcome != "NO"

            return TradeEvent(
                timestamp=float(data.get("timestamp") or data.get("ts") or time.time()),
                market_id=str(market_id),
                asset_id=str(asset_id),
                side=data.get("side", "buy"),
                price=price,
                size=size,
                is_yes=is_yes,
            )
        except (ValueError, TypeError) as exc:
            log.debug("trade_parse_error", error=str(exc), data=data)
            return None
