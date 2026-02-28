"""
Anti-adverse-selection module — the **"Fast-Kill"** mechanism.

Monitors external price feeds (Binance BTC/USDC WebSocket) and Polygon
RPC head lag to detect "toxic flow" before it reaches the Polymarket
CLOB.  When the external price moves by more than *N* ticks while the
local orderbook is stale, all resting orders are cancelled within a
sub-100 ms window.

Usage (wired into ``TradingBot._run()``)::

    guard = AdverseSelectionGuard(executor, book_trackers)
    tasks.append(asyncio.create_task(guard.start()))
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from src.core.config import settings
from src.core.logger import get_logger

if TYPE_CHECKING:
    from src.data.orderbook import OrderbookTracker
    from src.trading.executor import OrderExecutor

log = get_logger(__name__)


@dataclass
class _BinanceTick:
    price: float
    ts: float  # local receipt time


class AdverseSelectionGuard:
    """Predictive cancellation engine driven by external price deltas.

    Three concurrent coroutines:
      1. **Binance WS consumer** — maintains a rolling window of BTC/USDC
         trade ticks and computes the 2-second price delta in Polymarket
         "ticks" (1 tick = $0.01).
      2. **Polygon head-lag poller** — calls ``eth_blockNumber`` and
         checks the latest block timestamp vs VPS time.
      3. **Decision loop** (fast-polling at 50 ms) — evaluates the
         cancel condition and fires ``executor.cancel_all()``.

    Parameters
    ----------
    executor:
        Shared ``OrderExecutor`` for cancellation.
    book_trackers:
        Dict of ``asset_id → OrderbookTracker`` to measure local book
        staleness.
    fast_kill_event:
        Shared ``asyncio.Event`` — *cleared* during a kill cooldown so
        all ``OrderChaser`` instances pause.
    """

    def __init__(
        self,
        executor: OrderExecutor,
        book_trackers: dict[str, OrderbookTracker],
        fast_kill_event: asyncio.Event,
    ):
        strat = settings.strategy
        self._executor = executor
        self._books = book_trackers
        self._kill_event = fast_kill_event

        self._tick_threshold = strat.adverse_sel_tick_threshold
        self._book_stale_ms = strat.adverse_sel_book_stale_ms
        self._cooldown_s = strat.adverse_sel_cooldown_s
        self._poll_s = strat.adverse_sel_poll_ms / 1000.0
        self._polygon_lag_ms = strat.adverse_sel_polygon_head_lag_ms
        self._binance_url = strat.binance_ws_url
        self._enabled = strat.adverse_sel_enabled

        # Rolling Binance tick window (last 5 seconds)
        self._ticks: deque[_BinanceTick] = deque(maxlen=500)
        self._polygon_head_lag_ms: float = 0.0
        self._cooldown_until: float = 0.0
        self._running = False
        self._cancel_count: int = 0

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch all sub-tasks and run until stopped."""
        if not self._enabled:
            log.info("adverse_sel_disabled")
            return

        self._running = True
        self._kill_event.set()  # start clear (chasers may proceed)

        tasks = [
            asyncio.create_task(self._binance_consumer(), name="binance_ws"),
            asyncio.create_task(self._polygon_poller(), name="polygon_head"),
            asyncio.create_task(self._decision_loop(), name="adverse_sel_loop"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self._running = False

    async def stop(self) -> None:
        self._running = False
        self._kill_event.set()

    # ── External price delta ────────────────────────────────────────────────

    @property
    def ext_delta_ticks(self) -> float:
        """Price move (in Polymarket ticks) over the last 2 seconds."""
        if len(self._ticks) < 2:
            return 0.0
        now = time.time()
        cutoff = now - 2.0
        recent = [t for t in self._ticks if t.ts >= cutoff]
        if len(recent) < 2:
            return 0.0
        first = recent[0].price
        last = recent[-1].price
        if first == 0:
            return 0.0
        return abs(last - first) / 0.01  # 1 tick = $0.01

    @property
    def max_book_age_ms(self) -> float:
        """Age of the stalest orderbook tracker (ms)."""
        if not self._books:
            return 0.0
        now = time.time()
        ages = [
            (now - t._last_update) * 1000
            for t in self._books.values()
            if t._last_update > 0
        ]
        return max(ages) if ages else 0.0

    # ── Binance WebSocket ───────────────────────────────────────────────────

    async def _binance_consumer(self) -> None:
        """Consume BTC/USDC trade ticks from Binance."""
        import websockets
        import websockets.exceptions

        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    self._binance_url, ping_interval=20
                ) as ws:
                    log.info("binance_ws_connected", url=self._binance_url)
                    backoff = 1.0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw)
                            price = float(data.get("p", 0))
                            if price > 0:
                                self._ticks.append(
                                    _BinanceTick(price=price, ts=time.time())
                                )
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("binance_ws_error", error=str(exc), retry_in=backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    # ── Polygon RPC head-lag ────────────────────────────────────────────────

    async def _polygon_poller(self) -> None:
        """Poll Polygon RPC ``eth_blockNumber`` + ``eth_getBlockByNumber``
        to estimate head lag between the chain and VPS time."""
        rpc_url = settings.polygon_rpc_url
        if not rpc_url:
            log.info("polygon_rpc_disabled", reason="no POLYGON_RPC_URL")
            return

        async with httpx.AsyncClient(timeout=3.0) as client:
            while self._running:
                try:
                    # Get latest block number
                    resp = await client.post(
                        rpc_url,
                        json={
                            "jsonrpc": "2.0",
                            "method": "eth_blockNumber",
                            "params": [],
                            "id": 1,
                        },
                    )
                    block_hex = resp.json().get("result", "0x0")
                    block_num = int(block_hex, 16)

                    # Get block timestamp
                    resp2 = await client.post(
                        rpc_url,
                        json={
                            "jsonrpc": "2.0",
                            "method": "eth_getBlockByNumber",
                            "params": [hex(block_num), False],
                            "id": 2,
                        },
                    )
                    block_data = resp2.json().get("result", {})
                    block_ts = int(block_data.get("timestamp", "0x0"), 16)
                    self._polygon_head_lag_ms = abs(time.time() - block_ts) * 1000
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.debug("polygon_head_error", error=str(exc))
                await asyncio.sleep(0.5)

    # ── Decision loop ───────────────────────────────────────────────────────

    async def _decision_loop(self) -> None:
        """Fast-polling loop that evaluates the cancel condition."""
        while self._running:
            await asyncio.sleep(self._poll_s)
            now = time.time()

            # Respect cooldown
            if now < self._cooldown_until:
                continue

            should_cancel = False

            # Condition 1: external price moved + local book stale
            delta_ticks = self.ext_delta_ticks
            book_age = self.max_book_age_ms
            reason = ""
            if delta_ticks >= self._tick_threshold and book_age >= self._book_stale_ms:
                reason = "ext_delta+stale_book"
                should_cancel = True

            # Condition 2: Polygon head lag too high
            if self._polygon_head_lag_ms > self._polygon_lag_ms:
                reason = "polygon_head_lag"
                should_cancel = True

            if should_cancel:
                # Skip no-op kills when there are no resting orders
                open_count = self._executor.open_order_count
                if open_count == 0:
                    log.debug(
                        "adverse_sel_skip",
                        reason=reason,
                        delta_ticks=round(delta_ticks, 2),
                        book_age_ms=round(book_age, 1),
                    )
                    self._cooldown_until = time.time() + self._cooldown_s
                    continue
                log.warning(
                    "adverse_sel_trigger",
                    reason=reason,
                    delta_ticks=round(delta_ticks, 2),
                    book_age_ms=round(book_age, 1),
                    open_orders=open_count,
                )
                await self._execute_fast_kill()

    async def _execute_fast_kill(self) -> None:
        """Cancel all resting orders and enter cooldown."""
        t0 = time.perf_counter()
        self._kill_event.clear()  # pause all chasers

        count = await self._executor.cancel_all()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._cooldown_until = time.time() + self._cooldown_s
        self._cancel_count += 1

        log.warning(
            "fast_kill_executed",
            cancelled=count,
            elapsed_ms=round(elapsed_ms, 1),
            cooldown_s=self._cooldown_s,
            total_kills=self._cancel_count,
        )

        # Schedule cooldown release
        asyncio.get_event_loop().call_later(
            self._cooldown_s, self._kill_event.set
        )
