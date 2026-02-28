"""
Multi-source heartbeat — detects stale orderbook data and silently
disconnected WebSockets.

Compares the ``server_time`` in each ``OrderbookSnapshot`` against the
VPS local clock.  If the gap exceeds the configured threshold, all
execution is suspended until the WebSocket reconnects and proves
freshness.

Usage::

    heartbeat = BookHeartbeat(book_trackers, latency_guard, fast_kill_event, executor)
    asyncio.create_task(heartbeat.run())
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.core.config import settings
from src.core.logger import get_logger

if TYPE_CHECKING:
    from src.core.latency_guard import LatencyGuard
    from src.data.orderbook import OrderbookTracker
    from src.trading.executor import OrderExecutor

log = get_logger(__name__)


class BookHeartbeat:
    """Periodic staleness check across all orderbook trackers.

    Parameters
    ----------
    book_trackers:
        Dict of ``asset_id → OrderbookTracker``.  Shared reference — the
        bot may add/remove entries; heartbeat always reads the current
        contents.
    latency_guard:
        Shared ``LatencyGuard``.  Will be force-blocked when a stale
        book is detected.
    fast_kill_event:
        Shared ``asyncio.Event`` from the adverse-selection guard.
        Cleared on stale detection to pause all chasers.
    executor:
        Shared ``OrderExecutor`` — used to pull all resting quotes.
    telegram:
        Optional ``TelegramAlerter`` for notifications.
    """

    def __init__(
        self,
        book_trackers: dict[str, "OrderbookTracker"],
        latency_guard: "LatencyGuard",
        fast_kill_event: asyncio.Event,
        executor: "OrderExecutor",
        *,
        telegram: object | None = None,
    ):
        strat = settings.strategy
        self._books = book_trackers
        self._guard = latency_guard
        self._kill_event = fast_kill_event
        self._executor = executor
        self._telegram = telegram

        self._check_interval_s = strat.heartbeat_check_ms / 1000.0
        self._stale_ms = strat.heartbeat_stale_ms
        self._running = False
        self._is_suspended = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main heartbeat loop — runs every ``heartbeat_check_ms``."""
        self._running = True
        while self._running:
            try:
                await asyncio.sleep(self._check_interval_s)
                await self._check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("heartbeat_error", error=str(exc))

    def stop(self) -> None:
        self._running = False

    # ── Core check ──────────────────────────────────────────────────────────

    async def _check(self) -> None:
        now = time.time()

        if not self._books:
            return

        # ── Compute freshest and stalest gaps across all trackers ──────
        #
        # Purpose: detect dead WebSocket connections, NOT individual
        # market inactivity.  With 100+ markets subscribed, low-activity
        # markets will naturally have gaps >> 1.5 s even when the WS is
        # perfectly healthy.
        #
        # Strategy: use the *freshest* tracker (min gap) to judge WS
        # health.  If even the freshest tracker is stale, the connection
        # is genuinely dead and we should suspend.
        freshest_local_ms = float("inf")
        freshest_server_ms = float("inf")
        stalest_local_ms = 0.0
        stalest_asset = ""
        active_count = 0

        for asset_id, tracker in self._books.items():
            if tracker._last_update <= 0:
                continue  # never received data yet

            active_count += 1
            local_age = (now - tracker._last_update) * 1000

            if local_age < freshest_local_ms:
                freshest_local_ms = local_age
            if local_age > stalest_local_ms:
                stalest_local_ms = local_age
                stalest_asset = asset_id

            server_time = getattr(tracker, "_last_server_time", 0.0)
            if server_time > 0:
                server_gap = (now - server_time) * 1000
                if server_gap < freshest_server_ms:
                    freshest_server_ms = server_gap

        if active_count == 0:
            return  # no trackers have ever received data

        # Best (freshest) gap across both local and server clocks
        best_gap = freshest_local_ms
        if freshest_server_ms < float("inf"):
            best_gap = min(best_gap, freshest_server_ms)

        if best_gap > self._stale_ms:
            if not self._is_suspended:
                await self._suspend(best_gap, stalest_asset)
        else:
            if self._is_suspended:
                await self._resume()

    async def _suspend(self, gap_ms: float, asset_id: str) -> None:
        """Suspend all execution due to stale book data."""
        self._is_suspended = True

        # Force the latency guard into BLOCKED state
        self._guard.force_block("heartbeat_stale")

        # Pause chasers
        self._kill_event.clear()

        # Pull all resting quotes
        count = await self._executor.cancel_all()

        log.warning(
            "heartbeat_stale_detected",
            max_gap_ms=round(gap_ms, 1),
            threshold_ms=self._stale_ms,
            stalest_asset=asset_id[:16],
            orders_cancelled=count,
        )

        if self._telegram and hasattr(self._telegram, "send"):
            try:
                await self._telegram.send(
                    f"💓 <b>Heartbeat STALE</b> — execution suspended.\n"
                    f"Gap: {gap_ms:.0f}ms (threshold: {self._stale_ms}ms)\n"
                    f"Cancelled {count} resting orders."
                )
            except Exception:
                pass

    async def _resume(self) -> None:
        """Resume execution after fresh data arrives."""
        self._is_suspended = False
        self._kill_event.set()

        # Reset the latency guard so it transitions back to HEALTHY
        # immediately instead of waiting for N consecutive healthy ticks.
        self._guard.reset()

        log.info("heartbeat_recovered")

        if self._telegram and hasattr(self._telegram, "send"):
            try:
                await self._telegram.send("💚 <b>Heartbeat recovered</b> — execution resumed.")
            except Exception:
                pass

    @property
    def is_suspended(self) -> bool:
        return self._is_suspended
