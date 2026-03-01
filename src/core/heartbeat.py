"""
Tiered WebSocket health monitor — transport-layer heartbeat with
position-aware asset staleness checks.

Architecture
────────────
The monitor operates on two independent layers:

1. **Transport layer** — checks ``L2WebSocket._last_message_time``,
   which is updated on every incoming WS data frame (deltas,
   snapshots).  WS control frames (pong) do *not* update this
   timestamp, so during quiet markets the gap can naturally reach
   ~2 s without indicating a dead connection.

   To avoid false-positive flapping, suspension requires
   ``heartbeat_stale_count`` **consecutive** stale checks (default 2)
   before acting.  A single transient gap is tolerated; the 5-second
   ``ws_silence_timeout_s`` watchdog provides the hard safety net for
   truly dead connections.

2. **Position layer** — for assets with **open positions only**,
   checks per-book ``_last_update`` to ensure the specific books
   we are *trading against* are receiving data.  Inactive markets
   without positions are ignored — they are irrelevant to execution
   safety.

Suspension occurs when either layer is stale for ``stale_ms``
(transport requires consecutive confirmation).  Recovery requires
the transport layer to be healthy (the WS must be alive), at which
point individual books will naturally re-sync.

Usage::

    heartbeat = BookHeartbeat(
        book_trackers=trackers,
        latency_guard=guard,
        fast_kill_event=event,
        executor=executor,
        ws_transport=l2_ws,           # provides _last_message_time
        get_position_assets=fn,       # returns set of asset_ids with open positions
    )
    asyncio.create_task(heartbeat.run())
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable

import httpx

from src.core.config import settings
from src.core.logger import get_logger

if TYPE_CHECKING:
    from src.core.latency_guard import LatencyGuard
    from src.data.orderbook import OrderbookTracker
    from src.trading.executor import OrderExecutor

log = get_logger(__name__)


def _no_positions() -> set[str]:
    """Default callback when no position manager is wired."""
    return set()


class BookHeartbeat:
    """Tiered WebSocket health monitor.

    Parameters
    ----------
    book_trackers:
        Dict of ``asset_id → OrderbookTracker``.  Shared reference — the
        bot may add/remove entries; heartbeat always reads the current
        contents.
    latency_guard:
        Shared ``LatencyGuard``.  Will be force-blocked when the
        transport or a positioned asset goes stale.
    fast_kill_event:
        Shared ``asyncio.Event`` from the adverse-selection guard.
        Cleared on stale detection to pause all chasers.
    executor:
        Shared ``OrderExecutor`` — used to pull all resting quotes.
    ws_transport:
        The ``L2WebSocket`` instance (or any object with a
        ``_last_message_time: float`` attribute).  When ``None``, falls
        back to per-tracker scanning (legacy mode / tests).
    get_position_assets:
        Callable returning the set of ``asset_id`` strings that currently
        have open positions.  Only these assets are checked at the
        position layer.
    telegram:
        Optional ``TelegramAlerter`` for notifications.
    polygon_checker:
        Optional ``PolygonHeadLagChecker`` for Polygon RPC health.
        When provided, the heartbeat monitors blockchain head lag as
        a general health signal (affects settlement, not price
        discovery).  Suspend if lag exceeds threshold.
    """

    def __init__(
        self,
        book_trackers: dict[str, "OrderbookTracker"],
        latency_guard: "LatencyGuard",
        fast_kill_event: asyncio.Event,
        executor: "OrderExecutor",
        *,
        ws_transport: Any | None = None,
        get_position_assets: Callable[[], set[str]] | None = None,
        telegram: object | None = None,
        polygon_checker: "PolygonHeadLagChecker | None" = None,
    ):
        strat = settings.strategy
        self._books = book_trackers
        self._guard = latency_guard
        self._kill_event = fast_kill_event
        self._executor = executor
        self._telegram = telegram
        self._ws_transport = ws_transport
        self._get_position_assets = get_position_assets or _no_positions
        self._polygon_checker = polygon_checker

        self._check_interval_s = strat.heartbeat_check_ms / 1000.0
        self._stale_ms = strat.heartbeat_stale_ms
        self._stale_count_threshold = strat.heartbeat_stale_count
        self._running = False
        self._is_suspended = False
        self._suspend_reason: str = ""
        self._transport_stale_streak: int = 0

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

        # ── Layer 1: Transport health ─────────────────────────────────
        #
        # If we have a WS transport reference, check the last time *any*
        # message arrived on the wire.  This is the definitive signal.
        transport_gap_ms = self._transport_gap_ms(now)
        if transport_gap_ms is not None:
            if transport_gap_ms > self._stale_ms:
                self._transport_stale_streak += 1
                if self._transport_stale_streak >= self._stale_count_threshold:
                    if not self._is_suspended:
                        await self._suspend(
                            transport_gap_ms,
                            reason="ws_transport_dead",
                            detail="no WS message received",
                        )
                    return
                # Sub-threshold streak — tolerate transient gap
                log.debug(
                    "heartbeat_transient_gap",
                    gap_ms=round(transport_gap_ms, 1),
                    streak=self._transport_stale_streak,
                    threshold=self._stale_count_threshold,
                )
                return
            else:
                # Transport is healthy — reset streak counter.
                self._transport_stale_streak = 0
                # If we were suspended for a transport reason, resume.
                if self._is_suspended and self._suspend_reason == "ws_transport_dead":
                    await self._resume()
                    return

        # ── Layer 2: Positioned-asset health ──────────────────────────
        #
        # Only check assets we have open positions on.  Low-activity
        # markets without positions are irrelevant — a 10s gap on them
        # is not a safety concern.
        position_assets = self._get_position_assets()
        if position_assets:
            stalest_ms = 0.0
            stalest_asset = ""
            for asset_id in position_assets:
                tracker = self._books.get(asset_id)
                if tracker is None or tracker._last_update <= 0:
                    continue
                age_ms = (now - tracker._last_update) * 1000
                if age_ms > stalest_ms:
                    stalest_ms = age_ms
                    stalest_asset = asset_id

            # Position-level stale: the book we are actively trading
            # against hasn't updated.  Use a larger multiplier (3×)
            # since individual markets are naturally less frequent.
            position_stale_ms = self._stale_ms * 3
            if stalest_ms > position_stale_ms:
                if not self._is_suspended:
                    await self._suspend(
                        stalest_ms,
                        reason="position_book_stale",
                        detail=stalest_asset[:16],
                    )
                return

        # ── Layer 2.5: Polygon head-lag health ─────────────────────────
        #
        # If a PolygonHeadLagChecker is wired, check blockchain head lag.
        # Excessive lag indicates Polygon is stalled, which affects
        # settlement.  This is a general health signal, not an adverse
        # selection signal — it belongs in the heartbeat, not the guard.
        if self._polygon_checker is not None:
            try:
                is_healthy, lag_ms = await self._polygon_checker.check()
                if not is_healthy:
                    if not self._is_suspended:
                        await self._suspend(
                            lag_ms,
                            reason="polygon_head_lag",
                            detail=f"lag {lag_ms:.0f}ms",
                        )
                    return
                elif self._is_suspended and self._suspend_reason == "polygon_head_lag":
                    await self._resume()
                    return
            except Exception as exc:
                log.debug("polygon_check_error", error=str(exc))

        # ── Layer 3: Fallback — legacy per-tracker scan ───────────────
        #
        # When no WS transport is wired (tests, non-L2 mode), fall back
        # to the original freshest-tracker heuristic.
        if self._ws_transport is None:
            best_gap = self._freshest_tracker_gap_ms(now)
            if best_gap is not None and best_gap > self._stale_ms:
                if not self._is_suspended:
                    await self._suspend(
                        best_gap,
                        reason="legacy_stale",
                        detail="freshest tracker",
                    )
                return

        # ── All layers healthy — resume if suspended ──────────────────
        if self._is_suspended:
            await self._resume()

    # ── Transport layer ────────────────────────────────────────────────────

    def _transport_gap_ms(self, now: float) -> float | None:
        """Milliseconds since the last WS frame, or None if no transport."""
        if self._ws_transport is None:
            return None
        last = getattr(self._ws_transport, "_last_message_time", 0.0)
        if last <= 0:
            return None  # no messages received yet (still connecting)
        return (now - last) * 1000

    # ── Freshest tracker (legacy fallback) ─────────────────────────────────

    def _freshest_tracker_gap_ms(self, now: float) -> float | None:
        """Return the gap (ms) of the freshest active tracker, or None."""
        best = float("inf")
        for tracker in self._books.values():
            if tracker._last_update <= 0:
                continue
            age_ms = (now - tracker._last_update) * 1000
            if age_ms < best:
                best = age_ms
            server_time = getattr(tracker, "_last_server_time", 0.0)
            if server_time > 0:
                gap = (now - server_time) * 1000
                if gap < best:
                    best = gap
        return best if best < float("inf") else None

    # ── Suspend / Resume ───────────────────────────────────────────────────

    async def _suspend(self, gap_ms: float, *, reason: str, detail: str) -> None:
        """Suspend all execution due to stale data."""
        self._is_suspended = True
        self._suspend_reason = reason

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
            reason=reason,
            detail=detail,
            orders_cancelled=count,
        )

        if self._telegram and hasattr(self._telegram, "send"):
            try:
                await self._telegram.send(
                    f"💓 <b>Heartbeat STALE</b> — execution suspended.\n"
                    f"Reason: {reason}\n"
                    f"Gap: {gap_ms:.0f}ms (threshold: {self._stale_ms}ms)\n"
                    f"Cancelled {count} resting orders."
                )
            except Exception:
                pass

    async def _resume(self) -> None:
        """Resume execution after fresh data arrives."""
        self._is_suspended = False
        self._suspend_reason = ""
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


class PolygonHeadLagChecker:
    """Standalone Polygon RPC head-lag health check.

    Polls ``eth_blockNumber`` + ``eth_getBlockByNumber`` to estimate
    how far the chain head is behind wall-clock time.  This is useful
    as a general system health signal (settlement may be delayed) but
    is **not** an adverse-selection signal — it does not predict price
    movements on the CLOB.

    Previously lived inside ``AdverseSelectionGuard`` as a kill
    condition; now resides in the heartbeat module where it is checked
    as part of the tiered health monitor.

    Parameters
    ----------
    rpc_url:
        Polygon JSON-RPC endpoint URL.
    lag_threshold_ms:
        Maximum acceptable head lag in milliseconds.  If the latest
        block timestamp is older than this, the check reports unhealthy.
    """

    def __init__(self, rpc_url: str, lag_threshold_ms: float = 3000.0):
        self._rpc_url = rpc_url
        self._lag_threshold_ms = lag_threshold_ms
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=3.0)
        return self._client

    async def check(self) -> tuple[bool, float]:
        """Perform one head-lag check.

        Returns
        -------
        (is_healthy, lag_ms)
            ``is_healthy`` is ``True`` when lag is within threshold.
        """
        if not self._rpc_url:
            return True, 0.0

        client = self._get_client()

        # Get latest block number
        resp = await client.post(
            self._rpc_url,
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
            self._rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(block_num), False],
                "id": 2,
            },
        )
        block_data = resp2.json().get("result", {})
        block_ts = int(block_data.get("timestamp", "0x0"), 16)
        lag_ms = abs(time.time() - block_ts) * 1000

        return lag_ms <= self._lag_threshold_ms, lag_ms

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
