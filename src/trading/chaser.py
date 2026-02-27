"""
Passive-aggressive order chaser — ``OrderChaser``.

Manages a single resting ``POST_ONLY`` limit order for one leg of a
position (entry BUY or exit SELL).  Monitors the best bid/ask via a live
``OrderbookTracker`` and cancel-and-replaces the order when the BBO
moves, keeping the order at top-of-book to capture maker rebates.

The chaser respects:
  - A maximum chase depth (``max_chase_depth_cents``) beyond the
    original signal price — prevents chasing into adverse moves.
  - A rate-limited re-quote cadence (``chase_interval_ms``) to stay
    within the CLOB's API rate ceiling.
  - The ``LatencyGuard`` state — pauses requoting (but does NOT cancel
    the resting order) when data is stale.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import TYPE_CHECKING

from src.core.config import settings
from src.core.logger import get_logger
from src.data.orderbook import OrderbookTracker
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus

if TYPE_CHECKING:
    from src.core.latency_guard import LatencyGuard

log = get_logger(__name__)


class ChaserState(str, Enum):
    QUOTING = "QUOTING"
    CANCELLING = "CANCELLING"
    REQUOTING = "REQUOTING"
    FILLED = "FILLED"
    ABANDONED = "ABANDONED"


class OrderChaser:
    """State machine for a single passive-aggressive limit order.

    Parameters
    ----------
    executor:
        The shared ``OrderExecutor``.
    book:
        The ``OrderbookTracker`` for the asset being traded.
    market_id:
        The CLOB market / condition ID.
    asset_id:
        The specific outcome token ID.
    side:
        ``BUY`` for entries, ``SELL`` for exits.
    target_size:
        Number of shares to fill.
    anchor_price:
        The original signal price — used as a reference to measure chase
        drift.  For BUY this is the initial best-ask; for SELL it is the
        computed TP target.
    latency_guard:
        Shared ``LatencyGuard`` instance (optional — if ``None``, latency
        checks are skipped).
    max_chase_depth_cents:
        How far (in cents) from ``anchor_price`` the chaser may follow
        the BBO before abandoning.
    chase_interval_ms:
        Minimum milliseconds between requote iterations.
    """

    def __init__(
        self,
        executor: OrderExecutor,
        book: OrderbookTracker,
        market_id: str,
        asset_id: str,
        side: OrderSide,
        target_size: float,
        anchor_price: float,
        *,
        latency_guard: LatencyGuard | None = None,
        max_chase_depth_cents: float | None = None,
        chase_interval_ms: int | None = None,
    ):
        strat = settings.strategy
        self.executor = executor
        self.book = book
        self.market_id = market_id
        self.asset_id = asset_id
        self.side = side
        self.target_size = target_size
        self.anchor_price = anchor_price
        self._latency_guard = latency_guard

        self._max_chase_cents = (
            max_chase_depth_cents
            if max_chase_depth_cents is not None
            else strat.max_chase_depth_cents
        )
        # Ensure chase cadence respects API rate limit
        min_interval = 1000.0 / max(strat.api_rate_limit_per_sec, 1)
        configured = chase_interval_ms if chase_interval_ms is not None else strat.chase_interval_ms
        self._interval_s = max(configured, min_interval) / 1000.0

        self._post_only = strat.post_only_enabled

        # State
        self.state = ChaserState.QUOTING
        self.resting_order: Order | None = None
        self.filled_size: float = 0.0
        self.filled_avg_price: float = 0.0
        self._remaining: float = target_size

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(self) -> Order | None:
        """Execute the chase loop until filled, abandoned, or cancelled.

        Returns the filled ``Order`` or ``None`` if abandoned.
        """
        try:
            return await self._chase_loop()
        except asyncio.CancelledError:
            # External cancellation (timeout / shutdown) — cancel resting
            await self._cancel_resting()
            raise

    @property
    def is_terminal(self) -> bool:
        return self.state in (ChaserState.FILLED, ChaserState.ABANDONED)

    # ── Core loop ───────────────────────────────────────────────────────────

    async def _chase_loop(self) -> Order | None:
        """Main requote loop."""
        # Place the initial order
        initial_price = self._optimal_quote()
        if initial_price is None:
            log.warning(
                "chaser_no_bbo",
                side=self.side.value,
                asset=self.asset_id[:16],
            )
            self.state = ChaserState.ABANDONED
            return None

        await self._place(initial_price)

        while self.state not in (ChaserState.FILLED, ChaserState.ABANDONED):
            await asyncio.sleep(self._interval_s)

            # ── Latency gate ──────────────────────────────────────
            if self._latency_guard and self._latency_guard.is_blocked():
                # Pause chasing but keep resting order on the CLOB
                continue

            # ── Check fill ────────────────────────────────────────
            if self.resting_order and self.resting_order.status == OrderStatus.FILLED:
                self._on_filled(self.resting_order)
                break

            # ── Check partial fill ────────────────────────────────
            if self.resting_order and self.resting_order.status == OrderStatus.PARTIALLY_FILLED:
                self._accumulate_partial(self.resting_order)

            # ── Check if BBO moved ────────────────────────────────
            optimal = self._optimal_quote()
            if optimal is None:
                continue  # No book data yet

            if self.resting_order and abs(self.resting_order.price - optimal) < 1e-6:
                continue  # Still at top-of-book — nothing to do

            # ── Check drift ───────────────────────────────────────
            drift_cents = self._drift_cents(optimal)
            if drift_cents > self._max_chase_cents:
                log.info(
                    "chaser_abandoned",
                    side=self.side.value,
                    drift_cents=round(drift_cents, 2),
                    max_cents=self._max_chase_cents,
                    asset=self.asset_id[:16],
                )
                await self._cancel_resting()
                self.state = ChaserState.ABANDONED
                return None

            # ── Cancel-and-replace ────────────────────────────────
            self.state = ChaserState.CANCELLING
            await self._cancel_resting()
            # Yield to let cancel propagate
            await asyncio.sleep(0)

            self.state = ChaserState.REQUOTING
            await self._place(optimal)

        return self.resting_order

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _optimal_quote(self) -> float | None:
        """The price at which to rest the order (top-of-book, maker side)."""
        if not self.book.has_data:
            return None
        snap = self.book.snapshot()
        if self.side == OrderSide.BUY:
            return snap.best_bid if snap.best_bid > 0 else None
        else:
            return snap.best_ask if snap.best_ask > 0 else None

    def _drift_cents(self, current_quote: float) -> float:
        """How far the current BBO has drifted from the anchor (in cents)."""
        if self.side == OrderSide.BUY:
            # For BUY, adverse drift = price going UP (we pay more)
            return max(0.0, (current_quote - self.anchor_price) * 100)
        else:
            # For SELL, adverse drift = price going DOWN (we receive less)
            return max(0.0, (self.anchor_price - current_quote) * 100)

    async def _place(self, price: float) -> None:
        """Place a new resting order at *price*."""
        order = await self.executor.place_limit_order(
            market_id=self.market_id,
            asset_id=self.asset_id,
            side=self.side,
            price=round(price, 2),
            size=round(self._remaining, 2),
            post_only=self._post_only,
        )

        # Handle POST_ONLY rejection
        if order.rejection_reason == "would_cross":
            log.info(
                "chaser_post_only_rejected",
                side=self.side.value,
                price=price,
                asset=self.asset_id[:16],
            )
            # Will be retried on next iteration at a new price
            self.resting_order = None
            self.state = ChaserState.QUOTING
            return

        self.resting_order = order
        self.state = ChaserState.QUOTING
        log.debug(
            "chaser_quoted",
            side=self.side.value,
            price=price,
            remaining=self._remaining,
            order_id=order.order_id,
        )

    async def _cancel_resting(self) -> None:
        """Cancel the current resting order (if any)."""
        if self.resting_order and self.resting_order.status in (
            OrderStatus.LIVE,
            OrderStatus.PARTIALLY_FILLED,
        ):
            # Snapshot partial fill before cancel
            self._accumulate_partial(self.resting_order)
            await self.executor.cancel_order(self.resting_order)
        self.resting_order = None

    def _accumulate_partial(self, order: Order) -> None:
        """Track partial fill progress."""
        newly_filled = order.filled_size - (self.target_size - self._remaining)
        if newly_filled > 0:
            # Weighted-average the fill price
            prev_value = self.filled_avg_price * self.filled_size
            new_value = order.filled_avg_price * newly_filled
            self.filled_size += newly_filled
            if self.filled_size > 0:
                self.filled_avg_price = (prev_value + new_value) / self.filled_size
            self._remaining = max(0.0, self.target_size - self.filled_size)

        if self._remaining <= 0:
            self._on_filled(order)

    def _on_filled(self, order: Order) -> None:
        """Transition to terminal FILLED state."""
        self.filled_size = self.target_size
        self.filled_avg_price = order.filled_avg_price or order.price
        self._remaining = 0.0
        self.state = ChaserState.FILLED
        log.info(
            "chaser_filled",
            side=self.side.value,
            price=self.filled_avg_price,
            size=self.filled_size,
            order_id=order.order_id,
        )

    def force_check_fill(self, asset_id: str, market_price: float) -> bool:
        """Paper-mode helper: check if the resting order should fill.

        Called by the bot's paper-fill checking path.  Returns True if
        the chaser is now in FILLED state.
        """
        if not self.resting_order or self.resting_order.status != OrderStatus.LIVE:
            return False
        if asset_id != self.asset_id:
            return False

        should_fill = False
        if self.side == OrderSide.BUY and market_price <= self.resting_order.price:
            should_fill = True
        elif self.side == OrderSide.SELL and market_price >= self.resting_order.price:
            should_fill = True

        if should_fill:
            self.resting_order.status = OrderStatus.FILLED
            self.resting_order.filled_size = self.resting_order.size
            self.resting_order.filled_avg_price = self.resting_order.price
            self.resting_order.updated_at = time.time()
            self._on_filled(self.resting_order)
            return True

        return False
