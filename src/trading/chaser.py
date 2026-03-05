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
  - **Escalation**: After *N* consecutive POST_ONLY rejections, the
    chaser may cross the spread with a marketable limit order (subject
    to an alpha-sufficiency pre-check).
  - **Fast-kill event**: An ``asyncio.Event`` shared with the adverse-
    selection guard — when cleared, all placement is paused.
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
    from src.signals.iceberg_detector import IcebergDetector
    from src.trading.adverse_selection_monitor import AdverseSelectionMonitor

log = get_logger(__name__)


class ChaserState(str, Enum):
    QUOTING = "QUOTING"
    CANCELLING = "CANCELLING"
    REQUOTING = "REQUOTING"
    ESCALATING = "ESCALATING"
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
    tp_target_price:
        Expected take-profit exit price — used by the escalation alpha-
        sufficiency check.  Required for BUY-side escalation.
    fee_rate_bps:
        Taker fee for the token — used when escalating to a marketable
        limit order and for the alpha check.
    fast_kill_event:
        Shared ``asyncio.Event`` — when *cleared*, all order placement
        is paused until the event is set again.
    max_post_only_rejections:
        Number of consecutive POST_ONLY rejections before escalating.
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
        tp_target_price: float | None = None,
        fee_rate_bps: int = 0,
        fast_kill_event: asyncio.Event | None = None,
        max_post_only_rejections: int | None = None,
        urgent: bool = False,
        iceberg_detector: IcebergDetector | None = None,
        adverse_monitor: AdverseSelectionMonitor | None = None,
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
        self._urgent = urgent

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

        # Escalation parameters (Pillar 7)
        self._tp_target = tp_target_price
        self._fee_bps = fee_rate_bps
        self._fast_kill = fast_kill_event
        # Urgent chasers escalate immediately on first POST_ONLY
        # rejection, capturing the edge before it decays.
        if urgent:
            self._max_rejections = 0
        else:
            self._max_rejections = (
                max_post_only_rejections
                if max_post_only_rejections is not None
                else strat.chaser_max_rejections
            )
        self._escalation_ticks = strat.chaser_escalation_ticks
        self._desired_margin = strat.desired_margin_cents
        self._rejection_count: int = 0

        # SI-2: Iceberg-aware routing
        self._iceberg = iceberg_detector
        self._iceberg_peg_min_conf = getattr(
            strat, "iceberg_peg_min_confidence", 0.50
        )

        # Anti-toxicity guard: cancel chase instead of crossing when
        # adverse-selection p-value is dropping.
        self._adverse_monitor = adverse_monitor
        self._toxicity_market_id = market_id  # condition_id for stats lookup
        self._toxicity_ceil = getattr(
            strat, "chaser_toxicity_p_value_ceil", 0.10
        )
        # Snapshot the initial p-value to detect rapid deterioration.
        self._initial_p_value: float = 1.0
        if self._adverse_monitor is not None:
            _stats = self._adverse_monitor.get_stats(market_id)
            if _stats is not None:
                self._initial_p_value = _stats.last_p_value

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
        # Wait for fast-kill clearance before initial placement
        await self._wait_fast_kill()

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

            # ── Fast-kill gate ────────────────────────────────────
            if self._fast_kill and not self._fast_kill.is_set():
                # Adverse-selection guard has fired — pause all placement
                # but keep resting order (it may already be cancelled by
                # the guard's cancel_all)
                continue

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
            # ── Toxicity guard (mid-chase) ───────────────────
            if self._should_abort_toxicity():
                await self._cancel_resting()
                self.state = ChaserState.ABANDONED
                return None
            self.state = ChaserState.CANCELLING
            await self._cancel_resting()
            # Yield to let cancel propagate
            await asyncio.sleep(0)

            # Wait for fast-kill clearance before re-quoting
            await self._wait_fast_kill()

            self.state = ChaserState.REQUOTING
            await self._place(optimal)

        return self.resting_order

    # ── Helpers ─────────────────────────────────────────────────────────────

    async def _wait_fast_kill(self, timeout: float = 0.1) -> None:
        """Block until the fast-kill event is set (chasers may proceed)."""
        if self._fast_kill is None:
            return
        try:
            await asyncio.wait_for(self._fast_kill.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Will be retried on next loop iteration
            pass

    def _optimal_quote(self) -> float | None:
        """The price at which to rest the order (top-of-book, maker side).

        When an iceberg detector is available and detects a whale on
        **our side** of the book, pegs the order at the iceberg price
        to hide behind the institutional liquidity wall.
        """
        if not self.book.has_data:
            return None
        snap = self.book.snapshot()
        if self.side == OrderSide.BUY:
            bbo = snap.best_bid if snap.best_bid > 0 else None
        else:
            bbo = snap.best_ask if snap.best_ask > 0 else None

        if bbo is None:
            return None

        # ── SI-2: Iceberg peg ───────────────────────────────────
        if self._iceberg is not None:
            our_side = "BUY" if self.side == OrderSide.BUY else "SELL"
            iceberg = self._iceberg.strongest_iceberg(our_side)
            if (
                iceberg is not None
                and iceberg.confidence >= self._iceberg_peg_min_conf
            ):
                # Peg at iceberg price — sit behind whale wall
                peg = iceberg.price
                # Clamp so we never give up queue priority beyond BBO
                if self.side == OrderSide.BUY:
                    peg = min(peg, bbo)  # don't bid above best bid
                else:
                    peg = max(peg, bbo)  # don't ask below best ask

                log.debug(
                    "chaser_iceberg_peg",
                    side=self.side.value,
                    iceberg_price=iceberg.price,
                    iceberg_confidence=iceberg.confidence,
                    iceberg_refills=iceberg.refill_count,
                    bbo=bbo,
                    peg=peg,
                    asset=self.asset_id[:16],
                )
                return peg

        return bbo

    def _drift_cents(self, current_quote: float) -> float:
        """How far the current BBO has drifted from the anchor (in cents)."""
        if self.side == OrderSide.BUY:
            # For BUY, adverse drift = price going UP (we pay more)
            return max(0.0, (current_quote - self.anchor_price) * 100)
        else:
            # For SELL, adverse drift = price going DOWN (we receive less)
            return max(0.0, (self.anchor_price - current_quote) * 100)

    def _should_abort_toxicity(self) -> bool:
        """Check if the adverse-selection monitor signals rising toxicity.

        Returns ``True`` (abort the chase) when **both**:
          1. The market's current p-value is below ``chaser_toxicity_p_value_ceil``.
          2. The p-value has dropped by ≥ 50% since the chaser started.

        This prevents us from aggressively crossing the spread into a
        market that is becoming increasingly toxic — even if the formal
        suspension threshold has not yet been breached.
        """
        if self._adverse_monitor is None:
            return False
        stats = self._adverse_monitor.get_stats(self._toxicity_market_id)
        if stats is None:
            return False
        p = stats.last_p_value
        if p >= self._toxicity_ceil:
            return False
        # Also require a meaningful drop from the snapshot taken at init
        if self._initial_p_value > 0 and p < self._initial_p_value * 0.50:
            log.info(
                "chaser_toxicity_abort",
                side=self.side.value,
                asset=self.asset_id[:16],
                p_value=round(p, 6),
                initial_p=round(self._initial_p_value, 6),
                ceil=self._toxicity_ceil,
            )
            return True
        return False

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
            self._rejection_count += 1
            log.info(
                "chaser_post_only_rejected",
                side=self.side.value,
                price=price,
                asset=self.asset_id[:16],
                rejection_count=self._rejection_count,
                max_rejections=self._max_rejections,
            )

            # ── Escalation check (Pillar 7) ───────────────────────
            if self._rejection_count >= self._max_rejections:
                await self._escalate()
                return

            # Will be retried on next iteration at a new price
            self.resting_order = None
            self.state = ChaserState.QUOTING
            return

        # Successful placement — reset rejection counter
        self._rejection_count = 0
        self.resting_order = order
        self.state = ChaserState.QUOTING
        log.debug(
            "chaser_quoted",
            side=self.side.value,
            price=price,
            remaining=self._remaining,
            order_id=order.order_id,
        )

    async def _escalate(self) -> None:
        """Transition to ESCALATING: cross the spread with a marketable
        limit order after N consecutive POST_ONLY rejections.

        Before crossing, runs an alpha-sufficiency check to ensure the
        full round-trip cost (including taker fee) still beats the
        minimum profit margin.

        Also checks the adverse-selection monitor: if the market's
        toxicity p-value is below the ceiling, the chaser abandons
        rather than aggressively crossing the spread into toxic flow.
        """
        self.state = ChaserState.ESCALATING

        # ── Toxicity guard ─────────────────────────────────────
        if self._should_abort_toxicity():
            self.state = ChaserState.ABANDONED
            return

        snap = self.book.snapshot()
        tick = 0.01 * self._escalation_ticks

        if self.side == OrderSide.BUY:
            cross_price = (snap.best_ask + tick) if snap.best_ask > 0 else None
        else:
            cross_price = (snap.best_bid - tick) if snap.best_bid > 0 else None

        if cross_price is None or cross_price <= 0:
            log.warning("escalation_no_bbo", side=self.side.value)
            self.state = ChaserState.ABANDONED
            return

        # ── Alpha-sufficiency check ───────────────────────────────
        if not self._alpha_check(cross_price):
            log.info(
                "escalation_alpha_fail",
                side=self.side.value,
                cross_price=round(cross_price, 4),
                tp_target=self._tp_target,
                fee_bps=self._fee_bps,
            )
            self.state = ChaserState.ABANDONED
            return

        # Place marketable limit (post_only=False, include fee)
        log.info(
            "chaser_escalating",
            side=self.side.value,
            cross_price=round(cross_price, 4),
            fee_bps=self._fee_bps,
        )
        order = await self.executor.place_limit_order(
            market_id=self.market_id,
            asset_id=self.asset_id,
            side=self.side,
            price=round(cross_price, 2),
            size=round(self._remaining, 2),
            post_only=False,
            fee_rate_bps=self._fee_bps,
        )

        self._rejection_count = 0

        if order.status in (OrderStatus.FILLED,):
            self._on_filled(order)
        elif order.status in (OrderStatus.CANCELLED,):
            log.warning("escalation_order_failed", reason=order.rejection_reason)
            self.state = ChaserState.ABANDONED
        else:
            self.resting_order = order
            self.state = ChaserState.QUOTING

    def _alpha_check(self, cross_price: float) -> bool:
        """Verify the trade still has positive expected alpha after fees.

        For BUY entries:
            net_spread = (tp_target - cross_price) * 100
                         - entry_fee_cents - exit_fee_cents

        For SELL exits:
            net_spread = (cross_price - entry_anchor) * 100
                         - taker_fee_cents

        Must exceed ``desired_margin_cents``.
        """
        if self.side == OrderSide.BUY:
            if self._tp_target is None:
                # No TP target supplied — allow escalation (caller's risk)
                return True
            gross_spread = (self._tp_target - cross_price) * 100
            entry_fee = cross_price * self._fee_bps / 10_000 * 100
            # Assume maker exit (0 fee) as the optimistic case
            exit_fee = 0.0
            net = gross_spread - entry_fee - exit_fee
        else:
            # SELL side: anchor is the entry price we're trying to sell above
            gross_spread = (cross_price - self.anchor_price) * 100
            taker_fee = cross_price * self._fee_bps / 10_000 * 100
            net = gross_spread - taker_fee

        return net >= self._desired_margin

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
