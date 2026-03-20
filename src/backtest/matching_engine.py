"""
Pessimistic CLOB Matching Engine — institutional-grade order simulation.

Eliminates the three deadly sins of naïve backtesting:

1. **Look-ahead bias** — orders are invisible until ``submit_time + latency_ms``
2. **Queue-position optimism** — maker orders track ``queue_ahead`` and fill
   only after sufficient historical taker volume drains the queue
3. **Latency blindness** — configurable network latency penalty before
   orders enter the simulated book

The engine mirrors the historical L2 order book state using the production
``L2OrderBook`` class and matches simulated orders against it.

Fee model
─────────
Uses Polymarket's dynamic fee curve:  ``f = f_max × 4 × p × (1 − p)``
- Taker fills pay the fee (deducted from cash).
- Maker fills are fee-free (consistent with ``FeeCache.maker_fee_bps() → 0``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Literal

from sortedcontainers import SortedDict

from src.core.logger import get_logger
from src.trading.executor import OrderSide, OrderStatus
from src.trading.fees import get_fee_rate

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Fill:
    """Record of a single (partial) order fill."""

    order_id: str
    price: float
    size: float
    fee: float
    timestamp: float
    is_maker: bool
    side: OrderSide


@dataclass(slots=True)
class _LiquidityDebt:
    """Consumed volume at a price level, subject to exponential decay."""
    volume: float
    timestamp: float


@dataclass
class SimOrder:
    """A simulated order in the matching engine.

    Attributes
    ----------
    order_id:
        Unique identifier (``"SIM-{n}"``).
    side:
        ``OrderSide.BUY`` or ``OrderSide.SELL``.
    price:
        Limit price (0–1).  For market orders, 1.0 for buys / 0.01 for sells.
    size:
        Original order size in shares.
    remaining:
        Shares still awaiting fill.
    order_type:
        ``"limit"`` or ``"market"``.
    post_only:
        If True and the order would cross, it is immediately rejected.
    submit_time:
        Sim-time the strategy submitted the order.
    active_time:
        Sim-time the order becomes visible to the matching engine
        (``submit_time + latency_ms / 1000``).
    queue_ahead:
        Shares resting ahead of us at our price level (for FIFO simulation).
    fills:
        List of partial/full fills received so far.
    status:
        Current order lifecycle state.
    """

    order_id: str
    side: OrderSide
    price: float
    size: float
    remaining: float
    order_type: Literal["limit", "market"] = "limit"
    post_only: bool = False
    submit_time: float = 0.0
    active_time: float = 0.0
    queue_ahead: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: str = ""

    @property
    def filled_size(self) -> float:
        return self.size - self.remaining

    @property
    def filled_avg_price(self) -> float:
        if not self.fills:
            return 0.0
        total_qty = sum(f.size for f in self.fills)
        if total_qty <= 0:
            return 0.0
        return sum(f.price * f.size for f in self.fills) / total_qty

    @property
    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)


# ═══════════════════════════════════════════════════════════════════════════
#  Matching Engine
# ═══════════════════════════════════════════════════════════════════════════

class MatchingEngine:
    """Pessimistic CLOB simulator with FIFO queue tracking.

    Parameters
    ----------
    latency_ms:
        Network latency penalty in milliseconds.
    fee_max_pct:
        Maximum fee rate as a percentage (e.g. 1.56).
    fee_enabled:
        Whether to charge dynamic fees.
    """

    def __init__(
        self,
        *,
        latency_ms: float = 150.0,
        fee_max_pct: float = 2.00,
        fee_enabled: bool = True,
        impact_recovery_ms: float = 5000.0,
    ) -> None:
        self._latency_s = latency_ms / 1000.0
        self._f_max = fee_max_pct / 100.0
        self._fee_enabled = fee_enabled
        self._impact_tau = impact_recovery_ms / 1000.0  # decay time constant

        # ── Historical book mirror ─────────────────────────────
        # Uses plain SortedDict directly (lighter than full L2OrderBook
        # since we don't need async callbacks/desync machinery).
        self._bids: SortedDict = SortedDict()   # neg_price → size
        self._asks: SortedDict = SortedDict()   # price → size

        # ── Virtual Liquidity Debt (Synthetic Market Impact) ───────────
        # Tracks volume consumed by simulated orders, persists across book
        # updates, and decays exponentially to simulate MM replenishment.
        self._ask_debt: dict[float, list[_LiquidityDebt]] = {}  # price → debts
        self._bid_debt: dict[float, list[_LiquidityDebt]] = {}  # price → debts
        self._last_time: float = 0.0

        # ── Simulated orders ───────────────────────────────────────────
        self._pending: dict[str, SimOrder] = {}   # not yet past latency
        self._active_makers: dict[str, SimOrder] = {}  # resting limit orders
        self._all_orders: dict[str, SimOrder] = {}  # all ever created

        # ── ID generator ───────────────────────────────────────────────
        self._id_counter = itertools.count(1)

        # ── Book state cache ───────────────────────────────────────────
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    #  Book state — read accessors
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def best_bid(self) -> float:
        if not self._bids:
            return 0.0
        return -self._bids.keys()[0]

    @property
    def best_ask(self) -> float:
        if not self._asks:
            return 0.0
        return self._asks.keys()[0]

    @property
    def mid_price(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb <= 0 or ba <= 0:
            return 0.0
        return (bb + ba) / 2.0

    def depth_at_price(self, side: OrderSide, price: float) -> float:
        """Return the resting size at a specific price level."""
        if side == OrderSide.BUY:
            return self._bids.get(-price, 0.0)
        return self._asks.get(price, 0.0)

    def ask_levels(self, n: int = 20) -> list[tuple[float, float]]:
        """Return top *n* ask levels as ``(price, size)`` tuples."""
        return [
            (p, self._asks[p])
            for p in self._asks.islice(stop=n)
        ]

    def bid_levels(self, n: int = 20) -> list[tuple[float, float]]:
        """Return top *n* bid levels as ``(price, size)`` tuples."""
        return [
            (-neg_p, self._bids[neg_p])
            for neg_p in self._bids.islice(stop=n)
        ]

    # ═══════════════════════════════════════════════════════════════════════
    #  Book updates (fed from historical data)
    # ═══════════════════════════════════════════════════════════════════════

    def on_book_update(self, event_data: dict, *, current_time: float | None = None) -> None:
        """Apply an L2 delta or snapshot to the historical book mirror.

        Accepts the same payload format as ``L2OrderBook.on_delta()`` or
        ``L2OrderBook._apply_snapshot()``.

        After applying historical data the virtual liquidity debt is
        subtracted so that subsequent reads see impact-adjusted levels.
        """
        event_type = event_data.get("event_type", "")

        if event_type in ("book", "snapshot", "book_snapshot"):
            self._apply_snapshot(event_data)
        else:
            self._apply_delta(event_data)

        ts = current_time if current_time is not None else self._last_time
        self._apply_liquidity_debt(ts)
        self._refresh_bbo()

    def _apply_snapshot(self, data: dict) -> None:
        """Replace the book from a snapshot payload."""
        self._bids.clear()
        self._asks.clear()

        for b in data.get("bids") or []:
            try:
                price = float(b["price"])
                size = float(b["size"])
                if price > 0 and size > 0:
                    self._bids[-price] = size
            except (KeyError, TypeError, ValueError):
                continue

        for a in data.get("asks") or []:
            try:
                price = float(a["price"])
                size = float(a["size"])
                if price > 0 and size > 0:
                    self._asks[price] = size
            except (KeyError, TypeError, ValueError):
                continue

    def _apply_delta(self, data: dict) -> None:
        """Apply incremental changes from a delta payload."""
        changes = data.get("changes") or data.get("data") or []
        if isinstance(changes, dict):
            changes = [changes]
        if not changes and data.get("price"):
            changes = [data]

        for ch in changes:
            try:
                price = float(ch.get("price", 0))
                size = float(ch.get("size", 0))
                side = str(ch.get("side", "")).upper()
            except (TypeError, ValueError):
                continue

            if price <= 0:
                continue

            if side in ("BUY", "BID"):
                if size <= 0:
                    self._bids.pop(-price, None)
                else:
                    self._bids[-price] = size
            elif side in ("SELL", "ASK"):
                if size <= 0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = size

    def _refresh_bbo(self) -> None:
        self._best_bid = self.best_bid
        self._best_ask = self.best_ask

    # ═══════════════════════════════════════════════════════════════════════
    #  Order submission
    # ═══════════════════════════════════════════════════════════════════════

    def submit_order(
        self,
        side: OrderSide,
        price: float,
        size: float,
        *,
        order_type: Literal["limit", "market"] = "limit",
        post_only: bool = False,
        current_time: float = 0.0,
    ) -> SimOrder:
        """Submit an order to the simulated exchange.

        The order will not be matchable until ``current_time + latency_ms``
        (simulating network + exchange processing delay).

        Returns the ``SimOrder`` object (status = PENDING).
        """
        oid = f"SIM-{next(self._id_counter)}"
        order = SimOrder(
            order_id=oid,
            side=side,
            price=price,
            size=size,
            remaining=size,
            order_type=order_type,
            post_only=post_only,
            submit_time=current_time,
            active_time=current_time + self._latency_s,
        )

        self._pending[oid] = order
        self._all_orders[oid] = order

        log.debug(
            "sim_order_submitted",
            order_id=oid,
            side=side.value,
            price=price,
            size=size,
            type=order_type,
            active_at=order.active_time,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if successfully cancelled."""
        order = self._all_orders.get(order_id)
        if order is None:
            return False

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False

        order.status = OrderStatus.CANCELLED

        # Remove from whichever collection it's in
        self._pending.pop(order_id, None)
        self._active_makers.pop(order_id, None)

        log.debug("sim_order_cancelled", order_id=order_id)
        return True

    # ═══════════════════════════════════════════════════════════════════════
    #  Activation (latency expiry)
    # ═══════════════════════════════════════════════════════════════════════

    def activate_pending_orders(self, current_time: float) -> list[Fill]:
        """Move orders past their latency window into the active set.

        For taker/crossing orders, execute immediately against the book.
        For maker (non-crossing) limit orders, set queue_ahead and rest.

        Returns any fills generated from immediate taker execution.
        """
        fills: list[Fill] = []
        activated = []

        for oid, order in list(self._pending.items()):
            if current_time < order.active_time:
                continue
            if order.status == OrderStatus.CANCELLED:
                self._pending.pop(oid, None)
                continue

            activated.append(oid)

        for oid in activated:
            order = self._pending.pop(oid)
            order_fills = self._activate_order(order, current_time)
            fills.extend(order_fills)

        return fills

    def _activate_order(self, order: SimOrder, current_time: float) -> list[Fill]:
        """Activate a single order after its latency window expires."""
        # Determine if this is a crossing (taker) order
        is_crossing = self._is_crossing(order)

        if order.order_type == "market":
            # Market orders always cross
            return self._execute_taker(order, current_time)

        if is_crossing:
            if order.post_only:
                # POST_ONLY would cross → reject immediately
                order.status = OrderStatus.CANCELLED
                order.rejection_reason = "would_cross"
                log.debug(
                    "sim_post_only_rejected",
                    order_id=order.order_id,
                    price=order.price,
                    side=order.side.value,
                )
                return []
            # Aggressive limit order that crosses the spread
            return self._execute_taker(order, current_time)

        # Non-crossing limit order → rest as maker
        self._rest_as_maker(order)
        return []

    def _is_crossing(self, order: SimOrder) -> bool:
        """Check if a limit order would cross the spread."""
        if order.side == OrderSide.BUY:
            ba = self.best_ask
            return ba > 0 and order.price >= ba
        else:
            bb = self.best_bid
            return bb > 0 and order.price <= bb

    # ═══════════════════════════════════════════════════════════════════════
    #  Taker execution — walk the book (slippage simulation)
    # ═══════════════════════════════════════════════════════════════════════

    def _execute_taker(self, order: SimOrder, current_time: float) -> list[Fill]:
        """Walk the opposite side of the book to fill a taker order.

        Computes realistic VWAP slippage by consuming liquidity level-by-level.
        Virtual liquidity debt is subtracted from available size and new
        debt is recorded for each fill.
        """
        fills: list[Fill] = []
        self._last_time = current_time

        if order.side == OrderSide.BUY:
            # Walk the asks (ascending)
            levels_to_remove: list[float] = []

            for ask_price in list(self._asks.keys()):
                if order.remaining <= 0:
                    break
                if order.order_type == "limit" and ask_price > order.price:
                    break  # can't fill above our limit

                raw = self._asks[ask_price]
                avail = max(0.0, raw - self._decayed_debt(self._ask_debt.get(ask_price), current_time))
                if avail < 1e-9:
                    continue  # level fully consumed by prior impact
                fill_qty = min(order.remaining, avail)

                fee = self._compute_fee(ask_price, fill_qty)
                fill = Fill(
                    order_id=order.order_id,
                    price=ask_price,
                    size=fill_qty,
                    fee=fee,
                    timestamp=current_time,
                    is_maker=False,
                    side=order.side,
                )
                fills.append(fill)
                order.fills.append(fill)
                order.remaining -= fill_qty

                # Record synthetic market impact debt
                self._record_debt(self._ask_debt, ask_price, fill_qty, current_time)

                # Reduce book liquidity (for within-tick consistency)
                remaining_at_level = raw - fill_qty
                if remaining_at_level <= 1e-9:
                    levels_to_remove.append(ask_price)
                else:
                    self._asks[ask_price] = remaining_at_level

            for p in levels_to_remove:
                self._asks.pop(p, None)

        else:
            # SELL — walk the bids (descending = ascending neg keys)
            levels_to_remove: list[float] = []

            for neg_price in list(self._bids.keys()):
                if order.remaining <= 0:
                    break
                bid_price = -neg_price
                if order.order_type == "limit" and bid_price < order.price:
                    break  # can't fill below our limit

                raw = self._bids[neg_price]
                avail = max(0.0, raw - self._decayed_debt(self._bid_debt.get(bid_price), current_time))
                if avail < 1e-9:
                    continue  # level fully consumed by prior impact
                fill_qty = min(order.remaining, avail)

                fee = self._compute_fee(bid_price, fill_qty)
                fill = Fill(
                    order_id=order.order_id,
                    price=bid_price,
                    size=fill_qty,
                    fee=fee,
                    timestamp=current_time,
                    is_maker=False,
                    side=order.side,
                )
                fills.append(fill)
                order.fills.append(fill)
                order.remaining -= fill_qty

                # Record synthetic market impact debt
                self._record_debt(self._bid_debt, bid_price, fill_qty, current_time)

                remaining_at_level = raw - fill_qty
                if remaining_at_level <= 1e-9:
                    levels_to_remove.append(neg_price)
                else:
                    self._bids[neg_price] = remaining_at_level

            for np in levels_to_remove:
                self._bids.pop(np, None)

        # Update order status
        if order.remaining <= 1e-9:
            order.remaining = 0.0
            order.status = OrderStatus.FILLED
        elif order.fills:
            order.status = OrderStatus.PARTIALLY_FILLED
            # Remaining portion rests as maker
            self._rest_as_maker(order)
        else:
            # No fills at all (empty book or price too far)
            order.status = OrderStatus.LIVE
            self._rest_as_maker(order)

        self._refresh_bbo()
        return fills

    # ═══════════════════════════════════════════════════════════════════════
    #  Maker resting — FIFO queue tracking
    # ═══════════════════════════════════════════════════════════════════════

    def _rest_as_maker(self, order: SimOrder) -> None:
        """Place a non-crossing limit order at the back of the queue."""
        # Queue ahead = current depth at this price level
        if order.side == OrderSide.BUY:
            order.queue_ahead = self._bids.get(-order.price, 0.0)
        else:
            order.queue_ahead = self._asks.get(order.price, 0.0)

        order.status = OrderStatus.LIVE
        self._active_makers[order.order_id] = order

        log.debug(
            "sim_order_resting",
            order_id=order.order_id,
            side=order.side.value,
            price=order.price,
            queue_ahead=order.queue_ahead,
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Trade processing — drain maker queues
    # ═══════════════════════════════════════════════════════════════════════

    def on_trade(
        self,
        trade_price: float,
        trade_size: float,
        trade_side: str,
        current_time: float,
    ) -> list[Fill]:
        """Process a historical public trade against resting maker orders.

        FIFO queue logic: if a trade occurs at our price level on the
        correct aggressor side, decrement ``queue_ahead`` by the trade size.
        Our order fills only when ``queue_ahead`` is fully consumed.

        Parameters
        ----------
        trade_price:
            Price the historical trade executed at.
        trade_size:
            Number of shares in the historical trade.
        trade_side:
            Aggressor side: ``"buy"`` (lifts asks) or ``"sell"`` (hits bids).
        current_time:
            Simulated timestamp of the trade.

        Returns
        -------
        list[Fill]
            Any fills on our resting maker orders.
        """
        fills: list[Fill] = []
        completed: list[str] = []

        for oid, order in self._active_makers.items():
            if order.status != OrderStatus.LIVE:
                completed.append(oid)
                continue

            fill = self._try_drain_queue(
                order, trade_price, trade_size, trade_side, current_time
            )
            if fill is not None:
                fills.append(fill)
                if order.remaining <= 1e-9:
                    order.remaining = 0.0
                    order.status = OrderStatus.FILLED
                    completed.append(oid)
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED

        for oid in completed:
            self._active_makers.pop(oid, None)

        return fills

    def _try_drain_queue(
        self,
        order: SimOrder,
        trade_price: float,
        trade_size: float,
        trade_side: str,
        current_time: float,
    ) -> Fill | None:
        """Attempt to fill a maker order from a historical trade.

        A BUY maker order is filled when a SELL aggressor hits the bid
        at or below our price.  A SELL maker order is filled when a BUY
        aggressor lifts the ask at or above our price.
        """
        # Check side compatibility:
        #   Our BUY resting order is filled by an incoming SELL (someone market-sells into bids)
        #   Our SELL resting order is filled by an incoming BUY (someone market-buys from asks)
        if order.side == OrderSide.BUY:
            if trade_side.lower() != "sell":
                return None
            # Trade must be at or below our bid price (price improvement or match)
            if trade_price > order.price + 1e-9:
                return None
        else:  # SELL
            if trade_side.lower() != "buy":
                return None
            # Trade must be at or above our ask price
            if trade_price < order.price - 1e-9:
                return None

        # Drain the queue
        if order.queue_ahead >= trade_size:
            order.queue_ahead -= trade_size
            return None

        # We can fill some shares
        fillable_volume = trade_size - order.queue_ahead
        order.queue_ahead = 0.0

        fill_qty = min(order.remaining, fillable_volume)
        if fill_qty <= 1e-9:
            return None

        # Maker fills are fee-free on Polymarket
        fill = Fill(
            order_id=order.order_id,
            price=order.price,
            size=fill_qty,
            fee=0.0,
            timestamp=current_time,
            is_maker=True,
            side=order.side,
        )
        order.fills.append(fill)
        order.remaining -= fill_qty

        log.debug(
            "sim_maker_fill",
            order_id=order.order_id,
            price=order.price,
            fill_qty=fill_qty,
            remaining=order.remaining,
        )
        return fill

    # ═══════════════════════════════════════════════════════════════════════
    #  Price movement — invalidate maker orders
    # ═══════════════════════════════════════════════════════════════════════

    def check_maker_viability(self) -> list[str]:
        """Check if resting maker orders are still at viable price levels.

        Returns a list of order IDs that are no longer at the touch
        (price has moved away). These orders remain resting but their
        ``queue_ahead`` is recalculated from the current book depth if
        the level still exists, or the order becomes unfillable if the
        level has vanished.
        """
        stale: list[str] = []
        for oid, order in self._active_makers.items():
            if order.side == OrderSide.BUY:
                current_depth = self._bids.get(-order.price, 0.0)
            else:
                current_depth = self._asks.get(order.price, 0.0)

            if current_depth <= 0:
                # Our price level vanished — we're still resting but
                # there's nothing to queue behind. Reset queue to 0
                # so that the next trade at this level fills us first.
                if order.queue_ahead > 0:
                    order.queue_ahead = 0.0
                    stale.append(oid)

        return stale

    def simulate_order_fill(
        self,
        order_id: str,
        size: float,
        *,
        price: float | None = None,
        current_time: float,
        is_maker: bool = True,
    ) -> Fill | None:
        """Force a simulated fill on an existing order.

        Used by replay adapters when the historical book shows our resting
        quote has been crossed even though no explicit trade print exists.
        """
        order = self._all_orders.get(order_id)
        if order is None:
            return None
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return None
        if order.remaining <= 1e-9 or current_time < order.active_time:
            return None

        fill_qty = min(size, order.remaining)
        if fill_qty <= 1e-9:
            return None

        fill_price = order.price if price is None else price
        fee = 0.0 if is_maker else self._compute_fee(fill_price, fill_qty)
        fill = Fill(
            order_id=order.order_id,
            price=fill_price,
            size=fill_qty,
            fee=fee,
            timestamp=current_time,
            is_maker=is_maker,
            side=order.side,
        )
        order.fills.append(fill)
        order.remaining -= fill_qty

        if order.remaining <= 1e-9:
            order.remaining = 0.0
            order.status = OrderStatus.FILLED
            self._pending.pop(order_id, None)
            self._active_makers.pop(order_id, None)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        log.debug(
            "sim_order_force_filled",
            order_id=order_id,
            price=fill_price,
            fill_qty=fill_qty,
            remaining=order.remaining,
            is_maker=is_maker,
        )
        return fill

    # ═══════════════════════════════════════════════════════════════════════
    #  Fee computation
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_fee(self, fill_price: float, fill_size: float) -> float:
        """Compute the fee for a taker fill.

        Formula:  fee = fill_size × fill_price × f_max × 4 × p × (1 − p)
        """
        rate = get_fee_rate(
            fill_price, fee_enabled=self._fee_enabled, f_max=self._f_max
        )
        return fill_size * fill_price * rate
    # ═════════════════════════════════════════════════════════════════════
    #  Virtual Liquidity Debt (Synthetic Market Impact)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _record_debt(
        debt_map: dict[float, list[_LiquidityDebt]],
        price: float,
        volume: float,
        timestamp: float,
    ) -> None:
        """Record consumed volume as liquidity debt at *price*."""
        if volume <= 0:
            return
        if price not in debt_map:
            debt_map[price] = []
        debt_map[price].append(_LiquidityDebt(volume=volume, timestamp=timestamp))

    def _decayed_debt(self, debts: list[_LiquidityDebt] | None, now: float) -> float:
        """Sum decayed debt entries at a single price level."""
        if not debts:
            return 0.0
        tau = self._impact_tau
        if tau <= 0:
            return 0.0
        total = 0.0
        for d in debts:
            elapsed = now - d.timestamp
            if elapsed < 0:
                elapsed = 0.0
            total += d.volume * math.exp(-elapsed / tau)
        return total

    def _apply_liquidity_debt(self, current_time: float) -> None:
        """Subtract decayed virtual debt from the book after a historical update.

        Called inside ``on_book_update`` so that the book levels seen by
        subsequent order activations reflect our simulated market impact.
        Expired debt entries are pruned.
        """
        tau = self._impact_tau
        # Ask debt
        for price in list(self._ask_debt):
            debt = self._decayed_debt(self._ask_debt[price], current_time)
            if debt < 1e-9:
                del self._ask_debt[price]
                continue
            if price in self._asks:
                adjusted = self._asks[price] - debt
                if adjusted < 1e-9:
                    del self._asks[price]
                else:
                    self._asks[price] = adjusted
            # Prune entries decayed below 1% of original
            self._ask_debt[price] = [
                d for d in self._ask_debt[price]
                if d.volume * math.exp(-(current_time - d.timestamp) / tau) > d.volume * 0.01
            ] if tau > 0 else []
            if not self._ask_debt[price]:
                del self._ask_debt[price]

        # Bid debt
        for price in list(self._bid_debt):
            debt = self._decayed_debt(self._bid_debt[price], current_time)
            if debt < 1e-9:
                del self._bid_debt[price]
                continue
            neg_price = -price
            if neg_price in self._bids:
                adjusted = self._bids[neg_price] - debt
                if adjusted < 1e-9:
                    del self._bids[neg_price]
                else:
                    self._bids[neg_price] = adjusted
            # Prune entries decayed below 1% of original
            self._bid_debt[price] = [
                d for d in self._bid_debt[price]
                if d.volume * math.exp(-(current_time - d.timestamp) / tau) > d.volume * 0.01
            ] if tau > 0 else []
            if not self._bid_debt[price]:
                del self._bid_debt[price]
    # ═══════════════════════════════════════════════════════════════════════
    #  Accessors
    # ═══════════════════════════════════════════════════════════════════════

    def get_order(self, order_id: str) -> SimOrder | None:
        return self._all_orders.get(order_id)

    def get_open_orders(self) -> list[SimOrder]:
        """Return all non-terminal orders (pending + active makers)."""
        result: list[SimOrder] = []
        for o in self._pending.values():
            if o.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED):
                result.append(o)
        for o in self._active_makers.values():
            if o.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED):
                result.append(o)
        return result

    def get_all_orders(self) -> list[SimOrder]:
        return list(self._all_orders.values())

    def get_all_fills(self) -> list[Fill]:
        fills: list[Fill] = []
        for o in self._all_orders.values():
            fills.extend(o.fills)
        return sorted(fills, key=lambda f: f.timestamp)

    def reset(self) -> None:
        """Wipe all state (book + orders + liquidity debt)."""
        self._bids.clear()
        self._asks.clear()
        self._pending.clear()
        self._active_makers.clear()
        self._all_orders.clear()
        self._ask_debt.clear()
        self._bid_debt.clear()
        self._best_bid = 0.0
        self._best_ask = 0.0
        self._id_counter = itertools.count(1)
