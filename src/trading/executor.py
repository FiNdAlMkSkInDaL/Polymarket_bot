"""
Order executor — wraps the Polymarket CLOB client to place, cancel, and
monitor orders.  In paper mode, all calls are intercepted and simulated.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    LIVE = "LIVE"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Internal order tracking object."""

    order_id: str
    market_id: str
    asset_id: str
    side: OrderSide
    price: float
    size: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    filled_avg_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    clob_order_id: str = ""
    post_only: bool = False
    rejection_reason: str = ""          # "would_cross" if POST_ONLY rejected


class OrderExecutor:
    """Manages order lifecycle against the Polymarket CLOB.

    In **paper mode**, orders are tracked locally and fills are simulated
    when the live market price crosses the limit price.
    """

    def __init__(self, paper_mode: bool | None = None):
        self.paper_mode = paper_mode if paper_mode is not None else settings.paper_mode
        self._orders: dict[str, Order] = {}
        self._next_id = 1
        self._clob_client: Any = None  # Lazy init

    # ── CLOB client ─────────────────────────────────────────────────────────
    def _get_clob_client(self) -> Any:
        """Lazily initialise the py-clob-client."""
        if self._clob_client is not None:
            return self._clob_client

        if self.paper_mode:
            return None

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=settings.polymarket_api_key,
                api_secret=settings.polymarket_secret,
                api_passphrase=settings.polymarket_passphrase,
            )
            self._clob_client = ClobClient(
                settings.clob_http_url,
                key=settings.eoa_private_key,
                chain_id=137,  # Polygon mainnet
                creds=creds,
            )
            log.info("clob_client_initialised")
        except Exception as exc:
            log.error("clob_client_init_failed", error=str(exc))
            raise

        return self._clob_client

    # ── Order placement ─────────────────────────────────────────────────────
    async def place_limit_order(
        self,
        market_id: str,
        asset_id: str,
        side: OrderSide,
        price: float,
        size: float,
        *,
        post_only: bool = False,
        fee_rate_bps: int = 0,
    ) -> Order:
        """Place a GTC limit order.

        Parameters
        ----------
        post_only:
            If ``True``, the order will be rejected by the CLOB if it
            would immediately cross the spread (i.e., take liquidity).
            The returned ``Order`` will have ``status=CANCELLED`` and
            ``rejection_reason="would_cross"``.

        In paper mode the order is recorded locally and will be filled
        by :meth:`check_paper_fill` when the market price crosses.
        """
        order_id = f"{'PAPER' if self.paper_mode else 'LIVE'}-{self._next_id}"
        self._next_id += 1

        order = Order(
            order_id=order_id,
            market_id=market_id,
            asset_id=asset_id,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.LIVE,
            post_only=post_only,
        )

        if not self.paper_mode:
            clob = self._get_clob_client()
            try:
                from py_clob_client.clob_types import OrderArgs
                from py_clob_client.order_builder.constants import BUY, SELL

                from py_clob_client.clob_types import OrderType

                order_args = OrderArgs(
                    price=price,
                    size=size,
                    side=BUY if side == OrderSide.BUY else SELL,
                    token_id=asset_id,
                )
                # Attach fee rate to the signed payload when taking liquidity
                if fee_rate_bps > 0:
                    order_args.fee_rate_bps = fee_rate_bps

                # Two-step create+post so post_only reaches the wire.
                # ClobClient.create_and_post_order doesn't forward
                # post_only; ClobClient.post_order does.
                signed = await asyncio.to_thread(clob.create_order, order_args)
                resp = await asyncio.to_thread(
                    clob.post_order, signed, OrderType.GTC, post_only,
                )

                clob_id = ""
                if isinstance(resp, dict):
                    # Detect POST_ONLY rejection
                    if post_only and resp.get("status") in ("rejected", "REJECTED"):
                        order.status = OrderStatus.CANCELLED
                        order.rejection_reason = "would_cross"
                        log.info(
                            "post_only_rejected",
                            order_id=order.order_id,
                            side=side.value,
                            price=price,
                        )
                        self._orders[order.order_id] = order
                        return order
                    clob_id = resp.get("orderID", "") or resp.get("id", "")
                elif hasattr(resp, "orderID"):
                    clob_id = resp.orderID
                order.clob_order_id = str(clob_id)
                log.info(
                    "order_placed_live",
                    order_id=order.order_id,
                    clob_id=order.clob_order_id,
                    side=side.value,
                    price=price,
                    size=size,
                    post_only=post_only,
                )
            except Exception as exc:
                order.status = OrderStatus.CANCELLED
                log.error("order_place_failed", error=str(exc))
                self._orders[order.order_id] = order
                return order
        else:
            log.info(
                "order_placed_paper",
                order_id=order.order_id,
                side=side.value,
                price=price,
                size=size,
                asset=asset_id[:16],
            )

        self._orders[order.order_id] = order
        return order

    # ── Order cancellation ──────────────────────────────────────────────────
    async def cancel_order(self, order: Order) -> None:
        """Cancel a live order."""
        if order.status not in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED):
            return

        if not self.paper_mode and order.clob_order_id:
            try:
                clob = self._get_clob_client()
                await asyncio.to_thread(clob.cancel, order.clob_order_id)
                log.info("order_cancelled_live", clob_id=order.clob_order_id)
            except Exception as exc:
                log.warning("order_cancel_failed", error=str(exc))

        order.status = OrderStatus.CANCELLED
        order.updated_at = time.time()
        log.info("order_cancelled", order_id=order.order_id)

    # ── Cancel all ──────────────────────────────────────────────────────────
    async def cancel_all(self) -> int:
        """Cancel every open order.  Returns count cancelled."""
        count = 0
        for order in list(self._orders.values()):
            if order.status in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED):
                await self.cancel_order(order)
                count += 1
        return count

    # ── Paper-mode fill simulation ──────────────────────────────────────────
    def check_paper_fill(self, asset_id: str, market_price: float) -> list[Order]:
        """Simulate fills for paper orders when the market crosses.

        Applies configurable adverse slippage (``paper_slippage_cents``)
        to avoid overstating paper-mode performance.  Buy orders fill at
        ``limit + slippage`` and sell orders at ``limit - slippage``,
        clamped to [0.01, 0.99].
        """
        if not self.paper_mode:
            return []

        slippage = settings.strategy.paper_slippage_cents / 100.0  # cents → dollars

        filled: list[Order] = []
        for order in self._orders.values():
            if order.asset_id != asset_id:
                continue
            if order.status != OrderStatus.LIVE:
                continue

            should_fill = False
            if order.side == OrderSide.BUY and market_price <= order.price:
                should_fill = True
            elif order.side == OrderSide.SELL and market_price >= order.price:
                should_fill = True

            if should_fill:
                # Apply adverse slippage
                if order.side == OrderSide.BUY:
                    fill_price = min(0.99, order.price + slippage)
                else:
                    fill_price = max(0.01, order.price - slippage)

                order.status = OrderStatus.FILLED
                order.filled_size = order.size
                order.filled_avg_price = fill_price
                order.updated_at = time.time()
                log.info(
                    "paper_fill",
                    order_id=order.order_id,
                    side=order.side.value,
                    limit_price=order.price,
                    fill_price=round(fill_price, 4),
                    slippage_cents=round(slippage * 100, 1),
                    size=order.size,
                )
                filled.append(order)

        return filled

    # ── Query helpers ───────────────────────────────────────────────────────
    @property
    def open_order_count(self) -> int:
        """Number of currently resting (live or partially filled) orders."""
        return sum(
            1 for o in self._orders.values()
            if o.status in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED)
        )

    def get_open_orders(self, market_id: str | None = None) -> list[Order]:
        return [
            o
            for o in self._orders.values()
            if o.status in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED)
            and (market_id is None or o.market_id == market_id)
        ]

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def cleanup_old_orders(self, max_age_seconds: float = 3600) -> int:
        """Remove filled/cancelled orders older than *max_age_seconds*."""
        cutoff = time.time() - max_age_seconds
        to_remove = [
            oid for oid, o in self._orders.items()
            if o.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED)
            and o.updated_at < cutoff
        ]
        for oid in to_remove:
            del self._orders[oid]
        return len(to_remove)


# ═══════════════════════════════════════════════════════════════════════════
#  Order Status Poller — polls CLOB for live-order status updates
# ═══════════════════════════════════════════════════════════════════════════

# Callback type: async fn(Order) called when a fill is detected
FillCallback = Callable[[Order], Coroutine[Any, Any, None]]


class OrderStatusPoller:
    """Periodically polls the CLOB REST API for open-order status.

    For every LIVE or PARTIALLY_FILLED :class:`Order` that has a
    ``clob_order_id``, the poller fetches the latest status from
    ``GET /order/{id}``.  When a fill (full or partial) is detected it
    updates the local :class:`Order` object and invokes the registered
    *on_fill* callback so that :class:`PositionManager` can transition
    the owning position.

    In **paper mode** the poller is a no-op — fills are simulated
    locally by :meth:`OrderExecutor.check_paper_fill`.
    """

    # Mapping from CLOB status strings → internal OrderStatus
    _STATUS_MAP: dict[str, OrderStatus] = {
        "live": OrderStatus.LIVE,
        "LIVE": OrderStatus.LIVE,
        "open": OrderStatus.LIVE,
        "OPEN": OrderStatus.LIVE,
        "matched": OrderStatus.FILLED,
        "MATCHED": OrderStatus.FILLED,
        "filled": OrderStatus.FILLED,
        "FILLED": OrderStatus.FILLED,
        "cancelled": OrderStatus.CANCELLED,
        "CANCELLED": OrderStatus.CANCELLED,
        "canceled": OrderStatus.CANCELLED,
        "CANCELED": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
        "EXPIRED": OrderStatus.EXPIRED,
    }

    def __init__(
        self,
        executor: OrderExecutor,
        *,
        on_fill: FillCallback | None = None,
        poll_interval_s: float | None = None,
        max_retries: int | None = None,
    ):
        self._executor = executor
        self._on_fill = on_fill
        self._poll_s = poll_interval_s or settings.strategy.order_status_poll_s
        self._max_retries = max_retries if max_retries is not None else settings.strategy.order_status_max_retries
        self._running = False
        self._consecutive_errors: dict[str, int] = {}  # clob_order_id → error count

    # ── Public lifecycle ────────────────────────────────────────────────────
    async def run(self) -> None:
        """Long-running coroutine — poll until cancelled."""
        if self._executor.paper_mode:
            return  # paper mode: fills are simulated locally

        self._running = True
        log.info("order_status_poller_started", poll_s=self._poll_s)

        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("order_status_poll_error", error=str(exc))
            await asyncio.sleep(self._poll_s)

        log.info("order_status_poller_stopped")

    def stop(self) -> None:
        self._running = False

    # ── Core polling logic ──────────────────────────────────────────────────
    async def _poll_once(self) -> None:
        """Fetch status for every open order with a clob_order_id."""
        open_orders = [
            o for o in self._executor.get_open_orders()
            if o.clob_order_id
        ]
        if not open_orders:
            return

        clob = self._executor._get_clob_client()
        if clob is None:
            return

        for order in open_orders:
            try:
                resp = await asyncio.to_thread(clob.get_order, order.clob_order_id)
                if resp is None:
                    self._record_error(order.clob_order_id)
                    continue

                self._consecutive_errors.pop(order.clob_order_id, None)
                self._apply_update(order, resp)

            except Exception as exc:
                self._record_error(order.clob_order_id)
                log.warning(
                    "order_status_fetch_failed",
                    clob_id=order.clob_order_id,
                    error=str(exc),
                )

    def _record_error(self, clob_id: str) -> None:
        count = self._consecutive_errors.get(clob_id, 0) + 1
        self._consecutive_errors[clob_id] = count
        if count >= self._max_retries:
            log.warning(
                "order_status_max_retries",
                clob_id=clob_id,
                retries=count,
            )

    def _apply_update(self, order: Order, resp: Any) -> None:
        """Reconcile CLOB response with local order state."""
        if isinstance(resp, dict):
            raw_status = resp.get("status", "")
            filled_size = float(resp.get("size_matched", 0) or resp.get("filled_size", 0) or 0)
            avg_price = float(resp.get("average_price", 0) or resp.get("price", 0) or 0)
        elif hasattr(resp, "status"):
            raw_status = getattr(resp, "status", "")
            filled_size = float(getattr(resp, "size_matched", 0) or getattr(resp, "filled_size", 0) or 0)
            avg_price = float(getattr(resp, "average_price", 0) or getattr(resp, "price", 0) or 0)
        else:
            return

        new_status = self._STATUS_MAP.get(str(raw_status), None)
        if new_status is None:
            log.debug("order_status_unknown", clob_id=order.clob_order_id, raw=raw_status)
            return

        # Detect partial fill growth
        prev_filled = order.filled_size
        if filled_size > prev_filled:
            order.filled_size = filled_size
            if avg_price > 0:
                order.filled_avg_price = avg_price
            order.updated_at = time.time()

        # Detect terminal state transitions
        if new_status != order.status:
            old = order.status
            order.status = new_status
            order.updated_at = time.time()

            log.info(
                "order_status_updated",
                order_id=order.order_id,
                clob_id=order.clob_order_id,
                old_status=old.value,
                new_status=new_status.value,
                filled_size=order.filled_size,
            )

            # Fire fill callback on FILLED transition
            if new_status == OrderStatus.FILLED and self._on_fill:
                asyncio.create_task(self._safe_callback(order))
            elif (
                new_status == OrderStatus.PARTIALLY_FILLED
                and filled_size > prev_filled
                and self._on_fill
            ):
                # Also notify on new partial fills
                asyncio.create_task(self._safe_callback(order))

    async def _safe_callback(self, order: Order) -> None:
        """Invoke the fill callback with error isolation."""
        try:
            if self._on_fill:
                await self._on_fill(order)
        except Exception as exc:
            log.error(
                "fill_callback_error",
                order_id=order.order_id,
                error=str(exc),
            )
