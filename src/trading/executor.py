"""
Order executor — wraps the Polymarket CLOB client to place, cancel, and
monitor orders.  In paper mode, all calls are intercepted and simulated.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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

                order_args = OrderArgs(
                    price=price,
                    size=size,
                    side=BUY if side == OrderSide.BUY else SELL,
                    token_id=asset_id,
                )
                # Attach fee rate to the signed payload when taking liquidity
                if fee_rate_bps > 0:
                    order_args.fee_rate_bps = fee_rate_bps
                resp = clob.create_and_post_order(order_args)
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
                clob.cancel(order.clob_order_id)
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
        """Simulate fills for paper orders when the market crosses."""
        if not self.paper_mode:
            return []

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
                order.status = OrderStatus.FILLED
                order.filled_size = order.size
                order.filled_avg_price = order.price
                order.updated_at = time.time()
                log.info(
                    "paper_fill",
                    order_id=order.order_id,
                    side=order.side.value,
                    price=order.price,
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
