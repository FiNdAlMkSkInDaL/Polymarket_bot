"""Inventory-aware pure spread quoting for high-liquidity L2 markets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from src.core.config import EXCHANGE_MIN_SHARES, EXCHANGE_MIN_USD, settings
from src.core.logger import get_logger
from src.data.market_discovery import MarketInfo
from src.data.orderbook import OrderbookTracker
from src.signals.iceberg_detector import IcebergDetector
from src.trading.chaser import OrderChaser
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus

log = get_logger(__name__)


@dataclass(slots=True)
class QuoteState:
    market_id: str
    asset_id: str
    side: OrderSide
    order: Order | None = None
    acked_fill_size: float = 0.0


class PureMarketMaker:
    """Continuously maintains passive quotes on selected L2 markets.

    The current implementation quotes the NO token for each selected market.
    Bid quotes are always allowed. Ask quotes are inventory-aware and only
    rest when this engine has previously acquired inventory via filled bids.
    """

    def __init__(
        self,
        *,
        executor: OrderExecutor,
        get_active_markets: Callable[[], list[MarketInfo]],
        get_l2_books: Callable[[], dict[str, object]],
        get_book_trackers: Callable[[], dict[str, OrderbookTracker]],
        get_l2_active_set: Callable[[], set[str]],
        latency_guard: object | None = None,
        fast_kill_event: asyncio.Event | None = None,
        maker_monitor: object | None = None,
        iceberg_detectors: dict[str, IcebergDetector] | None = None,
    ):
        self._executor = executor
        self._get_active_markets = get_active_markets
        self._get_l2_books = get_l2_books
        self._get_book_trackers = get_book_trackers
        self._get_l2_active_set = get_l2_active_set
        self._latency_guard = latency_guard
        self._fast_kill_event = fast_kill_event
        self._maker_monitor = maker_monitor
        self._iceberg_detectors = iceberg_detectors or {}

        self._running = False
        self._quote_states: dict[tuple[str, OrderSide], QuoteState] = {}
        self._order_to_key: dict[str, tuple[str, OrderSide]] = {}
        self._inventory: dict[str, float] = {}

        strat = settings.strategy
        self._loop_interval_s = max(0.05, strat.pure_mm_loop_ms / 1000.0)
        self._max_markets = strat.pure_mm_max_markets
        self._quote_size_usd = strat.pure_mm_quote_size_usd
        self._inventory_cap_usd = strat.pure_mm_inventory_cap_usd
        self._toxic_ofi_ratio = strat.pure_mm_toxic_ofi_ratio
        self._depth_window_s = strat.pure_mm_depth_window_s
        self._depth_evaporation_pct = strat.pure_mm_depth_evaporation_pct

    async def run(self) -> None:
        self._running = True
        log.info(
            "pure_mm_started",
            max_markets=self._max_markets,
            quote_size_usd=self._quote_size_usd,
            inventory_cap_usd=self._inventory_cap_usd,
        )
        while self._running:
            try:
                await self._sync_quotes()
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("pure_mm_loop_error", exc_info=True)
            await asyncio.sleep(self._loop_interval_s)
        log.info("pure_mm_stopped")

    async def stop(self) -> None:
        self._running = False
        await self.cancel_all_quotes()

    async def cancel_all_quotes(self) -> None:
        for state in list(self._quote_states.values()):
            await self._cancel_state(state, reason="shutdown")

    async def on_order_fill(self, order: Order) -> bool:
        key = self._order_to_key.get(order.order_id)
        if key is None:
            return False

        state = self._quote_states.get(key)
        if state is None:
            self._order_to_key.pop(order.order_id, None)
            return False

        delta = max(0.0, order.filled_size - state.acked_fill_size)
        if delta > 0:
            signed = delta if order.side == OrderSide.BUY else -delta
            self._inventory[order.asset_id] = max(
                0.0,
                self._inventory.get(order.asset_id, 0.0) + signed,
            )
            state.acked_fill_size = order.filled_size
            log.info(
                "pure_mm_fill",
                order_id=order.order_id,
                asset_id=order.asset_id,
                side=order.side.value,
                fill_size=round(delta, 4),
                total_inventory=round(self._inventory.get(order.asset_id, 0.0), 4),
                status=order.status.value,
            )

        if order.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        ):
            state.order = None
            state.acked_fill_size = 0.0
            self._order_to_key.pop(order.order_id, None)

        return True

    def inventory_for_asset(self, asset_id: str) -> float:
        return self._inventory.get(asset_id, 0.0)

    async def _sync_quotes(self) -> None:
        selected = self._selected_markets()
        selected_assets = {market.no_token_id for market in selected}

        for key, state in list(self._quote_states.items()):
            if state.asset_id not in selected_assets:
                await self._cancel_state(state, reason="market_unselected")

        for market in selected:
            await self._sync_market(market)

    def _selected_markets(self) -> list[MarketInfo]:
        active = self._get_active_markets()
        l2_active = self._get_l2_active_set()
        eligible = [
            market
            for market in active
            if market.condition_id in l2_active
        ]
        eligible.sort(key=lambda market: market.daily_volume_usd, reverse=True)
        return eligible[: self._max_markets]

    async def _sync_market(self, market: MarketInfo) -> None:
        asset_id = market.no_token_id
        book_trackers = self._get_book_trackers()
        tracker = book_trackers.get(asset_id)
        l2_book = self._get_l2_books().get(asset_id)
        if tracker is None or not tracker.has_data or l2_book is None:
            await self._cancel_asset_quotes(asset_id, reason="no_l2_data")
            return

        snap = tracker.snapshot()
        if snap.best_bid <= 0 or snap.best_ask <= 0:
            await self._cancel_asset_quotes(asset_id, reason="empty_bbo")
            return

        bid_size = self._quote_size_for_bid(snap.best_bid, asset_id)
        ask_size = self._quote_size_for_ask(snap.best_ask, asset_id)

        await self._sync_side(
            market=market,
            asset_id=asset_id,
            side=OrderSide.BUY,
            target_size=bid_size,
            tracker=tracker,
            l2_book=l2_book,
        )
        await self._sync_side(
            market=market,
            asset_id=asset_id,
            side=OrderSide.SELL,
            target_size=ask_size,
            tracker=tracker,
            l2_book=l2_book,
        )

    async def _sync_side(
        self,
        *,
        market: MarketInfo,
        asset_id: str,
        side: OrderSide,
        target_size: float,
        tracker: OrderbookTracker,
        l2_book: object,
    ) -> None:
        key = (asset_id, side)
        state = self._quote_states.setdefault(
            key,
            QuoteState(market_id=market.condition_id, asset_id=asset_id, side=side),
        )
        state.market_id = market.condition_id

        if target_size <= 0:
            await self._cancel_state(state, reason="no_target_size")
            return

        helper = self._build_quote_helper(
            market_id=market.condition_id,
            asset_id=asset_id,
            side=side,
            tracker=tracker,
            anchor_price=tracker.best_bid if side == OrderSide.BUY else tracker.best_ask,
            size=target_size,
        )
        target_price = helper.quote_price()
        if target_price is None or target_price <= 0:
            await self._cancel_state(state, reason="no_quote_price")
            return

        if self._should_cancel_quote(
            order=state.order,
            target_size=target_size,
            side=side,
            l2_book=l2_book,
            helper=helper,
        ):
            await self._cancel_state(state, reason="toxic_flow")
            return

        if state.order is not None:
            if state.order.status not in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED):
                self._detach_order(state.order.order_id, state)
                state.order = None
                state.acked_fill_size = 0.0
            else:
                remaining = max(0.0, state.order.size - state.order.filled_size)
                price_match = abs(state.order.price - round(target_price, 2)) < 1e-6
                size_match = abs(remaining - round(target_size, 2)) < 1e-6
                if price_match and size_match:
                    return
                await self._cancel_state(state, reason="requote")

        order = await self._executor.place_limit_order(
            market_id=market.condition_id,
            asset_id=asset_id,
            side=side,
            price=round(target_price, 2),
            size=round(target_size, 2),
            post_only=True,
        )
        if order.status == OrderStatus.CANCELLED and order.rejection_reason == "would_cross":
            log.debug(
                "pure_mm_post_only_rejected",
                market_id=market.condition_id,
                asset_id=asset_id,
                side=side.value,
                price=round(target_price, 2),
            )
            return

        state.order = order
        state.acked_fill_size = order.filled_size
        self._order_to_key[order.order_id] = key
        log.info(
            "pure_mm_quote_resting",
            market_id=market.condition_id,
            asset_id=asset_id,
            side=side.value,
            price=round(order.price, 2),
            size=round(order.size, 2),
        )

    def _quote_size_for_bid(self, best_bid: float, asset_id: str) -> float:
        if best_bid <= 0:
            return 0.0
        inventory_value = self.inventory_for_asset(asset_id) * best_bid
        if inventory_value >= self._inventory_cap_usd:
            return 0.0
        return self._normalise_size(self._quote_size_usd / best_bid, best_bid)

    def _quote_size_for_ask(self, best_ask: float, asset_id: str) -> float:
        inventory = self.inventory_for_asset(asset_id)
        if inventory <= 0 or best_ask <= 0:
            return 0.0
        target = min(inventory, self._quote_size_usd / best_ask)
        return self._normalise_size(target, best_ask)

    @staticmethod
    def _normalise_size(raw_size: float, price: float) -> float:
        if raw_size <= 0 or price <= 0:
            return 0.0
        min_size = max(EXCHANGE_MIN_SHARES, EXCHANGE_MIN_USD / price)
        size = max(raw_size, min_size)
        return round(size, 2)

    async def _cancel_asset_quotes(self, asset_id: str, *, reason: str) -> None:
        for side in (OrderSide.BUY, OrderSide.SELL):
            state = self._quote_states.get((asset_id, side))
            if state is not None:
                await self._cancel_state(state, reason=reason)

    async def _cancel_state(self, state: QuoteState, *, reason: str) -> None:
        order = state.order
        if order is None:
            return
        self._detach_order(order.order_id, state)
        state.order = None
        state.acked_fill_size = 0.0
        await self._executor.cancel_order(order)
        log.info(
            "pure_mm_quote_cancelled",
            market_id=state.market_id,
            asset_id=state.asset_id,
            side=state.side.value,
            reason=reason,
        )

    def _detach_order(self, order_id: str, state: QuoteState) -> None:
        self._order_to_key.pop(order_id, None)
        key = (state.asset_id, state.side)
        if self._quote_states.get(key) is state and state.order is None:
            self._quote_states.pop(key, None)

    def _build_quote_helper(
        self,
        *,
        market_id: str,
        asset_id: str,
        side: OrderSide,
        tracker: OrderbookTracker,
        anchor_price: float,
        size: float,
    ) -> OrderChaser:
        return OrderChaser(
            executor=self._executor,
            book=tracker,
            market_id=market_id,
            asset_id=asset_id,
            side=side,
            target_size=max(size, EXCHANGE_MIN_SHARES),
            anchor_price=max(anchor_price, 0.01),
            latency_guard=self._latency_guard,
            fast_kill_event=self._fast_kill_event,
            max_post_only_rejections=10**9,
            iceberg_detector=self._iceberg_detectors.get(asset_id),
            adverse_monitor=self._maker_monitor,
        )

    def _should_cancel_quote(
        self,
        *,
        order: Order | None,
        target_size: float,
        side: OrderSide,
        l2_book: object,
        helper: OrderChaser,
    ) -> bool:
        if order is None:
            return False
        if helper.should_cancel_quote():
            return True

        depth_velocity_fn = getattr(l2_book, "depth_velocity", None)
        if callable(depth_velocity_fn):
            dv = depth_velocity_fn(self._depth_window_s)
            if dv is not None and dv <= -self._depth_evaporation_pct:
                return True

        ofi_fn = getattr(l2_book, "opposing_ofi_at_price", None)
        if not callable(ofi_fn):
            return False

        quote_side = "BUY" if side == OrderSide.BUY else "SELL"
        against_flow = float(ofi_fn(order.price, quote_side) or 0.0)
        return against_flow >= max(target_size, order.size) * self._toxic_ofi_ratio
