"""Inventory-aware pure spread quoting for high-liquidity L2 markets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

import src.core.config as config
from src.core.logger import get_logger
from src.data.market_discovery import MarketInfo
from src.data.orderbook import OrderbookTracker
from src.signals.iceberg_detector import IcebergDetector
from src.trading.chaser import OrderChaser
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus

log = get_logger(__name__)

TIGHT_TIER = "tight"
WIDE_TIER = "wide"


@dataclass(slots=True)
class QuoteState:
    market_id: str
    asset_id: str
    side: OrderSide
    tier: str
    order: Order | None = None
    acked_fill_size: float = 0.0


@dataclass(frozen=True, slots=True)
class IntendedQuote:
    side: OrderSide
    tier: str
    target_price: float
    target_size: float


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
        self._quote_states: dict[tuple[str, OrderSide, str], QuoteState] = {}
        self._order_to_key: dict[str, tuple[str, OrderSide, str]] = {}
        self._inventory: dict[str, float] = {}

        strat = config.settings.strategy
        self._loop_interval_s = max(0.05, strat.pure_mm_loop_ms / 1000.0)
        self._max_markets = strat.pure_mm_max_markets
        self._quote_size_usd = strat.pure_mm_quote_size_usd
        self._wide_tier_enabled = strat.pure_mm_wide_tier_enabled
        self._wide_spread_pct = strat.pure_mm_wide_spread_pct
        self._wide_size_usd = strat.pure_mm_wide_size_usd
        self._inventory_cap_usd = strat.pure_mm_inventory_cap_usd
        self._inventory_penalty_coef = max(0.0, strat.pure_mm_inventory_penalty_coef)
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

        for _, state in list(self._quote_states.items()):
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

        intended_quotes = self._build_intended_quotes(asset_id=asset_id, snapshot=snap)
        if self._is_market_toxic(asset_id=asset_id, l2_book=l2_book, intended_quotes=intended_quotes):
            await self._cancel_asset_quotes(asset_id, reason="toxic_flow")
            return

        active_keys: set[tuple[str, OrderSide, str]] = set()
        for intended_quote in intended_quotes:
            active_keys.add((asset_id, intended_quote.side, intended_quote.tier))
            await self._sync_quote(
                market=market,
                asset_id=asset_id,
                tracker=tracker,
                intended_quote=intended_quote,
            )

        for key, state in list(self._quote_states.items()):
            if state.asset_id == asset_id and key not in active_keys:
                await self._cancel_state(state, reason="quote_disabled")

    def _build_intended_quotes(self, *, asset_id: str, snapshot: object) -> list[IntendedQuote]:
        best_bid = float(snapshot.best_bid)
        best_ask = float(snapshot.best_ask)
        intended_quotes: list[IntendedQuote] = []

        tight_bid_size = self._quote_size_for_bid(best_bid, asset_id, self._quote_size_usd)
        tight_ask_size = self._quote_size_for_ask(best_ask, asset_id, self._quote_size_usd)
        if tight_bid_size > 0:
            intended_quotes.append(
                IntendedQuote(
                    side=OrderSide.BUY,
                    tier=TIGHT_TIER,
                    target_price=best_bid,
                    target_size=tight_bid_size,
                )
            )
        if tight_ask_size > 0:
            intended_quotes.append(
                IntendedQuote(
                    side=OrderSide.SELL,
                    tier=TIGHT_TIER,
                    target_price=best_ask,
                    target_size=tight_ask_size,
                )
            )

        if not self._wide_tier_enabled:
            return intended_quotes

        wide_bid_size = self._quote_size_for_bid(best_bid, asset_id, self._wide_size_usd)
        wide_ask_size = self._quote_size_for_ask(best_ask, asset_id, self._wide_size_usd)
        wide_bid_price = best_bid * (1.0 - self._wide_spread_pct)
        wide_ask_price = best_ask * (1.0 + self._wide_spread_pct)
        if wide_bid_size > 0 and wide_bid_price > 0:
            intended_quotes.append(
                IntendedQuote(
                    side=OrderSide.BUY,
                    tier=WIDE_TIER,
                    target_price=wide_bid_price,
                    target_size=wide_bid_size,
                )
            )
        if wide_ask_size > 0 and wide_ask_price > 0:
            intended_quotes.append(
                IntendedQuote(
                    side=OrderSide.SELL,
                    tier=WIDE_TIER,
                    target_price=wide_ask_price,
                    target_size=wide_ask_size,
                )
            )
        return intended_quotes

    async def _sync_quote(
        self,
        *,
        market: MarketInfo,
        asset_id: str,
        tracker: OrderbookTracker,
        intended_quote: IntendedQuote,
    ) -> None:
        key = (asset_id, intended_quote.side, intended_quote.tier)
        state = self._quote_states.setdefault(
            key,
            QuoteState(
                market_id=market.condition_id,
                asset_id=asset_id,
                side=intended_quote.side,
                tier=intended_quote.tier,
            ),
        )
        state.market_id = market.condition_id

        if intended_quote.target_size <= 0:
            await self._cancel_state(state, reason="no_target_size")
            return

        helper = self._build_quote_helper(
            market_id=market.condition_id,
            asset_id=asset_id,
            side=intended_quote.side,
            tracker=tracker,
            anchor_price=intended_quote.target_price,
            size=intended_quote.target_size,
        )
        target_price = intended_quote.target_price
        if target_price <= 0:
            await self._cancel_state(state, reason="no_quote_price")
            return

        if state.order is not None and helper.should_cancel_quote():
            await self._cancel_asset_quotes(asset_id, reason="toxic_flow")
            return

        if state.order is not None:
            if state.order.status not in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED):
                self._detach_order(state.order.order_id, state)
                state.order = None
                state.acked_fill_size = 0.0
            else:
                remaining = max(0.0, state.order.size - state.order.filled_size)
                price_match = abs(state.order.price - round(target_price, 2)) < 1e-6
                size_match = abs(remaining - round(intended_quote.target_size, 2)) < 1e-6
                if price_match and size_match:
                    return
                await self._cancel_state(state, reason="requote")

        order = await self._executor.place_limit_order(
            market_id=market.condition_id,
            asset_id=asset_id,
            side=intended_quote.side,
            price=round(target_price, 2),
            size=round(intended_quote.target_size, 2),
            post_only=True,
        )
        if order.status == OrderStatus.CANCELLED and order.rejection_reason == "would_cross":
            log.debug(
                "pure_mm_post_only_rejected",
                market_id=market.condition_id,
                asset_id=asset_id,
                side=intended_quote.side.value,
                price=round(target_price, 2),
                tier=intended_quote.tier,
            )
            return

        state.order = order
        state.acked_fill_size = order.filled_size
        self._order_to_key[order.order_id] = key
        log.info(
            "pure_mm_quote_resting",
            market_id=market.condition_id,
            asset_id=asset_id,
            side=intended_quote.side.value,
            price=round(order.price, 2),
            size=round(order.size, 2),
            tier=intended_quote.tier,
        )

    def _quote_size_for_bid(self, best_bid: float, asset_id: str, size_usd: float) -> float:
        if best_bid <= 0:
            return 0.0
        inventory_value = self.inventory_for_asset(asset_id) * best_bid
        if inventory_value >= self._inventory_cap_usd:
            return 0.0
        penalty_scale = self._inventory_penalty_scale(inventory_value)
        if penalty_scale <= 0:
            return 0.0
        target_usd = size_usd * penalty_scale
        return self._normalise_size(target_usd / best_bid, best_bid)

    def _quote_size_for_ask(self, best_ask: float, asset_id: str, size_usd: float) -> float:
        inventory = self.inventory_for_asset(asset_id)
        if inventory <= 0 or best_ask <= 0:
            return 0.0
        target = min(inventory, size_usd / best_ask)
        return self._normalise_size(target, best_ask)

    def _inventory_penalty_scale(self, inventory_value: float) -> float:
        if self._inventory_cap_usd <= 0:
            return 0.0
        fill_ratio = min(max(inventory_value / self._inventory_cap_usd, 0.0), 1.0)
        return max(0.0, (1.0 - fill_ratio) ** self._inventory_penalty_coef)

    @staticmethod
    def _normalise_size(raw_size: float, price: float) -> float:
        if raw_size <= 0 or price <= 0:
            return 0.0
        min_size = max(config.EXCHANGE_MIN_SHARES, config.EXCHANGE_MIN_USD / price)
        size = max(raw_size, min_size)
        return round(size, 2)

    async def _cancel_asset_quotes(self, asset_id: str, *, reason: str) -> None:
        for state in list(self._quote_states.values()):
            if state.asset_id == asset_id:
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
            tier=state.tier,
            reason=reason,
        )

    def _detach_order(self, order_id: str, state: QuoteState) -> None:
        self._order_to_key.pop(order_id, None)
        key = (state.asset_id, state.side, state.tier)
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
            target_size=max(size, config.EXCHANGE_MIN_SHARES),
            anchor_price=max(anchor_price, 0.01),
            latency_guard=self._latency_guard,
            fast_kill_event=self._fast_kill_event,
            max_post_only_rejections=10**9,
            iceberg_detector=self._iceberg_detectors.get(asset_id),
            adverse_monitor=self._maker_monitor,
        )

    def _is_market_toxic(
        self,
        *,
        asset_id: str,
        l2_book: object,
        intended_quotes: list[IntendedQuote],
    ) -> bool:
        depth_velocity_fn = getattr(l2_book, "depth_velocity", None)
        if callable(depth_velocity_fn):
            dv = depth_velocity_fn(self._depth_window_s)
            if dv is not None and dv <= -self._depth_evaporation_pct:
                return True

        ofi_fn = getattr(l2_book, "opposing_ofi_at_price", None)
        if not callable(ofi_fn):
            return False

        checks = [
            (quote.target_price, quote.side, quote.target_size)
            for quote in intended_quotes
        ]
        checks.extend(
            (state.order.price, state.side, state.order.size)
            for state in self._quote_states.values()
            if state.asset_id == asset_id and state.order is not None
        )
        for price, side, size in checks:
            quote_side = "BUY" if side == OrderSide.BUY else "SELL"
            against_flow = float(ofi_fn(price, quote_side) or 0.0)
            if against_flow >= size * self._toxic_ofi_ratio:
                return True
        return False
