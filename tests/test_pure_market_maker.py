from __future__ import annotations

from dataclasses import dataclass

import pytest

import src.core.config as cfg
from src.data.market_discovery import MarketInfo
from src.strategies.pure_market_maker import PureMarketMaker
from src.trading.executor import Order, OrderSide, OrderStatus


@dataclass
class _Snapshot:
    best_bid: float
    best_ask: float


class _FakeTracker:
    def __init__(self, best_bid: float = 0.45, best_ask: float = 0.46):
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.has_data = True
        self.book_depth_ratio = 1.0

    def snapshot(self):
        return _Snapshot(best_bid=self.best_bid, best_ask=self.best_ask)


class _FakeL2Book:
    def __init__(self):
        self._against = 0.0
        self._depth_velocity = 0.0

    def opposing_ofi_at_price(self, price: float, side: str) -> float:
        return self._against

    def depth_velocity(self, window_s: float) -> float:
        return self._depth_velocity


class _FakeExecutor:
    def __init__(self):
        self.paper_mode = True
        self.orders: list[Order] = []
        self.cancelled: list[str] = []

    async def place_limit_order(self, market_id, asset_id, side, price, size, *, post_only=False, fee_rate_bps=0):
        order = Order(
            order_id=f"ORD-{len(self.orders) + 1}",
            market_id=market_id,
            asset_id=asset_id,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.LIVE,
            post_only=post_only,
        )
        self.orders.append(order)
        return order

    async def cancel_order(self, order: Order) -> None:
        order.status = OrderStatus.CANCELLED
        self.cancelled.append(order.order_id)


def _build_market() -> MarketInfo:
    return MarketInfo(
        condition_id="COND1",
        question="Will X happen?",
        yes_token_id="YES1",
        no_token_id="NO1",
        daily_volume_usd=10000.0,
        end_date=None,
        active=True,
    )


@pytest.fixture
def restore_strategy_params():
    original = cfg.settings.strategy
    yield
    object.__setattr__(cfg.settings, "strategy", original)


@pytest.mark.asyncio
async def test_pure_mm_places_bid_then_inventory_backed_ask():
    market = _build_market()
    tracker = _FakeTracker()
    l2_book = _FakeL2Book()
    executor = _FakeExecutor()

    mm = PureMarketMaker(
        executor=executor,
        get_active_markets=lambda: [market],
        get_l2_books=lambda: {market.no_token_id: l2_book},
        get_book_trackers=lambda: {market.no_token_id: tracker},
        get_l2_active_set=lambda: {market.condition_id},
    )

    await mm._sync_quotes()

    assert len(executor.orders) == 1
    assert executor.orders[0].side == OrderSide.BUY

    bid = executor.orders[0]
    bid.status = OrderStatus.FILLED
    bid.filled_size = bid.size
    bid.filled_avg_price = bid.price
    assert await mm.on_order_fill(bid) is True

    await mm._sync_quotes()

    assert any(order.side == OrderSide.SELL for order in executor.orders)


@pytest.mark.asyncio
async def test_pure_mm_places_tight_and_wide_quotes_at_expected_prices(restore_strategy_params):
    market = _build_market()
    tracker = _FakeTracker()
    l2_book = _FakeL2Book()
    executor = _FakeExecutor()

    object.__setattr__(
        cfg.settings,
        "strategy",
        cfg.StrategyParams(
            pure_mm_inventory_cap_usd=1000.0,
            pure_mm_wide_tier_enabled=True,
            pure_mm_wide_spread_pct=0.15,
            pure_mm_wide_size_usd=50.0,
        ),
    )

    mm = PureMarketMaker(
        executor=executor,
        get_active_markets=lambda: [market],
        get_l2_books=lambda: {market.no_token_id: l2_book},
        get_book_trackers=lambda: {market.no_token_id: tracker},
        get_l2_active_set=lambda: {market.condition_id},
    )
    mm._inventory[market.no_token_id] = 200.0

    await mm._sync_quotes()

    assert len(executor.orders) == 4
    placed = {(order.side, order.price): order.size for order in executor.orders}
    assert placed[(OrderSide.BUY, 0.45)] == pytest.approx(11.11)
    assert placed[(OrderSide.SELL, 0.46)] == pytest.approx(10.87)
    assert placed[(OrderSide.BUY, 0.38)] == pytest.approx(111.11)
    assert placed[(OrderSide.SELL, 0.53)] == pytest.approx(108.7)


@pytest.mark.asyncio
async def test_pure_mm_cancels_all_tiers_on_toxic_flow(restore_strategy_params):
    market = _build_market()
    tracker = _FakeTracker()
    l2_book = _FakeL2Book()
    executor = _FakeExecutor()

    object.__setattr__(
        cfg.settings,
        "strategy",
        cfg.StrategyParams(
            pure_mm_inventory_cap_usd=1000.0,
            pure_mm_wide_tier_enabled=True,
            pure_mm_wide_spread_pct=0.15,
            pure_mm_wide_size_usd=50.0,
            pure_mm_toxic_ofi_ratio=0.5,
        ),
    )

    mm = PureMarketMaker(
        executor=executor,
        get_active_markets=lambda: [market],
        get_l2_books=lambda: {market.no_token_id: l2_book},
        get_book_trackers=lambda: {market.no_token_id: tracker},
        get_l2_active_set=lambda: {market.condition_id},
    )
    mm._inventory[market.no_token_id] = 200.0

    await mm._sync_quotes()
    assert len(executor.orders) == 4

    l2_book._against = max(order.size for order in executor.orders)
    await mm._sync_quotes()

    assert executor.cancelled == [order.order_id for order in executor.orders]