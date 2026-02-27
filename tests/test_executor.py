"""
Tests for the order executor in paper mode.
"""

import pytest

from src.trading.executor import OrderExecutor, OrderSide, OrderStatus


class TestOrderExecutorPaper:
    @pytest.fixture
    def executor(self):
        return OrderExecutor(paper_mode=True)

    @pytest.mark.asyncio
    async def test_place_limit_order(self, executor):
        order = await executor.place_limit_order(
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
        )
        assert order.status == OrderStatus.LIVE
        assert order.price == 0.45
        assert order.size == 10.0
        assert order.order_id.startswith("PAPER")

    @pytest.mark.asyncio
    async def test_paper_fill_buy(self, executor):
        order = await executor.place_limit_order(
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
        )

        # Price above → no fill
        filled = executor.check_paper_fill("NO_TOKEN", 0.50)
        assert len(filled) == 0
        assert order.status == OrderStatus.LIVE

        # Price at or below → fill
        filled = executor.check_paper_fill("NO_TOKEN", 0.45)
        assert len(filled) == 1
        assert filled[0].order_id == order.order_id
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_paper_fill_sell(self, executor):
        order = await executor.place_limit_order(
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.SELL,
            price=0.60,
            size=10.0,
        )

        # Price below → no fill
        filled = executor.check_paper_fill("NO_TOKEN", 0.55)
        assert len(filled) == 0

        # Price at or above → fill
        filled = executor.check_paper_fill("NO_TOKEN", 0.60)
        assert len(filled) == 1
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancel_order(self, executor):
        order = await executor.place_limit_order(
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
        )
        await executor.cancel_order(order)
        assert order.status == OrderStatus.CANCELLED

        # Cancelled order should not fill
        filled = executor.check_paper_fill("NO_TOKEN", 0.40)
        assert len(filled) == 0

    @pytest.mark.asyncio
    async def test_cancel_all(self, executor):
        await executor.place_limit_order("MKT", "T1", OrderSide.BUY, 0.40, 5)
        await executor.place_limit_order("MKT", "T2", OrderSide.BUY, 0.35, 5)
        await executor.place_limit_order("MKT", "T3", OrderSide.SELL, 0.70, 5)

        cancelled = await executor.cancel_all()
        assert cancelled == 3
        assert len(executor.get_open_orders()) == 0

    @pytest.mark.asyncio
    async def test_get_open_orders_filters_by_market(self, executor):
        await executor.place_limit_order("MKT_A", "T1", OrderSide.BUY, 0.40, 5)
        await executor.place_limit_order("MKT_B", "T2", OrderSide.BUY, 0.40, 5)

        assert len(executor.get_open_orders("MKT_A")) == 1
        assert len(executor.get_open_orders("MKT_B")) == 1
        assert len(executor.get_open_orders()) == 2
