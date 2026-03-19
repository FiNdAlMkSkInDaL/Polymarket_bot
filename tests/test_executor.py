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

        # Queue-aware paper fills require opposing taker flow and two touches.
        filled = executor.check_paper_fill(
            "NO_TOKEN", 0.45, trade_size=20.0, trade_side="sell", is_taker=True
        )
        assert len(filled) == 0
        filled = executor.check_paper_fill(
            "NO_TOKEN", 0.45, trade_size=1.0, trade_side="sell", is_taker=True
        )
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

        # Queue-aware paper fills require opposing taker flow and two touches.
        filled = executor.check_paper_fill(
            "NO_TOKEN", 0.60, trade_size=20.0, trade_side="buy", is_taker=True
        )
        assert len(filled) == 0
        filled = executor.check_paper_fill(
            "NO_TOKEN", 0.60, trade_size=1.0, trade_side="buy", is_taker=True
        )
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
        filled = executor.check_paper_fill(
            "NO_TOKEN", 0.40, trade_size=20.0, trade_side="sell", is_taker=True
        )
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

    @pytest.mark.asyncio
    async def test_place_with_fee_rate_bps(self, executor):
        """fee_rate_bps kwarg should be accepted without error."""
        order = await executor.place_limit_order(
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            fee_rate_bps=156,
        )
        assert order.status == OrderStatus.LIVE
        assert order.price == 0.45


class TestLatencyGuardForceBlock:
    def test_force_block_sets_blocked_state(self):
        from src.core.latency_guard import LatencyGuard, LatencyState

        guard = LatencyGuard(block_ms=500, warn_ms=200, recovery_count=3)
        assert guard.state == LatencyState.HEALTHY

        guard.force_block("test_reason")
        assert guard.state == LatencyState.BLOCKED
        assert guard.is_blocked() is True

    def test_force_block_resets_consecutive(self):
        from src.core.latency_guard import LatencyGuard, LatencyState

        guard = LatencyGuard(block_ms=500, warn_ms=200, recovery_count=3)
        guard._consecutive_healthy = 5

        guard.force_block("test")
        assert guard._consecutive_healthy == 0
