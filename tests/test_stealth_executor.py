"""Tests for the StealthExecutor (SI-4)."""

from __future__ import annotations

import asyncio

import pytest

from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.stealth_executor import StealthExecutor, StealthPlan


class TestStealthPlan:
    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=4,
            min_delay_ms=50.0,
            max_delay_ms=100.0,
            size_jitter_pct=0.15,
        )

    def test_plan_slice_count(self, stealth):
        plan = stealth._build_plan(total_size=20.0, price=0.50)
        # 20 * 0.50 = $10 → 10/3 + 1 ≈ 4 slices, capped at 4
        assert 2 <= plan.num_slices <= 4

    def test_plan_slices_sum_to_total(self, stealth):
        plan = stealth._build_plan(total_size=20.0, price=0.50)
        assert sum(plan.slice_sizes) == pytest.approx(20.0, abs=0.05)

    def test_plan_delays_between_bounds(self, stealth):
        plan = stealth._build_plan(total_size=20.0, price=0.50)
        for d in plan.delays_ms:
            assert 50.0 <= d <= 100.0

    def test_plan_small_order_gets_2_slices(self, stealth):
        plan = stealth._build_plan(total_size=6.0, price=0.50)
        # 6 * 0.50 = $3 → 3/3 + 1 = 2 slices
        assert plan.num_slices == 2


class TestStealthExecution:
    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=3,
            min_delay_ms=10.0,
            max_delay_ms=20.0,
            size_jitter_pct=0.10,
        )

    @pytest.mark.asyncio
    async def test_small_order_passes_through(self, stealth):
        """Orders below min_size_usd should not be split."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=5.0,  # 5 * 0.50 = $2.50 < $5.0
        )
        assert len(orders) == 1
        assert orders[0].size == 5.0

    @pytest.mark.asyncio
    async def test_large_order_is_split(self, stealth):
        """Orders above min_size_usd should be split into multiple slices."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,  # 20 * 0.50 = $10 > $5.0
        )
        assert len(orders) >= 2
        total_placed = sum(o.size for o in orders)
        assert total_placed == pytest.approx(20.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_all_orders_are_live(self, stealth):
        """In paper mode, all child orders should be LIVE."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
        )
        for o in orders:
            assert o.status == OrderStatus.LIVE

    @pytest.mark.asyncio
    async def test_sell_side(self, stealth):
        """Stealth execution should work for SELL orders too."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.SELL,
            price=0.50,
            total_size=20.0,
        )
        assert len(orders) >= 2
        for o in orders:
            assert o.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_post_only(self, stealth):
        """post_only should be forwarded to all child orders."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
            post_only=True,
        )
        for o in orders:
            assert o.post_only is True

    def test_executor_property(self, stealth):
        assert isinstance(stealth.executor, OrderExecutor)
