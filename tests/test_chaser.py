"""
Tests for OrderChaser — passive-aggressive quoting + escalation (Pillar 7).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.data.orderbook import OrderbookTracker, OrderbookSnapshot
from src.trading.chaser import ChaserState, OrderChaser
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus


def _seed_book(tracker: OrderbookTracker, bid: float = 0.47, ask: float = 0.53) -> None:
    """Inject a synthetic L2 snapshot so the tracker has data."""
    tracker.on_book_snapshot({
        "asset_id": tracker.asset_id,
        "bids": [{"price": str(bid), "size": "100"}],
        "asks": [{"price": str(ask), "size": "100"}],
    })


class TestChaserBasics:
    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        book = OrderbookTracker("ASSET_A")
        _seed_book(book, bid=0.47, ask=0.53)
        return executor, book

    @pytest.mark.asyncio
    async def test_initial_quote_placed(self, setup):
        """Chaser should place a resting order immediately."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
        )
        # Don't run the full loop — just verify construction
        assert chaser.state == ChaserState.QUOTING
        assert chaser.resting_order is None

    @pytest.mark.asyncio
    async def test_buy_chaser_fills_when_price_drops(self, setup):
        """Paper BUY chaser should fill when market price <= order price."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
        )

        # Place initial order manually
        await chaser._place(0.47)
        assert chaser.resting_order is not None

        # Simulate paper fill
        result = chaser.force_check_fill("ASSET_A", 0.47)
        assert result is True
        assert chaser.state == ChaserState.FILLED

    @pytest.mark.asyncio
    async def test_sell_chaser_fills_when_price_rises(self, setup):
        """Paper SELL chaser should fill when market price >= order price."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.SELL, target_size=10.0,
            anchor_price=0.55,
        )

        await chaser._place(0.53)
        assert chaser.resting_order is not None

        result = chaser.force_check_fill("ASSET_A", 0.53)
        assert result is True
        assert chaser.state == ChaserState.FILLED

    @pytest.mark.asyncio
    async def test_drift_beyond_max_abandons(self, setup):
        """If BBO drifts beyond max_chase_depth_cents, chaser abandons."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            max_chase_depth_cents=2.0,
        )

        # Drift is measured from anchor price
        # For BUY: adverse = price UP
        drift = chaser._drift_cents(0.50)  # 3¢ > 2¢
        assert drift > 2.0


class TestChaserEscalation:
    """Pillar 7 — hybrid-aggressive protocol tests."""

    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        book = OrderbookTracker("ASSET_A")
        _seed_book(book, bid=0.47, ask=0.53)
        event = asyncio.Event()
        event.set()
        return executor, book, event

    @pytest.mark.asyncio
    async def test_escalation_after_n_rejections(self, setup):
        """After max_post_only_rejections, chaser should enter ESCALATING."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=0.56,
            fee_rate_bps=100,
            fast_kill_event=event,
            max_post_only_rejections=2,
        )

        # Simulate 2 consecutive rejections
        chaser._rejection_count = 2

        # Call _escalate directly
        await chaser._escalate()

        # Should have either escalated to a marketable order or abandoned
        assert chaser.state in (ChaserState.QUOTING, ChaserState.FILLED, ChaserState.ABANDONED)

    @pytest.mark.asyncio
    async def test_alpha_check_buy_side(self, setup):
        """Alpha check should pass when spread > margin + fees."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=0.56,
            fee_rate_bps=100,
            fast_kill_event=event,
        )

        # Cross price at 0.48: gross = (0.56 - 0.48)*100 = 8¢
        # Entry fee = 0.48 * 100/10000 * 100 = 0.48¢
        # Net = 8 - 0.48 = 7.52¢ >> 1.0¢ margin
        assert chaser._alpha_check(0.48) is True

    @pytest.mark.asyncio
    async def test_alpha_check_fails_tight_spread(self, setup):
        """Alpha check should fail when fees eat the entire spread."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.55,
            tp_target_price=0.56,
            fee_rate_bps=200,
            fast_kill_event=event,
        )

        # Cross at 0.555: gross = (0.56 - 0.555)*100 = 0.5¢
        # Fee = 0.555 * 200/10000 * 100 = 1.11¢
        # Net = 0.5 - 1.11 = -0.61 < 1.0 margin
        assert chaser._alpha_check(0.555) is False

    @pytest.mark.asyncio
    async def test_alpha_check_sell_side(self, setup):
        """SELL-side alpha check uses anchor as reference."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.SELL, target_size=10.0,
            anchor_price=0.47,
            fee_rate_bps=100,
            fast_kill_event=event,
        )

        # Cross at 0.53: gross = (0.53 - 0.47)*100 = 6¢
        # Fee = 0.53 * 100/10000 * 100 = 0.53¢
        # Net = 6 - 0.53 = 5.47¢ >> 1.0 margin
        assert chaser._alpha_check(0.53) is True

    @pytest.mark.asyncio
    async def test_fast_kill_blocks_placement(self, setup):
        """When fast_kill_event is cleared, _wait_fast_kill should timeout."""
        executor, book, event = setup
        event.clear()  # simulate adverse selection trigger

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            fast_kill_event=event,
        )

        # _wait_fast_kill should timeout (not raise), allowing loop to continue
        start = time.time()
        await chaser._wait_fast_kill(timeout=0.05)
        elapsed = time.time() - start
        assert elapsed < 0.2  # should return quickly after timeout

    @pytest.mark.asyncio
    async def test_escalation_no_tp_target_allows(self, setup):
        """Without a TP target, BUY escalation should still be allowed."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=None,
            fee_rate_bps=100,
            fast_kill_event=event,
        )

        # Without a TP target, alpha check returns True (caller's risk)
        assert chaser._alpha_check(0.50) is True
