"""
Tests for the active stop-loss engine.
"""

import asyncio
import time

import pytest

from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import Position, PositionManager, PositionState
from src.trading.stop_loss import StopLossMonitor


# ── Fake components ─────────────────────────────────────────────────────────


class FakeAgg:
    """Minimal OHLCVAggregator stub."""

    def __init__(self, price: float = 0.0):
        self.current_price = price


class FakeBookSnap:
    def __init__(self, bid: float = 0.0, ask: float = 0.0):
        self.best_bid = bid
        self.best_ask = ask


class FakeBook:
    """Minimal OrderbookTracker stub."""

    def __init__(self, bid: float = 0.0, ask: float = 0.0):
        self.has_data = bid > 0 or ask > 0
        self._snap = FakeBookSnap(bid, ask)

    def snapshot(self):
        return self._snap


class FakeTradeStore:
    def __init__(self):
        self.recorded: list = []

    async def record(self, pos):
        self.recorded.append(pos)


class FakeTelegram:
    def __init__(self):
        self.exits: list = []

    async def notify_exit(self, *args):
        self.exits.append(args)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestStopLossMonitor:
    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(100.0)

        no_aggs: dict[str, FakeAgg] = {}
        books: dict[str, FakeBook] = {}
        store = FakeTradeStore()
        telegram = FakeTelegram()

        return executor, pm, no_aggs, books, store, telegram

    def _make_exit_pending_pos(self, executor, pm, entry_price=0.45, sl_trigger=8.0):
        """Create a position in EXIT_PENDING state."""
        order = Order(
            order_id="O-1",
            market_id="MKT",
            asset_id="NO_T",
            side=OrderSide.BUY,
            price=entry_price,
            size=10.0,
            status=OrderStatus.FILLED,
            filled_size=10.0,
            filled_avg_price=entry_price,
        )

        exit_order = Order(
            order_id="O-2",
            market_id="MKT",
            asset_id="NO_T",
            side=OrderSide.SELL,
            price=0.55,
            size=10.0,
            status=OrderStatus.LIVE,
        )
        executor._orders[exit_order.order_id] = exit_order

        pos = Position(
            id="POS-SL-1",
            market_id="MKT",
            no_asset_id="NO_T",
            state=PositionState.EXIT_PENDING,
            entry_order=order,
            entry_price=entry_price,
            entry_size=10.0,
            entry_time=time.time() - 60,
            exit_order=exit_order,
            target_price=0.55,
            sl_trigger_cents=sl_trigger,
            fee_enabled=False,
        )
        pm._positions[pos.id] = pos
        return pos

    @pytest.mark.asyncio
    async def test_triggers_stop_when_price_drops(self, setup):
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm, entry_price=0.45, sl_trigger=8.0)

        # Mid-price dropped 9¢ below entry → should trigger (threshold 8¢)
        no_aggs["NO_T"] = FakeAgg(price=0.36)

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
        )

        await monitor._check_once(8.0)

        # Position should be closed via force_stop_loss
        assert len(store.recorded) == 1
        assert len(telegram.exits) == 1

    @pytest.mark.asyncio
    async def test_no_trigger_when_within_threshold(self, setup):
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm, entry_price=0.45, sl_trigger=8.0)

        # Mid-price only 5¢ below → should NOT trigger
        no_aggs["NO_T"] = FakeAgg(price=0.40)

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
        )

        await monitor._check_once(8.0)

        assert len(store.recorded) == 0
        assert len(telegram.exits) == 0

    @pytest.mark.asyncio
    async def test_uses_orderbook_mid_price_over_agg(self, setup):
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm, entry_price=0.45, sl_trigger=5.0)

        # Aggregator says price is fine, but orderbook shows crash
        no_aggs["NO_T"] = FakeAgg(price=0.44)
        books["NO_T"] = FakeBook(bid=0.38, ask=0.40)  # mid = 0.39 → 6¢ loss

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
        )

        await monitor._check_once(5.0)

        assert len(store.recorded) == 1  # triggered because book mid is authoritative

    @pytest.mark.asyncio
    async def test_trailing_stop(self, setup):
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm, entry_price=0.45, sl_trigger=8.0)

        # Trailing offset = 3¢
        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
            trailing_offset_cents=3.0,
        )

        # Price rallies to 0.52 → HWM = 0.52
        no_aggs["NO_T"] = FakeAgg(price=0.52)
        await monitor._check_once(8.0)
        assert len(store.recorded) == 0
        assert monitor._hwm.get("POS-SL-1") == 0.52

        # Price falls to 0.49 → trailing floor = 0.52 - 0.03 = 0.49
        # trail_loss = (0.49 - 0.49) * 100 = 0, entry_loss = (0.45 - 0.49) * 100 = -4 (negative = profit)
        # Neither triggers
        no_aggs["NO_T"] = FakeAgg(price=0.49)
        await monitor._check_once(8.0)
        assert len(store.recorded) == 0

        # Price drops to 0.47 → trail_loss = (0.49 - 0.47) * 100 = 2, entry_loss = (0.45 - 0.47) * 100 = -2
        # max(2, -2) = 2 < 8 → no trigger yet
        no_aggs["NO_T"] = FakeAgg(price=0.47)
        await monitor._check_once(8.0)
        assert len(store.recorded) == 0

    @pytest.mark.asyncio
    async def test_skips_non_exit_pending_positions(self, setup):
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm)
        pos.state = PositionState.ENTRY_PENDING  # not eligible

        no_aggs["NO_T"] = FakeAgg(price=0.30)  # huge loss

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
        )

        await monitor._check_once(8.0)
        assert len(store.recorded) == 0  # ignored

    @pytest.mark.asyncio
    async def test_hwm_cleanup_on_position_close(self, setup):
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm, entry_price=0.45, sl_trigger=1.0)

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
            trailing_offset_cents=0.0,
        )

        no_aggs["NO_T"] = FakeAgg(price=0.50)
        await monitor._check_once(1.0)
        assert "POS-SL-1" in monitor._hwm

        # Simulate position close
        pos.state = PositionState.CLOSED

        # Next check should prune the HWM entry
        no_aggs["NO_T"] = FakeAgg(price=0.50)
        await monitor._check_once(1.0)
        assert "POS-SL-1" not in monitor._hwm
