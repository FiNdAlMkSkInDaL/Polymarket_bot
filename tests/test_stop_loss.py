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

    def __init__(self, bid: float = 0.0, ask: float = 0.0, book_depth_ratio: float = 1.0):
        self.has_data = bid > 0 or ask > 0
        self.best_bid = bid
        self.best_ask = ask
        self._snap = FakeBookSnap(bid, ask)
        self._book_depth_ratio = book_depth_ratio

    @property
    def book_depth_ratio(self) -> float:
        return self._book_depth_ratio

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

    # ── Pillar 11.3: Preemptive Liquidity Drain ───────────────────────────

    @pytest.mark.asyncio
    async def test_preemptive_stop_underwater_hollow_book(self, setup):
        """Hollowed book (ratio=0.05) + underwater → immediate stop."""
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm, entry_price=0.45, sl_trigger=8.0)

        # Mid = 0.43 → 2¢ underwater (pnl < 0)
        # book_depth_ratio = bid_depth / ask_depth.
        # our_support_ratio = bdr.  When bdr → 0, bid support vanishes
        # and our exit liquidity is draining.
        # bdr=0.05 → our_support = 0.05 < 0.10 → trigger.

        books["NO_T"] = FakeBook(bid=0.42, ask=0.44, book_depth_ratio=0.05)
        no_aggs["NO_T"] = FakeAgg(price=0.43)

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
        )

        await monitor._check_once(8.0)

        assert len(store.recorded) == 1
        assert pos.exit_reason == "preemptive_liquidity_drain"

    @pytest.mark.asyncio
    async def test_preemptive_stop_in_profit_no_trigger(self, setup):
        """Hollowed book but position in profit → should NOT trigger."""
        executor, pm, no_aggs, books, store, telegram = setup

        pos = self._make_exit_pending_pos(executor, pm, entry_price=0.45, sl_trigger=8.0)

        # Mid = 0.48 → 3¢ in profit (pnl > 0)
        books["NO_T"] = FakeBook(bid=0.47, ask=0.49, book_depth_ratio=0.1)
        no_aggs["NO_T"] = FakeAgg(price=0.48)

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
        )

        await monitor._check_once(8.0)

        # Not triggered because position is profitable
        assert len(store.recorded) == 0

    # ── Pillar 11.3: Time-Decay Stop Tightening ──────────────────────────

    @pytest.mark.asyncio
    async def test_time_decay_tightens_stop(self, setup):
        """After 20 mins, a vol-stretched stop-loss should decay back
        toward the 1.0× baseline, making it easier to trigger."""
        executor, pm, no_aggs, books, store, telegram = setup

        # Simulate a position with sl_vol_multiplier = 1.5
        # base_sl = 4.0, stretched = 4 * 1.5 = 6.0 (fees disabled)
        pos = self._make_exit_pending_pos(
            executor, pm, entry_price=0.45, sl_trigger=6.0,
        )
        pos.sl_vol_multiplier = 1.5
        # Position opened 20 minutes ago (beyond decay_start=5)
        pos.entry_time = time.time() - 20 * 60

        # Mid = 0.39 → unrealised_loss = 6¢
        # Without decay: 6 < 6 → no trigger (boundary)
        # With decay at 25min:
        #   elapsed=25, start=5, half_life=15
        #   decay_factor = exp(-ln(2)*(25-5)/15) = exp(-ln(2)*4/3) ≈ 0.397
        #   current_mult = 1.0 + (1.5-1.0)*0.397 ≈ 1.199
        #   fee_drag = 4.0*1.5 - 6.0 = 0.0 (fees disabled)
        #   decayed_sl = max(1.0, 4.0 * 1.199 - 0.0) ≈ 4.79
        #   6.0 >= 4.79 → TRIGGER!
        no_aggs["NO_T"] = FakeAgg(price=0.39)

        monitor = StopLossMonitor(
            position_manager=pm,
            no_aggs=no_aggs,
            book_trackers=books,
            trade_store=store,
            telegram=telegram,
        )

        await monitor._check_once(4.0)

        assert len(store.recorded) == 1, "Time-decayed stop should have triggered"
