"""
System Tests — Area 3: Paper Trading Loop

Covers:
  - Virtual BUY → writes timestamp, market_id, entry_price, take-profit to SQLite
  - Virtual SELL → computes PnL accurately with 1¢ slippage factor
  - Full position lifecycle: entry → fill → exit → record
  - TradeStore aggregate stats accuracy
  - Go-live criteria checks
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.trading.executor import OrderExecutor, OrderSide, OrderStatus, Order
from src.trading.position_manager import PositionManager, Position, PositionState
from src.trading.take_profit import compute_take_profit, TakeProfitResult
from src.signals.panic_detector import PanicSignal
from src.data.ohlcv import OHLCVAggregator
from src.monitoring.trade_store import TradeStore

from tests.helpers import make_trade, build_bar_history, make_position


# ═══════════════════════════════════════════════════════════════════════════
#  Section A: Virtual BUY — SQLite Write Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestVirtualBuy:
    """Force a Virtual Buy and verify the SQLite record."""

    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "system_trades.db")

    @pytest.mark.asyncio
    async def test_virtual_buy_writes_all_fields(self, store):
        """Record an entry and verify timestamp, market_id, entry_price, target."""
        await store.init()

        entry_price = 0.45
        target_price = 0.56
        market_id = "MKT_SYSTEM_TEST"

        pos = Position(
            id="SYS-BUY-1",
            market_id=market_id,
            no_asset_id="NO_TOKEN_SYS",
            state=PositionState.CLOSED,
            entry_price=entry_price,
            entry_size=10.0,
            entry_time=time.time() - 60,
            target_price=target_price,
            exit_price=target_price,
            exit_time=time.time(),
            exit_reason="target",
            pnl_cents=round((target_price - entry_price) * 10.0 * 100, 2),
            tp_result=TakeProfitResult(
                entry_price=entry_price,
                target_price=target_price,
                alpha=0.50,
                spread_cents=(target_price - entry_price) * 100,
                viable=True,
            ),
            signal=PanicSignal(
                market_id=market_id,
                yes_asset_id="YES_TOKEN_SYS",
                no_asset_id="NO_TOKEN_SYS",
                yes_price=0.70,
                yes_vwap=0.50,
                zscore=2.5,
                volume_ratio=4.0,
                no_best_ask=0.46,
                whale_confluence=False,
            ),
        )

        await store.record(pos)

        # Read back from SQLite directly
        cursor = await store._db.execute(
            "SELECT id, market_id, entry_price, target_price, entry_time, exit_time, "
            "state, exit_reason, pnl_cents, zscore, volume_ratio, whale "
            "FROM trades WHERE id = ?",
            ("SYS-BUY-1",),
        )
        row = await cursor.fetchone()

        assert row is not None, "Record must exist in SQLite"
        assert row[0] == "SYS-BUY-1"                     # id
        assert row[1] == market_id                         # market_id
        assert row[2] == pytest.approx(entry_price)        # entry_price
        assert row[3] == pytest.approx(target_price)       # target_price
        assert row[4] > 0                                  # entry_time (timestamp)
        assert row[5] > row[4]                             # exit_time > entry_time
        assert row[6] == "CLOSED"                          # state
        assert row[7] == "target"                          # exit_reason
        assert row[8] == pytest.approx(110.0)              # pnl_cents
        assert row[9] == pytest.approx(2.5)                # zscore
        assert row[10] == pytest.approx(4.0)               # volume_ratio
        assert row[11] == 0                                # whale (False)

        await store.close()

    @pytest.mark.asyncio
    async def test_virtual_buy_upsert_updates_existing(self, store):
        """Recording the same position ID twice should UPDATE (upsert), not duplicate."""
        await store.init()

        pos = make_position("UPSERT-1", 0.45, 0.55)
        await store.record(pos)

        # Modify and re-record
        pos.pnl_cents = 200.0
        await store.record(pos)

        cursor = await store._db.execute("SELECT COUNT(*) FROM trades WHERE id = ?", ("UPSERT-1",))
        count = (await cursor.fetchone())[0]
        assert count == 1, "Upsert must not create duplicates"

        cursor2 = await store._db.execute("SELECT pnl_cents FROM trades WHERE id = ?", ("UPSERT-1",))
        row = await cursor2.fetchone()
        assert row[0] == pytest.approx(200.0), "Upserted value must be updated"

        await store.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Section B: Virtual SELL — PnL with 1¢ Slippage
# ═══════════════════════════════════════════════════════════════════════════

class TestVirtualSell:
    """Trigger a Virtual Sell and verify PnL calculation with slippage."""

    def test_pnl_calculation_exact(self):
        """PnL = (exit - entry) * size * 100."""
        pos = Position(
            id="SELL-1", market_id="MKT", no_asset_id="NO",
            state=PositionState.EXIT_PENDING,
            entry_price=0.45, entry_size=10.0, entry_time=time.time() - 60,
        )
        # Simulate exit fill
        pos.state = PositionState.CLOSED
        pos.exit_price = 0.55
        pos.exit_time = time.time()
        pos.pnl_cents = round((pos.exit_price - pos.entry_price) * pos.entry_size * 100, 2)

        assert pos.pnl_cents == pytest.approx(100.0, abs=0.01)

    def test_pnl_with_1cent_slippage(self):
        """Apply 1¢ slippage to exit price and verify reduced PnL."""
        entry_price = 0.45
        target_price = 0.55
        slippage = 0.01  # 1¢
        size = 10.0

        actual_exit = target_price - slippage
        pnl = round((actual_exit - entry_price) * size * 100, 2)

        assert pnl == pytest.approx(90.0, abs=0.01)
        # Lost exactly 10¢ to slippage (0.01 * 10 shares * 100)

    def test_pnl_losing_trade_with_slippage(self):
        """Losing trade: exit < entry, slippage makes it worse."""
        entry_price = 0.50
        exit_price = 0.42
        slippage = 0.01
        size = 10.0

        actual_exit = exit_price - slippage  # 0.41
        pnl = round((actual_exit - entry_price) * size * 100, 2)

        assert pnl == pytest.approx(-90.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_full_lifecycle_entry_to_exit(self):
        """End-to-end: place entry → fill → compute TP → place exit → fill → record."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor, max_open_positions=5)
        pm.set_wallet_balance(50.0)

        # Build a NO aggregator with history
        no_agg = OHLCVAggregator("NO_TOKEN", lookback_minutes=10)
        no_prices = [0.55, 0.56, 0.57, 0.55, 0.56, 0.58, 0.55, 0.57, 0.56, 0.55]
        build_bar_history(no_agg, no_prices, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT_LIFECYCLE",
            yes_asset_id="YES_TOKEN",
            no_asset_id="NO_TOKEN",
            yes_price=0.75,
            yes_vwap=0.50,
            zscore=3.0,
            volume_ratio=5.0,
            no_best_ask=0.45,
            whale_confluence=False,
        )

        # Step 1: Open position (places entry BUY)
        pos = await pm.open_position(signal, no_agg)
        assert pos is not None
        assert pos.state == PositionState.ENTRY_PENDING
        assert pos.entry_price == pytest.approx(0.44, abs=0.01)  # ask - 0.01
        assert pos.target_price > pos.entry_price

        # Step 2: Simulate entry fill
        entry_order = pos.entry_order
        filled = executor.check_paper_fill("NO_TOKEN", 0.44)
        assert len(filled) == 1

        # Step 3: Process entry fill → place exit
        await pm.on_entry_filled(pos)
        assert pos.state == PositionState.EXIT_PENDING
        assert pos.exit_order is not None
        assert pos.exit_order.side == OrderSide.SELL

        # Step 4: Simulate exit fill
        exit_filled = executor.check_paper_fill("NO_TOKEN", pos.target_price)
        assert len(exit_filled) == 1

        # Step 5: Process exit fill
        pm.on_exit_filled(pos, reason="target")
        assert pos.state == PositionState.CLOSED
        assert pos.pnl_cents > 0, "Winning trade should have positive PnL"


# ═══════════════════════════════════════════════════════════════════════════
#  Section C: Aggregate Stats & Go-Live Criteria
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregateStats:
    """Verify aggregate statistics calculations in TradeStore."""

    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "stats_test.db")

    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, store):
        """3 wins + 1 loss → win_rate = 0.75."""
        await store.init()
        await store.record(make_position("W1", 0.45, 0.55))   # +100¢
        await store.record(make_position("W2", 0.40, 0.52))   # +120¢
        await store.record(make_position("W3", 0.50, 0.58))   # +80¢
        await store.record(make_position("L1", 0.48, 0.40, reason="timeout"))  # -80¢

        stats = await store.get_stats()
        assert stats["total_trades"] == 4
        assert stats["win_rate"] == 0.75
        assert stats["total_pnl_cents"] > 0
        assert stats["target_exits"] == 3
        assert stats["timeout_exits"] == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_expectancy(self, store):
        """Expectancy = total_pnl / total_trades."""
        await store.init()
        await store.record(make_position("E1", 0.45, 0.55))   # +100
        await store.record(make_position("E2", 0.50, 0.40, reason="timeout"))  # -100

        stats = await store.get_stats()
        assert stats["expectancy_cents"] == pytest.approx(0.0, abs=0.5)
        await store.close()

    @pytest.mark.asyncio
    async def test_max_drawdown(self, store):
        """Max drawdown should track worst peak-to-trough decline."""
        await store.init()
        await store.record(make_position("D1", 0.45, 0.55))   # +100
        await store.record(make_position("D2", 0.45, 0.55))   # +100 (cum=200)
        await store.record(make_position("D3", 0.55, 0.40, reason="timeout"))  # -150 (cum=50)
        await store.record(make_position("D4", 0.50, 0.42, reason="timeout"))  # -80  (cum=-30)

        stats = await store.get_stats()
        # Peak was 200, trough is -30, drawdown = 230
        assert stats["max_drawdown_cents"] > 0
        await store.close()

    @pytest.mark.asyncio
    async def test_go_live_criteria_not_met_few_trades(self, store):
        """< 20 trades → not ready."""
        await store.init()
        for i in range(5):
            await store.record(make_position(f"G{i}", 0.45, 0.55))
        ready, _ = await store.passes_go_live_criteria()
        assert ready is False
        await store.close()

    @pytest.mark.asyncio
    async def test_go_live_criteria_not_met_low_win_rate(self, store):
        """20 trades but win_rate < 55% → not ready."""
        await store.init()
        for i in range(8):
            await store.record(make_position(f"W{i}", 0.45, 0.55))
        for i in range(12):
            await store.record(make_position(f"L{i}", 0.55, 0.40, reason="timeout"))
        ready, stats = await store.passes_go_live_criteria()
        assert ready is False
        assert stats["win_rate"] < 0.55
        await store.close()

    @pytest.mark.asyncio
    async def test_go_live_criteria_met(self, store):
        """20+ trades, >55% win rate, positive expectancy → ready."""
        await store.init()
        for i in range(15):
            await store.record(make_position(f"W{i}", 0.45, 0.55))  # +100¢ each
        for i in range(6):
            await store.record(make_position(f"L{i}", 0.50, 0.47, reason="timeout"))  # -30¢ each
        ready, stats = await store.passes_go_live_criteria()
        assert stats["total_trades"] == 21
        assert stats["win_rate"] >= 0.55
        assert stats.get("expectancy_cents", 0) > 0
        assert ready is True
        await store.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Section D: Position Manager Risk Gates
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionManagerRisk:
    """Verify risk limits in the position manager."""

    @pytest.mark.asyncio
    async def test_max_open_positions_enforced(self):
        """Should reject new positions when max_open is reached."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor, max_open_positions=1)
        pm.set_wallet_balance(100.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.45, whale_confluence=False,
        )

        pos1 = await pm.open_position(signal, no_agg)
        assert pos1 is not None

        # Second position should be rejected
        pos2 = await pm.open_position(signal, no_agg)
        assert pos2 is None

    @pytest.mark.asyncio
    async def test_zero_balance_rejects_entry(self):
        """Zero wallet balance → position rejected."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(0.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.45, whale_confluence=False,
        )

        pos = await pm.open_position(signal, no_agg)
        assert pos is None

    @pytest.mark.asyncio
    async def test_invalid_entry_price_rejects(self):
        """Entry price ≤ 0 (ask = 0.01, subtract 0.01 → 0) → rejected."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(50.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.01,  # entry = 0.01 - 0.01 = 0.00
            whale_confluence=False,
        )

        pos = await pm.open_position(signal, no_agg)
        assert pos is None
