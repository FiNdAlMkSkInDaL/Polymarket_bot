"""
Tests for the trade store (SQLite persistence and stats).
"""

import asyncio
import time

import pytest

from src.monitoring.trade_store import TradeStore
from src.trading.position_manager import Position, PositionState
from src.trading.take_profit import TakeProfitResult
from src.signals.panic_detector import PanicSignal


def _make_position(
    pos_id: str,
    entry: float,
    exit_p: float,
    size: float = 10.0,
    reason: str = "target",
    whale: bool = False,
) -> Position:
    pnl = round((exit_p - entry) * size * 100, 2)
    now = time.time()
    return Position(
        id=pos_id,
        market_id="MKT_TEST",
        no_asset_id="NO_T",
        state=PositionState.CLOSED,
        entry_price=entry,
        entry_size=size,
        entry_time=now - 120,
        target_price=exit_p,
        exit_price=exit_p,
        exit_time=now,
        exit_reason=reason,
        pnl_cents=pnl,
        tp_result=TakeProfitResult(
            entry_price=entry, target_price=exit_p,
            alpha=0.5, spread_cents=abs(exit_p - entry) * 100, viable=True,
        ),
        signal=PanicSignal(
            market_id="MKT_TEST", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.70, yes_vwap=0.50, zscore=2.5, volume_ratio=4.0,
            no_best_ask=entry + 0.01, whale_confluence=whale,
        ),
    )


def _make_ofi_position(
    pos_id: str,
    entry: float,
    exit_p: float,
    *,
    toxicity_index: float,
    size: float = 10.0,
    entry_fee_bps: int = 0,
    exit_fee_bps: int = 0,
) -> Position:
    pos = _make_position(pos_id, entry, exit_p, size=size)
    pos.signal_type = "ofi_momentum"
    pos.entry_toxicity_index = toxicity_index
    pos.entry_fee_bps = entry_fee_bps
    pos.exit_fee_bps = exit_fee_bps
    pos.drawn_tp = exit_p
    pos.drawn_stop = round(entry * 0.985, 4)
    pos.drawn_time = 240.0
    return pos


class TestTradeStore:
    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "test_trades.db")

    @pytest.mark.asyncio
    async def test_record_and_stats(self, store):
        await store.init()

        # Record 3 winning trades and 1 losing trade
        await store.record(_make_position("P1", 0.45, 0.55))  # +100¢
        await store.record(_make_position("P2", 0.40, 0.52))  # +120¢
        await store.record(_make_position("P3", 0.50, 0.58))  # +80¢
        await store.record(_make_position("P4", 0.48, 0.40, reason="timeout"))  # -80¢

        stats = await store.get_stats()
        assert stats["total_trades"] == 4
        assert stats["win_rate"] == 0.75
        assert stats["total_pnl_cents"] > 0
        assert stats["target_exits"] == 3
        assert stats["timeout_exits"] == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_record_persists_fill_time_toxicity(self, store):
        await store.init()

        pos = _make_position("P_TOX", 0.45, 0.55)
        pos.entry_toxicity_index = 0.87
        pos.exit_toxicity_index = 0.24
        pos.drawn_tp = 0.56
        pos.drawn_stop = 0.441
        pos.drawn_time = 245.0
        pos.signal = None

        await store.record(pos)

        cursor = await store._db.execute(
            "SELECT entry_toxicity_index, exit_toxicity_index, drawn_tp, drawn_stop, drawn_time "
            "FROM trades WHERE id = ?",
            ("P_TOX",),
        )
        row = await cursor.fetchone()

        assert row is not None
        assert row[0] == pytest.approx(0.87)
        assert row[1] == pytest.approx(0.24)
        assert row[2] == pytest.approx(0.56)
        assert row[3] == pytest.approx(0.441)
        assert row[4] == pytest.approx(245.0)

        await store.close()

    @pytest.mark.asyncio
    async def test_go_live_criteria_not_met_few_trades(self, store):
        await store.init()
        await store.record(_make_position("P1", 0.45, 0.55))
        ready, stats = await store.passes_go_live_criteria()
        assert ready is False  # <20 trades
        await store.close()

    @pytest.mark.asyncio
    async def test_empty_stats(self, store):
        await store.init()
        stats = await store.get_stats()
        assert stats["total_trades"] == 0
        await store.close()

    @pytest.mark.asyncio
    async def test_get_ofi_toxicity_pnl_summary_buckets_pnl_and_fee_drag(self, store):
        await store.init()

        low_bucket = _make_ofi_position(
            "OFI-LOW",
            0.50,
            0.55,
            toxicity_index=0.12,
            entry_fee_bps=100,
            exit_fee_bps=0,
        )
        mid_bucket_win = _make_ofi_position(
            "OFI-MID-WIN",
            0.45,
            0.50,
            toxicity_index=0.44,
            entry_fee_bps=100,
            exit_fee_bps=50,
        )
        mid_bucket_loss = _make_ofi_position(
            "OFI-MID-LOSS",
            0.48,
            0.44,
            toxicity_index=0.47,
            entry_fee_bps=0,
            exit_fee_bps=100,
        )
        ignored = _make_position("PANIC-IGNORE", 0.40, 0.42)
        ignored.signal_type = "panic"
        ignored.entry_toxicity_index = 0.92
        ignored.entry_fee_bps = 100
        ignored.exit_fee_bps = 100

        await store.record(low_bucket)
        await store.record(mid_bucket_win)
        await store.record(mid_bucket_loss)
        await store.record(ignored)

        summary = await store.get_ofi_toxicity_pnl_summary(buckets=5)

        assert len(summary) == 2

        low = summary[0]
        assert low["bucket_index"] == 0
        assert low["trade_count"] == 1
        assert low["win_rate"] == pytest.approx(1.0)
        assert low["avg_net_pnl_cents"] == pytest.approx(50.0)
        assert low["total_taker_fee_drag_cents"] == pytest.approx(5.0)

        mid = summary[1]
        assert mid["bucket_index"] == 2
        assert mid["trade_count"] == 2
        assert mid["win_rate"] == pytest.approx(0.5)
        assert mid["avg_net_pnl_cents"] == pytest.approx(5.0)
        assert mid["total_net_pnl_cents"] == pytest.approx(10.0)
        assert mid["total_taker_fee_drag_cents"] == pytest.approx(11.4)

        await store.close()

    @pytest.mark.asyncio
    async def test_get_ofi_toxicity_pnl_summary_empty_when_no_ofi_trades(self, store):
        await store.init()
        await store.record(_make_position("PANIC-1", 0.45, 0.55))

        summary = await store.get_ofi_toxicity_pnl_summary()

        assert summary == []
        await store.close()


# ── State-persistence tests ─────────────────────────────────────────────────

from src.trading.executor import Order, OrderSide, OrderStatus


class TestStatePersistence:
    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "persist_test.db")

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore_orders(self, store):
        await store.init()

        orders = [
            Order(
                order_id="LIVE-1",
                market_id="MKT_A",
                asset_id="NO_A",
                side=OrderSide.BUY,
                price=0.45,
                size=10.0,
                status=OrderStatus.LIVE,
                clob_order_id="CLOB-1",
            ),
            Order(
                order_id="LIVE-2",
                market_id="MKT_B",
                asset_id="NO_B",
                side=OrderSide.SELL,
                price=0.60,
                size=5.0,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_size=2.0,
                filled_avg_price=0.59,
                clob_order_id="CLOB-2",
            ),
        ]

        await store.checkpoint_orders(orders)
        restored = await store.restore_orders()

        assert len(restored) == 2
        by_id = {r["order_id"]: r for r in restored}

        o1 = by_id["LIVE-1"]
        assert o1["market_id"] == "MKT_A"
        assert o1["side"] == "BUY"
        assert o1["price"] == 0.45
        assert o1["clob_order_id"] == "CLOB-1"
        assert o1["status"] == "LIVE"

        o2 = by_id["LIVE-2"]
        assert o2["filled_size"] == 2.0
        assert o2["filled_avg_price"] == 0.59

        await store.close()

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore_positions(self, store):
        await store.init()

        pos = _make_position("POS-1", 0.45, 0.55)
        pos.state = PositionState.EXIT_PENDING  # simulate open position

        await store.checkpoint_positions([pos])
        restored = await store.restore_positions()

        assert len(restored) == 1
        p = restored[0]
        assert p["id"] == "POS-1"
        assert p["market_id"] == "MKT_TEST"
        assert p["state"] == "EXIT_PENDING"
        assert p["entry_price"] == 0.45
        assert p["target_price"] == 0.55
        assert p["drawn_tp"] == 0.0
        assert p["drawn_stop"] == 0.0
        assert p["drawn_time"] == 0.0

        await store.close()

    @pytest.mark.asyncio
    async def test_checkpoint_replaces_previous_data(self, store):
        await store.init()

        orders_v1 = [
            Order(
                order_id="OLD-1", market_id="M", asset_id="A",
                side=OrderSide.BUY, price=0.40, size=5.0,
                status=OrderStatus.LIVE, clob_order_id="C-1",
            ),
        ]
        await store.checkpoint_orders(orders_v1)

        orders_v2 = [
            Order(
                order_id="NEW-1", market_id="M2", asset_id="A2",
                side=OrderSide.SELL, price=0.60, size=3.0,
                status=OrderStatus.LIVE, clob_order_id="C-2",
            ),
        ]
        await store.checkpoint_orders(orders_v2)

        restored = await store.restore_orders()
        assert len(restored) == 1
        assert restored[0]["order_id"] == "NEW-1"

        await store.close()

    @pytest.mark.asyncio
    async def test_clear_live_state(self, store):
        await store.init()

        orders = [
            Order(
                order_id="X-1", market_id="M", asset_id="A",
                side=OrderSide.BUY, price=0.45, size=10.0,
                status=OrderStatus.LIVE, clob_order_id="C-X",
            ),
        ]
        pos = _make_position("POS-X", 0.45, 0.55)
        pos.state = PositionState.ENTRY_PENDING

        await store.checkpoint_orders(orders)
        await store.checkpoint_positions([pos])

        await store.clear_live_state()

        assert len(await store.restore_orders()) == 0
        assert len(await store.restore_positions()) == 0

        await store.close()

    @pytest.mark.asyncio
    async def test_restore_empty_tables(self, store):
        await store.init()
        assert await store.restore_orders() == []
        assert await store.restore_positions() == []
        await store.close()


class TestTradeStorePragmas:
    """Verify WAL-mode concurrency hardening pragmas are applied."""

    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "pragma_test.db")

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, store):
        await store.init()
        async with store._db.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()
            assert row[0].lower() == "wal"
        await store.close()

    @pytest.mark.asyncio
    async def test_busy_timeout_set(self, store):
        await store.init()
        async with store._db.execute("PRAGMA busy_timeout") as cur:
            row = await cur.fetchone()
            assert row[0] == 5000
        await store.close()