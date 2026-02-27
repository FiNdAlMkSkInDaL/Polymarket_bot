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
