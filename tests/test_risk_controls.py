"""Tests for institutional risk controls in position_manager.py."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.ohlcv import OHLCVAggregator
from src.signals.panic_detector import PanicSignal
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import Position, PositionManager, PositionState


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_signal(market_id: str = "MKT_A") -> PanicSignal:
    return PanicSignal(
        market_id=market_id,
        yes_asset_id=f"YES_{market_id}",
        no_asset_id=f"NO_{market_id}",
        yes_price=0.40,
        yes_vwap=0.50,
        zscore=3.0,
        volume_ratio=5.0,
        no_best_ask=0.26,
        whale_confluence=False,
    )


def _make_no_agg(price: float = 0.65) -> OHLCVAggregator:
    agg = OHLCVAggregator("NO_TOKEN")
    agg.rolling_vwap = price
    return agg


# ── Circuit breaker tests ────────────────────────────────────────────────

class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_new_positions(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)
        pm.set_wallet_balance(100)
        pm._circuit_breaker_tripped = True

        result = await pm.open_position(_make_signal(), _make_no_agg())
        assert result is None

    def test_reset_daily_pnl_clears_breaker(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)
        pm._circuit_breaker_tripped = True
        pm.reset_daily_pnl()
        assert not pm.circuit_breaker_active


# ── Per-market & per-event limits ─────────────────────────────────────────

class TestConcentrationLimits:
    @pytest.mark.asyncio
    async def test_per_market_limit(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec, max_open_positions=10)
        pm.set_wallet_balance(1000)

        # Fill up per-market slots (default is 1 per market)
        sig = _make_signal("MKT_A")
        pos1 = await pm.open_position(sig, _make_no_agg(), fee_enabled=False)
        assert pos1 is not None

        # Second position on same market should be blocked
        pos2 = await pm.open_position(sig, _make_no_agg(), fee_enabled=False)
        assert pos2 is None

    @pytest.mark.asyncio
    async def test_per_event_limit(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec, max_open_positions=10)
        pm.set_wallet_balance(1000)

        # Two different markets, same event
        sig1 = _make_signal("MKT_A")
        sig2 = _make_signal("MKT_B")
        pos1 = await pm.open_position(sig1, _make_no_agg(), event_id="EVT_X", fee_enabled=False)
        assert pos1 is not None
        pos2 = await pm.open_position(sig2, _make_no_agg(), event_id="EVT_X", fee_enabled=False)
        assert pos2 is not None

        # Third should be blocked (default per-event limit is 2)
        sig3 = _make_signal("MKT_C")
        pos3 = await pm.open_position(sig3, _make_no_agg(), event_id="EVT_X", fee_enabled=False)
        assert pos3 is None


# ── Daily loss / drawdown tracking ────────────────────────────────────────

class TestPnLTracking:
    def test_on_exit_filled_tracks_pnl(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)

        pos = Position(
            id="POS-1",
            market_id="MKT_A",
            no_asset_id="NO_MKT_A",
            state=PositionState.EXIT_PENDING,
            entry_price=0.50,
            entry_size=10.0,
            target_price=0.55,
            fee_enabled=False,
        )
        pos.exit_order = MagicMock(filled_avg_price=0.55)
        pm._positions["POS-1"] = pos

        pm.on_exit_filled(pos, reason="target")
        assert pos.pnl_cents == 50.0  # (0.55 - 0.50) * 10 * 100
        assert pm._daily_pnl_cents == 50.0
        assert pm._cumulative_pnl_cents == 50.0

    def test_loss_accumulates(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)

        # Loss trade
        pos = Position(
            id="POS-2",
            market_id="MKT_A",
            no_asset_id="NO_MKT_A",
            state=PositionState.EXIT_PENDING,
            entry_price=0.50,
            entry_size=10.0,
            fee_enabled=False,
        )
        pos.exit_order = MagicMock(filled_avg_price=0.45)
        pm._positions["POS-2"] = pos

        pm.on_exit_filled(pos, reason="stop_loss")
        assert pos.pnl_cents == -50.0
        assert pm._daily_pnl_cents == -50.0


# ── Cleanup ───────────────────────────────────────────────────────────────

class TestCleanup:
    def test_cleanup_closed_keeps_recent(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)

        # Add 60 closed positions
        for i in range(60):
            pos = Position(
                id=f"POS-{i}",
                market_id="MKT_A",
                no_asset_id="NO_MKT_A",
                state=PositionState.CLOSED,
                exit_time=time.time() - i,  # most recent first
            )
            pm._positions[pos.id] = pos

        removed = pm.cleanup_closed()
        assert len(removed) == 10  # 60 - 50 kept
        # Most recent 50 should still be in positions
        assert len([p for p in pm._positions.values() if p.state == PositionState.CLOSED]) == 50

    def test_get_open_market_ids(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)

        pm._positions["P1"] = Position(
            id="P1", market_id="MKT_A", no_asset_id="NO",
            state=PositionState.EXIT_PENDING,
        )
        pm._positions["P2"] = Position(
            id="P2", market_id="MKT_B", no_asset_id="NO",
            state=PositionState.CLOSED,
        )

        open_ids = pm.get_open_market_ids()
        assert "MKT_A" in open_ids
        assert "MKT_B" not in open_ids


# ── Stop-loss ─────────────────────────────────────────────────────────────

class TestStopLoss:
    @pytest.mark.asyncio
    async def test_force_stop_loss(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)

        pos = Position(
            id="POS-SL",
            market_id="MKT_A",
            no_asset_id="NO_MKT_A",
            state=PositionState.EXIT_PENDING,
            entry_price=0.50,
            entry_size=10.0,
            entry_time=time.time(),
        )
        pm._positions["POS-SL"] = pos

        await pm.force_stop_loss(pos)
        assert pos.state == PositionState.CLOSED
        assert pos.exit_reason == "stop_loss"

    @pytest.mark.asyncio
    async def test_force_stop_loss_noop_if_not_exit_pending(self):
        exec = OrderExecutor(paper_mode=True)
        pm = PositionManager(exec)

        pos = Position(
            id="POS-NOP",
            market_id="MKT_A",
            no_asset_id="NO_MKT_A",
            state=PositionState.ENTRY_PENDING,
        )
        pm._positions["POS-NOP"] = pos

        await pm.force_stop_loss(pos)
        assert pos.state == PositionState.ENTRY_PENDING  # unchanged
