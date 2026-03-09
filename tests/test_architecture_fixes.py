"""
Tests for architecture-level fixes:
  1. post_only flag wired through to CLOB API
  2. filled_size tracking on Position → exit sizing
  3. Fee model validation (local formula vs REST endpoint)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import (
    Position,
    PositionManager,
    PositionState,
)
from src.trading.chaser import ChaserState, OrderChaser
from src.data.orderbook import OrderbookTracker
from src.trading.fee_cache import FeeCache, validate_fee_model


# ═══════════════════════════════════════════════════════════════════════════
#  Fix 1: post_only flag reaches the CLOB wire
# ═══════════════════════════════════════════════════════════════════════════


class TestPostOnlyWiring:
    """Verify that post_only is forwarded through create_order → post_order
    rather than the old create_and_post_order path that silently dropped it."""

    @pytest.mark.asyncio
    async def test_live_order_calls_post_order_with_post_only(self):
        """In live mode, placing with post_only=True should call
        clob.create_order then clob.post_order(signed, GTC, True)."""
        executor = OrderExecutor(paper_mode=False)

        mock_clob = MagicMock()
        mock_signed_order = MagicMock(name="signed_order")
        mock_clob.create_order.return_value = mock_signed_order
        mock_clob.post_order.return_value = {"orderID": "LIVE-123", "status": "live"}
        executor._clob_client = mock_clob

        # Patch asyncio.to_thread to run synchronously
        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with patch("src.trading.executor.asyncio.to_thread", side_effect=fake_to_thread):
            order = await executor.place_limit_order(
                market_id="MKT",
                asset_id="TOKEN_A",
                side=OrderSide.BUY,
                price=0.45,
                size=10.0,
                post_only=True,
            )

        # Verify two-step path was used, NOT create_and_post_order
        mock_clob.create_and_post_order.assert_not_called()
        mock_clob.create_order.assert_called_once()
        mock_clob.post_order.assert_called_once()

        # Verify post_only=True was the 3rd positional arg
        call_args = mock_clob.post_order.call_args
        assert call_args[0][0] is mock_signed_order  # signed order
        assert call_args[0][2] is True  # post_only flag

    @pytest.mark.asyncio
    async def test_live_order_post_only_false(self):
        """post_only=False should also use the two-step path
        (consistency — always use create_order + post_order)."""
        executor = OrderExecutor(paper_mode=False)

        mock_clob = MagicMock()
        mock_signed_order = MagicMock()
        mock_clob.create_order.return_value = mock_signed_order
        mock_clob.post_order.return_value = {"orderID": "LIVE-456"}

        executor._clob_client = mock_clob

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with patch("src.trading.executor.asyncio.to_thread", side_effect=fake_to_thread):
            order = await executor.place_limit_order(
                market_id="MKT",
                asset_id="TOKEN_A",
                side=OrderSide.BUY,
                price=0.45,
                size=10.0,
                post_only=False,
            )

        mock_clob.create_and_post_order.assert_not_called()
        call_args = mock_clob.post_order.call_args
        assert call_args[0][2] is False  # post_only flag

    @pytest.mark.asyncio
    async def test_paper_mode_unaffected(self):
        """Paper mode should not call CLOB at all."""
        executor = OrderExecutor(paper_mode=True)
        order = await executor.place_limit_order(
            market_id="MKT",
            asset_id="TOKEN_A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            post_only=True,
        )
        assert order.status == OrderStatus.LIVE
        assert order.post_only is True


# ═══════════════════════════════════════════════════════════════════════════
#  Fix 2: filled_size tracking — Position.effective_size
# ═══════════════════════════════════════════════════════════════════════════


class TestFilledSizeTracking:
    """Verify that Position.effective_size and filled_size correctly
    govern exit order sizing and PnL computation."""

    def test_effective_size_defaults_to_entry_size(self):
        """When filled_size is not set (0), effective_size = entry_size."""
        pos = Position(id="P1", market_id="M", no_asset_id="N")
        pos.entry_size = 10.0
        pos.filled_size = 0.0
        assert pos.effective_size == 10.0

    def test_effective_size_uses_filled_when_set(self):
        """When filled_size > 0, effective_size = filled_size."""
        pos = Position(id="P2", market_id="M", no_asset_id="N")
        pos.entry_size = 10.0
        pos.filled_size = 7.5
        assert pos.effective_size == 7.5

    def test_effective_size_full_fill(self):
        """When filled_size == entry_size, effective_size matches."""
        pos = Position(id="P3", market_id="M", no_asset_id="N")
        pos.entry_size = 10.0
        pos.filled_size = 10.0
        assert pos.effective_size == 10.0

    @pytest.mark.asyncio
    async def test_exit_order_sized_to_effective_size(self):
        """on_entry_filled should place exit order sized to effective_size
        when filled_size is set (partial fill scenario)."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)

        pos = Position(
            id="POS_PARTIAL",
            market_id="MKT_1",
            no_asset_id="NO_TOKEN",
            state=PositionState.ENTRY_PENDING,
            entry_size=10.0,
            filled_size=7.0,  # partial fill from chaser
        )
        # Create a mock entry order
        entry_order = Order(
            order_id="ENTRY-1",
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.FILLED,
            filled_avg_price=0.45,
            filled_size=7.0,
        )
        pos.entry_order = entry_order
        pos.target_price = 0.55
        pm._positions[pos.id] = pos

        await pm.on_entry_filled(pos)

        assert pos.state == PositionState.EXIT_PENDING
        assert pos.exit_order is not None
        # Exit order should be sized to 7.0 (partial fill), not 10.0
        assert pos.exit_order.size == 7.0

    @pytest.mark.asyncio
    async def test_exit_order_sized_to_entry_when_no_filled_size(self):
        """When filled_size == 0, on_entry_filled infers from the order
        and falls back to entry_size."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)

        pos = Position(
            id="POS_FULL",
            market_id="MKT_1",
            no_asset_id="NO_TOKEN",
            state=PositionState.ENTRY_PENDING,
            entry_size=10.0,
            filled_size=0.0,
        )
        entry_order = Order(
            order_id="ENTRY-2",
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.FILLED,
            filled_avg_price=0.45,
            filled_size=10.0,
        )
        pos.entry_order = entry_order
        pos.target_price = 0.55
        pm._positions[pos.id] = pos

        await pm.on_entry_filled(pos)

        # filled_size should have been inferred from the order
        assert pos.filled_size == 10.0
        assert pos.exit_order.size == 10.0

    @pytest.mark.asyncio
    async def test_pnl_uses_effective_size(self):
        """PnL computation in on_exit_filled should use effective_size."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)

        pos = Position(
            id="POS_PNL",
            market_id="MKT_1",
            no_asset_id="NO_TOKEN",
            state=PositionState.EXIT_PENDING,
            entry_price=0.45,
            entry_size=10.0,
            filled_size=6.0,
            fee_enabled=False,  # simplify PnL: no fee
        )
        exit_order = Order(
            order_id="EXIT-1",
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.SELL,
            price=0.55,
            size=6.0,
            status=OrderStatus.FILLED,
            filled_avg_price=0.55,
            filled_size=6.0,
        )
        pos.exit_order = exit_order
        pm._positions[pos.id] = pos

        pm.on_exit_filled(pos, reason="target")

        # PnL = (0.55 - 0.45) × 6.0 × 100 = 60.0¢ (no fees)
        assert pos.pnl_cents == pytest.approx(60.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_timeout_exit_uses_effective_size(self):
        """check_timeouts should size the market-sell to effective_size."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)

        pos = Position(
            id="POS_TMO",
            market_id="MKT_1",
            no_asset_id="NO_TOKEN",
            state=PositionState.EXIT_PENDING,
            entry_price=0.45,
            entry_size=10.0,
            filled_size=5.0,
            entry_time=time.time() - 100_000,  # long expired
            fee_enabled=False,
        )
        exit_order = Order(
            order_id="EXIT-REST",
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.SELL,
            price=0.55,
            size=10.0,
            status=OrderStatus.LIVE,
        )
        pos.exit_order = exit_order
        pm._positions[pos.id] = pos

        await pm.check_timeouts()

        # After timeout, the new exit order should be sized to 5.0
        assert pos.exit_order.size == 5.0
        assert pos.state == PositionState.CLOSED  # paper mode auto-closes

    @pytest.mark.asyncio
    async def test_force_stop_loss_uses_effective_size(self):
        """force_stop_loss should size the market-sell to effective_size."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)

        pos = Position(
            id="POS_SL",
            market_id="MKT_1",
            no_asset_id="NO_TOKEN",
            state=PositionState.EXIT_PENDING,
            entry_price=0.45,
            entry_size=10.0,
            filled_size=8.0,
            fee_enabled=False,
        )
        exit_order = Order(
            order_id="EXIT-SL",
            market_id="MKT_1",
            asset_id="NO_TOKEN",
            side=OrderSide.SELL,
            price=0.55,
            size=10.0,
            status=OrderStatus.LIVE,
        )
        pos.exit_order = exit_order
        pm._positions[pos.id] = pos

        await pm.force_stop_loss(pos)

        # Stop-loss exit should be sized to 8.0
        assert pos.exit_order.size == 8.0
        assert pos.state == PositionState.CLOSED


# ═══════════════════════════════════════════════════════════════════════════
#  Fix 3: Fee model validation
# ═══════════════════════════════════════════════════════════════════════════


class TestFeeModelValidation:
    """Test the startup fee model validation that compares local formula
    against the CLOB REST endpoint."""

    @pytest.mark.asyncio
    async def test_validate_passes_when_fees_match(self):
        """When CLOB returns the same BPS as our formula, validation passes."""
        # At mid=0.50, our formula: f_max × 4 × 0.5 × 0.5 = 0.0200 = 200 bps
        with patch.object(FeeCache, "get_fee_rate", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = 200
            result = await validate_fee_model(
                ["TOKEN_A"], [0.50], tolerance_bps=5
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_fails_on_divergence(self):
        """When CLOB returns significantly different BPS, validation fails."""
        with patch.object(FeeCache, "get_fee_rate", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = 250  # 50bps divergence from 200
            result = await validate_fee_model(
                ["TOKEN_A"], [0.50], tolerance_bps=5
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_within_tolerance(self):
        """Small divergence within tolerance should pass."""
        with patch.object(FeeCache, "get_fee_rate", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = 202  # 2bps off — within 5bps tolerance
            result = await validate_fee_model(
                ["TOKEN_A"], [0.50], tolerance_bps=5
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_empty_tokens(self):
        """No tokens to validate → True."""
        result = await validate_fee_model([], [])
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_handles_fetch_error(self):
        """If the REST call fails entirely, returns False."""
        with patch.object(
            FeeCache, "get_fee_rate", new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            result = await validate_fee_model(
                ["TOKEN_A"], [0.50], tolerance_bps=5
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_multiple_tokens(self):
        """Multiple tokens: one good, one bad → overall False."""
        call_count = 0

        async def mock_get(self_or_token, token=None):
            nonlocal call_count
            call_count += 1
            return 200 if call_count == 1 else 300

        with patch.object(FeeCache, "get_fee_rate", mock_get):
            result = await validate_fee_model(
                ["TOKEN_A", "TOKEN_B"], [0.50, 0.50], tolerance_bps=5
            )
            assert result is False
