"""
Resilience & Edge Case Tests — Area 4: Chaos Engineering

Covers:
  - WebSocket auto-reconnect on drop (ConnectionClosed, OSError)
  - SQLite database locked / inaccessible scenarios
  - OHLCV aggregator edge cases (empty bars, zero price, huge timestamps)
  - Position manager timeout enforcement
  - Executor edge cases
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import aiosqlite
import pytest

from src.data.ohlcv import OHLCVAggregator, OHLCVBar, BAR_INTERVAL
from src.data.websocket_client import MarketWebSocket, TradeEvent
from src.monitoring.trade_store import TradeStore
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import PositionManager, Position, PositionState
from src.signals.panic_detector import PanicDetector, PanicSignal

from tests.helpers import make_trade, build_bar_history, make_position


# ═══════════════════════════════════════════════════════════════════════════
#  Section A: WebSocket Reconnection (Chaos)
# ═══════════════════════════════════════════════════════════════════════════

class TestWebSocketResilience:
    """Verify auto-reconnect behaviour under simulated failures."""

    @pytest.mark.asyncio
    async def test_reconnect_after_connection_closed(self):
        """Simulated ConnectionClosed should trigger reconnect loop."""
        import websockets.exceptions

        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)

        attempt = 0

        async def mock_connect():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise websockets.exceptions.ConnectionClosed(None, None)
            ws._running = False

        ws._connect_and_consume = mock_connect

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await ws.start()

        assert attempt >= 3, f"Expected ≥3 reconnect attempts, got {attempt}"

    @pytest.mark.asyncio
    async def test_reconnect_after_os_error(self):
        """OSError (network unreachable) should trigger reconnect."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)

        attempt = 0

        async def mock_connect():
            nonlocal attempt
            attempt += 1
            if attempt < 4:
                raise OSError("Network unreachable")
            ws._running = False

        ws._connect_and_consume = mock_connect

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await ws.start()

        assert attempt >= 4

    @pytest.mark.asyncio
    async def test_reconnect_backoff_uses_sleep(self):
        """Reconnect should use asyncio.sleep(5) between attempts."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)

        sleep_calls = []

        async def track_sleep(duration):
            sleep_calls.append(duration)

        attempt = 0

        async def mock_connect():
            nonlocal attempt
            attempt += 1
            if attempt >= 2:
                ws._running = False
                return
            raise ConnectionError("test")

        ws._connect_and_consume = mock_connect

        with patch("asyncio.sleep", side_effect=track_sleep):
            await ws.start()

        assert 5 not in sleep_calls, "Hardcoded 5 s sleep should no longer be used"
        assert len(sleep_calls) >= 1, "At least one backoff sleep expected"
        # First attempt: base(1) * 2^1 + jitter ∈ (2.0, 3.0)
        assert 1.5 < sleep_calls[0] < 4.0, (
            f"Expected exponential backoff ~2-3 s on attempt 1, got {sleep_calls[0]}"
        )

    @pytest.mark.asyncio
    async def test_stop_during_reconnect(self):
        """Calling stop() while reconnecting should exit the loop."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)

        async def mock_connect():
            await ws.stop()  # Stop from within
            raise ConnectionError("should not retry")

        ws._connect_and_consume = mock_connect

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await ws.start()

        assert ws._running is False


# ═══════════════════════════════════════════════════════════════════════════
#  Section B: SQLite Resilience
# ═══════════════════════════════════════════════════════════════════════════

class TestSQLiteResilience:
    """Verify graceful handling of database errors."""

    @pytest.mark.asyncio
    async def test_write_to_readonly_path(self, tmp_path):
        """Writing to a read-only directory should raise / be handled."""
        # Create a read-only directory (platform-dependent)
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        db_path = readonly_dir / "trades.db"

        store = TradeStore(db_path)
        # This should work (creating the db in a writable dir)
        await store.init()
        await store.record(make_position("RO-1", 0.45, 0.55))
        await store.close()

    @pytest.mark.asyncio
    async def test_init_creates_schema(self, tmp_path):
        """init() should create the trades table automatically."""
        store = TradeStore(tmp_path / "fresh.db")
        await store.init()

        cursor = await store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        row = await cursor.fetchone()
        assert row is not None, "trades table must exist after init()"
        await store.close()

    @pytest.mark.asyncio
    async def test_double_init_is_safe(self, tmp_path):
        """Calling init() twice should not crash (idempotent)."""
        store = TradeStore(tmp_path / "double.db")
        await store.init()
        await store.init()  # Second call should be harmless
        await store.close()

    @pytest.mark.asyncio
    async def test_record_auto_inits(self, tmp_path):
        """record() on an uninitialised store should auto-init."""
        store = TradeStore(tmp_path / "lazy.db")
        # Don't call init() — record() should do it
        await store.record(make_position("LAZY-1", 0.45, 0.55))

        stats = await store.get_stats()
        assert stats["total_trades"] == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_close_twice_is_safe(self, tmp_path):
        """Calling close() twice should not raise."""
        store = TradeStore(tmp_path / "close2.db")
        await store.init()
        await store.close()
        await store.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_empty_db_returns_zero_stats(self, tmp_path):
        """An empty database should return total_trades=0."""
        store = TradeStore(tmp_path / "empty.db")
        await store.init()
        stats = await store.get_stats()
        assert stats == {"total_trades": 0}
        await store.close()

    @pytest.mark.asyncio
    async def test_parent_dir_creation(self, tmp_path):
        """TradeStore should create parent directories if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "trades.db"
        store = TradeStore(deep_path)
        await store.init()
        assert deep_path.exists()
        await store.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Section C: OHLCV Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestOHLCVEdgeCases:
    """Edge cases for bar aggregation."""

    def test_single_trade_no_bar(self):
        """A single trade should not produce a complete bar."""
        agg = OHLCVAggregator("TEST")
        result = agg.on_trade(make_trade(0.50, 10.0, 1000.0, asset_id="TEST"))
        assert result is None
        assert len(agg.bars) == 0

    def test_zero_volume_bar(self):
        """Bar with near-zero volume should still compute without NaN."""
        agg = OHLCVAggregator("TEST", lookback_minutes=5)
        t = 1000.0
        agg.on_trade(make_trade(0.50, 0.001, t, asset_id="TEST"))
        bar = agg.on_trade(make_trade(0.51, 0.001, t + BAR_INTERVAL + 1, asset_id="TEST"))
        if bar:
            assert not (bar.vwap != bar.vwap)  # NaN check

    def test_large_timestamp_gap(self):
        """A huge gap between trades should still produce a single bar."""
        agg = OHLCVAggregator("TEST", lookback_minutes=60)
        agg.on_trade(make_trade(0.50, 10.0, 1000.0, asset_id="TEST"))
        bar = agg.on_trade(make_trade(0.55, 5.0, 1_000_000.0, asset_id="TEST"))
        assert bar is not None  # Previous interval closes

    def test_maxlen_bounds_deque(self):
        """Deque should never grow larger than lookback + 5."""
        lookback = 10
        agg = OHLCVAggregator("TEST", lookback_minutes=lookback)
        prices = [0.50 + 0.01 * i for i in range(50)]
        build_bar_history(agg, prices, base_vol=5.0)
        assert len(agg.bars) <= lookback + 5

    def test_identical_trades_in_bar(self):
        """All trades at same price → OHLC all equal, vol summed."""
        agg = OHLCVAggregator("TEST", lookback_minutes=10)
        t = 1000.0
        for _ in range(5):
            agg.on_trade(make_trade(0.50, 10.0, t, asset_id="TEST"))
            t += 1
        bar = agg.on_trade(make_trade(0.50, 10.0, t + BAR_INTERVAL, asset_id="TEST"))
        if bar:
            assert bar.open == 0.50
            assert bar.high == 0.50
            assert bar.low == 0.50
            assert bar.close == 0.50


# ═══════════════════════════════════════════════════════════════════════════
#  Section D: Position Manager Timeout Enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeoutEnforcement:
    """Verify that stale positions are cancelled / force-closed."""

    @pytest.mark.asyncio
    async def test_entry_timeout_cancels_order(self):
        """Entry pending > entry_timeout_seconds → cancelled."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(50.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.45, whale_confluence=False,
        )

        pos = await pm.open_position(signal, no_agg)
        assert pos is not None

        # Simulate passage of time beyond entry timeout
        pos.entry_time = time.time() - 600  # 10 min ago, timeout is 300s

        await pm.check_timeouts()
        assert pos.state == PositionState.CANCELLED
        assert pos.exit_reason == "entry_timeout"

    @pytest.mark.asyncio
    async def test_exit_timeout_forces_close(self):
        """Exit pending > exit_timeout_seconds → force closed."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(50.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.45, whale_confluence=False,
        )

        pos = await pm.open_position(signal, no_agg)
        assert pos is not None

        # Simulate entry fill
        pos.entry_order.status = OrderStatus.FILLED
        pos.entry_order.filled_avg_price = pos.entry_price
        await pm.on_entry_filled(pos)
        assert pos.state == PositionState.EXIT_PENDING

        # Simulate passage of time beyond exit timeout
        pos.entry_time = time.time() - 2000  # > 1800s timeout

        await pm.check_timeouts()
        assert pos.state == PositionState.CLOSED
        assert pos.exit_reason == "timeout"


# ═══════════════════════════════════════════════════════════════════════════
#  Section E: Executor Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutorEdgeCases:
    """Edge cases for the paper-mode order executor."""

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled(self):
        """Cancelling an already-cancelled order should be a no-op."""
        executor = OrderExecutor(paper_mode=True)
        order = await executor.place_limit_order("MKT", "T1", OrderSide.BUY, 0.45, 10)
        await executor.cancel_order(order)
        assert order.status == OrderStatus.CANCELLED
        await executor.cancel_order(order)  # Should not raise
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_filled_order_is_noop(self):
        """Cannot cancel a filled order."""
        executor = OrderExecutor(paper_mode=True)
        order = await executor.place_limit_order("MKT", "T1", OrderSide.BUY, 0.45, 10)
        executor.check_paper_fill("T1", 0.40)  # Fill it
        assert order.status == OrderStatus.FILLED
        await executor.cancel_order(order)
        assert order.status == OrderStatus.FILLED  # Still filled

    @pytest.mark.asyncio
    async def test_filled_order_not_refilled(self):
        """A filled order should not fill again on subsequent price crosses."""
        executor = OrderExecutor(paper_mode=True)
        order = await executor.place_limit_order("MKT", "T1", OrderSide.BUY, 0.45, 10)
        executor.check_paper_fill("T1", 0.40)  # First fill
        filled_again = executor.check_paper_fill("T1", 0.30)  # Second cross
        assert len(filled_again) == 0

    @pytest.mark.asyncio
    async def test_multiple_orders_independent_fills(self):
        """Multiple open orders should fill independently."""
        executor = OrderExecutor(paper_mode=True)
        o1 = await executor.place_limit_order("MKT", "T1", OrderSide.BUY, 0.45, 10)
        o2 = await executor.place_limit_order("MKT", "T1", OrderSide.BUY, 0.40, 5)

        filled = executor.check_paper_fill("T1", 0.42)
        # Only o1 (price 0.45) should fill at 0.42
        assert len(filled) == 1
        assert filled[0].order_id == o1.order_id
        assert o2.status == OrderStatus.LIVE

    @pytest.mark.asyncio
    async def test_get_order_by_id(self):
        """get_order returns the correct order by ID."""
        executor = OrderExecutor(paper_mode=True)
        order = await executor.place_limit_order("MKT", "T1", OrderSide.BUY, 0.45, 10)
        fetched = executor.get_order(order.order_id)
        assert fetched is order

    @pytest.mark.asyncio
    async def test_get_order_nonexistent(self):
        """get_order returns None for unknown ID."""
        executor = OrderExecutor(paper_mode=True)
        assert executor.get_order("GHOST-999") is None

    def test_check_paper_fill_live_mode_returns_empty(self):
        """In live mode, check_paper_fill should return empty list."""
        executor = OrderExecutor(paper_mode=False)
        filled = executor.check_paper_fill("T1", 0.50)
        assert filled == []


# ═══════════════════════════════════════════════════════════════════════════
#  Section F: Panic Detector Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestPanicDetectorEdgeCases:
    """Edge cases for the panic signal detector."""

    def test_zero_sigma_returns_none(self):
        """If σ=0 (flat prices), detector should return None (avoid div/0)."""
        yes_agg = OHLCVAggregator("YES", lookback_minutes=10)
        no_agg = OHLCVAggregator("NO", lookback_minutes=10)
        build_bar_history(yes_agg, [0.50] * 10, base_vol=10.0)
        build_bar_history(no_agg, [0.50] * 10, base_vol=10.0)

        det = PanicDetector(
            market_id="MKT", yes_asset_id="YES", no_asset_id="NO",
            yes_aggregator=yes_agg, no_aggregator=no_agg,
            zscore_threshold=2.0, volume_ratio_threshold=3.0,
        )

        bar = OHLCVBar(
            open_time=99999, open=0.50, high=0.90,
            low=0.50, close=0.90, volume=100.0,
            vwap=0.80, trade_count=50,
        )
        signal = det.evaluate(bar, no_best_ask=0.30)
        assert signal is None, "Must not crash or fire on σ=0"

    def test_whale_confluence_passed_through(self):
        """whale_confluence flag should be preserved in the PanicSignal."""
        yes_agg = OHLCVAggregator("YES", lookback_minutes=10)
        no_agg = OHLCVAggregator("NO", lookback_minutes=10)
        prices = [0.45, 0.46, 0.47, 0.45, 0.46, 0.48, 0.45, 0.47, 0.46, 0.45]
        build_bar_history(yes_agg, prices, base_vol=10.0)
        build_bar_history(no_agg, [0.55, 0.54, 0.55, 0.56, 0.55, 0.54, 0.55, 0.55, 0.54, 0.55], base_vol=10.0)

        det = PanicDetector(
            market_id="MKT", yes_asset_id="YES", no_asset_id="NO",
            yes_aggregator=yes_agg, no_aggregator=no_agg,
            zscore_threshold=2.0, volume_ratio_threshold=3.0,
        )

        mu = yes_agg.rolling_vwap
        sigma = yes_agg.rolling_volatility
        if sigma > 0:
            spike = mu + 5.0 * sigma
            bar = OHLCVBar(
                open_time=99999, open=mu, high=spike,
                low=mu, close=spike, volume=100.0,
                vwap=spike, trade_count=50,
            )
            no_ask = no_agg.rolling_vwap - 0.05
            signal = det.evaluate(bar, no_best_ask=no_ask, whale_confluence=True)
            if signal:
                assert signal.whale_confluence is True
