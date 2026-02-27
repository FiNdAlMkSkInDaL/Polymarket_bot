"""
Tests for AdverseSelectionGuard — fast-kill predictive cancellation.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.signals.adverse_selection_guard import AdverseSelectionGuard, _BinanceTick
from src.data.orderbook import OrderbookTracker
from src.trading.executor import OrderExecutor, OrderSide


class TestAdverseSelectionGuard:
    @pytest.fixture
    def components(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers: dict[str, OrderbookTracker] = {}
        return executor, trackers, event

    def test_initial_state(self, components):
        executor, trackers, event = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        assert guard._running is False
        assert guard._cancel_count == 0
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_disabled_guard_returns_immediately(self, components):
        """When adverse_sel_enabled=False, start() returns immediately."""
        executor, trackers, event = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        guard._enabled = False
        # Should return without blocking
        await asyncio.wait_for(guard.start(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_execute_fast_kill_clears_event(self, components):
        """_execute_fast_kill should clear the kill event and cancel orders."""
        executor, trackers, event = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        guard._running = True

        # Place a resting order
        await executor.place_limit_order(
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, price=0.5, size=10.0,
        )

        await guard._execute_fast_kill()

        assert not event.is_set()  # event cleared
        assert guard._cancel_count == 1

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_kills(self, components):
        """After a kill, cooldown should prevent re-triggering."""
        executor, trackers, event = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        guard._running = True

        await guard._execute_fast_kill()
        assert guard._cooldown_until > time.time()

    def test_binance_tick_delta_computation(self, components):
        """Verify tick delta calculation from Binance price changes."""
        executor, trackers, event = components
        guard = AdverseSelectionGuard(executor, trackers, event)

        now = time.time()
        # Simulate two ticks 1 second apart with a $2.00 BTC move
        guard._ticks.append(_BinanceTick(price=100000.0, ts=now - 1.5))
        guard._ticks.append(_BinanceTick(price=100002.0, ts=now))

        # ext_delta_ticks is a property: |Δ| / $0.01 = 200 ticks
        delta = guard.ext_delta_ticks
        assert delta == pytest.approx(200.0, abs=1.0)

    def test_book_staleness_computation(self, components):
        """Max book age should reflect the time since last update."""
        executor, trackers, event = components
        tracker = OrderbookTracker("ASSET_A")
        tracker._last_update = time.time() - 1.0  # 1 second ago
        trackers["ASSET_A"] = tracker

        guard = AdverseSelectionGuard(executor, trackers, event)
        age = guard.max_book_age_ms
        assert age >= 900  # at least 900ms (allowing some timing slack)
        assert age < 2000

    def test_max_book_age_empty_trackers(self, components):
        """Empty tracker dict should return 0."""
        executor, trackers, event = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        assert guard.max_book_age_ms == 0.0
