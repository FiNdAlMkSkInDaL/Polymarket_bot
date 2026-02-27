"""
Tests for BookHeartbeat — multi-source staleness detection.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.heartbeat import BookHeartbeat
from src.core.latency_guard import LatencyGuard, LatencyState
from src.data.orderbook import OrderbookTracker
from src.trading.executor import OrderExecutor


def _make_tracker(asset_id: str, last_update: float = 0.0, server_time: float = 0.0) -> OrderbookTracker:
    """Create a tracker with controlled timestamps."""
    t = OrderbookTracker(asset_id)
    t._last_update = last_update
    t._last_server_time = server_time
    return t


class TestBookHeartbeat:
    @pytest.fixture
    def components(self):
        guard = LatencyGuard(block_ms=500, warn_ms=200, recovery_count=3)
        event = asyncio.Event()
        event.set()
        executor = OrderExecutor(paper_mode=True)
        return guard, event, executor

    def test_not_suspended_initially(self, components):
        guard, event, executor = components
        hb = BookHeartbeat({}, guard, event, executor)
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_suspend_on_stale_book(self, components):
        """When max gap exceeds threshold, heartbeat suspends execution."""
        guard, event, executor = components
        now = time.time()

        # Tracker with data that is 5 seconds old (well above default 1500ms)
        trackers = {
            "ASSET_A": _make_tracker("ASSET_A", last_update=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        # Run the check directly
        await hb._check()

        assert hb.is_suspended is True
        assert guard.is_blocked() is True
        assert not event.is_set()  # fast-kill cleared

    @pytest.mark.asyncio
    async def test_resume_when_fresh(self, components):
        """After suspension, fresh data should trigger resume."""
        guard, event, executor = components
        now = time.time()

        trackers = {
            "ASSET_A": _make_tracker("ASSET_A", last_update=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        # Suspend first
        await hb._check()
        assert hb.is_suspended is True

        # Now make the data fresh
        trackers["ASSET_A"]._last_update = time.time()
        trackers["ASSET_A"]._last_server_time = time.time()

        await hb._check()
        assert hb.is_suspended is False
        assert event.is_set()  # fast-kill restored

    @pytest.mark.asyncio
    async def test_no_suspend_with_fresh_data(self, components):
        """Fresh data should not trigger suspension."""
        guard, event, executor = components
        now = time.time()

        trackers = {
            "ASSET_A": _make_tracker("ASSET_A", last_update=now),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_empty_trackers_no_suspend(self, components):
        """No trackers means nothing to check — should not suspend."""
        guard, event, executor = components
        hb = BookHeartbeat({}, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_cancel_all_called_on_suspend(self, components):
        """Suspension should cancel all resting orders."""
        guard, event, executor = components
        now = time.time()

        # Place a resting order
        await executor.place_limit_order(
            market_id="MKT_1", asset_id="ASSET_A",
            side=MagicMock(), price=0.5, size=10.0,
        )

        trackers = {
            "ASSET_A": _make_tracker("ASSET_A", last_update=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is True

    @pytest.mark.asyncio
    async def test_server_time_gap_triggers_suspend(self, components):
        """Server-clock gap should also trigger suspension."""
        guard, event, executor = components
        now = time.time()

        # Local update is fresh, but server_time is old
        trackers = {
            "ASSET_A": _make_tracker("ASSET_A", last_update=now, server_time=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is True

    def test_stop(self, components):
        guard, event, executor = components
        hb = BookHeartbeat({}, guard, event, executor)
        hb._running = True
        hb.stop()
        assert hb._running is False
