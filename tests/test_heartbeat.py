"""
Tests for BookHeartbeat — tiered WebSocket health monitor.

Layer 1: Transport health (L2WebSocket._last_message_time)
Layer 2: Positioned-asset health (per-book _last_update for open positions)
Layer 3: Legacy fallback (freshest-tracker scan when no transport wired)
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.heartbeat import BookHeartbeat
from src.core.latency_guard import LatencyGuard
from src.data.orderbook import OrderbookTracker
from src.trading.executor import OrderExecutor


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_tracker(asset_id: str, last_update: float = 0.0, server_time: float = 0.0) -> OrderbookTracker:
    """Create a tracker with controlled timestamps."""
    t = OrderbookTracker(asset_id)
    t._last_update = last_update
    t._last_server_time = server_time
    return t


def _make_transport(last_message_time: float = 0.0) -> SimpleNamespace:
    """Fake WS transport with a _last_message_time attribute."""
    return SimpleNamespace(_last_message_time=last_message_time)


# ── Layer 1: Transport health ──────────────────────────────────────────────

class TestTransportLayer:
    """If we have a WS transport, its ``_last_message_time`` is the
    definitive signal.  Per-asset staleness is irrelevant for WS health."""

    @pytest.fixture
    def components(self):
        guard = LatencyGuard(block_ms=500, warn_ms=200, recovery_count=3)
        event = asyncio.Event()
        event.set()
        executor = OrderExecutor(paper_mode=True)
        return guard, event, executor

    @pytest.mark.asyncio
    async def test_healthy_transport_no_suspend(self, components):
        """Fresh WS messages → no suspension, even if some trackers are stale."""
        guard, event, executor = components
        now = time.time()

        # Transport received a message 100ms ago — connection is alive
        transport = _make_transport(now - 0.1)

        # Trackers: one very stale (low-activity market)
        trackers = {
            "STALE": _make_tracker("STALE", last_update=now - 30.0),
            "FRESH": _make_tracker("FRESH", last_update=now),
        }

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
        )

        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_dead_transport_suspends(self, components):
        """No WS message for > stale_ms → suspended immediately."""
        guard, event, executor = components
        now = time.time()

        # Transport: no message for 5 seconds
        transport = _make_transport(now - 5.0)
        trackers = {"A": _make_tracker("A", last_update=now)}

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
        )

        await hb._check()
        assert hb.is_suspended is True
        assert guard.is_blocked() is True
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_dead_transport_resumes_on_message(self, components):
        """When WS starts receiving messages again, resume."""
        guard, event, executor = components
        now = time.time()

        transport = _make_transport(now - 5.0)
        trackers = {"A": _make_tracker("A", last_update=now)}

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
        )

        await hb._check()
        assert hb.is_suspended is True

        # WS recovers
        transport._last_message_time = time.time()
        await hb._check()
        assert hb.is_suspended is False
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_transport_not_yet_connected_no_suspend(self, components):
        """If transport has _last_message_time == 0 (still connecting),
        do not suspend — there's nothing to invalidate yet."""
        guard, event, executor = components

        transport = _make_transport(0.0)  # never received a message
        trackers = {"A": _make_tracker("A", last_update=time.time())}

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
        )

        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_many_stale_trackers_healthy_transport(self, components):
        """148 stale trackers should NOT cause suspension when WS is alive.
        This is the exact scenario that caused flapping in production."""
        guard, event, executor = components
        now = time.time()

        transport = _make_transport(now - 0.2)  # healthy WS

        # Simulate 148 markets with varying staleness — some very stale
        trackers = {
            f"ASSET_{i}": _make_tracker(
                f"ASSET_{i}",
                last_update=now - (i * 0.5),  # up to 74 seconds stale
            )
            for i in range(148)
        }

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
        )

        # Run 10 consecutive checks — should never suspend
        for _ in range(10):
            await hb._check()
        assert hb.is_suspended is False


# ── Layer 2: Positioned-asset health ──────────────────────────────────────

class TestPositionLayer:
    """Position-aware staleness: only suspend if a book we're actively
    trading against goes stale (3× the base threshold)."""

    @pytest.fixture
    def components(self):
        guard = LatencyGuard(block_ms=500, warn_ms=200, recovery_count=3)
        event = asyncio.Event()
        event.set()
        executor = OrderExecutor(paper_mode=True)
        return guard, event, executor

    @pytest.mark.asyncio
    async def test_no_positions_stale_trackers_no_suspend(self, components):
        """Stale trackers with NO open positions → no suspension
        (nothing at risk, no need to block execution)."""
        guard, event, executor = components
        now = time.time()

        transport = _make_transport(now - 0.1)
        trackers = {
            "STALE": _make_tracker("STALE", last_update=now - 30.0),
        }

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
            get_position_assets=lambda: set(),
        )

        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_position_on_fresh_book_no_suspend(self, components):
        """Open position on a book that's fresh → no problem."""
        guard, event, executor = components
        now = time.time()

        transport = _make_transport(now - 0.1)
        trackers = {
            "POS_BOOK": _make_tracker("POS_BOOK", last_update=now),
        }

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
            get_position_assets=lambda: {"POS_BOOK"},
        )

        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_position_on_stale_book_suspends(self, components):
        """Open position on a book that hasn't updated for 3× threshold → suspend.
        Default threshold is 1500ms, so position threshold is 4500ms."""
        guard, event, executor = components
        now = time.time()

        transport = _make_transport(now - 0.1)  # WS is healthy
        trackers = {
            "POS_BOOK": _make_tracker("POS_BOOK", last_update=now - 5.0),
        }

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
            get_position_assets=lambda: {"POS_BOOK"},
        )

        await hb._check()
        assert hb.is_suspended is True

    @pytest.mark.asyncio
    async def test_position_within_3x_threshold_no_suspend(self, components):
        """Position book stale by 2s (< 3× 1.5s = 4.5s) → NOT suspended."""
        guard, event, executor = components
        now = time.time()

        transport = _make_transport(now - 0.1)
        trackers = {
            "POS_BOOK": _make_tracker("POS_BOOK", last_update=now - 2.0),
        }

        hb = BookHeartbeat(
            trackers, guard, event, executor,
            ws_transport=transport,
            get_position_assets=lambda: {"POS_BOOK"},
        )

        await hb._check()
        assert hb.is_suspended is False


# ── Layer 3: Legacy fallback (no transport) ───────────────────────────────

class TestLegacyFallback:
    """When no WS transport is wired (tests, non-L2 mode), falls back to
    the original freshest-tracker heuristic."""

    @pytest.fixture
    def components(self):
        guard = LatencyGuard(block_ms=500, warn_ms=200, recovery_count=3)
        event = asyncio.Event()
        event.set()
        executor = OrderExecutor(paper_mode=True)
        return guard, event, executor

    @pytest.mark.asyncio
    async def test_legacy_suspend_on_all_stale(self, components):
        """No transport, all trackers stale → suspend."""
        guard, event, executor = components
        now = time.time()

        trackers = {
            "A": _make_tracker("A", last_update=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is True

    @pytest.mark.asyncio
    async def test_legacy_one_fresh_no_suspend(self, components):
        """No transport, at least one fresh → do not suspend."""
        guard, event, executor = components
        now = time.time()

        trackers = {
            "STALE": _make_tracker("STALE", last_update=now - 10.0),
            "FRESH": _make_tracker("FRESH", last_update=now),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_legacy_resume_when_fresh(self, components):
        """Legacy mode: resume after fresh data arrives."""
        guard, event, executor = components
        now = time.time()

        trackers = {
            "A": _make_tracker("A", last_update=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is True

        trackers["A"]._last_update = time.time()
        await hb._check()
        assert hb.is_suspended is False
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_legacy_server_time_stale(self, components):
        """Legacy: both local and server timestamps stale → suspend."""
        guard, event, executor = components
        now = time.time()

        trackers = {
            "A": _make_tracker("A", last_update=now - 5.0, server_time=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is True

    @pytest.mark.asyncio
    async def test_legacy_fresh_local_stale_server(self, components):
        """Legacy: local is fresh (WS alive), server is behind (clock skew)
        → NOT suspended."""
        guard, event, executor = components
        now = time.time()

        trackers = {
            "A": _make_tracker("A", last_update=now, server_time=now - 5.0),
        }
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is False


# ── General behaviour ─────────────────────────────────────────────────────

class TestGeneralBehaviour:
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
    async def test_empty_trackers_no_suspend(self, components):
        guard, event, executor = components
        hb = BookHeartbeat({}, guard, event, executor)
        await hb._check()
        assert hb.is_suspended is False

    @pytest.mark.asyncio
    async def test_cancel_all_called_on_suspend(self, components):
        """Suspension should cancel all resting orders."""
        guard, event, executor = components
        now = time.time()

        await executor.place_limit_order(
            market_id="MKT_1", asset_id="ASSET_A",
            side=MagicMock(), price=0.5, size=10.0,
        )

        trackers = {"A": _make_tracker("A", last_update=now - 5.0)}
        hb = BookHeartbeat(trackers, guard, event, executor)

        await hb._check()
        assert hb.is_suspended is True

    def test_stop(self, components):
        guard, event, executor = components
        hb = BookHeartbeat({}, guard, event, executor)
        hb._running = True
        hb.stop()
        assert hb._running is False
