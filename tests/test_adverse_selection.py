"""
Tests for AdverseSelectionGuard v2 -- intrinsic microstructure detection.

Tests each of the four detection mechanisms individually and the 2-of-4
composite trigger logic.  All tests are deterministic and run without
live market data by directly manipulating internal guard state.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.orderbook import OrderbookTracker
from src.data.types import Level
from src.signals.adverse_selection_guard import AdverseSelectionGuard
from src.trading.executor import OrderExecutor, OrderSide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(asset_id: str, bid: float = 0.50, ask: float = 0.55,
                  bid_sizes: list[float] | None = None,
                  ask_sizes: list[float] | None = None) -> OrderbookTracker:
    """Create an OrderbookTracker with controlled BBO and depth."""
    tracker = OrderbookTracker(asset_id)
    if bid_sizes is None:
        bid_sizes = [100.0]
    if ask_sizes is None:
        ask_sizes = [100.0]

    bids_data = []
    for i, size in enumerate(bid_sizes):
        price = bid - i * 0.01
        bids_data.append({"side": "BUY", "price": str(price), "size": str(size)})
    asks_data = []
    for i, size in enumerate(ask_sizes):
        price = ask + i * 0.01
        asks_data.append({"side": "SELL", "price": str(price), "size": str(size)})

    tracker.on_price_change({"changes": bids_data + asks_data})
    return tracker


def _position_assets_factory(*asset_ids: str):
    """Return a callable that returns a fixed set of positioned asset IDs."""
    return lambda: set(asset_ids)


class TestAdverseSelectionGuard:
    """Core lifecycle and kill mechanics."""

    @pytest.fixture
    def components(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers: dict[str, OrderbookTracker] = {}
        taker_counts: dict[str, int] = {}
        total_counts: dict[str, int] = {}
        trade_counts: dict[str, float] = {}
        return executor, trackers, event, taker_counts, total_counts, trade_counts

    def test_initial_state(self, components):
        executor, trackers, event, tc, tot, trd = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        assert guard._running is False
        assert guard._cancel_count == 0
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_disabled_guard_returns_immediately(self, components):
        """When adverse_sel_enabled=False, start() returns immediately."""
        executor, trackers, event, tc, tot, trd = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        guard._enabled = False
        await asyncio.wait_for(guard.start(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_execute_fast_kill_clears_event(self, components):
        """_execute_fast_kill should clear the kill event and cancel orders."""
        executor, trackers, event, tc, tot, trd = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        guard._running = True

        await executor.place_limit_order(
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, price=0.5, size=10.0,
        )

        await guard._execute_fast_kill([{"signal": "test"}])

        assert not event.is_set()
        assert guard._cancel_count == 1

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_kills(self, components):
        """After a kill, cooldown should prevent re-triggering."""
        executor, trackers, event, tc, tot, trd = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        guard._running = True

        await guard._execute_fast_kill([{"signal": "test"}])
        assert guard._cooldown_until > time.time()

    @pytest.mark.asyncio
    async def test_cooldown_release_restores_event(self, components):
        """After cooldown, the event should be set again."""
        executor, trackers, event, tc, tot, trd = components
        guard = AdverseSelectionGuard(executor, trackers, event)
        guard._running = True
        guard._cooldown_s = 0.1

        await guard._execute_fast_kill([{"signal": "test"}])
        assert not event.is_set()

        await asyncio.sleep(0.2)
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_kill_sends_telegram_alert(self, components):
        """Fast kill should send a Telegram alert with signal details."""
        executor, trackers, event, tc, tot, trd = components
        telegram = AsyncMock()
        telegram.send = AsyncMock()
        guard = AdverseSelectionGuard(
            executor, trackers, event, telegram=telegram,
        )
        guard._running = True

        await guard._execute_fast_kill([
            {"signal": "flow_coherence"},
            {"signal": "depth_evaporation"},
        ])
        telegram.send.assert_called_once()
        msg = telegram.send.call_args[0][0]
        assert "Adverse Selection Kill" in msg
        assert "flow_coherence" in msg


class TestFlowCoherence:
    """Signal 1: Cross-market flow coherence."""

    @pytest.fixture
    def guard(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        taker_counts: dict[str, int] = {}
        total_counts: dict[str, int] = {}
        guard = AdverseSelectionGuard(
            executor, {}, event,
            taker_counts=taker_counts,
            total_counts=total_counts,
        )
        return guard

    def test_fires_when_enough_markets_hot(self, guard):
        """MTI > threshold on 3+ markets should fire."""
        now = time.time()
        old_snap = {f"ASSET_{i}": (0, 0) for i in range(4)}
        guard._mti_snapshots.append((now - 3.0, old_snap))

        guard._taker_counts.update({f"ASSET_{i}": 9 for i in range(4)})
        guard._total_counts.update({f"ASSET_{i}": 10 for i in range(4)})
        guard._snapshot_mti()

        fired, diag = guard._check_flow_coherence()
        assert fired is True
        assert diag["hot_markets"] >= 3

    def test_below_min_markets(self, guard):
        """Only 2 markets with high MTI should NOT fire (need 3)."""
        now = time.time()
        old_snap = {f"ASSET_{i}": (0, 0) for i in range(4)}
        guard._mti_snapshots.append((now - 3.0, old_snap))

        guard._taker_counts.update({"ASSET_0": 9, "ASSET_1": 9,
                                    "ASSET_2": 1, "ASSET_3": 1})
        guard._total_counts.update({"ASSET_0": 10, "ASSET_1": 10,
                                    "ASSET_2": 10, "ASSET_3": 10})
        guard._snapshot_mti()

        fired, diag = guard._check_flow_coherence()
        assert fired is False
        assert diag["hot_markets"] == 2

    def test_below_threshold(self, guard):
        """4 markets with MTI=0.5 (below 0.85) should NOT fire."""
        now = time.time()
        old_snap = {f"ASSET_{i}": (0, 0) for i in range(4)}
        guard._mti_snapshots.append((now - 3.0, old_snap))

        guard._taker_counts.update({f"ASSET_{i}": 5 for i in range(4)})
        guard._total_counts.update({f"ASSET_{i}": 10 for i in range(4)})
        guard._snapshot_mti()

        fired, diag = guard._check_flow_coherence()
        assert fired is False

    def test_insufficient_data(self, guard):
        """No snapshots -> should not fire."""
        fired, diag = guard._check_flow_coherence()
        assert fired is False


class TestDepthEvaporation:
    """Signal 2: Book depth evaporation."""

    @pytest.fixture
    def guard(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers: dict[str, OrderbookTracker] = {}

        tracker = _make_tracker(
            "ASSET_A", bid=0.50, ask=0.52,
            bid_sizes=[200.0, 150.0, 100.0],
            ask_sizes=[200.0, 150.0, 100.0],
        )
        trackers["ASSET_A"] = tracker

        guard = AdverseSelectionGuard(
            executor, trackers, event,
            get_position_assets=_position_assets_factory("ASSET_A"),
        )
        return guard, tracker

    def test_fires_on_depth_drop(self, guard):
        """60%+ depth drop within window should fire."""
        guard_obj, tracker = guard
        now = time.time()

        guard_obj._depth_near_mid_history["ASSET_A"] = deque(maxlen=100)
        guard_obj._depth_near_mid_history["ASSET_A"].append(
            (now - 3.0, 500.0)
        )
        guard_obj._depth_near_mid_history["ASSET_A"].append(
            (now, 100.0)
        )

        fired, diag = guard_obj._check_depth_evaporation()
        assert fired is True
        assert diag["worst_drop_pct"] >= 60.0

    def test_no_fire_on_small_drop(self, guard):
        """40% drop (below 60% threshold) should NOT fire."""
        guard_obj, tracker = guard
        now = time.time()

        guard_obj._depth_near_mid_history["ASSET_A"] = deque(maxlen=100)
        guard_obj._depth_near_mid_history["ASSET_A"].append(
            (now - 3.0, 500.0)
        )
        guard_obj._depth_near_mid_history["ASSET_A"].append(
            (now, 300.0)
        )

        fired, diag = guard_obj._check_depth_evaporation()
        assert fired is False

    def test_no_fire_on_non_positioned(self):
        """Snapshot step should not record depth for non-positioned assets."""
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers = {"ASSET_B": _make_tracker("ASSET_B")}

        guard = AdverseSelectionGuard(
            executor, trackers, event,
            get_position_assets=_position_assets_factory(),
        )

        guard._snapshot_depth()
        assert "ASSET_B" not in guard._depth_near_mid_history

    def test_depth_near_mid_computation(self, guard):
        """depth_near_mid should sum depth within N cents of mid."""
        guard_obj, tracker = guard
        depth = guard_obj._depth_near_mid(tracker)
        assert depth > 0

    def test_empty_book_returns_zero(self):
        """Tracker with no data should return 0 depth."""
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        tracker = OrderbookTracker("EMPTY")
        guard = AdverseSelectionGuard(executor, {"EMPTY": tracker}, event)
        assert guard._depth_near_mid(tracker) == 0.0


class TestSpreadBlowout:
    """Signal 3: Spread blow-out."""

    @pytest.fixture
    def guard(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers = {"ASSET_A": _make_tracker("ASSET_A", bid=0.50, ask=0.52)}

        guard = AdverseSelectionGuard(
            executor, trackers, event,
            get_position_assets=_position_assets_factory("ASSET_A"),
        )
        return guard

    def test_fires_on_blowout(self, guard):
        """Spread at 3x+ average should fire."""
        now = time.time()
        history = deque(maxlen=10000)
        for i in range(50):
            history.append((now - 300 + i * 6, 2.0))
        history.append((now, 7.0))
        guard._spread_ts_history["ASSET_A"] = history

        fired, diag = guard._check_spread_blowout()
        assert fired is True
        assert diag["worst_mult"] >= 3.0

    def test_no_fire_below_multiplier(self, guard):
        """Spread at 2x average should NOT fire (threshold is 3x)."""
        now = time.time()
        history = deque(maxlen=10000)
        for i in range(50):
            history.append((now - 300 + i * 6, 2.0))
        history.append((now, 4.0))
        guard._spread_ts_history["ASSET_A"] = history

        fired, diag = guard._check_spread_blowout()
        assert fired is False

    def test_no_fire_insufficient_history(self, guard):
        """Fewer than 10 spread observations should NOT fire."""
        now = time.time()
        history = deque(maxlen=10000)
        for i in range(5):
            history.append((now - 5 + i, 2.0))
        history.append((now, 10.0))
        guard._spread_ts_history["ASSET_A"] = history

        fired, _ = guard._check_spread_blowout()
        assert fired is False


class TestVelocityAnomaly:
    """Signal 4: Velocity anomaly on positioned assets."""

    @pytest.fixture
    def guard(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trade_counts: dict[str, float] = {"ASSET_A": 0.0}

        guard = AdverseSelectionGuard(
            executor, {}, event,
            trade_counts=trade_counts,
            get_position_assets=_position_assets_factory("ASSET_A"),
        )
        return guard

    def test_fires_on_velocity_spike(self, guard):
        """5x+ burst over baseline should fire."""
        now = time.time()

        guard._trade_rate_snapshots.clear()
        guard._trade_rate_snapshots.append(
            (now - 700, {"ASSET_A": 0.0})
        )
        guard._trade_rate_snapshots.append(
            (now - 15, {"ASSET_A": 100.0})
        )
        guard._trade_rate_snapshots.append(
            (now, {"ASSET_A": 200.0})
        )
        # Long rate: 200/700 = 0.286/s
        # Short rate: 100/15 = 6.67/s
        # Ratio: ~23.3x

        fired, diag = guard._check_velocity_anomaly()
        assert fired is True
        assert diag["worst_mult"] >= 5.0

    def test_no_fire_below_multiplier(self, guard):
        """Low burst should NOT fire (threshold is 5x)."""
        now = time.time()

        guard._trade_rate_snapshots.clear()
        guard._trade_rate_snapshots.append(
            (now - 700, {"ASSET_A": 0.0})
        )
        guard._trade_rate_snapshots.append(
            (now - 15, {"ASSET_A": 95.0})
        )
        guard._trade_rate_snapshots.append(
            (now, {"ASSET_A": 100.0})
        )
        # Long rate: 100/700 = 0.143/s
        # Short rate: 5/15 = 0.333/s
        # Ratio: ~2.33x

        fired, diag = guard._check_velocity_anomaly()
        assert fired is False

    def test_no_fire_insufficient_data(self, guard):
        """No snapshots -> should not fire."""
        guard._trade_rate_snapshots.clear()
        fired, _ = guard._check_velocity_anomaly()
        assert fired is False

    def test_no_fire_on_non_positioned(self, guard):
        """Velocity on non-positioned asset should not trigger."""
        now = time.time()
        guard._get_position_assets = lambda: set()

        guard._trade_rate_snapshots.clear()
        guard._trade_rate_snapshots.append((now - 700, {"ASSET_A": 0.0}))
        guard._trade_rate_snapshots.append((now - 15, {"ASSET_A": 95.0}))
        guard._trade_rate_snapshots.append((now, {"ASSET_A": 200.0}))

        fired, _ = guard._check_velocity_anomaly()
        assert fired is False


class TestCompositeTriggering:
    """2-of-4 composite trigger logic."""

    @pytest.fixture
    def guard(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        taker_counts: dict[str, int] = {}
        total_counts: dict[str, int] = {}
        trade_counts: dict[str, float] = {}
        trackers: dict[str, OrderbookTracker] = {}

        guard = AdverseSelectionGuard(
            executor, trackers, event,
            taker_counts=taker_counts,
            total_counts=total_counts,
            trade_counts=trade_counts,
            get_position_assets=_position_assets_factory("ASSET_A"),
        )
        return guard, executor, event

    @pytest.mark.asyncio
    async def test_two_signals_triggers_kill(self, guard):
        """When exactly 2 signals fire and orders exist -> kill fires."""
        guard_obj, executor, event = guard
        guard_obj._running = True

        await executor.place_limit_order(
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, price=0.5, size=10.0,
        )
        assert executor.open_order_count == 1

        guard_obj._check_flow_coherence = lambda: (True, {"signal": "flow_coherence"})
        guard_obj._check_spread_blowout = lambda: (True, {"signal": "spread_blowout"})
        guard_obj._check_depth_evaporation = lambda: (False, {})
        guard_obj._check_velocity_anomaly = lambda: (False, {})

        signals_fired = []
        for check_fn in [guard_obj._check_flow_coherence,
                         guard_obj._check_depth_evaporation,
                         guard_obj._check_spread_blowout,
                         guard_obj._check_velocity_anomaly]:
            fired, diag = check_fn()
            if fired:
                signals_fired.append(diag)

        assert len(signals_fired) == 2
        await guard_obj._execute_fast_kill(signals_fired)
        assert not event.is_set()
        assert guard_obj._cancel_count == 1

    @pytest.mark.asyncio
    async def test_single_signal_no_kill(self, guard):
        """When only 1 signal fires -> no kill."""
        guard_obj, executor, event = guard
        guard_obj._running = True

        await executor.place_limit_order(
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, price=0.5, size=10.0,
        )

        guard_obj._check_flow_coherence = lambda: (False, {})
        guard_obj._check_spread_blowout = lambda: (False, {})
        guard_obj._check_depth_evaporation = lambda: (True, {"signal": "depth_evaporation"})
        guard_obj._check_velocity_anomaly = lambda: (False, {})

        signals_fired = []
        for check_fn in [guard_obj._check_flow_coherence,
                         guard_obj._check_depth_evaporation,
                         guard_obj._check_spread_blowout,
                         guard_obj._check_velocity_anomaly]:
            fired, diag = check_fn()
            if fired:
                signals_fired.append(diag)

        assert len(signals_fired) == 1
        assert event.is_set()
        assert guard_obj._cancel_count == 0

    @pytest.mark.asyncio
    async def test_three_signals_triggers_kill(self, guard):
        """When 3 signals fire -> kill fires."""
        guard_obj, executor, event = guard
        guard_obj._running = True

        await executor.place_limit_order(
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, price=0.5, size=10.0,
        )

        guard_obj._check_flow_coherence = lambda: (True, {"signal": "flow_coherence"})
        guard_obj._check_spread_blowout = lambda: (True, {"signal": "spread_blowout"})
        guard_obj._check_depth_evaporation = lambda: (True, {"signal": "depth_evaporation"})
        guard_obj._check_velocity_anomaly = lambda: (False, {})

        signals_fired = []
        for check_fn in [guard_obj._check_flow_coherence,
                         guard_obj._check_depth_evaporation,
                         guard_obj._check_spread_blowout,
                         guard_obj._check_velocity_anomaly]:
            fired, diag = check_fn()
            if fired:
                signals_fired.append(diag)

        assert len(signals_fired) == 3
        await guard_obj._execute_fast_kill(signals_fired)
        assert not event.is_set()
        assert guard_obj._cancel_count == 1

    @pytest.mark.asyncio
    async def test_no_orders_skips_kill(self, guard):
        """When 2 signals fire but no open orders -> no kill (skip)."""
        guard_obj, executor, event = guard
        guard_obj._running = True

        assert executor.open_order_count == 0

        guard_obj._check_flow_coherence = lambda: (True, {"signal": "flow_coherence"})
        guard_obj._check_spread_blowout = lambda: (True, {"signal": "spread_blowout"})
        guard_obj._check_depth_evaporation = lambda: (False, {})
        guard_obj._check_velocity_anomaly = lambda: (False, {})

        signals_fired = []
        for check_fn in [guard_obj._check_flow_coherence,
                         guard_obj._check_depth_evaporation,
                         guard_obj._check_spread_blowout,
                         guard_obj._check_velocity_anomaly]:
            fired, diag = check_fn()
            if fired:
                signals_fired.append(diag)

        assert len(signals_fired) == 2
        assert event.is_set()
        assert guard_obj._cancel_count == 0


class TestSnapshotMechanics:
    """Test the data snapshot methods."""

    def test_snapshot_mti_records_counts(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        taker = {"A": 5, "B": 10}
        total = {"A": 10, "B": 15}
        guard = AdverseSelectionGuard(
            executor, {}, event,
            taker_counts=taker, total_counts=total,
        )
        guard._snapshot_mti()
        assert len(guard._mti_snapshots) == 1
        ts, snap = guard._mti_snapshots[0]
        assert snap["A"] == (5, 10)
        assert snap["B"] == (10, 15)

    def test_snapshot_depth_only_positioned(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers = {
            "POS_A": _make_tracker("POS_A"),
            "OTHER": _make_tracker("OTHER"),
        }
        guard = AdverseSelectionGuard(
            executor, trackers, event,
            get_position_assets=_position_assets_factory("POS_A"),
        )
        guard._snapshot_depth()
        assert "POS_A" in guard._depth_near_mid_history
        assert "OTHER" not in guard._depth_near_mid_history

    def test_snapshot_spreads_records(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        tracker = _make_tracker("ASSET_A", bid=0.45, ask=0.55)
        trackers = {"ASSET_A": tracker}
        guard = AdverseSelectionGuard(
            executor, trackers, event,
            get_position_assets=_position_assets_factory("ASSET_A"),
        )
        guard._snapshot_spreads()
        assert "ASSET_A" in guard._spread_ts_history
        assert len(guard._spread_ts_history["ASSET_A"]) == 1
        _, spread = guard._spread_ts_history["ASSET_A"][0]
        assert spread == pytest.approx(10.0, abs=0.1)

    def test_snapshot_trade_rates_records(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trade_counts = {"X": 42.0, "Y": 7.5}
        guard = AdverseSelectionGuard(
            executor, {}, event,
            trade_counts=trade_counts,
        )
        guard._snapshot_trade_rates()
        assert len(guard._trade_rate_snapshots) == 1
        _, snap = guard._trade_rate_snapshots[0]
        assert snap["X"] == 42.0
        assert snap["Y"] == 7.5


class TestPolygonHeadLagChecker:
    """Test the PolygonHeadLagChecker moved to heartbeat."""

    def test_import(self):
        from src.core.heartbeat import PolygonHeadLagChecker
        checker = PolygonHeadLagChecker(rpc_url="", lag_threshold_ms=3000)
        assert checker._lag_threshold_ms == 3000

    @pytest.mark.asyncio
    async def test_empty_url_returns_healthy(self):
        from src.core.heartbeat import PolygonHeadLagChecker
        checker = PolygonHeadLagChecker(rpc_url="", lag_threshold_ms=3000)
        healthy, lag = await checker.check()
        assert healthy is True
        assert lag == 0.0


class TestSpreadBlowoutWindowed:
    """Issue 3: Spread blow-out should use time-windowed average."""

    @pytest.fixture
    def guard(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers = {"ASSET_A": _make_tracker("ASSET_A", bid=0.50, ask=0.52)}
        guard = AdverseSelectionGuard(
            executor, trackers, event,
            get_position_assets=_position_assets_factory("ASSET_A"),
        )
        return guard

    def test_old_samples_excluded_from_average(self, guard):
        """Samples outside spread_avg_window_s should NOT affect average.

        Scenario: 50 old samples at spread=10 (outside window) + 50
        recent samples at spread=2.  If the old samples are included,
        the average would be ~6 and current=6 would only be 1x -> no
        fire.  With proper windowing, avg=2 and current=6 -> 3x -> fires.
        """
        now = time.time()
        window = guard._spread_avg_window_s  # 300s default
        history = deque(maxlen=10000)

        # Old samples OUTSIDE window (should be excluded)
        for i in range(50):
            history.append((now - window - 100 + i, 10.0))

        # Recent samples INSIDE window
        for i in range(50):
            history.append((now - 200 + i * 4, 2.0))

        # Current spike
        history.append((now, 7.0))
        guard._spread_ts_history["ASSET_A"] = history

        fired, diag = guard._check_spread_blowout()
        assert fired is True
        assert diag["worst_mult"] >= 3.0

    def test_all_in_window_behaves_normally(self, guard):
        """When all samples are within window, behaviour is unchanged."""
        now = time.time()
        history = deque(maxlen=10000)
        for i in range(50):
            history.append((now - 250 + i * 5, 2.0))
        history.append((now, 4.0))  # 2x -> below 3x threshold
        guard._spread_ts_history["ASSET_A"] = history

        fired, diag = guard._check_spread_blowout()
        assert fired is False


class TestVelocityAdaptiveMultiplier:
    """Issue 4: High-freq markets should require a boosted multiplier."""

    @pytest.fixture
    def guard(self):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trade_counts: dict[str, float] = {"ASSET_LO": 0.0, "ASSET_HI": 0.0}
        guard = AdverseSelectionGuard(
            executor, {}, event,
            trade_counts=trade_counts,
            get_position_assets=_position_assets_factory("ASSET_LO", "ASSET_HI"),
        )
        return guard

    def test_high_freq_needs_boosted_threshold(self, guard):
        """A high-frequency asset (>20 trades/min baseline) needs 7.5x
        to fire instead of 5x.  A 6x spike should NOT fire."""
        now = time.time()
        # High-freq baseline: 30 trades/min = 0.5/s over 700s = 350 trades
        guard._trade_rate_snapshots.clear()
        guard._trade_rate_snapshots.append(
            (now - 700, {"ASSET_HI": 0.0, "ASSET_LO": 0.0})
        )
        guard._trade_rate_snapshots.append(
            (now - 15, {"ASSET_HI": 342.5, "ASSET_LO": 0.0})
        )
        guard._trade_rate_snapshots.append(
            (now, {"ASSET_HI": 350.0 + 45.0, "ASSET_LO": 0.0})
        )
        # ASSET_HI: long_rate = 395/700 ≈ 0.564/s = 33.9/min (> 20 → boosted)
        #           short_rate = (395-342.5)/15 = 3.5/s
        #           ratio = 3.5/0.564 ≈ 6.2x
        #           effective threshold = 5 * 1.5 = 7.5x → NO FIRE

        fired, diag = guard._check_velocity_anomaly()
        # Should NOT fire because 6.2x < 7.5x (boosted threshold)
        hi_in_triggered = "ASSET_HI" in diag.get("triggered_assets", [])
        assert hi_in_triggered is False

    def test_low_freq_fires_at_default_threshold(self, guard):
        """A low-frequency asset (<20 trades/min baseline) uses the
        default 5x threshold.  A 6x spike should fire."""
        now = time.time()
        # Low-freq baseline: 5 trades/min = 0.083/s over 700s = 58.3 trades
        guard._trade_rate_snapshots.clear()
        guard._trade_rate_snapshots.append(
            (now - 700, {"ASSET_LO": 0.0, "ASSET_HI": 0.0})
        )
        guard._trade_rate_snapshots.append(
            (now - 15, {"ASSET_LO": 51.0, "ASSET_HI": 0.0})
        )
        guard._trade_rate_snapshots.append(
            (now, {"ASSET_LO": 58.3 + 7.5, "ASSET_HI": 0.0})
        )
        # ASSET_LO: long_rate = 65.8/700 ≈ 0.094/s = 5.64/min (< 20 → default)
        #           short_rate = (65.8-51)/15 ≈ 0.987/s
        #           ratio ≈ 10.5x → exceeds 5x → FIRE

        fired, diag = guard._check_velocity_anomaly()
        assert fired is True
        assert "ASSET_LO" in diag["triggered_assets"]


class TestKillOutcomeTracker:
    """Issue 2: Retrospective kill classification (TP/FP)."""

    @pytest.fixture
    def guard(self, tmp_path):
        executor = OrderExecutor(paper_mode=True)
        event = asyncio.Event()
        event.set()
        trackers = {
            "ASSET_A": _make_tracker("ASSET_A", bid=0.50, ask=0.52),
        }
        guard = AdverseSelectionGuard(
            executor, trackers, event,
            get_position_assets=_position_assets_factory("ASSET_A"),
        )
        guard._running = True
        guard._outcome_file = str(tmp_path / "outcomes.jsonl")
        return guard, trackers

    def test_record_true_positive(self, guard):
        """Kill is TP when price moves >= threshold (3¢)."""
        guard_obj, trackers = guard
        mids_at_kill = {"ASSET_A": 0.50}
        mids_after = {"ASSET_A": 0.54}  # +4¢ move

        guard_obj._record_kill_outcome(
            kill_number=1,
            kill_time=time.time(),
            signal_names=["flow_coherence", "depth_evaporation"],
            mids_at_kill=mids_at_kill,
            mids_after=mids_after,
        )

        with open(guard_obj._outcome_file) as f:
            line = json.loads(f.readline())
        assert line["classification"] == "true_positive"
        assert line["max_move_cents"] >= 3.0

    def test_record_false_positive(self, guard):
        """Kill is FP when price moves < threshold (3¢)."""
        guard_obj, trackers = guard
        mids_at_kill = {"ASSET_A": 0.50}
        mids_after = {"ASSET_A": 0.51}  # +1¢ move

        guard_obj._record_kill_outcome(
            kill_number=2,
            kill_time=time.time(),
            signal_names=["spread_blowout", "velocity_anomaly"],
            mids_at_kill=mids_at_kill,
            mids_after=mids_after,
        )

        with open(guard_obj._outcome_file) as f:
            line = json.loads(f.readline())
        assert line["classification"] == "false_positive"
        assert line["max_move_cents"] < 3.0

    def test_outcome_file_appendable(self, guard):
        """Multiple outcomes should append to the same JSONL file."""
        guard_obj, trackers = guard

        for i in range(3):
            guard_obj._record_kill_outcome(
                kill_number=i + 1,
                kill_time=time.time(),
                signal_names=["test"],
                mids_at_kill={"ASSET_A": 0.50},
                mids_after={"ASSET_A": 0.50 + (i + 1) * 0.01},
            )

        with open(guard_obj._outcome_file) as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 3
        assert lines[0]["kill_number"] == 1
        assert lines[2]["kill_number"] == 3

    @pytest.mark.asyncio
    async def test_outcome_recorded_after_delay(self, guard):
        """_schedule_outcome_check should trigger _record_kill_outcome
        after the configured delay."""
        guard_obj, trackers = guard
        guard_obj._outcome_delay_s = 0.1  # short delay for test
        guard_obj._cancel_count = 1

        guard_obj._schedule_outcome_check(["flow_coherence", "depth_evaporation"])

        await asyncio.sleep(0.3)

        import os
        assert os.path.exists(guard_obj._outcome_file)
        with open(guard_obj._outcome_file) as f:
            line = json.loads(f.readline())
        assert line["kill_number"] == 1
        assert line["signals"] == ["flow_coherence", "depth_evaporation"]