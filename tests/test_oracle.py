"""
Tests for the SI-8 Oracle Latency Arbitrage system.

Covers:
  - OffChainOracleAdapter base class (polling loop, circuit breaker, dynamic interval)
  - OracleSnapshot construction
  - OracleAdapterRegistry (registration, creation, unknown type)
  - OracleSignalEngine (divergence, threshold gating, monotonic latch, confidence floor, spread gate)
  - APElectionAdapter (response parsing, confidence graduation, event phases)
  - SportsAdapter (response parsing, winner/over_goals, confidence graduation)
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import time

import pytest

# Force paper mode for tests
os.environ.setdefault("PAPER_MODE", "true")
os.environ.setdefault("DEPLOYMENT_ENV", "PAPER")

from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
from src.data.oracle_adapter import (
    OffChainOracleAdapter,
    OracleAdapterRegistry,
    OracleMarketConfig,
    OracleSnapshot,
)
from src.data.adapters.ap_election_adapter import APElectionAdapter
from src.data.adapters.odds_api_websocket_adapter import OddsAPIWebSocketAdapter
from src.data.adapters.sports_adapter import SportsAdapter
from src.data.adapters.tree_news_websocket_adapter import TreeNewsWebSocketAdapter
from src.data.adapters import websocket_adapter_base
from src.signals.oracle_signal import OracleSignalEngine


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mc(**overrides) -> OracleMarketConfig:
    """Construct a minimal OracleMarketConfig with defaults."""
    defaults = dict(
        market_id="0xDEAD",
        oracle_type="test",
        oracle_params={},
        yes_asset_id="YES_1",
        no_asset_id="NO_1",
        event_id="EVT_1",
    )
    defaults.update(overrides)
    return OracleMarketConfig(**defaults)


class _StubAdapter(OffChainOracleAdapter):
    """Concrete adapter for testing the base class polling loop."""

    def __init__(self, market_config, *, on_trip=None, snapshots=None, error_after=None):
        super().__init__(market_config, on_trip=on_trip)
        self._snapshots = list(snapshots or [])
        self._call_count = 0
        self._error_after = error_after  # raise after this many successful polls

    @property
    def name(self) -> str:
        return "stub"

    async def poll(self) -> OracleSnapshot:
        self._call_count += 1
        if self._error_after is not None and self._call_count > self._error_after:
            raise RuntimeError("simulated poll failure")
        if self._snapshots:
            return self._snapshots.pop(0)
        return OracleSnapshot(
            adapter_name="stub",
            market_id=self._config.market_id,
            event_phase="in_progress",
        )


# ═══════════════════════════════════════════════════════════════════════════
#  OracleSnapshot tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOracleSnapshot:

    def test_default_values(self):
        snap = OracleSnapshot(adapter_name="test", market_id="MKT1")
        assert snap.resolved_outcome is None
        assert snap.confidence == 0.0
        assert snap.event_phase == "pre_event"
        assert snap.raw_state == {}

    def test_populated(self):
        snap = OracleSnapshot(
            adapter_name="ap",
            market_id="0xABC",
            raw_state={"called": True},
            resolved_outcome="YES",
            confidence=0.97,
            event_phase="final",
            timestamp=123.0,
        )
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.97
        assert snap.event_phase == "final"


# ═══════════════════════════════════════════════════════════════════════════
#  OracleAdapterRegistry tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOracleAdapterRegistry:

    def test_register_and_create(self):
        reg = OracleAdapterRegistry()
        reg.register("stub", _StubAdapter)
        adapter = reg.create("stub", _mc(oracle_type="stub"))
        assert isinstance(adapter, _StubAdapter)
        assert adapter.name == "stub"

    def test_unknown_type_raises(self):
        reg = OracleAdapterRegistry()
        with pytest.raises(KeyError):
            reg.create("nonexistent", _mc())

    def test_registered_types(self):
        reg = OracleAdapterRegistry()
        assert sorted(reg.registered_types) == [
            "ap_election",
            "odds_api_ws",
            "sports",
            "tree_news_ws",
        ]


# ═══════════════════════════════════════════════════════════════════════════
#  OffChainOracleAdapter base class tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOffChainOracleAdapter:

    def test_dynamic_interval_default(self):
        adapter = _StubAdapter(_mc())
        adapter._last_phase = "in_progress"
        # Default should be oracle_default_poll_ms from config
        assert adapter.current_polling_interval_ms > 0

    def test_dynamic_interval_critical(self):
        adapter = _StubAdapter(_mc())
        adapter._last_phase = "critical"
        assert adapter.current_polling_interval_ms == 200  # oracle_critical_poll_ms

    def test_dynamic_interval_idle(self):
        adapter = _StubAdapter(_mc())
        adapter._last_phase = "idle"
        assert adapter.current_polling_interval_ms == 30000  # oracle_idle_poll_ms

    @pytest.mark.asyncio
    async def test_polling_loop_pushes_to_queue(self):
        snap = OracleSnapshot(
            adapter_name="stub",
            market_id="0xDEAD",
            resolved_outcome="YES",
            confidence=0.97,
            event_phase="final",
        )
        adapter = _StubAdapter(_mc(), snapshots=[snap])
        q: asyncio.Queue = asyncio.Queue()

        # Run the loop briefly then stop it
        async def _run_then_stop():
            await asyncio.sleep(0.05)
            adapter.stop()

        await asyncio.gather(
            adapter.start(q),
            _run_then_stop(),
        )

        assert not q.empty()
        result = q.get_nowait()
        assert result.resolved_outcome == "YES"
        assert result.timestamp > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_on_errors(self):
        trip_called = []

        async def on_trip():
            trip_called.append(True)

        adapter = _StubAdapter(
            _mc(),
            on_trip=on_trip,
            error_after=0,  # fail on first poll
        )
        # Override breaker with low threshold for fast test
        adapter._breaker = ExceptionCircuitBreaker(threshold=2, window_s=60.0)
        q: asyncio.Queue = asyncio.Queue()

        await adapter.start(q)

        assert len(trip_called) == 1
        assert adapter._breaker.tripped

    def test_stop(self):
        adapter = _StubAdapter(_mc())
        adapter._running = True
        adapter.stop()
        assert not adapter._running


class TestWebSocketOracleAdapterStart:

    @pytest.mark.asyncio
    async def test_start_streams_unique_snapshots_into_queue(self, monkeypatch):
        payload_batches = [
            [json.dumps({
                "event": {
                    "id": "evt-1",
                    "status": "IN_PLAY",
                    "minute": 20,
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "scores": [
                        {"name": "Arsenal", "score": 1},
                        {"name": "Chelsea", "score": 0},
                    ],
                },
            })],
            [json.dumps({
                "event": {
                    "id": "evt-1",
                    "status": "FINISHED",
                    "minute": 90,
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "scores": [
                        {"name": "Arsenal", "score": 2},
                        {"name": "Chelsea", "score": 0},
                    ],
                    "completed": True,
                },
            })],
        ]

        class FakeWebSocket:
            def __init__(self, messages):
                self._messages = list(messages)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._messages:
                    return self._messages.pop(0)
                raise StopAsyncIteration

            async def send(self, payload):
                return None

            async def ping(self):
                loop = asyncio.get_running_loop()
                waiter = loop.create_future()
                waiter.set_result(None)
                return waiter

        class FakeConnection:
            def __init__(self, websocket):
                self._websocket = websocket

            async def __aenter__(self):
                return self._websocket

            async def __aexit__(self, exc_type, exc, tb):
                return False

        opened = []

        def fake_connect(*args, **kwargs):
            websocket = FakeWebSocket(payload_batches[len(opened)])
            opened.append(websocket)
            return FakeConnection(websocket)

        monkeypatch.setattr(websocket_adapter_base.websockets, "connect", fake_connect)

        adapter = OddsAPIWebSocketAdapter(
            _mc(
                oracle_type="odds_api_ws",
                external_id="evt-1",
                target_outcome="Arsenal",
                market_type="winner",
                oracle_params={},
            ),
            websocket_url="ws://example.test/feed",
            reconnect_base_s=0.01,
            reconnect_max_s=0.02,
            heartbeat_interval_s=60.0,
            heartbeat_timeout_s=60.0,
        )
        queue: asyncio.Queue = asyncio.Queue()

        async def stop_after_two():
            while queue.qsize() < 2:
                await asyncio.sleep(0.01)
            adapter.stop()

        await asyncio.wait_for(asyncio.gather(adapter.start(queue), stop_after_two()), timeout=2.0)

        first = queue.get_nowait()
        second = queue.get_nowait()
        assert first.event_phase == "in_progress"
        assert second.event_phase == "final"
        assert queue.empty()


# ═══════════════════════════════════════════════════════════════════════════
#  OracleSignalEngine tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOracleSignalEngine:

    def _engine(self, **kwargs):
        defaults = dict(
            confidence_threshold=0.06,
            min_confidence=0.80,
            max_spread_cents=15.0,
        )
        defaults.update(kwargs)
        return OracleSignalEngine(**defaults)

    def test_no_signal_on_none_outcome(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome=None, confidence=0.99,
        )
        assert engine.evaluate(snap, market_price=0.50) is None

    def test_no_signal_below_confidence_floor(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="YES", confidence=0.60,
        )
        assert engine.evaluate(snap, market_price=0.50) is None

    def test_no_signal_spread_too_wide(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="YES", confidence=0.97,
        )
        assert engine.evaluate(snap, market_price=0.50, spread_cents=20.0) is None

    def test_signal_fires_on_divergence(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="YES", confidence=0.97,
            event_phase="final",
        )
        # Market at 0.50, oracle says YES with 0.97 confidence → model_prob=0.97
        # Divergence = 0.50 - 0.97 = -0.47, abs = 0.47
        # Threshold = 0.06 * (1 + 0.03) ≈ 0.0618
        sig = engine.evaluate(snap, market_price=0.50)
        assert sig is not None
        assert sig.metadata["direction"] == "buy_yes"  # price < model_prob
        assert sig.metadata["model_probability"] == pytest.approx(0.97, abs=0.01)
        assert sig.metadata["confidence"] == 0.97

    def test_buy_no_direction(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="NO", confidence=0.97,
            event_phase="final",
        )
        # resolved_outcome=NO → model_prob = 1 - 0.97 = 0.03
        # Market at 0.50, divergence = 0.50 - 0.03 = 0.47 → buy_no
        sig = engine.evaluate(snap, market_price=0.50)
        assert sig is not None
        assert sig.metadata["direction"] == "buy_no"

    def test_monotonic_latch_suppresses_duplicate(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="YES", confidence=0.97,
            event_phase="final",
        )
        sig1 = engine.evaluate(snap, market_price=0.50)
        assert sig1 is not None

        # Same outcome again — should be suppressed by latch
        sig2 = engine.evaluate(snap, market_price=0.50)
        assert sig2 is None

    def test_latch_allows_new_outcome(self):
        engine = self._engine()
        snap_yes = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="YES", confidence=0.97,
        )
        snap_no = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="NO", confidence=0.97,
        )
        sig1 = engine.evaluate(snap_yes, market_price=0.50)
        assert sig1 is not None
        sig2 = engine.evaluate(snap_no, market_price=0.50)
        assert sig2 is not None

    def test_reset_latch(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="YES", confidence=0.97,
        )
        engine.evaluate(snap, market_price=0.50)
        engine.reset_latch("MKT1")
        sig = engine.evaluate(snap, market_price=0.50)
        assert sig is not None

    def test_no_signal_below_threshold(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="test", market_id="MKT1",
            resolved_outcome="YES", confidence=0.97,
        )
        # Market at 0.96 → divergence = 0.96 - 0.97 = -0.01
        # Threshold ≈ 0.0618 → below threshold
        sig = engine.evaluate(snap, market_price=0.96)
        assert sig is None

    def test_signal_metadata_format(self):
        engine = self._engine()
        snap = OracleSnapshot(
            adapter_name="ap", market_id="MKT1",
            resolved_outcome="YES", confidence=0.97,
            event_phase="critical",
            raw_state={"key": "val"},
        )
        sig = engine.evaluate(snap, market_price=0.50)
        assert sig is not None
        assert sig.name == "oracle_ap"
        assert "model_probability" in sig.metadata
        assert "divergence" in sig.metadata
        assert "oracle_event_phase" in sig.metadata
        assert sig.metadata["oracle_event_phase"] == "critical"
        assert sig.metadata["oracle_raw_state"] == {"key": "val"}


# ═══════════════════════════════════════════════════════════════════════════
#  APElectionAdapter tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAPElectionAdapter:

    def _adapter(self, **oracle_params):
        defaults = {"race_id": "R1", "target_candidate": "Smith"}
        defaults.update(oracle_params)
        mc = _mc(oracle_type="ap_election", oracle_params=defaults)
        return APElectionAdapter(mc)

    def test_name(self):
        assert self._adapter().name == "ap_election"

    def test_parse_race_called_for_target(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "raceID": "R1",
            "reportingPct": 0.95,
            "called": True,
            "calledFor": "Smith",
            "multiDeskConsensus": False,
            "candidates": [
                {"name": "Smith", "votes": 1000000},
                {"name": "Jones", "votes": 800000},
            ],
            "status": "called",
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.97
        assert snap.event_phase == "final"

    def test_parse_race_called_for_other(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "called": True,
            "calledFor": "Jones",
            "multiDeskConsensus": False,
            "candidates": [],
        })
        assert snap.resolved_outcome == "NO"
        assert snap.confidence == 0.97

    def test_parse_multi_desk_consensus(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "called": True,
            "calledFor": "Smith",
            "multiDeskConsensus": True,
            "candidates": [],
        })
        assert snap.confidence == 0.99

    def test_parse_high_reporting_not_called(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "reportingPct": 0.92,
            "called": False,
            "candidates": [
                {"name": "Smith", "votes": 1200000},
                {"name": "Jones", "votes": 800000},
            ],
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.90
        assert snap.event_phase == "critical"

    def test_parse_mid_reporting(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "reportingPct": 0.65,
            "called": False,
            "candidates": [
                {"name": "Smith", "votes": 700000},
                {"name": "Jones", "votes": 500000},
            ],
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.82
        assert snap.event_phase == "in_progress"

    def test_parse_low_reporting(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "reportingPct": 0.30,
            "called": False,
            "candidates": [
                {"name": "Jones", "votes": 300000},
                {"name": "Smith", "votes": 200000},
            ],
        })
        assert snap.resolved_outcome == "NO"  # Jones leads, not target
        assert snap.confidence == 0.70

    def test_parse_pre_event(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "reportingPct": 0.0,
            "called": False,
            "candidates": [],
            "status": "pre_event",
        })
        assert snap.resolved_outcome is None
        assert snap.confidence == 0.0
        assert snap.event_phase == "pre_event"

    def test_interval_phases(self):
        adapter = self._adapter()
        assert adapter._compute_interval("pre_event") == 5000
        assert adapter._compute_interval("in_progress") == 2000
        assert adapter._compute_interval("critical") == 200
        assert adapter._compute_interval("final") == 1000


# ═══════════════════════════════════════════════════════════════════════════
#  SportsAdapter tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSportsAdapter:

    def _adapter(self, market_type="winner", **overrides):
        if "team" in overrides:
            overrides["target_outcome"] = overrides.pop("team")
        defaults = {
            "oracle_type": "sports",
            "external_id": "M1",
            "target_outcome": "Arsenal",
            "market_type": market_type,
            "goal_line": 2.5,
        }
        defaults.update(overrides)
        mc = _mc(**defaults)
        return SportsAdapter(mc)

    def test_name(self):
        assert self._adapter().name == "sports"

    def test_parse_in_progress_leading(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "IN_PLAY",
                "minute": 60,
                "score": {"home": 2, "away": 1},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.85
        assert snap.event_phase == "in_progress"

    def test_parse_in_progress_trailing(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "IN_PLAY",
                "minute": 60,
                "score": {"home": 0, "away": 1},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "NO"
        assert snap.confidence == 0.85

    def test_parse_drawn_no_outcome(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "IN_PLAY",
                "minute": 60,
                "score": {"home": 1, "away": 1},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome is None

    def test_parse_finished_win(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "FINISHED",
                "minute": 90,
                "score": {"home": 3, "away": 0},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.98
        assert snap.event_phase == "final"

    def test_parse_finished_loss(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "FINISHED",
                "minute": 90,
                "score": {"home": 0, "away": 2},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "NO"
        assert snap.confidence == 0.98

    def test_parse_critical_with_red_card(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "IN_PLAY",
                "minute": 85,
                "score": {"home": 2, "away": 0},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [
                    {"type": "RED_CARD", "team": "away", "minute": 65},
                ],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.92
        assert snap.event_phase == "critical"

    def test_parse_late_game_two_goal_lead(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "IN_PLAY",
                "minute": 82,
                "score": {"home": 3, "away": 1},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.90

    def test_parse_halftime(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "HALFTIME",
                "minute": 45,
                "score": {"home": 1, "away": 0},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.event_phase == "idle"

    def test_parse_pre_event(self):
        adapter = self._adapter()
        snap = adapter._parse_response({
            "match": {
                "status": "TIMED",
                "minute": 0,
                "score": {"home": 0, "away": 0},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.event_phase == "pre_event"
        assert snap.resolved_outcome is None

    def test_away_team_target(self):
        adapter = self._adapter(team="Chelsea")
        snap = adapter._parse_response({
            "match": {
                "status": "FINISHED",
                "minute": 90,
                "score": {"home": 0, "away": 2},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.98

    # ── Over goals market type ─────────────────────────────────────

    def test_over_goals_finished_above(self):
        adapter = self._adapter(market_type="over_goals", goal_line=2.5)
        snap = adapter._parse_response({
            "match": {
                "status": "FINISHED",
                "minute": 90,
                "score": {"home": 2, "away": 1},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.98

    def test_over_goals_finished_below(self):
        adapter = self._adapter(market_type="over_goals", goal_line=2.5)
        snap = adapter._parse_response({
            "match": {
                "status": "FINISHED",
                "minute": 90,
                "score": {"home": 1, "away": 0},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "NO"

    def test_over_goals_in_play_already_over(self):
        adapter = self._adapter(market_type="over_goals", goal_line=2.5)
        snap = adapter._parse_response({
            "match": {
                "status": "IN_PLAY",
                "minute": 50,
                "score": {"home": 2, "away": 1},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.85

    def test_over_goals_in_play_not_yet(self):
        adapter = self._adapter(market_type="over_goals", goal_line=2.5)
        snap = adapter._parse_response({
            "match": {
                "status": "IN_PLAY",
                "minute": 50,
                "score": {"home": 1, "away": 0},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome is None

    def test_interval_phases(self):
        adapter = self._adapter()
        assert adapter._compute_interval("pre_event") == 200
        assert adapter._compute_interval("in_progress") == 1000
        assert adapter._compute_interval("critical") == 200
        assert adapter._compute_interval("idle") == 30000
        assert adapter._compute_interval("final") == 5000


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket oracle adapter tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOddsAPIWebSocketAdapter:

    def _adapter(self, **overrides):
        mc = _mc(
            oracle_type="odds_api_ws",
            external_id="evt-1",
            target_outcome="Arsenal",
            market_type="winner",
            oracle_params={},
            **overrides,
        )
        return OddsAPIWebSocketAdapter(mc, websocket_url="ws://127.0.0.1/test")

    def test_parse_final_winner_snapshot(self):
        adapter = self._adapter()
        snap = adapter._build_snapshot({
            "match": {
                "status": "FINISHED",
                "minute": 90,
                "score": {"home": 2, "away": 0},
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "events": [],
            },
        })
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.98
        assert snap.event_phase == "final"

    @pytest.mark.asyncio
    async def test_stream_reconnects_after_disconnect(self, monkeypatch):
        class FakeWebSocket:
            def __init__(self, messages):
                self._messages = list(messages)
                self.sent = []

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._messages:
                    return self._messages.pop(0)
                raise StopAsyncIteration

            async def send(self, payload):
                self.sent.append(payload)

            async def ping(self):
                loop = asyncio.get_running_loop()
                waiter = loop.create_future()
                waiter.set_result(None)
                return waiter

        class FakeConnection:
            def __init__(self, websocket):
                self._websocket = websocket

            async def __aenter__(self):
                return self._websocket

            async def __aexit__(self, exc_type, exc, tb):
                return False

        payload_batches = [
            [json.dumps({
                "event": {
                    "id": "evt-1",
                    "status": "IN_PLAY",
                    "minute": 70,
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "scores": [
                        {"name": "Arsenal", "score": 1},
                        {"name": "Chelsea", "score": 0},
                    ],
                },
            })],
            [json.dumps({
                "event": {
                    "id": "evt-1",
                    "status": "FINISHED",
                    "minute": 90,
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "scores": [
                        {"name": "Arsenal", "score": 2},
                        {"name": "Chelsea", "score": 0},
                    ],
                    "completed": True,
                },
            })],
        ]
        opened = []

        def fake_connect(*args, **kwargs):
            websocket = FakeWebSocket(payload_batches[len(opened)])
            opened.append((args, kwargs, websocket))
            return FakeConnection(websocket)

        monkeypatch.setattr(websocket_adapter_base.websockets, "connect", fake_connect)

        adapter = OddsAPIWebSocketAdapter(
            _mc(
                oracle_type="odds_api_ws",
                external_id="evt-1",
                target_outcome="Arsenal",
                market_type="winner",
                oracle_params={},
            ),
            websocket_url="ws://example.test/feed",
            reconnect_base_s=0.01,
            reconnect_max_s=0.02,
            heartbeat_interval_s=60.0,
            heartbeat_timeout_s=60.0,
        )

        async def collect_snapshots():
            snapshots = []
            async for snapshot in adapter.stream_snapshots():
                snapshots.append(snapshot)
                if len(snapshots) == 2:
                    adapter.stop()
                    break
            return snapshots

        snapshots = await asyncio.wait_for(collect_snapshots(), timeout=2.0)

        assert [snap.resolved_outcome for snap in snapshots] == ["YES", "YES"]
        assert snapshots[0].event_phase == "in_progress"
        assert snapshots[1].event_phase == "final"
        assert len(opened) == 2


class TestTreeNewsWebSocketAdapter:

    def _adapter(self, **overrides):
        mc = _mc(
            oracle_type="tree_news_ws",
            external_id="race-1",
            target_outcome="Smith",
            oracle_params={},
            **overrides,
        )
        return TreeNewsWebSocketAdapter(mc, websocket_url="ws://127.0.0.1/test")

    @pytest.mark.asyncio
    async def test_parse_final_resolved_payload(self):
        adapter = self._adapter()
        snap = await adapter._payload_to_snapshot(None, {
            "eventId": "race-1",
            "status": "resolved",
            "outcome": {
                "label": "Smith",
                "confidence": 0.98,
            },
            "verified": True,
        })
        assert snap is not None
        assert snap.resolved_outcome == "YES"
        assert snap.confidence == 0.98
        assert snap.event_phase == "final"

    @pytest.mark.asyncio
    async def test_parse_breaking_negative_payload(self):
        adapter = self._adapter()
        snap = await adapter._payload_to_snapshot(None, {
            "eventId": "race-1",
            "status": "developing",
            "urgency": "breaking",
            "outcome": {
                "label": "Jones",
            },
            "confidenceScore": 91,
        })
        assert snap is not None
        assert snap.resolved_outcome == "NO"
        assert snap.confidence == 0.91
        assert snap.event_phase == "critical"

    @pytest.mark.asyncio
    async def test_stream_emits_keepalive_snapshot_on_successful_pong(self, monkeypatch):
        class FakeWebSocket:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Future()
                raise StopAsyncIteration

            async def send(self, payload):
                return None

            async def ping(self):
                loop = asyncio.get_running_loop()
                waiter = loop.create_future()
                waiter.set_result(None)
                return waiter

        class FakeConnection:
            def __init__(self, websocket):
                self._websocket = websocket

            async def __aenter__(self):
                return self._websocket

            async def __aexit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(
            websocket_adapter_base.websockets,
            "connect",
            lambda *args, **kwargs: FakeConnection(FakeWebSocket()),
        )

        adapter = TreeNewsWebSocketAdapter(
            _mc(
                oracle_type="tree_news_ws",
                external_id="",
                target_outcome="YES",
                oracle_params={},
            ),
            websocket_url="ws://example.test/feed",
            reconnect_base_s=0.01,
            reconnect_max_s=0.02,
            heartbeat_interval_s=0.01,
            heartbeat_timeout_s=0.5,
        )

        async def collect_first_snapshot():
            async for snapshot in adapter.stream_snapshots():
                adapter.stop()
                return snapshot
            return None

        snapshot = await asyncio.wait_for(collect_first_snapshot(), timeout=1.0)

        assert snapshot is not None
        assert snapshot.raw_state == {"type": "keepalive", "transport": "pong"}
        assert snapshot.resolved_outcome is None
        assert snapshot.confidence == 0.0
        assert snapshot.event_phase == "idle"


# ═══════════════════════════════════════════════════════════════════════════
#  Config integration test
# ═══════════════════════════════════════════════════════════════════════════

class TestOracleConfig:

    @staticmethod
    def _reload_config(monkeypatch: pytest.MonkeyPatch, **env: str | None):
        keys = {
            "ORACLE_ARB_ENABLED",
            "ORACLE_ODDS_API_WS_URL",
            "ORACLE_ODDS_API_WS_KEY",
            "ORACLE_TREE_NEWS_WS_URL",
            "ORACLE_TREE_NEWS_WS_KEY",
        }
        keys.update(env.keys())
        for key in keys:
            monkeypatch.setenv(key, "")
        for key, value in env.items():
            if value is not None:
                monkeypatch.setenv(key, value)

        import src.core.config as config_module

        return importlib.reload(config_module)

    def test_si8_config_fields_exist(self):
        from src.core.config import settings
        strat = settings.strategy
        assert hasattr(strat, "oracle_arb_enabled")
        assert hasattr(strat, "oracle_default_poll_ms")
        assert hasattr(strat, "oracle_critical_poll_ms")
        assert hasattr(strat, "oracle_idle_poll_ms")
        assert hasattr(strat, "oracle_confidence_threshold")
        assert hasattr(strat, "oracle_min_confidence")
        assert hasattr(strat, "oracle_cooldown_seconds")
        assert hasattr(strat, "oracle_max_spread_cents")
        assert hasattr(strat, "oracle_market_configs")
        assert hasattr(strat, "oracle_ap_api_key")
        assert hasattr(strat, "oracle_ap_api_url")
        assert hasattr(strat, "oracle_sports_api_key")
        assert hasattr(strat, "oracle_sports_api_url")
        assert hasattr(strat, "oracle_odds_api_ws_url")
        assert hasattr(strat, "oracle_odds_api_ws_key")
        assert hasattr(strat, "oracle_tree_news_ws_url")
        assert hasattr(strat, "oracle_tree_news_ws_key")
        assert hasattr(strat, "oracle_shadow_mode")

    def test_si8_defaults(self, monkeypatch: pytest.MonkeyPatch):
        config_module = self._reload_config(monkeypatch)
        strat = config_module.StrategyParams()
        assert strat.oracle_arb_enabled is False
        assert strat.oracle_default_poll_ms == 1000
        assert strat.oracle_critical_poll_ms == 200
        assert strat.oracle_idle_poll_ms == 30000
        assert strat.oracle_confidence_threshold == pytest.approx(0.06)
        assert strat.oracle_min_confidence == pytest.approx(0.80)
        assert strat.oracle_cooldown_seconds == 300
        assert strat.oracle_max_spread_cents == pytest.approx(15.0)
        assert strat.oracle_odds_api_ws_url == ""
        assert strat.oracle_odds_api_ws_key == ""
        assert strat.oracle_tree_news_ws_url == ""
        assert strat.oracle_tree_news_ws_key == ""
        assert strat.oracle_shadow_mode is True

    def test_si8_tree_news_only_env_override(self, monkeypatch: pytest.MonkeyPatch):
        config_module = self._reload_config(
            monkeypatch,
            ORACLE_ARB_ENABLED="true",
            ORACLE_TREE_NEWS_WS_URL="wss://news.treeofalpha.com/ws",
            ORACLE_TREE_NEWS_WS_KEY="tree-key",
        )
        strat = config_module.StrategyParams()

        assert strat.oracle_arb_enabled is True
        assert strat.oracle_tree_news_ws_url == "wss://news.treeofalpha.com/ws"
        assert strat.oracle_tree_news_ws_key == "tree-key"
        assert strat.oracle_odds_api_ws_url == ""
        assert strat.oracle_odds_api_ws_key == ""
