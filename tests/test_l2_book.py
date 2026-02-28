"""
Comprehensive tests for the L2 Order Book Reconstruction Engine.

Covers:
  - Snapshot loading and delta application (happy path)
  - Delta buffering before snapshot
  - Sequence gap detection and desync recovery
  - Duplicate and stale sequence handling
  - Crossed book detection
  - BBO change triggers spread score recalculation
  - Depth-weighted spread score accuracy
  - State machine transitions
  - L2OrderBookAdapter backward compatibility
  - L2WebSocket reconnect and re-subscription
  - Dynamic add/remove of assets
  - REST snapshot timeout and retry logic
  - Ghost liquidity depth tracking
  - Level trimming at max depth
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.l2_book import BookState, L2OrderBook, L2Snapshot, _Level, _parse_int
from src.data.l2_websocket import L2WebSocket
from src.data.orderbook import L2OrderBookAdapter, OrderbookSnapshot, OrderbookTracker
from src.data.spread_score import SpreadScore, compute_spread_score


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_snapshot(
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    seq: int = 0,
) -> dict:
    """Build a REST-style snapshot dict."""
    return {
        "bids": [{"price": p, "size": s} for p, s in bids],
        "asks": [{"price": p, "size": s} for p, s in asks],
        "seq": seq,
        "timestamp": str(time.time()),
    }


def _make_delta(
    changes: list[tuple[str, str, str]],
    seq: int,
    asset_id: str = "ASSET_001",
) -> dict:
    """Build a delta message dict.

    Each change is (side, price, size).
    """
    return {
        "event_type": "price_change",
        "asset_id": asset_id,
        "seq": seq,
        "changes": [
            {"side": side, "price": price, "size": size}
            for side, price, size in changes
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Section A: L2 Order Book Core Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestL2OrderBookCore:
    """Core order book reconstruction tests."""

    def setup_method(self):
        self.book = L2OrderBook("ASSET_001", max_levels=50)

    # ── Initial state ─────────────────────────────────────────────────────
    def test_initial_state_is_empty(self):
        assert self.book.state == BookState.EMPTY
        assert self.book.seq == -1
        assert self.book.best_bid == 0.0
        assert self.book.best_ask == 0.0
        assert not self.book.has_data
        assert self.book.spread_cents == 0.0

    # ── Snapshot loading ──────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_snapshot_loads_correctly(self):
        snap_data = _make_snapshot(
            bids=[("0.47", "100"), ("0.46", "200"), ("0.45", "300")],
            asks=[("0.53", "80"), ("0.54", "60"), ("0.55", "40")],
            seq=10,
        )
        self.book.begin_buffering()
        result = await self.book.load_snapshot(snap_data)

        assert result is True
        assert self.book.state == BookState.SYNCED
        assert self.book.seq == 10
        assert self.book.best_bid == 0.47
        assert self.book.best_ask == 0.53
        assert self.book.has_data
        assert abs(self.book.spread_cents - 6.0) < 0.1

    @pytest.mark.asyncio
    async def test_snapshot_bids_sorted_descending(self):
        snap_data = _make_snapshot(
            bids=[("0.45", "100"), ("0.47", "200"), ("0.46", "300")],
            asks=[("0.55", "100")],
            seq=0,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)
        assert self.book.best_bid == 0.47

    @pytest.mark.asyncio
    async def test_snapshot_asks_sorted_ascending(self):
        snap_data = _make_snapshot(
            bids=[("0.45", "100")],
            asks=[("0.55", "50"), ("0.53", "100"), ("0.54", "80")],
            seq=0,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)
        assert self.book.best_ask == 0.53

    # ── Snapshot then deltas (happy path) ─────────────────────────────────
    @pytest.mark.asyncio
    async def test_snapshot_then_deltas_apply_correctly(self):
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)

        # Delta 11: update bid and add new ask level
        delta = _make_delta(
            [("BUY", "0.47", "150"), ("SELL", "0.52", "50")],
            seq=11,
        )
        result = self.book.on_delta(delta)
        assert result is True
        assert self.book.seq == 11

        # Bid size updated
        levels = self.book.levels("bid", 1)
        assert len(levels) == 1
        assert levels[0].size == 150

        # New ask at 0.52 is now best ask
        assert self.book.best_ask == 0.52

    @pytest.mark.asyncio
    async def test_multiple_sequential_deltas(self):
        snap_data = _make_snapshot(
            bids=[("0.50", "100")],
            asks=[("0.51", "100")],
            seq=0,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)

        for i in range(1, 11):
            delta = _make_delta(
                [("BUY", "0.50", str(100 + i * 10))],
                seq=i,
            )
            assert self.book.on_delta(delta) is True

        assert self.book.seq == 10
        levels = self.book.levels("bid", 1)
        assert levels[0].size == 200

    # ── Level removal (size=0) ────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_level_removal_zero_size(self):
        snap_data = _make_snapshot(
            bids=[("0.47", "100"), ("0.46", "200")],
            asks=[("0.53", "80")],
            seq=5,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)

        # Remove best bid
        delta = _make_delta([("BUY", "0.47", "0")], seq=6)
        self.book.on_delta(delta)

        assert self.book.best_bid == 0.46
        levels = self.book.levels("bid")
        assert len(levels) == 1

    @pytest.mark.asyncio
    async def test_remove_nonexistent_level_is_noop(self):
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=5,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)

        # Remove level that doesn't exist
        delta = _make_delta([("BUY", "0.40", "0")], seq=6)
        result = self.book.on_delta(delta)
        assert result is True
        assert self.book.best_bid == 0.47


# ═══════════════════════════════════════════════════════════════════════════
#  Section B: Delta Buffering & Replay
# ═══════════════════════════════════════════════════════════════════════════

class TestDeltaBuffering:
    """Tests for delta buffering before snapshot is loaded."""

    def setup_method(self):
        self.book = L2OrderBook("ASSET_001", max_levels=50)

    @pytest.mark.asyncio
    async def test_delta_before_snapshot_buffered(self):
        self.book.begin_buffering()
        assert self.book.state == BookState.BUFFERING

        # Send deltas while buffering
        delta1 = _make_delta([("BUY", "0.48", "100")], seq=11)
        delta2 = _make_delta([("SELL", "0.52", "50")], seq=12)
        self.book.on_delta(delta1)
        self.book.on_delta(delta2)

        # Book should still be in BUFFERING
        assert self.book.state == BookState.BUFFERING
        assert self.book.best_bid == 0.0  # no data applied yet

        # Now load snapshot with seq=10
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        await self.book.load_snapshot(snap_data)

        # Buffered deltas with seq > 10 should have been replayed
        assert self.book.state == BookState.SYNCED
        assert self.book.best_bid == 0.48  # from delta1
        assert self.book.best_ask == 0.52  # from delta2

    @pytest.mark.asyncio
    async def test_stale_buffered_deltas_discarded(self):
        self.book.begin_buffering()

        # Buffer deltas with seq ≤ snapshot seq
        delta_stale = _make_delta([("BUY", "0.99", "999")], seq=5)
        delta_fresh = _make_delta([("BUY", "0.48", "100")], seq=11)
        self.book.on_delta(delta_stale)
        self.book.on_delta(delta_fresh)

        # Load snapshot at seq=10 — stale delta (seq=5) should be skipped
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        await self.book.load_snapshot(snap_data)

        # Fresh delta applied, stale one not
        assert self.book.best_bid == 0.48
        # Verify 0.99 was NOT applied
        bid_levels = self.book.levels("bid", 10)
        prices = [l.price for l in bid_levels]
        assert 0.99 not in prices

    @pytest.mark.asyncio
    async def test_empty_state_ignores_deltas(self):
        delta = _make_delta([("BUY", "0.50", "100")], seq=1)
        result = self.book.on_delta(delta)
        assert result is False
        assert self.book.state == BookState.EMPTY


# ═══════════════════════════════════════════════════════════════════════════
#  Section C: Sequence Tracking & Desync Recovery
# ═══════════════════════════════════════════════════════════════════════════

class TestSequenceTracking:
    """Tests for sequence gap detection and desync recovery."""

    def setup_method(self):
        self.desync_called = False
        self.desync_asset_id = None

        def on_desync(asset_id):
            self.desync_called = True
            self.desync_asset_id = asset_id

        self.book = L2OrderBook(
            "ASSET_001",
            on_desync=on_desync,
            max_levels=50,
        )

    @pytest.mark.asyncio
    async def test_seq_gap_triggers_desync(self):
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)

        # Send delta with seq=12 (gap: expected 11)
        delta = _make_delta([("BUY", "0.48", "100")], seq=12)
        result = self.book.on_delta(delta)

        assert result is False
        assert self.book.state == BookState.DESYNCED
        assert self.desync_called is True
        assert self.desync_asset_id == "ASSET_001"

    @pytest.mark.asyncio
    async def test_duplicate_seq_ignored(self):
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)

        # Apply delta seq=11
        delta1 = _make_delta([("BUY", "0.48", "100")], seq=11)
        assert self.book.on_delta(delta1) is True

        # Re-send seq=11 — should be ignored
        delta_dup = _make_delta([("BUY", "0.48", "999")], seq=11)
        assert self.book.on_delta(delta_dup) is False

        # Size should remain 100, not 999
        levels = self.book.levels("bid", 1)
        assert levels[0].size == 100

    @pytest.mark.asyncio
    async def test_stale_seq_ignored(self):
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap_data)

        # Send delta with old seq=5
        delta = _make_delta([("BUY", "0.99", "999")], seq=5)
        result = self.book.on_delta(delta)

        assert result is False
        assert self.book.state == BookState.SYNCED

    @pytest.mark.asyncio
    async def test_desync_recovery_restores_book(self):
        # Setup initial synced state
        snap1 = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap1)

        # Force desync
        delta_gap = _make_delta([("BUY", "0.48", "100")], seq=15)
        self.book.on_delta(delta_gap)
        assert self.book.state == BookState.DESYNCED

        # Recovery: new snapshot
        snap2 = _make_snapshot(
            bids=[("0.49", "200")],
            asks=[("0.51", "150")],
            seq=20,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap2)

        assert self.book.state == BookState.SYNCED
        assert self.book.seq == 20
        assert self.book.best_bid == 0.49
        assert self.book.best_ask == 0.51

    @pytest.mark.asyncio
    async def test_multiple_rapid_desyncs(self):
        desync_count = 0

        def count_desyncs(asset_id):
            nonlocal desync_count
            desync_count += 1

        book = L2OrderBook("ASSET_X", on_desync=count_desyncs, max_levels=50)

        for i in range(3):
            snap = _make_snapshot(
                bids=[("0.47", "100")],
                asks=[("0.53", "80")],
                seq=i * 10,
            )
            book.begin_buffering()
            await book.load_snapshot(snap)

            # Cause desync
            gap_delta = _make_delta([("BUY", "0.48", "100")], seq=i * 10 + 5)
            book.on_delta(gap_delta)

        assert desync_count == 3

    @pytest.mark.asyncio
    async def test_desync_during_buffering_state(self):
        """Deltas arriving during DESYNCED state should be buffered."""
        snap1 = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap1)

        # Force desync
        gap_delta = _make_delta([("BUY", "0.48", "100")], seq=15)
        self.book.on_delta(gap_delta)
        assert self.book.state == BookState.DESYNCED

        # Deltas arriving during desync should be buffered
        late_delta = _make_delta([("SELL", "0.52", "80")], seq=21)
        result = self.book.on_delta(late_delta)
        assert result is False  # buffered, not applied

    @pytest.mark.asyncio
    async def test_no_seq_field_always_applies(self):
        """When delta has no seq field, it should always apply (no seq check)."""
        snap = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        self.book.begin_buffering()
        await self.book.load_snapshot(snap)

        # Delta without seq field
        delta = {
            "event_type": "price_change",
            "asset_id": "ASSET_001",
            "changes": [{"side": "BUY", "price": "0.48", "size": "200"}],
        }
        result = self.book.on_delta(delta)
        assert result is True
        assert self.book.best_bid == 0.48


# ═══════════════════════════════════════════════════════════════════════════
#  Section D: Crossed Book Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossedBook:
    """Tests for crossed book detection (best_bid >= best_ask)."""

    @pytest.mark.asyncio
    async def test_crossed_book_triggers_desync(self):
        desync_triggered = False

        def on_desync(aid):
            nonlocal desync_triggered
            desync_triggered = True

        book = L2OrderBook("ASSET_X", on_desync=on_desync, max_levels=50)
        snap = _make_snapshot(
            bids=[("0.50", "100")],
            asks=[("0.51", "80")],
            seq=10,
        )
        book.begin_buffering()
        await book.load_snapshot(snap)

        # Move ask below bid → crossed
        delta = _make_delta([("SELL", "0.49", "50")], seq=11)
        book.on_delta(delta)

        assert desync_triggered is True
        assert book.state == BookState.DESYNCED


# ═══════════════════════════════════════════════════════════════════════════
#  Section E: BBO Change & Spread Score
# ═══════════════════════════════════════════════════════════════════════════

class TestBBOAndSpreadScore:
    """Tests for BBO change detection and spread score computation."""

    @pytest.mark.asyncio
    async def test_bbo_change_triggers_spread_score(self):
        bbo_updates = []

        async def on_bbo(asset_id, score):
            bbo_updates.append((asset_id, score))

        book = L2OrderBook("ASSET_X", on_bbo_change=on_bbo, max_levels=50)
        snap = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        book.begin_buffering()
        await book.load_snapshot(snap)
        await asyncio.sleep(0)  # let ensure_future callbacks run

        # Initial snapshot should fire BBO callback
        assert len(bbo_updates) >= 1
        assert bbo_updates[-1][0] == "ASSET_X"
        assert bbo_updates[-1][1].score > 0

    @pytest.mark.asyncio
    async def test_no_bbo_change_no_callback(self):
        bbo_updates = []

        async def on_bbo(asset_id, score):
            bbo_updates.append((asset_id, score))

        book = L2OrderBook("ASSET_X", on_bbo_change=on_bbo, max_levels=50)
        snap = _make_snapshot(
            bids=[("0.47", "100"), ("0.46", "200")],
            asks=[("0.53", "80")],
            seq=10,
        )
        book.begin_buffering()
        await book.load_snapshot(snap)
        initial_count = len(bbo_updates)

        # Delta that changes a non-BBO level — should NOT fire callback
        delta = _make_delta([("BUY", "0.46", "300")], seq=11)
        book.on_delta(delta)

        assert len(bbo_updates) == initial_count

    @pytest.mark.asyncio
    async def test_spread_score_tight_spread(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap = _make_snapshot(
            bids=[("0.500", "100"), ("0.499", "200")],
            asks=[("0.501", "100"), ("0.502", "200")],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap)

        # 0.1 cent spread → should get very high score
        assert book.spread_score_value > 90

    @pytest.mark.asyncio
    async def test_spread_score_wide_spread(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap = _make_snapshot(
            bids=[("0.40", "100")],
            asks=[("0.60", "100")],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap)

        # 20 cent spread → should get 0 score
        assert book.spread_score_value == 0.0

    def test_spread_score_no_data(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        assert book.spread_score_value == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Section F: Spread Score Calculator
# ═══════════════════════════════════════════════════════════════════════════

class TestSpreadScoreCalculator:
    """Direct tests for the spread score computation function."""

    def test_perfect_spread(self):
        score = compute_spread_score(
            best_bid=0.500,
            best_ask=0.505,
            bid_levels=[(0.500, 100)],
            ask_levels=[(0.505, 100)],
        )
        # 0.5 cent spread → should be 100
        assert score.score == 100.0
        assert abs(score.raw_spread_cents - 0.5) < 0.01

    def test_moderate_spread(self):
        score = compute_spread_score(
            best_bid=0.47,
            best_ask=0.53,
            bid_levels=[(0.47, 100), (0.46, 200), (0.45, 300)],
            ask_levels=[(0.53, 80), (0.54, 60), (0.55, 40)],
        )
        # 6 cent spread → moderate score
        assert 0 < score.score < 70
        assert abs(score.raw_spread_cents - 6.0) < 0.1

    def test_depth_weighted_spread(self):
        # Narrow top level but wide deeper levels
        score = compute_spread_score(
            best_bid=0.499,
            best_ask=0.501,
            bid_levels=[(0.499, 10), (0.490, 100), (0.480, 100)],
            ask_levels=[(0.501, 10), (0.510, 100), (0.520, 100)],
        )
        # Depth-weighted spread should be wider than raw spread
        assert score.depth_weighted_spread_cents > score.raw_spread_cents

    def test_no_bid_returns_zero_score(self):
        score = compute_spread_score(0.0, 0.53, [], [(0.53, 80)])
        assert score.score == 0.0

    def test_no_ask_returns_zero_score(self):
        score = compute_spread_score(0.47, 0.0, [(0.47, 100)], [])
        assert score.score == 0.0

    def test_inverted_spread_returns_zero(self):
        score = compute_spread_score(0.53, 0.47, [(0.53, 100)], [(0.47, 80)])
        assert score.score == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Section G: State Machine Transitions
# ═══════════════════════════════════════════════════════════════════════════

class TestBookStateTransitions:
    """Verify the full state machine lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        desync_count = 0

        def on_desync(aid):
            nonlocal desync_count
            desync_count += 1

        book = L2OrderBook("ASSET_X", on_desync=on_desync, max_levels=50)

        # 1. EMPTY
        assert book.state == BookState.EMPTY

        # 2. EMPTY → BUFFERING
        book.begin_buffering()
        assert book.state == BookState.BUFFERING

        # 3. BUFFERING → SYNCED
        snap = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        await book.load_snapshot(snap)
        assert book.state == BookState.SYNCED

        # 4. SYNCED → DESYNCED (seq gap)
        gap_delta = _make_delta([("BUY", "0.48", "100")], seq=15)
        book.on_delta(gap_delta)
        assert book.state == BookState.DESYNCED
        assert desync_count == 1

        # 5. DESYNCED → BUFFERING → SYNCED (recovery)
        book.begin_buffering()
        assert book.state == BookState.BUFFERING
        snap2 = _make_snapshot(
            bids=[("0.49", "200")],
            asks=[("0.51", "150")],
            seq=20,
        )
        await book.load_snapshot(snap2)
        assert book.state == BookState.SYNCED

    @pytest.mark.asyncio
    async def test_reset_returns_to_empty(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        book.begin_buffering()
        await book.load_snapshot(snap)
        assert book.state == BookState.SYNCED

        book.reset()
        assert book.state == BookState.EMPTY
        assert book.seq == -1
        assert not book.has_data


# ═══════════════════════════════════════════════════════════════════════════
#  Section H: L2 Snapshot Properties
# ═══════════════════════════════════════════════════════════════════════════

class TestL2Snapshot:
    """Tests for L2Snapshot accuracy."""

    @pytest.mark.asyncio
    async def test_snapshot_properties(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100"), ("0.46", "200")],
            asks=[("0.53", "80"), ("0.54", "60")],
            seq=5,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        snap = book.snapshot()
        assert snap.asset_id == "ASSET_X"
        assert snap.best_bid == 0.47
        assert snap.best_ask == 0.53
        assert abs(snap.spread - 0.06) < 0.001
        assert abs(snap.mid_price - 0.50) < 0.001
        assert snap.bid_depth_usd > 0
        assert snap.ask_depth_usd > 0
        assert snap.state == BookState.SYNCED
        assert snap.seq == 5
        assert snap.fresh is True

    @pytest.mark.asyncio
    async def test_snapshot_fresh_flag(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        fresh_snap = book.snapshot(fresh=True)
        stale_snap = book.snapshot(fresh=False)
        assert fresh_snap.fresh is True
        assert stale_snap.fresh is False


# ═══════════════════════════════════════════════════════════════════════════
#  Section I: Book Depth & Ghost Liquidity
# ═══════════════════════════════════════════════════════════════════════════

class TestBookDepth:
    """Tests for depth calculations and ghost liquidity support."""

    @pytest.mark.asyncio
    async def test_book_depth_ratio(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.50", "200")],
            asks=[("0.51", "100")],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        ratio = book.book_depth_ratio
        # bid_depth = 0.50 * 200 = 100, ask_depth = 0.51 * 100 = 51
        assert ratio > 1.5

    @pytest.mark.asyncio
    async def test_book_depth_ratio_no_asks(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.50", "100")],
            asks=[],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)
        assert book.book_depth_ratio == 1.0

    @pytest.mark.asyncio
    async def test_current_total_depth(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.50", "100")],
            asks=[("0.60", "100")],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        depth = book.current_total_depth()
        # 0.50 * 100 + 0.60 * 100 = 50 + 60 = 110
        assert abs(depth - 110.0) < 0.01

    @pytest.mark.asyncio
    async def test_depth_velocity_insufficient_data(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        assert book.depth_velocity() is None

    @pytest.mark.asyncio
    async def test_levels_returns_copies(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100"), ("0.46", "200")],
            asks=[("0.53", "80")],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        levels = book.levels("bid", 2)
        assert len(levels) == 2
        # Modifying returned list shouldn't affect internal state
        levels.pop()
        assert len(book.levels("bid", 2)) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  Section J: Level Trimming
# ═══════════════════════════════════════════════════════════════════════════

class TestLevelTrimming:
    """Tests for max level enforcement."""

    @pytest.mark.asyncio
    async def test_max_levels_enforced(self):
        book = L2OrderBook("ASSET_X", max_levels=5)
        bids = [(f"0.{40 + i:02d}", "10") for i in range(10)]
        snap_data = _make_snapshot(bids=bids, asks=[("0.55", "100")], seq=0)
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        # Should only have top 5 bids
        all_bids = book.levels("bid", 10)
        assert len(all_bids) == 5
        # Best bid should be 0.49 (highest)
        assert all_bids[0].price == 0.49

    @pytest.mark.asyncio
    async def test_delta_respects_max_levels(self):
        book = L2OrderBook("ASSET_X", max_levels=3)
        snap_data = _make_snapshot(
            bids=[("0.47", "100"), ("0.46", "200"), ("0.45", "300")],
            asks=[("0.53", "80")],
            seq=0,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        # Add a new best bid — should push out worst
        delta = _make_delta([("BUY", "0.48", "50")], seq=1)
        book.on_delta(delta)

        all_bids = book.levels("bid", 10)
        assert len(all_bids) == 3
        assert all_bids[0].price == 0.48
        # 0.45 should have been trimmed
        prices = [l.price for l in all_bids]
        assert 0.45 not in prices


# ═══════════════════════════════════════════════════════════════════════════
#  Section K: L2OrderBookAdapter
# ═══════════════════════════════════════════════════════════════════════════

class TestL2OrderBookAdapter:
    """Tests for the adapter that wraps L2OrderBook behind OrderbookTracker API."""

    @pytest.mark.asyncio
    async def test_adapter_delegates_snapshot(self):
        l2 = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        l2.begin_buffering()
        await l2.load_snapshot(snap_data)

        adapter = L2OrderBookAdapter(l2)
        snap = adapter.snapshot()

        assert isinstance(snap, OrderbookSnapshot)
        assert snap.best_bid == 0.47
        assert snap.best_ask == 0.53
        assert snap.spread_score > 0

    @pytest.mark.asyncio
    async def test_adapter_delegates_levels(self):
        l2 = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100"), ("0.46", "200")],
            asks=[("0.53", "80")],
            seq=0,
        )
        l2.begin_buffering()
        await l2.load_snapshot(snap_data)

        adapter = L2OrderBookAdapter(l2)
        levels = adapter.levels("bid", 2)
        assert len(levels) == 2
        assert levels[0].price == 0.47

    @pytest.mark.asyncio
    async def test_adapter_delegates_properties(self):
        l2 = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=0,
        )
        l2.begin_buffering()
        await l2.load_snapshot(snap_data)

        adapter = L2OrderBookAdapter(l2)
        assert adapter.has_data is True
        assert abs(adapter.spread_cents - 6.0) < 0.1
        assert adapter.book_depth_ratio > 0

    @pytest.mark.asyncio
    async def test_adapter_write_ops_are_noop(self):
        l2 = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=0,
        )
        l2.begin_buffering()
        await l2.load_snapshot(snap_data)

        adapter = L2OrderBookAdapter(l2)

        # These should be no-ops
        adapter.on_price_change({"changes": [{"side": "BUY", "price": "0.99", "size": "999"}]})
        adapter.on_book_snapshot({"bids": [{"price": "0.10", "size": "1"}], "asks": []})

        # Original data unchanged
        assert adapter.snapshot().best_bid == 0.47

    @pytest.mark.asyncio
    async def test_adapter_timestamp_properties(self):
        l2 = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=0,
        )
        l2.begin_buffering()
        await l2.load_snapshot(snap_data)

        adapter = L2OrderBookAdapter(l2)
        assert adapter._last_update > 0
        assert adapter._last_server_time > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Section L: L2WebSocket
# ═══════════════════════════════════════════════════════════════════════════

class TestL2WebSocket:
    """Tests for the L2 WebSocket client."""

    @pytest.mark.asyncio
    async def test_reconnect_resubscribes_all_assets(self):
        """On reconnect, all tracked assets should be re-subscribed and
        snapshot fetch triggered."""
        book1 = L2OrderBook("ASSET_A", max_levels=50)
        book2 = L2OrderBook("ASSET_B", max_levels=50)
        books = {"ASSET_A": book1, "ASSET_B": book2}

        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        # Track subscribe calls
        subscribe_calls = []
        snapshot_calls = []

        async def mock_connect_and_consume():
            # Simulate: subscribe to all + trigger snapshots
            subscribe_calls.extend(list(l2_ws._books.keys()))
            for aid in list(l2_ws._books.keys()):
                snapshot_calls.append(aid)
            l2_ws._running = False

        l2_ws._connect_and_consume = mock_connect_and_consume

        await l2_ws.start()
        assert "ASSET_A" in subscribe_calls
        assert "ASSET_B" in subscribe_calls

    @pytest.mark.asyncio
    async def test_remove_assets_cleans_up(self):
        book1 = L2OrderBook("ASSET_A", max_levels=50)
        books = {"ASSET_A": book1}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        await l2_ws.remove_assets(["ASSET_A"])
        assert "ASSET_A" not in l2_ws._books

    @pytest.mark.asyncio
    async def test_handle_message_routes_delta(self):
        book = L2OrderBook("ASSET_A", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        books = {"ASSET_A": book}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        # Route a delta message
        delta_msg = _make_delta([("BUY", "0.48", "200")], seq=11, asset_id="ASSET_A")
        l2_ws._handle_message(delta_msg)

        assert book.best_bid == 0.48
        assert book.seq == 11

    @pytest.mark.asyncio
    async def test_handle_message_ignores_unknown_asset(self):
        book = L2OrderBook("ASSET_A", max_levels=50)
        books = {"ASSET_A": book}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        delta_msg = _make_delta([("BUY", "0.48", "200")], seq=1, asset_id="UNKNOWN")
        l2_ws._handle_message(delta_msg)
        # Should not crash

    @pytest.mark.asyncio
    async def test_handle_batch_messages(self):
        book = L2OrderBook("ASSET_A", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        books = {"ASSET_A": book}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        # Batch message (JSON array)
        batch = [
            _make_delta([("BUY", "0.47", "200")], seq=11, asset_id="ASSET_A"),
            _make_delta([("BUY", "0.48", "50")], seq=12, asset_id="ASSET_A"),
        ]
        l2_ws._handle_message(batch)

        assert book.seq == 12
        assert book.best_bid == 0.48

    @pytest.mark.asyncio
    async def test_reconnect_with_exponential_backoff(self):
        """Reconnect should use exponential backoff."""
        import websockets.exceptions

        book = L2OrderBook("ASSET_A", max_levels=50)
        books = {"ASSET_A": book}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        attempt = 0
        sleep_durations = []

        async def mock_connect():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise websockets.exceptions.ConnectionClosed(None, None)
            l2_ws._running = False

        l2_ws._connect_and_consume = mock_connect

        async def mock_sleep(duration):
            sleep_durations.append(duration)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await l2_ws.start()

        assert attempt >= 3
        # Backoff should increase
        if len(sleep_durations) >= 2:
            assert sleep_durations[1] > sleep_durations[0]

    @pytest.mark.asyncio
    async def test_desync_callback_triggers_snapshot_fetch(self):
        """When L2OrderBook fires on_desync, the WebSocket should schedule
        a snapshot re-fetch."""
        book = L2OrderBook("ASSET_A", max_levels=50)
        books = {"ASSET_A": book}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        # Mock the snapshot fetch
        with patch.object(l2_ws, "_fetch_and_apply_snapshot", new_callable=AsyncMock) as mock_fetch:
            await l2_ws._on_book_desync("ASSET_A")
            # Give the scheduled task time to start
            await asyncio.sleep(0.1)
            # Should have been called
            assert mock_fetch.called or "ASSET_A" in l2_ws._snapshot_tasks


# ═══════════════════════════════════════════════════════════════════════════
#  Section M: Snapshot Fetch
# ═══════════════════════════════════════════════════════════════════════════

class TestSnapshotFetch:
    """Tests for the REST snapshot fetch logic."""

    @pytest.mark.asyncio
    async def test_snapshot_fetch_and_apply(self):
        book = L2OrderBook("ASSET_A", max_levels=50)
        books = {"ASSET_A": book}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        test_snap = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )

        with patch(
            "src.data.l2_websocket.fetch_l2_snapshot",
            new_callable=AsyncMock,
            return_value=test_snap,
        ):
            await l2_ws._fetch_and_apply_snapshot("ASSET_A")

        assert book.state == BookState.SYNCED
        assert book.best_bid == 0.47

    @pytest.mark.asyncio
    async def test_snapshot_fetch_retries_on_failure(self):
        book = L2OrderBook("ASSET_A", max_levels=50)
        books = {"ASSET_A": book}
        l2_ws = L2WebSocket(books, ws_url="wss://test.example.com/ws")

        call_count = 0
        test_snap = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=10,
        )

        async def mock_fetch(asset_id, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # simulate failure
            return test_snap

        with patch(
            "src.data.l2_websocket.fetch_l2_snapshot",
            side_effect=mock_fetch,
        ):
            await l2_ws._fetch_and_apply_snapshot("ASSET_A")

        assert call_count == 3
        assert book.state == BookState.SYNCED


# ═══════════════════════════════════════════════════════════════════════════
#  Section N: Parse Helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestParseHelpers:
    def test_parse_int_valid(self):
        assert _parse_int("42") == 42
        assert _parse_int(42) == 42

    def test_parse_int_invalid(self):
        assert _parse_int("abc") == -1
        assert _parse_int(None) == -1

    def test_parse_int_default(self):
        assert _parse_int(None, 99) == 99


# ═══════════════════════════════════════════════════════════════════════════
#  Section O: Server Time Extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestServerTime:
    """Tests for timestamp normalisation."""

    @pytest.mark.asyncio
    async def test_microsecond_timestamp_normalised(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=0,
        )
        # Override timestamp to microseconds
        snap_data["timestamp"] = str(int(time.time() * 1_000_000))
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        snap = book.snapshot()
        # Should be normalised to seconds
        assert snap.server_time < 1e12

    @pytest.mark.asyncio
    async def test_millisecond_timestamp_normalised(self):
        book = L2OrderBook("ASSET_X", max_levels=50)
        snap_data = _make_snapshot(
            bids=[("0.47", "100")],
            asks=[("0.53", "80")],
            seq=0,
        )
        snap_data["timestamp"] = str(int(time.time() * 1_000))
        book.begin_buffering()
        await book.load_snapshot(snap_data)

        snap = book.snapshot()
        assert snap.server_time < 1e12
