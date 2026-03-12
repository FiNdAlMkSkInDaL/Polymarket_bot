"""
Tests for PositionManager — stink-bid cascade harvesting via L2 liquidity gaps.

Covers:
  - find_liquidity_gaps identifies thin price levels
  - harvest_cascades places stink bids at gap levels during a PanicSignal
  - Liquidity Hole scenario: a small trade wipes out 3 ticks and the bot
    captures a fill at the 4th tick
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.data.l2_book import BookState, L2OrderBook
from src.signals.panic_detector import PanicSignal
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import Position, PositionManager, PositionState


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_l2_book(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    *,
    depth_history_avg: float | None = None,
) -> L2OrderBook:
    """Create a SYNCED L2 book with given levels and optional depth history."""
    book = L2OrderBook("ASSET_001")
    snap = {
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
        "seq": 1,
    }
    book.begin_buffering()
    # Use the synchronous internal method to avoid event-loop conflicts
    book._apply_snapshot(snap, trigger="test")
    assert book.state == BookState.SYNCED

    # Seed depth history so the 5-minute average is defined
    if depth_history_avg is not None:
        now = time.time()
        for i in range(10):
            book._depth_history.append((now - 200 + i * 20, depth_history_avg))

    return book


def _make_panic_signal(
    market_id: str = "MKT_TEST",
    no_asset_id: str = "ASSET_001",
    no_best_ask: float = 0.50,
) -> PanicSignal:
    return PanicSignal(
        market_id=market_id,
        no_asset_id=no_asset_id,
        no_best_ask=no_best_ask,
        zscore=3.0,
        volume_ratio=5.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  L2OrderBook.find_liquidity_gaps
# ═══════════════════════════════════════════════════════════════════════════

class TestFindLiquidityGaps:
    """Tests for L2OrderBook.find_liquidity_gaps."""

    def test_no_gaps_when_depth_is_thick(self):
        """All ask levels have plenty of depth — no gaps returned."""
        book = _build_l2_book(
            bids=[(0.49, 500), (0.48, 500)],
            asks=[(0.51, 500), (0.52, 500), (0.53, 500)],
            depth_history_avg=100.0,  # avg depth = $100
        )
        gaps = book.find_liquidity_gaps()
        # Cumulative depth at 0.51 = 0.51*500 = $255, well above 10% of $100
        assert gaps == []

    def test_gaps_detected_at_thin_levels(self):
        """Thin ask levels are flagged as gaps."""
        book = _build_l2_book(
            bids=[(0.49, 200)],
            asks=[
                (0.50, 1),     # cumulative = 0.50*1 = $0.50
                (0.51, 1),     # cumulative = $0.50 + $0.51 = $1.01
                (0.52, 1),     # cumulative = $1.53
                (0.53, 500),   # cumulative = $1.53 + $265 → thick
            ],
            depth_history_avg=100.0,  # threshold = 10% * $100 = $10
        )
        gaps = book.find_liquidity_gaps()
        # First 3 levels are below $10 threshold
        assert len(gaps) == 3
        assert gaps[0].price == 0.50
        assert gaps[1].price == 0.51
        assert gaps[2].price == 0.52

    def test_no_gaps_without_seeded_depth_history(self):
        """Without seeded high-average history, snapshot-bootstrapped
        depth produces a low threshold that matches the actual book."""
        book = _build_l2_book(
            bids=[(0.49, 100)],
            asks=[(0.50, 100)],  # substantial depth → cumulative above threshold
        )
        # The snapshot itself records one depth sample; threshold is
        # 10% of that sample.  With 100 shares at $0.50 → $50 ask depth,
        # plus 100 shares at $0.49 → $49 bid depth ≈ $99 total depth.
        # Threshold = 10% * $99 ≈ $9.90.  Cumul at 0.50 = $50 → above.
        gaps = book.find_liquidity_gaps()
        assert gaps == []

    def test_min_depth_usd_overrides_average(self):
        """Explicit min_depth_usd enforces an absolute floor."""
        book = _build_l2_book(
            bids=[(0.49, 100)],
            asks=[
                (0.50, 5),   # cumulative = $2.50
                (0.51, 5),   # cumulative = $5.05
                (0.52, 200), # cumulative thick
            ],
            depth_history_avg=10.0,  # 10% avg = $1 (low)
        )
        # Without override, threshold = $1 → levels not flagged (cumulative > $1)
        gaps_low = book.find_liquidity_gaps(min_depth_usd=0.0)
        # With override at $10, both thin levels should be flagged
        gaps_high = book.find_liquidity_gaps(min_depth_usd=10.0)
        assert len(gaps_high) == 2
        assert gaps_high[0].price == 0.50


# ═══════════════════════════════════════════════════════════════════════════
#  PositionManager.harvest_cascades
# ═══════════════════════════════════════════════════════════════════════════

class TestHarvestCascades:
    """Tests for PositionManager.harvest_cascades."""

    @pytest.fixture
    def pm(self):
        """PositionManager backed by a paper executor."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor, max_open_positions=10)
        pm.set_wallet_balance(1000.0)
        return pm

    @pytest.mark.asyncio
    async def test_stink_bids_placed_at_gap_levels(self, pm):
        """Stink bids are placed at gap levels within the tick-offset window."""
        best_ask = 0.50
        # Build an ask side with thin levels from 0.46..0.49,
        # then a thick level at 0.50.
        # Tick offsets 2-4 below 0.50 → 0.46, 0.47, 0.48
        book = _build_l2_book(
            bids=[(0.45, 200)],
            asks=[
                (0.46, 1),    # gap
                (0.47, 1),    # gap
                (0.48, 1),    # gap
                (0.49, 1),    # gap (but outside default window: 0.50-0.02=0.48)
                (0.50, 500),  # thick
            ],
            depth_history_avg=100.0,
        )
        signal = _make_panic_signal(no_best_ask=best_ask)

        positions = await pm.harvest_cascades(signal, book)
        # Default tick_offset_range=(2,4): prices 0.46, 0.47, 0.48
        assert len(positions) >= 1
        placed_prices = {p.entry_price for p in positions}
        # All placed prices must be within the 2-4 tick window
        for price in placed_prices:
            assert 0.46 <= price <= 0.48
        # All positions are ENTRY_PENDING
        for pos in positions:
            assert pos.state == PositionState.ENTRY_PENDING
            assert pos.id.startswith("STINK-")

    @pytest.mark.asyncio
    async def test_no_bids_when_no_gaps(self, pm):
        """No stink bids placed when the book has no liquidity gaps."""
        book = _build_l2_book(
            bids=[(0.49, 500)],
            asks=[(0.50, 500), (0.51, 500)],
            depth_history_avg=10.0,
        )
        signal = _make_panic_signal(no_best_ask=0.50)

        positions = await pm.harvest_cascades(signal, book)
        assert positions == []

    @pytest.mark.asyncio
    async def test_no_bids_when_best_ask_zero(self, pm):
        """Early return when best ask is 0 (no valid book)."""
        book = _build_l2_book(bids=[], asks=[], depth_history_avg=100.0)
        signal = _make_panic_signal(no_best_ask=0.50)

        positions = await pm.harvest_cascades(signal, book)
        assert positions == []


# ═══════════════════════════════════════════════════════════════════════════
#  Liquidity Hole scenario — small trade wipes 3 ticks, fill at 4th
# ═══════════════════════════════════════════════════════════════════════════

class TestLiquidityHoleFill:
    """End-to-end scenario: stop-loss cascade wipes thin ticks and
    the bot captures a fill at the 4th tick via a stink bid."""

    @pytest.mark.asyncio
    async def test_cascade_fill_at_4th_tick(self):
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor, max_open_positions=10)
        pm.set_wallet_balance(1000.0)

        best_ask = 0.54

        # Build a book where ticks 0.51, 0.52, 0.53 are thin
        # and 0.50 is the deep "real" support.
        # A stop-loss cascade wipes 0.51-0.53 and fills at 0.50.
        book = _build_l2_book(
            bids=[(0.49, 200), (0.48, 200)],
            asks=[
                (0.50, 2),    # thin — gap (4 ticks below 0.54)
                (0.51, 2),    # thin — gap (3 ticks below)
                (0.52, 2),    # thin — gap (2 ticks below)
                (0.53, 200),  # thick (1 tick below — outside window)
                (0.54, 500),  # best ask
            ],
            depth_history_avg=200.0,  # threshold = $20
        )

        signal = _make_panic_signal(
            no_asset_id="ASSET_001",
            no_best_ask=best_ask,
        )

        # Place stink bids at gap levels 2-4 ticks below best ask (0.50-0.52)
        positions = await pm.harvest_cascades(signal, book)
        assert len(positions) >= 1

        # Find the position placed at 4 ticks below (0.50)
        stink_at_50 = [p for p in positions if p.entry_price == 0.50]
        assert len(stink_at_50) == 1, (
            f"Expected stink bid at 0.50; got prices "
            f"{[p.entry_price for p in positions]}"
        )
        pos = stink_at_50[0]

        # Simulate the cascade: a market sell wipes through thin levels.
        # In paper mode, simulate the fill by updating the order directly.
        order = pos.entry_order
        assert order is not None
        order.status = OrderStatus.FILLED
        order.filled_size = pos.entry_size
        order.filled_avg_price = 0.50

        # Mark the position as filled
        await pm.on_entry_filled(pos)

        assert pos.state == PositionState.EXIT_PENDING
        assert pos.entry_price == 0.50
        assert pos.filled_size == pos.entry_size


# ═══════════════════════════════════════════════════════════════════════════
#  Timeout exit cooldown
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeoutCooldown:
    """Step 2 fix: timeout exits must record a cooldown to prevent
    serial re-entry on the same market."""

    def test_timeout_exit_triggers_cooldown(self):
        """After a timeout exit, is_stop_loss_cooled_down must return False."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor, max_open_positions=10)
        pm.set_wallet_balance(1000.0)

        # Create a position and simulate a filled entry
        pos = Position(
            id="POS-TIMEOUT-1",
            market_id="MKT_TIMEOUT",
            no_asset_id="ASSET_TIMEOUT",
            state=PositionState.EXIT_PENDING,
            entry_price=0.30,
            entry_size=10.0,
            filled_size=10.0,
            entry_time=time.time() - 1800,
            target_price=0.35,
            fee_enabled=True,
        )
        pm._positions[pos.id] = pos

        # Call on_exit_filled with reason="timeout"
        pm.on_exit_filled(pos, reason="timeout")

        # Cooldown must be recorded
        assert pos.market_id in pm._stop_loss_cooldowns
        assert not pm.is_stop_loss_cooled_down(pos.market_id)

    def test_normal_target_exit_no_cooldown(self):
        """A normal 'target' exit should NOT trigger a cooldown."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor, max_open_positions=10)
        pm.set_wallet_balance(1000.0)

        pos = Position(
            id="POS-TARGET-1",
            market_id="MKT_TARGET",
            no_asset_id="ASSET_TARGET",
            state=PositionState.EXIT_PENDING,
            entry_price=0.30,
            entry_size=10.0,
            filled_size=10.0,
            entry_time=time.time() - 600,
            target_price=0.35,
            fee_enabled=True,
        )
        pm._positions[pos.id] = pos

        pm.on_exit_filled(pos, reason="target")

        # No cooldown for successful TP exits
        assert pos.market_id not in pm._stop_loss_cooldowns
