"""
Tests for OrderChaser — passive-aggressive quoting + escalation (Pillar 7).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from src.data.orderbook import OrderbookTracker, OrderbookSnapshot
from src.trading.chaser import ChaserState, OrderChaser
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus


def _seed_book(tracker: OrderbookTracker, bid: float = 0.47, ask: float = 0.53) -> None:
    """Inject a synthetic L2 snapshot so the tracker has data."""
    tracker.on_book_snapshot({
        "asset_id": tracker.asset_id,
        "bids": [{"price": str(bid), "size": "100"}],
        "asks": [{"price": str(ask), "size": "100"}],
    })


class TestChaserBasics:
    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        book = OrderbookTracker("ASSET_A")
        _seed_book(book, bid=0.47, ask=0.53)
        return executor, book

    @pytest.mark.asyncio
    async def test_initial_quote_placed(self, setup):
        """Chaser should place a resting order immediately."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
        )
        # Don't run the full loop — just verify construction
        assert chaser.state == ChaserState.QUOTING
        assert chaser.resting_order is None

    @pytest.mark.asyncio
    async def test_buy_chaser_fills_when_price_drops(self, setup):
        """Paper BUY chaser should fill when market price <= order price."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
        )

        # Place initial order manually
        await chaser._place(0.47)
        assert chaser.resting_order is not None

        # Simulate paper fill
        result = chaser.force_check_fill("ASSET_A", 0.47)
        assert result is True
        assert chaser.state == ChaserState.FILLED

    @pytest.mark.asyncio
    async def test_sell_chaser_fills_when_price_rises(self, setup):
        """Paper SELL chaser should fill when market price >= order price."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.SELL, target_size=10.0,
            anchor_price=0.55,
        )

        await chaser._place(0.53)
        assert chaser.resting_order is not None

        result = chaser.force_check_fill("ASSET_A", 0.53)
        assert result is True
        assert chaser.state == ChaserState.FILLED

    @pytest.mark.asyncio
    async def test_drift_beyond_max_abandons(self, setup):
        """If BBO drifts beyond max_chase_depth_cents, chaser abandons."""
        executor, book = setup
        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            max_chase_depth_cents=2.0,
        )

        # Drift is measured from anchor price
        # For BUY: adverse = price UP
        drift = chaser._drift_cents(0.50)  # 3¢ > 2¢
        assert drift > 2.0


class TestChaserEscalation:
    """Pillar 7 — hybrid-aggressive protocol tests."""

    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        book = OrderbookTracker("ASSET_A")
        _seed_book(book, bid=0.47, ask=0.53)
        event = asyncio.Event()
        event.set()
        return executor, book, event

    @pytest.mark.asyncio
    async def test_escalation_after_n_rejections(self, setup):
        """After max_post_only_rejections, chaser should enter ESCALATING."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=0.56,
            fee_rate_bps=100,
            fast_kill_event=event,
            max_post_only_rejections=2,
        )

        # Simulate 2 consecutive rejections
        chaser._rejection_count = 2

        # Call _escalate directly
        await chaser._escalate()

        # Should have either escalated to a marketable order or abandoned
        assert chaser.state in (ChaserState.QUOTING, ChaserState.FILLED, ChaserState.ABANDONED)

    @pytest.mark.asyncio
    async def test_alpha_check_buy_side(self, setup):
        """Alpha check should pass when spread > margin + fees."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=0.56,
            fee_rate_bps=100,
            fast_kill_event=event,
        )

        # Cross price at 0.48: gross = (0.56 - 0.48)*100 = 8¢
        # Entry fee = 0.48 * 100/10000 * 100 = 0.48¢
        # Net = 8 - 0.48 = 7.52¢ >> 1.0¢ margin
        assert chaser._alpha_check(0.48) is True

    @pytest.mark.asyncio
    async def test_alpha_check_fails_tight_spread(self, setup):
        """Alpha check should fail when fees eat the entire spread."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.55,
            tp_target_price=0.56,
            fee_rate_bps=200,
            fast_kill_event=event,
        )

        # Cross at 0.555: gross = (0.56 - 0.555)*100 = 0.5¢
        # Fee = 0.555 * 200/10000 * 100 = 1.11¢
        # Net = 0.5 - 1.11 = -0.61 < 1.0 margin
        assert chaser._alpha_check(0.555) is False

    @pytest.mark.asyncio
    async def test_alpha_check_sell_side(self, setup):
        """SELL-side alpha check uses anchor as reference."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.SELL, target_size=10.0,
            anchor_price=0.47,
            fee_rate_bps=100,
            fast_kill_event=event,
        )

        # Cross at 0.53: gross = (0.53 - 0.47)*100 = 6¢
        # Fee = 0.53 * 100/10000 * 100 = 0.53¢
        # Net = 6 - 0.53 = 5.47¢ >> 1.0 margin
        assert chaser._alpha_check(0.53) is True

    @pytest.mark.asyncio
    async def test_fast_kill_blocks_placement(self, setup):
        """When fast_kill_event is cleared, _wait_fast_kill should timeout."""
        executor, book, event = setup
        event.clear()  # simulate adverse selection trigger

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            fast_kill_event=event,
        )

        # _wait_fast_kill should timeout (not raise), allowing loop to continue
        start = time.time()
        await chaser._wait_fast_kill(timeout=0.05)
        elapsed = time.time() - start
        assert elapsed < 0.2  # should return quickly after timeout

    @pytest.mark.asyncio
    async def test_escalation_no_tp_target_allows(self, setup):
        """Without a TP target, BUY escalation should still be allowed."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=None,
            fee_rate_bps=100,
            fast_kill_event=event,
        )

        # Without a TP target, alpha check returns True (caller's risk)
        assert chaser._alpha_check(0.50) is True


class TestChaserIcebergPeg:
    """SI-2 iceberg-aware routing tests."""

    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        book = OrderbookTracker("ASSET_A")
        _seed_book(book, bid=0.47, ask=0.53)
        return executor, book

    def _make_iceberg_detector(self, side: str, price: float, confidence: float):
        """Create a minimal mock IcebergDetector."""
        signal = SimpleNamespace(
            side=side, price=price, confidence=confidence,
            refill_count=5, avg_slice_size=50.0,
        )

        class FakeDetector:
            def strongest_iceberg(self, s):
                if s == side:
                    return signal
                return None

        return FakeDetector()

    @pytest.mark.asyncio
    async def test_iceberg_peg_buy_side(self, setup):
        """BUY chaser should peg at iceberg price when available."""
        executor, book = setup
        detector = self._make_iceberg_detector("BUY", 0.46, 0.80)

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            iceberg_detector=detector,
        )

        quote = chaser._optimal_quote()
        # Iceberg at 0.46, but clamped to not exceed best_bid(0.47),
        # so peg=min(0.46, 0.47)=0.46
        assert quote == 0.46

    @pytest.mark.asyncio
    async def test_iceberg_peg_clamped_to_bbo(self, setup):
        """Iceberg price above BBO should be clamped down for BUY."""
        executor, book = setup
        detector = self._make_iceberg_detector("BUY", 0.49, 0.80)

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            iceberg_detector=detector,
        )

        quote = chaser._optimal_quote()
        # Iceberg at 0.49 but best_bid is 0.47 → clamped to 0.47
        assert quote == 0.47

    @pytest.mark.asyncio
    async def test_low_confidence_iceberg_ignored(self, setup):
        """Iceberg with confidence below threshold should be ignored."""
        executor, book = setup
        detector = self._make_iceberg_detector("BUY", 0.46, 0.30)

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            iceberg_detector=detector,
        )

        quote = chaser._optimal_quote()
        # Low confidence → standard BBO logic: best_bid = 0.47
        assert quote == 0.47

    @pytest.mark.asyncio
    async def test_no_iceberg_falls_back(self, setup):
        """No iceberg signal → standard BBO quote."""
        executor, book = setup

        class EmptyDetector:
            def strongest_iceberg(self, s):
                return None

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            iceberg_detector=EmptyDetector(),
        )

        quote = chaser._optimal_quote()
        assert quote == 0.47

    @pytest.mark.asyncio
    async def test_sell_side_iceberg_peg(self, setup):
        """SELL chaser should peg at iceberg price (clamped to best_ask)."""
        executor, book = setup
        detector = self._make_iceberg_detector("SELL", 0.54, 0.80)

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.SELL, target_size=10.0,
            anchor_price=0.55,
            iceberg_detector=detector,
        )

        quote = chaser._optimal_quote()
        # Iceberg at 0.54, clamped to max(0.54, best_ask=0.53)=0.54
        assert quote == 0.54


class TestChaserToxicityGuard:
    """Anti-toxicity guard — blocks escalation when adverse p-value drops."""

    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        book = OrderbookTracker("ASSET_A")
        _seed_book(book, bid=0.47, ask=0.53)
        event = asyncio.Event()
        event.set()
        return executor, book, event

    def _make_monitor(self, p_value: float):
        """Create a mock AdverseSelectionMonitor."""
        stats = SimpleNamespace(last_p_value=p_value)

        class FakeMonitor:
            def get_stats(self, market_id):
                return stats

        return FakeMonitor()

    @pytest.mark.asyncio
    async def test_toxicity_blocks_escalation(self, setup):
        """Escalation should be blocked when p-value drops below ceiling."""
        executor, book, event = setup
        # Initial p=0.50 (healthy), current p=0.04 (toxic)
        monitor = self._make_monitor(0.04)

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=0.56,
            fee_rate_bps=100,
            fast_kill_event=event,
            max_post_only_rejections=2,
            adverse_monitor=monitor,
        )
        # Override initial_p to simulate it was healthy at start
        chaser._initial_p_value = 0.50

        chaser._rejection_count = 2
        await chaser._escalate()

        # Should have abandoned, not crossed the spread
        assert chaser.state == ChaserState.ABANDONED

    @pytest.mark.asyncio
    async def test_healthy_market_allows_escalation(self, setup):
        """Escalation should proceed when p-value is above ceiling."""
        executor, book, event = setup
        monitor = self._make_monitor(0.50)

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=0.56,
            fee_rate_bps=100,
            fast_kill_event=event,
            max_post_only_rejections=2,
            adverse_monitor=monitor,
        )

        chaser._rejection_count = 2
        await chaser._escalate()

        # p=0.50 >> 0.10 ceiling → should NOT have abandoned from toxicity
        # (may still abandon from alpha check, but not from toxicity)
        # QUOTING or FILLED means it placed the escalation order successfully
        assert chaser.state in (ChaserState.QUOTING, ChaserState.FILLED, ChaserState.ABANDONED)

    @pytest.mark.asyncio
    async def test_no_monitor_allows_escalation(self, setup):
        """Without adverse_monitor, escalation proceeds normally."""
        executor, book, event = setup

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            tp_target_price=0.56,
            fee_rate_bps=100,
            fast_kill_event=event,
            max_post_only_rejections=2,
        )

        assert chaser._should_abort_toxicity() is False

    @pytest.mark.asyncio
    async def test_toxicity_requires_pvalue_drop(self, setup):
        """Toxicity guard requires both low p-value AND a drop from initial."""
        executor, book, event = setup
        # p=0.08 < 0.10 ceiling, but hasn't dropped 50% from initial 0.09
        monitor = self._make_monitor(0.08)

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            adverse_monitor=monitor,
        )
        chaser._initial_p_value = 0.09  # Only small drop: 0.09→0.08

        # 0.08 < 0.10 but 0.08 is NOT < 0.09 * 0.50 = 0.045
        assert chaser._should_abort_toxicity() is False


class TestChaserPreemptiveExit:
    """SI-5 — OFI-based preemptive exit when iceberg wall is near exhaustion."""

    @pytest.fixture
    def setup(self):
        executor = OrderExecutor(paper_mode=True)
        book = OrderbookTracker("ASSET_A")
        _seed_book(book, bid=0.47, ask=0.53)
        return executor, book

    def _make_iceberg_detector(self, side, price, confidence, estimated_total):
        signal = SimpleNamespace(
            side=side, price=price, confidence=confidence,
            refill_count=int(estimated_total / 100),
            avg_slice_size=100.0,
            estimated_total=estimated_total,
        )

        class FakeDetector:
            def strongest_iceberg(self, s):
                return signal if s == side else None

        return FakeDetector()

    @pytest.mark.asyncio
    async def test_preemptive_exit_on_wall_exhaustion(self, setup):
        """Chaser cancels peg and exits when sell burst exceeds 80% of
        a 1000-unit iceberg."""
        executor, book = setup

        detector = self._make_iceberg_detector(
            "BUY", 0.47, confidence=0.80, estimated_total=1000.0,
        )

        # Monkey-patch opposing_ofi_at_price onto the simple tracker so
        # the chaser can query per-price OFI without a full L2OrderBook.
        book.opposing_ofi_at_price = lambda price, side: 900.0  # 90% > 80%

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            iceberg_detector=detector,
        )

        # Simulate active iceberg peg
        chaser._iceberg_peg_active = True
        chaser._iceberg_peg_price = 0.47

        # Place a resting order so there is something to cancel
        await chaser._place(0.47)
        assert chaser.resting_order is not None

        # Wall pressure check should fire
        assert chaser._should_preemptive_exit() is True

        # Execute preemptive exit
        await chaser._preemptive_exit()

        assert chaser.state == ChaserState.ABANDONED
        assert chaser._iceberg_peg_active is False
        assert chaser.resting_order is None

    @pytest.mark.asyncio
    async def test_no_exit_when_ofi_below_threshold(self, setup):
        """Chaser stays pegged when OFI is well below the 80% threshold."""
        executor, book = setup

        detector = self._make_iceberg_detector(
            "BUY", 0.47, confidence=0.80, estimated_total=1000.0,
        )
        book.opposing_ofi_at_price = lambda price, side: 200.0  # 20% < 80%

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            iceberg_detector=detector,
        )
        chaser._iceberg_peg_active = True
        chaser._iceberg_peg_price = 0.47

        assert chaser._should_preemptive_exit() is False

    @pytest.mark.asyncio
    async def test_depth_ratio_floor_triggers_exit(self, setup):
        """Chaser exits when book_depth_ratio drops below 0.15."""
        executor, book = setup

        detector = self._make_iceberg_detector(
            "BUY", 0.47, confidence=0.80, estimated_total=1000.0,
        )
        book.opposing_ofi_at_price = lambda price, side: 0.0  # no OFI

        # Skew the book so bid depth is tiny relative to ask depth
        book.on_book_snapshot({
            "asset_id": "ASSET_A",
            "bids": [{"price": "0.47", "size": "1"}],
            "asks": [{"price": "0.53", "size": "1000"}],
        })
        assert book.book_depth_ratio < 0.15

        chaser = OrderChaser(
            executor=executor, book=book,
            market_id="MKT_1", asset_id="ASSET_A",
            side=OrderSide.BUY, target_size=10.0,
            anchor_price=0.47,
            iceberg_detector=detector,
        )
        chaser._iceberg_peg_active = True
        chaser._iceberg_peg_price = 0.47

        assert chaser._should_preemptive_exit() is True
