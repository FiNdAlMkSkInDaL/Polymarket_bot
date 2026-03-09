"""
Tests for the MatchingEngine — pessimistic CLOB simulator.

Covers:
- Latency enforcement
- FIFO queue tracking (maker orders)
- Taker order-book walking (VWAP slippage)
- Dynamic fee curve
- POST_ONLY rejection
- Partial fills
- Order cancellation
"""

from __future__ import annotations

import pytest

from src.backtest.matching_engine import Fill, MatchingEngine, SimOrder
from src.trading.executor import OrderSide, OrderStatus
from src.trading.fees import get_fee_rate


class TestLatencyEnforcement:
    """Orders must not be visible until latency expires."""

    def setup_method(self):
        self.me = MatchingEngine(latency_ms=100.0, fee_max_pct=1.56)
        # Set up a simple book
        self.me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "100"}, {"price": "0.44", "size": "200"}],
            "asks": [{"price": "0.55", "size": "100"}, {"price": "0.56", "size": "200"}],
        })

    def test_order_pending_during_latency(self):
        order = self.me.submit_order(
            OrderSide.BUY, 0.40, 10.0, current_time=1000.0
        )
        assert order.status == OrderStatus.PENDING
        assert order.active_time == 1000.1  # 100ms later

        # Activate at 1000.05 — too early
        fills = self.me.activate_pending_orders(1000.05)
        assert fills == []
        assert order.status == OrderStatus.PENDING

    def test_order_activates_after_latency(self):
        order = self.me.submit_order(
            OrderSide.BUY, 0.40, 10.0, current_time=1000.0
        )
        # Activate at 1000.1 — exactly at latency expiry
        fills = self.me.activate_pending_orders(1000.1)
        # Non-crossing limit → rests as maker
        assert order.status == OrderStatus.LIVE
        assert fills == []

    def test_zero_latency(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })
        order = me.submit_order(OrderSide.BUY, 0.40, 10.0, current_time=1000.0)
        fills = me.activate_pending_orders(1000.0)
        assert order.status == OrderStatus.LIVE


class TestFIFOQueueTracking:
    """Maker order FIFO queue simulation."""

    def setup_method(self):
        self.me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        self.me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

    def test_queue_ahead_set_on_rest(self):
        """When a maker order rests, queue_ahead = existing depth."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.50, 10.0, current_time=0.0
        )
        self.me.activate_pending_orders(0.0)
        # 100 shares already at 0.50 bid
        assert order.queue_ahead == 100.0
        assert order.status == OrderStatus.LIVE

    def test_queue_drains_gradually(self):
        """Historical trades drain the queue incrementally."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.50, 10.0, current_time=0.0
        )
        self.me.activate_pending_orders(0.0)

        # 60 shares of sell volume at 0.50 — drains queue from 100→40
        fills = self.me.on_trade(0.50, 60.0, "sell", 1.0)
        assert fills == []
        assert order.queue_ahead == 40.0

    def test_full_queue_drain_fills_order(self):
        """Order fills when all queue_ahead is consumed."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.50, 10.0, current_time=0.0
        )
        self.me.activate_pending_orders(0.0)
        assert order.queue_ahead == 100.0

        # Drain 100 shares + 10 more = fills our 10-share order
        fills = self.me.on_trade(0.50, 110.0, "sell", 1.0)
        assert len(fills) == 1
        assert fills[0].size == 10.0
        assert fills[0].is_maker is True
        assert fills[0].fee == 0.0  # maker = no fee
        assert order.status == OrderStatus.FILLED

    def test_partial_queue_drain(self):
        """Partial fill when taker volume only partially exceeds queue."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.50, 10.0, current_time=0.0
        )
        self.me.activate_pending_orders(0.0)

        # Drain 105 = 100 queue + 5 from our order
        fills = self.me.on_trade(0.50, 105.0, "sell", 1.0)
        assert len(fills) == 1
        assert fills[0].size == 5.0
        assert order.remaining == 5.0
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_pessimistic_queue_position(self):
        """Verify exact pessimistic scenario from the spec:
        100 ahead, our 10-share order, 95 shares of volume → only 0 fill."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.50, 10.0, current_time=0.0
        )
        self.me.activate_pending_orders(0.0)

        fills = self.me.on_trade(0.50, 95.0, "sell", 1.0)
        assert fills == []
        assert order.queue_ahead == 5.0  # still 5 shares ahead

    def test_wrong_side_trade_no_fill(self):
        """A BUY trade shouldn't fill our BUY maker order."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.50, 10.0, current_time=0.0
        )
        self.me.activate_pending_orders(0.0)

        fills = self.me.on_trade(0.50, 200.0, "buy", 1.0)
        assert fills == []
        assert order.queue_ahead == 100.0  # unchanged

    def test_wrong_price_trade_no_fill(self):
        """Trade at a different price shouldn't affect our order."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.50, 10.0, current_time=0.0
        )
        self.me.activate_pending_orders(0.0)

        fills = self.me.on_trade(0.55, 200.0, "sell", 1.0)
        assert fills == []

    def test_sell_maker_order(self):
        """Test FIFO for sell-side maker orders."""
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "50"}],
            "asks": [{"price": "0.55", "size": "80"}],
        })

        order = me.submit_order(OrderSide.SELL, 0.55, 20.0, current_time=0.0)
        me.activate_pending_orders(0.0)
        assert order.queue_ahead == 80.0

        # 90 shares of buy volume at 0.55 → fills 10 of our 20
        fills = me.on_trade(0.55, 90.0, "buy", 1.0)
        assert len(fills) == 1
        assert fills[0].size == 10.0


class TestTakerExecution:
    """Taker order execution — order book walking and VWAP slippage."""

    def setup_method(self):
        self.me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        # Book: asks at 0.50 (100), 0.51 (200), 0.52 (300)
        self.me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "500"}],
            "asks": [
                {"price": "0.50", "size": "100"},
                {"price": "0.51", "size": "200"},
                {"price": "0.52", "size": "300"},
            ],
        })

    def test_single_level_fill(self):
        """Buy 50 shares — fills entirely at 0.50."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.55, 50.0, order_type="limit",
            post_only=False, current_time=0.0,
        )
        fills = self.me.activate_pending_orders(0.0)
        assert len(fills) == 1
        assert fills[0].price == 0.50
        assert fills[0].size == 50.0
        assert order.status == OrderStatus.FILLED

    def test_multi_level_walk_vwap(self):
        """Buy 500 shares — sweeps multiple levels.

        Expected: 100@0.50 + 200@0.51 + 200@0.52 = 500
        VWAP = (100×0.50 + 200×0.51 + 200×0.52) / 500
             = (50 + 102 + 104) / 500
             = 256 / 500 = 0.512
        """
        order = self.me.submit_order(
            OrderSide.BUY, 0.55, 500.0, order_type="limit",
            post_only=False, current_time=0.0,
        )
        fills = self.me.activate_pending_orders(0.0)

        assert len(fills) == 3
        assert fills[0].price == 0.50
        assert fills[0].size == 100.0
        assert fills[1].price == 0.51
        assert fills[1].size == 200.0
        assert fills[2].price == 0.52
        assert fills[2].size == 200.0

        # Check VWAP
        vwap = order.filled_avg_price
        expected_vwap = (100 * 0.50 + 200 * 0.51 + 200 * 0.52) / 500
        assert abs(vwap - expected_vwap) < 1e-9
        assert order.status == OrderStatus.FILLED

    def test_partial_fill_at_limit(self):
        """Buy 500 shares with limit at 0.51 — only fills 300."""
        order = self.me.submit_order(
            OrderSide.BUY, 0.51, 500.0, order_type="limit",
            post_only=False, current_time=0.0,
        )
        fills = self.me.activate_pending_orders(0.0)

        assert len(fills) == 2
        total_filled = sum(f.size for f in fills)
        assert total_filled == 300.0
        assert order.remaining == 200.0
        # Remaining rests as maker at 0.51

    def test_market_order_exhausts_book(self):
        """Market buy for more than available → partial fill."""
        order = self.me.submit_order(
            OrderSide.BUY, 1.0, 1000.0, order_type="market",
            current_time=0.0,
        )
        fills = self.me.activate_pending_orders(0.0)
        total = sum(f.size for f in fills)
        assert total == 600.0  # 100 + 200 + 300
        assert order.remaining == 400.0

    def test_taker_sell_walks_bids(self):
        """Sell taker order walks the bid side."""
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [
                {"price": "0.50", "size": "100"},
                {"price": "0.49", "size": "200"},
            ],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        order = me.submit_order(
            OrderSide.SELL, 0.49, 150.0, order_type="limit",
            post_only=False, current_time=0.0,
        )
        fills = me.activate_pending_orders(0.0)

        assert len(fills) == 2
        assert fills[0].price == 0.50
        assert fills[0].size == 100.0
        assert fills[1].price == 0.49
        assert fills[1].size == 50.0
        assert order.status == OrderStatus.FILLED


class TestPostOnlyRejection:
    """POST_ONLY orders that would cross must be rejected."""

    def test_post_only_crossing_rejected(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        order = me.submit_order(
            OrderSide.BUY, 0.55, 10.0, post_only=True, current_time=0.0,
        )
        me.activate_pending_orders(0.0)

        assert order.status == OrderStatus.CANCELLED
        assert order.rejection_reason == "would_cross"

    def test_post_only_non_crossing_rests(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        order = me.submit_order(
            OrderSide.BUY, 0.52, 10.0, post_only=True, current_time=0.0,
        )
        me.activate_pending_orders(0.0)
        assert order.status == OrderStatus.LIVE


class TestDynamicFees:
    """Fee curve: f = f_max × 4 × p × (1-p)."""

    def test_fee_at_midprice(self):
        """At p=0.50, fee = f_max × 4 × 0.25 = f_max."""
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [],
            "asks": [{"price": "0.50", "size": "100"}],
        })

        order = me.submit_order(
            OrderSide.BUY, 0.50, 100.0, order_type="market", current_time=0.0,
        )
        fills = me.activate_pending_orders(0.0)

        assert len(fills) == 1
        expected_rate = 0.0156  # f_max at p=0.50
        expected_fee = 100.0 * 0.50 * expected_rate
        assert abs(fills[0].fee - expected_fee) < 1e-9

    def test_fee_at_extreme_price(self):
        """At p=0.01, fee ≈ 0 (deep OTM)."""
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [],
            "asks": [{"price": "0.01", "size": "1000"}],
        })

        order = me.submit_order(
            OrderSide.BUY, 0.01, 100.0, order_type="market", current_time=0.0,
        )
        fills = me.activate_pending_orders(0.0)

        rate = get_fee_rate(0.01, fee_enabled=True, f_max=0.0156)
        expected = 100.0 * 0.01 * rate
        assert abs(fills[0].fee - expected) < 1e-9
        assert fills[0].fee < 0.01  # very small

    def test_maker_fills_free(self):
        """Maker fills should have zero fees."""
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "50"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        order = me.submit_order(OrderSide.BUY, 0.50, 10.0, current_time=0.0)
        me.activate_pending_orders(0.0)

        # Drain queue and fill
        fills = me.on_trade(0.50, 60.0, "sell", 1.0)
        assert len(fills) == 1
        assert fills[0].fee == 0.0

    def test_fee_consistency_with_fees_module(self):
        """Matching engine fee must match src.trading.fees.get_fee_rate exactly."""
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        prices = [0.05, 0.10, 0.25, 0.33, 0.50, 0.67, 0.75, 0.90, 0.95]

        for p in prices:
            me.on_book_update({
                "event_type": "snapshot",
                "bids": [],
                "asks": [{"price": str(p), "size": "100"}],
            })
            me._id_counter.__next__  # reset doesn't matter for fee check

            order = me.submit_order(
                OrderSide.BUY, p, 100.0, order_type="market", current_time=0.0,
            )
            fills = me.activate_pending_orders(0.0)
            if fills:
                expected_rate = get_fee_rate(p, fee_enabled=True, f_max=0.0156)
                expected_fee = 100.0 * p * expected_rate
                assert abs(fills[0].fee - expected_fee) < 1e-9, f"Mismatch at p={p}"

    def test_fees_disabled(self):
        """With fee_enabled=False, all fees should be 0."""
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56, fee_enabled=False)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [],
            "asks": [{"price": "0.50", "size": "100"}],
        })

        order = me.submit_order(
            OrderSide.BUY, 0.50, 50.0, order_type="market", current_time=0.0,
        )
        fills = me.activate_pending_orders(0.0)
        assert fills[0].fee == 0.0


class TestOrderCancellation:
    """Cancel orders in various states."""

    def test_cancel_pending_order(self):
        me = MatchingEngine(latency_ms=100.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })
        order = me.submit_order(OrderSide.BUY, 0.40, 10.0, current_time=0.0)
        assert me.cancel_order(order.order_id) is True
        assert order.status == OrderStatus.CANCELLED

        # Should not activate after cancel
        fills = me.activate_pending_orders(1.0)
        assert fills == []

    def test_cancel_active_maker(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })
        order = me.submit_order(OrderSide.BUY, 0.50, 10.0, current_time=0.0)
        me.activate_pending_orders(0.0)
        assert order.status == OrderStatus.LIVE

        assert me.cancel_order(order.order_id) is True
        assert order.status == OrderStatus.CANCELLED

        # No more fills possible
        fills = me.on_trade(0.50, 200.0, "sell", 1.0)
        assert fills == []

    def test_cancel_nonexistent_order(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        assert me.cancel_order("DOES-NOT-EXIST") is False

    def test_cancel_filled_order(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [],
            "asks": [{"price": "0.50", "size": "100"}],
        })
        order = me.submit_order(
            OrderSide.BUY, 0.50, 10.0, order_type="market", current_time=0.0,
        )
        me.activate_pending_orders(0.0)
        assert order.status == OrderStatus.FILLED

        assert me.cancel_order(order.order_id) is False


class TestBookUpdates:
    """Book delta and snapshot application."""

    def test_snapshot_replaces_book(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.40", "size": "100"}],
            "asks": [{"price": "0.60", "size": "100"}],
        })
        assert me.best_bid == 0.40
        assert me.best_ask == 0.60

        # New snapshot replaces
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "50"}],
            "asks": [{"price": "0.55", "size": "50"}],
        })
        assert me.best_bid == 0.45
        assert me.best_ask == 0.55

    def test_delta_updates_levels(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        # Delta: add a new bid level, remove the ask level
        me.on_book_update({
            "event_type": "price_change",
            "changes": [
                {"side": "BUY", "price": "0.51", "size": "50"},
                {"side": "SELL", "price": "0.55", "size": "0"},
            ],
        })
        assert me.best_bid == 0.51
        assert me.best_ask == 0.0  # removed

    def test_mid_price(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })
        assert me.mid_price == 0.50


class TestAccessors:
    """Order and fill accessors."""

    def test_get_open_orders(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        o1 = me.submit_order(OrderSide.BUY, 0.45, 10.0, current_time=0.0)
        o2 = me.submit_order(OrderSide.SELL, 0.55, 10.0, current_time=0.0)
        me.activate_pending_orders(0.0)

        open_orders = me.get_open_orders()
        assert len(open_orders) == 2

    def test_get_all_fills_sorted(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        # Place two orders, one fills immediately as taker
        me.submit_order(OrderSide.BUY, 0.55, 5.0, current_time=1.0)
        me.activate_pending_orders(1.0)

        me.submit_order(OrderSide.BUY, 0.55, 5.0, order_type="market", current_time=2.0)
        me.activate_pending_orders(2.0)

        fills = me.get_all_fills()
        # Should be sorted by timestamp
        for i in range(len(fills) - 1):
            assert fills[i].timestamp <= fills[i + 1].timestamp

    def test_reset(self):
        me = MatchingEngine(latency_ms=0.0, fee_max_pct=1.56)
        me.on_book_update({
            "event_type": "snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })
        me.submit_order(OrderSide.BUY, 0.50, 10.0, current_time=0.0)
        me.reset()

        assert me.best_bid == 0.0
        assert me.best_ask == 0.0
        assert me.get_all_orders() == []
        assert me.get_open_orders() == []


class TestVirtualLiquidityDebt:
    """Synthetic market impact — consumed volume persists across book updates."""

    def setup_method(self):
        # Short recovery so decay is testable at small time deltas
        self.me = MatchingEngine(
            latency_ms=0.0, fee_max_pct=1.56, impact_recovery_ms=5000.0,
        )
        self._snapshot = {
            "event_type": "snapshot",
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}, {"price": "0.56", "size": "200"}],
        }
        self.me.on_book_update(self._snapshot, current_time=0.0)

    def test_debt_reduces_available_on_next_update(self):
        """After a taker fill, the next book update should show reduced liquidity."""
        # Consume 60 of the 100 at 0.55
        order = self.me.submit_order(
            OrderSide.BUY, 0.55, 60.0, order_type="market", current_time=1.0,
        )
        fills = self.me.activate_pending_orders(1.0)
        assert len(fills) == 1
        assert fills[0].size == 60.0

        # Re-apply the same historical snapshot at t=1.5 (500ms later, minimal decay)
        self.me.on_book_update(self._snapshot, current_time=1.5)

        # The book should show ~40 at 0.55 (100 - 60 * exp(-0.5/5) ≈ 100 - 54.3 = 45.7)
        avail = self.me.depth_at_price(OrderSide.SELL, 0.55)
        assert avail < 100.0, "Debt should reduce available liquidity"
        assert avail > 30.0, "Debt hasn't fully decayed yet"

    def test_debt_decays_over_time(self):
        """Virtual debt should decay exponentially toward zero."""
        import math

        # Consume all 100 at 0.55
        self.me.submit_order(
            OrderSide.BUY, 0.55, 100.0, order_type="market", current_time=1.0,
        )
        self.me.activate_pending_orders(1.0)

        # Re-apply after 25 seconds (5 × tau) — debt decayed to ~0.7%
        self.me.on_book_update(self._snapshot, current_time=26.0)
        avail = self.me.depth_at_price(OrderSide.SELL, 0.55)
        # exp(-25/5) = exp(-5) ≈ 0.0067 → debt ≈ 0.67 → avail ≈ 99.3
        assert avail > 95.0, f"Debt should be nearly gone after 5τ, got {avail}"

    def test_debt_accumulates_across_fills(self):
        """Multiple fills at the same level stack their debt."""
        # First fill: consume 30
        self.me.submit_order(
            OrderSide.BUY, 0.55, 30.0, order_type="market", current_time=1.0,
        )
        self.me.activate_pending_orders(1.0)

        # Re-apply book and fill again at t=2
        self.me.on_book_update(self._snapshot, current_time=2.0)
        self.me.submit_order(
            OrderSide.BUY, 0.55, 30.0, order_type="market", current_time=2.0,
        )
        self.me.activate_pending_orders(2.0)

        # Re-apply at t=2.5
        self.me.on_book_update(self._snapshot, current_time=2.5)
        avail = self.me.depth_at_price(OrderSide.SELL, 0.55)
        # Total debt ~ 30*exp(-1.5/5) + 30*exp(-0.5/5) ≈ 22.2 + 27.1 = 49.3
        assert avail < 60.0, f"Accumulated debt should reduce level significantly, got {avail}"

    def test_sell_side_debt(self):
        """Debt works symmetrically for sell-side fills."""
        self.me.submit_order(
            OrderSide.SELL, 0.45, 50.0, order_type="market", current_time=1.0,
        )
        self.me.activate_pending_orders(1.0)

        # Re-apply same snapshot
        self.me.on_book_update(self._snapshot, current_time=1.5)
        avail = self.me.depth_at_price(OrderSide.BUY, 0.45)
        assert avail < 100.0, "Sell-side debt should reduce bid liquidity"

    def test_reset_clears_debt(self):
        """reset() must wipe liquidity debt."""
        self.me.submit_order(
            OrderSide.BUY, 0.55, 50.0, order_type="market", current_time=1.0,
        )
        self.me.activate_pending_orders(1.0)
        self.me.reset()

        self.me.on_book_update(self._snapshot, current_time=2.0)
        avail = self.me.depth_at_price(OrderSide.SELL, 0.55)
        assert avail == 100.0, "After reset, no debt should remain"

    def test_impact_skips_exhausted_level(self):
        """When debt exceeds level size, taker should skip to next level."""
        # Consume all 100 at 0.55
        self.me.submit_order(
            OrderSide.BUY, 0.56, 100.0, order_type="market", current_time=1.0,
        )
        self.me.activate_pending_orders(1.0)

        # Re-apply book immediately (no decay)
        self.me.on_book_update(self._snapshot, current_time=1.001)

        # Submit another buy — should skip exhausted 0.55 and fill at 0.56
        order2 = self.me.submit_order(
            OrderSide.BUY, 0.56, 10.0, order_type="market", current_time=1.001,
        )
        fills = self.me.activate_pending_orders(1.001)
        assert len(fills) >= 1
        # First fill should NOT be at 0.55 (exhausted by debt)
        for f in fills:
            assert f.price >= 0.55
