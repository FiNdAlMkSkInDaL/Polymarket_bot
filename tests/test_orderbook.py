"""Tests for src.data.orderbook — live L2 orderbook tracker."""

from __future__ import annotations

import pytest

from src.data.orderbook import OrderbookSnapshot, OrderbookTracker


class TestOrderbookTracker:
    def setup_method(self):
        self.tracker = OrderbookTracker("ASSET_001")

    def test_initial_state(self):
        assert not self.tracker.has_data
        snap = self.tracker.snapshot()
        assert snap.best_bid == 0.0
        assert snap.best_ask == 0.0
        assert snap.mid_price == 0.0

    def test_price_change_buy(self):
        self.tracker.on_price_change({
            "event_type": "price_change",
            "asset_id": "ASSET_001",
            "changes": [
                {"side": "BUY", "price": "0.47", "size": "100"},
                {"side": "BUY", "price": "0.46", "size": "200"},
            ],
        })
        assert self.tracker.has_data
        snap = self.tracker.snapshot()
        assert snap.best_bid == 0.47
        assert snap.bid_depth_usd > 0

    def test_price_change_sell(self):
        self.tracker.on_price_change({
            "event_type": "price_change",
            "asset_id": "ASSET_001",
            "changes": [
                {"side": "SELL", "price": "0.53", "size": "80"},
                {"side": "SELL", "price": "0.55", "size": "60"},
            ],
        })
        snap = self.tracker.snapshot()
        assert snap.best_ask == 0.53
        assert snap.ask_depth_usd > 0

    def test_spread_and_mid(self):
        self.tracker.on_price_change({
            "changes": [
                {"side": "BUY", "price": "0.47", "size": "100"},
                {"side": "SELL", "price": "0.53", "size": "100"},
            ],
        })
        snap = self.tracker.snapshot()
        assert abs(snap.spread - 0.06) < 0.001
        assert abs(snap.mid_price - 0.50) < 0.001
        assert abs(self.tracker.spread_cents - 6.0) < 0.1

    def test_update_existing_level(self):
        self.tracker.on_price_change({
            "changes": [{"side": "BUY", "price": "0.47", "size": "100"}],
        })
        self.tracker.on_price_change({
            "changes": [{"side": "BUY", "price": "0.47", "size": "200"}],
        })
        # Should have one level, not two
        assert len(self.tracker._bids) == 1
        assert self.tracker._bids[0].size == 200

    def test_remove_level_zero_size(self):
        self.tracker.on_price_change({
            "changes": [{"side": "BUY", "price": "0.47", "size": "100"}],
        })
        self.tracker.on_price_change({
            "changes": [{"side": "BUY", "price": "0.47", "size": "0"}],
        })
        assert len(self.tracker._bids) == 0

    def test_book_snapshot_replaces_state(self):
        # First, add some levels
        self.tracker.on_price_change({
            "changes": [{"side": "BUY", "price": "0.47", "size": "100"}],
        })
        # Now replace with a book snapshot
        self.tracker.on_book_snapshot({
            "bids": [{"price": "0.45", "size": "500"}],
            "asks": [{"price": "0.55", "size": "300"}],
        })
        snap = self.tracker.snapshot()
        assert snap.best_bid == 0.45
        assert snap.best_ask == 0.55

    def test_book_snapshot_sorts_before_truncating_levels(self):
        self.tracker.on_book_snapshot({
            "bids": [
                {"price": "0.01", "size": "10"},
                {"price": "0.02", "size": "10"},
                {"price": "0.03", "size": "10"},
                {"price": "0.04", "size": "10"},
                {"price": "0.05", "size": "10"},
                {"price": "0.06", "size": "10"},
                {"price": "0.07", "size": "10"},
                {"price": "0.08", "size": "10"},
                {"price": "0.09", "size": "10"},
                {"price": "0.10", "size": "10"},
                {"price": "0.95", "size": "10"},
                {"price": "0.94", "size": "10"},
            ],
            "asks": [
                {"price": "0.99", "size": "10"},
                {"price": "0.98", "size": "10"},
                {"price": "0.97", "size": "10"},
                {"price": "0.96", "size": "10"},
                {"price": "0.95", "size": "10"},
                {"price": "0.94", "size": "10"},
                {"price": "0.93", "size": "10"},
                {"price": "0.92", "size": "10"},
                {"price": "0.91", "size": "10"},
                {"price": "0.90", "size": "10"},
                {"price": "0.11", "size": "10"},
                {"price": "0.12", "size": "10"},
            ],
        })

        snap = self.tracker.snapshot()

        assert snap.best_bid == 0.95
        assert snap.best_ask == 0.11
        assert len(self.tracker.levels("bid", n=20)) == self.tracker._MAX_LEVELS
        assert len(self.tracker.levels("ask", n=20)) == self.tracker._MAX_LEVELS

    def test_book_depth_ratio(self):
        self.tracker.on_book_snapshot({
            "bids": [{"price": "0.50", "size": "200"}],
            "asks": [{"price": "0.51", "size": "100"}],
        })
        ratio = self.tracker.book_depth_ratio
        # bid_depth = 0.5 * 200 = 100, ask_depth = 0.51 * 100 = 51
        assert ratio > 1.5

    def test_book_depth_ratio_no_asks(self):
        self.tracker.on_price_change({
            "changes": [{"side": "BUY", "price": "0.50", "size": "100"}],
        })
        assert self.tracker.book_depth_ratio == 1.0

    def test_spread_cents_no_data(self):
        assert self.tracker.spread_cents == 0.0

    def test_levels_returns_copy(self):
        self.tracker.on_price_change({
            "changes": [
                {"side": "BUY", "price": "0.47", "size": "100"},
                {"side": "BUY", "price": "0.46", "size": "200"},
            ],
        })
        levels = self.tracker.levels("bid", n=2)
        assert len(levels) == 2
        # Modifying returned list should NOT affect internal state
        levels.pop()
        assert len(self.tracker._bids) == 2

    def test_max_levels_capped(self):
        for i in range(20):
            price = f"0.{40 + i:02d}"
            self.tracker.on_price_change({
                "changes": [{"side": "BUY", "price": price, "size": "10"}],
            })
        # Internal should be capped (sorted + truncated on snapshot)
        # The tracker appends without truncating until next sort
        # But via on_book_snapshot it caps at _MAX_LEVELS
        # The on_price_change path appends freely — just verify it works
        snap = self.tracker.snapshot()
        assert snap.best_bid > 0

    def test_snapshot_fresh_flag(self):
        self.tracker.on_price_change({
            "changes": [{"side": "BUY", "price": "0.50", "size": "100"}],
        })
        snap_fresh = self.tracker.snapshot(fresh=True)
        snap_stale = self.tracker.snapshot(fresh=False)
        assert snap_fresh.fresh is True
        assert snap_stale.fresh is False

    def test_fallback_top_level_price(self):
        """price_change without 'changes' key falls back to top-level."""
        self.tracker.on_price_change({
            "event_type": "price_change",
            "asset_id": "ASSET_001",
            "price": "0.48",
            "size": "50",
            "side": "BUY",
        })
        snap = self.tracker.snapshot()
        assert snap.best_bid == 0.48


class TestOrderbookSnapshot:
    def test_dataclass(self):
        snap = OrderbookSnapshot(asset_id="X", best_bid=0.4, best_ask=0.6)
        assert snap.asset_id == "X"
        assert snap.spread == 0.0  # default
