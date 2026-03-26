from __future__ import annotations

import math
from decimal import Decimal

from src.data.orderbook import OrderbookTracker
from src.execution.live_book_interface import LiveBestBidProvider


class OrderbookBestBidProvider(LiveBestBidProvider):
    """Read-only live best-bid adapter over a single OrderbookTracker.

    The provider is intentionally defensive because the interface contract
    requires non-blocking reads that never raise.
    """

    def __init__(self, tracker: OrderbookTracker):
        if tracker is None:
            raise ValueError("tracker is required")
        self._tracker = tracker

    def get_best_bid(self, market_id: str) -> Decimal | None:
        try:
            if not self._matches_tracker(market_id):
                return None
            best_bid = float(self._tracker.best_bid)
            if not math.isfinite(best_bid) or best_bid <= 0.0:
                return None
            return Decimal(str(best_bid))
        except Exception:
            return None

    def get_best_bid_timestamp_ms(self, market_id: str) -> int | None:
        try:
            if not self._matches_tracker(market_id):
                return None
            snapshot = self._tracker.snapshot()
            timestamp = float(snapshot.timestamp)
            if not math.isfinite(timestamp) or timestamp <= 0.0:
                return None
            if timestamp >= 10_000_000_000:
                return int(timestamp)
            return int(timestamp * 1000)
        except Exception:
            return None

    def _matches_tracker(self, market_id: str) -> bool:
        market_key = str(market_id or "").strip()
        tracker_key = str(getattr(self._tracker, "asset_id", "") or "").strip()
        return bool(market_key) and market_key == tracker_key