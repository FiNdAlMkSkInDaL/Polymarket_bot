from __future__ import annotations

import math
from decimal import Decimal
from typing import Literal

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
            return self._coerce_positive_decimal(self._tracker.best_bid)
        except Exception:
            return None

    def get_best_ask(self, market_id: str) -> Decimal | None:
        try:
            if not self._matches_tracker(market_id):
                return None
            snapshot = self._tracker.snapshot()
            best_ask = self._coerce_positive_decimal(getattr(snapshot, "best_ask", 0.0))
            if best_ask is not None:
                return best_ask
            return self._coerce_positive_decimal(getattr(self._tracker, "best_ask", 0.0))
        except Exception:
            return None

    def get_spread(self, market_id: str) -> Decimal | None:
        if not self._matches_tracker(market_id):
            return None
        best_bid = self.get_best_bid(market_id)
        best_ask = self.get_best_ask(market_id)
        if best_bid is None or best_ask is None:
            return None
        spread = best_ask - best_bid
        if spread < Decimal("0"):
            return None
        return spread

    def get_top_depth(self, market_id: str, side: Literal["bid", "ask"]) -> Decimal | None:
        try:
            if not self._matches_tracker(market_id):
                return None
            if hasattr(self._tracker, "top_depths_usd"):
                bid_depth, ask_depth = self._tracker.top_depths_usd()
            else:
                snapshot = self._tracker.snapshot()
                bid_depth = getattr(snapshot, "bid_depth_usd", 0.0)
                ask_depth = getattr(snapshot, "ask_depth_usd", 0.0)
            depth = bid_depth if side == "bid" else ask_depth
            return self._coerce_non_negative_decimal(depth)
        except Exception:
            return None

    def get_top_depth_ewma(self, market_id: str, side: Literal["bid", "ask"]) -> Decimal | None:
        try:
            if not self._matches_tracker(market_id):
                return None
            top_depth_ewma = getattr(self._tracker, "top_depth_ewma", None)
            if callable(top_depth_ewma):
                baseline = self._coerce_non_negative_decimal(top_depth_ewma(side))
                if baseline is not None and baseline > Decimal("0"):
                    return baseline
            return self.get_top_depth(market_id, side)
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

    @staticmethod
    def _coerce_positive_decimal(value: object) -> Decimal | None:
        try:
            numeric = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric) or numeric <= 0.0:
            return None
        return Decimal(str(numeric))

    @staticmethod
    def _coerce_non_negative_decimal(value: object) -> Decimal | None:
        try:
            numeric = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric) or numeric < 0.0:
            return None
        return Decimal(str(numeric))