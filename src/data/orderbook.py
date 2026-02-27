"""
Orderbook tracker — maintains a live top-of-book view from WebSocket
``price_change`` and ``book`` events.

Each market gets two trackers (YES + NO) that expose real-time
bid/ask spread, depth, and mid-price used by:
  - Market scorer (spread & depth factors)
  - PanicDetector (real best_ask instead of last-trade proxy)
  - PositionManager (precise entry pricing)
  - TakeProfit (real book_depth_ratio)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class OrderbookSnapshot:
    """Point-in-time L2 orderbook summary for one asset."""

    asset_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    bid_depth_usd: float = 0.0   # sum of top-5 bid levels (price × size)
    ask_depth_usd: float = 0.0   # sum of top-5 ask levels
    mid_price: float = 0.0
    timestamp: float = 0.0
    fresh: bool = True            # False when latency guard is BLOCKED


@dataclass
class _Level:
    """A single price level in the book."""
    price: float
    size: float


class OrderbookTracker:
    """Per-asset orderbook maintained from WS ``price_change`` events.

    Tracks top-N levels on each side.  Not a full L2 book — Polymarket's
    WS provides diffed snapshots, not full order-by-order updates.
    """

    _MAX_LEVELS = 10  # keep top-10 per side

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self._bids: list[_Level] = []  # sorted desc by price
        self._asks: list[_Level] = []  # sorted asc by price
        self._last_update: float = 0.0

    def on_price_change(self, data: dict) -> None:
        """Process a ``price_change`` WS event.

        Expected schema (Polymarket WS ``market`` channel):
        ```json
        {
            "event_type": "price_change",
            "asset_id": "0x...",
            "changes": [
                {"side": "BUY", "price": "0.47", "size": "123.5"},
                {"side": "SELL", "price": "0.53", "size": "80.0"},
            ]
        }
        ```
        If ``changes`` is absent, fall back to top-level ``price``/``side``.
        """
        changes = data.get("changes") or data.get("data") or []
        if isinstance(changes, dict):
            changes = [changes]

        if not changes and data.get("price"):
            changes = [data]

        for ch in changes:
            try:
                price = float(ch.get("price", 0))
                size = float(ch.get("size", 0))
                side = str(ch.get("side", "")).upper()
            except (TypeError, ValueError):
                continue

            if price <= 0:
                continue

            if side == "BUY":
                self._update_side(self._bids, price, size, descending=True)
            elif side == "SELL":
                self._update_side(self._asks, price, size, descending=False)

        self._last_update = time.time()

    def on_book_snapshot(self, data: dict) -> None:
        """Process a full ``book`` snapshot (replaces current state)."""
        bids_raw = data.get("bids") or []
        asks_raw = data.get("asks") or []

        self._bids = []
        for b in bids_raw[:self._MAX_LEVELS]:
            try:
                self._bids.append(_Level(float(b["price"]), float(b["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        self._bids.sort(key=lambda l: l.price, reverse=True)

        self._asks = []
        for a in asks_raw[:self._MAX_LEVELS]:
            try:
                self._asks.append(_Level(float(a["price"]), float(a["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        self._asks.sort(key=lambda l: l.price)

        self._last_update = time.time()

    def snapshot(self, *, fresh: bool = True) -> OrderbookSnapshot:
        """Return current top-of-book summary.

        Parameters
        ----------
        fresh:
            Set to ``False`` when the latency guard is in BLOCKED state so
            downstream consumers know the data may be stale.
        """
        best_bid = self._bids[0].price if self._bids else 0.0
        best_ask = self._asks[0].price if self._asks else 0.0
        spread = (best_ask - best_bid) if (best_ask > 0 and best_bid > 0) else 0.0
        mid = (best_bid + best_ask) / 2.0 if (best_ask > 0 and best_bid > 0) else 0.0

        bid_depth = sum(l.price * l.size for l in self._bids[:5])
        ask_depth = sum(l.price * l.size for l in self._asks[:5])

        return OrderbookSnapshot(
            asset_id=self.asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=round(spread, 4),
            bid_depth_usd=round(bid_depth, 2),
            ask_depth_usd=round(ask_depth, 2),
            mid_price=round(mid, 4),
            timestamp=self._last_update,
            fresh=fresh,
        )

    def levels(self, side: str, n: int = 5) -> list[_Level]:
        """Return the top *n* levels for *side* (``"bid"`` or ``"ask"``).

        Returns a shallow copy so callers cannot mutate internal state.
        """
        if side.lower() in ("bid", "buy"):
            return list(self._bids[:n])
        return list(self._asks[:n])

    @property
    def spread_cents(self) -> float:
        """Current spread in cents (convenience)."""
        if not self._bids or not self._asks:
            return 0.0
        return round((self._asks[0].price - self._bids[0].price) * 100, 2)

    @property
    def book_depth_ratio(self) -> float:
        """Bid depth / ask depth ratio.  >1 = more resting buy interest."""
        bid_d = sum(l.price * l.size for l in self._bids[:5])
        ask_d = sum(l.price * l.size for l in self._asks[:5])
        if ask_d <= 0:
            return 1.0
        return round(bid_d / ask_d, 2)

    @property
    def has_data(self) -> bool:
        return self._last_update > 0

    # ── internal ───────────────────────────────────────────────────────
    def _update_side(
        self, levels: list[_Level], price: float, size: float, *, descending: bool
    ) -> None:
        """Insert or update a level.  Remove if size == 0."""
        # Find existing level at this price
        for i, lv in enumerate(levels):
            if abs(lv.price - price) < 1e-6:
                if size <= 0:
                    levels.pop(i)
                else:
                    lv.size = size
                return

        if size <= 0:
            return

        levels.append(_Level(price, size))
        levels.sort(key=lambda l: l.price, reverse=descending)

        # Trim to max levels
        if len(levels) > self._MAX_LEVELS:
            del levels[self._MAX_LEVELS:]
