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

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.logger import get_logger
from src.data.types import Level as _Level

log = get_logger(__name__)
_TOP_DEPTH_EWMA_ALPHA = 0.2


def _log_task_exception(task: asyncio.Task) -> None:
    """Done-callback for fire-and-forget tasks to surface exceptions."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error(
            "orderbook_callback_task_error",
            task_name=task.get_name(),
            error=repr(exc),
            exc_info=exc,
        )


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
    server_time: float = 0.0     # server-reported timestamp (epoch s)
    fresh: bool = True            # False when latency guard is BLOCKED
    spread_score: float = 0.0    # live L2 spread score (0-100)


class OrderbookTracker:
    """Per-asset orderbook maintained from WS ``price_change`` events.

    Tracks top-N levels on each side.  Not a full L2 book — Polymarket's
    WS provides diffed snapshots, not full order-by-order updates.
    """

    _MAX_LEVELS = 10  # keep top-10 per side

    def __init__(
        self,
        asset_id: str,
        *,
        on_bbo_change: Callable[..., Any] | None = None,
    ):
        self.asset_id = asset_id
        self._bids: list[_Level] = []  # sorted desc by price
        self._asks: list[_Level] = []  # sorted asc by price
        self._last_update: float = 0.0
        self._last_server_time: float = 0.0
        self._on_bbo_change = on_bbo_change
        # Previous BBO for change detection
        self._prev_best_bid: float = 0.0
        self._prev_best_ask: float = 0.0
        # Depth history ring buffer for Ghost Liquidity detection
        self._depth_history: deque[tuple[float, float]] = deque(maxlen=40)
        self._bid_depth_ewma: float = 0.0
        self._ask_depth_ewma: float = 0.0

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
        # Extract server timestamp if present
        srv = data.get("timestamp") or data.get("server_timestamp") or data.get("ts")
        if srv is not None:
            try:
                self._last_server_time = float(srv)
            except (TypeError, ValueError):
                pass
        self._record_depth()
        self._check_bbo_change()

    def on_book_snapshot(self, data: dict) -> None:
        """Process a full ``book`` snapshot (replaces current state)."""
        bids_raw = data.get("bids") or []
        asks_raw = data.get("asks") or []

        self._bids = []
        for b in bids_raw:
            try:
                self._bids.append(_Level(float(b["price"]), float(b["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        self._bids.sort(key=lambda l: l.price, reverse=True)
        self._bids = self._bids[:self._MAX_LEVELS]

        self._asks = []
        for a in asks_raw:
            try:
                self._asks.append(_Level(float(a["price"]), float(a["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        self._asks.sort(key=lambda l: l.price)
        self._asks = self._asks[:self._MAX_LEVELS]

        self._last_update = time.time()
        srv = data.get("timestamp") or data.get("server_timestamp") or data.get("ts")
        if srv is not None:
            try:
                self._last_server_time = float(srv)
            except (TypeError, ValueError):
                pass
        self._record_depth()
        self._check_bbo_change()

    @property
    def best_bid(self) -> float:
        """Best (highest) bid price, or 0.0 if empty."""
        return self._bids[0].price if self._bids else 0.0

    @property
    def best_ask(self) -> float:
        """Best (lowest) ask price, or 0.0 if empty."""
        return self._asks[0].price if self._asks else 0.0

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
            server_time=self._last_server_time,
            fresh=fresh,
        )

    def levels(self, side: str, n: int = 5) -> list[_Level]:
        """Return the top *n* levels for *side* (``"bid"`` or ``"ask"``).

        Returns a shallow copy so callers cannot mutate internal state.
        """
        if side.lower() in ("bid", "buy"):
            return list(self._bids[:n])
        return list(self._asks[:n])

    def depth_near_mid_usd(self, cents: float, max_levels: int = 50) -> float:
        """Sum resting depth (USD) within *cents* of mid."""
        bb = self.best_bid
        ba = self.best_ask
        if bb <= 0 or ba <= 0:
            return 0.0
        mid = (bb + ba) / 2.0
        threshold = cents / 100.0
        depth = 0.0
        for lv in self._bids[:max_levels]:
            if abs(lv.price - mid) <= threshold:
                depth += lv.price * lv.size
        for lv in self._asks[:max_levels]:
            if abs(lv.price - mid) <= threshold:
                depth += lv.price * lv.size
        return depth

    def toxicity_index(self, side: str = "BUY") -> float:
        del side
        return 0.0

    def toxicity_metrics(self, side: str = "BUY") -> dict[str, float]:
        del side
        return {
            "toxicity_index": 0.0,
            "toxicity_depth_evaporation": 0.0,
            "toxicity_sweep_ratio": 0.0,
        }

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

    @property
    def is_reliable(self) -> bool:
        """Legacy trackers have no desync tracking — always reliable."""
        return True

    @property
    def seq_gap_rate(self) -> float:
        """Legacy trackers have no sequence tracking."""
        return 0.0

    @property
    def delta_count(self) -> int:
        """Legacy trackers don't count deltas."""
        return 0

    @property
    def desync_total(self) -> int:
        """Legacy trackers don't track desyncs."""
        return 0

    # ── BBO change detection ─────────────────────────────────────────────
    def _check_bbo_change(self) -> None:
        """Fire ``on_bbo_change`` callback if the BBO has changed."""
        bb = self._bids[0].price if self._bids else 0.0
        ba = self._asks[0].price if self._asks else 0.0
        if bb == self._prev_best_bid and ba == self._prev_best_ask:
            return
        self._prev_best_bid = bb
        self._prev_best_ask = ba
        if self._on_bbo_change is not None:
            try:
                snap = self.snapshot()
                result = self._on_bbo_change(self.asset_id, snap)
                if asyncio.iscoroutine(result):
                    task = asyncio.ensure_future(result)
                    task.add_done_callback(_log_task_exception)
            except Exception:
                log.error(
                    "bbo_callback_dispatch_error",
                    asset_id=self.asset_id,
                    exc_info=True,
                )

    # ── depth tracking for Ghost Liquidity detection ─────────────────────
    def _record_depth(self) -> None:
        """Append current total depth to the ring buffer."""
        bid_depth, ask_depth = self.top_depths_usd()
        if self._bid_depth_ewma <= 0:
            self._bid_depth_ewma = bid_depth
        else:
            self._bid_depth_ewma = (
                _TOP_DEPTH_EWMA_ALPHA * bid_depth
                + (1.0 - _TOP_DEPTH_EWMA_ALPHA) * self._bid_depth_ewma
            )
        if self._ask_depth_ewma <= 0:
            self._ask_depth_ewma = ask_depth
        else:
            self._ask_depth_ewma = (
                _TOP_DEPTH_EWMA_ALPHA * ask_depth
                + (1.0 - _TOP_DEPTH_EWMA_ALPHA) * self._ask_depth_ewma
            )
        depth = round(bid_depth + ask_depth, 2)
        self._depth_history.append((time.time(), depth))

    def current_total_depth(self) -> float:
        """Return combined top-5 bid + ask depth in USD."""
        bid_d, ask_d = self.top_depths_usd()
        return round(bid_d + ask_d, 2)

    def top_depths_usd(self) -> tuple[float, float]:
        bid_d = sum(l.price * l.size for l in self._bids[:5])
        ask_d = sum(l.price * l.size for l in self._asks[:5])
        return round(bid_d, 2), round(ask_d, 2)

    def top_depth_ewma(self, side: str) -> float:
        current_bid, current_ask = self.top_depths_usd()
        if side.lower() in ("bid", "buy"):
            return round(self._bid_depth_ewma or current_bid, 2)
        return round(self._ask_depth_ewma or current_ask, 2)

    def depth_velocity(self, window_s: float = 2.0) -> float | None:
        """Compute the fractional depth change over *window_s* seconds.

        Returns
        -------
        float | None
            ``(D_now - D_past) / D_past`` if sufficient history, else ``None``.
            A value of -0.50 means depth dropped 50%.
        """
        if len(self._depth_history) < 2:
            return None

        now = time.time()
        target_time = now - window_s

        # Find the sample closest to target_time
        past_depth: float | None = None
        for ts, depth in self._depth_history:
            if ts <= target_time:
                past_depth = depth
            else:
                break

        if past_depth is None or past_depth <= 0:
            return None

        current_depth = self.current_total_depth()
        return (current_depth - past_depth) / past_depth

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


class L2OrderBookAdapter(OrderbookTracker):
    """Adapter that wraps an ``L2OrderBook`` behind the ``OrderbookTracker``
    interface so all existing consumers (heartbeat, sizer, chaser, ghost CB,
    scorer, TP rescale, adverse selection, MTI classification) work without
    modification.

    Delegates all reads to the underlying ``L2OrderBook`` instance.  Write
    methods (``on_price_change``, ``on_book_snapshot``) are no-ops since the
    L2 book is populated by its own WebSocket pipeline.
    """

    def __init__(self, l2_book: Any) -> None:
        # Do NOT call super().__init__() because we delegate everything
        self._l2 = l2_book
        self.asset_id = l2_book.asset_id

    # ── Write ops are no-ops (L2 book is fed by L2WebSocket) ───────────
    def on_price_change(self, data: dict) -> None:
        """No-op: L2 book receives deltas via its own pipeline."""
        pass

    def on_book_snapshot(self, data: dict) -> None:
        """No-op: L2 book receives snapshots via its own pipeline."""
        pass

    # ── Read ops delegate to L2OrderBook ───────────────────────────────
    def snapshot(self, *, fresh: bool = True) -> OrderbookSnapshot:
        l2_snap = self._l2.snapshot(fresh=fresh)
        return OrderbookSnapshot(
            asset_id=l2_snap.asset_id,
            best_bid=l2_snap.best_bid,
            best_ask=l2_snap.best_ask,
            spread=l2_snap.spread,
            bid_depth_usd=l2_snap.bid_depth_usd,
            ask_depth_usd=l2_snap.ask_depth_usd,
            mid_price=l2_snap.mid_price,
            timestamp=l2_snap.timestamp,
            server_time=l2_snap.server_time,
            fresh=l2_snap.fresh,
            spread_score=l2_snap.spread_score,
        )

    def levels(self, side: str, n: int = 5) -> list[_Level]:
        return self._l2.levels(side, n)

    @property
    def spread_cents(self) -> float:
        return self._l2.spread_cents

    @property
    def best_bid(self) -> float:
        return self._l2.best_bid

    @property
    def best_ask(self) -> float:
        return self._l2.best_ask

    @property
    def book_depth_ratio(self) -> float:
        return self._l2.book_depth_ratio

    @property
    def has_data(self) -> bool:
        return self._l2.has_data

    @property
    def is_reliable(self) -> bool:
        return self._l2.is_reliable

    @property
    def seq_gap_rate(self) -> float:
        return self._l2.seq_gap_rate

    @property
    def delta_count(self) -> int:
        return self._l2.delta_count

    @property
    def desync_total(self) -> int:
        return self._l2.desync_total

    @property
    def _last_update(self) -> float:
        return self._l2._last_update

    @_last_update.setter
    def _last_update(self, value: float) -> None:
        pass  # read-only delegate

    @property
    def _last_server_time(self) -> float:
        return self._l2._last_server_time

    @_last_server_time.setter
    def _last_server_time(self, value: float) -> None:
        pass  # read-only delegate

    def current_total_depth(self) -> float:
        return self._l2.current_total_depth()

    def top_depths_usd(self) -> tuple[float, float]:
        if hasattr(self._l2, "top_depths_usd"):
            return self._l2.top_depths_usd()
        snap = self._l2.snapshot()
        return (
            round(getattr(snap, "bid_depth_usd", 0.0), 2),
            round(getattr(snap, "ask_depth_usd", 0.0), 2),
        )

    def top_depth_ewma(self, side: str) -> float:
        if hasattr(self._l2, "top_depth_ewma"):
            return self._l2.top_depth_ewma(side)
        bid_depth, ask_depth = self.top_depths_usd()
        return bid_depth if side.lower() in ("bid", "buy") else ask_depth

    def depth_velocity(self, window_s: float = 2.0) -> float | None:
        return self._l2.depth_velocity(window_s)

    def depth_near_mid_usd(self, cents: float, max_levels: int = 50) -> float:
        return self._l2.depth_near_mid_usd(cents, max_levels)

    def toxicity_index(self, side: str = "BUY") -> float:
        return self._l2.toxicity_index(side)

    def toxicity_metrics(self, side: str = "BUY") -> dict[str, float]:
        return self._l2.toxicity_metrics(side)


class SharedBookReaderAdapter(OrderbookTracker):
    """Adapter that wraps a ``SharedBookReader`` (cross-process shared memory)
    behind the ``OrderbookTracker`` interface so all existing consumers work
    without modification.

    The underlying ``SharedBookReader`` reads from a shared memory segment
    written by an L2 worker process — no in-process L2OrderBook exists.
    """

    def __init__(self, reader: "Any") -> None:
        # Do NOT call super().__init__()
        self._reader = reader
        self.asset_id = reader.asset_id
        self.__last_update: float = 0.0

    # ── Write ops are no-ops (workers write via shared memory) ─────────
    def on_price_change(self, data: dict) -> None:
        pass

    def on_book_snapshot(self, data: dict) -> None:
        pass

    # ── Read ops delegate to SharedBookReader ──────────────────────────
    def snapshot(self, *, fresh: bool = True) -> OrderbookSnapshot:
        snap = self._reader.read_header()
        self.__last_update = snap.timestamp
        return OrderbookSnapshot(
            asset_id=snap.asset_id,
            best_bid=snap.best_bid,
            best_ask=snap.best_ask,
            spread=snap.spread,
            bid_depth_usd=snap.bid_depth_usd,
            ask_depth_usd=snap.ask_depth_usd,
            mid_price=snap.mid_price,
            timestamp=snap.timestamp,
            server_time=snap.server_time,
            fresh=snap.fresh if fresh else False,
            spread_score=snap.spread_score,
        )

    def levels(self, side: str, n: int = 5) -> list[_Level]:
        snap = self._reader.read_full()
        if side.lower() in ("bid", "buy"):
            raw = snap.bid_levels or []
            return [_Level(p, s) for p, s in raw[:n]]
        raw = snap.ask_levels or []
        return [_Level(p, s) for p, s in raw[:n]]

    @property
    def spread_cents(self) -> float:
        snap = self._reader.read_header()
        bb, ba = snap.best_bid, snap.best_ask
        if bb <= 0 or ba <= 0:
            return 0.0
        return round((ba - bb) * 100, 2)

    @property
    def best_bid(self) -> float:
        return self._reader.read_header().best_bid

    @property
    def best_ask(self) -> float:
        return self._reader.read_header().best_ask

    @property
    def book_depth_ratio(self) -> float:
        snap = self._reader.read_header()
        if snap.ask_depth_usd <= 0:
            return 1.0
        return round(snap.bid_depth_usd / snap.ask_depth_usd, 2)

    @property
    def has_data(self) -> bool:
        return self._reader.read_header().timestamp > 0

    @property
    def is_reliable(self) -> bool:
        return self._reader.read_header().is_reliable

    @property
    def seq_gap_rate(self) -> float:
        snap = self._reader.read_header()
        if snap.delta_count <= 0:
            return 0.0
        return snap.desync_total / snap.delta_count

    @property
    def delta_count(self) -> int:
        return self._reader.read_header().delta_count

    @property
    def desync_total(self) -> int:
        return self._reader.read_header().desync_total

    @property
    def _last_update(self) -> float:
        snap = self._reader.read_header()
        self.__last_update = snap.timestamp
        return snap.timestamp

    @_last_update.setter
    def _last_update(self, value: float) -> None:
        pass  # read-only

    @property
    def _last_server_time(self) -> float:
        return self._reader.read_header().server_time

    @_last_server_time.setter
    def _last_server_time(self, value: float) -> None:
        pass  # read-only

    def current_total_depth(self) -> float:
        snap = self._reader.read_header()
        return round(snap.bid_depth_usd + snap.ask_depth_usd, 2)

    def top_depths_usd(self) -> tuple[float, float]:
        snap = self._reader.read_header()
        return round(snap.bid_depth_usd, 2), round(snap.ask_depth_usd, 2)

    def top_depth_ewma(self, side: str) -> float:
        bid_depth, ask_depth = self.top_depths_usd()
        return bid_depth if side.lower() in ("bid", "buy") else ask_depth

    def depth_velocity(self, window_s: float = 2.0) -> float | None:
        # Depth history is maintained in-process by L2OrderBook —
        # not available via shared memory.  Return None (safe default:
        # consumers treat None as "insufficient data, skip check").
        return None

    def depth_near_mid_usd(self, cents: float, max_levels: int = 50) -> float:
        # Use the pre-computed value from the worker (1-cent default).
        # For the fast 50ms ASG loop, this avoids reading all 50 levels.
        if abs(cents - 1.0) < 0.01:
            return self._reader.read_header().depth_near_mid
        # For non-standard cent values, fall back to level-by-level calc.
        snap = self._reader.read_full()
        bb, ba = snap.best_bid, snap.best_ask
        if bb <= 0 or ba <= 0:
            return 0.0
        mid = (bb + ba) / 2.0
        threshold = cents / 100.0
        depth = 0.0
        for levels in (snap.bid_levels or [], snap.ask_levels or []):
            for price, size in levels[:max_levels]:
                if abs(price - mid) <= threshold:
                    depth += price * size
        return depth

    def toxicity_index(self, side: str = "BUY") -> float:
        snap = self._reader.read_header()
        if side.upper() == "SELL":
            return float(snap.sell_toxicity or 0.0)
        return float(snap.buy_toxicity or 0.0)

    def toxicity_metrics(self, side: str = "BUY") -> dict[str, float]:
        return {
            "toxicity_index": round(self.toxicity_index(side), 6),
            "toxicity_depth_evaporation": 0.0,
            "toxicity_sweep_ratio": 0.0,
        }
