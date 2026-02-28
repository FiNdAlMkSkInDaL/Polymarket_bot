"""
L2 Order Book Reconstruction Engine — maintains a full-depth local order
book from WebSocket deltas with sequence-number tracking and automatic
desync recovery.

State Machine
─────────────
  EMPTY → subscribe sent → BUFFERING
  BUFFERING → REST snapshot loaded + buffered deltas replayed → SYNCED
  SYNCED → each delta seq checked → if gap → DESYNCED
  DESYNCED → wipe book, re-fetch snapshot → BUFFERING → SYNCED

Public API is a superset of ``OrderbookTracker`` so all existing consumers
(heartbeat, sizer, chaser, ghost CB, scorer, TP rescale, adverse-selection,
MTI classification) work without modification.

Design notes:
  - Uses ``sortedcontainers.SortedDict`` for O(log n) insert/delete.
  - Delta application is synchronous and non-blocking — no queue needed.
  - Spread score is recomputed only when BBO actually changes.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from sortedcontainers import SortedDict

from src.core.config import settings
from src.core.logger import get_logger
from src.data.spread_score import SpreadScore, compute_spread_score
from src.data.types import Level as _Level

log = get_logger(__name__)


# ── Book state enum ────────────────────────────────────────────────────────
class BookState(enum.Enum):
    EMPTY = "empty"
    BUFFERING = "buffering"
    SYNCED = "synced"
    DESYNCED = "desynced"


# ── Snapshot dataclass (re-exported for convenience) ───────────────────────
@dataclass
class L2Snapshot:
    """Point-in-time L2 order book summary for one asset."""

    asset_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    mid_price: float = 0.0
    timestamp: float = 0.0
    server_time: float = 0.0
    fresh: bool = True
    spread_score: float = 0.0
    state: BookState = BookState.EMPTY
    seq: int = 0


class L2OrderBook:
    """Full-depth L2 order book with sequence tracking and desync recovery.

    Parameters
    ----------
    asset_id:
        The token ID this book tracks.
    on_bbo_change:
        Optional async callback invoked whenever the BBO changes.
        Signature: ``async def cb(asset_id: str, score: SpreadScore) -> None``
    on_desync:
        Optional async callback invoked on sequence gap detection.
        Signature: ``async def cb(asset_id: str) -> None``
    max_levels:
        Maximum depth to maintain per side (default from config).
    """

    def __init__(
        self,
        asset_id: str,
        *,
        on_bbo_change: Callable[..., Any] | None = None,
        on_desync: Callable[..., Any] | None = None,
        max_levels: int | None = None,
    ):
        self.asset_id = asset_id
        self._on_bbo_change = on_bbo_change
        self._on_desync = on_desync
        self._max_levels = max_levels or settings.strategy.l2_max_levels

        # ── Core book data (SortedDict: price → size) ─────────────────
        # Bids: negated keys so the highest bid is first in iteration
        self._bids: SortedDict = SortedDict()   # neg_price → size
        # Asks: natural ordering so the lowest ask is first
        self._asks: SortedDict = SortedDict()   # price → size

        # ── Sequence tracking ─────────────────────────────────────────
        self._seq: int = -1
        self._state: BookState = BookState.EMPTY
        self._desync_count: int = 0

        # ── Delta buffer (used during BUFFERING state) ────────────────
        self._delta_buffer: deque[dict] = deque(
            maxlen=settings.strategy.l2_delta_buffer_size
        )

        # ── BBO cache for change detection ────────────────────────────
        self._prev_best_bid: float = 0.0
        self._prev_best_ask: float = 0.0

        # ── Spread score ──────────────────────────────────────────────
        self._spread_score: SpreadScore = SpreadScore()

        # ── Timestamps ────────────────────────────────────────────────
        self._last_update: float = 0.0
        self._last_server_time: float = 0.0

        # ── Depth history for ghost liquidity detection ───────────────
        self._depth_history: deque[tuple[float, float]] = deque(maxlen=40)

        # ── Snapshot fetch lock (prevents concurrent fetches) ─────────
        self._snapshot_lock = asyncio.Lock()

    # ═══════════════════════════════════════════════════════════════════════
    #  State machine
    # ═══════════════════════════════════════════════════════════════════════
    @property
    def state(self) -> BookState:
        return self._state

    @property
    def seq(self) -> int:
        return self._seq

    # ═══════════════════════════════════════════════════════════════════════
    #  Snapshot loading (called by L2WebSocket on connect/desync)
    # ═══════════════════════════════════════════════════════════════════════
    def begin_buffering(self) -> None:
        """Transition to BUFFERING state — deltas are buffered until
        a REST snapshot is loaded."""
        self._state = BookState.BUFFERING
        self._delta_buffer.clear()
        log.debug("l2_buffering", asset_id=self.asset_id)

    async def load_snapshot(
        self,
        snapshot_data: dict,
    ) -> bool:
        """Apply a REST snapshot and replay any buffered deltas.

        Parameters
        ----------
        snapshot_data:
            Dict with keys ``bids``, ``asks`` (lists of ``{price, size}``),
            and optionally ``seq`` / ``sequence`` / ``timestamp``.

        Returns
        -------
        bool
            True if the snapshot was loaded and book is now SYNCED.
        """
        async with self._snapshot_lock:
            return self._apply_snapshot(snapshot_data)

    def _apply_snapshot(self, data: dict) -> bool:
        """Internal: replace book state from snapshot dict."""
        bids_raw = data.get("bids") or []
        asks_raw = data.get("asks") or []

        # Parse sequence number
        snap_seq = _parse_int(
            data.get("seq") or data.get("sequence") or data.get("seq_num"), -1
        )

        # Rebuild bids
        self._bids.clear()
        for b in bids_raw:
            try:
                price = float(b["price"])
                size = float(b["size"])
                if price > 0 and size > 0:
                    self._bids[-price] = size   # negated for desc order
            except (KeyError, TypeError, ValueError):
                continue
        self._trim_side(self._bids)

        # Rebuild asks
        self._asks.clear()
        for a in asks_raw:
            try:
                price = float(a["price"])
                size = float(a["size"])
                if price > 0 and size > 0:
                    self._asks[price] = size
            except (KeyError, TypeError, ValueError):
                continue
        self._trim_side(self._asks)

        # Set sequence
        if snap_seq >= 0:
            self._seq = snap_seq
        else:
            self._seq = 0

        self._last_update = time.time()
        self._extract_server_time(data)

        # Replay buffered deltas that are newer than the snapshot
        replayed = 0
        for delta in self._delta_buffer:
            delta_seq = _parse_int(
                delta.get("seq") or delta.get("sequence") or delta.get("seq_num"), -1
            )
            if delta_seq > self._seq:
                self._apply_delta_changes(delta)
                if delta_seq == self._seq + 1:
                    self._seq = delta_seq
                elif delta_seq > self._seq + 1:
                    # Gap in buffered deltas — skip to this seq
                    self._seq = delta_seq
                replayed += 1

        self._delta_buffer.clear()
        self._state = BookState.SYNCED
        self._desync_count = 0

        # Update BBO + spread score (also records depth if BBO changed)
        self._update_bbo_and_score()

        log.info(
            "l2_synced",
            asset_id=self.asset_id,
            seq=self._seq,
            bids=len(self._bids),
            asks=len(self._asks),
            replayed=replayed,
        )
        return True

    # ═══════════════════════════════════════════════════════════════════════
    #  Delta processing
    # ═══════════════════════════════════════════════════════════════════════
    def on_delta(self, data: dict) -> bool:
        """Process an incoming L2 delta message.

        Returns True if the delta was applied successfully, False if it
        was buffered or caused a desync.

        Expected schema::

            {
                "event_type": "price_change",
                "asset_id": "0x...",
                "seq": 12345,
                "changes": [
                    {"side": "BUY", "price": "0.47", "size": "123.5"},
                    {"side": "SELL", "price": "0.53", "size": "0"},
                ]
            }
        """
        if self._state == BookState.EMPTY:
            # Not yet subscribed — ignore
            return False

        if self._state == BookState.BUFFERING:
            self._delta_buffer.append(data)
            return False

        if self._state == BookState.DESYNCED:
            # We're waiting for a new snapshot — buffer in case recovery starts
            self._delta_buffer.append(data)
            return False

        # ── SYNCED: check sequence continuity ─────────────────────────
        delta_seq = _parse_int(
            data.get("seq") or data.get("sequence") or data.get("seq_num"), -1
        )

        if delta_seq >= 0 and self._seq >= 0:
            if delta_seq <= self._seq:
                # Duplicate or stale — ignore
                return False
            if delta_seq != self._seq + 1:
                # Sequence gap detected → DESYNCED
                log.warning(
                    "l2_seq_gap",
                    asset_id=self.asset_id,
                    expected=self._seq + 1,
                    received=delta_seq,
                )
                self._trigger_desync()
                self._delta_buffer.append(data)
                return False
            # Sequence is correct — advance
            self._seq = delta_seq

        # ── Apply the delta ───────────────────────────────────────────
        self._apply_delta_changes(data)
        now = time.time()
        self._last_update = now
        self._extract_server_time(data)
        self._update_bbo_and_score(now)
        return True

    def _apply_delta_changes(self, data: dict) -> None:
        """Apply price/size changes from a delta message."""
        changes = data.get("changes") or data.get("data") or []
        if isinstance(changes, dict):
            changes = [changes]

        # Fallback: top-level price/side/size
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

            if side in ("BUY", "BID"):
                neg_price = -price
                if size <= 0:
                    self._bids.pop(neg_price, None)
                else:
                    self._bids[neg_price] = size
            elif side in ("SELL", "ASK"):
                if size <= 0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = size

        # Trim if over max levels
        self._trim_side(self._bids)
        self._trim_side(self._asks)

    def _trigger_desync(self) -> None:
        """Move to DESYNCED state, wipe the book, signal for re-snapshot."""
        self._state = BookState.DESYNCED
        self._desync_count += 1
        self._bids.clear()
        self._asks.clear()
        self._spread_score = SpreadScore()
        self._prev_best_bid = 0.0
        self._prev_best_ask = 0.0
        self._delta_buffer.clear()
        log.warning(
            "l2_desynced",
            asset_id=self.asset_id,
            desync_count=self._desync_count,
        )
        # Signal callback (L2WebSocket will trigger re-snapshot)
        if self._on_desync is not None:
            try:
                result = self._on_desync(self.asset_id)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════
    #  BBO tracking & spread score
    # ═══════════════════════════════════════════════════════════════════════
    def _update_bbo_and_score(self, now: float | None = None) -> None:
        """Recompute spread score if BBO changed."""
        bb = self.best_bid
        ba = self.best_ask

        if bb == self._prev_best_bid and ba == self._prev_best_ask:
            return  # No change

        self._prev_best_bid = bb
        self._prev_best_ask = ba

        if now is None:
            now = time.time()

        if bb <= 0 or ba <= 0:
            self._spread_score = SpreadScore(timestamp=now)
            return

        # Detect crossed book → treat as desync
        if bb >= ba and self._state == BookState.SYNCED:
            log.warning(
                "l2_crossed_book",
                asset_id=self.asset_id,
                best_bid=bb,
                best_ask=ba,
            )
            self._trigger_desync()
            return

        top_n = settings.strategy.l2_spread_score_top_n

        bid_levels = [
            (-neg_p, self._bids[neg_p])
            for neg_p in self._bids.islice(stop=top_n)
        ]
        ask_levels = [
            (p, self._asks[p])
            for p in self._asks.islice(stop=top_n)
        ]

        self._spread_score = compute_spread_score(
            bb, ba, bid_levels, ask_levels, top_n=top_n, timestamp=now,
        )

        # Record depth only when BBO actually changed
        self._record_depth(now)

        # Fire BBO change callback
        if self._on_bbo_change is not None:
            try:
                result = self._on_bbo_change(self.asset_id, self._spread_score)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════
    #  Public API — compatible with OrderbookTracker
    # ═══════════════════════════════════════════════════════════════════════
    @property
    def best_bid(self) -> float:
        if not self._bids:
            return 0.0
        first_neg = self._bids.keys()[0]
        return -first_neg

    @property
    def best_ask(self) -> float:
        if not self._asks:
            return 0.0
        return self._asks.keys()[0]

    @property
    def spread_cents(self) -> float:
        """Current spread in cents."""
        bb = self.best_bid
        ba = self.best_ask
        if bb <= 0 or ba <= 0:
            return 0.0
        return round((ba - bb) * 100, 2)

    @property
    def book_depth_ratio(self) -> float:
        """Bid depth / ask depth ratio.  >1 = more resting buy interest."""
        bid_d = sum(
            (-neg_p) * self._bids[neg_p]
            for neg_p in self._bids.islice(stop=5)
        )
        ask_d = sum(
            p * self._asks[p]
            for p in self._asks.islice(stop=5)
        )
        if ask_d <= 0:
            return 1.0
        return round(bid_d / ask_d, 2)

    @property
    def has_data(self) -> bool:
        return self._last_update > 0

    @property
    def spread_score_value(self) -> float:
        """Current spread score (0-100)."""
        return self._spread_score.score

    @property
    def spread_score_obj(self) -> SpreadScore:
        """Full spread score object."""
        return self._spread_score

    def snapshot(self, *, fresh: bool = True) -> L2Snapshot:
        """Return current top-of-book summary."""
        bb = self.best_bid
        ba = self.best_ask
        spread = (ba - bb) if (ba > 0 and bb > 0) else 0.0
        mid = (bb + ba) / 2.0 if (ba > 0 and bb > 0) else 0.0

        bid_depth = sum(
            (-neg_p) * self._bids[neg_p]
            for neg_p in self._bids.islice(stop=5)
        )
        ask_depth = sum(
            p * self._asks[p]
            for p in self._asks.islice(stop=5)
        )

        return L2Snapshot(
            asset_id=self.asset_id,
            best_bid=bb,
            best_ask=ba,
            spread=round(spread, 4),
            bid_depth_usd=round(bid_depth, 2),
            ask_depth_usd=round(ask_depth, 2),
            mid_price=round(mid, 4),
            timestamp=self._last_update,
            server_time=self._last_server_time,
            fresh=fresh,
            spread_score=self._spread_score.score,
            state=self._state,
            seq=self._seq,
        )

    def levels(self, side: str, n: int = 5) -> list[_Level]:
        """Return top *n* levels for *side* (``"bid"``/``"buy"``
        or ``"ask"``/``"sell"``).  Returns copies."""
        if side.lower() in ("bid", "buy"):
            return [
                _Level(-neg_p, self._bids[neg_p])
                for neg_p in self._bids.islice(stop=n)
            ]
        return [
            _Level(p, self._asks[p])
            for p in self._asks.islice(stop=n)
        ]

    # ── Ghost liquidity support ────────────────────────────────────────
    def _record_depth(self, now: float | None = None) -> None:
        depth = self.current_total_depth()
        self._depth_history.append(((now or time.time()), depth))

    def current_total_depth(self) -> float:
        bid_d = sum(
            (-neg_p) * self._bids[neg_p]
            for neg_p in self._bids.islice(stop=5)
        )
        ask_d = sum(
            p * self._asks[p]
            for p in self._asks.islice(stop=5)
        )
        return round(bid_d + ask_d, 2)

    def depth_velocity(self, window_s: float = 2.0) -> float | None:
        """Compute fractional depth change over *window_s* seconds."""
        if len(self._depth_history) < 2:
            return None

        now = time.time()
        target_time = now - window_s

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

    # ── Internal helpers ───────────────────────────────────────────────
    def _trim_side(self, side: SortedDict) -> None:
        """Keep only the top ``_max_levels`` entries."""
        while len(side) > self._max_levels:
            side.popitem()       # removes last (worst) level

    def _extract_server_time(self, data: dict) -> None:
        """Extract and normalise server timestamp."""
        srv = data.get("timestamp") or data.get("server_timestamp") or data.get("ts")
        if srv is not None:
            try:
                raw = float(srv)
                if raw > 1e15:
                    raw /= 1_000_000
                elif raw > 1e12:
                    raw /= 1_000
                self._last_server_time = raw
            except (TypeError, ValueError):
                pass

    def reset(self) -> None:
        """Full wipe — returns book to EMPTY state."""
        self._bids.clear()
        self._asks.clear()
        self._seq = -1
        self._state = BookState.EMPTY
        self._delta_buffer.clear()
        self._spread_score = SpreadScore()
        self._prev_best_bid = 0.0
        self._prev_best_ask = 0.0
        self._last_update = 0.0
        self._last_server_time = 0.0
        self._depth_history.clear()
        self._desync_count = 0


# ── REST snapshot fetcher ──────────────────────────────────────────────────
async def fetch_l2_snapshot(
    asset_id: str,
    *,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> dict | None:
    """Fetch the current L2 book snapshot from the CLOB REST API.

    Returns the parsed JSON dict on success, or ``None`` on failure.
    """
    url = base_url or settings.clob_book_url
    timeout = timeout_s or settings.strategy.l2_snapshot_timeout_s

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params={"token_id": asset_id})
            resp.raise_for_status()
            data = resp.json()
            # The REST API may wrap the book in a top-level key
            if "market" in data and isinstance(data["market"], dict):
                data = data["market"]
            return data
    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
        log.warning(
            "l2_snapshot_fetch_failed",
            asset_id=asset_id,
            error=str(exc),
        )
        return None


# ── Helpers ────────────────────────────────────────────────────────────────
def _parse_int(val: Any, default: int = -1) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default
