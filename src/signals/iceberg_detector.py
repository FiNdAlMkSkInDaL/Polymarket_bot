"""
Iceberg Order Detector — identifies hidden liquidity (iceberg/reserve
orders) by tracking size replenishment at persistent price levels in
the L2 order book.

Theory
──────
An iceberg order is a large order that exchanges (and Polymarket's CLOB)
allow to be submitted with only a portion visible.  When the visible slice
is consumed, the exchange automatically replenishes it from the hidden
reserve.  The signature is:

  1. A level is **fully consumed** (size goes to zero in a delta).
  2. Within a short window, the **same price level** reappears with
     fresh size — typically the same quantity as the previous slice.
  3. This pattern repeats ≥ ``min_refills`` times at the same price.

Implications for mean-reversion:
  - An iceberg **on the same side as our entry** (e.g. bid-side when
    we're buying NO) is favourable — hidden support reduces fill risk.
  - An iceberg **opposing our position** is toxic — a large informed
    seller is consuming the price one slice at a time.

This module is purely informational: it emits ``IcebergSignal`` events
that the bot can log, factor into EQS, or use to adjust sizing.

Integration
───────────
The ``IcebergDetector`` is instantiated per-asset.  The L2OrderBook's
``_apply_delta_changes()`` calls ``detector.on_delta(side, price, old_size,
new_size)`` for every level change, and the detector accumulates evidence.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class IcebergSignal:
    """Emitted when an iceberg order is detected at a price level."""

    asset_id: str
    side: str          # "BUY" or "SELL"
    price: float
    refill_count: int  # how many replenishments observed
    avg_slice_size: float
    estimated_total: float  # refill_count × avg_slice_size
    timestamp: float
    confidence: float  # 0-1 how confident we are this is really an iceberg


class IcebergDetector:
    """Per-asset iceberg order detector.

    Tracks level removals and subsequent replenishments to identify
    hidden reserve orders.

    Parameters
    ----------
    asset_id:
        The token ID this detector monitors.
    refill_window_s:
        Maximum seconds between consumption and replenishment to count
        as a single iceberg refill cycle.
    min_refills:
        Minimum number of refill cycles at the same price before an
        iceberg signal is emitted.
    size_tolerance_pct:
        Maximum fractional deviation between refill sizes to be
        considered the same iceberg (e.g. 0.30 = ±30%).
    """

    def __init__(
        self,
        asset_id: str,
        *,
        refill_window_s: float = 0.0,
        min_refills: int = 0,
        size_tolerance_pct: float = 0.0,
    ):
        strat = settings.strategy
        self.asset_id = asset_id
        self._refill_window = refill_window_s or getattr(strat, "iceberg_refill_window_s", 5.0)
        self._min_refills = min_refills or getattr(strat, "iceberg_min_refills", 3)
        self._size_tolerance = size_tolerance_pct or getattr(strat, "iceberg_size_tolerance_pct", 0.30)

        # Tracking state per (side, price) level
        # Key: (side, price) → refill history
        self._removed_levels: dict[tuple[str, float], float] = {}  # → timestamp of last removal
        self._removed_sizes: dict[tuple[str, float], float] = {}   # → size at removal
        self._refill_counts: dict[tuple[str, float], int] = defaultdict(int)
        self._slice_sizes: dict[tuple[str, float], list[float]] = defaultdict(list)

        # Active iceberg signals (keyed by (side, price))
        self._active_icebergs: dict[tuple[str, float], IcebergSignal] = {}

        # Cleanup counter — evict stale tracking entries periodically
        self._delta_count = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def on_level_change(
        self,
        side: str,
        price: float,
        old_size: float,
        new_size: float,
    ) -> IcebergSignal | None:
        """Process a single level change from a book delta.

        Parameters
        ----------
        side:
            "BUY" or "SELL"
        price:
            The price level that changed.
        old_size:
            Previous size at this level (0 if the level didn't exist).
        new_size:
            New size at this level (0 if the level was removed).

        Returns
        -------
        IcebergSignal or None:
            Emitted when an iceberg is detected or updated with a new
            refill cycle.
        """
        key = (side, price)
        now = time.time()

        self._delta_count += 1
        if self._delta_count % 500 == 0:
            self._cleanup_stale(now)

        # ── Level consumed (size → 0) ─────────────────────────────────
        if new_size <= 0 and old_size > 0:
            self._removed_levels[key] = now
            self._removed_sizes[key] = old_size
            return None

        # ── Level (re)appeared ────────────────────────────────────────
        if new_size > 0 and key in self._removed_levels:
            removed_ts = self._removed_levels[key]
            removed_size = self._removed_sizes.get(key, 0)

            # Check timing window
            elapsed = now - removed_ts
            if elapsed > self._refill_window:
                # Too long — not an iceberg refill, clean up
                self._removed_levels.pop(key, None)
                self._removed_sizes.pop(key, None)
                return None

            # Check size similarity
            if removed_size > 0:
                size_ratio = new_size / removed_size
                if abs(size_ratio - 1.0) > self._size_tolerance:
                    # Size too different — not a refill of the same iceberg
                    self._removed_levels.pop(key, None)
                    self._removed_sizes.pop(key, None)
                    return None

            # ── This looks like a refill cycle ────────────────────────
            self._refill_counts[key] += 1
            self._slice_sizes[key].append(new_size)

            # Reset removal state for next detection cycle
            self._removed_levels.pop(key, None)
            self._removed_sizes.pop(key, None)

            refill_count = self._refill_counts[key]
            if refill_count >= self._min_refills:
                sizes = self._slice_sizes[key]
                avg_slice = sum(sizes) / len(sizes)
                confidence = min(1.0, refill_count / (self._min_refills * 2))

                signal = IcebergSignal(
                    asset_id=self.asset_id,
                    side=side,
                    price=price,
                    refill_count=refill_count,
                    avg_slice_size=round(avg_slice, 2),
                    estimated_total=round(refill_count * avg_slice, 2),
                    timestamp=now,
                    confidence=round(confidence, 3),
                )

                self._active_icebergs[key] = signal

                log.info(
                    "iceberg_detected",
                    asset_id=self.asset_id,
                    side=side,
                    price=price,
                    refills=refill_count,
                    avg_slice=round(avg_slice, 2),
                    est_total=round(refill_count * avg_slice, 2),
                    confidence=round(confidence, 3),
                )

                return signal

        return None

    @property
    def active_icebergs(self) -> dict[tuple[str, float], IcebergSignal]:
        """Currently active iceberg signals, keyed by (side, price)."""
        return dict(self._active_icebergs)

    def has_iceberg(self, side: str) -> bool:
        """Check if any iceberg is active on a given side."""
        return any(s == side for (s, _) in self._active_icebergs)

    def strongest_iceberg(self, side: str) -> IcebergSignal | None:
        """Return the highest-confidence iceberg on the given side."""
        candidates = [
            sig for (s, _), sig in self._active_icebergs.items()
            if s == side
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.confidence)

    def reset(self) -> None:
        """Clear all state."""
        self._removed_levels.clear()
        self._removed_sizes.clear()
        self._refill_counts.clear()
        self._slice_sizes.clear()
        self._active_icebergs.clear()

    # ── Internal ───────────────────────────────────────────────────────────

    def _cleanup_stale(self, now: float) -> None:
        """Evict tracking entries older than 2× the refill window."""
        cutoff = now - self._refill_window * 2
        stale_keys = [
            k for k, ts in self._removed_levels.items()
            if ts < cutoff
        ]
        for k in stale_keys:
            self._removed_levels.pop(k, None)
            self._removed_sizes.pop(k, None)

        # Also expire icebergs not refreshed recently
        stale_icebergs = [
            k for k, sig in self._active_icebergs.items()
            if sig.timestamp < cutoff
        ]
        for k in stale_icebergs:
            self._active_icebergs.pop(k, None)
            self._refill_counts.pop(k, None)
            self._slice_sizes.pop(k, None)
