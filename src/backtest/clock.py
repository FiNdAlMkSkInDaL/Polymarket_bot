"""
Simulated clock for deterministic backtesting.

Monkey-patches ``time.time`` during a backtest run so that all existing
code that calls ``time.time()`` — Order.created_at default_factory,
Position.created_at, L2OrderBook._last_update, depth_velocity, spread
score timestamps, signal timestamps, fee cache TTL, latency guard, …
— transparently uses the simulated timestamp of the event being replayed.

Usage
─────
    clock = SimClock(start_time=1_700_000_000.0)
    clock.install()        # patches time.time globally
    clock.advance(1_700_000_001.5)
    assert time.time() == 1_700_000_001.5
    clock.uninstall()      # restores real time.time

Or as a context manager::

    with SimClock(start_time=1_700_000_000.0) as clock:
        clock.advance(ts)
        ...
"""

from __future__ import annotations

import time


class SimClock:
    """Simulated wall-clock that replaces ``time.time`` during backtests.

    Parameters
    ----------
    start_time:
        Initial simulated timestamp (Unix epoch seconds).
    """

    __slots__ = ("_current", "_real_time", "_installed")

    def __init__(self, start_time: float = 0.0) -> None:
        self._current: float = start_time
        self._real_time = time.time  # keep a reference before patching
        self._installed: bool = False

    # ── Public API ─────────────────────────────────────────────────────

    def now(self) -> float:
        """Return the current simulated time."""
        return self._current

    def advance(self, ts: float) -> None:
        """Advance simulated time to *ts*.

        Raises ``ValueError`` if *ts* < current time (time must be
        monotonically non-decreasing).
        """
        if ts < self._current:
            raise ValueError(
                f"SimClock: cannot go backwards "
                f"({ts:.6f} < current {self._current:.6f})"
            )
        self._current = ts

    def install(self) -> None:
        """Monkey-patch ``time.time`` with ``self.now``."""
        if self._installed:
            return
        self._real_time = time.time  # snapshot the real function
        time.time = self.now  # type: ignore[assignment]
        self._installed = True

    def uninstall(self) -> None:
        """Restore the real ``time.time``."""
        if not self._installed:
            return
        time.time = self._real_time  # type: ignore[assignment]
        self._installed = False

    @property
    def real_time(self):
        """Access the un-patched ``time.time`` (for wall-clock logging)."""
        return self._real_time

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self) -> SimClock:
        self.install()
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()

    # ── Repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SimClock(current={self._current:.6f}, "
            f"installed={self._installed})"
        )
