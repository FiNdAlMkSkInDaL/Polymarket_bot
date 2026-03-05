"""
Exception Circuit Breaker — rolling error counter for critical async loops.

Prevents blanket ``except Exception`` from silently masking severe state
corruption.  Each guarded loop instantiates a breaker with a threshold
(default 5) and window (default 60 s).  When consecutive unexpected
exceptions within the window exceed the threshold, the breaker trips and
the caller must trigger a graceful shutdown.

Usage::

    breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)

    while self._running:
        try:
            ...
        except (KeyError, asyncio.TimeoutError):
            # expected, non-fatal — continue
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            if breaker.record():
                log.critical("circuit_breaker_tripped", ...)
                # trigger shutdown
                break
            log.error("...", exc_info=True)
"""

from __future__ import annotations

import time
from collections import deque


class ExceptionCircuitBreaker:
    """Rolling-window error counter for async loop exception guards.

    Parameters
    ----------
    threshold:
        Maximum number of unexpected exceptions allowed within *window_s*
        before the breaker trips.
    window_s:
        Rolling time window in seconds.  Errors older than this are
        automatically evicted.
    """

    def __init__(self, threshold: int = 5, window_s: float = 60.0) -> None:
        self._threshold = threshold
        self._window_s = window_s
        self._timestamps: deque[float] = deque()
        self._tripped = False

    # ------------------------------------------------------------------
    @property
    def tripped(self) -> bool:
        """Whether the breaker has already been tripped."""
        return self._tripped

    @property
    def recent_errors(self) -> int:
        """Number of errors currently within the rolling window."""
        self._evict()
        return len(self._timestamps)

    # ------------------------------------------------------------------
    def record(self, now: float | None = None) -> bool:
        """Record an unexpected exception timestamp.

        Returns ``True`` if the threshold is breached (breaker trips).
        Once tripped, always returns ``True`` without further recording.
        """
        if self._tripped:
            return True

        now = now if now is not None else time.monotonic()
        self._timestamps.append(now)
        self._evict(now)

        if len(self._timestamps) >= self._threshold:
            self._tripped = True
            return True
        return False

    def reset(self) -> None:
        """Reset the breaker (for testing or recovery scenarios)."""
        self._timestamps.clear()
        self._tripped = False

    # ------------------------------------------------------------------
    def _evict(self, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        cutoff = now - self._window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
