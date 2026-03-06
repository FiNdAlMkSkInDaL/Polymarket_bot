"""
Worker heartbeat protocol — lightweight liveness signalling for child
processes managed by the ``ProcessManager``.

Each worker process gets a ``WorkerHeartbeatSender`` that periodically
writes ``time.monotonic()`` into a shared ``multiprocessing.Value``.
The main process uses ``WorkerHeartbeatChecker`` to detect stale workers.

This is *separate* from Pillar 8 ``BookHeartbeat`` which monitors
order book data freshness.  This module monitors process liveness only.
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorkerHealth:
    """Point-in-time health snapshot for a single worker."""

    worker_id: str
    pid: int | None
    last_heartbeat: float  # monotonic timestamp
    alive: bool  # True if process is still running
    stale: bool  # True if heartbeat is older than threshold
    stale_duration: float  # seconds since last heartbeat


class WorkerHeartbeatSender:
    """Used inside a worker process to publish liveness.

    Call ``beat()`` periodically (e.g. every 500 ms) from the worker's
    main loop.

    Parameters
    ----------
    heartbeat_value:
        A ``multiprocessing.Value('d')`` shared with the main process.
    """

    def __init__(self, heartbeat_value: Any) -> None:
        self._value = heartbeat_value

    def beat(self) -> None:
        """Record the current monotonic timestamp as the heartbeat."""
        self._value.value = time.monotonic()


class WorkerHeartbeatChecker:
    """Used by the main process to monitor worker liveness.

    Parameters
    ----------
    stale_threshold_s:
        A worker is considered stale if its heartbeat is older than this
        many seconds.  Default 3.0 s.
    """

    def __init__(self, stale_threshold_s: float = 3.0) -> None:
        self._threshold = stale_threshold_s
        # worker_id → (heartbeat_value, Process)
        self._workers: dict[str, tuple[Any, multiprocessing.Process | None]] = {}

    def register(
        self,
        worker_id: str,
        heartbeat_value: Any,
        process: multiprocessing.Process | None = None,
    ) -> None:
        """Register a worker to monitor."""
        self._workers[worker_id] = (heartbeat_value, process)

    def unregister(self, worker_id: str) -> None:
        """Stop monitoring a worker."""
        self._workers.pop(worker_id, None)

    def check(self, worker_id: str) -> WorkerHealth:
        """Check the health of a specific worker."""
        entry = self._workers.get(worker_id)
        if entry is None:
            return WorkerHealth(
                worker_id=worker_id,
                pid=None,
                last_heartbeat=0.0,
                alive=False,
                stale=True,
                stale_duration=float("inf"),
            )

        hb_value, proc = entry
        last_hb = hb_value.value
        now = time.monotonic()
        stale_duration = now - last_hb if last_hb > 0 else float("inf")
        is_alive = proc.is_alive() if proc is not None else False
        is_stale = stale_duration > self._threshold

        return WorkerHealth(
            worker_id=worker_id,
            pid=proc.pid if proc is not None else None,
            last_heartbeat=last_hb,
            alive=is_alive,
            stale=is_stale,
            stale_duration=stale_duration,
        )

    def check_all(self) -> list[WorkerHealth]:
        """Check health of all registered workers."""
        return [self.check(wid) for wid in self._workers]

    def any_stale(self) -> bool:
        """Return True if any registered worker is stale or dead."""
        for health in self.check_all():
            if health.stale or not health.alive:
                return True
        return False

    def dead_workers(self) -> list[WorkerHealth]:
        """Return list of workers that are not alive."""
        return [h for h in self.check_all() if not h.alive]

    def stale_workers(self) -> list[WorkerHealth]:
        """Return list of workers whose heartbeat is stale."""
        return [h for h in self.check_all() if h.stale]
