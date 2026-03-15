"""
Process Manager — orchestrates the lifecycle of worker processes for the
multi-core distributed architecture.

Responsibilities
----------------
- Spawn and stop L2 reconstruction workers and the PCE computation worker.
- Monitor worker liveness via ``WorkerHeartbeatChecker``.
- Trigger emergency stop (cancel all orders, halt bot) if any worker
  crashes or becomes unresponsive.
- Allocate and clean up shared memory blocks for IPC.

The ProcessManager runs a periodic health-check coroutine inside the main
asyncio event loop.  Workers themselves run their own asyncio loops in
separate OS processes.
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing
import os
import queue as _queue_mod
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.ipc import (
    BLOCK_SIZE,
    SharedBookReader,
    allocate_shm,
    cleanup_shm,
)
from src.core.logger import get_logger
from src.core.worker_heartbeat import WorkerHeartbeatChecker

log = get_logger(__name__)

# Use "spawn" on all platforms for safety (fork + asyncio = trouble).
_MP_CTX = multiprocessing.get_context("spawn")


@dataclass
class WorkerRecord:
    """Internal bookkeeping for a managed worker process."""

    worker_id: str
    process: multiprocessing.Process | None = None
    heartbeat_value: Any = None  # multiprocessing.Value('d')
    shutdown_event: Any = None  # multiprocessing.Event
    circuit_breaker_event: Any = None  # multiprocessing.Event — trips on fatal
    started: bool = False


@dataclass
class L2ShardAssignment:
    """Maps a set of asset IDs to a specific L2 worker."""

    worker_id: str
    asset_ids: list[str]
    shm_names: dict[str, str]  # asset_id → shm segment name


class ProcessManager:
    """Orchestrates worker process lifecycle and health monitoring.

    Parameters
    ----------
    on_emergency_stop:
        Async callback invoked when a worker failure requires the bot to
        halt.  Typically ``bot.stop()``.
    n_l2_workers:
        Number of L2 reconstruction workers.  Defaults to
        ``min(cpu_count() - 2, 4)`` but at least 1.
    heartbeat_stale_s:
        Seconds after which a silent worker is considered dead.
    health_check_interval_s:
        How often the health-check loop runs.
    """

    def __init__(
        self,
        *,
        on_emergency_stop: Callable[..., Any] | None = None,
        n_l2_workers: int | None = None,
        heartbeat_stale_s: float = 3.0,
        health_check_interval_s: float = 1.0,
        stale_startup_grace_s: float = 15.0,
    ) -> None:
        cpu = os.cpu_count() or 4
        self._n_l2_workers = n_l2_workers or max(1, min(cpu - 2, 4))
        self._on_emergency_stop = on_emergency_stop
        self._health_interval = health_check_interval_s
        self._stale_startup_grace_s = stale_startup_grace_s
        self._heartbeat_checker = WorkerHeartbeatChecker(stale_threshold_s=heartbeat_stale_s)

        # Worker registry
        self._workers: dict[str, WorkerRecord] = {}
        # Shared memory blocks owned by this manager (asset_id → shm obj)
        self._shm_blocks: dict[str, multiprocessing.shared_memory.SharedMemory] = {}
        # asset_id → shm_name mapping (passed to readers)
        self._shm_names: dict[str, str] = {}
        # Readers for the main process
        self._readers: dict[str, SharedBookReader] = {}

        # L2 worker shard assignments
        self._l2_shards: list[L2ShardAssignment] = []
        # Per-worker control queues for dynamic market add/remove
        self._l2_control_queues: dict[str, multiprocessing.Queue] = {}

        # PCE worker queues
        self._pce_input_queue: multiprocessing.Queue | None = None
        self._pce_output_queue: multiprocessing.Queue | None = None
        self._pce_var_request_queue: multiprocessing.Queue | None = None
        self._pce_var_response_queue: multiprocessing.Queue | None = None

        # BBO event queue (all L2 workers → main process)
        self._bbo_queue: multiprocessing.Queue | None = None

        self._running = False
        self._emergency_triggered = False

        # Belt-and-suspenders: ensure cleanup runs even if bot.stop() is
        # bypassed by an unhandled exception.  Won't fire on SIGKILL but
        # covers interpreter exit paths.
        atexit.register(self._atexit_cleanup)

    # ═══════════════════════════════════════════════════════════════════════
    #  Properties
    # ═══════════════════════════════════════════════════════════════════════
    @property
    def n_l2_workers(self) -> int:
        return self._n_l2_workers

    @property
    def bbo_queue(self) -> multiprocessing.Queue:
        assert self._bbo_queue is not None
        return self._bbo_queue

    @property
    def pce_input_queue(self) -> multiprocessing.Queue:
        assert self._pce_input_queue is not None
        return self._pce_input_queue

    @property
    def pce_output_queue(self) -> multiprocessing.Queue:
        assert self._pce_output_queue is not None
        return self._pce_output_queue

    @property
    def pce_var_request_queue(self) -> multiprocessing.Queue:
        assert self._pce_var_request_queue is not None
        return self._pce_var_request_queue

    @property
    def pce_var_response_queue(self) -> multiprocessing.Queue:
        assert self._pce_var_response_queue is not None
        return self._pce_var_response_queue

    @property
    def heartbeat_checker(self) -> WorkerHeartbeatChecker:
        return self._heartbeat_checker

    def get_reader(self, asset_id: str) -> SharedBookReader | None:
        return self._readers.get(asset_id)

    def get_all_readers(self) -> dict[str, SharedBookReader]:
        return dict(self._readers)

    # ═══════════════════════════════════════════════════════════════════════
    #  Shared Memory Allocation
    # ═══════════════════════════════════════════════════════════════════════
    def allocate_books(self, asset_ids: list[str]) -> dict[str, str]:
        """Allocate shared memory blocks for all asset IDs.

        Returns a mapping of ``asset_id → shm_name``.
        """
        for aid in asset_ids:
            if aid in self._shm_blocks:
                continue
            shm, name = allocate_shm(aid)
            self._shm_blocks[aid] = shm
            self._shm_names[aid] = name
            self._readers[aid] = SharedBookReader(aid, name)
        return dict(self._shm_names)

    # ═══════════════════════════════════════════════════════════════════════
    #  L2 Worker Management
    # ═══════════════════════════════════════════════════════════════════════
    def _build_l2_shards(self, asset_ids: list[str]) -> list[L2ShardAssignment]:
        """Partition assets into N shards, keeping same-market tokens together.

        We sort by asset_id so YES/NO tokens (which are adjacent hex strings
        on the same condition) land on the same shard.
        """
        sorted_ids = sorted(asset_ids)
        shards: list[L2ShardAssignment] = []
        n = self._n_l2_workers

        # Round-robin assignment
        buckets: list[list[str]] = [[] for _ in range(n)]
        for i, aid in enumerate(sorted_ids):
            buckets[i % n].append(aid)

        for idx, bucket in enumerate(buckets):
            wid = f"l2_worker_{idx}"
            shard_shm = {aid: self._shm_names[aid] for aid in bucket}
            shards.append(L2ShardAssignment(
                worker_id=wid,
                asset_ids=bucket,
                shm_names=shard_shm,
            ))
        return shards

    def start_l2_workers(self, asset_ids: list[str]) -> None:
        """Allocate shared memory and spawn L2 reconstruction workers.

        Must be called from the main process before starting the event loop
        (or from an async context via ``await asyncio.to_thread(...)``).
        """
        from src.data.l2_worker import l2_worker_main

        # Allocate shared memory for all assets
        self.allocate_books(asset_ids)

        # Create the BBO event queue
        self._bbo_queue = _MP_CTX.Queue(maxsize=2000)

        # Build shard assignments
        self._l2_shards = self._build_l2_shards(asset_ids)

        for shard in self._l2_shards:
            wid = shard.worker_id
            hb_value = _MP_CTX.Value("d", 0.0)
            shutdown_evt = _MP_CTX.Event()
            cb_evt = _MP_CTX.Event()  # circuit breaker event
            ctrl_queue = _MP_CTX.Queue(maxsize=100)

            proc = _MP_CTX.Process(
                target=l2_worker_main,
                args=(
                    wid,
                    shard.asset_ids,
                    shard.shm_names,
                    self._bbo_queue,
                    hb_value,
                    shutdown_evt,
                    cb_evt,
                    ctrl_queue,
                ),
                name=wid,
                daemon=True,
            )

            record = WorkerRecord(
                worker_id=wid,
                process=proc,
                heartbeat_value=hb_value,
                shutdown_event=shutdown_evt,
                circuit_breaker_event=cb_evt,
            )
            self._workers[wid] = record
            self._l2_control_queues[wid] = ctrl_queue
            self._heartbeat_checker.register(wid, hb_value, proc)

            proc.start()
            record.started = True
            log.info(
                "l2_worker_started",
                worker_id=wid,
                pid=proc.pid,
                n_assets=len(shard.asset_ids),
            )

    # ═══════════════════════════════════════════════════════════════════════
    #  PCE Worker Management
    # ═══════════════════════════════════════════════════════════════════════
    def start_pce_worker(
        self,
        *,
        data_dir: str = "",
        strategy_params_dict: dict | None = None,
    ) -> None:
        """Spawn the PCE / SI-3 computation worker.

        Parameters
        ----------
        data_dir:
            Directory for PCE state persistence (SQLite).
        strategy_params_dict:
            Serialized strategy params for the worker (pickled primitives).
        """
        from src.trading.pce_worker import pce_worker_main

        wid = "pce_worker"
        hb_value = _MP_CTX.Value("d", 0.0)
        shutdown_evt = _MP_CTX.Event()
        cb_evt = _MP_CTX.Event()

        self._pce_input_queue = _MP_CTX.Queue(maxsize=5000)
        self._pce_output_queue = _MP_CTX.Queue(maxsize=500)
        self._pce_var_request_queue = _MP_CTX.Queue(maxsize=100)
        self._pce_var_response_queue = _MP_CTX.Queue(maxsize=100)

        proc = _MP_CTX.Process(
            target=pce_worker_main,
            args=(
                wid,
                self._pce_input_queue,
                self._pce_output_queue,
                self._pce_var_request_queue,
                self._pce_var_response_queue,
                hb_value,
                shutdown_evt,
                cb_evt,
                data_dir,
                strategy_params_dict or {},
            ),
            name=wid,
            daemon=True,
        )

        record = WorkerRecord(
            worker_id=wid,
            process=proc,
            heartbeat_value=hb_value,
            shutdown_event=shutdown_evt,
            circuit_breaker_event=cb_evt,
        )
        self._workers[wid] = record
        self._heartbeat_checker.register(wid, hb_value, proc)

        proc.start()
        record.started = True
        log.info("pce_worker_started", pid=proc.pid)

    # ═══════════════════════════════════════════════════════════════════════
    #  Health Monitoring
    # ═══════════════════════════════════════════════════════════════════════
    async def health_check_loop(self) -> None:
        """Async coroutine that periodically checks worker liveness.

        If any worker is dead or stale, triggers emergency stop.
        Run this as an asyncio task in the main event loop.
        """
        self._running = True
        # Give workers time to initialize
        await asyncio.sleep(5.0)
        monitor_started_at = time.monotonic()

        while self._running:
            try:
                await asyncio.sleep(self._health_interval)
                if not self._running:
                    break

                # Check for circuit breaker trips
                for wid, record in self._workers.items():
                    if record.circuit_breaker_event and record.circuit_breaker_event.is_set():
                        log.critical(
                            "worker_circuit_breaker_tripped",
                            worker_id=wid,
                        )
                        await self._trigger_emergency_stop(
                            f"Worker {wid} circuit breaker tripped"
                        )
                        return

                # Check heartbeats
                dead = self._heartbeat_checker.dead_workers()
                if dead:
                    names = [w.worker_id for w in dead]
                    log.critical("workers_dead", workers=names)
                    await self._trigger_emergency_stop(
                        f"Dead workers: {', '.join(names)}"
                    )
                    return

                stale = self._heartbeat_checker.stale_workers()
                if stale:
                    now = time.monotonic()
                    in_startup_grace = (now - monitor_started_at) < self._stale_startup_grace_s
                    names = [w.worker_id for w in stale]
                    durations = {w.worker_id: round(w.stale_duration, 1) for w in stale}
                    log.warning(
                        "workers_stale",
                        workers=names,
                        durations=durations,
                    )
                    if in_startup_grace:
                        log.info(
                            "workers_stale_ignored_startup_grace",
                            workers=names,
                            stale_durations=durations,
                            grace_s=self._stale_startup_grace_s,
                        )
                        continue
                    # Only emergency-stop if stale beyond 2× threshold (gives
                    # workers a chance to recover after GC pauses etc.)
                    critical = [w for w in stale if w.stale_duration > 6.0]
                    if critical:
                        await self._trigger_emergency_stop(
                            f"Critically stale workers: {[w.worker_id for w in critical]}"
                        )
                        return

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("health_check_error", error=str(exc))

    async def _trigger_emergency_stop(self, reason: str) -> None:
        """Initiate emergency shutdown due to worker failure."""
        if self._emergency_triggered:
            return
        self._emergency_triggered = True
        log.critical("emergency_stop_triggered", reason=reason)

        # Signal all workers to shut down
        self._signal_all_shutdown()

        # Invoke the bot's emergency stop callback
        if self._on_emergency_stop is not None:
            try:
                result = self._on_emergency_stop()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.error("emergency_stop_callback_error", error=str(exc))

    def _signal_all_shutdown(self) -> None:
        """Set shutdown events for all workers."""
        for record in self._workers.values():
            if record.shutdown_event is not None:
                record.shutdown_event.set()

    # ═══════════════════════════════════════════════════════════════════════
    #  Dynamic Market Management
    # ═══════════════════════════════════════════════════════════════════════
    def add_asset(self, asset_id: str) -> str | None:
        """Add a new asset to the appropriate L2 worker shard.

        Returns the shm_name for the new asset, or None if no workers.
        """
        if asset_id in self._shm_names:
            return self._shm_names[asset_id]

        if not self._l2_shards:
            return None

        # Allocate shared memory
        shm_map = self.allocate_books([asset_id])
        shm_name = shm_map[asset_id]

        # Find the worker with the fewest assets
        min_shard = min(self._l2_shards, key=lambda s: len(s.asset_ids))
        min_shard.asset_ids.append(asset_id)
        min_shard.shm_names[asset_id] = shm_name

        # Send control message to that worker
        ctrl_q = self._l2_control_queues.get(min_shard.worker_id)
        if ctrl_q is not None:
            try:
                ctrl_q.put_nowait(("add_asset", asset_id, shm_name))
            except _queue_mod.Full:
                log.warning("ctrl_queue_full_on_add", asset_id=asset_id)

        return shm_name

    def remove_asset(self, asset_id: str) -> None:
        """Remove an asset from its L2 worker shard."""
        for shard in self._l2_shards:
            if asset_id in shard.asset_ids:
                shard.asset_ids.remove(asset_id)
                shard.shm_names.pop(asset_id, None)
                ctrl_q = self._l2_control_queues.get(shard.worker_id)
                if ctrl_q is not None:
                    try:
                        ctrl_q.put_nowait(("remove_asset", asset_id))
                    except _queue_mod.Full:
                        log.warning("ctrl_queue_full_on_remove", asset_id=asset_id)
                break

        # Clean up shared memory and reader
        reader = self._readers.pop(asset_id, None)
        if reader:
            reader.close()
        shm = self._shm_blocks.pop(asset_id, None)
        if shm:
            cleanup_shm(shm)
        self._shm_names.pop(asset_id, None)

    # ═══════════════════════════════════════════════════════════════════════
    #  Graceful Shutdown
    # ═══════════════════════════════════════════════════════════════════════
    def stop_all(self, timeout: float = 10.0) -> None:
        """Gracefully stop all workers and clean up shared memory.

        Blocks for up to *timeout* seconds waiting for each worker to
        finish.  Forcefully terminates any stragglers.
        """
        self._running = False
        self._signal_all_shutdown()

        for wid, record in self._workers.items():
            proc = record.process
            if proc is None:
                continue
            if not proc.is_alive():
                # Reap already-dead processes to prevent <defunct> zombies.
                proc.join(timeout=1.0)
                continue
            proc.join(timeout=timeout)
            if proc.is_alive():
                log.warning("worker_force_terminate", worker_id=wid, pid=proc.pid)
                proc.terminate()
                proc.join(timeout=3.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)  # reap after kill

        # Clean up shared memory
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

        for shm in self._shm_blocks.values():
            cleanup_shm(shm)
        self._shm_blocks.clear()
        self._shm_names.clear()

        # Close queues
        for q in self._l2_control_queues.values():
            q.close()
        self._l2_control_queues.clear()

        for q_attr in (
            "_bbo_queue",
            "_pce_input_queue",
            "_pce_output_queue",
            "_pce_var_request_queue",
            "_pce_var_response_queue",
        ):
            q = getattr(self, q_attr, None)
            if q is not None:
                q.close()
            setattr(self, q_attr, None)

        self._workers.clear()
        log.info("process_manager_stopped")

    def _atexit_cleanup(self) -> None:
        """Last-resort cleanup registered via ``atexit``."""
        if self._workers:
            try:
                self.stop_all(timeout=5.0)
            except Exception:
                pass
