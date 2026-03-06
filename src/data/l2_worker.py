"""
L2 Reconstruction Worker — runs in a dedicated child process, handling
WebSocket connections and L2 order book reconstruction for a shard of
markets.

Entry point is ``l2_worker_main()``, invoked by ``ProcessManager``.
The worker runs its own asyncio event loop, creates ``L2WebSocket`` and
``L2OrderBook`` instances for its assigned assets, and publishes updates
to shared memory via ``SharedBookWriter``.

BBO change notifications are pushed to a shared ``multiprocessing.Queue``
so the main process can react (callbacks, stop-loss, signal evaluation).
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue as _queue_mod
import signal
import sys
import time
from typing import Any

# Ensure the project root is importable when spawned as a new process.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def l2_worker_main(
    worker_id: str,
    asset_ids: list[str],
    shm_names: dict[str, str],
    bbo_queue: multiprocessing.Queue,
    heartbeat_value: Any,  # multiprocessing.Value('d')
    shutdown_event: Any,  # multiprocessing.Event
    circuit_breaker_event: Any,  # multiprocessing.Event
    control_queue: Any,  # multiprocessing.Queue
) -> None:
    """Top-level entry point for an L2 reconstruction worker process.

    Runs an asyncio event loop internally.  Returns when the shutdown
    event is set or an unrecoverable error occurs.
    """
    # Install uvloop if available (Linux production)
    try:
        import uvloop  # type: ignore[import-untyped]
        uvloop.install()
    except ImportError:
        pass

    try:
        asyncio.run(
            _l2_worker_async(
                worker_id,
                asset_ids,
                shm_names,
                bbo_queue,
                heartbeat_value,
                shutdown_event,
                circuit_breaker_event,
                control_queue,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception:
        # Signal circuit breaker to main process
        circuit_breaker_event.set()


async def _l2_worker_async(
    worker_id: str,
    asset_ids: list[str],
    shm_names: dict[str, str],
    bbo_queue: multiprocessing.Queue,
    heartbeat_value: Any,
    shutdown_event: Any,
    circuit_breaker_event: Any,
    control_queue: Any,
) -> None:
    """Async implementation of the L2 worker."""
    from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
    from src.core.ipc import SharedBookWriter, LATENCY_HEALTHY, _STATE_MAP
    from src.core.logger import get_logger, setup_logging
    from src.core.config import settings
    from src.core.worker_heartbeat import WorkerHeartbeatSender
    from src.data.l2_book import L2OrderBook
    from src.data.l2_websocket import L2WebSocket

    setup_logging(settings.log_dir)
    log = get_logger(f"l2_worker.{worker_id}")
    log.info("l2_worker_starting", worker_id=worker_id, n_assets=len(asset_ids))

    heartbeat = WorkerHeartbeatSender(heartbeat_value)
    breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)

    # Capture parent PID for orphan detection in heartbeat loop
    _original_ppid = os.getppid()

    # ── Create SharedBookWriters ──────────────────────────────────────
    writers: dict[str, SharedBookWriter] = {}
    for aid in asset_ids:
        writers[aid] = SharedBookWriter(aid, shm_names[aid])

    # ── Helper: publish book state to shared memory ───────────────────
    def _publish_book(book: L2OrderBook) -> None:
        """Serialize the current book state into shared memory."""
        aid = book.asset_id
        writer = writers.get(aid)
        if writer is None:
            return

        bb = book.best_bid
        ba = book.best_ask
        bid_levels_raw = [
            (-neg_p, book._bids[neg_p])
            for neg_p in book._bids.islice(stop=50)
        ]
        ask_levels_raw = [
            (p, book._asks[p])
            for p in book._asks.islice(stop=50)
        ]

        # Pre-compute depth near mid for AdverseSelectionGuard fast-path
        depth_near_mid = book.depth_near_mid_usd(1.0, 50)

        writer.write(
            seq=book.seq,
            timestamp=book._last_update,
            server_time=book._last_server_time,
            best_bid=bb,
            best_ask=ba,
            bid_depth_usd=sum(p * s for p, s in bid_levels_raw[:5]),
            ask_depth_usd=sum(p * s for p, s in ask_levels_raw[:5]),
            spread_score=book.spread_score_value,
            depth_near_mid=depth_near_mid,
            state=_STATE_MAP.get(book.state.value, 0),
            latency_state=LATENCY_HEALTHY,
            is_reliable=book.is_reliable,
            n_bid_levels=len(bid_levels_raw),
            n_ask_levels=len(ask_levels_raw),
            delta_count=book.delta_count,
            desync_total=book.desync_total,
            bid_levels=bid_levels_raw,
            ask_levels=ask_levels_raw,
        )

    # ── BBO change callback — publishes to shm + bbo_queue ───────────
    _bbo_queue_drops = 0

    async def _on_bbo_change(asset_id: str, score: Any) -> None:
        nonlocal _bbo_queue_drops
        try:
            book = books.get(asset_id)
            if book is None:
                return
            _publish_book(book)
            # Notify main process (non-blocking)
            try:
                bbo_queue.put_nowait(("bbo", asset_id, book.seq))
            except _queue_mod.Full:
                _bbo_queue_drops += 1
                if _bbo_queue_drops % 100 == 1:
                    log.warning("queue_full_drop", queue="bbo", total_drops=_bbo_queue_drops)
        except Exception as exc:
            log.error("bbo_publish_error", asset_id=asset_id, error=str(exc))
            if breaker.record():
                circuit_breaker_event.set()

    # ── Desync callback ───────────────────────────────────────────────
    async def _on_desync(asset_id: str) -> None:
        log.warning("l2_desync_in_worker", asset_id=asset_id, worker_id=worker_id)
        # L2WebSocket handles re-fetching the snapshot internally

    # ── Create L2OrderBook instances ──────────────────────────────────
    books: dict[str, L2OrderBook] = {}
    for aid in asset_ids:
        book = L2OrderBook(
            aid,
            on_bbo_change=_on_bbo_change,
            on_desync=_on_desync,
        )
        books[aid] = book

    # ── Create L2WebSocket ────────────────────────────────────────────
    l2_ws = L2WebSocket(books)

    # ── Control queue consumer (dynamic add/remove) ───────────────────
    async def _control_loop() -> None:
        """Process control messages for dynamic asset management."""
        while not shutdown_event.is_set():
            try:
                # Non-blocking poll
                await asyncio.sleep(1.0)
                while True:
                    try:
                        msg = control_queue.get_nowait()
                    except Exception:
                        break
                    cmd = msg[0]
                    if cmd == "add_asset":
                        _, new_aid, new_shm_name = msg
                        if new_aid not in books:
                            new_book = L2OrderBook(
                                new_aid,
                                on_bbo_change=_on_bbo_change,
                                on_desync=_on_desync,
                            )
                            books[new_aid] = new_book
                            writers[new_aid] = SharedBookWriter(new_aid, new_shm_name)
                            await l2_ws.add_assets({new_aid: new_book})
                            log.info("asset_added", asset_id=new_aid, worker_id=worker_id)
                    elif cmd == "remove_asset":
                        _, rm_aid = msg
                        if rm_aid in books:
                            await l2_ws.remove_assets([rm_aid])
                            books.pop(rm_aid, None)
                            w = writers.pop(rm_aid, None)
                            if w:
                                w.close()
                            log.info("asset_removed", asset_id=rm_aid, worker_id=worker_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("control_loop_error", error=str(exc))

    # ── Heartbeat loop ────────────────────────────────────────────────
    async def _heartbeat_loop() -> None:
        while not shutdown_event.is_set():            # Detect orphaned worker (parent died)
            if os.getppid() != _original_ppid:
                log.warning("parent_died_orphan_exit", worker_id=worker_id)
                shutdown_event.set()
                return            heartbeat.beat()
            await asyncio.sleep(0.5)

    # ── Shutdown watcher ──────────────────────────────────────────────
    async def _watch_shutdown() -> None:
        """Poll the shutdown event and cancel tasks when set."""
        while not shutdown_event.is_set():
            await asyncio.sleep(0.5)
        # Give a brief window for cleanup
        await asyncio.sleep(0.5)

    # ── Run all tasks ─────────────────────────────────────────────────
    log.info("l2_worker_ready", worker_id=worker_id, n_assets=len(asset_ids))
    heartbeat.beat()

    tasks = [
        asyncio.create_task(l2_ws.start(), name=f"{worker_id}_ws"),
        asyncio.create_task(_heartbeat_loop(), name=f"{worker_id}_heartbeat"),
        asyncio.create_task(_control_loop(), name=f"{worker_id}_control"),
        asyncio.create_task(_watch_shutdown(), name=f"{worker_id}_shutdown_watch"),
    ]

    try:
        # Wait for shutdown watcher to complete (signals shutdown)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # If the shutdown watcher finished, cancel everything else
        for t in pending:
            t.cancel()
        # Wait for cancellations to propagate
        await asyncio.gather(*pending, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        # Cleanup
        await l2_ws.stop()
        for w in writers.values():
            w.close()
        log.info("l2_worker_stopped", worker_id=worker_id)
