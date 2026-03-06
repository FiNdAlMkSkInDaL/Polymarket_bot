"""
PCE / SI-3 Computation Worker — runs O(N²) matrix math in a dedicated
child process, keeping the main asyncio event loop free.

Entry point is ``pce_worker_main()``, invoked by ``ProcessManager``.

Communication protocol
──────────────────────
**Input queue** (main → worker):
  ``("bar_return", market_id, log_return, yes_asset_id, no_asset_id)``
  ``("register_market", market_id, event_id, tags)``
  ``("set_aggregator_bars", market_id, closes)``  — initial bar sync

**Output queue** (worker → main):
  ``("pce_refreshed", dashboard_data_dict)``
  ``("cm_signals", [signal_dict, ...])``
  ``("prior_validation", summary_dict)``

**VaR gate** (synchronous request-response):
  Request: ``("check_var", request_id, positions_list, market_id, size_usd, direction)``
  Response: ``("var_result", request_id, allowed, var_result_dict)``
  Response: ``("haircut_result", request_id, haircut_factor, avg_correlation)``
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue as _queue_mod
import sys
import time
from typing import Any

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def pce_worker_main(
    worker_id: str,
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    var_request_queue: multiprocessing.Queue,
    var_response_queue: multiprocessing.Queue,
    heartbeat_value: Any,
    shutdown_event: Any,
    circuit_breaker_event: Any,
    data_dir: str,
    strategy_params_dict: dict,
) -> None:
    """Top-level entry point for the PCE computation worker."""
    try:
        import uvloop  # type: ignore[import-untyped]
        uvloop.install()
    except ImportError:
        pass

    try:
        asyncio.run(
            _pce_worker_async(
                worker_id,
                input_queue,
                output_queue,
                var_request_queue,
                var_response_queue,
                heartbeat_value,
                shutdown_event,
                circuit_breaker_event,
                data_dir,
                strategy_params_dict,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception:
        circuit_breaker_event.set()


async def _pce_worker_async(
    worker_id: str,
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    var_request_queue: multiprocessing.Queue,
    var_response_queue: multiprocessing.Queue,
    heartbeat_value: Any,
    shutdown_event: Any,
    circuit_breaker_event: Any,
    data_dir: str,
    strategy_params_dict: dict,
) -> None:
    """Async implementation of the PCE worker."""
    from src.core.config import settings
    from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
    from src.core.logger import get_logger, setup_logging
    from src.core.worker_heartbeat import WorkerHeartbeatSender
    from src.trading.portfolio_correlation import PortfolioCorrelationEngine
    from src.signals.cross_market import CrossMarketSignalGenerator

    setup_logging(settings.log_dir)
    log = get_logger(f"pce_worker.{worker_id}")
    log.info("pce_worker_starting", worker_id=worker_id)

    heartbeat = WorkerHeartbeatSender(heartbeat_value)
    breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)

    # Capture parent PID for orphan detection in heartbeat loop
    _original_ppid = os.getppid()

    # ── Initialize PCE ────────────────────────────────────────────────
    pce = PortfolioCorrelationEngine(data_dir=data_dir or settings.record_data_dir)
    pce.load_state()

    # ── Initialize SI-3 ──────────────────────────────────────────────
    cross_market: CrossMarketSignalGenerator | None = None
    if settings.strategy.cross_mkt_enabled:
        cross_market = CrossMarketSignalGenerator(pce)

    # ── Refresh interval ──────────────────────────────────────────────
    refresh_interval = settings.strategy.pce_correlation_refresh_minutes * 60
    last_refresh = time.monotonic()
    cycle_count = 0

    # ── Input queue consumer ──────────────────────────────────────────
    _output_queue_drops = 0
    _var_response_drops = 0

    async def _consume_inputs() -> None:
        """Process bar returns and registration commands."""
        nonlocal last_refresh, cycle_count

        while not shutdown_event.is_set():
            # Process all available messages (non-blocking drain)
            processed = 0
            while processed < 200:  # batch limit per cycle
                try:
                    msg = input_queue.get_nowait()
                except Exception:
                    break

                processed += 1
                try:
                    cmd = msg[0]
                    if cmd == "bar_return":
                        _, market_id, log_return, yes_aid, no_aid = msg
                        if cross_market is not None:
                            cross_market.record_return(
                                market_id, log_return,
                                yes_asset_id=yes_aid,
                                no_asset_id=no_aid,
                            )
                    elif cmd == "register_market":
                        _, market_id, event_id, tags = msg
                        pce.register_market_by_ids(market_id, event_id, tags)
                except Exception as exc:
                    log.error("input_process_error", error=str(exc))
                    if breaker.record():
                        circuit_breaker_event.set()
                        return

            # After draining input, do SI-3 scan if we processed any bars
            if processed > 0 and cross_market is not None:
                try:
                    cm_signals = cross_market.scan()
                    if cm_signals:
                        signal_dicts = [
                            {
                                "lagging_market_id": s.lagging_market_id,
                                "leading_market_id": s.leading_market_id,
                                "lagging_asset_id": getattr(s, "lagging_asset_id", ""),
                                "leading_asset_id": getattr(s, "leading_asset_id", ""),
                                "direction": s.direction,
                                "z_score": s.z_score,
                                "correlation": s.correlation,
                                "confidence": getattr(s, "confidence", 0.0),
                            }
                            for s in cm_signals
                        ]
                        try:
                            output_queue.put_nowait(("cm_signals", signal_dicts))
                        except _queue_mod.Full:
                            nonlocal _output_queue_drops
                            _output_queue_drops += 1
                            if _output_queue_drops % 100 == 1:
                                log.warning("queue_full_drop", queue="pce_output", total_drops=_output_queue_drops)
                except Exception as exc:
                    log.error("cm_scan_error", error=str(exc))

            # Periodic correlation refresh
            now = time.monotonic()
            if now - last_refresh >= refresh_interval:
                last_refresh = now
                cycle_count += 1
                try:
                    pce.refresh_correlations()
                    pce.save_state()

                    # Every 12th cycle (~6h) validate structural priors
                    if cycle_count % 12 == 0:
                        summary = pce.validate_structural_priors()
                        try:
                            output_queue.put_nowait(("prior_validation", summary))
                        except _queue_mod.Full:
                            _output_queue_drops += 1
                            if _output_queue_drops % 100 == 1:
                                log.warning("queue_full_drop", queue="pce_output", total_drops=_output_queue_drops)

                    # Send dashboard
                    dashboard = pce.get_dashboard_data()
                    try:
                        output_queue.put_nowait(("pce_refreshed", dashboard))
                    except _queue_mod.Full:
                        _output_queue_drops += 1
                        if _output_queue_drops % 100 == 1:
                            log.warning("queue_full_drop", queue="pce_output", total_drops=_output_queue_drops)

                    log.info(
                        "pce_refresh_complete",
                        cycle=cycle_count,
                        pairs=dashboard.get("total_pairs_tracked", 0),
                    )
                except Exception as exc:
                    log.error("pce_refresh_error", error=str(exc))

            await asyncio.sleep(0.1)

    # ── VaR gate responder ────────────────────────────────────────────
    async def _var_gate_loop() -> None:
        """Handle synchronous VaR gate requests from the main process."""
        while not shutdown_event.is_set():
            try:
                msg = var_request_queue.get_nowait()
            except Exception:
                await asyncio.sleep(0.01)  # 10ms poll
                continue

            try:
                cmd = msg[0]
                if cmd == "check_var":
                    _, req_id, positions_data, market_id, size_usd, direction = msg
                    # Reconstruct minimal position-like objects
                    pos_list = [_MinimalPosition(**p) for p in positions_data]
                    allowed, var_result = pce.check_var_gate(
                        pos_list, market_id, size_usd, direction,
                    )
                    result_dict = {
                        "portfolio_var": getattr(var_result, "portfolio_var", 0.0),
                        "var_limit": getattr(var_result, "var_limit", 0.0),
                        "position_count": getattr(var_result, "position_count", 0),
                    }
                    try:
                        var_response_queue.put_nowait(("var_result", req_id, allowed, result_dict))
                    except _queue_mod.Full:
                        nonlocal _var_response_drops
                        _var_response_drops += 1
                        if _var_response_drops % 100 == 1:
                            log.warning("queue_full_drop", queue="var_response", total_drops=_var_response_drops)

                elif cmd == "check_haircut":
                    _, req_id, market_id, exposure_usd = msg
                    haircut = pce.compute_concentration_haircut(
                        market_id, exposure_usd,
                    )
                    try:
                        var_response_queue.put_nowait(("haircut_result", req_id, haircut, 0.0))
                    except _queue_mod.Full:
                        _var_response_drops += 1
                        if _var_response_drops % 100 == 1:
                            log.warning("queue_full_drop", queue="var_response", total_drops=_var_response_drops)

            except Exception as exc:
                log.error("var_gate_error", error=str(exc), exc_info=True)
                if breaker.record():
                    circuit_breaker_event.set()
                    return

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
        while not shutdown_event.is_set():
            await asyncio.sleep(0.5)
        await asyncio.sleep(0.5)

    # ── Run ───────────────────────────────────────────────────────────
    log.info("pce_worker_ready", worker_id=worker_id)
    heartbeat.beat()

    tasks = [
        asyncio.create_task(_consume_inputs(), name="pce_inputs"),
        asyncio.create_task(_var_gate_loop(), name="pce_var_gate"),
        asyncio.create_task(_heartbeat_loop(), name="pce_heartbeat"),
        asyncio.create_task(_watch_shutdown(), name="pce_shutdown_watch"),
    ]

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            pce.save_state()
        except Exception:
            pass
        log.info("pce_worker_stopped", worker_id=worker_id)


class _MinimalPosition:
    """Lightweight stand-in for Position objects across process boundary.

    Only includes the fields accessed by ``check_var_gate()`` and
    ``compute_concentration_haircut()``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.market_id = kwargs.get("market_id", "")
        self.entry_price = kwargs.get("entry_price", 0.0)
        self.size = kwargs.get("size", 0.0)
        self.filled_size = kwargs.get("filled_size", 0.0)
        self.trade_asset_id = kwargs.get("trade_asset_id", "")
        self.no_asset_id = kwargs.get("no_asset_id", "")
        self.event_id = kwargs.get("event_id", "")
