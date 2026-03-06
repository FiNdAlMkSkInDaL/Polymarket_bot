"""
Main bot orchestrator — wires the data pipeline, signal detection,
order execution, and monitoring into a single async event loop.

V2 — Institutional-grade lifecycle:
  - Three-tier market lifecycle (observing → active → draining)
  - Live orderbook tracking (spread, depth, mid-price)
  - Composite market scoring with dynamic re-evaluation
  - Per-market/event position limits, daily loss circuit breaker,
    trailing stop-loss, signal cooldown
  - Automatic resolution detection and market eviction
  - Memory cleanup for stale positions, orders, aggregators
"""

from __future__ import annotations

import asyncio
import queue as _queue_mod
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings, DeploymentEnv
from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
from src.core.guard import DeploymentGuard
from src.core.heartbeat import BookHeartbeat, PolygonHeadLagChecker
from src.core.latency_guard import LatencyGuard, LatencyState
from src.core.logger import get_logger, setup_logging
from src.core.process_manager import ProcessManager
from src.data.market_discovery import MarketInfo, fetch_active_markets
from src.data.market_lifecycle import MarketLifecycleManager
from src.data.market_scorer import compute_score
from src.data.ohlcv import OHLCVAggregator
from src.data.orderbook import L2OrderBookAdapter, OrderbookTracker, SharedBookReaderAdapter
from src.data.websocket_client import MarketWebSocket, TradeEvent
from src.data.l2_book import BookState, L2OrderBook
from src.data.l2_websocket import L2WebSocket
from src.monitoring.telegram import TelegramAlerter
from src.monitoring.trade_store import TradeStore
from src.signals.adverse_selection_guard import AdverseSelectionGuard
from src.signals.edge_filter import compute_edge_score
from src.signals.panic_detector import PanicDetector, PanicSignal
from src.signals.resolution_probability import (
    CryptoPriceModel,
    GenericBayesianModel,
    RPECalibrationTracker,
    ResolutionProbabilityEngine,
)
from src.signals.signal_framework import (
    BaseSignal,
    CompositeSignalEvaluator,
    OrderbookImbalanceSignal,
    SignalResult,
    SpreadCompressionSignal,
)
from src.signals.whale_monitor import WhaleMonitor
from src.trading.chaser import ChaserState, OrderChaser
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus, OrderStatusPoller
from src.trading.position_manager import PositionManager, PositionState
from src.trading.portfolio_correlation import PortfolioCorrelationEngine
from src.trading.stop_loss import StopLossMonitor
from src.trading.stealth_executor import StealthExecutor
from src.trading.take_profit import compute_dynamic_spread, compute_take_profit
from src.signals.regime_detector import RegimeDetector
from src.signals.iceberg_detector import IcebergDetector
from src.signals.cross_market import CrossMarketSignalGenerator
from src.signals.drift_signal import DriftSignal, MeanReversionDrift
from src.trading.adverse_selection_monitor import AdverseSelectionMonitor, make_fill_record

# Data recording (lazy import — only used when RECORD_DATA=true)
try:
    from src.backtest.data_recorder import MarketDataRecorder
except ImportError:
    MarketDataRecorder = None  # type: ignore[misc,assignment]

log = get_logger(__name__)


def _safe_task_done_callback(task: asyncio.Task) -> None:
    """Callback attached to fire-and-forget tasks to log unhandled exceptions.

    Without this, exceptions in tasks created via ``asyncio.ensure_future``
    or ``asyncio.create_task`` are silently dropped until the task is GC'd,
    producing the infamous *"Task exception was never retrieved"* message.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error(
            "unhandled_task_exception",
            task_name=task.get_name(),
            error=repr(exc),
            exc_info=exc,
        )


def _safe_fire_and_forget(coro, *, name: str | None = None) -> asyncio.Task:
    """Schedule a coroutine as a fire-and-forget task with error logging.

    Replaces bare ``asyncio.ensure_future`` calls throughout the bot to
    ensure that exceptions are always captured and logged.
    """
    task = asyncio.ensure_future(coro)
    if name:
        task.set_name(name)
    task.add_done_callback(_safe_task_done_callback)
    return task


class TradingBot:
    """Top-level orchestrator for the mean-reversion market maker."""

    def __init__(
        self,
        paper_mode: bool | None = None,
        *,
        deployment_env: DeploymentEnv | None = None,
        confirmed_production: bool = False,
    ):
        # Deployment env is the canonical source of truth.
        # Legacy paper_mode kwarg is supported for backward compatibility.
        if deployment_env is not None:
            self.deployment_env = deployment_env
        elif paper_mode is not None:
            self.deployment_env = (
                DeploymentEnv.PAPER if paper_mode else DeploymentEnv.PRODUCTION
            )
        else:
            self.deployment_env = settings.deployment_env

        self.guard = DeploymentGuard(
            self.deployment_env,
            confirmed_production=confirmed_production,
        )
        self.paper_mode = self.guard.is_paper

        # Components (initialised in start())
        self.executor = OrderExecutor(paper_mode=self.paper_mode)
        self.trade_store = TradeStore()

        # Portfolio Correlation Engine (Pillar 15)
        self.pce = PortfolioCorrelationEngine(
            data_dir=settings.record_data_dir,
        )
        self.pce.load_state()

        # Per-market state — initialise book_trackers dict early so the
        # PositionManager (which receives a reference to it for paper-mode
        # fill simulation) sees subsequent additions.
        self._book_trackers: dict[str, OrderbookTracker] = {}  # asset_id → tracker

        # SI-2: Per-asset iceberg detectors (initialised before PositionManager
        # so the dict reference is shared — new detectors added later are visible).
        self._iceberg_detectors: dict[str, IcebergDetector] = {}  # asset_id → detector

        self.positions = PositionManager(
            self.executor, trade_store=self.trade_store, guard=self.guard,
            pce=self.pce,
            book_trackers=self._book_trackers,
            iceberg_detectors=self._iceberg_detectors,
        )
        self.whale_monitor = WhaleMonitor(zscore_fn=self._latest_zscore)
        self.telegram = TelegramAlerter()
        self.lifecycle = MarketLifecycleManager()
        self.latency_guard = LatencyGuard()

        # Shared fast-kill event for adverse-selection guard
        self._fast_kill_event = asyncio.Event()
        self._fast_kill_event.set()  # start SET — chasers may proceed (clear = paused)

        # Per-market state
        self._markets: list[MarketInfo] = []
        self._yes_aggs: dict[str, OHLCVAggregator] = {}  # keyed by yes_token_id
        self._no_aggs: dict[str, OHLCVAggregator] = {}   # keyed by no_token_id
        self._detectors: dict[str, PanicDetector] = {}    # keyed by condition_id
        self._market_map: dict[str, MarketInfo] = {}      # asset_id → MarketInfo
        # _book_trackers initialised above (before PositionManager)
        self._l2_books: dict[str, L2OrderBook] = {}  # asset_id → L2 book (when L2 enabled)
        self._trade_counts: dict[str, float] = {}  # asset_id → trades/min
        self._taker_counts: dict[str, int] = {}    # asset_id → taker-initiated trades
        self._total_counts: dict[str, int] = {}    # asset_id → total classified trades
        self._recent_trade_volume: dict[str, list[tuple[float, float]]] = {}  # cond_id → [(ts, size)]

        # Spread-based signal evaluators (Problem 3)
        self._spread_evaluators: dict[str, CompositeSignalEvaluator] = {}  # condition_id → evaluator
        self._spread_cooldowns: dict[str, float] = {}  # condition_id → last fire timestamp

        # SI-1: Per-market regime detectors
        self._regime_detectors: dict[str, RegimeDetector] = {}  # condition_id → detector

        # V3: Per-market drift signal detectors
        self._drift_detectors: dict[str, MeanReversionDrift] = {}  # condition_id → detector
        self._drift_cooldowns: dict[str, float] = {}  # condition_id → last fire timestamp

        # Maker adverse-selection monitor (V1/V4 calibration)
        self._maker_monitor: AdverseSelectionMonitor | None = None

        # SI-2: _iceberg_detectors initialised above (before PositionManager)

        # SI-3: Cross-market signal generator
        self._cross_market = CrossMarketSignalGenerator(self.pce) if settings.strategy.cross_mkt_enabled else None

        # SI-4: Stealth executor wrapper
        self._stealth = StealthExecutor(self.executor) if settings.strategy.stealth_enabled else None

        # Lifecycle
        self._running = False
        self._start_time: float = 0.0       # set in _run() for uptime tracking
        self._latest_z: float = 0.0        # tracks most recent panic Z-score
        self._trade_queue: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=1000)
        self._book_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._ws: MarketWebSocket | None = None
        self._l2_ws: L2WebSocket | None = None
        self._tasks: list[asyncio.Task] = []
        self._stop_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Multi-core ProcessManager — orchestrates L2 and PCE workers
        self._process_manager: ProcessManager | None = None
        self._multicore_enabled: bool = settings.strategy.l2_enabled  # gate on L2
        self._pce_queue_drops: int = 0  # rate-limited counter for PCE input queue drops

        # Singleton RPE (Pillar 14) — shared across all markets.
        # The CryptoPriceModel and GenericBayesianModel are stateless
        # per call; the per-market estimate cache inside the RPE already
        # keys by market_id, so one instance suffices.
        self._rpe = ResolutionProbabilityEngine(
            models=[
                CryptoPriceModel(
                    price_fn=lambda: self._get_crypto_spot(),
                ),
                GenericBayesianModel(),
            ],
        )
        # Crypto retrigger state (Fix 2): last evaluated spot price
        self._rpe_last_spot: float | None = None

        # Cached fee-category set (avoid re-parsing on every trade eval)
        _fee_cats_raw = settings.strategy.fee_enabled_categories.lower().split(",")
        self._fee_category_set: frozenset[str] = frozenset(
            cat.strip() for cat in _fee_cats_raw if cat.strip()
        )

        # RPE calibration tracker (Deliverable A)
        self._rpe_calibration = RPECalibrationTracker()

        # Exception circuit breakers for critical async loops
        self._trade_loop_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)

        # RPE per-market cooldown (Deliverable C)
        self._rpe_last_signal: dict[str, float] = {}

        # Data recorder (enabled via RECORD_DATA=true, or forced on in PAPER)
        self._recorder = None
        if self.guard.enforce_data_recording() and MarketDataRecorder is not None:
            self._recorder = MarketDataRecorder(data_dir=settings.record_data_dir)

    # ═══════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════════════
    async def start(self) -> None:
        """Discover markets, wire components, and enter the main loop."""
        setup_logging(settings.log_dir)
        mode_label = self.deployment_env.value
        strat = settings.strategy
        log.info(
            "bot_starting",
            mode=mode_label,
            **self.guard.startup_summary(),
            zscore_threshold=strat.zscore_threshold,
            volume_ratio_threshold=strat.volume_ratio_threshold,
            min_edge_score=strat.min_edge_score,
            signal_cooldown_min=strat.signal_cooldown_minutes,
            heartbeat_stale_ms=strat.heartbeat_stale_ms,
            heartbeat_stale_count=strat.heartbeat_stale_count,
            ws_silence_timeout=strat.ws_silence_timeout_s,
            min_kelly_trades=strat.min_kelly_trades,
            spread_signal_enabled=strat.spread_signal_enabled,
            rpe_shadow_mode=strat.rpe_shadow_mode,
            rpe_generic_enabled=strat.rpe_generic_enabled,
            kelly_fraction=strat.kelly_fraction,
        )

        # Fail-fast if running with real capital without required credentials
        if self.guard.is_live:
            missing = settings.validate_credentials()
            if missing:
                for msg in missing:
                    log.error("credential_missing", detail=msg)
                raise SystemExit(
                    f"Cannot start in {mode_label} mode — missing credentials: "
                    f"{", ".join(missing)}"
                )

        await self.telegram.send(f"🤖 Bot starting in <b>{mode_label}</b> mode")

        try:
            await self._run(mode_label)
        except Exception as exc:
            log.exception("bot_crashed", error=str(exc))
            await self.telegram.send(
                f"💥 <b>Bot crashed unexpectedly</b>\n"
                f"<code>{type(exc).__name__}: {exc}</code>\n"
                "Please restart the bot."
            )
            raise

    async def _run(self, mode_label: str) -> None:
        """Internal: initialise and run after startup message is sent."""
        self._start_time = time.monotonic()

        # 1. Init trade store
        await self.trade_store.init()

        # 2. Discover eligible markets via lifecycle manager
        self._markets = await self.lifecycle.initial_discovery()
        if not self._markets:
            log.error("no_eligible_markets")
            await self.telegram.send(
                "⚠️ <b>Bot stopped — no eligible markets found.</b>\n"
                "No markets currently meet the configured criteria.\n"
                "Restart the bot manually when conditions change."
            )
            return

        log.info("markets_selected", count=len(self._markets))

        # 3. Build per-market aggregators, detectors & book trackers
        all_asset_ids: list[str] = []
        whale_map: dict[str, tuple[str, str]] = {}

        for m in self._markets:
            self._wire_market(m)
            all_asset_ids.extend([m.yes_token_id, m.no_token_id])
            whale_map[m.yes_token_id] = (m.condition_id, "yes")
            whale_map[m.no_token_id] = (m.condition_id, "no")

        # Also wire observing-tier markets (collect data, no trading)
        for om in self.lifecycle.observing.values():
            m = om.info
            self._wire_market(m)
            all_asset_ids.extend([m.yes_token_id, m.no_token_id])
            whale_map[m.yes_token_id] = (m.condition_id, "yes")
            whale_map[m.no_token_id] = (m.condition_id, "no")

        # Register whale token→market mapping
        self.whale_monitor.set_market_map(whale_map)

        # 4. Set initial wallet balance
        self.positions.set_wallet_balance(self.guard.get_wallet_balance())

        # 4b. Validate fee model against CLOB REST endpoint (live only).
        #     Sample up to 3 active markets to cross-check the local
        #     parabolic formula against the exchange's actual fee curve.
        if self.guard.is_live and self._markets:
            from src.trading.fee_cache import validate_fee_model
            probe_markets = self._markets[:3]
            probe_tokens = [m.no_token_id for m in probe_markets]
            # Use 0.50 mid-price as the worst-case probe (peak fee)
            probe_prices = [0.50] * len(probe_tokens)
            fee_ok = await validate_fee_model(probe_tokens, probe_prices)
            if not fee_ok:
                await self.telegram.send(
                    "⚠️ <b>Fee model divergence detected</b>\n"
                    "Local fee formula disagrees with CLOB endpoint.\n"
                    "Review logs — bot continues with local model."
                )

        # 5. Launch concurrent tasks
        self._running = True
        self._ws = MarketWebSocket(
            all_asset_ids, self._trade_queue, book_queue=self._book_queue,
            recorder=self._recorder,
        )

        # L2 WebSocket (Pillar 11) — dedicated connection for L2 book data
        # In multi-core mode, L2 processing is offloaded to worker processes
        # managed by the ProcessManager.  In single-process fallback, the
        # original L2WebSocket runs inside the main asyncio loop.
        if self._multicore_enabled and self._l2_books:
            self._process_manager = ProcessManager(
                on_emergency_stop=self._schedule_stop,
            )
            l2_asset_ids = list(self._l2_books.keys())
            self._process_manager.start_l2_workers(l2_asset_ids)
            # Replace in-process L2 book trackers with shared-memory readers
            for aid in l2_asset_ids:
                reader = self._process_manager.get_reader(aid)
                if reader is not None:
                    self._book_trackers[aid] = SharedBookReaderAdapter(reader)
            # Start PCE worker
            self._process_manager.start_pce_worker(
                data_dir=settings.record_data_dir,
            )
            # Wire VaR gate queues to position manager for remote PCE checks
            self.positions.set_var_gate_queues(
                self._process_manager.pce_var_request_queue,
                self._process_manager.pce_var_response_queue,
            )
            self._l2_ws = None
            log.info(
                "multicore_enabled",
                l2_workers=self._process_manager.n_l2_workers,
                l2_assets=len(l2_asset_ids),
            )
        elif settings.strategy.l2_enabled and self._l2_books:
            # Fallback: single-process L2 (original path)
            for l2_book in self._l2_books.values():
                l2_book._on_desync = self._on_l2_desync
            self._l2_ws = L2WebSocket(self._l2_books, recorder=self._recorder)
        else:
            self._l2_ws = None

        # Adverse-selection guard (Pillar 5)
        self._adverse_guard = AdverseSelectionGuard(
            executor=self.executor,
            book_trackers=self._book_trackers,
            fast_kill_event=self._fast_kill_event,
            taker_counts=self._taker_counts,
            total_counts=self._total_counts,
            trade_counts=self._trade_counts,
            get_position_assets=self._positioned_asset_ids,
            telegram=self.telegram,
            on_shutdown=self.stop,
        )

        # Polygon head-lag checker (moved from adverse-selection guard)
        polygon_checker = (
            PolygonHeadLagChecker(
                rpc_url=settings.polygon_rpc_url,
                lag_threshold_ms=settings.strategy.adverse_sel_polygon_head_lag_ms,
            )
            if settings.polygon_rpc_url
            else None
        )

        # Book heartbeat (Pillar 8)
        self._heartbeat = BookHeartbeat(
            book_trackers=self._book_trackers,
            latency_guard=self.latency_guard,
            fast_kill_event=self._fast_kill_event,
            executor=self.executor,
            ws_transport=self._l2_ws,
            get_position_assets=self._positioned_asset_ids,
            telegram=self.telegram,
            polygon_checker=polygon_checker,
            process_manager=self._process_manager,
        )

        # Order status poller (Pillar 10) — live mode only
        self._order_poller = OrderStatusPoller(
            self.executor,
            on_fill=self._on_clob_fill,
        )

        # Event-driven stop-loss monitor (Pillar 11) — no polling task
        # V4: on_probe_breakeven wires back to PositionManager.scale_probe_to_full
        def _probe_breakeven_cb(pos):
            return self.positions.scale_probe_to_full(
                pos,
                get_mid_price_fn=lambda asset_id: (
                    self._stop_loss_monitor._get_mid_price(asset_id)
                    if hasattr(self, "_stop_loss_monitor")
                    else 0.0
                ),
            )

        self._stop_loss_monitor = StopLossMonitor(
            position_manager=self.positions,
            no_aggs=self._no_aggs,
            book_trackers=self._book_trackers,
            trade_store=self.trade_store,
            telegram=self.telegram,
            on_probe_breakeven=_probe_breakeven_cb,
        )
        self._stop_loss_monitor.start()  # mark active, no coroutine

        # Maker adverse-selection monitor — tracks T+5/15/60 PnL for POST_ONLY fills
        async def _mid_price_for_monitor(asset_id: str) -> float | None:
            mid = self._stop_loss_monitor._get_mid_price(asset_id)
            return mid if mid > 0 else None

        def _vol_provider_for_monitor(market_id: str) -> float | None:
            """Return current EWMA vol for *market_id* from the NO aggregator."""
            agg = self._no_aggs.get(market_id)
            if agg is None:
                return None
            return agg.rolling_volatility_ewma

        self._vol_provider_for_monitor = _vol_provider_for_monitor

        self._maker_monitor = AdverseSelectionMonitor(
            mid_price_fn=_mid_price_for_monitor,
            alpha=settings.strategy.adverse_monitor_alpha_base,
            vol_provider=self._vol_provider_for_monitor,
            vol_ref=settings.strategy.adverse_monitor_vol_ref,
            alpha_gamma=settings.strategy.adverse_monitor_alpha_gamma,
            alpha_min=settings.strategy.adverse_monitor_alpha_min,
            alpha_max=settings.strategy.adverse_monitor_alpha_max,
        )
        # Wire monitor into PositionManager for exec-mode downgrade on suspension
        self.positions._maker_monitor = self._maker_monitor
        # Wire stealth executor into PositionManager for sliced probe scale-ups
        if self._stealth is not None:
            self.positions._stealth = self._stealth

        self._tasks = [
            asyncio.create_task(self._ws.start(), name="ws"),
            asyncio.create_task(self._process_trades(), name="trade_processor"),
            asyncio.create_task(self._process_book_updates(), name="book_processor"),
            asyncio.create_task(self._timeout_loop(), name="timeout_loop"),
            asyncio.create_task(self._stats_loop(), name="stats_loop"),
            asyncio.create_task(self.whale_monitor.start(), name="whale_monitor"),
            asyncio.create_task(self._market_refresh_loop(), name="market_refresh"),
            asyncio.create_task(self._cleanup_loop(), name="cleanup"),
            asyncio.create_task(self._tp_rescale_loop(), name="tp_rescale"),
            asyncio.create_task(self._adverse_guard.start(), name="adverse_sel"),
            asyncio.create_task(self._heartbeat.run(), name="heartbeat"),
            asyncio.create_task(self._ghost_liquidity_loop(), name="ghost_liquidity"),
            asyncio.create_task(self._order_poller.run(), name="order_status_poller"),
            asyncio.create_task(self._health_reporter(), name="health_reporter"),
            asyncio.create_task(self._rpe_crypto_retrigger_loop(), name="rpe_retrigger"),
            asyncio.create_task(self._stale_bar_flush_loop(), name="stale_bar_flush"),
            asyncio.create_task(self._paper_summary_loop(), name="paper_summary"),
        ]

        # Multi-core mode: add worker consumer tasks, skip in-process PCE loop
        if self._process_manager is not None:
            self._tasks.append(
                asyncio.create_task(self._consume_bbo_events(), name="bbo_consumer")
            )
            self._tasks.append(
                asyncio.create_task(self._consume_pce_results(), name="pce_consumer")
            )
            self._tasks.append(
                asyncio.create_task(
                    self._process_manager.health_check_loop(), name="worker_health"
                )
            )
        else:
            # Single-process fallback: PCE refresh runs in-process
            self._tasks.append(
                asyncio.create_task(self._pce_refresh_loop(), name="pce_refresh")
            )

        # Launch data recorder if enabled
        if self._recorder is not None:
            self._tasks.append(
                asyncio.create_task(self._recorder.run(), name="data_recorder")
            )
            log.info("data_recorder_enabled", data_dir=settings.record_data_dir)

        # Launch L2 WebSocket task if enabled
        if self._l2_ws is not None:
            self._tasks.append(
                asyncio.create_task(self._l2_ws.start(), name="l2_ws")
            )

        # ── Global asyncio exception handler (last line of defence) ─────
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._asyncio_exception_handler)

        # Handle graceful shutdown via SIGINT / SIGTERM
        self._loop = loop
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._schedule_stop)
        else:
            # On Windows, loop.add_signal_handler is not supported.
            # Register via the signal module instead (handles Ctrl-C).
            signal.signal(
                signal.SIGINT,
                lambda *_: loop.call_soon_threadsafe(self._schedule_stop),
            )

        log.info("bot_running", markets=len(self._markets), mode=mode_label)

        # Wait for all tasks (they run indefinitely until stop())
        try:
            await asyncio.gather(*self._tasks, return_exceptions=False)
        except asyncio.CancelledError:
            pass

        # Ensure stop() runs to completion before start() returns
        # (otherwise asyncio.run() tears down the loop while stop()
        #  is still doing cleanup — the real cause of the hang).
        if self._stop_task is not None and not self._stop_task.done():
            await self._stop_task

    # ── Global asyncio exception handler ────────────────────────────────────
    def _asyncio_exception_handler(self, loop: asyncio.AbstractEventLoop, context: dict) -> None:
        """Catch-all for unhandled exceptions in asyncio tasks/callbacks.

        Prevents the silent *"Task exception was never retrieved"* problem
        by logging with full context instead of relying on GC finalisation.
        """
        exc = context.get("exception")
        msg = context.get("message", "Unhandled exception in event loop")
        task = context.get("future")
        task_name = task.get_name() if task and hasattr(task, "get_name") else str(task)
        log.error(
            "asyncio_unhandled_exception",
            message=msg,
            task=task_name,
            error=repr(exc) if exc else "unknown",
            exc_info=exc,
        )

    # ── Market wiring / unwiring ───────────────────────────────────────────
    def _wire_market(self, m: MarketInfo) -> None:
        """Create aggregators, detectors, and book trackers for a market."""
        if m.yes_token_id in self._market_map:
            return  # already wired

        self._market_map[m.yes_token_id] = m
        self._market_map[m.no_token_id] = m

        yes_agg = OHLCVAggregator(m.yes_token_id)
        no_agg = OHLCVAggregator(m.no_token_id)
        self._yes_aggs[m.yes_token_id] = yes_agg
        self._no_aggs[m.no_token_id] = no_agg

        detector = PanicDetector(
            market_id=m.condition_id,
            yes_asset_id=m.yes_token_id,
            no_asset_id=m.no_token_id,
            yes_aggregator=yes_agg,
            no_aggregator=no_agg,
        )
        self._detectors[m.condition_id] = detector

        # Spread-based composite signal evaluator (Problem 3)
        if settings.strategy.spread_signal_enabled:
            imbalance_sig = OrderbookImbalanceSignal(m.condition_id)
            spread_comp_sig = SpreadCompressionSignal(m.condition_id)
            evaluator = CompositeSignalEvaluator(
                generators=[(imbalance_sig, 0.6), (spread_comp_sig, 0.4)],
                min_composite_score=settings.strategy.min_composite_signal_score,
            )
            self._spread_evaluators[m.condition_id] = evaluator

        # RPE: Resolution Probability Engine (Pillar 14)
        # Wiring is handled by the singleton self._rpe;
        # no per-market instance needed.

        # SI-1: Per-market regime detector
        if settings.strategy.regime_enabled:
            self._regime_detectors[m.condition_id] = RegimeDetector(m.condition_id)

        # V3: Per-market drift signal detector
        if settings.strategy.drift_signal_enabled:
            self._drift_detectors[m.condition_id] = MeanReversionDrift(m.condition_id)

        # PCE: register market for correlation tracking (Pillar 15)
        self.pce.register_market(
            m.condition_id, m.event_id, getattr(m, 'tags', '') or '',
            yes_agg,
        )

        # Orderbook trackers for both tokens
        if settings.strategy.l2_enabled:
            for token_id in (m.yes_token_id, m.no_token_id):
                # SI-2: Create iceberg detector and wire as level-change callback
                ice_cb = None
                if settings.strategy.iceberg_enabled:
                    ice_det = IcebergDetector(token_id)
                    self._iceberg_detectors[token_id] = ice_det
                    ice_cb = ice_det.on_level_change

                l2_book = L2OrderBook(
                    token_id,
                    on_bbo_change=self._on_l2_bbo_change,
                    on_level_change=ice_cb,
                )
                self._l2_books[token_id] = l2_book
                self._book_trackers[token_id] = L2OrderBookAdapter(l2_book)
        else:
            self._book_trackers[m.yes_token_id] = OrderbookTracker(
                m.yes_token_id,
                on_bbo_change=self._on_orderbook_bbo_change,
            )
            self._book_trackers[m.no_token_id] = OrderbookTracker(
                m.no_token_id,
                on_bbo_change=self._on_orderbook_bbo_change,
            )

    def _unwire_market(self, m: MarketInfo) -> None:
        """Remove all state for an evicted market."""
        self._market_map.pop(m.yes_token_id, None)
        self._market_map.pop(m.no_token_id, None)
        self._yes_aggs.pop(m.yes_token_id, None)
        self._no_aggs.pop(m.no_token_id, None)
        self._detectors.pop(m.condition_id, None)
        self._spread_evaluators.pop(m.condition_id, None)
        self._spread_cooldowns.pop(m.condition_id, None)
        self._regime_detectors.pop(m.condition_id, None)
        self._drift_detectors.pop(m.condition_id, None)
        self._drift_cooldowns.pop(m.condition_id, None)
        self._rpe.clear_market(m.condition_id)
        self.pce.unregister_market(m.condition_id)
        self._book_trackers.pop(m.yes_token_id, None)
        self._book_trackers.pop(m.no_token_id, None)
        # Clean up L2 book instances
        for token_id in (m.yes_token_id, m.no_token_id):
            l2_book = self._l2_books.pop(token_id, None)
            if l2_book is not None:
                l2_book.reset()
            ice = self._iceberg_detectors.pop(token_id, None)
            if ice is not None:
                ice.reset()
        self._trade_counts.pop(m.yes_token_id, None)
        self._trade_counts.pop(m.no_token_id, None)
        self._taker_counts.pop(m.yes_token_id, None)
        self._taker_counts.pop(m.no_token_id, None)
        self._total_counts.pop(m.yes_token_id, None)
        self._total_counts.pop(m.no_token_id, None)
        self._recent_trade_volume.pop(m.condition_id, None)
        self._markets = [x for x in self._markets if x.condition_id != m.condition_id]

    def _positioned_asset_ids(self) -> set[str]:
        """Return asset IDs that currently have open positions.

        Used by the heartbeat to decide which books need fresh data.
        Includes trade_asset_id (YES or NO token) for bidirectional
        RPE positions.
        """
        ids: set[str] = set()
        for pos in self.positions.get_open_positions():
            ids.add(pos.trade_asset_id or pos.no_asset_id)
            ids.add(pos.no_asset_id)
        return ids

    def _is_fee_enabled(self, market: MarketInfo) -> bool:
        """Check if a market's tags match any fee-enabled category."""
        market_tags = (getattr(market, 'tags', '') or '').lower()
        if not market_tags:
            return True
        return any(cat in market_tags for cat in self._fee_category_set)

    def _get_crypto_spot(self) -> float | None:
        """Return latest BTC spot price from the adverse-selection guard.

        Returns None when the guard hasn't started or has no ticks.
        Used as the ``price_fn`` for the shared CryptoPriceModel.
        """
        guard = getattr(self, "_adverse_guard", None)
        if guard is None:
            return None
        # AdverseSelectionGuard doesn't expose get_latest_price();
        # it doesn't maintain external Binance ticks.  The RPE crypto
        # model will return None (no hallucination) until a real
        # external price source is wired.
        fn = getattr(guard, "get_latest_price", None)
        if fn is not None:
            return fn()
        return None

    def _schedule_stop(self) -> None:
        """Schedule stop() as an independent task (idempotent).

        MUST be used instead of calling stop() directly from within any
        of self._tasks — calling stop() inline creates a circular future
        chain (task → stop → gather(tasks) → task) that causes
        RecursionError on cancellation.
        """
        if self._stop_task is None:
            loop = self._loop or asyncio.get_running_loop()
            self._stop_task = loop.create_task(self.stop())

    async def stop(self) -> None:
        """Graceful shutdown: cancel orders, flatten, stop tasks."""
        if not self._running:
            return
        self._running = False
        log.info("bot_stopping")

        cancelled = await self.executor.cancel_all()
        log.info("orders_cancelled", count=cancelled)

        # Cancel all active chaser tasks on open positions
        for pos in self.positions.get_open_positions():
            if pos.entry_chaser_task and not pos.entry_chaser_task.done():
                pos.entry_chaser_task.cancel()
            if pos.exit_chaser_task and not pos.exit_chaser_task.done():
                pos.exit_chaser_task.cancel()

        if self._ws:
            await self._ws.stop()
        if self._l2_ws:
            await self._l2_ws.stop()

        # Cancel all tasks BEFORE stopping process manager so that
        # queue-consumer threads (bbo_consumer, pce_consumer) exit
        # before their underlying queues are closed.
        for task in self._tasks:
            task.cancel()
        # Allow cancelled tasks (especially to_thread consumers) to
        # observe the cancellation and finish their current iteration.
        # Timeout prevents hanging if a thread is stuck on a closed queue.
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                log.warning("task_cancel_timeout", stuck_tasks=[
                    t.get_name() for t in self._tasks if not t.done()
                ])

        # Stop multi-core worker processes (queues safe to close now)
        if self._process_manager is not None:
            await asyncio.to_thread(self._process_manager.stop_all)
            self._process_manager = None

        await self.whale_monitor.stop()

        # Stop new guard and heartbeat
        if hasattr(self, "_adverse_guard"):
            await self._adverse_guard.stop()
        if hasattr(self, "_heartbeat"):
            self._heartbeat.stop()
        if hasattr(self, "_order_poller"):
            self._order_poller.stop()
        if hasattr(self, "_stop_loss_monitor"):
            self._stop_loss_monitor.stop()

        try:
            stats = await asyncio.wait_for(self.trade_store.get_stats(), timeout=5)
            log.info("final_stats", **stats)
            await asyncio.wait_for(self.telegram.notify_stats(stats), timeout=5)
        except Exception:
            log.warning("final_stats_skipped")
        try:
            await asyncio.wait_for(self.trade_store.clear_live_state(), timeout=5)
            await asyncio.wait_for(self.trade_store.close(), timeout=5)
        except Exception:
            log.warning("trade_store_close_error", exc_info=True)

        log.info("bot_stopped")

    # ═══════════════════════════════════════════════════════════════════════
    #  Kill switch
    # ═══════════════════════════════════════════════════════════════════════
    async def kill(self) -> None:
        """Emergency shutdown — cancel everything and exit."""
        log.warning("KILL_SWITCH_ACTIVATED")
        await self.telegram.notify_kill()
        await self.stop()

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 10 — CLOB fill callback for order-status poller
    # ═══════════════════════════════════════════════════════════════════════
    async def _on_clob_fill(self, order: "Order") -> None:
        """Called by :class:`OrderStatusPoller` when a LIVE order fills.

        Routes the fill to the correct position lifecycle handler:
        entry fill → :meth:`_handle_entry_fill`, exit fill → :meth:`_handle_exit_fill`.
        """
        from src.trading.executor import Order  # avoid circular at module-level

        for pos in self.positions.get_open_positions():
            if pos.entry_order and pos.entry_order.order_id == order.order_id:
                if pos.state == PositionState.ENTRY_PENDING:
                    await self._handle_entry_fill(pos)
                    # V1 Adverse-selection monitor: record POST_ONLY entry fills
                    if (
                        self._maker_monitor is not None
                        and getattr(order, "post_only", False)
                        and order.filled_avg_price > 0
                        and order.filled_size > 0
                    ):
                        fill_rec = make_fill_record(
                            market_id=pos.market_id,
                            asset_id=pos.trade_asset_id or pos.no_asset_id,
                            fill_price=order.filled_avg_price,
                            fill_side="BUY",
                            size_usd=order.filled_avg_price * order.filled_size,
                        )
                        self._maker_monitor.record_maker_fill(fill_rec)
                return
            if pos.exit_order and pos.exit_order.order_id == order.order_id:
                if pos.state == PositionState.EXIT_PENDING:
                    self._handle_exit_fill(pos)
                return

        log.debug("clob_fill_no_position", order_id=order.order_id)

    # ═══════════════════════════════════════════════════════════════════════
    #  Adaptive polling helper
    # ═══════════════════════════════════════════════════════════════════════
    def _latest_zscore(self) -> float:
        """Callback for WhaleMonitor — returns the most recent panic Z-score."""
        return self._latest_z

    # ═══════════════════════════════════════════════════════════════════════
    #  Core processing loops
    # ═══════════════════════════════════════════════════════════════════════
    async def _process_trades(self) -> None:
        """Main loop: consume TradeEvents, update aggregators, check signals.

        Pillar 4 integration: every message is latency-checked.  If the
        guard is BLOCKED, aggregators still receive the price (so bars
        don't gap on recovery) but signal evaluation and fill checking
        are skipped.
        """
        while self._running:
            try:
                event = await asyncio.wait_for(self._trade_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # ── Pillar 4: Latency gate ─────────────────────────────────
                latency_state = self.latency_guard.check(event.timestamp)
                is_blocked = latency_state == LatencyState.BLOCKED

                if is_blocked and latency_state != getattr(self, "_prev_latency", None):
                    await self.telegram.send(
                        "🚨 <b>Latency BLOCKED</b> — execution halted until WS stabilises.\n"
                        f"Delta: {self.latency_guard.last_delta_ms:.0f}ms"
                    )
                if not is_blocked and getattr(self, "_prev_latency", None) == LatencyState.BLOCKED:
                    await self.telegram.send("✅ <b>Latency recovered</b> — execution resumed.")
                self._prev_latency = latency_state

                # Track trade frequency
                self._trade_counts[event.asset_id] = (
                    self._trade_counts.get(event.asset_id, 0.0) + 1.0
                )

                # ── MTI: classify trade as taker or maker ──────────────────
                book = self._book_trackers.get(event.asset_id)
                if book and book.has_data:
                    snap = book.snapshot()
                    is_taker = False
                    if event.side == "buy" and snap.best_ask > 0 and event.price >= snap.best_ask:
                        is_taker = True
                    elif event.side == "sell" and snap.best_bid > 0 and event.price <= snap.best_bid:
                        is_taker = True
                    event.is_taker = is_taker

                    self._total_counts[event.asset_id] = (
                        self._total_counts.get(event.asset_id, 0) + 1
                    )
                    if is_taker:
                        self._taker_counts[event.asset_id] = (
                            self._taker_counts.get(event.asset_id, 0) + 1
                        )

                # ── Ghost liquidity: record recent trade volume ────────────
                market_info = self._market_map.get(event.asset_id)
                if market_info:
                    cid = market_info.condition_id
                    if cid not in self._recent_trade_volume:
                        self._recent_trade_volume[cid] = []
                    self._recent_trade_volume[cid].append(
                        (event.timestamp, event.price * event.size)
                    )

                # Route to the correct aggregator (always, even when blocked)
                yes_agg = self._yes_aggs.get(event.asset_id)
                no_agg = self._no_aggs.get(event.asset_id)

                if yes_agg:
                    bar = yes_agg.on_trade(event)
                    if bar and not is_blocked:
                        await self._on_yes_bar_closed(event.asset_id, bar)

                if no_agg:
                    no_agg.on_trade(event)
                    if self.paper_mode and not is_blocked:
                        self._check_paper_fills(event)

                if self.paper_mode and yes_agg and not is_blocked:
                    self._check_paper_fills(event)
            except asyncio.CancelledError:
                raise
            except KeyError:
                # Stale / missing asset data — non-fatal, skip this event
                log.warning(
                    "trade_processing_stale_key",
                    asset_id=event.asset_id,
                    exc_info=True,
                )
            except Exception:
                log.error(
                    "trade_processing_error",
                    asset_id=event.asset_id,
                    exc_info=True,
                )
                if self._trade_loop_breaker.record():
                    log.critical(
                        "trade_processor_circuit_breaker_tripped",
                        errors_in_window=self._trade_loop_breaker.recent_errors,
                        msg="Too many unexpected errors in trade processor — initiating shutdown",
                    )
                    await self.telegram.send(
                        "🔴 <b>CIRCUIT BREAKER</b>: trade_processor tripped "
                        "(5 unexpected errors in 60s) — shutting down."
                    )
                    self._schedule_stop()
                    return

    async def _process_book_updates(self) -> None:
        """Consume orderbook WS events and update trackers.

        When the latency guard is BLOCKED, snapshots are tagged as
        ``fresh=False`` so downstream consumers (sizer, chaser) know
        the data may be stale.

        When L2 is enabled, trackers backed by L2OrderBookAdapter are
        fed directly by the L2WebSocket — ``on_price_change`` /
        ``on_book_snapshot`` are no-ops on adapters.  This loop still
        runs for any remaining non-L2 trackers and as a fallback.
        """
        while self._running:
            try:
                msg = await asyncio.wait_for(self._book_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if not isinstance(msg, dict):
                continue

            event_type = msg.get("event_type") or msg.get("type", "")
            asset_id = msg.get("asset_id") or ""

            tracker = self._book_trackers.get(asset_id)
            if not tracker:
                continue

            # L2-backed trackers are no-ops for these calls, but we
            # still route for backward compat and non-L2 fallback.
            try:
                if event_type == "price_change":
                    tracker.on_price_change(msg)
                elif event_type == "book":
                    tracker.on_book_snapshot(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.error(
                    "book_update_error",
                    asset_id=asset_id,
                    event_type=event_type,
                    exc_info=True,
                )

    # ── Multi-core consumer loops ─────────────────────────────────────────
    async def _consume_bbo_events(self) -> None:
        """Consume BBO change notifications from L2 worker processes.

        Reads ``(\"bbo\", asset_id, seq)`` tuples from the shared
        ``multiprocessing.Queue``, then drives the same callbacks the
        in-process ``_on_l2_bbo_change`` would have fired.
        """
        if self._process_manager is None:
            return
        bbo_queue = self._process_manager.bbo_queue

        while self._running:
            try:
                # Poll the multiprocessing Queue from a thread to avoid
                # blocking the event loop.
                try:
                    msg = await asyncio.to_thread(bbo_queue.get, timeout=0.5)
                except Exception:
                    # Empty queue or timeout
                    continue

                if msg is None:
                    continue

                cmd = msg[0]
                if cmd == "bbo":
                    _, asset_id, seq = msg
                    # Read spread score from shared memory for the callback
                    reader = self._process_manager.get_reader(asset_id)
                    if reader is None:
                        continue
                    snap = reader.read_header()
                    # Build a minimal SpreadScore-like object
                    from src.data.spread_score import SpreadScore
                    score = SpreadScore(
                        raw_spread_cents=round(snap.spread * 100, 2) if snap.spread else 0.0,
                        depth_weighted_spread_cents=0.0,
                        score=snap.spread_score,
                        timestamp=snap.timestamp,
                    )
                    await self._on_l2_bbo_change(asset_id, score)
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("bbo_consumer_error", exc_info=True)

    async def _consume_pce_results(self) -> None:
        """Consume results from the PCE computation worker.

        Dispatches dashboard data, cross-market signals, etc.
        """
        if self._process_manager is None:
            return
        output_queue = self._process_manager.pce_output_queue

        while self._running:
            try:
                try:
                    msg = await asyncio.to_thread(output_queue.get, timeout=1.0)
                except Exception:
                    continue

                if msg is None:
                    continue

                cmd = msg[0]
                if cmd == "pce_refreshed":
                    _, dashboard = msg
                    await self.telegram.notify_pce_dashboard(dashboard)
                    log.info(
                        "pce_dashboard_received",
                        portfolio_var=dashboard.get("portfolio_var", 0.0),
                        pairs_tracked=dashboard.get("total_pairs_tracked", 0),
                    )
                elif cmd == "cm_signals":
                    _, signal_dicts = msg
                    if signal_dicts:
                        log.info("cross_market_signals_received", count=len(signal_dicts))
                elif cmd == "prior_validation":
                    _, summary = msg
                    log.info("pce_prior_validation_received", **summary)
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("pce_consumer_error", exc_info=True)

    # ── L2 callbacks ──────────────────────────────────────────────────────
    async def _on_l2_bbo_change(self, asset_id: str, score: Any) -> None:
        """Callback from L2OrderBook when the BBO changes.

        Logs the live spread score for monitoring and drives the
        event-driven stop-loss engine.  Also evaluates the spread-based
        signal path (Problem 3) on NO-token BBO ticks.

        Fully exception-guarded: any failure is logged and contained
        so that a single bad tick cannot kill the callback pipeline.
        """
        try:
            await self._on_l2_bbo_change_inner(asset_id, score)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error(
                "l2_bbo_callback_error",
                asset_id=asset_id,
                exc_info=True,
            )

    async def _on_l2_bbo_change_inner(self, asset_id: str, score: Any) -> None:
        """Inner implementation — called by the guarded wrapper."""
        log.debug(
            "l2_bbo_update",
            asset_id=asset_id,
            spread_score=round(score.score, 1),
            raw_spread_cents=round(score.raw_spread_cents, 2),
        )
        # Drive event-driven stop-loss evaluation
        await self._stop_loss_monitor.on_bbo_update(asset_id)

        # Tick the maker adverse-selection monitor (schedules T+5/15/60 marks)
        if self._maker_monitor is not None:
            await self._maker_monitor.tick()

        # ── Spread-based signal evaluation (Problem 3) ─────────────────
        if not settings.strategy.spread_signal_enabled:
            return

        market_info = self._market_map.get(asset_id)
        if not market_info:
            return

        # Only evaluate on NO-token BBO changes
        if asset_id != market_info.no_token_id:
            return

        # Only trade active-tier markets
        if not self.lifecycle.is_tradeable(market_info.condition_id):
            return

        # Signal cooldown check (lifecycle cooldown)
        if not self.lifecycle.is_cooled_down(market_info.condition_id):
            return

        # Stop-loss cooldown — prevent re-entry into recently stopped-out markets
        if not self.positions.is_stop_loss_cooled_down(market_info.condition_id):
            return

        # Per-signal-source cooldown (configurable, default 30s)
        now = time.time()
        last_fire = self._spread_cooldowns.get(market_info.condition_id, 0.0)
        if now - last_fire < settings.strategy.spread_signal_cooldown_s:
            return

        # Check minimum spread
        strat = settings.strategy
        if score.raw_spread_cents < strat.min_spread_cents:
            return

        no_book = self._book_trackers.get(market_info.no_token_id)
        if not no_book or not no_book.has_data:
            return

        evaluator = self._spread_evaluators.get(market_info.condition_id)
        if not evaluator:
            return

        actionable, composite_score, fired = evaluator.is_actionable(
            no_book=no_book,
        )

        if not actionable:
            return

        # Generate a synthetic PanicSignal to enter the standard flow
        no_agg = self._no_aggs.get(market_info.no_token_id)
        if not no_agg:
            return

        snap = no_book.snapshot()
        no_best_ask = snap.best_ask if snap.best_ask > 0 else no_agg.current_price
        if no_best_ask <= 0:
            return

        yes_agg = self._yes_aggs.get(market_info.yes_token_id)
        yes_price = yes_agg.current_price if yes_agg else 0.50
        yes_vwap = yes_agg.rolling_vwap if yes_agg else 0.50

        # ── Price band guard (consistent with _on_yes_bar_closed) ──────
        strat_band = settings.strategy
        if yes_price <= strat_band.min_tradeable_price or yes_price >= strat_band.max_tradeable_price:
            return
        if yes_price >= 0.97 or yes_price <= 0.03:
            self.lifecycle.drain_market(market_info.condition_id, reason="near_resolved_price")
            return
        if not market_info.accepting_orders:
            self.lifecycle.drain_market(market_info.condition_id, reason="not_accepting_orders")
            return

        # Build synthetic signal with moderate values
        synthetic_sig = PanicSignal(
            market_id=market_info.condition_id,
            yes_asset_id=market_info.yes_token_id,
            no_asset_id=market_info.no_token_id,
            yes_price=yes_price,
            yes_vwap=yes_vwap,
            zscore=strat.zscore_threshold * 1.2,  # just above threshold
            volume_ratio=strat.volume_ratio_threshold * 1.1,
            no_best_ask=no_best_ask,
            whale_confluence=False,
        )

        # Build signal metadata with uncertainty from evaluator +
        # a sizing multiplier of 50% (lower conviction)
        spread_metadata: dict[str, Any] = {}
        if fired:
            spread_metadata = dict(fired[0].metadata)
        spread_metadata["signal_source"] = "spread_opportunity"
        spread_metadata["spread_sizing_mult"] = 0.50

        self._spread_cooldowns[market_info.condition_id] = now

        log.info(
            "spread_signal_fired",
            market=market_info.condition_id,
            composite_score=round(composite_score, 3),
            signal_source="spread_opportunity",
            raw_spread_cents=round(score.raw_spread_cents, 2),
        )

        self.lifecycle.record_signal(market_info.condition_id)

        # Compute days to resolution
        days = 30
        if market_info.end_date:
            days = max(1, (market_info.end_date - datetime.now(timezone.utc)).days)

        # Get real book depth ratio if available
        book_depth = 1.0
        if no_book and no_book.has_data:
            book_depth = no_book.book_depth_ratio

        # Determine fee category
        fee_enabled = self._is_fee_enabled(market_info)

        pos = await self.positions.open_position(
            synthetic_sig,
            no_agg,
            no_book=no_book,
            event_id=market_info.event_id,
            days_to_resolution=days,
            book_depth_ratio=book_depth,
            fee_enabled=fee_enabled,
            signal_metadata=spread_metadata,
        )
        if pos:
            if no_book and no_book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, no_book),
                    name=f"chaser_entry_{pos.id}",
                )
                chaser_task.add_done_callback(_safe_task_done_callback)
                pos.entry_chaser_task = chaser_task

    def _on_orderbook_bbo_change(self, asset_id: str, snapshot: Any) -> None:
        """Callback from basic OrderbookTracker when the BBO changes.

        Schedules an async stop-loss evaluation for this asset.
        """
        _safe_fire_and_forget(
            self._stop_loss_monitor.on_bbo_update(asset_id),
            name=f"stop_loss_bbo_{asset_id[:12]}",
        )

    async def _on_l2_desync(self, asset_id: str) -> None:
        """Callback from L2OrderBook on sequence gap detection.

        Routes to the L2WebSocket for snapshot re-fetch.
        Exception-guarded so a desync error cannot kill the callback.
        """
        try:
            if self._l2_ws is not None:
                await self._l2_ws._on_book_desync(asset_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("l2_desync_callback_error", asset_id=asset_id, exc_info=True)

    async def _on_yes_bar_closed(self, yes_asset_id: str, bar: Any) -> None:
        """A 1-min YES bar just closed — evaluate the panic detector."""
        market_info = self._market_map.get(yes_asset_id)
        if not market_info:
            return

        # Only trade active-tier markets
        if not self.lifecycle.is_tradeable(market_info.condition_id):
            return

        # ── Fix 4: Real-time drain if market stopped accepting orders ──
        if not market_info.accepting_orders:
            self.lifecycle.drain_market(market_info.condition_id, reason="not_accepting_orders")
            return

        # Signal cooldown check
        if not self.lifecycle.is_cooled_down(market_info.condition_id):
            return

        # ── L2 book reliability gate ───────────────────────────────
        # Chronically desyncing books produce unreliable BBO/depth data.
        # Skip signal evaluation but do NOT evict — let recovery continue.
        l2_yes = self._l2_books.get(market_info.yes_token_id)
        if l2_yes is not None and not l2_yes.is_reliable:
            log.info(
                "l2_book_unreliable",
                asset_id=market_info.yes_token_id,
                seq_gap_rate=round(l2_yes.seq_gap_rate, 4),
                delta_count=l2_yes.delta_count,
            )
            return

        detector = self._detectors.get(market_info.condition_id)
        if not detector:
            return

        no_agg = self._no_aggs.get(market_info.no_token_id)
        if not no_agg:
            return

        # Use real orderbook best_ask if available, else fall back to last trade
        no_book = self._book_trackers.get(market_info.no_token_id)
        if no_book and no_book.has_data:
            snap = no_book.snapshot()
            no_best_ask = snap.best_ask if snap.best_ask > 0 else no_agg.current_price
        else:
            no_best_ask = no_agg.current_price

        if no_best_ask <= 0:
            return

        # ── Derive YES close price for price-band checks ───────────────
        yes_price = bar.close if hasattr(bar, 'close') else 0.0

        # ── Fix 2: Near-resolved price auto-drain ──────────────────────
        if yes_price >= 0.97 or yes_price <= 0.03:
            self.lifecycle.drain_market(market_info.condition_id, reason="near_resolved_price")
            return

        # ── Fix 1: Tradeable price band guard ──────────────────────────
        min_price = settings.strategy.min_tradeable_price
        max_price = settings.strategy.max_tradeable_price
        price_in_band = min_price < yes_price < max_price

        # Whale confluence check
        whale = self.whale_monitor.has_confluence(market_info.no_token_id)

        if price_in_band:
            # ── SI-1: Update regime detector and gate entries ───────────
            regime_det = self._regime_detectors.get(market_info.condition_id)
            yes_agg_regime = self._yes_aggs.get(market_info.yes_token_id)
            if regime_det and yes_agg_regime:
                # Compute log-return from the bar
                closes = [b.close for b in yes_agg_regime.bars]
                if len(closes) >= 2 and closes[-2] > 0:
                    import math as _math
                    lr = _math.log(closes[-1] / closes[-2])
                else:
                    lr = 0.0
                regime_det.update(
                    log_return=lr,
                    ewma_vol=yes_agg_regime.rolling_volatility_ewma,
                    ew_vol=yes_agg_regime.rolling_volatility,
                )
                # SI-3: record return for cross-market signal generator
                if self._multicore_enabled and self._process_manager is not None:
                    try:
                        self._process_manager.pce_input_queue.put_nowait(
                            ("bar_return", market_info.condition_id, lr,
                             market_info.yes_token_id, market_info.no_token_id),
                        )
                    except _queue_mod.Full:
                        self._pce_queue_drops += 1
                        if self._pce_queue_drops % 100 == 1:
                            log.warning("queue_full_drop", queue="pce_input", total_drops=self._pce_queue_drops)
                elif self._cross_market is not None:
                    self._cross_market.record_return(
                        market_info.condition_id,
                        lr,
                        yes_asset_id=market_info.yes_token_id,
                        no_asset_id=market_info.no_token_id,
                    )

            sig = detector.evaluate(bar, no_best_ask=no_best_ask, whale_confluence=whale)
            if sig:
                # SI-1: Regime gate — suppress panic in trending regime
                if regime_det and not regime_det.is_mean_revert:
                    log.info(
                        "regime_gate_suppressed",
                        market_id=market_info.condition_id[:16],
                        regime_score=round(regime_det.regime_score, 3),
                    )
                elif not self.positions.is_stop_loss_cooled_down(market_info.condition_id):
                    log.info(
                        "stop_loss_cooldown_suppressed",
                        market_id=market_info.condition_id[:16],
                    )
                else:
                    self._latest_z = sig.zscore
                    self.lifecycle.record_signal(market_info.condition_id)
                    await self._on_panic_signal(
                        sig, no_agg, market_info,
                        signal_metadata={
                            "regime_mean_revert": (
                                regime_det.is_mean_revert if regime_det else False
                            ),
                        },
                    )

            # ── V3: Drift signal (low-volatility mean-reversion) ───────────
            # Only evaluate when PanicDetector did NOT fire — ensures
            # uncorrelated signal source.  Requires MR regime.
            if not sig and settings.strategy.drift_signal_enabled:
                drift_det = self._drift_detectors.get(market_info.condition_id)
                if drift_det and no_agg:
                    # Drift cooldown
                    now_drift = time.time()
                    last_drift = self._drift_cooldowns.get(market_info.condition_id, 0.0)
                    if now_drift - last_drift >= settings.strategy.drift_cooldown_s:
                        # Check L2 reliability
                        l2_no = self._l2_books.get(market_info.no_token_id)
                        l2_ok = l2_no is None or l2_no.is_reliable

                        drift_sig = drift_det.evaluate(
                            no_agg,
                            no_asset_id=market_info.no_token_id,
                            no_best_ask=no_best_ask,
                            regime_is_mean_revert=(
                                regime_det.is_mean_revert if regime_det else False
                            ),
                            l2_reliable=l2_ok,
                        )
                        if drift_sig and drift_sig.direction == "BUY_NO":
                            if not self.positions.is_stop_loss_cooled_down(market_info.condition_id):
                                log.info(
                                    "stop_loss_cooldown_suppressed_drift",
                                    market_id=market_info.condition_id[:16],
                                )
                            else:
                                self._drift_cooldowns[market_info.condition_id] = now_drift
                                self.lifecycle.record_signal(market_info.condition_id)
                                await self._on_panic_signal(
                                    drift_sig, no_agg, market_info,
                                    signal_metadata={
                                        "signal_source": "drift",  # Flaw 2: suppresses regime discount
                                        "source": "drift",         # legacy compat
                                        "drift_score": drift_sig.score,
                                        "regime_mean_revert": True,
                                        "spread_compressed": False,
                                    },
                                )

        # ── RPE evaluation (Pillar 14) ───────────────────────────────────
        rpe = self._rpe
        if rpe and price_in_band:
            # Deliverable D: Data freshness gate
            # Use last_trade_time (updated on every trade) instead of
            # bars[-1].open_time (only updated when a bar closes).
            # This prevents false stale-data rejections between bars
            # on healthy markets with moderate trade frequency.
            yes_agg_rpe = self._yes_aggs.get(market_info.yes_token_id)
            max_age = settings.strategy.rpe_max_data_age_seconds
            if yes_agg_rpe:
                if yes_agg_rpe.last_trade_time <= 0:
                    # No trades received at all — genuinely stale
                    log.info(
                        "rpe_no_trade_data",
                        market=market_info.condition_id,
                    )
                    return
                data_age = time.time() - yes_agg_rpe.last_trade_time
                if data_age > max_age:
                    log.info(
                        "rpe_stale_data",
                        market=market_info.condition_id,
                        data_age_s=round(data_age, 1),
                        max_age_s=max_age,
                    )
                    return

            days = 30
            total_duration_days = 90.0
            if market_info.end_date:
                days = max(1, (market_info.end_date - datetime.now(timezone.utc)).days)
                # Estimate total market duration from creation to end.
                # Polymarket markets typically run 30-180 days.  If we
                # don't know the creation date, assume 90 days total.
                total_duration_days = max(float(days), 90.0)

            # Thread L2 order-book data into RPE for continuous
            # observation updates (Dynamic Prior Generation Engine).
            no_book_rpe = self._book_trackers.get(market_info.no_token_id)
            book_ratio = None
            l2_ok = False
            if no_book_rpe and no_book_rpe.has_data:
                book_ratio = no_book_rpe.book_depth_ratio
                l2_ok = no_book_rpe.is_reliable

            try:
                rpe_signal = rpe.evaluate(
                    market=market_info,
                    market_price=yes_price,
                    days_to_resolution=days,
                    total_duration_days=total_duration_days,
                    book_depth_ratio=book_ratio,
                    l2_reliable=l2_ok,
                )
                if rpe_signal:
                    await self._on_rpe_signal(rpe_signal, market_info, days, current_price=yes_price)
            except Exception:
                log.warning("rpe_evaluation_error", market=market_info.condition_id, exc_info=True)

    async def _on_panic_signal(
        self, sig: BaseSignal, no_agg: OHLCVAggregator, market: MarketInfo,
        signal_metadata: dict | None = None,
    ) -> None:
        """Handle a confirmed entry signal (PanicSignal or DriftSignal)."""
        # Fire-and-forget — Telegram notification must NOT block the
        # alpha-critical execution path (saves 50-500ms per trade).
        _notify_zscore = sig.zscore if isinstance(sig, PanicSignal) else abs(getattr(sig, "displacement", 0.0))
        _notify_vratio = sig.volume_ratio if isinstance(sig, PanicSignal) else 1.0
        asyncio.ensure_future(self.telegram.notify_signal(
            sig.market_id, _notify_zscore, _notify_vratio
        ))

        # Compute days to resolution
        days = 30
        if market.end_date:
            days = max(1, (market.end_date - datetime.now(timezone.utc)).days)

        # Get real book depth ratio if available
        book_depth = 1.0
        no_book = self._book_trackers.get(market.no_token_id)
        if no_book and no_book.has_data:
            book_depth = no_book.book_depth_ratio

            # ── Minimum ask-depth gate ─────────────────────────────────
            snap = no_book.snapshot()
            min_depth = settings.strategy.min_ask_depth_usd
            if snap.ask_depth_usd < min_depth:
                log.info(
                    "panic_rejected_thin_asks",
                    market=market.condition_id,
                    ask_depth_usd=round(snap.ask_depth_usd, 2),
                    min_required=min_depth,
                )
                return

        # Fetch fee rates for this token
        # Determine if market is fee-enabled based on category
        fee_enabled = self._is_fee_enabled(market)

        pos = await self.positions.open_position(
            sig,
            no_agg,
            no_book=no_book,
            event_id=market.event_id,
            days_to_resolution=days,
            book_depth_ratio=book_depth,
            fee_enabled=fee_enabled,
            signal_metadata=signal_metadata,
        )
        if pos:
            # Launch entry chaser as a child task (Pillar 1)
            # Panic signals use urgent=True to escalate immediately
            # and capture the edge before it decays.
            if no_book and no_book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, no_book, urgent=True),
                    name=f"chaser_entry_{pos.id}",
                )
                chaser_task.add_done_callback(_safe_task_done_callback)
                pos.entry_chaser_task = chaser_task
            # else: order already placed directly by PositionManager

        if pos and self.positions.circuit_breaker_active:
            await self.telegram.send(
                "🛑 <b>Circuit breaker tripped</b> — pausing new positions."
            )

    async def _on_rpe_signal(
        self,
        signal: SignalResult,
        market: MarketInfo,
        days_to_resolution: int,
        *,
        current_price: float | None = None,
    ) -> None:
        """Handle an RPE divergence signal — may be shadow or live.

        Shadow mode control flow
        ────────────────────────
        RPE shadow mode is controlled by the ``RPE_SHADOW_MODE`` env var
        (default ``False`` → **live** mode).  When ``rpe_shadow_mode=True``:

        1. The RPE still evaluates markets normally and fires signals.
        2. Signals pass through EQS filtering and cooldown checks.
        3. Signals are recorded in ``RPECalibrationTracker`` with
           ``shadow=True`` for offline Brier/log-loss scoring.
        4. A Telegram notification is sent (prefixed "👻 SHADOW").
        5. **No position is opened** — the method returns early at the
           ``if shadow: return`` gate below the notification block.

        This allows the operator to collect calibration data on RPE
        accuracy without risking real (or paper) capital, then switch
        to live by setting ``RPE_SHADOW_MODE=false``.
        """
        meta = signal.metadata
        direction = meta.get("direction", "buy_no")
        model_prob = meta.get("model_probability", 0.5)
        confidence = meta.get("confidence", 0.0)
        shadow = meta.get("shadow_mode", True)
        strat = settings.strategy

        # Fix 3: Use actual market price, not the normalised divergence score
        display_price = current_price if current_price is not None else meta.get("market_price", signal.score)

        # ── Deliverable C: Per-market cooldown ─────────────────────────
        now = time.monotonic()
        last_fire = self._rpe_last_signal.get(market.condition_id, 0.0)
        if now - last_fire < strat.rpe_cooldown_seconds:
            log.debug(
                "rpe_cooldown_active",
                market=market.condition_id,
                seconds_remaining=round(strat.rpe_cooldown_seconds - (now - last_fire), 1),
            )
            return

        # ── Deliverable B: EQS gate (all paths, before Telegram) ────────
        # Determine entry price for EQS calculation
        yes_agg = self._yes_aggs.get(market.yes_token_id)
        no_agg = self._no_aggs.get(market.no_token_id)
        if direction == "buy_no":
            book = self._book_trackers.get(market.no_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                eqs_entry = snap.best_ask - 0.01 if snap.best_ask > 0 else 0.0
            else:
                eqs_entry = no_agg.current_price if no_agg else 0.0
        else:
            book = self._book_trackers.get(market.yes_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                eqs_entry = snap.best_ask - 0.01 if snap.best_ask > 0 else 0.0
            else:
                eqs_entry = yes_agg.current_price if yes_agg else 0.0

        if eqs_entry > 0:
            fee_enabled = self._is_fee_enabled(market)

            edge = compute_edge_score(
                entry_price=eqs_entry,
                no_vwap=model_prob,
                zscore=0.0,
                volume_ratio=0.0,
                whale_confluence=False,
                fee_enabled=fee_enabled,
                model_confidence=confidence,
                min_score=strat.rpe_min_eqs,
            )
            log.debug(
                "rpe_eqs_assessment",
                market=market.condition_id,
                eqs=edge.score,
                viable=edge.viable,
                regime=edge.regime_quality,
                fee_eff=edge.fee_efficiency,
                signal_q=edge.signal_quality,
                reason=edge.rejection_reason,
            )
            if not edge.viable:
                log.info(
                    "rpe_eqs_rejected",
                    market=market.condition_id,
                    score=edge.score,
                    min_required=strat.rpe_min_eqs,
                    reason=edge.rejection_reason,
                    direction=direction,
                    confidence=round(confidence, 3),
                )
                # Do NOT stamp the cooldown on EQS rejection — market
                # conditions (price, spread, depth) can change rapidly and
                # we want to re-evaluate as soon as a new bar arrives.
                # The per-source cooldown on *successful* fires prevents
                # log spam from repeated passes.
                return

        # ── Record cooldown timestamp (after EQS passes) ─────────────
        self._rpe_last_signal[market.condition_id] = now

        # ── Record in calibration tracker (Deliverable A) ────────────
        model_meta = meta.get("model_metadata", {})
        self._rpe_calibration.record_signal(
            market_id=market.condition_id,
            model_prob=model_prob,
            market_price=display_price,
            direction=direction,
            timestamp=time.time(),
            shadow=shadow,
            prior_source=model_meta.get("prior_source", ""),
            l2_active=model_meta.get("l2_active", False),
            theta_w_prior=model_meta.get("w_prior", 0.0),
        )

        # ── Telegram notification (with calibration footer) ───────────
        cal_footer = ""
        cal_stats = self._rpe_calibration.calibration_summary()
        if cal_stats.get("resolved", 0) >= 20:
            brier = cal_stats.get("brier_score", "n/a")
            logloss = cal_stats.get("log_loss", "n/a")
            acc = cal_stats.get("direction_accuracy", "n/a")
            cal_footer = f"\nCalibration (N={cal_stats['resolved']}): Brier={brier} LL={logloss} Acc={acc}"

        # Fire-and-forget — keep Telegram off the execution hot path.
        asyncio.ensure_future(self.telegram.notify_rpe_signal(
            market_id=market.condition_id,
            model_prob=model_prob,
            market_price=display_price,
            direction=direction,
            confidence=confidence,
            shadow=shadow,
            question=market.question,
            calibration_footer=cal_footer,
        ))

        # Record in data recorder for backtesting
        if self._recorder is not None:
            self._recorder.enqueue("rpe_signal", {
                "market": market.condition_id,
                "asset_id": market.yes_token_id,
                "model_probability": model_prob,
                "confidence": confidence,
                "direction": direction,
                "shadow": shadow,
                "model_metadata": meta.get("model_metadata", {}),
            })

        if shadow:
            log.info(
                "rpe_shadow_signal",
                market=market.condition_id,
                model_prob=round(model_prob, 4),
                confidence=round(confidence, 3),
                direction=direction,
            )
            return

        # ── Live-mode safety gates (conservative RPE activation) ───────
        # Gate 1: Require higher confidence than the base threshold
        if confidence < 0.15:
            log.info(
                "rpe_live_confidence_too_low",
                market=market.condition_id,
                confidence=round(confidence, 3),
                min_required=0.15,
            )
            return

        # Gate 2: Require >= 10 OHLCV bars of history for meaningful data
        bar_count = max(
            len(yes_agg.bars) if yes_agg else 0,
            len(no_agg.bars) if no_agg else 0,
        )
        if bar_count < 10:
            log.info(
                "rpe_live_insufficient_bars",
                market=market.condition_id,
                bar_count=bar_count,
                min_required=10,
            )
            return

        # ── Deliverable C: Position deduplication ────────────────────
        positioned = self._positioned_asset_ids()
        if market.yes_token_id in positioned or market.no_token_id in positioned:
            log.info(
                "rpe_already_positioned",
                market=market.condition_id,
                yes_token=market.yes_token_id,
                no_token=market.no_token_id,
            )
            return

        # Live mode — open a position
        if not fee_enabled:
            fee_enabled = self._is_fee_enabled(market)

        # Determine entry price from the appropriate book
        if direction == "buy_no":
            book = self._book_trackers.get(market.no_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                entry_price = round(snap.best_ask - 0.01, 2) if snap.best_ask > 0 else 0.0
            else:
                entry_price = round(no_agg.current_price, 2) if no_agg else 0.0
        else:
            book = self._book_trackers.get(market.yes_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                entry_price = round(snap.best_ask - 0.01, 2) if snap.best_ask > 0 else 0.0
            else:
                entry_price = round(yes_agg.current_price, 2) if yes_agg else 0.0

        if entry_price <= 0:
            return

        # ── Minimum ask-depth gate (RPE path) ────────────────────────
        if book and book.has_data:
            snap_depth = book.snapshot()
            min_depth = settings.strategy.min_ask_depth_usd
            if snap_depth.ask_depth_usd < min_depth:
                log.info(
                    "rpe_rejected_thin_asks",
                    market=market.condition_id,
                    ask_depth_usd=round(snap_depth.ask_depth_usd, 2),
                    min_required=min_depth,
                    direction=direction,
                )
                return

        pos = await self.positions.open_rpe_position(
            market_id=market.condition_id,
            yes_asset_id=market.yes_token_id,
            no_asset_id=market.no_token_id,
            direction=direction,
            model_probability=model_prob,
            confidence=confidence,
            entry_price=entry_price,
            event_id=market.event_id,
            days_to_resolution=days_to_resolution,
            fee_enabled=fee_enabled,
            book=book,
            signal_metadata=meta,
        )

        # ── Model-only probe routing (Dynamic Prior Engine) ──────────
        # When GenericBayesianModel fires a massive divergence but
        # PanicDetector is silent, route through V4 probe sizing for
        # bounded-risk model exploration.
        if pos is None and meta.get("model_name") == "generic_bayesian":
            model_meta = meta.get("model_metadata", {})
            divergence_abs = abs(meta.get("divergence", 0.0))
            probe_min = settings.strategy.rpe_probe_divergence_min

            # Check PanicDetector is truly silent for this market
            panic_det = self._detectors.get(market.condition_id)
            panic_silent = True
            if panic_det and no_agg and no_agg.bars:
                latest_bar = no_agg.bars[-1]
                no_book_probe = self._book_trackers.get(market.no_token_id)
                no_ask_probe = 0.0
                if no_book_probe and no_book_probe.has_data:
                    no_ask_probe = no_book_probe.snapshot().best_ask
                try:
                    panic_sig = panic_det.evaluate(latest_bar, no_ask_probe)
                    if panic_sig is not None:
                        panic_silent = False
                except Exception:
                    pass

            if (
                divergence_abs >= probe_min
                and panic_silent
                and model_meta.get("dynamic_prior_enabled", False)
            ):
                probe_meta = dict(meta)
                probe_meta["force_probe"] = True
                probe_meta["probe_reason"] = "model_divergence_no_panic"
                log.info(
                    "rpe_model_probe_attempt",
                    market=market.condition_id,
                    divergence=round(divergence_abs, 4),
                    probe_min=probe_min,
                    prior_source=model_meta.get("prior_source", "unknown"),
                )
                pos = await self.positions.open_rpe_position(
                    market_id=market.condition_id,
                    yes_asset_id=market.yes_token_id,
                    no_asset_id=market.no_token_id,
                    direction=direction,
                    model_probability=model_prob,
                    confidence=confidence,
                    entry_price=entry_price,
                    event_id=market.event_id,
                    days_to_resolution=days_to_resolution,
                    fee_enabled=fee_enabled,
                    book=book,
                    signal_metadata=probe_meta,
                )

        if pos:
            if book and book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, book),
                    name=f"chaser_entry_{pos.id}",
                )
                chaser_task.add_done_callback(_safe_task_done_callback)
                pos.entry_chaser_task = chaser_task

    async def _rpe_crypto_retrigger_loop(self) -> None:
        """Periodically check if BTC spot has moved enough to re-evaluate RPE.

        Runs every 5 seconds.  When the spot price changes by more than
        ``rpe_crypto_retrigger_cents`` since the last evaluation, re-runs
        the RPE for all active crypto-tagged markets.  This catches
        mispricings that appear between 1-minute bar closes.

        Does NOT open a new WebSocket — reuses the same ``price_fn``
        that the CryptoPriceModel already reads from.
        """
        interval_s = 5.0
        while self._running:
            await asyncio.sleep(interval_s)
            try:
                spot = self._get_crypto_spot()
                if spot is None:
                    continue

                threshold = settings.strategy.rpe_crypto_retrigger_cents
                last = self._rpe_last_spot
                if last is not None and abs(spot - last) < threshold:
                    continue

                # Spot moved enough — re-evaluate crypto markets
                self._rpe_last_spot = spot
                for m in list(self._markets):
                    if not self.lifecycle.is_tradeable(m.condition_id):
                        continue
                    # Quick check: is this a crypto market?
                    tags = (getattr(m, "tags", "") or "").lower()
                    question_lower = m.question.lower()
                    is_crypto = (
                        "crypto" in tags
                        or any(
                            kw in question_lower
                            for kw in ("bitcoin", "btc", "ethereum", "eth")
                        )
                    )
                    if not is_crypto:
                        continue

                    # Use latest YES price from aggregator or book
                    yes_agg = self._yes_aggs.get(m.yes_token_id)
                    yes_book = self._book_trackers.get(m.yes_token_id)
                    if yes_book and yes_book.has_data:
                        snap = yes_book.snapshot()
                        yes_price = snap.mid_price if snap.mid_price > 0 else (
                            yes_agg.current_price if yes_agg else 0.0
                        )
                    else:
                        yes_price = yes_agg.current_price if yes_agg else 0.0

                    if not (
                        settings.strategy.min_tradeable_price
                        < yes_price
                        < settings.strategy.max_tradeable_price
                    ):
                        continue

                    # L2 book reliability gate (retrigger path)
                    l2_yes = self._l2_books.get(m.yes_token_id)
                    if l2_yes is not None and not l2_yes.is_reliable:
                        log.info(
                            "l2_book_unreliable",
                            asset_id=m.yes_token_id,
                            seq_gap_rate=round(l2_yes.seq_gap_rate, 4),
                            delta_count=l2_yes.delta_count,
                            path="retrigger",
                        )
                        continue

                    # Deliverable D: Data freshness gate (retrigger path)
                    # Use last_trade_time for accurate freshness.
                    max_age = settings.strategy.rpe_max_data_age_seconds
                    if yes_agg:
                        if yes_agg.last_trade_time <= 0:
                            continue
                        data_age = time.time() - yes_agg.last_trade_time
                        if data_age > max_age:
                            log.info(
                                "rpe_stale_data",
                                market=m.condition_id,
                                data_age_s=round(data_age, 1),
                                max_age_s=max_age,
                                path="retrigger",
                            )
                            continue

                    days = 30
                    total_dur = 90.0
                    if m.end_date:
                        days = max(1, (m.end_date - datetime.now(timezone.utc)).days)
                        total_dur = max(float(days), 90.0)

                    # Thread L2 data for retrigger path
                    no_book_rt = self._book_trackers.get(m.no_token_id)
                    rt_ratio = None
                    rt_l2_ok = False
                    if no_book_rt and no_book_rt.has_data:
                        rt_ratio = no_book_rt.book_depth_ratio
                        rt_l2_ok = no_book_rt.is_reliable

                    rpe_signal = self._rpe.evaluate(
                        market=m,
                        market_price=yes_price,
                        days_to_resolution=days,
                        total_duration_days=total_dur,
                        book_depth_ratio=rt_ratio,
                        l2_reliable=rt_l2_ok,
                    )
                    if rpe_signal:
                        await self._on_rpe_signal(rpe_signal, m, days, current_price=yes_price)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("rpe_retrigger_error", exc_info=True)

    def _check_paper_fills(self, event: TradeEvent) -> None:
        """Check if any paper orders should fill based on this trade."""
        filled_orders = self.executor.check_paper_fill(event.asset_id, event.price)
        for order in filled_orders:
            # Find the position that owns this order
            for pos in self.positions.get_open_positions():
                if pos.entry_order and pos.entry_order.order_id == order.order_id:
                    # Entry filled — schedule exit placement
                    _safe_fire_and_forget(
                        self._handle_entry_fill(pos),
                        name=f"entry_fill_{pos.id}",
                    )
                elif pos.exit_order and pos.exit_order.order_id == order.order_id:
                    # Exit filled — close position
                    self._handle_exit_fill(pos)

    async def _handle_entry_fill(self, pos: Any) -> None:
        """Entry order filled — compute target and place exit.

        Exception-guarded: failures are logged but do not propagate
        (the task was launched via fire-and-forget).
        """
        try:
            await self.positions.on_entry_filled(pos)
            await self.telegram.notify_entry(
                pos.id,
                pos.market_id,
                pos.entry_price,
                pos.entry_size,
                pos.target_price,
            )
            # Launch exit chaser (Pillar 1)
            # Use trade_asset_id to route to the correct book —
            # panic positions trade NO, RPE positions may trade YES or NO.
            exit_asset = pos.trade_asset_id or pos.no_asset_id
            exit_book = self._book_trackers.get(exit_asset)
            if exit_book and exit_book.has_data:
                exit_task = asyncio.create_task(
                    self._exit_chaser_flow(pos, exit_book),
                    name=f"chaser_exit_{pos.id}",
                )
                exit_task.add_done_callback(_safe_task_done_callback)
                pos.exit_chaser_task = exit_task
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("handle_entry_fill_error", pos_id=pos.id, exc_info=True)

    def _handle_exit_fill(self, pos: Any) -> None:
        """Exit order filled — close position and record."""
        self.positions.on_exit_filled(pos, reason="target")
        # Refresh lifecycle cooldown from close time (not signal fire time)
        self.lifecycle.record_signal(pos.market_id)
        _safe_fire_and_forget(
            self._record_and_notify_exit(pos),
            name=f"record_exit_{pos.id}",
        )

    async def _record_and_notify_exit(self, pos: Any) -> None:
        try:
            await self.trade_store.record(pos)
            self.positions._invalidate_stats_cache()
            await self.telegram.notify_exit(
                pos.id, pos.entry_price, pos.exit_price, pos.pnl_cents, pos.exit_reason
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("record_notify_exit_error", pos_id=pos.id, exc_info=True)

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 1 — Passive-aggressive chaser flows
    # ═══════════════════════════════════════════════════════════════════════
    async def _entry_chaser_flow(
        self, pos: Any, no_book: OrderbookTracker, *, urgent: bool = False,
    ) -> None:
        """Run the entry-side chaser for a position.

        If the chaser fills, triggers the entry-fill handler.
        If abandoned, cancels the position.

        Parameters
        ----------
        urgent:
            When True the chaser escalates immediately on the first
            POST_ONLY rejection, capturing fast-decaying panic edges
            by accepting a taker fill.
        """
        # Resolve the correct asset_id — RPE positions may trade YES.
        entry_asset = pos.trade_asset_id or pos.no_asset_id
        try:
            chaser = OrderChaser(
                executor=self.executor,
                book=no_book,
                market_id=pos.market_id,
                asset_id=entry_asset,
                side=OrderSide.BUY,
                target_size=pos.entry_size,
                anchor_price=pos.entry_price,
                latency_guard=self.latency_guard,
                tp_target_price=pos.target_price,
                fee_rate_bps=pos.entry_fee_bps,
                fast_kill_event=self._fast_kill_event,
                urgent=urgent,
                iceberg_detector=self._iceberg_detectors.get(entry_asset),
                adverse_monitor=self._maker_monitor,
            )
            result = await chaser.run()

            if chaser.state == ChaserState.FILLED and result:
                # Guard: don't resurrect a position that was cancelled
                # while the chaser was running (e.g. ghost liquidity eviction)
                if pos.state == PositionState.CANCELLED:
                    log.warning(
                        "entry_chaser_filled_but_cancelled",
                        pos_id=pos.id,
                    )
                else:
                    pos.entry_order = result
                    # Propagate actual fill quantity from chaser
                    pos.filled_size = chaser.filled_size
                    await self._handle_entry_fill(pos)
            elif chaser.state == ChaserState.ABANDONED:
                # Cancel the original order placed by PositionManager
                if pos.entry_order:
                    await self.executor.cancel_order(pos.entry_order)
                pos.state = PositionState.CANCELLED
                pos.exit_reason = "chaser_abandoned"
                log.info("entry_chaser_abandoned", pos_id=pos.id)
        except asyncio.CancelledError:
            log.info("entry_chaser_cancelled", pos_id=pos.id)
        except Exception as exc:
            log.error("entry_chaser_error", pos_id=pos.id, error=str(exc))

    async def _exit_chaser_flow(self, pos: Any, no_book: OrderbookTracker) -> None:
        """Run the exit-side chaser for a position.

        If the chaser fills, triggers the exit-fill handler.
        If abandoned, logs but does NOT force-close — the timeout loop
        will handle it via an IOC market sell.
        """
        # Resolve the correct asset_id — RPE positions may trade YES.
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        try:
            chaser = OrderChaser(
                executor=self.executor,
                book=no_book,
                market_id=pos.market_id,
                asset_id=exit_asset,
                side=OrderSide.SELL,
                target_size=pos.effective_size,
                anchor_price=pos.target_price,
                latency_guard=self.latency_guard,
                fee_rate_bps=pos.exit_fee_bps,
                fast_kill_event=self._fast_kill_event,
                iceberg_detector=self._iceberg_detectors.get(exit_asset),
                adverse_monitor=self._maker_monitor,
            )
            result = await chaser.run()

            if chaser.state == ChaserState.FILLED and result:
                pos.exit_order = result
                self._handle_exit_fill(pos)
            elif chaser.state == ChaserState.ABANDONED:
                log.info(
                    "exit_chaser_abandoned",
                    pos_id=pos.id,
                    target=pos.target_price,
                )
                # Let timeout_loop handle the force-exit
        except asyncio.CancelledError:
            log.info("exit_chaser_cancelled", pos_id=pos.id)
        except Exception as exc:
            log.error("exit_chaser_error", pos_id=pos.id, error=str(exc))

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 3 — Adaptive TP rescaling loop
    # ═══════════════════════════════════════════════════════════════════════
    async def _tp_rescale_loop(self) -> None:
        """Periodically re-evaluate take-profit targets for open positions.

        Runs every ``tp_rescale_interval_s`` seconds.  For each position
        in EXIT_PENDING state, recomputes the target using current σ₃₀
        and VWAP.  If the new target diverges by more than 0.5¢ from the
        current resting exit order, cancels and requotes.
        """
        interval = settings.strategy.tp_rescale_interval_s

        while self._running:
            await asyncio.sleep(interval)
            try:
                for pos in self.positions.get_open_positions():
                    if pos.state != PositionState.EXIT_PENDING:
                        continue
                    if not pos.exit_order:
                        continue

                    asset_id = pos.trade_asset_id or pos.no_asset_id
                    no_agg = self._no_aggs.get(asset_id)
                    if not no_agg or no_agg.rolling_vwap <= 0:
                        continue

                    # Current 30-min volatility
                    sigma_30 = no_agg.rolling_volatility_30m

                    # Dynamic spread floor
                    dynamic_spread = compute_dynamic_spread(sigma_30)

                    # Recompute take-profit with fresh market state
                    no_book = self._book_trackers.get(asset_id)
                    depth_ratio = 1.0
                    if no_book and no_book.has_data:
                        depth_ratio = no_book.book_depth_ratio

                    whale = self.whale_monitor.has_confluence(asset_id)

                    new_tp = compute_take_profit(
                        entry_price=pos.entry_price,
                        no_vwap=no_agg.rolling_vwap,
                        realised_vol=sigma_30,
                        book_depth_ratio=depth_ratio,
                        whale_confluence=whale,
                        entry_fee_bps=pos.entry_fee_bps,
                        exit_fee_bps=pos.exit_fee_bps,
                        desired_margin_cents=settings.strategy.desired_margin_cents,
                    )

                    # Tighten if calm: if current BBO is already profitable
                    # and σ is low, lock the scalp
                    current_best_bid = 0.0
                    tp_book = self._book_trackers.get(asset_id)
                    if tp_book and tp_book.has_data:
                        snap = tp_book.snapshot()
                        current_best_bid = snap.best_bid

                    new_target = new_tp.target_price
                    if (
                        dynamic_spread < settings.strategy.min_spread_cents
                        and current_best_bid > pos.entry_price
                    ):
                        scalp_target = max(
                            current_best_bid - 0.01,
                            pos.entry_price + dynamic_spread / 100.0,
                        )
                        new_target = min(new_target, scalp_target)

                    # Anti-ratchet guard: never move TP FURTHER from entry
                    # than the original target.  Rescaling should only
                    # tighten (lock profits) or track mean-reversion,
                    # not chase a moving VWAP upward indefinitely.
                    original_target = pos.tp_result.target_price if pos.tp_result else pos.target_price
                    if new_target > original_target:
                        new_target = original_target

                    # Only requote if target moved by more than 0.5¢
                    delta_cents = abs(new_target - pos.target_price) * 100
                    if delta_cents < 0.5:
                        continue

                    old_target = pos.target_price
                    pos.target_price = round(new_target, 4)
                    pos.tp_result = new_tp

                    # Cancel exit chaser — it will be restarted with new target
                    if pos.exit_chaser_task and not pos.exit_chaser_task.done():
                        pos.exit_chaser_task.cancel()

                    # Cancel resting exit order
                    if pos.exit_order and pos.exit_order.status in (
                        OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED,
                    ):
                        await self.executor.cancel_order(pos.exit_order)

                    # Place new exit order (or restart chaser)
                    exit_book = self._book_trackers.get(asset_id)
                    if exit_book and exit_book.has_data:
                        exit_task = asyncio.create_task(
                            self._exit_chaser_flow(pos, exit_book),
                            name=f"chaser_exit_rescale_{pos.id}",
                        )
                        exit_task.add_done_callback(_safe_task_done_callback)
                        pos.exit_chaser_task = exit_task
                    else:
                        exit_order = await self.executor.place_limit_order(
                            market_id=pos.market_id,
                            asset_id=asset_id,
                            side=OrderSide.SELL,
                            price=pos.target_price,
                            size=pos.effective_size,
                            fee_rate_bps=pos.exit_fee_bps,
                        )
                        pos.exit_order = exit_order

                    log.info(
                        "tp_rescaled",
                        pos_id=pos.id,
                        old_target=old_target,
                        new_target=pos.target_price,
                        sigma_30=round(sigma_30, 5),
                        dynamic_spread=dynamic_spread,
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("tp_rescale_error", error=str(exc))

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 9 — Ghost Liquidity Circuit Breaker
    # ═══════════════════════════════════════════════════════════════════════
    def _trade_volume_in_window(self, condition_id: str, window_s: float) -> float:
        """Sum of trade volume (USD) for *condition_id* in the last *window_s* seconds."""
        entries = self._recent_trade_volume.get(condition_id, [])
        if not entries:
            return 0.0
        cutoff = time.time() - window_s
        return sum(vol for ts, vol in entries if ts >= cutoff)

    def _prune_trade_volume(self) -> None:
        """Remove trade-volume entries older than 5 seconds."""
        cutoff = time.time() - 5.0
        for cid in list(self._recent_trade_volume):
            self._recent_trade_volume[cid] = [
                (ts, vol) for ts, vol in self._recent_trade_volume[cid] if ts >= cutoff
            ]
            if not self._recent_trade_volume[cid]:
                del self._recent_trade_volume[cid]

    async def _ghost_liquidity_loop(self) -> None:
        """Fast loop (500ms): detect ghost liquidity and trigger emergency drain.

        Ghost = depth drops > 50% in < 2s WITHOUT matching trade volume.

        Safety guards to prevent false positives:
        - Require book in SYNCED state (no resync artifacts)
        - Require minimum absolute depth ($50) before evaluating
        - Per-market cooldown of 60s between ghost triggers
        - Depth velocity requires >=4 history samples
        """
        interval_s = settings.strategy.ghost_check_interval_ms / 1000.0
        ghost_window = settings.strategy.ghost_window_s
        ghost_threshold = -settings.strategy.ghost_depth_drop_threshold  # e.g. -0.50
        # Minimum absolute depth to evaluate ghost detection (very thin
        # books naturally fluctuate dramatically)
        min_depth_for_ghost = 50.0  # $50 USD
        ghost_cooldown_s = 60.0  # seconds between ghost triggers per market
        ghost_cooldowns: dict[str, float] = {}  # cid -> last trigger time

        while self._running:
            try:
                self._prune_trade_volume()

                # Check active markets for ghost liquidity
                for cid in list(self.lifecycle.active):
                    am = self.lifecycle.active.get(cid)
                    if not am:
                        continue

                    # Per-market cooldown
                    last_ghost = ghost_cooldowns.get(cid, 0.0)
                    if time.time() - last_ghost < ghost_cooldown_s:
                        continue

                    # Check both YES and NO book trackers
                    for token_id in (am.info.yes_token_id, am.info.no_token_id):
                        tracker = self._book_trackers.get(token_id)
                        if not tracker or not tracker.has_data:
                            continue

                        # Require minimum absolute depth to avoid false
                        # positives on naturally thin books
                        current_depth = tracker.current_total_depth()
                        if current_depth < min_depth_for_ghost:
                            continue

                        dv = tracker.depth_velocity(ghost_window)
                        if dv is None:
                            continue

                        if dv <= ghost_threshold:
                            # Check if there were trades to explain the depth drop
                            trade_vol = self._trade_volume_in_window(cid, ghost_window)
                            if trade_vol > 0:
                                continue  # legitimate consumption, not ghost

                            # GHOST DETECTED — emergency drain
                            baseline = tracker.current_total_depth()
                            # Use pre-drop depth approximation
                            if dv < -0.99:
                                baseline_pre = baseline * 100.0  # near-total wipe
                            else:
                                baseline_pre = baseline / (1.0 + dv)

                            ghost_cooldowns[cid] = time.time()
                            self.lifecycle.emergency_drain(cid, baseline_pre)

                            # Force-exit any open positions in this market
                            for pos in self.positions.get_open_positions():
                                if pos.market_id == cid:
                                    if pos.state == PositionState.EXIT_PENDING:
                                        await self.positions.force_stop_loss(pos)
                                    elif pos.state == PositionState.ENTRY_PENDING:
                                        if pos.entry_order:
                                            await self.executor.cancel_order(pos.entry_order)
                                        pos.state = PositionState.CANCELLED
                                        pos.exit_reason = "ghost_liquidity"

                            log.warning(
                                "ghost_liquidity_detected",
                                condition_id=cid,
                                token_id=token_id,
                                depth_velocity=round(dv, 3),
                                baseline_depth=round(baseline_pre, 2),
                                l2_state=(
                                    self._l2_books[token_id].state.value
                                    if token_id in self._l2_books
                                    else "n/a"
                                ),
                            )
                            await self.telegram.send(
                                f"👻 <b>Ghost Liquidity</b> detected!\n"
                                f"Market: <code>{cid[:16]}</code>\n"
                                f"Depth drop: {abs(dv)*100:.0f}% in {ghost_window}s\n"
                                f"Emergency drain activated."
                            )
                            break  # only need one token to trigger per market

                # Check emergency markets for recovery
                recovered = self.lifecycle.check_emergency_recovery(self._book_trackers)
                for cid in recovered:
                    log.info("ghost_liquidity_recovered", condition_id=cid)
                    await self.telegram.send(
                        f"✅ <b>Ghost recovered</b>: <code>{cid[:16]}</code> back to ACTIVE"
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("ghost_liquidity_error", error=str(exc))
            await asyncio.sleep(interval_s)

    # ═══════════════════════════════════════════════════════════════════════
    #  Background loops
    # ═══════════════════════════════════════════════════════════════════════
    async def _timeout_loop(self) -> None:
        """Every 10 seconds: check timeouts and record closed positions.

        Stop-loss checks have been moved to :class:`StopLossMonitor`
        for faster (500ms) reaction time.
        """
        while self._running:
            try:
                await self.positions.check_timeouts()

                # Record timeouts (only once — mark as recorded)
                for pos in self.positions.get_all_positions():
                    if (
                        pos.state == PositionState.CLOSED
                        and pos.exit_reason == "timeout"
                        and not getattr(pos, "_recorded", False)
                    ):
                        await self.trade_store.record(pos)
                        self.positions._invalidate_stats_cache()
                        await self.telegram.notify_exit(
                            pos.id, pos.entry_price, pos.exit_price,
                            pos.pnl_cents, pos.exit_reason,
                        )
                        pos._recorded = True  # type: ignore[attr-defined]
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("timeout_loop_error", error=str(exc))
            await asyncio.sleep(10)

    async def _stale_bar_flush_loop(self) -> None:
        """Every 30 seconds, flush stale OHLCV bars on all aggregators.

        On low-volume markets, minutes can pass without a single trade.
        Without this loop, bars only close when the next trade arrives,
        causing rolling VWAP and σ to freeze — which in turn makes the
        panic detector and RPE evaluate against stale statistics.

        This loop calls ``flush_stale_bar()`` on every aggregator,
        closing any bar that has been open longer than BAR_INTERVAL
        even if no new trade has arrived.
        """
        while self._running:
            await asyncio.sleep(30)
            try:
                now = time.time()
                for yes_agg in self._yes_aggs.values():
                    bar = yes_agg.flush_stale_bar(now)
                    if bar:
                        # Drive the same signal evaluation as a normal bar close
                        asset_id = yes_agg.asset_id
                        latency_state = self.latency_guard.check(now)
                        if latency_state != LatencyState.BLOCKED:
                            await self._on_yes_bar_closed(asset_id, bar)
                for no_agg in self._no_aggs.values():
                    no_agg.flush_stale_bar(now)

                # SI-3: Cross-market divergence scan after all bars updated
                # In multicore mode, PCE worker handles scanning autonomously
                if not self._multicore_enabled and self._cross_market is not None:
                    cm_signals = self._cross_market.scan()
                    if cm_signals:
                        log.info("cross_market_scan", signals=len(cm_signals))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("stale_bar_flush_error", error=str(exc))

    async def _stats_loop(self) -> None:
        """Every 15 minutes, log and broadcast aggregate stats."""
        while self._running:
            await asyncio.sleep(900)
            try:
                stats = await self.trade_store.get_stats()
                log.info("periodic_stats", **stats)
                await self.telegram.notify_stats(stats)

                # Log market scores
                for am in self.lifecycle.active.values():
                    bd = am.score
                    log.info(
                        "market_scored",
                        question=am.info.question[:50],
                        **bd.as_dict(),
                    )

                if self.paper_mode:
                    ready, _ = await self.trade_store.passes_go_live_criteria()
                    if ready:
                        await self.telegram.send(
                            "🟢 <b>Paper-trading criteria MET</b> — ready for live deployment!"
                        )

                # ── RPE calibration gate check ────────────────────────
                cal_stats = self._rpe_calibration.calibration_summary()
                resolved_n = cal_stats.get("resolved", 0)

                if settings.strategy.rpe_generic_enabled and resolved_n < 30:
                    log.warning(
                        "rpe_generic_insufficient_calibration",
                        resolved=resolved_n,
                        required=30,
                    )

                if (
                    self.paper_mode
                    and resolved_n >= 30
                    and cal_stats.get("direction_accuracy", 0) >= 0.55
                ):
                    await self.telegram.send(
                        "🎯 <b>Generic RPE calibrated</b>\n"
                        f"Direction accuracy: {cal_stats['direction_accuracy']:.1%} "
                        f"on {resolved_n} resolved signals.\n"
                        "Consider enabling <code>RPE_GENERIC_ENABLED=true</code>."
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("stats_loop_error", error=str(exc))

    async def _paper_summary_loop(self) -> None:
        """Every 30 minutes, send a formatted paper trade summary to Telegram.

        Gives the operator immediate visibility into paper performance
        without having to inspect the DB or raw log entries.
        """
        interval = 1800  # 30 minutes
        while self._running:
            await asyncio.sleep(interval)
            try:
                stats = await self.trade_store.get_stats()
                uptime_h = (time.monotonic() - self._start_time) / 3600.0
                await self.telegram.notify_paper_summary(stats, uptime_h)
                log.info(
                    "paper_summary_sent",
                    total_trades=stats.get("total_trades", 0),
                    uptime_h=round(uptime_h, 1),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("paper_summary_error", error=str(exc))

    async def _health_reporter(self) -> None:
        """Every 60 seconds, write ``system_health.json`` to the log dir.

        Tracks memory usage, WebSocket reconnect counts, SQLite lock
        retries, latency guard state, and heartbeat status — critical
        telemetry for the long-running data-harvesting soak test.
        """
        import json as _json
        import os as _os
        import tempfile
        from pathlib import Path as _Path

        try:
            health_dir = _Path(settings.log_dir)
            health_dir.mkdir(parents=True, exist_ok=True)
            health_path = health_dir / "system_health.json"
        except Exception:
            log.warning("health_reporter_dir_error", exc_info=True)
            health_dir = _Path(tempfile.gettempdir())
            health_path = health_dir / "system_health.json"

        # Try to import psutil for memory tracking; degrade gracefully.
        try:
            import psutil  # type: ignore[import-untyped]
            _process = psutil.Process(_os.getpid())
        except (ImportError, Exception):
            _process = None

        while self._running:
            await asyncio.sleep(60)
            try:
                ws_reconnects = (
                    getattr(self._ws, "reconnect_count", 0)
                    if self._ws else 0
                )
                l2_reconnects = (
                    getattr(self._l2_ws, "reconnect_count", 0)
                    if self._l2_ws else 0
                )
                memory_bytes = (
                    (await asyncio.to_thread(_process.memory_info)).rss
                    if _process else None
                )
                uptime_s = round(time.monotonic() - self._start_time, 1)

                heartbeat_state = "unknown"
                if hasattr(self, "_heartbeat"):
                    heartbeat_state = (
                        "suspended" if self._heartbeat.is_suspended else "alive"
                    )
                latency_state = self.latency_guard.state.value

                health = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "deployment_env": self.deployment_env.value,
                    "uptime_s": uptime_s,
                    "memory_bytes": memory_bytes,
                    "ws_reconnects": ws_reconnects,
                    "l2_ws_reconnects": l2_reconnects,
                    "db_lock_retries": self.trade_store.db_lock_retries,
                    "latency_guard_state": latency_state,
                    "heartbeat_state": heartbeat_state,
                    "active_positions": len(self.positions.get_open_positions()),
                    "active_markets": len(self.lifecycle.active),
                    "observing_markets": len(self.lifecycle.observing),
                }

                # Aggregate L2 book health metrics across all tracked books
                l2_total_deltas = 0
                l2_total_desyncs = 0
                l2_synced_count = 0
                for book in self._l2_books.values():
                    l2_total_deltas += book.delta_count
                    l2_total_desyncs += book.desync_total
                    if book.state == BookState.SYNCED:
                        l2_synced_count += 1

                health["l2_total_deltas"] = l2_total_deltas
                health["l2_total_desyncs"] = l2_total_desyncs
                health["l2_synced_books"] = l2_synced_count
                health["l2_total_books"] = len(self._l2_books)
                health["l2_seq_gap_rate"] = round(
                    l2_total_desyncs / max(1, l2_total_deltas), 6
                )
                health["l2_unreliable_books"] = sum(
                    1 for book in self._l2_books.values()
                    if not book.is_reliable
                )

                # Atomic write: offload blocking file I/O to a thread
                await asyncio.to_thread(
                    self._write_health_sync, health, health_dir, health_path,
                )

                log.debug("health_report_written", path=str(health_path))

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("health_reporter_error", error=str(exc))

    @staticmethod
    def _write_health_sync(health: dict, health_dir: Any, health_path: Any) -> None:
        """Synchronous atomic write — called via ``asyncio.to_thread``."""
        import json as _json
        import os as _os
        import tempfile

        fd, tmp_path = tempfile.mkstemp(dir=str(health_dir), suffix=".tmp")
        try:
            with _os.fdopen(fd, "w") as f:
                _json.dump(health, f, indent=2)
            _os.replace(tmp_path, str(health_path))
        except Exception:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            raise

    async def _market_refresh_loop(self) -> None:
        """Periodically re-discover, re-score, promote/demote/evict."""
        interval = settings.strategy.market_refresh_minutes * 60
        while self._running:
            await asyncio.sleep(interval)
            try:
                # Decay trade frequency counts (approximate trades/min)
                decay_mins = settings.strategy.market_refresh_minutes
                for aid in self._trade_counts:
                    self._trade_counts[aid] = self._trade_counts[aid] / max(decay_mins, 1)

                # Decay taker/total counts (halve each refresh cycle)
                for aid in list(self._taker_counts):
                    self._taker_counts[aid] = max(0, self._taker_counts[aid] // 2)
                for aid in list(self._total_counts):
                    self._total_counts[aid] = max(0, self._total_counts[aid] // 2)

                open_markets = self.positions.get_open_market_ids()
                whale_tokens = self.whale_monitor.get_whale_tokens()

                newly_added, evicted = await self.lifecycle.refresh(
                    orderbook_trackers=self._book_trackers,
                    trade_counts=self._trade_counts,
                    whale_tokens=whale_tokens,
                    open_position_markets=open_markets,
                    taker_counts=self._taker_counts,
                    total_counts=self._total_counts,
                )

                # Evict markets
                for cid in evicted:
                    for m in list(self._markets):
                        if m.condition_id == cid:
                            if self._ws:
                                await self._ws.remove_assets(
                                    [m.yes_token_id, m.no_token_id]
                                )
                            if self._l2_ws:
                                await self._l2_ws.remove_assets(
                                    [m.yes_token_id, m.no_token_id]
                                )
                            self._unwire_market(m)
                            break

                # Wire + subscribe new markets
                new_asset_ids: list[str] = []
                new_l2_books: dict[str, L2OrderBook] = {}
                for m in newly_added:
                    self._wire_market(m)
                    new_asset_ids.extend([m.yes_token_id, m.no_token_id])
                    # Collect L2 books for dynamic subscription
                    for token_id in (m.yes_token_id, m.no_token_id):
                        if token_id in self._l2_books:
                            new_l2_books[token_id] = self._l2_books[token_id]

                if self._ws and new_asset_ids:
                    await self._ws.add_assets(new_asset_ids)
                if self._l2_ws and new_l2_books:
                    await self._l2_ws.add_assets(new_l2_books)

                # ── Stale trade eviction ──────────────────────────────
                stale_evicted = self.lifecycle.check_stale_markets(
                    yes_aggs=self._yes_aggs,
                    open_position_markets=open_markets,
                    stale_threshold_s=settings.strategy.stale_market_eviction_s,
                )
                for cid in stale_evicted:
                    for m in list(self._markets):
                        if m.condition_id == cid:
                            if self._ws:
                                await self._ws.remove_assets(
                                    [m.yes_token_id, m.no_token_id]
                                )
                            if self._l2_ws:
                                await self._l2_ws.remove_assets(
                                    [m.yes_token_id, m.no_token_id]
                                )
                            self._unwire_market(m)
                            break

                # ── Per-market health log ─────────────────────────────
                for cid, am in self.lifecycle.active.items():
                    tid = am.info.yes_token_id
                    agg = self._yes_aggs.get(tid)
                    bt = self._book_trackers.get(tid)
                    tpm = self._trade_counts.get(tid, 0.0)
                    last_trade_age = (
                        round(time.time() - agg.last_trade_time, 1)
                        if agg and agg.last_trade_time > 0
                        else -1
                    )
                    bars = len(agg.bars) if agg else 0
                    spread = bt.spread_cents if bt and bt.has_data else -1
                    log.info(
                        "market_health",
                        condition_id=cid,
                        score=round(am.score.total, 1),
                        trades_per_min=round(tpm, 2),
                        last_trade_age_s=last_trade_age,
                        bars=bars,
                        spread_cents=spread,
                    )

                # Update _markets list with promoted markets
                for cid, am in self.lifecycle.active.items():
                    if am.info not in self._markets:
                        self._markets.append(am.info)

                total = len(self.lifecycle.active)
                obs = len(self.lifecycle.observing)
                drn = len(self.lifecycle.draining)

                if newly_added or evicted:
                    log.info(
                        "market_refresh_done",
                        added=len(newly_added),
                        evicted=len(evicted),
                        active=total,
                        observing=obs,
                        draining=drn,
                    )
                    await self.telegram.send(
                        f"🔄 <b>Market refresh</b>\n"
                        f"Active: {total}  |  Observing: {obs}  |  Draining: {drn}\n"
                        f"+{len(newly_added)} new  |  -{len(evicted)} evicted"
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("market_refresh_error", error=str(exc))

    async def _pce_refresh_loop(self) -> None:
        """Periodically recompute correlations, persist, and send dashboard."""
        interval = settings.strategy.pce_correlation_refresh_minutes * 60
        cycle_count = 0
        while self._running:
            await asyncio.sleep(interval)
            try:
                cycle_count += 1
                await asyncio.to_thread(self.pce.refresh_correlations)
                await asyncio.to_thread(self.pce.save_state)

                # Every 12th cycle (~6 hours) validate structural priors
                if cycle_count % 12 == 0:
                    prior_summary = await asyncio.to_thread(
                        self.pce.validate_structural_priors
                    )
                    log.info("pce_prior_validation", **prior_summary)

                # Send Telegram dashboard
                open_positions = self.positions.get_open_positions()
                dashboard = self.pce.get_dashboard_data(open_positions)
                await self.telegram.notify_pce_dashboard(dashboard)

                log.info(
                    "pce_dashboard_sent",
                    portfolio_var=dashboard.get("portfolio_var", 0.0),
                    pairs_tracked=dashboard.get("total_pairs_tracked", 0),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("pce_refresh_error", error=str(exc))

    async def _cleanup_loop(self) -> None:
        """Every 5 minutes: clean up stale positions, orders, reset daily PnL,
        and checkpoint live state for crash recovery."""
        while self._running:
            await asyncio.sleep(300)
            try:
                removed_pos = self.positions.cleanup_closed()
                if removed_pos:
                    log.info("positions_cleaned", count=len(removed_pos))

                removed_orders = self.executor.cleanup_old_orders()
                if removed_orders:
                    log.info("orders_cleaned", count=removed_orders)

                # Check for midnight UTC → reset daily PnL (exactly once)
                now = datetime.now(timezone.utc)
                today = now.date()
                if not hasattr(self, "_last_pnl_reset_date"):
                    self._last_pnl_reset_date = today
                if today != self._last_pnl_reset_date:
                    self.positions.reset_daily_pnl()
                    self._last_pnl_reset_date = today

                # Checkpoint live state for crash recovery
                await self.trade_store.checkpoint_orders(
                    self.executor.get_open_orders()
                )
                await self.trade_store.checkpoint_positions(
                    self.positions.get_open_positions()
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("cleanup_loop_error", error=str(exc))