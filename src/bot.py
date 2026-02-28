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
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings, DeploymentEnv
from src.core.guard import DeploymentGuard
from src.core.heartbeat import BookHeartbeat
from src.core.latency_guard import LatencyGuard, LatencyState
from src.core.logger import get_logger, setup_logging
from src.data.market_discovery import MarketInfo, fetch_active_markets
from src.data.market_lifecycle import MarketLifecycleManager
from src.data.market_scorer import compute_score
from src.data.ohlcv import OHLCVAggregator
from src.data.orderbook import L2OrderBookAdapter, OrderbookTracker
from src.data.websocket_client import MarketWebSocket, TradeEvent
from src.data.l2_book import L2OrderBook
from src.data.l2_websocket import L2WebSocket
from src.monitoring.telegram import TelegramAlerter
from src.monitoring.trade_store import TradeStore
from src.signals.adverse_selection_guard import AdverseSelectionGuard
from src.signals.panic_detector import PanicDetector, PanicSignal
from src.signals.whale_monitor import WhaleMonitor
from src.trading.chaser import ChaserState, OrderChaser
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus, OrderStatusPoller
from src.trading.position_manager import PositionManager, PositionState
from src.trading.stop_loss import StopLossMonitor
from src.trading.take_profit import compute_dynamic_spread, compute_take_profit

# Data recording (lazy import — only used when RECORD_DATA=true)
try:
    from src.backtest.data_recorder import MarketDataRecorder
except ImportError:
    MarketDataRecorder = None  # type: ignore[misc,assignment]

log = get_logger(__name__)


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
        self.positions = PositionManager(
            self.executor, trade_store=self.trade_store, guard=self.guard,
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
        self._book_trackers: dict[str, OrderbookTracker] = {}  # asset_id → tracker
        self._l2_books: dict[str, L2OrderBook] = {}  # asset_id → L2 book (when L2 enabled)
        self._trade_counts: dict[str, float] = {}  # asset_id → trades/min
        self._taker_counts: dict[str, int] = {}    # asset_id → taker-initiated trades
        self._total_counts: dict[str, int] = {}    # asset_id → total classified trades
        self._recent_trade_volume: dict[str, list[tuple[float, float]]] = {}  # cond_id → [(ts, size)]

        # Lifecycle
        self._running = False
        self._start_time: float = 0.0       # set in _run() for uptime tracking
        self._latest_z: float = 0.0        # tracks most recent panic Z-score
        self._trade_queue: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=1000)
        self._book_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._ws: MarketWebSocket | None = None
        self._l2_ws: L2WebSocket | None = None
        self._tasks: list[asyncio.Task] = []

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
        log.info(
            "bot_starting",
            mode=mode_label,
            **self.guard.startup_summary(),
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

        # 5. Launch concurrent tasks
        self._running = True
        self._ws = MarketWebSocket(
            all_asset_ids, self._trade_queue, book_queue=self._book_queue,
            recorder=self._recorder,
        )

        # L2 WebSocket (Pillar 11) — dedicated connection for L2 book data
        if settings.strategy.l2_enabled and self._l2_books:
            # Register desync callbacks for all L2 books
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
        )

        # Order status poller (Pillar 10) — live mode only
        self._order_poller = OrderStatusPoller(
            self.executor,
            on_fill=self._on_clob_fill,
        )

        # Event-driven stop-loss monitor (Pillar 11) — no polling task
        self._stop_loss_monitor = StopLossMonitor(
            position_manager=self.positions,
            no_aggs=self._no_aggs,
            book_trackers=self._book_trackers,
            trade_store=self.trade_store,
            telegram=self.telegram,
        )
        self._stop_loss_monitor.start()  # mark active, no coroutine

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
        ]

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

        # Handle graceful shutdown via SIGINT / SIGTERM
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        else:
            # On Windows, loop.add_signal_handler is not supported.
            # Register via the signal module instead (handles Ctrl-C).
            signal.signal(
                signal.SIGINT,
                lambda *_: loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.stop())
                ),
            )

        log.info("bot_running", markets=len(self._markets), mode=mode_label)

        # Wait for all tasks (they run indefinitely until stop())
        try:
            await asyncio.gather(*self._tasks, return_exceptions=False)
        except asyncio.CancelledError:
            pass

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

        # Orderbook trackers for both tokens
        if settings.strategy.l2_enabled:
            for token_id in (m.yes_token_id, m.no_token_id):
                l2_book = L2OrderBook(
                    token_id,
                    on_bbo_change=self._on_l2_bbo_change,
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
        self._book_trackers.pop(m.yes_token_id, None)
        self._book_trackers.pop(m.no_token_id, None)
        # Clean up L2 book instances
        for token_id in (m.yes_token_id, m.no_token_id):
            l2_book = self._l2_books.pop(token_id, None)
            if l2_book is not None:
                l2_book.reset()
        self._trade_counts.pop(m.yes_token_id, None)
        self._trade_counts.pop(m.no_token_id, None)
        self._markets = [x for x in self._markets if x.condition_id != m.condition_id]

    def _positioned_asset_ids(self) -> set[str]:
        """Return asset IDs that currently have open positions.

        Used by the heartbeat to decide which books need fresh data.
        """
        return {
            pos.no_asset_id
            for pos in self.positions.get_open_positions()
        }

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

        for task in self._tasks:
            task.cancel()

        stats = await self.trade_store.get_stats()
        log.info("final_stats", **stats)
        await self.telegram.notify_stats(stats)
        await self.trade_store.clear_live_state()
        await self.trade_store.close()

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
            if event_type == "price_change":
                tracker.on_price_change(msg)
            elif event_type == "book":
                tracker.on_book_snapshot(msg)

    # ── L2 callbacks ──────────────────────────────────────────────────────
    async def _on_l2_bbo_change(self, asset_id: str, score: Any) -> None:
        """Callback from L2OrderBook when the BBO changes.

        Logs the live spread score for monitoring and drives the
        event-driven stop-loss engine.  This fires on every BBO tick
        so keep it lightweight.
        """
        log.debug(
            "l2_bbo_update",
            asset_id=asset_id,
            spread_score=round(score.score, 1),
            raw_spread_cents=round(score.raw_spread_cents, 2),
        )
        # Drive event-driven stop-loss evaluation
        await self._stop_loss_monitor.on_bbo_update(asset_id)

    def _on_orderbook_bbo_change(self, asset_id: str, snapshot: Any) -> None:
        """Callback from basic OrderbookTracker when the BBO changes.

        Schedules an async stop-loss evaluation for this asset.
        """
        asyncio.ensure_future(self._stop_loss_monitor.on_bbo_update(asset_id))

    async def _on_l2_desync(self, asset_id: str) -> None:
        """Callback from L2OrderBook on sequence gap detection.

        Routes to the L2WebSocket for snapshot re-fetch.
        """
        if self._l2_ws is not None:
            await self._l2_ws._on_book_desync(asset_id)

    async def _on_yes_bar_closed(self, yes_asset_id: str, bar: Any) -> None:
        """A 1-min YES bar just closed — evaluate the panic detector."""
        market_info = self._market_map.get(yes_asset_id)
        if not market_info:
            return

        # Only trade active-tier markets
        if not self.lifecycle.is_tradeable(market_info.condition_id):
            return

        # Signal cooldown check
        if not self.lifecycle.is_cooled_down(market_info.condition_id):
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

        # Whale confluence check
        whale = self.whale_monitor.has_confluence(market_info.no_token_id)

        sig = detector.evaluate(bar, no_best_ask=no_best_ask, whale_confluence=whale)
        if sig:
            self._latest_z = sig.zscore
            self.lifecycle.record_signal(market_info.condition_id)
            await self._on_panic_signal(sig, no_agg, market_info)

    async def _on_panic_signal(
        self, sig: PanicSignal, no_agg: OHLCVAggregator, market: MarketInfo
    ) -> None:
        """Handle a confirmed panic signal."""
        await self.telegram.notify_signal(
            sig.market_id, sig.zscore, sig.volume_ratio
        )

        # Compute days to resolution
        days = 30
        if market.end_date:
            days = max(1, (market.end_date - datetime.now(timezone.utc)).days)

        # Get real book depth ratio if available
        book_depth = 1.0
        no_book = self._book_trackers.get(market.no_token_id)
        if no_book and no_book.has_data:
            book_depth = no_book.book_depth_ratio

        # Fetch fee rates for this token
        # Determine if market is fee-enabled based on category
        fee_cats = settings.strategy.fee_enabled_categories.lower().split(",")
        market_tags = (getattr(market, 'tags', '') or '').lower()
        fee_enabled = any(cat.strip() in market_tags for cat in fee_cats) if market_tags else True

        pos = await self.positions.open_position(
            sig,
            no_agg,
            no_book=no_book,
            event_id=market.event_id,
            days_to_resolution=days,
            book_depth_ratio=book_depth,
            fee_enabled=fee_enabled,
        )
        if pos:
            # Launch entry chaser as a child task (Pillar 1)
            if no_book and no_book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, no_book),
                    name=f"chaser_entry_{pos.id}",
                )
                pos.entry_chaser_task = chaser_task
            # else: order already placed directly by PositionManager

        if pos and self.positions.circuit_breaker_active:
            await self.telegram.send(
                "🛑 <b>Circuit breaker tripped</b> — pausing new positions."
            )

    def _check_paper_fills(self, event: TradeEvent) -> None:
        """Check if any paper orders should fill based on this trade."""
        filled_orders = self.executor.check_paper_fill(event.asset_id, event.price)
        for order in filled_orders:
            # Find the position that owns this order
            for pos in self.positions.get_open_positions():
                if pos.entry_order and pos.entry_order.order_id == order.order_id:
                    # Entry filled — schedule exit placement
                    asyncio.create_task(self._handle_entry_fill(pos))
                elif pos.exit_order and pos.exit_order.order_id == order.order_id:
                    # Exit filled — close position
                    self._handle_exit_fill(pos)

    async def _handle_entry_fill(self, pos: Any) -> None:
        """Entry order filled — compute target and place exit."""
        await self.positions.on_entry_filled(pos)
        await self.telegram.notify_entry(
            pos.id,
            pos.market_id,
            pos.entry_price,
            pos.entry_size,
            pos.target_price,
        )
        # Launch exit chaser (Pillar 1)
        no_book = self._book_trackers.get(pos.no_asset_id)
        if no_book and no_book.has_data:
            exit_task = asyncio.create_task(
                self._exit_chaser_flow(pos, no_book),
                name=f"chaser_exit_{pos.id}",
            )
            pos.exit_chaser_task = exit_task

    def _handle_exit_fill(self, pos: Any) -> None:
        """Exit order filled — close position and record."""
        self.positions.on_exit_filled(pos, reason="target")
        asyncio.create_task(self._record_and_notify_exit(pos))

    async def _record_and_notify_exit(self, pos: Any) -> None:
        await self.trade_store.record(pos)
        await self.telegram.notify_exit(
            pos.id, pos.entry_price, pos.exit_price, pos.pnl_cents, pos.exit_reason
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 1 — Passive-aggressive chaser flows
    # ═══════════════════════════════════════════════════════════════════════
    async def _entry_chaser_flow(self, pos: Any, no_book: OrderbookTracker) -> None:
        """Run the entry-side chaser for a position.

        If the chaser fills, triggers the entry-fill handler.
        If abandoned, cancels the position.
        """
        try:
            chaser = OrderChaser(
                executor=self.executor,
                book=no_book,
                market_id=pos.market_id,
                asset_id=pos.no_asset_id,
                side=OrderSide.BUY,
                target_size=pos.entry_size,
                anchor_price=pos.entry_price,
                latency_guard=self.latency_guard,
                tp_target_price=pos.target_price,
                fee_rate_bps=pos.entry_fee_bps,
                fast_kill_event=self._fast_kill_event,
            )
            result = await chaser.run()

            if chaser.state == ChaserState.FILLED and result:
                pos.entry_order = result
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
        try:
            chaser = OrderChaser(
                executor=self.executor,
                book=no_book,
                market_id=pos.market_id,
                asset_id=pos.no_asset_id,
                side=OrderSide.SELL,
                target_size=pos.entry_size,
                anchor_price=pos.target_price,
                latency_guard=self.latency_guard,
                fee_rate_bps=pos.exit_fee_bps,
                fast_kill_event=self._fast_kill_event,
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

                    no_agg = self._no_aggs.get(pos.no_asset_id)
                    if not no_agg or no_agg.rolling_vwap <= 0:
                        continue

                    # Current 30-min volatility
                    sigma_30 = no_agg.rolling_volatility_30m

                    # Dynamic spread floor
                    dynamic_spread = compute_dynamic_spread(sigma_30)

                    # Recompute take-profit with fresh market state
                    no_book = self._book_trackers.get(pos.no_asset_id)
                    depth_ratio = 1.0
                    if no_book and no_book.has_data:
                        depth_ratio = no_book.book_depth_ratio

                    whale = self.whale_monitor.has_confluence(pos.no_asset_id)

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
                    if no_book and no_book.has_data:
                        snap = no_book.snapshot()
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
                    no_book = self._book_trackers.get(pos.no_asset_id)
                    if no_book and no_book.has_data:
                        exit_task = asyncio.create_task(
                            self._exit_chaser_flow(pos, no_book),
                            name=f"chaser_exit_rescale_{pos.id}",
                        )
                        pos.exit_chaser_task = exit_task
                    else:
                        exit_order = await self.executor.place_limit_order(
                            market_id=pos.market_id,
                            asset_id=pos.no_asset_id,
                            side=OrderSide.SELL,
                            price=pos.target_price,
                            size=pos.entry_size,
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
        """
        interval_s = settings.strategy.ghost_check_interval_ms / 1000.0
        ghost_window = settings.strategy.ghost_window_s
        ghost_threshold = -settings.strategy.ghost_depth_drop_threshold  # e.g. -0.50

        while self._running:
            try:
                self._prune_trade_volume()

                # Check active markets for ghost liquidity
                for cid in list(self.lifecycle.active):
                    am = self.lifecycle.active.get(cid)
                    if not am:
                        continue

                    # Check both YES and NO book trackers
                    for token_id in (am.info.yes_token_id, am.info.no_token_id):
                        tracker = self._book_trackers.get(token_id)
                        if not tracker or not tracker.has_data:
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
                                depth_velocity=round(dv, 3),
                                baseline_depth=round(baseline_pre, 2),
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

                # Record timeouts
                for pos in self.positions.get_all_positions():
                    if pos.state == PositionState.CLOSED and pos.exit_reason == "timeout":
                        await self.trade_store.record(pos)
                        await self.telegram.notify_exit(
                            pos.id, pos.entry_price, pos.exit_price,
                            pos.pnl_cents, pos.exit_reason,
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("timeout_loop_error", error=str(exc))
            await asyncio.sleep(10)

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
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("stats_loop_error", error=str(exc))

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

        health_dir = _Path(settings.log_dir)
        health_dir.mkdir(parents=True, exist_ok=True)
        health_path = health_dir / "system_health.json"

        # Try to import psutil for memory tracking; degrade gracefully.
        try:
            import psutil  # type: ignore[import-untyped]
            _process = psutil.Process(_os.getpid())
        except ImportError:
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

                # Check for midnight UTC → reset daily PnL
                now = datetime.now(timezone.utc)
                if now.hour == 0 and now.minute < 6:
                    self.positions.reset_daily_pnl()

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