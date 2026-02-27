"""
Main bot orchestrator — wires the data pipeline, signal detection,
order execution, and monitoring into a single async event loop.
"""

from __future__ import annotations

import asyncio
import signal
import time
from typing import Any

from src.core.config import settings
from src.core.logger import get_logger, setup_logging
from src.data.market_discovery import MarketInfo, fetch_active_markets
from src.data.ohlcv import OHLCVAggregator
from src.data.websocket_client import MarketWebSocket, TradeEvent
from src.monitoring.telegram import TelegramAlerter
from src.monitoring.trade_store import TradeStore
from src.signals.panic_detector import PanicDetector, PanicSignal
from src.signals.whale_monitor import WhaleMonitor
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import PositionManager, PositionState

log = get_logger(__name__)


class TradingBot:
    """Top-level orchestrator for the mean-reversion market maker."""

    def __init__(self, paper_mode: bool | None = None):
        self.paper_mode = paper_mode if paper_mode is not None else settings.paper_mode

        # Components (initialised in start())
        self.executor = OrderExecutor(paper_mode=self.paper_mode)
        self.positions = PositionManager(self.executor)
        self.whale_monitor = WhaleMonitor(zscore_fn=self._latest_zscore)
        self.trade_store = TradeStore()
        self.telegram = TelegramAlerter()

        # Per-market state
        self._markets: list[MarketInfo] = []
        self._yes_aggs: dict[str, OHLCVAggregator] = {}  # keyed by yes_token_id
        self._no_aggs: dict[str, OHLCVAggregator] = {}   # keyed by no_token_id
        self._detectors: dict[str, PanicDetector] = {}    # keyed by condition_id
        self._market_map: dict[str, MarketInfo] = {}      # asset_id → MarketInfo

        # Lifecycle
        self._running = False
        self._latest_z: float = 0.0        # tracks most recent panic Z-score
        self._trade_queue: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=1000)
        self._ws: MarketWebSocket | None = None
        self._tasks: list[asyncio.Task] = []

    # ═══════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════════════
    async def start(self) -> None:
        """Discover markets, wire components, and enter the main loop."""
        setup_logging(settings.log_dir)
        mode_label = "PAPER" if self.paper_mode else "LIVE"
        log.info("bot_starting", mode=mode_label)
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
        # 1. Init trade store
        await self.trade_store.init()

        # 2. Discover eligible markets
        self._markets = await fetch_active_markets()
        if not self._markets:
            log.error("no_eligible_markets")
            await self.telegram.send(
                "⚠️ <b>Bot stopped — no eligible markets found.</b>\n"
                "No markets currently meet the configured criteria.\n"
                "Restart the bot manually when conditions change."
            )
            return

        log.info("markets_selected", count=len(self._markets))

        # 3. Build per-market aggregators & detectors
        all_asset_ids: list[str] = []
        for m in self._markets:
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

            all_asset_ids.extend([m.yes_token_id, m.no_token_id])

        # 4. Set initial wallet balance (simplified — hard-coded for PoC)
        self.positions.set_wallet_balance(
            50.0 if self.paper_mode else settings.strategy.max_trade_size_usd * 5
        )

        # 5. Launch concurrent tasks
        self._running = True
        self._ws = MarketWebSocket(all_asset_ids, self._trade_queue)

        self._tasks = [
            asyncio.create_task(self._ws.start(), name="ws"),
            asyncio.create_task(self._process_trades(), name="trade_processor"),
            asyncio.create_task(self._timeout_loop(), name="timeout_loop"),
            asyncio.create_task(self._stats_loop(), name="stats_loop"),
            asyncio.create_task(self.whale_monitor.start(), name="whale_monitor"),
            asyncio.create_task(self._market_refresh_loop(), name="market_refresh"),
        ]

        # Handle graceful shutdown via SIGINT / SIGTERM
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        log.info("bot_running", markets=len(self._markets), mode=mode_label)

        # Wait for all tasks (they run indefinitely until stop())
        try:
            await asyncio.gather(*self._tasks, return_exceptions=False)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Graceful shutdown: cancel orders, flatten, stop tasks."""
        if not self._running:
            return
        self._running = False
        log.info("bot_stopping")

        # Cancel all open orders
        cancelled = await self.executor.cancel_all()
        log.info("orders_cancelled", count=cancelled)

        # Stop WebSocket
        if self._ws:
            await self._ws.stop()

        # Stop whale monitor
        await self.whale_monitor.stop()

        # Cancel async tasks
        for task in self._tasks:
            task.cancel()

        # Persist final stats
        stats = await self.trade_store.get_stats()
        log.info("final_stats", **stats)
        await self.telegram.notify_stats(stats)
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
    #  Adaptive polling helper
    # ═══════════════════════════════════════════════════════════════════════
    def _latest_zscore(self) -> float:
        """Callback for WhaleMonitor — returns the most recent panic Z-score."""
        return self._latest_z

    # ═══════════════════════════════════════════════════════════════════════
    #  Core processing loop
    # ═══════════════════════════════════════════════════════════════════════
    async def _process_trades(self) -> None:
        """Main loop: consume TradeEvents, update aggregators, check signals."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._trade_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Route to the correct aggregator
            yes_agg = self._yes_aggs.get(event.asset_id)
            no_agg = self._no_aggs.get(event.asset_id)

            if yes_agg:
                bar = yes_agg.on_trade(event)
                if bar:
                    await self._on_yes_bar_closed(event.asset_id, bar)

            if no_agg:
                bar = no_agg.on_trade(event)
                # Also use NO trades to check paper fills
                if self.paper_mode:
                    self._check_paper_fills(event)

            # In paper mode, also check fills on YES-side trades
            if self.paper_mode and yes_agg:
                self._check_paper_fills(event)

    async def _on_yes_bar_closed(self, yes_asset_id: str, bar: Any) -> None:
        """A 1-min YES bar just closed — evaluate the panic detector."""
        market_info = self._market_map.get(yes_asset_id)
        if not market_info:
            return

        detector = self._detectors.get(market_info.condition_id)
        if not detector:
            return

        no_agg = self._no_aggs.get(market_info.no_token_id)
        if not no_agg:
            return

        # Best ask for NO — use current last traded price as proxy
        no_best_ask = no_agg.current_price
        if no_best_ask <= 0:
            return

        # Whale confluence check
        whale = self.whale_monitor.has_confluence(market_info.no_token_id)

        signal = detector.evaluate(bar, no_best_ask=no_best_ask, whale_confluence=whale)
        if signal:
            self._latest_z = signal.zscore     # feed adaptive whale poller
            await self._on_panic_signal(signal, no_agg)

    async def _on_panic_signal(self, signal: PanicSignal, no_agg: OHLCVAggregator) -> None:
        """Handle a confirmed panic signal — attempt to open a position."""
        await self.telegram.notify_signal(
            signal.market_id, signal.zscore, signal.volume_ratio
        )

        pos = await self.positions.open_position(signal, no_agg)
        if pos:
            # In paper mode, immediately check if the entry fill can simulate
            pass

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
    #  Background loops
    # ═══════════════════════════════════════════════════════════════════════
    async def _timeout_loop(self) -> None:
        """Every 10 seconds, check for stale orders/positions."""
        while self._running:
            try:
                await self.positions.check_timeouts()
                # Record any positions that were force-closed by timeout
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
            await asyncio.sleep(900)  # 15 min
            try:
                stats = await self.trade_store.get_stats()
                log.info("periodic_stats", **stats)
                await self.telegram.notify_stats(stats)

                # In paper mode, check go-live readiness
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
    async def _market_refresh_loop(self) -> None:
        """Periodically re-discover markets, drop resolved ones, add new."""
        interval = settings.strategy.market_refresh_minutes * 60
        while self._running:
            await asyncio.sleep(interval)
            try:
                fresh = await fetch_active_markets()
                if not fresh:
                    log.warning("market_refresh_no_results")
                    continue

                existing_ids = {m.condition_id for m in self._markets}
                new_markets = [m for m in fresh if m.condition_id not in existing_ids]

                if not new_markets:
                    log.info("market_refresh_no_new", total=len(self._markets))
                    continue

                # Wire up new markets
                new_asset_ids: list[str] = []
                for m in new_markets:
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
                    new_asset_ids.extend([m.yes_token_id, m.no_token_id])

                self._markets.extend(new_markets)

                # Subscribe WebSocket to new tokens
                if self._ws and new_asset_ids:
                    await self._ws.add_assets(new_asset_ids)

                log.info(
                    "market_refresh_added",
                    new=len(new_markets),
                    total=len(self._markets),
                )
                await self.telegram.send(
                    f"\U0001f504 <b>Market refresh</b>: added {len(new_markets)} new market(s), "
                    f"now watching {len(self._markets)} total."
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("market_refresh_error", error=str(exc))