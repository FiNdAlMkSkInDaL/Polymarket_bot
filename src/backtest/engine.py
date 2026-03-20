"""
BacktestEngine — top-level event-driven replay orchestrator.

Replays historical market events tick-by-tick through a pessimistic
CLOB matching engine, calling strategy hooks at each step.  Produces
a ``BacktestResult`` with full telemetry.

**Synchronous design** — no ``async`` needed during replay (no network I/O).
Simpler to reason about determinism; measurably faster than an async loop.

Example
───────
    from src.backtest import BacktestEngine, BacktestConfig, DataLoader

    config = BacktestConfig(initial_cash=1000.0, latency_ms=150.0)
    loader = DataLoader.from_directory("data/raw_ticks/2026-02-25")
    engine = BacktestEngine(strategy=my_strategy, data_loader=loader, config=config)
    result = engine.run()
    print(result.metrics.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backtest.clock import SimClock
from src.backtest.data_loader import DataLoader, MarketEvent
from src.backtest.matching_engine import Fill, MatchingEngine, SimOrder
from src.backtest.strategy import StrategyABC
from src.backtest.telemetry import BacktestMetrics, Telemetry
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator, OHLCVBar
from src.data.websocket_client import TradeEvent
from src.trading.executor import OrderSide

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the backtest engine.

    Parameters
    ----------
    initial_cash:
        Starting cash balance in USD.
    latency_ms:
        Simulated network + exchange latency in milliseconds.
    fee_max_pct:
        Maximum fee rate as a percentage (Polymarket default: 1.56%).
    fee_enabled:
        Whether to apply dynamic fee curve.
    equity_sample_interval:
        Record equity every N events (for Sharpe / drawdown).
    bar_interval_s:
        OHLCV bar interval in seconds (default: 60).
    """

    initial_cash: float = 1000.0
    latency_ms: float = 150.0
    fee_max_pct: float = 2.00
    fee_enabled: bool = True
    equity_sample_interval: int = 100
    bar_interval_s: float = 60.0


# ═══════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Output of a completed backtest run.

    Contains computed metrics, the raw trade log, and the equity curve.
    """

    metrics: BacktestMetrics
    config: BacktestConfig
    events_processed: int = 0
    final_cash: float = 0.0
    final_equity: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)
    all_fills: list[Fill] = field(default_factory=list)
    all_orders: list[SimOrder] = field(default_factory=list)

    def summary(self) -> str:
        """Formatted summary string."""
        header = (
            f"\n  Events processed: {self.events_processed:,}\n"
            f"  Final cash:       ${self.final_cash:,.4f}\n"
            f"  Final equity:     ${self.final_equity:,.4f}\n"
            f"  Open positions:   {len(self.positions)}\n"
        )
        return header + self.metrics.summary()


# ═══════════════════════════════════════════════════════════════════════════
#  Engine
# ═══════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Event-driven backtest engine with pessimistic matching.

    The engine owns:
    - A ``SimClock`` that patches ``time.time`` during replay.
    - A ``MatchingEngine`` for order simulation.
    - A ``Telemetry`` instance for metrics accumulation.
    - Per-asset ``OHLCVAggregator`` instances for bar generation.

    Parameters
    ----------
    strategy:
        A ``StrategyABC`` subclass instance.
    data_loader:
        A ``DataLoader`` providing chronological ``MarketEvent`` objects.
    config:
        A ``BacktestConfig`` with run parameters.
    """

    def __init__(
        self,
        strategy: StrategyABC,
        data_loader: DataLoader,
        config: BacktestConfig | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.strategy = strategy
        self.data_loader = data_loader

        # ── Core components ────────────────────────────────────────────
        self.clock = SimClock(start_time=0.0)
        self.matching_engine = MatchingEngine(
            latency_ms=self.config.latency_ms,
            fee_max_pct=self.config.fee_max_pct,
            fee_enabled=self.config.fee_enabled,
        )
        self.telemetry = Telemetry(initial_cash=self.config.initial_cash)

        # ── State ──────────────────────────────────────────────────────
        self._cash: float = self.config.initial_cash
        self._positions: dict[str, float] = {}  # asset_id → shares held
        self._aggregators: dict[str, OHLCVAggregator] = {}  # asset_id → agg
        self._events_processed: int = 0
        # Set to True once a real L2 book event has been processed; used to
        # suppress synthetic BBO injection when real order-book data exists.
        self._has_real_book: bool = False
        # Per-asset BBO so strategies can query bid/ask for a specific token
        # rather than the unified matching-engine book (which reflects only the
        # most recently processed asset in trade-only mode).
        self._bbo_per_asset: dict[str, tuple[float, float]] = {}  # asset_id → (bid, ask)

    # ═══════════════════════════════════════════════════════════════════════
    #  Order submission API (called by strategies)
    # ═══════════════════════════════════════════════════════════════════════

    def submit_order(
        self,
        side: OrderSide,
        price: float,
        size: float,
        *,
        order_type: str = "limit",
        post_only: bool = False,
    ) -> SimOrder:
        """Submit an order to the simulated exchange.

        This is the primary interface for strategies to place orders.
        The order is subject to the configured latency penalty.
        """
        # Record mid at submission for slippage calculation
        mid = self.matching_engine.mid_price

        order = self.matching_engine.submit_order(
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            post_only=post_only,
            current_time=self.clock.now(),
        )

        # Store mid for slippage calc when fills arrive
        order._mid_at_submit = mid  # type: ignore[attr-defined]
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        return self.matching_engine.cancel_order(order_id)

    def simulate_fill(
        self,
        order_id: str,
        size: float,
        price: float | None = None,
        *,
        is_maker: bool = True,
    ) -> Fill | None:
        """Inject a simulated fill for an existing order.

        Replay adapters use this when a historical L2 snapshot crosses a
        resting quote but the recorded dataset does not contain the exact
        public trade print that would have drained the level.
        """
        fill = self.matching_engine.simulate_order_fill(
            order_id,
            size,
            price=price,
            current_time=self.clock.now(),
            is_maker=is_maker,
        )
        if fill is not None:
            self._process_fills([fill])
        return fill

    def get_asset_best_ask(self, asset_id: str) -> float:
        """Return the best ask for a specific asset.

        Falls back to the unified matching-engine best ask when no
        per-asset BBO has been recorded yet.
        """
        bbo = self._bbo_per_asset.get(asset_id)
        return bbo[1] if bbo else self.matching_engine.best_ask

    def get_asset_best_bid(self, asset_id: str) -> float:
        """Return the best bid for a specific asset.

        Falls back to the unified matching-engine best bid when no
        per-asset BBO has been recorded yet.
        """
        bbo = self._bbo_per_asset.get(asset_id)
        return bbo[0] if bbo else self.matching_engine.best_bid

    # ═══════════════════════════════════════════════════════════════════════
    #  Main replay loop
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> BacktestResult:
        """Execute the tick-by-tick backtest.

        Installs the simulated clock, processes all events, computes
        metrics, and returns a ``BacktestResult``.
        """
        log.info(
            "backtest_start",
            initial_cash=self.config.initial_cash,
            latency_ms=self.config.latency_ms,
            fee_max_pct=self.config.fee_max_pct,
        )

        # Wire strategy ↔ engine reference
        self.strategy.engine = self

        with self.clock:
            self.strategy.on_init()

            for event in self.data_loader:
                self._process_event(event)
                self._events_processed += 1

                # Periodic equity sampling
                if self._events_processed % self.config.equity_sample_interval == 0:
                    equity = self._compute_equity()
                    self.telemetry.record_equity(self.clock.now(), equity)

            # Final equity snapshot
            final_equity = self._compute_equity()
            self.telemetry.record_equity(self.clock.now(), final_equity)

            # Notify strategy that replay is done
            self.strategy.on_end()

        # ── Compute metrics ────────────────────────────────────────────
        metrics = self.telemetry.finalize(final_equity=final_equity)

        result = BacktestResult(
            metrics=metrics,
            config=self.config,
            events_processed=self._events_processed,
            final_cash=self._cash,
            final_equity=final_equity,
            positions=dict(self._positions),
            all_fills=self.matching_engine.get_all_fills(),
            all_orders=self.matching_engine.get_all_orders(),
        )

        log.info(
            "backtest_done",
            events=self._events_processed,
            pnl=round(metrics.total_pnl, 4),
            sharpe=round(metrics.sharpe_ratio, 2),
            max_dd=round(metrics.max_drawdown, 4),
        )

        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  Per-event processing
    # ═══════════════════════════════════════════════════════════════════════

    def _process_event(self, event: MarketEvent) -> None:
        """Process a single market event."""
        # Advance simulated clock
        self.clock.advance(event.timestamp)

        # Activate orders past their latency window
        activation_fills = self.matching_engine.activate_pending_orders(
            event.timestamp
        )
        self._process_fills(activation_fills)

        if event.event_type in ("l2_delta", "l2_snapshot"):
            self._process_book_event(event)
        elif event.event_type == "trade":
            self._process_trade_event(event)
        elif event.event_type == "external_price":
            self._process_external_price_event(event)

    def _process_book_event(self, event: MarketEvent) -> None:
        """Handle an L2 delta or snapshot."""
        self._has_real_book = True
        # Update the matching engine's historical book mirror
        self.matching_engine.on_book_update(event.data, current_time=event.timestamp)

        # Check if resting maker orders are still viable
        self.matching_engine.check_maker_viability()

        # Build a snapshot dict for the strategy
        snapshot = {
            "best_bid": self.matching_engine.best_bid,
            "best_ask": self.matching_engine.best_ask,
            "mid_price": self.matching_engine.mid_price,
            "bid_levels": self.matching_engine.bid_levels(5),
            "ask_levels": self.matching_engine.ask_levels(5),
            "event_data": event.data,
            "timestamp": event.timestamp,
        }

        # Record per-asset BBO from real book data
        self._bbo_per_asset[event.asset_id] = (
            self.matching_engine.best_bid,
            self.matching_engine.best_ask,
        )

        self.strategy.on_book_update(event.asset_id, snapshot)

    # Half-spread used to synthesize a BBO when only trade data is available
    # (no L2 book snapshots).  0.02 yields a 4-cent spread which passes the
    # default min_spread_cents=4 gate while staying realistic for Polymarket.
    _SYNTH_HALF_SPREAD: float = 0.02

    def _process_trade_event(self, event: MarketEvent) -> None:
        """Handle a public trade event."""
        data = event.data

        # Parse trade fields
        trade_price = float(data.get("price", 0))
        trade_size = float(data.get("size", 0))
        trade_side = str(data.get("side", "buy")).lower()

        if trade_price <= 0 or trade_size <= 0:
            return

        # When no L2 book data is available (trade-only replay) synthesize a
        # one-level BBO centred on the trade price so that signal gates that
        # require best_bid / best_ask can fire.  Update on every trade so the
        # BBO tracks price movements; skip once real L2 data has arrived.
        if not self._has_real_book:
            half = self._SYNTH_HALF_SPREAD
            synth_bid = max(0.01, trade_price - half)
            synth_ask = min(0.99, trade_price + half)
            self.matching_engine.on_book_update({
                "event_type": "book_snapshot",
                "bids": [{"price": synth_bid, "size": 9999.0}],
                "asks": [{"price": synth_ask, "size": 9999.0}],
            }, current_time=event.timestamp)
            # Track per-asset synthetic BBO so strategies can query the
            # correct bid/ask for a specific token, even when the shared
            # matching engine reflects a different token's last trade.
            self._bbo_per_asset[event.asset_id] = (synth_bid, synth_ask)

        # Check maker order queue drainage
        trade_fills = self.matching_engine.on_trade(
            trade_price=trade_price,
            trade_size=trade_size,
            trade_side=trade_side,
            current_time=event.timestamp,
        )
        self._process_fills(trade_fills)

        # Build TradeEvent for the strategy
        trade_event = TradeEvent(
            timestamp=event.timestamp,
            market_id=data.get("market_id", data.get("market", "")),
            asset_id=event.asset_id,
            side=trade_side,
            price=trade_price,
            size=trade_size,
            is_yes=data.get("is_yes", True),
            is_taker=data.get("is_taker", False),
        )

        # Feed aggregator for bar generation
        agg = self._get_aggregator(event.asset_id)
        bar = agg.on_trade(trade_event)
        if bar is not None:
            self.strategy.on_bar(event.asset_id, bar)

        # Notify strategy
        self.strategy.on_trade(event.asset_id, trade_event)

    def _process_external_price_event(self, event: MarketEvent) -> None:
        """Handle an external price observation (e.g. Binance BTC price).

        These events are recorded by the live bot's data recorder and
        replayed here so that the RPE's crypto model has access to the
        same external prices it would see in production.
        """
        data = event.data
        price = float(data.get("price", data.get("model_probability", 0)))
        self.strategy.on_external_price(
            event.asset_id, price, event.timestamp
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Fill processing
    # ═══════════════════════════════════════════════════════════════════════

    def _process_fills(self, fills: list[Fill]) -> None:
        """Update cash, positions, telemetry on new fills."""
        for fill in fills:
            # Get the mid at submission from the order
            order = self.matching_engine.get_order(fill.order_id)
            mid_at_submit = getattr(order, "_mid_at_submit", 0.0) if order else 0.0

            # Update cash balance
            if fill.side == OrderSide.BUY:
                cost = fill.price * fill.size + fill.fee
                self._cash -= cost
                self._positions[fill.order_id.rsplit("-", 1)[0]] = (
                    self._positions.get(fill.order_id.rsplit("-", 1)[0], 0.0)
                    + fill.size
                )
            else:
                proceeds = fill.price * fill.size - fill.fee
                self._cash += proceeds
                # Reduce position — use the order context to find asset
                # For simplicity, track by a synthetic key
                pos_key = fill.order_id.rsplit("-", 1)[0]
                held = self._positions.get(pos_key, 0.0)
                self._positions[pos_key] = max(0.0, held - fill.size)
                if self._positions[pos_key] <= 1e-9:
                    self._positions.pop(pos_key, None)

            # Record in telemetry
            self.telemetry.record_fill(fill, mid_at_submission=mid_at_submit)

            # Notify strategy
            self.strategy.on_fill(fill)

    # ═══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_equity(self) -> float:
        """Mark-to-market equity = cash + sum(position_value)."""
        equity = self._cash
        mid = self.matching_engine.mid_price
        if mid > 0:
            for _key, shares in self._positions.items():
                equity += shares * mid
        return equity

    def _get_aggregator(self, asset_id: str) -> OHLCVAggregator:
        """Lazily create per-asset OHLCV aggregators."""
        if asset_id not in self._aggregators:
            self._aggregators[asset_id] = OHLCVAggregator(asset_id)
        return self._aggregators[asset_id]

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    @property
    def equity(self) -> float:
        return self._compute_equity()
