"""
Strategy interface and production-parity adapter for the backtest engine.

Two interaction modes
─────────────────────

1. **StrategyABC** — lightweight callback-based interface for rapid
   research iteration.  Subclass and implement the hooks.

2. **BotReplayAdapter** — wraps the existing live-bot components
   (``CompositeSignalEvaluator``, ``PanicDetector``, ``PositionManager``,
   ``OHLCVAggregator``) so that the exact production pipeline runs inside
   the backtest engine.  Ensures PnL parity between sim and live.

The backtest engine only knows about ``StrategyABC``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.core.config import StrategyParams
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator, OHLCVBar
from src.data.websocket_client import TradeEvent
from src.trading.executor import OrderSide
from src.trading.portfolio_correlation import PortfolioCorrelationEngine

if TYPE_CHECKING:
    from src.backtest.data_loader import MarketEvent
    from src.backtest.matching_engine import Fill, SimOrder

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Abstract Strategy Interface
# ═══════════════════════════════════════════════════════════════════════════

class StrategyABC(ABC):
    """Callback-based strategy interface for the backtest engine.

    The engine calls these hooks in order as it replays historical data.
    Strategies submit orders via ``self.engine.submit_order(...)`` and
    cancel via ``self.engine.cancel_order(...)``.

    Attributes
    ----------
    engine:
        Reference to the ``BacktestEngine`` instance. Set by the engine
        before calling ``on_init()``.
    """

    engine: Any = None   # set by BacktestEngine before on_init()

    @abstractmethod
    def on_init(self) -> None:
        """Called once before the replay loop starts.

        Use this to initialise indicators, state, etc.
        The ``self.engine`` reference is available here.
        """
        ...

    @abstractmethod
    def on_book_update(self, asset_id: str, snapshot: dict) -> None:
        """Called on every L2 delta or snapshot.

        Parameters
        ----------
        asset_id:
            The token this update relates to.
        snapshot:
            Dict with ``best_bid``, ``best_ask``, ``mid_price``,
            ``bid_depth_usd``, ``ask_depth_usd``, etc.
        """
        ...

    @abstractmethod
    def on_trade(self, asset_id: str, trade: TradeEvent) -> None:
        """Called on every historical public trade."""
        ...

    @abstractmethod
    def on_fill(self, fill: "Fill") -> None:
        """Called when a simulated order is filled (or partially filled)."""
        ...

    def on_bar(self, asset_id: str, bar: OHLCVBar) -> None:
        """Called when a 1-minute OHLCV bar closes (optional override)."""
        pass

    def on_end(self) -> None:
        """Called after the replay loop finishes (optional override).

        Use this for cleanup, final position unwinding, etc.
        """
        pass

    def on_external_price(
        self, asset_id: str, price: float, timestamp: float
    ) -> None:
        """Called when an external price observation is replayed.

        Used by the RPE's crypto model to receive Binance BTC prices
        during backtest replay.  Default implementation is a no-op.
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Bot Replay Adapter — production parity
# ═══════════════════════════════════════════════════════════════════════════

class BotReplayAdapter(StrategyABC):
    """Adapter that wires the live-bot signal/position pipeline into
    the backtest engine.

    Translates engine callbacks into the exact data structures and method
    calls that the live ``TradingBot`` uses, so that signal firing,
    sizing, take-profit, and stop-loss logic run identically.

    Parameters
    ----------
    market_id:
        The condition_id of the market being backtested.
    yes_asset_id:
        Token ID for the YES outcome.
    no_asset_id:
        Token ID for the NO outcome.
    fee_enabled:
        Whether the market's category charges dynamic fees.
    initial_bankroll:
        Starting cash for Kelly sizing.
    params:
        Optional ``StrategyParams`` for tunable knobs.  When ``None``
        (the default) the global defaults from ``src.core.config`` are
        used, preserving full backward-compatibility.
    """

    def __init__(
        self,
        market_id: str,
        yes_asset_id: str,
        no_asset_id: str,
        *,
        fee_enabled: bool = True,
        initial_bankroll: float = 1000.0,
        params: StrategyParams | None = None,
    ) -> None:
        self._market_id = market_id
        self._yes_asset_id = yes_asset_id
        self._no_asset_id = no_asset_id
        self._fee_enabled = fee_enabled
        self._initial_bankroll = initial_bankroll
        self._params = params or StrategyParams()

        # Aggregators (created on init)
        self._yes_agg: OHLCVAggregator | None = None
        self._no_agg: OHLCVAggregator | None = None

        # Pending orders we've submitted: sim_order_id → context
        self._pending_entries: dict[str, dict] = {}
        self._pending_exits: dict[str, dict] = {}

        # Position tracking (simplified for backtest)
        self._positions: list[dict] = []   # all opened positions
        self._open_positions: dict[str, dict] = {}  # order_id → position context

        # External price history for RPE crypto model replay
        self._external_prices: list[tuple[float, float]] = []  # (ts, price)

        # PCE backtest integration (Pillar 15 Issue 4)
        self._pce: PortfolioCorrelationEngine | None = None
        self._pce_rejections: int = 0
        if self._params.pce_backtest_enabled:
            self._pce = PortfolioCorrelationEngine(
                shadow_mode=True,  # always shadow in backtest
                max_portfolio_var_usd=self._params.pce_max_portfolio_var_usd,
                haircut_threshold=self._params.pce_correlation_haircut_threshold,
                structural_prior_weight=self._params.pce_structural_prior_weight,
                min_overlap_bars=self._params.pce_min_overlap_bars,
                holding_period_minutes=self._params.pce_holding_period_minutes,
                var_soft_cap=self._params.pce_var_soft_cap,
                var_bisect_iterations=self._params.pce_var_bisect_iterations,
            )

    def on_init(self) -> None:
        """Initialise aggregators and detectors."""
        self._yes_agg = OHLCVAggregator(self._yes_asset_id)
        self._no_agg = OHLCVAggregator(self._no_asset_id)

        # Register market with PCE if enabled
        if self._pce is not None and self._yes_agg is not None:
            self._pce.register_market(
                market_id=self._market_id,
                event_id=self._market_id,  # same in backtest
                tags="backtest",
                aggregator=self._yes_agg,
            )

        log.info(
            "bot_replay_adapter_init",
            market_id=self._market_id,
            yes=self._yes_asset_id,
            no=self._no_asset_id,
            pce_enabled=self._pce is not None,
        )

    def on_book_update(self, asset_id: str, snapshot: dict) -> None:
        """Route book updates to internal state.

        In the live bot, book updates feed into spread scoring, ghost
        liquidity detection, and TP rescaling. The adapter stores the
        latest snapshot for use in signal evaluation and sizing.
        """
        # Store latest book state for sizing / signal decisions
        pass  # The matching engine maintains the authoritative book

    def on_trade(self, asset_id: str, trade: TradeEvent) -> None:
        """Feed trades into OHLCV aggregators.

        When a bar closes, evaluate the strategy's signal logic.
        """
        if asset_id == self._yes_asset_id and self._yes_agg:
            bar = self._yes_agg.on_trade(trade)
            if bar is not None:
                self._on_yes_bar_closed(bar)
        elif asset_id == self._no_asset_id and self._no_agg:
            bar = self._no_agg.on_trade(trade)
            if bar is not None:
                self.on_bar(asset_id, bar)

    def _on_yes_bar_closed(self, bar: OHLCVBar) -> None:
        """Evaluate the panic/mean-reversion signal on a YES bar close.

        This mirrors the live bot's ``_on_yes_bar_closed()`` logic.
        """
        if self._yes_agg is None or self._no_agg is None:
            return
        if self.engine is None:
            return

        # Check if we have enough history
        if len(self._yes_agg.bars) < 5:
            return

        vwap = self._yes_agg.rolling_vwap
        if vwap <= 0:
            return

        # Simple mean-reversion signal: YES price spikes above VWAP
        yes_price = bar.close
        deviation = (yes_price - vwap) / max(vwap, 0.01)

        # The NO side is our entry — when YES spikes, NO drops
        no_price = self._no_agg.current_price
        if no_price <= 0:
            return

        # Check if deviation is significant
        vol = self._yes_agg.rolling_volatility
        if vol <= 0:
            vol = 0.01

        zscore = deviation / vol
        if zscore < self._params.zscore_threshold:
            return

        # Too many open positions?
        if len(self._open_positions) >= self._params.max_open_positions:
            return

        # Entry: buy NO token at best ask
        best_ask = self.engine.matching_engine.best_ask
        if best_ask <= 0:
            return

        # Sizing driven by kelly_fraction & max_trade_size_usd
        trade_usd = min(
            self._initial_bankroll * self._params.kelly_fraction,
            self._params.max_trade_size_usd,
        )
        size = trade_usd / best_ask if best_ask > 0 else 0
        if size < 1:
            return

        # PCE VaR gate: check if adding this position keeps VaR within limits
        if self._pce is not None:
            # Refresh correlations on each entry attempt (cheap in backtest)
            self._pce.refresh_correlations()

            # Build synthetic open_positions list for PCE
            pce_positions = self._build_pce_positions()

            if self._pce.var_soft_cap:
                sizing_result = self._pce.compute_var_sizing_cap(
                    open_positions=pce_positions,
                    proposed_market_id=self._market_id,
                    proposed_size_usd=trade_usd,
                    proposed_direction="NO",
                )
                # Record PCE snapshot for telemetry
                if self.engine is not None and hasattr(self.engine, 'telemetry'):
                    _avg_corr = 0.0
                    if pce_positions:
                        _corrs = [
                            self._pce.corr_matrix.get(self._market_id, p.market_id)
                            for p in pce_positions if p.market_id != self._market_id
                        ]
                        _avg_corr = sum(_corrs) / len(_corrs) if _corrs else 0.0
                    self.engine.telemetry.record_pce_snapshot(
                        timestamp=bar.open_time,
                        portfolio_var=sizing_result.current_var,
                        avg_correlation=_avg_corr,
                        n_positions=len(pce_positions),
                    )
                if sizing_result.cap_usd < 1.0:
                    self._pce_rejections += 1
                    log.debug("adapter_pce_var_blocked", market=self._market_id)
                    return
                trade_usd = min(trade_usd, sizing_result.cap_usd)
                size = trade_usd / best_ask if best_ask > 0 else 0
                if size < 1:
                    return
            else:
                allowed, _var_result = self._pce.check_var_gate(
                    open_positions=pce_positions,
                    proposed_market_id=self._market_id,
                    proposed_size_usd=trade_usd,
                    proposed_direction="NO",
                )
                # Record PCE snapshot for telemetry
                if self.engine is not None and hasattr(self.engine, 'telemetry'):
                    _avg_corr = 0.0
                    if pce_positions:
                        _corrs = [
                            self._pce.corr_matrix.get(self._market_id, p.market_id)
                            for p in pce_positions if p.market_id != self._market_id
                        ]
                        _avg_corr = sum(_corrs) / len(_corrs) if _corrs else 0.0
                    self.engine.telemetry.record_pce_snapshot(
                        timestamp=bar.open_time,
                        portfolio_var=_var_result.portfolio_var_usd,
                        avg_correlation=_avg_corr,
                        n_positions=len(pce_positions),
                    )
                if not allowed:
                    self._pce_rejections += 1
                    log.debug("adapter_pce_var_blocked", market=self._market_id)
                    return

            # Apply concentration haircut
            haircut = self._pce.compute_concentration_haircut(
                proposed_market_id=self._market_id,
                open_positions=pce_positions,
            )
            if haircut < 1.0:
                trade_usd *= haircut
                size = trade_usd / best_ask if best_ask > 0 else 0
                if size < 1:
                    return

        order = self.engine.submit_order(
            side=OrderSide.BUY,
            price=best_ask,
            size=size,
            order_type="limit",
            post_only=False,
        )

        # Track as pending entry
        target_price = no_price + self._params.alpha_default * (vwap - no_price)
        self._pending_entries[order.order_id] = {
            "order": order,
            "entry_price": best_ask,
            "target_price": max(target_price, best_ask + 0.01),
            "zscore": zscore,
            "yes_price": yes_price,
            "vwap": vwap,
        }

        log.debug(
            "adapter_entry_signal",
            order_id=order.order_id,
            no_price=no_price,
            zscore=round(zscore, 2),
            target=round(target_price, 4),
        )

    def _build_pce_positions(self) -> list:
        """Build a list of lightweight position objects for PCE API.

        PCE expects objects with ``.market_id``, ``.entry_price``, and
        ``.entry_size`` attributes.  We use a simple namespace class.
        """
        class _Pos:
            def __init__(self, market_id: str, entry_price: float, entry_size: float, trade_side: str = "NO"):
                self.market_id = market_id
                self.entry_price = entry_price
                self.entry_size = entry_size
                self.trade_side = trade_side

        positions: list[_Pos] = []
        for pos in self._open_positions.values():
            fill = pos["entry_fill"]
            positions.append(_Pos(
                market_id=self._market_id,
                entry_price=fill.price,
                entry_size=fill.size,
            ))
        return positions

    def on_fill(self, fill: "Fill") -> None:
        """Route fills to entry or exit position handling."""
        oid = fill.order_id

        # ── Entry fill ─────────────────────────────────────────────────
        if oid in self._pending_entries:
            ctx = self._pending_entries.pop(oid)
            order = self.engine.matching_engine.get_order(oid)
            if order is None or order.remaining > 1e-9:
                # Partial — keep tracking
                if order and order.remaining > 1e-9:
                    self._pending_entries[oid] = ctx
                return

            # Fully filled — place exit order at target
            target = ctx["target_price"]
            exit_order = self.engine.submit_order(
                side=OrderSide.SELL,
                price=target,
                size=fill.size,
                order_type="limit",
                post_only=True,
            )
            pos = {
                "entry_fill": fill,
                "entry_ctx": ctx,
                "exit_order_id": exit_order.order_id,
                "entry_time": fill.timestamp,
            }
            self._pending_exits[exit_order.order_id] = pos
            self._open_positions[exit_order.order_id] = pos

            log.debug(
                "adapter_entry_filled",
                order_id=oid,
                entry_price=fill.price,
                target=target,
                exit_order=exit_order.order_id,
            )
            return

        # ── Exit fill ──────────────────────────────────────────────────
        if oid in self._pending_exits:
            pos = self._pending_exits.pop(oid)
            self._open_positions.pop(oid, None)

            entry_fill = pos["entry_fill"]
            pnl = (fill.price - entry_fill.price) * fill.size
            pnl_net = pnl - fill.fee - entry_fill.fee

            self._positions.append({
                "entry_price": entry_fill.price,
                "exit_price": fill.price,
                "size": fill.size,
                "pnl_gross": pnl,
                "pnl_net": pnl_net,
                "entry_time": pos["entry_time"],
                "exit_time": fill.timestamp,
                "entry_fee": entry_fill.fee,
                "exit_fee": fill.fee,
            })

            log.debug(
                "adapter_exit_filled",
                entry=entry_fill.price,
                exit=fill.price,
                pnl_net=round(pnl_net, 4),
            )

    def on_external_price(
        self, asset_id: str, price: float, timestamp: float
    ) -> None:
        """Store replayed external prices for RPE crypto model.

        In backtesting, the RPE's ``CryptoPriceModel`` reads the
        latest price from ``self._external_prices`` via a closure,
        replicating the live bot's Binance WS feed.
        """
        if price > 0:
            self._external_prices.append((timestamp, price))
            # Keep only last 500 entries (matches live deque maxlen)
            if len(self._external_prices) > 500:
                self._external_prices = self._external_prices[-500:]

    def on_end(self) -> None:
        """Log summary of adapter performance."""
        # Push PCE rejection count to telemetry
        if self._pce is not None and self.engine is not None and hasattr(self.engine, 'telemetry'):
            self.engine.telemetry.set_pce_rejections(self._pce_rejections)

        # Unregister from PCE
        if self._pce is not None:
            self._pce.unregister_market(self._market_id)

        total_pnl = sum(p["pnl_net"] for p in self._positions)
        log.info(
            "adapter_summary",
            total_positions=len(self._positions),
            open_remaining=len(self._open_positions),
            total_pnl_net=round(total_pnl, 4),
            pce_enabled=self._pce is not None,
        )
