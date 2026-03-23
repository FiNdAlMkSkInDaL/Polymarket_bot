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
from collections import deque
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from src.core.config import EXCHANGE_MIN_SHARES, EXCHANGE_MIN_USD, StrategyParams
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator, OHLCVBar
from src.data.websocket_client import TradeEvent
from src.signals.edge_filter import compute_edge_score
from src.signals.panic_detector import PanicDetector
from src.trading.executor import OrderSide, OrderStatus
from src.trading.portfolio_correlation import PortfolioCorrelationEngine

if TYPE_CHECKING:
    from src.backtest.data_loader import MarketEvent
    from src.backtest.matching_engine import Fill, SimOrder

log = get_logger(__name__)

LEGACY_BACKTEST_SIGNAL_DEFAULTS: dict[str, float | int] = {
    "zscore_threshold": 0.20,
    "volume_ratio_threshold": 0.5,
    "trend_guard_pct": 0.08,
    "trend_guard_bars": 15,
}


def split_strategy_and_legacy_params(
    overrides: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    """Partition WFO/backtest overrides into live config vs legacy signal knobs."""
    overrides = overrides or {}
    strategy_field_names = {field.name for field in fields(StrategyParams)}
    strategy_params = {
        name: value for name, value in overrides.items() if name in strategy_field_names
    }
    legacy_params = {
        name: value
        for name, value in overrides.items()
        if name in LEGACY_BACKTEST_SIGNAL_DEFAULTS
    }
    return strategy_params, legacy_params


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
        legacy_signal_params: dict[str, float | int] | None = None,
    ) -> None:
        self._market_id = market_id
        self._yes_asset_id = yes_asset_id
        self._no_asset_id = no_asset_id
        self._fee_enabled = fee_enabled
        self._initial_bankroll = initial_bankroll
        self._params = params or StrategyParams()
        self._legacy_signal_params = dict(LEGACY_BACKTEST_SIGNAL_DEFAULTS)
        if legacy_signal_params:
            self._legacy_signal_params.update(legacy_signal_params)

        # Aggregators and detector (created on init)
        self._yes_agg: OHLCVAggregator | None = None
        self._no_agg: OHLCVAggregator | None = None
        self._detector: PanicDetector | None = None

        # Signal cooldown tracking
        self._last_signal_time: float = 0.0
        self._cooldown_seconds: float = self._params.signal_cooldown_minutes * 60.0

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

        # Use the real PanicDetector — same gates as live bot
        self._detector = PanicDetector(
            market_id=self._market_id,
            yes_asset_id=self._yes_asset_id,
            no_asset_id=self._no_asset_id,
            yes_aggregator=self._yes_agg,
            no_aggregator=self._no_agg,
            zscore_threshold=float(self._legacy_signal_params["zscore_threshold"]),
            volume_ratio_threshold=float(self._legacy_signal_params["volume_ratio_threshold"]),
            trend_guard_pct=float(self._legacy_signal_params["trend_guard_pct"]),
            trend_guard_bars=int(self._legacy_signal_params["trend_guard_bars"]),
        )

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

        Uses the real ``PanicDetector`` for full signal parity with the
        live bot, including z-score, volume ratio, intra-bar retracement,
        NO-discount, and trend-guard gates.
        """
        if self._yes_agg is None or self._no_agg is None:
            return
        if self.engine is None or self._detector is None:
            return

        yes_price = bar.close
        bar_time = getattr(bar, 'open_time', 0.0)
        if not isinstance(bar_time, (int, float)):
            bar_time = 0.0

        # ── Near-resolved price guard (same as live bot) ───────────────
        if yes_price >= 0.97 or yes_price <= 0.03:
            return

        # ── Tradeable price band guard ─────────────────────────────────
        if not (self._params.min_tradeable_price < yes_price < self._params.max_tradeable_price):
            return

        # ── Signal cooldown ────────────────────────────────────────────
        if bar_time - self._last_signal_time < self._cooldown_seconds:
            return

        # ── Too many open positions? ───────────────────────────────────
        if len(self._open_positions) >= self._params.max_open_positions:
            return

        # ── NO best ask from per-asset BBO ────────────────────────────
        # The unified matching engine reflects only the most recently
        # processed token (the YES trade that just closed this bar in
        # trade-only mode).  Use the NO-specific BBO instead so the
        # PanicDetector's NO-discount check compares against the actual
        # NO token price, not the YES token price.
        best_ask = self.engine.get_asset_best_ask(self._no_asset_id)
        if best_ask <= 0:
            return

        # ── Run the real PanicDetector ─────────────────────────────────
        # This checks: z-score, volume ratio, intra-bar retracement,
        # NO discount factor, and trend guard — all identical to live
        sig = self._detector.evaluate(bar, no_best_ask=best_ask)
        if sig is None:
            return

        # Record signal time for cooldown
        self._last_signal_time = bar_time

        best_bid = self.engine.get_asset_best_bid(self._no_asset_id)
        if best_bid <= 0:
            return

        # ── Spread gate ────────────────────────────────────────────────
        spread = best_ask - best_bid
        if spread * 100.0 < self._params.min_spread_cents:
            return

        # ── Edge quality score gate ────────────────────────────────────
        no_vwap = self._no_agg.rolling_vwap
        exec_mode = "maker" if self._params.maker_routing_enabled else "taker"
        eqs = compute_edge_score(
            entry_price=best_ask,
            no_vwap=no_vwap if no_vwap > 0 else sig.no_best_ask,
            zscore=sig.zscore,
            volume_ratio=sig.volume_ratio,
            whale_confluence=sig.whale_confluence,
            fee_enabled=self._fee_enabled,
            alpha=self._params.alpha_default,
            zscore_threshold=float(self._legacy_signal_params["zscore_threshold"]),
            min_score=self._params.min_edge_score,
            current_ewma_vol=self._no_agg.rolling_volatility_ewma or None,
            execution_mode=exec_mode,
        )
        if not eqs.viable:
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
        no_price = sig.no_best_ask
        target_price = no_price + self._params.alpha_default * (max(no_vwap, no_price) - no_price)
        self._pending_entries[order.order_id] = {
            "order": order,
            "entry_price": best_ask,
            "target_price": max(target_price, best_ask + 0.01),
            "zscore": sig.zscore,
            "yes_price": sig.yes_price,
            "vwap": sig.yes_vwap,
        }

        log.debug(
            "adapter_entry_signal",
            order_id=order.order_id,
            no_price=no_price,
            zscore=round(sig.zscore, 2),
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


@dataclass(slots=True)
class _ReplayQuoteState:
    side: OrderSide
    tier: str
    order_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ReplayIntendedQuote:
    side: OrderSide
    tier: str
    target_price: float
    target_size: float


_REPLAY_TIGHT_TIER = "tight"
_REPLAY_WIDE_TIER = "wide"


class _ReplayOrderBook:
    """Minimal L2 replay book with live-parity OFI and depth metrics."""

    _OFI_WINDOW_S = 2.0

    def __init__(self, asset_id: str) -> None:
        self.asset_id = asset_id
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._depth_history: deque[tuple[float, float]] = deque(maxlen=40)
        self._ofi_window: deque[tuple[float, float, float]] = deque(maxlen=500)
        self._ofi_price_window: deque[tuple[float, str, float, float]] = deque(maxlen=1000)
        self._last_update: float = 0.0

    @property
    def has_data(self) -> bool:
        return self._last_update > 0.0

    @property
    def best_bid(self) -> float:
        return max(self._bids) if self._bids else 0.0

    @property
    def best_ask(self) -> float:
        return min(self._asks) if self._asks else 0.0

    @property
    def book_depth_ratio(self) -> float:
        bid_depth = sum(price * size for price, size in sorted(self._bids.items(), reverse=True)[:5])
        ask_depth = sum(price * size for price, size in sorted(self._asks.items())[:5])
        if ask_depth <= 0:
            return 1.0
        return round(bid_depth / ask_depth, 2)

    def apply_event(self, event_data: dict[str, Any], timestamp: float) -> None:
        event_type = str(event_data.get("event_type", "")).lower()
        if event_type in ("book", "snapshot", "book_snapshot", "l2_snapshot"):
            self._apply_snapshot(event_data, timestamp)
        else:
            self._apply_delta(event_data, timestamp)
        self._last_update = timestamp
        self._record_depth(timestamp)

    def current_total_depth(self) -> float:
        bid_depth = sum(price * size for price, size in sorted(self._bids.items(), reverse=True)[:5])
        ask_depth = sum(price * size for price, size in sorted(self._asks.items())[:5])
        return round(bid_depth + ask_depth, 2)

    def depth_velocity(self, window_s: float) -> float | None:
        if len(self._depth_history) < 2:
            return None
        target_time = self._last_update - window_s
        past_depth: float | None = None
        for ts, depth in self._depth_history:
            if ts <= target_time:
                past_depth = depth
            else:
                break
        if past_depth is None or past_depth <= 0:
            return None
        return (self.current_total_depth() - past_depth) / past_depth

    def opposing_ofi_at_price(self, price: float, side: str) -> float:
        self._prune_ofi(self._last_update)
        consumed = 0.0
        target_side = side.upper()
        for _ts, event_side, event_price, delta in self._ofi_price_window:
            if event_side == target_side and abs(event_price - price) < 1e-6 and delta < 0:
                consumed += abs(delta)
        return consumed

    def _apply_snapshot(self, data: dict[str, Any], timestamp: float) -> None:
        new_bids = self._parse_levels(data.get("bids") or [])
        new_asks = self._parse_levels(data.get("asks") or [])

        delta_bid_qty = 0.0
        delta_ask_qty = 0.0

        for price in set(self._bids) | set(new_bids):
            level_delta = new_bids.get(price, 0.0) - self._bids.get(price, 0.0)
            if level_delta != 0.0:
                self._ofi_price_window.append((timestamp, "BUY", price, level_delta))
                delta_bid_qty += level_delta

        for price in set(self._asks) | set(new_asks):
            level_delta = new_asks.get(price, 0.0) - self._asks.get(price, 0.0)
            if level_delta != 0.0:
                self._ofi_price_window.append((timestamp, "SELL", price, level_delta))
                delta_ask_qty += level_delta

        self._bids = new_bids
        self._asks = new_asks
        self._record_ofi(delta_bid_qty, delta_ask_qty, timestamp)

    def _apply_delta(self, data: dict[str, Any], timestamp: float) -> None:
        changes = data.get("changes") or data.get("price_changes") or data.get("data") or []
        if isinstance(changes, dict):
            changes = [changes]
        if not changes and data.get("price") is not None:
            changes = [data]

        delta_bid_qty = 0.0
        delta_ask_qty = 0.0
        for change in changes:
            try:
                price = float(change.get("price", 0.0))
                size = float(change.get("size", 0.0))
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue

            side = str(change.get("side", "")).upper()
            if side in ("BUY", "BID"):
                old_size = self._bids.get(price, 0.0)
                if size <= 0:
                    self._bids.pop(price, None)
                else:
                    self._bids[price] = size
                level_delta = size - old_size
                if level_delta != 0.0:
                    self._ofi_price_window.append((timestamp, "BUY", price, level_delta))
                    delta_bid_qty += level_delta
            elif side in ("SELL", "ASK"):
                old_size = self._asks.get(price, 0.0)
                if size <= 0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = size
                level_delta = size - old_size
                if level_delta != 0.0:
                    self._ofi_price_window.append((timestamp, "SELL", price, level_delta))
                    delta_ask_qty += level_delta

        self._record_ofi(delta_bid_qty, delta_ask_qty, timestamp)

    @staticmethod
    def _parse_levels(levels: list[dict[str, Any]]) -> dict[float, float]:
        parsed: dict[float, float] = {}
        for level in levels:
            try:
                price = float(level["price"])
                size = float(level["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if price > 0 and size > 0:
                parsed[price] = size
        return parsed

    def _record_depth(self, timestamp: float) -> None:
        self._depth_history.append((timestamp, self.current_total_depth()))

    def _record_ofi(self, delta_bid_qty: float, delta_ask_qty: float, timestamp: float) -> None:
        if delta_bid_qty != 0.0 or delta_ask_qty != 0.0:
            self._ofi_window.append((timestamp, delta_bid_qty, delta_ask_qty))
        self._prune_ofi(timestamp)

    def _prune_ofi(self, timestamp: float) -> None:
        cutoff = timestamp - self._OFI_WINDOW_S
        while self._ofi_window and self._ofi_window[0][0] < cutoff:
            self._ofi_window.popleft()
        while self._ofi_price_window and self._ofi_price_window[0][0] < cutoff:
            self._ofi_price_window.popleft()


class PureMarketMakerReplayAdapter(StrategyABC):
    """Backtest replay harness for the passive pure market maker."""

    def __init__(
        self,
        market_id: str,
        yes_asset_id: str,
        no_asset_id: str,
        *,
        fee_enabled: bool = True,
        initial_bankroll: float = 1000.0,
        params: StrategyParams | None = None,
        legacy_signal_params: dict[str, float | int] | None = None,
    ) -> None:
        self._market_id = market_id
        self._yes_asset_id = yes_asset_id
        self._no_asset_id = no_asset_id
        self._fee_enabled = fee_enabled
        self._initial_bankroll = initial_bankroll
        self._params = params or StrategyParams()
        self._legacy_signal_params = legacy_signal_params or {}

        self._book = _ReplayOrderBook(no_asset_id)
        self._book_update_count = 0
        self._inventory: float = 0.0
        self._quote_states: dict[tuple[OrderSide, str], _ReplayQuoteState] = {}
        self._order_to_key: dict[str, tuple[OrderSide, str]] = {}

        self._quote_size_usd = self._params.pure_mm_quote_size_usd
        self._wide_tier_enabled = self._params.pure_mm_wide_tier_enabled
        self._wide_spread_pct = self._params.pure_mm_wide_spread_pct
        self._wide_size_usd = self._params.pure_mm_wide_size_usd
        self._inventory_cap_usd = self._params.pure_mm_inventory_cap_usd
        self._toxic_ofi_ratio = self._params.pure_mm_toxic_ofi_ratio
        self._depth_window_s = self._params.pure_mm_depth_window_s
        self._depth_evaporation_pct = self._params.pure_mm_depth_evaporation_pct

    def on_init(self) -> None:
        log.info(
            "pure_mm_replay_adapter_init",
            market_id=self._market_id,
            asset_id=self._no_asset_id,
            quote_size_usd=self._quote_size_usd,
            inventory_cap_usd=self._inventory_cap_usd,
        )

    def on_book_update(self, asset_id: str, snapshot: dict) -> None:
        if asset_id != self._no_asset_id or self.engine is None:
            return

        event_data = snapshot.get("event_data")
        if not isinstance(event_data, dict):
            return

        timestamp = float(snapshot.get("timestamp", 0.0) or self.engine.clock.now())
        self._book_update_count += 1
        self._book.apply_event(event_data, timestamp)
        self._simulate_crossed_fills()
        self._sync_quotes()

    def on_trade(self, asset_id: str, trade: TradeEvent) -> None:
        return

    def on_fill(self, fill: "Fill") -> None:
        key = self._order_to_key.get(fill.order_id)
        if key is None:
            return

        side, tier = key

        if fill.side == OrderSide.BUY:
            self._inventory += fill.size
        else:
            self._inventory = max(0.0, self._inventory - fill.size)

        order = self.engine.matching_engine.get_order(fill.order_id) if self.engine else None
        if order is None or order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            state = self._quote_states[(side, tier)]
            state.order_id = None
            self._order_to_key.pop(fill.order_id, None)

    def on_end(self) -> None:
        inventory_mark_price = max(self._book.best_bid, 0.0)
        log.info(
            "pure_mm_replay_adapter_summary",
            asset_id=self._no_asset_id,
            inventory=round(self._inventory, 4),
            inventory_mark_value=round(self._inventory_value(inventory_mark_price), 4),
            open_quotes=sum(1 for state in self._quote_states.values() if state.order_id),
            book_updates=self._book_update_count,
        )

    @property
    def has_l2_book(self) -> bool:
        return self._book_update_count > 0

    def _sync_quotes(self) -> None:
        best_bid = self._book.best_bid
        best_ask = self._book.best_ask
        if best_bid <= 0 or best_ask <= 0:
            self._cancel_all_quotes(reason="empty_bbo")
            return

        intended_quotes = self._build_intended_quotes(best_bid, best_ask)
        if self._is_market_toxic(intended_quotes):
            self._cancel_all_quotes(reason="toxic_flow")
            return

        active_keys: set[tuple[OrderSide, str]] = set()
        for intended_quote in intended_quotes:
            key = (intended_quote.side, intended_quote.tier)
            active_keys.add(key)
            self._sync_quote(intended_quote)

        for key, state in list(self._quote_states.items()):
            if key not in active_keys:
                self._cancel_state(state, reason="quote_disabled")

    def _simulate_crossed_fills(self) -> None:
        if self.engine is None:
            return

        for state in self._quote_states.values():
            order = self._get_live_order(state)
            if order is None or order.remaining <= 1e-9:
                continue
            if order.active_time > self.engine.clock.now():
                continue

            if state.side == OrderSide.BUY and self._book.best_ask > 0 and self._book.best_ask <= order.price:
                fillable = self._fillable_buy_size(order)
                if fillable <= 0:
                    self._cancel_state(state, reason="inventory_cap_reached")
                    continue
                self.engine.simulate_fill(
                    order.order_id,
                    min(order.remaining, fillable),
                    order.price,
                    is_maker=True,
                )
            elif state.side == OrderSide.SELL and self._book.best_bid > 0 and self._book.best_bid >= order.price:
                fillable = self._fillable_sell_size(order)
                if fillable <= 0:
                    self._cancel_state(state, reason="inventory_depleted")
                    continue
                self.engine.simulate_fill(
                    order.order_id,
                    min(order.remaining, fillable),
                    order.price,
                    is_maker=True,
                )

    def _build_intended_quotes(self, best_bid: float, best_ask: float) -> list[_ReplayIntendedQuote]:
        intended_quotes: list[_ReplayIntendedQuote] = []
        reserved_buy_notional = self._reserved_buy_notional()
        reserved_sell_shares = self._reserved_sell_shares()

        tight_bid_size = self._quote_size_for_bid(
            best_bid,
            self._quote_size_usd,
            reserved_buy_notional=reserved_buy_notional,
        )
        tight_ask_size = self._quote_size_for_ask(
            best_ask,
            self._quote_size_usd,
            reserved_sell_shares=reserved_sell_shares,
        )
        if tight_bid_size > 0:
            intended_quotes.append(
                _ReplayIntendedQuote(OrderSide.BUY, _REPLAY_TIGHT_TIER, best_bid, tight_bid_size)
            )
            reserved_buy_notional += tight_bid_size * best_bid
        if tight_ask_size > 0:
            intended_quotes.append(
                _ReplayIntendedQuote(OrderSide.SELL, _REPLAY_TIGHT_TIER, best_ask, tight_ask_size)
            )
            reserved_sell_shares += tight_ask_size

        if not self._wide_tier_enabled:
            return intended_quotes

        wide_bid_price = best_bid * (1.0 - self._wide_spread_pct)
        wide_ask_price = best_ask * (1.0 + self._wide_spread_pct)
        wide_bid_size = self._quote_size_for_bid(
            best_bid,
            self._wide_size_usd,
            reserved_buy_notional=reserved_buy_notional,
        )
        wide_ask_size = self._quote_size_for_ask(
            best_ask,
            self._wide_size_usd,
            reserved_sell_shares=reserved_sell_shares,
        )
        if wide_bid_size > 0 and wide_bid_price > 0:
            intended_quotes.append(
                _ReplayIntendedQuote(OrderSide.BUY, _REPLAY_WIDE_TIER, wide_bid_price, wide_bid_size)
            )
            reserved_buy_notional += wide_bid_size * best_bid
        if wide_ask_size > 0 and wide_ask_price > 0:
            intended_quotes.append(
                _ReplayIntendedQuote(OrderSide.SELL, _REPLAY_WIDE_TIER, wide_ask_price, wide_ask_size)
            )
            reserved_sell_shares += wide_ask_size
        return intended_quotes

    def _sync_quote(self, intended_quote: _ReplayIntendedQuote) -> None:
        key = (intended_quote.side, intended_quote.tier)
        state = self._quote_states.setdefault(key, _ReplayQuoteState(intended_quote.side, intended_quote.tier))
        order = self._get_live_order(state)

        if intended_quote.target_size <= 0 or intended_quote.target_price <= 0:
            self._cancel_state(state, reason="no_target_size")
            return

        rounded_price = round(intended_quote.target_price, 2)
        rounded_size = round(intended_quote.target_size, 2)

        if order is not None:
            remaining = max(0.0, order.remaining)
            price_match = abs(order.price - rounded_price) < 1e-9
            size_match = abs(remaining - rounded_size) < 1e-9
            if price_match and size_match:
                return
            self._cancel_state(state, reason="requote")

        order = self.engine.submit_order(
            side=intended_quote.side,
            price=rounded_price,
            size=rounded_size,
            order_type="limit",
            post_only=True,
            asset_id=self._no_asset_id,
        )
        state.order_id = order.order_id
        self._order_to_key[order.order_id] = key

    def _is_market_toxic(self, intended_quotes: list[_ReplayIntendedQuote]) -> bool:
        depth_velocity = self._book.depth_velocity(self._depth_window_s)
        if depth_velocity is not None and depth_velocity <= -self._depth_evaporation_pct:
            return True

        checks = [
            (quote.target_price, quote.side, quote.target_size)
            for quote in intended_quotes
        ]
        for state in self._quote_states.values():
            order = self._get_live_order(state)
            if order is not None:
                checks.append((order.price, state.side, order.size))

        for price, side, size in checks:
            quote_side = "BUY" if side == OrderSide.BUY else "SELL"
            against_flow = self._book.opposing_ofi_at_price(price, quote_side)
            if against_flow >= size * self._toxic_ofi_ratio:
                return True
        return False

    def _get_live_order(self, state: _ReplayQuoteState) -> "SimOrder" | None:
        if state.order_id is None or self.engine is None:
            return None
        order = self.engine.matching_engine.get_order(state.order_id)
        if order is None or order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            if state.order_id is not None:
                self._order_to_key.pop(state.order_id, None)
            state.order_id = None
            return None
        return order

    def _cancel_all_quotes(self, *, reason: str) -> None:
        for state in self._quote_states.values():
            self._cancel_state(state, reason=reason)

    def _cancel_state(self, state: _ReplayQuoteState, *, reason: str) -> None:
        order = self._get_live_order(state)
        if order is None or self.engine is None:
            return
        self.engine.cancel_order(order.order_id)
        self._order_to_key.pop(order.order_id, None)
        state.order_id = None
        log.debug(
            "pure_mm_replay_quote_cancelled",
            asset_id=self._no_asset_id,
            side=state.side.value,
            tier=state.tier,
            reason=reason,
        )

    def _quote_size_for_bid(
        self,
        best_bid: float,
        size_usd: float,
        *,
        reserved_buy_notional: float = 0.0,
    ) -> float:
        if best_bid <= 0:
            return 0.0
        inventory_value = self._inventory * best_bid
        remaining_headroom_usd = self._inventory_cap_usd - inventory_value - reserved_buy_notional
        if remaining_headroom_usd <= 0:
            return 0.0
        allocatable_usd = min(size_usd, remaining_headroom_usd)
        return self._normalise_size(allocatable_usd / best_bid, best_bid)

    def _quote_size_for_ask(
        self,
        best_ask: float,
        size_usd: float,
        *,
        reserved_sell_shares: float = 0.0,
    ) -> float:
        if best_ask <= 0 or self._inventory <= 0:
            return 0.0
        available_inventory = max(0.0, self._inventory - reserved_sell_shares)
        if available_inventory <= 0:
            return 0.0
        return self._normalise_size(min(available_inventory, size_usd / best_ask), best_ask)

    def _reserved_buy_notional(self, *, exclude_order_id: str | None = None) -> float:
        if self.engine is None:
            return 0.0
        total = 0.0
        for order in self.engine.matching_engine.get_open_orders():
            if order.order_id == exclude_order_id or order.side != OrderSide.BUY:
                continue
            total += order.remaining * order.price
        return total

    def _reserved_sell_shares(self, *, exclude_order_id: str | None = None) -> float:
        if self.engine is None:
            return 0.0
        total = 0.0
        for order in self.engine.matching_engine.get_open_orders():
            if order.order_id == exclude_order_id or order.side != OrderSide.SELL:
                continue
            total += order.remaining
        return total

    def _fillable_buy_size(self, order: "SimOrder") -> float:
        book_price = max(self._book.best_bid, order.price, 1e-9)
        inventory_value = self._inventory_value(book_price)
        reserved_buy_notional = self._reserved_buy_notional(exclude_order_id=order.order_id)
        remaining_headroom_usd = self._inventory_cap_usd - inventory_value - reserved_buy_notional
        if remaining_headroom_usd <= 0:
            return 0.0
        return remaining_headroom_usd / book_price

    def _fillable_sell_size(self, order: "SimOrder") -> float:
        available_inventory = self._inventory - self._reserved_sell_shares(exclude_order_id=order.order_id)
        if available_inventory <= 0:
            return 0.0
        return min(order.remaining, available_inventory)

    def _inventory_value(self, mark_price: float) -> float:
        if mark_price <= 0:
            return 0.0
        return self._inventory * mark_price

    @staticmethod
    def _normalise_size(raw_size: float, price: float) -> float:
        if raw_size <= 0 or price <= 0:
            return 0.0
        min_size = max(EXCHANGE_MIN_SHARES, EXCHANGE_MIN_USD / price)
        if raw_size + 1e-9 < min_size:
            return 0.0
        return round(raw_size, 2)
