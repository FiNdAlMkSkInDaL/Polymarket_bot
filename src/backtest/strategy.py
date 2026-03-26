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
from datetime import datetime, timezone
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from src.core.config import EXCHANGE_MIN_SHARES, EXCHANGE_MIN_USD, StrategyParams, settings
from src.core.logger import get_logger
from src.backtest.matching_engine import Fill
from src.data.market_discovery import MarketInfo
from src.data.ohlcv import OHLCVAggregator, OHLCVBar
from src.data.websocket_client import TradeEvent
from src.signals.bayesian_arb import (
    BayesianArbDetector,
    BayesianArbRelationshipManager,
    BayesianRelationship,
)
from src.signals.contagion_arb import ContagionArbDetector, ContagionArbSignal
from src.signals.edge_filter import compute_edge_score
from src.signals.ofi_momentum import OFIMomentumDetector, compute_toxicity_size_multiplier
from src.signals.panic_detector import PanicDetector
from src.signals.resolution_probability import (
    CryptoPriceModel,
    GenericBayesianModel,
    ResolutionProbabilityEngine,
)
from src.signals.signal_framework import MetaStrategyController
from src.execution.momentum_taker import MomentumBracket
from src.execution.momentum_taker import draw_stochastic_momentum_bracket
from src.trading.executor import OrderSide, OrderStatus
from src.trading.portfolio_correlation import PortfolioCorrelationEngine

if TYPE_CHECKING:
    from src.backtest.data_loader import MarketEvent
    from src.backtest.matching_engine import Fill, SimOrder

log = get_logger(__name__)
OFI_REPLAY_SMART_PASSIVE_TIMEOUT_SECONDS = 15.0
OFI_REPLAY_TIME_STOP_VACUUM_RATIO = 0.35
OFI_REPLAY_TIME_STOP_RECOVERY_RATIO = 0.60
OFI_REPLAY_TIME_STOP_SPREAD_MULTIPLIER = 1.75
_REPLAY_TOP_DEPTH_EWMA_ALPHA = 0.2

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


def _parse_market_end_date(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


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

    self_aggregates_trades = True

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
        stochastic_seed: int | None = None,
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
        self._stochastic_seed = stochastic_seed
        self._rng = random.Random(stochastic_seed)

        # Aggregators and detector (created on init)
        self._yes_agg: OHLCVAggregator | None = None
        self._no_agg: OHLCVAggregator | None = None
        self._detector: PanicDetector | None = None
        self._ofi_detector: OFIMomentumDetector | None = None
        self._book: _ReplayOrderBook | None = None
        self._meta_controller = MetaStrategyController()

        # Signal cooldown tracking
        self._last_signal_time: float = 0.0
        self._cooldown_seconds: float = self._params.signal_cooldown_minutes * 60.0

        # Pending orders we've submitted: sim_order_id → context
        self._pending_entries: dict[str, dict] = {}
        self._pending_exits: dict[str, dict] = {}

        # Position tracking (simplified for backtest)
        self._positions: list[dict] = []   # all opened positions
        self._open_positions: dict[str, dict] = {}  # order_id → position context
        self._smart_passive_started: int = 0
        self._smart_passive_maker_filled: int = 0
        self._smart_passive_fallbacks: int = 0

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
        self._book = _ReplayOrderBook(self._no_asset_id)

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
        self._ofi_detector = OFIMomentumDetector(
            market_id=self._market_id,
            no_asset_id=self._no_asset_id,
            window_ms=self._params.window_ms,
            threshold=self._params.ofi_threshold,
            tvi_kappa=self._params.ofi_tvi_kappa,
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
            stochastic_seed=self._stochastic_seed,
        )

    def on_book_update(self, asset_id: str, snapshot: dict) -> None:
        """Route book updates to internal state.

        In the live bot, book updates feed into spread scoring, ghost
        liquidity detection, and TP rescaling. The adapter stores the
        latest snapshot for use in signal evaluation and sizing.
        """
        if asset_id != self._no_asset_id or self._ofi_detector is None or self.engine is None:
            return

        self._check_momentum_brackets(snapshot)

        bid_levels = snapshot.get("bid_levels") or []
        ask_levels = snapshot.get("ask_levels") or []
        if not bid_levels or not ask_levels:
            return

        best_bid = float(snapshot.get("best_bid", 0.0) or 0.0)
        best_ask = float(snapshot.get("best_ask", 0.0) or 0.0)
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return

        timestamp = float(snapshot.get("timestamp", 0.0) or 0.0)
        if self._book is not None and timestamp > 0:
            self._book.apply_event(
                {
                    "event_type": "l2_snapshot",
                    "bids": [
                        {"price": str(price), "size": str(size)}
                        for price, size in bid_levels
                    ],
                    "asks": [
                        {"price": str(price), "size": str(size)}
                        for price, size in ask_levels
                    ],
                },
                timestamp,
            )
        if timestamp - self._last_signal_time < self._cooldown_seconds:
            return
        if len(self._open_positions) >= self._params.max_open_positions:
            return

        sig = self._ofi_detector.generate_signal(
            no_book=self._book,
            trade_aggregator=self._no_agg,
            timestamp_ms=int(timestamp * 1000),
        )
        if sig is None or sig.direction != "BUY":
            return

        meta_decision = self._meta_controller.evaluate("ofi_momentum", 0.5)
        if meta_decision.vetoed:
            return

        trade_usd = min(
            self._initial_bankroll * self._params.kelly_fraction,
            self._params.max_trade_size_usd,
        )
        size = (trade_usd / best_ask) * meta_decision.weight if best_ask > 0 else 0.0
        toxicity_mult = compute_toxicity_size_multiplier(
            getattr(sig, "toxicity_index", 0.0),
            elevated_threshold=self._params.ofi_toxicity_scale_threshold,
            max_multiplier=self._params.ofi_toxicity_size_boost_max,
        )
        if toxicity_mult > 1.0:
            size *= toxicity_mult
        if size < 1:
            return

        order = self.engine.submit_order(
            side=OrderSide.BUY,
            price=best_ask,
            size=size,
            order_type="limit",
            post_only=False,
        )
        drawn_bracket = draw_stochastic_momentum_bracket(
            mean_max_hold_seconds=MomentumBracket().max_hold_seconds,
            rng=self._rng,
        )
        target_price = drawn_bracket.target_price(best_ask)

        self._last_signal_time = timestamp
        self._pending_entries[order.order_id] = {
            "order": order,
            "entry_price": best_ask,
            "target_price": target_price,
            "stop_price": drawn_bracket.stop_price(best_ask),
            "max_hold_seconds": drawn_bracket.max_hold_seconds,
            "drawn_tp": target_price,
            "drawn_stop": drawn_bracket.stop_price(best_ask),
            "drawn_time": drawn_bracket.max_hold_seconds,
            "drawn_tp_pct": drawn_bracket.take_profit_pct,
            "drawn_stop_pct": drawn_bracket.stop_loss_pct,
            "ofi": sig.ofi,
            "signal_source": "ofi_momentum",
        }

        log.debug(
            "adapter_ofi_entry_signal",
            order_id=order.order_id,
            ofi=round(sig.ofi, 4),
            entry_price=best_ask,
            target_price=target_price,
        )

    def on_trade(self, asset_id: str, trade: TradeEvent) -> None:
        """Feed trades into OHLCV aggregators.

        When a bar closes, evaluate the strategy's signal logic.
        """
        if asset_id == self._no_asset_id and self.engine is not None:
            self._check_momentum_brackets({
                "best_bid": self.engine.get_asset_best_bid(self._no_asset_id),
                "timestamp": trade.timestamp,
            })

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

            pos = {
                "entry_fill": fill,
                "entry_ctx": ctx,
                "exit_order_id": None,
                "entry_time": fill.timestamp,
                "target_price": float(ctx.get("target_price", 0.0) or 0.0),
                "stop_price": float(ctx.get("stop_price", 0.0) or 0.0),
                "max_hold_seconds": float(ctx.get("max_hold_seconds", 0.0) or 0.0),
                "drawn_tp": float(ctx.get("drawn_tp", 0.0) or 0.0),
                "drawn_stop": float(ctx.get("drawn_stop", 0.0) or 0.0),
                "drawn_time": float(ctx.get("drawn_time", 0.0) or 0.0),
                "drawn_tp_pct": float(ctx.get("drawn_tp_pct", 0.0) or 0.0),
                "drawn_stop_pct": float(ctx.get("drawn_stop_pct", 0.0) or 0.0),
                "signal_source": ctx.get("signal_source", ""),
                "exit_reason": "take_profit",
            }

            if ctx.get("signal_source") == "ofi_momentum":
                tracking_id = f"OFI-LOCAL-{oid}"
                self._open_positions[tracking_id] = pos
            else:
                target = ctx["target_price"]
                exit_order = self.engine.submit_order(
                    side=OrderSide.SELL,
                    price=target,
                    size=fill.size,
                    order_type="limit",
                    post_only=True,
                )
                pos["exit_order_id"] = exit_order.order_id
                self._pending_exits[exit_order.order_id] = pos
                self._open_positions[exit_order.order_id] = pos

            log.debug(
                "adapter_entry_filled",
                order_id=oid,
                entry_price=fill.price,
                target=pos.get("target_price", 0.0),
                exit_order=pos.get("exit_order_id"),
            )
            return

        # ── Exit fill ──────────────────────────────────────────────────
        if oid in self._pending_exits:
            pos = self._pending_exits.pop(oid)
            self._open_positions.pop(oid, None)

            if (
                pos.get("signal_source") == "ofi_momentum"
                and pos.get("exit_reason") == "time_stop"
                and pos.get("smart_passive_deadline", 0.0) > 0
                and fill.is_maker
            ):
                self._smart_passive_maker_filled += 1

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
                "exit_reason": pos.get("exit_reason", "take_profit"),
                "smart_passive_started_at": pos.get("smart_passive_started_at", 0.0),
                "drawn_tp": pos.get("drawn_tp", 0.0),
                "drawn_stop": pos.get("drawn_stop", 0.0),
                "drawn_time": pos.get("drawn_time", 0.0),
            })

            log.debug(
                "adapter_exit_filled",
                entry=entry_fill.price,
                exit=fill.price,
                pnl_net=round(pnl_net, 4),
            )

    def _check_momentum_brackets(self, snapshot: dict[str, Any]) -> None:
        """Force-close OFI momentum positions on stop-loss or time-stop."""
        if self.engine is None or not self._open_positions:
            return

        best_bid = float(
            snapshot.get("best_bid", 0.0) or self.engine.get_asset_best_bid(self._no_asset_id) or 0.0
        )
        best_ask = float(
            snapshot.get("best_ask", 0.0) or self.engine.get_asset_best_ask(self._no_asset_id) or 0.0
        )
        timestamp = float(snapshot.get("timestamp", 0.0) or 0.0)
        if timestamp <= 0 or (best_bid <= 0 and best_ask <= 0):
            return

        for exit_order_id, pos in list(self._open_positions.items()):
            if pos.get("signal_source") != "ofi_momentum":
                continue

            smart_passive_deadline = float(pos.get("smart_passive_deadline", 0.0) or 0.0)
            if smart_passive_deadline > 0:
                if timestamp >= smart_passive_deadline and best_bid > 0:
                    if self._should_suppress_replay_time_stop(best_bid, best_ask, pos):
                        continue
                    self._trigger_smart_passive_fallback(exit_order_id, pos, best_bid)
                continue

            target_price = float(pos.get("target_price", 0.0) or 0.0)
            if target_price > 0 and best_bid >= target_price:
                self._trigger_local_replay_exit(
                    tracking_id=exit_order_id,
                    pos=pos,
                    best_bid=best_bid,
                    reason="target",
                )
                continue

            stop_price = float(pos.get("stop_price", 0.0) or 0.0)
            max_hold_seconds = float(pos.get("max_hold_seconds", 0.0) or 0.0)
            entry_time = float(pos.get("entry_time", 0.0) or 0.0)

            if stop_price > 0 and best_bid <= stop_price:
                self._trigger_local_replay_exit(
                    tracking_id=exit_order_id,
                    pos=pos,
                    best_bid=best_bid,
                    reason="stop_loss",
                )
                continue

            if max_hold_seconds > 0 and timestamp - entry_time >= max_hold_seconds and best_ask > 0:
                if self._should_suppress_replay_time_stop(best_bid, best_ask, pos):
                    continue
                self._start_smart_passive_time_stop(exit_order_id, pos, best_ask, timestamp)

    def _should_suppress_replay_time_stop(
        self,
        best_bid: float,
        best_ask: float,
        pos: dict[str, Any],
    ) -> bool:
        if self._book is None:
            return False

        bid_depth, ask_depth = self._book.top_depths_usd()
        bid_baseline = self._book.top_depth_ewma("bid")
        ask_baseline = self._book.top_depth_ewma("ask")
        spread = max(0.0, best_ask - best_bid)
        baseline_spread = max(
            0.01,
            float(pos.get("entry_fill").price if pos.get("entry_fill") else 0.5)
            * float(pos.get("drawn_stop_pct", 0.0) or MomentumBracket().stop_loss_pct),
        )
        bid_vacuum = bid_depth > 0 and bid_baseline > 0 and bid_depth < bid_baseline * OFI_REPLAY_TIME_STOP_VACUUM_RATIO
        ask_vacuum = ask_depth > 0 and ask_baseline > 0 and ask_depth < ask_baseline * OFI_REPLAY_TIME_STOP_VACUUM_RATIO
        spread_blown_out = spread >= baseline_spread * OFI_REPLAY_TIME_STOP_SPREAD_MULTIPLIER

        if (bid_vacuum or ask_vacuum) and spread_blown_out:
            return True

        bid_recovered = bid_baseline <= 0 or bid_depth >= bid_baseline * OFI_REPLAY_TIME_STOP_RECOVERY_RATIO
        ask_recovered = ask_baseline <= 0 or ask_depth >= ask_baseline * OFI_REPLAY_TIME_STOP_RECOVERY_RATIO
        return not (bid_recovered and ask_recovered)

    def _trigger_local_replay_exit(
        self,
        *,
        tracking_id: str,
        pos: dict[str, Any],
        best_bid: float,
        reason: str,
    ) -> None:
        if self.engine is None:
            return

        self._open_positions.pop(tracking_id, None)
        taker_exit_order = self.engine.submit_order(
            side=OrderSide.SELL,
            price=best_bid,
            size=pos["entry_fill"].size,
            order_type="limit",
            post_only=False,
        )
        pos["exit_order_id"] = taker_exit_order.order_id
        pos["exit_reason"] = reason
        pos["smart_passive_deadline"] = 0.0
        self._pending_exits[taker_exit_order.order_id] = pos
        self._open_positions[taker_exit_order.order_id] = pos
        self.engine.simulate_fill(
            taker_exit_order.order_id,
            pos["entry_fill"].size,
            price=best_bid,
            is_maker=False,
        )

    def _start_smart_passive_time_stop(
        self,
        exit_order_id: str,
        pos: dict[str, Any],
        best_ask: float,
        timestamp: float,
    ) -> None:
        if self.engine is None:
            return

        if pos.get("exit_order_id"):
            self.engine.cancel_order(exit_order_id)
        self._pending_exits.pop(exit_order_id, None)
        self._open_positions.pop(exit_order_id, None)

        passive_exit_order = self.engine.submit_order(
            side=OrderSide.SELL,
            price=best_ask,
            size=pos["entry_fill"].size,
            order_type="limit",
            post_only=True,
        )

        pos["exit_order_id"] = passive_exit_order.order_id
        pos["exit_reason"] = "time_stop"
        pos["smart_passive_started_at"] = timestamp
        pos["smart_passive_deadline"] = timestamp + OFI_REPLAY_SMART_PASSIVE_TIMEOUT_SECONDS
        self._pending_exits[passive_exit_order.order_id] = pos
        self._open_positions[passive_exit_order.order_id] = pos
        self._smart_passive_started += 1

    def _trigger_smart_passive_fallback(
        self,
        exit_order_id: str,
        pos: dict[str, Any],
        best_bid: float,
    ) -> None:
        if self.engine is None:
            return

        if pos.get("exit_order_id"):
            self.engine.cancel_order(exit_order_id)
        self._pending_exits.pop(exit_order_id, None)
        self._open_positions.pop(exit_order_id, None)

        taker_exit_order = self.engine.submit_order(
            side=OrderSide.SELL,
            price=best_bid,
            size=pos["entry_fill"].size,
            order_type="limit",
            post_only=False,
        )
        pos["exit_order_id"] = taker_exit_order.order_id
        pos["exit_reason"] = "time_stop"
        pos["smart_passive_deadline"] = 0.0
        self._pending_exits[taker_exit_order.order_id] = pos
        self._open_positions[taker_exit_order.order_id] = pos
        self._smart_passive_fallbacks += 1
        self.engine.simulate_fill(
            taker_exit_order.order_id,
            pos["entry_fill"].size,
            price=best_bid,
            is_maker=False,
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
        if self.engine is not None and hasattr(self.engine, 'telemetry'):
            self.engine.telemetry.set_smart_passive_counters(
                started=self._smart_passive_started,
                maker_filled=self._smart_passive_maker_filled,
                fallback_triggered=self._smart_passive_fallbacks,
            )

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
            smart_passive_started=self._smart_passive_started,
            smart_passive_maker_filled=self._smart_passive_maker_filled,
            smart_passive_fallbacks=self._smart_passive_fallbacks,
        )


class ContagionReplayAdapter(StrategyABC):
    """Shared-market replay adapter for the Domino contagion arb.

    Reuses the production contagion detector, shared PCE correlations, and
    the RPE dislocation math inside a single replay strategy that sees the
    whole market universe at once.
    """

    self_aggregates_trades = True

    def __init__(
        self,
        market_configs: list[dict[str, Any]],
        *,
        fee_enabled: bool = True,
        initial_bankroll: float = 1000.0,
        params: StrategyParams | None = None,
    ) -> None:
        self._market_configs = [dict(config) for config in market_configs]
        self._fee_enabled = fee_enabled
        self._initial_bankroll = initial_bankroll
        self._params = params or StrategyParams()

        self._markets: dict[str, MarketInfo] = {}
        self._market_by_asset: dict[str, MarketInfo] = {}
        self._asset_role: dict[str, str] = {}
        self._yes_aggs: dict[str, OHLCVAggregator] = {}
        self._no_aggs: dict[str, OHLCVAggregator] = {}
        self._books: dict[str, _ReplayOrderBook] = {}
        self._external_prices: list[tuple[float, float]] = []

        self._pending_entries: dict[str, dict[str, Any]] = {}
        self._pending_exits: dict[str, dict[str, Any]] = {}
        self._open_positions: dict[str, dict[str, Any]] = {}
        self._positions: list[dict[str, Any]] = []

        self._pce = PortfolioCorrelationEngine(
            shadow_mode=True,
            max_portfolio_var_usd=self._params.pce_max_portfolio_var_usd,
            haircut_threshold=self._params.pce_correlation_haircut_threshold,
            structural_prior_weight=self._params.pce_structural_prior_weight,
            min_overlap_bars=self._params.pce_min_overlap_bars,
            holding_period_minutes=self._params.pce_holding_period_minutes,
            var_soft_cap=self._params.pce_var_soft_cap,
            var_bisect_iterations=self._params.pce_var_bisect_iterations,
        )
        self._rpe = ResolutionProbabilityEngine(
            models=[
                CryptoPriceModel(price_fn=self._latest_external_price),
                GenericBayesianModel(),
            ],
            shadow_mode=False,
        )
        self._contagion = ContagionArbDetector(
            self._pce,
            self._rpe,
            universe_size=self._params.max_active_l2_markets,
            min_correlation=self._params.contagion_arb_min_correlation,
            trigger_percentile=self._params.contagion_arb_trigger_percentile,
            min_history=self._params.contagion_arb_min_history,
            min_leader_shift=self._params.contagion_arb_min_leader_shift,
            min_residual_shift=self._params.contagion_arb_min_residual_shift,
            toxicity_impulse_scale=self._params.contagion_arb_toxicity_impulse_scale,
            cooldown_seconds=self._params.contagion_arb_cooldown_seconds,
            max_pairs_per_leader=self._params.contagion_arb_max_pairs_per_leader,
            shadow_mode=False,
            max_cross_book_desync_ms=self._params.max_cross_book_desync_ms,
            max_leader_age_ms=self._params.contagion_arb_max_leader_age_ms,
            max_lagger_age_ms=self._params.contagion_arb_max_lagger_age_ms,
            max_causal_lag_ms=self._params.contagion_arb_max_causal_lag_ms,
            allow_negative_lag=self._params.contagion_arb_allow_negative_lag,
        )

    def detector_diagnostics(self) -> dict[str, Any]:
        return self._contagion.diagnostics_snapshot()

    def on_init(self) -> None:
        for index, config in enumerate(self._market_configs):
            market_id = str(config.get("market_id") or f"CONTAGION_{index}").strip()
            yes_asset_id = str(config.get("yes_asset_id") or config.get("yes_id") or "").strip()
            no_asset_id = str(config.get("no_asset_id") or config.get("no_id") or "").strip()
            if not market_id or not yes_asset_id or not no_asset_id:
                continue

            market = MarketInfo(
                condition_id=market_id,
                question=str(config.get("question") or market_id),
                yes_token_id=yes_asset_id,
                no_token_id=no_asset_id,
                daily_volume_usd=float(config.get("daily_volume_usd", 0.0) or 0.0),
                end_date=None,
                active=bool(config.get("active", True)),
                event_id=str(config.get("event_id") or config.get("group") or ""),
                liquidity_usd=float(config.get("liquidity_usd", 0.0) or 0.0),
                score=float(config.get("score", 0.0) or 0.0),
                accepting_orders=bool(config.get("accepting_orders", True)),
                tags=str(config.get("tags") or config.get("theme") or ""),
                neg_risk=bool(config.get("neg_risk", False)),
            )
            self._markets[market_id] = market
            self._market_by_asset[yes_asset_id] = market
            self._market_by_asset[no_asset_id] = market
            self._asset_role[yes_asset_id] = "yes"
            self._asset_role[no_asset_id] = "no"
            self._yes_aggs[yes_asset_id] = OHLCVAggregator(yes_asset_id)
            self._no_aggs[no_asset_id] = OHLCVAggregator(no_asset_id)
            self._books[yes_asset_id] = _ReplayOrderBook(yes_asset_id)
            self._books[no_asset_id] = _ReplayOrderBook(no_asset_id)

            self._pce.register_market(
                market_id=market.condition_id,
                event_id=market.event_id or market.condition_id,
                tags=market.tags or "replay",
                aggregator=self._yes_aggs[yes_asset_id],
            )
            self._contagion.register_market(market)

        log.info("contagion_replay_adapter_init", markets=len(self._markets))

    def on_book_update(self, asset_id: str, snapshot: dict) -> None:
        market = self._market_by_asset.get(asset_id)
        book = self._books.get(asset_id)
        if market is None or book is None:
            return

        bid_levels = snapshot.get("bid_levels") or []
        ask_levels = snapshot.get("ask_levels") or []
        timestamp = float(snapshot.get("timestamp", 0.0) or 0.0)
        if timestamp <= 0 or not bid_levels or not ask_levels:
            return

        book.apply_event(
            {
                "event_type": "l2_snapshot",
                "bids": [{"price": str(price), "size": str(size)} for price, size in bid_levels],
                "asks": [{"price": str(price), "size": str(size)} for price, size in ask_levels],
            },
            timestamp,
        )

        self._check_contagion_exits(asset_id, timestamp)
        self._evaluate_contagion_market(market.condition_id, timestamp)

    def on_trade(self, asset_id: str, trade: TradeEvent) -> None:
        role = self._asset_role.get(asset_id)
        if role == "yes":
            self._yes_aggs[asset_id].on_trade(trade)
        elif role == "no":
            self._no_aggs[asset_id].on_trade(trade)

    def on_fill(self, fill: "Fill") -> None:
        order_id = fill.order_id

        if order_id in self._pending_entries:
            ctx = self._pending_entries.pop(order_id)
            order = self.engine.matching_engine.get_order(order_id) if self.engine is not None else None
            if order is not None and order.remaining > 1e-9:
                self._pending_entries[order_id] = ctx
                return

            self._open_positions[order_id] = {
                "entry_fill": fill,
                "entry_time": fill.timestamp,
                "asset_id": ctx["asset_id"],
                "market_id": ctx["market_id"],
                "target_price": ctx["target_price"],
                "stop_price": ctx["stop_price"],
                "max_hold_seconds": ctx["max_hold_seconds"],
                "signal_source": ctx["signal_source"],
                "exit_reason": "take_profit",
            }
            return

        if order_id in self._pending_exits:
            pos = self._pending_exits.pop(order_id)
            entry_fill = pos["entry_fill"]
            pnl = (fill.price - entry_fill.price) * fill.size
            pnl_net = pnl - fill.fee - entry_fill.fee
            self._positions.append(
                {
                    "entry_price": entry_fill.price,
                    "exit_price": fill.price,
                    "size": fill.size,
                    "pnl_gross": pnl,
                    "pnl_net": pnl_net,
                    "entry_time": pos["entry_time"],
                    "exit_time": fill.timestamp,
                    "entry_fee": entry_fill.fee,
                    "exit_fee": fill.fee,
                    "exit_reason": pos.get("exit_reason", "take_profit"),
                    "signal_source": pos.get("signal_source", "contagion_arb"),
                }
            )

    def on_external_price(self, asset_id: str, price: float, timestamp: float) -> None:
        del asset_id
        if price > 0:
            self._external_prices.append((timestamp, price))
            if len(self._external_prices) > 500:
                self._external_prices = self._external_prices[-500:]

    def on_end(self) -> None:
        if self.engine is not None:
            for entry_order_id, pos in list(self._open_positions.items()):
                asset_id = str(pos.get("asset_id", ""))
                book = self._books.get(asset_id)
                if book is None or not book.has_data:
                    continue
                best_bid = book.best_bid
                if best_bid <= 0:
                    continue
                pos["exit_reason"] = "end_of_replay"
                exit_order = self.engine.submit_order(
                    side=OrderSide.SELL,
                    price=best_bid,
                    size=pos["entry_fill"].size,
                    order_type="limit",
                    post_only=False,
                    asset_id=asset_id,
                )
                self._pending_exits[exit_order.order_id] = pos
                self._open_positions.pop(entry_order_id, None)
                self._flush_pending_orders(asset_id, self.engine.clock.now())

        for market_id in list(self._markets):
            self._pce.unregister_market(market_id)

        total_pnl = sum(pos.get("pnl_net", 0.0) for pos in self._positions)
        detector_diagnostics = self._contagion.diagnostics_snapshot()
        log.info(
            "contagion_replay_adapter_summary",
            positions=len(self._positions),
            open_remaining=len(self._open_positions),
            total_pnl_net=round(total_pnl, 4),
            detector_diagnostics=detector_diagnostics,
        )

    def _latest_external_price(self) -> float | None:
        if not self._external_prices:
            return None
        return float(self._external_prices[-1][1])

    def _evaluate_contagion_market(self, market_id: str, timestamp: float) -> None:
        market = self._markets.get(market_id)
        if market is None or self.engine is None:
            return

        yes_book = self._books.get(market.yes_token_id)
        no_book = self._books.get(market.no_token_id)
        if yes_book is None or no_book is None or not yes_book.has_data or not no_book.has_data:
            return

        yes_snapshot = yes_book.snapshot()
        no_snapshot = no_book.snapshot()
        yes_bid = float(getattr(yes_snapshot, "best_bid", 0.0) or 0.0)
        yes_ask = float(getattr(yes_snapshot, "best_ask", 0.0) or 0.0)
        if yes_bid <= 0 or yes_ask <= 0 or yes_ask <= yes_bid:
            return

        self._pce.refresh_correlations()
        signals = self._contagion.evaluate_market(
            market=market,
            yes_price=(yes_bid + yes_ask) / 2.0,
            yes_buy_toxicity=yes_book.toxicity_index("BUY"),
            no_buy_toxicity=no_book.toxicity_index("BUY"),
            timestamp=timestamp,
            universe=list(self._markets.values()),
            book_snapshots=(yes_snapshot, no_snapshot),
        )
        for signal in signals:
            self._open_contagion_position(signal, timestamp)

    def _open_contagion_position(
        self,
        signal: ContagionArbSignal,
        timestamp: float,
    ) -> None:
        if self.engine is None:
            return

        market = self._markets.get(signal.lagging_market_id)
        if market is None or not market.accepting_orders:
            return
        if any(pos.get("market_id") == market.condition_id for pos in self._open_positions.values()):
            return

        asset_id = signal.lagging_asset_id
        book = self._books.get(asset_id)
        if book is None or not book.has_data or book.best_bid <= 0 or book.best_ask <= 0:
            return

        agg = self._yes_aggs.get(asset_id) or self._no_aggs.get(asset_id)
        if agg is None or agg.last_trade_time <= 0:
            return
        if (timestamp - agg.last_trade_time) > self._params.contagion_arb_max_last_trade_age_s:
            return

        spread_pct = ((book.best_ask - book.best_bid) / book.best_ask) * 100.0
        if spread_pct > self._params.contagion_arb_max_lagging_spread_pct:
            return

        entry_price = round(book.best_ask, 2)
        if entry_price <= 0:
            return

        trade_usd = min(
            self._initial_bankroll * self._params.kelly_fraction,
            self._params.max_trade_size_usd,
        )
        size = trade_usd / entry_price if entry_price > 0 else 0.0
        if size < 1:
            return

        fair_value = signal.implied_probability
        if signal.direction == "buy_no":
            fair_value = max(0.01, min(0.99, 1.0 - signal.implied_probability))
        target_price = round(max(entry_price + 0.01, min(0.99, fair_value)), 2)
        stop_price = round(max(0.01, entry_price * (1.0 - self._params.stop_loss_pct)), 2)

        order = self.engine.submit_order(
            side=OrderSide.BUY,
            price=entry_price,
            size=size,
            order_type="limit",
            post_only=False,
            asset_id=asset_id,
        )
        self._pending_entries[order.order_id] = {
            "asset_id": asset_id,
            "market_id": market.condition_id,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_seconds": max(60.0, self._params.contagion_arb_cooldown_seconds),
            "signal_source": signal.signal_source,
        }
        self._flush_pending_orders(asset_id, timestamp)

    def _check_contagion_exits(self, asset_id: str, timestamp: float) -> None:
        if self.engine is None:
            return

        book = self._books.get(asset_id)
        if book is None or not book.has_data or book.best_bid <= 0:
            return

        for entry_order_id, pos in list(self._open_positions.items()):
            if pos.get("asset_id") != asset_id:
                continue

            best_bid = round(book.best_bid, 2)
            exit_reason = ""
            if best_bid >= float(pos.get("target_price", 0.0) or 0.0):
                exit_reason = "take_profit"
            elif best_bid <= float(pos.get("stop_price", 0.0) or 0.0):
                exit_reason = "stop_loss"
            elif timestamp - float(pos.get("entry_time", 0.0) or 0.0) >= float(pos.get("max_hold_seconds", 0.0) or 0.0):
                exit_reason = "time_stop"

            if not exit_reason:
                continue

            exit_order = self.engine.submit_order(
                side=OrderSide.SELL,
                price=best_bid,
                size=pos["entry_fill"].size,
                order_type="limit",
                post_only=False,
                asset_id=asset_id,
            )
            pos["exit_reason"] = exit_reason
            self._pending_exits[exit_order.order_id] = pos
            self._open_positions.pop(entry_order_id, None)
            self._flush_pending_orders(asset_id, timestamp)

    def _flush_pending_orders(self, asset_id: str, timestamp: float) -> None:
        if self.engine is None:
            return

        self._sync_matching_engine_book(asset_id)
        current_time = timestamp + (self.engine.config.latency_ms / 1000.0) + 1e-6
        fills = self.engine.matching_engine.activate_pending_orders(current_time)
        if fills:
            self.engine._process_fills(fills)

    def _sync_matching_engine_book(self, asset_id: str) -> None:
        if self.engine is None:
            return

        book = self._books.get(asset_id)
        if book is None or not book.has_data:
            return

        self.engine.matching_engine.on_book_update(
            {
                "event_type": "book_snapshot",
                "bids": [{"price": level.price, "size": level.size} for level in book.levels("bid")],
                "asks": [{"price": level.price, "size": level.size} for level in book.levels("ask")],
            },
            current_time=self.engine.clock.now(),
        )


class BayesianReplayAdapter(StrategyABC):
    """Shared-market replay adapter for SI-10 Bayesian joint-probability arb.

    The replay books the detector's deterministic Dutch-book payout immediately
    after a signal so WFO optimises the live economic gates rather than a
    separate liquidation model.
    """

    self_aggregates_trades = True

    def __init__(
        self,
        market_configs: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        *,
        fee_enabled: bool = True,
        initial_bankroll: float = 1000.0,
        params: StrategyParams | None = None,
    ) -> None:
        self._market_configs = [dict(config) for config in market_configs]
        self._relationship_configs = [dict(item) for item in relationships]
        self._fee_enabled = fee_enabled
        self._initial_bankroll = initial_bankroll
        self._params = params or StrategyParams()

        self._markets: dict[str, MarketInfo] = {}
        self._books: dict[str, _ReplayOrderBook] = {}
        self._cluster_mgr = BayesianArbRelationshipManager(
            relationships=self._build_relationships(self._relationship_configs)
        )
        self._detector: BayesianArbDetector | None = None
        self._cluster_by_asset: dict[str, set[str]] = {}
        self._cluster_lookup: dict[str, Any] = {}
        self._last_signature: dict[str, tuple[Any, ...]] = {}
        self._last_signal_ts: dict[str, float] = {}
        self._trade_counter: int = 0

    def on_init(self) -> None:
        for index, config in enumerate(self._market_configs):
            market_id = str(config.get("market_id") or f"SI10_{index}").strip()
            yes_asset_id = str(config.get("yes_asset_id") or config.get("yes_id") or "").strip()
            no_asset_id = str(config.get("no_asset_id") or config.get("no_id") or "").strip()
            if not market_id or not yes_asset_id or not no_asset_id:
                continue

            market = MarketInfo(
                condition_id=market_id,
                question=str(config.get("question") or market_id),
                yes_token_id=yes_asset_id,
                no_token_id=no_asset_id,
                daily_volume_usd=float(config.get("daily_volume_usd", 0.0) or 0.0),
                end_date=_parse_market_end_date(
                    config.get("end_date")
                    or config.get("end_date_iso")
                    or config.get("end_time")
                ),
                active=bool(config.get("active", True)),
                event_id=str(config.get("event_id") or config.get("group") or ""),
                liquidity_usd=float(config.get("liquidity_usd", 0.0) or 0.0),
                score=float(config.get("score", 0.0) or 0.0),
                accepting_orders=bool(config.get("accepting_orders", True)),
                tags=str(config.get("tags") or config.get("theme") or ""),
                neg_risk=bool(config.get("neg_risk", False)),
            )
            self._markets[market_id] = market
            self._books[yes_asset_id] = _ReplayOrderBook(yes_asset_id)
            self._books[no_asset_id] = _ReplayOrderBook(no_asset_id)

        self._cluster_mgr.scan_clusters(list(self._markets.values()))
        self._cluster_lookup = {
            cluster.relationship_id: cluster for cluster in self._cluster_mgr.active_clusters
        }
        for cluster in self._cluster_mgr.active_clusters:
            for asset_id in (
                cluster.base_a.yes_token_id,
                cluster.base_a.no_token_id,
                cluster.base_b.yes_token_id,
                cluster.base_b.no_token_id,
                cluster.joint.yes_token_id,
                cluster.joint.no_token_id,
            ):
                self._cluster_by_asset.setdefault(asset_id, set()).add(cluster.relationship_id)

        self._detector = BayesianArbDetector(
            self._books,
            fee_enabled_resolver=self._is_fee_enabled,
            now_provider=self._now,
        )
        log.info(
            "bayesian_replay_adapter_init",
            markets=len(self._markets),
            relationships=len(self._cluster_lookup),
        )

    def on_book_update(self, asset_id: str, snapshot: dict) -> None:
        book = self._books.get(asset_id)
        if book is None:
            return

        bid_levels = snapshot.get("bid_levels") or []
        ask_levels = snapshot.get("ask_levels") or []
        timestamp = float(snapshot.get("timestamp", 0.0) or 0.0)
        if timestamp <= 0 or not bid_levels or not ask_levels:
            return

        book.apply_event(
            {
                "event_type": "l2_snapshot",
                "bids": [{"price": str(price), "size": str(size)} for price, size in bid_levels],
                "asks": [{"price": str(price), "size": str(size)} for price, size in ask_levels],
            },
            timestamp,
        )

        for relationship_id in self._cluster_by_asset.get(asset_id, ()):
            cluster = self._cluster_lookup.get(relationship_id)
            if cluster is None:
                continue
            self._evaluate_cluster(cluster, timestamp)

    def on_trade(self, asset_id: str, trade: TradeEvent) -> None:
        del asset_id
        del trade

    def on_fill(self, fill: "Fill") -> None:
        del fill

    def _evaluate_cluster(self, cluster: Any, timestamp: float) -> None:
        if self.engine is None or self._detector is None:
            return

        signal = self._detector.evaluate_cluster(cluster, wallet_balance=self.engine.cash)
        if signal is None:
            return

        signature = (
            signal.violation_type,
            tuple(
                sorted(
                    (
                        asset_id,
                        round(float(details.get("target_price", 0.0) or 0.0), 4),
                    )
                    for asset_id, details in signal.traded_leg_prices.items()
                )
            ),
            round(signal.net_edge_cents, 2),
        )
        last_signature = self._last_signature.get(cluster.relationship_id)
        last_ts = self._last_signal_ts.get(cluster.relationship_id, 0.0)
        if signature == last_signature and (timestamp - last_ts) < 60.0:
            return

        total_payout = signal.target_shares * signal.guaranteed_payout
        total_fee_usd = max(0.0, total_payout - signal.total_collateral - signal.net_ev_usd)
        total_entry_cost = signal.total_collateral + total_fee_usd
        if total_entry_cost > self.engine.cash + 1e-9:
            return

        self._trade_counter += 1
        avg_entry_price = total_entry_cost / max(signal.target_shares, 1e-9)
        exit_time = timestamp + (signal.days_to_resolution * 86_400.0)

        self.engine._cash -= total_entry_cost
        self.engine._cash += total_payout
        self.engine.telemetry.record_fill(
            Fill(
                order_id=f"SI10-{self._trade_counter}-ENTRY",
                price=avg_entry_price,
                size=signal.target_shares,
                fee=total_fee_usd,
                timestamp=timestamp,
                is_maker=False,
                side=OrderSide.BUY,
            ),
            mid_at_submission=avg_entry_price,
        )
        self.engine.telemetry.record_fill(
            Fill(
                order_id=f"SI10-{self._trade_counter}-EXIT",
                price=signal.guaranteed_payout,
                size=signal.target_shares,
                fee=0.0,
                timestamp=timestamp + 1e-6,
                is_maker=False,
                side=OrderSide.SELL,
            ),
            mid_at_submission=signal.guaranteed_payout,
        )
        self.engine.telemetry.record_round_trip(
            entry_price=avg_entry_price,
            exit_price=signal.guaranteed_payout,
            size=signal.target_shares,
            entry_fee=total_fee_usd,
            exit_fee=0.0,
            entry_time=timestamp,
            exit_time=exit_time,
        )
        self.engine.telemetry.record_equity(timestamp, self.engine.cash)

        self._last_signature[cluster.relationship_id] = signature
        self._last_signal_ts[cluster.relationship_id] = timestamp

        log.info(
            "bayesian_replay_trade",
            relationship_id=cluster.relationship_id,
            net_ev_usd=round(signal.net_ev_usd, 4),
            annualized_yield=round(signal.annualized_yield, 6),
            shares=signal.target_shares,
        )

    def _is_fee_enabled(self, market: MarketInfo) -> bool:
        if not self._fee_enabled:
            return False
        market_tags = (getattr(market, "tags", "") or "").lower()
        if not market_tags:
            return False
        fee_categories = {
            part.strip().lower()
            for part in (settings.strategy.fee_enabled_categories or "").split(",")
            if part.strip()
        }
        return any(category in market_tags for category in fee_categories)

    def _now(self) -> datetime:
        current_ts = self.engine.clock.now() if self.engine is not None else 0.0
        return datetime.fromtimestamp(current_ts, tz=timezone.utc)

    @staticmethod
    def _build_relationships(raw: list[dict[str, Any]]) -> list[BayesianRelationship]:
        relationships: list[BayesianRelationship] = []
        for index, item in enumerate(raw):
            base_a = str(item.get("base_a_condition_id") or item.get("base_a") or "").strip()
            base_b = str(item.get("base_b_condition_id") or item.get("base_b") or "").strip()
            joint = str(item.get("joint_condition_id") or item.get("joint") or "").strip()
            if not base_a or not base_b or not joint:
                continue
            relationship_id = str(
                item.get("relationship_id")
                or item.get("id")
                or f"SI10_REL_{index}"
            )
            relationships.append(
                BayesianRelationship(
                    relationship_id=relationship_id,
                    base_a_condition_id=base_a,
                    base_b_condition_id=base_b,
                    joint_condition_id=joint,
                    label=str(item.get("label") or item.get("name") or relationship_id),
                )
            )
        return relationships


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
        self._bid_depth_ewma: float = 0.0
        self._ask_depth_ewma: float = 0.0
        self._ofi_window: deque[tuple[float, float, float]] = deque(maxlen=500)
        self._ofi_price_window: deque[tuple[float, str, float, float]] = deque(maxlen=1000)
        self._toxicity_window_s: float = max(
            0.25,
            settings.strategy.toxicity_window_ms / 1000.0,
        )
        self._toxicity_window: deque[tuple[float, float, float, float, float]] = deque(maxlen=1000)
        self._rolling_bid_sweep_usd: float = 0.0
        self._rolling_ask_sweep_usd: float = 0.0
        self._toxicity_depth_norm: float = max(
            1e-6,
            settings.strategy.toxicity_depth_evaporation_pct,
        )
        self._toxicity_sweep_norm: float = max(
            1e-6,
            settings.strategy.toxicity_sweep_depth_ratio,
        )
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

    def levels(self, side: str, n: int = 5):
        if side.lower() in ("bid", "buy"):
            levels = sorted(self._bids.items(), reverse=True)[:n]
        else:
            levels = sorted(self._asks.items())[:n]
        return [SimpleNamespace(price=price, size=size) for price, size in levels]

    def snapshot(self):
        return SimpleNamespace(
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            timestamp=self._last_update,
        )

    def apply_event(self, event_data: dict[str, Any], timestamp: float) -> None:
        event_type = str(event_data.get("event_type", "")).lower()
        if event_type in ("book", "snapshot", "book_snapshot", "l2_snapshot"):
            self._apply_snapshot(event_data, timestamp)
        else:
            self._apply_delta(event_data, timestamp)
        self._last_update = timestamp
        self._record_depth(timestamp)

    def current_total_depth(self) -> float:
        bid_depth, ask_depth = self.top_depths_usd()
        return round(bid_depth + ask_depth, 2)

    def top_depths_usd(self) -> tuple[float, float]:
        bid_depth = sum(price * size for price, size in sorted(self._bids.items(), reverse=True)[:5])
        ask_depth = sum(price * size for price, size in sorted(self._asks.items())[:5])
        return round(bid_depth, 2), round(ask_depth, 2)

    def top_depth_ewma(self, side: str) -> float:
        bid_depth, ask_depth = self.top_depths_usd()
        if side.lower() in ("bid", "buy"):
            return round(self._bid_depth_ewma or bid_depth, 2)
        return round(self._ask_depth_ewma or ask_depth, 2)

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

    def toxicity_index(self, side: str = "BUY") -> float:
        return self.toxicity_metrics(side)["toxicity_index"]

    def toxicity_metrics(self, side: str = "BUY") -> dict[str, float]:
        self._prune_toxicity(self._last_update)
        if len(self._toxicity_window) < 2:
            return {
                "toxicity_index": 0.0,
                "toxicity_depth_evaporation": 0.0,
                "toxicity_sweep_ratio": 0.0,
            }

        baseline = self._toxicity_window[0]
        current = self._toxicity_window[-1]
        direction = side.upper()
        if direction == "SELL":
            baseline_depth = baseline[1]
            current_depth = current[1]
            sweep_ratio = self._rolling_bid_sweep_usd / max(1e-9, baseline_depth)
        else:
            baseline_depth = baseline[2]
            current_depth = current[2]
            sweep_ratio = self._rolling_ask_sweep_usd / max(1e-9, baseline_depth)

        if baseline_depth <= 0:
            return {
                "toxicity_index": 0.0,
                "toxicity_depth_evaporation": 0.0,
                "toxicity_sweep_ratio": 0.0,
            }

        depth_evaporation = max(0.0, (baseline_depth - current_depth) / baseline_depth)
        evap_score = min(1.0, depth_evaporation / self._toxicity_depth_norm)
        sweep_score = min(1.0, sweep_ratio / self._toxicity_sweep_norm)
        return {
            "toxicity_index": round(min(1.0, 0.5 * evap_score + 0.5 * sweep_score), 6),
            "toxicity_depth_evaporation": round(depth_evaporation, 6),
            "toxicity_sweep_ratio": round(sweep_ratio, 6),
        }

    def _apply_snapshot(self, data: dict[str, Any], timestamp: float) -> None:
        new_bids = self._parse_levels(data.get("bids") or [])
        new_asks = self._parse_levels(data.get("asks") or [])

        delta_bid_qty = 0.0
        delta_ask_qty = 0.0
        bid_sweep_usd = 0.0
        ask_sweep_usd = 0.0

        for price in set(self._bids) | set(new_bids):
            level_delta = new_bids.get(price, 0.0) - self._bids.get(price, 0.0)
            if level_delta != 0.0:
                self._ofi_price_window.append((timestamp, "BUY", price, level_delta))
                delta_bid_qty += level_delta
            if level_delta < 0.0:
                bid_sweep_usd += price * abs(level_delta)

        for price in set(self._asks) | set(new_asks):
            level_delta = new_asks.get(price, 0.0) - self._asks.get(price, 0.0)
            if level_delta != 0.0:
                self._ofi_price_window.append((timestamp, "SELL", price, level_delta))
                delta_ask_qty += level_delta
            if level_delta < 0.0:
                ask_sweep_usd += price * abs(level_delta)

        self._bids = new_bids
        self._asks = new_asks
        self._record_ofi(delta_bid_qty, delta_ask_qty, timestamp)
        self._record_toxicity(timestamp, bid_sweep_usd, ask_sweep_usd)

    def _apply_delta(self, data: dict[str, Any], timestamp: float) -> None:
        changes = data.get("changes") or data.get("price_changes") or data.get("data") or []
        if isinstance(changes, dict):
            changes = [changes]
        if not changes and data.get("price") is not None:
            changes = [data]

        delta_bid_qty = 0.0
        delta_ask_qty = 0.0
        bid_sweep_usd = 0.0
        ask_sweep_usd = 0.0
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
                if level_delta < 0.0:
                    bid_sweep_usd += price * abs(level_delta)
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
                if level_delta < 0.0:
                    ask_sweep_usd += price * abs(level_delta)

        self._record_ofi(delta_bid_qty, delta_ask_qty, timestamp)
        self._record_toxicity(timestamp, bid_sweep_usd, ask_sweep_usd)

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
        bid_depth, ask_depth = self.top_depths_usd()
        if self._bid_depth_ewma <= 0:
            self._bid_depth_ewma = bid_depth
        else:
            self._bid_depth_ewma = (
                _REPLAY_TOP_DEPTH_EWMA_ALPHA * bid_depth
                + (1.0 - _REPLAY_TOP_DEPTH_EWMA_ALPHA) * self._bid_depth_ewma
            )
        if self._ask_depth_ewma <= 0:
            self._ask_depth_ewma = ask_depth
        else:
            self._ask_depth_ewma = (
                _REPLAY_TOP_DEPTH_EWMA_ALPHA * ask_depth
                + (1.0 - _REPLAY_TOP_DEPTH_EWMA_ALPHA) * self._ask_depth_ewma
            )
        self._depth_history.append((timestamp, round(bid_depth + ask_depth, 2)))

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

    def _top_depths_usd(self) -> tuple[float, float]:
        return self.top_depths_usd()

    def _record_toxicity(self, timestamp: float, bid_sweep_usd: float, ask_sweep_usd: float) -> None:
        bid_depth, ask_depth = self._top_depths_usd()
        self._toxicity_window.append((timestamp, bid_depth, ask_depth, bid_sweep_usd, ask_sweep_usd))
        self._rolling_bid_sweep_usd += bid_sweep_usd
        self._rolling_ask_sweep_usd += ask_sweep_usd
        self._prune_toxicity(timestamp)

    def _prune_toxicity(self, timestamp: float) -> None:
        cutoff = timestamp - self._toxicity_window_s
        while self._toxicity_window and self._toxicity_window[0][0] < cutoff:
            _, _bid_depth, _ask_depth, bid_sweep_usd, ask_sweep_usd = self._toxicity_window.popleft()
            self._rolling_bid_sweep_usd -= bid_sweep_usd
            self._rolling_ask_sweep_usd -= ask_sweep_usd
        self._rolling_bid_sweep_usd = max(0.0, self._rolling_bid_sweep_usd)
        self._rolling_ask_sweep_usd = max(0.0, self._rolling_ask_sweep_usd)


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
        self._inventory_penalty_coef = max(0.0, self._params.pure_mm_inventory_penalty_coef)
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
        penalty_scale = self._inventory_penalty_scale(inventory_value)
        if penalty_scale <= 0:
            return 0.0
        allocatable_usd = min(size_usd * penalty_scale, remaining_headroom_usd)
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

    def _inventory_penalty_scale(self, inventory_value: float) -> float:
        if self._inventory_cap_usd <= 0:
            return 0.0
        fill_ratio = min(max(inventory_value / self._inventory_cap_usd, 0.0), 1.0)
        return max(0.0, (1.0 - fill_ratio) ** self._inventory_penalty_coef)

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
