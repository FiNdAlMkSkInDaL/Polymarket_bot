"""
Position manager — tracks open positions, orchestrates the entry → exit
lifecycle, and enforces risk limits.

Risk controls:
  - Configurable max open positions (``MAX_OPEN_POSITIONS``)
  - Per-market concentration limit (``MAX_POSITIONS_PER_MARKET``)
  - Per-event concentration limit (``MAX_POSITIONS_PER_EVENT``)
  - Daily loss circuit breaker (``DAILY_LOSS_LIMIT_USD``)
  - Max drawdown kill switch (``MAX_DRAWDOWN_CENTS``)
  - Trailing stop-loss (``STOP_LOSS_CENTS``)
  - Total exposure cap (``MAX_TOTAL_EXPOSURE_PCT``)
"""

from __future__ import annotations

import asyncio
import multiprocessing
import queue as _queue_mod
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.config import DeploymentEnv, settings
from src.core.guard import DeploymentGuard
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator
from src.data.orderbook import OrderbookTracker
from src.execution.momentum_taker import (
    DEFAULT_MOMENTUM_MAX_HOLD_SECONDS,
    DrawnMomentumBracket,
    MomentumBracket,
    MomentumTakerExecutor,
    draw_stochastic_momentum_bracket,
)
from src.execution.ofi_local_exit_monitor import OfiExitDecision, OfiLocalExitMonitor
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.signals.edge_filter import ConfluenceContext, compute_confluence_discount, compute_edge_score
from src.signals.iceberg_detector import IcebergDetector
from src.signals.ofi_momentum import OFIMomentumSignal, compute_toxicity_size_multiplier
from src.signals.panic_detector import PanicSignal
from src.signals.drift_signal import DriftSignal
from src.signals.signal_framework import BaseSignal, VacuumSignal
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.ensemble_risk import EnsembleRiskManager
from src.trading.fees import (
    compute_adaptive_stop_loss_cents,
    compute_adaptive_trailing_offset_cents,
    compute_net_pnl_cents,
    get_fee_rate,
)
from src.trading.sizer import (
    KellyResult,
    SizingResult,
    compute_depth_aware_size,
    compute_kelly_size,
)
from src.trading.take_profit import TakeProfitResult, compute_take_profit

log = get_logger(__name__)

LEGACY_SIGNAL_ZSCORE_THRESHOLD = 0.20
OFI_SMART_PASSIVE_TIMEOUT_SECONDS = 15.0
OFI_TIME_STOP_VACUUM_RATIO = 0.35
OFI_TIME_STOP_RECOVERY_RATIO = 0.60
OFI_TIME_STOP_SPREAD_MULTIPLIER = 1.75


class PositionState(str, Enum):
    ENTRY_PENDING = "ENTRY_PENDING"
    ENTRY_FILLED = "ENTRY_FILLED"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ComboState(str, Enum):
    """Lifecycle state for a multi-leg combinatorial arb position."""

    ENTRY_PENDING = "ENTRY_PENDING"     # orders placed, waiting for fills
    PARTIAL_FILL = "PARTIAL_FILL"       # some legs filled, others pending
    ALL_FILLED = "ALL_FILLED"           # all legs filled — risk-free
    ABANDONED = "ABANDONED"             # leg(s) timed out, unwinding
    CLOSED = "CLOSED"                   # resolved or emergency-drained


@dataclass
class Position:
    """Tracks a single round-trip trade (entry + exit)."""

    id: str
    market_id: str
    no_asset_id: str
    event_id: str = ""

    state: PositionState = PositionState.ENTRY_PENDING

    # Entry
    entry_order: Order | None = None
    entry_price: float = 0.0
    entry_size: float = 0.0          # intended (pre-fill) size
    filled_size: float = 0.0          # actual shares acquired (set on fill)
    entry_time: float = 0.0

    # Exit
    exit_order: Order | None = None
    target_price: float = 0.0
    stop_price: float = 0.0
    tp_result: TakeProfitResult | None = None
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    max_hold_seconds: float = 0.0
    smart_passive_exit_deadline: float = 0.0
    drawn_tp: float = 0.0
    drawn_stop: float = 0.0
    drawn_time: float = 0.0
    drawn_tp_pct: float = 0.0
    drawn_stop_pct: float = 0.0
    time_stop_delay_seconds: float = 0.0
    time_stop_suppression_count: int = 0
    exit_price_minus_drawn_stop_cents: float = 0.0

    # Bidirectional support (RPE)
    trade_asset_id: str = ""   # actual token traded (YES or NO); fallback to no_asset_id
    trade_side: str = "NO"     # "YES" or "NO" — which outcome token we bought
    yes_asset_id: str = ""     # YES token id (for RPE YES-side entries)

    # Signal metadata
    signal: BaseSignal | None = None
    signal_zscore: float | None = None
    signal_volume_ratio: float | None = None
    signal_whale_confluence: bool = False
    entry_toxicity_index: float = 0.0
    exit_toxicity_index: float = 0.0

    # Sizing metadata
    sizing: SizingResult | None = None
    kelly_result: KellyResult | None = None

    # Fee tracking (basis points)
    entry_fee_bps: int = 0
    exit_fee_bps: int = 0

    # Chaser task handles (for cancellation on timeout / shutdown)
    entry_chaser_task: asyncio.Task | None = field(default=None, repr=False)
    exit_chaser_task: asyncio.Task | None = field(default=None, repr=False)

    # PnL
    pnl_cents: float = 0.0

    # Fee tracking
    fee_enabled: bool = True
    sl_trigger_cents: float = 0.0     # fee-adaptive stop-loss threshold
    trailing_offset_cents: float = 0.0  # vol-adaptive trailing stop offset
    expected_net_target_per_share_cents: float = 0.0
    expected_net_target_minus_one_tick_per_share_cents: float = 0.0

    # V4: Probe sizing flag
    is_probe: bool = False

    # Pillar 16: Alpha-source attribution
    signal_type: str = ""         # "panic", "drift", "rpe", "stink_bid"
    strategy_source: str = ""
    meta_weight: float = 1.0     # SI-6 MetaStrategyController weight at entry

    # Pillar 11.3: initial vol multiplier used at open time so the
    # stop-loss monitor can decay it back toward 1.0 for stale trades.
    sl_vol_multiplier: float = 1.0

    created_at: float = field(default_factory=time.time)

    @property
    def effective_size(self) -> float:
        """Shares to use for exit orders / PnL — filled_size if set, else entry_size."""
        return self.filled_size if self.filled_size > 0 else self.entry_size


@dataclass
class ComboPosition:
    """Tracks a multi-leg combinatorial arbitrage position (SI-9).

    All legs target the same share count.  The combo is risk-free
    once ALL legs fill (guaranteed $1.00 payout at resolution).
    """

    combo_id: str
    event_id: str
    state: ComboState = ComboState.ENTRY_PENDING
    target_size: float = 0.0                           # uniform shares per leg
    maker_market_id: str = ""
    legs: dict[str, Position] = field(default_factory=dict)   # market_id → Position
    planned_collateral: float = 0.0
    sweep_triggered: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def maker_position(self) -> Position | None:
        return self.legs.get(self.maker_market_id)

    @property
    def taker_positions(self) -> list[Position]:
        return [
            pos for market_id, pos in self.legs.items()
            if market_id != self.maker_market_id
        ]

    @property
    def filled_legs(self) -> list[Position]:
        return [p for p in self.legs.values() if p.state == PositionState.ENTRY_FILLED]

    @property
    def pending_legs(self) -> list[Position]:
        return [p for p in self.legs.values() if p.state == PositionState.ENTRY_PENDING]

    @property
    def all_filled(self) -> bool:
        return bool(self.legs) and all(
            p.state == PositionState.ENTRY_FILLED for p in self.legs.values()
        )

    @property
    def total_collateral(self) -> float:
        if self.planned_collateral > 0:
            return self.planned_collateral
        return sum(p.entry_price * p.entry_size for p in self.legs.values())

    @property
    def pnl_cents_if_resolved(self) -> float:
        """Expected PnL if all legs fill and the event resolves."""
        sum_entry = sum(p.entry_price for p in self.legs.values())
        return (1.0 - sum_entry) * self.target_size * 100.0


class PositionManager:
    """Orchestrates the full lifecycle of mean-reversion trades.

    Workflow:
      1. A PanicSignal fires → open_position() is called.
      2. A GTC limit BUY for NO is placed at a discount.
      3. When the entry fills → compute take-profit → place limit SELL.
      4. When the exit fills or times out → position is closed.
    """

    # Maximum closed positions kept in memory for debugging
    _MAX_CLOSED_KEPT = 50

    def __init__(
        self,
        executor: OrderExecutor,
        *,
        max_open_positions: int | None = None,
        trade_store: Any | None = None,
        guard: DeploymentGuard | None = None,
        pce: Any | None = None,
        book_trackers: dict[str, Any] | None = None,
        maker_monitor: Any | None = None,
        iceberg_detectors: dict[str, IcebergDetector] | None = None,
        ensemble_risk: EnsembleRiskManager | None = None,
    ):
        self.executor = executor
        self.max_open = max_open_positions or settings.strategy.max_open_positions
        self._trade_store = trade_store
        self._guard = guard
        self._pce = pce  # PortfolioCorrelationEngine (Pillar 15)
        # Multicore: VaR gate via PCE worker queues (set by bot.py)
        self._var_request_q: multiprocessing.Queue | None = None
        self._var_response_q: multiprocessing.Queue | None = None
        self._book_trackers = book_trackers or {}  # asset_id → OrderbookTracker
        self._maker_monitor = maker_monitor  # AdverseSelectionMonitor (V1/V4)
        self._iceberg_detectors = iceberg_detectors or {}  # asset_id → IcebergDetector
        self._ensemble_risk = ensemble_risk or EnsembleRiskManager()
        self._stealth = None  # StealthExecutor — injected by bot.py when stealth_enabled
        self._ohlcv_aggs: dict[str, OHLCVAggregator] | None = None  # asset_id → aggregator; injected by bot.py
        self._positions: dict[str, Position] = {}
        self._next_id = 1
        self._wallet_balance_usd: Decimal | float = 0.0
        self._daily_pnl_cents: float = 0.0
        self._momentum_taker = MomentumTakerExecutor(executor)
        self._ofi_rng = random.Random()

        # ── Trade store stats cache (OE-1) ─────────────────────────────
        # Caches get_stats() results with a 5-second TTL to avoid
        # blocking aiosqlite reads on every signal evaluation.
        self._stats_cache: dict[str | None, dict[str, Any]] = {}
        self._stats_cache_time: dict[str | None, float] = {}
        self._STATS_CACHE_TTL: float = 5.0
        self._daily_pnl_date: str = ""  # YYYY-MM-DD
        self._cumulative_pnl_cents: float = 0.0
        self._peak_pnl_cents: float = 0.0
        self._max_drawdown_cents: float = 0.0
        self._circuit_breaker_tripped: bool = False
        self._last_entry_rejection_reasons: dict[tuple[str, str], str] = {}
        # Serialize entry attempts to prevent concurrent entries on
        # the same market/event from racing past risk gates.
        self._entry_lock = asyncio.Lock()
        # V4 Probe harvesting: track which probes have been scaled in already
        self._probe_scaled_set: set[str] = set()
        # V4 Probe harvesting: track which probes have been scaled in already
        self._probe_scaled_set: set[str] = set()
        # Per-market cooldown after stop-loss to prevent rapid re-entry
        self._stop_loss_cooldowns: dict[str, float] = {}
        # Combo order routing: order_id → (event_id, market_id)
        self._combo_order_map: dict[str, tuple[str, str]] = {}
        self._smart_passive_started_count: int = 0
        self._smart_passive_maker_filled_count: int = 0
        self._smart_passive_fallback_triggered_count: int = 0
        self._ofi_exit_monitors: dict[str, OfiLocalExitMonitor] = {}

    @property
    def ensemble_risk(self) -> EnsembleRiskManager:
        return self._ensemble_risk

    @staticmethod
    def _normalize_strategy_source(strategy_source: str | None) -> str:
        value = (strategy_source or "").strip()
        return value or "unknown_strategy"

    def _resolve_strategy_source(
        self,
        signal: BaseSignal | None = None,
        signal_metadata: dict[str, Any] | None = None,
        *,
        fallback: str = "unknown_strategy",
    ) -> str:
        meta = signal_metadata or {}
        source = meta.get("signal_source") or getattr(signal, "signal_source", "")
        return self._normalize_strategy_source(source or fallback)

    def _remember_entry_rejection_reason(
        self,
        strategy_source: str,
        market_id: str,
        reason: str,
    ) -> None:
        key = (self._normalize_strategy_source(strategy_source), str(market_id or ""))
        self._last_entry_rejection_reasons[key] = str(reason or "")

    def pop_last_entry_rejection_reason(
        self,
        strategy_source: str,
        market_id: str,
    ) -> str | None:
        key = (self._normalize_strategy_source(strategy_source), str(market_id or ""))
        return self._last_entry_rejection_reasons.pop(key, None)

    def _ensemble_allows_entry(
        self,
        *,
        market_id: str,
        direction: str,
        strategy_source: str,
        log_event: str,
    ) -> bool:
        allowed, reason = self._ensemble_risk.can_enter(
            market_id=market_id,
            strategy_source=strategy_source,
            direction=direction,
        )
        if allowed:
            return True
        log.info(
            log_event,
            market_id=market_id,
            direction=reason["direction"] if reason else direction,
            strategy_source=strategy_source,
            blocking_strategy=(reason or {}).get("blocking_strategy", ""),
        )
        return False

    def _ensemble_allows_batch_entry(
        self,
        *,
        strategy_source: str,
        exposures: list[tuple[str, str]],
        log_event: str,
        entry_id: str,
    ) -> bool:
        allowed, reason = self._ensemble_risk.can_enter_batch(
            strategy_source=strategy_source,
            exposures=exposures,
        )
        if allowed:
            return True
        log.info(
            log_event,
            entry_id=entry_id,
            market_id=(reason or {}).get("market_id", ""),
            direction=(reason or {}).get("direction", ""),
            strategy_source=strategy_source,
            blocking_strategy=(reason or {}).get("blocking_strategy", ""),
        )
        return False

    def _register_ensemble_position(self, pos: Position) -> None:
        if pos.filled_size <= 0 and pos.entry_size <= 0:
            return
        self._ensemble_risk.register_position(
            position_id=pos.id,
            market_id=pos.market_id,
            strategy_source=pos.strategy_source,
            direction=pos.trade_side or "NO",
        )

    def _release_ensemble_position(self, pos: Position) -> None:
        self._ensemble_risk.release_position(pos.id)

    # ── Wallet balance ─────────────────────────────────────────────────────
    def set_wallet_balance(self, usd: Decimal | float) -> None:
        self._wallet_balance_usd = usd

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_tripped

    def is_combo_order(self, order_id: str) -> bool:
        return order_id in self._combo_order_map

    @property
    def smart_passive_counters(self) -> dict[str, int]:
        return {
            "smart_passive_started": self._smart_passive_started_count,
            "maker_filled": self._smart_passive_maker_filled_count,
            "fallback_triggered": self._smart_passive_fallback_triggered_count,
        }

    def _register_combo_order(self, combo: ComboPosition, pos: Position) -> None:
        if pos.entry_order is None:
            return
        self._combo_order_map[pos.entry_order.order_id] = (combo.event_id, pos.market_id)

    def _unregister_combo_order(self, order_id: str | None) -> None:
        if order_id:
            self._combo_order_map.pop(order_id, None)

    def _sync_combo_state(self, combo: ComboPosition) -> None:
        if combo.state in (ComboState.ABANDONED, ComboState.CLOSED):
            return
        if combo.all_filled:
            combo.state = ComboState.ALL_FILLED
        elif combo.sweep_triggered or combo.filled_legs:
            combo.state = ComboState.PARTIAL_FILL
        else:
            combo.state = ComboState.ENTRY_PENDING

    async def _trigger_combo_taker_sweep(self, combo: ComboPosition) -> None:
        """Cross all deferred taker legs once the maker leg fully fills."""
        if combo.sweep_triggered:
            return

        strat = settings.strategy
        aggression = strat.si9_emergency_taker_max_cents / 100.0
        combo.sweep_triggered = True

        log.info(
            "combo_taker_sweep_triggered",
            combo_id=combo.combo_id,
            event_id=combo.event_id,
            n_takers=len(combo.taker_positions),
        )

        for pos in combo.taker_positions:
            asset_id = pos.trade_asset_id or pos.yes_asset_id
            book = self._book_trackers.get(asset_id)
            best_ask = book.best_ask if book is not None else 0.0
            reference_ask = best_ask if best_ask > 0 else pos.entry_price
            cross_price = round(min(0.99, max(reference_ask, 0.01) + aggression), 2)

            order = await self.executor.place_limit_order(
                market_id=pos.market_id,
                asset_id=asset_id,
                side=OrderSide.BUY,
                price=cross_price,
                size=pos.entry_size,
                post_only=False,
            )
            pos.entry_order = order
            pos.entry_time = time.time()
            pos.entry_price = cross_price
            self._positions[pos.id] = pos

            if order.status != OrderStatus.CANCELLED:
                self._register_combo_order(combo, pos)

            if self.executor.paper_mode and order.status == OrderStatus.LIVE:
                order.status = OrderStatus.FILLED
                order.filled_size = pos.entry_size
                order.filled_avg_price = cross_price
                order.updated_at = time.time()

            if order.status == OrderStatus.FILLED:
                pos.state = PositionState.ENTRY_FILLED
                pos.filled_size = order.filled_size or pos.entry_size
                pos.entry_price = order.filled_avg_price or cross_price
                self._register_ensemble_position(pos)
                log.info(
                    "combo_taker_leg_filled",
                    combo_id=combo.combo_id,
                    market_id=pos.market_id[:16],
                    price=pos.entry_price,
                )
            elif order.status == OrderStatus.CANCELLED:
                pos.exit_reason = "taker_sweep_rejected"
                log.warning(
                    "combo_taker_sweep_rejected",
                    combo_id=combo.combo_id,
                    market_id=pos.market_id[:16],
                    price=cross_price,
                )

        self._sync_combo_state(combo)

    async def on_combo_order_update(
        self,
        order: Order,
        combo_positions: dict[str, "ComboPosition"],
    ) -> bool:
        """Process maker/taker combo fills outside the single-position path."""
        combo_ref = self._combo_order_map.get(order.order_id)
        if combo_ref is None:
            return False

        event_id, market_id = combo_ref
        combo = combo_positions.get(event_id)
        if combo is None:
            return False

        pos = combo.legs.get(market_id)
        if pos is None:
            return False

        pos.entry_order = order
        if order.filled_size > 0:
            pos.filled_size = order.filled_size

        if order.status == OrderStatus.FILLED:
            pos.state = PositionState.ENTRY_FILLED
            pos.filled_size = order.filled_size or pos.entry_size
            pos.entry_price = order.filled_avg_price or order.price or pos.entry_price
            self._register_ensemble_position(pos)

            if market_id == combo.maker_market_id and not combo.sweep_triggered:
                log.info(
                    "combo_maker_leg_filled",
                    combo_id=combo.combo_id,
                    event_id=combo.event_id,
                    market_id=market_id[:16],
                )
                await self._trigger_combo_taker_sweep(combo)

            self._sync_combo_state(combo)
            return True

        self._sync_combo_state(combo)
        return True

    def reset_daily_pnl(self) -> None:
        """Call at UTC midnight to reset the daily loss tracker."""
        self._daily_pnl_cents = 0.0
        self._daily_pnl_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._circuit_breaker_tripped = False
        log.info("daily_pnl_reset")

    def is_stop_loss_cooled_down(self, market_id: str) -> bool:
        """Return True if enough time has passed since the last stop-loss on this market."""
        last_sl = self._stop_loss_cooldowns.get(market_id)
        if last_sl is None:
            return True
        return (time.time() - last_sl) >= settings.strategy.stop_loss_cooldown_s

    # ── Cached trade-store stats ───────────────────────────────────────────
    async def _get_cached_stats(self, signal_type: str | None = None) -> dict[str, Any]:
        """Return trade-store stats with a 5-second TTL cache.

        Avoids a blocking aiosqlite read on every signal evaluation.
        Cache is invalidated whenever a trade is recorded.
        """
        now = time.time()
        cache_key = signal_type
        if (
            cache_key in self._stats_cache
            and (now - self._stats_cache_time.get(cache_key, 0.0)) < self._STATS_CACHE_TTL
        ):
            return self._stats_cache[cache_key]
        if self._trade_store is None:
            return {}
        try:
            if signal_type:
                stats = await self._trade_store.get_stats(signal_type=signal_type)
            else:
                stats = await self._trade_store.get_stats()
            self._stats_cache[cache_key] = stats
            self._stats_cache_time[cache_key] = now
        except Exception:
            log.warning("trade_store_stats_unavailable")
            if cache_key not in self._stats_cache:
                self._stats_cache[cache_key] = {}
        return self._stats_cache[cache_key]

    def _invalidate_stats_cache(self) -> None:
        """Invalidate the stats cache (call after recording a trade)."""
        self._stats_cache = {}
        self._stats_cache_time = {}

    # ── Pillar 16.2: Self-healing alpha throttle ─────────────────────────
    async def _compute_strategy_multiplier(self, signal_type: str) -> float:
        """Return a Kelly multiplier in (0, 1] based on rolling strategy EV.

        Penalty scale (requires ≥ 20 trades for activation):
          - rolling_ev < -2.0¢ (significant bleed) → 0.1  (90% reduction)
          - rolling_ev < 0                          → 0.5  (50% reduction)
          - otherwise                               → 1.0  (no penalty)
        """
        if self._trade_store is None or not signal_type:
            return 1.0
        try:
            rolling_ev, n_trades = await self._trade_store.get_strategy_expectancy(
                signal_type, window=50,
            )
        except Exception:
            return 1.0

        if n_trades < 20:
            return 1.0

        if rolling_ev < -2.0:
            mult = 0.1
        elif rolling_ev < 0:
            mult = 0.5
        else:
            return 1.0

        log.warning(
            "alpha_source_throttled",
            signal_type=signal_type,
            rolling_ev=round(rolling_ev, 3),
            n_trades=n_trades,
            strategy_mult=mult,
        )
        return mult

    # ── Multicore VaR gate wiring ────────────────────────────────────────
    def set_var_gate_queues(
        self,
        request_q: multiprocessing.Queue,
        response_q: multiprocessing.Queue,
    ) -> None:
        """Wire the PCE worker queues for remote VaR gating."""
        self._var_request_q = request_q
        self._var_response_q = response_q

    async def _check_var_remote(
        self,
        open_positions: list,
        market_id: str,
        max_trade: float,
        trade_direction: str,
    ) -> tuple[bool, float, bool]:
        """Send VaR check to PCE worker and await response (500ms timeout).

        Fail-closed: if the worker doesn't respond, block the trade.
        Returns (allowed, max_trade, var_was_bisected).
        """
        req_id = uuid.uuid4().hex[:12]
        positions_data = [
            {
                "market_id": p.market_id,
                "entry_price": p.entry_price,
                "size": p.entry_size,
                "filled_size": p.filled_size,
                "trade_asset_id": p.trade_asset_id,
                "no_asset_id": p.no_asset_id,
                "event_id": p.event_id,
                "trade_side": p.trade_side,
            }
            for p in open_positions
        ]
        try:
            self._var_request_q.put_nowait(
                ("check_var", req_id, positions_data, market_id, max_trade, trade_direction),
            )
        except _queue_mod.Full:
            log.warning("var_request_queue_full", market_id=market_id)
            return False, 0.0, False  # fail-closed
        # Poll response queue with 500ms total timeout to absorb short
        # VPS scheduling hiccups without tripping false fail-closed blocks.
        deadline = time.monotonic() + 0.50
        while time.monotonic() < deadline:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._var_response_q.get(timeout=0.05),
                )
                if msg[0] == "var_result" and msg[1] == req_id:
                    allowed = msg[2]
                    result_dict = msg[3]
                    if not allowed:
                        log.warning(
                            "pce_var_gate_blocked",
                            market=market_id,
                            portfolio_var=result_dict.get("portfolio_var_usd", 0),
                        )
                        return False, 0.0, False
                    capped = result_dict.get("capped_size", max_trade)
                    bisected = result_dict.get("bisected", False)
                    return True, capped, bisected
            except Exception:
                pass
        # Timeout: fail-closed
        log.warning("pce_var_gate_timeout", market=market_id)
        return False, 0.0, False

    # ── Shared risk gates ──────────────────────────────────────────────────
    async def _check_risk_gates(
        self, market_id: str, event_id: str = "", *, trade_direction: str = "NO"
    ) -> tuple[bool, float, bool]:
        """Run all risk gates shared between panic and RPE entries.

        Returns (passed: bool, max_trade_usd: float, var_was_bisected: bool).
        When passed is False, the caller must abort.  max_trade_usd is the
        capital-allocation cap for this entry attempt.  var_was_bisected
        indicates whether the PCE VaR soft-cap already reduced sizing via
        bisection (used to avoid double-penalization with the concentration
        haircut).
        """
        strat = settings.strategy

        # ── Circuit breaker check ──────────────────────────────────────────
        if self._circuit_breaker_tripped:
            log.warning("circuit_breaker_active", reason="daily_loss_or_drawdown")
            return False, 0.0, False

        # ── Daily loss check ───────────────────────────────────────────────
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_pnl_date != today:
            self.reset_daily_pnl()

        daily_loss_limit_cents = strat.daily_loss_limit_usd * 100
        if self._daily_pnl_cents <= -daily_loss_limit_cents:
            self._circuit_breaker_tripped = True
            log.warning(
                "daily_loss_limit_hit",
                daily_pnl=round(self._daily_pnl_cents, 2),
                limit=daily_loss_limit_cents,
            )
            return False, 0.0, False

        # ── Max drawdown check ─────────────────────────────────────────────
        if self._max_drawdown_cents >= strat.max_drawdown_cents:
            self._circuit_breaker_tripped = True
            log.warning(
                "max_drawdown_hit",
                drawdown=round(self._max_drawdown_cents, 2),
                limit=strat.max_drawdown_cents,
            )
            return False, 0.0, False

        # ── Risk gate: max open positions ──────────────────────────────────
        open_positions = self.get_open_positions()
        if len(open_positions) >= self.max_open:
            log.warning("max_positions_reached", open=len(open_positions))
            return False, 0.0, False

        # ── Risk gate: per-market concentration ────────────────────────────
        market_count = sum(
            1 for p in open_positions if p.market_id == market_id
        )
        if market_count >= strat.max_positions_per_market:
            log.warning(
                "per_market_limit",
                market=market_id,
                count=market_count,
                limit=strat.max_positions_per_market,
            )
            return False, 0.0, False

        # ── Risk gate: per-event concentration ─────────────────────────────
        if event_id:
            event_count = sum(
                1 for p in open_positions if p.event_id == event_id
            )
            if event_count >= strat.max_positions_per_event:
                log.warning(
                    "per_event_limit",
                    event_id=event_id,
                    count=event_count,
                    limit=strat.max_positions_per_event,
                )
                return False, 0.0, False

        # ── Risk gate: total exposure ──────────────────────────────────────
        total_exposure = sum(
            p.entry_price * p.entry_size for p in open_positions
        )
        max_exposure = self._wallet_balance_usd * strat.max_total_exposure_pct / 100.0
        if total_exposure >= max_exposure and max_exposure > 0:
            log.warning(
                "exposure_limit",
                exposure=round(total_exposure, 2),
                limit=round(max_exposure, 2),
            )
            return False, 0.0, False

        # ── Risk gate: capital allocation ──────────────────────────────────
        max_trade = min(
            strat.max_trade_size_usd,
            self._wallet_balance_usd * strat.max_wallet_risk_pct / 100.0,
        )
        if max_trade <= 0:
            log.warning("insufficient_balance", balance=self._wallet_balance_usd)
            return False, 0.0, False

        # ── Risk gate: Portfolio Correlation Engine (PCE) VaR ──────────────
        var_was_bisected = False
        if self._var_request_q is not None:
            # Multicore: VaR gate via PCE worker queue
            allowed, max_trade, var_was_bisected = await self._check_var_remote(
                open_positions, market_id, max_trade, trade_direction,
            )
            if not allowed:
                return False, 0.0, False
        elif self._pce is not None:
            if self._pce.var_soft_cap:
                # Soft cap: skip check_var_gate(); compute_var_sizing_cap
                # already evaluates VaR internally (avoids redundant O(N²))
                sizing_result = self._pce.compute_var_sizing_cap(
                    open_positions=open_positions,
                    proposed_market_id=market_id,
                    proposed_size_usd=max_trade,
                    proposed_direction=trade_direction,
                )
                if sizing_result.cap_usd < 1.0:
                    log.warning(
                        "pce_var_gate_blocked",
                        market=market_id,
                        portfolio_var=sizing_result.current_var,
                        threshold=self._pce.max_portfolio_var_usd,
                    )
                    return False, 0.0, False
                max_trade = min(max_trade, sizing_result.cap_usd)
                var_was_bisected = sizing_result.bisect_iterations > 0
            else:
                # Hard block mode: binary accept/reject
                allowed, var_result = self._pce.check_var_gate(
                    open_positions=open_positions,
                    proposed_market_id=market_id,
                    proposed_size_usd=max_trade,
                    proposed_direction=trade_direction,
                )
                if not allowed:
                    log.warning(
                        "pce_var_gate_blocked",
                        market=market_id,
                        portfolio_var=var_result.portfolio_var_usd,
                        threshold=self._pce.max_portfolio_var_usd,
                    )
                    return False, 0.0, False

        return True, max_trade, var_was_bisected

    # ── Open a new position on signal ──────────────────────────────────────
    async def open_position(
        self,
        signal: BaseSignal,
        no_aggregator: OHLCVAggregator,
        *,
        no_book: OrderbookTracker | None = None,
        event_id: str = "",
        days_to_resolution: int = 30,
        book_depth_ratio: float = 1.0,
        fee_enabled: bool = True,
        signal_metadata: dict | None = None,
    ) -> Position | None:
        """Attempt to open a mean-reversion position on a panic or drift signal.

        Parameters
        ----------
        signal:
            A :class:`~src.signals.signal_framework.BaseSignal` (either a
            :class:`~src.signals.panic_detector.PanicSignal` or a
            :class:`~src.signals.drift_signal.DriftSignal`).  Signal-type-
            specific fields (z-score, volume ratio, whale confluence) are
            extracted polymorphically inside ``_open_position_inner``.
        no_book:
            Live L2 order book for the NO token.  Used by the depth-aware
            sizer to cap order size.  Falls back to fixed sizing with
            a 50% haircut if ``None``.
        fee_enabled:
            Whether this market category charges dynamic fees (crypto/sports).
            Used to compute the fee-adaptive stop-loss and net PnL.
        """
        async with self._entry_lock:
            if self._is_momentum_signal(signal, signal_metadata):
                return await self._open_momentum_position_inner(
                    signal,
                    no_aggregator,
                    no_book=no_book,
                    event_id=event_id,
                    fee_enabled=fee_enabled,
                    signal_metadata=signal_metadata,
                )
            return await self._open_position_inner(
                signal, no_aggregator,
                no_book=no_book, event_id=event_id,
                days_to_resolution=days_to_resolution,
                book_depth_ratio=book_depth_ratio,
                fee_enabled=fee_enabled,
                signal_metadata=signal_metadata,
            )

    @staticmethod
    def _is_momentum_signal(
        signal: BaseSignal,
        signal_metadata: dict | None = None,
    ) -> bool:
        if getattr(signal, "signal_source", "") == "ofi_momentum":
            return True
        meta = signal_metadata or {}
        return meta.get("signal_source") == "ofi_momentum"

    async def _open_momentum_position_inner(
        self,
        signal: BaseSignal,
        no_aggregator: OHLCVAggregator,
        *,
        no_book: OrderbookTracker | None = None,
        event_id: str = "",
        fee_enabled: bool = True,
        signal_metadata: dict | None = None,
    ) -> Position | None:
        """Open a strict taker-routed OFI momentum position."""
        del no_aggregator
        strategy_source = self._resolve_strategy_source(
            signal,
            signal_metadata,
            fallback="ofi_momentum",
        )

        if not self._ensemble_allows_entry(
            market_id=signal.market_id,
            direction="NO",
            strategy_source=strategy_source,
            log_event="ensemble_risk_blocked_momentum_entry",
        ):
            return None

        passed, max_trade, var_was_bisected = await self._check_risk_gates(
            signal.market_id, event_id,
        )
        if not passed:
            return None

        best_ask = 0.0
        if no_book is not None and no_book.has_data:
            snap = no_book.snapshot()
            best_ask = getattr(snap, "best_ask", 0.0)
        if best_ask <= 0:
            best_ask = getattr(signal, "best_ask", 0.0) or signal.no_best_ask
        if best_ask <= 0 or best_ask >= 1:
            log.warning("momentum_entry_price_invalid", ask=best_ask, market=signal.market_id)
            return None

        if no_book is not None:
            sizing = compute_depth_aware_size(
                book=no_book,
                entry_price=best_ask,
                max_trade_usd=max_trade,
                side="BUY",
            )
        else:
            fallback_usd = max_trade * 0.50
            shares = round(fallback_usd / best_ask, 2)
            if shares < 1:
                shares = 0.0
                fallback_usd = 0.0
            sizing = SizingResult(
                size_usd=fallback_usd,
                size_shares=shares,
                available_liq_usd=0.0,
                method="fallback_no_book",
                capped=False,
            )

        meta = signal_metadata or {}
        strategy_source = self._resolve_strategy_source(signal, signal_metadata, fallback="panic")
        meta_weight = max(0.0, float(meta.get("meta_weight", 1.0)))
        entry_size = round(sizing.size_shares * meta_weight, 2)
        toxicity_index = float(meta.get("toxicity_index", 0.0) or 0.0)
        if toxicity_index <= 0.0 and no_book is not None and hasattr(no_book, "toxicity_metrics"):
            try:
                toxicity_index = float(
                    no_book.toxicity_metrics("BUY").get("toxicity_index", 0.0) or 0.0
                )
            except Exception:
                log.warning(
                    "momentum_toxicity_snapshot_failed",
                    market=signal.market_id,
                    exc_info=True,
                )
        if toxicity_index <= 0.0:
            toxicity_index = float(getattr(signal, "toxicity_index", 0.0) or 0.0)
        toxicity_index = max(0.0, toxicity_index)
        if toxicity_index >= settings.strategy.ofi_toxicity_veto_threshold:
            self._remember_entry_rejection_reason(
                strategy_source,
                signal.market_id,
                "toxicity_veto",
            )
            log.info(
                "momentum_toxicity_veto",
                market=signal.market_id,
                toxicity_index=round(toxicity_index, 4),
                threshold=round(settings.strategy.ofi_toxicity_veto_threshold, 4),
            )
            return None
        toxicity_size_mult = compute_toxicity_size_multiplier(
            toxicity_index,
            elevated_threshold=settings.strategy.ofi_toxicity_scale_threshold,
            min_multiplier=settings.strategy.ofi_toxicity_size_haircut_floor,
        )
        if toxicity_size_mult < 1.0:
            entry_size = round(entry_size * toxicity_size_mult, 2)
            log.info(
                "momentum_toxicity_size_haircut_applied",
                market=signal.market_id,
                toxicity_index=round(toxicity_index, 4),
                size_multiplier=round(toxicity_size_mult, 4),
                new_size=entry_size,
            )

        if self._pce is not None and not var_was_bisected:
            pce_haircut = self._pce.compute_concentration_haircut(
                signal.market_id, self.get_open_positions()
            )
            if pce_haircut < 1.0:
                entry_size = max(1.0, round(entry_size * pce_haircut, 2))
                log.info(
                    "momentum_pce_size_haircut_applied",
                    haircut=round(pce_haircut, 4),
                    new_size=entry_size,
                )
        elif self._pce is not None and var_was_bisected:
            log.info(
                "momentum_pce_haircut_skipped_bisected",
                market=signal.market_id,
                reason="var_bisection_already_applied",
            )

        if self._guard is not None:
            entry_size = self._guard.get_allowed_trade_shares(entry_size, best_ask)

        configured_max_hold_seconds = min(
            float(DEFAULT_MOMENTUM_MAX_HOLD_SECONDS),
            float(settings.strategy.ofi_momentum_max_hold_seconds),
        )
        if isinstance(signal, OFIMomentumSignal):
            max_hold_seconds = float(
                min(
                    float(signal.max_hold_seconds or configured_max_hold_seconds),
                    configured_max_hold_seconds,
                )
            )
        else:
            max_hold_seconds = float(
                min(
                    float(getattr(signal, "max_hold_seconds", 0) or configured_max_hold_seconds),
                    configured_max_hold_seconds,
                )
            )
        drawn_bracket = draw_stochastic_momentum_bracket(
            mean_take_profit_pct=settings.strategy.ofi_momentum_take_profit_pct,
            mean_stop_loss_pct=settings.strategy.ofi_momentum_stop_loss_pct,
            mean_max_hold_seconds=max_hold_seconds,
            rng=self._ofi_rng,
        )
        sl_trigger = drawn_bracket.stop_loss_cents(best_ask)
        stop_price = drawn_bracket.stop_price(best_ask)
        target_price = drawn_bracket.target_price(best_ask)
        max_hold_seconds = float(drawn_bracket.max_hold_seconds)

        if ((target_price - best_ask) * 100.0) < settings.strategy.ofi_min_target_edge_cents:
            self._remember_entry_rejection_reason(
                strategy_source,
                signal.market_id,
                "insufficient_edge",
            )
            log.info(
                "skip_entry_insufficient_edge",
                spread_cents=round((target_price - best_ask) * 100.0, 2),
                min_viable=round(settings.strategy.ofi_min_target_edge_cents, 2),
                exec_mode="taker",
                margin=settings.strategy.desired_margin_cents,
            )
            return None

        max_loss_cents = settings.strategy.max_loss_per_trade_cents
        if max_loss_cents > 0 and best_ask > 0 and sl_trigger > 0:
            max_shares_for_risk = max_loss_cents / sl_trigger
            if entry_size > max_shares_for_risk:
                entry_size = round(max_shares_for_risk, 2)
                log.info(
                    "momentum_dollar_risk_cap_applied",
                    max_loss_cents=max_loss_cents,
                    capped_size=entry_size,
                )

        if entry_size < 1:
            log.info(
                "skip_momentum_entry_insufficient_size",
                sizing_method=sizing.method,
                meta_weight=meta_weight,
            )
            return None

        entry_fee_frac = get_fee_rate(best_ask, fee_enabled=fee_enabled)
        entry_fee_bps = round(entry_fee_frac * 10000)
        exit_fee_frac = get_fee_rate(target_price, fee_enabled=True)
        expected_net_target_per_share_cents = max(
            0.0,
            ((target_price - best_ask) * 100.0) - (entry_fee_frac * 100.0) - (exit_fee_frac * 100.0),
        )
        expected_net_target_minus_one_tick_per_share_cents = max(
            0.0,
            expected_net_target_per_share_cents - 1.0,
        )

        pos_id = f"MOMO-{self._next_id}"
        self._next_id += 1

        order, entry_price = await self._momentum_taker.place_buy(
            market_id=signal.market_id,
            asset_id=signal.no_asset_id,
            size=entry_size,
            best_ask=best_ask,
        )

        pos = Position(
            id=pos_id,
            market_id=signal.market_id,
            no_asset_id=signal.no_asset_id,
            trade_asset_id=signal.no_asset_id,
            event_id=event_id,
            state=PositionState.ENTRY_PENDING,
            entry_order=order,
            entry_price=entry_price,
            entry_size=entry_size,
            entry_time=time.time(),
            signal=signal,
            signal_zscore=getattr(signal, "zscore", None),
            signal_volume_ratio=getattr(signal, "volume_ratio", None),
            signal_whale_confluence=bool(getattr(signal, "whale_confluence", False)),
            target_price=target_price,
            stop_price=stop_price,
            sizing=sizing,
            fee_enabled=fee_enabled,
            sl_trigger_cents=sl_trigger,
            trailing_offset_cents=0.0,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=0,
            signal_type="ofi_momentum",
            strategy_source=strategy_source,
            meta_weight=meta_weight,
            expected_net_target_per_share_cents=round(expected_net_target_per_share_cents, 4),
            expected_net_target_minus_one_tick_per_share_cents=round(
                expected_net_target_minus_one_tick_per_share_cents,
                4,
            ),
            max_hold_seconds=max_hold_seconds,
            entry_toxicity_index=round(toxicity_index, 6),
            drawn_tp=target_price,
            drawn_stop=stop_price,
            drawn_time=max_hold_seconds,
            drawn_tp_pct=drawn_bracket.take_profit_pct,
            drawn_stop_pct=drawn_bracket.stop_loss_pct,
        )
        self._positions[pos.id] = pos

        log.info(
            "momentum_position_opened",
            pos_id=pos.id,
            market=signal.market_id,
            entry=entry_price,
            size=entry_size,
            target=pos.target_price,
            stop=pos.stop_price,
            max_hold_seconds=max_hold_seconds,
        )

        if order.status == OrderStatus.FILLED:
            await self.on_entry_filled(pos)

        return pos

    async def _open_position_inner(
        self,
        signal: BaseSignal,
        no_aggregator: OHLCVAggregator,
        *,
        no_book: OrderbookTracker | None = None,
        event_id: str = "",
        days_to_resolution: int = 30,
        book_depth_ratio: float = 1.0,
        fee_enabled: bool = True,
        signal_metadata: dict | None = None,
    ) -> Position | None:
        """Inner implementation of open_position (runs under _entry_lock)."""
        strat = settings.strategy
        strategy_source = self._resolve_strategy_source(
            signal,
            signal_metadata,
            fallback="panic",
        )

        if not self._ensemble_allows_entry(
            market_id=signal.market_id,
            direction="NO",
            strategy_source=strategy_source,
            log_event="ensemble_risk_blocked_position_entry",
        ):
            return None

        passed, max_trade, var_was_bisected = await self._check_risk_gates(
            signal.market_id, event_id,
        )
        if not passed:
            return None

        # ── Polymorphic signal-field extraction ────────────────────────────
        # PanicSignal carries live z-score, volume ratio, and whale flag.
        # DriftSignal carries a cumulative displacement; map it to the same
        # semantic slots so downstream EQS / Kelly / TP logic is unchanged.
        if isinstance(signal, PanicSignal):
            _zscore: float = signal.zscore
            _volume_ratio: float = signal.volume_ratio
            _whale_confluence: bool = signal.whale_confluence
        else:
            # DriftSignal (and any future BaseSignal subclass)
            _zscore = abs(signal.displacement) if hasattr(signal, "displacement") else 0.0
            _volume_ratio = 1.0
            _whale_confluence = False

        # ── SI-2: Query iceberg detector for alpha signal ──────────────
        _iceberg_active = False
        ice_det = self._iceberg_detectors.get(signal.no_asset_id)
        if ice_det is not None:
            strongest = ice_det.strongest_iceberg("BUY")
            if strongest is not None and strongest.confidence >= strat.iceberg_peg_min_confidence:
                _iceberg_active = True

        # Entry price: undercut the best ask by 1¢ to ensure maker
        entry_price = round(signal.no_best_ask - 0.01, 2)
        if entry_price <= 0:
            log.warning("entry_price_invalid", ask=signal.no_best_ask)
            return None

        # ── Edge quality gate (information-theoretic) ──────────────────
        # Weighted geometric mean of regime entropy, fee efficiency,
        # tick viability, and signal quality.  Replaces the old naive
        # min/max price-range clamp with a principled EV gate.
        #
        # V1: Maker routing — entries priced at best_ask - 1¢ are maker
        # orders with 0 bps fee; use execution_mode="maker" for EQS.
        # V2: Confluence routing — multiple confirmed signals lower the
        # EQS threshold dynamically.
        exec_mode = "maker" if strat.maker_routing_enabled else "taker"

        # V1 Adverse-selection monitor: downgrade to taker if maker is suspended
        # for this specific market (statistical toxic-flow detection).
        if exec_mode == "maker" and self._maker_monitor is not None:
            if not self._maker_monitor.is_maker_allowed(signal.market_id):
                exec_mode = "taker"
                log.info(
                    "maker_downgraded_asl_suspension",
                    market_id=signal.market_id,
                )

        # Determine signal source for flaw guards
        meta = signal_metadata or {}
        is_drift_signal: bool = meta.get("signal_source") == "drift"

        # Build confluence context for dynamic threshold (V2)
        confluence = ConfluenceContext(
            whale_strong_confluence=_whale_confluence,
            spread_compressed=bool(meta.get("spread_compressed", False)),
            l2_reliable=(
                no_book is not None
                and no_book.has_data
                and getattr(no_book, "is_reliable", True)
            ),
            regime_mean_revert=bool(meta.get("regime_mean_revert", False)),
        )
        base_threshold = strat.min_edge_score
        # V1: maker discount on threshold
        if exec_mode == "maker":
            base_threshold = base_threshold * strat.maker_eqs_discount
        # V2: confluence discount on threshold; propagate flaw-guard flags
        eqs_threshold = compute_confluence_discount(
            confluence,
            base_threshold,
            is_drift_signal=is_drift_signal,      # Flaw 2: suppress regime double-count
            maker_routing_active=(exec_mode == "maker"),  # Flaw 1: tighter combined floor
        )

        edge = compute_edge_score(
            entry_price=entry_price,
            no_vwap=no_aggregator.rolling_vwap,
            zscore=_zscore,
            volume_ratio=_volume_ratio,
            whale_confluence=_whale_confluence,
            iceberg_active=_iceberg_active,
            fee_enabled=fee_enabled,
            current_ewma_vol=no_aggregator.rolling_volatility_ewma or None,
            execution_mode=exec_mode,
            min_score=eqs_threshold,
        )

        # V4: Probe sizing — sub-threshold entries at micro-size
        is_probe = False
        if not edge.viable:
            if (
                strat.probe_sizing_enabled
                and edge.score >= strat.probe_eqs_floor
                and edge.rejection_reason == "score_below_threshold"
            ):
                is_probe = True
                # Flaw 3: probes are sub-threshold data-gathering entries.
                # Confluence discounts (particularly the 5-pt whale discount)
                # must NOT lower the probe threshold — the probe is admitted
                # via the confluence-independent probe_eqs_floor, so applying
                # discounts would double-count unconfirmed signals.  Verify
                # that the raw score still clears the floor without confluence.
                if strat.probe_suppress_confluence and confluence.whale_strong_confluence:
                    raw_threshold = strat.min_edge_score  # no discounts at all
                    if edge.score < strat.probe_eqs_floor:
                        log.info(
                            "probe_rejected_stale_whale_confluence",
                            score=edge.score,
                            probe_eqs_floor=strat.probe_eqs_floor,
                        )
                        return None
                    log.info(
                        "probe_whale_confluence_suppressed",
                        score=edge.score,
                        probe_eqs_floor=strat.probe_eqs_floor,
                        raw_threshold=raw_threshold,
                    )
                log.info(
                    "probe_entry_accepted",
                    score=edge.score,
                    threshold=eqs_threshold,
                    floor=strat.probe_eqs_floor,
                )
            else:
                log.info(
                    "eqs_rejected",
                    market_id=signal.market_id,
                    score=round(edge.score, 4),
                    threshold=round(eqs_threshold, 4),
                    reason=edge.rejection_reason,
                    is_drift=is_drift_signal,
                    entry_price=entry_price,
                    exec_mode=exec_mode,
                    regime_q=round(edge.regime_quality, 4) if hasattr(edge, 'regime_quality') else None,
                    fee_eff=round(edge.fee_efficiency, 4) if hasattr(edge, 'fee_efficiency') else None,
                    signal_q=round(edge.signal_quality, 4) if hasattr(edge, 'signal_quality') else None,
                )
                return None

        # ── Depth-aware sizing (Pillar 2) ──────────────────────────────────
        if no_book is not None:
            sizing = compute_depth_aware_size(
                book=no_book,
                entry_price=entry_price,
                max_trade_usd=max_trade,
                side="BUY",
            )
        else:
            # Fallback: no book available — use fixed sizing with 50% haircut
            fallback_usd = max_trade * 0.50
            shares = round(fallback_usd / entry_price, 2)
            if shares < 1:
                shares = 0.0
                fallback_usd = 0.0
            from src.trading.sizer import SizingResult
            sizing = SizingResult(
                size_usd=fallback_usd,
                size_shares=shares,
                available_liq_usd=0.0,
                method="fallback_no_book",
                capped=False,
            )

        # ── Kelly criterion sizing (Pillar 3) ─────────────────────────────
        kelly_result: KellyResult | None = None
        win_rate = 0.0
        avg_win_cents = 0.0
        avg_loss_cents = 0.0
        total_trades = 0

        # ── Pillar 16: Derive signal type early for throttle gate ─────────
        _meta = signal_metadata or {}
        _signal_type = "drift" if _meta.get("signal_source") == "drift" else "panic"
        _meta_weight = float(_meta.get("meta_weight", 1.0))

        stats = await self._get_cached_stats()
        if stats:
            win_rate = stats.get("win_rate", 0.0)
            avg_win_cents = stats.get("avg_win_cents", 0.0)
            avg_loss_cents = stats.get("avg_loss_cents", 0.0)
            total_trades = stats.get("total_trades", 0)

        # Normalise signal strength to 0.0–1.0 for Kelly sizer.
        # For PanicSignal: map z-score excess above threshold to [0, 1].
        # For DriftSignal: use pre-normalised score directly.
        z_threshold = LEGACY_SIGNAL_ZSCORE_THRESHOLD
        signal_score = min(1.0, max(0.0, (_zscore - z_threshold) / z_threshold)) if z_threshold > 0 else 0.5

        # Inject rolling expectancy for adaptive cold-start sizing
        kelly_meta = dict(signal_metadata or {})
        # Inject decayed win rate for non-stationary Kelly (OE-4)
        decayed_wr = stats.get("decayed_win_rate", 0.0)
        if decayed_wr > 0:
            kelly_meta["decayed_win_rate"] = decayed_wr
        if self._trade_store is not None and total_trades > 0:
            try:
                rolling_exp = await self._trade_store.get_rolling_expectancy(
                    window=strat.cold_start_halt_window
                )
                kelly_meta["rolling_expectancy_cents"] = rolling_exp
            except Exception:
                pass

        # ── Pillar 16.2: Self-healing alpha throttle ──────────────────────
        strategy_mult = await self._compute_strategy_multiplier(_signal_type)

        kelly_result = compute_kelly_size(
            signal_score=signal_score,
            win_rate=win_rate,
            avg_win_cents=avg_win_cents,
            avg_loss_cents=avg_loss_cents,
            bankroll_usd=self._wallet_balance_usd,
            entry_price=entry_price,
            max_trade_usd=max_trade,
            book=no_book,
            signal_metadata=kelly_meta,
            total_trades=total_trades,
            _precomputed_depth_result=sizing,
            pce=self._pce,
            proposed_market_id=signal.market_id,
            open_positions=self.get_open_positions(),
            strategy_multiplier=strategy_mult,
        )

        # Reject if Kelly finds no edge (cold-start is allowed through)
        if kelly_result.method == "kelly_no_edge":
            log.info(
                "skip_entry_kelly_no_edge",
                estimated_p=kelly_result.estimated_p,
                adjusted_p=kelly_result.adjusted_p,
                uncertainty=kelly_result.uncertainty_penalty,
            )
            return None

        # Conservative sizing: min(depth-aware, kelly)
        entry_size = min(sizing.size_shares, kelly_result.size_shares)

        # ── V4: Probe sizing cap ──────────────────────────────────────────
        # Probe entries use micro-size (5% of standard Kelly, hard cap).
        # In PENNY_LIVE mode the $1 deployment cap clamps probe and full
        # sizes to the same value, destroying probe math.  Skip probes
        # entirely in PENNY_LIVE — defer to PRODUCTION runs.
        _is_penny_live = (
            self._guard is not None
            and self._guard.deployment_env == DeploymentEnv.PENNY_LIVE
        )
        if is_probe and _is_penny_live:
            is_probe = False
            log.info(
                "probe_sizing_skipped_penny_live",
                reason="penny_live_clamp_destroys_probe_math",
            )
        if is_probe:
            probe_max_shares = round(strat.probe_max_usd / entry_price, 2) if entry_price > 0 else 0.0
            probe_kelly_shares = round(entry_size * strat.probe_kelly_fraction, 2)
            entry_size = max(1.0, min(probe_kelly_shares, probe_max_shares))
            log.info(
                "probe_sizing_applied",
                probe_shares=entry_size,
                probe_max_usd=strat.probe_max_usd,
                probe_kelly_fraction=strat.probe_kelly_fraction,
            )

        # ── Spread signal sizing multiplier ────────────────────────────────
        # Spread-opportunity entries are lower conviction; apply the
        # explicit sizing multiplier set by the spread signal source.
        meta = signal_metadata or {}
        spread_mult = float(meta.get("spread_sizing_mult", 1.0))
        if spread_mult < 1.0:
            entry_size = max(1.0, round(entry_size * spread_mult, 2))
            log.info(
                "spread_sizing_mult_applied",
                multiplier=spread_mult,
                new_size=entry_size,
            )

        # ── PCE concentration haircut (Pillar 15) ─────────────────────────
        # Skip haircut when VaR bisection already reduced sizing to avoid
        # double-penalizing correlated entries (blueprint §III-B).
        if self._pce is not None and not var_was_bisected:
            pce_haircut = self._pce.compute_concentration_haircut(
                signal.market_id, self.get_open_positions()
            )
            if pce_haircut < 1.0:
                entry_size = max(1.0, round(entry_size * pce_haircut, 2))
                log.info(
                    "pce_size_haircut_applied",
                    haircut=round(pce_haircut, 4),
                    new_size=entry_size,
                )
        elif self._pce is not None and var_was_bisected:
            log.info(
                "pce_haircut_skipped_bisected",
                market=signal.market_id,
                reason="var_bisection_already_applied",
            )

        # ── Deployment guard: enforce phase-specific size cap ──────────────
        if self._guard is not None:
            entry_size = self._guard.get_allowed_trade_shares(
                entry_size, entry_price
            )

        if entry_size < 1:
            log.info(
                "skip_entry_insufficient_size",
                sizing_method=sizing.method,
                kelly_method=kelly_result.method,
                available_liq=sizing.available_liq_usd,
            )
            return None

        # Pre-check: will the take-profit be viable?
        actual_depth_ratio = book_depth_ratio
        if no_book is not None and no_book.has_data:
            actual_depth_ratio = no_book.book_depth_ratio

        # Derive fee bps from the dynamic fee curve for TP computation
        from src.trading.fees import get_fee_rate
        entry_fee_frac = get_fee_rate(entry_price, fee_enabled=fee_enabled)
        entry_fee_bps = round(entry_fee_frac * 10000)
        exit_fee_bps = 0  # maker exit → no taker fee

        tp = compute_take_profit(
            entry_price=entry_price,
            no_vwap=no_aggregator.rolling_vwap,
            realised_vol=no_aggregator.rolling_volatility,
            whale_confluence=_whale_confluence,
            iceberg_active=_iceberg_active,
            book_depth_ratio=actual_depth_ratio,
            days_to_resolution=days_to_resolution,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
            fee_enabled=fee_enabled,
        )
        if not tp.viable:
            log.info(
                "skip_entry_low_spread",
                spread=tp.spread_cents,
                min_required=strat.min_spread_cents,
            )
            return None

        # ── Minimum viable edge gate ───────────────────────────────────
        # Reject trades where TP spread cannot cover slippage + fees.
        # This was the #1 cause of losing "target" exits in post-mortem.
        # V1: Maker routing — when execution is maker (POST_ONLY), both
        # entry and exit have 0 slippage and 0 fees; the gate reduces to
        # just the desired margin.
        if exec_mode == "maker":
            min_viable_spread = strat.desired_margin_cents
        else:
            slippage_rt_cents = 2 * strat.paper_slippage_cents
            entry_fee_cents = entry_fee_frac * 100.0
            exit_est_price = tp.target_price
            # Exit is taker — always model fee regardless of market category.
            exit_fee_cents = get_fee_rate(exit_est_price, fee_enabled=True) * 100.0
            min_viable_spread = slippage_rt_cents + entry_fee_cents + exit_fee_cents + strat.desired_margin_cents
        if tp.spread_cents < min_viable_spread:
            log.info(
                "skip_entry_insufficient_edge",
                spread_cents=round(tp.spread_cents, 2),
                min_viable=round(min_viable_spread, 2),
                exec_mode=exec_mode,
                margin=strat.desired_margin_cents,
            )
            return None

        # ── Dollar-risk cap ────────────────────────────────────────────
        # Cap size so that a gap-to-zero cannot lose more than
        # max_loss_per_trade_cents.
        max_loss_cents = strat.max_loss_per_trade_cents
        if max_loss_cents > 0 and entry_price > 0:
            max_shares_for_risk = max_loss_cents / (entry_price * 100.0)
            if entry_size > max_shares_for_risk:
                entry_size = round(max_shares_for_risk, 2)
                log.info(
                    "dollar_risk_cap_applied",
                    max_loss_cents=max_loss_cents,
                    capped_size=entry_size,
                )
                if entry_size < 1:
                    log.info("skip_entry_risk_cap_too_small")
                    return None

        # Place entry order
        pos_id = f"POS-{self._next_id}"
        self._next_id += 1

        # Compute volatility- and fee-adaptive stop-loss (downside semi-variance)
        _downside_vol: float | None = no_aggregator.rolling_downside_vol_ewma or None
        sl_trigger = compute_adaptive_stop_loss_cents(
            sl_base_cents=strat.stop_loss_cents,
            entry_price=entry_price,
            fee_enabled=fee_enabled,
            ewma_vol=_downside_vol,
            ref_vol=strat.sl_vol_ref,
            is_adaptive=strat.sl_vol_adaptive,
            max_multiplier=strat.sl_vol_multiplier_max,
        )

        # Pillar 11.3: record the vol multiplier used at open time
        if strat.sl_vol_adaptive and _downside_vol and _downside_vol > 0 and strat.sl_vol_ref > 0:
            _sl_vol_mult = max(1.0, min(_downside_vol / strat.sl_vol_ref, strat.sl_vol_multiplier_max))
        else:
            _sl_vol_mult = 1.0

        # Compute volatility-adaptive trailing stop offset
        _trailing_offset = compute_adaptive_trailing_offset_cents(
            base_offset_cents=strat.trailing_stop_offset_cents,
            ewma_downside_vol=_downside_vol,
            ref_vol=strat.sl_vol_ref,
            is_adaptive=strat.sl_vol_adaptive,
            max_multiplier=strat.sl_vol_multiplier_max,
        )

        order = await self.executor.place_limit_order(
            market_id=signal.market_id,
            asset_id=signal.no_asset_id,
            side=OrderSide.BUY,
            price=entry_price,
            size=entry_size,
        )

        pos = Position(
            id=pos_id,
            market_id=signal.market_id,
            no_asset_id=signal.no_asset_id,
            event_id=event_id,
            state=PositionState.ENTRY_PENDING,
            entry_order=order,
            entry_price=entry_price,
            entry_size=entry_size,
            entry_time=time.time(),
            signal=signal,
            tp_result=tp,
            target_price=tp.target_price,
            sizing=sizing,
            kelly_result=kelly_result,
            fee_enabled=fee_enabled,
            sl_trigger_cents=sl_trigger,
            trailing_offset_cents=_trailing_offset,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
            is_probe=is_probe,
            sl_vol_multiplier=_sl_vol_mult,
            signal_type=_signal_type,
            strategy_source=strategy_source,
            meta_weight=_meta_weight,
        )

        self._positions[pos.id] = pos

        log.info(
            "position_opened",
            pos_id=pos.id,
            market=signal.market_id,
            event_id=event_id,
            entry=entry_price,
            size=entry_size,
            target=tp.target_price,
            alpha=tp.alpha,
            sl_trigger=sl_trigger,
            fee_enabled=fee_enabled,
            is_probe=is_probe,
            exec_mode=exec_mode,
            signal_type=_signal_type,
        )

        return pos

    # ── Open RPE position (bidirectional) ──────────────────────────────────
    async def open_rpe_position(
        self,
        *,
        market_id: str,
        yes_asset_id: str,
        no_asset_id: str,
        direction: str,
        model_probability: float,
        confidence: float,
        entry_price: float,
        event_id: str = "",
        days_to_resolution: int = 30,
        fee_enabled: bool = True,
        book: OrderbookTracker | None = None,
        signal_metadata: dict | None = None,
        latency_healthy: bool = False,
    ) -> Position | None:
        """Open a position driven by the Resolution Probability Engine.

        This is a parallel entry method to ``open_position()`` — it
        shares all risk gates but supports bidirectional trades
        (buy YES or buy NO) and uses RPE-specific edge evaluation.

        Design decision — **parallel method vs extending open_position()**:
            The two alphas (panic mean-reversion and RPE model-based) are
            structurally independent.  Extending ``open_position()`` to
            handle both would introduce conditional branches in a proven
            code path.  A separate method isolates RPE logic while
            sharing risk infrastructure via ``_check_risk_gates()``.

        Parameters
        ----------
        direction:
            ``"buy_yes"`` or ``"buy_no"`` — which outcome token to buy.
        model_probability:
            The RPE's estimated YES resolution probability.
        confidence:
            Model confidence (0–1), used as ``1 - uncertainty_penalty``
            for Kelly sizing.
        entry_price:
            Price to place the limit BUY order at.
        latency_healthy:
            True when LatencyGuard is HEALTHY — enables the fast-strike
            taker path when divergence exceeds 2¢.
        """
        async with self._entry_lock:
            return await self._open_rpe_position_inner(
                market_id=market_id, yes_asset_id=yes_asset_id,
                no_asset_id=no_asset_id, direction=direction,
                model_probability=model_probability, confidence=confidence,
                entry_price=entry_price, event_id=event_id,
                days_to_resolution=days_to_resolution,
                fee_enabled=fee_enabled, book=book,
                signal_metadata=signal_metadata,
                latency_healthy=latency_healthy,
            )

    async def _open_rpe_position_inner(
        self,
        *,
        market_id: str,
        yes_asset_id: str,
        no_asset_id: str,
        direction: str,
        model_probability: float,
        confidence: float,
        entry_price: float,
        event_id: str = "",
        days_to_resolution: int = 30,
        fee_enabled: bool = True,
        book: OrderbookTracker | None = None,
        signal_metadata: dict | None = None,
        latency_healthy: bool = False,
    ) -> Position | None:
        """Inner implementation of open_rpe_position (runs under _entry_lock)."""
        strat = settings.strategy
        strategy_source = self._resolve_strategy_source(
            signal_metadata=signal_metadata,
            fallback="rpe",
        )

        # Shadow mode: log but do not trade
        if strat.rpe_shadow_mode:
            log.info(
                "rpe_shadow_entry_skipped",
                market=market_id,
                direction=direction,
                model_prob=round(model_probability, 4),
                confidence=round(confidence, 3),
                entry_price=entry_price,
            )
            return None

        # Determine trade direction before risk gates for PCE directional awareness
        trade_direction = "YES" if direction == "buy_yes" else "NO"

        if not self._ensemble_allows_entry(
            market_id=market_id,
            direction=trade_direction,
            strategy_source=strategy_source,
            log_event="ensemble_risk_blocked_rpe_entry",
        ):
            return None

        passed, max_trade, var_was_bisected = await self._check_risk_gates(
            market_id, event_id, trade_direction=trade_direction,
        )
        if not passed:
            return None

        if entry_price <= 0 or entry_price >= 1:
            log.warning("rpe_entry_price_invalid", price=entry_price)
            return None

        # Determine which token to trade
        if direction == "buy_yes":
            trade_asset = yes_asset_id
            trade_side = "YES"
        else:
            trade_asset = no_asset_id
            trade_side = "NO"

        # ── Edge quality gate ──────────────────────────────────────────
        # Map RPE divergence to the EQS framework.  The divergence
        # divided by (1 - confidence) produces a z-score-like metric.
        # Volume ratio is set to 1.0 (RPE is not volume-triggered).
        divergence_z = abs(model_probability - entry_price) / max(
            1.0 - confidence, 0.05
        )
        edge = compute_edge_score(
            entry_price=entry_price,
            no_vwap=model_probability,  # model estimate as "fair value"
            zscore=divergence_z,
            volume_ratio=1.0,
            whale_confluence=False,
            fee_enabled=fee_enabled,
        )
        if not edge.viable:
            log.info(
                "rpe_edge_rejected",
                market=market_id,
                reason=edge.rejection_reason,
                score=edge.score,
            )
            return None

        # ── Fast-Strike taker path (Pillar 14) ────────────────────────
        # When RPE fair value diverges by >2¢ from the CLOB price and
        # LatencyGuard is HEALTHY, adjust entry to cross the spread and
        # take stale liquidity immediately.  The position is tagged so
        # the bot skips the OrderChaser.
        if direction == "buy_yes":
            _fair_vs_clob = abs(model_probability - entry_price)
        else:
            _fair_vs_clob = abs((1.0 - model_probability) - entry_price)

        fast_strike = _fair_vs_clob > 0.02 and latency_healthy
        _fast_strike_fired_at: float | None = None
        if fast_strike:
            _fast_strike_fired_at = time.monotonic()
            entry_price = round(entry_price + 0.01, 2)
            log.info(
                "rpe_fast_strike_triggered",
                market=market_id,
                divergence=round(_fair_vs_clob, 4),
                taker_price=entry_price,
                direction=direction,
            )

        # ── Depth-aware sizing ─────────────────────────────────────────
        if book is not None:
            sizing = compute_depth_aware_size(
                book=book,
                entry_price=entry_price,
                max_trade_usd=max_trade,
                side="BUY",
            )
        else:
            fallback_usd = max_trade * 0.50
            shares = round(fallback_usd / entry_price, 2)
            if shares < 1:
                shares = 0.0
                fallback_usd = 0.0
            sizing = SizingResult(
                size_usd=fallback_usd,
                size_shares=shares,
                available_liq_usd=0.0,
                method="fallback_no_book",
                capped=False,
            )

        # ── Kelly sizing ───────────────────────────────────────────────
        # RPE confidence flows directly as uncertainty_penalty
        rpe_metadata = dict(signal_metadata or {})
        rpe_metadata.setdefault("uncertainty_penalty", 1.0 - confidence)

        win_rate = 0.0
        avg_win_cents = 0.0
        avg_loss_cents = 0.0
        total_trades_rpe = 0
        stats = await self._get_cached_stats()
        if stats:
            win_rate = stats.get("win_rate", 0.0)
            avg_win_cents = stats.get("avg_win_cents", 0.0)
            avg_loss_cents = stats.get("avg_loss_cents", 0.0)
            total_trades_rpe = stats.get("total_trades", 0)

        signal_score = min(1.0, divergence_z / 3.0)

        # Half Kelly for RPE entries — lower conviction than panic signals
        rpe_kelly_fraction = (strat.kelly_fraction * 0.5)

        # ── Pillar 16.2: Self-healing alpha throttle ──────────────────────
        strategy_mult = await self._compute_strategy_multiplier("rpe")

        kelly_result = compute_kelly_size(
            signal_score=signal_score,
            win_rate=win_rate,
            avg_win_cents=avg_win_cents,
            avg_loss_cents=avg_loss_cents,
            bankroll_usd=self._wallet_balance_usd,
            entry_price=entry_price,
            max_trade_usd=max_trade,
            book=book,
            kelly_fraction_mult=rpe_kelly_fraction,
            signal_metadata=rpe_metadata,
            total_trades=total_trades_rpe,
            pce=self._pce,
            proposed_market_id=market_id,
            open_positions=self.get_open_positions(),
            strategy_multiplier=strategy_mult,
        )

        if kelly_result.method == "kelly_no_edge":
            log.info(
                "rpe_skip_kelly_no_edge",
                estimated_p=kelly_result.estimated_p,
                adjusted_p=kelly_result.adjusted_p,
                uncertainty=kelly_result.uncertainty_penalty,
            )
            return None

        entry_size = min(sizing.size_shares, kelly_result.size_shares)

        # ── V4 Probe sizing for model-only divergences ─────────────────
        # When the Dynamic Prior Engine flags a divergence but no panic
        # signal exists, open at probe size (5% Kelly, $2 cap) to
        # explore model-based alpha with bounded downside.
        is_model_probe = bool(rpe_metadata.get("force_probe", False))
        # In PENNY_LIVE mode, the $1 deployment cap makes probe sizing
        # identical to full sizing.  Skip probes — defer to PRODUCTION.
        _is_penny_live_rpe = (
            self._guard is not None
            and self._guard.deployment_env == DeploymentEnv.PENNY_LIVE
        )
        if is_model_probe and _is_penny_live_rpe:
            is_model_probe = False
            log.info(
                "rpe_probe_skipped_penny_live",
                reason="penny_live_clamp_destroys_probe_math",
            )
        if is_model_probe:
            probe_max_usd = strat.probe_max_usd
            probe_fraction = strat.probe_kelly_fraction
            probe_max_shares = round(probe_max_usd / entry_price, 2) if entry_price > 0 else 0.0
            probe_kelly_shares = round(entry_size * probe_fraction, 2)
            entry_size = max(1.0, min(probe_kelly_shares, probe_max_shares))
            log.info(
                "rpe_model_probe_sizing",
                probe_shares=entry_size,
                probe_max_usd=probe_max_usd,
                probe_fraction=probe_fraction,
                probe_reason=rpe_metadata.get("probe_reason", ""),
            )

        # ── PCE concentration haircut (Pillar 15) ─────────────────────
        # Skip haircut when VaR bisection already reduced sizing to avoid
        # double-penalizing correlated entries (blueprint §III-B).
        if self._pce is not None and not var_was_bisected:
            pce_haircut = self._pce.compute_concentration_haircut(
                market_id, self.get_open_positions()
            )
            if pce_haircut < 1.0:
                entry_size = max(1.0, round(entry_size * pce_haircut, 2))
                log.info(
                    "rpe_pce_size_haircut_applied",
                    haircut=round(pce_haircut, 4),
                    new_size=entry_size,
                )
        elif self._pce is not None and var_was_bisected:
            log.info(
                "rpe_pce_haircut_skipped_bisected",
                market=market_id,
                reason="var_bisection_already_applied",
            )

        if self._guard is not None:
            entry_size = self._guard.get_allowed_trade_shares(
                entry_size, entry_price
            )

        if entry_size < 1:
            log.info("rpe_skip_insufficient_size")
            return None

        # ── Dollar-risk cap (RPE) ──────────────────────────────────────
        max_loss_cents = strat.max_loss_per_trade_cents
        if max_loss_cents > 0 and entry_price > 0:
            max_shares_for_risk = max_loss_cents / (entry_price * 100.0)
            if entry_size > max_shares_for_risk:
                entry_size = round(max_shares_for_risk, 2)
                log.info(
                    "rpe_dollar_risk_cap_applied",
                    max_loss_cents=max_loss_cents,
                    capped_size=entry_size,
                )
                if entry_size < 1:
                    log.info("rpe_skip_risk_cap_too_small")
                    return None

        # ── Take-profit target ─────────────────────────────────────────
        # Mean-reversion toward model estimate: price should converge
        # to model_probability.
        # Use a confidence-scaled alpha: high confidence → capture more
        # of the divergence; low confidence → conservative partial capture.
        alpha_base = strat.alpha_default
        # Scale alpha by confidence: [0.15..0.95] → [0.6..1.0] of base alpha
        conf_scale = 0.6 + 0.4 * max(0.0, min(1.0, confidence))
        alpha = alpha_base * conf_scale

        if direction == "buy_no":
            # NO token price ≈ 1 - YES price.  We enter at entry_price
            # (NO best ask - 1¢), target is higher (NO reverts up).
            no_entry = entry_price
            no_fair = 1.0 - model_probability
            target_price = round(
                no_entry + alpha * (no_fair - no_entry), 2
            )
            target_price = max(target_price, no_entry + 0.01)
        else:
            # YES side: entry at entry_price, target toward model estimate
            target_price = round(
                entry_price + alpha * (model_probability - entry_price), 2
            )
            target_price = max(target_price, entry_price + 0.01)

        # Fee-adaptive stop-loss (RPE path — no aggregator available)
        sl_trigger = compute_adaptive_stop_loss_cents(
            sl_base_cents=strat.stop_loss_cents,
            entry_price=entry_price,
            fee_enabled=fee_enabled,
            ewma_vol=None,
            ref_vol=strat.sl_vol_ref,
            is_adaptive=strat.sl_vol_adaptive,
            max_multiplier=strat.sl_vol_multiplier_max,
        )

        # Trailing offset (RPE path — no downside vol, defaults to 1.0×)
        _trailing_offset = compute_adaptive_trailing_offset_cents(
            base_offset_cents=strat.trailing_stop_offset_cents,
            ewma_downside_vol=None,
            ref_vol=strat.sl_vol_ref,
            is_adaptive=strat.sl_vol_adaptive,
            max_multiplier=strat.sl_vol_multiplier_max,
        )

        from src.trading.fees import get_fee_rate
        entry_fee_frac = get_fee_rate(entry_price, fee_enabled=fee_enabled)
        entry_fee_bps = round(entry_fee_frac * 10000)

        # Place entry order
        pos_id = f"RPE-{self._next_id}"
        self._next_id += 1

        order = await self.executor.place_limit_order(
            market_id=market_id,
            asset_id=trade_asset,
            side=OrderSide.BUY,
            price=entry_price,
            size=entry_size,
            signal_fired_at=_fast_strike_fired_at,
        )

        # ── Pillar 16: Alpha-source attribution ──────────────────────
        _rpe_meta_weight = float(rpe_metadata.get("meta_weight", 1.0))

        pos = Position(
            id=pos_id,
            market_id=market_id,
            no_asset_id=no_asset_id,
            yes_asset_id=yes_asset_id,
            trade_asset_id=trade_asset,
            trade_side=trade_side,
            event_id=event_id,
            state=PositionState.ENTRY_PENDING,
            entry_order=order,
            entry_price=entry_price,
            entry_size=entry_size,
            entry_time=time.time(),
            target_price=target_price,
            sizing=sizing,
            kelly_result=kelly_result,
            fee_enabled=fee_enabled,
            sl_trigger_cents=sl_trigger,
            trailing_offset_cents=_trailing_offset,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=0,
            signal_type="rpe",
            strategy_source=strategy_source,
            meta_weight=_rpe_meta_weight,
        )

        self._positions[pos.id] = pos

        # Mark model-only probes so they can graduate via
        # scale_probe_to_full() on breakeven.
        if is_model_probe:
            pos.is_probe = True

        # Tag fast-strike positions so bot.py skips the OrderChaser.
        if fast_strike:
            pos.fast_strike = True

        log.info(
            "rpe_position_opened",
            pos_id=pos.id,
            market=market_id,
            direction=direction,
            trade_side=trade_side,
            entry=entry_price,
            size=entry_size,
            target=target_price,
            model_prob=round(model_probability, 4),
            confidence=round(confidence, 3),
            sl_trigger=sl_trigger,
            is_model_probe=is_model_probe,
            fast_strike=fast_strike,
        )

        return pos

    # ── Handle entry fill ──────────────────────────────────────────────────
    async def on_entry_filled(self, pos: Position) -> None:
        """Called when the entry buy order is filled.  Places the exit sell."""
        pos.state = PositionState.ENTRY_FILLED
        pos.entry_price = pos.entry_order.filled_avg_price or pos.entry_price
        pos.entry_toxicity_index = self._capture_fill_toxicity_index(pos, side="BUY")
        self._register_ensemble_position(pos)

        if pos.signal_type == "ofi_momentum":
            log.info(
                "momentum_entry_fill_toxicity_snapshot",
                pos_id=pos.id,
                market=pos.market_id,
                toxicity_index=round(pos.entry_toxicity_index, 4),
            )

        if pos.signal_type == "ofi_momentum":
            self._refresh_ofi_drawn_bracket(pos)

        # Reconcile actual fill quantity.  The chaser or paper-fill may
        # have set filled_size already; if not, infer from the order.
        if pos.filled_size <= 0:
            if pos.entry_order and pos.entry_order.filled_size > 0:
                pos.filled_size = pos.entry_order.filled_size
            else:
                pos.filled_size = pos.entry_size

        if pos.signal_type == "ofi_momentum":
            pos.exit_order = None
            pos.state = PositionState.EXIT_PENDING
            log.info(
                "momentum_local_exit_armed",
                pos_id=pos.id,
                target=pos.target_price,
                stop=pos.stop_price,
                max_hold_seconds=round(pos.max_hold_seconds, 2),
                trade_side=pos.trade_side,
            )
            return

        # Use trade_asset_id for exit — supports both YES and NO side entries.
        # Fallback to no_asset_id preserves backward compatibility for
        # existing panic-detector positions created before RPE was added.
        exit_asset = pos.trade_asset_id or pos.no_asset_id

        # Place exit limit sell at the computed target, sized to *actual* fill
        exit_order = await self.executor.place_limit_order(
            market_id=pos.market_id,
            asset_id=exit_asset,
            side=OrderSide.SELL,
            price=pos.target_price,
            size=pos.effective_size,
        )

        pos.exit_order = exit_order
        pos.state = PositionState.EXIT_PENDING

        log.info(
            "exit_order_placed",
            pos_id=pos.id,
            target=pos.target_price,
            size=pos.effective_size,
            trade_side=pos.trade_side,
        )

    def _refresh_ofi_drawn_bracket(self, pos: Position) -> None:
        take_profit_pct = pos.drawn_tp_pct or MomentumBracket().take_profit_pct
        stop_loss_pct = pos.drawn_stop_pct or MomentumBracket().stop_loss_pct
        max_hold_seconds = pos.drawn_time or pos.max_hold_seconds or DEFAULT_MOMENTUM_MAX_HOLD_SECONDS
        drawn_bracket = DrawnMomentumBracket(
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_hold_seconds=max_hold_seconds,
        )
        pos.target_price = drawn_bracket.target_price(pos.entry_price)
        pos.stop_price = drawn_bracket.stop_price(pos.entry_price)
        pos.sl_trigger_cents = drawn_bracket.stop_loss_cents(pos.entry_price)
        pos.max_hold_seconds = float(drawn_bracket.max_hold_seconds)
        pos.drawn_tp = pos.target_price
        pos.drawn_stop = pos.stop_price
        pos.drawn_time = pos.max_hold_seconds

    # ── Handle exit fill ───────────────────────────────────────────────────
    def on_exit_filled(self, pos: Position, reason: str = "target") -> None:
        """Close the position after exit fill or forced liquidation."""
        if pos.state == PositionState.CLOSED:
            log.warning("position_already_closed_ignore_fill", pos_id=pos.id, reason=reason)
            return

        pos.exit_toxicity_index = self._capture_fill_toxicity_index(pos, side="SELL")

        if (
            pos.signal_type == "ofi_momentum"
            and pos.exit_reason == "time_stop"
            and pos.smart_passive_exit_deadline > 0
            and pos.exit_order is not None
            and pos.exit_order.post_only
            and pos.exit_order.status == OrderStatus.FILLED
        ):
            self._smart_passive_maker_filled_count += 1

        self._release_ensemble_position(pos)

        pos.state = PositionState.CLOSED
        pos.exit_price = (
            pos.exit_order.filled_avg_price if pos.exit_order else pos.target_price
        )
        pos.exit_time = time.time()
        pos.exit_reason = pos.exit_reason or reason
        pos.smart_passive_exit_deadline = 0.0

        # Net-of-fee PnL using actual fill size
        pos.pnl_cents = compute_net_pnl_cents(
            entry_price=pos.entry_price,
            exit_price=pos.exit_price,
            size=pos.effective_size,
            fee_enabled=pos.fee_enabled,
            is_maker_entry=(pos.entry_fee_bps == 0 and pos.fee_enabled),
        )

        # Release heavy references to allow GC of closures and metadata
        pos.entry_chaser_task = None
        pos.exit_chaser_task = None
        pos.signal = None
        pos.sizing = None
        pos.kelly_result = None
        pos.tp_result = None

        # Track daily PnL and drawdown
        self._daily_pnl_cents += pos.pnl_cents
        self._cumulative_pnl_cents += pos.pnl_cents
        self._peak_pnl_cents = max(self._peak_pnl_cents, self._cumulative_pnl_cents)
        self._max_drawdown_cents = max(
            self._max_drawdown_cents,
            self._peak_pnl_cents - self._cumulative_pnl_cents,
        )

        # Record stop-loss cooldown to prevent rapid re-entry on same market.
        # Includes preemptive_liquidity_drain which is a loss-exit that was
        # previously missing — its omission caused a loser-loop where the bot
        # re-entered immediately after a preemptive OBI exit.
        if reason in ("stop_loss", "preemptive_liquidity_drain", "timeout", "time_stop"):
            self._stop_loss_cooldowns[pos.market_id] = pos.exit_time

        log.info(
            "position_closed",
            pos_id=pos.id,
            entry=pos.entry_price,
            exit=pos.exit_price,
            pnl_cents=pos.pnl_cents,
            reason=reason,
            hold_seconds=round(pos.exit_time - pos.entry_time, 1),
            daily_pnl=round(self._daily_pnl_cents, 2),
            drawdown=round(self._max_drawdown_cents, 2),
            exit_toxicity_index=round(pos.exit_toxicity_index, 4),
        )

        # Auto-cleanup old closed/cancelled positions
        self.cleanup_closed()

    # ── Timeout enforcement ────────────────────────────────────────────────
    async def check_timeouts(self, *, exclude_signal_types: set[str] | None = None) -> None:
        """Cancel stale entry orders, check stop-losses, and force-exit stale positions."""
        now = time.time()
        stop_loss_cents = settings.strategy.stop_loss_cents
        excluded_signal_types = exclude_signal_types or set()

        for pos in list(self._positions.values()):
            if pos.state == PositionState.EXIT_PENDING and pos.signal_type in excluded_signal_types:
                continue
            # Entry timeout
            if pos.state == PositionState.ENTRY_PENDING:
                elapsed = now - pos.entry_time
                if elapsed > settings.strategy.entry_timeout_seconds:
                    # Cancel chaser task if running
                    if pos.entry_chaser_task and not pos.entry_chaser_task.done():
                        pos.entry_chaser_task.cancel()
                    if pos.entry_order:
                        await self.executor.cancel_order(pos.entry_order)
                    pos.state = PositionState.CANCELLED
                    pos.exit_reason = "entry_timeout"
                    log.info("entry_timeout", pos_id=pos.id, elapsed_s=round(elapsed))

            # Exit timeout or stop-loss
            elif pos.state == PositionState.EXIT_PENDING:
                if pos.signal_type == "ofi_momentum":
                    await self.evaluate_ofi_local_exit(pos, now=now)
                    continue

                elapsed = now - pos.entry_time

                # Stop-loss: check if current NO price moved against us
                should_stop = False
                if stop_loss_cents > 0 and pos.entry_price > 0:
                    # Unrealised loss check deferred to bot loop where
                    # we have access to current prices — here we only
                    # handle cases signalled by the bot via force_stop_loss()
                    pass

                max_hold_seconds = (
                    pos.max_hold_seconds
                    if pos.max_hold_seconds > 0
                    else settings.strategy.exit_timeout_seconds
                )
                if elapsed > max_hold_seconds:
                    exit_reason = "time_stop" if pos.signal_type == "ofi_momentum" else "timeout"
                    if pos.signal_type == "ofi_momentum":
                        await self._start_smart_passive_time_stop(pos, now)
                    else:
                        await self._place_aggressive_timeout_exit(pos, reason=exit_reason)

    def _is_smart_passive_time_stop_active(self, pos: Position) -> bool:
        return (
            pos.signal_type == "ofi_momentum"
            and pos.exit_reason == "time_stop"
            and pos.smart_passive_exit_deadline > 0
            and pos.exit_order is not None
            and pos.exit_order.status in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED)
        )

    async def evaluate_ofi_local_exit(self, pos: Position, *, now: float | None = None) -> bool:
        if pos.signal_type != "ofi_momentum" or pos.state != PositionState.EXIT_PENDING:
            return False

        now = now if now is not None else time.time()
        time_stop_deadline = pos.entry_time + (pos.drawn_time or pos.max_hold_seconds or 0.0)

        if (
            pos.exit_order is not None
            and pos.exit_reason in ("target", "stop_loss")
            and pos.exit_order.status in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED)
        ):
            return True

        if self._is_smart_passive_time_stop_active(pos):
            if now >= pos.smart_passive_exit_deadline:
                decision = self._evaluate_ofi_exit_decision(pos, current_timestamp_ms=int(now * 1000))
                if decision.action == "SUPPRESSED_BY_VACUUM":
                    pos.time_stop_suppression_count += 1
                    if time_stop_deadline > 0:
                        pos.time_stop_delay_seconds = max(pos.time_stop_delay_seconds, now - time_stop_deadline)
                    if pos.time_stop_delay_seconds >= OFI_SMART_PASSIVE_TIMEOUT_SECONDS:
                        forced_fill_price = self._paper_market_sell_price(pos)
                        if forced_fill_price > 0 and pos.drawn_stop > 0:
                            pos.exit_price_minus_drawn_stop_cents = round(
                                (forced_fill_price - pos.drawn_stop) * 100.0,
                                4,
                            )
                        await self._place_aggressive_timeout_exit(pos, reason="time_stop")
                    return True
                await self._promote_smart_passive_to_taker(pos)
            return True

        decision = self._evaluate_ofi_exit_decision(pos, current_timestamp_ms=int(now * 1000))
        if decision.action == "TARGET_HIT":
            await self.force_target_exit(pos)
            return True

        if decision.action == "STOP_HIT":
            await self.force_stop_loss(pos, reason="stop_loss")
            return True

        if decision.action == "SUPPRESSED_BY_VACUUM":
            pos.time_stop_suppression_count += 1
            if time_stop_deadline > 0:
                pos.time_stop_delay_seconds = max(pos.time_stop_delay_seconds, now - time_stop_deadline)
            if pos.time_stop_delay_seconds >= OFI_SMART_PASSIVE_TIMEOUT_SECONDS:
                forced_fill_price = self._paper_market_sell_price(pos)
                if forced_fill_price > 0 and pos.drawn_stop > 0:
                    pos.exit_price_minus_drawn_stop_cents = round(
                        (forced_fill_price - pos.drawn_stop) * 100.0,
                        4,
                    )
                await self._place_aggressive_timeout_exit(pos, reason="time_stop")
            return True

        if decision.action == "TIME_STOP_TRIGGERED":
            if time_stop_deadline > 0:
                pos.time_stop_delay_seconds = max(pos.time_stop_delay_seconds, now - time_stop_deadline)
            projected_fill_price = self._paper_market_sell_price(pos)
            if projected_fill_price > 0 and pos.drawn_stop > 0:
                pos.exit_price_minus_drawn_stop_cents = round(
                    (projected_fill_price - pos.drawn_stop) * 100.0,
                    4,
                )
            if projected_fill_price > 0 and pos.stop_price > 0 and projected_fill_price <= pos.stop_price:
                await self._place_aggressive_timeout_exit(pos, reason="time_stop")
                return True
            if pos.time_stop_delay_seconds >= OFI_TIME_STOP_RECOVERY_RATIO * OFI_SMART_PASSIVE_TIMEOUT_SECONDS:
                await self._place_aggressive_timeout_exit(pos, reason="time_stop")
                return True
            await self._start_smart_passive_time_stop(pos, now)
        return True

    def _evaluate_ofi_exit_decision(self, pos: Position, *, current_timestamp_ms: int) -> OfiExitDecision:
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        max_hold_seconds = pos.drawn_time or pos.max_hold_seconds or DEFAULT_MOMENTUM_MAX_HOLD_SECONDS
        drawn_tp = pos.drawn_tp if pos.drawn_tp > 0 else pos.target_price
        drawn_stop = pos.drawn_stop if pos.drawn_stop > 0 else pos.stop_price
        baseline_spread = max(0.01, pos.entry_price * pos.drawn_stop_pct) if pos.drawn_stop_pct > 0 else 0.01
        position_state = {
            "market_id": exit_asset,
            "drawn_tp": Decimal(str(max(0.0, drawn_tp))),
            "drawn_stop": Decimal(str(max(0.0, drawn_stop))),
            "drawn_time_ms": int((pos.entry_time + max_hold_seconds) * 1000),
            "baseline_spread": Decimal(str(baseline_spread)),
        }
        monitor = self._ofi_exit_monitor(exit_asset)
        if monitor is None:
            current_best_bid = Decimal(str(max(0.0, self._current_best_bid(pos))))
            if position_state["drawn_tp"] > Decimal("0") and current_best_bid >= position_state["drawn_tp"]:
                return OfiExitDecision(action="TARGET_HIT", trigger_price=position_state["drawn_tp"])
            if position_state["drawn_stop"] > Decimal("0") and current_best_bid > Decimal("0") and current_best_bid <= position_state["drawn_stop"]:
                return OfiExitDecision(action="STOP_HIT", trigger_price=position_state["drawn_stop"])
            if current_timestamp_ms > position_state["drawn_time_ms"]:
                return OfiExitDecision(action="TIME_STOP_TRIGGERED", trigger_price=current_best_bid)
            return OfiExitDecision(action="HOLD", trigger_price=current_best_bid)
        decision = monitor.evaluate_exit(position_state, current_timestamp_ms)
        if decision.action == "SUPPRESSED_BY_VACUUM":
            log.info(
                "ofi_time_stop_suppressed_liquidity_vacuum",
                pos_id=pos.id,
                asset_id=exit_asset,
                trigger_price=float(decision.trigger_price),
            )
        return decision

    def _ofi_exit_monitor(self, asset_id: str) -> OfiLocalExitMonitor | None:
        asset_key = str(asset_id or "").strip()
        if not asset_key:
            return None
        monitor = self._ofi_exit_monitors.get(asset_key)
        if monitor is not None:
            return monitor
        tracker = self._book_trackers.get(asset_key)
        if tracker is None:
            return None
        monitor = OfiLocalExitMonitor(
            OrderbookBestBidProvider(tracker),
            vacuum_ratio=Decimal(str(OFI_TIME_STOP_VACUUM_RATIO)),
            spread_multiple=Decimal(str(OFI_TIME_STOP_SPREAD_MULTIPLIER)),
        )
        self._ofi_exit_monitors[asset_key] = monitor
        return monitor

    def _capture_fill_toxicity_index(self, pos: Position, *, side: str) -> float:
        asset_id = pos.trade_asset_id or pos.no_asset_id
        book = self._book_trackers.get(asset_id)
        toxicity_index = 0.0

        if book is not None and hasattr(book, "toxicity_metrics"):
            try:
                metrics = book.toxicity_metrics(side)
                toxicity_index = float(metrics.get("toxicity_index", 0.0) or 0.0)
            except Exception:
                log.warning(
                    "fill_toxicity_snapshot_failed",
                    pos_id=pos.id,
                    asset_id=asset_id,
                    side=side,
                    exc_info=True,
                )

        if toxicity_index <= 0.0:
            toxicity_index = (
                pos.entry_toxicity_index if side.upper() == "BUY" else pos.exit_toxicity_index
            )

        if toxicity_index <= 0.0:
            toxicity_index = float(getattr(pos.signal, "toxicity_index", 0.0) or 0.0)

        return round(max(0.0, toxicity_index), 6)

    async def _start_smart_passive_time_stop(self, pos: Position, now: float) -> None:
        await self._cancel_exit_flow(pos)

        passive_price = self._current_best_ask(pos)
        if passive_price <= 0:
            log.info(
                "time_stop_smart_passive_skipped",
                pos_id=pos.id,
                reason="no_best_ask",
            )
            await self._place_aggressive_timeout_exit(pos, reason="time_stop")
            return

        exit_asset = pos.trade_asset_id or pos.no_asset_id
        exit_order = await self.executor.place_limit_order(
            market_id=pos.market_id,
            asset_id=exit_asset,
            side=OrderSide.SELL,
            price=passive_price,
            size=pos.effective_size,
            post_only=True,
        )

        if exit_order.status == OrderStatus.CANCELLED:
            log.info(
                "time_stop_smart_passive_rejected",
                pos_id=pos.id,
                price=passive_price,
                rejection_reason=exit_order.rejection_reason,
            )
            await self._place_aggressive_timeout_exit(pos, reason="time_stop")
            return

        pos.exit_order = exit_order
        pos.exit_reason = "time_stop"
        pos.smart_passive_exit_deadline = now + OFI_SMART_PASSIVE_TIMEOUT_SECONDS
        self._smart_passive_started_count += 1
        log.info(
            "time_stop_smart_passive_started",
            pos_id=pos.id,
            price=passive_price,
            deadline_s=OFI_SMART_PASSIVE_TIMEOUT_SECONDS,
        )

    async def _promote_smart_passive_to_taker(self, pos: Position) -> None:
        await self._cancel_exit_flow(pos)
        self._smart_passive_fallback_triggered_count += 1
        log.info(
            "time_stop_smart_passive_expired",
            pos_id=pos.id,
            fallback="aggressive_taker_exit",
        )
        await self._place_aggressive_timeout_exit(pos, reason="time_stop")

    async def _cancel_exit_flow(self, pos: Position) -> None:
        if pos.exit_chaser_task and not pos.exit_chaser_task.done():
            pos.exit_chaser_task.cancel()
        if pos.exit_order:
            await self.executor.cancel_order(pos.exit_order)

    async def _place_aggressive_timeout_exit(self, pos: Position, *, reason: str) -> None:
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        pos.exit_reason = reason
        pos.smart_passive_exit_deadline = 0.0
        exit_order = await self.executor.place_limit_order(
            market_id=pos.market_id,
            asset_id=exit_asset,
            side=OrderSide.SELL,
            price=0.01,
            size=pos.effective_size,
        )
        pos.exit_order = exit_order
        if self.executor.paper_mode:
            fill_price = self._paper_market_sell_price(pos)
            exit_order.filled_avg_price = fill_price
            exit_order.filled_size = pos.effective_size
            exit_order.status = OrderStatus.FILLED
            self.on_exit_filled(pos, reason=reason)
        else:
            log.info(
                "time_stop_exit_placed" if reason == "time_stop" else "timeout_exit_placed",
                pos_id=pos.id,
                note="awaiting CLOB fill confirmation",
            )

    def _current_best_bid(self, pos: Position) -> float:
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        tracker = self._book_trackers.get(exit_asset)
        if tracker is None:
            return 0.0

        best_bid = 0.0
        try:
            snap = tracker.snapshot()
            best_bid = float(getattr(snap, "best_bid", 0.0) or 0.0)
        except Exception:
            best_bid = float(getattr(tracker, "best_bid", 0.0) or 0.0)

        if best_bid <= 0:
            best_bid = float(getattr(tracker, "best_bid", 0.0) or 0.0)

        if best_bid <= 0:
            return 0.0
        return round(min(0.99, max(0.01, best_bid)), 4)

    def _current_best_ask(self, pos: Position) -> float:
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        tracker = self._book_trackers.get(exit_asset)
        if tracker is None:
            return 0.0

        best_ask = 0.0
        try:
            snap = tracker.snapshot()
            best_ask = float(getattr(snap, "best_ask", 0.0) or 0.0)
        except Exception:
            best_ask = float(getattr(tracker, "best_ask", 0.0) or 0.0)

        if best_ask <= 0:
            best_ask = float(getattr(tracker, "best_ask", 0.0) or 0.0)

        if best_ask <= 0:
            return 0.0
        return round(min(0.99, max(0.01, best_ask)), 4)

    async def force_target_exit(self, pos: Position, *, reason: str = "target") -> None:
        if pos.state != PositionState.EXIT_PENDING:
            return

        if pos.exit_chaser_task and not pos.exit_chaser_task.done():
            pos.exit_chaser_task.cancel()

        if pos.exit_order:
            await self.executor.cancel_order(pos.exit_order)

        exit_asset = pos.trade_asset_id or pos.no_asset_id
        exit_price = pos.target_price if pos.target_price > 0 else self._current_best_bid(pos)
        exit_price = round(min(0.99, max(0.01, exit_price)), 4)
        exit_order = await self.executor.place_limit_order(
            market_id=pos.market_id,
            asset_id=exit_asset,
            side=OrderSide.SELL,
            price=exit_price,
            size=pos.effective_size,
        )
        pos.exit_order = exit_order
        pos.exit_reason = reason
        pos.smart_passive_exit_deadline = 0.0
        if self.executor.paper_mode:
            fill_price = self._current_best_bid(pos) or exit_price
            exit_order.filled_avg_price = fill_price
            exit_order.filled_size = pos.effective_size
            exit_order.status = OrderStatus.FILLED
            self.on_exit_filled(pos, reason=reason)
        else:
            log.info(
                "target_exit_placed",
                pos_id=pos.id,
                price=exit_price,
                note="awaiting CLOB fill confirmation",
            )

    def _paper_market_sell_price(self, pos: Position) -> float:
        """Estimate a realistic fill price for paper-mode market sells.

        Looks up the current best bid from the book tracker for the
        position's asset.  Falls back to ``entry - sl_trigger/100``
        if no book data is available.  Applies ``paper_slippage_cents``
        adversely (subtracts from bid) and floors at 0.01.
        """
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        best_bid = 0.0

        tracker = self._book_trackers.get(exit_asset)
        if tracker is not None:
            try:
                snap = tracker.snapshot()
                best_bid = getattr(snap, "best_bid", 0.0)
            except Exception:
                pass

        if best_bid <= 0:
            # Fallback: estimate exit price from entry and stop-loss
            sl_cents = pos.sl_trigger_cents if pos.sl_trigger_cents > 0 else settings.strategy.stop_loss_cents
            best_bid = pos.entry_price - sl_cents / 100.0

        slippage = settings.strategy.paper_slippage_cents / 100.0
        fill_price = best_bid - slippage
        return max(0.01, round(fill_price, 4))

    async def force_stop_loss(self, pos: Position, *, reason: str = "stop_loss") -> None:
        """Force-close a position via market sell due to stop-loss trigger."""
        if pos.state != PositionState.EXIT_PENDING:
            return

        # Cancel exit chaser if running
        if pos.exit_chaser_task and not pos.exit_chaser_task.done():
            pos.exit_chaser_task.cancel()

        if pos.exit_order:
            await self.executor.cancel_order(pos.exit_order)

        exit_asset = pos.trade_asset_id or pos.no_asset_id
        pos.exit_reason = reason
        exit_order = await self.executor.place_limit_order(
            market_id=pos.market_id,
            asset_id=exit_asset,
            side=OrderSide.SELL,
            price=0.01,
            size=pos.effective_size,
        )
        pos.exit_order = exit_order
        # In paper mode, simulate a realistic fill at the current best
        # bid rather than the literal 0.01 limit price.
        if self.executor.paper_mode:
            fill_price = self._paper_market_sell_price(pos)
            exit_order.filled_avg_price = fill_price
            exit_order.filled_size = pos.effective_size
            exit_order.status = OrderStatus.FILLED
            self.on_exit_filled(pos, reason=reason)
        else:
            log.info(
                "stop_loss_exit_placed",
                pos_id=pos.id,
                note="awaiting CLOB fill confirmation",
            )

        log.warning(
            "stop_loss_triggered",
            pos_id=pos.id,
            entry=pos.entry_price,
            exit=pos.exit_price if pos.state == PositionState.CLOSED else 0.0,
            pnl_cents=pos.pnl_cents if pos.state == PositionState.CLOSED else 0.0,
        )

    # ── V4: Probe harvesting ───────────────────────────────────────────────
    async def scale_probe_to_full(
        self,
        pos: "Position",
        get_mid_price_fn,
    ) -> "Order | None":
        """Pyramid a confirmed probe into a full-Kelly position.

        Called by the ``on_probe_breakeven`` callback from
        :class:`~src.trading.stop_loss.StopLossMonitor`.  Six sequential
        guards ensure the scale-in remains safe, timely, and still has edge.

        When ``StealthExecutor`` is available, orders above the stealth
        threshold are sliced into 2–4 randomised clips with inter-slice
        delays and mid-price abandonment checks.

        Guards
        ------
        1. Position still open and currently net-positive
        2. Not already scaled (idempotent — uses ``_probe_scaled_set``)
        3. "Not too late": current profit < ``harvest_max_progress`` × TP span
        4. Incremental USD ≥ ``harvest_min_increment_usd``
        5. PCE VaR budget allows additional exposure
        6. Deployment guard phase size cap respected
        """
        if pos.state != PositionState.EXIT_PENDING:
            return None

        strat = settings.strategy

        # Guard 0: probe harvest is meaningless in PENNY_LIVE — the $1
        # clamp makes the incremental scale-up identical to the original
        # probe size.  Defer probe harvesting to PRODUCTION runs.
        if (
            self._guard is not None
            and self._guard.deployment_env == DeploymentEnv.PENNY_LIVE
        ):
            log.info(
                "probe_harvest_skipped_penny_live",
                pos_id=pos.id,
                reason="penny_live_clamp_destroys_probe_math",
            )
            return None

        # Guard 1: still open and net-positive
        asset_id = pos.trade_asset_id or pos.no_asset_id
        mid = get_mid_price_fn(asset_id)
        if not mid or mid <= 0:
            return None
        direction = 1.0 if (pos.trade_side or "NO") == "NO" else -1.0
        current_pnl_cents = direction * (mid - pos.entry_price) * 100.0
        if current_pnl_cents < strat.harvest_min_profit_cents:
            log.info(
                "probe_harvest_not_profitable",
                pos_id=pos.id,
                pnl_cents=round(current_pnl_cents, 2),
                min_required=strat.harvest_min_profit_cents,
            )
            return None

        # Guard 2: idempotent — only scale once per probe
        if pos.id in self._probe_scaled_set:
            return None

        # Guard 3: not too late — avoid pyramiding into a nearly-completed move
        if pos.tp_result is not None and pos.tp_result.target_price > pos.entry_price:
            tp_span_cents = (pos.tp_result.target_price - pos.entry_price) * 100.0
            if tp_span_cents > 0:
                progress = current_pnl_cents / tp_span_cents
                if progress > strat.harvest_max_progress:
                    log.info(
                        "probe_harvest_skipped_late",
                        pos_id=pos.id,
                        progress=round(progress, 3),
                        max_allowed=strat.harvest_max_progress,
                    )
                    return None

        # Guard 4: incremental size must be worthwhile
        if pos.kelly_result is None:
            return None
        # The probe was sized at probe_kelly_fraction of full Kelly;
        # reverse-engineer the full-Kelly share count.
        full_kelly_shares = pos.kelly_result.size_shares / max(
            strat.probe_kelly_fraction, 0.001
        )
        incremental_shares = max(0.0, full_kelly_shares - pos.entry_size)
        incremental_usd = incremental_shares * pos.entry_price
        if incremental_usd < strat.harvest_min_increment_usd:
            log.info(
                "probe_harvest_increment_too_small",
                pos_id=pos.id,
                incremental_usd=round(incremental_usd, 3),
                min_required=strat.harvest_min_increment_usd,
            )
            return None

        # Guard 5: PCE VaR budget (bisection — allows partial scale-up)
        if self._pce is not None and hasattr(self._pce, "compute_var_sizing_cap"):
            open_positions = self.get_open_positions()
            sizing_result = self._pce.compute_var_sizing_cap(
                open_positions=open_positions,
                proposed_market_id=pos.market_id,
                proposed_size_usd=incremental_usd,
                proposed_direction=pos.trade_side or "NO",
            )
            if sizing_result.cap_usd < strat.harvest_min_increment_usd:
                log.warning(
                    "probe_harvest_var_rejected",
                    pos_id=pos.id,
                    market_id=pos.market_id,
                    incremental_usd=round(incremental_usd, 2),
                    var_cap=round(sizing_result.cap_usd, 2),
                    headroom=round(sizing_result.headroom, 2),
                )
                return None
            if sizing_result.cap_usd < incremental_usd:
                original_usd = incremental_usd
                incremental_usd = sizing_result.cap_usd
                incremental_shares = round(
                    incremental_usd / pos.entry_price, 2
                ) if pos.entry_price > 0 else 0.0
                log.info(
                    "probe_harvest_var_sized",
                    pos_id=pos.id,
                    original_usd=round(original_usd, 2),
                    capped_usd=round(incremental_usd, 2),
                    bisect_iterations=sizing_result.bisect_iterations,
                )

        # Guard 6: deployment guard phase cap
        if self._guard is not None:
            incremental_shares = self._guard.get_allowed_trade_shares(
                incremental_shares, pos.entry_price
            )
            incremental_usd = incremental_shares * pos.entry_price
            if incremental_usd < strat.harvest_min_increment_usd:
                return None

        # Place scale-in limit order — price-cap: don't chase above entry + 1¢
        scale_price = min(mid, pos.entry_price + 0.01)
        scale_price = max(0.01, round(scale_price, 2))
        scale_shares = round(incremental_shares, 2)
        if scale_shares < 1:
            return None

        # ── SI-4: Stealth slicing for scale-ups ───────────────────────
        if self._stealth is not None and incremental_usd >= self._stealth._min_size:
            # Build a mid-price reader that closes over the asset_id
            _asset = asset_id
            _mid_fn = get_mid_price_fn

            def _get_mid() -> float | None:
                return _mid_fn(_asset)

            # Fetch recent volume from OHLCV aggregator for POV cap
            # Burst-volume acceleration: use max(avg, current) so volume
            # spikes allow faster execution instead of over-slicing.
            _recent_vol = 0.0
            if self._ohlcv_aggs is not None:
                _agg = self._ohlcv_aggs.get(asset_id)
                if _agg is not None:
                    _eff_vol = max(_agg.avg_bar_volume, _agg.current_bar_volume)
                    _recent_vol = _eff_vol * max(scale_price, 0.01)

            # Fetch L2 book depth ratio for OBI pacing
            _depth_ratio = 1.0
            _tracker = self._book_trackers.get(asset_id)
            if _tracker is not None and getattr(_tracker, 'is_reliable', False):
                _depth_ratio = _tracker.book_depth_ratio

            # SI-4.3: Detect opposing icebergs for delay skewing
            _opposing_iceberg = False
            _ice_det = self._iceberg_detectors.get(asset_id)
            if _ice_det is not None:
                _opp_side = "SELL"  # we are BUYing, so opposing is SELL
                _strongest = _ice_det.strongest_iceberg(_opp_side)
                if _strongest is not None and _strongest.confidence >= 0.50:
                    _opposing_iceberg = True

            stealth_orders = await self._stealth.place_stealth_order(
                market_id=pos.market_id,
                asset_id=asset_id,
                side=OrderSide.BUY,
                price=scale_price,
                total_size=scale_shares,
                post_only=True,
                get_mid_fn=_get_mid,
                recent_volume_usd=_recent_vol,
                book_depth_ratio=_depth_ratio,
                opposing_iceberg_detected=_opposing_iceberg,
            )
            order = stealth_orders[0] if stealth_orders else None
        else:
            order = await self.executor.place_limit_order(
                market_id=pos.market_id,
                asset_id=asset_id,
                side=OrderSide.BUY,
                price=scale_price,
                size=scale_shares,
            )

        self._probe_scaled_set.add(pos.id)
        pos.is_probe = False  # graduated: now tracked as a full position

        new_avg = (
            (pos.entry_price * pos.entry_size + scale_price * scale_shares)
            / (pos.entry_size + scale_shares)
        )
        log.info(
            "probe_harvested",
            pos_id=pos.id,
            scale_shares=scale_shares,
            incremental_usd=round(incremental_usd, 2),
            scale_price=round(scale_price, 4),
            new_avg_entry=round(new_avg, 4),
            pnl_cents_at_scale=round(current_pnl_cents, 2),
        )
        return order

    # ── Cleanup ────────────────────────────────────────────────────────────
    def cleanup_closed(self) -> list[Position]:
        """Remove and return closed/cancelled positions (keep last N)."""
        closed = [
            p for p in self._positions.values()
            if p.state in (PositionState.CLOSED, PositionState.CANCELLED)
        ]
        # Sort by exit_time desc, keep only the most recent
        closed.sort(key=lambda p: p.exit_time or p.created_at, reverse=True)

        to_remove = closed[self._MAX_CLOSED_KEPT:]
        for p in to_remove:
            del self._positions[p.id]

        return to_remove

    def get_open_market_ids(self) -> set[str]:
        """Return set of market_ids with open positions."""
        return {
            p.market_id for p in self._positions.values()
            if p.state in (
                PositionState.ENTRY_PENDING,
                PositionState.ENTRY_FILLED,
                PositionState.EXIT_PENDING,
            )
        }

    # ── Query helpers ───────────────────────────────────────────────────────
    def get_open_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.state in (
                PositionState.ENTRY_PENDING,
                PositionState.ENTRY_FILLED,
                PositionState.EXIT_PENDING,
            )
        ]

    def get_all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_closed_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.state == PositionState.CLOSED]

    # ── Stink-bid cascade harvesting ───────────────────────────────────
    async def harvest_cascades(
        self,
        signal: PanicSignal,
        l2_book: Any,
        *,
        event_id: str = "",
        min_depth_usd: float = 0.0,
        tick_offset_range: tuple[int, int] = (2, 4),
    ) -> list[Position]:
        """Place passive stink bids at liquidity-gap levels during a panic.

        When a :class:`PanicSignal` fires, this method queries the L2
        book for thin-depth gaps 2–4 ticks below the current best ask
        and places limit BUY orders at those levels.  These orders are
        designed to fill only when a stop-loss cascade pushes through
        the gap.

        Parameters
        ----------
        signal:
            The PanicSignal that triggered this call.
        l2_book:
            An :class:`~src.data.l2_book.L2OrderBook` with a
            ``find_liquidity_gaps()`` method.
        event_id:
            Market event id for risk-gate checks.
        min_depth_usd:
            Forwarded to ``find_liquidity_gaps()``.
        tick_offset_range:
            Inclusive (lo, hi) tick offsets below best_ask to consider.
            Default is (2, 4) → prices 0.02–0.04 below best ask.

        Returns
        -------
        list[Position]
            Positions created for each stink bid placed.
        """
        best_ask = signal.no_best_ask
        if best_ask <= 0:
            return []

        gaps = l2_book.find_liquidity_gaps(min_depth_usd=min_depth_usd)
        if not gaps:
            return []

        lo, hi = tick_offset_range
        tick = 0.01
        lower_bound = round(best_ask - hi * tick, 2)
        upper_bound = round(best_ask - lo * tick, 2)

        # Filter gaps to only those within the offset window
        target_levels = [
            g for g in gaps
            if lower_bound <= g.price <= upper_bound and g.price > 0
        ]
        if not target_levels:
            return []

        positions: list[Position] = []
        async with self._entry_lock:
            for gap in target_levels:
                # Recheck risk gates for each order
                passed, max_trade, _ = await self._check_risk_gates(
                    signal.market_id, event_id,
                )
                if not passed:
                    break

                entry_price = gap.price
                # Conservative fixed sizing: 25 % of max trade, min 1 share
                entry_size = max(1.0, round((max_trade * 0.25) / entry_price, 2))

                order = await self.executor.place_limit_order(
                    market_id=signal.market_id,
                    asset_id=signal.no_asset_id,
                    side=OrderSide.BUY,
                    price=entry_price,
                    size=entry_size,
                )

                pos_id = f"STINK-{self._next_id}"
                self._next_id += 1

                pos = Position(
                    id=pos_id,
                    market_id=signal.market_id,
                    no_asset_id=signal.no_asset_id,
                    event_id=event_id,
                    state=PositionState.ENTRY_PENDING,
                    entry_order=order,
                    entry_price=entry_price,
                    entry_size=entry_size,
                    entry_time=time.time(),
                    signal=signal,
                    signal_type="stink_bid",
                )
                self._positions[pos.id] = pos
                positions.append(pos)

                log.info(
                    "stink_bid_placed",
                    pos_id=pos.id,
                    market=signal.market_id,
                    price=entry_price,
                    size=entry_size,
                    gap_depth=gap.size,
                )

        return positions

    # ── Vacuum stink-bid strategy (ghost liquidity exploit) ────────────
    async def open_vacuum_stink_bids(
        self,
        signal: VacuumSignal,
        *,
        event_id: str = "",
    ) -> list[Position]:
        """Place POST_ONLY limit orders on both sides during a ghost liquidity vacuum.

        When a :class:`VacuumSignal` fires (>50% depth drop), this method
        places deeply OTM maker orders to catch flash-crash wicks caused
        by retail market orders hitting the thinned book.

        All orders are strictly POST_ONLY to guarantee maker execution
        (0 bps fee).  Sizing is capped by Kelly / VaR gates and the
        ``vacuum_stink_bid_max_risk_usd`` hard cap ($1–$2).

        Parameters
        ----------
        signal:
            The VacuumSignal carrying mid-price and asset IDs.
        event_id:
            Market event id for risk-gate checks.

        Returns
        -------
        list[Position]
            Positions created for each stink bid placed (bid + ask sides).
        """
        strat = settings.strategy
        if not strat.vacuum_stink_bid_enabled:
            return []

        mid = signal.mid_price
        if mid <= 0:
            return []

        offset = strat.vacuum_stink_bid_offset_cents / 100.0  # cents → dollars
        max_risk_usd = strat.vacuum_stink_bid_max_risk_usd

        # mid is the YES token mid-price; NO mid = 1.0 - mid.
        # Each stink bid sits *below* that side's mid to catch flash-crash wicks.
        yes_bid = round(max(0.01, mid - offset), 2)
        no_bid = round(max(0.01, (1.0 - mid) - offset), 2)

        # ── Spread-aware clamping ──────────────────────────────────────────────────────
        # During ghost-liquidity events the spread blows out.  If the
        # calculated bid >= best_ask, POST_ONLY would reject because the
        # order would cross.  Clamp each bid to best_ask - 0.01 so it
        # always rests as a maker order.
        yes_tracker = self._book_trackers.get(signal.yes_asset_id)
        no_tracker = self._book_trackers.get(signal.no_asset_id)

        if yes_tracker is not None and yes_tracker.best_ask > 0:
            yes_bid = round(min(yes_bid, yes_tracker.best_ask - 0.01), 2)
        if no_tracker is not None and no_tracker.best_ask > 0:
            no_bid = round(min(no_bid, no_tracker.best_ask - 0.01), 2)

        sides: list[tuple[float, str, str, float]] = []
        # BUY YES token below YES mid — catches YES flash-crash
        if 0.01 <= yes_bid <= 0.99 and signal.yes_asset_id:
            sides.append((yes_bid, signal.yes_asset_id, "BUY", mid))
        # BUY NO token below NO mid — catches NO flash-crash
        if 0.01 <= no_bid <= 0.99 and signal.no_asset_id:
            sides.append((no_bid, signal.no_asset_id, "BUY", 1.0 - mid))

        if not sides:
            return []

        positions: list[Position] = []
        async with self._entry_lock:
            for entry_price, asset_id, _, side_mid in sides:
                if entry_price <= 0 or entry_price >= 1.0:
                    continue

                # Risk gates (Kelly, VaR, max positions, etc.)
                passed, max_trade, _ = await self._check_risk_gates(
                    signal.market_id, event_id,
                )
                if not passed:
                    break

                # Cap at vacuum_stink_bid_max_risk_usd
                capped_usd = min(max_trade, max_risk_usd)
                entry_size = max(1.0, round(capped_usd / entry_price, 2))

                # Deployment guard cap
                if self._guard is not None:
                    entry_size = self._guard.get_allowed_trade_shares(
                        entry_size, entry_price,
                    )
                if entry_size < 1:
                    continue

                order = await self.executor.place_limit_order(
                    market_id=signal.market_id,
                    asset_id=asset_id,
                    side=OrderSide.BUY,
                    price=entry_price,
                    size=entry_size,
                    post_only=True,
                )

                # POST_ONLY rejection — would cross the spread
                if order.status == OrderStatus.CANCELLED:
                    log.info(
                        "vacuum_stink_bid_rejected_post_only",
                        asset=asset_id[:16],
                        price=entry_price,
                    )
                    continue

                pos_id = f"VACUUM-{self._next_id}"
                self._next_id += 1

                # Take-profit: revert to side mid (where price was before the crash)
                target = round(min(side_mid, 0.99), 2)

                pos = Position(
                    id=pos_id,
                    market_id=signal.market_id,
                    no_asset_id=signal.no_asset_id,
                    event_id=event_id,
                    state=PositionState.ENTRY_PENDING,
                    entry_order=order,
                    entry_price=entry_price,
                    entry_size=entry_size,
                    entry_time=time.time(),
                    signal=signal,
                    signal_type="vacuum_stink_bid",
                    trade_asset_id=asset_id,
                    yes_asset_id=signal.yes_asset_id,
                    target_price=target,
                )
                self._positions[pos.id] = pos
                positions.append(pos)

                log.info(
                    "vacuum_stink_bid_placed",
                    pos_id=pos.id,
                    market=signal.market_id[:16],
                    asset=asset_id[:16],
                    price=entry_price,
                    size=entry_size,
                    mid=mid,
                    depth_velocity=round(signal.depth_velocity, 3),
                )

        return positions

    async def cancel_vacuum_stink_bids(self, market_id: str) -> int:
        """Cancel all unfilled vacuum stink bids for a market.

        Called when the ghost liquidity event recovers and the market
        returns to ACTIVE — the structural advantage has disappeared.

        Returns the number of orders cancelled.
        """
        cancelled = 0
        for pos in list(self._positions.values()):
            if pos.market_id != market_id:
                continue
            if pos.signal_type != "vacuum_stink_bid":
                continue
            if pos.state != PositionState.ENTRY_PENDING:
                continue
            if pos.entry_order and pos.entry_order.status in (
                OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED,
            ):
                await self.executor.cancel_order(pos.entry_order)
                pos.state = PositionState.CANCELLED
                pos.exit_reason = "ghost_recovered"
                cancelled += 1
                log.info(
                    "vacuum_stink_bid_cancelled",
                    pos_id=pos.id,
                    market=market_id[:16],
                    reason="ghost_recovered",
                )
        return cancelled

    # ═══════════════════════════════════════════════════════════════════════
    #  SI-9: Combinatorial Arbitrage — multi-leg entry + Hanging Leg SM
    # ═══════════════════════════════════════════════════════════════════════

    async def open_combo_position(
        self,
        signal: Any,
        combo_positions: dict[str, "ComboPosition"],
    ) -> ComboPosition | None:
        """Quote only the maker leg and defer taker legs until filled.

        The upgraded SI-9 flow is maker-first:
          1. Quote the bottleneck leg passively via POST_ONLY.
          2. Cache the taker legs locally but do not execute them yet.
          3. When the maker leg fully fills, immediately sweep the takers.

        Parameters
        ----------
        signal :
            A ``ComboArbSignal`` carrying the legs and computed edge.
            The signal already carries ``target_shares`` (uniform share
            count computed by ``ComboSizer``).
        combo_positions :
            Active combo positions dict (for concurrency checks).

        Returns
        -------
        ComboPosition | None
            The created combo, or ``None`` if risk gates reject.
        """
        if not self._is_combo_like_signal(signal):
            return None

        strategy_source = self._normalize_strategy_source(
            getattr(signal, "signal_source", "")
            or ("si10_bayesian_arb" if hasattr(signal, "relationship_label") else "si9_combo_arb")
        )
        proposed_exposures = [
            (
                getattr(leg, "market_id", ""),
                getattr(leg, "trade_side", "YES") or "YES",
            )
            for leg in signal.legs
        ]

        strat = settings.strategy
        async with self._entry_lock:
            if not self._ensemble_allows_batch_entry(
                strategy_source=strategy_source,
                exposures=proposed_exposures,
                log_event="ensemble_risk_blocked_combo_entry",
                entry_id=getattr(signal, "cluster_event_id", ""),
            ):
                return None

            # ── Combo-specific risk gates ──────────────────────────────
            if self._circuit_breaker_tripped:
                log.warning("combo_blocked_circuit_breaker")
                return None

            if self._daily_pnl_cents <= -(strat.daily_loss_limit_usd * 100):
                log.warning("combo_blocked_daily_loss")
                return None
            if self._max_drawdown_cents >= strat.max_drawdown_cents:
                log.warning("combo_blocked_drawdown")
                return None

            # Max concurrent combos
            active_combos = [
                c for c in combo_positions.values()
                if c.state in (ComboState.ENTRY_PENDING, ComboState.PARTIAL_FILL)
            ]
            if len(active_combos) >= strat.si9_max_concurrent_combos:
                log.warning("combo_blocked_max_concurrent", count=len(active_combos))
                return None

            # No duplicate combo for same event
            if signal.cluster_event_id in combo_positions:
                existing = combo_positions[signal.cluster_event_id]
                if existing.state in (ComboState.ENTRY_PENDING, ComboState.PARTIAL_FILL, ComboState.ALL_FILLED):
                    return None

            # Total exposure across all combos
            total_exposure = sum(
                c.total_collateral for c in combo_positions.values()
                if c.state in (ComboState.ENTRY_PENDING, ComboState.PARTIAL_FILL, ComboState.ALL_FILLED)
            )
            # Use signal's pre-computed collateral (shares * Σ prices)
            new_collateral = signal.total_collateral
            if total_exposure + new_collateral > strat.si9_max_total_exposure_usd:
                log.warning(
                    "combo_blocked_exposure",
                    current=round(total_exposure, 2),
                    proposed=round(new_collateral, 2),
                )
                return None

            # Wallet balance check
            if new_collateral > self._wallet_balance_usd * strat.max_wallet_risk_pct / 100.0:
                log.warning("combo_blocked_wallet", collateral=round(new_collateral, 2))
                return None

            if signal.maker_leg is None:
                log.warning("combo_missing_maker_leg", event_id=signal.cluster_event_id)
                return None

            # ── Quote only the maker leg ───────────────────────────────
            combo_id = f"COMBO-{signal.cluster_event_id[:12]}-{self._next_id}"
            self._next_id += 1
            legs: dict[str, Position] = {}
            uniform_shares = signal.target_shares

            maker_leg = signal.maker_leg
            maker_asset_id = getattr(maker_leg, "asset_id", maker_leg.yes_token_id)
            maker_trade_side = getattr(maker_leg, "trade_side", "YES")
            maker_order = await self.executor.place_limit_order(
                market_id=maker_leg.market_id,
                asset_id=maker_asset_id,
                side=OrderSide.BUY,
                price=maker_leg.target_price,
                size=uniform_shares,
                post_only=True,
            )

            if (
                maker_order.status == OrderStatus.CANCELLED
                and maker_order.rejection_reason == "would_cross"
            ):
                fallback_price = round(maker_leg.best_bid, 2)
                if fallback_price > 0:
                    maker_order = await self.executor.place_limit_order(
                        market_id=maker_leg.market_id,
                        asset_id=maker_asset_id,
                        side=OrderSide.BUY,
                        price=fallback_price,
                        size=uniform_shares,
                        post_only=True,
                    )

            if maker_order.status == OrderStatus.CANCELLED:
                log.warning("combo_no_legs_placed", event_id=signal.cluster_event_id)
                return None

            maker_pos = Position(
                id=f"{combo_id}-L0",
                market_id=maker_leg.market_id,
                no_asset_id=maker_leg.no_token_id,
                event_id=signal.cluster_event_id,
                state=PositionState.ENTRY_PENDING,
                entry_order=maker_order,
                entry_price=maker_order.price,
                entry_size=uniform_shares,
                entry_time=time.time(),
                trade_asset_id=maker_asset_id,
                trade_side=maker_trade_side,
                yes_asset_id=maker_leg.yes_token_id,
                signal_type="combo_arb",
                strategy_source=strategy_source,
                fee_enabled=False,
            )
            self._positions[maker_pos.id] = maker_pos
            legs[maker_pos.market_id] = maker_pos

            for idx, leg in enumerate(signal.taker_legs, start=1):
                leg_asset_id = getattr(leg, "asset_id", leg.yes_token_id)
                leg_trade_side = getattr(leg, "trade_side", "YES")
                pos = Position(
                    id=f"{combo_id}-L{idx}",
                    market_id=leg.market_id,
                    no_asset_id=leg.no_token_id,
                    event_id=signal.cluster_event_id,
                    state=PositionState.ENTRY_PENDING,
                    entry_order=None,
                    entry_price=leg.target_price,
                    entry_size=uniform_shares,
                    entry_time=0.0,
                    trade_asset_id=leg_asset_id,
                    trade_side=leg_trade_side,
                    yes_asset_id=leg.yes_token_id,
                    signal_type="combo_arb",
                    strategy_source=strategy_source,
                    fee_enabled=True,
                )
                legs[pos.market_id] = pos

            combo = ComboPosition(
                combo_id=combo_id,
                event_id=signal.cluster_event_id,
                state=ComboState.ENTRY_PENDING,
                target_size=uniform_shares,
                maker_market_id=maker_leg.market_id,
                legs=legs,
                planned_collateral=new_collateral,
            )
            self._register_combo_order(combo, maker_pos)

            log.info(
                "combo_position_opened",
                combo_id=combo_id,
                event_id=signal.cluster_event_id,
                n_legs=len(legs),
                maker_market_id=maker_leg.market_id,
                pending_takers=len(signal.taker_legs),
                routing_reason=maker_leg.routing_reason,
                collateral=round(new_collateral, 2),
                edge_cents=signal.edge_cents,
                shares=uniform_shares,
            )
            return combo

    def _is_combo_like_signal(self, signal: Any) -> bool:
        maker_leg = getattr(signal, "maker_leg", None)
        taker_legs = getattr(signal, "taker_legs", None)
        return (
            maker_leg is not None
            and isinstance(taker_legs, list)
            and hasattr(signal, "cluster_event_id")
            and hasattr(signal, "target_shares")
            and hasattr(signal, "total_collateral")
        )

    async def abandon_combo(
        self,
        combo: ComboPosition,
        reason: str = "leg_timeout",
    ) -> None:
        """Hanging-Leg State Machine — safely unwind a partial combo.

        This is the CRITICAL safety mechanism for combinatorial arbitrage.
        Unlike single-position exits, a combo with partial fills leaves
        the portfolio with naked directional risk.  We must handle two
        fundamentally different states:

        ══════════════════════════════════════════════════════════════════
        STATE 1: ZERO FILLS (no leg has been filled)
        ══════════════════════════════════════════════════════════════════
        Safe exit.  Simply cancel all resting maker BUY orders.
        No shares acquired, no directional risk, no unwind needed.

        ══════════════════════════════════════════════════════════════════
        STATE 2: PARTIAL FILLS (≥1 leg filled, ≥1 leg unfilled)
        ══════════════════════════════════════════════════════════════════
        DANGEROUS.  We hold YES shares on some outcomes but not all.
        The arb is incomplete — we have naked directional risk.

        Emergency hedge decision tree for each UNFILLED leg:
          a) Read the YES best_ask from the orderbook.
          b) Compute the taker cost = (best_ask - original_bid) * shares * 100.
          c) If taker_cost ≤ si9_emergency_taker_max_cents AND
             spread ≤ si9_emergency_max_spread_cents:
               → CROSS THE SPREAD: send a TAKER BUY at best_ask to
                 forcibly complete the leg and preserve the arb structure.
                 We pay the taker fee but the arb payout covers it.
          d) If the spread is too wide or cost too high:
               → DUMP FILLED LEGS: the arb is unsalvageable. Immediately
                 market-sell all YES shares we've acquired on the filled
                 legs to eliminate directional risk entirely.

        The bot MUST NOT leave the portfolio holding partial combo shares.
        Every code path terminates with either a complete arb or zero
        exposure.
        ══════════════════════════════════════════════════════════════════
        """
        strat = settings.strategy
        max_taker_cents = strat.si9_emergency_taker_max_cents
        max_spread_cents = strat.si9_emergency_max_spread_cents

        maker_pos = combo.maker_position

        # ── Maker-first safe timeout: no maker fill, no taker sweep ────
        if (
            not combo.sweep_triggered
            and maker_pos is not None
            and maker_pos.state == PositionState.ENTRY_PENDING
            and (maker_pos.entry_order is None or maker_pos.entry_order.filled_size <= 0)
        ):
            combo.state = ComboState.ABANDONED
            for pos in combo.legs.values():
                if pos.entry_order and pos.entry_order.status in (
                    OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED,
                ):
                    await self.executor.cancel_order(pos.entry_order)
                    self._unregister_combo_order(pos.entry_order.order_id)
                pos.state = PositionState.CANCELLED
                pos.exit_reason = reason
            log.info(
                "combo_abandoned_safe",
                combo_id=combo.combo_id,
                event_id=combo.event_id,
                reason=reason,
                note="maker_unfilled_no_taker_exposure",
            )
            return

        # ── Partial maker fill before sweep: flatten only the maker ────
        if (
            not combo.sweep_triggered
            and maker_pos is not None
            and maker_pos.entry_order is not None
            and 0 < maker_pos.entry_order.filled_size < maker_pos.entry_size
        ):
            if maker_pos.entry_order.status in (OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED):
                await self.executor.cancel_order(maker_pos.entry_order)
                self._unregister_combo_order(maker_pos.entry_order.order_id)

            maker_pos.filled_size = maker_pos.entry_order.filled_size
            maker_pos.state = PositionState.ENTRY_FILLED
            self._register_ensemble_position(maker_pos)

            book = self._book_trackers.get(maker_pos.trade_asset_id or maker_pos.yes_asset_id)
            bid = book.best_bid if book is not None else 0.01
            bid = max(0.01, bid)

            exit_order = await self.executor.place_limit_order(
                market_id=maker_pos.market_id,
                asset_id=maker_pos.trade_asset_id or maker_pos.yes_asset_id,
                side=OrderSide.SELL,
                price=round(bid, 2),
                size=maker_pos.filled_size,
                post_only=False,
            )
            maker_pos.exit_order = exit_order
            maker_pos.state = PositionState.EXIT_PENDING
            maker_pos.exit_reason = f"{reason}_partial_maker_dump"

            for pos in combo.taker_positions:
                pos.state = PositionState.CANCELLED
                pos.exit_reason = reason

            combo.state = ComboState.ABANDONED
            log.warning(
                "combo_abandoned_partial_maker_dump",
                combo_id=combo.combo_id,
                event_id=combo.event_id,
                filled_size=maker_pos.filled_size,
                sell_price=round(bid, 2),
            )
            return

        filled = [p for p in combo.legs.values() if p.state == PositionState.ENTRY_FILLED]
        unfilled = [p for p in combo.legs.values() if p.state == PositionState.ENTRY_PENDING]

        # ── STATE 2: Partial fills — emergency hedge required ──────────
        # First, cancel all still-resting unfilled orders so they can't
        # fill while we're evaluating the hedge.
        for pos in unfilled:
            if pos.entry_order and pos.entry_order.status in (
                OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED,
            ):
                await self.executor.cancel_order(pos.entry_order)
                self._unregister_combo_order(pos.entry_order.order_id)

        # Attempt to CROSS THE SPREAD on each unfilled leg to complete arb.
        # Track which legs we successfully taker-fill vs which are too wide.
        hedge_failures: list[Position] = []

        for pos in unfilled:
            book = self._book_trackers.get(pos.trade_asset_id or pos.yes_asset_id)
            best_ask = book.best_ask if book else 0.0

            if best_ask <= 0:
                # No ask available — cannot cross, mark as failure
                hedge_failures.append(pos)
                pos.state = PositionState.CANCELLED
                pos.exit_reason = f"{reason}_no_ask"
                log.warning(
                    "combo_hedge_no_ask",
                    combo_id=combo.combo_id,
                    market_id=pos.market_id[:16],
                )
                continue

            # Compute cost of crossing the spread
            spread_cents = (best_ask - pos.entry_price) * 100.0
            taker_cost_cents = spread_cents * pos.entry_size

            # Decision: cross if within tolerance, else mark as failure
            if spread_cents <= max_spread_cents and taker_cost_cents <= max_taker_cents:
                # ── CROSS THE SPREAD: taker BUY to force-fill the leg ──
                # We send post_only=False (taker) and accept the fee hit.
                taker_order = await self.executor.place_limit_order(
                    market_id=pos.market_id,
                    asset_id=pos.trade_asset_id or pos.yes_asset_id,
                    side=OrderSide.BUY,
                    price=round(best_ask, 2),
                    size=pos.entry_size,
                    post_only=False,  # TAKER — deliberately crossing
                )
                pos.entry_order = taker_order
                if taker_order.status != OrderStatus.CANCELLED:
                    self._register_combo_order(combo, pos)
                if taker_order.status == OrderStatus.CANCELLED:
                    # Taker order also failed — treat as hedge failure
                    hedge_failures.append(pos)
                    pos.state = PositionState.CANCELLED
                    pos.exit_reason = f"{reason}_taker_rejected"
                    log.warning(
                        "combo_taker_rejected",
                        combo_id=combo.combo_id,
                        market_id=pos.market_id[:16],
                    )
                else:
                    # Taker order placed — assume fill (GTC at best_ask)
                    pos.state = PositionState.ENTRY_FILLED
                    pos.filled_size = pos.entry_size
                    pos.entry_price = best_ask
                    self._register_ensemble_position(pos)
                    log.info(
                        "combo_emergency_taker_fill",
                        combo_id=combo.combo_id,
                        market_id=pos.market_id[:16],
                        taker_price=best_ask,
                        spread_cents=round(spread_cents, 1),
                    )
            else:
                # Spread too wide — mark as hedge failure
                hedge_failures.append(pos)
                pos.state = PositionState.CANCELLED
                pos.exit_reason = f"{reason}_spread_too_wide"
                log.warning(
                    "combo_spread_too_wide",
                    combo_id=combo.combo_id,
                    market_id=pos.market_id[:16],
                    spread_cents=round(spread_cents, 1),
                    taker_cost_cents=round(taker_cost_cents, 1),
                )

        # ── Check outcome: did we complete the arb or must we dump? ────
        if not hedge_failures:
            # All unfilled legs were successfully taker-filled.
            # The arb is now complete (all legs filled).
            combo.state = ComboState.ALL_FILLED
            log.info(
                "combo_emergency_hedge_complete",
                combo_id=combo.combo_id,
                event_id=combo.event_id,
                n_taker_fills=len(unfilled),
            )
            return

        # ── DUMP FILLED LEGS: arb is unsalvageable, eliminate risk ─────
        # We MUST sell every YES token we hold to zero out directional
        # exposure.  Use taker SELL at best_bid to guarantee execution.
        combo.state = ComboState.ABANDONED
        all_filled_now = [
            p for p in combo.legs.values()
            if p.state == PositionState.ENTRY_FILLED
        ]
        for pos in all_filled_now:
            book = self._book_trackers.get(pos.trade_asset_id or pos.yes_asset_id)
            bid = book.best_bid if book else 0.01
            if bid <= 0:
                bid = 0.01

            # Market-sell at bid (taker) to guarantee execution
            exit_order = await self.executor.place_limit_order(
                market_id=pos.market_id,
                asset_id=pos.trade_asset_id or pos.yes_asset_id,
                side=OrderSide.SELL,
                price=round(bid, 2),
                size=pos.effective_size,
                post_only=False,  # TAKER — must exit now
            )
            pos.exit_order = exit_order
            pos.state = PositionState.EXIT_PENDING
            pos.exit_reason = f"{reason}_emergency_dump"
            log.warning(
                "combo_emergency_dump",
                combo_id=combo.combo_id,
                market_id=pos.market_id[:16],
                sell_price=bid,
                shares=pos.effective_size,
            )

        log.warning(
            "combo_abandoned_with_dump",
            combo_id=combo.combo_id,
            event_id=combo.event_id,
            reason=reason,
            filled_dumped=len(all_filled_now),
            unfilled_failed=len(hedge_failures),
        )

    def combo_positions_for_pce(
        self,
        combo_positions: dict[str, "ComboPosition"],
    ) -> list[dict[str, Any]]:
        """Serialize active combo positions for the PCE worker.

        Returns a list of dicts matching ``_MinimalPosition`` kwargs:
        ``market_id, entry_price, size, filled_size, trade_asset_id,
        no_asset_id, event_id, entry_size, trade_side``.

        MUST be full dicts, NOT tuples — the PCE worker reconstructs
        ``_MinimalPosition(**p)`` and would crash on positional args.
        """
        result: list[dict[str, Any]] = []
        for combo in combo_positions.values():
            if combo.state not in (ComboState.ENTRY_PENDING, ComboState.PARTIAL_FILL, ComboState.ALL_FILLED):
                continue
            for pos in combo.legs.values():
                if pos.state not in (PositionState.ENTRY_PENDING, PositionState.ENTRY_FILLED):
                    continue
                if pos.entry_order is None and pos.state != PositionState.ENTRY_FILLED:
                    continue
                result.append({
                    "market_id": pos.market_id,
                    "entry_price": pos.entry_price,
                    "size": pos.entry_size,
                    "filled_size": pos.filled_size,
                    "trade_asset_id": pos.trade_asset_id,
                    "no_asset_id": pos.no_asset_id,
                    "event_id": pos.event_id,
                    "entry_size": pos.entry_size,
                    "trade_side": pos.trade_side,
                })
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  Shadow Performance Tracker — virtual execution for shadow strategies
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ShadowPosition:
    """In-memory virtual position for a shadow strategy signal."""

    id: str
    signal_source: str       # e.g. "SI-3", "RPE-Experimental"
    market_id: str
    asset_id: str
    direction: str            # "YES" or "NO"
    entry_price: float
    entry_size: float
    entry_time: float
    target_price: float
    stop_price: float
    fee_enabled: bool = True
    entry_fee_bps: int = 0
    exit_fee_bps: int = 0
    zscore: float = 0.0
    confidence: float = 0.0
    reference_price: float = 0.0
    reference_price_band: str = ""
    toxicity_index: float = 0.0


class ShadowExecutionTracker:
    """Tracks virtual shadow positions and closes them when the L2 book
    touches the take-profit or stop-loss level.

    Lifecycle:
      1. ``open_shadow_position()`` — creates a virtual entry from a
         shadow signal, simulating maker/taker routing and computing
         TP/SL with fee deductions.
      2. ``tick()`` — called on every L2 book update.  Checks whether
         any open shadow position's TP or SL has been touched.
      3. On closure, ``record_shadow_trade()`` is called on the
         ``TradeStore`` to persist the counterfactual result.
    """

    def __init__(self, trade_store: Any | None = None):
        self._trade_store = trade_store
        self._positions: dict[str, ShadowPosition] = {}
        self._next_id = 1

    def open_shadow_position(
        self,
        *,
        signal_source: str,
        market_id: str,
        asset_id: str,
        direction: str,
        best_ask: float,
        best_bid: float,
        entry_price: float | None = None,
        target_price: float | None = None,
        stop_price: float | None = None,
        entry_size: float = 10.0,
        fee_enabled: bool = True,
        zscore: float = 0.0,
        confidence: float = 0.0,
        reference_price: float = 0.0,
        reference_price_band: str = "",
        toxicity_index: float = 0.0,
        tp_alpha: float = 0.05,
        sl_cents: float = 4.0,
    ) -> ShadowPosition | None:
        """Create a virtual shadow position simulating maker entry.

        Entry price: best_ask - 0.01 (maker, undercutting the ask).
        TP: entry_price + tp_alpha (capped at 0.99).
        SL: entry_price - sl_cents/100 (floored at 0.01).
        Fees: Polymarket dynamic fee curve applied to both legs.
        """
        if entry_price is None:
            entry_price = round(best_ask - 0.01, 2)
        else:
            entry_price = round(float(entry_price), 2)
        if entry_price <= 0.01 or entry_price >= 0.99:
            return None

        # Compute fees via the dynamic fee curve
        entry_fee = get_fee_rate(entry_price, fee_enabled=fee_enabled)
        entry_fee_bps = round(entry_fee * 10000)

        # TP target: partial mean-reversion alpha capture
        if target_price is None:
            target_price = round(min(0.99, entry_price + tp_alpha), 2)
        else:
            target_price = round(float(target_price), 2)
        exit_fee = get_fee_rate(target_price, fee_enabled=fee_enabled)
        exit_fee_bps = round(exit_fee * 10000)

        # SL: fee-adaptive stop-loss
        if stop_price is None:
            sl_price = round(max(0.01, entry_price - sl_cents / 100.0), 2)
        else:
            sl_price = round(float(stop_price), 2)

        pos_id = f"SHADOW-{signal_source}-{self._next_id}"
        self._next_id += 1

        pos = ShadowPosition(
            id=pos_id,
            signal_source=signal_source,
            market_id=market_id,
            asset_id=asset_id,
            direction=direction,
            entry_price=entry_price,
            entry_size=entry_size,
            entry_time=time.time(),
            target_price=target_price,
            stop_price=sl_price,
            fee_enabled=fee_enabled,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
            zscore=zscore,
            confidence=confidence,
            reference_price=reference_price,
            reference_price_band=reference_price_band,
            toxicity_index=toxicity_index,
        )

        self._positions[pos.id] = pos
        log.info(
            "shadow_position_opened",
            pos_id=pos.id,
            signal_source=signal_source,
            market=market_id,
            direction=direction,
            entry=entry_price,
            target=target_price,
            stop=sl_price,
            size=entry_size,
        )
        return pos

    async def tick(
        self,
        asset_id: str,
        best_bid: float,
        best_ask: float,
    ) -> list[ShadowPosition]:
        """Check all open shadow positions against current L2 prices.

        A shadow position is closed when:
          - best_bid >= target_price  → take-profit hit
          - best_ask <= stop_price    → stop-loss hit

        Returns a list of positions that were closed on this tick.
        """
        closed: list[ShadowPosition] = []

        for pos in list(self._positions.values()):
            if pos.asset_id != asset_id:
                continue

            exit_price = 0.0
            exit_reason = ""

            if best_bid > 0 and best_bid >= pos.target_price:
                exit_price = pos.target_price
                exit_reason = "target"
            elif best_ask > 0 and best_ask <= pos.stop_price:
                exit_price = pos.stop_price
                exit_reason = "stop_loss"
            else:
                continue

            # Compute net PnL with fee deductions
            pnl = compute_net_pnl_cents(
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size=pos.entry_size,
                fee_enabled=pos.fee_enabled,
                is_maker_entry=(pos.entry_fee_bps == 0 and pos.fee_enabled),
            )

            now = time.time()

            log.info(
                "shadow_position_closed",
                pos_id=pos.id,
                signal_source=pos.signal_source,
                entry=pos.entry_price,
                exit=exit_price,
                pnl=pnl,
                reason=exit_reason,
            )

            # Persist to DB
            if self._trade_store is not None:
                try:
                    await self._trade_store.record_shadow_trade(
                        trade_id=pos.id,
                        signal_source=pos.signal_source,
                        market_id=pos.market_id,
                        asset_id=pos.asset_id,
                        direction=pos.direction,
                        reference_price=pos.reference_price,
                        reference_price_band=pos.reference_price_band,
                        entry_price=pos.entry_price,
                        entry_size=pos.entry_size,
                        entry_time=pos.entry_time,
                        target_price=pos.target_price,
                        stop_price=pos.stop_price,
                        exit_price=exit_price,
                        exit_time=now,
                        exit_reason=exit_reason,
                        pnl_cents=pnl,
                        entry_fee_bps=pos.entry_fee_bps,
                        exit_fee_bps=pos.exit_fee_bps,
                        zscore=pos.zscore,
                        confidence=pos.confidence,
                        toxicity_index=pos.toxicity_index,
                    )
                except Exception:
                    log.warning("shadow_trade_persist_failed", pos_id=pos.id, exc_info=True)

            del self._positions[pos.id]
            closed.append(pos)

        return closed

    @property
    def open_count(self) -> int:
        return len(self._positions)

    def get_open_shadow_positions(self) -> list[ShadowPosition]:
        return list(self._positions.values())
