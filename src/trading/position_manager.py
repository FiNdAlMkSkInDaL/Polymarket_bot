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
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.config import DeploymentEnv, settings
from src.core.guard import DeploymentGuard
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator
from src.data.orderbook import OrderbookTracker
from src.signals.edge_filter import ConfluenceContext, compute_confluence_discount, compute_edge_score
from src.signals.iceberg_detector import IcebergDetector
from src.signals.panic_detector import PanicSignal
from src.signals.drift_signal import DriftSignal
from src.signals.signal_framework import BaseSignal, VacuumSignal
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.fees import (
    compute_adaptive_stop_loss_cents,
    compute_adaptive_trailing_offset_cents,
    compute_net_pnl_cents,
)
from src.trading.sizer import (
    KellyResult,
    SizingResult,
    compute_depth_aware_size,
    compute_kelly_size,
)
from src.trading.take_profit import TakeProfitResult, compute_take_profit

log = get_logger(__name__)


class PositionState(str, Enum):
    ENTRY_PENDING = "ENTRY_PENDING"
    ENTRY_FILLED = "ENTRY_FILLED"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


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
    tp_result: TakeProfitResult | None = None
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""

    # Bidirectional support (RPE)
    trade_asset_id: str = ""   # actual token traded (YES or NO); fallback to no_asset_id
    trade_side: str = "NO"     # "YES" or "NO" — which outcome token we bought
    yes_asset_id: str = ""     # YES token id (for RPE YES-side entries)

    # Signal metadata
    signal: BaseSignal | None = None

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

    # V4: Probe sizing flag
    is_probe: bool = False

    # Pillar 16: Alpha-source attribution
    signal_type: str = ""         # "panic", "drift", "rpe", "stink_bid"
    meta_weight: float = 1.0     # SI-6 MetaStrategyController weight at entry

    # Pillar 11.3: initial vol multiplier used at open time so the
    # stop-loss monitor can decay it back toward 1.0 for stale trades.
    sl_vol_multiplier: float = 1.0

    created_at: float = field(default_factory=time.time)

    @property
    def effective_size(self) -> float:
        """Shares to use for exit orders / PnL — filled_size if set, else entry_size."""
        return self.filled_size if self.filled_size > 0 else self.entry_size


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
        self._stealth = None  # StealthExecutor — injected by bot.py when stealth_enabled
        self._ohlcv_aggs: dict[str, OHLCVAggregator] | None = None  # asset_id → aggregator; injected by bot.py
        self._positions: dict[str, Position] = {}
        self._next_id = 1
        self._wallet_balance_usd: float = 0.0
        self._daily_pnl_cents: float = 0.0

        # ── Trade store stats cache (OE-1) ─────────────────────────────
        # Caches get_stats() results with a 5-second TTL to avoid
        # blocking aiosqlite reads on every signal evaluation.
        self._stats_cache: dict[str, Any] | None = None
        self._stats_cache_time: float = 0.0
        self._STATS_CACHE_TTL: float = 5.0
        self._daily_pnl_date: str = ""  # YYYY-MM-DD
        self._cumulative_pnl_cents: float = 0.0
        self._peak_pnl_cents: float = 0.0
        self._max_drawdown_cents: float = 0.0
        self._circuit_breaker_tripped: bool = False
        # Serialize entry attempts to prevent concurrent entries on
        # the same market/event from racing past risk gates.
        self._entry_lock = asyncio.Lock()
        # V4 Probe harvesting: track which probes have been scaled in already
        self._probe_scaled_set: set[str] = set()
        # V4 Probe harvesting: track which probes have been scaled in already
        self._probe_scaled_set: set[str] = set()
        # Per-market cooldown after stop-loss to prevent rapid re-entry
        self._stop_loss_cooldowns: dict[str, float] = {}

    # ── Wallet balance ─────────────────────────────────────────────────────
    def set_wallet_balance(self, usd: float) -> None:
        self._wallet_balance_usd = usd

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_tripped

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
    async def _get_cached_stats(self) -> dict[str, Any]:
        """Return trade-store stats with a 5-second TTL cache.

        Avoids a blocking aiosqlite read on every signal evaluation.
        Cache is invalidated whenever a trade is recorded.
        """
        now = time.time()
        if (
            self._stats_cache is not None
            and (now - self._stats_cache_time) < self._STATS_CACHE_TTL
        ):
            return self._stats_cache
        if self._trade_store is None:
            return {}
        try:
            self._stats_cache = await self._trade_store.get_stats()
            self._stats_cache_time = now
        except Exception:
            log.warning("trade_store_stats_unavailable")
            if self._stats_cache is None:
                self._stats_cache = {}
        return self._stats_cache

    def _invalidate_stats_cache(self) -> None:
        """Invalidate the stats cache (call after recording a trade)."""
        self._stats_cache = None
        self._stats_cache_time = 0.0

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
        """Send VaR check to PCE worker and await response (50ms timeout).

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
        # Poll response queue with 50ms total timeout
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._var_response_q.get(timeout=0.01),
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
            return await self._open_position_inner(
                signal, no_aggregator,
                no_book=no_book, event_id=event_id,
                days_to_resolution=days_to_resolution,
                book_depth_ratio=book_depth_ratio,
                fee_enabled=fee_enabled,
                signal_metadata=signal_metadata,
            )

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
        z_threshold = strat.zscore_threshold
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
            exit_fee_cents = get_fee_rate(exit_est_price, fee_enabled=fee_enabled) * 100.0
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

        # Reconcile actual fill quantity.  The chaser or paper-fill may
        # have set filled_size already; if not, infer from the order.
        if pos.filled_size <= 0:
            if pos.entry_order and pos.entry_order.filled_size > 0:
                pos.filled_size = pos.entry_order.filled_size
            else:
                pos.filled_size = pos.entry_size

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

    # ── Handle exit fill ───────────────────────────────────────────────────
    def on_exit_filled(self, pos: Position, reason: str = "target") -> None:
        """Close the position after exit fill or forced liquidation."""
        pos.state = PositionState.CLOSED
        pos.exit_price = (
            pos.exit_order.filled_avg_price if pos.exit_order else pos.target_price
        )
        pos.exit_time = time.time()
        pos.exit_reason = reason

        # Net-of-fee PnL using actual fill size
        pos.pnl_cents = compute_net_pnl_cents(
            entry_price=pos.entry_price,
            exit_price=pos.exit_price,
            size=pos.effective_size,
            fee_enabled=pos.fee_enabled,
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
        if reason in ("stop_loss", "preemptive_liquidity_drain"):
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
        )

        # Auto-cleanup old closed/cancelled positions
        self.cleanup_closed()

    # ── Timeout enforcement ────────────────────────────────────────────────
    async def check_timeouts(self) -> None:
        """Cancel stale entry orders, check stop-losses, and force-exit stale positions."""
        now = time.time()
        stop_loss_cents = settings.strategy.stop_loss_cents

        for pos in list(self._positions.values()):
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
                elapsed = now - pos.entry_time

                # Stop-loss: check if current NO price moved against us
                should_stop = False
                if stop_loss_cents > 0 and pos.entry_price > 0:
                    # Unrealised loss check deferred to bot loop where
                    # we have access to current prices — here we only
                    # handle cases signalled by the bot via force_stop_loss()
                    pass

                if elapsed > settings.strategy.exit_timeout_seconds:
                    # Cancel chaser task if running
                    if pos.exit_chaser_task and not pos.exit_chaser_task.done():
                        pos.exit_chaser_task.cancel()
                    # Cancel the existing limit sell
                    if pos.exit_order:
                        await self.executor.cancel_order(pos.exit_order)

                    # Place a market sell (IOC at best bid — simulated by
                    # placing at a very low price in paper mode)
                    exit_asset = pos.trade_asset_id or pos.no_asset_id
                    exit_order = await self.executor.place_limit_order(
                        market_id=pos.market_id,
                        asset_id=exit_asset,
                        side=OrderSide.SELL,
                        price=0.01,  # effectively market sell
                        size=pos.effective_size,
                    )
                    pos.exit_order = exit_order
                    # In paper mode, simulate a realistic fill at the
                    # current best bid rather than the literal 0.01.
                    if self.executor.paper_mode:
                        fill_price = self._paper_market_sell_price(pos)
                        exit_order.filled_avg_price = fill_price
                        exit_order.filled_size = pos.effective_size
                        exit_order.status = OrderStatus.FILLED
                        self.on_exit_filled(pos, reason="timeout")
                    else:
                        pos.exit_reason = "timeout"
                        log.info(
                            "timeout_exit_placed",
                            pos_id=pos.id,
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
            pos.exit_reason = reason
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
