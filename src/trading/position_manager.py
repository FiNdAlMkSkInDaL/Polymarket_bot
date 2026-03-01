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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.config import settings
from src.core.guard import DeploymentGuard
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator
from src.data.orderbook import OrderbookTracker
from src.signals.edge_filter import compute_edge_score
from src.signals.panic_detector import PanicSignal
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.fees import compute_adaptive_stop_loss_cents, compute_net_pnl_cents
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
    signal: PanicSignal | None = None

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
    ):
        self.executor = executor
        self.max_open = max_open_positions or settings.strategy.max_open_positions
        self._trade_store = trade_store
        self._guard = guard
        self._pce = pce  # PortfolioCorrelationEngine (Pillar 15)
        self._positions: dict[str, Position] = {}
        self._next_id = 1
        self._wallet_balance_usd: float = 0.0
        self._daily_pnl_cents: float = 0.0
        self._daily_pnl_date: str = ""  # YYYY-MM-DD
        self._cumulative_pnl_cents: float = 0.0
        self._peak_pnl_cents: float = 0.0
        self._max_drawdown_cents: float = 0.0
        self._circuit_breaker_tripped: bool = False

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

    # ── Shared risk gates ──────────────────────────────────────────────────
    def _check_risk_gates(
        self, market_id: str, event_id: str = "", *, trade_direction: str = "NO"
    ) -> tuple[bool, float]:
        """Run all risk gates shared between panic and RPE entries.

        Returns (passed: bool, max_trade_usd: float).  When passed
        is False, the caller must abort.  max_trade_usd is the
        capital-allocation cap for this entry attempt.
        """
        strat = settings.strategy

        # ── Circuit breaker check ──────────────────────────────────────────
        if self._circuit_breaker_tripped:
            log.warning("circuit_breaker_active", reason="daily_loss_or_drawdown")
            return False, 0.0

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
            return False, 0.0

        # ── Max drawdown check ─────────────────────────────────────────────
        if self._max_drawdown_cents >= strat.max_drawdown_cents:
            self._circuit_breaker_tripped = True
            log.warning(
                "max_drawdown_hit",
                drawdown=round(self._max_drawdown_cents, 2),
                limit=strat.max_drawdown_cents,
            )
            return False, 0.0

        # ── Risk gate: max open positions ──────────────────────────────────
        open_positions = self.get_open_positions()
        if len(open_positions) >= self.max_open:
            log.warning("max_positions_reached", open=len(open_positions))
            return False, 0.0

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
            return False, 0.0

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
                return False, 0.0

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
            return False, 0.0

        # ── Risk gate: capital allocation ──────────────────────────────────
        max_trade = min(
            strat.max_trade_size_usd,
            self._wallet_balance_usd * strat.max_wallet_risk_pct / 100.0,
        )
        if max_trade <= 0:
            log.warning("insufficient_balance", balance=self._wallet_balance_usd)
            return False, 0.0

        # ── Risk gate: Portfolio Correlation Engine (PCE) VaR ──────────────
        if self._pce is not None:
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
                    return False, 0.0
                max_trade = min(max_trade, sizing_result.cap_usd)
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
                    return False, 0.0

        return True, max_trade

    # ── Open a new position on signal ──────────────────────────────────────
    async def open_position(
        self,
        signal: PanicSignal,
        no_aggregator: OHLCVAggregator,
        *,
        no_book: OrderbookTracker | None = None,
        event_id: str = "",
        days_to_resolution: int = 30,
        book_depth_ratio: float = 1.0,
        fee_enabled: bool = True,
        signal_metadata: dict | None = None,
    ) -> Position | None:
        """Attempt to open a mean-reversion position on a panic signal.

        Parameters
        ----------
        no_book:
            Live L2 order book for the NO token.  Used by the depth-aware
            sizer to cap order size.  Falls back to fixed sizing with
            a 50% haircut if ``None``.
        fee_enabled:
            Whether this market category charges dynamic fees (crypto/sports).
            Used to compute the fee-adaptive stop-loss and net PnL.
        """
        strat = settings.strategy

        passed, max_trade = self._check_risk_gates(signal.market_id, event_id)
        if not passed:
            return None

        # Entry price: undercut the best ask by 1¢ to ensure maker
        entry_price = round(signal.no_best_ask - 0.01, 2)
        if entry_price <= 0:
            log.warning("entry_price_invalid", ask=signal.no_best_ask)
            return None

        # ── Edge quality gate (information-theoretic) ──────────────────
        # Weighted geometric mean of regime entropy, fee efficiency,
        # tick viability, and signal quality.  Replaces the old naive
        # min/max price-range clamp with a principled EV gate.
        edge = compute_edge_score(
            entry_price=entry_price,
            no_vwap=no_aggregator.rolling_vwap,
            zscore=signal.zscore,
            volume_ratio=signal.volume_ratio,
            whale_confluence=signal.whale_confluence,
            fee_enabled=fee_enabled,
        )
        if not edge.viable:
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
        if self._trade_store is not None:
            try:
                stats = await self._trade_store.get_stats()
                win_rate = stats.get("win_rate", 0.0)
                avg_win_cents = stats.get("avg_win_cents", 0.0)
                avg_loss_cents = stats.get("avg_loss_cents", 0.0)
                total_trades = stats.get("total_trades", 0)
            except Exception:
                log.warning("trade_store_stats_unavailable")

        # Normalise signal strength to 0.0–1.0 for Kelly sizer.
        # zscore is typically 2–5+; map the excess above threshold to [0, 1].
        z_threshold = strat.zscore_threshold
        signal_score = min(1.0, max(0.0, (signal.zscore - z_threshold) / z_threshold)) if z_threshold > 0 else 0.5

        kelly_result = compute_kelly_size(
            signal_score=signal_score,
            win_rate=win_rate,
            avg_win_cents=avg_win_cents,
            avg_loss_cents=avg_loss_cents,
            bankroll_usd=self._wallet_balance_usd,
            entry_price=entry_price,
            max_trade_usd=max_trade,
            book=no_book,
            signal_metadata=signal_metadata,
            total_trades=total_trades,
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
        if self._pce is not None:
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
            whale_confluence=signal.whale_confluence,
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

        # Place entry order
        pos_id = f"POS-{self._next_id}"
        self._next_id += 1

        # Compute fee-adaptive stop-loss
        sl_trigger = compute_adaptive_stop_loss_cents(
            sl_base_cents=strat.stop_loss_cents,
            entry_price=entry_price,
            fee_enabled=fee_enabled,
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
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
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
        """
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

        passed, max_trade = self._check_risk_gates(
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
        if self._trade_store is not None:
            try:
                stats = await self._trade_store.get_stats()
                win_rate = stats.get("win_rate", 0.0)
                avg_win_cents = stats.get("avg_win_cents", 0.0)
                avg_loss_cents = stats.get("avg_loss_cents", 0.0)
                total_trades_rpe = stats.get("total_trades", 0)
            except Exception:
                log.warning("trade_store_stats_unavailable")

        signal_score = min(1.0, divergence_z / 3.0)

        # Half Kelly for RPE entries — lower conviction than panic signals
        rpe_kelly_fraction = (strat.kelly_fraction * 0.5)

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

        # ── PCE concentration haircut (Pillar 15) ─────────────────────
        if self._pce is not None:
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

        if self._guard is not None:
            entry_size = self._guard.get_allowed_trade_shares(
                entry_size, entry_price
            )

        if entry_size < 1:
            log.info("rpe_skip_insufficient_size")
            return None

        # ── Take-profit target ─────────────────────────────────────────
        # Mean-reversion toward model estimate: price should converge
        # to model_probability.
        alpha = strat.alpha_default
        target_price = round(
            entry_price + alpha * (model_probability - entry_price), 2
        )
        # When buying NO (market overpriced), model_prob < market_price,
        # so we bought at 1-market_price on the NO side.  The target
        # needs to be above entry for a profitable exit.
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

        # Fee-adaptive stop-loss
        sl_trigger = compute_adaptive_stop_loss_cents(
            sl_base_cents=strat.stop_loss_cents,
            entry_price=entry_price,
            fee_enabled=fee_enabled,
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
        )

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
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=0,
        )

        self._positions[pos.id] = pos

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
                    # In paper mode, check for an immediate fill;
                    # in live mode the fill will be detected by the
                    # order-status poller or the next check_timeouts cycle.
                    if self.executor.paper_mode:
                        self.on_exit_filled(pos, reason="timeout")
                    else:
                        pos.exit_reason = "timeout"
                        log.info(
                            "timeout_exit_placed",
                            pos_id=pos.id,
                            note="awaiting CLOB fill confirmation",
                        )

    async def force_stop_loss(self, pos: Position) -> None:
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
        # In paper mode, close immediately; in live mode wait for CLOB
        # fill confirmation from the order-status poller.
        if self.executor.paper_mode:
            self.on_exit_filled(pos, reason="stop_loss")
        else:
            pos.exit_reason = "stop_loss"
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
