"""
SI-9 Combinatorial Arbitrage — signal detection and sizing for mutually
exclusive Polymarket event clusters.

When the sum of YES best-bids across all legs of a negRisk cluster
drops below ``1.0 - margin``, a risk-free arbitrage exists:
buy every YES token at best_bid (maker), hold to resolution, collect
the guaranteed $1.00 payout on the winning leg.

This module provides:
    - ``ComboLeg`` / ``ComboArbSignal`` — maker/taker-aware combo payloads.
    - ``ComboSizer`` — share-count-pegged sizer (NOT Kelly/USD).
    - ``ComboArbDetector`` — scans clusters and emits signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.config import settings
from src.core.logger import get_logger
from src.data.arb_clusters import ArbCluster
from src.data.orderbook import OrderbookTracker
from src.signals.microstructure_utils import CrossBookSyncGate
from src.signals.ofi_momentum import OFIMomentumDetector
from src.signals.signal_framework import BaseSignal, SignalGenerator, SignalResult

log = get_logger(__name__)


SI9_MIN_SUM_BIDS = 0.85
SI9_MAX_EDGE_DOLLARS = 0.15
SI9_GUARDRAIL_LOG_INTERVAL_SECONDS = 60 * 60
SI9_GUARDRAIL_LOG_SUM_BIDS_DELTA = 0.05


# ═══════════════════════════════════════════════════════════════════════════
#  Data payloads
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ComboLeg:
    """One leg of a combinatorial arbitrage opportunity."""

    market_id: str         # condition_id
    yes_token_id: str      # token to BUY
    no_token_id: str       # NO token (needed for emergency unwind reference)
    best_bid: float        # current YES best_bid from book
    best_ask: float        # current YES best_ask
    target_price: float    # price to place the BUY order at
    target_shares: float   # shares to buy (IDENTICAL across all legs)
    question: str = ""     # human label for logging
    spread_width: float = 0.0
    bid_depth_usd: float = 0.0
    bid_depth_shares: float = 0.0
    market_liquidity_usd: float = 0.0
    daily_volume_usd: float = 0.0
    routing_reason: str = ""


@dataclass
class ComboArbSignal(BaseSignal):
    """Emitted when a mutually exclusive cluster is mis-priced.

    Carries the routing split and computed edge so the execution layer
    can work the bottleneck leg passively and cross the remaining legs.
    """

    cluster_event_id: str = ""
    maker_leg: ComboLeg | None = None
    taker_legs: list[ComboLeg] = field(default_factory=list)
    sum_best_bids: float = 0.0   # Σ YES best_bids across legs
    edge_cents: float = 0.0      # (1.0 - sum_best_bids)*100 - min_margin_cents
    margin_used: float = 0.0     # si9_min_margin_cents at evaluation time
    latency_option_premium_dollars: float = 0.0
    required_edge_dollars: float = 0.0
    target_shares: float = 0.0   # uniform share count across all legs
    total_collateral: float = 0.0  # Σ(target_price_i * target_shares)

    @property
    def legs(self) -> list[ComboLeg]:
        """Backward-compatible all-legs view with maker leg first."""
        if self.maker_leg is None:
            return list(self.taker_legs)
        return [self.maker_leg, *self.taker_legs]


@dataclass
class ComboArbDeferral:
    """Structured reason for an SI-9 cluster being deferred."""

    event_id: str
    reason: str
    maker_leg: str
    market_id: str
    question: str = ""
    rolling_vi: float = 0.0
    current_vi: float = 0.0
    threshold: float = 0.0
    deferred_at: float = 0.0
    defer_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  ComboSizer — share-count pegging (NOT Kelly/USD)
# ═══════════════════════════════════════════════════════════════════════════


class ComboSizer:
    """Compute a uniform share count for all legs of a combo arb.

    Combinatorial arbitrage REQUIRES buying the exact same number of
    shares S across all N outcomes.  The total collateral is:

        C = S * Σ(target_price_i)

    We solve for S given:
      - wallet_balance: available USDC
      - max_per_combo_usd: hard cap on total combo collateral
      - max_wallet_risk_pct: fraction of wallet we may commit
      - target_prices: list of per-leg prices

    The sizer ensures S₁ = S₂ = ... = Sₙ, which is the fundamental
    invariant of Dutch Book arbitrage.  Using Kelly/USD sizing would
    produce unequal share counts and leave the portfolio unhedged.
    """

    @staticmethod
    def compute_shares(
        target_prices: list[float],
        wallet_balance: float,
        max_per_combo_usd: float,
        max_wallet_risk_pct: float,
        min_leg_depth_shares: float,
    ) -> float:
        """Return the uniform share count, or 0.0 if infeasible.

        Parameters
        ----------
        target_prices :
            Price per share for each leg (usually best_bid or best_bid+1¢).
        wallet_balance :
            Current USDC balance.
        max_per_combo_usd :
            Hard cap on total collateral for one combo.
        max_wallet_risk_pct :
            Maximum percent of wallet to risk on combos (0-100).
        min_leg_depth_shares :
            Thinnest leg depth in shares — we cap S to avoid slamming
            an illiquid leg.
        """
        if not target_prices or any(p <= 0 for p in target_prices):
            return 0.0

        sum_prices = sum(target_prices)
        if sum_prices <= 0:
            return 0.0

        # Budget: min of (hard cap, wallet fraction)
        wallet_budget = wallet_balance * max_wallet_risk_pct / 100.0
        budget = min(max_per_combo_usd, wallet_budget)
        if budget <= 0:
            return 0.0

        # S = budget / Σ(price_i)  — same share count across all legs
        shares = budget / sum_prices

        # Cap to thinnest leg depth (take no more than 50% of depth)
        if min_leg_depth_shares > 0:
            shares = min(shares, min_leg_depth_shares * 0.5)

        # Minimum viable: at least 1 share
        if shares < 1.0:
            return 0.0

        return round(shares, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════════


class ComboArbDetector(SignalGenerator):
    """Scans arb clusters for actionable mis-pricings.

    For each active cluster:
      1. Read ``best_bid`` from each leg's YES-token OrderbookTracker.
      2. Compute ``sum_bids = Σ best_bid(YES_i)``.
      3. If ``sum_bids < 1.0 - min_margin_cents/100``: arb detected.
      4. Size via ``ComboSizer`` (share-count pegging, NOT Kelly).

    Parameters
    ----------
    book_trackers :
        Mapping ``{yes_token_id → OrderbookTracker}`` for live BBO reads.
    """

    def __init__(
        self,
        book_trackers: dict[str, OrderbookTracker],
        aggregators: dict[str, Any] | None = None,
        on_sync_block: Callable[[Any], None] | None = None,
    ) -> None:
        self._books = book_trackers
        self._aggregators = aggregators or {}
        self._sizer = ComboSizer()
        self._sync_gate = CrossBookSyncGate()
        self._on_sync_block = on_sync_block
        self._ofi_detectors: dict[str, OFIMomentumDetector] = {}
        self._guardrail_rejection_log_state: dict[str, tuple[float, float]] = {}
        self._maker_leg_ofi_paused: dict[str, bool] = {}
        self._active_deferrals: dict[str, ComboArbDeferral] = {}
        self._recent_resumes: dict[str, ComboArbDeferral] = {}

    @property
    def name(self) -> str:
        return "combo_arb"

    # ── SignalGenerator interface (single-market, not used for combos) ──
    def evaluate(self, **kwargs: Any) -> ComboArbSignal | None:
        cluster = kwargs.get("cluster")
        if not isinstance(cluster, ArbCluster):
            return None

        wallet_balance = float(kwargs.get("wallet_balance", 0.0))
        return self._evaluate_cluster(cluster, wallet_balance=wallet_balance)

    # ── Cluster-level evaluation (called by the combo loop) ────────────
    def evaluate_cluster(
        self,
        cluster: ArbCluster,
        wallet_balance: float = 0.0,
    ) -> ComboArbSignal | None:
        return self._evaluate_cluster(cluster, wallet_balance=wallet_balance)

    def _rank_legs_for_routing(self, legs: list[ComboLeg]) -> list[ComboLeg]:
        """Rank legs so the hardest leg to fill is worked passively first.

        The bottleneck leg is the one with:
          1. the widest spread
          2. the thinnest bid-side depth when spreads tie
        """
        return sorted(
            legs,
            key=lambda leg: (
                -leg.spread_width,
                leg.bid_depth_usd,
                leg.bid_depth_shares,
                leg.market_liquidity_usd,
                leg.daily_volume_usd,
                leg.question,
            ),
        )

    def _routing_reason_for_maker(
        self,
        maker_leg: ComboLeg,
        ranked_legs: list[ComboLeg],
    ) -> str:
        """Explain why the maker leg was selected for passive routing."""
        if len(ranked_legs) == 1:
            return (
                "widest_spread "
                f"spread_width={maker_leg.spread_width:.4f}"
            )

        next_leg = ranked_legs[1]
        if maker_leg.spread_width > next_leg.spread_width:
            return (
                "widest_spread "
                f"spread_width={maker_leg.spread_width:.4f} "
                f"next_spread_width={next_leg.spread_width:.4f}"
            )

        return (
            "thinnest_bid_depth "
            f"bid_depth_usd={maker_leg.bid_depth_usd:.2f} "
            f"spread_width={maker_leg.spread_width:.4f}"
        )

    def _evaluate_cluster(
        self,
        cluster: ArbCluster,
        wallet_balance: float = 0.0,
    ) -> ComboArbSignal | None:
        """Check a single cluster for arb.  Returns signal or ``None``.

        Parameters
        ----------
        cluster :
            The cluster to evaluate.
        wallet_balance :
            Current USDC balance for sizing.
        """
        strat = settings.strategy
        min_margin = strat.si9_min_margin_cents / 100.0  # convert to $
        buffer = strat.si9_margin_buffer_cents / 100.0
        min_depth_usd = strat.si9_min_leg_depth_usd

        legs: list[ComboLeg] = []
        snapshots: list[Any] = []
        sum_bids = 0.0
        min_bid_depth_shares = float("inf")

        for mkt in cluster.legs:
            book = self._books.get(mkt.yes_token_id)
            if book is None:
                return None  # no book data — skip cluster

            snap = book.snapshot()
            if not snap.fresh:
                return None  # stale data — unsafe for arb
            snapshots.append(snap)

            bid = snap.best_bid
            ask = snap.best_ask
            if bid <= 0:
                return None  # no bid on a leg — no arb

            # Check bid-side depth meets minimum
            bid_depth_usd = snap.bid_depth_usd
            if bid_depth_usd < min_depth_usd:
                return None  # too thin to fill

            # Compute depth in shares at bid price for sizer cap
            depth_shares = bid_depth_usd / bid if bid > 0 else 0
            if depth_shares < min_bid_depth_shares:
                min_bid_depth_shares = depth_shares

            sum_bids += bid

            legs.append(ComboLeg(
                market_id=mkt.condition_id,
                yes_token_id=mkt.yes_token_id,
                no_token_id=mkt.no_token_id,
                best_bid=bid,
                best_ask=ask,
                target_price=0.0,  # computed below
                target_shares=0.0,  # computed below
                question=mkt.question[:60],
                spread_width=round(snap.spread, 4),
                bid_depth_usd=round(bid_depth_usd, 2),
                bid_depth_shares=round(depth_shares, 4),
                market_liquidity_usd=round(mkt.liquidity_usd, 2),
                daily_volume_usd=round(mkt.daily_volume_usd, 2),
            ))

        sync_assessment = self._sync_gate.assess(snapshots)
        if not sync_assessment.is_synchronized:
            if self._on_sync_block is not None:
                self._on_sync_block(sync_assessment)
            return None

        edge_dollars = 1.0 - sum_bids

        # ── Arb check: Σ(YES best_bids) < 1.0 - margin ───────────────
        base_threshold = 1.0 - min_margin
        if sum_bids >= base_threshold:
            return None  # no arb

        # Reject ghost-town books with implausibly large gaps. SI-9 should
        # only work tight, liquid mispricings rather than dead markets.
        if sum_bids < SI9_MIN_SUM_BIDS or edge_dollars > SI9_MAX_EDGE_DOLLARS:
            if self._should_log_guardrail_rejection(cluster.event_id, sum_bids):
                log.info(
                    "combo_arb_guardrail_rejected",
                    event_id=cluster.event_id,
                    sum_bids=round(sum_bids, 4),
                    edge_dollars=round(edge_dollars, 4),
                    min_sum_bids=SI9_MIN_SUM_BIDS,
                    max_edge_dollars=SI9_MAX_EDGE_DOLLARS,
                    throttle_interval_s=SI9_GUARDRAIL_LOG_INTERVAL_SECONDS,
                    sum_bids_delta_threshold=SI9_GUARDRAIL_LOG_SUM_BIDS_DELTA,
                )
            return None

        ranked_legs = self._rank_legs_for_routing(legs)
        maker_leg = ranked_legs[0]
        taker_legs = ranked_legs[1:]
        maker_leg.routing_reason = self._routing_reason_for_maker(
            maker_leg,
            ranked_legs,
        )

        latency_option_premium = self._latency_option_premium(taker_legs)
        required_edge_dollars = min_margin + latency_option_premium
        if edge_dollars <= required_edge_dollars:
            return None

        edge_cents = (edge_dollars - required_edge_dollars) * 100.0

        if self._maker_leg_has_toxic_ofi(cluster.event_id, maker_leg):
            return None

        # ── Target prices: maker works passively, takers cross ─────────
        if edge_dollars > required_edge_dollars + buffer:
            maker_leg.target_price = round(min(0.99, maker_leg.best_bid + 0.01), 2)
        else:
            maker_leg.target_price = round(maker_leg.best_bid, 2)

        for leg in taker_legs:
            cross_price = leg.best_ask if leg.best_ask > 0 else leg.best_bid
            leg.target_price = round(min(0.99, max(cross_price, leg.best_bid)), 2)

        # ── Sizing: ComboSizer — share-count pegging ───────────────────
        # CRITICAL: All legs get the SAME share count S.
        # Total collateral = S * Σ(target_price_i).
        target_prices = [lg.target_price for lg in ranked_legs]
        shares = self._sizer.compute_shares(
            target_prices=target_prices,
            wallet_balance=wallet_balance,
            max_per_combo_usd=strat.si9_max_per_combo_usd,
            max_wallet_risk_pct=strat.max_wallet_risk_pct,
            min_leg_depth_shares=min_bid_depth_shares,
        )
        if shares < 1.0:
            return None  # infeasible size

        total_collateral = shares * sum(target_prices)

        for leg in ranked_legs:
            leg.target_shares = shares

        signal = ComboArbSignal(
            market_id=cluster.event_id,  # BaseSignal.market_id → event_id
            cluster_event_id=cluster.event_id,
            maker_leg=maker_leg,
            taker_legs=taker_legs,
            sum_best_bids=round(sum_bids, 4),
            edge_cents=round(edge_cents, 2),
            margin_used=round(required_edge_dollars * 100.0, 2),
            latency_option_premium_dollars=round(latency_option_premium, 6),
            required_edge_dollars=round(required_edge_dollars, 6),
            target_shares=shares,
            total_collateral=round(total_collateral, 2),
        )

        log.info(
            "combo_arb_signal",
            event_id=cluster.event_id,
            maker_leg=maker_leg.yes_token_id,
            routing_reason=maker_leg.routing_reason,
            taker_legs=[leg.yes_token_id for leg in taker_legs],
            n_legs=len(ranked_legs),
            sum_bids=round(sum_bids, 4),
            edge_cents=round(edge_cents, 2),
            latency_option_premium_dollars=round(latency_option_premium, 6),
            shares=shares,
            total_collateral=round(total_collateral, 2),
        )
        return signal

    def _maker_leg_has_toxic_ofi(self, event_id: str, maker_leg: ComboLeg) -> bool:
        book = self._books.get(maker_leg.yes_token_id)
        if book is None:
            return False

        detector = self._ofi_detectors.get(maker_leg.yes_token_id)
        if detector is None:
            strat = settings.strategy
            detector = OFIMomentumDetector(
                market_id=maker_leg.market_id,
                window_ms=strat.si9_ofi_window_ms,
                threshold=strat.si9_toxic_ofi_threshold,
                tvi_kappa=strat.ofi_tvi_kappa,
            )
            self._ofi_detectors[maker_leg.yes_token_id] = detector

        signal = detector.generate_signal(
            book=book,
            trade_aggregator=self._aggregators.get(maker_leg.yes_token_id),
        )
        is_toxic = signal is not None and signal.direction == "SELL"
        was_paused = self._maker_leg_ofi_paused.get(maker_leg.yes_token_id, False)

        if is_toxic:
            previous = self._active_deferrals.get(event_id)
            self._active_deferrals[event_id] = ComboArbDeferral(
                event_id=event_id,
                reason="toxic_ofi",
                maker_leg=maker_leg.yes_token_id,
                market_id=maker_leg.market_id,
                question=maker_leg.question,
                rolling_vi=signal.rolling_vi,
                current_vi=signal.current_vi,
                threshold=signal.threshold,
                deferred_at=time.time(),
                defer_count=(previous.defer_count + 1) if previous else 1,
            )
            if not was_paused:
                log.info(
                    "combo_arb_ofi_paused",
                    event_id=event_id,
                    maker_leg=maker_leg.yes_token_id,
                    market_id=maker_leg.market_id,
                    direction=signal.direction,
                    rolling_vi=signal.rolling_vi,
                    current_vi=signal.current_vi,
                    threshold=signal.threshold,
                )
            self._maker_leg_ofi_paused[maker_leg.yes_token_id] = True
            return True

        previous = self._active_deferrals.pop(event_id, None)
        if was_paused:
            log.info(
                "combo_arb_ofi_resumed",
                event_id=event_id,
                maker_leg=maker_leg.yes_token_id,
                market_id=maker_leg.market_id,
                rolling_vi=round(detector.rolling_vi, 6),
                current_vi=round(detector.current_vi, 6),
                threshold=detector.threshold,
            )
            if previous is not None:
                self._recent_resumes[event_id] = previous
        self._maker_leg_ofi_paused[maker_leg.yes_token_id] = False
        return False

    def _latency_option_premium(self, taker_legs: list[ComboLeg]) -> float:
        horizon_ms = max(0, int(settings.strategy.si9_latency_option_window_ms))
        if horizon_ms <= 0 or not taker_legs:
            return 0.0

        horizon_minutes = horizon_ms / (1000.0 * 60.0)
        if horizon_minutes <= 0:
            return 0.0

        sqrt_horizon = horizon_minutes ** 0.5
        premium = 0.0
        for leg in taker_legs:
            agg = self._aggregators.get(leg.yes_token_id)
            if agg is None:
                continue

            sigma = float(getattr(agg, "rolling_volatility_ewma", 0.0) or 0.0)
            if sigma <= 0:
                continue

            reference_price = float(getattr(agg, "current_price", 0.0) or 0.0)
            if reference_price <= 0:
                reference_price = leg.best_ask if leg.best_ask > 0 else leg.best_bid
            if reference_price <= 0:
                continue

            premium += reference_price * sigma * sqrt_horizon

        return premium

    def get_active_deferral(self, event_id: str) -> ComboArbDeferral | None:
        return self._active_deferrals.get(event_id)

    def active_deferrals(self) -> list[ComboArbDeferral]:
        return list(self._active_deferrals.values())

    def pop_recent_resume(self, event_id: str) -> ComboArbDeferral | None:
        return self._recent_resumes.pop(event_id, None)

    def _should_log_guardrail_rejection(self, event_id: str, sum_bids: float) -> bool:
        now = time.time()
        last_state = self._guardrail_rejection_log_state.get(event_id)
        if last_state is None:
            self._guardrail_rejection_log_state[event_id] = (now, sum_bids)
            return True

        last_logged_at, last_sum_bids = last_state
        if now - last_logged_at >= SI9_GUARDRAIL_LOG_INTERVAL_SECONDS:
            self._guardrail_rejection_log_state[event_id] = (now, sum_bids)
            return True

        if abs(sum_bids - last_sum_bids) > SI9_GUARDRAIL_LOG_SUM_BIDS_DELTA:
            self._guardrail_rejection_log_state[event_id] = (now, sum_bids)
            return True

        return False
