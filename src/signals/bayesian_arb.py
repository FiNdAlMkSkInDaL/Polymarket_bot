"""
SI-10 Bayesian joint-probability arbitrage.

This module scans configured base/base/joint triplets and looks for
Fr\u00e9chet-bound violations across Polymarket books:

  - Upper bound:  P(A \u2229 B) > P(A)  or  P(A \u2229 B) > P(B)
  - Lower bound:  P(A \u2229 B) < P(A) + P(B) - 1

Each violation is converted into a Dutch-book portfolio with a fixed,
state-independent minimum payout of $1.00 per unit:

  - Upper-A: YES(A) + NO(A\u2229B)
  - Upper-B: YES(B) + NO(A\u2229B)
  - Lower:   YES(A\u2229B) + NO(A) + NO(B)

The hedge ratio is exactly 1:1 across the selected legs, so the existing
maker-first combo execution path can reuse the same uniform-share sizing
invariant as SI-9.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings
from src.core.logger import get_logger
from src.data.market_discovery import MarketInfo
from src.data.orderbook import OrderbookTracker
from src.signals.combinatorial_arb import ComboArbDeferral, ComboSizer
from src.signals.ofi_momentum import OFIMomentumDetector
from src.signals.signal_framework import BaseSignal, SignalGenerator
from src.trading.fees import get_fee_rate

log = get_logger(__name__)


@dataclass(frozen=True)
class BayesianRelationship:
    """Configured base/base/joint relationship for SI-10."""

    relationship_id: str
    base_a_condition_id: str
    base_b_condition_id: str
    joint_condition_id: str
    label: str = ""


@dataclass
class BayesianArbCluster:
    """One configured SI-10 triplet."""

    relationship_id: str
    base_a: MarketInfo
    base_b: MarketInfo
    joint: MarketInfo
    label: str = ""
    last_scan_ts: float = 0.0
    active: bool = True

    @property
    def event_id(self) -> str:
        return self.relationship_id

    @property
    def legs(self) -> list[MarketInfo]:
        return [self.base_a, self.base_b, self.joint]


class BayesianArbRelationshipManager:
    """Maintains configured SI-10 triplets over the live market set."""

    def __init__(self, relationships: list[BayesianRelationship] | None = None) -> None:
        self._relationships = relationships or self._load_relationships_from_settings()
        self._clusters: dict[str, BayesianArbCluster] = {}

    def _load_relationships_from_settings(self) -> list[BayesianRelationship]:
        raw = settings.strategy.si10_relationships_json.strip()
        if not raw:
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning("bayesian_arb_relationships_invalid_json", error=str(exc))
            return []

        if not isinstance(payload, list):
            log.warning("bayesian_arb_relationships_invalid_payload")
            return []

        relationships: list[BayesianRelationship] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue

            base_a = str(item.get("base_a_condition_id") or item.get("base_a") or "")
            base_b = str(item.get("base_b_condition_id") or item.get("base_b") or "")
            joint = str(item.get("joint_condition_id") or item.get("joint") or "")
            if not base_a or not base_b or not joint:
                log.warning(
                    "bayesian_arb_relationship_skipped",
                    index=index,
                    reason="missing_condition_id",
                )
                continue

            relationship_id = str(
                item.get("relationship_id")
                or item.get("id")
                or f"SI10-{joint[:12]}-{index}"
            )
            label = str(item.get("label") or item.get("name") or relationship_id)
            relationships.append(
                BayesianRelationship(
                    relationship_id=relationship_id,
                    base_a_condition_id=base_a,
                    base_b_condition_id=base_b,
                    joint_condition_id=joint,
                    label=label,
                )
            )

        return relationships

    def scan_clusters(self, markets: list[MarketInfo]) -> list[BayesianArbCluster]:
        by_condition = {market.condition_id: market for market in markets}
        now = time.monotonic()
        new_clusters: list[BayesianArbCluster] = []
        seen_relationships: set[str] = set()

        for rel in self._relationships:
            seen_relationships.add(rel.relationship_id)
            base_a = by_condition.get(rel.base_a_condition_id)
            base_b = by_condition.get(rel.base_b_condition_id)
            joint = by_condition.get(rel.joint_condition_id)

            if base_a is None or base_b is None or joint is None:
                self._clusters.pop(rel.relationship_id, None)
                continue

            cluster = self._clusters.get(rel.relationship_id)
            if cluster is None:
                cluster = BayesianArbCluster(
                    relationship_id=rel.relationship_id,
                    base_a=base_a,
                    base_b=base_b,
                    joint=joint,
                    label=rel.label,
                    last_scan_ts=now,
                    active=True,
                )
                self._clusters[rel.relationship_id] = cluster
                new_clusters.append(cluster)
                log.info(
                    "bayesian_arb_cluster_discovered",
                    relationship_id=rel.relationship_id,
                    label=rel.label,
                    base_a=base_a.condition_id,
                    base_b=base_b.condition_id,
                    joint=joint.condition_id,
                )
            else:
                cluster.base_a = base_a
                cluster.base_b = base_b
                cluster.joint = joint
                cluster.label = rel.label
                cluster.last_scan_ts = now
                cluster.active = True

        for relationship_id in list(self._clusters):
            if relationship_id not in seen_relationships:
                self._clusters[relationship_id].active = False

        return new_clusters

    @property
    def active_clusters(self) -> list[BayesianArbCluster]:
        return [cluster for cluster in self._clusters.values() if cluster.active]

    def all_cluster_asset_ids(self) -> set[str]:
        asset_ids: set[str] = set()
        for cluster in self.active_clusters:
            for market in cluster.legs:
                asset_ids.add(market.yes_token_id)
                asset_ids.add(market.no_token_id)
        return asset_ids


@dataclass
class BayesianLeg:
    """One SI-10 leg, which may trade either the YES or NO token."""

    market_id: str
    asset_id: str
    trade_side: str
    yes_token_id: str
    no_token_id: str
    best_bid: float
    best_ask: float
    target_price: float
    target_shares: float
    question: str = ""
    role: str = ""
    hedge_weight: float = 1.0
    spread_width: float = 0.0
    bid_depth_usd: float = 0.0
    bid_depth_shares: float = 0.0
    market_liquidity_usd: float = 0.0
    daily_volume_usd: float = 0.0
    routing_reason: str = ""
    fee_enabled: bool = False


@dataclass
class BayesianArbSignal(BaseSignal):
    """SI-10 payload for the existing maker-first combo execution path."""

    cluster_event_id: str = ""
    maker_leg: BayesianLeg | None = None
    taker_legs: list[BayesianLeg] = field(default_factory=list)
    edge_cents: float = 0.0
    margin_used: float = 0.0
    target_shares: float = 0.0
    total_collateral: float = 0.0
    guaranteed_payout: float = 1.0
    relationship_label: str = ""
    violation_type: str = ""
    raw_gap_cents: float = 0.0
    base_a_yes_bid: float = 0.0
    base_b_yes_bid: float = 0.0
    joint_yes_bid: float = 0.0
    hedge_weights: dict[str, float] = field(default_factory=dict)
    bound_title: str = ""
    bound_expression: str = ""
    bound_lhs: float = 0.0
    bound_rhs: float = 0.0
    observed_yes_prices: dict[str, float] = field(default_factory=dict)
    traded_leg_prices: dict[str, dict[str, float | str]] = field(default_factory=dict)
    gross_edge_cents: float = 0.0
    spread_cost_cents: float = 0.0
    taker_fee_cents: float = 0.0
    net_edge_cents: float = 0.0
    net_ev_usd: float = 0.0
    annualized_yield: float = 0.0
    days_to_resolution: float = 0.0

    @property
    def legs(self) -> list[BayesianLeg]:
        if self.maker_leg is None:
            return list(self.taker_legs)
        return [self.maker_leg, *self.taker_legs]


@dataclass(frozen=True)
class _CandidateSpec:
    violation_type: str
    leg_specs: tuple[tuple[str, str, MarketInfo], ...]
    raw_gap_cents: float


class BayesianArbDetector(SignalGenerator):
    """Evaluates one configured base/base/joint triplet in O(1) time."""

    def __init__(
        self,
        book_trackers: dict[str, OrderbookTracker],
        *,
        fee_enabled_resolver: Callable[[MarketInfo], bool] | None = None,
        fee_rate_bps_lookup: Callable[[str], int] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._books = book_trackers
        self._sizer = ComboSizer()
        self._ofi_detectors: dict[str, OFIMomentumDetector] = {}
        self._maker_leg_ofi_paused: dict[str, bool] = {}
        self._active_deferrals: dict[str, ComboArbDeferral] = {}
        self._recent_resumes: dict[str, ComboArbDeferral] = {}
        self._fee_enabled_resolver = fee_enabled_resolver or self._default_fee_enabled
        self._fee_rate_bps_lookup = fee_rate_bps_lookup
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    @property
    def name(self) -> str:
        return "bayesian_arb"

    def evaluate(self, **kwargs: Any) -> BayesianArbSignal | None:
        cluster = kwargs.get("cluster")
        if not isinstance(cluster, BayesianArbCluster):
            return None
        wallet_balance = float(kwargs.get("wallet_balance", 0.0))
        return self.evaluate_cluster(cluster, wallet_balance=wallet_balance)

    def evaluate_cluster(
        self,
        cluster: BayesianArbCluster,
        wallet_balance: float = 0.0,
    ) -> BayesianArbSignal | None:
        strat = settings.strategy
        min_margin = strat.si10_min_margin_cents / 100.0
        buffer = strat.si10_margin_buffer_cents / 100.0
        min_depth_usd = strat.si10_min_leg_depth_usd

        base_a_bid = self._yes_best_bid(cluster.base_a)
        base_b_bid = self._yes_best_bid(cluster.base_b)
        joint_bid = self._yes_best_bid(cluster.joint)
        if min(base_a_bid, base_b_bid, joint_bid) <= 0:
            return None

        candidates = (
            _CandidateSpec(
                violation_type="upper_a",
                leg_specs=(("base_a", "YES", cluster.base_a), ("joint", "NO", cluster.joint)),
                raw_gap_cents=(joint_bid - base_a_bid) * 100.0,
            ),
            _CandidateSpec(
                violation_type="upper_b",
                leg_specs=(("base_b", "YES", cluster.base_b), ("joint", "NO", cluster.joint)),
                raw_gap_cents=(joint_bid - base_b_bid) * 100.0,
            ),
            _CandidateSpec(
                violation_type="lower",
                leg_specs=(
                    ("joint", "YES", cluster.joint),
                    ("base_a", "NO", cluster.base_a),
                    ("base_b", "NO", cluster.base_b),
                ),
                raw_gap_cents=((base_a_bid + base_b_bid - 1.0) - joint_bid) * 100.0,
            ),
        )

        best_signal: BayesianArbSignal | None = None
        for candidate in candidates:
            signal = self._evaluate_candidate(
                cluster=cluster,
                candidate=candidate,
                wallet_balance=wallet_balance,
                min_margin=min_margin,
                buffer=buffer,
                min_depth_usd=min_depth_usd,
                base_a_bid=base_a_bid,
                base_b_bid=base_b_bid,
                joint_bid=joint_bid,
            )
            if signal is None:
                continue
            if best_signal is None or signal.edge_cents > best_signal.edge_cents:
                best_signal = signal

        return best_signal

    def _evaluate_candidate(
        self,
        *,
        cluster: BayesianArbCluster,
        candidate: _CandidateSpec,
        wallet_balance: float,
        min_margin: float,
        buffer: float,
        min_depth_usd: float,
        base_a_bid: float,
        base_b_bid: float,
        joint_bid: float,
    ) -> BayesianArbSignal | None:
        if candidate.raw_gap_cents <= 0:
            return None

        bound_title, bound_expression, bound_lhs, bound_rhs = self._bound_details(
            candidate.violation_type,
            base_a_bid=base_a_bid,
            base_b_bid=base_b_bid,
            joint_bid=joint_bid,
        )

        legs: list[BayesianLeg] = []
        min_bid_depth_shares = float("inf")
        quote_sum = 0.0

        for role, trade_side, market in candidate.leg_specs:
            leg = self._build_leg(
                market=market,
                role=role,
                trade_side=trade_side,
                min_depth_usd=min_depth_usd,
            )
            if leg is None:
                return None

            legs.append(leg)
            quote_sum += leg.best_bid
            min_bid_depth_shares = min(min_bid_depth_shares, leg.bid_depth_shares)

        if quote_sum >= 1.0 - min_margin:
            return None

        ranked_legs = self._rank_legs_for_routing(legs)
        maker_leg = ranked_legs[0]
        taker_legs = ranked_legs[1:]
        maker_leg.routing_reason = self._routing_reason_for_maker(maker_leg, ranked_legs)

        if self._maker_leg_has_toxic_ofi(cluster.relationship_id, maker_leg):
            return None

        if (1.0 - quote_sum) > min_margin + buffer:
            maker_leg.target_price = round(min(0.99, maker_leg.best_bid + 0.01), 2)
        else:
            maker_leg.target_price = round(maker_leg.best_bid, 2)

        for leg in taker_legs:
            cross_price = leg.best_ask if leg.best_ask > 0 else leg.best_bid
            leg.target_price = round(min(0.99, max(cross_price, leg.best_bid)), 2)

        target_prices = [leg.target_price for leg in ranked_legs]
        total_cost = sum(target_prices)
        gross_edge = max(0.0, 1.0 - quote_sum)
        executable_edge = 1.0 - total_cost
        if executable_edge < min_margin:
            return None

        taker_fee_cost = 0.0
        spread_cost = 0.0
        for leg in taker_legs:
            spread_cost += max(0.0, leg.target_price - leg.best_bid)
            fee_rate = self._taker_fee_rate_for_leg(leg)
            taker_fee_cost += leg.target_price * fee_rate

        net_edge_per_share = 1.0 - total_cost - taker_fee_cost
        if net_edge_per_share <= 0.0:
            return None

        shares = self._sizer.compute_shares(
            target_prices=target_prices,
            wallet_balance=wallet_balance,
            max_per_combo_usd=settings.strategy.si9_max_per_combo_usd,
            max_wallet_risk_pct=settings.strategy.max_wallet_risk_pct,
            min_leg_depth_shares=min_bid_depth_shares,
        )
        if shares < 1.0:
            return None

        net_ev_usd = net_edge_per_share * shares
        if net_ev_usd < settings.strategy.si10_min_net_edge_usd:
            return None

        days_to_resolution = self._days_to_latest_resolution(cluster)
        annualized_yield = self._annualized_yield(
            net_edge_per_share=net_edge_per_share,
            total_cost=total_cost,
            days_to_resolution=days_to_resolution,
        )
        if annualized_yield < settings.strategy.si10_min_annualized_yield:
            return None

        for leg in ranked_legs:
            leg.target_shares = shares

        signal = BayesianArbSignal(
            market_id=cluster.relationship_id,
            cluster_event_id=cluster.relationship_id,
            maker_leg=maker_leg,
            taker_legs=taker_legs,
            edge_cents=round(net_edge_per_share * 100.0, 2),
            margin_used=settings.strategy.si10_min_margin_cents,
            target_shares=shares,
            total_collateral=round(total_cost * shares, 2),
            guaranteed_payout=1.0,
            relationship_label=cluster.label,
            violation_type=candidate.violation_type,
            raw_gap_cents=round(candidate.raw_gap_cents, 2),
            base_a_yes_bid=round(base_a_bid, 4),
            base_b_yes_bid=round(base_b_bid, 4),
            joint_yes_bid=round(joint_bid, 4),
            hedge_weights={leg.asset_id: leg.hedge_weight for leg in ranked_legs},
            bound_title=bound_title,
            bound_expression=bound_expression,
            bound_lhs=round(bound_lhs, 4),
            bound_rhs=round(bound_rhs, 4),
            observed_yes_prices={
                "base_a_yes": round(base_a_bid, 4),
                "base_b_yes": round(base_b_bid, 4),
                "joint_yes": round(joint_bid, 4),
            },
            traded_leg_prices={
                leg.asset_id: {
                    "market_id": leg.market_id,
                    "role": leg.role,
                    "trade_side": leg.trade_side,
                    "best_bid": round(leg.best_bid, 4),
                    "best_ask": round(leg.best_ask, 4),
                    "target_price": round(leg.target_price, 4),
                    "fee_bps": self._leg_fee_bps(leg) if leg in taker_legs else 0,
                }
                for leg in ranked_legs
            },
            gross_edge_cents=round(gross_edge * 100.0, 2),
            spread_cost_cents=round(spread_cost * 100.0, 2),
            taker_fee_cents=round(taker_fee_cost * 100.0, 2),
            net_edge_cents=round(net_edge_per_share * 100.0, 2),
            net_ev_usd=round(net_ev_usd, 4),
            annualized_yield=round(annualized_yield, 6),
            days_to_resolution=round(days_to_resolution, 4),
        )

        log.info(
            "bayesian_arb_signal",
            relationship_id=cluster.relationship_id,
            label=cluster.label,
            violation_type=candidate.violation_type,
            bound_title=bound_title,
            bound_expression=bound_expression,
            bound_lhs=round(bound_lhs, 4),
            bound_rhs=round(bound_rhs, 4),
            maker_leg=maker_leg.asset_id,
            taker_legs=[leg.asset_id for leg in taker_legs],
            edge_cents=signal.edge_cents,
            gross_edge_cents=signal.gross_edge_cents,
            spread_cost_cents=signal.spread_cost_cents,
            taker_fee_cents=signal.taker_fee_cents,
            net_ev_usd=signal.net_ev_usd,
            annualized_yield=signal.annualized_yield,
            days_to_resolution=signal.days_to_resolution,
            raw_gap_cents=signal.raw_gap_cents,
            shares=shares,
            total_collateral=signal.total_collateral,
            observed_yes_prices=signal.observed_yes_prices,
            traded_leg_prices=signal.traded_leg_prices,
        )
        return signal

    def _bound_details(
        self,
        violation_type: str,
        *,
        base_a_bid: float,
        base_b_bid: float,
        joint_bid: float,
    ) -> tuple[str, str, float, float]:
        if violation_type in {"upper_a", "upper_b"}:
            rhs = min(base_a_bid, base_b_bid)
            return (
                "Upper Bound Violation",
                f"P(A ∩ B) = {joint_bid:.4f} > min(P(A), P(B)) = {rhs:.4f}",
                joint_bid,
                rhs,
            )

        rhs = base_a_bid + base_b_bid - 1.0
        return (
            "Lower Bound Violation",
            f"P(A ∩ B) = {joint_bid:.4f} < P(A) + P(B) - 1 = {rhs:.4f}",
            joint_bid,
            rhs,
        )

    def _build_leg(
        self,
        *,
        market: MarketInfo,
        role: str,
        trade_side: str,
        min_depth_usd: float,
    ) -> BayesianLeg | None:
        asset_id = market.yes_token_id if trade_side == "YES" else market.no_token_id
        book = self._books.get(asset_id)
        if book is None:
            return None

        snap = book.snapshot()
        if not self._is_snapshot_fresh(snap):
            return None

        bid = float(getattr(snap, "best_bid", 0.0) or 0.0)
        ask = float(getattr(snap, "best_ask", 0.0) or 0.0)
        if bid <= 0:
            return None

        bid_depth_usd = self._bid_depth_usd(book, snap)
        if bid_depth_usd < min_depth_usd:
            return None

        depth_shares = bid_depth_usd / bid if bid > 0 else 0.0
        return BayesianLeg(
            market_id=market.condition_id,
            asset_id=asset_id,
            trade_side=trade_side,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            best_bid=bid,
            best_ask=ask,
            target_price=0.0,
            target_shares=0.0,
            question=market.question[:60],
            role=role,
            hedge_weight=1.0,
            spread_width=round(max(0.0, ask - bid), 4),
            bid_depth_usd=round(bid_depth_usd, 2),
            bid_depth_shares=round(depth_shares, 4),
            market_liquidity_usd=round(market.liquidity_usd, 2),
            daily_volume_usd=round(market.daily_volume_usd, 2),
            fee_enabled=self._fee_enabled_resolver(market),
        )

    def _yes_best_bid(self, market: MarketInfo) -> float:
        book = self._books.get(market.yes_token_id)
        if book is None:
            return 0.0
        snap = book.snapshot()
        if not self._is_snapshot_fresh(snap):
            return 0.0
        return float(getattr(snap, "best_bid", 0.0) or 0.0)

    def _rank_legs_for_routing(self, legs: list[BayesianLeg]) -> list[BayesianLeg]:
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
        maker_leg: BayesianLeg,
        ranked_legs: list[BayesianLeg],
    ) -> str:
        if len(ranked_legs) == 1:
            return f"widest_spread spread_width={maker_leg.spread_width:.4f}"

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

    def _maker_leg_has_toxic_ofi(self, cluster_id: str, maker_leg: BayesianLeg) -> bool:
        book = self._books.get(maker_leg.asset_id)
        if book is None:
            return False

        detector = self._ofi_detectors.get(maker_leg.asset_id)
        if detector is None:
            detector = OFIMomentumDetector(
                market_id=maker_leg.market_id,
                window_ms=settings.strategy.si9_ofi_window_ms,
                threshold=settings.strategy.si10_maker_ofi_tolerance,
                tvi_kappa=settings.strategy.ofi_tvi_kappa,
            )
            self._ofi_detectors[maker_leg.asset_id] = detector

        signal = detector.generate_signal(book=book)
        is_toxic = signal is not None and signal.direction == "SELL"
        was_paused = self._maker_leg_ofi_paused.get(maker_leg.asset_id, False)

        if is_toxic:
            previous = self._active_deferrals.get(cluster_id)
            self._active_deferrals[cluster_id] = ComboArbDeferral(
                event_id=cluster_id,
                reason="toxic_ofi",
                maker_leg=maker_leg.asset_id,
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
                    "bayesian_arb_ofi_paused",
                    relationship_id=cluster_id,
                    maker_leg=maker_leg.asset_id,
                    market_id=maker_leg.market_id,
                    direction=signal.direction,
                    rolling_vi=signal.rolling_vi,
                    current_vi=signal.current_vi,
                    threshold=signal.threshold,
                )
            self._maker_leg_ofi_paused[maker_leg.asset_id] = True
            return True

        previous = self._active_deferrals.pop(cluster_id, None)
        if was_paused:
            log.info(
                "bayesian_arb_ofi_resumed",
                relationship_id=cluster_id,
                maker_leg=maker_leg.asset_id,
                market_id=maker_leg.market_id,
                rolling_vi=round(detector.rolling_vi, 6),
                current_vi=round(detector.current_vi, 6),
                threshold=detector.threshold,
            )
            if previous is not None:
                self._recent_resumes[cluster_id] = previous
        self._maker_leg_ofi_paused[maker_leg.asset_id] = False
        return False

    def get_active_deferral(self, cluster_id: str) -> ComboArbDeferral | None:
        return self._active_deferrals.get(cluster_id)

    def active_deferrals(self) -> list[ComboArbDeferral]:
        return list(self._active_deferrals.values())

    def pop_recent_resume(self, cluster_id: str) -> ComboArbDeferral | None:
        return self._recent_resumes.pop(cluster_id, None)

    def _default_fee_enabled(self, market: MarketInfo) -> bool:
        market_tags = (getattr(market, "tags", "") or "").lower()
        if not market_tags:
            return False
        fee_categories = {
            part.strip().lower()
            for part in (settings.strategy.fee_enabled_categories or "").split(",")
            if part.strip()
        }
        return any(category in market_tags for category in fee_categories)

    def _leg_fee_bps(self, leg: BayesianLeg) -> int:
        if self._fee_rate_bps_lookup is not None:
            return max(0, int(self._fee_rate_bps_lookup(leg.asset_id)))
        if leg.fee_enabled:
            return int(round(get_fee_rate(leg.target_price, fee_enabled=True) * 10_000.0))
        return 0

    def _taker_fee_rate_for_leg(self, leg: BayesianLeg) -> float:
        if not leg.fee_enabled:
            return 0.0
        fee_bps = self._leg_fee_bps(leg)
        if fee_bps > 0:
            return fee_bps / 10_000.0
        return get_fee_rate(leg.target_price, fee_enabled=True)

    def _days_to_latest_resolution(self, cluster: BayesianArbCluster) -> float:
        latest_end = max(
            (market.end_date for market in cluster.legs if market.end_date is not None),
            default=None,
        )
        if latest_end is None:
            return max(1.0, float(settings.strategy.si10_default_days_to_resolution))

        now = self._now_provider()
        if latest_end.tzinfo is None:
            latest_end = latest_end.replace(tzinfo=timezone.utc)
        delta_days = (latest_end - now).total_seconds() / 86_400.0
        return max(1.0, delta_days)

    @staticmethod
    def _annualized_yield(
        *,
        net_edge_per_share: float,
        total_cost: float,
        days_to_resolution: float,
    ) -> float:
        if total_cost <= 0 or days_to_resolution <= 0:
            return 0.0
        simple_return = net_edge_per_share / total_cost
        return simple_return * (365.0 / days_to_resolution)

    @staticmethod
    def _is_snapshot_fresh(snapshot: Any) -> bool:
        if hasattr(snapshot, "fresh"):
            return bool(getattr(snapshot, "fresh"))
        return float(getattr(snapshot, "timestamp", 0.0) or 0.0) > 0.0

    @staticmethod
    def _bid_depth_usd(book: Any, snapshot: Any) -> float:
        if hasattr(snapshot, "bid_depth_usd"):
            return float(getattr(snapshot, "bid_depth_usd", 0.0) or 0.0)
        if hasattr(book, "levels"):
            levels = book.levels("bid", n=1)
            if levels:
                level = levels[0]
                return float(getattr(level, "price", 0.0) or 0.0) * float(
                    getattr(level, "size", 0.0) or 0.0
                )
        best_bid = float(getattr(snapshot, "best_bid", 0.0) or 0.0)
        return best_bid * 100.0 if best_bid > 0 else 0.0