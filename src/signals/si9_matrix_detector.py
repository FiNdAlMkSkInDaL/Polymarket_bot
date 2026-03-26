from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.detectors.si9_cluster_config import Si9ClusterConfig


SuppressionReason = Literal[
    "STALE_LEG",
    "GHOST_TOWN",
    "IMPLAUSIBLE_EDGE",
    "INSUFFICIENT_YIELD",
    "INCOMPLETE_CLUSTER",
]


@dataclass(frozen=True, slots=True)
class Si9TopOfBookState:
    market_id: str
    ask_price: Decimal
    ask_size: Decimal
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class Si9RawLegSnapshot:
    market_id: str
    ask_price: Decimal | None
    ask_size: Decimal | None
    timestamp_ms: int | None


@dataclass(frozen=True, slots=True)
class Si9TradeableLegSnapshot:
    market_id: str
    ask_price: Decimal
    ask_size: Decimal
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class Si9RawClusterSnapshot:
    cluster_id: str
    legs: tuple[Si9RawLegSnapshot, ...]
    cluster_ask_sum: Decimal | None
    net_edge: Decimal | None
    bottleneck_market_id: str | None
    eval_timestamp_ms: int | None
    would_be_tradeable: bool
    suppression_reason: SuppressionReason | None


@dataclass(frozen=True, slots=True)
class Si9TradeableSnapshot:
    cluster_id: str
    legs: tuple[Si9TradeableLegSnapshot, ...]
    cluster_ask_sum: Decimal
    net_edge: Decimal
    bottleneck_market_id: str
    required_share_counts: Decimal
    tradeable_at_ms: int


@dataclass(frozen=True, slots=True)
class Si9MatrixSignal:
    cluster_id: str
    market_ids: tuple[str, ...]
    best_yes_asks: dict[str, Decimal]
    ask_sizes: dict[str, Decimal]
    total_yes_ask: Decimal
    gross_edge: Decimal
    net_edge: Decimal
    target_yield: Decimal
    bottleneck_market_id: str
    required_share_counts: Decimal
    signal_source: str = "SI9-Matrix"


@dataclass(frozen=True, slots=True)
class _ClusterAssessment:
    cluster_id: str
    legs: tuple[Si9TopOfBookState, ...]
    cluster_ask_sum: Decimal | None
    gross_edge: Decimal | None
    net_edge: Decimal | None
    bottleneck_market_id: str | None
    required_share_counts: Decimal | None
    suppression_reason: SuppressionReason | None
    eval_timestamp_ms: int

    @property
    def is_tradeable(self) -> bool:
        return self.suppression_reason is None


class Si9MatrixDetector:
    """Tracks mutually exclusive clusters and emits isolated Dutch-book signals."""

    __slots__ = (
        "config",
        "_cluster_members",
        "_top_of_book_by_market",
    )

    def __init__(self, config: Si9ClusterConfig) -> None:
        self.config = config
        self._cluster_members: dict[str, tuple[str, ...]] = {}
        self._top_of_book_by_market: dict[str, Si9TopOfBookState] = {}

    @property
    def cluster_members(self) -> dict[str, tuple[str, ...]]:
        return self._cluster_members

    @property
    def top_of_book_by_market(self) -> dict[str, Si9TopOfBookState]:
        return self._top_of_book_by_market

    def register_cluster(self, cluster_id: str, market_ids: list[str] | tuple[str, ...]) -> None:
        cluster_key = str(cluster_id).strip()
        members = tuple(str(market_id).strip() for market_id in market_ids)
        if len(members) < self.config.min_cluster_size:
            raise ValueError("market_ids must satisfy config.min_cluster_size")
        self._cluster_members[cluster_key] = members

    def update_best_yes_ask(
        self,
        market_id: str,
        ask_price: Decimal,
        ask_size: Decimal,
        timestamp_ms: int,
    ) -> None:
        market_key = str(market_id).strip()
        price = Decimal(ask_price)
        size = Decimal(ask_size)
        if price <= Decimal("0") or price >= Decimal("1"):
            return
        if size <= Decimal("0"):
            return
        self._top_of_book_by_market[market_key] = Si9TopOfBookState(
            market_id=market_key,
            ask_price=price,
            ask_size=size,
            timestamp_ms=int(timestamp_ms),
        )

    # FLAG-4-RESOLVED: raw replay state and live-tradeable state are split into
    # separate contracts so monitoring never consumes executable state by mistake.
    def cluster_snapshot(
        self,
        cluster_id: str,
        eval_timestamp_ms: int | None = None,
    ) -> Si9RawClusterSnapshot:
        cluster_key = str(cluster_id).strip()
        market_ids = self._cluster_members.get(cluster_key)
        if market_ids is None:
            raise KeyError(f"Unknown cluster_id: {cluster_id!r}")

        raw_legs = tuple(
            self._raw_leg_snapshot(market_id)
            for market_id in market_ids
        )
        present_legs = [
            self._top_of_book_by_market[market_id]
            for market_id in market_ids
            if market_id in self._top_of_book_by_market
        ]
        if eval_timestamp_ms is None:
            # Deprecated legacy path: raw snapshots without an injected
            # evaluation timestamp infer freshness from the latest visible leg
            # timestamp for replay compatibility only.
            assessment_eval_timestamp_ms = max((leg.timestamp_ms for leg in present_legs), default=0)
            snapshot_eval_timestamp_ms = None
        else:
            assessment_eval_timestamp_ms = int(eval_timestamp_ms)
            snapshot_eval_timestamp_ms = assessment_eval_timestamp_ms
        assessment = self._assess_cluster(cluster_key, assessment_eval_timestamp_ms)
        return Si9RawClusterSnapshot(
            cluster_id=cluster_key,
            legs=raw_legs,
            cluster_ask_sum=assessment.cluster_ask_sum,
            net_edge=assessment.net_edge,
            bottleneck_market_id=assessment.bottleneck_market_id,
            eval_timestamp_ms=snapshot_eval_timestamp_ms,
            would_be_tradeable=assessment.is_tradeable,
            suppression_reason=assessment.suppression_reason,
        )

    def tradeable_snapshot(
        self,
        cluster_id: str,
        eval_timestamp_ms: int,
    ) -> Si9TradeableSnapshot | None:
        cluster_key = str(cluster_id).strip()
        assessment = self._assess_cluster(cluster_key, int(eval_timestamp_ms))
        if not assessment.is_tradeable:
            return None
        assert assessment.cluster_ask_sum is not None
        assert assessment.net_edge is not None
        assert assessment.bottleneck_market_id is not None
        assert assessment.required_share_counts is not None
        return Si9TradeableSnapshot(
            cluster_id=cluster_key,
            legs=tuple(
                Si9TradeableLegSnapshot(
                    market_id=leg.market_id,
                    ask_price=leg.ask_price,
                    ask_size=leg.ask_size,
                    timestamp_ms=leg.timestamp_ms,
                )
                for leg in assessment.legs
            ),
            cluster_ask_sum=assessment.cluster_ask_sum,
            net_edge=assessment.net_edge,
            bottleneck_market_id=assessment.bottleneck_market_id,
            required_share_counts=assessment.required_share_counts,
            tradeable_at_ms=assessment.eval_timestamp_ms,
        )

    def evaluate_cluster(
        self,
        cluster_id: str,
        eval_timestamp_ms: int,
    ) -> Si9MatrixSignal | None:
        cluster_key = str(cluster_id).strip()
        assessment = self._assess_cluster(cluster_key, int(eval_timestamp_ms))
        if not assessment.is_tradeable:
            return None
        assert assessment.cluster_ask_sum is not None
        assert assessment.gross_edge is not None
        assert assessment.net_edge is not None
        assert assessment.bottleneck_market_id is not None
        assert assessment.required_share_counts is not None
        return Si9MatrixSignal(
            cluster_id=cluster_key,
            market_ids=tuple(leg.market_id for leg in assessment.legs),
            best_yes_asks={leg.market_id: leg.ask_price for leg in assessment.legs},
            ask_sizes={leg.market_id: leg.ask_size for leg in assessment.legs},
            total_yes_ask=assessment.cluster_ask_sum,
            gross_edge=assessment.gross_edge,
            net_edge=assessment.net_edge,
            target_yield=self.config.target_yield,
            bottleneck_market_id=assessment.bottleneck_market_id,
            required_share_counts=assessment.required_share_counts,
        )

    def _assess_cluster(self, cluster_id: str, eval_timestamp_ms: int) -> _ClusterAssessment:
        market_ids = self._cluster_members.get(cluster_id)
        if market_ids is None:
            raise KeyError(f"Unknown cluster_id: {cluster_id!r}")

        legs: list[Si9TopOfBookState] = []
        for market_id in market_ids:
            leg = self._top_of_book_by_market.get(market_id)
            if leg is None:
                return _ClusterAssessment(
                    cluster_id=cluster_id,
                    legs=tuple(legs),
                    cluster_ask_sum=None,
                    gross_edge=None,
                    net_edge=None,
                    bottleneck_market_id=None,
                    required_share_counts=None,
                    suppression_reason="INCOMPLETE_CLUSTER",
                    eval_timestamp_ms=eval_timestamp_ms,
                )
            legs.append(leg)

        for leg in legs:
            if eval_timestamp_ms - leg.timestamp_ms > self.config.max_ask_age_ms:
                return _ClusterAssessment(
                    cluster_id=cluster_id,
                    legs=tuple(legs),
                    cluster_ask_sum=None,
                    gross_edge=None,
                    net_edge=None,
                    bottleneck_market_id=None,
                    required_share_counts=None,
                    suppression_reason="STALE_LEG",
                    eval_timestamp_ms=eval_timestamp_ms,
                )

        cluster_ask_sum = sum((leg.ask_price for leg in legs), start=Decimal("0"))
        if cluster_ask_sum < self.config.ghost_town_floor:
            return _ClusterAssessment(
                cluster_id=cluster_id,
                legs=tuple(legs),
                cluster_ask_sum=cluster_ask_sum,
                gross_edge=None,
                net_edge=None,
                bottleneck_market_id=None,
                required_share_counts=None,
                suppression_reason="GHOST_TOWN",
                eval_timestamp_ms=eval_timestamp_ms,
            )

        gross_edge = Decimal("1") - cluster_ask_sum
        if gross_edge > self.config.implausible_edge_ceil:
            return _ClusterAssessment(
                cluster_id=cluster_id,
                legs=tuple(legs),
                cluster_ask_sum=cluster_ask_sum,
                gross_edge=gross_edge,
                net_edge=None,
                bottleneck_market_id=None,
                required_share_counts=None,
                suppression_reason="IMPLAUSIBLE_EDGE",
                eval_timestamp_ms=eval_timestamp_ms,
            )

        net_edge = self._compute_net_edge(cluster_ask_sum, len(legs))
        if net_edge < self.config.target_yield:
            return _ClusterAssessment(
                cluster_id=cluster_id,
                legs=tuple(legs),
                cluster_ask_sum=cluster_ask_sum,
                gross_edge=gross_edge,
                net_edge=net_edge,
                bottleneck_market_id=None,
                required_share_counts=None,
                suppression_reason="INSUFFICIENT_YIELD",
                eval_timestamp_ms=eval_timestamp_ms,
            )

        bottleneck_market_id, required_share_counts = self._select_bottleneck_leg(legs, market_ids)
        return _ClusterAssessment(
            cluster_id=cluster_id,
            legs=tuple(legs),
            cluster_ask_sum=cluster_ask_sum,
            gross_edge=gross_edge,
            net_edge=net_edge,
            bottleneck_market_id=bottleneck_market_id,
            required_share_counts=required_share_counts,
            suppression_reason=None,
            eval_timestamp_ms=eval_timestamp_ms,
        )

    def _compute_net_edge(self, total_yes_ask: Decimal, leg_count: int) -> Decimal:
        gross_edge = Decimal("1") - total_yes_ask
        total_fee_cost = Decimal(leg_count) * self.config.taker_fee_per_leg
        total_slippage_cost = Decimal(leg_count) * self.config.slippage_budget
        return gross_edge - total_fee_cost - total_slippage_cost

    def _raw_leg_snapshot(self, market_id: str) -> Si9RawLegSnapshot:
        leg = self._top_of_book_by_market.get(market_id)
        if leg is None:
            return Si9RawLegSnapshot(
                market_id=market_id,
                ask_price=None,
                ask_size=None,
                timestamp_ms=None,
            )
        return Si9RawLegSnapshot(
            market_id=leg.market_id,
            ask_price=leg.ask_price,
            ask_size=leg.ask_size,
            timestamp_ms=leg.timestamp_ms,
        )

    def _select_bottleneck_leg(
        self,
        legs: list[Si9TopOfBookState],
        market_ids: tuple[str, ...],
    ) -> tuple[str, Decimal]:
        max_shares = min(leg.ask_size for leg in legs)
        depth_candidates = [leg for leg in legs if leg.ask_size == max_shares]
        highest_ask = max(leg.ask_price for leg in depth_candidates)
        highest_ask_candidates = [leg for leg in depth_candidates if leg.ask_price == highest_ask]

        # FLAG-3-RESOLVED: the live default uses lexicographic market_id for
        # equal-depth, equal-price ties because it is deterministic across process
        # restarts and replay runs. stable_index is retained only for backwards
        # compatibility with historical registration-order behavior.
        if self.config.tiebreak_policy == "lowest_market_id":
            bottleneck_leg = min(highest_ask_candidates, key=lambda leg: leg.market_id)
        else:
            registration_index = {market_id: index for index, market_id in enumerate(market_ids)}
            bottleneck_leg = min(
                highest_ask_candidates,
                key=lambda leg: registration_index[leg.market_id],
            )
        return bottleneck_leg.market_id, max_shares
