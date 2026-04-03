from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, Sequence

from src.execution.priority_context import PriorityOrderContext
from src.signals.base_strategy import BaseStrategy
from src.signals.obi_scalper import NormalizedBookLevel, ObiScalper


_ONE = Decimal("1")


class ClockMs(Protocol):
    def __call__(self) -> int:
        ...


@dataclass(frozen=True, slots=True)
class _WallCandidate:
    wall_side: str
    price: Decimal
    size: Decimal
    wall_size_usd: Decimal
    price_level_vs_mid_ticks: Decimal
    opposing_average_size: Decimal
    ratio: Decimal


@dataclass(slots=True)
class _TrackedWall:
    wall_id: str
    wall_side: str
    wall_price: Decimal
    initial_size: Decimal
    quote_side: str
    quote_price: Decimal


@dataclass(slots=True)
class _ObservedWall:
    wall_side: str
    wall_price: Decimal
    first_seen_at_ms: int
    age_qualified: bool = False

    @property
    def wall_id(self) -> str:
        return f"{self.wall_side}:{self.wall_price}:{self.first_seen_at_ms}"


@dataclass(slots=True)
class _MarketState:
    tracked_wall: _TrackedWall | None = None
    observed_wall: _ObservedWall | None = None


@dataclass(slots=True)
class WallJumper(BaseStrategy):
    dispatcher: Any | None = None
    market_catalog: Any | None = None
    depth_levels: int = 5
    min_wall_size: Decimal = Decimal("10000")
    min_distance_from_mid_ticks: Decimal = Decimal("2")
    min_structural_wall_size_usd: Decimal = Decimal("100000")
    wall_to_opposing_ratio: Decimal = Decimal("5")
    collapse_fraction: Decimal = Decimal("0.5")
    wall_age_ms: int = 0
    order_size: Decimal = Decimal("10")
    tick_size: Decimal = Decimal("0.01")
    signal_source: str = "MANUAL"
    clock_ms: ClockMs = field(default=lambda: int(time.time() * 1000))
    _market_state: dict[str, _MarketState] = field(init=False, repr=False)
    _jump_quotes_emitted: int = field(init=False, repr=False)
    _cancel_all_triggered: int = field(init=False, repr=False)
    _walls_identified: int = field(init=False, repr=False)
    _walls_aged_past_threshold: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock_ms,
        )
        if not isinstance(self.depth_levels, int) or self.depth_levels <= 0:
            raise ValueError("depth_levels must be a strictly positive int")
        self.min_wall_size = self._as_decimal(self.min_wall_size, name="min_wall_size")
        self.min_distance_from_mid_ticks = self._as_decimal(
            self.min_distance_from_mid_ticks,
            name="min_distance_from_mid_ticks",
        )
        self.min_structural_wall_size_usd = self._as_decimal(
            self.min_structural_wall_size_usd,
            name="min_structural_wall_size_usd",
        )
        self.wall_to_opposing_ratio = self._as_decimal(
            self.wall_to_opposing_ratio,
            name="wall_to_opposing_ratio",
        )
        self.collapse_fraction = self._as_decimal(self.collapse_fraction, name="collapse_fraction")
        self.order_size = self._as_decimal(self.order_size, name="order_size")
        self.tick_size = self._as_decimal(self.tick_size, name="tick_size")
        if self.min_wall_size <= Decimal("0"):
            raise ValueError("min_wall_size must be strictly positive")
        if self.min_distance_from_mid_ticks <= Decimal("0"):
            raise ValueError("min_distance_from_mid_ticks must be strictly positive")
        if self.min_structural_wall_size_usd <= Decimal("0"):
            raise ValueError("min_structural_wall_size_usd must be strictly positive")
        if self.wall_to_opposing_ratio <= Decimal("1"):
            raise ValueError("wall_to_opposing_ratio must be greater than 1")
        if self.collapse_fraction <= Decimal("0") or self.collapse_fraction >= _ONE:
            raise ValueError("collapse_fraction must be between 0 and 1")
        if not isinstance(self.wall_age_ms, int) or self.wall_age_ms < 0:
            raise ValueError("wall_age_ms must be a non-negative int")
        if self.order_size <= Decimal("0"):
            raise ValueError("order_size must be strictly positive")
        if self.tick_size <= Decimal("0"):
            raise ValueError("tick_size must be strictly positive")
        self.signal_source = str(self.signal_source or "").strip().upper() or "MANUAL"
        self._market_state = {}
        self._jump_quotes_emitted = 0
        self._cancel_all_triggered = 0
        self._walls_identified = 0
        self._walls_aged_past_threshold = 0

    def bind_dispatcher(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher
        super().bind_dispatcher(dispatcher)

    def bind_market_catalog(self, market_catalog: Any) -> None:
        self.market_catalog = market_catalog
        super().bind_market_catalog(market_catalog)

    def bind_clock(self, clock) -> None:
        self.clock_ms = clock
        super().bind_clock(clock)

    def on_bbo_update(
        self,
        market_id: str,
        top_bids: Sequence[dict[str, Any]],
        top_asks: Sequence[dict[str, Any]],
    ) -> None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            return

        bids = ObiScalper._normalize_levels(top_bids)[: self.depth_levels]
        asks = ObiScalper._normalize_levels(top_asks)[: self.depth_levels]
        if not bids or not asks:
            return

        state = self._state_for(normalized_market_id)
        tracked_wall = state.tracked_wall
        if tracked_wall is not None:
            current_size = self._current_wall_size(tracked_wall, bids=bids, asks=asks)
            collapse_threshold = tracked_wall.initial_size * self.collapse_fraction
            if current_size <= collapse_threshold:
                self._cancel_all_quotes(
                    normalized_market_id,
                    tracked_wall=tracked_wall,
                    current_size=current_size,
                )
                state.tracked_wall = None
                tracked_wall = None

        candidate = self._select_wall_candidate(bids=bids, asks=asks)
        if candidate is None:
            state.observed_wall = None
            return

        observed_wall, wall_age_ms = self._record_wall_observation(state, candidate=candidate)
        if wall_age_ms < self.wall_age_ms:
            return

        quote_side, quote_price = self._build_jump_quote(candidate=candidate, bids=bids, asks=asks)
        if quote_price is None:
            return

        if tracked_wall is not None and tracked_wall.quote_side != quote_side:
            self._cancel_all_quotes(
                normalized_market_id,
                tracked_wall=tracked_wall,
                current_size=tracked_wall.initial_size,
            )
            state.tracked_wall = None
            tracked_wall = None

        if (
            tracked_wall is not None
            and tracked_wall.wall_side == candidate.wall_side
            and tracked_wall.wall_price == candidate.price
            and tracked_wall.quote_price == quote_price
        ):
            tracked_wall.initial_size = max(tracked_wall.initial_size, candidate.size)
            return

        self.submit_order(
            PriorityOrderContext(
                market_id=normalized_market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=min(_ONE, candidate.ratio / self.wall_to_opposing_ratio),
                target_price=quote_price,
                anchor_volume=self.order_size,
                max_capital=quote_price * self.order_size,
                signal_metadata={
                    "strategy": "wall_jumper",
                    "post_only": True,
                    "time_in_force": "GTC",
                    "liquidity_intent": "MAKER",
                    "quote_side": quote_side,
                    "quote_id": f"wall_jumper:{normalized_market_id}:{quote_side.lower()}",
                    "wall_id": f"{normalized_market_id}:{observed_wall.wall_id}",
                    "wall_side": candidate.wall_side,
                    "wall_price": str(candidate.price),
                    "wall_size": str(candidate.size),
                    "wall_size_usd": str(candidate.wall_size_usd),
                    "wall_age_ms": wall_age_ms,
                    "price_level_vs_mid_ticks": str(candidate.price_level_vs_mid_ticks),
                    "wall_to_opposing_ratio": str(candidate.ratio),
                    "entry_theory": "penny_jump_wall_support",
                },
            )
        )
        self._jump_quotes_emitted += 1
        state.tracked_wall = _TrackedWall(
            wall_id=f"{normalized_market_id}:{observed_wall.wall_id}",
            wall_side=candidate.wall_side,
            wall_price=candidate.price,
            initial_size=candidate.size,
            quote_side=quote_side,
            quote_price=quote_price,
        )

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        del market_id, trade_data

    def on_tick(self) -> None:
        return None

    def diagnostics_snapshot(self) -> dict[str, Any]:
        return {
            "walls_identified": self._walls_identified,
            "walls_aged_past_threshold": self._walls_aged_past_threshold,
            "wall_age_ms_threshold": self.wall_age_ms,
            "min_distance_from_mid_ticks": str(self.min_distance_from_mid_ticks),
            "min_structural_wall_size_usd": str(self.min_structural_wall_size_usd),
            "jump_quotes_emitted": self._jump_quotes_emitted,
            "cancel_all_triggered": self._cancel_all_triggered,
        }

    def _record_wall_observation(
        self,
        state: _MarketState,
        *,
        candidate: _WallCandidate,
    ) -> tuple[_ObservedWall, int]:
        observed_wall = state.observed_wall
        current_ms = self.current_timestamp_ms
        if (
            observed_wall is None
            or observed_wall.wall_side != candidate.wall_side
            or observed_wall.wall_price != candidate.price
        ):
            observed_wall = _ObservedWall(
                wall_side=candidate.wall_side,
                wall_price=candidate.price,
                first_seen_at_ms=current_ms,
            )
            state.observed_wall = observed_wall
            self._walls_identified += 1

        wall_age_ms = max(0, current_ms - observed_wall.first_seen_at_ms)
        if wall_age_ms >= self.wall_age_ms and not observed_wall.age_qualified:
            observed_wall.age_qualified = True
            self._walls_aged_past_threshold += 1
        return observed_wall, wall_age_ms

    def _select_wall_candidate(
        self,
        *,
        bids: Sequence[NormalizedBookLevel],
        asks: Sequence[NormalizedBookLevel],
    ) -> _WallCandidate | None:
        best_bid = bids[0].price
        best_ask = asks[0].price
        mid_price = (best_bid + best_ask) / Decimal("2")
        bid_candidate = self._best_wall_candidate(
            wall_side="BID",
            levels=bids,
            opposing_levels=asks,
            mid_price=mid_price,
        )
        ask_candidate = self._best_wall_candidate(
            wall_side="ASK",
            levels=asks,
            opposing_levels=bids,
            mid_price=mid_price,
        )
        candidates = [candidate for candidate in (bid_candidate, ask_candidate) if candidate is not None]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                candidate.ratio,
                candidate.wall_size_usd,
                candidate.price_level_vs_mid_ticks,
                candidate.size,
            ),
        )

    def _best_wall_candidate(
        self,
        *,
        wall_side: str,
        levels: Sequence[NormalizedBookLevel],
        opposing_levels: Sequence[NormalizedBookLevel],
        mid_price: Decimal,
    ) -> _WallCandidate | None:
        if not levels or not opposing_levels:
            return None
        opposing_average = sum(level.size for level in opposing_levels) / Decimal(len(opposing_levels))
        if opposing_average <= Decimal("0"):
            return None

        best_candidate: _WallCandidate | None = None
        for level in levels:
            if level.size < self.min_wall_size:
                continue
            wall_size_usd = level.price * level.size
            if wall_size_usd < self.min_structural_wall_size_usd:
                continue
            price_level_vs_mid_ticks = abs(level.price - mid_price) / self.tick_size
            if price_level_vs_mid_ticks < self.min_distance_from_mid_ticks:
                continue
            ratio = level.size / opposing_average
            if ratio < self.wall_to_opposing_ratio:
                continue
            candidate = _WallCandidate(
                wall_side=wall_side,
                price=level.price,
                size=level.size,
                wall_size_usd=wall_size_usd,
                price_level_vs_mid_ticks=price_level_vs_mid_ticks,
                opposing_average_size=opposing_average,
                ratio=ratio,
            )
            if best_candidate is None or (
                candidate.ratio,
                candidate.wall_size_usd,
                candidate.price_level_vs_mid_ticks,
                candidate.size,
            ) > (
                best_candidate.ratio,
                best_candidate.wall_size_usd,
                best_candidate.price_level_vs_mid_ticks,
                best_candidate.size,
            ):
                best_candidate = candidate
        return best_candidate

    def _build_jump_quote(
        self,
        *,
        candidate: _WallCandidate,
        bids: Sequence[NormalizedBookLevel],
        asks: Sequence[NormalizedBookLevel],
    ) -> tuple[str, Decimal | None]:
        best_bid = bids[0].price
        best_ask = asks[0].price
        if candidate.wall_side == "BID":
            quote_price = candidate.price + self.tick_size
            if quote_price >= best_ask:
                return "BID", None
            return "BID", quote_price

        quote_price = candidate.price - self.tick_size
        if quote_price <= best_bid:
            return "ASK", None
        return "ASK", quote_price

    def _current_wall_size(
        self,
        tracked_wall: _TrackedWall,
        *,
        bids: Sequence[NormalizedBookLevel],
        asks: Sequence[NormalizedBookLevel],
    ) -> Decimal:
        levels = bids if tracked_wall.wall_side == "BID" else asks
        for level in levels:
            if level.price == tracked_wall.wall_price:
                return level.size
        return Decimal("0")

    def _cancel_all_quotes(
        self,
        market_id: str,
        *,
        tracked_wall: _TrackedWall,
        current_size: Decimal,
    ) -> None:
        self._cancel_all_triggered += 1
        self.submit_order(
            PriorityOrderContext(
                market_id=market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=_ONE,
                target_price=Decimal("0.01"),
                anchor_volume=Decimal("1"),
                max_capital=Decimal("0.01"),
                signal_metadata={
                    "strategy": "wall_jumper",
                    "action": "CANCEL_ALL",
                    "wall_id": tracked_wall.wall_id,
                    "wall_side": tracked_wall.wall_side,
                    "wall_price": str(tracked_wall.wall_price),
                    "initial_wall_size": str(tracked_wall.initial_size),
                    "current_wall_size": str(current_size),
                },
            )
        )

    def _state_for(self, market_id: str) -> _MarketState:
        state = self._market_state.get(market_id)
        if state is None:
            state = _MarketState()
            self._market_state[market_id] = state
        return state

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value