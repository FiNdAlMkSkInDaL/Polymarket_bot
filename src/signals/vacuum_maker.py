from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, Sequence

from src.execution.priority_context import PriorityOrderContext
from src.signals.base_strategy import BaseStrategy
from src.signals.obi_scalper import ObiScalper


_ONE = Decimal("1")


class ClockMs(Protocol):
    def __call__(self) -> int:
        ...


@dataclass(slots=True)
class _MarketState:
    status: str = "IDLE"
    crash_direction: str | None = None
    crash_obi: Decimal | None = None
    crash_detected_at_ms: int | None = None
    vacuum_started_at_ms: int | None = None
    last_best_bid: Decimal | None = None
    last_best_ask: Decimal | None = None
    last_spread: Decimal | None = None
    last_bid_top3_depth: Decimal = Decimal("0")
    last_ask_top3_depth: Decimal = Decimal("0")
    last_top3_depth: Decimal = Decimal("0")
    recent_large_trade_direction: str | None = None
    recent_large_trade_size: Decimal = Decimal("0")
    recent_large_trade_at_ms: int | None = None
    has_live_quotes: bool = False


@dataclass(slots=True)
class VacuumMaker(BaseStrategy):
    dispatcher: Any | None = None
    market_catalog: Any | None = None
    clock: ClockMs = field(default=lambda: int(time.time() * 1000))
    depth_levels: int = 3
    crash_abs_obi: Decimal = Decimal("0.95")
    large_trade_min_size: Decimal = Decimal("25")
    large_trade_depth_ratio: Decimal = Decimal("0.25")
    recent_trade_memory_ms: int = 500
    vacuum_min_spread_ticks: int = 1
    max_vacuum_window_ms: int = 2000
    order_size: Decimal = Decimal("10")
    tick_size: Decimal = Decimal("0.01")
    signal_source: str = "MANUAL"
    _market_state: dict[str, _MarketState] = field(init=False, repr=False)
    _crash_imminent_entries: int = field(init=False, repr=False)
    _vacuum_entries: int = field(init=False, repr=False)
    _recent_trade_confirmed_vacuums: int = field(init=False, repr=False)
    _skipped_missing_bbo: int = field(init=False, repr=False)
    _skipped_spread_too_tight: int = field(init=False, repr=False)
    _spread_wide_at_quote_time: int = field(init=False, repr=False)
    _ticks_scanned_in_vacuum: int = field(init=False, repr=False)
    _vacuum_aborted_max_window: int = field(init=False, repr=False)
    _quotes_attempted: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock,
        )
        if not isinstance(self.depth_levels, int) or self.depth_levels <= 0:
            raise ValueError("depth_levels must be a strictly positive int")
        self.crash_abs_obi = self._as_decimal(self.crash_abs_obi, name="crash_abs_obi")
        self.large_trade_min_size = self._as_decimal(self.large_trade_min_size, name="large_trade_min_size")
        self.large_trade_depth_ratio = self._as_decimal(self.large_trade_depth_ratio, name="large_trade_depth_ratio")
        self.order_size = self._as_decimal(self.order_size, name="order_size")
        self.tick_size = self._as_decimal(self.tick_size, name="tick_size")
        if self.crash_abs_obi <= Decimal("0") or self.crash_abs_obi >= _ONE:
            raise ValueError("crash_abs_obi must be between 0 and 1")
        if self.large_trade_min_size <= Decimal("0"):
            raise ValueError("large_trade_min_size must be strictly positive")
        if self.large_trade_depth_ratio <= Decimal("0"):
            raise ValueError("large_trade_depth_ratio must be strictly positive")
        if not isinstance(self.recent_trade_memory_ms, int) or self.recent_trade_memory_ms <= 0:
            raise ValueError("recent_trade_memory_ms must be a strictly positive int")
        if not isinstance(self.vacuum_min_spread_ticks, int) or self.vacuum_min_spread_ticks <= 0:
            raise ValueError("vacuum_min_spread_ticks must be a strictly positive int")
        if not isinstance(self.max_vacuum_window_ms, int) or self.max_vacuum_window_ms <= 0:
            raise ValueError("max_vacuum_window_ms must be a strictly positive int")
        if self.order_size <= Decimal("0"):
            raise ValueError("order_size must be strictly positive")
        if self.tick_size <= Decimal("0"):
            raise ValueError("tick_size must be strictly positive")
        self.signal_source = str(self.signal_source or "").strip().upper() or "MANUAL"
        self._market_state = {}
        self._crash_imminent_entries = 0
        self._vacuum_entries = 0
        self._recent_trade_confirmed_vacuums = 0
        self._skipped_missing_bbo = 0
        self._skipped_spread_too_tight = 0
        self._spread_wide_at_quote_time = 0
        self._ticks_scanned_in_vacuum = 0
        self._vacuum_aborted_max_window = 0
        self._quotes_attempted = 0

    def bind_dispatcher(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher
        super().bind_dispatcher(dispatcher)

    def bind_market_catalog(self, market_catalog: Any) -> None:
        self.market_catalog = market_catalog
        super().bind_market_catalog(market_catalog)

    def bind_clock(self, clock) -> None:
        self.clock = clock
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
        state.last_best_bid = bids[0].price
        state.last_best_ask = asks[0].price
        state.last_spread = asks[0].price - bids[0].price
        state.last_bid_top3_depth = sum(level.size for level in bids)
        state.last_ask_top3_depth = sum(level.size for level in asks)

        obi = ObiScalper.calculate_obi(bids, asks, depth_levels=self.depth_levels)
        if obi is None:
            return

        if obi >= self.crash_abs_obi:
            self._handle_crash_signal(state, obi=obi, crash_direction="BUY", depth=state.last_ask_top3_depth)
            return
        if obi <= -self.crash_abs_obi:
            self._handle_crash_signal(state, obi=obi, crash_direction="SELL", depth=state.last_bid_top3_depth)

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            return
        state = self._state_for(normalized_market_id)

        trade_side = str(trade_data.get("side") or "").strip().upper()
        trade_size = self._as_decimal(trade_data.get("size", "0"), name="trade_size")
        if trade_size <= Decimal("0"):
            return
        trade_timestamp_ms = int(trade_data.get("timestamp_ms") or self.current_timestamp_ms)
        trade_direction = self._trade_direction_from_side(trade_side)
        if trade_direction is not None:
            self._maybe_record_recent_large_trade(
                state,
                trade_direction=trade_direction,
                trade_size=trade_size,
                trade_timestamp_ms=trade_timestamp_ms,
            )
        if state.status != "CRASH_IMMINENT":
            return
        if state.crash_direction == "BUY" and trade_side not in {"BUY", ""}:
            return
        if state.crash_direction == "SELL" and trade_side not in {"SELL", ""}:
            return

        large_trade_threshold = self._large_trade_threshold_for_direction(state, state.crash_direction)
        if trade_size < large_trade_threshold:
            return

        self._enter_vacuum(state, vacuum_started_at_ms=trade_timestamp_ms)

    def on_tick(self) -> None:
        current_ms = self.current_timestamp_ms
        for market_id, state in self._market_state.items():
            if state.status != "VACUUM" or state.vacuum_started_at_ms is None:
                continue
            self._ticks_scanned_in_vacuum += 1
            if state.last_best_bid is None or state.last_best_ask is None or state.last_spread is None:
                self._skipped_missing_bbo += 1
                if current_ms - state.vacuum_started_at_ms > self.max_vacuum_window_ms:
                    self._abort_vacuum(state)
                continue
            if self._spread_is_wide_enough(state.last_spread):
                self._spread_wide_at_quote_time += 1
                self._quotes_attempted += 1
                self._quote_widened_spread(market_id, state)
                state.status = "QUOTED"
                state.has_live_quotes = True
                continue
            self._skipped_spread_too_tight += 1
            if current_ms - state.vacuum_started_at_ms > self.max_vacuum_window_ms:
                self._abort_vacuum(state)

    def _abort_vacuum(self, state: _MarketState) -> None:
        self._vacuum_aborted_max_window += 1
        state.status = "IDLE"
        state.vacuum_started_at_ms = None
        state.has_live_quotes = False

    def _spread_is_wide_enough(self, spread: Decimal) -> bool:
        threshold = self.tick_size * Decimal(self.vacuum_min_spread_ticks)
        return spread > threshold

    def _enter_crash_imminent(
        self,
        state: _MarketState,
        *,
        obi: Decimal,
        crash_direction: str,
        depth: Decimal,
    ) -> None:
        if state.status != "CRASH_IMMINENT":
            self._crash_imminent_entries += 1
        state.status = "CRASH_IMMINENT"
        state.crash_obi = obi
        state.crash_direction = crash_direction
        state.crash_detected_at_ms = self.current_timestamp_ms
        state.vacuum_started_at_ms = None
        state.last_top3_depth = depth

    def diagnostics_snapshot(self) -> dict[str, Any]:
        return {
            "crash_imminent_entries": self._crash_imminent_entries,
            "vacuum_entries": self._vacuum_entries,
            "recent_trade_confirmed_vacuums": self._recent_trade_confirmed_vacuums,
            "skipped_missing_bbo": self._skipped_missing_bbo,
            "skipped_spread_too_tight": self._skipped_spread_too_tight,
            "spread_wide_at_quote_time": self._spread_wide_at_quote_time,
            "ticks_scanned_in_vacuum": self._ticks_scanned_in_vacuum,
            "vacuum_aborted_max_window": self._vacuum_aborted_max_window,
            "quotes_attempted": self._quotes_attempted,
        }

    def _handle_crash_signal(
        self,
        state: _MarketState,
        *,
        obi: Decimal,
        crash_direction: str,
        depth: Decimal,
    ) -> None:
        recent_trade_at_ms = state.recent_large_trade_at_ms
        if self._recent_large_trade_matches(
            state,
            crash_direction=crash_direction,
            reference_timestamp_ms=self.current_timestamp_ms,
        ) and recent_trade_at_ms is not None:
            state.crash_obi = obi
            state.crash_direction = crash_direction
            state.crash_detected_at_ms = self.current_timestamp_ms
            state.last_top3_depth = depth
            self._recent_trade_confirmed_vacuums += 1
            self._enter_vacuum(state, vacuum_started_at_ms=recent_trade_at_ms)
            return
        self._enter_crash_imminent(state, obi=obi, crash_direction=crash_direction, depth=depth)

    def _enter_vacuum(self, state: _MarketState, *, vacuum_started_at_ms: int) -> None:
        if state.status != "VACUUM":
            self._vacuum_entries += 1
        state.status = "VACUUM"
        state.vacuum_started_at_ms = vacuum_started_at_ms
        state.recent_large_trade_direction = None
        state.recent_large_trade_size = Decimal("0")
        state.recent_large_trade_at_ms = None

    def _maybe_record_recent_large_trade(
        self,
        state: _MarketState,
        *,
        trade_direction: str,
        trade_size: Decimal,
        trade_timestamp_ms: int,
    ) -> None:
        large_trade_threshold = self._large_trade_threshold_for_direction(state, trade_direction)
        if trade_size < large_trade_threshold:
            return
        state.recent_large_trade_direction = trade_direction
        state.recent_large_trade_size = trade_size
        state.recent_large_trade_at_ms = trade_timestamp_ms

    def _recent_large_trade_matches(
        self,
        state: _MarketState,
        *,
        crash_direction: str,
        reference_timestamp_ms: int,
    ) -> bool:
        recent_trade_at_ms = state.recent_large_trade_at_ms
        if recent_trade_at_ms is None or state.recent_large_trade_direction != crash_direction:
            return False
        return abs(reference_timestamp_ms - recent_trade_at_ms) <= self.recent_trade_memory_ms

    def _large_trade_threshold_for_direction(self, state: _MarketState, crash_direction: str | None) -> Decimal:
        reference_depth = Decimal("0")
        if crash_direction == "BUY":
            reference_depth = state.last_ask_top3_depth
        elif crash_direction == "SELL":
            reference_depth = state.last_bid_top3_depth
        return max(self.large_trade_min_size, reference_depth * self.large_trade_depth_ratio)

    @staticmethod
    def _trade_direction_from_side(trade_side: str) -> str | None:
        normalized_side = str(trade_side or "").strip().upper()
        if normalized_side == "BUY":
            return "BUY"
        if normalized_side == "SELL":
            return "SELL"
        return None

    def _quote_widened_spread(self, market_id: str, state: _MarketState) -> None:
        assert state.last_best_bid is not None
        assert state.last_best_ask is not None
        assert state.last_spread is not None

        bid_price = state.last_best_bid + self.tick_size
        ask_price = state.last_best_ask - self.tick_size
        if ask_price <= bid_price:
            midpoint = (state.last_best_bid + state.last_best_ask) / Decimal("2")
            bid_price = midpoint - (self.tick_size / Decimal("2"))
            ask_price = midpoint + (self.tick_size / Decimal("2"))
            bid_price = self._round_down_to_tick(bid_price)
            ask_price = self._round_up_to_tick(ask_price)
            if ask_price <= bid_price:
                ask_price = bid_price + self.tick_size

        conviction_scalar = Decimal("1")
        self.submit_order(self._build_quote_context(market_id, state, quote_side="BID", target_price=bid_price, conviction_scalar=conviction_scalar))
        self.submit_order(self._build_quote_context(market_id, state, quote_side="ASK", target_price=ask_price, conviction_scalar=conviction_scalar))

    def _build_quote_context(
        self,
        market_id: str,
        state: _MarketState,
        *,
        quote_side: str,
        target_price: Decimal,
        conviction_scalar: Decimal,
    ) -> PriorityOrderContext:
        return PriorityOrderContext(
            market_id=market_id,
            side="YES",
            signal_source=self.signal_source,
            conviction_scalar=conviction_scalar,
            target_price=target_price,
            anchor_volume=self.order_size,
            max_capital=target_price * self.order_size,
            signal_metadata={
                "strategy": "vacuum_maker",
                "post_only": True,
                "time_in_force": "GTC",
                "liquidity_intent": "MAKER",
                "quote_side": quote_side,
                "quote_id": f"vacuum_maker:{market_id}:{quote_side.lower()}",
                "crash_direction": state.crash_direction,
                "crash_obi": str(state.crash_obi) if state.crash_obi is not None else None,
                "vacuum_min_spread_ticks": self.vacuum_min_spread_ticks,
                "max_vacuum_window_ms": self.max_vacuum_window_ms,
                "entry_theory": "post_steamroller_vacuum",
            },
        )

    def _state_for(self, market_id: str) -> _MarketState:
        state = self._market_state.get(market_id)
        if state is None:
            state = _MarketState()
            self._market_state[market_id] = state
        return state

    def _round_down_to_tick(self, price: Decimal) -> Decimal:
        return (price / self.tick_size).to_integral_value(rounding="ROUND_FLOOR") * self.tick_size

    def _round_up_to_tick(self, price: Decimal) -> Decimal:
        return (price / self.tick_size).to_integral_value(rounding="ROUND_CEILING") * self.tick_size

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value