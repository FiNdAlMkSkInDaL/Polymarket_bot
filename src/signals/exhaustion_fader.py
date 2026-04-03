from __future__ import annotations

import time
from collections import deque
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
class _TradeSample:
    timestamp_ms: int
    side: str
    size: Decimal
    mid_price: Decimal


@dataclass(slots=True)
class _MarketState:
    last_best_bid: Decimal | None = None
    last_best_ask: Decimal | None = None
    last_mid_price: Decimal | None = None
    last_obi: Decimal | None = None
    trade_window: deque[_TradeSample] = field(default_factory=deque)
    last_signal_side: str | None = None
    last_signal_at_ms: int | None = None


@dataclass(slots=True)
class ExhaustionFader(BaseStrategy):
    dispatcher: Any | None = None
    market_catalog: Any | None = None
    clock: ClockMs = field(default=lambda: int(time.time() * 1000))
    depth_levels: int = 3
    trade_window_ms: int = 5_000
    spike_threshold: Decimal = Decimal("0.04")
    flat_obi_abs: Decimal = Decimal("0.5")
    toxic_obi_abs: Decimal = Decimal("0.85")
    order_size: Decimal = Decimal("5")
    signal_source: str = "MANUAL"
    _market_state: dict[str, _MarketState] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock,
        )
        if not isinstance(self.depth_levels, int) or self.depth_levels <= 0:
            raise ValueError("depth_levels must be a strictly positive int")
        if not isinstance(self.trade_window_ms, int) or self.trade_window_ms <= 0:
            raise ValueError("trade_window_ms must be a strictly positive int")
        self.spike_threshold = self._as_decimal(self.spike_threshold, name="spike_threshold")
        self.flat_obi_abs = self._as_decimal(self.flat_obi_abs, name="flat_obi_abs")
        self.toxic_obi_abs = self._as_decimal(self.toxic_obi_abs, name="toxic_obi_abs")
        self.order_size = self._as_decimal(self.order_size, name="order_size")
        if self.spike_threshold <= Decimal("0"):
            raise ValueError("spike_threshold must be strictly positive")
        if self.flat_obi_abs <= Decimal("0") or self.flat_obi_abs >= _ONE:
            raise ValueError("flat_obi_abs must be between 0 and 1")
        if self.toxic_obi_abs <= self.flat_obi_abs or self.toxic_obi_abs >= _ONE:
            raise ValueError("toxic_obi_abs must be greater than flat_obi_abs and less than 1")
        if self.order_size <= Decimal("0"):
            raise ValueError("order_size must be strictly positive")
        self.signal_source = str(self.signal_source or "").strip().upper() or "MANUAL"
        self._market_state = {}

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
        state.last_mid_price = (bids[0].price + asks[0].price) / Decimal("2")
        state.last_obi = ObiScalper.calculate_obi(bids, asks, depth_levels=self.depth_levels)

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            return
        state = self._market_state.get(normalized_market_id)
        if state is None or state.last_mid_price is None:
            return

        trade_side = str(trade_data.get("side") or "").strip().upper()
        if trade_side not in {"BUY", "SELL"}:
            return

        trade_size = self._as_decimal(trade_data.get("size", "0"), name="trade_size")
        if trade_size <= Decimal("0"):
            return
        timestamp_ms = int(trade_data.get("timestamp_ms") or self.current_timestamp_ms)

        state.trade_window.append(
            _TradeSample(
                timestamp_ms=timestamp_ms,
                side=trade_side,
                size=trade_size,
                mid_price=state.last_mid_price,
            )
        )
        self._prune_trade_window(state.trade_window, current_timestamp_ms=timestamp_ms)
        self._maybe_fade_spike(normalized_market_id, state, current_timestamp_ms=timestamp_ms)

    def on_tick(self) -> None:
        return None

    def _maybe_fade_spike(self, market_id: str, state: _MarketState, *, current_timestamp_ms: int) -> None:
        if len(state.trade_window) < 2:
            return
        if state.last_obi is None or abs(state.last_obi) >= self.flat_obi_abs:
            return
        if abs(state.last_obi) > self.toxic_obi_abs:
            return

        sides = {sample.side for sample in state.trade_window}
        if len(sides) != 1:
            return
        spike_side = next(iter(sides))

        first_sample = state.trade_window[0]
        last_sample = state.trade_window[-1]
        if spike_side == "BUY":
            price_change = last_sample.mid_price - first_sample.mid_price
        else:
            price_change = first_sample.mid_price - last_sample.mid_price
        if price_change <= self.spike_threshold:
            return

        if state.last_signal_side == spike_side and state.last_signal_at_ms is not None:
            if current_timestamp_ms - state.last_signal_at_ms < self.trade_window_ms:
                return

        context = self._build_fade_context(market_id, state, spike_side=spike_side, price_change=price_change)
        if context is None:
            return
        self.submit_order(context)
        state.last_signal_side = spike_side
        state.last_signal_at_ms = current_timestamp_ms

    def _build_fade_context(
        self,
        market_id: str,
        state: _MarketState,
        *,
        spike_side: str,
        price_change: Decimal,
    ) -> PriorityOrderContext | None:
        if state.last_best_bid is None or state.last_best_ask is None:
            return None

        if spike_side == "BUY":
            quote_side = "ASK"
            target_price = state.last_best_ask
        else:
            quote_side = "BID"
            target_price = state.last_best_bid
        if target_price <= Decimal("0"):
            return None

        return PriorityOrderContext(
            market_id=market_id,
            side="YES",
            signal_source=self.signal_source,
            conviction_scalar=Decimal("1"),
            target_price=target_price,
            anchor_volume=self.order_size,
            max_capital=target_price * self.order_size,
            signal_metadata={
                "strategy": "exhaustion_fader",
                "post_only": True,
                "time_in_force": "GTC",
                "liquidity_intent": "MAKER",
                "quote_side": quote_side,
                "quote_id": f"exhaustion_fader:{market_id}:{quote_side.lower()}",
                "spike_side": spike_side,
                "spike_window_ms": self.trade_window_ms,
                "spike_mid_move": str(price_change),
                "obi": str(state.last_obi),
                "entry_theory": "retail_fomo_exhaustion",
            },
        )

    def _state_for(self, market_id: str) -> _MarketState:
        state = self._market_state.get(market_id)
        if state is None:
            state = _MarketState()
            self._market_state[market_id] = state
        return state

    def _prune_trade_window(self, window: deque[_TradeSample], *, current_timestamp_ms: int) -> None:
        cutoff = current_timestamp_ms - self.trade_window_ms
        while window and window[0].timestamp_ms < cutoff:
            window.popleft()

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value