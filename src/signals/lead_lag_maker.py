from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, Sequence

from src.execution.priority_context import PriorityOrderContext
from src.signals.base_strategy import BaseStrategy


class ClockMs(Protocol):
    def __call__(self) -> int:
        ...


_ONE = Decimal("1")


@dataclass(slots=True)
class LeadLagMaker(BaseStrategy):
    primary_market_id: str
    secondary_market_id: str
    dispatcher: Any | None = None
    market_catalog: Any | None = None
    order_size: Decimal = Decimal("5")
    contagion_move_cents: Decimal = Decimal("2")
    lookback_window_ms: int = 1_000
    cooldown_ms: int = 5_000
    signal_source: str = "CONTAGION"
    clock_ms: ClockMs = field(default=lambda: int(time.time() * 1000))
    _mid_history: dict[str, deque[tuple[int, Decimal]]] = field(init=False, repr=False)
    _contagion_until_ms: int = field(init=False, default=0, repr=False)
    _secondary_mode: str = field(init=False, default="IDLE", repr=False)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock_ms,
        )
        self.primary_market_id = str(self.primary_market_id or "").strip()
        self.secondary_market_id = str(self.secondary_market_id or "").strip()
        if not self.primary_market_id:
            raise ValueError("primary_market_id must be a non-empty string")
        if not self.secondary_market_id:
            raise ValueError("secondary_market_id must be a non-empty string")
        self.order_size = self._as_decimal(self.order_size, name="order_size")
        self.contagion_move_cents = self._as_decimal(self.contagion_move_cents, name="contagion_move_cents")
        if self.order_size <= Decimal("0"):
            raise ValueError("order_size must be strictly positive")
        if self.contagion_move_cents <= Decimal("0"):
            raise ValueError("contagion_move_cents must be strictly positive")
        if self.lookback_window_ms <= 0:
            raise ValueError("lookback_window_ms must be strictly positive")
        if self.cooldown_ms <= 0:
            raise ValueError("cooldown_ms must be strictly positive")
        self.signal_source = str(self.signal_source or "").strip().upper() or "CONTAGION"
        self._mid_history = {
            self.primary_market_id: deque(),
            self.secondary_market_id: deque(),
        }

    def bind_dispatcher(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher
        super().bind_dispatcher(dispatcher)

    def bind_market_catalog(self, market_catalog: Any) -> None:
        self.market_catalog = market_catalog
        super().bind_market_catalog(market_catalog)

    def bind_clock(self, clock) -> None:
        self.clock_ms = clock
        super().bind_clock(clock)

    def on_bbo_update(self, market_id: str, top_bids: Sequence[dict[str, Any]], top_asks: Sequence[dict[str, Any]]) -> None:
        normalized_market_id = str(market_id or "").strip()
        if normalized_market_id not in {self.primary_market_id, self.secondary_market_id}:
            return
        best_bid, best_ask = self._best_prices(top_bids, top_asks)
        if best_bid is None or best_ask is None or best_ask <= best_bid:
            return

        timestamp_ms = self.current_timestamp_ms
        mid_price = (best_bid + best_ask) / Decimal("2")
        self._record_mid(normalized_market_id, timestamp_ms, mid_price)

        if normalized_market_id == self.primary_market_id and self._primary_contagion_triggered():
            self._trigger_contagion(timestamp_ms)
            return

        if normalized_market_id != self.secondary_market_id:
            return
        if timestamp_ms < self._contagion_until_ms:
            return
        self._quote_secondary(best_bid, best_ask)
        self._secondary_mode = "QUOTING"

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        del market_id, trade_data

    def on_tick(self) -> None:
        return None

    def _record_mid(self, market_id: str, timestamp_ms: int, mid_price: Decimal) -> None:
        history = self._mid_history.setdefault(market_id, deque())
        history.append((timestamp_ms, mid_price))
        cutoff_ms = timestamp_ms - self.lookback_window_ms
        while history and history[0][0] < cutoff_ms:
            history.popleft()

    def _primary_contagion_triggered(self) -> bool:
        history = self._mid_history.get(self.primary_market_id)
        if not history or len(history) < 2:
            return False
        current_mid = history[-1][1]
        oldest_mid = history[0][1]
        move_cents = abs(current_mid - oldest_mid) * Decimal("100")
        return move_cents >= self.contagion_move_cents

    def _trigger_contagion(self, timestamp_ms: int) -> None:
        self._contagion_until_ms = max(self._contagion_until_ms, timestamp_ms + self.cooldown_ms)
        if self._secondary_mode != "CONTAGION_DETECTED":
            self._cancel_secondary_quotes()
        self._secondary_mode = "CONTAGION_DETECTED"

    def _quote_secondary(self, best_bid: Decimal, best_ask: Decimal) -> None:
        self.submit_order(
            PriorityOrderContext(
                market_id=self.secondary_market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=_ONE,
                target_price=best_bid,
                anchor_volume=self.order_size,
                max_capital=best_bid * self.order_size,
                signal_metadata={
                    "strategy": "lead_lag_maker",
                    "role": "secondary_maker",
                    "post_only": True,
                    "time_in_force": "GTC",
                    "liquidity_intent": "MAKER",
                    "quote_side": "BID",
                    "quote_id": f"lead_lag_maker:{self.secondary_market_id}:bid",
                    "primary_market_id": self.primary_market_id,
                    "secondary_market_id": self.secondary_market_id,
                },
            )
        )
        self.submit_order(
            PriorityOrderContext(
                market_id=self.secondary_market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=_ONE,
                target_price=best_ask,
                anchor_volume=self.order_size,
                max_capital=best_ask * self.order_size,
                signal_metadata={
                    "strategy": "lead_lag_maker",
                    "role": "secondary_maker",
                    "post_only": True,
                    "time_in_force": "GTC",
                    "liquidity_intent": "MAKER",
                    "quote_side": "ASK",
                    "quote_id": f"lead_lag_maker:{self.secondary_market_id}:ask",
                    "primary_market_id": self.primary_market_id,
                    "secondary_market_id": self.secondary_market_id,
                },
            )
        )

    def _cancel_secondary_quotes(self) -> None:
        self.submit_order(
            PriorityOrderContext(
                market_id=self.secondary_market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=_ONE,
                target_price=Decimal("0.01"),
                anchor_volume=Decimal("1"),
                max_capital=Decimal("0.01"),
                signal_metadata={
                    "strategy": "lead_lag_maker",
                    "role": "secondary_maker",
                    "action": "CANCEL_ALL",
                    "primary_market_id": self.primary_market_id,
                    "secondary_market_id": self.secondary_market_id,
                },
            )
        )

    @staticmethod
    def _best_prices(
        top_bids: Sequence[dict[str, Any]],
        top_asks: Sequence[dict[str, Any]],
    ) -> tuple[Decimal | None, Decimal | None]:
        best_bid = LeadLagMaker._price_from_levels(top_bids)
        best_ask = LeadLagMaker._price_from_levels(top_asks)
        return best_bid, best_ask

    @staticmethod
    def _price_from_levels(levels: Sequence[dict[str, Any]]) -> Decimal | None:
        if not levels:
            return None
        raw_price = levels[0].get("price")
        if raw_price in (None, ""):
            return None
        return Decimal(str(raw_price))

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value