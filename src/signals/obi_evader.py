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
class ObiEvader(BaseStrategy):
    dispatcher: Any | None = None
    market_catalog: Any | None = None
    depth_levels: int = 3
    order_size: Decimal = Decimal("5")
    safe_zone_abs_obi: Decimal = Decimal("0.5")
    toxic_zone_abs_obi: Decimal = Decimal("0.85")
    signal_source: str = "MANUAL"
    clock_ms: ClockMs = field(default=lambda: int(time.time() * 1000))
    _active_mode: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock_ms,
        )
        self.order_size = self._as_decimal(self.order_size, name="order_size")
        self.safe_zone_abs_obi = self._as_decimal(self.safe_zone_abs_obi, name="safe_zone_abs_obi")
        self.toxic_zone_abs_obi = self._as_decimal(self.toxic_zone_abs_obi, name="toxic_zone_abs_obi")
        if self.depth_levels <= 0:
            raise ValueError("depth_levels must be strictly positive")
        if self.order_size <= Decimal("0"):
            raise ValueError("order_size must be strictly positive")
        if self.safe_zone_abs_obi <= Decimal("0") or self.safe_zone_abs_obi >= _ONE:
            raise ValueError("safe_zone_abs_obi must be between 0 and 1")
        if self.toxic_zone_abs_obi <= self.safe_zone_abs_obi or self.toxic_zone_abs_obi >= _ONE:
            raise ValueError("toxic_zone_abs_obi must be greater than safe_zone_abs_obi and less than 1")
        self.signal_source = str(self.signal_source or "").strip().upper() or "MANUAL"
        self._active_mode = {}

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
        if not normalized_market_id:
            return

        bids = ObiScalper._normalize_levels(top_bids)[: self.depth_levels]
        asks = ObiScalper._normalize_levels(top_asks)[: self.depth_levels]
        if not bids or not asks:
            return

        obi = ObiScalper.calculate_obi(bids, asks, depth_levels=self.depth_levels)
        if obi is None:
            return

        abs_obi = abs(obi)
        if abs_obi > self.toxic_zone_abs_obi:
            if self._active_mode.get(normalized_market_id) != "TOXIC":
                self._cancel_all_quotes(normalized_market_id, obi)
                self._active_mode[normalized_market_id] = "TOXIC"
            return

        if abs_obi < self.safe_zone_abs_obi:
            self._quote_market(normalized_market_id, bids[0].price, asks[0].price, obi)
            self._active_mode[normalized_market_id] = "SAFE"

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        del market_id, trade_data

    def on_tick(self) -> None:
        return None

    def _quote_market(self, market_id: str, best_bid: Decimal, best_ask: Decimal, obi: Decimal) -> None:
        if best_bid <= Decimal("0") or best_ask <= Decimal("0") or best_ask <= best_bid:
            return

        self.submit_order(
            PriorityOrderContext(
                market_id=market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=min(_ONE, abs(obi)),
                target_price=best_bid,
                anchor_volume=self.order_size,
                max_capital=best_bid * self.order_size,
                signal_metadata={
                    "strategy": "obi_evader",
                    "obi": str(obi),
                    "depth_levels": self.depth_levels,
                    "post_only": True,
                    "time_in_force": "GTC",
                    "liquidity_intent": "MAKER",
                    "quote_side": "BID",
                    "quote_id": f"obi_evader:{market_id}:bid",
                },
            )
        )
        self.submit_order(
            PriorityOrderContext(
                market_id=market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=min(_ONE, abs(obi)),
                target_price=best_ask,
                anchor_volume=self.order_size,
                max_capital=best_ask * self.order_size,
                signal_metadata={
                    "strategy": "obi_evader",
                    "obi": str(obi),
                    "depth_levels": self.depth_levels,
                    "post_only": True,
                    "time_in_force": "GTC",
                    "liquidity_intent": "MAKER",
                    "quote_side": "ASK",
                    "quote_id": f"obi_evader:{market_id}:ask",
                },
            )
        )

    def _cancel_all_quotes(self, market_id: str, obi: Decimal) -> None:
        self.submit_order(
            PriorityOrderContext(
                market_id=market_id,
                side="YES",
                signal_source=self.signal_source,
                conviction_scalar=min(_ONE, abs(obi)),
                target_price=Decimal("0.01"),
                anchor_volume=Decimal("1"),
                max_capital=Decimal("0.01"),
                signal_metadata={
                    "strategy": "obi_evader",
                    "obi": str(obi),
                    "action": "CANCEL_ALL",
                },
            )
        )

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value