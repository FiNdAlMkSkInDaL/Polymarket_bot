from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Protocol, Sequence

from src.execution.priority_context import PriorityOrderContext
from src.models.inventory_skew import InventorySkewQuote, InventorySkewInputs, compute_inventory_skew_quotes
from src.signals.base_strategy import BaseStrategy
from src.signals.obi_scalper import ObiScalper


_ONE = Decimal("1")
_ONE_HUNDRED = Decimal("100")
_ZERO = Decimal("0")


class ClockMs(Protocol):
    def __call__(self) -> int:
        ...


@dataclass(slots=True)
class _MarketState:
    last_best_bid: Decimal | None = None
    last_best_ask: Decimal | None = None
    last_spread: Decimal | None = None
    last_inventory_usd: Decimal = Decimal("0")
    last_quote: InventorySkewQuote | None = None


@dataclass(slots=True)
class LongTailMarketMaker(BaseStrategy):
    dispatcher: Any | None = None
    market_catalog: Any | None = None
    clock: ClockMs = field(default=lambda: int(time.time() * 1000))
    min_spread_cents: Decimal = Decimal("1")
    max_inventory_usd: Decimal = Decimal("100")
    base_order_size_usd: Decimal = Decimal("10")
    inventory_provider: Callable[[str], Any] | None = None
    signal_source: str = "MANUAL"
    _market_state: dict[str, _MarketState] = field(init=False, repr=False)
    _inventory_provider: Callable[[str], Decimal] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock,
        )
        if not callable(self.clock):
            raise ValueError("clock must be callable")

        self.min_spread_cents = self._as_decimal(self.min_spread_cents, name="min_spread_cents")
        self.max_inventory_usd = self._as_decimal(self.max_inventory_usd, name="max_inventory_usd")
        self.base_order_size_usd = self._as_decimal(self.base_order_size_usd, name="base_order_size_usd")
        if self.min_spread_cents < _ZERO:
            raise ValueError("min_spread_cents must be non-negative")
        if self.max_inventory_usd <= _ZERO:
            raise ValueError("max_inventory_usd must be strictly positive")
        if self.base_order_size_usd <= _ZERO:
            raise ValueError("base_order_size_usd must be strictly positive")

        self.signal_source = str(self.signal_source or "").strip().upper() or "MANUAL"
        if self.signal_source not in {"OFI", "SI9", "SI10", "CONTAGION", "MANUAL", "CTF", "REWARD"}:
            raise ValueError(f"Unsupported signal_source: {self.signal_source!r}")

        if self.inventory_provider is not None and not callable(self.inventory_provider):
            raise ValueError("inventory_provider must be callable")
        self._inventory_provider = self._wrap_inventory_provider(self.inventory_provider)
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

    def bind_inventory_provider(self, inventory_provider: Callable[[str], Any] | None) -> None:
        if inventory_provider is not None and not callable(inventory_provider):
            raise ValueError("inventory_provider must be callable")
        self.inventory_provider = inventory_provider
        self._inventory_provider = self._wrap_inventory_provider(inventory_provider)

    def on_bbo_update(
        self,
        market_id: str,
        top_bids: Sequence[dict[str, Any]],
        top_asks: Sequence[dict[str, Any]],
    ) -> None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            return

        bids = ObiScalper._normalize_levels(top_bids)
        asks = ObiScalper._normalize_levels(top_asks)
        if not bids or not asks:
            return

        best_bid = bids[0].price
        best_ask = asks[0].price
        spread = best_ask - best_bid
        if spread < self._min_spread_dollars:
            return

        mid_price = (best_bid + best_ask) / Decimal("2")
        current_inventory_usd = self._inventory_provider(normalized_market_id)
        quote = compute_inventory_skew_quotes(
            InventorySkewInputs(
                current_inventory_usd=current_inventory_usd,
                max_inventory_usd=self.max_inventory_usd,
                base_spread=spread,
                mid_price=mid_price,
                best_bid=best_bid,
                best_ask=best_ask,
            )
        )

        state = self._state_for(normalized_market_id)
        state.last_best_bid = best_bid
        state.last_best_ask = best_ask
        state.last_spread = spread
        state.last_inventory_usd = current_inventory_usd
        state.last_quote = quote

        self.submit_order(
            self._build_quote_context(
                market_id=normalized_market_id,
                quote_side="BID",
                target_price=quote.bid_price,
                spread=spread,
                inventory_usd=current_inventory_usd,
                quote=quote,
            )
        )
        self.submit_order(
            self._build_quote_context(
                market_id=normalized_market_id,
                quote_side="ASK",
                target_price=quote.ask_price,
                spread=spread,
                inventory_usd=current_inventory_usd,
                quote=quote,
            )
        )

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        return None

    def on_tick(self) -> None:
        return None

    @property
    def min_spread_dollars(self) -> Decimal:
        return self._min_spread_dollars

    @property
    def latest_quotes_by_market(self) -> dict[str, InventorySkewQuote | None]:
        return {
            market_id: state.last_quote
            for market_id, state in self._market_state.items()
        }

    @property
    def _min_spread_dollars(self) -> Decimal:
        return self.min_spread_cents / _ONE_HUNDRED

    def _build_quote_context(
        self,
        *,
        market_id: str,
        quote_side: str,
        target_price: Decimal,
        spread: Decimal,
        inventory_usd: Decimal,
        quote: InventorySkewQuote,
    ) -> PriorityOrderContext:
        order_size = self.base_order_size_usd / target_price
        return PriorityOrderContext(
            market_id=market_id,
            side="YES",
            signal_source=self.signal_source,
            conviction_scalar=_ONE,
            target_price=target_price,
            anchor_volume=order_size,
            max_capital=self.base_order_size_usd,
            signal_metadata={
                "strategy": "long_tail_maker",
                "post_only": True,
                "time_in_force": "GTC",
                "liquidity_intent": "MAKER",
                "quote_side": quote_side,
                "quote_id": f"long_tail_maker:{market_id}:{quote_side.lower()}",
                "inventory_usd": str(inventory_usd),
                "inventory_ratio": str(quote.inventory_ratio),
                "urgency": str(quote.urgency),
                "center_shift": str(quote.center_shift),
                "adjusted_half_spread": str(quote.adjusted_half_spread),
                "observed_spread": str(spread),
                "min_spread_cents": str(self.min_spread_cents),
                "max_inventory_usd": str(self.max_inventory_usd),
                "base_order_size_usd": str(self.base_order_size_usd),
                "aggressive_exit": quote.aggressive_exit,
                "aggressive_side": quote.aggressive_side,
                "entry_theory": "inventory_skewed_long_tail_market_making",
            },
        )

    def _state_for(self, market_id: str) -> _MarketState:
        state = self._market_state.get(market_id)
        if state is None:
            state = _MarketState()
            self._market_state[market_id] = state
        return state

    def _wrap_inventory_provider(
        self,
        inventory_provider: Callable[[str], Any] | None,
    ) -> Callable[[str], Decimal]:
        if inventory_provider is None:
            return lambda market_id: Decimal("0")

        def _provider(market_id: str) -> Decimal:
            raw_inventory = inventory_provider(market_id)
            if raw_inventory in (None, ""):
                return Decimal("0")
            return self._as_decimal(raw_inventory, name="current_inventory_usd")

        return _provider

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value