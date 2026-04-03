from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Protocol, Sequence

from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchReceipt, PriorityDispatcher
from src.signals.base_strategy import BaseStrategy


_ONE = Decimal("1")


class ClockMs(Protocol):
    def __call__(self) -> int:
        ...


@dataclass(frozen=True, slots=True)
class NormalizedBookLevel:
    price: Decimal
    size: Decimal


@dataclass(slots=True)
class ObiScalper(BaseStrategy):
    """Directional taker strategy driven by top-of-book order imbalance.

    The strategy consumes replayed L2 updates via :meth:`on_bbo_update`,
    computes OBI using the top ``depth_levels`` levels, and turns extreme
    pressure into a ``PriorityOrderContext``. When a shared
    ``PriorityDispatcher`` is injected, the intent is dispatched immediately;
    otherwise the context is queued for the replay engine to drain.
    """

    dispatcher: PriorityDispatcher | None = None
    market_catalog: Any | None = None
    threshold: Decimal = Decimal("0.85")
    depth_levels: int = 3
    order_size: Decimal = Decimal("5")
    max_capital_per_order: Decimal | None = None
    cooldown_ms: int = 1_000
    signal_source: str = "MANUAL"
    clock_ms: ClockMs = field(default=lambda: int(time.time() * 1000))
    _pending_intents: deque[PriorityOrderContext] = field(init=False, repr=False)
    _last_signal_at_ms: dict[tuple[str, str], int] = field(init=False, repr=False)
    _latest_obi_by_market: dict[str, Decimal] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock_ms,
        )
        self.threshold = self._as_decimal(self.threshold, name="threshold")
        if self.threshold <= Decimal("0") or self.threshold >= _ONE:
            raise ValueError("threshold must be between 0 and 1")

        if not isinstance(self.depth_levels, int) or self.depth_levels <= 0:
            raise ValueError("depth_levels must be a strictly positive int")

        self.order_size = self._as_decimal(self.order_size, name="order_size")
        if self.order_size <= Decimal("0"):
            raise ValueError("order_size must be strictly positive")

        if self.max_capital_per_order is not None:
            self.max_capital_per_order = self._as_decimal(
                self.max_capital_per_order,
                name="max_capital_per_order",
            )
            if self.max_capital_per_order <= Decimal("0"):
                raise ValueError("max_capital_per_order must be strictly positive")

        if not isinstance(self.cooldown_ms, int) or self.cooldown_ms < 0:
            raise ValueError("cooldown_ms must be a non-negative int")

        self.signal_source = str(self.signal_source or "").strip().upper()
        if self.signal_source not in {"OFI", "SI9", "SI10", "CONTAGION", "MANUAL", "CTF", "REWARD"}:
            raise ValueError(f"Unsupported signal_source: {self.signal_source!r}")

        if not callable(self.clock_ms):
            raise ValueError("clock_ms must be callable")

        self._pending_intents = deque()
        self._last_signal_at_ms = {}
        self._latest_obi_by_market = {}

    def bind_dispatcher(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher
        super().bind_dispatcher(dispatcher)

    def bind_market_catalog(self, market_catalog: Any) -> None:
        self.market_catalog = market_catalog
        super().bind_market_catalog(market_catalog)

    def bind_clock(self, clock) -> None:
        self.clock_ms = clock
        super().bind_clock(clock)

    @property
    def latest_obi_by_market(self) -> dict[str, Decimal]:
        return dict(self._latest_obi_by_market)

    def drain_pending_intents(self) -> tuple[PriorityOrderContext, ...]:
        intents = tuple(self._pending_intents)
        self._pending_intents.clear()
        return intents

    def on_bbo_update(
        self,
        market_id: str,
        top_bids: Sequence[Any],
        top_asks: Sequence[Any],
    ) -> DispatchReceipt | PriorityOrderContext | None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            raise ValueError("market_id must be a non-empty string")

        bids = self._normalize_levels(top_bids)
        asks = self._normalize_levels(top_asks)
        if not bids or not asks:
            return None

        obi = self.calculate_obi(bids, asks, depth_levels=self.depth_levels)
        if obi is None:
            return None
        self._latest_obi_by_market[normalized_market_id] = obi

        if obi > self.threshold:
            side = "YES"
        elif obi < -self.threshold:
            side = "NO"
        else:
            return None

        timestamp_ms = int(self.clock_ms())
        dedup_key = (normalized_market_id, side)
        last_signal_at_ms = self._last_signal_at_ms.get(dedup_key)
        if last_signal_at_ms is not None and timestamp_ms - last_signal_at_ms < self.cooldown_ms:
            return None

        context = self._build_context(
            market_id=normalized_market_id,
            side=side,
            bids=bids,
            asks=asks,
            obi=obi,
        )
        self._last_signal_at_ms[dedup_key] = timestamp_ms

        if self.dispatcher is not None:
            return self.dispatcher.dispatch(context, timestamp_ms)

        self._pending_intents.append(context)
        return context

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        return None

    def on_tick(self) -> None:
        return None

    @classmethod
    def calculate_obi(
        cls,
        top_bids: Sequence[Any],
        top_asks: Sequence[Any],
        *,
        depth_levels: int = 3,
    ) -> Decimal | None:
        if depth_levels <= 0:
            raise ValueError("depth_levels must be strictly positive")

        bids = cls._normalize_levels(top_bids)[:depth_levels]
        asks = cls._normalize_levels(top_asks)[:depth_levels]
        if not bids or not asks:
            return None

        bid_volume = sum(level.size for level in bids)
        ask_volume = sum(level.size for level in asks)
        total_volume = bid_volume + ask_volume
        if total_volume <= Decimal("0"):
            return None

        return (bid_volume - ask_volume) / total_volume

    def _build_context(
        self,
        *,
        market_id: str,
        side: str,
        bids: Sequence[NormalizedBookLevel],
        asks: Sequence[NormalizedBookLevel],
        obi: Decimal,
    ) -> PriorityOrderContext:
        if side == "YES":
            aggressive_price = asks[0].price
            available_volume = sum(level.size for level in asks[: self.depth_levels])
        else:
            aggressive_price = _ONE - bids[0].price
            available_volume = sum(level.size for level in bids[: self.depth_levels])

        if aggressive_price <= Decimal("0"):
            raise ValueError("Derived aggressive price must be strictly positive")

        anchor_volume = min(self.order_size, available_volume)
        if anchor_volume <= Decimal("0"):
            raise ValueError("anchor_volume must be strictly positive")

        natural_capital = aggressive_price * anchor_volume
        max_capital = natural_capital
        if self.max_capital_per_order is not None:
            max_capital = min(self.max_capital_per_order, natural_capital)
            if max_capital <= Decimal("0"):
                raise ValueError("max_capital must be strictly positive")
            affordable_volume = max_capital / aggressive_price
            anchor_volume = min(anchor_volume, affordable_volume)
            if anchor_volume <= Decimal("0"):
                raise ValueError("anchor_volume must remain strictly positive after capital clamp")

        conviction_scalar = min(_ONE, max(Decimal("0"), abs(obi)))
        return PriorityOrderContext(
            market_id=market_id,
            side=side,
            signal_source=self.signal_source,
            conviction_scalar=conviction_scalar,
            target_price=aggressive_price,
            anchor_volume=anchor_volume,
            max_capital=aggressive_price * anchor_volume,
            signal_metadata={
                "strategy": "obi_scalper",
                "obi": str(obi),
                "depth_levels": self.depth_levels,
                "threshold": str(self.threshold),
                "book_pressure": "BUY" if side == "YES" else "SELL",
            },
        )

    @staticmethod
    def _normalize_levels(levels: Sequence[Any]) -> list[NormalizedBookLevel]:
        normalized: list[NormalizedBookLevel] = []
        for raw_level in levels:
            if raw_level is None:
                continue
            normalized_level = ObiScalper._normalize_level(raw_level)
            if normalized_level is None:
                continue
            normalized.append(normalized_level)
        return normalized

    @staticmethod
    def _normalize_level(raw_level: Any) -> NormalizedBookLevel | None:
        price: Any | None = None
        size: Any | None = None

        if isinstance(raw_level, dict):
            price = raw_level.get("price", raw_level.get("px", raw_level.get("rate")))
            size = raw_level.get("size", raw_level.get("qty", raw_level.get("quantity", raw_level.get("volume"))))
        elif isinstance(raw_level, (tuple, list)) and len(raw_level) >= 2:
            price, size = raw_level[0], raw_level[1]
        else:
            price = getattr(raw_level, "price", getattr(raw_level, "px", None))
            size = getattr(
                raw_level,
                "size",
                getattr(raw_level, "qty", getattr(raw_level, "quantity", getattr(raw_level, "volume", None))),
            )

        if price is None or size is None:
            return None

        normalized_price = ObiScalper._as_decimal(price, name="price")
        normalized_size = ObiScalper._as_decimal(size, name="size")
        if normalized_price <= Decimal("0") or normalized_size <= Decimal("0"):
            return None
        return NormalizedBookLevel(price=normalized_price, size=normalized_size)

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            decimal_value = Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value
