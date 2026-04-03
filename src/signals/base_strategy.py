from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from src.execution.priority_context import PriorityOrderContext


class BaseStrategy(ABC):
    def __init__(
        self,
        *,
        dispatcher: Any | None = None,
        market_catalog: Any | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._market_catalog = market_catalog
        self._clock = clock or (lambda: 0)

    @property
    def dispatcher(self) -> Any:
        return self._dispatcher

    @property
    def market_catalog(self) -> Any:
        return self._market_catalog

    @property
    def current_timestamp_ms(self) -> int:
        return int(self._clock())

    def bind_dispatcher(self, dispatcher: Any) -> None:
        self._dispatcher = dispatcher

    def bind_market_catalog(self, market_catalog: Any) -> None:
        self._market_catalog = market_catalog

    def bind_clock(self, clock: Callable[[], int]) -> None:
        self._clock = clock

    def submit_order(self, context: PriorityOrderContext):
        if self._dispatcher is None:
            raise RuntimeError("Strategy dispatcher is not bound")
        return self._dispatcher.dispatch(context, self.current_timestamp_ms)

    @abstractmethod
    def on_bbo_update(
        self,
        market_id: str,
        top_bids: Sequence[dict[str, Any]],
        top_asks: Sequence[dict[str, Any]],
    ) -> None:
        ...

    @abstractmethod
    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_tick(self) -> None:
        ...