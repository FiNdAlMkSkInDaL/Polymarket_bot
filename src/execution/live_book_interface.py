from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal


class LiveBestBidProvider(ABC):
    @abstractmethod
    def get_best_bid(self, market_id: str) -> Decimal | None:
        """
        Returns current best bid for market_id or None if unavailable.
        Must be O(1). Must never block. Must never raise.
        """

    @abstractmethod
    def get_best_bid_timestamp_ms(self, market_id: str) -> int | None:
        """
        Returns timestamp of last best-bid update or None if unavailable.
        """


class PaperBestBidProvider(LiveBestBidProvider):
    """
    Paper-mode implementation. Returns ask price as bid proxy.
    Used by paper adapters until live book state is wired.
    Documents the approximation explicitly at call sites.
    """

    def __init__(self, ask_proxy: dict[str, Decimal]):
        self._ask_proxy = ask_proxy

    def get_best_bid(self, market_id: str) -> Decimal | None:
        try:
            return self._ask_proxy.get(str(market_id).strip())
        except Exception:
            return None

    def get_best_bid_timestamp_ms(self, market_id: str) -> int | None:
        _ = market_id
        return None