from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.execution.live_book_interface import LiveBestBidProvider
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider


class _StubTracker:
    def __init__(
        self,
        *,
        asset_id: str = "asset-1",
        best_bid: float = 0.47,
        best_ask: float = 0.49,
        bid_depth_usd: float = 120.0,
        ask_depth_usd: float = 150.0,
        bid_depth_ewma: float = 200.0,
        ask_depth_ewma: float = 220.0,
        timestamp: float = 1712345.678,
    ) -> None:
        self.asset_id = asset_id
        self._best_bid = best_bid
        self._best_ask = best_ask
        self._bid_depth_usd = bid_depth_usd
        self._ask_depth_usd = ask_depth_usd
        self._bid_depth_ewma = bid_depth_ewma
        self._ask_depth_ewma = ask_depth_ewma
        self._timestamp = timestamp

    @property
    def best_bid(self) -> float:
        return self._best_bid

    @property
    def best_ask(self) -> float:
        return self._best_ask

    def top_depths_usd(self) -> tuple[float, float]:
        return self._bid_depth_usd, self._ask_depth_usd

    def top_depth_ewma(self, side: str) -> float:
        return self._bid_depth_ewma if side == "bid" else self._ask_depth_ewma

    def snapshot(self):
        return SimpleNamespace(
            timestamp=self._timestamp,
            best_ask=self._best_ask,
            bid_depth_usd=self._bid_depth_usd,
            ask_depth_usd=self._ask_depth_usd,
        )


class _ExplodingBidTracker(_StubTracker):
    @property
    def best_bid(self) -> float:
        raise RuntimeError("boom")


class _ExplodingSnapshotTracker(_StubTracker):
    def snapshot(self):
        raise RuntimeError("boom")


def test_provider_satisfies_live_best_bid_provider_abc() -> None:
    assert isinstance(OrderbookBestBidProvider(_StubTracker()), LiveBestBidProvider)


def test_constructor_requires_tracker() -> None:
    with pytest.raises(ValueError, match="tracker"):
        OrderbookBestBidProvider(None)  # type: ignore[arg-type]


def test_get_best_bid_returns_decimal_for_matching_asset_id() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(best_bid=0.47))

    assert provider.get_best_bid("asset-1") == Decimal("0.47")


def test_get_best_bid_returns_none_for_mismatched_asset_id() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(asset_id="asset-1", best_bid=0.47))

    assert provider.get_best_bid("asset-2") is None


def test_get_best_bid_returns_none_for_zero_or_negative_bid() -> None:
    zero_provider = OrderbookBestBidProvider(_StubTracker(best_bid=0.0))
    negative_provider = OrderbookBestBidProvider(_StubTracker(best_bid=-0.1))

    assert zero_provider.get_best_bid("asset-1") is None
    assert negative_provider.get_best_bid("asset-1") is None


def test_get_best_bid_swallows_tracker_exceptions() -> None:
    provider = OrderbookBestBidProvider(_ExplodingBidTracker())

    assert provider.get_best_bid("asset-1") is None


def test_get_best_bid_timestamp_ms_converts_seconds_to_milliseconds() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(timestamp=1712345.678))

    assert provider.get_best_bid_timestamp_ms("asset-1") == 1712345678


def test_get_best_bid_timestamp_ms_returns_none_for_invalid_or_missing_timestamp() -> None:
    zero_provider = OrderbookBestBidProvider(_StubTracker(timestamp=0.0))
    negative_provider = OrderbookBestBidProvider(_StubTracker(timestamp=-1.0))

    assert zero_provider.get_best_bid_timestamp_ms("asset-1") is None
    assert negative_provider.get_best_bid_timestamp_ms("asset-1") is None


def test_get_best_bid_timestamp_ms_preserves_millisecond_timestamps() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(timestamp=1712345678901.0))

    assert provider.get_best_bid_timestamp_ms("asset-1") == 1712345678901


def test_get_best_bid_timestamp_ms_returns_none_for_mismatched_asset_id() -> None:
    provider = OrderbookBestBidProvider(_StubTracker())

    assert provider.get_best_bid_timestamp_ms("other-asset") is None


def test_get_best_bid_timestamp_ms_swallows_snapshot_exceptions() -> None:
    provider = OrderbookBestBidProvider(_ExplodingSnapshotTracker())

    assert provider.get_best_bid_timestamp_ms("asset-1") is None


def test_get_best_ask_returns_decimal_for_matching_asset_id() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(best_ask=0.49))

    assert provider.get_best_ask("asset-1") == Decimal("0.49")


def test_get_spread_returns_decimal_difference_between_best_ask_and_bid() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(best_bid=0.47, best_ask=0.49))

    assert provider.get_spread("asset-1") == Decimal("0.02")


def test_get_top_depth_reads_current_depth_for_each_side() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(bid_depth_usd=120.0, ask_depth_usd=150.0))

    assert provider.get_top_depth("asset-1", "bid") == Decimal("120.0")
    assert provider.get_top_depth("asset-1", "ask") == Decimal("150.0")


def test_get_top_depth_ewma_prefers_tracker_ewma_over_current_depth() -> None:
    provider = OrderbookBestBidProvider(_StubTracker(bid_depth_usd=120.0, bid_depth_ewma=200.0))

    assert provider.get_top_depth_ewma("asset-1", "bid") == Decimal("200.0")