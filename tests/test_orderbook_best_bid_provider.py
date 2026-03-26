from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.execution.live_book_interface import LiveBestBidProvider
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider


class _StubTracker:
    def __init__(self, *, asset_id: str = "asset-1", best_bid: float = 0.47, timestamp: float = 1712345.678) -> None:
        self.asset_id = asset_id
        self._best_bid = best_bid
        self._timestamp = timestamp

    @property
    def best_bid(self) -> float:
        return self._best_bid

    def snapshot(self):
        return SimpleNamespace(timestamp=self._timestamp)


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