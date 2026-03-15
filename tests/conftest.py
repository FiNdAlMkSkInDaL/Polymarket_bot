"""
Shared pytest fixtures for the Polymarket bot test suite.

Factory helpers live in tests/helpers.py and are imported as
``from tests.helpers import ...`` by individual test modules.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from src.data.ohlcv import OHLCVAggregator
from src.signals.panic_detector import PanicDetector
from src.trading.executor import OrderExecutor
from src.monitoring.trade_store import TradeStore


# ── Ensure paper mode is always on during tests ──────────────────────────
os.environ.setdefault("PAPER_MODE", "true")
os.environ.setdefault("DEPLOYMENT_ENV", "PAPER")


# ── Global: eliminate real-time delays in tests ──────────────────────────
_real_sleep = asyncio.sleep


@pytest.fixture(autouse=True)
def _fast_asyncio_sleep(monkeypatch):
    """Replace asyncio.sleep with an instant yield in every test.

    Tests that need to observe real elapsed time (or that mock sleep
    themselves) are unaffected because their own patches override this.
    """
    async def _instant_sleep(delay, *args, **kwargs):
        # Yield control without actually waiting.
        await _real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


# ── Global: block stray network calls ────────────────────────────────────
@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Prevent any test from accidentally making real HTTP/WS requests.

    httpx.AsyncClient.send and httpx.Client.send are replaced with
    mocks that raise immediately, so forgotten mocks surface as fast
    failures instead of 5-second timeouts.
    """
    import httpx

    def _blocked(*a, **kw):
        raise RuntimeError(
            "Unmocked network call detected in test — add a mock/patch"
        )

    async def _ablocked(*a, **kw):
        raise RuntimeError(
            "Unmocked async network call detected in test — add a mock/patch"
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _ablocked)
    monkeypatch.setattr(httpx.Client, "send", _blocked)


# ── Fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def paper_executor():
    """OrderExecutor in paper mode (session-scoped, stateless)."""
    return OrderExecutor(paper_mode=True)


@pytest.fixture
def yes_aggregator():
    """Fresh YES token aggregator."""
    return OHLCVAggregator("YES_TOKEN", lookback_minutes=10)


@pytest.fixture
def no_aggregator():
    """Fresh NO token aggregator."""
    return OHLCVAggregator("NO_TOKEN", lookback_minutes=10)


@pytest.fixture
def trade_store():
    """TradeStore backed by an in-memory SQLite database."""
    return TradeStore(":memory:")


@pytest.fixture
def detector(yes_aggregator, no_aggregator):
    """PanicDetector wired to fresh aggregators."""
    return PanicDetector(
        market_id="MKT_TEST",
        yes_asset_id="YES_TOKEN",
        no_asset_id="NO_TOKEN",
        yes_aggregator=yes_aggregator,
        no_aggregator=no_aggregator,
        zscore_threshold=2.0,
        volume_ratio_threshold=3.0,
    )
