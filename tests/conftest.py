"""
Shared pytest fixtures for the Polymarket bot test suite.

Factory helpers live in tests/helpers.py and are imported as
``from tests.helpers import ...`` by individual test modules.
"""

from __future__ import annotations

import os

import pytest

from src.data.ohlcv import OHLCVAggregator
from src.signals.panic_detector import PanicDetector
from src.trading.executor import OrderExecutor
from src.monitoring.trade_store import TradeStore


# ── Ensure paper mode is always on during tests ──────────────────────────
os.environ.setdefault("PAPER_MODE", "true")


# ── Fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture
def paper_executor():
    """OrderExecutor in paper mode."""
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
def trade_store(tmp_path):
    """TradeStore backed by a temporary SQLite database."""
    return TradeStore(tmp_path / "test_trades.db")


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
