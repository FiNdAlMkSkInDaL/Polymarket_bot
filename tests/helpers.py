"""
Shared test helpers — factory functions for creating test objects.

Importable as `from tests.helpers import ...`
"""

from __future__ import annotations

import time

from src.data.ohlcv import OHLCVAggregator, OHLCVBar, BAR_INTERVAL
from src.data.websocket_client import TradeEvent
from src.signals.panic_detector import PanicSignal
from src.trading.take_profit import TakeProfitResult
from src.trading.position_manager import Position, PositionState


def make_trade(
    price: float,
    size: float,
    ts: float,
    asset_id: str = "YES_TOKEN",
    market_id: str = "MKT_TEST",
    is_yes: bool = True,
) -> TradeEvent:
    """Create a TradeEvent for testing."""
    return TradeEvent(
        timestamp=ts,
        market_id=market_id,
        asset_id=asset_id,
        side="buy",
        price=price,
        size=size,
        is_yes=is_yes,
    )


def build_bar_history(
    agg: OHLCVAggregator,
    prices: list[float],
    base_vol: float = 10.0,
    start_ts: float = 1000.0,
) -> float:
    """Feed trades to build completed bar history. Returns the next timestamp."""
    t = start_ts
    for p in prices:
        agg.on_trade(make_trade(p, base_vol, t, asset_id=agg.asset_id))
        t += BAR_INTERVAL + 0.1
        # Small closing tick to trigger bar finalisation
        agg.on_trade(make_trade(p + 0.001, 1.0, t, asset_id=agg.asset_id))
        t += 0.1
    return t


def make_position(
    pos_id: str,
    entry: float,
    exit_p: float,
    size: float = 10.0,
    reason: str = "target",
    whale: bool = False,
    market_id: str = "MKT_TEST",
) -> Position:
    """Create a closed Position for trade store tests."""
    pnl = round((exit_p - entry) * size * 100, 2)
    now = time.time()
    return Position(
        id=pos_id,
        market_id=market_id,
        no_asset_id="NO_TOKEN",
        state=PositionState.CLOSED,
        entry_price=entry,
        entry_size=size,
        entry_time=now - 120,
        target_price=exit_p,
        exit_price=exit_p,
        exit_time=now,
        exit_reason=reason,
        pnl_cents=pnl,
        tp_result=TakeProfitResult(
            entry_price=entry,
            target_price=exit_p,
            alpha=0.5,
            spread_cents=abs(exit_p - entry) * 100,
            viable=True,
        ),
        signal=PanicSignal(
            market_id=market_id,
            yes_asset_id="YES_TOKEN",
            no_asset_id="NO_TOKEN",
            yes_price=0.70,
            yes_vwap=0.50,
            zscore=2.5,
            volume_ratio=4.0,
            no_best_ask=entry + 0.01,
            whale_confluence=whale,
        ),
    )
