from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot import TradingBot
from src.signals.contagion_arb import ContagionArbSignal


@dataclass
class FakeMarketInfo:
    condition_id: str
    yes_token_id: str
    no_token_id: str
    question: str = "Will BTC rally?"
    accepting_orders: bool = True
    end_date: object | None = None
    event_id: str = "EVENT-1"


@dataclass
class FakeSnapshot:
    best_bid: float
    best_ask: float
    ask_depth_usd: float


class FakeBook:
    has_data = True

    def __init__(self, best_bid: float, best_ask: float, ask_depth_usd: float = 50.0) -> None:
        self._snapshot = FakeSnapshot(best_bid=best_bid, best_ask=best_ask, ask_depth_usd=ask_depth_usd)

    def snapshot(self):
        return self._snapshot


def _signal(market: FakeMarketInfo, direction: str = "buy_yes") -> ContagionArbSignal:
    return ContagionArbSignal(
        leading_market_id="LEADER",
        lagging_market_id=market.condition_id,
        lagging_asset_id=market.yes_token_id if direction == "buy_yes" else market.no_token_id,
        direction=direction,
        implied_probability=0.64,
        lagging_market_price=0.58,
        confidence=0.82,
        correlation=0.71,
        thematic_group="crypto",
        toxicity_percentile=0.80,
        leader_toxicity=0.93,
        leader_price_shift=0.03,
        expected_probability_shift=0.025,
        timestamp=time.time(),
        score=0.9,
        is_shadow=False,
        metadata={},
    )


@pytest.mark.asyncio
async def test_contagion_live_gate_suppresses_wide_lagger_spread() -> None:
    bot = TradingBot(paper_mode=True)
    bot.telegram.notify_contagion_matrix = AsyncMock()
    bot.lifecycle.is_tradeable = MagicMock(return_value=True)
    bot.lifecycle.is_cooled_down = MagicMock(return_value=True)
    bot.positions.is_stop_loss_cooled_down = MagicMock(return_value=True)
    bot.positions.open_rpe_position = AsyncMock()

    market = FakeMarketInfo(condition_id="MKT-C", yes_token_id="YES-C", no_token_id="NO-C")
    bot._book_trackers[market.yes_token_id] = FakeBook(0.50, 0.53)
    bot._yes_aggs[market.yes_token_id] = MagicMock(current_price=0.52, last_trade_time=time.time())

    await bot._on_contagion_signal(_signal(market), market)

    bot.positions.open_rpe_position.assert_not_awaited()
    assert bot._recent_contagion_matrix[0]["suppression_reason"] == "lagger_spread_wide"


@pytest.mark.asyncio
async def test_contagion_live_gate_suppresses_stale_lagger_trade() -> None:
    bot = TradingBot(paper_mode=True)
    bot.telegram.notify_contagion_matrix = AsyncMock()
    bot.lifecycle.is_tradeable = MagicMock(return_value=True)
    bot.lifecycle.is_cooled_down = MagicMock(return_value=True)
    bot.positions.is_stop_loss_cooled_down = MagicMock(return_value=True)
    bot.positions.open_rpe_position = AsyncMock()

    market = FakeMarketInfo(condition_id="MKT-S", yes_token_id="YES-S", no_token_id="NO-S")
    bot._book_trackers[market.yes_token_id] = FakeBook(0.50, 0.51)
    bot._yes_aggs[market.yes_token_id] = MagicMock(current_price=0.51, last_trade_time=time.time() - 600.0)

    await bot._on_contagion_signal(_signal(market), market)

    bot.positions.open_rpe_position.assert_not_awaited()
    assert bot._recent_contagion_matrix[0]["suppression_reason"] == "lagger_trade_stale"


@pytest.mark.asyncio
async def test_contagion_live_gate_suppresses_same_direction_ensemble_overlap() -> None:
    bot = TradingBot(paper_mode=True)
    bot.telegram.notify_contagion_matrix = AsyncMock()
    bot.lifecycle.is_tradeable = MagicMock(return_value=True)
    bot.lifecycle.is_cooled_down = MagicMock(return_value=True)
    bot.positions.is_stop_loss_cooled_down = MagicMock(return_value=True)
    bot.positions.open_rpe_position = AsyncMock()

    market = FakeMarketInfo(condition_id="MKT-E", yes_token_id="YES-E", no_token_id="NO-E")
    bot._book_trackers[market.yes_token_id] = FakeBook(0.50, 0.51)
    bot._yes_aggs[market.yes_token_id] = MagicMock(current_price=0.51, last_trade_time=time.time())
    bot.ensemble_risk.register_position(
        position_id="LIVE-OFI",
        market_id=market.condition_id,
        strategy_source="ofi_momentum",
        direction="YES",
    )

    await bot._on_contagion_signal(_signal(market, direction="buy_yes"), market)

    bot.positions.open_rpe_position.assert_not_awaited()