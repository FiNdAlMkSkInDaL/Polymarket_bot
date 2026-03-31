from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot import TradingBot
from src.data.ohlcv import OHLCVAggregator
from src.signals.ofi_momentum import OFIMomentumSignal
from src.signals.ofi_momentum import OFIMomentumDetector


@dataclass
class FakeMarketInfo:
    condition_id: str
    yes_token_id: str
    no_token_id: str
    accepting_orders: bool = True
    end_date: object | None = None
    event_id: str = "EVENT-1"


@dataclass
class FakeLevel:
    price: float
    size: float


@dataclass
class FakeSnapshot:
    best_bid: float
    best_ask: float
    timestamp: float
    ask_depth_usd: float = 50.0


class FakeBook:
    has_data = True
    book_depth_ratio = 2.0

    def __init__(
        self,
        best_bid: float,
        bid_size: float,
        best_ask: float,
        ask_size: float,
        timestamp_s: float,
        ask_depth_usd: float = 50.0,
        *,
        toxicity_buy: float = 0.82,
        toxicity_sell: float = 0.0,
        depth_evaporation: float = 0.24,
        sweep_ratio: float = 0.41,
    ) -> None:
        self._bid = FakeLevel(best_bid, bid_size)
        self._ask = FakeLevel(best_ask, ask_size)
        self._snapshot = FakeSnapshot(
            best_bid=best_bid,
            best_ask=best_ask,
            timestamp=timestamp_s,
            ask_depth_usd=ask_depth_usd,
        )
        self._toxicity_buy = toxicity_buy
        self._toxicity_sell = toxicity_sell
        self._depth_evaporation = depth_evaporation
        self._sweep_ratio = sweep_ratio

    def levels(self, side: str, n: int = 1):
        if side.lower() in ("bid", "buy"):
            return [self._bid][:n]
        return [self._ask][:n]

    def snapshot(self):
        return self._snapshot

    def toxicity_metrics(self, side: str = "BUY"):
        if side.upper() == "BUY":
            return {
                "toxicity_index": self._toxicity_buy,
                "toxicity_depth_evaporation": self._depth_evaporation,
                "toxicity_sweep_ratio": self._sweep_ratio,
            }
        return {
            "toxicity_index": self._toxicity_sell,
            "toxicity_depth_evaporation": self._depth_evaporation,
            "toxicity_sweep_ratio": self._sweep_ratio,
        }


@dataclass
class FakeSpreadScore:
    score: float = 75.0
    raw_spread_cents: float = 2.0


@pytest.mark.asyncio
async def test_l2_bbo_change_routes_ofi_signal_through_live_bot() -> None:
    bot = TradingBot(paper_mode=True)

    market = FakeMarketInfo(
        condition_id="MKT-OFI",
        yes_token_id="YES-1",
        no_token_id="NO-1",
    )
    bot._market_map[market.no_token_id] = market
    bot._no_aggs[market.no_token_id] = MagicMock()
    bot._book_trackers[market.no_token_id] = FakeBook(0.49, 95.0, 0.51, 5.0, 1.0)
    bot._l2_books[market.no_token_id] = MagicMock(is_reliable=True)
    bot._ofi_detectors[market.condition_id] = MagicMock(generate_signal=MagicMock(return_value=OFIMomentumSignal(
        market_id=market.condition_id,
        no_asset_id=market.no_token_id,
        no_best_ask=0.51,
        signal_source="ofi_momentum",
        direction="BUY",
        trade_flow_imbalance=0.35,
        tvi_multiplier=1.1,
        top_ask_size=5.0,
        rolling_vi=0.82,
        timestamp_ms=1000,
        window_ms=2000,
        toxicity_index=0.62,
    )))
    bot._live_orchestrator = MagicMock(on_ofi_signal=MagicMock())
    bot._orchestrator_health_monitor = MagicMock()
    bot._meta_controller = MagicMock(
        evaluate=MagicMock(return_value=MagicMock(vetoed=False, weight=1.0, veto_reason=None))
    )
    bot._ofi_lane_enabled = MagicMock(return_value=True)

    bot.lifecycle.is_tradeable = MagicMock(return_value=True)
    bot.lifecycle.is_cooled_down = MagicMock(return_value=True)
    bot.lifecycle.record_signal = MagicMock()
    bot.positions.is_stop_loss_cooled_down = MagicMock(return_value=True)
    bot._stop_loss_monitor = MagicMock(on_bbo_update=AsyncMock())
    bot._maker_monitor = None
    bot._on_panic_signal = AsyncMock()

    await bot._on_l2_bbo_change_inner(market.no_token_id, FakeSpreadScore())

    bot.lifecycle.record_signal.assert_called_once_with(market.condition_id)
    bot._live_orchestrator.on_ofi_signal.assert_called_once()
    bot._on_panic_signal.assert_not_awaited()


def test_top_toxicity_rankings_uses_active_l2_markets() -> None:
    bot = TradingBot(paper_mode=True)

    market_a = FakeMarketInfo(condition_id="MKT-A", yes_token_id="YES-A", no_token_id="NO-A")
    market_b = FakeMarketInfo(condition_id="MKT-B", yes_token_id="YES-B", no_token_id="NO-B")
    bot._markets = [market_a, market_b]
    bot._l2_active_set = {"MKT-A", "MKT-B"}
    bot._book_trackers[market_a.no_token_id] = FakeBook(
        0.49,
        95.0,
        0.51,
        5.0,
        1.0,
        toxicity_buy=0.82,
        toxicity_sell=0.10,
    )
    bot._book_trackers[market_b.no_token_id] = FakeBook(
        0.45,
        100.0,
        0.55,
        10.0,
        1.0,
        toxicity_buy=0.25,
        toxicity_sell=0.91,
        depth_evaporation=0.40,
        sweep_ratio=0.55,
    )

    rankings = bot._top_toxicity_rankings(limit=None)

    assert [row["condition_id"] for row in rankings] == ["MKT-B", "MKT-A"]
    assert rankings[0]["dominant_side"] == "SELL"
    assert rankings[0]["toxicity_index"] == pytest.approx(0.91, abs=1e-6)
    assert rankings[1]["dominant_side"] == "BUY"
    assert rankings[1]["toxicity_index"] == pytest.approx(0.82, abs=1e-6)


@pytest.mark.asyncio
async def test_on_panic_signal_suppresses_same_direction_ensemble_overlap() -> None:
    bot = TradingBot(paper_mode=True)
    market = FakeMarketInfo(condition_id="MKT-OFI", yes_token_id="YES-1", no_token_id="NO-1")
    bot._book_trackers[market.no_token_id] = FakeBook(0.49, 95.0, 0.51, 5.0, 1.0)
    bot.positions.open_position = AsyncMock()

    bot.ensemble_risk.register_position(
        position_id="LIVE-CONTAGION",
        market_id=market.condition_id,
        strategy_source="si10_contagion_arb",
        direction="NO",
    )

    signal = OFIMomentumSignal(
        market_id=market.condition_id,
        no_asset_id=market.no_token_id,
        no_best_ask=0.51,
        signal_source="ofi_momentum",
        direction="BUY",
    )
    no_agg = OHLCVAggregator(market.no_token_id)
    no_agg.rolling_vwap = 0.50

    await bot._on_panic_signal(
        signal,
        no_agg,
        market,
        signal_metadata={"signal_source": "ofi_momentum"},
    )

    bot.positions.open_position.assert_not_awaited()