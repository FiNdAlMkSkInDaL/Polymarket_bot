from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.backtest.strategy import ContagionReplayAdapter
from src.signals.contagion_arb import ContagionArbSignal


def test_contagion_replay_adapter_opens_position_for_signal() -> None:
    adapter = ContagionReplayAdapter(
        market_configs=[
            {
                "market_id": "LEADER",
                "yes_asset_id": "YES-L",
                "no_asset_id": "NO-L",
                "tags": "crypto",
            },
            {
                "market_id": "LAGGER",
                "yes_asset_id": "YES-G",
                "no_asset_id": "NO-G",
                "tags": "crypto",
            },
        ]
    )
    engine = MagicMock()
    engine.config = SimpleNamespace(latency_ms=0.0)
    engine.submit_order.return_value = SimpleNamespace(order_id="SIM-1")
    engine.matching_engine.activate_pending_orders.return_value = []
    engine.matching_engine.on_book_update = MagicMock()
    engine._process_fills = MagicMock()
    adapter.engine = engine
    adapter.on_init()

    book = adapter._books["YES-G"]
    book.apply_event(
        {
            "event_type": "l2_snapshot",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.51", "size": "100"}],
        },
        time.time(),
    )
    adapter._yes_aggs["YES-G"].last_trade_time = time.time()

    signal = ContagionArbSignal(
        leading_market_id="LEADER",
        lagging_market_id="LAGGER",
        lagging_asset_id="YES-G",
        direction="buy_yes",
        implied_probability=0.60,
        lagging_market_price=0.54,
        confidence=0.85,
        correlation=0.72,
        thematic_group="crypto",
        toxicity_percentile=0.80,
        leader_toxicity=0.91,
        leader_price_shift=0.03,
        expected_probability_shift=0.025,
        timestamp=time.time(),
        score=0.9,
        metadata={},
    )

    adapter._open_contagion_position(signal, time.time())

    engine.submit_order.assert_called_once()
    assert engine.submit_order.call_args.kwargs["asset_id"] == "YES-G"
    engine.matching_engine.activate_pending_orders.assert_called_once()