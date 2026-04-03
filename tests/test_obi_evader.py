from __future__ import annotations

from decimal import Decimal

from src.signals.obi_evader import ObiEvader
from src.signals.obi_evader_v2 import ObiEvaderV2


class _DispatcherProbe:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, context, dispatch_timestamp_ms: int):
        self.calls.append((context, dispatch_timestamp_ms))
        return None


def test_obi_evader_quotes_bid_and_ask_in_safe_zone() -> None:
    dispatcher = _DispatcherProbe()
    strategy = ObiEvader(dispatcher=dispatcher, clock_ms=lambda: 1000, order_size=Decimal("5"))

    strategy.on_bbo_update(
        "market-1",
        top_bids=[{"price": "0.49", "size": "20"}, {"price": "0.48", "size": "15"}, {"price": "0.47", "size": "10"}],
        top_asks=[{"price": "0.51", "size": "20"}, {"price": "0.52", "size": "15"}, {"price": "0.53", "size": "10"}],
    )

    assert len(dispatcher.calls) == 2
    bid_context = dispatcher.calls[0][0]
    ask_context = dispatcher.calls[1][0]
    assert bid_context.signal_metadata["quote_side"] == "BID"
    assert ask_context.signal_metadata["quote_side"] == "ASK"
    assert bid_context.signal_metadata["post_only"] is True
    assert ask_context.signal_metadata["time_in_force"] == "GTC"


def test_obi_evader_cancels_quotes_in_toxic_zone() -> None:
    dispatcher = _DispatcherProbe()
    strategy = ObiEvader(dispatcher=dispatcher, clock_ms=lambda: 2000)

    strategy.on_bbo_update(
        "market-2",
        top_bids=[{"price": "0.60", "size": "80"}, {"price": "0.59", "size": "50"}, {"price": "0.58", "size": "40"}],
        top_asks=[{"price": "0.61", "size": "2"}, {"price": "0.62", "size": "2"}, {"price": "0.63", "size": "2"}],
    )

    assert len(dispatcher.calls) == 1
    cancel_context = dispatcher.calls[0][0]
    assert cancel_context.signal_metadata["action"] == "CANCEL_ALL"


def test_obi_evader_v2_blocks_quotes_when_spread_not_wide_enough() -> None:
    dispatcher = _DispatcherProbe()
    strategy = ObiEvaderV2(dispatcher=dispatcher, clock_ms=lambda: 3000, order_size=Decimal("5"))

    strategy.on_bbo_update(
        "market-3",
        top_bids=[{"price": "0.49", "size": "20"}, {"price": "0.48", "size": "15"}, {"price": "0.47", "size": "10"}],
        top_asks=[{"price": "0.51", "size": "20"}, {"price": "0.52", "size": "15"}, {"price": "0.53", "size": "10"}],
    )

    assert dispatcher.calls == []


def test_obi_evader_v2_quotes_bid_and_ask_when_spread_exceeds_gate() -> None:
    dispatcher = _DispatcherProbe()
    strategy = ObiEvaderV2(dispatcher=dispatcher, clock_ms=lambda: 4000, order_size=Decimal("5"))

    strategy.on_bbo_update(
        "market-4",
        top_bids=[{"price": "0.48", "size": "20"}, {"price": "0.47", "size": "15"}, {"price": "0.46", "size": "10"}],
        top_asks=[{"price": "0.505", "size": "20"}, {"price": "0.515", "size": "15"}, {"price": "0.525", "size": "10"}],
    )

    assert len(dispatcher.calls) == 2
    bid_context = dispatcher.calls[0][0]
    ask_context = dispatcher.calls[1][0]
    assert bid_context.signal_metadata["quote_side"] == "BID"
    assert ask_context.signal_metadata["quote_side"] == "ASK"
    assert Decimal(bid_context.signal_metadata["spread_cents"]) > Decimal("2.0")


def test_obi_evader_v2_cancels_quotes_in_toxic_zone_even_when_spread_is_tight() -> None:
    dispatcher = _DispatcherProbe()
    strategy = ObiEvaderV2(dispatcher=dispatcher, clock_ms=lambda: 5000)

    strategy.on_bbo_update(
        "market-5",
        top_bids=[{"price": "0.60", "size": "80"}, {"price": "0.59", "size": "50"}, {"price": "0.58", "size": "40"}],
        top_asks=[{"price": "0.61", "size": "2"}, {"price": "0.62", "size": "2"}, {"price": "0.63", "size": "2"}],
    )

    assert len(dispatcher.calls) == 1
    cancel_context = dispatcher.calls[0][0]
    assert cancel_context.signal_metadata["action"] == "CANCEL_ALL"