from __future__ import annotations

from decimal import Decimal

from src.signals.exhaustion_fader import ExhaustionFader


class _DispatcherProbe:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, context, dispatch_timestamp_ms: int):
        self.calls.append((context, dispatch_timestamp_ms))
        return None


def test_exhaustion_fader_posts_ask_after_buy_fomo_spike_with_flat_obi() -> None:
    now = {"value": 1000}
    dispatcher = _DispatcherProbe()
    strategy = ExhaustionFader(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        spike_threshold=Decimal("0.04"),
        order_size=Decimal("7"),
    )

    strategy.on_bbo_update(
        "market-1",
        top_bids=[{"price": "0.48", "size": "20"}, {"price": "0.47", "size": "20"}, {"price": "0.46", "size": "20"}],
        top_asks=[{"price": "0.50", "size": "20"}, {"price": "0.51", "size": "20"}, {"price": "0.52", "size": "20"}],
    )
    strategy.on_trade("market-1", {"side": "BUY", "size": 20.0, "price": 0.50, "timestamp_ms": 1000})

    now["value"] = 5000
    strategy.on_bbo_update(
        "market-1",
        top_bids=[{"price": "0.53", "size": "20"}, {"price": "0.52", "size": "20"}, {"price": "0.51", "size": "20"}],
        top_asks=[{"price": "0.55", "size": "20"}, {"price": "0.56", "size": "20"}, {"price": "0.57", "size": "20"}],
    )
    strategy.on_trade("market-1", {"side": "BUY", "size": 30.0, "price": 0.55, "timestamp_ms": 5000})

    assert len(dispatcher.calls) == 1
    context = dispatcher.calls[0][0]
    assert context.signal_metadata["quote_side"] == "ASK"
    assert context.signal_metadata["post_only"] is True
    assert context.signal_metadata["time_in_force"] == "GTC"
    assert context.target_price == Decimal("0.55")
    assert context.anchor_volume == Decimal("7")


def test_exhaustion_fader_posts_bid_after_sell_fomo_spike_with_flat_obi() -> None:
    now = {"value": 2000}
    dispatcher = _DispatcherProbe()
    strategy = ExhaustionFader(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        spike_threshold=Decimal("0.04"),
    )

    strategy.on_bbo_update(
        "market-2",
        top_bids=[{"price": "0.54", "size": "20"}, {"price": "0.53", "size": "20"}, {"price": "0.52", "size": "20"}],
        top_asks=[{"price": "0.56", "size": "20"}, {"price": "0.57", "size": "20"}, {"price": "0.58", "size": "20"}],
    )
    strategy.on_trade("market-2", {"side": "SELL", "size": 25.0, "price": 0.54, "timestamp_ms": 2000})

    now["value"] = 6500
    strategy.on_bbo_update(
        "market-2",
        top_bids=[{"price": "0.49", "size": "20"}, {"price": "0.48", "size": "20"}, {"price": "0.47", "size": "20"}],
        top_asks=[{"price": "0.51", "size": "20"}, {"price": "0.52", "size": "20"}, {"price": "0.53", "size": "20"}],
    )
    strategy.on_trade("market-2", {"side": "SELL", "size": 40.0, "price": 0.49, "timestamp_ms": 6500})

    assert len(dispatcher.calls) == 1
    context = dispatcher.calls[0][0]
    assert context.signal_metadata["quote_side"] == "BID"
    assert context.target_price == Decimal("0.49")


def test_exhaustion_fader_rejects_spike_when_obi_is_structurally_toxic() -> None:
    now = {"value": 1000}
    dispatcher = _DispatcherProbe()
    strategy = ExhaustionFader(dispatcher=dispatcher, clock=lambda: now["value"])

    strategy.on_bbo_update(
        "market-3",
        top_bids=[{"price": "0.48", "size": "20"}, {"price": "0.47", "size": "20"}, {"price": "0.46", "size": "20"}],
        top_asks=[{"price": "0.50", "size": "20"}, {"price": "0.51", "size": "20"}, {"price": "0.52", "size": "20"}],
    )
    strategy.on_trade("market-3", {"side": "BUY", "size": 20.0, "price": 0.50, "timestamp_ms": 1000})

    now["value"] = 5000
    strategy.on_bbo_update(
        "market-3",
        top_bids=[{"price": "0.53", "size": "200"}, {"price": "0.52", "size": "150"}, {"price": "0.51", "size": "120"}],
        top_asks=[{"price": "0.55", "size": "2"}, {"price": "0.56", "size": "2"}, {"price": "0.57", "size": "2"}],
    )
    strategy.on_trade("market-3", {"side": "BUY", "size": 30.0, "price": 0.55, "timestamp_ms": 5000})

    assert dispatcher.calls == []