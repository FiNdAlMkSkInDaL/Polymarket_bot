from __future__ import annotations

from decimal import Decimal

from src.signals.lead_lag_maker import LeadLagMaker


class _DispatcherProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def dispatch(self, context, dispatch_timestamp_ms: int):
        self.calls.append((context, dispatch_timestamp_ms))
        return None


def test_lead_lag_maker_quotes_secondary_bid_and_ask_under_normal_conditions() -> None:
    dispatcher = _DispatcherProbe()
    strategy = LeadLagMaker(
        primary_market_id="primary",
        secondary_market_id="secondary",
        dispatcher=dispatcher,
        clock_ms=lambda: 1_000,
        order_size=Decimal("5"),
    )

    strategy.on_bbo_update(
        "secondary",
        top_bids=[{"price": "0.49", "size": "10"}],
        top_asks=[{"price": "0.51", "size": "10"}],
    )

    assert len(dispatcher.calls) == 2
    bid_context = dispatcher.calls[0][0]
    ask_context = dispatcher.calls[1][0]
    assert bid_context.market_id == "secondary"
    assert ask_context.market_id == "secondary"
    assert bid_context.signal_metadata["quote_side"] == "BID"
    assert ask_context.signal_metadata["quote_side"] == "ASK"
    assert bid_context.signal_metadata["post_only"] is True
    assert ask_context.signal_metadata["time_in_force"] == "GTC"


def test_lead_lag_maker_cancels_secondary_quotes_on_primary_contagion() -> None:
    dispatcher = _DispatcherProbe()
    current_ts = {"value": 1_000}
    strategy = LeadLagMaker(
        primary_market_id="primary",
        secondary_market_id="secondary",
        dispatcher=dispatcher,
        clock_ms=lambda: current_ts["value"],
        contagion_move_cents=Decimal("2"),
    )

    strategy.on_bbo_update(
        "primary",
        top_bids=[{"price": "0.49", "size": "10"}],
        top_asks=[{"price": "0.51", "size": "10"}],
    )
    current_ts["value"] = 1_800
    strategy.on_bbo_update(
        "primary",
        top_bids=[{"price": "0.46", "size": "10"}],
        top_asks=[{"price": "0.48", "size": "10"}],
    )

    assert len(dispatcher.calls) == 1
    cancel_context = dispatcher.calls[0][0]
    assert cancel_context.market_id == "secondary"
    assert cancel_context.signal_metadata["action"] == "CANCEL_ALL"


def test_lead_lag_maker_holds_secondary_quotes_during_cooldown_then_resumes() -> None:
    dispatcher = _DispatcherProbe()
    current_ts = {"value": 1_000}
    strategy = LeadLagMaker(
        primary_market_id="primary",
        secondary_market_id="secondary",
        dispatcher=dispatcher,
        clock_ms=lambda: current_ts["value"],
        cooldown_ms=5_000,
    )

    strategy.on_bbo_update(
        "primary",
        top_bids=[{"price": "0.49", "size": "10"}],
        top_asks=[{"price": "0.51", "size": "10"}],
    )
    current_ts["value"] = 1_700
    strategy.on_bbo_update(
        "primary",
        top_bids=[{"price": "0.45", "size": "10"}],
        top_asks=[{"price": "0.47", "size": "10"}],
    )
    current_ts["value"] = 3_000
    strategy.on_bbo_update(
        "secondary",
        top_bids=[{"price": "0.39", "size": "10"}],
        top_asks=[{"price": "0.41", "size": "10"}],
    )
    current_ts["value"] = 6_800
    strategy.on_bbo_update(
        "secondary",
        top_bids=[{"price": "0.40", "size": "10"}],
        top_asks=[{"price": "0.42", "size": "10"}],
    )

    assert len(dispatcher.calls) == 3
    assert dispatcher.calls[0][0].signal_metadata["action"] == "CANCEL_ALL"
    resumed_bid = dispatcher.calls[1][0]
    resumed_ask = dispatcher.calls[2][0]
    assert resumed_bid.signal_metadata["quote_side"] == "BID"
    assert resumed_ask.signal_metadata["quote_side"] == "ASK"