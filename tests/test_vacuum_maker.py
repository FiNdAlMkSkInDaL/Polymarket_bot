from __future__ import annotations

from decimal import Decimal

from src.signals.vacuum_maker import VacuumMaker


class _DispatcherProbe:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, context, dispatch_timestamp_ms: int):
        self.calls.append((context, dispatch_timestamp_ms))
        return None


def test_vacuum_maker_quotes_immediately_on_first_wide_tick_in_vacuum() -> None:
    now = {"value": 1050}
    dispatcher = _DispatcherProbe()
    strategy = VacuumMaker(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        order_size=Decimal("10"),
        large_trade_min_size=Decimal("20"),
    )

    strategy.on_bbo_update(
        "market-1",
        top_bids=[{"price": "0.70", "size": "500"}, {"price": "0.69", "size": "400"}, {"price": "0.68", "size": "300"}],
        top_asks=[{"price": "0.76", "size": "5"}, {"price": "0.77", "size": "4"}, {"price": "0.78", "size": "3"}],
    )

    strategy.on_trade(
        "market-1",
        {"price": 0.77, "size": 25.0, "side": "BUY", "timestamp_ms": 1050},
    )

    strategy.on_tick()
    assert len(dispatcher.calls) == 2
    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["quotes_attempted"] == 1
    assert diagnostics["spread_wide_at_quote_time"] == 1
    assert diagnostics["ticks_scanned_in_vacuum"] == 1
    assert diagnostics["vacuum_aborted_max_window"] == 0
    bid_context = dispatcher.calls[0][0]
    ask_context = dispatcher.calls[1][0]
    assert bid_context.signal_metadata["quote_side"] == "BID"
    assert ask_context.signal_metadata["quote_side"] == "ASK"
    assert bid_context.signal_metadata["post_only"] is True
    assert ask_context.signal_metadata["time_in_force"] == "GTC"
    assert bid_context.target_price == Decimal("0.71")
    assert ask_context.target_price == Decimal("0.75")


def test_vacuum_maker_ignores_small_trade_while_crash_imminent() -> None:
    now = {"value": 2000}
    dispatcher = _DispatcherProbe()
    strategy = VacuumMaker(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        large_trade_min_size=Decimal("50"),
    )

    strategy.on_bbo_update(
        "market-2",
        top_bids=[{"price": "0.20", "size": "3"}, {"price": "0.19", "size": "2"}, {"price": "0.18", "size": "1"}],
        top_asks=[{"price": "0.23", "size": "120"}, {"price": "0.24", "size": "110"}, {"price": "0.25", "size": "100"}],
    )

    strategy.on_trade(
        "market-2",
        {"price": 0.20, "size": 10.0, "side": "SELL", "timestamp_ms": 2050},
    )
    now["value"] = 2600
    strategy.on_tick()

    assert dispatcher.calls == []


def test_vacuum_maker_requires_trade_side_to_match_crash_direction() -> None:
    now = {"value": 3000}
    dispatcher = _DispatcherProbe()
    strategy = VacuumMaker(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        large_trade_min_size=Decimal("20"),
    )

    strategy.on_bbo_update(
        "market-3",
        top_bids=[{"price": "0.70", "size": "500"}, {"price": "0.69", "size": "400"}, {"price": "0.68", "size": "300"}],
        top_asks=[{"price": "0.76", "size": "5"}, {"price": "0.77", "size": "4"}, {"price": "0.78", "size": "3"}],
    )

    strategy.on_trade(
        "market-3",
        {"price": 0.77, "size": 100.0, "side": "SELL", "timestamp_ms": 3050},
    )
    now["value"] = 3600
    strategy.on_tick()

    assert dispatcher.calls == []


def test_vacuum_maker_promotes_recent_trade_first_sequence_directly_to_vacuum() -> None:
    now = {"value": 4000}
    dispatcher = _DispatcherProbe()
    strategy = VacuumMaker(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        large_trade_min_size=Decimal("20"),
        recent_trade_memory_ms=500,
    )

    strategy.on_bbo_update(
        "market-4",
        top_bids=[{"price": "0.48", "size": "200"}, {"price": "0.47", "size": "150"}, {"price": "0.46", "size": "100"}],
        top_asks=[{"price": "0.52", "size": "30"}, {"price": "0.53", "size": "20"}, {"price": "0.54", "size": "10"}],
    )

    strategy.on_trade(
        "market-4",
        {"price": 0.54, "size": 25.0, "side": "BUY", "timestamp_ms": 4050},
    )

    now["value"] = 4100
    strategy.on_bbo_update(
        "market-4",
        top_bids=[{"price": "0.70", "size": "500"}, {"price": "0.69", "size": "400"}, {"price": "0.68", "size": "300"}],
        top_asks=[{"price": "0.76", "size": "5"}, {"price": "0.77", "size": "4"}, {"price": "0.78", "size": "3"}],
    )

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["crash_imminent_entries"] == 0
    assert diagnostics["vacuum_entries"] == 1
    assert diagnostics["recent_trade_confirmed_vacuums"] == 1
    strategy.on_tick()
    assert len(dispatcher.calls) == 2


def test_vacuum_maker_recent_trade_confirmation_expires() -> None:
    now = {"value": 5000}
    dispatcher = _DispatcherProbe()
    strategy = VacuumMaker(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        large_trade_min_size=Decimal("20"),
        recent_trade_memory_ms=100,
    )

    strategy.on_bbo_update(
        "market-5",
        top_bids=[{"price": "0.48", "size": "200"}, {"price": "0.47", "size": "150"}, {"price": "0.46", "size": "100"}],
        top_asks=[{"price": "0.52", "size": "90"}, {"price": "0.53", "size": "80"}, {"price": "0.54", "size": "70"}],
    )

    strategy.on_trade(
        "market-5",
        {"price": 0.54, "size": 25.0, "side": "BUY", "timestamp_ms": 5050},
    )

    now["value"] = 5300
    strategy.on_bbo_update(
        "market-5",
        top_bids=[{"price": "0.70", "size": "500"}, {"price": "0.69", "size": "400"}, {"price": "0.68", "size": "300"}],
        top_asks=[{"price": "0.76", "size": "5"}, {"price": "0.77", "size": "4"}, {"price": "0.78", "size": "3"}],
    )

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["crash_imminent_entries"] == 1
    assert diagnostics["vacuum_entries"] == 0
    assert diagnostics["recent_trade_confirmed_vacuums"] == 0

    strategy.on_trade(
        "market-5",
        {"price": 0.77, "size": 25.0, "side": "BUY", "timestamp_ms": 5310},
    )

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["vacuum_entries"] == 1


def test_vacuum_maker_waits_for_wide_tick_after_narrow_tick() -> None:
    now = {"value": 6000}
    dispatcher = _DispatcherProbe()
    strategy = VacuumMaker(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        large_trade_min_size=Decimal("20"),
        tick_size=Decimal("0.01"),
    )

    strategy.on_bbo_update(
        "market-6",
        top_bids=[{"price": "0.70", "size": "500"}, {"price": "0.69", "size": "400"}, {"price": "0.68", "size": "300"}],
        top_asks=[{"price": "0.76", "size": "5"}, {"price": "0.77", "size": "4"}, {"price": "0.78", "size": "3"}],
    )
    strategy.on_trade(
        "market-6",
        {"price": 0.77, "size": 25.0, "side": "BUY", "timestamp_ms": 6050},
    )
    strategy.on_bbo_update(
        "market-6",
        top_bids=[{"price": "0.74", "size": "100"}, {"price": "0.73", "size": "90"}, {"price": "0.72", "size": "80"}],
        top_asks=[{"price": "0.75", "size": "100"}, {"price": "0.76", "size": "90"}, {"price": "0.77", "size": "80"}],
    )

    now["value"] = 6051
    strategy.on_tick()

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["skipped_spread_too_tight"] == 1
    assert diagnostics["spread_wide_at_quote_time"] == 0
    assert diagnostics["quotes_attempted"] == 0
    assert diagnostics["ticks_scanned_in_vacuum"] == 1
    assert dispatcher.calls == []

    strategy.on_bbo_update(
        "market-6",
        top_bids=[{"price": "0.70", "size": "100"}, {"price": "0.69", "size": "90"}, {"price": "0.68", "size": "80"}],
        top_asks=[{"price": "0.76", "size": "100"}, {"price": "0.77", "size": "90"}, {"price": "0.78", "size": "80"}],
    )
    now["value"] = 6052
    strategy.on_tick()

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["spread_wide_at_quote_time"] == 1
    assert diagnostics["quotes_attempted"] == 1
    assert diagnostics["ticks_scanned_in_vacuum"] == 2
    assert len(dispatcher.calls) == 2
    bid_context = dispatcher.calls[0][0]
    ask_context = dispatcher.calls[1][0]
    assert bid_context.target_price == Decimal("0.71")
    assert ask_context.target_price == Decimal("0.75")


def test_vacuum_maker_aborts_when_no_wide_tick_arrives_before_max_window() -> None:
    now = {"value": 7000}
    dispatcher = _DispatcherProbe()
    strategy = VacuumMaker(
        dispatcher=dispatcher,
        clock=lambda: now["value"],
        large_trade_min_size=Decimal("20"),
        tick_size=Decimal("0.01"),
        max_vacuum_window_ms=100,
    )

    strategy.on_bbo_update(
        "market-7",
        top_bids=[{"price": "0.70", "size": "500"}, {"price": "0.69", "size": "400"}, {"price": "0.68", "size": "300"}],
        top_asks=[{"price": "0.76", "size": "5"}, {"price": "0.77", "size": "4"}, {"price": "0.78", "size": "3"}],
    )
    strategy.on_trade(
        "market-7",
        {"price": 0.77, "size": 25.0, "side": "BUY", "timestamp_ms": 7050},
    )
    strategy.on_bbo_update(
        "market-7",
        top_bids=[{"price": "0.75", "size": "100"}, {"price": "0.74", "size": "90"}, {"price": "0.73", "size": "80"}],
        top_asks=[{"price": "0.75", "size": "100"}, {"price": "0.76", "size": "90"}, {"price": "0.77", "size": "80"}],
    )

    now["value"] = 7100
    strategy.on_tick()
    assert dispatcher.calls == []

    now["value"] = 7151
    strategy.on_tick()

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["skipped_spread_too_tight"] == 2
    assert diagnostics["quotes_attempted"] == 0
    assert diagnostics["ticks_scanned_in_vacuum"] == 2
    assert diagnostics["vacuum_aborted_max_window"] == 1
    assert dispatcher.calls == []