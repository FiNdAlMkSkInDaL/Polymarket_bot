from __future__ import annotations

from decimal import Decimal

from src.signals.wall_jumper import WallJumper


class _DispatcherProbe:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, context, dispatch_timestamp_ms: int):
        self.calls.append((context, dispatch_timestamp_ms))
        return None


def test_wall_jumper_penny_jumps_large_bid_wall() -> None:
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: 1000, order_size=Decimal("25"), wall_age_ms=0)

    strategy.on_bbo_update(
        "market-1",
        top_bids=[
            {"price": "0.20", "size": "600000"},
            {"price": "0.19", "size": "2000"},
            {"price": "0.18", "size": "1800"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "900"},
            {"price": "0.27", "size": "1100"},
        ],
    )

    assert len(dispatcher.calls) == 1
    context = dispatcher.calls[0][0]
    assert context.target_price == Decimal("0.21")
    assert context.anchor_volume == Decimal("25")
    assert context.signal_metadata["quote_side"] == "BID"
    assert context.signal_metadata["post_only"] is True
    assert context.signal_metadata["time_in_force"] == "GTC"
    assert context.signal_metadata["wall_side"] == "BID"
    assert context.signal_metadata["wall_size_usd"] == "120000.00"
    assert Decimal(context.signal_metadata["price_level_vs_mid_ticks"]) >= Decimal("2")


def test_wall_jumper_penny_jumps_large_ask_wall() -> None:
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: 1000, wall_age_ms=0)

    strategy.on_bbo_update(
        "market-2",
        top_bids=[
            {"price": "0.70", "size": "1200"},
            {"price": "0.69", "size": "1000"},
            {"price": "0.68", "size": "900"},
        ],
        top_asks=[
            {"price": "0.75", "size": "150000"},
            {"price": "0.76", "size": "200"},
            {"price": "0.77", "size": "180"},
        ],
    )

    assert len(dispatcher.calls) == 1
    context = dispatcher.calls[0][0]
    assert context.target_price == Decimal("0.74")
    assert context.signal_metadata["quote_side"] == "ASK"
    assert context.signal_metadata["wall_side"] == "ASK"
    assert context.signal_metadata["wall_size_usd"] == "112500.00"


def test_wall_jumper_skips_quote_when_jump_would_cross_spread() -> None:
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: 1000, wall_age_ms=0)

    strategy.on_bbo_update(
        "market-3",
        top_bids=[
            {"price": "0.60", "size": "12000"},
            {"price": "0.59", "size": "200"},
            {"price": "0.58", "size": "180"},
        ],
        top_asks=[
            {"price": "0.61", "size": "1000"},
            {"price": "0.62", "size": "1000"},
            {"price": "0.63", "size": "1000"},
        ],
    )

    assert dispatcher.calls == []


def test_wall_jumper_cancels_when_tracked_wall_loses_half_its_depth() -> None:
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: 1000, wall_age_ms=0)

    strategy.on_bbo_update(
        "market-4",
        top_bids=[
            {"price": "0.20", "size": "600000"},
            {"price": "0.19", "size": "300"},
            {"price": "0.18", "size": "250"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "950"},
            {"price": "0.27", "size": "900"},
        ],
    )
    strategy.on_bbo_update(
        "market-4",
        top_bids=[
            {"price": "0.20", "size": "290000"},
            {"price": "0.19", "size": "300"},
            {"price": "0.18", "size": "250"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "950"},
            {"price": "0.27", "size": "900"},
        ],
    )

    assert len(dispatcher.calls) == 2
    cancel_context = dispatcher.calls[1][0]
    assert cancel_context.signal_metadata["action"] == "CANCEL_ALL"
    assert cancel_context.signal_metadata["wall_price"] == "0.20"
    assert cancel_context.signal_metadata["current_wall_size"] == "290000"


def test_wall_jumper_reports_jump_and_cancel_diagnostics() -> None:
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: 1000, wall_age_ms=0)

    strategy.on_bbo_update(
        "market-5",
        top_bids=[
            {"price": "0.20", "size": "600000"},
            {"price": "0.19", "size": "300"},
            {"price": "0.18", "size": "250"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "1000"},
            {"price": "0.27", "size": "1000"},
        ],
    )
    strategy.on_bbo_update(
        "market-5",
        top_bids=[
            {"price": "0.20", "size": "290000"},
            {"price": "0.19", "size": "300"},
            {"price": "0.18", "size": "250"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "1000"},
            {"price": "0.27", "size": "1000"},
        ],
    )

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics == {
        "walls_identified": 1,
        "walls_aged_past_threshold": 1,
        "wall_age_ms_threshold": 0,
        "min_distance_from_mid_ticks": "2",
        "min_structural_wall_size_usd": "100000",
        "jump_quotes_emitted": 1,
        "cancel_all_triggered": 1,
    }


def test_wall_jumper_waits_until_wall_ages_past_threshold() -> None:
    now = {"value": 1_000}
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: now["value"], wall_age_ms=30_000)

    strategy.on_bbo_update(
        "market-6",
        top_bids=[
            {"price": "0.20", "size": "600000"},
            {"price": "0.19", "size": "200"},
            {"price": "0.18", "size": "180"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "900"},
            {"price": "0.27", "size": "1100"},
        ],
    )

    assert dispatcher.calls == []

    now["value"] = 31_000
    strategy.on_bbo_update(
        "market-6",
        top_bids=[
            {"price": "0.20", "size": "600000"},
            {"price": "0.19", "size": "200"},
            {"price": "0.18", "size": "180"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "900"},
            {"price": "0.27", "size": "1100"},
        ],
    )

    assert len(dispatcher.calls) == 1
    context = dispatcher.calls[0][0]
    assert context.signal_metadata["wall_age_ms"] == 30000

    diagnostics = strategy.diagnostics_snapshot()
    assert diagnostics["walls_identified"] == 1
    assert diagnostics["walls_aged_past_threshold"] == 1
    assert diagnostics["jump_quotes_emitted"] == 1


def test_wall_jumper_rejects_near_mid_wall_even_when_ratio_is_large() -> None:
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: 1000, wall_age_ms=0)

    strategy.on_bbo_update(
        "market-7",
        top_bids=[
            {"price": "0.21", "size": "600000"},
            {"price": "0.20", "size": "200"},
            {"price": "0.19", "size": "180"},
        ],
        top_asks=[
            {"price": "0.22", "size": "1000"},
            {"price": "0.23", "size": "900"},
            {"price": "0.24", "size": "1100"},
        ],
    )

    assert dispatcher.calls == []


def test_wall_jumper_rejects_wall_below_structural_notional_gate() -> None:
    dispatcher = _DispatcherProbe()
    strategy = WallJumper(dispatcher=dispatcher, clock_ms=lambda: 1000, wall_age_ms=0)

    strategy.on_bbo_update(
        "market-8",
        top_bids=[
            {"price": "0.20", "size": "400000"},
            {"price": "0.19", "size": "200"},
            {"price": "0.18", "size": "180"},
        ],
        top_asks=[
            {"price": "0.25", "size": "1000"},
            {"price": "0.26", "size": "900"},
            {"price": "0.27", "size": "1100"},
        ],
    )

    assert dispatcher.calls == []