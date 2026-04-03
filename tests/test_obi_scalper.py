from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.execution.priority_context import PriorityOrderContext
from src.signals.obi_scalper import ObiScalper


@dataclass
class _StubReceipt:
    context: PriorityOrderContext


class _StubDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[PriorityOrderContext, int]] = []

    def dispatch(self, context: PriorityOrderContext, dispatch_timestamp_ms: int):
        self.calls.append((context, dispatch_timestamp_ms))
        return _StubReceipt(context=context)


def test_obi_scalper_dispatches_yes_on_extreme_buy_pressure() -> None:
    dispatcher = _StubDispatcher()
    strategy = ObiScalper(
        dispatcher=dispatcher,
        clock_ms=lambda: 1234,
        order_size=Decimal("4"),
    )

    receipt = strategy.on_bbo_update(
        "market-1",
        top_bids=[(Decimal("0.44"), Decimal("60")), (Decimal("0.43"), Decimal("40")), (Decimal("0.42"), Decimal("20"))],
        top_asks=[(Decimal("0.46"), Decimal("2")), (Decimal("0.47"), Decimal("2")), (Decimal("0.48"), Decimal("4"))],
    )

    assert isinstance(receipt, _StubReceipt)
    assert len(dispatcher.calls) == 1
    context, timestamp_ms = dispatcher.calls[0]
    assert timestamp_ms == 1234
    assert context.side == "YES"
    assert context.signal_source == "MANUAL"
    assert context.target_price == Decimal("0.46")
    assert context.anchor_volume == Decimal("4")
    assert context.max_capital == Decimal("1.84")
    assert Decimal(context.signal_metadata["obi"]) > Decimal("0.85")


def test_obi_scalper_queues_no_intent_on_extreme_sell_pressure() -> None:
    strategy = ObiScalper(
        clock_ms=lambda: 2000,
        order_size=Decimal("6"),
    )

    context = strategy.on_bbo_update(
        "market-2",
        top_bids=[{"price": "0.39", "size": "2"}, {"price": "0.38", "size": "2"}, {"price": "0.37", "size": "1"}],
        top_asks=[{"price": "0.41", "size": "40"}, {"price": "0.42", "size": "30"}, {"price": "0.43", "size": "20"}],
    )

    assert isinstance(context, PriorityOrderContext)
    assert context.side == "NO"
    assert context.target_price == Decimal("0.61")
    assert context.anchor_volume == Decimal("5")
    drained = strategy.drain_pending_intents()
    assert drained == (context,)


def test_obi_scalper_ignores_neutral_imbalance() -> None:
    strategy = ObiScalper(clock_ms=lambda: 3000)

    result = strategy.on_bbo_update(
        "market-3",
        top_bids=[(0.49, 10), (0.48, 10), (0.47, 10)],
        top_asks=[(0.51, 10), (0.52, 10), (0.53, 10)],
    )

    assert result is None
    assert strategy.drain_pending_intents() == ()


def test_calculate_obi_uses_top_three_levels_only() -> None:
    obi = ObiScalper.calculate_obi(
        top_bids=[(1, 90), (0.9, 5), (0.8, 5), (0.7, 10_000)],
        top_asks=[(1.1, 1), (1.2, 1), (1.3, 1), (1.4, 10_000)],
        depth_levels=3,
    )

    assert obi == Decimal("97") / Decimal("103")
