from __future__ import annotations

import pytest

from src.signals.bayesian_arb import BayesianArbSignal, BayesianLeg
from src.signals.combinatorial_arb import ComboArbSignal, ComboLeg
from src.trading.executor import Order, OrderSide, OrderStatus
from src.trading.position_manager import ComboState, PositionManager, PositionState


class _Book:
    def __init__(self, best_bid: float, best_ask: float):
        self.best_bid = best_bid
        self.best_ask = best_ask


class _Executor:
    def __init__(self):
        self.paper_mode = False
        self.orders: list[Order] = []
        self.cancelled: list[str] = []

    async def place_limit_order(
        self,
        market_id,
        asset_id,
        side,
        price,
        size,
        *,
        post_only=False,
        fee_rate_bps=0,
        signal_fired_at=None,
    ):
        order = Order(
            order_id=f"ORD-{len(self.orders) + 1}",
            market_id=market_id,
            asset_id=asset_id,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.LIVE,
            post_only=post_only,
        )
        self.orders.append(order)
        return order

    async def cancel_order(self, order: Order) -> None:
        order.status = OrderStatus.CANCELLED
        self.cancelled.append(order.order_id)


def _signal() -> ComboArbSignal:
    maker = ComboLeg(
        market_id="MKT_MAKER",
        yes_token_id="YES_MAKER",
        no_token_id="NO_MAKER",
        best_bid=0.30,
        best_ask=0.34,
        target_price=0.31,
        target_shares=25.0,
        question="Maker",
        routing_reason="widest_spread spread_width=0.0400 next_spread_width=0.0200",
    )
    taker_a = ComboLeg(
        market_id="MKT_TAKER_A",
        yes_token_id="YES_TAKER_A",
        no_token_id="NO_TAKER_A",
        best_bid=0.28,
        best_ask=0.52,
        target_price=0.52,
        target_shares=25.0,
        question="Taker A",
    )
    taker_b = ComboLeg(
        market_id="MKT_TAKER_B",
        yes_token_id="YES_TAKER_B",
        no_token_id="NO_TAKER_B",
        best_bid=0.27,
        best_ask=0.53,
        target_price=0.53,
        target_shares=25.0,
        question="Taker B",
    )
    return ComboArbSignal(
        market_id="EVENT-1",
        cluster_event_id="EVENT-1",
        maker_leg=maker,
        taker_legs=[taker_a, taker_b],
        target_shares=25.0,
        total_collateral=34.0,
        edge_cents=6.0,
    )


@pytest.mark.asyncio
async def test_open_combo_position_quotes_only_maker_leg():
    executor = _Executor()
    pm = PositionManager(executor, max_open_positions=10)
    pm.set_wallet_balance(1000.0)

    combo = await pm.open_combo_position(_signal(), {})

    assert combo is not None
    assert combo.maker_market_id == "MKT_MAKER"
    assert combo.state == ComboState.ENTRY_PENDING
    assert len(executor.orders) == 1
    assert executor.orders[0].market_id == "MKT_MAKER"
    assert executor.orders[0].post_only is True
    assert len(combo.legs) == 3
    assert combo.legs["MKT_TAKER_A"].entry_order is None
    assert combo.legs["MKT_TAKER_B"].entry_order is None
    assert [pos.market_id for pos in pm.get_open_positions()] == ["MKT_MAKER"]


@pytest.mark.asyncio
async def test_maker_fill_triggers_aggressive_taker_sweep():
    executor = _Executor()
    pm = PositionManager(
        executor,
        max_open_positions=10,
        book_trackers={
            "YES_TAKER_A": _Book(best_bid=0.51, best_ask=0.52),
            "YES_TAKER_B": _Book(best_bid=0.52, best_ask=0.53),
        },
    )
    pm.set_wallet_balance(1000.0)

    combo = await pm.open_combo_position(_signal(), {})
    assert combo is not None

    maker_order = executor.orders[0]
    maker_order.status = OrderStatus.FILLED
    maker_order.filled_size = 25.0
    maker_order.filled_avg_price = 0.31

    handled = await pm.on_combo_order_update(maker_order, {combo.event_id: combo})

    assert handled is True
    assert combo.sweep_triggered is True
    assert len(executor.orders) == 3
    assert [order.post_only for order in executor.orders] == [True, False, False]
    assert executor.orders[1].price == 0.56
    assert executor.orders[2].price == 0.57
    assert combo.state == ComboState.PARTIAL_FILL
    assert len(pm.get_open_positions()) == 3
    assert combo.legs["MKT_TAKER_A"].state == PositionState.ENTRY_PENDING
    assert combo.legs["MKT_TAKER_A"].entry_order is not None


@pytest.mark.asyncio
async def test_abandon_combo_safely_cancels_unfilled_maker():
    executor = _Executor()
    pm = PositionManager(executor, max_open_positions=10)
    pm.set_wallet_balance(1000.0)

    combo = await pm.open_combo_position(_signal(), {})
    assert combo is not None

    await pm.abandon_combo(combo, reason="leg_timeout")

    assert combo.state == ComboState.ABANDONED
    assert executor.cancelled == [executor.orders[0].order_id]
    assert len(executor.orders) == 1
    assert all(pos.state == PositionState.CANCELLED for pos in combo.legs.values())


@pytest.mark.asyncio
async def test_hanging_taker_after_sweep_triggers_emergency_dump():
    executor = _Executor()
    books = {
        "YES_MAKER": _Book(best_bid=0.30, best_ask=0.31),
        "YES_TAKER_A": _Book(best_bid=0.51, best_ask=0.52),
        "YES_TAKER_B": _Book(best_bid=0.49, best_ask=0.80),
    }
    pm = PositionManager(
        executor,
        max_open_positions=10,
        book_trackers=books,
    )
    pm.set_wallet_balance(1000.0)

    combo = await pm.open_combo_position(_signal(), {})
    assert combo is not None

    maker_order = executor.orders[0]
    maker_order.status = OrderStatus.FILLED
    maker_order.filled_size = 25.0
    maker_order.filled_avg_price = 0.31
    await pm.on_combo_order_update(maker_order, {combo.event_id: combo})

    taker_a_order = combo.legs["MKT_TAKER_A"].entry_order
    assert taker_a_order is not None
    taker_a_order.status = OrderStatus.FILLED
    taker_a_order.filled_size = 25.0
    taker_a_order.filled_avg_price = 0.56
    await pm.on_combo_order_update(taker_a_order, {combo.event_id: combo})

    taker_b_pos = combo.legs["MKT_TAKER_B"]
    assert taker_b_pos.entry_order is not None
    assert taker_b_pos.state == PositionState.ENTRY_PENDING

    # Liquidity vanishes after the sweep is triggered, making the hedge
    # uneconomic and forcing the emergency dump path.
    books["YES_TAKER_B"].best_ask = 0.95

    await pm.abandon_combo(combo, reason="leg_timeout")

    assert combo.state == ComboState.ABANDONED
    assert taker_b_pos.entry_order.status == OrderStatus.CANCELLED
    assert taker_b_pos.exit_reason == "leg_timeout_spread_too_wide"
    assert executor.cancelled == [taker_b_pos.entry_order.order_id]

    maker_pos = combo.legs["MKT_MAKER"]
    taker_a_pos = combo.legs["MKT_TAKER_A"]
    assert maker_pos.state == PositionState.EXIT_PENDING
    assert taker_a_pos.state == PositionState.EXIT_PENDING
    assert maker_pos.exit_order is not None
    assert taker_a_pos.exit_order is not None
    assert maker_pos.exit_order.side == OrderSide.SELL
    assert taker_a_pos.exit_order.side == OrderSide.SELL
    assert maker_pos.exit_order.post_only is False
    assert taker_a_pos.exit_order.post_only is False
    assert maker_pos.exit_order.price == 0.30
    assert taker_a_pos.exit_order.price == 0.51
    assert len(executor.orders) == 5


def _bayesian_signal() -> BayesianArbSignal:
    maker = BayesianLeg(
        market_id="MKT_JOINT",
        asset_id="YES_JOINT",
        trade_side="YES",
        yes_token_id="YES_JOINT",
        no_token_id="NO_JOINT",
        best_bid=0.20,
        best_ask=0.21,
        target_price=0.20,
        target_shares=25.0,
        question="Joint",
        role="joint",
    )
    taker_a = BayesianLeg(
        market_id="MKT_A",
        asset_id="NO_A",
        trade_side="NO",
        yes_token_id="YES_A",
        no_token_id="NO_A",
        best_bid=0.31,
        best_ask=0.35,
        target_price=0.35,
        target_shares=25.0,
        question="Base A",
        role="base_a",
    )
    taker_b = BayesianLeg(
        market_id="MKT_B",
        asset_id="NO_B",
        trade_side="NO",
        yes_token_id="YES_B",
        no_token_id="NO_B",
        best_bid=0.36,
        best_ask=0.37,
        target_price=0.37,
        target_shares=25.0,
        question="Base B",
        role="base_b",
    )
    return BayesianArbSignal(
        market_id="REL-LOWER",
        cluster_event_id="REL-LOWER",
        maker_leg=maker,
        taker_legs=[taker_a, taker_b],
        target_shares=25.0,
        total_collateral=23.0,
        edge_cents=9.0,
        violation_type="lower",
    )


@pytest.mark.asyncio
async def test_combo_position_uses_trade_asset_ids_for_bayesian_no_legs():
    executor = _Executor()
    pm = PositionManager(
        executor,
        max_open_positions=10,
        book_trackers={
            "NO_A": _Book(best_bid=0.31, best_ask=0.35),
            "NO_B": _Book(best_bid=0.36, best_ask=0.37),
        },
    )
    pm.set_wallet_balance(1000.0)

    combo = await pm.open_combo_position(_bayesian_signal(), {})

    assert combo is not None
    assert executor.orders[0].asset_id == "YES_JOINT"

    maker_order = executor.orders[0]
    maker_order.status = OrderStatus.FILLED
    maker_order.filled_size = 25.0
    maker_order.filled_avg_price = 0.20

    handled = await pm.on_combo_order_update(maker_order, {combo.event_id: combo})

    assert handled is True
    assert [order.asset_id for order in executor.orders] == ["YES_JOINT", "NO_A", "NO_B"]
    assert combo.legs["MKT_A"].trade_side == "NO"
    assert combo.legs["MKT_B"].trade_side == "NO"