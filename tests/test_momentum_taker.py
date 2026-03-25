from __future__ import annotations

import random
import time
from dataclasses import dataclass

import pytest

from src.execution.momentum_taker import draw_stochastic_momentum_bracket
from src.signals.ofi_momentum import OFIMomentumSignal
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import PositionManager, PositionState


@dataclass
class _FakeLevel:
    price: float
    size: float


class _FakeSnapshot:
    def __init__(self, bid: float, ask: float):
        self.best_bid = bid
        self.best_ask = ask
        self.mid_price = (bid + ask) / 2.0


class _FakeBook:
    def __init__(
        self,
        bid: float = 0.49,
        ask: float = 0.50,
        buy_toxicity: float = 0.82,
        sell_toxicity: float = 0.31,
    ):
        self.has_data = True
        self.is_reliable = True
        self.best_bid = bid
        self.best_ask = ask
        self._buy_toxicity = buy_toxicity
        self._sell_toxicity = sell_toxicity
        self._snapshot = _FakeSnapshot(bid, ask)
        self._asks = [_FakeLevel(ask, 500.0), _FakeLevel(round(ask + 0.01, 2), 500.0)]
        self._bids = [_FakeLevel(bid, 500.0), _FakeLevel(round(bid - 0.01, 2), 500.0)]

    @property
    def book_depth_ratio(self) -> float:
        return 1.0

    def snapshot(self):
        return self._snapshot

    def levels(self, side: str, depth: int):
        del depth
        return self._asks if side == "ask" else self._bids

    def toxicity_metrics(self, side: str = "BUY"):
        toxicity = self._buy_toxicity if side.upper() == "BUY" else self._sell_toxicity
        return {
            "toxicity_index": toxicity,
            "toxicity_depth_evaporation": 0.0,
            "toxicity_sweep_ratio": 0.0,
        }


@pytest.mark.asyncio
async def test_momentum_signal_routes_as_taker_with_hidden_stochastic_brackets():
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10)
    pm._ofi_rng.seed(7)
    pm.set_wallet_balance(1000.0)

    signal = OFIMomentumSignal(
        market_id="MKT-MOMO",
        no_asset_id="NO-MOMO",
        no_best_ask=0.50,
        current_vi=0.91,
        rolling_vi=0.93,
    )
    book = _FakeBook()

    pos = await pm.open_position(
        signal,
        no_aggregator=object(),
        no_book=book,
        fee_enabled=False,
        signal_metadata={"signal_source": "ofi_momentum", "meta_weight": 1.0},
    )

    expected = draw_stochastic_momentum_bracket(
        mean_max_hold_seconds=300.0,
        rng=random.Random(7),
    )

    assert pos is not None
    assert pos.signal_type == "ofi_momentum"
    assert pos.state == PositionState.EXIT_PENDING
    assert pos.entry_order is not None
    assert pos.entry_order.side == OrderSide.BUY
    assert pos.entry_order.status == OrderStatus.FILLED
    assert pos.entry_price == pytest.approx(0.50)
    assert pos.target_price == pytest.approx(expected.target_price(0.50))
    assert pos.stop_price == pytest.approx(expected.stop_price(0.50))
    assert pos.sl_trigger_cents == pytest.approx(expected.stop_loss_cents(0.50))
    assert pos.max_hold_seconds == pytest.approx(expected.max_hold_seconds)
    assert pos.drawn_tp == pytest.approx(pos.target_price)
    assert pos.drawn_stop == pytest.approx(pos.stop_price)
    assert pos.drawn_time == pytest.approx(pos.max_hold_seconds)
    assert pos.exit_order is None
    assert pos.entry_toxicity_index == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_momentum_position_time_stop_starts_smart_passive_exit():
    executor = OrderExecutor(paper_mode=True)
    book = _FakeBook()
    pm = PositionManager(executor, max_open_positions=10, book_trackers={"NO-MOMO": book})
    pm.set_wallet_balance(1000.0)

    signal = OFIMomentumSignal(
        market_id="MKT-MOMO",
        no_asset_id="NO-MOMO",
        no_best_ask=0.50,
        current_vi=0.91,
        rolling_vi=0.93,
    )

    pos = await pm.open_position(
        signal,
        no_aggregator=object(),
        no_book=book,
        fee_enabled=False,
        signal_metadata={"signal_source": "ofi_momentum"},
    )

    assert pos is not None
    assert pos.exit_order is None
    book.best_bid = round(pos.entry_price, 2)
    book.best_ask = round(min(0.99, pos.entry_price + 0.01), 2)
    book._snapshot = _FakeSnapshot(book.best_bid, book.best_ask)
    pos.entry_time = time.time() - (pos.max_hold_seconds + 1)

    await pm.check_timeouts()

    assert pos.state == PositionState.EXIT_PENDING
    assert pos.exit_reason == "time_stop"
    assert pos.exit_order is not None
    assert pos.exit_order.post_only is True
    assert pos.smart_passive_exit_deadline > time.time()
    assert pm.smart_passive_counters["smart_passive_started"] == 1


    @pytest.mark.asyncio
    async def test_momentum_toxicity_index_scales_taker_size():
        base_executor = OrderExecutor(paper_mode=True)
        base_pm = PositionManager(base_executor, max_open_positions=10)
        base_pm.set_wallet_balance(1000.0)

        scaled_executor = OrderExecutor(paper_mode=True)
        scaled_pm = PositionManager(scaled_executor, max_open_positions=10)
        scaled_pm.set_wallet_balance(1000.0)

        signal = OFIMomentumSignal(
            market_id="MKT-MOMO",
            no_asset_id="NO-MOMO",
            no_best_ask=0.50,
            current_vi=0.91,
            rolling_vi=0.93,
        )
        book = _FakeBook()

        base_pos = await base_pm.open_position(
            signal,
            no_aggregator=object(),
            no_book=book,
            fee_enabled=False,
            signal_metadata={"signal_source": "ofi_momentum", "meta_weight": 1.0},
        )
        scaled_pos = await scaled_pm.open_position(
            signal,
            no_aggregator=object(),
            no_book=book,
            fee_enabled=False,
            signal_metadata={
                "signal_source": "ofi_momentum",
                "meta_weight": 1.0,
                "toxicity_index": 0.9,
            },
        )

        assert base_pos is not None
        assert scaled_pos is not None
        assert scaled_pos.entry_size > base_pos.entry_size


@pytest.mark.asyncio
async def test_momentum_time_stop_enters_smart_passive_before_taker_fallback():
    book = _FakeBook(bid=0.49, ask=0.51)
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10, book_trackers={"NO-MOMO": book})
    pm.set_wallet_balance(1000.0)

    signal = OFIMomentumSignal(
        market_id="MKT-MOMO",
        no_asset_id="NO-MOMO",
        no_best_ask=0.50,
        current_vi=0.91,
        rolling_vi=0.93,
    )

    pos = await pm.open_position(
        signal,
        no_aggregator=object(),
        no_book=book,
        fee_enabled=False,
        signal_metadata={"signal_source": "ofi_momentum"},
    )

    assert pos is not None
    assert pos.exit_order is None
    book.best_bid = round(pos.entry_price, 2)
    book.best_ask = round(min(0.99, pos.entry_price + 0.01), 2)
    book._snapshot = _FakeSnapshot(book.best_bid, book.best_ask)
    pos.entry_time = time.time() - (pos.max_hold_seconds + 1)

    await pm.check_timeouts()

    assert pos.state == PositionState.EXIT_PENDING
    assert pos.exit_reason == "time_stop"
    assert pos.smart_passive_exit_deadline > time.time()
    assert pos.exit_order is not None
    assert pos.exit_order.post_only is True
    assert pos.exit_order.price == pytest.approx(book.best_ask)

    pos.smart_passive_exit_deadline = time.time() - 1
    await pm.check_timeouts()

    assert pos.state == PositionState.CLOSED
    assert pos.exit_reason == "time_stop"
    assert pos.smart_passive_exit_deadline == 0.0
    assert pos.market_id in pm._stop_loss_cooldowns
    assert pm.smart_passive_counters["fallback_triggered"] == 1
    assert pos.exit_toxicity_index == pytest.approx(0.31)