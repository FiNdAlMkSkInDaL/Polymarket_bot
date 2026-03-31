from __future__ import annotations

import random
import time
from dataclasses import dataclass

import pytest

from src.core.config import settings
from src.execution.momentum_taker import draw_stochastic_momentum_bracket
from src.signals.ofi_momentum import OFIMomentumSignal
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus
from src.trading.position_manager import PositionManager, PositionState


@dataclass
class _FakeLevel:
    price: float
    size: float


class _FakeSnapshot:
    def __init__(
        self,
        bid: float,
        ask: float,
        *,
        bid_depth_usd: float = 500.0,
        ask_depth_usd: float = 500.0,
        timestamp: float | None = None,
    ):
        self.best_bid = bid
        self.best_ask = ask
        self.mid_price = (bid + ask) / 2.0
        self.bid_depth_usd = bid_depth_usd
        self.ask_depth_usd = ask_depth_usd
        self.timestamp = time.time() if timestamp is None else timestamp


class _FakeBook:
    def __init__(
        self,
        bid: float = 0.49,
        ask: float = 0.50,
        *,
        asset_id: str = "NO-MOMO",
        buy_toxicity: float = 0.2,
        sell_toxicity: float = 0.31,
        bid_depth_usd: float = 500.0,
        ask_depth_usd: float = 500.0,
        bid_depth_ewma: float = 500.0,
        ask_depth_ewma: float = 500.0,
    ):
        self.asset_id = asset_id
        self.has_data = True
        self.is_reliable = True
        self.best_bid = bid
        self.best_ask = ask
        self._buy_toxicity = buy_toxicity
        self._sell_toxicity = sell_toxicity
        self._bid_depth_usd = bid_depth_usd
        self._ask_depth_usd = ask_depth_usd
        self._bid_depth_ewma = bid_depth_ewma
        self._ask_depth_ewma = ask_depth_ewma
        self._snapshot = _FakeSnapshot(
            bid,
            ask,
            bid_depth_usd=bid_depth_usd,
            ask_depth_usd=ask_depth_usd,
        )
        self._asks = [_FakeLevel(ask, 500.0), _FakeLevel(round(ask + 0.01, 2), 500.0)]
        self._bids = [_FakeLevel(bid, 500.0), _FakeLevel(round(bid - 0.01, 2), 500.0)]

    @property
    def book_depth_ratio(self) -> float:
        return 1.0

    def snapshot(self):
        return self._snapshot

    def refresh_snapshot(self) -> None:
        self._snapshot = _FakeSnapshot(
            self.best_bid,
            self.best_ask,
            bid_depth_usd=self._bid_depth_usd,
            ask_depth_usd=self._ask_depth_usd,
        )

    def levels(self, side: str, depth: int):
        del depth
        return self._asks if side == "ask" else self._bids

    def top_depths_usd(self) -> tuple[float, float]:
        return self._bid_depth_usd, self._ask_depth_usd

    def top_depth_ewma(self, side: str) -> float:
        return self._bid_depth_ewma if side == "bid" else self._ask_depth_ewma

    def toxicity_metrics(self, side: str = "BUY"):
        toxicity = self._buy_toxicity if side.upper() == "BUY" else self._sell_toxicity
        return {
            "toxicity_index": toxicity,
            "toxicity_depth_evaporation": 0.0,
            "toxicity_sweep_ratio": 0.0,
        }


def _override_strategy(**overrides: float) -> dict[str, float]:
    previous: dict[str, float] = {}
    for key, value in overrides.items():
        previous[key] = getattr(settings.strategy, key)
        object.__setattr__(settings.strategy, key, value)
    return previous


def _restore_strategy(previous: dict[str, float]) -> None:
    for key, value in previous.items():
        object.__setattr__(settings.strategy, key, value)


@pytest.mark.asyncio
async def test_momentum_signal_routes_as_taker_with_hidden_stochastic_brackets():
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10)
    pm._ofi_rng.seed(7)
    pm.set_wallet_balance(1000.0)
    previous = _override_strategy(
        ofi_toxicity_veto_threshold=0.95,
        ofi_min_target_edge_cents=0.5,
        ofi_momentum_take_profit_pct=0.08,
        desired_margin_cents=0.0,
        paper_slippage_cents=0.0,
    )
    try:
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
            mean_take_profit_pct=settings.strategy.ofi_momentum_take_profit_pct,
            mean_stop_loss_pct=settings.strategy.ofi_momentum_stop_loss_pct,
            mean_max_hold_seconds=min(300.0, float(settings.strategy.ofi_momentum_max_hold_seconds)),
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
        assert pos.entry_toxicity_index == pytest.approx(0.2)
        assert pos.expected_net_target_per_share_cents > 0.0
        assert pos.expected_net_target_minus_one_tick_per_share_cents <= pos.expected_net_target_per_share_cents
    finally:
        _restore_strategy(previous)


@pytest.mark.asyncio
async def test_momentum_position_time_stop_starts_smart_passive_exit():
    executor = OrderExecutor(paper_mode=True)
    book = _FakeBook()
    pm = PositionManager(executor, max_open_positions=10, book_trackers={"NO-MOMO": book})
    pm.set_wallet_balance(1000.0)
    previous = _override_strategy(
        ofi_toxicity_veto_threshold=0.95,
        ofi_min_target_edge_cents=0.5,
        ofi_momentum_take_profit_pct=0.08,
        desired_margin_cents=0.0,
        paper_slippage_cents=0.0,
    )
    try:
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
    finally:
        _restore_strategy(previous)


@pytest.mark.asyncio
async def test_momentum_toxicity_veto_blocks_entry():
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10)
    pm.set_wallet_balance(1000.0)

    signal = OFIMomentumSignal(
        market_id="MKT-MOMO",
        no_asset_id="NO-MOMO",
        no_best_ask=0.50,
        current_vi=0.91,
        rolling_vi=0.93,
    )
    book = _FakeBook(buy_toxicity=0.9)

    pos = await pm.open_position(
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

    assert pos is None
    assert pm.pop_last_entry_rejection_reason("ofi_momentum", signal.market_id) == "toxicity_veto"


@pytest.mark.asyncio
async def test_momentum_toxicity_haircuts_size_instead_of_boosting() -> None:
    base_executor = OrderExecutor(paper_mode=True)
    base_pm = PositionManager(base_executor, max_open_positions=10)
    base_pm.set_wallet_balance(1000.0)

    scaled_executor = OrderExecutor(paper_mode=True)
    scaled_pm = PositionManager(scaled_executor, max_open_positions=10)
    scaled_pm.set_wallet_balance(1000.0)
    previous = _override_strategy(
        ofi_toxicity_veto_threshold=0.95,
        ofi_min_target_edge_cents=0.5,
        ofi_momentum_take_profit_pct=0.08,
        desired_margin_cents=0.0,
        paper_slippage_cents=0.0,
    )
    try:
        signal = OFIMomentumSignal(
            market_id="MKT-MOMO",
            no_asset_id="NO-MOMO",
            no_best_ask=0.50,
            current_vi=0.91,
            rolling_vi=0.93,
        )
        book = _FakeBook(buy_toxicity=0.62)

        base_pos = await base_pm.open_position(
            signal,
            no_aggregator=object(),
            no_book=book,
            fee_enabled=False,
            signal_metadata={"signal_source": "ofi_momentum", "meta_weight": 1.0, "toxicity_index": 0.2},
        )
        scaled_pos = await scaled_pm.open_position(
            signal,
            no_aggregator=object(),
            no_book=book,
            fee_enabled=False,
            signal_metadata={
                "signal_source": "ofi_momentum",
                "meta_weight": 1.0,
                "toxicity_index": 0.62,
            },
        )

        assert base_pos is not None
        assert scaled_pos is not None
        assert scaled_pos.entry_size < base_pos.entry_size
    finally:
        _restore_strategy(previous)


@pytest.mark.asyncio
async def test_momentum_insufficient_edge_rejects_high_price_entry() -> None:
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10)
    pm.set_wallet_balance(1000.0)

    signal = OFIMomentumSignal(
        market_id="MKT-MOMO",
        no_asset_id="NO-MOMO",
        no_best_ask=0.98,
        current_vi=0.91,
        rolling_vi=0.93,
    )
    book = _FakeBook(bid=0.97, ask=0.98, buy_toxicity=0.2)

    pos = await pm.open_position(
        signal,
        no_aggregator=object(),
        no_book=book,
        fee_enabled=False,
        signal_metadata={"signal_source": "ofi_momentum", "meta_weight": 1.0, "toxicity_index": 0.2},
    )

    assert pos is None
    assert pm.pop_last_entry_rejection_reason("ofi_momentum", signal.market_id) == "insufficient_edge"


@pytest.mark.asyncio
async def test_momentum_time_stop_bypasses_smart_passive_when_projected_taker_fill_hits_stop() -> None:
    book = _FakeBook(bid=0.49, ask=0.50, asset_id="NO-MOMO")
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10, book_trackers={"NO-MOMO": book})
    pm._ofi_rng.seed(0)
    pm.set_wallet_balance(1000.0)
    previous = _override_strategy(
        ofi_toxicity_veto_threshold=0.95,
        ofi_min_target_edge_cents=0.5,
        ofi_momentum_take_profit_pct=0.12,
        desired_margin_cents=0.0,
        paper_slippage_cents=0.5,
    )
    try:
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
            fee_enabled=True,
            signal_metadata={"signal_source": "ofi_momentum", "meta_weight": 1.0},
        )

        assert pos is not None
        pos.drawn_stop = 0.49
        pos.stop_price = 0.49
        book.best_bid = pos.drawn_stop + settings.strategy.paper_slippage_cents / 100.0
        book.best_ask = round(book.best_bid + 0.01, 4)
        book.refresh_snapshot()
        pos.entry_time = time.time() - (pos.max_hold_seconds + 1)

        await pm.check_timeouts()

        assert pos.state == PositionState.CLOSED
        assert pos.exit_reason == "time_stop"
        assert pm.smart_passive_counters["smart_passive_started"] == 0
        assert pos.time_stop_suppression_count == 0
        assert pos.time_stop_delay_seconds >= 1.0
        assert pos.exit_price_minus_drawn_stop_cents == pytest.approx(0.0)
    finally:
        _restore_strategy(previous)


@pytest.mark.asyncio
async def test_momentum_time_stop_suppression_is_capped_and_recorded() -> None:
    book = _FakeBook(bid=0.49, ask=0.50, asset_id="NO-MOMO")
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10, book_trackers={"NO-MOMO": book})
    pm._ofi_rng.seed(0)
    pm.set_wallet_balance(1000.0)
    previous = _override_strategy(
        ofi_toxicity_veto_threshold=0.95,
        ofi_min_target_edge_cents=0.5,
        ofi_momentum_take_profit_pct=0.12,
        desired_margin_cents=0.0,
        paper_slippage_cents=0.5,
    )
    try:
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
            fee_enabled=True,
            signal_metadata={"signal_source": "ofi_momentum", "meta_weight": 1.0},
        )

        assert pos is not None
        pos.drawn_stop = 0.48
        pos.stop_price = 0.48
        book.best_bid = 0.50
        book.best_ask = 0.55
        book._bid_depth_usd = 40.0
        book._ask_depth_usd = 190.0
        book._bid_depth_ewma = 200.0
        book._ask_depth_ewma = 220.0
        book.refresh_snapshot()
        pos.entry_time = time.time() - (pos.max_hold_seconds + 31)

        await pm.check_timeouts()

        assert pos.state == PositionState.CLOSED
        assert pos.exit_reason == "time_stop"
        assert pm.smart_passive_counters["smart_passive_started"] == 0
        assert pos.time_stop_suppression_count >= 1
        assert pos.time_stop_delay_seconds >= 31.0
        assert pos.exit_price_minus_drawn_stop_cents == pytest.approx((pos.exit_price - pos.drawn_stop) * 100.0)
    finally:
        _restore_strategy(previous)


@pytest.mark.asyncio
async def test_momentum_time_stop_enters_smart_passive_before_taker_fallback():
    book = _FakeBook(bid=0.49, ask=0.51)
    executor = OrderExecutor(paper_mode=True)
    pm = PositionManager(executor, max_open_positions=10, book_trackers={"NO-MOMO": book})
    pm.set_wallet_balance(1000.0)
    previous = _override_strategy(
        ofi_toxicity_veto_threshold=0.95,
        ofi_min_target_edge_cents=0.5,
        ofi_momentum_take_profit_pct=0.08,
        desired_margin_cents=0.0,
        paper_slippage_cents=0.0,
    )
    try:
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
    finally:
        _restore_strategy(previous)