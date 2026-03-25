"""
End-to-end tests for the BacktestEngine.

Uses synthetic L2 + trade data piped through a DataLoader to verify
the full pipeline: clock → event processing → matching → strategy
callbacks → position tracking → telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.backtest.matching_engine import Fill
from src.backtest.strategy import StrategyABC
from src.data.ohlcv import OHLCVBar
from src.data.websocket_client import TradeEvent
from src.trading.executor import OrderSide


# ═══════════════════════════════════════════════════════════════════════════
#  Test strategy implementations
# ═══════════════════════════════════════════════════════════════════════════

class PassiveStrategy(StrategyABC):
    """Does nothing — just counts events."""

    def __init__(self):
        self.book_updates = 0
        self.trades = 0
        self.fills: list[Fill] = []
        self.bars: list[OHLCVBar] = []
        self.inited = False
        self.ended = False

    def on_init(self):
        self.inited = True

    def on_book_update(self, asset_id, snapshot):
        self.book_updates += 1

    def on_trade(self, asset_id, trade):
        self.trades += 1

    def on_fill(self, fill):
        self.fills.append(fill)

    def on_bar(self, asset_id, bar):
        self.bars.append(bar)

    def on_end(self):
        self.ended = True


class SelfAggregatingStrategy(PassiveStrategy):
    self_aggregates_trades = True


class BuyOnFirstTradeStrategy(StrategyABC):
    """Places a market buy of 10 shares on the first trade event."""

    def __init__(self):
        self._bought = False
        self.fills: list[Fill] = []
        self.orders = []

    def on_init(self):
        pass

    def on_book_update(self, asset_id, snapshot):
        pass

    def on_trade(self, asset_id, trade):
        if not self._bought:
            order = self.engine.submit_order(
                OrderSide.BUY, price=1.0, size=10.0, order_type="market"
            )
            self.orders.append(order)
            self._bought = True

    def on_fill(self, fill):
        self.fills.append(fill)

    def on_end(self):
        pass


class MakerStrategy(StrategyABC):
    """Places a resting bid on the first book update, then waits for fills."""

    def __init__(self, bid_price: float = 0.45, size: float = 10.0):
        self._bid_price = bid_price
        self._size = size
        self._placed = False
        self.fills: list[Fill] = []
        self.order = None

    def on_init(self):
        pass

    def on_book_update(self, asset_id, snapshot):
        if not self._placed:
            self.order = self.engine.submit_order(
                OrderSide.BUY, price=self._bid_price, size=self._size,
                post_only=True,
            )
            self._placed = True

    def on_trade(self, asset_id, trade):
        pass

    def on_fill(self, fill):
        self.fills.append(fill)

    def on_end(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic data helpers
# ═══════════════════════════════════════════════════════════════════════════

def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _make_l2_snapshot(
    ts: float,
    asset_id: str,
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
) -> dict:
    return {
        "local_ts": ts,
        "source": "l2",
        "asset_id": asset_id,
        "payload": {
            "event_type": "snapshot",
            "bids": [{"price": p, "size": s} for p, s in bids],
            "asks": [{"price": p, "size": s} for p, s in asks],
        },
    }


def _make_trade(
    ts: float,
    asset_id: str,
    price: str,
    size: str,
    side: str = "buy",
) -> dict:
    return {
        "local_ts": ts,
        "source": "trade",
        "asset_id": asset_id,
        "payload": {
            "price": price,
            "size": size,
            "side": side,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineLifecycle:
    """Strategy lifecycle hooks: on_init → on_end."""

    def test_init_and_end_called(self, tmp_path: Path):
        events = [_make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")])]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = PassiveStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0)
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        assert strategy.inited is True
        assert strategy.ended is True

    def test_engine_reference_set(self, tmp_path: Path):
        events = [_make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")])]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = PassiveStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        engine = BacktestEngine(
            strategy=strategy, data_loader=loader,
            config=BacktestConfig(latency_ms=0.0),
        )
        engine.run()

        assert strategy.engine is engine


class TestEventRouting:
    """Events dispatched to the correct strategy callbacks."""

    def test_book_updates_counted(self, tmp_path: Path):
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")]),
            _make_l2_snapshot(2.0, "A", [("0.46", "100")], [("0.54", "100")]),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = PassiveStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        engine = BacktestEngine(
            strategy=strategy, data_loader=loader,
            config=BacktestConfig(latency_ms=0.0),
        )
        result = engine.run()

        assert strategy.book_updates == 2

    def test_self_aggregating_strategy_skips_engine_bar_generation(self, tmp_path: Path):
        events = [
            _make_trade(1.0, "A", "0.50", "10"),
            _make_trade(61.5, "A", "0.55", "10"),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = SelfAggregatingStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        engine = BacktestEngine(
            strategy=strategy, data_loader=loader,
            config=BacktestConfig(latency_ms=0.0),
        )
        result = engine.run()

        assert strategy.trades == 2
        assert strategy.bars == []
        assert result.events_processed == 2

    def test_trades_counted(self, tmp_path: Path):
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")]),
            _make_trade(2.0, "A", "0.50", "10"),
            _make_trade(3.0, "A", "0.51", "20"),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = PassiveStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        engine = BacktestEngine(
            strategy=strategy, data_loader=loader,
            config=BacktestConfig(latency_ms=0.0),
        )
        engine.run()

        assert strategy.trades == 2


class TestTakerOrder:
    """Market / aggressive limit order execution through the engine."""

    def test_market_buy_fills(self, tmp_path: Path):
        """Market buy → fills from the ask side → cash decreases."""
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")]),
            _make_trade(2.0, "A", "0.55", "5", "buy"),
            # Third event activates the pending order submitted during the trade callback
            _make_l2_snapshot(3.0, "A", [("0.45", "100")], [("0.55", "100")]),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = BuyOnFirstTradeStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_enabled=False)
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        # Should have filled 10 shares at 0.55
        assert len(strategy.fills) == 1
        assert strategy.fills[0].price == 0.55
        assert strategy.fills[0].size == 10.0

        # Cash = 1000 - (10 × 0.55) = 994.5
        assert abs(result.final_cash - 994.5) < 1e-6

    def test_market_buy_with_fees(self, tmp_path: Path):
        """Taker fee is deducted from cash."""
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.50", "100")]),
            _make_trade(2.0, "A", "0.50", "5", "buy"),
            # Third event activates the pending order from the trade callback
            _make_l2_snapshot(3.0, "A", [("0.45", "100")], [("0.50", "100")]),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = BuyOnFirstTradeStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(
            initial_cash=1000.0, latency_ms=0.0,
            fee_enabled=True, fee_max_pct=1.56,
        )
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        # Fill price = 0.50, fee_rate = 0.0156 × 4 × 0.25 = 0.0156
        # Fee = 10 × 0.50 × 0.0156 = 0.078
        assert len(strategy.fills) == 1
        expected_fee = 10.0 * 0.50 * 0.0156
        assert abs(strategy.fills[0].fee - expected_fee) < 1e-9

        expected_cash = 1000.0 - (10.0 * 0.50 + expected_fee)
        assert abs(result.final_cash - expected_cash) < 1e-6


class TestMakerOrder:
    """Maker order with FIFO queue drain through the engine."""

    def test_maker_fill_after_queue_drain(self, tmp_path: Path):
        """Resting bid fills after sufficient sell volume."""
        events = [
            # Snapshot: bid 0.45@100, ask 0.55@100
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")]),
            # Sell trades at 0.45 drain the queue (100) then fill our 10
            _make_trade(3.0, "A", "0.45", "110", "sell"),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = MakerStrategy(bid_price=0.45, size=10.0)
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_enabled=False)
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        assert len(strategy.fills) == 1
        assert strategy.fills[0].is_maker is True
        assert strategy.fills[0].fee == 0.0
        assert strategy.fills[0].size == 10.0

    def test_maker_no_fill_insufficient_volume(self, tmp_path: Path):
        """Not enough sell volume → no fill on our maker order."""
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")]),
            # Only 50 shares of sell at 0.45 — not enough to drain 100 queue
            _make_trade(3.0, "A", "0.45", "50", "sell"),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = MakerStrategy(bid_price=0.45, size=10.0)
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0)
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        assert len(strategy.fills) == 0


class TestLatencyInEngine:
    """Latency penalty through the full engine pipeline."""

    def test_order_delayed_by_latency(self, tmp_path: Path):
        """With 500ms latency, order submitted at t=2 activates at t=2.5."""
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.50", "100")]),
            # First trade triggers order submission at t=2.0
            _make_trade(2.0, "A", "0.50", "5", "buy"),
            # The order won't activate until t=2.5 — next event at t=2.3 too early
            _make_trade(2.3, "A", "0.50", "5", "buy"),
            # Event at t=2.5 should activate the pending order
            _make_trade(2.5, "A", "0.50", "5", "buy"),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = BuyOnFirstTradeStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(
            initial_cash=1000.0, latency_ms=500.0, fee_enabled=False,
        )
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        # Should still fill eventually
        assert len(strategy.fills) >= 1


class TestResultObject:
    """BacktestResult output validation."""

    def test_result_has_metrics(self, tmp_path: Path):
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")]),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = PassiveStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(initial_cash=500.0)
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert result.config == config
        assert result.events_processed == 1
        assert abs(result.final_cash - 500.0) < 1e-9

    def test_summary_string(self, tmp_path: Path):
        events = [
            _make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")]),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = PassiveStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        engine = BacktestEngine(
            strategy=strategy, data_loader=loader,
            config=BacktestConfig(),
        )
        result = engine.run()
        s = result.summary()
        assert "Events processed" in s
        assert "Final cash" in s


class TestEquityComputation:
    """Mark-to-market equity = cash + positions × mid."""

    def test_equity_no_positions(self, tmp_path: Path):
        events = [_make_l2_snapshot(1.0, "A", [("0.45", "100")], [("0.55", "100")])]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = PassiveStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0)
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        assert abs(result.final_equity - 1000.0) < 1e-6

    def test_equity_uses_asset_specific_raw_book(self, tmp_path: Path):
        events = [
            _make_l2_snapshot(
                1.0,
                "A",
                [("0.007", "1000")],
                [("0.999", "1000"), ("0.008", "1000")],
            ),
            _make_trade(2.0, "A", "0.008", "1", "buy"),
            _make_l2_snapshot(
                3.0,
                "A",
                [("0.007", "1000")],
                [("0.999", "1000"), ("0.008", "1000")],
            ),
        ]
        _write_events(tmp_path / "a.jsonl", events)

        strategy = BuyOnFirstTradeStrategy()
        loader = DataLoader.from_files(tmp_path / "a.jsonl")
        config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_enabled=False)
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
        result = engine.run()

        assert abs(result.final_cash - 999.92) < 1e-6
        assert abs(result.final_equity - 999.99) < 1e-6
