"""
Tests for the BotReplayAdapter — production-parity strategy wrapper.

Covers:
- Adapter initialisation (aggregators created)
- Trade routing to correct YES/NO aggregators
- Signal gating (zscore check, position limit)
- Entry/exit order lifecycle
- Fill routing (entry vs exit fill handling)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.matching_engine import Fill
from src.backtest.strategy import BotReplayAdapter
from src.data.ohlcv import OHLCVAggregator
from src.data.websocket_client import TradeEvent
from src.trading.executor import OrderSide


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


class TestAdapterInit:
    """Adapter initialisation and aggregator creation."""

    def test_aggregators_created(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES-TOKEN",
            no_asset_id="NO-TOKEN",
        )
        # Mock the engine
        adapter.engine = MagicMock()
        adapter.on_init()

        assert adapter._yes_agg is not None
        assert adapter._no_agg is not None
        assert isinstance(adapter._yes_agg, OHLCVAggregator)
        assert isinstance(adapter._no_agg, OHLCVAggregator)

    def test_separate_aggregator_ids(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        # Each aggregator should track its own asset
        assert adapter._yes_agg.asset_id == "YES"
        assert adapter._no_agg.asset_id == "NO"


class TestTradeRouting:
    """Trade events route to the correct aggregator."""

    def _make_trade_event(self, asset_id: str, price: float = 0.50, size: float = 10.0) -> TradeEvent:
        return TradeEvent(
            timestamp=1000.0,
            market_id="MKT-1",
            asset_id=asset_id,
            side="buy",
            price=price,
            size=size,
            is_yes=(asset_id == "YES"),
            is_taker=False,
        )

    def test_yes_trade_routes_to_yes_agg(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        trade = self._make_trade_event("YES", 0.60)
        adapter.on_trade("YES", trade)

        # YES aggregator should have processed the trade
        assert adapter._yes_agg.current_price > 0

    def test_no_trade_routes_to_no_agg(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        trade = self._make_trade_event("NO", 0.40)
        adapter.on_trade("NO", trade)

        assert adapter._no_agg.current_price > 0

    def test_unknown_asset_ignored(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        trade = self._make_trade_event("OTHER", 0.50)
        adapter.on_trade("OTHER", trade)

        # Neither aggregator should be affected
        assert adapter._yes_agg.current_price == 0
        assert adapter._no_agg.current_price == 0


class TestOfiBookRouting:
    def test_no_side_book_update_feeds_ofi_detector_and_submits_order(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.engine.submit_order.return_value = MagicMock(order_id="SIM-OFI")
        adapter.on_init()

        adapter.on_book_update(
            "NO",
            {
                "best_bid": 0.49,
                "best_ask": 0.51,
                "timestamp": 3600.0,
                "bid_levels": [(0.49, 95.0)],
                "ask_levels": [(0.51, 5.0)],
            },
        )

        adapter.engine.submit_order.assert_called_once()
        assert "SIM-OFI" in adapter._pending_entries

    def test_ofi_bracket_draws_are_reproducible_per_seed(self):
        def _pending_entry_for(seed: int) -> dict:
            adapter = BotReplayAdapter(
                market_id="MKT-1",
                yes_asset_id="YES",
                no_asset_id="NO",
                stochastic_seed=seed,
            )
            adapter.engine = MagicMock()
            adapter.engine.submit_order.return_value = MagicMock(order_id=f"SIM-OFI-{seed}")
            adapter.on_init()
            adapter.on_book_update(
                "NO",
                {
                    "best_bid": 0.49,
                    "best_ask": 0.51,
                    "timestamp": 3600.0,
                    "bid_levels": [(0.49, 95.0)],
                    "ask_levels": [(0.51, 5.0)],
                },
            )
            return adapter._pending_entries[f"SIM-OFI-{seed}"]

        seeded_a = _pending_entry_for(11)
        seeded_b = _pending_entry_for(11)
        seeded_c = _pending_entry_for(12)

        assert seeded_a["target_price"] == pytest.approx(seeded_b["target_price"])
        assert seeded_a["stop_price"] == pytest.approx(seeded_b["stop_price"])
        assert seeded_a["max_hold_seconds"] == pytest.approx(seeded_b["max_hold_seconds"])
        assert (
            seeded_a["target_price"],
            seeded_a["stop_price"],
            seeded_a["max_hold_seconds"],
        ) != (
            seeded_c["target_price"],
            seeded_c["stop_price"],
            seeded_c["max_hold_seconds"],
        )

    def test_yes_side_book_update_does_not_trigger_ofi_entry(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.engine.submit_order.return_value = MagicMock(order_id="SIM-OFI")
        adapter.on_init()

        adapter.on_book_update(
            "YES",
            {
                "best_bid": 0.49,
                "best_ask": 0.51,
                "timestamp": 3600.0,
                "bid_levels": [(0.49, 95.0)],
                "ask_levels": [(0.51, 5.0)],
            },
        )

        adapter.engine.submit_order.assert_not_called()


class TestSignalGating:
    """Signal gating: zscore threshold, position limits."""

    def test_no_signal_when_insufficient_bars(self):
        """Need at least 5 bars of history."""
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.engine.matching_engine = MagicMock()
        adapter.engine.matching_engine.best_ask = 0.40
        adapter.engine.submit_order = MagicMock()
        adapter.on_init()

        # Only 2 bars — not enough for signal
        assert len(adapter._yes_agg.bars) == 0
        assert len(adapter._pending_entries) == 0

    def test_position_limit_enforced(self):
        """Max 3 open positions."""
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        # Simulate 3 open positions
        adapter._open_positions = {"a": {}, "b": {}, "c": {}}

        # Even with a strong signal, should not place another order
        adapter._on_yes_bar_closed(MagicMock(close=0.80))

        # No new orders
        adapter.engine.submit_order.assert_not_called()


class TestFillRouting:
    """Fill events route to entry or exit handling."""

    def test_entry_fill_triggers_exit_order(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )

        # Mock engine
        mock_engine = MagicMock()
        mock_exit_order = MagicMock()
        mock_exit_order.order_id = "SIM-EXIT"
        mock_engine.submit_order.return_value = mock_exit_order

        mock_order = MagicMock()
        mock_order.remaining = 0.0  # fully filled
        mock_engine.matching_engine.get_order.return_value = mock_order

        adapter.engine = mock_engine
        adapter.on_init()

        # Simulate a pending entry
        adapter._pending_entries["SIM-1"] = {
            "order": MagicMock(),
            "entry_price": 0.40,
            "target_price": 0.45,
            "zscore": 2.5,
            "yes_price": 0.60,
            "vwap": 0.55,
        }

        # Simulate entry fill
        fill = Fill(
            order_id="SIM-1",
            price=0.40,
            size=10.0,
            fee=0.05,
            timestamp=1000.0,
            is_maker=False,
            side=OrderSide.BUY,
        )
        adapter.on_fill(fill)

        # Should have placed an exit order
        mock_engine.submit_order.assert_called_once()
        call_kwargs = mock_engine.submit_order.call_args
        assert call_kwargs.kwargs.get("order_type") == "limit"
        assert "SIM-EXIT" in adapter._pending_exits

    def test_exit_fill_records_round_trip(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        # Simulate an open position awaiting exit fill
        entry_fill = Fill(
            order_id="SIM-1",
            price=0.40,
            size=10.0,
            fee=0.05,
            timestamp=500.0,
            is_maker=False,
            side=OrderSide.BUY,
        )
        pos = {
            "entry_fill": entry_fill,
            "entry_ctx": {},
            "exit_order_id": "SIM-2",
            "entry_time": 500.0,
        }
        adapter._pending_exits["SIM-2"] = pos
        adapter._open_positions["SIM-2"] = pos

        # Simulate exit fill
        exit_fill = Fill(
            order_id="SIM-2",
            price=0.45,
            size=10.0,
            fee=0.03,
            timestamp=600.0,
            is_maker=True,
            side=OrderSide.SELL,
        )
        adapter.on_fill(exit_fill)

        # Round-trip should be recorded
        assert len(adapter._positions) == 1
        rt = adapter._positions[0]
        assert rt["entry_price"] == 0.40
        assert rt["exit_price"] == 0.45
        # PnL = (0.45 - 0.40) × 10 - 0.05 - 0.03 = 0.42
        expected_pnl = 0.5 - 0.05 - 0.03
        assert abs(rt["pnl_net"] - expected_pnl) < 1e-9

        # Open position should be removed
        assert len(adapter._open_positions) == 0
        assert len(adapter._pending_exits) == 0

    def test_unknown_fill_ignored(self):
        """Fill for an untracked order ID is silently ignored."""
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        fill = Fill(
            order_id="UNKNOWN-99",
            price=0.50,
            size=5.0,
            fee=0.0,
            timestamp=100.0,
            is_maker=False,
            side=OrderSide.BUY,
        )
        # Should not raise
        adapter.on_fill(fill)
        assert len(adapter._positions) == 0

    def test_ofi_stop_loss_forces_taker_exit(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        entry_fill = Fill(
            order_id="SIM-IN",
            price=0.50,
            size=10.0,
            fee=0.05,
            timestamp=500.0,
            is_maker=False,
            side=OrderSide.BUY,
        )
        pos = {
            "entry_fill": entry_fill,
            "entry_ctx": {"signal_source": "ofi_momentum"},
            "exit_order_id": "SIM-EXIT",
            "entry_time": 500.0,
            "stop_price": 0.4925,
            "max_hold_seconds": 300.0,
            "signal_source": "ofi_momentum",
            "exit_reason": "take_profit",
        }
        adapter._pending_exits["SIM-EXIT"] = pos
        adapter._open_positions["SIM-EXIT"] = pos

        def _simulate_fill(order_id, size, price=None, is_maker=True):
            fill = Fill(
                order_id=order_id,
                price=price if price is not None else 0.49,
                size=size,
                fee=0.03,
                timestamp=700.0,
                is_maker=is_maker,
                side=OrderSide.SELL,
            )
            adapter.on_fill(fill)
            return fill

        adapter.engine.simulate_fill.side_effect = _simulate_fill

        adapter.on_book_update(
            "NO",
            {
                "best_bid": 0.49,
                "best_ask": 0.51,
                "timestamp": 700.0,
                "bid_levels": [(0.49, 100.0)],
                "ask_levels": [(0.51, 100.0)],
            },
        )

        adapter.engine.submit_order.assert_called_once_with(
            side=OrderSide.SELL,
            price=0.49,
            size=10.0,
            order_type="limit",
            post_only=False,
        )
        adapter.engine.simulate_fill.assert_called_once_with(
            adapter.engine.submit_order.return_value.order_id,
            10.0,
            price=0.49,
            is_maker=False,
        )
        assert len(adapter._open_positions) == 0
        assert adapter._positions[-1]["exit_reason"] == "stop_loss"

    def test_ofi_time_stop_starts_smart_passive_maker_exit(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        entry_fill = Fill(
            order_id="SIM-IN",
            price=0.50,
            size=10.0,
            fee=0.05,
            timestamp=500.0,
            is_maker=False,
            side=OrderSide.BUY,
        )
        pos = {
            "entry_fill": entry_fill,
            "entry_ctx": {"signal_source": "ofi_momentum"},
            "exit_order_id": "SIM-EXIT",
            "entry_time": 500.0,
            "stop_price": 0.4925,
            "max_hold_seconds": 300.0,
            "signal_source": "ofi_momentum",
            "exit_reason": "take_profit",
        }
        adapter._pending_exits["SIM-EXIT"] = pos
        adapter._open_positions["SIM-EXIT"] = pos
        adapter.engine.cancel_order.return_value = True

        passive_order = MagicMock()
        passive_order.order_id = "SIM-TIMESTOP-MAKER"
        adapter.engine.submit_order.return_value = passive_order

        adapter._check_momentum_brackets(
            {
                "best_bid": 0.50,
                "best_ask": 0.52,
                "timestamp": 801.0,
            }
        )

        adapter.engine.cancel_order.assert_called_once_with("SIM-EXIT")
        adapter.engine.submit_order.assert_called_once_with(
            side=OrderSide.SELL,
            price=0.52,
            size=10.0,
            order_type="limit",
            post_only=True,
        )
        assert "SIM-TIMESTOP-MAKER" in adapter._pending_exits
        assert adapter._pending_exits["SIM-TIMESTOP-MAKER"]["exit_reason"] == "time_stop"
        assert adapter._pending_exits["SIM-TIMESTOP-MAKER"]["smart_passive_deadline"] == pytest.approx(816.0)

    def test_ofi_time_stop_passive_exit_falls_back_to_taker(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        entry_fill = Fill(
            order_id="SIM-IN",
            price=0.50,
            size=10.0,
            fee=0.05,
            timestamp=500.0,
            is_maker=False,
            side=OrderSide.BUY,
        )
        pos = {
            "entry_fill": entry_fill,
            "entry_ctx": {"signal_source": "ofi_momentum"},
            "exit_order_id": "SIM-TIMESTOP-MAKER",
            "entry_time": 500.0,
            "stop_price": 0.4925,
            "max_hold_seconds": 300.0,
            "signal_source": "ofi_momentum",
            "exit_reason": "time_stop",
            "smart_passive_started_at": 801.0,
            "smart_passive_deadline": 816.0,
        }
        adapter._pending_exits["SIM-TIMESTOP-MAKER"] = pos
        adapter._open_positions["SIM-TIMESTOP-MAKER"] = pos
        adapter.engine.cancel_order.return_value = True

        taker_order = MagicMock()
        taker_order.order_id = "SIM-TIMESTOP-TAKER"
        adapter.engine.submit_order.return_value = taker_order

        def _simulate_fill(order_id, size, price=None, is_maker=True):
            fill = Fill(
                order_id=order_id,
                price=price if price is not None else 0.50,
                size=size,
                fee=0.03,
                timestamp=801.0,
                is_maker=is_maker,
                side=OrderSide.SELL,
            )
            adapter.on_fill(fill)
            return fill

        adapter.engine.simulate_fill.side_effect = _simulate_fill

        adapter._check_momentum_brackets(
            {
                "best_bid": 0.50,
                "best_ask": 0.52,
                "timestamp": 816.0,
            },
        )

        adapter.engine.cancel_order.assert_called_once_with("SIM-TIMESTOP-MAKER")
        adapter.engine.submit_order.assert_called_once_with(
            side=OrderSide.SELL,
            price=0.50,
            size=10.0,
            order_type="limit",
            post_only=False,
        )
        adapter.engine.simulate_fill.assert_called_once_with(
            "SIM-TIMESTOP-TAKER",
            10.0,
            price=0.50,
            is_maker=False,
        )
        assert len(adapter._open_positions) == 0
        assert adapter._positions[-1]["exit_reason"] == "time_stop"
        assert adapter._smart_passive_fallbacks == 1

    def test_ofi_time_stop_passive_exit_can_fill_as_maker(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        entry_fill = Fill(
            order_id="SIM-IN",
            price=0.50,
            size=10.0,
            fee=0.05,
            timestamp=500.0,
            is_maker=False,
            side=OrderSide.BUY,
        )
        pos = {
            "entry_fill": entry_fill,
            "entry_ctx": {"signal_source": "ofi_momentum"},
            "exit_order_id": "SIM-TIMESTOP-MAKER",
            "entry_time": 500.0,
            "stop_price": 0.4925,
            "max_hold_seconds": 300.0,
            "signal_source": "ofi_momentum",
            "exit_reason": "time_stop",
            "smart_passive_started_at": 801.0,
            "smart_passive_deadline": 816.0,
        }
        adapter._pending_exits["SIM-TIMESTOP-MAKER"] = pos
        adapter._open_positions["SIM-TIMESTOP-MAKER"] = pos

        maker_fill = Fill(
            order_id="SIM-TIMESTOP-MAKER",
            price=0.52,
            size=10.0,
            fee=0.0,
            timestamp=810.0,
            is_maker=True,
            side=OrderSide.SELL,
        )

        adapter.on_fill(maker_fill)

        assert len(adapter._open_positions) == 0
        assert adapter._positions[-1]["exit_reason"] == "time_stop"
        assert adapter._smart_passive_maker_filled == 1


class TestOnEnd:
    """End-of-backtest summary."""

    def test_on_end_runs_cleanly(self):
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        # Simulate some completed positions
        adapter._positions = [{"pnl_net": 5.0}, {"pnl_net": -2.0}]
        adapter._open_positions = {"SIM-X": {}}

        # Should not raise
        adapter.on_end()


class TestIntegrationMinimal:
    """Minimal end-to-end with BotReplayAdapter through BacktestEngine."""

    def test_adapter_runs_in_engine(self, tmp_path: Path):
        """Smoke test: adapter can run through the engine without errors."""
        events = [
            {
                "local_ts": 1.0,
                "source": "l2",
                "asset_id": "YES",
                "payload": {
                    "event_type": "snapshot",
                    "bids": [{"price": "0.55", "size": "100"}],
                    "asks": [{"price": "0.60", "size": "100"}],
                },
            },
            {
                "local_ts": 2.0,
                "source": "trade",
                "asset_id": "YES",
                "payload": {"price": "0.58", "size": "10", "side": "buy"},
            },
            {
                "local_ts": 3.0,
                "source": "trade",
                "asset_id": "NO",
                "payload": {"price": "0.42", "size": "10", "side": "sell"},
            },
        ]
        _write_events(tmp_path / "data.jsonl", events)

        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES",
            no_asset_id="NO",
        )
        loader = DataLoader.from_files(tmp_path / "data.jsonl")
        config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0)
        engine = BacktestEngine(
            strategy=adapter, data_loader=loader, config=config,
        )
        result = engine.run()

        assert result.events_processed == 3
        assert abs(result.final_cash - 1000.0) < 1e-6  # no orders placed
