"""Tests for the dedicated pure market maker replay adapter."""

from __future__ import annotations

import json
from pathlib import Path

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.strategy import PureMarketMakerReplayAdapter
from src.core.config import StrategyParams


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _snapshot(ts: float, asset_id: str, bid: str, bid_size: str, ask: str, ask_size: str) -> dict:
    return {
        "local_ts": ts,
        "source": "l2",
        "asset_id": asset_id,
        "payload": {
            "event_type": "snapshot",
            "bids": [{"price": bid, "size": bid_size}],
            "asks": [{"price": ask, "size": ask_size}],
        },
    }


def _delta(ts: float, asset_id: str, side: str, price: str, size: str) -> dict:
    return {
        "local_ts": ts,
        "source": "l2",
        "asset_id": asset_id,
        "payload": {
            "event_type": "price_change",
            "changes": [{"side": side, "price": price, "size": size}],
        },
    }


class TestPureMarketMakerReplayAdapter:
    def test_posts_resting_bid(self, tmp_path: Path):
        events = [_snapshot(1.0, "NO", "0.40", "20", "0.42", "20")]
        _write_events(tmp_path / "ticks.jsonl", events)

        adapter = PureMarketMakerReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=StrategyParams(),
        )
        engine = BacktestEngine(
            strategy=adapter,
            data_loader=DataLoader.from_files(tmp_path / "ticks.jsonl"),
            config=BacktestConfig(initial_cash=1000.0, latency_ms=0.0),
        )

        result = engine.run()

        assert result.events_processed == 1
        assert len(result.all_orders) == 1
        assert result.all_orders[0].side.value == "BUY"
        assert abs(result.all_orders[0].price - 0.40) < 1e-9

    def test_posts_tight_and_wide_bids(self, tmp_path: Path):
        events = [_snapshot(1.0, "NO", "0.40", "20", "0.42", "20")]
        _write_events(tmp_path / "ticks.jsonl", events)

        adapter = PureMarketMakerReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=StrategyParams(
                pure_mm_wide_tier_enabled=True,
                pure_mm_wide_spread_pct=0.15,
                pure_mm_inventory_cap_usd=1000.0,
            ),
        )
        engine = BacktestEngine(
            strategy=adapter,
            data_loader=DataLoader.from_files(tmp_path / "ticks.jsonl"),
            config=BacktestConfig(initial_cash=1000.0, latency_ms=0.0),
        )

        result = engine.run()

        bid_orders = [order for order in result.all_orders if order.side.value == "BUY"]
        assert len(bid_orders) == 2
        prices = sorted(order.price for order in bid_orders)
        assert prices == [0.34, 0.4]

    def test_fills_resting_bid_when_ask_crosses(self, tmp_path: Path):
        events = [
            _snapshot(1.0, "NO", "0.40", "20", "0.42", "20"),
            _snapshot(2.0, "NO", "0.39", "20", "0.40", "20"),
        ]
        _write_events(tmp_path / "ticks.jsonl", events)

        adapter = PureMarketMakerReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=StrategyParams(),
        )
        engine = BacktestEngine(
            strategy=adapter,
            data_loader=DataLoader.from_files(tmp_path / "ticks.jsonl"),
            config=BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_enabled=False),
        )

        result = engine.run()

        assert result.metrics.total_fills >= 1
        assert result.all_fills[0].is_maker is True
        assert abs(result.all_fills[0].price - 0.40) < 1e-9
        assert result.final_cash < 1000.0
        assert any(order.side.value == "SELL" for order in result.all_orders)

    def test_cancels_bid_on_toxic_ofi(self, tmp_path: Path):
        events = [
            _snapshot(1.0, "NO", "0.40", "20", "0.42", "20"),
            _snapshot(2.0, "NO", "0.40", "20", "0.42", "20"),
            _delta(2.5, "NO", "BUY", "0.40", "10"),
        ]
        _write_events(tmp_path / "ticks.jsonl", events)

        adapter = PureMarketMakerReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=StrategyParams(
                pure_mm_toxic_ofi_ratio=0.5,
                pure_mm_depth_evaporation_pct=0.95,
            ),
        )
        engine = BacktestEngine(
            strategy=adapter,
            data_loader=DataLoader.from_files(tmp_path / "ticks.jsonl"),
            config=BacktestConfig(initial_cash=1000.0, latency_ms=0.0),
        )

        result = engine.run()

        bid_orders = [order for order in result.all_orders if order.side.value == "BUY"]
        assert bid_orders
        assert bid_orders[-1].status.value == "CANCELLED"

    def test_cancels_all_bid_tiers_on_toxic_depth_evaporation(self, tmp_path: Path):
        events = [
            _snapshot(1.0, "NO", "0.40", "20", "0.42", "20"),
            _snapshot(2.0, "NO", "0.40", "1", "0.42", "1"),
        ]
        _write_events(tmp_path / "ticks.jsonl", events)

        adapter = PureMarketMakerReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=StrategyParams(
                pure_mm_wide_tier_enabled=True,
                pure_mm_inventory_cap_usd=1000.0,
                pure_mm_depth_window_s=0.5,
                pure_mm_depth_evaporation_pct=0.1,
            ),
        )
        engine = BacktestEngine(
            strategy=adapter,
            data_loader=DataLoader.from_files(tmp_path / "ticks.jsonl"),
            config=BacktestConfig(initial_cash=1000.0, latency_ms=0.0),
        )

        result = engine.run()

        bid_orders = [order for order in result.all_orders if order.side.value == "BUY"]
        assert len(bid_orders) == 2
        assert all(order.status.value == "CANCELLED" for order in bid_orders)