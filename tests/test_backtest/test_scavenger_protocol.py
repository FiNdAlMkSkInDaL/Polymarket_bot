from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.backtest.scavenger_protocol import ScavengerConfig, run_scavenger_backtest


def _row(
    *,
    timestamp: datetime,
    resolution_timestamp: datetime,
    market_id: str,
    event_id: str,
    token_id: str = "YES",
    best_bid: float,
    best_ask: float,
    bid_depth: float = 100.0,
    ask_depth: float = 100.0,
    final_resolution_value: float = 1.0,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "event_id": event_id,
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "resolution_timestamp": resolution_timestamp,
        "final_resolution_value": final_resolution_value,
    }


def test_scavenger_waits_for_true_touch_through() -> None:
    resolution = datetime(2026, 4, 5, 12, 0, 0)
    rows = [
        _row(
            timestamp=resolution - timedelta(hours=80),
            resolution_timestamp=resolution,
            market_id="m1",
            event_id="outside_window",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=resolution - timedelta(hours=48),
            resolution_timestamp=resolution,
            market_id="m1",
            event_id="signal",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=resolution - timedelta(hours=47),
            resolution_timestamp=resolution,
            market_id="m1",
            event_id="near_but_not_fill",
            best_bid=0.95,
            best_ask=0.96,
        ),
        _row(
            timestamp=resolution - timedelta(hours=44),
            resolution_timestamp=resolution,
            market_id="m1",
            event_id="real_fill",
            best_bid=0.94,
            best_ask=0.96,
        ),
    ]

    result = run_scavenger_backtest(pl.DataFrame(rows), config=ScavengerConfig())

    assert result.orders.height == 1
    assert result.fills.height == 1
    trade = result.fills.row(0, named=True)
    assert trade["signal_event_id"] == "signal"
    assert trade["fill_event_id"] == "real_fill"
    assert trade["fill_reason"] == "best_bid_lt_0p95"
    assert trade["fill_timestamp"] == resolution - timedelta(hours=44)
    assert trade["raw_roi"] == pytest.approx((1.0 - 0.95) / 0.95)
    assert trade["apr"] == pytest.approx(((1.0 - 0.95) / 0.95) * 365.0 * 24.0 / 44.0)


def test_scavenger_resets_cycle_after_a_fill() -> None:
    resolution = datetime(2026, 4, 6, 12, 0, 0)
    rows = [
        _row(
            timestamp=resolution - timedelta(hours=30),
            resolution_timestamp=resolution,
            market_id="m2",
            event_id="signal_1",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=resolution - timedelta(hours=29),
            resolution_timestamp=resolution,
            market_id="m2",
            event_id="fill_1",
            best_bid=0.95,
            best_ask=0.95,
        ),
        _row(
            timestamp=resolution - timedelta(hours=28),
            resolution_timestamp=resolution,
            market_id="m2",
            event_id="stays_through_bid",
            best_bid=0.94,
            best_ask=0.94,
        ),
        _row(
            timestamp=resolution - timedelta(hours=20),
            resolution_timestamp=resolution,
            market_id="m2",
            event_id="signal_2",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=resolution - timedelta(hours=19),
            resolution_timestamp=resolution,
            market_id="m2",
            event_id="fill_2",
            best_bid=0.95,
            best_ask=0.95,
        ),
    ]

    result = run_scavenger_backtest(pl.DataFrame(rows), config=ScavengerConfig())

    assert result.orders.height == 2
    assert result.fills.height == 2
    assert result.fills["fill_event_id"].to_list() == ["fill_1", "fill_2"]
    assert result.fills["fill_reason"].to_list() == ["best_ask_le_0p95", "best_ask_le_0p95"]

    summary = result.summary.row(0, named=True)
    assert summary["orders_posted"] == 2
    assert summary["fills"] == 2
    assert summary["fill_rate"] == pytest.approx(1.0)
    assert summary["signal_rows"] == 2