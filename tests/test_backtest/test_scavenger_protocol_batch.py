from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from scripts.run_scavenger_protocol_batch import run_scavenger_batch
from src.backtest.scavenger_protocol import ScavengerConfig


def _row(
    *,
    timestamp: datetime,
    resolution_timestamp: datetime,
    market_id: str,
    event_id: str,
    token_id: str = "YES",
    best_bid: float,
    best_ask: float,
    final_resolution_value: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "event_id": event_id,
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": 100.0,
        "ask_depth": 100.0,
        "resolution_timestamp": resolution_timestamp,
        "final_resolution_value": final_resolution_value,
    }


def test_scavenger_batch_runner_tracks_daily_lockup_and_catastrophic_losses(tmp_path: Path) -> None:
    input_root = tmp_path / "lake"
    day_one = datetime(2026, 4, 1, 0, 0, 0)
    day_two = day_one + timedelta(days=1)
    resolution = datetime(2026, 4, 2, 9, 0, 0)

    (input_root / "2026-04-01").mkdir(parents=True)
    (input_root / "2026-04-02").mkdir(parents=True)

    pl.DataFrame(
        [
            _row(
                timestamp=datetime(2026, 4, 1, 10, 0, 0),
                resolution_timestamp=resolution,
                market_id="m1",
                event_id="signal_accept",
                best_bid=0.96,
                best_ask=0.99,
                final_resolution_value=0.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 1, 11, 0, 0),
                resolution_timestamp=resolution,
                market_id="m1",
                event_id="fill_accept",
                best_bid=0.94,
                best_ask=0.96,
                final_resolution_value=0.0,
            ),
            _row(
                timestamp=datetime(2026, 4, 1, 12, 0, 0),
                resolution_timestamp=resolution,
                market_id="m2",
                event_id="signal_reject",
                best_bid=0.96,
                best_ask=0.99,
                final_resolution_value=1.0,
            ),
        ]
    ).write_parquet(input_root / "2026-04-01" / "ticks.parquet")

    pl.DataFrame(
        [
            _row(
                timestamp=datetime(2026, 4, 2, 8, 0, 0),
                resolution_timestamp=resolution,
                market_id="m1",
                event_id="day_two_marker",
                best_bid=0.99,
                best_ask=1.0,
                final_resolution_value=0.0,
            )
        ]
    ).write_parquet(input_root / "2026-04-02" / "ticks.parquet")

    result = run_scavenger_batch(
        input_root,
        ScavengerConfig(starting_bankroll_usdc=250.0, max_notional_per_market_usdc=250.0),
        window_lookback_days=1,
        window_lookahead_days=1,
    )

    tearsheet = result.tearsheet.sort("date")
    day_one_row = tearsheet.row(0, named=True)
    day_two_row = tearsheet.row(1, named=True)

    assert day_one_row["date"] == day_one.date().isoformat()
    assert day_one_row["starting_capital_usdc"] == 250.0
    assert day_one_row["ending_capital_usdc"] == 0.0
    assert day_one_row["fills"] == 1
    assert day_one_row["rejected_signals_capital_lockup"] == 1
    assert day_one_row["catastrophic_losses"] == 0

    assert day_two_row["date"] == day_two.date().isoformat()
    assert day_two_row["starting_capital_usdc"] == 0.0
    assert day_two_row["ending_capital_usdc"] == 0.0
    assert day_two_row["fills"] == 0
    assert day_two_row["rejected_signals_capital_lockup"] == 0
    assert day_two_row["catastrophic_losses"] == 1


def test_scavenger_batch_runner_reports_near_miss_days_and_top_markets(tmp_path: Path) -> None:
    input_root = tmp_path / "lake"
    resolution = datetime(2026, 4, 3, 12, 0, 0)

    (input_root / "2026-04-01").mkdir(parents=True)
    (input_root / "2026-04-02").mkdir(parents=True)

    pl.DataFrame(
        [
            _row(
                timestamp=datetime(2026, 4, 1, 10, 0, 0),
                resolution_timestamp=resolution,
                market_id="m-near",
                event_id="near_ask_day_one",
                best_bid=0.96,
                best_ask=0.98,
                final_resolution_value=1.0,
            )
        ]
    ).write_parquet(input_root / "2026-04-01" / "ticks.parquet")

    pl.DataFrame(
        [
            _row(
                timestamp=datetime(2026, 4, 2, 10, 0, 0),
                resolution_timestamp=resolution,
                market_id="m-near",
                event_id="near_bid_day_two",
                best_bid=0.975,
                best_ask=0.99,
                final_resolution_value=1.0,
            )
        ]
    ).write_parquet(input_root / "2026-04-02" / "ticks.parquet")

    result = run_scavenger_batch(
        input_root,
        ScavengerConfig(),
        window_lookback_days=1,
        window_lookahead_days=1,
    )

    tearsheet = result.tearsheet.sort("date")
    assert tearsheet["near_misses"].to_list() == [1, 1]
    assert result.summary["near_misses"] == 2
    assert result.summary["near_miss_daily_counts"] == [
        {"date": "2026-04-01", "near_miss_count": 1, "near_miss_market_count": 1},
        {"date": "2026-04-02", "near_miss_count": 1, "near_miss_market_count": 1},
    ]
    assert result.summary["near_miss_top_markets"][0]["market_id"] == "m-near"
    assert result.summary["near_miss_top_markets"][0]["near_miss_count"] == 2