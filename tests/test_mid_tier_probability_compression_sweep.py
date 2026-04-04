from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from scripts.sweep_mid_tier_probability_compression import discover_daily_partitions, run_sweep


def _row(
    *,
    timestamp: datetime,
    resolution_timestamp: datetime,
    event_id: str,
    market_id: str,
    token_id: str,
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
        "bid_depth": 200.0,
        "ask_depth": 200.0,
        "resolution_timestamp": resolution_timestamp,
        "final_resolution_value": final_resolution_value,
    }


def test_run_sweep_builds_daily_grid_and_scales_notionals(tmp_path: Path) -> None:
    day_dir = tmp_path / "2026-04-03"
    day_dir.mkdir()

    start = datetime(2026, 4, 3, 12, 0, 0)
    resolution = start + timedelta(hours=6)
    cross = start + timedelta(minutes=5)
    rows = [
        _row(
            timestamp=start,
            resolution_timestamp=resolution,
            event_id="evt-1",
            market_id="mkt-a",
            token_id="tok-a",
            best_bid=0.49,
            best_ask=0.50,
            final_resolution_value=0.0,
        ),
        _row(
            timestamp=start,
            resolution_timestamp=resolution,
            event_id="evt-1",
            market_id="mkt-b",
            token_id="tok-b",
            best_bid=0.47,
            best_ask=0.48,
            final_resolution_value=0.0,
        ),
        _row(
            timestamp=start,
            resolution_timestamp=resolution,
            event_id="evt-1",
            market_id="mkt-c",
            token_id="tok-c",
            best_bid=0.10,
            best_ask=0.11,
            final_resolution_value=1.0,
        ),
        _row(
            timestamp=start,
            resolution_timestamp=resolution,
            event_id="evt-1",
            market_id="mkt-d",
            token_id="tok-d",
            best_bid=0.08,
            best_ask=0.09,
            final_resolution_value=0.0,
        ),
        _row(
            timestamp=cross,
            resolution_timestamp=resolution,
            event_id="evt-1",
            market_id="mkt-c",
            token_id="tok-c",
            best_bid=0.12,
            best_ask=0.13,
            final_resolution_value=1.0,
        ),
        _row(
            timestamp=cross,
            resolution_timestamp=resolution,
            event_id="evt-1",
            market_id="mkt-d",
            token_id="tok-d",
            best_bid=0.07,
            best_ask=0.08,
            final_resolution_value=0.0,
        ),
    ]
    pl.DataFrame(rows).write_parquet(day_dir / "ticks.parquet")

    artifacts = run_sweep(
        tmp_path,
        thresholds=(0.95, 0.98),
        notionals=(10.0, 50.0),
        timestamp_unit="us",
        resolution_timestamp_unit="us",
    )

    assert artifacts.daily_rows.height == 4
    active = artifacts.daily_rows.filter(pl.col("top2_yes_threshold") == 0.95).sort("max_leg_notional_usd")
    inactive = artifacts.daily_rows.filter(pl.col("top2_yes_threshold") == 0.98).sort("max_leg_notional_usd")

    assert active.get_column("total_realized_pnl_usd").to_list() == [-10.0, -50.0]
    assert active.get_column("total_legging_loss_usd").to_list() == [10.0, 50.0]
    assert active.get_column("avg_filled_entry_spread").to_list() == [0.01, 0.01]
    assert active.get_column("simulated_exit_fills").to_list() == [0, 0]
    assert inactive.get_column("candidate_orders").to_list() == [0, 0]
    assert inactive.get_column("avg_filled_entry_spread").to_list() == [0.0, 0.0]
    assert artifacts.summary["days_processed"] == 1
    assert artifacts.summary["notionals"] == [10.0, 50.0]


def test_discover_daily_partitions_supports_hive_l2_book_layout(tmp_path: Path) -> None:
    hour_dir = tmp_path / "l2_book" / "date=2026-03-20" / "hour=00"
    hour_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            _row(
                timestamp=datetime(2026, 3, 20, 12, 0, 0),
                resolution_timestamp=datetime(2026, 3, 20, 18, 0, 0),
                event_id="evt-1",
                market_id="mkt-1",
                token_id="tok-1",
                best_bid=0.49,
                best_ask=0.50,
                final_resolution_value=0.0,
            )
        ]
    ).write_parquet(hour_dir / "part-1.parquet")

    discovered = discover_daily_partitions(tmp_path)

    assert discovered == [("2026-03-20", tmp_path / "l2_book" / "date=2026-03-20")]