from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import polars as pl
import pytest

from scripts.backtest_mid_tier_probability_compression import MidTierCompressionConfig, run_backtest


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


def _strict_row(
    *,
    timestamp: datetime,
    event_id: str,
    market_id: str,
    token_id: str,
    best_bid: float,
    best_ask: float,
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
    }


def test_single_fill_favorite_collapse_is_counted_as_legging_drawdown(tmp_path: Path) -> None:
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

    parquet_path = tmp_path / "midtier_collapse.parquet"
    pl.DataFrame(rows).write_parquet(parquet_path)

    result = run_backtest(parquet_path, MidTierCompressionConfig(timestamp_unit="us", resolution_timestamp_unit="us"))

    assert result.summary["candidate_orders"] == 2
    assert result.summary["accepted_orders"] == 2
    assert result.summary["filled_orders"] == 1
    assert result.summary["favorite_collapse_hits"] == 1
    assert result.summary["total_legging_loss_usd"] == 50.0
    assert result.summary["max_legging_drawdown_usd"] == 50.0
    assert result.summary["total_realized_pnl_usd"] == -50.0
    assert result.summary["quote_schema"] == "yes_bbo_inverted_to_no"
    assert result.summary["concurrent_name_cap_mode"] == "rolling"
    assert result.summary["concurrent_name_cap_release_key"] == "resolution_timestamp"

    collapse_event = result.event_tail.row(0, named=True)
    assert collapse_event["event_id"] == "evt-1"
    assert collapse_event["accepted_legs"] == 2
    assert collapse_event["filled_legs"] == 1
    assert collapse_event["favorite_collapse_hit"] is True
    assert collapse_event["legging_loss_usd"] == 50.0

    collapse_order = result.orders.filter(pl.col("market_id") == "mkt-c").row(0, named=True)
    assert collapse_order["entry_yes_ask"] == pytest.approx(0.11)
    assert collapse_order["entry_no_bid"] == pytest.approx(0.89)
    assert collapse_order["future_min_no_ask_exclusive"] == pytest.approx(0.88)


def test_concurrent_name_cap_rejects_overlapping_candidates(tmp_path: Path) -> None:
    start = datetime(2026, 4, 3, 14, 0, 0)
    resolution = start + timedelta(hours=4)
    rows: list[dict[str, object]] = []

    for idx in range(1, 4):
        event_id = f"evt-{idx}"
        rows.extend(
            [
                _row(
                    timestamp=start,
                    resolution_timestamp=resolution,
                    event_id=event_id,
                    market_id=f"mkt-{idx}-a",
                    token_id=f"tok-{idx}-a",
                    best_bid=0.50,
                    best_ask=0.51,
                    final_resolution_value=0.0,
                ),
                _row(
                    timestamp=start,
                    resolution_timestamp=resolution,
                    event_id=event_id,
                    market_id=f"mkt-{idx}-b",
                    token_id=f"tok-{idx}-b",
                    best_bid=0.46,
                    best_ask=0.47,
                    final_resolution_value=0.0,
                ),
                _row(
                    timestamp=start,
                    resolution_timestamp=resolution,
                    event_id=event_id,
                    market_id=f"mkt-{idx}-c",
                    token_id=f"tok-{idx}-c",
                    best_bid=0.20,
                    best_ask=0.21,
                    final_resolution_value=0.0,
                ),
            ]
        )

    parquet_path = tmp_path / "midtier_cap.parquet"
    pl.DataFrame(rows).write_parquet(parquet_path)

    result = run_backtest(
        parquet_path,
        MidTierCompressionConfig(
            timestamp_unit="us",
            resolution_timestamp_unit="us",
            max_concurrent_names=2,
        ),
    )

    accepted = result.orders.filter(pl.col("accepted")).sort("event_id")

    assert result.summary["candidate_orders"] == 3
    assert result.summary["accepted_orders"] == 2
    assert result.summary["rejected_orders"] == 1
    assert accepted.get_column("event_id").to_list() == ["evt-1", "evt-2"]


def test_concurrent_name_cap_releases_slot_after_resolution_timestamp(tmp_path: Path) -> None:
    start = datetime(2026, 4, 3, 16, 0, 0)
    rows: list[dict[str, object]] = []

    event_specs = [
        ("evt-1", start, start + timedelta(hours=1)),
        ("evt-2", start + timedelta(minutes=30), start + timedelta(hours=4)),
        ("evt-3", start + timedelta(hours=2), start + timedelta(hours=5)),
    ]

    for event_id, timestamp, resolution_timestamp in event_specs:
        rows.extend(
            [
                _row(
                    timestamp=timestamp,
                    resolution_timestamp=resolution_timestamp,
                    event_id=event_id,
                    market_id=f"{event_id}-a",
                    token_id=f"{event_id}-ta",
                    best_bid=0.50,
                    best_ask=0.51,
                    final_resolution_value=0.0,
                ),
                _row(
                    timestamp=timestamp,
                    resolution_timestamp=resolution_timestamp,
                    event_id=event_id,
                    market_id=f"{event_id}-b",
                    token_id=f"{event_id}-tb",
                    best_bid=0.46,
                    best_ask=0.47,
                    final_resolution_value=0.0,
                ),
                _row(
                    timestamp=timestamp,
                    resolution_timestamp=resolution_timestamp,
                    event_id=event_id,
                    market_id=f"{event_id}-c",
                    token_id=f"{event_id}-tc",
                    best_bid=0.20,
                    best_ask=0.21,
                    final_resolution_value=0.0,
                ),
            ]
        )

    parquet_path = tmp_path / "midtier_rolling_cap.parquet"
    pl.DataFrame(rows).write_parquet(parquet_path)

    result = run_backtest(
        parquet_path,
        MidTierCompressionConfig(
            timestamp_unit="us",
            resolution_timestamp_unit="us",
            max_concurrent_names=1,
        ),
    )

    accepted = result.orders.filter(pl.col("accepted")).sort("order_timestamp")

    assert result.summary["accepted_orders"] == 2
    assert result.summary["rejected_orders"] == 1
    assert accepted.get_column("event_id").to_list() == ["evt-1", "evt-3"]


def test_backtest_joins_enriched_manifest_and_marks_open_markets_to_last_no_bid(tmp_path: Path) -> None:
    lake_root = tmp_path / "lake"
    day_dir = lake_root / "l2_book" / "date=2026-04-03"
    day_dir.mkdir(parents=True)

    start = datetime(2026, 4, 3, 12, 0, 0)
    resolution = start + timedelta(hours=6)
    cross = start + timedelta(minutes=5)
    rows = [
        _strict_row(
            timestamp=start,
            event_id="evt-1",
            market_id="mkt-a",
            token_id="tok-a",
            best_bid=0.49,
            best_ask=0.50,
        ),
        _strict_row(
            timestamp=start,
            event_id="evt-1",
            market_id="mkt-b",
            token_id="tok-b",
            best_bid=0.47,
            best_ask=0.48,
        ),
        _strict_row(
            timestamp=start,
            event_id="evt-1",
            market_id="mkt-c",
            token_id="tok-c",
            best_bid=0.10,
            best_ask=0.11,
        ),
        _strict_row(
            timestamp=start,
            event_id="evt-1",
            market_id="mkt-d",
            token_id="tok-d",
            best_bid=0.08,
            best_ask=0.09,
        ),
        _strict_row(
            timestamp=cross,
            event_id="evt-1",
            market_id="mkt-c",
            token_id="tok-c",
            best_bid=0.14,
            best_ask=0.15,
        ),
        _strict_row(
            timestamp=cross,
            event_id="evt-1",
            market_id="mkt-d",
            token_id="tok-d",
            best_bid=0.07,
            best_ask=0.08,
        ),
    ]
    pl.DataFrame(rows).write_parquet(day_dir / "ticks.parquet")

    (lake_root / "enriched_manifest.json").write_text(
        json.dumps(
            {
                "markets": [
                    {
                        "market_id": "mkt-a",
                        "event_id": "evt-1",
                        "gamma_market_status": "resolved",
                        "gamma_closed": True,
                        "resolution_timestamp": resolution.isoformat(),
                        "final_resolution_value": 0.0,
                    },
                    {
                        "market_id": "mkt-b",
                        "event_id": "evt-1",
                        "gamma_market_status": "resolved",
                        "gamma_closed": True,
                        "resolution_timestamp": resolution.isoformat(),
                        "final_resolution_value": 0.0,
                    },
                    {
                        "market_id": "mkt-c",
                        "event_id": "evt-1",
                        "gamma_market_status": "open",
                        "gamma_closed": False,
                        "resolution_timestamp": None,
                        "final_resolution_value": None,
                    },
                    {
                        "market_id": "mkt-d",
                        "event_id": "evt-1",
                        "gamma_market_status": "resolved",
                        "gamma_closed": True,
                        "resolution_timestamp": resolution.isoformat(),
                        "final_resolution_value": 0.0,
                    },
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_backtest(day_dir, MidTierCompressionConfig(timestamp_unit="us", resolution_timestamp_unit="us"))

    assert result.summary["candidate_orders"] == 2
    assert result.summary["filled_orders"] == 1
    assert result.summary["simulated_exit_candidate_orders"] == 1
    assert result.summary["simulated_exit_accepted_orders"] == 1
    assert result.summary["simulated_exit_fills"] == 1
    assert result.summary["settlement_exit_fills"] == 0
    assert result.summary["avg_filled_entry_spread"] == pytest.approx(0.01)
    assert result.summary["total_realized_pnl_usd"] == pytest.approx(-2.25)

    open_order = result.orders.filter(pl.col("market_id") == "mkt-c").row(0, named=True)
    assert open_order["exit_mode"] == "simulated_last_no_bid"
    assert open_order["market_last_no_bid"] == pytest.approx(0.85)
    assert open_order["effective_exit_no_price"] == pytest.approx(0.85)
    assert open_order["entry_spread"] == pytest.approx(0.01)