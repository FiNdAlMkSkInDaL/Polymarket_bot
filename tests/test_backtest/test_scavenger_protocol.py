from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.backtest.scavenger_protocol import (
    ScavengerConfig,
    collect_scavenger_price_distribution,
    run_scavenger_backtest,
    summarize_scavenger_price_distribution,
)


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
    assert result.candidates.height == 1
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
    assert result.candidates.height == 2
    assert result.fills.height == 2
    assert result.fills["fill_event_id"].to_list() == ["fill_1", "fill_2"]
    assert result.fills["fill_reason"].to_list() == ["best_ask_le_0p95", "best_ask_le_0p95"]

    summary = result.summary.row(0, named=True)
    assert summary["orders_posted"] == 2
    assert summary["fills"] == 2
    assert summary["fill_rate"] == pytest.approx(1.0)
    assert summary["signal_rows"] == 2


def test_scavenger_reinvests_only_after_resolution() -> None:
    first_resolution = datetime(2026, 4, 5, 12, 0, 0)
    second_resolution = datetime(2026, 4, 6, 12, 0, 0)
    rows = [
        _row(
            timestamp=datetime(2026, 4, 5, 8, 0, 0),
            resolution_timestamp=first_resolution,
            market_id="m3",
            event_id="signal_1",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=datetime(2026, 4, 5, 9, 0, 0),
            resolution_timestamp=first_resolution,
            market_id="m3",
            event_id="fill_1",
            best_bid=0.95,
            best_ask=0.95,
        ),
        _row(
            timestamp=datetime(2026, 4, 5, 11, 0, 0),
            resolution_timestamp=first_resolution,
            market_id="m4",
            event_id="signal_2",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=datetime(2026, 4, 5, 13, 0, 0),
            resolution_timestamp=second_resolution,
            market_id="m5",
            event_id="signal_3",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=datetime(2026, 4, 5, 14, 0, 0),
            resolution_timestamp=second_resolution,
            market_id="m5",
            event_id="fill_3",
            best_bid=0.95,
            best_ask=0.95,
        ),
    ]

    result = run_scavenger_backtest(
        pl.DataFrame(rows),
        config=ScavengerConfig(starting_bankroll_usdc=250.0, max_notional_per_market_usdc=250.0),
    )

    assert result.portfolio["position_status"].to_list() == [
        "accepted_filled",
        "rejected_capital_lockup",
        "accepted_filled",
    ]
    assert result.summary.row(0, named=True)["signals_rejected_capital_lockup"] == 1
    expected_final_bankroll = 250.0 + (250.0 * (1.0 / 0.95 - 1.0)) * 2.0
    assert result.summary.row(0, named=True)["ending_bankroll_usdc"] == pytest.approx(
        expected_final_bankroll
    )


def test_scavenger_reserves_capital_for_unfilled_quotes_until_resolution() -> None:
    resolution = datetime(2026, 4, 5, 12, 0, 0)
    rows = [
        _row(
            timestamp=datetime(2026, 4, 5, 8, 0, 0),
            resolution_timestamp=resolution,
            market_id="m6",
            event_id="signal_1",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=datetime(2026, 4, 5, 10, 0, 0),
            resolution_timestamp=resolution,
            market_id="m7",
            event_id="signal_2",
            best_bid=0.96,
            best_ask=0.99,
        ),
    ]

    result = run_scavenger_backtest(
        pl.DataFrame(rows),
        config=ScavengerConfig(starting_bankroll_usdc=250.0, max_notional_per_market_usdc=250.0),
    )

    assert result.fills.height == 0
    assert result.portfolio["position_status"].to_list() == [
        "accepted_unfilled",
        "rejected_capital_lockup",
    ]
    assert result.summary.row(0, named=True)["signals_rejected_capital_lockup"] == 1


def test_scavenger_prioritizes_shorter_time_to_resolution_for_same_timestamp() -> None:
    signal_ts = datetime(2026, 4, 5, 8, 0, 0)
    short_resolution = datetime(2026, 4, 5, 12, 0, 0)
    long_resolution = datetime(2026, 4, 6, 12, 0, 0)
    rows = [
        _row(
            timestamp=signal_ts,
            resolution_timestamp=long_resolution,
            market_id="a-long-lockup",
            event_id="long_signal",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=signal_ts + timedelta(minutes=10),
            resolution_timestamp=long_resolution,
            market_id="a-long-lockup",
            event_id="long_fill",
            best_bid=0.95,
            best_ask=0.95,
        ),
        _row(
            timestamp=signal_ts,
            resolution_timestamp=short_resolution,
            market_id="z-short-turnover",
            event_id="short_signal",
            best_bid=0.96,
            best_ask=0.99,
        ),
        _row(
            timestamp=signal_ts + timedelta(minutes=5),
            resolution_timestamp=short_resolution,
            market_id="z-short-turnover",
            event_id="short_fill",
            best_bid=0.94,
            best_ask=0.96,
        ),
    ]

    result = run_scavenger_backtest(
        pl.DataFrame(rows),
        config=ScavengerConfig(starting_bankroll_usdc=250.0, max_notional_per_market_usdc=250.0),
    )

    assert result.portfolio["signal_event_id"].to_list() == ["short_signal", "long_signal"]
    assert result.portfolio["position_status"].to_list() == [
        "accepted_filled",
        "rejected_capital_lockup",
    ]


def test_scavenger_tracks_near_miss_states_and_summary() -> None:
    resolution = datetime(2026, 4, 7, 12, 0, 0)
    rows = [
        _row(
            timestamp=resolution - timedelta(hours=40),
            resolution_timestamp=resolution,
            market_id="near-miss-market",
            event_id="near_ask",
            best_bid=0.96,
            best_ask=0.98,
        ),
        _row(
            timestamp=resolution - timedelta(hours=39),
            resolution_timestamp=resolution,
            market_id="near-miss-market",
            event_id="near_bid",
            best_bid=0.975,
            best_ask=0.99,
        ),
        _row(
            timestamp=resolution - timedelta(hours=38),
            resolution_timestamp=resolution,
            market_id="near-miss-market",
            event_id="far_miss",
            best_bid=0.99,
            best_ask=0.94,
        ),
    ]

    result = run_scavenger_backtest(pl.DataFrame(rows), config=ScavengerConfig())

    assert result.orders.height == 0
    assert result.fills.height == 0
    assert result.near_misses["event_id"].to_list() == ["near_ask", "near_bid"]
    assert result.near_misses["near_miss_reason"].to_list() == [
        "best_ask_below_threshold",
        "best_bid_above_threshold",
    ]
    summary = result.summary.row(0, named=True)
    assert summary["near_misses"] == 2
    assert summary["near_miss_markets"] == 1
    assert summary["closest_near_miss_price_gap"] == pytest.approx(0.01)


def test_scavenger_accepts_raw_book_sources_with_metadata_enrichment() -> None:
    resolution = datetime(2026, 4, 8, 12, 0, 0)
    raw_rows = [
        {
            "timestamp": resolution - timedelta(hours=20),
            "market_id": "raw-market",
            "event_id": "raw_signal",
            "token_id": "YES",
            "best_bid": 0.96,
            "best_ask": 0.99,
            "bid_depth": 100.0,
            "ask_depth": 100.0,
        },
        {
            "timestamp": resolution - timedelta(hours=19),
            "market_id": "raw-market",
            "event_id": "raw_fill",
            "token_id": "YES",
            "best_bid": 0.95,
            "best_ask": 0.95,
            "bid_depth": 100.0,
            "ask_depth": 100.0,
        },
    ]
    metadata_frame = pl.DataFrame(
        [
            {
                "market_id": "raw-market",
                "token_id": "YES",
                "metadata_event_id": "metadata_event",
                "resolution_timestamp": resolution,
                "final_resolution_value": 1.0,
            }
        ]
    )

    result = run_scavenger_backtest(
        pl.DataFrame(raw_rows),
        config=ScavengerConfig(),
        metadata_frame=metadata_frame,
    )

    assert result.orders.height == 1
    assert result.fills.height == 1
    assert result.fills.row(0, named=True)["fill_event_id"] == "raw_fill"


def test_scavenger_records_deepest_dip_and_highest_spike_distribution() -> None:
    resolution = datetime(2026, 4, 9, 12, 0, 0)
    rows = [
        _row(
            timestamp=resolution - timedelta(hours=30),
            resolution_timestamp=resolution,
            market_id="m-range",
            event_id="obs_1",
            best_bid=0.97,
            best_ask=0.99,
            final_resolution_value=1.0,
        ),
        _row(
            timestamp=resolution - timedelta(hours=20),
            resolution_timestamp=resolution,
            market_id="m-range",
            event_id="obs_2",
            best_bid=0.94,
            best_ask=0.93,
            final_resolution_value=1.0,
        ),
        _row(
            timestamp=resolution - timedelta(hours=10),
            resolution_timestamp=resolution,
            market_id="m-range",
            event_id="obs_3",
            best_bid=0.98,
            best_ask=0.95,
            final_resolution_value=1.0,
        ),
    ]

    result = run_scavenger_backtest(pl.DataFrame(rows), config=ScavengerConfig())

    assert result.price_distribution.height == 1
    distribution_row = result.price_distribution.row(0, named=True)
    assert distribution_row["market_id"] == "m-range"
    assert distribution_row["token_id"] == "YES"
    assert distribution_row["final_result"] == 1
    assert distribution_row["deepest_dip"] == pytest.approx(0.93)
    assert distribution_row["highest_spike"] == pytest.approx(0.98)

    distribution_summary = summarize_scavenger_price_distribution(
        result.price_distribution,
        current_bid_price=0.95,
    )
    assert distribution_summary["unit_count"] == 1
    assert distribution_summary["median_deepest_dip"] == pytest.approx(0.93)
    assert distribution_summary["recommended_realistic_scavenge_bid"] == pytest.approx(0.92)


def test_scavenger_collects_price_distribution_in_chunks_for_raw_book_sources(
    tmp_path: Path,
) -> None:
    resolution = datetime(2026, 4, 10, 12, 0, 0)
    first_chunk_path = tmp_path / "chunk_1.parquet"
    second_chunk_path = tmp_path / "chunk_2.parquet"
    pl.DataFrame(
        [
            {
                "timestamp": resolution - timedelta(hours=30),
                "market_id": "raw-market",
                "event_id": "obs_1",
                "token_id": "YES",
                "best_bid": 0.97,
                "best_ask": 0.99,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            }
        ]
    ).write_parquet(first_chunk_path)
    pl.DataFrame(
        [
            {
                "timestamp": resolution - timedelta(hours=10),
                "market_id": "raw-market",
                "event_id": "obs_2",
                "token_id": "YES",
                "best_bid": 0.98,
                "best_ask": 0.94,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            }
        ]
    ).write_parquet(second_chunk_path)

    metadata_frame = pl.DataFrame(
        [
            {
                "market_id": "raw-market",
                "token_id": "YES",
                "metadata_event_id": "metadata_event",
                "resolution_timestamp": resolution,
                "final_resolution_value": 1.0,
            }
        ]
    )

    distribution = collect_scavenger_price_distribution(
        [first_chunk_path, second_chunk_path],
        config=ScavengerConfig(),
        metadata_frame=metadata_frame,
        chunk_size=1,
    )

    assert distribution.height == 1
    distribution_row = distribution.row(0, named=True)
    assert distribution_row["market_id"] == "raw-market"
    assert distribution_row["token_id"] == "YES"
    assert distribution_row["final_result"] == 1
    assert distribution_row["deepest_dip"] == pytest.approx(0.94)
    assert distribution_row["highest_spike"] == pytest.approx(0.98)
    assert distribution_row["observation_count"] == 2