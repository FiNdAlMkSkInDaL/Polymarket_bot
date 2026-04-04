from __future__ import annotations

import polars as pl
import pytest

from src.backtest.conditional_probability_squeeze import (
    ConditionalProbabilitySqueezeConfig,
    MarketSlice,
    align_nested_market_books,
    default_minimum_theoretical_edge_dollars,
    run_conditional_probability_squeeze_backtest,
    run_conditional_probability_squeeze_backtest_from_frames,
)


def _market_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


BASE_TS = 1_700_000_000_000


def test_align_nested_market_books_uses_backward_asof_without_lookahead() -> None:
    market_a = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.62,
                "best_ask": 0.63,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.64,
                "best_ask": 0.65,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )
    market_b = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_500,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "B_YES",
                "best_bid": 0.60,
                "best_ask": 0.61,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_500,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "B_YES",
                "best_bid": 0.58,
                "best_ask": 0.59,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )

    aligned = align_nested_market_books(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(max_quote_age_ms=10_000, z_window_events=2),
    ).collect()

    row = aligned.filter(pl.col("timestamp") == BASE_TS + 1_500).to_dicts()[0]
    assert row["quote_ts_a"] == BASE_TS + 1_000
    assert row["quote_ts_b"] == BASE_TS + 1_500
    assert row["best_ask_a"] == pytest.approx(0.63)
    assert row["best_bid_b"] == pytest.approx(0.60)


def test_pretrade_fok_rejects_signal_when_decision_depth_is_missing() -> None:
    market_a = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.61,
                "best_ask": 0.62,
                "bid_depth": 100.0,
                "ask_depth": 5.0,
            }
        ]
    )
    market_b = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "B_YES",
                "best_bid": 0.60,
                "best_ask": 0.61,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            }
        ]
    )

    result = run_conditional_probability_squeeze_backtest_from_frames(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(
            order_size=10.0,
            entry_gap_threshold=0.03,
            entry_zscore_threshold=99.0,
            minimum_edge_over_combined_spread_ratio=0.0,
            route_latency_ms=0,
            max_hold_ms=1_000,
            z_window_events=2,
        ),
    )

    trade = result.trades.to_dicts()[0]
    assert trade["trade_state"] == "rejected_pretrade_depth"
    assert trade["reason"] == "decision_depth"


def test_successful_basket_uses_fill_prices_and_exits_on_recovery() -> None:
    market_a = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.61,
                "best_ask": 0.62,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.65,
                "best_ask": 0.66,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )
    market_b = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "B_YES",
                "best_bid": 0.60,
                "best_ask": 0.61,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "B_YES",
                "best_bid": 0.58,
                "best_ask": 0.59,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )

    result = run_conditional_probability_squeeze_backtest_from_frames(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(
            order_size=10.0,
            entry_gap_threshold=0.03,
            exit_gap_threshold=0.05,
            entry_zscore_threshold=99.0,
            minimum_edge_over_combined_spread_ratio=0.0,
            route_latency_ms=0,
            max_hold_ms=1_000,
            z_window_events=2,
        ),
    )

    trade = result.trades.to_dicts()[0]
    assert trade["trade_state"] == "basket_closed"
    assert trade["exit_type"] == "signal"
    assert trade["entry_price_a"] == pytest.approx(0.62)
    assert trade["entry_price_b"] == pytest.approx(0.60)
    assert trade["exit_price_a"] == pytest.approx(0.65)
    assert trade["exit_price_b"] == pytest.approx(0.59)
    assert trade["net_pnl"] > 0


def test_partial_fill_flattens_orphaned_long_leg_on_next_tick() -> None:
    market_a = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.61,
                "best_ask": 0.62,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 1_100,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.61,
                "best_ask": 0.62,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 1_200,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "A_YES",
                "best_bid": 0.60,
                "best_ask": 0.61,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )
    market_b = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "B_YES",
                "best_bid": 0.60,
                "best_ask": 0.61,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 1_100,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "B_YES",
                "best_bid": 0.60,
                "best_ask": 0.61,
                "bid_depth": 1.0,
                "ask_depth": 100.0,
            },
        ]
    )

    result = run_conditional_probability_squeeze_backtest_from_frames(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(
            order_size=10.0,
            entry_gap_threshold=0.03,
            entry_zscore_threshold=99.0,
            minimum_edge_over_combined_spread_ratio=0.0,
            route_latency_ms=100,
            max_hold_ms=1_000,
            z_window_events=2,
        ),
    )

    trade = result.trades.to_dicts()[0]
    assert trade["trade_state"] == "flattened_stage1"
    assert trade["orphan_leg"] == "a"
    assert trade["flatten_stage"] == 1
    assert trade["entry_price_a"] == pytest.approx(0.62)
    assert trade["exit_price_a"] == pytest.approx(0.60)
    assert trade["net_pnl"] < 0


def test_default_market_slice_filters_to_yes_token(tmp_path: Path) -> None:
    market_a = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.61,
                "best_ask": 0.62,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "NO",
                "best_bid": 0.38,
                "best_ask": 0.39,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.65,
                "best_ask": 0.66,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "NO",
                "best_bid": 0.34,
                "best_ask": 0.35,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )
    market_b = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.60,
                "best_ask": 0.61,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "NO",
                "best_bid": 0.39,
                "best_ask": 0.40,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.58,
                "best_ask": 0.59,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "NO",
                "best_bid": 0.41,
                "best_ask": 0.42,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )

    lake_path = tmp_path / "tokenized_lake.parquet"
    pl.concat([market_a, market_b], how="vertical").write_parquet(lake_path)

    result = run_conditional_probability_squeeze_backtest(
        lake_path,
        market_a=MarketSlice("A", token_id=None),
        market_b=MarketSlice("B", token_id=None),
        config=ConditionalProbabilitySqueezeConfig(
            order_size=10.0,
            entry_gap_threshold=0.03,
            exit_gap_threshold=0.05,
            entry_zscore_threshold=99.0,
            minimum_edge_over_combined_spread_ratio=0.0,
            route_latency_ms=0,
            max_hold_ms=1_000,
            process_by_day=False,
            z_window_events=2,
        ),
        return_aligned=True,
    )

    assert result.aligned is not None
    aligned_row = result.aligned.row(0, named=True)
    assert aligned_row["best_bid_a"] == pytest.approx(0.61)
    assert aligned_row["best_ask_a"] == pytest.approx(0.62)
    assert aligned_row["best_bid_b"] == pytest.approx(0.60)
    assert aligned_row["best_ask_b"] == pytest.approx(0.61)


def test_default_minimum_theoretical_edge_dollars_scales_with_order_size() -> None:
    assert default_minimum_theoretical_edge_dollars(100.0) == pytest.approx(2.0)
    assert default_minimum_theoretical_edge_dollars(500.0) == pytest.approx(10.0)


def test_minimum_edge_filter_blocks_zscore_only_signals() -> None:
    market_a = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.94,
                "best_ask": 0.95,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.94,
                "best_ask": 0.95,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 3_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.79,
                "best_ask": 0.80,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )
    market_b = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.01,
                "best_ask": 0.02,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 2_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.01,
                "best_ask": 0.02,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
            {
                "timestamp": BASE_TS + 3_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.019,
                "best_ask": 0.029,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            },
        ]
    )

    ungated = run_conditional_probability_squeeze_backtest_from_frames(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(
            order_size=10.0,
            entry_gap_threshold=0.03,
            entry_zscore_threshold=0.5,
            minimum_edge_over_combined_spread_ratio=0.0,
            route_latency_ms=0,
            max_hold_ms=1_000,
            process_by_day=False,
            z_window_events=2,
        ),
    )
    gated = run_conditional_probability_squeeze_backtest_from_frames(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(
            order_size=10.0,
            entry_gap_threshold=0.03,
            entry_zscore_threshold=0.5,
            minimum_edge_over_combined_spread_ratio=0.03,
            route_latency_ms=0,
            max_hold_ms=1_000,
            process_by_day=False,
            z_window_events=2,
        ),
    )

    assert ungated.summary["total_valid_signals_generated"] == 1
    assert gated.summary["total_valid_signals_generated"] == 0


def test_absolute_theoretical_edge_floor_blocks_small_crossed_edge() -> None:
    market_a = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "A",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.499,
                "best_ask": 0.500,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            }
        ]
    )
    market_b = _market_frame(
        [
            {
                "timestamp": BASE_TS + 1_000,
                "market_id": "B",
                "event_id": "evt",
                "token_id": "YES",
                "best_bid": 0.515,
                "best_ask": 0.516,
                "bid_depth": 100.0,
                "ask_depth": 100.0,
            }
        ]
    )

    ungated = run_conditional_probability_squeeze_backtest_from_frames(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(
            order_size=100.0,
            entry_gap_threshold=0.03,
            entry_zscore_threshold=99.0,
            minimum_edge_over_combined_spread_ratio=0.0,
            minimum_theoretical_edge_dollars=0.0,
            route_latency_ms=0,
            max_hold_ms=1_000,
            process_by_day=False,
            z_window_events=2,
        ),
    )
    gated = run_conditional_probability_squeeze_backtest_from_frames(
        market_a,
        market_b,
        config=ConditionalProbabilitySqueezeConfig(
            order_size=100.0,
            entry_gap_threshold=0.03,
            entry_zscore_threshold=99.0,
            minimum_edge_over_combined_spread_ratio=0.0,
            minimum_theoretical_edge_dollars=2.0,
            route_latency_ms=0,
            max_hold_ms=1_000,
            process_by_day=False,
            z_window_events=2,
        ),
    )

    assert ungated.summary["total_valid_signals_generated"] == 1
    assert gated.summary["minimum_theoretical_edge_dollars"] == pytest.approx(2.0)
    assert gated.summary["total_valid_signals_generated"] == 0