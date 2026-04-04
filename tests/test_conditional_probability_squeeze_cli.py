from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from src.backtest.conditional_probability_squeeze import (
    ConditionalProbabilitySqueezeConfig,
    MarketSlice,
    run_conditional_probability_squeeze_backtest,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_conditional_probability_squeeze_backtest as squeeze_cli


BASE_TS = 1_700_000_000_000
DAY_MS = 86_400_000


def _row(
    timestamp: int,
    market_id: str,
    *,
    best_bid: float,
    best_ask: float,
    bid_depth: float,
    ask_depth: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "event_id": "event-1",
        "token_id": "YES",
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
    }


def _build_two_day_lake() -> pl.DataFrame:
    day_one = [
        _row(BASE_TS + 1_000, "A", best_bid=0.61, best_ask=0.62, bid_depth=200.0, ask_depth=200.0),
        _row(BASE_TS + 1_100, "A", best_bid=0.61, best_ask=0.62, bid_depth=200.0, ask_depth=200.0),
        _row(BASE_TS + 2_000, "A", best_bid=0.65, best_ask=0.66, bid_depth=200.0, ask_depth=200.0),
        _row(BASE_TS + 1_000, "B", best_bid=0.60, best_ask=0.61, bid_depth=200.0, ask_depth=200.0),
        _row(BASE_TS + 1_100, "B", best_bid=0.60, best_ask=0.61, bid_depth=200.0, ask_depth=200.0),
        _row(BASE_TS + 2_000, "B", best_bid=0.58, best_ask=0.59, bid_depth=200.0, ask_depth=200.0),
    ]
    day_two_base = BASE_TS + DAY_MS
    day_two = [
        _row(day_two_base + 1_000, "A", best_bid=0.61, best_ask=0.62, bid_depth=200.0, ask_depth=200.0),
        _row(day_two_base + 1_100, "A", best_bid=0.61, best_ask=0.62, bid_depth=200.0, ask_depth=200.0),
        _row(day_two_base + 1_200, "A", best_bid=0.55, best_ask=0.56, bid_depth=200.0, ask_depth=200.0),
        _row(day_two_base + 1_000, "B", best_bid=0.60, best_ask=0.61, bid_depth=200.0, ask_depth=200.0),
        _row(day_two_base + 1_100, "B", best_bid=0.60, best_ask=0.61, bid_depth=10.0, ask_depth=200.0),
        _row(day_two_base + 1_200, "B", best_bid=0.40, best_ask=0.41, bid_depth=200.0, ask_depth=200.0),
    ]
    return pl.DataFrame(day_one + day_two).sort(["timestamp", "market_id"])


def _config(*, process_by_day: bool) -> ConditionalProbabilitySqueezeConfig:
    return ConditionalProbabilitySqueezeConfig(
        order_size=100.0,
        entry_gap_threshold=0.025,
        entry_zscore_threshold=10.0,
        minimum_edge_over_combined_spread_ratio=0.0,
        minimum_theoretical_edge_dollars=0.0,
        exit_gap_threshold=0.05,
        exit_zscore_threshold=10.0,
        z_window_events=2,
        route_latency_ms=100,
        max_quote_age_ms=1_000,
        max_hold_ms=5_000,
        process_by_day=process_by_day,
        chunk_days=1,
        warmup_days=1,
        collect_engine="streaming",
    )


def test_config_defaults_to_100ms_route_latency() -> None:
    config = ConditionalProbabilitySqueezeConfig()
    assert config.route_latency_ms == 100
    assert config.route_latency_us == 100_000
    assert config.minimum_theoretical_edge_dollars == 0.0


def test_cli_default_minimum_theoretical_edge_floor_scales_with_order_size() -> None:
    args = squeeze_cli._parse_args(
        [
            "lake.parquet",
            "--market-a-id",
            "A",
            "--market-b-id",
            "B",
            "--order-size",
            "500",
        ]
    )

    config = squeeze_cli.build_config_from_args(args)

    assert config.minimum_theoretical_edge_dollars == pytest.approx(10.0)


def test_source_runner_matches_chunked_and_unchunked_results(tmp_path: Path) -> None:
    lake_path = tmp_path / "lake.parquet"
    _build_two_day_lake().write_parquet(lake_path)

    chunked = run_conditional_probability_squeeze_backtest(
        lake_path,
        market_a=MarketSlice("A"),
        market_b=MarketSlice("B"),
        config=_config(process_by_day=True),
    )
    unchunked = run_conditional_probability_squeeze_backtest(
        lake_path,
        market_a=MarketSlice("A"),
        market_b=MarketSlice("B"),
        config=_config(process_by_day=False),
    )

    assert chunked.summary["route_latency_us"] == 100_000
    assert chunked.summary["chunks_processed"] == 2
    assert chunked.summary["total_valid_signals_generated"] == 2
    assert chunked.summary["successful_fok_baskets"] == 1
    assert chunked.summary["partial_fills_requiring_flatten"] == 1
    assert chunked.summary["route_arrival_full_rejections"] == 0
    assert chunked.summary["flattened_stage1"] == 1

    for key in [
        "total_valid_signals_generated",
        "decision_time_fok_passes",
        "successful_fok_baskets",
        "route_arrival_full_rejections",
        "partial_fills_requiring_flatten",
        "flattened_stage1",
    ]:
        assert chunked.summary[key] == unchunked.summary[key]

    assert chunked.summary["net_pnl"] == pytest.approx(unchunked.summary["net_pnl"])
    assert chunked.trades.height == unchunked.trades.height == 2


def test_cli_writes_expected_diagnostics(tmp_path: Path) -> None:
    lake_path = tmp_path / "lake.parquet"
    output_dir = tmp_path / "diagnostics"
    _build_two_day_lake().write_parquet(lake_path)

    exit_code = squeeze_cli.main(
        [
            str(lake_path),
            "--market-a-id",
            "A",
            "--market-b-id",
            "B",
            "--entry-gap-threshold",
            "0.025",
            "--entry-zscore-threshold",
            "10.0",
            "--minimum-edge-over-combined-spread-ratio",
            "0.0",
            "--minimum-theoretical-edge-dollars",
            "0.0",
            "--exit-gap-threshold",
            "0.05",
            "--exit-zscore-threshold",
            "10.0",
            "--z-window-events",
            "2",
            "--max-quote-age-ms",
            "1000",
            "--max-hold-ms",
            "5000",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "signals.parquet").exists()
    assert (output_dir / "trades.parquet").exists()
    assert (output_dir / "successful_fok_baskets.parquet").exists()
    assert (output_dir / "route_arrival_rejections.parquet").exists()
    assert (output_dir / "flatten_baskets.parquet").exists()
    assert (output_dir / "chunk_stats.parquet").exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    success_baskets = pl.read_parquet(output_dir / "successful_fok_baskets.parquet")
    flatten_baskets = pl.read_parquet(output_dir / "flatten_baskets.parquet")

    assert summary["route_latency_ms"] == 100
    assert summary["route_latency_us"] == 100_000
    assert summary["minimum_theoretical_edge_dollars"] == 0.0
    assert summary["total_valid_signals_generated"] == 2
    assert summary["successful_fok_baskets"] == 1
    assert summary["partial_fills_requiring_flatten"] == 1
    assert "FOK Survival Rate at 100ms" in report
    assert "Minimum theoretical edge dollars: 0.00" in report
    assert success_baskets.height == 1
    assert flatten_baskets.height == 1