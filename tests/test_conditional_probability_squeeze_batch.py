from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from src.backtest.conditional_probability_squeeze import ConditionalProbabilitySqueezeConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_conditional_probability_squeeze_batch as squeeze_batch
import smoke_test_squeeze as squeeze_smoke


def _timestamp(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    millisecond: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, millisecond * 1000, tzinfo=timezone.utc)


def _row(
    timestamp: datetime,
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
        "event_id": f"event-{market_id}",
        "token_id": "YES",
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
    }


def _build_batch_fixture_lake(input_root: Path) -> None:
    fixture_rows: dict[str, list[dict[str, object]]] = {
        "2026-04-01": [
            _row(_timestamp(2026, 4, 1, 8, 0, 0), "parent-alpha", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 0), "child-alpha", best_bid=0.60, best_ask=0.61, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 0, 100), "parent-alpha", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 0, 100), "child-alpha", best_bid=0.60, best_ask=0.61, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 1), "parent-alpha", best_bid=0.65, best_ask=0.66, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 1, 8, 0, 1), "child-alpha", best_bid=0.58, best_ask=0.59, bid_depth=500.0, ask_depth=500.0),
        ],
        "2026-04-02": [
            _row(_timestamp(2026, 4, 2, 9, 0, 0), "parent-beta", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 0), "child-beta", best_bid=0.60, best_ask=0.61, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 0, 100), "parent-beta", best_bid=0.61, best_ask=0.62, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 0, 100), "child-beta", best_bid=0.60, best_ask=0.61, bid_depth=10.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 1), "parent-beta", best_bid=0.55, best_ask=0.56, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 2, 9, 0, 1), "child-beta", best_bid=0.40, best_ask=0.41, bid_depth=500.0, ask_depth=500.0),
        ],
        "2026-04-03": [
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "parent-alpha", best_bid=0.70, best_ask=0.71, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "child-alpha", best_bid=0.20, best_ask=0.21, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "parent-beta", best_bid=0.70, best_ask=0.71, bid_depth=500.0, ask_depth=500.0),
            _row(_timestamp(2026, 4, 3, 12, 0, 0), "child-beta", best_bid=0.20, best_ask=0.21, bid_depth=500.0, ask_depth=500.0),
        ],
    }

    for day_str, rows in fixture_rows.items():
        day_dir = input_root / day_str
        day_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(day_dir / "fixture.parquet")


def _write_mapping_config(config_path: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "parent-alpha": "child-alpha",
                "parent-beta": "child-beta",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _config() -> ConditionalProbabilitySqueezeConfig:
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
        process_by_day=True,
        chunk_days=1,
        warmup_days=1,
        collect_engine="streaming",
    )


def test_batch_runner_ranks_pairs_by_fok_minus_flatten_loss(tmp_path: Path) -> None:
    input_root = tmp_path / "lake"
    config_path = tmp_path / "squeeze_pairs.json"
    output_dir = tmp_path / "output"
    _build_batch_fixture_lake(input_root)
    _write_mapping_config(config_path)

    result = squeeze_batch.run_squeeze_batch(
        input_root,
        config_path,
        _config(),
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 3),
    )

    assert result.summary["pairs_completed"] == 2
    assert result.summary["pairs_failed"] == 0
    assert result.summary["minimum_theoretical_edge_dollars"] == 0.0
    assert result.summary["top_pair_id"] == "parent-alpha__child-alpha"
    assert result.ranking.height == 2

    top_row = result.ranking.row(0, named=True)
    second_row = result.ranking.row(1, named=True)
    assert top_row["pair_id"] == "parent-alpha__child-alpha"
    assert top_row["successful_fok_baskets"] == 1
    assert top_row["partial_fills_requiring_flatten"] == 0
    assert top_row["fok_survival_rate_at_route_latency"] == 1.0
    assert top_row["ranking_net_pnl"] > second_row["ranking_net_pnl"]
    assert second_row["pair_id"] == "parent-beta__child-beta"
    assert second_row["partial_fills_requiring_flatten"] == 1
    assert second_row["flattened_basket_net_loss"] > 0.0

    squeeze_batch.write_batch_outputs(result, output_dir, input_root=input_root, pairs_config_path=config_path)
    markdown = (output_dir / "ranking.md").read_text(encoding="utf-8")
    assert (output_dir / "ranking.csv").exists()
    assert (output_dir / "ranking.parquet").exists()
    assert (output_dir / "batch_summary.json").exists()
    assert "FOK Survival" in markdown
    assert "Minimum theoretical edge dollars: 0.00" in markdown
    assert "parent-alpha__child-alpha" in markdown


def test_batch_cli_default_minimum_theoretical_edge_floor_scales_with_order_size() -> None:
    args = squeeze_batch._parse_args(
        [
            "lake_root",
            "--order-size",
            "500",
        ]
    )

    config = squeeze_batch.build_config_from_args(args)

    assert config.minimum_theoretical_edge_dollars == 10.0


def test_smoke_test_bootstraps_and_prints_report(tmp_path: Path, capsys) -> None:
    input_root = tmp_path / "parquet_lake"
    pairs_config_path = tmp_path / "squeeze_pairs.smoke.json"
    output_dir = tmp_path / "diagnostics"

    report = squeeze_smoke.run_smoke_test(
        input_root=input_root,
        pairs_config_path=pairs_config_path,
        output_dir=output_dir,
    )
    captured = capsys.readouterr()

    assert report.batch_summary["pairs_completed"] == 2
    assert report.batch_summary["top_pair_id"] == "smoke-alpha"
    assert len(report.ranking_preview) == 2
    assert report.peak_memory_mb >= 0.0
    assert (output_dir / "ranking.csv").exists()
    assert (output_dir / "ranking.md").exists()
    assert (output_dir / "batch_summary.json").exists()
    assert "Aggregated Diagnostics Preview:" in captured.out
    assert "Smoke Test Complete. Peak Memory Usage:" in captured.out


def test_batch_runner_accepts_full_lake_root_and_ignores_validation_parquet(tmp_path: Path) -> None:
    input_root = tmp_path / "lake_root"
    l2_book_root = input_root / "l2_book"
    validation_root = input_root / "validation"
    config_path = tmp_path / "squeeze_pairs.json"

    _build_batch_fixture_lake(l2_book_root)
    validation_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"noise": [1]}).write_parquet(validation_root / "broken.parquet")
    _write_mapping_config(config_path)

    result = squeeze_batch.run_squeeze_batch(
        input_root,
        config_path,
        _config(),
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 3),
    )

    assert result.summary["pairs_completed"] == 2
    assert result.summary["scan_root"] == str(l2_book_root)
    assert result.summary["source_file_count"] == 3
    assert all("validation" not in str(path) for path in result.source_files)