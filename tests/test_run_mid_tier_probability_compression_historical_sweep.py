from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from scripts.run_mid_tier_probability_compression_historical_sweep import run_historical_sweep


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


def test_run_historical_sweep_writes_execution_summary_and_pareto_artifacts(tmp_path: Path) -> None:
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

    artifacts = run_historical_sweep(
        tmp_path,
        output_dir=tmp_path / "historical_output",
        thresholds=(0.95,),
        notionals=(10.0,),
        timestamp_unit="us",
        resolution_timestamp_unit="us",
    )

    execution_summary = json.loads(Path(artifacts.execution_summary_output).read_text(encoding="utf-8"))

    assert Path(artifacts.daily_output).exists()
    assert Path(artifacts.rankings_output).exists()
    assert Path(artifacts.sweep_summary_output).exists()
    assert Path(artifacts.sweep_markdown_output).exists()
    assert Path(artifacts.reducer_summary_output).exists()
    assert Path(artifacts.pareto_markdown_output).exists()
    assert Path(artifacts.execution_summary_output).exists()

    assert execution_summary["days_processed"] == 1
    assert execution_summary["combination_count"] == 1
    assert execution_summary["pareto_frontier_count"] == 1
    assert execution_summary["top3_pareto_coordinates"][0]["top2_yes_threshold"] == 0.95
    assert execution_summary["top3_pareto_coordinates"][0]["aggregate_net_pnl_usd"] == -10.0
    assert execution_summary["memory"]["sweep_peak_memory_mb"] >= 0.0
    assert execution_summary["memory"]["reducer_peak_memory_mb"] >= 0.0
    assert "Pareto Frontier" in Path(artifacts.pareto_markdown_output).read_text(encoding="utf-8")