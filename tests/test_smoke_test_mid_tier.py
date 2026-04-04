from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from scripts.smoke_test_mid_tier import run_smoke_test


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


def _write_day_partition(day_dir: Path, base_time: datetime) -> None:
    resolution = base_time + timedelta(hours=6)
    cross = base_time + timedelta(minutes=5)
    rows = [
        _row(
            timestamp=base_time,
            resolution_timestamp=resolution,
            event_id=f"evt-{day_dir.name}",
            market_id=f"mkt-a-{day_dir.name}",
            token_id=f"tok-a-{day_dir.name}",
            best_bid=0.49,
            best_ask=0.50,
            final_resolution_value=0.0,
        ),
        _row(
            timestamp=base_time,
            resolution_timestamp=resolution,
            event_id=f"evt-{day_dir.name}",
            market_id=f"mkt-b-{day_dir.name}",
            token_id=f"tok-b-{day_dir.name}",
            best_bid=0.47,
            best_ask=0.48,
            final_resolution_value=0.0,
        ),
        _row(
            timestamp=base_time,
            resolution_timestamp=resolution,
            event_id=f"evt-{day_dir.name}",
            market_id=f"mkt-c-{day_dir.name}",
            token_id=f"tok-c-{day_dir.name}",
            best_bid=0.10,
            best_ask=0.11,
            final_resolution_value=1.0,
        ),
        _row(
            timestamp=base_time,
            resolution_timestamp=resolution,
            event_id=f"evt-{day_dir.name}",
            market_id=f"mkt-d-{day_dir.name}",
            token_id=f"tok-d-{day_dir.name}",
            best_bid=0.08,
            best_ask=0.09,
            final_resolution_value=0.0,
        ),
        _row(
            timestamp=cross,
            resolution_timestamp=resolution,
            event_id=f"evt-{day_dir.name}",
            market_id=f"mkt-c-{day_dir.name}",
            token_id=f"tok-c-{day_dir.name}",
            best_bid=0.12,
            best_ask=0.13,
            final_resolution_value=1.0,
        ),
        _row(
            timestamp=cross,
            resolution_timestamp=resolution,
            event_id=f"evt-{day_dir.name}",
            market_id=f"mkt-d-{day_dir.name}",
            token_id=f"tok-d-{day_dir.name}",
            best_bid=0.07,
            best_ask=0.08,
            final_resolution_value=0.0,
        ),
    ]
    pl.DataFrame(rows).write_parquet(day_dir / "ticks.parquet")


def test_run_smoke_test_executes_three_day_window_and_writes_frontier(tmp_path: Path) -> None:
    for offset, trade_date in enumerate(["2026-04-01", "2026-04-02", "2026-04-03"], start=1):
        day_dir = tmp_path / trade_date
        day_dir.mkdir()
        _write_day_partition(day_dir, datetime(2026, 4, offset, 12, 0, 0))

    artifacts = run_smoke_test(
        tmp_path,
        output_dir=tmp_path / "smoke_output",
        thresholds=(0.95,),
        notionals=(10.0,),
    )

    assert artifacts.selected_dates == ("2026-04-01", "2026-04-02", "2026-04-03")
    assert artifacts.peak_memory_mb >= 0.0
    assert "Smoke Test Complete. Peak Memory Usage:" in Path(artifacts.markdown_output).read_text(encoding="utf-8")
    assert "Pareto Frontier" in artifacts.pareto_frontier_markdown
    assert Path(artifacts.daily_output).exists()
    assert Path(artifacts.reduced_output).exists()