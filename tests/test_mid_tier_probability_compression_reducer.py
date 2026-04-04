from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from scripts.reduce_mid_tier_probability_compression_sweep import run_reducer


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_run_reducer_ranks_by_risk_adjusted_score_and_handles_zero_drawdown(tmp_path: Path, suffix: str) -> None:
    panel = pl.DataFrame(
        [
            {
                "trade_date": "2026-04-01",
                "top2_yes_threshold": 0.95,
                "max_leg_notional_usd": 10.0,
                "filled_orders": 10,
                "simulated_exit_fills": 4,
                "winning_fills": 8,
                "fill_win_rate": 0.8,
                "filled_entry_spread_sum": 0.2,
                "total_realized_pnl_usd": 100.0,
                "max_legging_drawdown_usd": 20.0,
            },
            {
                "trade_date": "2026-04-02",
                "top2_yes_threshold": 0.95,
                "max_leg_notional_usd": 10.0,
                "filled_orders": 5,
                "simulated_exit_fills": 1,
                "winning_fills": 2,
                "fill_win_rate": 0.4,
                "filled_entry_spread_sum": 0.05,
                "total_realized_pnl_usd": -20.0,
                "max_legging_drawdown_usd": 10.0,
            },
            {
                "trade_date": "2026-04-01",
                "top2_yes_threshold": 0.96,
                "max_leg_notional_usd": 25.0,
                "filled_orders": 8,
                "simulated_exit_fills": 2,
                "winning_fills": 6,
                "fill_win_rate": 0.75,
                "filled_entry_spread_sum": 0.12,
                "total_realized_pnl_usd": 50.0,
                "max_legging_drawdown_usd": 0.0,
            },
            {
                "trade_date": "2026-04-02",
                "top2_yes_threshold": 0.96,
                "max_leg_notional_usd": 25.0,
                "filled_orders": 4,
                "simulated_exit_fills": 0,
                "winning_fills": 4,
                "fill_win_rate": 1.0,
                "filled_entry_spread_sum": 0.04,
                "total_realized_pnl_usd": 10.0,
                "max_legging_drawdown_usd": 0.0,
            },
            {
                "trade_date": "2026-04-01",
                "top2_yes_threshold": 0.97,
                "max_leg_notional_usd": 50.0,
                "filled_orders": 3,
                "simulated_exit_fills": 0,
                "winning_fills": 0,
                "fill_win_rate": 0.0,
                "filled_entry_spread_sum": 0.09,
                "total_realized_pnl_usd": -5.0,
                "max_legging_drawdown_usd": 0.0,
            },
        ]
    )

    input_path = tmp_path / f"reducer_panel{suffix}"
    if suffix == ".parquet":
        panel.write_parquet(input_path)
    else:
        panel.write_csv(input_path)

    artifacts = run_reducer(input_path)
    rankings = artifacts.rankings
    frontier = artifacts.pareto_frontier

    assert rankings.get_column("top2_yes_threshold").to_list() == [0.96, 0.95, 0.97]
    assert rankings.get_column("max_leg_notional_usd").to_list() == [25.0, 10.0, 50.0]
    assert rankings.get_column("pareto_frontier").to_list() == [True, True, False]
    assert frontier.get_column("top2_yes_threshold").to_list() == [0.96, 0.95]
    assert frontier.get_column("max_leg_notional_usd").to_list() == [25.0, 10.0]

    best = rankings.row(0, named=True)
    assert best["aggregate_net_pnl_usd"] == 60.0
    assert best["total_fills"] == 12
    assert best["simulated_exit_fills"] == 2
    assert best["winning_fills"] == 10
    assert best["win_rate"] == pytest.approx(10.0 / 12.0)
    assert best["avg_filled_entry_spread"] == pytest.approx(0.16 / 12.0)
    assert best["absolute_worst_case_daily_legging_drawdown_usd"] == 0.0
    assert best["risk_adjusted_score"] is None
    assert best["risk_adjusted_score_display"] == "inf"

    second = rankings.row(1, named=True)
    assert second["aggregate_net_pnl_usd"] == 80.0
    assert second["absolute_worst_case_daily_legging_drawdown_usd"] == 20.0
    assert second["risk_adjusted_score"] == pytest.approx(4.0)

    worst = rankings.row(2, named=True)
    assert worst["risk_adjusted_score_display"] == "-inf"