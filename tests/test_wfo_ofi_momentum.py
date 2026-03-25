from pathlib import Path
from unittest.mock import patch

from src.backtest.wfo_optimizer import WfoReport

import wfo_ofi_momentum


def test_target_params_match_ofi_momentum_grid():
    assert wfo_ofi_momentum.TARGET_PARAMS == (
        "ofi_threshold",
        "window_ms",
        "ofi_tvi_kappa",
        "ofi_toxicity_scale_threshold",
        "ofi_toxicity_size_boost_max",
        "take_profit_pct",
        "stop_loss_pct",
    )


def test_parse_args_defaults_to_top25_vps_dataset():
    args = wfo_ofi_momentum.parse_args([])

    assert Path(args.data_dir) == Path("data/vps_march2026")
    assert Path(args.market_configs) == Path("data/market_map_top25.json")
    assert Path(args.search_space_bounds) == Path("data/wfo_ofi_momentum_bounds.json")
    assert args.train_days == 35
    assert args.test_days == 7
    assert args.step_days == 7
    assert args.embargo_days == 1
    assert args.n_trials == 500
    assert args.trial_timeout_s == 60.0
    assert args.max_markets == 25


def test_default_bounds_file_exists_and_is_narrowed_for_ofi():
    bounds_path = Path(wfo_ofi_momentum.DEFAULT_BOUNDS_PATH)
    payload = bounds_path.read_text(encoding="utf-8")

    assert '"ofi_threshold": [0.85, 0.99]' in payload
    assert '"window_ms": [1000, 5000]' in payload


def test_main_passes_params_export_and_bounds_paths(tmp_path):
    report_path = tmp_path / "report.json"
    params_path = tmp_path / "champion_params.json"
    trials_path = tmp_path / "trials.json"
    storage_path = tmp_path / "wfo_optuna.db"
    bounds_path = tmp_path / "bounds.json"
    bounds_path.write_text('{"ofi_threshold": [0.85, 0.99]}', encoding="utf-8")

    report = WfoReport(
        folds=[],
        stitched_equity_curve=[],
        aggregate_oos_sharpe=0.0,
        aggregate_oos_max_drawdown=0.0,
        aggregate_oos_total_pnl=0.0,
        aggregate_oos_win_rate=0.0,
        aggregate_oos_profit_factor=0.0,
        aggregate_oos_trade_count=0,
        parameter_stability={},
        total_elapsed_s=0.0,
        avg_sharpe_decay_pct=0.0,
        overfit_probability=0.0,
        unstable_params=[],
        champion_params={},
        champion_fold_index=-1,
        champion_degradation_pct=0.0,
    )

    with patch("wfo_ofi_momentum.run_wfo", return_value=report) as run_wfo_mock, patch(
        "wfo_ofi_momentum._trial_payload", return_value=[]
    ):
        exit_code = wfo_ofi_momentum.main(
            [
                "--report-path",
                str(report_path),
                "--params-path",
                str(params_path),
                "--trial-export-path",
                str(trials_path),
                "--storage-path",
                str(storage_path),
                "--search-space-bounds",
                str(bounds_path),
            ]
        )

    assert exit_code == 0
    cfg = run_wfo_mock.call_args.args[0]
    assert cfg.output_params_path == str(params_path)
    assert cfg.search_space_bounds_path == str(bounds_path)