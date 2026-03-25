from pathlib import Path

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
    assert args.train_days == 35
    assert args.test_days == 7
    assert args.step_days == 7
    assert args.embargo_days == 1
    assert args.n_trials == 500
    assert args.trial_timeout_s == 60.0
    assert args.max_markets == 25