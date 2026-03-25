from pathlib import Path

import wfo_bayesian_arb


def test_target_params_match_bayesian_arb_grid():
    assert wfo_bayesian_arb.TARGET_PARAMS == (
        "si10_min_net_edge_usd",
        "si10_maker_ofi_tolerance",
        "si9_latency_option_window_ms",
    )


def test_parse_args_defaults_to_bayesian_wfo_dataset():
    args = wfo_bayesian_arb.parse_args([])

    assert Path(args.data_dir) == Path("data/vps_march2026")
    assert Path(args.market_configs) == Path("data/market_map_top25.json")
    assert Path(args.relationships) == Path("si10_relationships.example.json")
    assert args.train_days == 35
    assert args.test_days == 7
    assert args.step_days == 7
    assert args.embargo_days == 1
    assert args.n_trials == 300
    assert args.trial_timeout_s == 60.0
    assert args.max_markets == 25