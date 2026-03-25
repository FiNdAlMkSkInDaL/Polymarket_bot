from pathlib import Path

import wfo_contagion_arb


def test_target_params_match_contagion_arb_grid():
    assert wfo_contagion_arb.TARGET_PARAMS == (
        "contagion_arb_min_correlation",
        "contagion_arb_trigger_percentile",
        "contagion_arb_min_leader_shift",
        "contagion_arb_min_residual_shift",
        "contagion_arb_toxicity_impulse_scale",
        "max_cross_book_desync_ms",
    )


def test_parse_args_defaults_to_contagion_wfo_dataset():
    args = wfo_contagion_arb.parse_args([])

    assert Path(args.data_dir) == Path("data/vps_march2026")
    assert Path(args.market_configs) == Path("data/domino_micro_market_map.json")
    assert args.train_days == 5
    assert args.test_days == 1
    assert args.step_days == 1
    assert args.embargo_days == 0
    assert args.n_trials == 30
    assert args.trial_timeout_s == 60.0
    assert args.max_markets == 5