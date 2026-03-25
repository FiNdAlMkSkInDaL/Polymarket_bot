from pathlib import Path

import optuna
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

import wfo_bayesian_arb


def test_target_params_match_bayesian_arb_grid():
    assert wfo_bayesian_arb.TARGET_PARAMS == (
        "si10_min_net_edge_usd",
        "si10_maker_ofi_tolerance",
        "si9_latency_option_window_ms",
        "max_cross_book_desync_ms",
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
    assert Path(args.trial_export_path) == Path("data/wfo_bayesian_arb_top25/wfo_trials.json")
    assert args.study_prefix == "polymarket_wfo"
    assert args.checkpoint_interval_trials == 50
    assert args.watch_checkpoints_only is False


def test_build_checkpoint_payloads_prefers_completed_fold(tmp_path: Path) -> None:
    storage_path = tmp_path / "wfo_optuna.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    complete_study = optuna.create_study(
        study_name="polymarket_wfo_fold_0",
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )
    complete_study.add_trial(
        create_trial(
            params={
                "si10_min_net_edge_usd": 0.15,
                "si10_maker_ofi_tolerance": 0.8,
                "si9_latency_option_window_ms": 4000,
                "max_cross_book_desync_ms": 300,
            },
            distributions={
                "si10_min_net_edge_usd": FloatDistribution(0.01, 2.0, log=True),
                "si10_maker_ofi_tolerance": FloatDistribution(0.70, 0.98),
                "si9_latency_option_window_ms": IntDistribution(1000, 10000),
                "max_cross_book_desync_ms": IntDistribution(100, 1200),
            },
            value=1.5,
            state=TrialState.COMPLETE,
        )
    )

    partial_study = optuna.create_study(
        study_name="polymarket_wfo_fold_1",
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )
    partial_study.add_trial(
        create_trial(
            params={
                "si10_min_net_edge_usd": 0.25,
                "si10_maker_ofi_tolerance": 0.9,
                "si9_latency_option_window_ms": 5000,
                "max_cross_book_desync_ms": 250,
            },
            distributions={
                "si10_min_net_edge_usd": FloatDistribution(0.01, 2.0, log=True),
                "si10_maker_ofi_tolerance": FloatDistribution(0.70, 0.98),
                "si9_latency_option_window_ms": IntDistribution(1000, 10000),
                "max_cross_book_desync_ms": IntDistribution(100, 1200),
            },
            value=2.5,
            state=TrialState.COMPLETE,
        )
    )

    report_payload, params_payload = wfo_bayesian_arb._build_checkpoint_payloads(
        storage_path=str(storage_path),
        study_prefix="polymarket_wfo",
        n_trials=1,
    )

    assert report_payload is not None
    assert report_payload["summary"]["complete_folds"] == 2
    assert params_payload is not None
    assert params_payload["meta"]["selection_basis"] == "best_complete_fold_by_objective"
    assert params_payload["meta"]["fold_index"] == 1
    assert params_payload["params"]["si10_min_net_edge_usd"] == 0.25


def test_emit_checkpoint_snapshot_writes_files(tmp_path: Path) -> None:
    storage_path = tmp_path / "wfo_optuna.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study = optuna.create_study(
        study_name="polymarket_wfo_fold_0",
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )
    study.add_trial(
        create_trial(
            params={
                "si10_min_net_edge_usd": 0.11,
                "si10_maker_ofi_tolerance": 0.81,
                "si9_latency_option_window_ms": 4500,
                "max_cross_book_desync_ms": 275,
            },
            distributions={
                "si10_min_net_edge_usd": FloatDistribution(0.01, 2.0, log=True),
                "si10_maker_ofi_tolerance": FloatDistribution(0.70, 0.98),
                "si9_latency_option_window_ms": IntDistribution(1000, 10000),
                "max_cross_book_desync_ms": IntDistribution(100, 1200),
            },
            value=1.0,
            state=TrialState.COMPLETE,
        )
    )

    checkpoint_report_path = tmp_path / "wfo_checkpoint_report.json"
    checkpoint_params_path = tmp_path / "wfo_checkpoint_champion_params.json"
    snapshot = wfo_bayesian_arb._emit_checkpoint_snapshot(
        storage_path=str(storage_path),
        study_prefix="polymarket_wfo",
        n_trials=1,
        checkpoint_report_path=checkpoint_report_path,
        checkpoint_params_path=checkpoint_params_path,
    )

    assert snapshot is not None
    assert checkpoint_report_path.exists()
    assert checkpoint_params_path.exists()


def test_checkpoint_watcher_exits_with_stale_waiting_trials(tmp_path: Path) -> None:
    storage_path = tmp_path / "wfo_optuna.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study = optuna.create_study(
        study_name="polymarket_wfo_fold_0",
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )
    study.add_trial(
        create_trial(
            params={
                "si10_min_net_edge_usd": 0.11,
                "si10_maker_ofi_tolerance": 0.81,
                "si9_latency_option_window_ms": 4500,
                "max_cross_book_desync_ms": 275,
            },
            distributions={
                "si10_min_net_edge_usd": FloatDistribution(0.01, 2.0, log=True),
                "si10_maker_ofi_tolerance": FloatDistribution(0.70, 0.98),
                "si9_latency_option_window_ms": IntDistribution(1000, 10000),
                "max_cross_book_desync_ms": IntDistribution(100, 1200),
            },
            value=1.0,
            state=TrialState.COMPLETE,
        )
    )
    study.enqueue_trial(
        {
            "si10_min_net_edge_usd": 0.12,
            "si10_maker_ofi_tolerance": 0.82,
            "si9_latency_option_window_ms": 4600,
            "max_cross_book_desync_ms": 300,
        }
    )

    checkpoint_report_path = tmp_path / "wfo_checkpoint_report.json"
    checkpoint_params_path = tmp_path / "wfo_checkpoint_champion_params.json"

    wfo_bayesian_arb._checkpoint_watcher_loop(
        storage_path=str(storage_path),
        study_prefix="polymarket_wfo",
        n_trials=1,
        checkpoint_interval_trials=50,
        checkpoint_poll_interval_s=0.01,
        checkpoint_report_path=checkpoint_report_path,
        checkpoint_params_path=checkpoint_params_path,
        stop_event=None,
        stop_when_complete=True,
    )

    assert checkpoint_report_path.exists()
    assert checkpoint_params_path.exists()