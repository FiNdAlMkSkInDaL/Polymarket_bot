#!/usr/bin/env python3
"""Convenience entrypoint for SI-10 Bayesian arb WFO on a shared market universe."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.backtest.wfo_optimizer import WfoConfig, WfoReport, run_wfo
from src.core.logger import setup_logging

setup_logging(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

DEFAULT_DATA_DIR = Path("data/vps_march2026")
DEFAULT_MARKET_CONFIGS = Path("data/market_map_top25.json")
DEFAULT_RELATIONSHIPS = Path("si10_relationships.example.json")
DEFAULT_REPORT_DIR = Path("data/wfo_bayesian_arb_top25")
DEFAULT_REPORT_PATH = DEFAULT_REPORT_DIR / "wfo_report.json"
DEFAULT_PARAMS_PATH = DEFAULT_REPORT_DIR / "champion_params.json"
DEFAULT_STORAGE_PATH = DEFAULT_REPORT_DIR / "wfo_optuna.db"
DEFAULT_TRIALS_PATH = DEFAULT_REPORT_DIR / "wfo_trials.json"
DEFAULT_STUDY_PREFIX = "polymarket_wfo"
DEFAULT_CHECKPOINT_INTERVAL_TRIALS = 50
DEFAULT_CHECKPOINT_POLL_INTERVAL_S = 15.0
TARGET_PARAMS = (
    "si10_min_net_edge_usd",
    "si10_maker_ofi_tolerance",
    "si9_latency_option_window_ms",
    "max_cross_book_desync_ms",
)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write(text.encode(encoding, errors="replace") + b"\n")
        sys.stdout.flush()


def _risk_adjusted_ratio(total_pnl: float, max_drawdown: float) -> float:
    if max_drawdown <= 0:
        return float("inf") if total_pnl > 0 else float("-inf")
    return total_pnl / max_drawdown


def _select_best_ratio(report: WfoReport) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for fold in report.folds:
        ratio = _risk_adjusted_ratio(fold.oos_total_pnl, fold.oos_max_drawdown)
        candidate = {
            "fold_index": fold.fold_index,
            "params": {
                name: fold.best_params[name]
                for name in TARGET_PARAMS
                if name in fold.best_params
            },
            "oos_total_pnl": fold.oos_total_pnl,
            "oos_max_drawdown": fold.oos_max_drawdown,
            "oos_ratio": ratio,
            "oos_fills": fold.oos_total_fills,
            "oos_sharpe": fold.oos_sharpe,
        }
        if best is None or candidate["oos_ratio"] > best["oos_ratio"]:
            best = candidate
    return best or {
        "fold_index": -1,
        "params": {},
        "oos_total_pnl": 0.0,
        "oos_max_drawdown": 0.0,
        "oos_ratio": float("-inf"),
        "oos_fills": 0,
        "oos_sharpe": 0.0,
    }


def _report_payload(report: WfoReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["best_return_drawdown"] = _select_best_ratio(report)
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_report_path(report_path: Path) -> Path:
    return report_path.with_name("wfo_checkpoint_report.json")


def _checkpoint_params_path(params_path: Path) -> Path:
    return params_path.with_name("wfo_checkpoint_champion_params.json")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _fold_index_from_study_name(study_name: str, study_prefix: str) -> int:
    prefix = f"{study_prefix}_fold_"
    if not study_name.startswith(prefix):
        return -1
    suffix = study_name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else -1


def _terminal_trials_count(state_counts: dict[str, int]) -> int:
    return sum(
        count
        for state, count in state_counts.items()
        if state not in {"RUNNING", "WAITING"}
    )


def _build_checkpoint_payloads(
    *,
    storage_path: str,
    study_prefix: str,
    n_trials: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    import optuna

    storage_url = f"sqlite:///{Path(storage_path).as_posix()}"
    storage_file = Path(storage_path)
    if not storage_file.exists():
        return None, None

    summaries = [
        summary
        for summary in optuna.study.get_all_study_summaries(storage=storage_url)
        if summary.study_name.startswith(f"{study_prefix}_fold_")
    ]
    if not summaries:
        return None, None

    folds_payload: list[dict[str, Any]] = []
    checkpoint_candidate: dict[str, Any] | None = None
    complete_candidate: dict[str, Any] | None = None
    total_terminal_trials = 0
    complete_folds = 0
    total_running = 0
    total_waiting = 0

    for summary in sorted(summaries, key=lambda item: _fold_index_from_study_name(item.study_name, study_prefix)):
        study = optuna.load_study(study_name=summary.study_name, storage=storage_url)
        state_counts: dict[str, int] = {}
        for trial in study.trials:
            state_name = trial.state.name
            state_counts[state_name] = state_counts.get(state_name, 0) + 1

        terminal_trials = _terminal_trials_count(state_counts)
        running_trials = state_counts.get("RUNNING", 0)
        waiting_trials = state_counts.get("WAITING", 0)
        is_complete = terminal_trials >= n_trials and bool(study.best_trials)
        if is_complete:
            complete_folds += 1

        total_terminal_trials += terminal_trials
        total_running += running_trials
        total_waiting += waiting_trials

        best_value = None
        best_trial_number = None
        best_params: dict[str, Any] = {}
        if study.best_trials:
            best_value = study.best_value
            best_trial_number = study.best_trial.number
            best_params = {
                name: study.best_params[name]
                for name in TARGET_PARAMS
                if name in study.best_params
            }
            candidate = {
                "fold_index": _fold_index_from_study_name(summary.study_name, study_prefix),
                "study_name": summary.study_name,
                "best_value": best_value,
                "best_trial_number": best_trial_number,
                "best_params": best_params,
                "is_complete": is_complete,
            }
            if checkpoint_candidate is None or best_value > checkpoint_candidate["best_value"]:
                checkpoint_candidate = candidate
            if is_complete and (complete_candidate is None or best_value > complete_candidate["best_value"]):
                complete_candidate = candidate

        folds_payload.append(
            {
                "fold_index": _fold_index_from_study_name(summary.study_name, study_prefix),
                "study_name": summary.study_name,
                "trial_state_counts": state_counts,
                "terminal_trials": terminal_trials,
                "remaining_to_target": max(n_trials - terminal_trials, 0),
                "is_complete": is_complete,
                "best_value": best_value,
                "best_trial_number": best_trial_number,
                "best_params": best_params,
            }
        )

    selected_candidate = complete_candidate or checkpoint_candidate
    report_payload = {
        "generated_at": _utc_now_iso(),
        "checkpoint": True,
        "study_prefix": study_prefix,
        "storage_path": str(storage_file),
        "target_params": list(TARGET_PARAMS),
        "target_trials_per_fold": n_trials,
        "summary": {
            "fold_count": len(folds_payload),
            "complete_folds": complete_folds,
            "terminal_trials": total_terminal_trials,
            "running_trials": total_running,
            "waiting_trials": total_waiting,
            "selection_basis": (
                "best_complete_fold_by_objective"
                if complete_candidate is not None
                else "best_partial_fold_by_objective"
            ),
        },
        "folds": folds_payload,
    }

    params_payload = None
    if selected_candidate and selected_candidate["best_params"]:
        params_payload = {
            "params": selected_candidate["best_params"],
            "meta": {
                "checkpoint": True,
                "generated_at": report_payload["generated_at"],
                "selection_basis": report_payload["summary"]["selection_basis"],
                "fold_index": selected_candidate["fold_index"],
                "study_name": selected_candidate["study_name"],
                "best_objective": selected_candidate["best_value"],
                "best_trial_number": selected_candidate["best_trial_number"],
                "fold_complete": selected_candidate["is_complete"],
                "complete_folds": complete_folds,
                "target_trials_per_fold": n_trials,
            },
        }

    return report_payload, params_payload


def _emit_checkpoint_snapshot(
    *,
    storage_path: str,
    study_prefix: str,
    n_trials: int,
    checkpoint_report_path: Path,
    checkpoint_params_path: Path,
) -> dict[str, int] | None:
    report_payload, params_payload = _build_checkpoint_payloads(
        storage_path=storage_path,
        study_prefix=study_prefix,
        n_trials=n_trials,
    )
    if report_payload is None:
        return None

    _write_json_atomic(checkpoint_report_path, report_payload)
    if params_payload is not None:
        _write_json_atomic(checkpoint_params_path, params_payload)

    summary = report_payload["summary"]
    return {
        "fold_count": int(summary["fold_count"]),
        "terminal_trials": int(summary["terminal_trials"]),
        "complete_folds": int(summary["complete_folds"]),
        "running_trials": int(summary["running_trials"]),
        "waiting_trials": int(summary["waiting_trials"]),
    }


def _checkpoint_snapshot_is_complete(snapshot: dict[str, int]) -> bool:
    fold_count = snapshot.get("fold_count", 0)
    if fold_count <= 0:
        return False
    return (
        snapshot["running_trials"] == 0
        and snapshot["complete_folds"] >= fold_count
    )


def _checkpoint_watcher_loop(
    *,
    storage_path: str,
    study_prefix: str,
    n_trials: int,
    checkpoint_interval_trials: int,
    checkpoint_poll_interval_s: float,
    checkpoint_report_path: Path,
    checkpoint_params_path: Path,
    stop_event: threading.Event | None,
    stop_when_complete: bool,
) -> None:
    next_trial_threshold = max(1, checkpoint_interval_trials)
    last_complete_folds = -1
    force_emit = True

    while True:
        snapshot = _emit_checkpoint_snapshot(
            storage_path=storage_path,
            study_prefix=study_prefix,
            n_trials=n_trials,
            checkpoint_report_path=checkpoint_report_path,
            checkpoint_params_path=checkpoint_params_path,
        ) if force_emit else None

        if snapshot is None:
            report_payload, _ = _build_checkpoint_payloads(
                storage_path=storage_path,
                study_prefix=study_prefix,
                n_trials=n_trials,
            )
            if report_payload is not None:
                summary = report_payload["summary"]
                terminal_trials = int(summary["terminal_trials"])
                complete_folds = int(summary["complete_folds"])
                fold_count = int(summary["fold_count"])
                should_emit = (
                    terminal_trials >= next_trial_threshold
                    or complete_folds > last_complete_folds
                    or (
                        stop_when_complete
                        and int(summary["running_trials"]) == 0
                        and complete_folds >= fold_count
                    )
                )
                if should_emit:
                    snapshot = _emit_checkpoint_snapshot(
                        storage_path=storage_path,
                        study_prefix=study_prefix,
                        n_trials=n_trials,
                        checkpoint_report_path=checkpoint_report_path,
                        checkpoint_params_path=checkpoint_params_path,
                    )
        if snapshot is not None:
            terminal_trials = snapshot["terminal_trials"]
            complete_folds = snapshot["complete_folds"]
            if checkpoint_interval_trials > 0:
                next_trial_threshold = ((terminal_trials // checkpoint_interval_trials) + 1) * checkpoint_interval_trials
            last_complete_folds = complete_folds
            force_emit = False
            if stop_when_complete and _checkpoint_snapshot_is_complete(snapshot):
                return

        if stop_event is not None and stop_event.wait(timeout=checkpoint_poll_interval_s):
            return
        if stop_event is None:
            time.sleep(checkpoint_poll_interval_s)


def _trial_payload(storage_path: str, study_prefix: str, report: WfoReport) -> list[dict[str, Any]]:
    import optuna

    rows: list[dict[str, Any]] = []
    storage_url = f"sqlite:///{Path(storage_path).as_posix()}"
    for fold in report.folds:
        study_name = f"{study_prefix}_fold_{fold.fold_index}"
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        for trial in study.trials:
            if trial.value is None:
                continue
            rows.append(
                {
                    "study_name": study_name,
                    "fold_index": fold.fold_index,
                    "trial_number": trial.number,
                    "objective": trial.value,
                    "score": trial.value,
                    "state": str(trial.state),
                    "train_dates": fold.train_dates,
                    "test_dates": fold.test_dates,
                    "params": trial.params,
                }
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SI-10 Bayesian arb WFO on a shared market universe."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dates", nargs="+", default=None)
    parser.add_argument("--train-days", type=int, default=35)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--trial-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-markets", type=int, default=25)
    parser.add_argument("--market-configs", default=str(DEFAULT_MARKET_CONFIGS))
    parser.add_argument("--relationships", default=str(DEFAULT_RELATIONSHIPS))
    parser.add_argument("--storage-path", default=str(DEFAULT_STORAGE_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--params-path", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--trial-export-path", default=str(DEFAULT_TRIALS_PATH))
    parser.add_argument("--study-prefix", default=DEFAULT_STUDY_PREFIX)
    parser.add_argument("--checkpoint-interval-trials", type=int, default=DEFAULT_CHECKPOINT_INTERVAL_TRIALS)
    parser.add_argument("--checkpoint-poll-interval-s", type=float, default=DEFAULT_CHECKPOINT_POLL_INTERVAL_S)
    parser.add_argument("--watch-checkpoints-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.report_path)
    params_path = Path(args.params_path)
    trial_export_path = Path(args.trial_export_path)
    checkpoint_report_path = _checkpoint_report_path(report_path)
    checkpoint_params_path = _checkpoint_params_path(params_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    trial_export_path.parent.mkdir(parents=True, exist_ok=True)

    if args.watch_checkpoints_only:
        _checkpoint_watcher_loop(
            storage_path=args.storage_path,
            study_prefix=args.study_prefix,
            n_trials=args.n_trials,
            checkpoint_interval_trials=args.checkpoint_interval_trials,
            checkpoint_poll_interval_s=args.checkpoint_poll_interval_s,
            checkpoint_report_path=checkpoint_report_path,
            checkpoint_params_path=checkpoint_params_path,
            stop_event=None,
            stop_when_complete=True,
        )
        _safe_print(f"Checkpoint report written to {checkpoint_report_path}")
        if checkpoint_params_path.exists():
            _safe_print(f"Checkpoint params written to {checkpoint_params_path}")
        return 0

    cfg = WfoConfig(
        data_dir=args.data_dir,
        allowed_dates=tuple(args.dates) if args.dates else None,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        n_trials=args.n_trials,
        max_workers=args.max_workers,
        trial_timeout_s=args.trial_timeout_s,
        storage_url=f"sqlite:///{Path(args.storage_path).as_posix()}",
        study_prefix=args.study_prefix,
        output_params_path=str(params_path),
        max_markets=args.max_markets,
        market_configs_path=args.market_configs,
        bayesian_relationships_path=args.relationships,
        strategy_adapter="bayesian_arb",
        search_space_params=TARGET_PARAMS,
    )
    checkpoint_stop_event = threading.Event()
    checkpoint_thread = threading.Thread(
        target=_checkpoint_watcher_loop,
        kwargs={
            "storage_path": args.storage_path,
            "study_prefix": cfg.study_prefix,
            "n_trials": args.n_trials,
            "checkpoint_interval_trials": args.checkpoint_interval_trials,
            "checkpoint_poll_interval_s": args.checkpoint_poll_interval_s,
            "checkpoint_report_path": checkpoint_report_path,
            "checkpoint_params_path": checkpoint_params_path,
            "stop_event": checkpoint_stop_event,
            "stop_when_complete": False,
        },
        daemon=True,
    )
    checkpoint_thread.start()
    try:
        report = run_wfo(cfg)
    finally:
        checkpoint_stop_event.set()
        checkpoint_thread.join(timeout=2.0)
    payload = _report_payload(report)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    trials_payload = _trial_payload(args.storage_path, cfg.study_prefix, report)
    trial_export_path.write_text(json.dumps(trials_payload, indent=2), encoding="utf-8")

    best_ratio = payload["best_return_drawdown"]
    _safe_print(report.summary())
    _safe_print("Best return/drawdown candidate:")
    _safe_print(json.dumps(best_ratio, indent=2))
    _safe_print(f"WFO report written to {report_path}")
    _safe_print(f"WFO trial export written to {trial_export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())