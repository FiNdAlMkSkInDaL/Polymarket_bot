from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna

from src.backtest.wfo_optimizer import WfoConfig, run_wfo
import src.backtest.wfo_optimizer as wfo_mod


OUTPUT_DIR = ROOT / "data" / "wfo_si9_wide_spread_2026_03_20_22"
TARGET_IDS_PATH = ROOT / "data" / "si9_target_markets.json"
DATES = ("2026-03-20", "2026-03-21", "2026-03-22")
SEARCH_PARAMS = (
    "pure_mm_wide_tier_enabled",
    "pure_mm_wide_spread_pct",
    "pure_mm_inventory_penalty_coef",
    "pure_mm_toxic_ofi_ratio",
    "pure_mm_depth_evaporation_pct",
)


def _build_market_configs(root: Path, target_ids: list[str]) -> list[dict[str, str]]:
    configs: list[dict[str, str]] = []
    for market_id in target_ids:
        pair: list[str] | None = None
        for date in DATES:
            path = root / "data" / "raw_ticks" / date / f"{market_id}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for _ in range(400):
                    line = handle.readline()
                    if not line:
                        break
                    record = json.loads(line)
                    payload = record.get("payload") or {}
                    changes = payload.get("price_changes") or []
                    asset_ids = sorted(
                        {
                            str(change.get("asset_id") or "")
                            for change in changes
                            if change.get("asset_id")
                        }
                    )
                    if len(asset_ids) >= 2:
                        pair = asset_ids[:2]
                        break
            if pair:
                break
        if pair:
            configs.append(
                {
                    "market_id": market_id,
                    "yes_asset_id": pair[0],
                    "no_asset_id": pair[1],
                }
            )
    return configs


def _export_trials(storage_path: Path, study_prefix: str, report) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    storage_url = f"sqlite:///{storage_path.as_posix()}"
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
                    "state": str(trial.state),
                    "train_dates": fold.train_dates,
                    "test_dates": fold.test_dates,
                    "params": trial.params,
                }
            )
    return rows


def main() -> int:
    os.environ.setdefault("BOT_DISABLE_FILE_LOGGING", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_ids = json.loads(TARGET_IDS_PATH.read_text(encoding="utf-8"))
    market_configs = _build_market_configs(ROOT, target_ids)
    if not market_configs:
        raise RuntimeError("No SI9 target markets with token pairs found in weekend capture")

    print(f"captured_markets={len(market_configs)}")
    for config in market_configs:
        print(json.dumps(config, sort_keys=True))

    storage_path = OUTPUT_DIR / "wfo_optuna.db"
    report_path = OUTPUT_DIR / "wfo_report.json"
    params_path = OUTPUT_DIR / "champion_params.json"
    trials_path = OUTPUT_DIR / "wfo_trials.json"

    original_loader = wfo_mod._load_market_configs
    wfo_mod._load_market_configs = lambda data_dir: list(market_configs)
    try:
        cfg = WfoConfig(
            data_dir="data",
            allowed_dates=DATES,
            train_days=2,
            test_days=1,
            step_days=1,
            embargo_days=0,
            n_trials=100,
            max_workers=8,
            storage_url=f"sqlite:///{storage_path.as_posix()}",
            output_params_path=str(params_path),
            max_markets=None,
            strategy_adapter="pure_market_maker",
            search_space_params=SEARCH_PARAMS,
        )
        report = run_wfo(cfg)
    finally:
        wfo_mod._load_market_configs = original_loader

    payload = asdict(report)
    payload["captured_market_configs"] = market_configs
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    trials_path.write_text(
        json.dumps(_export_trials(storage_path, cfg.study_prefix, report), indent=2),
        encoding="utf-8",
    )

    print(f"aggregate_oos_sharpe={report.aggregate_oos_sharpe}")
    print(f"aggregate_oos_total_pnl={report.aggregate_oos_total_pnl}")
    print(f"aggregate_oos_max_drawdown={report.aggregate_oos_max_drawdown}")
    print(f"champion_fold_index={report.champion_fold_index}")
    print("champion_params=" + json.dumps(report.champion_params, sort_keys=True))
    for fold in report.folds:
        print(
            "fold_result="
            + json.dumps(
                {
                    "fold_index": fold.fold_index,
                    "oos_sharpe": fold.oos_sharpe,
                    "oos_total_pnl": fold.oos_total_pnl,
                    "oos_max_drawdown": fold.oos_max_drawdown,
                    "best_params": fold.best_params,
                },
                sort_keys=True,
            )
        )
    print(f"output_dir={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())