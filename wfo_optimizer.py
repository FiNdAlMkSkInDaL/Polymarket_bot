#!/usr/bin/env python3
"""Convenience entrypoint for pure market-maker WFO on the PMXT 7-day dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.backtest.wfo_optimizer import WfoConfig, WfoReport, run_wfo

DEFAULT_REPORT_DIR = Path("data/wfo_mm_last7")
DEFAULT_REPORT_PATH = DEFAULT_REPORT_DIR / "wfo_report.json"
DEFAULT_PARAMS_PATH = DEFAULT_REPORT_DIR / "champion_params.json"
DEFAULT_STORAGE_PATH = DEFAULT_REPORT_DIR / "wfo_optuna.db"
TARGET_PARAMS = (
    "pure_mm_toxic_ofi_ratio",
    "pure_mm_depth_evaporation_pct",
)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure market-maker WFO on the 7-day PMXT dataset.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-days", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-markets", type=int, default=25)
    parser.add_argument("--storage-path", default=str(DEFAULT_STORAGE_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--params-path", default=str(DEFAULT_PARAMS_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.report_path)
    params_path = Path(args.params_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = WfoConfig(
        data_dir=args.data_dir,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        n_trials=args.n_trials,
        max_workers=args.max_workers,
        storage_url=f"sqlite:///{Path(args.storage_path).as_posix()}",
        output_params_path=str(params_path),
        max_markets=args.max_markets,
        strategy_adapter="pure_market_maker",
        search_space_params=TARGET_PARAMS,
    )
    report = run_wfo(cfg)
    payload = _report_payload(report)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best_ratio = payload["best_return_drawdown"]
    print(report.summary())
    print("Best return/drawdown candidate:")
    print(json.dumps(best_ratio, indent=2))
    print(f"WFO report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())