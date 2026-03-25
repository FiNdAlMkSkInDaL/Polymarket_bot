#!/usr/bin/env python3
"""Convenience entrypoint for the Domino contagion arb micro-sweep."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import sys
from pathlib import Path

from src.core.logger import setup_logging

setup_logging(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

from src.backtest.wfo_optimizer import WfoConfig, run_wfo

DEFAULT_DATA_DIR = Path("data/vps_march2026")
DEFAULT_ALLOWED_DATES = (
    "2026-02-26",
    "2026-02-27",
    "2026-02-28",
    "2026-03-01",
    "2026-03-02",
    "2026-03-03",
    "2026-03-04",
)
DEFAULT_MARKET_CONFIGS = Path("data/domino_micro_market_map.json")
DEFAULT_REPORT_DIR = Path("data/wfo_contagion_arb_micro_2026_03_25")
DEFAULT_REPORT_PATH = DEFAULT_REPORT_DIR / "wfo_report.json"
DEFAULT_PARAMS_PATH = DEFAULT_REPORT_DIR / "champion_params.json"
DEFAULT_STORAGE_PATH = DEFAULT_REPORT_DIR / "wfo_optuna.db"
TARGET_PARAMS = (
    "contagion_arb_min_correlation",
    "contagion_arb_trigger_percentile",
    "contagion_arb_min_leader_shift",
    "contagion_arb_min_residual_shift",
    "contagion_arb_toxicity_impulse_scale",
    "max_cross_book_desync_ms",
)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write(text.encode(encoding, errors="replace") + b"\n")
        sys.stdout.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Domino contagion arb micro-sweep on a 5-market debug universe."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_ALLOWED_DATES))
    parser.add_argument("--train-days", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--trial-timeout-s", type=float, default=60.0)
    parser.add_argument("--gap-threshold", type=float, default=0.05)
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--market-configs", default=str(DEFAULT_MARKET_CONFIGS))
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
        allowed_dates=tuple(args.dates) if args.dates else None,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        n_trials=args.n_trials,
        max_workers=args.max_workers,
        trial_timeout_s=args.trial_timeout_s,
        gap_threshold=args.gap_threshold,
        storage_url=f"sqlite:///{Path(args.storage_path).as_posix()}",
        output_params_path=str(params_path),
        max_markets=args.max_markets,
        min_trades=args.min_trades,
        market_configs_path=args.market_configs,
        strategy_adapter="contagion_arb",
        search_space_params=TARGET_PARAMS,
    )
    report = run_wfo(cfg)
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    _safe_print(report.summary())
    _safe_print(f"WFO report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())