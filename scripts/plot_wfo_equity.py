#!/usr/bin/env python3
"""Plot the stitched OOS equity curve from a WFO run.

Usage
─────
    # Re-run the WFO replay and plot (uses existing Optuna DB):
    python scripts/plot_wfo_equity.py --data-dir data/vps_march2026/ticks

    # Or provide a pre-computed WFO report JSON:
    python scripts/plot_wfo_equity.py --report wfo_report.json

    # Save to custom path:
    python scripts/plot_wfo_equity.py --data-dir data/vps_march2026/ticks -o my_curve.png
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _load_report_from_json(path: str) -> dict:
    """Load a serialised WfoReport dict from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_equity_curve(
    stitched_curve: list[tuple[float, float]],
    fold_boundaries: list[float],
    output_path: str = "wfo_equity_curve.png",
    initial_cash: float = 1000.0,
) -> None:
    """Plot equity curve with fold boundaries and drawdown subplot."""
    if not stitched_curve:
        print("No equity curve data to plot.")
        return

    timestamps = [datetime.fromtimestamp(ts) for ts, _ in stitched_curve]
    equities = np.array([eq for _, eq in stitched_curve])

    # Drawdown computation
    hwm = np.maximum.accumulate(equities)
    drawdown_pct = np.where(hwm > 0, (hwm - equities) / hwm * 100, 0.0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True
    )
    fig.suptitle("Walk-Forward OOS Equity Curve (Stitched)", fontsize=14, fontweight="bold")

    # Equity curve
    ax1.plot(timestamps, equities, color="#2196F3", linewidth=1.2, label="OOS Equity")
    ax1.axhline(y=initial_cash, color="gray", linestyle="--", alpha=0.5, label=f"Initial (${initial_cash:,.0f})")
    ax1.fill_between(
        timestamps, initial_cash, equities,
        where=equities >= initial_cash, alpha=0.15, color="green",
    )
    ax1.fill_between(
        timestamps, initial_cash, equities,
        where=equities < initial_cash, alpha=0.15, color="red",
    )

    # Fold boundaries
    for i, boundary_ts in enumerate(fold_boundaries):
        boundary_dt = datetime.fromtimestamp(boundary_ts)
        ax1.axvline(x=boundary_dt, color="#FF9800", linestyle=":", alpha=0.7, linewidth=1)
        ax2.axvline(x=boundary_dt, color="#FF9800", linestyle=":", alpha=0.7, linewidth=1)
        ax1.text(
            boundary_dt, ax1.get_ylim()[1] * 0.98, f"F{i + 1}",
            fontsize=8, color="#FF9800", ha="center", va="top",
        )

    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Drawdown subplot
    ax2.fill_between(timestamps, 0, -drawdown_pct, color="#F44336", alpha=0.4)
    ax2.plot(timestamps, -drawdown_pct, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved equity curve to {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WFO OOS equity curve.")
    parser.add_argument(
        "--report", type=str, default=None,
        help="Path to WFO report JSON (strategy_params_optimized.json or wfo_report.json).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to tick data directory (re-runs WFO replay to extract curves).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="wfo_equity_curve.png",
        help="Output image path (default: wfo_equity_curve.png).",
    )
    parser.add_argument(
        "--initial-cash", type=float, default=1000.0,
        help="Initial cash balance for reference line.",
    )
    args = parser.parse_args()

    if args.report:
        report_data = _load_report_from_json(args.report)
        stitched = report_data.get("stitched_equity_curve", [])
        folds = report_data.get("folds", [])
        fold_boundaries = []
        for fold in folds:
            curve = fold.get("oos_equity_curve", [])
            if curve:
                fold_boundaries.append(curve[0][0])
    elif args.data_dir:
        # Re-run the WFO to extract curves (import lazily to avoid
        # pulling in the full bot stack when just plotting from JSON)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.backtest.wfo_optimizer import WfoConfig, run_wfo

        cfg = WfoConfig(
            data_dir=args.data_dir,
            anchored=True,
            train_days=35,
            test_days=7,
            step_days=7,
            embargo_days=1,
            n_trials=5,  # minimal — just replaying best params
            max_workers=1,
            initial_cash=args.initial_cash,
        )
        report = run_wfo(cfg)
        stitched = report.stitched_equity_curve
        fold_boundaries = []
        for fr in report.folds:
            if fr.oos_equity_curve:
                fold_boundaries.append(fr.oos_equity_curve[0][0])
    else:
        parser.error("Provide either --report or --data-dir.")
        return

    plot_equity_curve(
        stitched_curve=stitched,
        fold_boundaries=fold_boundaries,
        output_path=args.output,
        initial_cash=args.initial_cash,
    )


if __name__ == "__main__":
    main()
