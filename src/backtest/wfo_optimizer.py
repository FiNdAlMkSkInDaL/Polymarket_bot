"""
Walk-Forward Optimization (WFO) pipeline — Step 20 of the Upgrade Plan.

Systematically optimises strategy parameters on rolling In-Sample (IS)
windows and evaluates them on Out-of-Sample (OOS) windows using Optuna
with a risk-adjusted objective function.

Architecture
────────────
    1. **Time-Series CV** — rolling (train=30 d, test=7 d, step=7 d).
    2. **Optuna Study per fold** — TPE sampler over 6-parameter search
       space, coordinated via SQLite storage and ``ProcessPoolExecutor``.
    3. **Objective** — Sharpe × drawdown penalty; collapses to 0 when
       ``max_drawdown ≥ MAX_ACCEPTABLE_DRAWDOWN``.
    4. **OOS Stitching** — best-IS params are replayed on OOS data;
       equity curves are concatenated into the "True Backtest".

Public API
──────────
    WfoConfig        – pipeline configuration
    WfoReport        – aggregated results (folds + stitched OOS curve)
    FoldResult       – per-fold IS/OOS metrics + best params
    run_wfo()        – entry point (blocking, uses ProcessPoolExecutor)
    generate_folds() – time-series cross-validation window generator
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from src.core.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WfoConfig:
    """Walk-Forward Optimization configuration.

    Parameters
    ----------
    data_dir:
        Root directory containing ``raw_ticks/YYYY-MM-DD/`` subdirs.
    market_id:
        Polymarket condition ID for the backtest adapter.
    yes_asset_id:
        Token ID for the YES outcome.
    no_asset_id:
        Token ID for the NO outcome.
    train_days:
        Length of each In-Sample training window in days.
    test_days:
        Length of each Out-of-Sample testing window in days.
    step_days:
        How many days to step forward between folds.
    n_trials:
        Total Optuna trials per fold (split across workers).
    max_workers:
        Number of parallel processes.  Defaults to ``cpu_count - 1``.
    max_acceptable_drawdown:
        Drawdown threshold (fraction) above which the objective score
        collapses to zero.
    initial_cash:
        Starting cash balance for each backtest run.
    storage_url:
        Optuna RDB storage URL for cross-process coordination.
    study_prefix:
        Prefix for Optuna study names (one study per fold).
    latency_ms:
        Simulated exchange latency in ms.
    fee_max_pct:
        Maximum Polymarket fee rate %.
    fee_enabled:
        Whether dynamic fees are active.
    """

    data_dir: str = "data"
    market_id: str = "BACKTEST"
    yes_asset_id: str = "YES_TOKEN"
    no_asset_id: str = "NO_TOKEN"

    train_days: int = 30
    test_days: int = 7
    step_days: int = 7

    n_trials: int = 100
    max_workers: int = field(default_factory=lambda: max((os.cpu_count() or 2) - 1, 1))
    max_acceptable_drawdown: float = 0.15

    initial_cash: float = 1000.0
    storage_url: str = "sqlite:///wfo_optuna.db"
    study_prefix: str = "polymarket_wfo"

    latency_ms: float = 150.0
    fee_max_pct: float = 1.56
    fee_enabled: bool = True


# ═══════════════════════════════════════════════════════════════════════════
#  Fold definition
# ═══════════════════════════════════════════════════════════════════════════

class Fold(NamedTuple):
    """A single train/test split."""

    index: int
    train_dates: list[str]  # YYYY-MM-DD strings
    test_dates: list[str]


# ═══════════════════════════════════════════════════════════════════════════
#  Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """Per-fold optimisation results."""

    fold_index: int
    best_params: dict[str, float]
    is_sharpe: float = 0.0
    is_max_drawdown: float = 0.0
    is_total_pnl: float = 0.0
    oos_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_total_pnl: float = 0.0
    is_equity_curve: list[tuple[float, float]] = field(default_factory=list)
    oos_equity_curve: list[tuple[float, float]] = field(default_factory=list)
    n_trials_completed: int = 0
    best_trial_score: float = 0.0
    train_dates: list[str] = field(default_factory=list)
    test_dates: list[str] = field(default_factory=list)


@dataclass
class WfoReport:
    """Aggregated Walk-Forward Optimization report."""

    folds: list[FoldResult] = field(default_factory=list)
    stitched_equity_curve: list[tuple[float, float]] = field(default_factory=list)
    aggregate_oos_sharpe: float = 0.0
    aggregate_oos_max_drawdown: float = 0.0
    aggregate_oos_total_pnl: float = 0.0
    parameter_stability: dict[str, list[float]] = field(default_factory=dict)
    total_elapsed_s: float = 0.0

    def summary(self) -> str:
        """Human-readable WFO report with IS-vs-OOS comparison."""
        lines = [
            "",
            "═" * 80,
            "          WALK-FORWARD OPTIMIZATION REPORT",
            "═" * 80,
            "",
            f"  Folds completed:     {len(self.folds)}",
            f"  Total elapsed:       {self.total_elapsed_s:.1f}s",
            "",
            "  ┌────────┬──────────────┬──────────────┬──────────────┬──────────────┐",
            "  │  Fold  │  IS Sharpe   │  OOS Sharpe  │   IS MaxDD   │  OOS MaxDD   │",
            "  ├────────┼──────────────┼──────────────┼──────────────┼──────────────┤",
        ]

        for fr in self.folds:
            decay = ""
            if fr.is_sharpe != 0:
                pct = ((fr.oos_sharpe - fr.is_sharpe) / abs(fr.is_sharpe)) * 100
                decay = f" ({pct:+.0f}%)"
            lines.append(
                f"  │  {fr.fold_index:>4}  │  {fr.is_sharpe:>+10.2f}  "
                f"│  {fr.oos_sharpe:>+10.2f}  │  {fr.is_max_drawdown:>10.2%}  "
                f"│  {fr.oos_max_drawdown:>10.2%}  │{decay}"
            )

        lines.append(
            "  └────────┴──────────────┴──────────────┴──────────────┴──────────────┘"
        )

        lines.extend([
            "",
            "  Stitched OOS Metrics (True Backtest)",
            "  ─────────────────────────────────────",
            f"    Aggregate Sharpe:      {self.aggregate_oos_sharpe:>+.2f}",
            f"    Aggregate Max DD:      {self.aggregate_oos_max_drawdown:>.2%}",
            f"    Aggregate Total PnL:  ${self.aggregate_oos_total_pnl:>+.2f}",
            "",
        ])

        # Parameter stability
        if self.parameter_stability:
            lines.append("  Parameter Stability Across Folds")
            lines.append("  ────────────────────────────────")
            for pname, vals in sorted(self.parameter_stability.items()):
                if len(vals) > 1:
                    arr = np.array(vals)
                    mu, sd = float(arr.mean()), float(arr.std(ddof=1))
                    cv = sd / abs(mu) if abs(mu) > 1e-9 else float("inf")
                    lines.append(
                        f"    {pname:<30s}  mean={mu:.4f}  std={sd:.4f}  CV={cv:.2f}"
                    )
                else:
                    lines.append(f"    {pname:<30s}  value={vals[0]:.4f}")
            lines.append("")

        lines.append("═" * 80)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Search Space
# ═══════════════════════════════════════════════════════════════════════════

#: Names and ranges for the Optuna search space.
#: Each entry: (suggest_method, low, high)
SEARCH_SPACE: dict[str, tuple[str, float, float]] = {
    "zscore_threshold": ("suggest_float", 1.5, 3.5),
    "spread_compression_pct": ("suggest_float", 0.05, 0.25),
    "stop_loss_cents": ("suggest_float", 2.0, 12.0),
    "trailing_stop_offset_cents": ("suggest_float", 1.0, 5.0),
    "kelly_fraction": ("suggest_float", 0.05, 0.35),
    "max_impact_pct": ("suggest_float", 0.05, 0.25),
}


def _suggest_params(trial: Any) -> dict[str, float]:
    """Sample hyperparameters from the Optuna trial."""
    params: dict[str, float] = {}
    for name, (method, lo, hi) in SEARCH_SPACE.items():
        fn = getattr(trial, method)
        params[name] = fn(name, lo, hi)
    return params


# ═══════════════════════════════════════════════════════════════════════════
#  Time-Series Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════

def generate_folds(
    available_dates: list[str],
    train_days: int = 30,
    test_days: int = 7,
    step_days: int = 7,
) -> list[Fold]:
    """Generate rolling IS/OOS folds from a sorted list of YYYY-MM-DD date strings.

    Parameters
    ----------
    available_dates:
        Sorted list of dates for which recorded tick data exists.
    train_days:
        Number of calendar days in each training window.
    test_days:
        Number of calendar days in each testing window.
    step_days:
        Number of calendar days to step forward between folds.

    Returns
    -------
    list[Fold]:
        Each fold contains the subset of ``available_dates`` that fall
        within its train and test windows.  Folds with empty test sets
        are skipped.
    """
    if not available_dates:
        return []

    all_dates_dt = [datetime.strptime(d, "%Y-%m-%d").date() for d in available_dates]
    first_date = all_dates_dt[0]
    last_date = all_dates_dt[-1]

    folds: list[Fold] = []
    fold_idx = 0
    train_start = first_date

    while True:
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)

        # Stop if test window exceeds available data
        if test_start > last_date:
            break

        # Collect dates that actually have data within each window
        train_dates = [
            d for d in available_dates
            if train_start <= datetime.strptime(d, "%Y-%m-%d").date() <= train_end
        ]
        test_dates = [
            d for d in available_dates
            if test_start <= datetime.strptime(d, "%Y-%m-%d").date() <= test_end
        ]

        # Only emit fold if both windows have data
        if train_dates and test_dates:
            folds.append(Fold(index=fold_idx, train_dates=train_dates, test_dates=test_dates))
            fold_idx += 1

        train_start += timedelta(days=step_days)

    return folds


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loader helpers
# ═══════════════════════════════════════════════════════════════════════════

def _collect_files_for_dates(data_dir: str, dates: list[str]) -> list[Path]:
    """Gather all JSONL tick files for the given date strings."""
    from src.backtest.data_recorder import MarketDataRecorder

    files: list[Path] = []
    for d in dates:
        files.extend(MarketDataRecorder.data_files_for_date(data_dir, d))
    return files


def _build_data_loader(
    data_dir: str,
    dates: list[str],
    asset_ids: set[str] | None = None,
) -> Any:
    """Construct a DataLoader for the given date window.

    Returns ``None`` if no files are found (instead of raising).
    """
    from src.backtest.data_loader import DataLoader

    files = _collect_files_for_dates(data_dir, dates)
    if not files:
        return None
    return DataLoader(files, asset_ids=asset_ids)


# ═══════════════════════════════════════════════════════════════════════════
#  Single-backtest runner (must be picklable → top-level function)
# ═══════════════════════════════════════════════════════════════════════════

def _run_single_backtest(
    data_dir: str,
    dates: list[str],
    param_overrides: dict[str, float],
    market_id: str,
    yes_asset_id: str,
    no_asset_id: str,
    initial_cash: float,
    latency_ms: float,
    fee_max_pct: float,
    fee_enabled: bool,
) -> dict[str, Any] | None:
    """Run one backtest and return serialised metrics.

    This function is designed to be called inside a child process via
    ``ProcessPoolExecutor``.  It returns a plain dict (JSON-safe) so
    that results can cross the process boundary without pickling issues.

    Returns ``None`` if the data loader cannot find any files.
    """
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.strategy import BotReplayAdapter
    from src.core.config import StrategyParams

    loader = _build_data_loader(
        data_dir, dates, asset_ids={yes_asset_id, no_asset_id}
    )
    if loader is None:
        return None

    params = StrategyParams(**param_overrides)

    strategy = BotReplayAdapter(
        market_id=market_id,
        yes_asset_id=yes_asset_id,
        no_asset_id=no_asset_id,
        fee_enabled=fee_enabled,
        initial_bankroll=initial_cash,
        params=params,
    )

    config = BacktestConfig(
        initial_cash=initial_cash,
        latency_ms=latency_ms,
        fee_max_pct=fee_max_pct,
        fee_enabled=fee_enabled,
    )

    engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
    result = engine.run()

    return result.metrics.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
#  Objective function
# ═══════════════════════════════════════════════════════════════════════════

def compute_wfo_score(
    sharpe_ratio: float,
    max_drawdown: float,
    max_acceptable_drawdown: float,
) -> float:
    """Risk-adjusted objective: Sharpe × drawdown penalty.

    .. math::

        \\text{Score} = \\text{Sharpe} \\times \\max\\!\\left(0,\\;
        1 - \\frac{\\text{MaxDD}}{\\text{MaxAcceptableDD}}\\right)

    If ``max_drawdown >= max_acceptable_drawdown`` the score collapses
    to 0, forcing Optuna to discard catastrophic parameter sets.
    """
    penalty = max(0.0, 1.0 - (max_drawdown / max_acceptable_drawdown))
    return sharpe_ratio * penalty


def _objective(
    trial: Any,
    train_dates: list[str],
    wfo_cfg: WfoConfig,
) -> float:
    """Optuna objective — runs an IS backtest and returns the WFO score."""
    suggested = _suggest_params(trial)

    metrics = _run_single_backtest(
        data_dir=wfo_cfg.data_dir,
        dates=train_dates,
        param_overrides=suggested,
        market_id=wfo_cfg.market_id,
        yes_asset_id=wfo_cfg.yes_asset_id,
        no_asset_id=wfo_cfg.no_asset_id,
        initial_cash=wfo_cfg.initial_cash,
        latency_ms=wfo_cfg.latency_ms,
        fee_max_pct=wfo_cfg.fee_max_pct,
        fee_enabled=wfo_cfg.fee_enabled,
    )

    if metrics is None:
        return float("-inf")

    sharpe = metrics.get("sharpe_ratio", 0.0)
    mdd = metrics.get("max_drawdown", 0.0)
    score = compute_wfo_score(sharpe, mdd, wfo_cfg.max_acceptable_drawdown)

    # Log trial outcome
    log.info(
        "wfo_trial",
        trial=trial.number,
        score=round(score, 4),
        sharpe=round(sharpe, 4),
        max_dd=round(mdd, 4),
        params=suggested,
    )

    return score


# ═══════════════════════════════════════════════════════════════════════════
#  Worker function (runs inside a child process)
# ═══════════════════════════════════════════════════════════════════════════

def _worker(
    study_name: str,
    storage: str,
    train_dates: list[str],
    wfo_cfg_dict: dict[str, Any],
    n_trials: int,
) -> None:
    """Optuna worker — loads the shared study and runs ``n_trials``.

    Parameters are passed as plain dicts so that they are safely
    picklable across the process boundary.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    wfo_cfg = WfoConfig(**wfo_cfg_dict)
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(
        lambda trial: _objective(trial, train_dates, wfo_cfg),
        n_trials=n_trials,
        n_jobs=1,  # single-threaded inside each process (avoid GIL)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  OOS equity-curve stitching
# ═══════════════════════════════════════════════════════════════════════════

def _stitch_equity_curves(
    fold_results: list[FoldResult],
    initial_cash: float,
) -> list[tuple[float, float]]:
    """Concatenate OOS equity curves, adjusting for carry-over equity.

    Each fold's OOS curve starts at its own ``initial_cash``.  We
    re-base each curve so that the starting equity of fold *N+1*
    equals the ending equity of fold *N*.
    """
    stitched: list[tuple[float, float]] = []
    carry_equity = initial_cash

    for fr in fold_results:
        curve = fr.oos_equity_curve
        if not curve:
            continue

        # The first point's equity is the IS-ending cash (= initial_cash
        # for each independent run).  Compute the offset so that we
        # continue from where the previous fold ended.
        base_equity = curve[0][1] if curve else initial_cash
        offset = carry_equity - base_equity

        for ts, eq in curve:
            stitched.append((ts, eq + offset))

        # Update carry for the next fold
        if stitched:
            carry_equity = stitched[-1][1]

    return stitched


def _compute_stitched_metrics(
    curve: list[tuple[float, float]],
    initial_cash: float,
) -> tuple[float, float, float]:
    """Compute Sharpe, max-drawdown, total PnL from a stitched equity curve.

    Returns (sharpe_ratio, max_drawdown_frac, total_pnl).
    """
    if len(curve) < 3:
        pnl = (curve[-1][1] - initial_cash) if curve else 0.0
        return 0.0, 0.0, pnl

    equities = np.array([eq for _, eq in curve])
    total_pnl = float(equities[-1] - initial_cash)

    # Max drawdown
    hwm = np.maximum.accumulate(equities)
    dd_pct = np.where(hwm > 0, (hwm - equities) / hwm, 0.0)
    max_dd = float(dd_pct.max())

    # Sharpe
    returns = np.diff(equities) / np.maximum(equities[:-1], 1e-9)
    if len(returns) > 1:
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        # Annualise assuming 1-minute bars (same convention as Telemetry)
        ann = math.sqrt(525_960.0)
        sharpe = (mu / sigma * ann) if sigma > 0 else 0.0
    else:
        sharpe = 0.0

    return sharpe, max_dd, total_pnl


# ═══════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_wfo(cfg: WfoConfig) -> WfoReport:
    """Execute the full Walk-Forward Optimization pipeline.

    1. Enumerate available dates and generate rolling IS/OOS folds.
    2. For each fold:
       a. Create an Optuna study and run parallel trials on IS data.
       b. Extract the best parameters.
       c. Replay IS + OOS with best params for metrics & equity curves.
    3. Stitch OOS curves and compute aggregate metrics.
    4. Return a ``WfoReport``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.monotonic()

    # ── Discover available dates ───────────────────────────────────────
    from src.backtest.data_recorder import MarketDataRecorder

    available = MarketDataRecorder.available_dates(cfg.data_dir)
    if not available:
        raise FileNotFoundError(
            f"No recorded tick data found in {cfg.data_dir}/raw_ticks/"
        )

    log.info("wfo_start", available_dates=len(available), config=cfg)

    folds = generate_folds(
        available,
        train_days=cfg.train_days,
        test_days=cfg.test_days,
        step_days=cfg.step_days,
    )

    if not folds:
        raise ValueError(
            f"Cannot generate any folds from {len(available)} available dates "
            f"with train={cfg.train_days}d / test={cfg.test_days}d / "
            f"step={cfg.step_days}d."
        )

    log.info("wfo_folds_generated", n_folds=len(folds))

    # ── Serialise config for child processes ───────────────────────────
    cfg_dict = {
        "data_dir": cfg.data_dir,
        "market_id": cfg.market_id,
        "yes_asset_id": cfg.yes_asset_id,
        "no_asset_id": cfg.no_asset_id,
        "train_days": cfg.train_days,
        "test_days": cfg.test_days,
        "step_days": cfg.step_days,
        "n_trials": cfg.n_trials,
        "max_workers": cfg.max_workers,
        "max_acceptable_drawdown": cfg.max_acceptable_drawdown,
        "initial_cash": cfg.initial_cash,
        "storage_url": cfg.storage_url,
        "study_prefix": cfg.study_prefix,
        "latency_ms": cfg.latency_ms,
        "fee_max_pct": cfg.fee_max_pct,
        "fee_enabled": cfg.fee_enabled,
    }

    fold_results: list[FoldResult] = []

    for fold in folds:
        log.info(
            "wfo_fold_start",
            fold=fold.index,
            train=f"{fold.train_dates[0]}..{fold.train_dates[-1]}",
            test=f"{fold.test_dates[0]}..{fold.test_dates[-1]}",
        )

        study_name = f"{cfg.study_prefix}_fold_{fold.index}"

        # Create a fresh study for this fold
        study = optuna.create_study(
            study_name=study_name,
            storage=cfg.storage_url,
            direction="maximize",
            load_if_exists=True,
        )

        # ── Parallel trial execution ──────────────────────────────────
        trials_per_worker = max(1, math.ceil(cfg.n_trials / cfg.max_workers))

        if cfg.max_workers > 1:
            with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
                futures = [
                    pool.submit(
                        _worker,
                        study_name,
                        cfg.storage_url,
                        fold.train_dates,
                        cfg_dict,
                        trials_per_worker,
                    )
                    for _ in range(cfg.max_workers)
                ]
                for fut in futures:
                    fut.result()  # propagate exceptions
        else:
            # Single-process mode (simpler debugging / testing)
            study.optimize(
                lambda trial: _objective(trial, fold.train_dates, cfg),
                n_trials=cfg.n_trials,
                n_jobs=1,
            )

        # Re-load study to pick up all worker results
        study = optuna.load_study(study_name=study_name, storage=cfg.storage_url)

        if not study.best_trials:
            log.warning("wfo_fold_no_trials", fold=fold.index)
            continue

        best_params = study.best_params
        best_score = study.best_value

        log.info(
            "wfo_fold_best",
            fold=fold.index,
            score=round(best_score, 4),
            params=best_params,
        )

        # ── Replay IS with best params (for decay comparison) ─────────
        is_metrics = _run_single_backtest(
            data_dir=cfg.data_dir,
            dates=fold.train_dates,
            param_overrides=best_params,
            market_id=cfg.market_id,
            yes_asset_id=cfg.yes_asset_id,
            no_asset_id=cfg.no_asset_id,
            initial_cash=cfg.initial_cash,
            latency_ms=cfg.latency_ms,
            fee_max_pct=cfg.fee_max_pct,
            fee_enabled=cfg.fee_enabled,
        )

        # ── Replay OOS with best params ───────────────────────────────
        oos_metrics = _run_single_backtest(
            data_dir=cfg.data_dir,
            dates=fold.test_dates,
            param_overrides=best_params,
            market_id=cfg.market_id,
            yes_asset_id=cfg.yes_asset_id,
            no_asset_id=cfg.no_asset_id,
            initial_cash=cfg.initial_cash,
            latency_ms=cfg.latency_ms,
            fee_max_pct=cfg.fee_max_pct,
            fee_enabled=cfg.fee_enabled,
        )

        fr = FoldResult(
            fold_index=fold.index,
            best_params=best_params,
            n_trials_completed=len(study.trials),
            best_trial_score=best_score,
            train_dates=fold.train_dates,
            test_dates=fold.test_dates,
        )

        if is_metrics:
            fr.is_sharpe = is_metrics.get("sharpe_ratio", 0.0)
            fr.is_max_drawdown = is_metrics.get("max_drawdown", 0.0)
            fr.is_total_pnl = is_metrics.get("total_pnl", 0.0)
            fr.is_equity_curve = is_metrics.get("equity_curve", [])

        if oos_metrics:
            fr.oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
            fr.oos_max_drawdown = oos_metrics.get("max_drawdown", 0.0)
            fr.oos_total_pnl = oos_metrics.get("total_pnl", 0.0)
            fr.oos_equity_curve = oos_metrics.get("equity_curve", [])

        fold_results.append(fr)

        log.info(
            "wfo_fold_done",
            fold=fold.index,
            is_sharpe=round(fr.is_sharpe, 4),
            oos_sharpe=round(fr.oos_sharpe, 4),
            decay_pct=(
                round(((fr.oos_sharpe - fr.is_sharpe) / abs(fr.is_sharpe)) * 100, 1)
                if abs(fr.is_sharpe) > 1e-6
                else None
            ),
        )

    # ── Stitch OOS equity curves ──────────────────────────────────────
    stitched = _stitch_equity_curves(fold_results, cfg.initial_cash)
    agg_sharpe, agg_dd, agg_pnl = _compute_stitched_metrics(stitched, cfg.initial_cash)

    # ── Parameter stability ───────────────────────────────────────────
    stability: dict[str, list[float]] = {}
    for fr in fold_results:
        for pname, pval in fr.best_params.items():
            stability.setdefault(pname, []).append(pval)

    elapsed = time.monotonic() - t0

    report = WfoReport(
        folds=fold_results,
        stitched_equity_curve=stitched,
        aggregate_oos_sharpe=agg_sharpe,
        aggregate_oos_max_drawdown=agg_dd,
        aggregate_oos_total_pnl=agg_pnl,
        parameter_stability=stability,
        total_elapsed_s=elapsed,
    )

    log.info(
        "wfo_complete",
        folds=len(fold_results),
        oos_sharpe=round(agg_sharpe, 4),
        oos_max_dd=round(agg_dd, 4),
        oos_pnl=round(agg_pnl, 4),
        elapsed_s=round(elapsed, 1),
    )

    return report
