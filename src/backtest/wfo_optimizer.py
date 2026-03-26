"""
Walk-Forward Optimization (WFO) pipeline — enterprise-grade.

Systematically optimises strategy parameters on rolling (or anchored)
In-Sample (IS) windows and evaluates them on embargo-purged Out-of-Sample
(OOS) windows using Optuna with a multi-metric composite objective.

Architecture
────────────
    1. **Time-Series CV** — rolling *or* anchored (expanding IS window).
       Configurable embargo gap between IS and OOS to prevent data
       leakage at the boundary.
    2. **Optuna Study per fold** — TPE sampler with ``MedianPruner``,
       warm-started from the previous fold's best parameters.
       Coordinated via SQLite storage and ``ProcessPoolExecutor``.
    3. **Multi-metric objective** — composite of Sharpe, Sortino,
       profit-factor, and drawdown penalty.  Minimum trade-count gate
       rejects inactive parameter sets.  Log-scale suggestion for
       spread-like parameters.
    4. **Overfitting detection** — IS/OOS Sharpe decay penalty per fold,
       aggregate probability-of-overfitting estimate, and parameter
       stability enforcement (high CV → flagged).
    5. **OOS Stitching** — best-IS params are replayed on OOS data;
       equity curves are concatenated into the "True Backtest".

Public API
──────────
    WfoConfig         – pipeline configuration
    WfoReport         – aggregated results (folds + stitched OOS curve)
    FoldResult        – per-fold IS/OOS metrics + best params
    run_wfo()         – entry point (blocking, uses ProcessPoolExecutor)
    generate_folds()  – time-series cross-validation window generator
    compute_wfo_score – multi-metric composite objective function
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, NamedTuple

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
    embargo_days:
        Gap in calendar days between IS and OOS windows.  Prevents
        autocorrelation leakage at the IS/OOS boundary.  Set to 0
        for backward-compatible behaviour.
    anchored:
        If True, use an expanding (anchored) IS window that always
        starts from the first available date.  If False (default),
        use a rolling window of fixed ``train_days`` width.
    n_trials:
        Total Optuna trials per fold (split across workers).
    max_workers:
        Number of parallel processes.  Defaults to ``cpu_count - 1``.
    max_acceptable_drawdown:
        Drawdown threshold (fraction) above which the objective score
        collapses to zero.
    min_trades:
        Minimum number of fills required for a valid trial.  Trials
        with fewer fills are rejected (score = -inf).
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
    warm_start:
        If True, seed each fold's Optuna study with the previous
        fold's best parameters as an initial trial (enqueue_trial).
    sortino_weight:
        Weight for Sortino ratio in the composite objective [0, 1].
    profit_factor_weight:
        Weight for profit factor in the composite objective [0, 1].
    sharpe_weight:
        Weight for Sharpe ratio in the composite objective [0, 1].
    """

    data_dir: str = "data"
    allowed_dates: tuple[str, ...] | None = None
    market_id: str = "BACKTEST"
    yes_asset_id: str = "YES_TOKEN"
    no_asset_id: str = "NO_TOKEN"

    train_days: int = 30
    test_days: int = 7
    step_days: int = 7
    embargo_days: int = 1
    anchored: bool = False

    n_trials: int = 100
    max_workers: int = field(default_factory=lambda: max((os.cpu_count() or 2) - 1, 1))
    max_acceptable_drawdown: float = 0.15
    min_trades: int = 5

    initial_cash: float = 1000.0
    storage_url: str = "sqlite:///wfo_optuna.db"
    study_prefix: str = "polymarket_wfo"

    latency_ms: float = 150.0
    fee_max_pct: float = 2.00
    fee_enabled: bool = True

    warm_start: bool = True

    # Composite objective weights (must sum to 1.0)
    sharpe_weight: float = 0.50
    sortino_weight: float = 0.30
    profit_factor_weight: float = 0.20

    # sqrt(n_trades) bonus — continuous reward for statistical significance
    trade_bonus_weight: float = 0.05

    # Data-quality gate: abort fold if gap ratio exceeds this threshold
    gap_threshold: float = 0.01       # 1% of events with >5-min gap → abort
    gap_max_interval_s: float = 300.0  # 5-minute gap definition

    # Output path for champion parameters JSON (None = no export)
    output_params_path: str | None = None

    # Limit multi-market universe to N markets (None = use all)
    max_markets: int | None = None

    # Optional JSON file with narrowed search-space bounds (overrides SEARCH_SPACE)
    search_space_bounds_path: str | None = None

    # Optional JSON file with explicit WFO target markets.
    market_configs_path: str | None = None

    # Optional JSON file describing SI-10 triplet relationships.
    bayesian_relationships_path: str | None = None

    # Strategy adapter used by the single-backtest runner.
    # Supported: "bot_replay", "pure_market_maker"
    strategy_adapter: str = "bot_replay"

    # Optional subset of SEARCH_SPACE parameter names to optimise.
    search_space_params: tuple[str, ...] | None = None

    # Hard wall-clock cap for a single trial replay.
    trial_timeout_s: float = 60.0


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
    is_sortino: float = 0.0
    is_win_rate: float = 0.0
    is_profit_factor: float = 0.0
    is_total_fills: int = 0
    oos_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_total_pnl: float = 0.0
    oos_sortino: float = 0.0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_total_fills: int = 0
    is_equity_curve: list[tuple[float, float]] = field(default_factory=list)
    oos_equity_curve: list[tuple[float, float]] = field(default_factory=list)
    n_trials_completed: int = 0
    best_trial_score: float = 0.0
    train_dates: list[str] = field(default_factory=list)
    test_dates: list[str] = field(default_factory=list)
    sharpe_decay_pct: float = 0.0  # (OOS-IS)/|IS| × 100


@dataclass
class WfoReport:
    """Aggregated Walk-Forward Optimization report."""

    folds: list[FoldResult] = field(default_factory=list)
    stitched_equity_curve: list[tuple[float, float]] = field(default_factory=list)
    aggregate_oos_sharpe: float = 0.0
    aggregate_oos_max_drawdown: float = 0.0
    aggregate_oos_total_pnl: float = 0.0
    aggregate_oos_win_rate: float = 0.0
    aggregate_oos_profit_factor: float = 0.0
    aggregate_oos_trade_count: int = 0
    parameter_stability: dict[str, list[float]] = field(default_factory=dict)
    total_elapsed_s: float = 0.0

    # ── Overfitting diagnostics ────────────────────────────────────────
    avg_sharpe_decay_pct: float = 0.0      # mean IS→OOS Sharpe decay %
    overfit_probability: float = 0.0       # fraction of folds where OOS < 0
    unstable_params: list[str] = field(default_factory=list)  # params with CV > 0.50

    # ── Champion selection (lowest OOS degradation) ────────────────────
    champion_params: dict[str, float] = field(default_factory=dict)
    champion_fold_index: int = -1
    champion_degradation_pct: float = 0.0

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
            "  ┌────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┐",
            "  │  Fold  │ IS Sharpe│OOS Sharpe│  IS MaxDD│ OOS MaxDD│ IS Fills │ Decay%     │",
            "  ├────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────┤",
        ]

        for fr in self.folds:
            decay_str = f"{fr.sharpe_decay_pct:+.0f}%" if abs(fr.is_sharpe) > 1e-6 else "N/A"
            lines.append(
                f"  │  {fr.fold_index:>4}  │ {fr.is_sharpe:>+8.2f}│ {fr.oos_sharpe:>+8.2f}"
                f"│ {fr.is_max_drawdown:>8.2%}│ {fr.oos_max_drawdown:>8.2%}"
                f"│ {fr.is_total_fills:>8}│ {decay_str:>10} │"
            )

        lines.append(
            "  └────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────────┘"
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

        # ── Overfitting analysis ──────────────────────────────────────
        lines.append("  Overfitting Analysis")
        lines.append("  ────────────────────")
        lines.append(f"    Avg IS→OOS Sharpe Decay:  {self.avg_sharpe_decay_pct:+.1f}%")
        lines.append(f"    Overfit Probability:       {self.overfit_probability:.1%}")
        if self.unstable_params:
            lines.append(f"    Unstable Parameters:       {', '.join(self.unstable_params)}")
        else:
            lines.append("    Unstable Parameters:       (none — all stable)")
        lines.append("")

        # Parameter stability
        if self.parameter_stability:
            lines.append("  Parameter Stability Across Folds")
            lines.append("  ────────────────────────────────")
            for pname, vals in sorted(self.parameter_stability.items()):
                if len(vals) > 1:
                    arr = np.array(vals)
                    mu, sd = float(arr.mean()), float(arr.std(ddof=1))
                    cv = sd / abs(mu) if abs(mu) > 1e-9 else float("inf")
                    flag = " ⚠" if cv > 0.50 else ""
                    lines.append(
                        f"    {pname:<30s}  mean={mu:.4f}  std={sd:.4f}  CV={cv:.2f}{flag}"
                    )
                else:
                    lines.append(f"    {pname:<30s}  value={vals[0]:.4f}")
            lines.append("")

        # Champion params
        if self.champion_params:
            lines.append("  Champion Parameter Set")
            lines.append("  ──────────────────────")
            lines.append(f"    Selected Fold:             {self.champion_fold_index}")
            lines.append(f"    OOS Degradation:           {self.champion_degradation_pct:.1f}%")
            lines.append("    Parameters:")
            for pname, pval in sorted(self.champion_params.items()):
                lines.append(f"      {pname:<30s}  {pval:.6f}")
            lines.append("")

        lines.append("═" * 80)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Search Space
# ═══════════════════════════════════════════════════════════════════════════

#: Names and ranges for the Optuna search space.
#: Numeric entries use (suggest_method, low, high[, log]).
#: Categorical entries use ("suggest_categorical", choices).
#: Log-scale for parameters where relative magnitude matters more than
#: absolute distance (e.g. kelly_fraction, spread_compression_pct).
SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    # Core signal parameters
    # Hardened bounds (March 9 WFO Reset): admit only statistically
    # significant panics.  Prior [0.15, 1.5] allowed sub-σ noise entries.
    "zscore_threshold": ("suggest_float", 0.2, 2.5),
    "spread_compression_pct": ("suggest_float", 0.02, 0.30, True),     # log-scale
    "volume_ratio_threshold": ("suggest_float", 0.1, 4.0),
    # Trend regime guard (wide range so WFO can find the right threshold)
    "trend_guard_pct": ("suggest_float", 0.05, 1.0),
    # OFI momentum entry confirmation
    "ofi_threshold": ("suggest_float", 0.60, 0.95),
    "window_ms": ("suggest_int", 500, 5000),
    "ofi_tvi_kappa": ("suggest_float", 0.0, 2.0),
    "ofi_toxicity_scale_threshold": ("suggest_float", 0.40, 0.90),
    "ofi_toxicity_size_boost_max": ("suggest_float", 1.0, 3.0),
    # Risk management
    # Hardened: SL must be ≥ min_spread_cents (4.0) to survive the spread.
    # Prior [2.0, 12.0] allowed Chop Trap: positive Sharpe, negative PnL.
    "stop_loss_cents": ("suggest_float", 4.0, 12.0),
    "trailing_stop_offset_cents": ("suggest_float", 0.5, 6.0),
    "kelly_fraction": ("suggest_float", 0.03, 0.40, True),             # log-scale
    "max_impact_pct": ("suggest_float", 0.03, 0.30, True),             # log-scale
    "take_profit_pct": ("suggest_float", 0.01, 0.05),
    "stop_loss_pct": ("suggest_float", 0.01, 0.05),
    # Take-profit
    "alpha_default": ("suggest_float", 0.25, 0.75),
    "tp_vol_sensitivity": ("suggest_float", 0.5, 3.0),
    # Edge quality — institutional filter (hardened March 9)
    "min_edge_score": ("suggest_float", 50.0, 85.0),
    # RPE (Pillar 14)
    "rpe_confidence_threshold": ("suggest_float", 0.03, 0.20),
    "rpe_bayesian_obs_weight": ("suggest_float", 1.0, 15.0),
    "rpe_crypto_vol_default": ("suggest_float", 0.50, 1.20),
    # Drift signal
    "drift_z_threshold": ("suggest_float", 0.5, 2.0),
    "drift_vol_ceiling": ("suggest_float", 0.02, 0.15, True),     # log-scale
    # Pure market maker microstructure
    "pure_mm_wide_tier_enabled": ("suggest_categorical", (True,)),
    "pure_mm_wide_spread_pct": ("suggest_float", 0.10, 0.25),
    "pure_mm_inventory_penalty_coef": ("suggest_float", 0.5, 1.5),
    "pure_mm_toxic_ofi_ratio": ("suggest_float", 0.95, 1.0),
    "pure_mm_depth_evaporation_pct": ("suggest_float", 0.95, 1.0),
    # PCE (Pillar 15)
    "pce_max_portfolio_var_usd": ("suggest_float", 20.0, 100.0),
    "pce_correlation_haircut_threshold": ("suggest_float", 0.30, 0.80),
    "pce_structural_prior_weight": ("suggest_int", 5, 30),
    "pce_holding_period_minutes": ("suggest_int", 30, 360),
    # SI-10: Domino contagion arb
    "contagion_arb_min_correlation": ("suggest_float", 0.30, 0.95),
    "contagion_arb_trigger_percentile": ("suggest_float", 0.50, 0.99),
    "contagion_arb_min_history": ("suggest_int", 8, 64),
    "contagion_arb_min_leader_shift": ("suggest_float", 0.001, 0.03, True),
    "contagion_arb_min_residual_shift": ("suggest_float", 0.001, 0.03, True),
    "contagion_arb_toxicity_impulse_scale": ("suggest_float", 0.01, 0.20, True),
    "contagion_arb_cooldown_seconds": ("suggest_float", 10.0, 180.0),
    "contagion_arb_max_lagging_spread_pct": ("suggest_float", 0.5, 3.0),
    "contagion_arb_max_last_trade_age_s": ("suggest_float", 30.0, 300.0),
    "max_cross_book_desync_ms": ("suggest_int", 100, 1200),
    # SI-10: Bayesian joint-probability arb
    "si10_min_net_edge_usd": ("suggest_float", 0.01, 2.0, True),
    "si10_maker_ofi_tolerance": ("suggest_float", 0.70, 0.98),
    "si9_latency_option_window_ms": ("suggest_int", 1000, 10000),
    # SI-2: Iceberg detector alpha modifiers
    "iceberg_eqs_bonus": ("suggest_float", 0.05, 0.25),
    "iceberg_tp_alpha": ("suggest_float", 0.02, 0.10),
}


def _load_search_space_bounds(path: str | None) -> dict[str, tuple[float, float]] | None:
    """Load narrowed search-space bounds from a JSON file.

    Expected format::

        {
          "zscore_threshold": [0.20, 0.60],
          "kelly_fraction": [0.05, 0.15],
          ...
        }

    Returns ``None`` if *path* is falsy or the file doesn't exist.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("search_space_bounds_not_found", path=path)
        return None
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    bounds: dict[str, tuple[float, float]] = {}
    for name, pair in raw.items():
        if name in SEARCH_SPACE and isinstance(pair, (list, tuple)) and len(pair) == 2:
            bounds[name] = (float(pair[0]), float(pair[1]))
    log.info("search_space_bounds_loaded", path=path, n_overrides=len(bounds))
    return bounds


def _suggest_params(
    trial: Any,
    bounds_override: dict[str, tuple[float, float]] | None = None,
    search_space_params: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Sample hyperparameters from the Optuna trial.

    If *bounds_override* is provided, its ``(lo, hi)`` values replace the
    defaults in ``SEARCH_SPACE`` for the matching parameter names.
    Supports optional 4th element ``log=True`` for log-uniform sampling.
    Categorical specs use ``("suggest_categorical", choices)``.
    """
    params: dict[str, Any] = {}
    selected_names = search_space_params or tuple(SEARCH_SPACE.keys())
    for name in selected_names:
        spec = SEARCH_SPACE[name]
        method = spec[0]
        if method == "suggest_categorical":
            choices = tuple(spec[1])
            if hasattr(trial, "suggest_categorical"):
                params[name] = trial.suggest_categorical(name, choices)
            elif len(choices) == 1:
                params[name] = choices[0]
            else:
                raise AttributeError(f"Trial does not support suggest_categorical for {name}")
            continue
        lo, hi = spec[1], spec[2]
        log_scale = spec[3] if len(spec) > 3 else False  # type: ignore[arg-type]
        # Apply bounds override if available
        if bounds_override and name in bounds_override:
            lo, hi = bounds_override[name]
        if method == "suggest_int":
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            fn = getattr(trial, method)
            if log_scale:
                params[name] = fn(name, lo, hi, log=True)
            else:
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
    embargo_days: int = 0,
    anchored: bool = False,
) -> list[Fold]:
    """Generate rolling (or anchored) IS/OOS folds from sorted date strings.

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
    embargo_days:
        Gap between IS end and OOS start (prevents boundary leakage).
        The embargo period is excluded from both IS and OOS.
    anchored:
        If True, IS window always starts from ``available_dates[0]``
        (expanding train window).  ``train_days`` is still the *minimum*
        IS width — the first fold won't fire until IS ≥ train_days.

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
        # In anchored mode, IS always starts from first_date
        if anchored:
            effective_train_start = first_date
        else:
            effective_train_start = train_start

        train_end = train_start + timedelta(days=train_days - 1)

        # Embargo gap between IS and OOS
        test_start = train_end + timedelta(days=1 + embargo_days)
        test_end = test_start + timedelta(days=test_days - 1)

        # Stop if test window exceeds available data
        if test_start > last_date:
            break

        # Collect dates that actually have data within each window
        train_dates = [
            d for d in available_dates
            if effective_train_start <= datetime.strptime(d, "%Y-%m-%d").date() <= train_end
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

def _collect_files_for_dates(
    data_dir: str,
    dates: list[str],
    *,
    market_id: str | None = None,
    asset_ids: set[str] | None = None,
) -> list[Path]:
    """Gather all tick files (JSONL and Parquet) for the given date strings."""
    from src.backtest.data_recorder import MarketDataRecorder

    files: list[Path] = []
    base = Path(data_dir)
    target_ids = {str(value) for value in (asset_ids or set()) if value}
    if market_id:
        target_ids.add(str(market_id))

    def _include_file(path: Path) -> bool:
        if not target_ids:
            return True
        return path.stem in target_ids

    for d in dates:
        # Raw JSONL from raw_ticks/<date>/
        files.extend(
            path for path in MarketDataRecorder.data_files_for_date(data_dir, d)
            if _include_file(path)
        )
        # Processed Parquet from <data_dir>/<date>/
        parquet_dir = base / d
        if parquet_dir.exists():
            files.extend(path for path in sorted(parquet_dir.glob("*.parquet")) if _include_file(path))
    return files


def _build_data_loader(
    data_dir: str,
    dates: list[str],
    market_id: str | None = None,
    asset_ids: set[str] | None = None,
) -> Any:
    """Construct a DataLoader for the given date window.

    Returns ``None`` if no files are found (instead of raising).
    """
    from src.backtest.data_loader import DataLoader

    files = _collect_files_for_dates(
        data_dir,
        dates,
        market_id=market_id,
        asset_ids=asset_ids,
    )
    if not files:
        return None
    return DataLoader(files, asset_ids=asset_ids)


def _compute_gap_ratio(
    data_dir: str,
    dates: list[str],
    market_id: str | None,
    asset_ids: set[str],
    max_interval_s: float,
) -> float:
    """Scan tick files for the given dates and return the fraction of events
    where the timestamp gap from the previous event exceeds ``max_interval_s``.

    This is a lightweight pre-scan — it reads timestamps only (no full
    backtest replay) to decide whether the data window is too sparse.
    """
    from src.backtest.data_loader import DataLoader

    files = _collect_files_for_dates(
        data_dir,
        dates,
        market_id=market_id,
        asset_ids=asset_ids,
    )
    if not files:
        return 0.0

    loader = DataLoader(files, asset_ids=asset_ids)

    total = 0
    gaps = 0
    prev_ts = 0.0
    for event in loader:
        total += 1
        if prev_ts > 0 and (event.timestamp - prev_ts) > max_interval_s:
            gaps += 1
        prev_ts = event.timestamp

    return gaps / total if total > 0 else 0.0


def _normalise_market_configs(raw: list[dict[str, Any]], source: Path) -> list[dict[str, str]]:
    """Normalise market config JSON into {market_id, yes_asset_id, no_asset_id} rows."""
    configs: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        market_id = str(entry.get("market_id") or "").strip()
        yes_asset_id = str(entry.get("yes_asset_id") or entry.get("yes_id") or "").strip()
        no_asset_id = str(entry.get("no_asset_id") or entry.get("no_id") or "").strip()
        if yes_asset_id and no_asset_id:
            configs.append(
                {
                    "market_id": market_id or f"TARGET_{index}_{yes_asset_id}_{no_asset_id}",
                    "yes_asset_id": yes_asset_id,
                    "no_asset_id": no_asset_id,
                    "question": str(entry.get("question") or ""),
                    "event_id": str(entry.get("event_id") or entry.get("group") or ""),
                    "tags": str(entry.get("tags") or entry.get("theme") or ""),
                    "liquidity_usd": float(entry.get("liquidity_usd", 0.0) or 0.0),
                    "daily_volume_usd": float(entry.get("daily_volume_usd", 0.0) or 0.0),
                    "accepting_orders": bool(entry.get("accepting_orders", True)),
                    "end_date": entry.get("end_date") or entry.get("end_date_iso") or entry.get("end_time"),
                }
            )
    if configs:
        log.info("wfo_market_configs_loaded", n_markets=len(configs), path=str(source))
    return configs


def _load_market_configs(data_dir: str, market_configs_path: str | None = None) -> list[dict]:
    """Load market configs and return {market_id, yes_asset_id, no_asset_id} rows.

    If ``market_configs_path`` is provided, that file is used directly.
    Otherwise searches ``data_dir``, its parent, and the ``data/`` directory
    relative to the current working directory for ``market_map.json``.
    Returns an empty list when nothing valid is found.
    """
    import json

    candidates: list[Path]
    if market_configs_path:
        candidates = [Path(market_configs_path)]
    else:
        base = Path(data_dir)
        candidates = [base / "market_map.json", base.parent / "market_map.json", Path("data") / "market_map.json"]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                return _normalise_market_configs(raw, candidate)
            log.warning("wfo_market_configs_invalid", path=str(candidate), reason="expected JSON list")
            return []
    return []


def _load_bayesian_relationships(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    candidate = Path(path)
    if not candidate.exists():
        log.warning("wfo_bayesian_relationships_not_found", path=path)
        return []
    with open(candidate, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        log.warning("wfo_bayesian_relationships_invalid", path=path)
        return []
    return [item for item in raw if isinstance(item, dict)]


def _run_multi_market_backtest(
    data_dir: str,
    dates: list[str],
    param_overrides: dict,
    market_configs: list[dict],
    initial_cash: float,
    latency_ms: float,
    fee_max_pct: float,
    fee_enabled: bool,
    strategy_adapter: str = "bot_replay",
    gap_threshold: float = 0.01,
    gap_max_interval_s: float = 300.0,
    bayesian_relationships: list[dict[str, Any]] | None = None,
    stochastic_seed: int | None = None,
) -> dict | None:
    """Run one backtest per market config and return averaged metrics.

    Markets that produce zero fills are excluded from the average.
    Returns ``None`` when no market yields any fills.
    """
    if strategy_adapter == "contagion_arb":
        return _run_contagion_multi_market_backtest(
            data_dir=data_dir,
            dates=dates,
            param_overrides=param_overrides,
            market_configs=market_configs,
            initial_cash=initial_cash,
            latency_ms=latency_ms,
            fee_max_pct=fee_max_pct,
            fee_enabled=fee_enabled,
            gap_threshold=gap_threshold,
            gap_max_interval_s=gap_max_interval_s,
        )
    if strategy_adapter == "bayesian_arb":
        return _run_bayesian_multi_market_backtest(
            data_dir=data_dir,
            dates=dates,
            param_overrides=param_overrides,
            market_configs=market_configs,
            initial_cash=initial_cash,
            latency_ms=latency_ms,
            fee_max_pct=fee_max_pct,
            fee_enabled=fee_enabled,
            gap_threshold=gap_threshold,
            gap_max_interval_s=gap_max_interval_s,
            bayesian_relationships=bayesian_relationships,
        )

    all_metrics = []
    for mc in market_configs:
        result = _run_single_backtest(
            data_dir=data_dir,
            dates=dates,
            param_overrides=param_overrides,
            market_id=mc["market_id"],
            yes_asset_id=mc["yes_asset_id"],
            no_asset_id=mc["no_asset_id"],
            initial_cash=initial_cash,
            latency_ms=latency_ms,
            fee_max_pct=fee_max_pct,
            fee_enabled=fee_enabled,
            strategy_adapter=strategy_adapter,
            gap_threshold=gap_threshold,
            gap_max_interval_s=gap_max_interval_s,
            stochastic_seed=stochastic_seed,
        )
        if result is not None and result.get("total_fills", 0) >= 1:
            all_metrics.append(result)

    if not all_metrics:
        return None

    # Average scalar metrics across active markets; sum additive counters
    ref = all_metrics[0]
    avg: dict = {}
    for key, val in ref.items():
        if key == "total_fills":
            avg[key] = sum(m.get(key, 0) for m in all_metrics)
        elif isinstance(val, (int, float)):
            values = [m.get(key, 0) for m in all_metrics]
            avg[key] = sum(values) / len(values)
        else:
            avg[key] = val  # non-numeric (e.g. equity_curve) → first market's value
    return avg


def _run_contagion_multi_market_backtest(
    data_dir: str,
    dates: list[str],
    param_overrides: dict[str, float],
    market_configs: list[dict[str, Any]],
    initial_cash: float,
    latency_ms: float,
    fee_max_pct: float,
    fee_enabled: bool,
    gap_threshold: float = 0.01,
    gap_max_interval_s: float = 300.0,
) -> dict[str, Any] | None:
    """Run a true shared replay for the contagion arb across all configured markets."""
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.strategy import ContagionReplayAdapter, split_strategy_and_legacy_params
    from src.core.config import StrategyParams

    asset_ids: set[str] = set()
    for config in market_configs:
        asset_ids.add(str(config.get("yes_asset_id") or ""))
        asset_ids.add(str(config.get("no_asset_id") or ""))
    asset_ids.discard("")
    loader = _build_data_loader(data_dir, dates, asset_ids=asset_ids)
    if loader is None:
        return None

    if gap_threshold > 0:
        gap_ratio = _compute_gap_ratio(
            data_dir,
            dates,
            None,
            asset_ids,
            gap_max_interval_s,
        )
        if gap_ratio > gap_threshold:
            log.warning(
                "wfo_data_quality_abort",
                dates=f"{dates[0]}..{dates[-1]}",
                gap_ratio=round(gap_ratio, 4),
                threshold=gap_threshold,
                strategy_adapter="contagion_arb",
            )
            return None

    strategy_param_overrides, _legacy_signal_params = split_strategy_and_legacy_params(
        param_overrides
    )
    params = StrategyParams(**strategy_param_overrides)
    strategy = ContagionReplayAdapter(
        market_configs=market_configs,
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
    metrics = result.metrics.to_dict()
    metrics["contagion_detector_diagnostics"] = strategy.detector_diagnostics()
    return metrics


def _run_bayesian_multi_market_backtest(
    data_dir: str,
    dates: list[str],
    param_overrides: dict[str, float],
    market_configs: list[dict[str, Any]],
    initial_cash: float,
    latency_ms: float,
    fee_max_pct: float,
    fee_enabled: bool,
    gap_threshold: float = 0.01,
    gap_max_interval_s: float = 300.0,
    bayesian_relationships: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Run a shared replay for SI-10 across all configured markets."""
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.strategy import BayesianReplayAdapter, split_strategy_and_legacy_params
    from src.core.config import StrategyParams

    if not bayesian_relationships:
        return None

    asset_ids: set[str] = set()
    for config in market_configs:
        asset_ids.add(str(config.get("yes_asset_id") or ""))
        asset_ids.add(str(config.get("no_asset_id") or ""))
    asset_ids.discard("")
    loader = _build_data_loader(data_dir, dates, asset_ids=asset_ids)
    if loader is None:
        return None

    if gap_threshold > 0:
        gap_ratio = _compute_gap_ratio(
            data_dir,
            dates,
            None,
            asset_ids,
            gap_max_interval_s,
        )
        if gap_ratio > gap_threshold:
            log.warning(
                "wfo_data_quality_abort",
                dates=f"{dates[0]}..{dates[-1]}",
                gap_ratio=round(gap_ratio, 4),
                threshold=gap_threshold,
                strategy_adapter="bayesian_arb",
            )
            return None

    strategy_param_overrides, _legacy_signal_params = split_strategy_and_legacy_params(
        param_overrides
    )
    params = StrategyParams(**strategy_param_overrides)
    strategy = BayesianReplayAdapter(
        market_configs=market_configs,
        relationships=bayesian_relationships,
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


def _trial_backtest_worker(
    queue: Any,
    *,
    multi_market: bool,
    kwargs: dict[str, Any],
) -> None:
    """Run a trial replay in a separate process so it can be hard-killed."""
    try:
        if multi_market:
            result = _run_multi_market_backtest(**kwargs)
        else:
            result = _run_single_backtest(**kwargs)
        queue.put(("ok", result))
    except Exception as exc:  # pragma: no cover - defensive child-process path
        queue.put(("error", repr(exc)))


def _run_backtest_with_timeout(
    *,
    timeout_s: float,
    multi_market: bool,
    kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any] | None | str]:
    """Execute a backtest in a child process and enforce a wall-clock timeout."""
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    proc = ctx.Process(
        target=_trial_backtest_worker,
        kwargs={
            "queue": queue,
            "multi_market": multi_market,
            "kwargs": kwargs,
        },
    )
    proc.start()

    try:
        proc.join(timeout_s)
    except KeyboardInterrupt:
        if proc.is_alive():
            proc.terminate()
            proc.join(5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(5.0)
        return ("interrupted", None)

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(5.0)
        return ("timeout", None)

    if queue.empty():
        return ("ok", None)

    status, payload = queue.get()
    return status, payload


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
    strategy_adapter: str = "bot_replay",
    gap_threshold: float = 0.01,
    gap_max_interval_s: float = 300.0,
    bayesian_relationships: list[dict[str, Any]] | None = None,
    stochastic_seed: int | None = None,
) -> dict[str, Any] | None:
    """Run one backtest and return serialised metrics.

    This function is designed to be called inside a child process via
    ``ProcessPoolExecutor``.  It returns a plain dict (JSON-safe) so
    that results can cross the process boundary without pickling issues.

    Returns ``None`` if the data loader cannot find any files or if
    the gap/stale-data ratio exceeds ``gap_threshold``.
    """
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.strategy import (
        BotReplayAdapter,
        ContagionReplayAdapter,
        PureMarketMakerReplayAdapter,
        split_strategy_and_legacy_params,
    )
    from src.core.config import StrategyParams

    del bayesian_relationships

    loader = _build_data_loader(
        data_dir,
        dates,
        market_id=market_id,
        asset_ids={yes_asset_id, no_asset_id},
    )
    if loader is None:
        return None

    # ── Pre-scan for gap/stale data quality ─────────────────────────
    if gap_threshold > 0:
        gap_ratio = _compute_gap_ratio(
            data_dir,
            dates,
            market_id,
            {yes_asset_id, no_asset_id},
            gap_max_interval_s,
        )
        if gap_ratio > gap_threshold:
            log.warning(
                "wfo_data_quality_abort",
                dates=f"{dates[0]}..{dates[-1]}",
                gap_ratio=round(gap_ratio, 4),
                threshold=gap_threshold,
            )
            return None

    strategy_param_overrides, legacy_signal_params = split_strategy_and_legacy_params(
        param_overrides
    )
    params = StrategyParams(**strategy_param_overrides)

    if strategy_adapter == "contagion_arb":
        return None
    if strategy_adapter == "pure_market_maker":
        strategy = PureMarketMakerReplayAdapter(
            market_id=market_id,
            yes_asset_id=yes_asset_id,
            no_asset_id=no_asset_id,
            fee_enabled=fee_enabled,
            initial_bankroll=initial_cash,
            params=params,
            legacy_signal_params=legacy_signal_params,
        )
    else:
        strategy = BotReplayAdapter(
            market_id=market_id,
            yes_asset_id=yes_asset_id,
            no_asset_id=no_asset_id,
            fee_enabled=fee_enabled,
            initial_bankroll=initial_cash,
            params=params,
            legacy_signal_params=legacy_signal_params,
            stochastic_seed=stochastic_seed,
        )

    config = BacktestConfig(
        initial_cash=initial_cash,
        latency_ms=latency_ms,
        fee_max_pct=fee_max_pct,
        fee_enabled=fee_enabled,
    )

    engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
    result = engine.run()

    if strategy_adapter == "pure_market_maker" and not getattr(strategy, "has_l2_book", False):
        log.warning(
            "wfo_pure_mm_missing_l2",
            dates=f"{dates[0]}..{dates[-1]}",
            data_dir=data_dir,
        )
        return None

    return result.metrics.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
#  Objective function
# ═══════════════════════════════════════════════════════════════════════════

def compute_wfo_score(
    sharpe_ratio: float,
    max_drawdown: float,
    max_acceptable_drawdown: float,
    *,
    sortino_ratio: float = 0.0,
    profit_factor: float = 0.0,
    total_fills: int = 0,
    min_trades: int = 5,
    sharpe_weight: float = 0.50,
    sortino_weight: float = 0.30,
    profit_factor_weight: float = 0.20,
    trade_bonus_weight: float = 0.05,
) -> float:
    """Multi-metric composite objective with drawdown penalty, trade gate,
    and sqrt(n_trades) bonus.

    .. math::

        \\text{Composite} = w_S \\cdot \\text{Sharpe} +
                            w_{So} \\cdot \\text{Sortino} +
                            w_{PF} \\cdot \\ln(1 + \\text{PF}) +
                            w_T \\cdot \\sqrt{\\text{fills}}

        \\text{Score} = \\text{Composite} \\times \\max\\!\\left(0,\\;
        1 - \\frac{\\text{MaxDD}}{\\text{MaxAcceptableDD}}\\right)

    Returns ``-inf`` when ``total_fills < min_trades`` (inactivity gate).
    If ``max_drawdown >= max_acceptable_drawdown`` the score collapses
    to 0, forcing Optuna to discard catastrophic parameter sets.

    ``profit_factor`` is transformed via ``ln(1 + PF)`` to normalise
    its scale relative to Sharpe/Sortino (PF > 2 is excellent, PF=1
    is breakeven).  ``trade_bonus_weight * sqrt(fills)`` continuously
    rewards parameter sets that generate more trades for statistical
    significance.
    """
    # Gate: reject parameter sets that produce too few trades
    if total_fills < min_trades:
        return float("-inf")

    # Drawdown penalty: linear collapse from 1 → 0
    dd_penalty = max(0.0, 1.0 - (max_drawdown / max_acceptable_drawdown))

    # Normalised profit factor via log transform
    pf_score = math.log(1.0 + max(profit_factor, 0.0))

    # Weighted composite with sqrt(n_trades) bonus
    composite = (
        sharpe_weight * sharpe_ratio
        + sortino_weight * sortino_ratio
        + profit_factor_weight * pf_score
        + trade_bonus_weight * math.sqrt(total_fills)
    )

    return composite * dd_penalty


ObjectiveRejectionReason = Literal[
    "below_min_trades",
    "no_metrics",
    "drawdown_collapse",
    "valid",
]


def _record_objective_trial_attrs(
    trial: Any,
    *,
    total_fills: int,
    rejection_reason: ObjectiveRejectionReason,
    raw_sharpe: float | None = None,
    max_drawdown: float | None = None,
    profit_factor: float | None = None,
) -> None:
    trial.set_user_attr("total_fills", int(total_fills))
    trial.set_user_attr("rejection_reason", rejection_reason)
    if rejection_reason == "drawdown_collapse":
        trial.set_user_attr("raw_sharpe", float(raw_sharpe or 0.0))
        trial.set_user_attr("max_drawdown", float(max_drawdown or 0.0))
        trial.set_user_attr("profit_factor", float(profit_factor or 0.0))


def _objective(
    trial: Any,
    train_dates: list[str],
    wfo_cfg: WfoConfig,
    market_configs: list[dict] | None = None,
    bounds_override: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Optuna objective — runs an IS backtest and returns the WFO score."""
    suggested = _suggest_params(
        trial,
        bounds_override=bounds_override,
        search_space_params=wfo_cfg.search_space_params,
    )

    _common = dict(
        data_dir=wfo_cfg.data_dir,
        dates=train_dates,
        param_overrides=suggested,
        initial_cash=wfo_cfg.initial_cash,
        latency_ms=wfo_cfg.latency_ms,
        fee_max_pct=wfo_cfg.fee_max_pct,
        fee_enabled=wfo_cfg.fee_enabled,
        strategy_adapter=wfo_cfg.strategy_adapter,
        gap_threshold=wfo_cfg.gap_threshold,
        gap_max_interval_s=wfo_cfg.gap_max_interval_s,
        bayesian_relationships=_load_bayesian_relationships(wfo_cfg.bayesian_relationships_path),
        stochastic_seed=trial.number,
    )

    if market_configs:
        run_status, payload = _run_backtest_with_timeout(
            timeout_s=wfo_cfg.trial_timeout_s,
            multi_market=True,
            kwargs={"market_configs": market_configs, **_common},
        )
    else:
        run_status, payload = _run_backtest_with_timeout(
            timeout_s=wfo_cfg.trial_timeout_s,
            multi_market=False,
            kwargs={
                "market_id": wfo_cfg.market_id,
                "yes_asset_id": wfo_cfg.yes_asset_id,
                "no_asset_id": wfo_cfg.no_asset_id,
                **_common,
            },
        )

    if run_status == "timeout":
        _record_objective_trial_attrs(
            trial,
            total_fills=0,
            rejection_reason="no_metrics",
        )
        trial.set_user_attr("timed_out", True)
        trial.set_user_attr("timed_out_params", suggested)
        log.warning(
            "wfo_trial_timeout",
            trial=trial.number,
            timeout_s=wfo_cfg.trial_timeout_s,
            window_ms=suggested.get("window_ms"),
            params=suggested,
        )
        return -9999.0

    if run_status == "interrupted":
        _record_objective_trial_attrs(
            trial,
            total_fills=0,
            rejection_reason="no_metrics",
        )
        trial.set_user_attr("interrupted", True)
        trial.set_user_attr("interrupted_params", suggested)
        log.warning(
            "wfo_trial_join_interrupted",
            trial=trial.number,
            timeout_s=wfo_cfg.trial_timeout_s,
            window_ms=suggested.get("window_ms"),
            params=suggested,
        )
        return -9999.0

    if run_status == "error":
        _record_objective_trial_attrs(
            trial,
            total_fills=0,
            rejection_reason="no_metrics",
        )
        log.warning(
            "wfo_trial_worker_error",
            trial=trial.number,
            window_ms=suggested.get("window_ms"),
            params=suggested,
            error=payload,
        )
        return float("-inf")

    metrics = payload

    if metrics is None:
        _record_objective_trial_attrs(
            trial,
            total_fills=0,
            rejection_reason="no_metrics",
        )
        return float("-inf")

    sharpe = metrics.get("sharpe_ratio", 0.0)
    sortino = metrics.get("sortino_ratio", 0.0)
    mdd = metrics.get("max_drawdown", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    fills = metrics.get("total_fills", 0)

    rejection_reason: ObjectiveRejectionReason = "valid"
    if fills < wfo_cfg.min_trades:
        rejection_reason = "below_min_trades"
    else:
        dd_penalty = max(0.0, 1.0 - (mdd / wfo_cfg.max_acceptable_drawdown))
        if dd_penalty == 0.0:
            rejection_reason = "drawdown_collapse"

    score = compute_wfo_score(
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        max_acceptable_drawdown=wfo_cfg.max_acceptable_drawdown,
        sortino_ratio=sortino,
        profit_factor=pf,
        total_fills=fills,
        min_trades=wfo_cfg.min_trades,
        sharpe_weight=wfo_cfg.sharpe_weight,
        sortino_weight=wfo_cfg.sortino_weight,
        profit_factor_weight=wfo_cfg.profit_factor_weight,
        trade_bonus_weight=wfo_cfg.trade_bonus_weight,
    )

    _record_objective_trial_attrs(
        trial,
        total_fills=int(fills),
        rejection_reason=rejection_reason,
        raw_sharpe=sharpe,
        max_drawdown=mdd,
        profit_factor=pf,
    )

    # Log trial outcome
    log.info(
        "wfo_trial",
        trial=trial.number,
        score=round(score, 4) if score != float("-inf") else "-inf",
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        max_dd=round(mdd, 4),
        pf=round(pf, 2),
        fills=fills,
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
    market_configs: list[dict] | None = None,
    bounds_override: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Optuna worker — loads the shared study and runs ``n_trials``.

    Parameters are passed as plain dicts so that they are safely
    picklable across the process boundary.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    wfo_cfg = WfoConfig(**wfo_cfg_dict)
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Manual ask/tell loop instead of study.optimize() to handle race
    # conditions on the warm-start enqueued trial shared across workers.
    for _ in range(n_trials):
        trial = None
        try:
            trial = study.ask()
            value = _objective(trial, train_dates, wfo_cfg, market_configs, bounds_override)
            study.tell(trial, value)
        except ValueError as exc:
            if "COMPLETE" in str(exc):
                # Race: another worker already completed this trial — skip it.
                log.warning("wfo_worker_trial_race", error=str(exc))
            else:
                # Unexpected ValueError: mark trial failed so it doesn't stay RUNNING.
                if trial is not None:
                    try:
                        study.tell(trial, float("-inf"))
                    except Exception:
                        pass
                raise
        except Exception:
            # Any other error: mark trial failed before re-raising.
            if trial is not None:
                try:
                    study.tell(trial, float("-inf"))
                except Exception:
                    pass
            raise


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


def _cleanup_stale_running_trials(study: Any) -> int:
    """Mark stale RUNNING trials as FAIL before resuming a fold study."""
    import optuna

    running_trials = study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.RUNNING,),
    )
    if not running_trials:
        return 0

    cleaned = 0
    cleaned_numbers: list[int] = []
    for frozen_trial in running_trials:
        trial_id = getattr(frozen_trial, "_trial_id", None)
        if trial_id is None:
            continue
        try:
            transitioned = study._storage.set_trial_state_values(  # type: ignore[attr-defined]
                trial_id,
                optuna.trial.TrialState.FAIL,
                values=None,
            )
        except Exception as exc:
            log.warning(
                "wfo_stale_trial_cleanup_failed",
                study_name=study.study_name,
                trial_number=frozen_trial.number,
                error=str(exc),
            )
            continue

        if transitioned:
            cleaned += 1
            cleaned_numbers.append(frozen_trial.number)

    if cleaned:
        log.warning(
            "wfo_stale_trials_cleaned",
            study_name=study.study_name,
            cleaned_trials=cleaned,
            trial_numbers=cleaned_numbers,
        )

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_wfo(cfg: WfoConfig) -> WfoReport:
    """Execute the full Walk-Forward Optimization pipeline.

    1. Enumerate available dates and generate rolling/anchored IS/OOS folds
       with configurable embargo gap.
    2. For each fold:
       a. Create an Optuna study (with MedianPruner) and optionally
          warm-start from the previous fold's best parameters.
       b. Run parallel trials on IS data.
       c. Extract the best parameters.
       d. Replay IS + OOS with best params for metrics & equity curves.
    3. Stitch OOS curves and compute aggregate metrics.
    4. Compute overfitting diagnostics (decay, probability, stability).
    5. Return a ``WfoReport``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.monotonic()

    # ── Load search-space bounds override (if provided) ────────────────
    bounds_override = _load_search_space_bounds(cfg.search_space_bounds_path)

    # ── Discover available dates ───────────────────────────────────────
    from src.backtest.data_recorder import MarketDataRecorder

    available = MarketDataRecorder.available_dates(cfg.data_dir)
    if cfg.allowed_dates is not None:
        requested_dates = list(dict.fromkeys(cfg.allowed_dates))
        requested_set = set(requested_dates)
        all_available = available
        available = [date_str for date_str in all_available if date_str in requested_set]
        missing_dates = [date_str for date_str in requested_dates if date_str not in requested_set.intersection(all_available)]
        if missing_dates:
            raise FileNotFoundError(
                "Requested WFO dates were not found in "
                f"{cfg.data_dir}/raw_ticks/: {', '.join(missing_dates)}"
            )
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
        embargo_days=cfg.embargo_days,
        anchored=cfg.anchored,
    )

    if not folds:
        raise ValueError(
            f"Cannot generate any folds from {len(available)} available dates "
            f"with train={cfg.train_days}d / test={cfg.test_days}d / "
            f"step={cfg.step_days}d / embargo={cfg.embargo_days}d."
        )

    log.info("wfo_folds_generated", n_folds=len(folds))

    # ── Load market configs (auto-detect from market_map.json) ─────────
    market_configs: list[dict] | None = None
    _SYNTH_PLACEHOLDERS = {"YES_TOKEN", "0x" + "a1" * 16}
    if cfg.market_configs_path or cfg.yes_asset_id in _SYNTH_PLACEHOLDERS:
        # No specific market provided → multi-market mode
        loaded = _load_market_configs(cfg.data_dir, cfg.market_configs_path)
        if loaded:
            if cfg.max_markets is not None and cfg.max_markets < len(loaded):
                loaded = loaded[:cfg.max_markets]
                log.info("wfo_market_universe_capped", max_markets=cfg.max_markets)
            market_configs = loaded
            log.info("wfo_multi_market_mode", n_markets=len(market_configs))
        else:
            log.warning(
                "wfo_no_market_configs",
                msg=(
                    f"market config file not found: {cfg.market_configs_path}"
                    if cfg.market_configs_path
                    else "market_map.json not found; using placeholder IDs"
                ),
            )

    # ── Serialise config for child processes ───────────────────────────
    cfg_dict = {
        "data_dir": cfg.data_dir,
        "allowed_dates": cfg.allowed_dates,
        "market_id": cfg.market_id,
        "yes_asset_id": cfg.yes_asset_id,
        "no_asset_id": cfg.no_asset_id,
        "train_days": cfg.train_days,
        "test_days": cfg.test_days,
        "step_days": cfg.step_days,
        "embargo_days": cfg.embargo_days,
        "anchored": cfg.anchored,
        "n_trials": cfg.n_trials,
        "max_workers": cfg.max_workers,
        "max_acceptable_drawdown": cfg.max_acceptable_drawdown,
        "min_trades": cfg.min_trades,
        "initial_cash": cfg.initial_cash,
        "storage_url": cfg.storage_url,
        "study_prefix": cfg.study_prefix,
        "latency_ms": cfg.latency_ms,
        "fee_max_pct": cfg.fee_max_pct,
        "fee_enabled": cfg.fee_enabled,
        "warm_start": cfg.warm_start,
        "sharpe_weight": cfg.sharpe_weight,
        "sortino_weight": cfg.sortino_weight,
        "profit_factor_weight": cfg.profit_factor_weight,
        "trade_bonus_weight": cfg.trade_bonus_weight,
        "gap_threshold": cfg.gap_threshold,
        "gap_max_interval_s": cfg.gap_max_interval_s,
        "output_params_path": cfg.output_params_path,
        "market_configs_path": cfg.market_configs_path,
        "bayesian_relationships_path": cfg.bayesian_relationships_path,
        "search_space_bounds_path": cfg.search_space_bounds_path,
        "strategy_adapter": cfg.strategy_adapter,
        "search_space_params": cfg.search_space_params,
        "trial_timeout_s": cfg.trial_timeout_s,
    }

    fold_results: list[FoldResult] = []
    prev_best_params: dict[str, float] | None = None

    for fold in folds:
        log.info(
            "wfo_fold_start",
            fold=fold.index,
            train=f"{fold.train_dates[0]}..{fold.train_dates[-1]}",
            test=f"{fold.test_dates[0]}..{fold.test_dates[-1]}",
        )

        study_name = f"{cfg.study_prefix}_fold_{fold.index}"

        # Create a fresh study with MedianPruner for early stopping
        study = optuna.create_study(
            study_name=study_name,
            storage=cfg.storage_url,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
            ),
        )

        _cleanup_stale_running_trials(study)

        # ── Warm-start: seed with previous fold's best params ─────────
        if cfg.warm_start and prev_best_params is not None:
            try:
                study.enqueue_trial(prev_best_params)
                log.info(
                    "wfo_warm_start",
                    fold=fold.index,
                    seed_params=prev_best_params,
                )
            except Exception:
                # If enqueue fails (e.g. param name mismatch), proceed
                log.warning("wfo_warm_start_failed", fold=fold.index)

        # ── Parallel trial execution ──────────────────────────────────
        completed_trials = sum(
            1
            for trial in study.trials
            if trial.state != optuna.trial.TrialState.WAITING
        )
        remaining_trials = max(cfg.n_trials - completed_trials, 0)

        if remaining_trials == 0:
            log.info(
                "wfo_fold_trials_already_complete",
                fold=fold.index,
                existing_trials=completed_trials,
                target_trials=cfg.n_trials,
            )
        else:
            log.info(
                "wfo_fold_resume_progress",
                fold=fold.index,
                existing_trials=completed_trials,
                remaining_trials=remaining_trials,
                target_trials=cfg.n_trials,
            )

        if remaining_trials > 0:
            trials_per_worker = max(1, math.ceil(remaining_trials / cfg.max_workers))

            if cfg.max_workers > 1:
                worker_count = min(cfg.max_workers, remaining_trials)
                with ProcessPoolExecutor(max_workers=worker_count) as pool:
                    futures = [
                        pool.submit(
                            _worker,
                            study_name,
                            cfg.storage_url,
                            fold.train_dates,
                            cfg_dict,
                            trials_per_worker,
                            market_configs,
                            bounds_override,
                        )
                        for _ in range(worker_count)
                    ]
                    for fut in futures:
                        fut.result()  # propagate exceptions
            else:
                # Single-process mode (simpler debugging / testing)
                study.optimize(
                    lambda trial: _objective(trial, fold.train_dates, cfg, market_configs, bounds_override),
                    n_trials=remaining_trials,
                    n_jobs=1,
                )

        # Re-load study to pick up all worker results
        study = optuna.load_study(study_name=study_name, storage=cfg.storage_url)

        if not study.best_trials:
            log.warning("wfo_fold_no_trials", fold=fold.index)
            continue

        best_params = study.best_params
        best_score = study.best_value
        best_trial_number = study.best_trial.number
        prev_best_params = dict(best_params)  # for warm-start

        log.info(
            "wfo_fold_best",
            fold=fold.index,
            score=round(best_score, 4),
            params=best_params,
        )

        # ── Helper to dispatch single vs multi-market replay ──────────
        _replay_kwargs = dict(
            data_dir=cfg.data_dir,
            param_overrides=best_params,
            initial_cash=cfg.initial_cash,
            latency_ms=cfg.latency_ms,
            fee_max_pct=cfg.fee_max_pct,
            fee_enabled=cfg.fee_enabled,
            strategy_adapter=cfg.strategy_adapter,
            gap_threshold=cfg.gap_threshold,
            gap_max_interval_s=cfg.gap_max_interval_s,
            bayesian_relationships=_load_bayesian_relationships(cfg.bayesian_relationships_path),
            stochastic_seed=best_trial_number,
        )

        def _replay(dates: list[str]) -> dict | None:
            if market_configs:
                run_status, payload = _run_backtest_with_timeout(
                    timeout_s=cfg.trial_timeout_s,
                    multi_market=True,
                    kwargs={"market_configs": market_configs, "dates": dates, **_replay_kwargs},
                )
            else:
                run_status, payload = _run_backtest_with_timeout(
                    timeout_s=cfg.trial_timeout_s,
                    multi_market=False,
                    kwargs={
                        "dates": dates,
                        "market_id": cfg.market_id,
                        "yes_asset_id": cfg.yes_asset_id,
                        "no_asset_id": cfg.no_asset_id,
                        **_replay_kwargs,
                    },
                )

            if run_status == "ok":
                return payload if isinstance(payload, dict) or payload is None else None

            log.warning(
                "wfo_replay_unavailable",
                status=run_status,
                dates=f"{dates[0]}..{dates[-1]}",
                timeout_s=cfg.trial_timeout_s,
                fold=fold.index,
            )
            return None

        # ── Replay IS with best params (for decay comparison) ─────────
        is_metrics = _replay(fold.train_dates)

        # ── Replay OOS with best params ───────────────────────────────
        oos_metrics = _replay(fold.test_dates)

        fr = FoldResult(
            fold_index=fold.index,
            best_params=best_params,
            n_trials_completed=sum(
                1
                for trial in study.trials
                if trial.state != optuna.trial.TrialState.WAITING
            ),
            best_trial_score=best_score,
            train_dates=fold.train_dates,
            test_dates=fold.test_dates,
        )

        if is_metrics:
            fr.is_sharpe = is_metrics.get("sharpe_ratio", 0.0)
            fr.is_max_drawdown = is_metrics.get("max_drawdown", 0.0)
            fr.is_total_pnl = is_metrics.get("total_pnl", 0.0)
            fr.is_sortino = is_metrics.get("sortino_ratio", 0.0)
            fr.is_win_rate = is_metrics.get("win_rate", 0.0)
            fr.is_profit_factor = is_metrics.get("profit_factor", 0.0)
            fr.is_total_fills = is_metrics.get("total_fills", 0)
            fr.is_equity_curve = is_metrics.get("equity_curve", [])

        if oos_metrics:
            fr.oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
            fr.oos_max_drawdown = oos_metrics.get("max_drawdown", 0.0)
            fr.oos_total_pnl = oos_metrics.get("total_pnl", 0.0)
            fr.oos_sortino = oos_metrics.get("sortino_ratio", 0.0)
            fr.oos_win_rate = oos_metrics.get("win_rate", 0.0)
            fr.oos_profit_factor = oos_metrics.get("profit_factor", 0.0)
            fr.oos_total_fills = oos_metrics.get("total_fills", 0)
            fr.oos_equity_curve = oos_metrics.get("equity_curve", [])

        # Compute IS→OOS Sharpe decay
        if abs(fr.is_sharpe) > 1e-6:
            fr.sharpe_decay_pct = ((fr.oos_sharpe - fr.is_sharpe) / abs(fr.is_sharpe)) * 100
        else:
            fr.sharpe_decay_pct = 0.0

        fold_results.append(fr)

        log.info(
            "wfo_fold_done",
            fold=fold.index,
            is_sharpe=round(fr.is_sharpe, 4),
            oos_sharpe=round(fr.oos_sharpe, 4),
            decay_pct=round(fr.sharpe_decay_pct, 1),
            is_fills=fr.is_total_fills,
        )

    # ── Stitch OOS equity curves ──────────────────────────────────────
    stitched = _stitch_equity_curves(fold_results, cfg.initial_cash)
    agg_sharpe, agg_dd, agg_pnl = _compute_stitched_metrics(stitched, cfg.initial_cash)

    # ── Parameter stability ───────────────────────────────────────────
    stability: dict[str, list[float]] = {}
    for fr in fold_results:
        for pname, pval in fr.best_params.items():
            stability.setdefault(pname, []).append(pval)

    # ── Overfitting diagnostics ───────────────────────────────────────
    sharpe_decays = [fr.sharpe_decay_pct for fr in fold_results if abs(fr.is_sharpe) > 1e-6]
    avg_decay = float(np.mean(sharpe_decays)) if sharpe_decays else 0.0

    total_oos_trades = int(sum(fr.oos_total_fills for fr in fold_results))
    if total_oos_trades > 0:
        agg_oos_win_rate = float(
            sum(fr.oos_win_rate * fr.oos_total_fills for fr in fold_results) / total_oos_trades
        )
        agg_oos_profit_factor = float(
            sum(fr.oos_profit_factor * fr.oos_total_fills for fr in fold_results) / total_oos_trades
        )
    else:
        agg_oos_win_rate = 0.0
        agg_oos_profit_factor = 0.0

    # Probability of overfitting: fraction of folds where OOS Sharpe < 0
    n_overfit = sum(1 for fr in fold_results if fr.oos_sharpe < 0)
    overfit_prob = n_overfit / len(fold_results) if fold_results else 0.0

    # Unstable parameters: CV > 0.50
    unstable: list[str] = []
    _STABILITY_CV_THRESHOLD = 0.50
    for pname, vals in stability.items():
        if len(vals) > 1:
            arr = np.array(vals)
            mu = float(arr.mean())
            sd = float(arr.std(ddof=1))
            cv = sd / abs(mu) if abs(mu) > 1e-9 else float("inf")
            if cv > _STABILITY_CV_THRESHOLD:
                unstable.append(pname)

    elapsed = time.monotonic() - t0

    # ── Champion selection: lowest OOS degradation ────────────────────
    champion_params: dict[str, float] = {}
    champion_fold_idx = -1
    champion_degradation = float("inf")

    for fr in fold_results:
        if fr.oos_sharpe <= 0:
            continue  # require positive OOS Sharpe

        # Compute IS composite score for this fold's best params
        is_composite = compute_wfo_score(
            sharpe_ratio=fr.is_sharpe,
            max_drawdown=fr.is_max_drawdown,
            max_acceptable_drawdown=cfg.max_acceptable_drawdown,
            sortino_ratio=fr.is_sortino,
            profit_factor=fr.is_profit_factor,
            total_fills=fr.is_total_fills,
            min_trades=1,  # don't gate — already validated
            sharpe_weight=cfg.sharpe_weight,
            sortino_weight=cfg.sortino_weight,
            profit_factor_weight=cfg.profit_factor_weight,
            trade_bonus_weight=cfg.trade_bonus_weight,
        )
        oos_composite = compute_wfo_score(
            sharpe_ratio=fr.oos_sharpe,
            max_drawdown=fr.oos_max_drawdown,
            max_acceptable_drawdown=cfg.max_acceptable_drawdown,
            sortino_ratio=fr.oos_sortino,
            profit_factor=fr.oos_profit_factor,
            total_fills=fr.oos_total_fills,
            min_trades=1,
            sharpe_weight=cfg.sharpe_weight,
            sortino_weight=cfg.sortino_weight,
            profit_factor_weight=cfg.profit_factor_weight,
            trade_bonus_weight=cfg.trade_bonus_weight,
        )

        degradation = abs(is_composite - oos_composite) / max(abs(is_composite), 1e-6)
        if degradation < champion_degradation:
            champion_degradation = degradation
            champion_fold_idx = fr.fold_index
            champion_params = dict(fr.best_params)

    # Fallback: if no fold has positive OOS Sharpe, pick lowest abs-decay
    if not champion_params and fold_results:
        best_fr = min(fold_results, key=lambda f: abs(f.sharpe_decay_pct))
        champion_params = dict(best_fr.best_params)
        champion_fold_idx = best_fr.fold_index
        champion_degradation = abs(best_fr.sharpe_decay_pct) / 100.0

    report = WfoReport(
        folds=fold_results,
        stitched_equity_curve=stitched,
        aggregate_oos_sharpe=agg_sharpe,
        aggregate_oos_max_drawdown=agg_dd,
        aggregate_oos_total_pnl=agg_pnl,
        aggregate_oos_win_rate=agg_oos_win_rate,
        aggregate_oos_profit_factor=agg_oos_profit_factor,
        aggregate_oos_trade_count=total_oos_trades,
        parameter_stability=stability,
        total_elapsed_s=elapsed,
        avg_sharpe_decay_pct=avg_decay,
        overfit_probability=overfit_prob,
        unstable_params=unstable,
        champion_params=champion_params,
        champion_fold_index=champion_fold_idx,
        champion_degradation_pct=champion_degradation * 100,
    )

    # ── Export champion params to JSON ────────────────────────────────
    if cfg.output_params_path and champion_params:
        _export_champion_params(cfg, report)

    log.info(
        "wfo_complete",
        folds=len(fold_results),
        oos_sharpe=round(agg_sharpe, 4),
        oos_max_dd=round(agg_dd, 4),
        oos_pnl=round(agg_pnl, 4),
        avg_decay_pct=round(avg_decay, 1),
        overfit_prob=round(overfit_prob, 2),
        unstable_params=unstable,
        champion_fold=champion_fold_idx,
        champion_degradation_pct=round(champion_degradation * 100, 1),
        elapsed_s=round(elapsed, 1),
    )

    return report


def _export_champion_params(cfg: WfoConfig, report: WfoReport) -> None:
    """Write champion parameters to a JSON file with metadata."""
    data_dates = []
    for fr in report.folds:
        data_dates.extend(fr.train_dates)
        data_dates.extend(fr.test_dates)

    champion_fold = next(
        (fr for fr in report.folds if fr.fold_index == report.champion_fold_index),
        None,
    )

    date_range = ""
    if data_dates:
        date_range = f"{min(data_dates)} to {max(data_dates)}"

    output = {
        "params": report.champion_params,
        "meta": {
            "champion_fold": report.champion_fold_index,
            "oos_sharpe": round(champion_fold.oos_sharpe if champion_fold else 0.0, 4),
            "oos_win_rate": round(champion_fold.oos_win_rate if champion_fold else 0.0, 4),
            "oos_profit_factor": round(
                champion_fold.oos_profit_factor if champion_fold else 0.0,
                4,
            ),
            "oos_trade_count": int(champion_fold.oos_total_fills if champion_fold else 0),
            "oos_degradation_pct": round(report.champion_degradation_pct, 2),
            "n_folds": len(report.folds),
            "n_trials_per_fold": cfg.n_trials,
            "data_range": date_range,
            "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "avg_sharpe_decay_pct": round(report.avg_sharpe_decay_pct, 2),
            "overfit_probability": round(report.overfit_probability, 4),
            "unstable_params": report.unstable_params,
        },
    }

    out_path = Path(cfg.output_params_path)  # type: ignore[arg-type]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=False), encoding="utf-8")

    log.info("wfo_params_exported", path=str(out_path))
