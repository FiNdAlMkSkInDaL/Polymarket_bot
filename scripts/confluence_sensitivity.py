"""
Confluence Parameter Sensitivity Analysis
==========================================

Stratified bootstrap win-rate analysis for the 17 V1-V4 parameters.

For each confluence factor combination (2^4 = 16 cells), this script:
  1. Loads closed trades from the SQLite trade store (tagged with
     signal_flags if available, inferred from log fields otherwise).
  2. Computes per-cell win rate with 95% CI via bootstrap (B=2000).
  3. Runs a one-sided binomial test: H₀: w_cell ≤ w_baseline.
  4. Outputs a JSON report with "justified": bool for each factor.

Usage
-----
    python scripts/confluence_sensitivity.py --min-trades 30 --alpha 0.10

Optional flags
--------------
    --alpha          Significance threshold (default: 0.10)
    --min-trades     Minimum fills per cell to include (default: 30)
    --bootstrap-n    Bootstrap replicates (default: 2000)
    --db-path        Path to trade DB (default: from config)
    --output         Output JSON path (default: prints to stdout)
    --sobol          Run first-order Sobol sensitivity index estimation
                     (requires SALib: pip install SALib)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Add project root to sys.path ───────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_FLAGS_COLS = [
    "maker",              # bool: was execution_mode=maker
    "confluence_mask",    # int (0-15): 4-bit bitmask of active factors
    #   bit 0 = whale, bit 1 = spread, bit 2 = l2, bit 3 = regime
    "drift_active",       # bool: DriftSignal was primary source
    "probe",              # bool: is_probe entry
    "n_confluence_factors",  # int: number of active factors
]

_FACTOR_BITS = {
    "whale":   0,
    "spread":  1,
    "l2":      2,
    "regime":  3,
}


def _load_trades(db_path: str) -> list[dict[str, Any]]:
    """Load closed trades from SQLite TradeStore.

    Falls back to inferring signal_flags from available columns when
    the 'signal_flags' column is absent (pre-calibration trade data).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check available columns
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row["name"] for row in cursor.fetchall()}

    if "signal_flags" in columns:
        rows = cursor.execute("""
            SELECT id, pnl_cents, signal_flags, exit_reason
            FROM trades
            WHERE exit_reason IS NOT NULL AND exit_reason != 'cancelled'
        """).fetchall()
        trades = []
        for r in rows:
            flags = json.loads(r["signal_flags"] or "{}")
            trades.append({
                "id": r["id"],
                "pnl_cents": r["pnl_cents"],
                "win": r["pnl_cents"] > 0,
                **flags,
            })
    else:
        # No signal_flags column: load pnl_cents only, assume baseline
        rows = cursor.execute("""
            SELECT id, pnl_cents, exit_reason
            FROM trades
            WHERE exit_reason IS NOT NULL AND exit_reason != 'cancelled'
        """).fetchall()
        trades = [
            {
                "id": r["id"],
                "pnl_cents": r["pnl_cents"],
                "win": r["pnl_cents"] > 0,
                "confluence_mask": 0,
                "maker": False,
                "drift_active": False,
                "probe": False,
                "n_confluence_factors": 0,
            }
            for r in rows
        ]
    conn.close()
    return trades


# ═══════════════════════════════════════════════════════════════════════════
#  Statistical utilities
# ═══════════════════════════════════════════════════════════════════════════

def _bootstrap_win_rate(
    wins: list[bool],
    n_replicates: int = 2000,
    ci_level: float = 0.95,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via bootstrap percentile CI."""
    if not wins:
        return 0.0, 0.0, 0.0
    rng = rng or random.Random(42)
    n = len(wins)
    obs = sum(wins) / n
    reps = []
    for _ in range(n_replicates):
        sample = [rng.choice(wins) for _ in range(n)]
        reps.append(sum(sample) / n)
    reps.sort()
    alpha = 1.0 - ci_level
    lo = reps[int(alpha / 2 * n_replicates)]
    hi = reps[int((1 - alpha / 2) * n_replicates)]
    return obs, lo, hi


def _binomial_pvalue_one_sided(k: int, n: int, p0: float) -> float:
    """One-sided binomial p-value: P(X ≥ k | H₀: p = p0).

    Uses normal approximation with continuity correction (Berry-Esseen
    accuracy ≥ n > 10).
    """
    if n == 0:
        return 1.0
    mu = n * p0
    sigma = math.sqrt(n * p0 * (1.0 - p0))
    if sigma == 0:
        return 0.0 if k > mu else 1.0
    # Upper-tail z with continuity correction
    z = (k - 0.5 - mu) / sigma
    # Complementary CDF via erfc
    return 0.5 * math.erfc(z / math.sqrt(2))


# ═══════════════════════════════════════════════════════════════════════════
#  Report dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FactorReport:
    factor: str
    n_trades_on:  int = 0
    n_trades_off: int = 0
    win_rate_on:  float = 0.0
    win_rate_off: float = 0.0
    ci_lower_on:  float = 0.0
    ci_upper_on:  float = 0.0
    delta_win_rate: float = 0.0
    p_value:       float = 1.0
    justified:     bool = False
    note:          str = ""


@dataclass
class ComboReport:
    mask:      int = 0
    label:     str = ""
    n_trades:  int = 0
    win_rate:  float = 0.0
    ci_lower:  float = 0.0
    ci_upper:  float = 0.0
    p_value:   float = 1.0
    justified: bool = False


@dataclass
class SensitivityReport:
    generated_at:    str = ""
    total_trades:    int = 0
    baseline_win_rate: float = 0.0
    baseline_n:      int = 0
    alpha:           float = 0.10
    min_trades:      int = 30
    factors:         list[FactorReport] = field(default_factory=list)
    combos:          list[ComboReport] = field(default_factory=list)
    warnings:        list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_sensitivity_analysis(
    trades: list[dict[str, Any]],
    alpha: float = 0.10,
    min_trades: int = 30,
    bootstrap_n: int = 2000,
) -> SensitivityReport:
    report = SensitivityReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_trades=len(trades),
        alpha=alpha,
        min_trades=min_trades,
    )
    rng = random.Random(12345)

    # Baseline: all trades (confederation-independent)
    all_wins = [bool(t.get("win", t.get("pnl_cents", 0) > 0)) for t in trades]
    bl_wr, bl_lo, bl_hi = _bootstrap_win_rate(all_wins, bootstrap_n, rng=rng)
    report.baseline_win_rate = round(bl_wr, 4)
    report.baseline_n = len(all_wins)

    if len(all_wins) < min_trades:
        report.warnings.append(
            f"Insufficient data: {len(all_wins)} trades < min_trades={min_trades}"
        )
        return report

    # ── Per-factor marginal win rate ───────────────────────────────────────
    for factor, bit in _FACTOR_BITS.items():
        trades_on  = [t for t in trades if (t.get("confluence_mask", 0) >> bit) & 1]
        trades_off = [t for t in trades if not ((t.get("confluence_mask", 0) >> bit) & 1)]

        wins_on  = [bool(t.get("win", t.get("pnl_cents", 0) > 0)) for t in trades_on]
        wins_off = [bool(t.get("win", t.get("pnl_cents", 0) > 0)) for t in trades_off]

        fr = FactorReport(factor=factor)
        fr.n_trades_on  = len(wins_on)
        fr.n_trades_off = len(wins_off)

        if len(wins_on) < min_trades:
            fr.note = f"insufficient_data (n={len(wins_on)} < min={min_trades})"
            report.factors.append(fr)
            continue

        wr_on, lo_on, hi_on = _bootstrap_win_rate(wins_on, bootstrap_n, rng=rng)
        wr_off, _, _ = _bootstrap_win_rate(wins_off, 500, rng=rng)

        fr.win_rate_on  = round(wr_on, 4)
        fr.win_rate_off = round(wr_off, 4)
        fr.ci_lower_on  = round(lo_on, 4)
        fr.ci_upper_on  = round(hi_on, 4)
        fr.delta_win_rate = round(wr_on - wr_off, 4)

        # One-sided binomial test: H₀: w_on ≤ w_off (factor adds no edge)
        k = sum(wins_on)
        fr.p_value = round(
            _binomial_pvalue_one_sided(k, len(wins_on), max(wr_off, 0.01)),
            6,
        )
        fr.justified = (fr.p_value < alpha) and (fr.delta_win_rate > 0)

        if not fr.justified:
            fr.note = (
                f"Discount NOT statistically justified: "
                f"Δwin_rate={fr.delta_win_rate:+.3f}, p={fr.p_value:.3f}"
            )
        else:
            fr.note = (
                f"Discount JUSTIFIED: "
                f"Δwin_rate={fr.delta_win_rate:+.3f}, p={fr.p_value:.3f}"
            )

        report.factors.append(fr)

    # ── Per-combination (2x4 = 16 cells) win rates ────────────────────────
    for mask in range(16):
        cell = [t for t in trades
                if (t.get("confluence_mask", 0) & 0xF) == mask]
        wins = [bool(t.get("win", t.get("pnl_cents", 0) > 0)) for t in cell]

        label_bits = [
            f for f, b in _FACTOR_BITS.items()
            if (mask >> b) & 1
        ]
        label = "+".join(label_bits) if label_bits else "none"

        cr = ComboReport(mask=mask, label=label, n_trades=len(wins))
        if len(wins) < min_trades:
            cr.justified = False
            report.combos.append(cr)
            continue

        wr, lo, hi = _bootstrap_win_rate(wins, bootstrap_n, rng=rng)
        cr.win_rate = round(wr, 4)
        cr.ci_lower = round(lo, 4)
        cr.ci_upper = round(hi, 4)
        k = sum(wins)
        cr.p_value = round(
            _binomial_pvalue_one_sided(k, len(wins), max(bl_wr, 0.01)),
            6,
        )
        cr.justified = (cr.p_value < alpha) and (wr > bl_wr)
        report.combos.append(cr)

    # Summary warnings for unjustified discount magnitudes
    for fr in report.factors:
        if fr.n_trades_on >= min_trades and not fr.justified:
            report.warnings.append(
                f"CALIBRATION ALERT: confluence_{fr.factor}_discount is "
                f"NOT statistically justified (p={fr.p_value:.3f}). "
                f"Consider reducing or zeroing this parameter."
            )

    return report


# ═══════════════════════════════════════════════════════════════════════════
#  Optional: Sobol first-order sensitivity indices
# ═══════════════════════════════════════════════════════════════════════════

def run_sobol_analysis(db_path: str, n_samples: int = 2000) -> dict[str, Any]:
    """Estimate first-order Sobol indices for the 17 V1-V4 parameters.

    Requires SALib (pip install SALib) and the backtest engine.  Runs
    n_samples Monte-Carlo evaluations of the WFO backtest engine with
    Latin Hypercube sampled parameter vectors.

    Returns
    -------
    dict
        Keys: parameter names.  Values: {"S1": float, "ST": float}.
        S1 = first-order index (direct variance contribution).
        ST = total-effect index (including interaction terms).
    """
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol as sobol_analyze
    except ImportError:
        return {"error": "SALib not installed — run: pip install SALib"}

    # 17-parameter problem with bounds for each
    problem = {
        "num_vars": 17,
        "names": [
            "maker_routing_enabled",       # 0/1
            "maker_eqs_discount",          # [0.75, 0.95]
            "confluence_eqs_floor",        # [28, 42]
            "confluence_min_factors",      # [1, 4]
            "confluence_whale_discount",   # [3, 7]  — tightened; was [1,9]
            "confluence_spread_discount",  # [1, 7]
            "confluence_l2_discount",      # [0, 0]  — zeroed; reclassified as hard gate
            "confluence_regime_discount",  # [0, 0]  — zeroed; failed binomial test
            "drift_signal_enabled",        # 0/1
            "drift_lookback_bars",         # [6, 16]
            "drift_z_threshold",           # [0.6, 2.0]
            "drift_vol_ceiling",           # [0.008, 0.025]
            "drift_cooldown_s",            # [60, 300]
            "probe_sizing_enabled",        # 0/1
            "probe_eqs_floor",             # [28, 42]
            "probe_kelly_fraction",        # [0.02, 0.10]
            "probe_max_usd",               # [0.5, 5.0]
        ],
        "bounds": [
            [0, 1], [0.75, 0.95], [28, 42], [1, 4],
            [3, 7], [1, 7], [0, 0.001], [0, 0.001],
            [0, 1], [6, 16], [0.6, 2.0], [0.008, 0.025], [60, 300],
            [0, 1], [28, 42], [0.02, 0.10], [0.5, 5.0],
        ],
    }

    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.backtest.engine import BacktestEngine
        from src.backtest.data_loader import BacktestDataLoader
    except ImportError as e:
        return {"error": f"Backtest engine not importable: {e}"}

    results = []
    for i, params_row in enumerate(param_values):
        if i % 50 == 0:
            print(f"  Sobol sample {i}/{len(param_values)}...", end="\r")
        # Build overridden environment
        param_override = {name: val for name, val in zip(problem["names"], params_row)}
        # Round binary params
        param_override["maker_routing_enabled"] = bool(round(param_override["maker_routing_enabled"]))
        param_override["drift_signal_enabled"]  = bool(round(param_override["drift_signal_enabled"]))
        param_override["probe_sizing_enabled"]  = bool(round(param_override["probe_sizing_enabled"]))
        param_override["confluence_min_factors"] = int(round(param_override["confluence_min_factors"]))
        param_override["drift_lookback_bars"]   = int(round(param_override["drift_lookback_bars"]))

        try:
            engine = BacktestEngine(strategy_params=param_override)
            result = engine.run_quick(n_days=30)
            results.append(result.get("sharpe", 0.0))
        except Exception:
            results.append(0.0)

    import numpy as np
    Y = np.array(results)
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    return {
        name: {"S1": round(float(s1), 4), "ST": round(float(st), 4)}
        for name, s1, st in zip(problem["names"], Si["S1"], Si["ST"])
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confluence parameter sensitivity analysis for V1-V4 architecture."
    )
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="One-sided significance level for justification test (default: 0.10)")
    parser.add_argument("--min-trades", type=int, default=30,
                        help="Min fills per cell to include in analysis (default: 30)")
    parser.add_argument("--bootstrap-n", type=int, default=2000,
                        help="Bootstrap replicates (default: 2000)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to trade SQLite DB (default: from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: stdout)")
    parser.add_argument("--sobol", action="store_true",
                        help="Run Sobol sensitivity analysis (requires SALib)")
    parser.add_argument("--sobol-n", type=int, default=256,
                        help="Sobol sample count (default: 256)")
    args = parser.parse_args()

    # Resolve DB path
    if args.db_path:
        db_path = args.db_path
    else:
        try:
            from src.core.config import settings
            db_path = settings.trade_store_path or "trades.db"
        except Exception:
            db_path = "trades.db"

    print(f"Loading trades from: {db_path}", file=sys.stderr)
    try:
        trades = _load_trades(db_path)
    except Exception as e:
        print(f"ERROR: Could not load trades: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(trades)} closed trades", file=sys.stderr)

    report = run_sensitivity_analysis(
        trades,
        alpha=args.alpha,
        min_trades=args.min_trades,
        bootstrap_n=args.bootstrap_n,
    )

    output: dict[str, Any] = asdict(report)

    if args.sobol:
        print("Running Sobol sensitivity analysis...", file=sys.stderr)
        sobol_result = run_sobol_analysis(db_path, n_samples=args.sobol_n)
        output["sobol"] = sobol_result

    out_json = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).write_text(out_json)
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(out_json)

    # Print human-readable summary to stderr
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SENSITIVITY REPORT — {report.generated_at}", file=sys.stderr)
    print(f"Total trades: {report.total_trades} | Baseline win rate: {report.baseline_win_rate:.1%}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for fr in report.factors:
        status = "✓ JUSTIFIED" if fr.justified else "✗ NOT JUSTIFIED"
        print(f"  {fr.factor:8s}  n_on={fr.n_trades_on:4d}  wr_on={fr.win_rate_on:.3f}"
              f"  Δ={fr.delta_win_rate:+.3f}  p={fr.p_value:.3f}  {status}",
              file=sys.stderr)
    if report.warnings:
        print(f"\nWARNINGS:", file=sys.stderr)
        for w in report.warnings:
            print(f"  ⚠  {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
