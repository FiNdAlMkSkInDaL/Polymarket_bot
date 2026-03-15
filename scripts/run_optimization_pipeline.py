#!/usr/bin/env python3
"""
3-Stage Optimization Pipeline Orchestrator
===========================================

Automates the full Discovery → Refinement → Full-Universe Validation
sequence for Walk-Forward Optimization.

Phase 1: Discovery Sweep
    - Small universe (5 markets), short window (14 days), 50 trials.
    - Saves champion → logs/phase1_champion.json.

Bounds Narrowing
        - Reads Phase 1 champion, computes strict ±15% around
            each winning parameter value.
    - Clamps to original absolute min/max.
    - Saves → logs/phase2_bounds.json.

Phase 2: Refinement Sweep
    - Medium universe (10 markets), 30-day window, 100 trials.
    - Uses narrowed bounds from Phase 1.
    - Saves champion → logs/phase2_champion.json.

Phase 3: Full-Universe Validation
    - Full 31-market backtest with Phase 2 champion params injected.
    - Saves telemetry → logs/final_validation_tearsheet.json.

Usage
-----
    python scripts/run_optimization_pipeline.py --data-dir data/vps_march2026

All intermediate artefacts are written to the ``logs/`` directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx

from src.backtest.wfo_optimizer import SEARCH_SPACE
from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


# ── Telegram helpers (sync) ───────────────────────────────────────────────

def _notify(message: str) -> None:
    """Send a Telegram message synchronously.  Fails silently."""
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        log.warning("telegram_not_configured")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = httpx.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning("telegram_send_failed", status=resp.status_code)
    except Exception as exc:
        log.warning("telegram_send_error", error=str(exc))

# ── Defaults ──────────────────────────────────────────────────────────────
LOGS_DIR = Path("logs")

PHASE1_OUTPUT = LOGS_DIR / "phase1_champion.json"
PHASE2_BOUNDS = LOGS_DIR / "phase2_bounds.json"
PHASE2_OUTPUT = LOGS_DIR / "phase2_champion.json"
PHASE3_OUTPUT = LOGS_DIR / "final_validation_tearsheet.json"

# Per-phase Optuna DBs prevent study-name collisions
PHASE1_DB = LOGS_DIR / "wfo_phase1.db"
PHASE2_DB = LOGS_DIR / "wfo_phase2.db"
PHASE3_DB = LOGS_DIR / "wfo_phase3.db"

# Fractional half-width of the narrowed range relative to the original domain.
NARROWING_HALF_WIDTH = 0.15


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run_cli(args: list[str], phase: str) -> None:
    """Execute a ``polybot`` CLI command as a subprocess.

    Raises ``SystemExit`` on non-zero return code.
    """
    cmd = [sys.executable, "-m", "src.cli"] + args
    log.info("pipeline_exec", phase=phase, cmd=" ".join(cmd))

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        log.error(
            "pipeline_phase_failed",
            phase=phase,
            returncode=result.returncode,
            elapsed_s=round(elapsed, 1),
        )
        sys.exit(result.returncode)

    log.info("pipeline_phase_done", phase=phase, elapsed_s=round(elapsed, 1))


def _load_json(path: Path) -> dict:
    """Read and parse a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: dict) -> None:
    """Write a dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    log.info("pipeline_artefact_written", path=str(path))


# ═══════════════════════════════════════════════════════════════════════════
#  Bounds-Narrowing Algorithm
# ═══════════════════════════════════════════════════════════════════════════

def compute_narrowed_bounds(
    champion_params: dict[str, float],
    half_width_frac: float = NARROWING_HALF_WIDTH,
) -> dict[str, list[float]]:
    """Narrow the search space around Phase 1 champion values.

    For every parameter in ``SEARCH_SPACE``:

        new_lo          = clamp(champion_value × (1 - half_width_frac), lo, hi)
        new_hi          = clamp(champion_value × (1 + half_width_frac), lo, hi)

    Integer parameters produce integer-rounded bounds.

    Returns a dict suitable for serialisation to JSON and consumption
    by ``--search-space-bounds``.
    """
    bounds: dict[str, list[float]] = {}

    for name, spec in SEARCH_SPACE.items():
        method = spec[0]
        orig_lo, orig_hi = float(spec[1]), float(spec[2])

        value = champion_params.get(name)
        if value is None:
            # Parameter not produced by Phase 1 — keep original range
            bounds[name] = [orig_lo, orig_hi]
            continue

        margin_mult = float(half_width_frac)
        new_lo = max(orig_lo, value * (1.0 - margin_mult))
        new_hi = min(orig_hi, value * (1.0 + margin_mult))

        # Guarantee new_lo < new_hi (defend against edge cases)
        if new_lo >= new_hi:
            epsilon = max(abs(value) * 0.001, 1e-9)
            new_lo = max(orig_lo, value - epsilon)
            new_hi = min(orig_hi, value + epsilon)
            if new_lo >= new_hi:
                new_lo, new_hi = orig_lo, orig_hi

        if method == "suggest_int":
            new_lo = float(int(round(new_lo)))
            new_hi = float(int(round(new_hi)))
            new_lo = max(new_lo, float(int(orig_lo)))
            new_hi = min(new_hi, float(int(orig_hi)))
            if new_lo >= new_hi:
                if new_hi < float(int(orig_hi)):
                    new_hi += 1.0
                elif new_lo > float(int(orig_lo)):
                    new_lo -= 1.0
                else:
                    new_lo, new_hi = float(int(orig_lo)), float(int(orig_hi))

        bounds[name] = [new_lo, new_hi]

    return bounds


# ═══════════════════════════════════════════════════════════════════════════
#  Phase runners
# ═══════════════════════════════════════════════════════════════════════════

# ── Production defaults per phase ─────────────────────────────────────────
PHASE1_DEFAULTS = {"max_markets": 5, "train_days": 14, "n_trials": 50, "max_workers": 2}
PHASE2_DEFAULTS = {"max_markets": 10, "train_days": 30, "n_trials": 100, "max_workers": 2}
PHASE3_DEFAULTS = {"max_markets": 31, "train_days": 30, "n_trials": 1}

# Smoke-test overrides (minimal resources for local validation)
SMOKE_OVERRIDES = {
    "max_markets": 1, "train_days": 5, "n_trials": 10,
    "max_workers": 1, "test_days": 2, "step_days": 30,
}


def _wfo_base_args(data_dir: str, cfg: dict) -> list[str]:
    """Build common WFO CLI args from a config dict."""
    args = [
        "wfo",
        "--data-dir", data_dir,
        "--max-markets", str(cfg["max_markets"]),
        "--train-days", str(cfg["train_days"]),
        "--n-trials", str(cfg["n_trials"]),
    ]
    if "max_workers" in cfg:
        args.extend(["--max-workers", str(cfg["max_workers"])])
    if "test_days" in cfg:
        args.extend(["--test-days", str(cfg["test_days"])])
    if "step_days" in cfg:
        args.extend(["--step-days", str(cfg["step_days"])])
    return args


def run_phase1(data_dir: str, *, overrides: dict | None = None) -> None:
    """Phase 1: Discovery Sweep — wide search, small universe."""
    cfg = {**PHASE1_DEFAULTS, **(overrides or {})}
    log.info("pipeline_phase1_start", **cfg)
    _run_cli(
        _wfo_base_args(data_dir, cfg) + [
            "--storage", f"sqlite:///{PHASE1_DB}",
            "--output-params", str(PHASE1_OUTPUT),
        ],
        phase="phase1_discovery",
    )

    if not PHASE1_OUTPUT.exists():
        log.error("pipeline_phase1_no_output", path=str(PHASE1_OUTPUT))
        sys.exit(1)


def run_bounds_narrowing() -> None:
    """Intermediate: read Phase 1 champion, compute narrowed bounds."""
    log.info("pipeline_narrowing_start")

    champion_data = _load_json(PHASE1_OUTPUT)
    champion_params = champion_data.get("params", {})

    if not champion_params:
        log.error("pipeline_narrowing_empty_params", path=str(PHASE1_OUTPUT))
        sys.exit(1)

    narrowed = compute_narrowed_bounds(champion_params)
    _write_json(PHASE2_BOUNDS, narrowed)

    log.info(
        "pipeline_narrowing_done",
        n_params=len(narrowed),
        output=str(PHASE2_BOUNDS),
    )


def run_phase2(data_dir: str, *, overrides: dict | None = None) -> None:
    """Phase 2: Refinement Sweep — narrowed bounds, medium universe."""
    cfg = {**PHASE2_DEFAULTS, **(overrides or {})}
    log.info("pipeline_phase2_start", **cfg)
    _run_cli(
        _wfo_base_args(data_dir, cfg) + [
            "--storage", f"sqlite:///{PHASE2_DB}",
            "--search-space-bounds", str(PHASE2_BOUNDS),
            "--output-params", str(PHASE2_OUTPUT),
        ],
        phase="phase2_refinement",
    )

    if not PHASE2_OUTPUT.exists():
        log.error("pipeline_phase2_no_output", path=str(PHASE2_OUTPUT))
        sys.exit(1)


def run_phase3(data_dir: str, *, overrides: dict | None = None) -> None:
    """Phase 3: Full-Universe Validation backtest with champion params."""
    cfg = {**PHASE3_DEFAULTS, **(overrides or {})}
    log.info("pipeline_phase3_start", **cfg)

    champion_data = _load_json(PHASE2_OUTPUT)
    champion_params = champion_data.get("params", {})

    if not champion_params:
        log.error("pipeline_phase3_empty_params", path=str(PHASE2_OUTPUT))
        sys.exit(1)

    # We run WFO in validation mode: full universe, using the champion
    # params as a fixed search space (very tight bounds → deterministic).
    # Build a "locked" bounds file where lo == hi for each param so Optuna
    # simply reproduces the champion.
    locked_bounds_path = LOGS_DIR / "phase3_locked_bounds.json"
    locked: dict[str, list[float]] = {}
    for name, spec in SEARCH_SPACE.items():
        value = champion_params.get(name)
        if value is not None:
            if spec[0] == "suggest_int":
                locked[name] = [int(value), int(value) + 1]
            else:
                # Optuna requires lo < hi; use a negligible epsilon
                eps = abs(value) * 1e-8 if abs(value) > 1e-12 else 1e-12
                locked[name] = [value, value + eps]
        else:
            locked[name] = [float(spec[1]), float(spec[2])]

    _write_json(locked_bounds_path, locked)

    _run_cli(
        _wfo_base_args(data_dir, cfg) + [
            "--storage", f"sqlite:///{PHASE3_DB}",
            "--search-space-bounds", str(locked_bounds_path),
            "--output-params", str(PHASE3_OUTPUT),
        ],
        phase="phase3_validation",
    )

    if not PHASE3_OUTPUT.exists():
        log.error("pipeline_phase3_no_output", path=str(PHASE3_OUTPUT))
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-Stage Optimization Pipeline (Discovery → Refinement → Validation)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to recorded tick data directory.",
    )
    parser.add_argument(
        "--skip-to",
        choices=["phase1", "narrowing", "phase2", "phase3"],
        default="phase1",
        help="Resume the pipeline from a specific stage (default: phase1).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Use minimal parameters (1 market, 2 days, 2 trials) for local validation.",
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        default=False,
        help="Send a test Telegram alert and exit.",
    )
    args = parser.parse_args()

    phase_overrides = SMOKE_OVERRIDES if args.smoke_test else None

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_t0 = time.monotonic()
    stages = ["phase1", "narrowing", "phase2", "phase3"]
    start_idx = stages.index(args.skip_to)

    log.info(
        "pipeline_start",
        data_dir=args.data_dir,
        skip_to=args.skip_to,
        smoke_test=args.smoke_test,
    )

    # ── Milestone 0: Test alert mode ──────────────────────────────────
    if args.test_alert:
        _notify("🔔 <b>WFO Test Alert</b>\nTelegram integration is working.")
        print("Test alert sent — check Telegram.")
        return

    # ── Milestone 1: Pipeline Start ───────────────────────────────────
    try:
        _notify(
            "🚀 <b>WFO Pipeline Started</b>\n"
            f"Target: <code>{args.data_dir}</code>\n"
            "Optimizing for Sharpe Ratio across 3 Phases."
        )
    except Exception:
        pass  # non-critical

    try:
        if start_idx <= 0:
            run_phase1(args.data_dir, overrides=phase_overrides)

            # ── Milestone 2: Phase 1 Complete ─────────────────────────
            try:
                p1 = _load_json(PHASE1_OUTPUT)
                n_candidates = len(p1.get("params", {}))
                _notify(
                    "✅ <b>Phase 1 (Discovery) Complete</b>\n"
                    f"Found <b>{n_candidates}</b> valid parameter candidates.\n"
                    "Moving to Phase 2 Refinement."
                )
            except Exception:
                pass

        if start_idx <= 1:
            run_bounds_narrowing()

        if start_idx <= 2:
            run_phase2(args.data_dir, overrides=phase_overrides)

            # ── Milestone 3: Phase 2 Complete ─────────────────────────
            try:
                p2 = _load_json(PHASE2_OUTPUT)
                best_sharpe = p2.get("value", p2.get("sharpe", "N/A"))
                if isinstance(best_sharpe, (int, float)):
                    best_sharpe = f"{best_sharpe:.2f}"
                _notify(
                    "🎯 <b>Phase 2 (Refinement) Complete</b>\n"
                    f"Best Sharpe: <b>{best_sharpe}</b>\n"
                    "Starting Phase 3 Out-of-Sample Validation."
                )
            except Exception:
                pass

        if start_idx <= 3:
            run_phase3(args.data_dir, overrides=phase_overrides)

    except SystemExit as exc:
        # ── Milestone 5: Critical Failure ─────────────────────────────
        tb = traceback.format_exc()
        snippet = tb[-500:] if len(tb) > 500 else tb
        _notify(
            "⚠️ <b>WFO PIPELINE CRASHED</b>\n"
            f"<pre>{snippet}</pre>\n"
            "Check logs/pipeline.log."
        )
        raise
    except Exception:
        tb = traceback.format_exc()
        snippet = tb[-500:] if len(tb) > 500 else tb
        _notify(
            "⚠️ <b>WFO PIPELINE CRASHED</b>\n"
            f"<pre>{snippet}</pre>\n"
            "Check logs/pipeline.log."
        )
        raise

    elapsed = time.monotonic() - pipeline_t0
    log.info(
        "pipeline_complete",
        elapsed_s=round(elapsed, 1),
        phase3_output=str(PHASE3_OUTPUT),
    )

    # ── Milestone 4: Pipeline Finished ────────────────────────────────
    try:
        p3 = _load_json(PHASE3_OUTPUT)
        champion_sharpe = p3.get("value", p3.get("sharpe", "N/A"))
        total_trades = p3.get("total_trades", p3.get("n_trades", "N/A"))
        if isinstance(champion_sharpe, (int, float)):
            champion_sharpe = f"{champion_sharpe:.2f}"
        _notify(
            "🏆 <b>WFO Successfully Finished!</b>\n"
            f"  • Champion Sharpe: <b>{champion_sharpe}</b>\n"
            f"  • Total Trades: <b>{total_trades}</b>\n"
            f"  • Configuration saved to <code>phase3_locked_bounds.json</code>\n"
            f"  • Elapsed: {elapsed:.0f}s"
        )
    except Exception:
        pass

    print(f"\n{'='*70}")
    print("  OPTIMIZATION PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Phase 1 champion:   {PHASE1_OUTPUT}")
    print(f"  Phase 2 bounds:     {PHASE2_BOUNDS}")
    print(f"  Phase 2 champion:   {PHASE2_OUTPUT}")
    print(f"  Validation output:  {PHASE3_OUTPUT}")
    print(f"  Total elapsed:      {elapsed:.0f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
