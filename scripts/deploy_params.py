#!/usr/bin/env python3
"""Deploy selected WFO champion params + manual overrides to .env on VPS.

Usage (from project root):
    python scripts/deploy_params.py              # dry-run (show what would change)
    python scripts/deploy_params.py --apply      # actually append to .env

Reads logs/final_validation_tearsheet.json, selects a conservative subset
of champion params, and appends them (plus manual tuning overrides) to .env.
Skips params flagged as likely overfit (e.g. trend_guard_pct 7× default).
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEARSHEET = PROJECT_ROOT / "logs" / "final_validation_tearsheet.json"
ENV_FILE = PROJECT_ROOT / ".env"

# ── Params to deploy from WFO champion ────────────────────────────────────
# Only params that are clearly better than defaults AND not extreme.
DEPLOY_PARAMS: dict[str, str] = {
    "zscore_threshold":   "ZSCORE_THRESHOLD",
    "alpha_default":      "ALPHA_DEFAULT",
    "min_edge_score":     "MIN_EDGE_SCORE",
    "kelly_fraction":     "KELLY_FRACTION",
    "stop_loss_cents":    "STOP_LOSS_CENTS",
    "drift_z_threshold":  "DRIFT_Z_THRESHOLD",
}

# Params SKIPPED (likely overfit — extreme deviation from defaults):
#   trend_guard_pct:            0.561 vs 0.08 default (7× — disables guard)
#   pce_max_portfolio_var_usd:  98.4  vs 50.0 default (doubles risk budget)

# ── Manual tuning overrides (not in WFO) ──────────────────────────────────
MANUAL_OVERRIDES: dict[str, str] = {
    "EXIT_TIMEOUT_SECONDS":   "900",      # was 1800 — align with MR halflife
    "MAX_ACTIVE_L2_MARKETS":  "25",       # was 50  — reduce L2 desync load
    "SIGNAL_COOLDOWN_MINUTES": "5.0",     # was 0.25 — prevent serial re-entry
}


def load_champion_params() -> dict[str, float]:
    with open(TEARSHEET) as f:
        data = json.load(f)
    return data["params"]


def read_existing_env(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def build_env_lines(champion: dict[str, float]) -> list[str]:
    """Build the list of KEY=VALUE lines to append."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"# ── Strategy params deployed {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} ──")

    # WFO champion (selected subset)
    lines.append("# WFO champion params (conservative subset)")
    for param_name, env_name in DEPLOY_PARAMS.items():
        value = champion.get(param_name)
        if value is not None:
            # Round to 6 decimal places for readability
            if isinstance(value, float):
                lines.append(f"{env_name}={value:.6f}")
            else:
                lines.append(f"{env_name}={value}")

    # Manual overrides
    lines.append("# Manual tuning overrides")
    for env_name, value in MANUAL_OVERRIDES.items():
        lines.append(f"{env_name}={value}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy WFO params to .env")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write to .env (default: dry-run)")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE,
                        help="Path to .env file (default: project root .env)")
    args = parser.parse_args()

    champion = load_champion_params()
    new_lines = build_env_lines(champion)

    print("Parameters to deploy:")
    print("-" * 50)
    for line in new_lines:
        print(line)
    print("-" * 50)

    if not args.apply:
        print("\nDry run — pass --apply to write to .env")
        return

    env_path: Path = args.env_file
    # Backup existing .env
    if env_path.exists():
        backup = env_path.with_suffix(f".env.bak.{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(env_path, backup)
        print(f"Backed up existing .env → {backup.name}")

    with open(env_path, "a") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"\n✅ Appended {len(new_lines) - 3} params to {env_path}")


if __name__ == "__main__":
    main()
