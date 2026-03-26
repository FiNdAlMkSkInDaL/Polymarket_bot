from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.strategy import ContagionReplayAdapter
from src.backtest.wfo_optimizer import _build_data_loader, _load_market_configs
from src.core.config import StrategyParams


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = ROOT / "data" / "wfo_contagion_arb_micro_2026_03_25_microscale_fast" / "champion_params.json"
MARKET_CONFIG_PATH = ROOT / "data" / "domino_micro_fast_market_map.json"
DATA_DIR = ROOT / "data" / "vps_march2026"
OUTPUT_PATH = ROOT / "_tmp_contagion_causal_validation.json"
VALIDATION_DATES = ["2026-03-03"]
FIXED_CAUSAL_CONFIG = {
    "contagion_arb_max_leader_age_ms": 5000.0,
    "contagion_arb_max_lagger_age_ms": 30000.0,
    "contagion_arb_max_causal_lag_ms": 600000.0,
    "contagion_arb_allow_negative_lag": False,
}

logging.getLogger().setLevel(logging.WARNING)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def main() -> None:
    champion = _read_json(CHAMPION_PATH)
    params = dict(champion.get("params", {}))
    params.update(FIXED_CAUSAL_CONFIG)

    market_configs = _load_market_configs(str(MARKET_CONFIG_PATH))
    asset_ids = {
        str(config.get("yes_asset_id") or "")
        for config in market_configs
    } | {
        str(config.get("no_asset_id") or "")
        for config in market_configs
    }
    asset_ids.discard("")

    loader = _build_data_loader(str(DATA_DIR), VALIDATION_DATES, asset_ids=asset_ids)
    if loader is None:
        raise RuntimeError("Failed to build archived data loader for causal validation")

    strategy = ContagionReplayAdapter(
        market_configs=market_configs,
        fee_enabled=True,
        initial_bankroll=1000.0,
        params=StrategyParams(**params),
    )
    engine = BacktestEngine(
        strategy=strategy,
        data_loader=loader,
        config=BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_max_pct=2.0, fee_enabled=True),
    )
    result = engine.run()
    diagnostics = strategy.detector_diagnostics()

    cross_market_pairs = int(diagnostics.get("cross_market_pairs_evaluated", 0) or 0)
    accepted_causal = int(diagnostics.get("accepted_causal_lag_count", 0) or 0)
    legacy_sync_passed = int(diagnostics.get("legacy_sync_pairs_passed", 0) or 0)

    suppressor_counts = {
        "no_toxicity_spike": int(diagnostics.get("reject_no_toxicity_spike", 0) or 0),
        "insufficient_leader_impulse": int(diagnostics.get("reject_insufficient_leader_impulse", 0) or 0),
        "lagger_snapshot_stale": int(diagnostics.get("reject_lagger_snapshot_stale", 0) or 0),
        "causal_lag_too_large": int(diagnostics.get("reject_causal_lag_too_large", 0) or 0),
        "lagger_newer_than_leader": int(diagnostics.get("reject_lagger_newer_than_leader", 0) or 0),
        "leader_snapshot_stale": int(diagnostics.get("reject_leader_snapshot_stale", 0) or 0),
        "correlation_too_low": int(diagnostics.get("reject_correlation_too_low", 0) or 0),
        "residual_shift_too_small": int(diagnostics.get("reject_residual_shift_too_small", 0) or 0),
    }
    dominant_suppressor = max(
        suppressor_counts.items(),
        key=lambda item: (item[1], item[0]),
        default=("none", 0),
    )

    payload = {
        "validation_dates": VALIDATION_DATES,
        "champion_params": champion.get("params", {}),
        "fixed_causal_config": FIXED_CAUSAL_CONFIG,
        "metrics": result.metrics.to_dict(),
        "diagnostics": diagnostics,
        "summary": {
            "leader_events_evaluated": int(diagnostics.get("evaluations_with_previous_snapshot", 0) or 0),
            "events_reaching_spike_check": int(diagnostics.get("toxicity_spikes_detected", 0) or 0),
            "signals_fired": int(diagnostics.get("signals_emitted", 0) or 0),
            "fills_executed": int(result.metrics.total_fills),
            "cross_market_pairs_evaluated": cross_market_pairs,
            "causal_gate_pass_rate": _safe_rate(accepted_causal, cross_market_pairs),
            "legacy_sync_pass_rate": _safe_rate(legacy_sync_passed, cross_market_pairs),
            "dominant_suppressor": {
                "reason": dominant_suppressor[0],
                "count": int(dominant_suppressor[1]),
            },
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()