from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import send_shadow_hourly_telegram as telegram_script


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_build_shadow_telegram_summary_and_message(tmp_path: Path) -> None:
    run_root = tmp_path / "2026-04-06T11-05-01Z"
    _write_json(
        run_root / "metadata_refresh_preflight.json",
        {
            "active_market_count": 202,
            "covered_market_count": 202,
            "coverage_alarm": False,
            "coverage_pct": 100.0,
            "refresh_status": "success",
        },
    )
    _write_json(
        run_root / "scavenger_protocol_historical_sweep" / "summary.json",
        {
            "portfolio_fills": 0,
            "portfolio_orders_accepted": 0,
            "price_distribution_summary": {"unit_count": 3},
            "source_unique_markets": 202,
        },
    )
    (run_root / "conditional_probability_squeeze_batch").mkdir(parents=True, exist_ok=True)
    _write_json(
        run_root / "conditional_probability_squeeze_batch" / "batch_summary.json",
        {
            "pairs_requested": 49,
            "pairs_completed": 49,
            "pairs_failed": 0,
            "top_pair_id": "pair-a",
        },
    )
    (run_root / "conditional_probability_squeeze_batch" / "ranking.csv").write_text(
        "pair_id,status,total_valid_signals_generated,successful_fok_baskets\n"
        "pair-a,ok,2,1\n"
        "pair-b,ok,3,0\n",
        encoding="utf-8",
    )
    mid_dir = run_root / "mid_tier_probability_compression_historical_sweep"
    mid_dir.mkdir(parents=True, exist_ok=True)
    daily_panel = mid_dir / "daily_panel.parquet"
    pl.DataFrame(
        [
            {"signal_snapshots": 2, "candidate_orders": 4, "accepted_orders": 1, "filled_orders": 1},
            {"signal_snapshots": 1, "candidate_orders": 2, "accepted_orders": 0, "filled_orders": 0},
        ]
    ).write_parquet(daily_panel)
    _write_json(
        mid_dir / "execution_summary.json",
        {
            "artifacts": {
                "daily_panel": str(daily_panel),
            }
        },
    )

    summary = telegram_script.build_shadow_telegram_summary(run_root)
    message = telegram_script.format_shadow_telegram_message(summary)

    assert summary["preflight"]["coverage_pct"] == 100.0
    assert summary["scavenger"]["targets_found"] == 3
    assert summary["squeeze"]["total_valid_signals_generated"] == 5
    assert summary["squeeze"]["successful_fok_baskets"] == 1
    assert summary["mid_tier"]["signal_snapshots"] == 3
    assert summary["mid_tier"]["candidate_orders"] == 6
    assert "Coverage: ✅ 100.00% (202/202) | refresh=success" in message
    assert "Scavenger: targets=3 | orders=0 | fills=0" in message
    assert "Squeeze: signals=5 | baskets=1 | pairs=49/49" in message
    assert "Mid-Tier: snapshots=3 | candidates=6 | fills=1" in message


def test_format_shadow_telegram_message_marks_fallback_and_missing_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "2026-04-06T12-05-01Z"
    _write_json(
        run_root / "metadata_refresh_preflight.json",
        {
            "active_market_count": 200,
            "covered_market_count": 180,
            "coverage_alarm": True,
            "coverage_pct": 90.0,
            "refresh_status": "fallback",
        },
    )

    summary = telegram_script.build_shadow_telegram_summary(run_root)
    message = telegram_script.format_shadow_telegram_message(summary)

    assert summary["preflight"]["fallback_triggered"] is True
    assert summary["scavenger"]["status"] == "missing"
    assert "Coverage: ⚠️ 90.00% (180/200) | refresh=fallback | fallback" in message
    assert "Scavenger: unavailable" in message
    assert "Squeeze: unavailable" in message
    assert "Mid-Tier: unavailable" in message