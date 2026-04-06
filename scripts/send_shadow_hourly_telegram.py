#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "shadow_mode" else PROJECT_ROOT
for candidate in (PROJECT_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.core.config import settings
from src.monitoring.telegram import TelegramAlerter


DEFAULT_RUN_ROOT = REPO_ROOT / "shadow_logs"
DEFAULT_SUMMARY_NAME = "shadow_telegram_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a compact Telegram summary for a completed shadow hourly run.")
    parser.add_argument("--run-root", type=Path, required=True, help="Timestamped shadow run directory.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the telegram summary JSON. Defaults to <run-root>/shadow_telegram_summary.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and print the message without sending it.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summary_output_path(run_root: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return run_root / DEFAULT_SUMMARY_NAME


def load_preflight_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / "metadata_refresh_preflight.json"
    payload = _read_json(path)
    if payload is None:
        return {
            "status": "missing",
            "path": str(path),
            "refresh_status": "missing",
            "coverage_pct": 0.0,
            "covered_market_count": 0,
            "active_market_count": 0,
            "coverage_alarm": False,
            "fallback_triggered": False,
        }
    refresh_status = str(payload.get("refresh_status") or "unknown")
    return {
        "status": "ok",
        "path": str(path),
        "refresh_status": refresh_status,
        "coverage_pct": _coerce_float(payload.get("coverage_pct")),
        "covered_market_count": _coerce_int(payload.get("covered_market_count")),
        "active_market_count": _coerce_int(payload.get("active_market_count")),
        "coverage_alarm": bool(payload.get("coverage_alarm")),
        "fallback_triggered": refresh_status != "success",
    }


def load_scavenger_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / "scavenger_protocol_historical_sweep" / "summary.json"
    payload = _read_json(path)
    if payload is None:
        return {
            "status": "missing",
            "path": str(path),
            "targets_found": 0,
            "portfolio_orders_accepted": 0,
            "portfolio_fills": 0,
            "source_unique_markets": 0,
        }
    price_distribution = payload.get("price_distribution_summary") or {}
    return {
        "status": "ok",
        "path": str(path),
        "targets_found": _coerce_int(price_distribution.get("unit_count")),
        "portfolio_orders_accepted": _coerce_int(payload.get("portfolio_orders_accepted")),
        "portfolio_fills": _coerce_int(payload.get("portfolio_fills")),
        "source_unique_markets": _coerce_int(payload.get("source_unique_markets")),
    }


def load_squeeze_summary(run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "conditional_probability_squeeze_batch" / "batch_summary.json"
    ranking_path = run_root / "conditional_probability_squeeze_batch" / "ranking.csv"
    summary_payload = _read_json(summary_path) or {}
    requested = _coerce_int(summary_payload.get("pairs_requested"))
    completed = _coerce_int(summary_payload.get("pairs_completed"))
    failed = _coerce_int(summary_payload.get("pairs_failed"))
    top_pair_id = str(summary_payload.get("top_pair_id") or "") or None
    total_valid_signals = 0
    successful_fok_baskets = 0

    if ranking_path.is_file():
        with ranking_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                total_valid_signals += _coerce_int(row.get("total_valid_signals_generated"))
                successful_fok_baskets += _coerce_int(row.get("successful_fok_baskets"))

    return {
        "status": "ok" if summary_path.is_file() or ranking_path.is_file() else "missing",
        "summary_path": str(summary_path),
        "ranking_path": str(ranking_path),
        "pairs_requested": requested,
        "pairs_completed": completed,
        "pairs_failed": failed,
        "top_pair_id": top_pair_id,
        "total_valid_signals_generated": total_valid_signals,
        "successful_fok_baskets": successful_fok_baskets,
    }


def load_mid_tier_summary(run_root: Path) -> dict[str, Any]:
    execution_path = run_root / "mid_tier_probability_compression_historical_sweep" / "execution_summary.json"
    execution_payload = _read_json(execution_path) or {}
    artifacts = execution_payload.get("artifacts") or {}
    daily_panel_path = Path(str(artifacts.get("daily_panel"))) if artifacts.get("daily_panel") else run_root / "mid_tier_probability_compression_historical_sweep" / "daily_panel.parquet"
    if not daily_panel_path.is_file():
        return {
            "status": "missing",
            "execution_path": str(execution_path),
            "daily_panel_path": str(daily_panel_path),
            "signal_snapshots": 0,
            "candidate_orders": 0,
            "accepted_orders": 0,
            "filled_orders": 0,
        }

    frame = pl.read_parquet(daily_panel_path)
    return {
        "status": "ok",
        "execution_path": str(execution_path),
        "daily_panel_path": str(daily_panel_path),
        "signal_snapshots": _coerce_int(frame["signal_snapshots"].sum()) if "signal_snapshots" in frame.columns else 0,
        "candidate_orders": _coerce_int(frame["candidate_orders"].sum()) if "candidate_orders" in frame.columns else 0,
        "accepted_orders": _coerce_int(frame["accepted_orders"].sum()) if "accepted_orders" in frame.columns else 0,
        "filled_orders": _coerce_int(frame["filled_orders"].sum()) if "filled_orders" in frame.columns else 0,
    }


def build_shadow_telegram_summary(run_root: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_root": str(run_root),
        "run_stamp": run_root.name,
        "preflight": load_preflight_summary(run_root),
        "scavenger": load_scavenger_summary(run_root),
        "squeeze": load_squeeze_summary(run_root),
        "mid_tier": load_mid_tier_summary(run_root),
    }


def format_shadow_telegram_message(summary: dict[str, Any]) -> str:
    preflight = summary["preflight"]
    scavenger = summary["scavenger"]
    squeeze = summary["squeeze"]
    mid_tier = summary["mid_tier"]

    coverage_icon = "⚠️" if preflight.get("coverage_alarm") else "✅"
    fallback_suffix = " | fallback" if preflight.get("fallback_triggered") else ""
    preflight_line = (
        f"Coverage: {coverage_icon} {float(preflight.get('coverage_pct', 0.0)):.2f}% "
        f"({int(preflight.get('covered_market_count', 0))}/{int(preflight.get('active_market_count', 0))}) "
        f"| refresh={html.escape(str(preflight.get('refresh_status', 'unknown')))}{fallback_suffix}"
    )

    if scavenger.get("status") == "ok":
        scavenger_line = (
            f"Scavenger: targets={int(scavenger.get('targets_found', 0))} "
            f"| orders={int(scavenger.get('portfolio_orders_accepted', 0))} "
            f"| fills={int(scavenger.get('portfolio_fills', 0))}"
        )
    else:
        scavenger_line = "Scavenger: unavailable"

    if squeeze.get("status") == "ok":
        squeeze_line = (
            f"Squeeze: signals={int(squeeze.get('total_valid_signals_generated', 0))} "
            f"| baskets={int(squeeze.get('successful_fok_baskets', 0))} "
            f"| pairs={int(squeeze.get('pairs_completed', 0))}/{int(squeeze.get('pairs_requested', 0))}"
        )
    else:
        squeeze_line = "Squeeze: unavailable"

    if mid_tier.get("status") == "ok":
        mid_tier_line = (
            f"Mid-Tier: snapshots={int(mid_tier.get('signal_snapshots', 0))} "
            f"| candidates={int(mid_tier.get('candidate_orders', 0))} "
            f"| fills={int(mid_tier.get('filled_orders', 0))}"
        )
    else:
        mid_tier_line = "Mid-Tier: unavailable"

    return (
        f"👻 <b>Shadow Hourly</b> <code>{html.escape(str(summary['run_stamp']))}</code>\n"
        f"{preflight_line}\n"
        f"{scavenger_line}\n"
        f"{squeeze_line}\n"
        f"{mid_tier_line}"
    )


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_telegram_credentials() -> tuple[str, str]:
    bot_token = os.getenv("SHADOW_TELEGRAM_BOT_TOKEN") or settings.telegram_bot_token
    chat_id = os.getenv("SHADOW_TELEGRAM_CHAT_ID") or settings.telegram_chat_id
    return bot_token, chat_id


async def _send_summary(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    summary = build_shadow_telegram_summary(run_root)
    message = format_shadow_telegram_message(summary)

    summary_output = _summary_output_path(run_root, args.summary_output)
    send_status = "dry_run" if args.dry_run else "pending"
    summary["telegram"] = {
        "message": message,
        "status": send_status,
        "summary_output": str(summary_output),
    }

    if args.dry_run:
        _write_summary(summary_output, summary)
        print(message)
        return 0

    bot_token, chat_id = resolve_telegram_credentials()
    if not bot_token or not chat_id:
        summary["telegram"]["status"] = "disabled"
        _write_summary(summary_output, summary)
        print(
            "WARNING SHADOW_TELEGRAM_SUMMARY_SKIPPED "
            f"configured=false summary_output={summary_output}"
        )
        return 0

    alerter = TelegramAlerter(bot_token=bot_token, chat_id=chat_id)
    try:
        ok = await alerter.send_checked(message)
    finally:
        await alerter.close()

    summary["telegram"]["status"] = "sent" if ok else "failed"
    _write_summary(summary_output, summary)
    if ok:
        print(
            "SHADOW_TELEGRAM_SUMMARY_SENT "
            f"summary_output={summary_output} run_root={run_root}"
        )
    else:
        print(
            "WARNING SHADOW_TELEGRAM_SUMMARY_FAILED "
            f"summary_output={summary_output} run_root={run_root}"
        )
    return 0


def main() -> int:
    return asyncio.run(_send_summary(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())