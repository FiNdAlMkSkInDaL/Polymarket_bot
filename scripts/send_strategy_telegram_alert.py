#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.telegram import TelegramAlerter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send Shield/Sword VPS scheduler alerts to Telegram.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shield = subparsers.add_parser("shield", help="Send a Shield paper summary alert.")
    shield.add_argument("--launch-summary", type=Path, required=True, help="Path to underwriter launch summary JSON.")

    sword = subparsers.add_parser("sword", help="Send a Sword paper summary alert.")
    sword.add_argument("--scan-summary", type=Path, required=True, help="Path to live_bbo_arb_scanner output JSON.")
    sword.add_argument("--launch-summary", type=Path, help="Optional path to clob arb launch summary JSON.")

    failure = subparsers.add_parser("failure", help="Send a pipeline failure alert.")
    failure.add_argument("--strategy", required=True, choices=("SHIELD", "SWORD", "MASTER"))
    failure.add_argument("--stage", required=True)
    failure.add_argument("--message", required=True)

    comm_check = subparsers.add_parser("comm-check", help="Send a strict Telegram connectivity check.")
    comm_check.add_argument("--message", default="VPS Telegram comm check OK", help="Connectivity-check message body.")

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


async def _run(args: argparse.Namespace) -> int:
    alerter = TelegramAlerter()
    try:
        if args.command == "shield":
            await alerter.notify_shield_paper_update(load_json(args.launch_summary))
            return 0
        if args.command == "sword":
            scan_summary = load_json(args.scan_summary)
            launch_summary = load_json(args.launch_summary) if args.launch_summary else None
            await alerter.notify_sword_paper_update(scan_summary=scan_summary, launch_summary=launch_summary)
            return 0
        if args.command == "failure":
            await alerter.notify_pipeline_failure(args.strategy, args.stage, args.message)
            return 0
        ok = await alerter.send_checked(args.message)
        return 0 if ok else 1
    finally:
        await alerter.close()


def main() -> None:
    raise SystemExit(asyncio.run(_run(parse_args())))


if __name__ == "__main__":
    main()