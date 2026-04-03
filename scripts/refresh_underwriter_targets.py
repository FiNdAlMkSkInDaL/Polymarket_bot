#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = PROJECT_ROOT / "data" / "flb_results_final.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "flb_results_live.json"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the active Shield target set from Gamma without remning historical ticks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Base FLB results file with active_markets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Refreshed live target output path.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Gamma HTTP timeout in seconds.")
    return parser.parse_args()


def _normalize_condition_id(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized and not normalized.startswith("0x"):
        normalized = "0x" + normalized
    return normalized


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


def fetch_market(client: httpx.Client, condition_id: str) -> dict[str, Any] | None:
    response = client.get(GAMMA_MARKETS_URL, params={"condition_ids": condition_id})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return None
    return payload[0]


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)
    active_markets = payload.get("active_markets")
    if not isinstance(active_markets, list):
        raise ValueError(f"Expected active_markets list in {args.input}")

    refreshed_active: list[dict[str, Any]] = []
    retired_markets: list[dict[str, Any]] = []

    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0))
    with httpx.Client(timeout=timeout) as client:
        for row in active_markets:
            if not isinstance(row, dict):
                continue
            condition_id = _normalize_condition_id(row.get("condition_id"))
            if not condition_id:
                continue
            market = fetch_market(client, condition_id)
            if market is None:
                retired_markets.append({"condition_id": condition_id, "reason": "missing_gamma_metadata"})
                continue
            is_active = bool(market.get("active", False))
            is_closed = bool(market.get("closed", False))
            accepting_orders = bool(market.get("acceptingOrders", True))
            enable_orderbook = bool(market.get("enableOrderBook", True))
            if not is_active or is_closed or not accepting_orders or not enable_orderbook:
                retired_markets.append(
                    {
                        "condition_id": condition_id,
                        "question": str(row.get("question") or market.get("question") or ""),
                        "reason": "inactive_or_closed",
                        "active": is_active,
                        "closed": is_closed,
                        "accepting_orders": accepting_orders,
                        "enable_orderbook": enable_orderbook,
                    }
                )
                continue
            refreshed_row = dict(row)
            refreshed_row["question"] = str(market.get("question") or row.get("question") or condition_id)
            refreshed_row["market_slug"] = str(market.get("slug") or row.get("market_slug") or "")
            refreshed_row["event_title"] = str((market.get("events") or [{}])[0].get("title") if isinstance(market.get("events"), list) and market.get("events") else row.get("event_title") or "")
            refreshed_row["refreshed_at"] = _utc_now()
            refreshed_active.append(refreshed_row)

    summary = dict(payload.get("summary") or {})
    active_bucket = dict(summary.get("active_bucket") or {})
    active_bucket["count"] = len(refreshed_active)
    summary["active_bucket"] = active_bucket

    refreshed_payload = {
        **payload,
        "generated_at": _utc_now(),
        "refresh_source": str(args.input),
        "summary": summary,
        "active_markets": refreshed_active,
        "retired_markets": retired_markets,
        "refresh_stats": {
            "input_active_markets": len(active_markets),
            "refreshed_active_markets": len(refreshed_active),
            "retired_markets": len(retired_markets),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(refreshed_payload, indent=2), encoding="utf-8")

    print(f"Input active markets: {len(active_markets)}")
    print(f"Refreshed active markets: {len(refreshed_active)}")
    print(f"Retired markets: {len(retired_markets)}")
    print(f"Refreshed target file written to {args.output}")


if __name__ == "__main__":
    main()