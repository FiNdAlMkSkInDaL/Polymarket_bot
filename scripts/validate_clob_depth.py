#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from py_clob_client.client import ClobClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.logger import get_logger, setup_logging


log = get_logger(__name__)

DEFAULT_INPUT = PROJECT_ROOT / "config" / "negative_risk_targets.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "executable_neg_risk.json"
DEFAULT_CLOB_URL = "https://clob.polymarket.com"


@dataclass(slots=True)
class ExecutableLeg:
    condition_id: str
    market_id: str
    question: str
    outcome_label: str
    yes_token_id: str
    no_token_id: str
    best_bid: float
    best_ask: float
    execution_side: str
    execution_price: float
    execution_size_shares: float
    execution_notional_usd: float
    market_volume_24h: float


@dataclass(slots=True)
class ExecutableGroup:
    event_id: str
    event_title: str
    event_slug: str
    outcome_count: int
    event_volume_24h: float
    inefficiency_type: str
    recommended_action: str
    launcher_family: str
    fee_tolerance: float
    validation_side: str
    execution_price_sum: float
    execution_edge_vs_fair_value: float
    threshold_to_beat: float
    min_leg_depth_usd_required: float
    min_leg_depth_usd_observed: float
    strip_max_size_shares_at_bbo: float
    strip_executable_notional_usd: float
    legs: list[ExecutableLeg]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate negative-risk Dutch-book candidates using live executable BBO prices and depth.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSON from find_negative_risk.py.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Refined executable output path.")
    parser.add_argument(
        "--fee-tolerance",
        type=float,
        default=0.02,
        help="Fee/slippage buffer applied against the 1.0 strip boundary, in dollars per full strip.",
    )
    parser.add_argument(
        "--min-leg-depth-usd",
        type=float,
        default=10.0,
        help="Minimum resting executable notional required at the BBO for every leg.",
    )
    parser.add_argument("--clob-url", default=DEFAULT_CLOB_URL, help="Polymarket CLOB base URL.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Reserved for future HTTP clients.")
    parser.add_argument("--log-dir", default="logs", help="Structured log directory.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _load_candidates(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError(f"Expected 'targets' list in {path}")
    return payload


def _best_level(levels: list[Any]) -> tuple[float, float]:
    if not levels:
        return 0.0, 0.0
    level = levels[-1]
    return _safe_float(getattr(level, "price", 0.0)), _safe_float(getattr(level, "size", 0.0))


def _extract_live_leg(clob_client: ClobClient, leg: dict[str, Any], *, execution_side: str) -> ExecutableLeg | None:
    yes_token_id = _clean_text(leg.get("yes_token_id"))
    no_token_id = _clean_text(leg.get("no_token_id"))
    if not yes_token_id or not no_token_id:
        return None

    book = clob_client.get_order_book(yes_token_id)
    bids = getattr(book, "bids", []) or []
    asks = getattr(book, "asks", []) or []
    best_bid, best_bid_size = _best_level(bids)
    best_ask, best_ask_size = _best_level(asks)

    if execution_side == "ask":
        execution_price = best_ask
        execution_size = best_ask_size
    else:
        execution_price = best_bid
        execution_size = best_bid_size

    if execution_price <= 0.0 or execution_size <= 0.0:
        return None

    return ExecutableLeg(
        condition_id=_clean_text(leg.get("condition_id")),
        market_id=_clean_text(leg.get("market_id")),
        question=_clean_text(leg.get("question")),
        outcome_label=_clean_text(leg.get("outcome_label")),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        best_bid=round(best_bid, 4),
        best_ask=round(best_ask, 4),
        execution_side=execution_side,
        execution_price=round(execution_price, 4),
        execution_size_shares=round(execution_size, 4),
        execution_notional_usd=round(execution_price * execution_size, 4),
        market_volume_24h=round(_safe_float(leg.get("market_volume_24h")), 2),
    )


def _validate_group(
    group: dict[str, Any],
    *,
    clob_client: ClobClient,
    fee_tolerance: float,
    min_leg_depth_usd: float,
) -> tuple[ExecutableGroup | None, str | None]:
    recommended_action = _clean_text(group.get("recommended_action"))
    if recommended_action == "BUY_YES_STRIP":
        execution_side = "ask"
        threshold = 1.0 - fee_tolerance
        comparator = lambda total: total < threshold
    elif recommended_action == "SELL_NO_STRIP":
        execution_side = "bid"
        threshold = 1.0 + fee_tolerance
        comparator = lambda total: total > threshold
    else:
        return None, "unsupported_action"

    raw_legs = group.get("legs")
    if not isinstance(raw_legs, list) or not raw_legs:
        return None, "missing_legs"

    live_legs: list[ExecutableLeg] = []
    for leg in raw_legs:
        if not isinstance(leg, dict):
            return None, "invalid_leg_payload"
        try:
            live_leg = _extract_live_leg(clob_client, leg, execution_side=execution_side)
        except Exception as exc:
            log.warning(
                "executable_neg_risk_leg_fetch_failed",
                event_id=_clean_text(group.get("event_id")),
                condition_id=_clean_text(leg.get("condition_id")),
                error=str(exc),
            )
            return None, "book_fetch_failed"
        if live_leg is None:
            return None, "missing_bbo_side"
        if live_leg.execution_notional_usd < min_leg_depth_usd:
            return None, "insufficient_leg_depth"
        live_legs.append(live_leg)

    execution_price_sum = round(sum(leg.execution_price for leg in live_legs), 4)
    if not comparator(execution_price_sum):
        return None, "no_executable_edge"

    min_leg_depth_observed = round(min(leg.execution_notional_usd for leg in live_legs), 4)
    strip_max_size_shares = round(min(leg.execution_size_shares for leg in live_legs), 4)
    strip_executable_notional_usd = round(strip_max_size_shares * execution_price_sum, 4)
    execution_edge_vs_fair_value = round(execution_price_sum - 1.0, 4)

    return (
        ExecutableGroup(
            event_id=_clean_text(group.get("event_id")),
            event_title=_clean_text(group.get("event_title")),
            event_slug=_clean_text(group.get("event_slug")),
            outcome_count=int(group.get("outcome_count") or len(live_legs)),
            event_volume_24h=round(_safe_float(group.get("event_volume_24h")), 2),
            inefficiency_type=_clean_text(group.get("inefficiency_type")),
            recommended_action=recommended_action,
            launcher_family=_clean_text(group.get("launcher_family")) or "CLOB_GROUP_ARB",
            fee_tolerance=round(fee_tolerance, 4),
            validation_side=execution_side,
            execution_price_sum=execution_price_sum,
            execution_edge_vs_fair_value=execution_edge_vs_fair_value,
            threshold_to_beat=round(threshold, 4),
            min_leg_depth_usd_required=round(min_leg_depth_usd, 4),
            min_leg_depth_usd_observed=min_leg_depth_observed,
            strip_max_size_shares_at_bbo=strip_max_size_shares,
            strip_executable_notional_usd=strip_executable_notional_usd,
            legs=sorted(live_legs, key=lambda row: row.execution_price, reverse=True),
        ),
        None,
    )


def main() -> int:
    args = _parse_args()
    setup_logging(
        log_dir=args.log_dir,
        level=getattr(__import__("logging"), args.log_level.upper()),
        log_file="validate_clob_depth.jsonl",
    )

    payload = _load_candidates(args.input)
    targets = payload.get("targets") or []
    clob_client = ClobClient(args.clob_url)

    passed_groups: list[ExecutableGroup] = []
    rejection_counts: Counter[str] = Counter()
    for group in targets:
        if not isinstance(group, dict):
            rejection_counts["invalid_group_payload"] += 1
            continue
        validated_group, rejection_reason = _validate_group(
            group,
            clob_client=clob_client,
            fee_tolerance=args.fee_tolerance,
            min_leg_depth_usd=args.min_leg_depth_usd,
        )
        if validated_group is None:
            rejection_counts[rejection_reason or "unknown_rejection"] += 1
            continue
        passed_groups.append(validated_group)

    passed_groups.sort(key=lambda row: (abs(row.execution_edge_vs_fair_value), row.event_volume_24h), reverse=True)

    output_payload = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_generated_at": payload.get("generated_at"),
        "candidate_groups": len(targets),
        "executable_groups": len(passed_groups),
        "filters": {
            "fee_tolerance": args.fee_tolerance,
            "min_leg_depth_usd": args.min_leg_depth_usd,
        },
        "rejections": dict(sorted(rejection_counts.items())),
        "targets": [asdict(group) for group in passed_groups],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    log.info(
        "executable_neg_risk_validation_complete",
        input_groups=len(targets),
        executable_groups=len(passed_groups),
        rejections=dict(rejection_counts),
        output_path=str(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())