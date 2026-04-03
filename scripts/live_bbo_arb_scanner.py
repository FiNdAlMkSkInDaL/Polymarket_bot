#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from py_clob_client.client import ClobClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.logger import get_logger, setup_logging


log = get_logger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "live_executable_strips.json"
DEFAULT_CLOB_URL = "https://clob.polymarket.com"
THRESHOLD_PATTERN = re.compile(
    r"(at least|at most|or more|or less|more than|less than|above|below|under|over|>=|<=|>|<|\b\d+(?:\.\d+)?\+)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class LiveExecutableLeg:
    condition_id: str
    market_id: str
    question: str
    outcome_label: str
    yes_token_id: str
    no_token_id: str
    best_bid: float
    best_bid_size_shares: float
    best_bid_notional_usd: float
    best_ask: float
    best_ask_size_shares: float
    best_ask_notional_usd: float
    market_volume_24h: float


@dataclass(slots=True)
class LiveExecutableStrip:
    event_id: str
    event_title: str
    event_slug: str
    outcome_count: int
    event_volume_24h: float
    validation_mode: str
    recommended_action: str
    launcher_family: str
    execution_price_sum: float
    execution_edge_vs_fair_value: float
    fee_buffer: float
    min_leg_depth_usd_required: float
    min_leg_depth_usd_observed: float
    strip_max_size_shares_at_bbo: float
    strip_executable_notional_usd: float
    legs: list[LiveExecutableLeg]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan live Polymarket grouped markets for executable Dutch-book strips using BBO prices and depth only.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path for executable strip targets.")
    parser.add_argument("--page-size", type=int, default=500, help="Gamma page size.")
    parser.add_argument("--max-pages", type=int, default=40, help="Maximum number of Gamma pages to scan.")
    parser.add_argument("--fee-buffer", type=float, default=0.02, help="Execution buffer around the 1.0 strip boundary.")
    parser.add_argument("--min-leg-depth-usd", type=float, default=10.0, help="Minimum BBO notional required for every leg.")
    parser.add_argument("--min-outcomes", type=int, default=3, help="Minimum number of active legs required per grouped event.")
    parser.add_argument("--clob-url", default=DEFAULT_CLOB_URL, help="Polymarket CLOB base URL.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--log-dir", default="logs", help="Structured log directory.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _fetch_gamma_markets(*, page_size: int, max_pages: int, timeout: float) -> list[dict[str, Any]]:
    timeout_config = httpx.Timeout(timeout, connect=min(timeout, 10.0))
    items: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_config) as client:
        offset = 0
        for page in range(max_pages):
            response = client.get(
                GAMMA_MARKETS_URL,
                params={
                    "limit": page_size,
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            page_items = [item for item in payload if isinstance(item, dict)]
            items.extend(page_items)
            log.info("live_bbo_gamma_page_scanned", page=page + 1, page_items=len(page_items), total_items=len(items))
            if len(payload) < page_size:
                break
            offset += page_size
    return items


def _group_open_markets(markets: list[dict[str, Any]], *, min_outcomes: int) -> tuple[dict[str, tuple[dict[str, Any], list[dict[str, Any]]]], Counter[str]]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    counters: Counter[str] = Counter()

    for market in markets:
        if not market.get("active") or market.get("closed"):
            counters["inactive_or_closed"] += 1
            continue
        if not market.get("acceptingOrders", True) or not market.get("enableOrderBook", True):
            counters["not_accepting_or_no_orderbook"] += 1
            continue

        events = market.get("events") or []
        event = events[0] if isinstance(events, list) and events else {}
        event_id = _clean_text(event.get("id") or market.get("eventId"))
        if not event_id:
            counters["missing_event_id"] += 1
            continue

        token_ids = _parse_listish(market.get("clobTokenIds"))
        outcomes = _parse_listish(market.get("outcomes"))
        if len(token_ids) != 2 or len(outcomes) != 2:
            counters["invalid_token_or_outcome_shape"] += 1
            continue

        if event_id not in grouped:
            grouped[event_id] = (event, [])
        grouped[event_id][1].append(market)
        counters["markets_grouped"] += 1

    filtered: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for event_id, (event, event_markets) in grouped.items():
        if len(event_markets) < min_outcomes:
            counters["too_few_outcomes"] += 1
            continue
        filtered[event_id] = (event, event_markets)
        counters["events_with_min_outcomes"] += 1
        if any(market.get("negRisk", False) for market in event_markets):
            counters["grouped_events_with_any_negrisk_leg"] += 1
    return filtered, counters


def _best_level(levels: list[Any]) -> tuple[float, float]:
    if not levels:
        return 0.0, 0.0
    level = levels[-1]
    return _safe_float(getattr(level, "price", 0.0)), _safe_float(getattr(level, "size", 0.0))


def _is_cumulative_threshold_market(market: dict[str, Any]) -> bool:
    question = _clean_text(market.get("question"))
    label = _clean_text(market.get("groupItemTitle"))
    combined = f"{question} {label}".strip()
    if not combined:
        return False
    return bool(THRESHOLD_PATTERN.search(combined))


def _is_cumulative_threshold_group(event_markets: list[dict[str, Any]]) -> bool:
    threshold_hits = sum(1 for market in event_markets if _is_cumulative_threshold_market(market))
    return threshold_hits >= 2


def _extract_live_leg(clob_client: ClobClient, market: dict[str, Any]) -> LiveExecutableLeg | None:
    token_ids = _parse_listish(market.get("clobTokenIds"))
    if len(token_ids) != 2:
        return None

    yes_token_id = _clean_text(token_ids[0])
    no_token_id = _clean_text(token_ids[1])
    if not yes_token_id or not no_token_id:
        return None

    book = clob_client.get_order_book(yes_token_id)
    bids = getattr(book, "bids", []) or []
    asks = getattr(book, "asks", []) or []

    best_bid, best_bid_size = _best_level(bids)
    best_ask, best_ask_size = _best_level(asks)
    if best_bid <= 0.0 and best_ask <= 0.0:
        return None

    return LiveExecutableLeg(
        condition_id=_clean_text(market.get("conditionId") or market.get("condition_id")),
        market_id=_clean_text(market.get("id")),
        question=_clean_text(market.get("question")),
        outcome_label=_clean_text(market.get("groupItemTitle") or market.get("question")),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        best_bid=round(best_bid, 4),
        best_bid_size_shares=round(best_bid_size, 4),
        best_bid_notional_usd=round(best_bid * best_bid_size, 4),
        best_ask=round(best_ask, 4),
        best_ask_size_shares=round(best_ask_size, 4),
        best_ask_notional_usd=round(best_ask * best_ask_size, 4),
        market_volume_24h=round(_safe_float(market.get("volume24hrClob") or market.get("volume24hr") or market.get("volumeNum24hr")), 2),
    )


def _evaluate_group(
    event_id: str,
    event: dict[str, Any],
    event_markets: list[dict[str, Any]],
    *,
    clob_client: ClobClient,
    fee_buffer: float,
    min_leg_depth_usd: float,
) -> tuple[list[LiveExecutableStrip], str | None]:
    if not event.get("enableNegRisk", False):
        return [], "event_not_negrisk"
    if not all(market.get("negRisk", False) for market in event_markets):
        return [], "market_not_negrisk"
    if _is_cumulative_threshold_group(event_markets):
        return [], "cumulative_threshold_ladder"

    live_legs: list[LiveExecutableLeg] = []
    for market in event_markets:
        try:
            leg = _extract_live_leg(clob_client, market)
        except Exception as exc:
            log.warning(
                "live_bbo_leg_fetch_failed",
                event_id=event_id,
                condition_id=_clean_text(market.get("conditionId") or market.get("condition_id")),
                error=str(exc),
            )
            return [], "book_fetch_failed"
        if leg is None:
            return [], "missing_bbo"
        live_legs.append(leg)

    if len(live_legs) < 3:
        return [], "too_few_live_legs"

    results: list[LiveExecutableStrip] = []

    ask_sum = round(sum(leg.best_ask for leg in live_legs), 4)
    ask_threshold = 1.0 - fee_buffer
    ask_depths = [leg.best_ask_notional_usd for leg in live_legs]
    if ask_sum < ask_threshold and min(ask_depths) >= min_leg_depth_usd:
        min_depth = round(min(leg.best_ask_notional_usd for leg in live_legs), 4)
        strip_size = round(min(leg.best_ask_size_shares for leg in live_legs), 4)
        results.append(
            LiveExecutableStrip(
                event_id=event_id,
                event_title=_clean_text(event.get("title")),
                event_slug=_clean_text(event.get("slug")),
                outcome_count=len(live_legs),
                event_volume_24h=round(_safe_float(event.get("volume24hr")), 2),
                validation_mode="BUY_YES_STRIP",
                recommended_action="BUY_YES_STRIP",
                launcher_family="CLOB_GROUP_ARB",
                execution_price_sum=ask_sum,
                execution_edge_vs_fair_value=round(ask_sum - 1.0, 4),
                fee_buffer=round(fee_buffer, 4),
                min_leg_depth_usd_required=round(min_leg_depth_usd, 4),
                min_leg_depth_usd_observed=min_depth,
                strip_max_size_shares_at_bbo=strip_size,
                strip_executable_notional_usd=round(strip_size * ask_sum, 4),
                legs=sorted(live_legs, key=lambda row: row.best_ask, reverse=True),
            )
        )

    bid_sum = round(sum(leg.best_bid for leg in live_legs), 4)
    bid_threshold = 1.0 + fee_buffer
    bid_depths = [leg.best_bid_notional_usd for leg in live_legs]
    if bid_sum > bid_threshold and min(bid_depths) >= min_leg_depth_usd:
        min_depth = round(min(leg.best_bid_notional_usd for leg in live_legs), 4)
        strip_size = round(min(leg.best_bid_size_shares for leg in live_legs), 4)
        results.append(
            LiveExecutableStrip(
                event_id=event_id,
                event_title=_clean_text(event.get("title")),
                event_slug=_clean_text(event.get("slug")),
                outcome_count=len(live_legs),
                event_volume_24h=round(_safe_float(event.get("volume24hr")), 2),
                validation_mode="SELL_NO_STRIP",
                recommended_action="SELL_NO_STRIP",
                launcher_family="CLOB_GROUP_ARB",
                execution_price_sum=bid_sum,
                execution_edge_vs_fair_value=round(bid_sum - 1.0, 4),
                fee_buffer=round(fee_buffer, 4),
                min_leg_depth_usd_required=round(min_leg_depth_usd, 4),
                min_leg_depth_usd_observed=min_depth,
                strip_max_size_shares_at_bbo=strip_size,
                strip_executable_notional_usd=round(strip_size * bid_sum, 4),
                legs=sorted(live_legs, key=lambda row: row.best_bid, reverse=True),
            )
        )

    if results:
        return results, None
    if ask_sum < ask_threshold or bid_sum > bid_threshold:
        return [], "insufficient_leg_depth"
    return [], "no_executable_edge"


def main() -> int:
    args = _parse_args()
    setup_logging(
        log_dir=args.log_dir,
        level=getattr(__import__("logging"), args.log_level.upper()),
        log_file="live_bbo_arb_scanner.jsonl",
    )

    gamma_markets = _fetch_gamma_markets(page_size=args.page_size, max_pages=args.max_pages, timeout=args.timeout)
    grouped_events, grouping_counters = _group_open_markets(gamma_markets, min_outcomes=args.min_outcomes)

    clob_client = ClobClient(args.clob_url)
    results: list[LiveExecutableStrip] = []
    rejection_counts: Counter[str] = Counter()

    for event_id, (event, event_markets) in grouped_events.items():
        strips, rejection_reason = _evaluate_group(
            event_id,
            event,
            event_markets,
            clob_client=clob_client,
            fee_buffer=args.fee_buffer,
            min_leg_depth_usd=args.min_leg_depth_usd,
        )
        if strips:
            results.extend(strips)
            continue
        rejection_counts[rejection_reason or "unknown_rejection"] += 1

    results.sort(key=lambda row: (abs(row.execution_edge_vs_fair_value), row.event_volume_24h), reverse=True)

    output_payload = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "gamma_markets_scanned": len(gamma_markets),
        "grouped_events_considered": len(grouped_events),
        "executable_strips": len(results),
        "filters": {
            "fee_buffer": args.fee_buffer,
            "buy_yes_threshold": round(1.0 - args.fee_buffer, 4),
            "sell_no_threshold": round(1.0 + args.fee_buffer, 4),
            "min_leg_depth_usd": args.min_leg_depth_usd,
            "min_outcomes": args.min_outcomes,
        },
        "grouping_counters": dict(sorted(grouping_counters.items())),
        "rejections": dict(sorted(rejection_counts.items())),
        "targets": [asdict(result) for result in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    log.info(
        "live_bbo_arb_scan_complete",
        gamma_markets_scanned=len(gamma_markets),
        grouped_events_considered=len(grouped_events),
        executable_strips=len(results),
        rejections=dict(rejection_counts),
        output_path=str(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())