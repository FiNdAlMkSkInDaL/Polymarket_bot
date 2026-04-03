#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "config" / "negative_risk_targets.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "docs" / "negative_risk_report.md"


@dataclass(slots=True)
class OutcomeLeg:
    condition_id: str
    market_id: str
    question: str
    outcome_label: str
    yes_token_id: str
    no_token_id: str
    best_bid: float
    best_ask: float
    mid_price: float
    market_volume_24h: float


@dataclass(slots=True)
class NegativeRiskGroup:
    event_id: str
    event_title: str
    event_slug: str
    outcome_count: int
    event_volume_24h: float
    total_mid_price: float
    inefficiency_type: str
    edge_to_fair_value: float
    recommended_action: str
    launcher_family: str
    legs: list[OutcomeLeg]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find negative-risk Dutch-book opportunities across grouped Polymarket CLOB events.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT, help="Machine-readable output path.")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT, help="Markdown report output path.")
    parser.add_argument("--page-size", type=int, default=500, help="Gamma page size.")
    parser.add_argument("--max-pages", type=int, default=40, help="Maximum Gamma pages to scan.")
    parser.add_argument("--min-event-volume-24h", type=float, default=1000.0, help="Minimum 24h event volume in USD.")
    parser.add_argument("--over-round-threshold", type=float, default=1.05, help="Total YES mid-price threshold for sell-NO strip candidates.")
    parser.add_argument("--under-round-threshold", type=float, default=0.95, help="Total YES mid-price threshold for buy-YES strip candidates.")
    parser.add_argument("--min-outcomes", type=int, default=3, help="Minimum number of active outcome markets per event.")
    parser.add_argument("--top", type=int, default=10, help="Maximum number of opportunities to write.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--log-dir", default="logs", help="Structured log directory.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


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


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
            log.info("negative_risk_gamma_page_scanned", page=page + 1, page_items=len(page_items), total_items=len(items))
            if len(payload) < page_size:
                break
            offset += page_size
    return items


def _group_candidate_markets(markets: list[dict[str, Any]], *, min_outcomes: int, min_event_volume_24h: float) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for market in markets:
        if not market.get("active") or market.get("closed"):
            continue
        if not market.get("acceptingOrders", True) or not market.get("enableOrderBook", True):
            continue
        if not market.get("negRisk", False):
            continue

        events = market.get("events") or []
        event = events[0] if isinstance(events, list) and events else {}
        if not event or not event.get("enableNegRisk", False):
            continue

        event_id = _clean_text(event.get("id") or market.get("eventId"))
        if not event_id:
            continue

        token_ids = _parse_listish(market.get("clobTokenIds"))
        outcomes = _parse_listish(market.get("outcomes"))
        if len(token_ids) != 2 or len(outcomes) != 2:
            continue

        if event_id not in grouped:
            grouped[event_id] = (event, [])
        grouped[event_id][1].append(market)

    filtered: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for event_id, (event, event_markets) in grouped.items():
        event_volume = _safe_float(event.get("volume24hr"))
        if len(event_markets) < min_outcomes:
            continue
        if event_volume <= min_event_volume_24h:
            continue
        filtered[event_id] = (event, event_markets)
    return filtered


def _extract_leg_bbo(clob_client: ClobClient, market: dict[str, Any]) -> OutcomeLeg | None:
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
    best_bid = _safe_float(bids[-1].price if bids else 0.0)
    best_ask = _safe_float(asks[-1].price if asks else 0.0)
    if best_bid > 0.0 and best_ask > 0.0:
        mid_price = (best_bid + best_ask) / 2.0
    elif best_bid > 0.0:
        mid_price = best_bid
    elif best_ask > 0.0:
        mid_price = best_ask
    else:
        return None

    return OutcomeLeg(
        condition_id=_clean_text(market.get("conditionId") or market.get("condition_id")),
        market_id=_clean_text(market.get("id")),
        question=_clean_text(market.get("question")),
        outcome_label=_clean_text(market.get("groupItemTitle") or market.get("question")),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        best_bid=round(best_bid, 4),
        best_ask=round(best_ask, 4),
        mid_price=round(mid_price, 4),
        market_volume_24h=round(_safe_float(market.get("volume24hrClob") or market.get("volume24hr") or market.get("volumeNum24hr")), 2),
    )


def _discover_negative_risk_groups(
    grouped_markets: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    over_round_threshold: float,
    under_round_threshold: float,
) -> list[NegativeRiskGroup]:
    clob_client = ClobClient("https://clob.polymarket.com")
    results: list[NegativeRiskGroup] = []

    for event_id, (event, event_markets) in grouped_markets.items():
        legs: list[OutcomeLeg] = []
        incomplete = False
        for market in event_markets:
            leg = _extract_leg_bbo(clob_client, market)
            if leg is None:
                incomplete = True
                break
            legs.append(leg)
        if incomplete or len(legs) < 3:
            continue

        total_mid_price = round(sum(leg.mid_price for leg in legs), 4)
        if total_mid_price > over_round_threshold:
            inefficiency_type = "over_round"
            recommended_action = "SELL_NO_STRIP"
        elif total_mid_price < under_round_threshold:
            inefficiency_type = "under_round"
            recommended_action = "BUY_YES_STRIP"
        else:
            continue

        results.append(
            NegativeRiskGroup(
                event_id=event_id,
                event_title=_clean_text(event.get("title")),
                event_slug=_clean_text(event.get("slug")),
                outcome_count=len(legs),
                event_volume_24h=round(_safe_float(event.get("volume24hr")), 2),
                total_mid_price=total_mid_price,
                inefficiency_type=inefficiency_type,
                edge_to_fair_value=round(total_mid_price - 1.0, 4),
                recommended_action=recommended_action,
                launcher_family="CLOB_GROUP_ARB",
                legs=sorted(legs, key=lambda row: row.mid_price, reverse=True),
            )
        )

    results.sort(key=lambda row: (abs(row.edge_to_fair_value), row.event_volume_24h), reverse=True)
    return results


def _write_markdown_report(groups: list[NegativeRiskGroup], output_path: Path) -> None:
    lines = [
        "# Negative Risk Discovery",
        "",
        f"Identified {len(groups)} live grouped CLOB inefficiencies.",
        "",
    ]
    for group in groups:
        lines.append(f"## {group.event_title}")
        lines.append("")
        lines.append(f"- event_id: {group.event_id}")
        lines.append(f"- inefficiency_type: {group.inefficiency_type}")
        lines.append(f"- recommended_action: {group.recommended_action}")
        lines.append(f"- total_mid_price: {group.total_mid_price:.4f}")
        lines.append(f"- edge_to_fair_value: {group.edge_to_fair_value:+.4f}")
        lines.append(f"- event_volume_24h: ${group.event_volume_24h:,.2f}")
        lines.append(f"- outcome_count: {group.outcome_count}")
        lines.append("")
        lines.append("| Outcome | Mid | Bid | Ask | 24h Vol | Condition ID |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for leg in group.legs:
            lines.append(
                f"| {leg.outcome_label} | {leg.mid_price:.4f} | {leg.best_bid:.4f} | {leg.best_ask:.4f} | ${leg.market_volume_24h:,.2f} | {leg.condition_id} |"
            )
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    setup_logging(
        log_dir=args.log_dir,
        level=getattr(__import__("logging"), args.log_level.upper()),
        log_file="negative_risk_discovery.jsonl",
    )

    gamma_markets = _fetch_gamma_markets(
        page_size=args.page_size,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    grouped_markets = _group_candidate_markets(
        gamma_markets,
        min_outcomes=args.min_outcomes,
        min_event_volume_24h=args.min_event_volume_24h,
    )
    groups = _discover_negative_risk_groups(
        grouped_markets,
        over_round_threshold=args.over_round_threshold,
        under_round_threshold=args.under_round_threshold,
    )
    selected_groups = groups[: max(1, args.top)]

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "gamma_markets_scanned": len(gamma_markets),
        "eligible_grouped_events": len(grouped_markets),
        "reported_opportunities": len(selected_groups),
        "filters": {
            "min_event_volume_24h": args.min_event_volume_24h,
            "over_round_threshold": args.over_round_threshold,
            "under_round_threshold": args.under_round_threshold,
            "min_outcomes": args.min_outcomes,
        },
        "targets": [asdict(group) for group in selected_groups],
    }
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown_report(selected_groups, args.markdown_output)

    log.info(
        "negative_risk_discovery_complete",
        gamma_markets_scanned=len(gamma_markets),
        eligible_grouped_events=len(grouped_markets),
        reported_opportunities=len(selected_groups),
        json_output=str(args.json_output),
        markdown_output=str(args.markdown_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())