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

import httpx
from py_clob_client.client import ClobClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logger import get_logger, setup_logging


log = get_logger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "flb_results_live.json"
DEFAULT_CLOB_URL = "https://clob.polymarket.com"
DEFAULT_MAX_SHIELD_TARGETS = 100
DISPLAY_CATEGORIES = {"crypto", "geopolitics", "sports", "politics", "policy", "macro", "business", "technology", "culture"}
THEME_KEYWORDS = {
    "crypto": {
        "airdrop", "bitcoin", "btc", "consensys", "crypto", "defi", "eth", "ethereum", "kraken", "metamask", "microstrategy", "ostium", "pump", "sol", "solana", "theo", "token", "usdai",
    },
    "geopolitics": {
        "annex", "capture", "china", "clash", "cuba", "disarm", "hezbollah", "india", "iran", "israel", "nato", "nuke", "qassem", "russia", "territory", "ukraine", "withdraw",
    },
    "sports": {
        "champions", "conference", "esports", "fifa", "finals", "league", "lck", "lpl", "nba", "playoffs", "qualify", "season", "tournament", "win", "world",
    },
    "policy": {
        "cap", "deductions", "fed", "gambling", "interest", "rates", "repealed", "tax", "withdraw",
    },
    "macro": {
        "economy", "gdp", "inflation", "rates", "recession", "unemployment",
    },
    "politics": {
        "candidate", "democratic", "election", "electoral", "impeach", "nomination", "nominee", "party", "presidential", "president", "republican", "senate", "trump",
    },
    "business": {
        "bankruptcy", "company", "ipo", "market", "merger", "revenue", "stock", "valuation",
    },
    "technology": {
        "ai", "launch", "software", "tech", "token",
    },
    "culture": {
        "album", "film", "game", "gta", "movie", "released", "release", "tv",
    },
}


@dataclass(slots=True)
class LiveFlbTarget:
    condition_id: str
    market_id: str
    event_id: str
    question: str
    category: str
    market_slug: str
    event_title: str
    yes_token_id: str
    no_token_id: str
    entry_yes_ask: float
    terminal_yes_ask: float
    max_yes_ask: float
    max_no_drawdown_cents: float | None
    best_yes_bid: float
    best_yes_ask: float
    yes_midpoint: float | None
    gamma_yes_price: float | None
    price_source: str
    market_volume_24h: float
    liquidity_clob_usd: float
    discovered_at: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan live Gamma markets and write fresh sub-5c YES longshot targets for the Shield underwriter.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path for live Shield targets.")
    parser.add_argument("--page-size", type=int, default=500, help="Gamma page size.")
    parser.add_argument("--max-pages", type=int, default=40, help="Maximum number of Gamma pages to scan.")
    parser.add_argument("--clob-url", default=DEFAULT_CLOB_URL, help="Polymarket CLOB base URL.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-yes-price", type=float, default=0.05, help="Maximum live YES price to qualify as a longshot target.")
    parser.add_argument("--min-yes-price", type=float, default=0.001, help="Minimum live YES price to retain in the target set.")
    parser.add_argument("--prefilter-max-yes-price", type=float, default=0.08, help="Gamma outcome-price prefilter before hitting the CLOB book.")
    parser.add_argument("--max-shield-targets", type=int, default=DEFAULT_MAX_SHIELD_TARGETS, help="Hard cap on live Shield targets emitted after ranking.")
    parser.add_argument("--log-dir", default="logs", help="Structured log directory.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _tokenise(text: str) -> set[str]:
    cleaned: list[str] = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return {part for part in "".join(cleaned).split() if part}


def infer_category(*, question: str, market_slug: str, event_title: str, market_category: str = "", event_category: str = "") -> str:
    for source in (event_category, market_category):
        text = _clean_text(source).lower()
        if text in DISPLAY_CATEGORIES:
            return text
    tokens = _tokenise(" ".join(filter(None, [question, market_slug, event_title, market_category, event_category])))
    best_category = "unknown"
    best_score = 0
    for category, keywords in THEME_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def _best_level(levels: list[Any]) -> tuple[float, float]:
    if not levels:
        return 0.0, 0.0
    level = levels[-1]
    return float(getattr(level, "price", 0.0) or 0.0), float(getattr(level, "size", 0.0) or 0.0)


def _extract_yes_no_metadata(market: dict[str, Any]) -> tuple[str, str, float | None]:
    token_ids = [str(item) for item in _parse_listish(market.get("clobTokenIds"))]
    outcomes = [str(item) for item in _parse_listish(market.get("outcomes"))]
    outcome_prices = [_safe_float(item) for item in _parse_listish(market.get("outcomePrices"))]

    yes_token_id = ""
    no_token_id = ""
    gamma_yes_price: float | None = None
    for index, outcome in enumerate(outcomes):
        normalized = outcome.strip().lower()
        token_id = token_ids[index] if index < len(token_ids) else ""
        price = outcome_prices[index] if index < len(outcome_prices) else None
        if normalized == "yes":
            yes_token_id = token_id
            gamma_yes_price = price
        elif normalized == "no":
            no_token_id = token_id
    return yes_token_id, no_token_id, gamma_yes_price


def select_reference_yes_price(*, best_bid: float, best_ask: float, gamma_yes_price: float | None, max_yes_price: float, min_yes_price: float) -> tuple[float | None, float | None, str | None]:
    midpoint: float | None = None
    midpoint_source: str | None = None
    if best_bid > 0.0 and best_ask > 0.0:
        midpoint = round((best_bid + best_ask) / 2.0, 6)
        midpoint_source = "midpoint"
    elif gamma_yes_price is not None and gamma_yes_price > 0.0:
        midpoint = gamma_yes_price
        midpoint_source = "gamma_outcome_price"

    if min_yes_price < best_ask < max_yes_price:
        return best_ask, midpoint, "best_ask"
    if midpoint is not None and min_yes_price < midpoint < max_yes_price:
        return midpoint, midpoint, midpoint_source
    return None, midpoint, None


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
            log.info("live_flb_gamma_page_scanned", page=page + 1, page_items=len(page_items), total_items=len(items))
            if len(payload) < page_size:
                break
            offset += page_size
    return items


def _build_target(
    market: dict[str, Any],
    *,
    clob_client: ClobClient,
    max_yes_price: float,
    min_yes_price: float,
) -> LiveFlbTarget | None:
    yes_token_id, no_token_id, gamma_yes_price = _extract_yes_no_metadata(market)
    if not yes_token_id or not no_token_id:
        return None

    book = clob_client.get_order_book(yes_token_id)
    bids = getattr(book, "bids", []) or []
    asks = getattr(book, "asks", []) or []
    best_bid, _ = _best_level(bids)
    best_ask, _ = _best_level(asks)
    reference_yes_price, midpoint, price_source = select_reference_yes_price(
        best_bid=best_bid,
        best_ask=best_ask,
        gamma_yes_price=gamma_yes_price,
        max_yes_price=max_yes_price,
        min_yes_price=min_yes_price,
    )
    if reference_yes_price is None or price_source is None:
        return None

    events = market.get("events") if isinstance(market.get("events"), list) else []
    first_event = events[0] if events and isinstance(events[0], dict) else {}
    question = _clean_text(market.get("question"))
    market_slug = _clean_text(market.get("slug"))
    event_title = _clean_text(first_event.get("title"))
    category = infer_category(
        question=question,
        market_slug=market_slug,
        event_title=event_title,
        market_category=_clean_text(market.get("category")),
        event_category=_clean_text(first_event.get("category")),
    )
    return LiveFlbTarget(
        condition_id=_clean_text(market.get("conditionId") or market.get("condition_id")),
        market_id=_clean_text(market.get("id")),
        event_id=_clean_text(first_event.get("id") or market.get("eventId") or market.get("event_id")),
        question=question,
        category=category,
        market_slug=market_slug,
        event_title=event_title,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        entry_yes_ask=round(reference_yes_price, 6),
        terminal_yes_ask=round(reference_yes_price, 6),
        max_yes_ask=round(reference_yes_price, 6),
        max_no_drawdown_cents=None,
        best_yes_bid=round(best_bid, 6),
        best_yes_ask=round(best_ask, 6),
        yes_midpoint=round(midpoint, 6) if midpoint is not None else None,
        gamma_yes_price=round(gamma_yes_price, 6) if gamma_yes_price is not None else None,
        price_source=price_source,
        market_volume_24h=round(float(market.get("volume24hrClob") or market.get("volume24hr") or market.get("volumeNum24hr") or 0.0), 2),
        liquidity_clob_usd=round(float(market.get("liquidityClob") or market.get("liquidity") or 0.0), 2),
        discovered_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )


def _build_payload(*, args: argparse.Namespace, gamma_markets: list[dict[str, Any]], targets: list[LiveFlbTarget], rejection_counts: Counter[str], gamma_prefilter_candidates: int, clob_books_requested: int, eligible_before_cap: int) -> dict[str, Any]:
    category_counts = Counter(target.category for target in targets)
    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scope": {
            "scan_source": "live_gamma_clob",
            "gamma_markets_scanned": len(gamma_markets),
            "yes_price_threshold": args.max_yes_price,
            "min_yes_price": args.min_yes_price,
            "prefilter_max_yes_price": args.prefilter_max_yes_price,
            "max_shield_targets": args.max_shield_targets,
            "clob_url": args.clob_url,
        },
        "summary": {
            "qualified_yes_longshots": len(targets),
            "resolved_bucket": {
                "count": 0,
                "resolved_yes": 0,
                "resolved_no": 0,
                "category_counts": {},
            },
            "active_bucket": {
                "count": len(targets),
                "resolved_yes": 0,
                "resolved_no": 0,
                "category_counts": dict(sorted(category_counts.items())),
            },
            "spike_above_10c_count": 0,
        },
        "discovery_stats": {
            "gamma_markets_scanned": len(gamma_markets),
            "gamma_prefilter_candidates": gamma_prefilter_candidates,
            "clob_books_requested": clob_books_requested,
            "eligible_longshots": eligible_before_cap,
            "eligible_longshots_before_cap": eligible_before_cap,
            "selected_longshots": len(targets),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "spike_markets_above_10c": [],
        "resolved_markets": [],
        "active_markets": [asdict(target) for target in targets],
    }


def rank_targets(targets: list[LiveFlbTarget], *, max_targets: int) -> list[LiveFlbTarget]:
    ranked = sorted(
        targets,
        key=lambda row: (
            -row.market_volume_24h,
            -row.liquidity_clob_usd,
            row.entry_yes_ask,
            row.question.lower(),
        ),
    )
    if max_targets <= 0:
        return ranked
    return ranked[:max_targets]


def main() -> int:
    args = _parse_args()
    setup_logging(
        log_dir=args.log_dir,
        level=getattr(__import__("logging"), args.log_level.upper()),
        log_file="live_flb_scanner.jsonl",
    )

    gamma_markets = _fetch_gamma_markets(page_size=args.page_size, max_pages=args.max_pages, timeout=args.timeout)
    clob_client = ClobClient(args.clob_url)
    targets: list[LiveFlbTarget] = []
    rejection_counts: Counter[str] = Counter()
    gamma_prefilter_candidates = 0
    clob_books_requested = 0

    for market in gamma_markets:
        if not market.get("active") or market.get("closed"):
            rejection_counts["inactive_or_closed"] += 1
            continue
        if not market.get("acceptingOrders", True) or not market.get("enableOrderBook", True):
            rejection_counts["not_accepting_or_no_orderbook"] += 1
            continue

        yes_token_id, no_token_id, gamma_yes_price = _extract_yes_no_metadata(market)
        if not yes_token_id or not no_token_id:
            rejection_counts["invalid_yes_no_shape"] += 1
            continue
        if gamma_yes_price is not None and gamma_yes_price >= args.prefilter_max_yes_price:
            rejection_counts["gamma_prefilter_not_longshot"] += 1
            continue

        gamma_prefilter_candidates += 1
        clob_books_requested += 1
        try:
            target = _build_target(
                market,
                clob_client=clob_client,
                max_yes_price=float(args.max_yes_price),
                min_yes_price=float(args.min_yes_price),
            )
        except Exception as exc:
            rejection_counts["book_fetch_failed"] += 1
            log.warning(
                "live_flb_book_fetch_failed",
                condition_id=_clean_text(market.get("conditionId") or market.get("condition_id")),
                error=str(exc),
            )
            continue

        if target is None:
            rejection_counts["not_sub5c_on_book"] += 1
            continue
        targets.append(target)

    eligible_before_cap = len(targets)
    targets = rank_targets(targets, max_targets=int(args.max_shield_targets))
    payload = _build_payload(
        args=args,
        gamma_markets=gamma_markets,
        targets=targets,
        rejection_counts=rejection_counts,
        gamma_prefilter_candidates=gamma_prefilter_candidates,
        clob_books_requested=clob_books_requested,
        eligible_before_cap=eligible_before_cap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info(
        "live_flb_scan_complete",
        gamma_markets_scanned=len(gamma_markets),
        gamma_prefilter_candidates=gamma_prefilter_candidates,
        clob_books_requested=clob_books_requested,
        eligible_longshots_before_cap=eligible_before_cap,
        selected_longshots=len(targets),
        output_path=str(args.output),
    )
    print(f"Gamma markets scanned: {len(gamma_markets)}")
    print(f"Gamma prefilter candidates: {gamma_prefilter_candidates}")
    print(f"CLOB books requested: {clob_books_requested}")
    print(f"Eligible live longshots before cap: {eligible_before_cap}")
    print(f"Selected live longshots: {len(targets)}")
    print(f"Live Shield target file written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())