#!/usr/bin/env python3
"""Rank active Polymarket markets by wick potential.

This standalone script queries the public Gamma API, filters to active binary
markets with an enabled order book, and ranks them by a simple proxy for wick
potential:

    wick_score = volume_24h / resting_liquidity

High 24h volume combined with thin resting liquidity tends to produce books
that are more vulnerable to sudden sweeps, flash crashes, and large temporary
dislocations.

Examples
--------
    python scripts/screen_wick_markets.py
    python scripts/screen_wick_markets.py --top 25 --min-volume 1000
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from typing import Any

import httpx


GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


@dataclass(slots=True)
class RankedMarket:
    market_id: str
    condition_id: str
    title: str
    volume_24h: float
    liquidity: float
    wick_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen active Polymarket markets for high wick potential using "
            "24h volume divided by total resting liquidity."
        )
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of ranked markets to print (default: 25).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Gamma page size per request (default: 500).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum number of pages to fetch (default: 10).",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Drop markets below this 24h volume in USD (default: 0).",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=1.0,
        help=(
            "Drop markets below this resting liquidity in USD. Values at or below "
            "zero are excluded from ranking because the score would be unstable "
            "(default: 1.0)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds (default: 20).",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _market_title(payload: dict[str, Any]) -> str:
    for key in ("question", "title", "slug"):
        value = payload.get(key)
        if value:
            return str(value).strip()
    return "<untitled market>"


def _is_binary_market(payload: dict[str, Any]) -> bool:
    tokens = payload.get("tokens")
    if isinstance(tokens, list) and len(tokens) == 2:
        return True
    outcomes = _parse_listish(payload.get("outcomes"))
    return len(outcomes) == 2


def _extract_market(payload: dict[str, Any]) -> RankedMarket | None:
    active = bool(payload.get("active", False))
    closed = bool(payload.get("closed", True))
    accepting_orders = bool(payload.get("acceptingOrders", True))
    enable_order_book = bool(payload.get("enableOrderBook", True))
    neg_risk = bool(payload.get("negRisk", False))

    if not active or closed or not accepting_orders or not enable_order_book or neg_risk:
        return None
    if not _is_binary_market(payload):
        return None

    market_id = str(payload.get("id") or "").strip()
    condition_id = str(payload.get("conditionId") or payload.get("condition_id") or "").strip()
    if not market_id:
        return None

    volume_24h = _safe_float(
        payload.get("volume24hrClob", payload.get("volume24hr", payload.get("volumeNum24hr", 0.0)))
    )
    liquidity = _safe_float(
        payload.get("liquidityClob", payload.get("liquidityNum", payload.get("liquidity", 0.0)))
    )
    if liquidity <= 0.0:
        return None

    return RankedMarket(
        market_id=market_id,
        condition_id=condition_id,
        title=_market_title(payload),
        volume_24h=volume_24h,
        liquidity=liquidity,
        wick_score=volume_24h / liquidity,
    )


def fetch_active_markets(*, page_size: int, max_pages: int, timeout: float) -> list[RankedMarket]:
    markets: list[RankedMarket] = []
    seen_market_ids: set[str] = set()
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))

    with httpx.Client(timeout=client_timeout, headers={"User-Agent": "polymarket-wick-screener/1.0"}) as client:
        for page_index in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": page_index * page_size,
            }
            response = _get_with_retries(client, GAMMA_MARKETS_URL, params)
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(items, list) or not items:
                break

            new_items = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                market = _extract_market(item)
                if market is None or market.market_id in seen_market_ids:
                    continue
                seen_market_ids.add(market.market_id)
                markets.append(market)
                new_items += 1

            if len(items) < page_size or new_items == 0:
                break

    return markets


def _get_with_retries(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    attempts: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(base_delay * attempt)
    if last_error is None:
        raise RuntimeError("Gamma request failed without an exception")
    raise RuntimeError(f"Gamma request failed after {attempts} attempts: {last_error}") from last_error


def rank_markets(
    markets: list[RankedMarket],
    *,
    min_volume: float,
    min_liquidity: float,
) -> list[RankedMarket]:
    filtered = [
        market
        for market in markets
        if market.volume_24h >= min_volume and market.liquidity >= min_liquidity
    ]
    return sorted(
        filtered,
        key=lambda market: (market.wick_score, market.volume_24h),
        reverse=True,
    )


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_table(markets: list[RankedMarket], *, top_n: int) -> None:
    selected = markets[:top_n]
    if not selected:
        print("No qualifying active markets matched the requested filters.")
        return

    id_width = max(len("Market ID"), max(len(market.market_id) for market in selected))
    volume_width = max(len("24h Volume"), max(len(_format_money(market.volume_24h)) for market in selected))
    liquidity_width = max(len("Liquidity"), max(len(_format_money(market.liquidity)) for market in selected))
    score_width = max(len("Wick Score"), max(len(f"{market.wick_score:,.2f}x") for market in selected))

    terminal_width = shutil.get_terminal_size(fallback=(120, 20)).columns
    fixed_width = 4 + id_width + 2 + 2 + volume_width + 2 + liquidity_width + 2 + score_width
    title_width = max(24, min(60, terminal_width - fixed_width))

    header = (
        f"{'#':>2}  {'Market ID':<{id_width}}  {'Title':<{title_width}}  "
        f"{'24h Volume':>{volume_width}}  {'Liquidity':>{liquidity_width}}  {'Wick Score':>{score_width}}"
    )
    divider = "-" * len(header)

    print("Top Polymarket markets by wick potential")
    print("Score = 24h volume / resting liquidity")
    print(divider)
    print(header)
    print(divider)

    for index, market in enumerate(selected, start=1):
        print(
            f"{index:>2}  "
            f"{market.market_id:<{id_width}}  "
            f"{_truncate(market.title, title_width):<{title_width}}  "
            f"{_format_money(market.volume_24h):>{volume_width}}  "
            f"{_format_money(market.liquidity):>{liquidity_width}}  "
            f"{market.wick_score:>{score_width - 1},.2f}x"
        )

    print(divider)
    print(f"Ranked {len(markets):,} qualifying active markets; showing top {len(selected):,}.")


def main() -> int:
    args = parse_args()
    markets = fetch_active_markets(
        page_size=args.page_size,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    ranked = rank_markets(
        markets,
        min_volume=args.min_volume,
        min_liquidity=args.min_liquidity,
    )
    print_table(ranked, top_n=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
