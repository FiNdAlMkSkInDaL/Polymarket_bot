#!/usr/bin/env python3
"""Audit mid-tier Polymarket markets with active reward pools.

This script:
1. Fetches active binary markets from Gamma.
2. Filters to a user-defined 24h volume band.
3. Keeps markets with live CLOB reward pools.
4. Pulls current order books for the top reward-heavy markets.
5. Estimates current competition from weighted liquidity resting inside the
   reward spread window.

The competition estimate is a conservative proxy, not Polymarket's internal
score: for each token book we count only levels that both satisfy the reward
minimum size and sit inside the max-spread window around the current mid. Each
level is linearly down-weighted as it gets farther from the mid. The market's
competition score is the sum of paired two-sided weighted depth across both
token books.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from py_clob_client.client import ClobClient


GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"


@dataclass(slots=True)
class TokenAudit:
    token_id: str
    outcome: str
    mid_price: float
    best_bid: float
    best_ask: float
    reward_bid_depth_usd: float
    reward_ask_depth_usd: float
    paired_depth_usd: float


@dataclass(slots=True)
class CandidateMarket:
    market_id: str
    condition_id: str
    question: str
    volume_24h: float
    liquidity: float
    daily_reward_usd: float
    reward_max_spread_cents: float
    reward_min_size: float
    competitive_flag: bool
    token_audit: list[TokenAudit]

    @property
    def competition_usd(self) -> float:
        return round(sum(item.paired_depth_usd for item in self.token_audit), 2)

    @property
    def total_reward_zone_depth_usd(self) -> float:
        return round(
            sum(item.reward_bid_depth_usd + item.reward_ask_depth_usd for item in self.token_audit),
            2,
        )

    @property
    def capital_for_majority_usd(self) -> float:
        return round(self.competition_usd * 1.05, 2)

    @property
    def reward_to_competition(self) -> float:
        if self.competition_usd <= 0:
            return math.inf
        return self.daily_reward_usd / self.competition_usd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit mid-tier Polymarket markets with active reward pools."
    )
    parser.add_argument("--min-volume", type=float, default=10_000.0)
    parser.add_argument("--max-volume", type=float, default=75_000.0)
    parser.add_argument("--gamma-page-size", type=int, default=500)
    parser.add_argument("--gamma-max-pages", type=int, default=20)
    parser.add_argument("--top-reward-markets", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--clob-host", default=DEFAULT_CLOB_HOST)
    parser.add_argument("--export-json", type=Path, default=None)
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


def _parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if "T" in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        return date.fromisoformat(text)
    except ValueError:
        return None


def _active_rewards(rewards: Any, *, as_of: date | None = None) -> list[dict[str, Any]]:
    if not isinstance(rewards, list):
        return []
    today = as_of or datetime.now(timezone.utc).date()
    active: list[dict[str, Any]] = []
    for reward in rewards:
        if not isinstance(reward, dict):
            continue
        rate = _safe_float(reward.get("rewardsDailyRate"))
        if rate <= 0:
            continue
        start_date = _parse_iso_date(reward.get("startDate"))
        end_date = _parse_iso_date(reward.get("endDate"))
        if start_date and today < start_date:
            continue
        if end_date and today > end_date:
            continue
        active.append(reward)
    return active


def _get_with_retries(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    attempts: int = 4,
    base_delay: float = 1.0,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(base_delay * attempt)
    if last_error is None:
        raise RuntimeError("Gamma request failed without an exception")
    raise RuntimeError(f"Gamma request failed after {attempts} attempts: {last_error}") from last_error


def fetch_active_reward_markets(
    *,
    min_volume: float,
    max_volume: float,
    page_size: int,
    max_pages: int,
    timeout: float,
) -> list[dict[str, Any]]:
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))
    seen_market_ids: set[str] = set()
    qualified: list[dict[str, Any]] = []

    with httpx.Client(
        timeout=client_timeout,
        headers={"User-Agent": "polymarket-mid-tier-reward-audit/1.0"},
    ) as client:
        for page_index in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": page_index * page_size,
            }
            payload = _get_with_retries(client, GAMMA_MARKETS_URL, params).json()
            items = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(items, list) or not items:
                break

            new_count = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                market_id = str(item.get("id") or "").strip()
                if not market_id or market_id in seen_market_ids:
                    continue
                seen_market_ids.add(market_id)
                new_count += 1

                if not item.get("active") or item.get("closed"):
                    continue
                if not item.get("acceptingOrders") or not item.get("enableOrderBook"):
                    continue
                if item.get("negRisk"):
                    continue

                token_ids = _parse_listish(item.get("clobTokenIds"))
                outcomes = _parse_listish(item.get("outcomes"))
                if len(token_ids) != 2 or len(outcomes) != 2:
                    continue

                volume_24h = _safe_float(
                    item.get("volume24hrClob", item.get("volume24hr", item.get("volumeNum", 0.0)))
                )
                if volume_24h < min_volume or volume_24h > max_volume:
                    continue

                rewards = _active_rewards(item.get("clobRewards"))
                if not rewards:
                    continue
                daily_reward = sum(_safe_float(reward.get("rewardsDailyRate")) for reward in rewards)
                if daily_reward <= 0:
                    continue

                qualified.append(item)

            if len(items) < page_size or new_count == 0:
                break

    return qualified


def _book_mid_price(book: Any) -> float:
    bids = getattr(book, "bids", []) or []
    asks = getattr(book, "asks", []) or []
    best_bid = _safe_float(bids[-1].price if bids else 0.0)
    best_ask = _safe_float(asks[-1].price if asks else 0.0)
    if best_bid > 0 and best_ask > 0:
        return (best_bid + best_ask) / 2.0
    if best_bid > 0:
        return best_bid
    if best_ask > 0:
        return best_ask
    return 0.0


def _reward_zone_depth(
    levels: list[Any],
    *,
    mid_price: float,
    half_spread: float,
    min_size: float,
    side: str,
) -> float:
    if mid_price <= 0 or half_spread <= 0:
        return 0.0

    depth = 0.0
    lower = mid_price - half_spread
    upper = mid_price + half_spread
    for level in levels:
        price = _safe_float(getattr(level, "price", 0.0))
        size = _safe_float(getattr(level, "size", 0.0))
        if size < min_size or price <= 0:
            continue
        if side == "bid":
            if price > mid_price or price < lower:
                continue
            weight = (price - lower) / half_spread
        else:
            if price < mid_price or price > upper:
                continue
            weight = (upper - price) / half_spread
        if weight <= 0:
            continue
        depth += price * size * weight
    return round(depth, 2)


def audit_market_books(markets: list[dict[str, Any]], *, clob_host: str) -> list[CandidateMarket]:
    client = ClobClient(clob_host)
    audited: list[CandidateMarket] = []

    for market in markets:
        token_ids = _parse_listish(market.get("clobTokenIds"))
        outcomes = _parse_listish(market.get("outcomes"))
        reward_min_size = _safe_float(market.get("rewardsMinSize"), 0.0)
        reward_max_spread_cents = _safe_float(market.get("rewardsMaxSpread"), 0.0)
        half_spread = reward_max_spread_cents / 200.0
        token_audits: list[TokenAudit] = []

        for token_id, outcome in zip(token_ids, outcomes, strict=True):
            book = client.get_order_book(token_id)
            bids = getattr(book, "bids", []) or []
            asks = getattr(book, "asks", []) or []
            best_bid = _safe_float(bids[-1].price if bids else 0.0)
            best_ask = _safe_float(asks[-1].price if asks else 0.0)
            mid_price = _book_mid_price(book)
            reward_bid_depth = _reward_zone_depth(
                bids,
                mid_price=mid_price,
                half_spread=half_spread,
                min_size=reward_min_size,
                side="bid",
            )
            reward_ask_depth = _reward_zone_depth(
                asks,
                mid_price=mid_price,
                half_spread=half_spread,
                min_size=reward_min_size,
                side="ask",
            )
            token_audits.append(
                TokenAudit(
                    token_id=str(token_id),
                    outcome=str(outcome),
                    mid_price=round(mid_price, 4),
                    best_bid=round(best_bid, 4),
                    best_ask=round(best_ask, 4),
                    reward_bid_depth_usd=reward_bid_depth,
                    reward_ask_depth_usd=reward_ask_depth,
                    paired_depth_usd=round(min(reward_bid_depth, reward_ask_depth), 2),
                )
            )

        audited.append(
            CandidateMarket(
                market_id=str(market.get("id") or ""),
                condition_id=str(market.get("conditionId") or ""),
                question=str(market.get("question") or "<untitled market>"),
                volume_24h=_safe_float(market.get("volume24hrClob", market.get("volume24hr"))),
                liquidity=_safe_float(market.get("liquidityClob", market.get("liquidityNum", market.get("liquidity")))),
                daily_reward_usd=round(
                    sum(_safe_float(reward.get("rewardsDailyRate")) for reward in _active_rewards(market.get("clobRewards"))),
                    2,
                ),
                reward_max_spread_cents=reward_max_spread_cents,
                reward_min_size=reward_min_size,
                competitive_flag=bool(market.get("competitive")),
                token_audit=token_audits,
            )
        )

    return audited


def rank_candidates(markets: list[CandidateMarket], *, top_n: int) -> list[CandidateMarket]:
    ranked = sorted(
        markets,
        key=lambda market: (
            market.reward_to_competition,
            market.daily_reward_usd,
            -market.capital_for_majority_usd,
            market.volume_24h,
        ),
        reverse=True,
    )
    return ranked[:top_n]


def _format_money(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"${value:,.2f}"


def _format_ratio(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:,.2f}x"


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_table(markets: list[CandidateMarket]) -> None:
    if not markets:
        print("No qualifying rewarded mid-tier markets matched the requested filters.")
        return

    terminal_width = shutil.get_terminal_size(fallback=(160, 24)).columns
    vol_width = max(len("24h Vol"), max(len(_format_money(m.volume_24h)) for m in markets))
    reward_width = max(len("Reward/day"), max(len(_format_money(m.daily_reward_usd)) for m in markets))
    comp_width = max(len("Comp."), max(len(_format_money(m.competition_usd)) for m in markets))
    cap_width = max(len(">50% Cap"), max(len(_format_money(m.capital_for_majority_usd)) for m in markets))
    roi_width = max(len("Reward/Comp"), max(len(_format_ratio(m.reward_to_competition)) for m in markets))

    fixed_width = 4 + vol_width + reward_width + comp_width + cap_width + roi_width + 18
    title_width = max(28, min(70, terminal_width - fixed_width))

    header = (
        f"{'#':>2}  {'Market':<{title_width}}  {'24h Vol':>{vol_width}}  {'Reward/day':>{reward_width}}  "
        f"{'Comp.':>{comp_width}}  {'>50% Cap':>{cap_width}}  {'Reward/Comp':>{roi_width}}  Flag"
    )
    divider = "-" * len(header)

    print("Mid-tier rewarded market audit")
    print("Competition = paired weighted reward-zone depth across both token books")
    print(divider)
    print(header)
    print(divider)
    for index, market in enumerate(markets, start=1):
        flag = "hot" if market.competitive_flag else "cool"
        print(
            f"{index:>2}  {_truncate(market.question, title_width):<{title_width}}  "
            f"{_format_money(market.volume_24h):>{vol_width}}  "
            f"{_format_money(market.daily_reward_usd):>{reward_width}}  "
            f"{_format_money(market.competition_usd):>{comp_width}}  "
            f"{_format_money(market.capital_for_majority_usd):>{cap_width}}  "
            f"{_format_ratio(market.reward_to_competition):>{roi_width}}  {flag}"
        )
    print(divider)


def export_json(markets: list[CandidateMarket], output_path: Path) -> None:
    payload = []
    for market in markets:
        payload.append(
            {
                "market_id": market.market_id,
                "condition_id": market.condition_id,
                "question": market.question,
                "volume_24h": market.volume_24h,
                "liquidity": market.liquidity,
                "daily_reward_usd": market.daily_reward_usd,
                "reward_max_spread_cents": market.reward_max_spread_cents,
                "reward_min_size": market.reward_min_size,
                "competitive_flag": market.competitive_flag,
                "competition_usd": market.competition_usd,
                "total_reward_zone_depth_usd": market.total_reward_zone_depth_usd,
                "capital_for_majority_usd": market.capital_for_majority_usd,
                "reward_to_competition": None if math.isinf(market.reward_to_competition) else market.reward_to_competition,
                "token_audit": [
                    {
                        "token_id": item.token_id,
                        "outcome": item.outcome,
                        "mid_price": item.mid_price,
                        "best_bid": item.best_bid,
                        "best_ask": item.best_ask,
                        "reward_bid_depth_usd": item.reward_bid_depth_usd,
                        "reward_ask_depth_usd": item.reward_ask_depth_usd,
                        "paired_depth_usd": item.paired_depth_usd,
                    }
                    for item in market.token_audit
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    reward_markets = fetch_active_reward_markets(
        min_volume=args.min_volume,
        max_volume=args.max_volume,
        page_size=args.gamma_page_size,
        max_pages=args.gamma_max_pages,
        timeout=args.timeout,
    )

    reward_markets = sorted(
        reward_markets,
        key=lambda market: (
            sum(_safe_float(reward.get("rewardsDailyRate")) for reward in _active_rewards(market.get("clobRewards"))),
            _safe_float(market.get("volume24hrClob", market.get("volume24hr"))),
        ),
        reverse=True,
    )

    audited = audit_market_books(
        reward_markets[: args.top_reward_markets],
        clob_host=args.clob_host,
    )
    ranked = rank_candidates(audited, top_n=args.top_candidates)

    print_table(ranked)
    if args.export_json is not None:
        export_json(ranked, args.export_json)
        print(f"\nWrote {len(ranked)} candidate rows to {args.export_json}")


if __name__ == "__main__":
    main()