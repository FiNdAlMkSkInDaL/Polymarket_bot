#!/usr/bin/env python3
"""Rank candidate markets by farming efficiency.

For each conditionId in the input universe, this script:
1. Fetches market metadata from Gamma.
2. Reads the active reward program from ``clobRewards``.
3. Pulls the last N hours of YES-token price history from the public CLOB
   ``/prices-history`` endpoint.
4. Computes daily realized volatility from hourly simple returns.
5. Uses the active reward program's daily USD amount from ``clobRewards`` as:

    daily_reward_usd = sum(active rewardsDailyRate)

   and expected daily drawdown as:

       expected_daily_drawdown_usd = daily_volatility * rewardsMinSize

6. Ranks markets by:

       farming_efficiency = daily_reward_usd / expected_daily_drawdown_usd

The risk denominator uses ``rewardsMinSize`` as the minimum qualifying
collateral required to farm the reward.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import stdev
from typing import Any

import httpx


GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"


@dataclass(slots=True)
class MarketEfficiency:
    condition_id: str
    title: str
    yes_token_id: str
    yes_price: float
    rewards_daily_rate: float
    collateral_usd: float
    daily_reward_usd: float
    daily_volatility: float
    expected_daily_drawdown_usd: float
    farming_efficiency: float
    history_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidate Polymarket markets by farming-efficiency score."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data/target_markets_monday.json"),
        help="JSON file containing a list of candidate conditionIds.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="History window in hours for realized volatility (default: 72).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of ranked markets to print (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional JSON export path for the computed rankings.",
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


def _parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _market_title(payload: dict[str, Any]) -> str:
    for key in ("question", "title", "slug"):
        value = payload.get(key)
        if value:
            return str(value).strip()
    return "<untitled market>"


def _active_reward_rate(payload: dict[str, Any], *, as_of: datetime) -> float:
    rewards = _parse_listish(payload.get("clobRewards"))
    total_rate = 0.0
    for reward in rewards:
        if not isinstance(reward, dict):
            continue
        start = _parse_date(reward.get("startDate"))
        end = _parse_date(reward.get("endDate"))
        if start is not None and as_of < start:
            continue
        if end is not None and as_of > end:
            continue
        total_rate += _safe_float(reward.get("rewardsDailyRate"))
    return total_rate


def _extract_yes_price(payload: dict[str, Any]) -> float:
    prices = _parse_listish(payload.get("outcomePrices"))
    if prices:
        return _safe_float(prices[0])
    return 0.0


def _extract_yes_token_id(payload: dict[str, Any]) -> str:
    token_ids = _parse_listish(payload.get("clobTokenIds"))
    return str(token_ids[0]) if token_ids else ""


def _fetch_market_payload(
    client: httpx.Client,
    *,
    condition_id: str,
) -> dict[str, Any] | None:
    response = client.get(
        GAMMA_MARKETS_URL,
        params={"condition_ids": condition_id, "limit": 10},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        return None
    item = payload[0]
    return item if isinstance(item, dict) else None


def _fetch_price_history(
    client: httpx.Client,
    *,
    token_id: str,
    hours: int,
    end_ts: int,
) -> list[tuple[int, float]]:
    start_ts = end_ts - hours * 3600
    response = client.get(
        CLOB_PRICE_HISTORY_URL,
        params={
            "market": token_id,
            "interval": "1h",
            "fidelity": 60,
            "startTs": start_ts,
            "endTs": end_ts,
        },
    )
    response.raise_for_status()
    payload = response.json()
    history = payload.get("history", []) if isinstance(payload, dict) else []
    result: list[tuple[int, float]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        ts = int(row.get("t", 0))
        price = _safe_float(row.get("p"))
        if ts <= 0 or price <= 0.0:
            continue
        result.append((ts, price))
    return result


def _compute_daily_volatility(history: list[tuple[int, float]]) -> float:
    if len(history) < 3:
        return 0.0
    prices = [price for _, price in history]
    hourly_returns: list[float] = []
    for prev_price, next_price in zip(prices, prices[1:]):
        if prev_price <= 0.0:
            continue
        hourly_returns.append((next_price / prev_price) - 1.0)
    if len(hourly_returns) < 2:
        return 0.0
    return stdev(hourly_returns) * math.sqrt(24.0)


def _format_money(value: float) -> str:
    return f"${value:,.4f}"


def load_condition_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [str(item).strip() for item in payload if str(item).strip()]


def compute_rankings(
    condition_ids: list[str],
    *,
    hours: int,
    timeout: float,
) -> list[MarketEfficiency]:
    results: list[MarketEfficiency] = []
    now = datetime.now(UTC)
    end_ts = int(now.timestamp())
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))

    with httpx.Client(timeout=client_timeout, headers={"User-Agent": "polymarket-farming-efficiency/1.0"}) as client:
        for condition_id in condition_ids:
            market = _fetch_market_payload(client, condition_id=condition_id)
            if market is None:
                continue

            yes_token_id = _extract_yes_token_id(market)
            if not yes_token_id:
                continue

            collateral_usd = _safe_float(market.get("rewardsMinSize"))
            rewards_daily_rate = _active_reward_rate(market, as_of=now)
            if collateral_usd <= 0.0 or rewards_daily_rate <= 0.0:
                continue

            history = _fetch_price_history(
                client,
                token_id=yes_token_id,
                hours=hours,
                end_ts=end_ts,
            )
            daily_volatility = _compute_daily_volatility(history)
            if daily_volatility <= 0.0:
                continue

            daily_reward_usd = rewards_daily_rate
            expected_daily_drawdown_usd = daily_volatility * collateral_usd
            farming_efficiency = daily_reward_usd / expected_daily_drawdown_usd

            results.append(
                MarketEfficiency(
                    condition_id=condition_id,
                    title=_market_title(market),
                    yes_token_id=yes_token_id,
                    yes_price=_extract_yes_price(market),
                    rewards_daily_rate=rewards_daily_rate,
                    collateral_usd=collateral_usd,
                    daily_reward_usd=daily_reward_usd,
                    daily_volatility=daily_volatility,
                    expected_daily_drawdown_usd=expected_daily_drawdown_usd,
                    farming_efficiency=farming_efficiency,
                    history_points=len(history),
                )
            )

    return sorted(
        results,
        key=lambda item: (item.farming_efficiency, item.daily_reward_usd),
        reverse=True,
    )


def print_rankings(rankings: list[MarketEfficiency], *, top_n: int, hours: int) -> None:
    selected = rankings[:top_n]
    if not selected:
        print("No candidate markets had both active farming rewards and enough price history.")
        return

    print(
        f"Top {len(selected)} markets by farming efficiency "
        f"(reward / (daily_volatility x collateral), {hours}h lookback)"
    )
    print(
        "Assumption: daily reward = sum(active clobRewards.rewardsDailyRate) in USD, "
        "and collateral = rewardsMinSize, using the YES token's hourly history."
    )
    print()

    for index, item in enumerate(selected, start=1):
        print(f"{index}. {item.title}")
        print(f"   condition_id: {item.condition_id}")
        print(f"   yes_price: {item.yes_price:.4f}")
        print(f"   reward_rate_daily: {item.rewards_daily_rate:.6f}")
        print(f"   collateral_usd: {_format_money(item.collateral_usd)}")
        print(f"   daily_reward_usd: {_format_money(item.daily_reward_usd)}")
        print(f"   daily_volatility: {item.daily_volatility:.4%}")
        print(
            "   expected_daily_drawdown_usd: "
            f"{_format_money(item.expected_daily_drawdown_usd)}"
        )
        print(f"   farming_efficiency: {item.farming_efficiency:.6f}")
        print(f"   history_points: {item.history_points}")
        print()


def export_rankings(rankings: list[MarketEfficiency], output_path: Path) -> None:
    payload = [
        {
            "condition_id": item.condition_id,
            "title": item.title,
            "yes_token_id": item.yes_token_id,
            "yes_price": item.yes_price,
            "rewards_daily_rate": item.rewards_daily_rate,
            "collateral_usd": item.collateral_usd,
            "daily_reward_usd": item.daily_reward_usd,
            "daily_volatility": item.daily_volatility,
            "expected_daily_drawdown_usd": item.expected_daily_drawdown_usd,
            "farming_efficiency": item.farming_efficiency,
            "history_points": item.history_points,
        }
        for item in rankings
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    condition_ids = load_condition_ids(args.input_json)
    rankings = compute_rankings(
        condition_ids,
        hours=args.hours,
        timeout=args.timeout,
    )
    print_rankings(rankings, top_n=args.top, hours=args.hours)
    if args.export_json is not None:
        export_rankings(rankings, args.export_json)
        print(f"Exported {len(rankings)} rows to {args.export_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())