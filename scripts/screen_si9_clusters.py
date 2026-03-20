#!/usr/bin/env python3
"""Screen Polymarket for SI-9 negRisk event clusters.

This standalone script queries the public Gamma /events API, extracts active
mutually exclusive negRisk event groups, and prints the top clusters ranked by
their aggregate turnover relative to displayed liquidity.

Examples
--------
    python scripts/screen_si9_clusters.py
    python scripts/screen_si9_clusters.py --top 10 --min-volume 5000
    python scripts/screen_si9_clusters.py --export-json data/si9_clusters.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any

import httpx


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


@dataclass(slots=True)
class ClusterLeg:
    market_id: str
    condition_id: str
    question: str
    group_item_title: str
    volume_24h: float
    liquidity: float


@dataclass(slots=True)
class RankedCluster:
    event_id: str
    event_slug: str
    title: str
    leg_count: int
    total_volume_24h: float
    total_liquidity: float
    cluster_score: float
    legs: list[ClusterLeg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen active Polymarket negRisk event clusters for SI-9 using "
            "aggregate 24h volume divided by aggregate resting liquidity."
        )
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of clusters to print (default: 25).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Gamma event page size per request (default: 100).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum number of event pages to fetch (default: 20).",
    )
    parser.add_argument(
        "--min-legs",
        type=int,
        default=2,
        help="Minimum active negRisk legs required per cluster (default: 2).",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=6,
        help="Maximum active negRisk legs allowed per cluster (default: 6).",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Drop clusters below this aggregate 24h volume in USD (default: 0).",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=1.0,
        help=(
            "Drop clusters below this aggregate resting liquidity in USD. Values "
            "at or below zero are excluded from ranking because the score would "
            "be unstable (default: 1.0)."
        ),
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
        help=(
            "Write the top ranked SI-9 clusters to FILE as JSON, preserving "
            "CLOB-compatible conditionId hex strings for every leg."
        ),
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


def _clean_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _condition_id(value: Any) -> str:
    condition_id = str(value or "").strip()
    if not condition_id.startswith("0x"):
        return ""
    return condition_id


def _is_binary_market(payload: dict[str, Any]) -> bool:
    tokens = payload.get("tokens")
    if isinstance(tokens, list) and len(tokens) == 2:
        return True
    outcomes = _parse_listish(payload.get("outcomes"))
    return len(outcomes) == 2


def _extract_leg(payload: dict[str, Any]) -> ClusterLeg | None:
    active = bool(payload.get("active", False))
    closed = bool(payload.get("closed", True))
    accepting_orders = bool(payload.get("acceptingOrders", True))
    enable_order_book = bool(payload.get("enableOrderBook", True))
    neg_risk = bool(payload.get("negRisk", False))

    if not active or closed or not accepting_orders or not enable_order_book or not neg_risk:
        return None
    if not _is_binary_market(payload):
        return None

    market_id = _clean_text(payload.get("id"))
    condition_id = _condition_id(payload.get("conditionId") or payload.get("condition_id"))
    if not market_id or not condition_id:
        return None

    return ClusterLeg(
        market_id=market_id,
        condition_id=condition_id,
        question=_clean_text(payload.get("question"), "<untitled market>"),
        group_item_title=_clean_text(payload.get("groupItemTitle")),
        volume_24h=_safe_float(
            payload.get("volume24hrClob", payload.get("volume24hr", payload.get("volumeNum24hr", 0.0)))
        ),
        liquidity=_safe_float(
            payload.get("liquidityClob", payload.get("liquidityNum", payload.get("liquidity", 0.0)))
        ),
    )


def _extract_cluster(
    payload: dict[str, Any],
    *,
    min_legs: int,
    max_legs: int,
) -> RankedCluster | None:
    event_id = _clean_text(payload.get("id"))
    if not event_id:
        return None

    active = bool(payload.get("active", False))
    closed = bool(payload.get("closed", True))
    enable_neg_risk = bool(payload.get("enableNegRisk", payload.get("negRisk", False)))
    raw_markets = payload.get("markets")
    if not active or closed or not enable_neg_risk or not isinstance(raw_markets, list):
        return None

    seen_condition_ids: set[str] = set()
    legs: list[ClusterLeg] = []
    for raw_market in raw_markets:
        if not isinstance(raw_market, dict):
            continue
        leg = _extract_leg(raw_market)
        if leg is None or leg.condition_id in seen_condition_ids:
            continue
        seen_condition_ids.add(leg.condition_id)
        legs.append(leg)

    if len(legs) < min_legs or len(legs) > max_legs:
        return None

    total_volume_24h = sum(leg.volume_24h for leg in legs)
    total_liquidity = sum(leg.liquidity for leg in legs)
    if total_liquidity <= 0.0:
        return None

    return RankedCluster(
        event_id=event_id,
        event_slug=_clean_text(payload.get("slug")),
        title=_clean_text(payload.get("title"), "<untitled event>"),
        leg_count=len(legs),
        total_volume_24h=total_volume_24h,
        total_liquidity=total_liquidity,
        cluster_score=total_volume_24h / total_liquidity,
        legs=sorted(legs, key=lambda leg: (leg.group_item_title, leg.question)),
    )


def fetch_clusters(
    *,
    page_size: int,
    max_pages: int,
    timeout: float,
    min_legs: int,
    max_legs: int,
) -> list[RankedCluster]:
    clusters: list[RankedCluster] = []
    seen_event_ids: set[str] = set()
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))

    with httpx.Client(timeout=client_timeout, headers={"User-Agent": "polymarket-si9-screener/1.0"}) as client:
        for page_index in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": page_index * page_size,
            }
            response = _get_with_retries(client, GAMMA_EVENTS_URL, params)
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(items, list) or not items:
                break

            new_items = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                cluster = _extract_cluster(item, min_legs=min_legs, max_legs=max_legs)
                if cluster is None or cluster.event_id in seen_event_ids:
                    continue
                seen_event_ids.add(cluster.event_id)
                clusters.append(cluster)
                new_items += 1

            if len(items) < page_size or new_items == 0:
                break

    return clusters


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


def rank_clusters(
    clusters: list[RankedCluster],
    *,
    min_volume: float,
    min_liquidity: float,
) -> list[RankedCluster]:
    filtered = [
        cluster
        for cluster in clusters
        if cluster.total_volume_24h >= min_volume and cluster.total_liquidity >= min_liquidity
    ]
    return sorted(
        filtered,
        key=lambda cluster: (cluster.cluster_score, cluster.total_volume_24h, cluster.leg_count),
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


def print_table(clusters: list[RankedCluster], *, top_n: int) -> None:
    selected = clusters[:top_n]
    if not selected:
        print("No qualifying active SI-9 negRisk clusters matched the requested filters.")
        return

    event_width = max(len("Event ID"), max(len(cluster.event_id) for cluster in selected))
    legs_width = max(len("Legs"), max(len(str(cluster.leg_count)) for cluster in selected))
    volume_width = max(len("24h Volume"), max(len(_format_money(cluster.total_volume_24h)) for cluster in selected))
    liquidity_width = max(len("Liquidity"), max(len(_format_money(cluster.total_liquidity)) for cluster in selected))
    score_width = max(len("Cluster Score"), max(len(f"{cluster.cluster_score:,.2f}x") for cluster in selected))

    terminal_width = shutil.get_terminal_size(fallback=(140, 20)).columns
    fixed_width = 4 + event_width + 2 + legs_width + 2 + volume_width + 2 + liquidity_width + 2 + score_width
    title_width = max(28, min(64, terminal_width - fixed_width))

    header = (
        f"{'#':>2}  {'Event ID':<{event_width}}  {'Title':<{title_width}}  {'Legs':>{legs_width}}  "
        f"{'24h Volume':>{volume_width}}  {'Liquidity':>{liquidity_width}}  {'Cluster Score':>{score_width}}"
    )
    divider = "-" * len(header)

    print("Top Polymarket SI-9 negRisk clusters")
    print("Score = aggregate 24h volume / aggregate resting liquidity")
    print(divider)
    print(header)
    print(divider)

    for index, cluster in enumerate(selected, start=1):
        print(
            f"{index:>2}  "
            f"{cluster.event_id:<{event_width}}  "
            f"{_truncate(cluster.title, title_width):<{title_width}}  "
            f"{cluster.leg_count:>{legs_width}}  "
            f"{_format_money(cluster.total_volume_24h):>{volume_width}}  "
            f"{_format_money(cluster.total_liquidity):>{liquidity_width}}  "
            f"{cluster.cluster_score:>{score_width - 1},.2f}x"
        )

    print(divider)
    print(f"Ranked {len(clusters):,} qualifying SI-9 clusters; showing top {len(selected):,}.")
    print()
    print("Cluster legs")
    print("------------")
    for index, cluster in enumerate(selected, start=1):
        print(f"{index:>2}. {cluster.title} ({cluster.event_id})")
        for leg in cluster.legs:
            suffix = f" [{leg.group_item_title}]" if leg.group_item_title else ""
            print(f"    - {leg.question}{suffix}: {leg.condition_id}")
        print()


def _cluster_to_json(cluster: RankedCluster) -> dict[str, Any]:
    return {
        "event_id": cluster.event_id,
        "event_slug": cluster.event_slug,
        "title": cluster.title,
        "leg_count": cluster.leg_count,
        "total_volume_24h": cluster.total_volume_24h,
        "total_liquidity": cluster.total_liquidity,
        "cluster_score": cluster.cluster_score,
        "condition_ids": [leg.condition_id for leg in cluster.legs],
        "markets": [
            {
                "market_id": leg.market_id,
                "condition_id": leg.condition_id,
                "question": leg.question,
                "group_item_title": leg.group_item_title,
                "volume_24h": leg.volume_24h,
                "liquidity": leg.liquidity,
            }
            for leg in cluster.legs
        ],
    }


def export_clusters(clusters: list[RankedCluster], *, top_n: int, output_path: Path) -> None:
    selected = [_cluster_to_json(cluster) for cluster in clusters[:top_n]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.min_legs < 2:
        raise SystemExit("--min-legs must be at least 2 for mutually exclusive clusters")
    if args.max_legs < args.min_legs:
        raise SystemExit("--max-legs must be greater than or equal to --min-legs")

    clusters = fetch_clusters(
        page_size=args.page_size,
        max_pages=args.max_pages,
        timeout=args.timeout,
        min_legs=args.min_legs,
        max_legs=args.max_legs,
    )
    ranked = rank_clusters(
        clusters,
        min_volume=args.min_volume,
        min_liquidity=args.min_liquidity,
    )
    print_table(ranked, top_n=args.top)
    if args.export_json is not None:
        export_clusters(ranked, top_n=args.top, output_path=args.export_json)
        print(f"Exported {min(args.top, len(ranked))} SI-9 clusters to {args.export_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())