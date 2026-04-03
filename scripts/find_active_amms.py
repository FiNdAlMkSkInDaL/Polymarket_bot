#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.logger import get_logger, setup_logging
from src.data.alchemy_rpc_client import AlchemyRpcClient


log = get_logger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "config" / "active_amm_targets.json"


@dataclass(slots=True)
class AmmCandidate:
    condition_id: str
    market_id: str
    question: str
    market_maker_address: str
    yes_token_id: str
    no_token_id: str
    fpmm_live: bool
    gamma_liquidity: float
    gamma_volume_24hr: float


@dataclass(slots=True)
class VerifiedAmmTarget:
    condition_id: str
    market_id: str
    question: str
    market_maker_address: str
    yes_token_id: str
    no_token_id: str
    yes_reserve: str
    no_reserve: str
    combined_reserves: str
    fpmm_live: bool
    gamma_liquidity: float
    gamma_volume_24hr: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find active Polymarket AMM pools with verified on-chain liquidity.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path.")
    parser.add_argument("--page-size", type=int, default=500, help="Gamma page size.")
    parser.add_argument("--max-pages", type=int, default=40, help="Maximum Gamma market pages to scan.")
    parser.add_argument("--min-combined-reserves", type=Decimal, default=Decimal("500"), help="Minimum YES+NO reserve threshold.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of verified AMM pools to export.")
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


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_condition_id(value: Any) -> str:
    normalized = _clean_text(value).lower()
    if not normalized:
        return ""
    if not normalized.startswith("0x"):
        normalized = "0x" + normalized
    body = normalized[2:]
    if len(body) != 64:
        return ""
    try:
        int(body, 16)
    except ValueError:
        return ""
    return normalized


def _extract_yes_no_token_ids(market: dict[str, Any]) -> tuple[str, str] | None:
    native_tokens = market.get("tokens")
    tokens: list[dict[str, str]] = []
    if isinstance(native_tokens, list) and native_tokens:
        tokens = [
            {
                "token_id": _clean_text(token.get("token_id") or token.get("id")),
                "outcome": _clean_text(token.get("outcome")),
            }
            for token in native_tokens
            if isinstance(token, dict)
        ]
    else:
        clob_ids = _parse_listish(market.get("clobTokenIds"))
        outcomes = _parse_listish(market.get("outcomes"))
        tokens = [
            {"token_id": _clean_text(token_id), "outcome": _clean_text(outcome)}
            for token_id, outcome in zip(clob_ids, outcomes)
        ]

    yes_token_id = ""
    no_token_id = ""
    for token in tokens:
        outcome = token["outcome"].upper()
        if outcome == "YES":
            yes_token_id = token["token_id"]
        elif outcome == "NO":
            no_token_id = token["token_id"]
    if not yes_token_id or not no_token_id:
        return None
    return yes_token_id, no_token_id


def _iter_gamma_markets(client: httpx.Client, *, page_size: int, max_pages: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
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
        page_items = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(page_items, list) or not page_items:
            break
        dict_items = [item for item in page_items if isinstance(item, dict)]
        items.extend(dict_items)
        log.info("amm_gamma_page_scanned", page=page + 1, page_items=len(dict_items), total_items=len(items))
        if len(page_items) < page_size:
            break
        offset += page_size
    return items


def _collect_candidates(markets: list[dict[str, Any]]) -> list[AmmCandidate]:
    candidates: list[AmmCandidate] = []
    seen_condition_ids: set[str] = set()
    for market in markets:
        condition_id = _normalize_condition_id(market.get("conditionId") or market.get("condition_id"))
        if not condition_id or condition_id in seen_condition_ids:
            continue
        if not bool(market.get("active", False)):
            continue
        if bool(market.get("closed", True)):
            continue
        if not bool(market.get("acceptingOrders", True)):
            continue
        market_maker_address = _clean_text(market.get("marketMakerAddress"))
        if not market_maker_address:
            continue
        token_ids = _extract_yes_no_token_ids(market)
        if token_ids is None:
            continue
        seen_condition_ids.add(condition_id)
        candidates.append(
            AmmCandidate(
                condition_id=condition_id,
                market_id=_clean_text(market.get("id")),
                question=_clean_text(market.get("question")),
                market_maker_address=market_maker_address,
                yes_token_id=token_ids[0],
                no_token_id=token_ids[1],
                fpmm_live=bool(market.get("fpmmLive", False)),
                gamma_liquidity=_to_float(market.get("liquidityClob") or market.get("liquidity")),
                gamma_volume_24hr=_to_float(market.get("volume24hr") or market.get("volume24hrClob") or market.get("volumeNum24hr")),
            )
        )
    return candidates


def _verify_candidates(
    candidates: list[AmmCandidate],
    *,
    min_combined_reserves: Decimal,
) -> tuple[list[VerifiedAmmTarget], dict[str, int]]:
    stats = {
        "reserve_checks": 0,
        "reserve_errors": 0,
        "below_threshold": 0,
        "verified": 0,
    }
    verified: list[VerifiedAmmTarget] = []
    with AlchemyRpcClient() as alchemy_client:
        for candidate in candidates:
            stats["reserve_checks"] += 1
            try:
                reserves = alchemy_client.get_pool_reserves(candidate.condition_id, force_refresh=True)
            except Exception as exc:
                stats["reserve_errors"] += 1
                log.warning(
                    "amm_reserve_check_failed",
                    condition_id=candidate.condition_id,
                    market_maker_address=candidate.market_maker_address,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            combined_reserves = reserves.yes_reserve + reserves.no_reserve
            if combined_reserves <= min_combined_reserves:
                stats["below_threshold"] += 1
                continue

            verified.append(
                VerifiedAmmTarget(
                    condition_id=candidate.condition_id,
                    market_id=candidate.market_id,
                    question=candidate.question,
                    market_maker_address=reserves.market_maker_address,
                    yes_token_id=reserves.yes_token_id,
                    no_token_id=reserves.no_token_id,
                    yes_reserve=str(reserves.yes_reserve),
                    no_reserve=str(reserves.no_reserve),
                    combined_reserves=str(combined_reserves),
                    fpmm_live=candidate.fpmm_live,
                    gamma_liquidity=candidate.gamma_liquidity,
                    gamma_volume_24hr=candidate.gamma_volume_24hr,
                )
            )
            stats["verified"] += 1
    verified.sort(key=lambda row: Decimal(row.combined_reserves), reverse=True)
    return verified, stats


def main() -> int:
    args = _parse_args()
    setup_logging(
        log_dir=args.log_dir,
        level=getattr(__import__("logging"), args.log_level.upper()),
        log_file="active_amm_discovery.jsonl",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0))
    with httpx.Client(timeout=timeout) as client:
        markets = _iter_gamma_markets(client, page_size=args.page_size, max_pages=args.max_pages)

    candidates = _collect_candidates(markets)
    verified, stats = _verify_candidates(
        candidates,
        min_combined_reserves=args.min_combined_reserves,
    )
    selected = verified[: max(1, args.top_n)]

    payload = {
        "generated_at": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "gamma_markets_scanned": len(markets),
        "amm_candidates": len(candidates),
        "reserve_checks": stats["reserve_checks"],
        "reserve_errors": stats["reserve_errors"],
        "min_combined_reserves": str(args.min_combined_reserves),
        "verified_pool_count": len(verified),
        "selected_pool_count": len(selected),
        "targets": [asdict(row) for row in selected],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info(
        "active_amm_discovery_complete",
        gamma_markets_scanned=len(markets),
        amm_candidates=len(candidates),
        reserve_checks=stats["reserve_checks"],
        reserve_errors=stats["reserve_errors"],
        verified_pool_count=len(verified),
        selected_pool_count=len(selected),
        output_path=str(args.output),
    )
    if len(selected) < 10:
        log.warning(
            "active_amm_target_count_low",
            selected_pool_count=len(selected),
            threshold=10,
            min_combined_reserves=str(args.min_combined_reserves),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())