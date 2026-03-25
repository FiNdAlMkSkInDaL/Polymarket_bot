#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.data.market_discovery import MarketInfo, _normalize_gamma_market, _parse_market


DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 0.5
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RelationshipValidationIssue:
    relationship_id: str
    field: str
    reason: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SI-10 relationship JSON against live Polymarket market metadata."
    )
    parser.add_argument("relationships_file", help="Path to the SI-10 relationship JSON file.")
    parser.add_argument(
        "--gamma-url",
        default="https://gamma-api.polymarket.com/markets",
        help="Gamma markets endpoint to query for validation.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retry attempts for Gamma API requests.",
    )
    parser.add_argument(
        "--backoff-base-s",
        type=float,
        default=DEFAULT_BACKOFF_BASE_S,
        help="Base exponential backoff delay between retries.",
    )
    return parser.parse_args()


def load_relationships(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("SI-10 relationships JSON must be a list of objects")

    relationships: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {index} is not an object")

        relationship_id = str(item.get("relationship_id") or item.get("id") or f"si10-{index}")
        relationships.append(
            {
                "relationship_id": relationship_id,
                "label": str(item.get("label") or item.get("name") or relationship_id),
                "base_a_condition_id": str(item.get("base_a_condition_id") or item.get("base_a") or ""),
                "base_b_condition_id": str(item.get("base_b_condition_id") or item.get("base_b") or ""),
                "joint_condition_id": str(item.get("joint_condition_id") or item.get("joint") or ""),
            }
        )
    return relationships


async def _async_get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    *,
    attempts: int,
    backoff_base_s: float,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url, params=params)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise httpx.HTTPStatusError(
                    f"retryable Gamma status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            if attempt == attempts:
                break
            await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))
    if last_error is None:
        raise RuntimeError("Gamma request failed without an exception")
    raise RuntimeError(f"Gamma request failed after {attempts} attempts: {last_error}") from last_error


async def fetch_gamma_market_index(
    *,
    gamma_url: str,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
) -> dict[str, dict[str, Any]]:
    timeout = httpx.Timeout(timeout_s)
    page_size = 500
    offset = 0
    markets: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            resp = await _async_get_with_retries(
                client,
                gamma_url,
                {"limit": page_size, "offset": offset},
                attempts=max_retries,
                backoff_base_s=backoff_base_s,
            )
            payload = resp.json()
            if not isinstance(payload, list) or not payload:
                break

            for item in payload:
                if not isinstance(item, dict):
                    continue
                condition_id = str(item.get("conditionId") or item.get("condition_id") or "")
                if condition_id:
                    markets[condition_id] = item

            if len(payload) < page_size:
                break
            offset += page_size

    return markets


def build_market_info(raw_market: dict[str, Any]) -> MarketInfo | None:
    schema = _normalize_gamma_market(raw_market)
    market_info, _reason = _parse_market(schema, min_volume=0.0, min_days_to_resolution=0)
    return market_info


def validate_relationships(
    relationships: list[dict[str, Any]],
    market_index: dict[str, dict[str, Any]],
) -> tuple[list[RelationshipValidationIssue], dict[str, dict[str, Any]]]:
    issues: list[RelationshipValidationIssue] = []
    resolved: dict[str, dict[str, Any]] = {}

    for row in relationships:
        relationship_id = row["relationship_id"]
        seen_ids: set[str] = set()
        resolved_row: dict[str, Any] = {
            "relationship_id": relationship_id,
            "label": row["label"],
            "markets": {},
        }

        for field in ("base_a_condition_id", "base_b_condition_id", "joint_condition_id"):
            condition_id = row.get(field, "")
            if not condition_id:
                issues.append(
                    RelationshipValidationIssue(
                        relationship_id=relationship_id,
                        field=field,
                        reason="missing_condition_id",
                    )
                )
                continue

            if condition_id in seen_ids:
                issues.append(
                    RelationshipValidationIssue(
                        relationship_id=relationship_id,
                        field=field,
                        reason="duplicate_condition_id",
                        detail=condition_id,
                    )
                )
            seen_ids.add(condition_id)

            raw_market = market_index.get(condition_id)
            if raw_market is None:
                issues.append(
                    RelationshipValidationIssue(
                        relationship_id=relationship_id,
                        field=field,
                        reason="condition_id_not_found",
                        detail=condition_id,
                    )
                )
                continue

            is_active = bool(raw_market.get("active", False)) and not bool(raw_market.get("closed", True))
            accepting_orders = bool(raw_market.get("acceptingOrders", True))
            if not is_active or not accepting_orders:
                issues.append(
                    RelationshipValidationIssue(
                        relationship_id=relationship_id,
                        field=field,
                        reason="market_not_active",
                        detail=condition_id,
                    )
                )
                continue

            market_info = build_market_info(raw_market)
            if market_info is None:
                issues.append(
                    RelationshipValidationIssue(
                        relationship_id=relationship_id,
                        field=field,
                        reason="market_failed_parse",
                        detail=condition_id,
                    )
                )
                continue

            resolved_row["markets"][field] = {
                "condition_id": market_info.condition_id,
                "question": market_info.question,
                "yes_token_id": market_info.yes_token_id,
                "no_token_id": market_info.no_token_id,
                "active": market_info.active,
                "accepting_orders": market_info.accepting_orders,
            }

        resolved[relationship_id] = resolved_row

    return issues, resolved


async def _async_main() -> int:
    args = parse_args()
    relationships = load_relationships(Path(args.relationships_file))
    market_index = await fetch_gamma_market_index(
        gamma_url=args.gamma_url,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        backoff_base_s=args.backoff_base_s,
    )
    issues, resolved = validate_relationships(relationships, market_index)

    report = {
        "relationships_file": str(Path(args.relationships_file).resolve()),
        "market_count": len(market_index),
        "relationship_count": len(relationships),
        "valid": len(issues) == 0,
        "issues": [issue.__dict__ for issue in issues],
        "resolved": resolved,
    }
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())