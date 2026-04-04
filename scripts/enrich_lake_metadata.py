#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable

import httpx
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "artifacts" / "l2_parquet_lake_full"
DEFAULT_BATCH_SIZE = 20
DEFAULT_TIMEOUT_SECONDS = 30.0
ENRICHED_MANIFEST_NAME = "enriched_manifest.json"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


@dataclass(slots=True)
class GammaMarketRow:
    market_id: str
    event_id: str | None
    question: str
    gamma_closed: bool
    gamma_market_status: str
    resolution_timestamp: datetime | None
    final_resolution_value: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve centralized lake market metadata from Gamma and write an enriched manifest.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Lake root containing l2_book/ and manifest.json.",
    )
    parser.add_argument(
        "--gamma-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of condition_ids to request per Gamma API call.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout for Gamma requests.",
    )
    return parser.parse_args()


def _parquet_root(input_root: Path) -> Path:
    candidate = input_root / "l2_book"
    if candidate.is_dir():
        return candidate
    return input_root


def _discover_days(input_root: Path) -> list[str]:
    parquet_root = _parquet_root(input_root)
    days: set[str] = set()
    for child in parquet_root.iterdir():
        if not child.is_dir() or not child.name.startswith("date="):
            continue
        days.add(child.name.removeprefix("date="))
    return sorted(days)


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _discover_market_catalog(input_root: Path) -> pl.DataFrame:
    parquet_root = _parquet_root(input_root)
    target = str(parquet_root / "**" / "*.parquet")
    return (
        pl.scan_parquet(target, glob=True)
        .select(
            pl.col("market_id").cast(pl.String).str.to_lowercase().alias("market_id"),
            pl.col("event_id").cast(pl.String).alias("event_id"),
        )
        .unique(subset=["market_id"], keep="first")
        .sort("market_id")
        .collect()
    )


def fetch_gamma_market_rows(
    condition_ids: list[str],
    *,
    batch_size: int,
    timeout_seconds: float,
) -> dict[str, GammaMarketRow]:
    rows: dict[str, GammaMarketRow] = {}
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for offset in range(0, len(condition_ids), batch_size):
            batch = condition_ids[offset : offset + batch_size]
            response = client.get(
                GAMMA_MARKETS_URL,
                params=[("condition_ids", condition_id) for condition_id in batch],
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Gamma /markets response was not a list")

            for item in payload:
                if not isinstance(item, dict):
                    continue

                market_id = str(item.get("conditionId") or "").strip().lower()
                if not market_id:
                    continue

                event_id = str(item.get("eventId") or "").strip() or None
                events = item.get("events") or []
                first_event = events[0] if isinstance(events, list) and events and isinstance(events[0], dict) else {}
                if event_id is None:
                    event_id = str(first_event.get("id") or "").strip() or None

                outcomes = [str(value).strip().lower() for value in _parse_listish(item.get("outcomes"))]
                outcome_prices = _parse_listish(item.get("outcomePrices"))
                yes_price: float | None = None
                if "yes" in outcomes:
                    yes_index = outcomes.index("yes")
                    if yes_index < len(outcome_prices):
                        try:
                            yes_price = float(outcome_prices[yes_index])
                        except (TypeError, ValueError):
                            yes_price = None

                gamma_closed = bool(item.get("closed"))
                resolution_timestamp = _parse_timestamp(item.get("closedTime")) or _parse_timestamp(
                    first_event.get("closedTime")
                )
                if resolution_timestamp is None:
                    resolution_timestamp = _parse_timestamp(item.get("endDate")) or _parse_timestamp(
                        first_event.get("endDate")
                    )

                final_resolution_value = yes_price if gamma_closed and yes_price in {0.0, 1.0} else None
                gamma_market_status = "resolved"
                if final_resolution_value is None:
                    gamma_market_status = "closed_unresolved" if gamma_closed else "open"

                rows[market_id] = GammaMarketRow(
                    market_id=market_id,
                    event_id=event_id,
                    question=str(item.get("question") or "").strip(),
                    gamma_closed=gamma_closed,
                    gamma_market_status=gamma_market_status,
                    resolution_timestamp=resolution_timestamp,
                    final_resolution_value=final_resolution_value,
                )
    return rows


def _source_manifest_payload(input_root: Path) -> dict[str, Any] | None:
    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _manifest_market_rows(catalog: pl.DataFrame, gamma_rows: dict[str, GammaMarketRow]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in catalog.iter_rows(named=True):
        market_id = str(row["market_id"])
        gamma_row = gamma_rows.get(market_id)
        if gamma_row is None:
            rows.append(
                {
                    "market_id": market_id,
                    "event_id": str(row["event_id"]),
                    "question": "",
                    "gamma_closed": False,
                    "gamma_market_status": "missing_gamma",
                    "resolution_timestamp": None,
                    "final_resolution_value": None,
                }
            )
            continue

        rows.append(
            {
                "market_id": market_id,
                "event_id": gamma_row.event_id or str(row["event_id"]),
                "question": gamma_row.question,
                "gamma_closed": gamma_row.gamma_closed,
                "gamma_market_status": gamma_row.gamma_market_status,
                "resolution_timestamp": gamma_row.resolution_timestamp.isoformat() if gamma_row.resolution_timestamp else None,
                "final_resolution_value": gamma_row.final_resolution_value,
            }
        )
    return rows


def run_enrichment(
    input_root: str | Path,
    *,
    gamma_batch_size: int = DEFAULT_BATCH_SIZE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    fetcher: Callable[[list[str]], dict[str, GammaMarketRow]] | None = None,
) -> dict[str, Any]:
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Lake root does not exist: {input_root}")

    catalog = _discover_market_catalog(root)
    if catalog.is_empty():
        raise ValueError(f"No parquet-backed markets found under {root}")

    market_ids = catalog.get_column("market_id").to_list()
    gamma_rows = (
        fetcher(market_ids)
        if fetcher is not None
        else fetch_gamma_market_rows(
            market_ids,
            batch_size=gamma_batch_size,
            timeout_seconds=timeout_seconds,
        )
    )

    markets = _manifest_market_rows(catalog, gamma_rows)
    resolved_market_count = sum(1 for row in markets if row["gamma_market_status"] == "resolved")
    open_market_count = sum(1 for row in markets if row["gamma_market_status"] == "open")
    closed_unresolved_market_count = sum(1 for row in markets if row["gamma_market_status"] == "closed_unresolved")
    missing_gamma_market_count = sum(1 for row in markets if row["gamma_market_status"] == "missing_gamma")
    source_manifest = _source_manifest_payload(root)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_root": str(root.resolve()),
        "parquet_root": str(_parquet_root(root).resolve()),
        "gamma_market_count": len(gamma_rows),
        "market_count": len(markets),
        "resolved_market_count": resolved_market_count,
        "open_market_count": open_market_count,
        "closed_unresolved_market_count": closed_unresolved_market_count,
        "missing_gamma_market_count": missing_gamma_market_count,
        "days": source_manifest.get("days") if isinstance(source_manifest, dict) and source_manifest.get("days") else _discover_days(root),
        "strict_schema": source_manifest.get("strict_schema") if isinstance(source_manifest, dict) else None,
        "join_schema": {
            "gamma_market_status": "Utf8 resolved|open|closed_unresolved|missing_gamma",
            "gamma_closed": "Boolean",
            "resolution_timestamp": "ISO8601 UTC timestamp or null",
            "final_resolution_value": "Float64 YES settlement value when resolved, otherwise null",
        },
        "markets": markets,
    }

    (root / ENRICHED_MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = parse_args()
    payload = run_enrichment(
        args.input_root,
        gamma_batch_size=int(args.gamma_batch_size),
        timeout_seconds=float(args.timeout_seconds),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())