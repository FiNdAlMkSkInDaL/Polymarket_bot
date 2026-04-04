#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_conditional_probability_squeeze_batch import discover_source_files, resolve_input_scan_root


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "artifacts" / "l2_parquet_lake_full"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "clob_arb_baseline_metadata.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "config" / "squeeze_pairs.json"
MARKET_ID_FROM_FILE_RE = re.compile(r"part-(0x[0-9a-f]+)-", re.IGNORECASE)
SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ResolvedMarket:
    market_id: str
    question: str
    slug: str
    event_id: str
    event_slug: str
    event_title: str
    yes_token_id: str
    no_token_id: str
    neg_risk: bool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover Conditional Probability Squeeze pairs from a parquet lake and metadata cache.",
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Parquet lake root or l2_book root used to determine which markets are available.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Resolved metadata cache with markets_by_token entries.",
    )
    parser.add_argument(
        "--seed-config",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Existing squeeze pair config to preserve and extend.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON file for the merged squeeze pair config.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON path for discovery summary. Defaults to <output>.summary.json.",
    )
    return parser.parse_args(argv)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else []
    return []


def _slugify(value: str, *, fallback: str, max_length: int) -> str:
    slug = SLUG_RE.sub("-", value.lower()).strip("-")
    if not slug:
        slug = fallback
    if len(slug) <= max_length:
        return slug
    return slug[:max_length].rstrip("-")


def _normalize_pair_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and "pairs" in raw:
        raw_pairs = raw["pairs"]
    elif isinstance(raw, list):
        raw_pairs = raw
    elif isinstance(raw, dict):
        raw_pairs = []
        for parent_market_id, child_value in raw.items():
            if isinstance(child_value, str):
                raw_pairs.append(
                    {
                        "parent_market_id": parent_market_id,
                        "child_market_id": child_value,
                    }
                )
                continue
            if isinstance(child_value, dict):
                raw_pairs.append({"parent_market_id": parent_market_id, **child_value})
                continue
            raise ValueError("Unsupported squeeze pair mapping value in config JSON")
    else:
        raise ValueError("Pairs config must be a list, a {'pairs': [...]} object, or a parent->child mapping")

    entries: list[dict[str, Any]] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError("Each squeeze pair entry must be a JSON object")
        parent_market_id = _clean_text(raw_pair.get("parent_market_id"))
        child_market_id = _clean_text(raw_pair.get("child_market_id"))
        if not parent_market_id or not child_market_id:
            raise ValueError("Each squeeze pair entry must include parent_market_id and child_market_id")
        entry: dict[str, Any] = {
            "pair_id": _clean_text(raw_pair.get("pair_id")) or f"{parent_market_id}__{child_market_id}",
            "parent_market_id": parent_market_id.lower(),
            "child_market_id": child_market_id.lower(),
        }
        for key in (
            "parent_token_id",
            "child_token_id",
            "notes",
            "relationship_type",
            "event_id",
            "event_slug",
            "event_title",
            "parent_outcome",
            "child_outcome",
        ):
            value = raw_pair.get(key)
            if value is not None:
                entry[key] = _clean_text(value)
        entries.append(entry)
    return entries


def load_seed_pair_entries(config_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists():
        return []
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return _normalize_pair_entries(raw)


def _resolve_market_from_metadata_row(row: dict[str, Any]) -> ResolvedMarket | None:
    market_id = _clean_text(row.get("conditionId")).lower()
    if not market_id:
        return None

    events = _parse_listish(row.get("events"))
    first_event = events[0] if events and isinstance(events[0], dict) else {}
    event_id = _clean_text(first_event.get("id") or row.get("eventId") or row.get("event_id"))
    if not event_id:
        return None

    token_ids = [_clean_text(item) for item in _parse_listish(row.get("clobTokenIds"))]
    outcomes = [_clean_text(item).upper() for item in _parse_listish(row.get("outcomes"))]
    yes_token_id = ""
    no_token_id = ""
    if token_ids and outcomes and len(token_ids) == len(outcomes):
        for outcome in outcomes:
            if outcome == "YES":
                yes_token_id = "YES"
            elif outcome == "NO":
                no_token_id = "NO"
    elif len(token_ids) >= 2:
        yes_token_id = "YES"
        no_token_id = "NO"

    if not yes_token_id or not no_token_id:
        return None

    return ResolvedMarket(
        market_id=market_id,
        question=_clean_text(row.get("question")),
        slug=_clean_text(row.get("slug")),
        event_id=event_id,
        event_slug=_clean_text(first_event.get("slug")),
        event_title=_clean_text(first_event.get("title")),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        neg_risk=bool(row.get("negRisk") or first_event.get("negRisk")),
    )


def discover_available_market_ids(input_root: Path) -> tuple[set[str], tuple[Path, ...], Path]:
    scan_root = resolve_input_scan_root(input_root)
    source_files = discover_source_files(input_root)
    available_market_ids = {
        match.group(1).lower()
        for path in source_files
        if (match := MARKET_ID_FROM_FILE_RE.search(path.stem)) is not None
    }
    if available_market_ids:
        return available_market_ids, source_files, scan_root

    frame = (
        pl.scan_parquet([str(path) for path in source_files])
        .select(pl.col("market_id").cast(pl.String).str.to_lowercase().alias("market_id"))
        .unique()
        .collect(engine="streaming")
    )
    return set(frame.get_column("market_id").to_list()), source_files, scan_root


def load_resolved_markets(metadata_path: Path, available_market_ids: set[str]) -> dict[str, ResolvedMarket]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = payload.get("markets_by_token")
    if not isinstance(rows, dict):
        raise ValueError("Metadata cache is missing a markets_by_token mapping")

    resolved: dict[str, ResolvedMarket] = {}
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        market = _resolve_market_from_metadata_row(row)
        if market is None or market.market_id not in available_market_ids:
            continue
        resolved.setdefault(market.market_id, market)
    return resolved


def discover_mutually_exclusive_pairs(markets: dict[str, ResolvedMarket]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    markets_by_event: dict[str, list[ResolvedMarket]] = defaultdict(list)
    for market in markets.values():
        if market.neg_risk and market.event_id:
            markets_by_event[market.event_id].append(market)

    pair_entries: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    for event_id, event_markets in sorted(markets_by_event.items(), key=lambda item: item[0]):
        eligible = sorted(
            [market for market in event_markets if market.yes_token_id and market.no_token_id],
            key=lambda market: (market.slug or market.question, market.market_id),
        )
        if len(eligible) < 2:
            continue
        event_summaries.append(
            {
                "event_id": event_id,
                "event_slug": eligible[0].event_slug,
                "event_title": eligible[0].event_title,
                "market_count": len(eligible),
                "market_ids": [market.market_id for market in eligible],
                "questions": [market.question for market in eligible],
            }
        )
        event_slug = _slugify(
            eligible[0].event_slug or eligible[0].event_title,
            fallback=f"event-{event_id}",
            max_length=48,
        )
        for child_market in eligible:
            child_slug = _slugify(
                child_market.slug or child_market.question,
                fallback=f"market-{child_market.market_id[2:10]}",
                max_length=32,
            )
            for parent_market in eligible:
                if parent_market.market_id == child_market.market_id:
                    continue
                parent_slug = _slugify(
                    parent_market.slug or parent_market.question,
                    fallback=f"market-{parent_market.market_id[2:10]}",
                    max_length=32,
                )
                pair_entries.append(
                    {
                        "pair_id": (
                            f"mutex-{event_slug}-{child_slug}-yes-to-{parent_slug}-no-"
                            f"{child_market.market_id[2:10]}-{parent_market.market_id[2:10]}"
                        ),
                        "parent_market_id": parent_market.market_id,
                        "child_market_id": child_market.market_id,
                        "parent_token_id": parent_market.no_token_id,
                        "child_token_id": child_market.yes_token_id,
                        "notes": (
                            f"{child_market.question} YES implies {parent_market.question} NO because these are "
                            f"mutually exclusive outcomes within the same event."
                        ),
                        "relationship_type": "mutually_exclusive",
                        "event_id": event_id,
                        "event_slug": child_market.event_slug,
                        "event_title": child_market.event_title,
                        "parent_outcome": "NO",
                        "child_outcome": "YES",
                    }
                )
    pair_entries.sort(key=lambda item: item["pair_id"])
    event_summaries.sort(key=lambda item: (-int(item["market_count"]), item["event_slug"], item["event_id"]))
    return pair_entries, event_summaries


def _pair_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _clean_text(entry.get("parent_market_id")).lower(),
        _clean_text(entry.get("parent_token_id") or "YES").upper(),
        _clean_text(entry.get("child_market_id")).lower(),
        _clean_text(entry.get("child_token_id") or "YES").upper(),
    )


def build_squeeze_pairs_config(
    input_root: Path,
    metadata_path: Path,
    seed_config_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available_market_ids, source_files, scan_root = discover_available_market_ids(input_root)
    resolved_markets = load_resolved_markets(metadata_path, available_market_ids)
    seed_pairs = load_seed_pair_entries(seed_config_path)
    mutual_exclusion_pairs, event_summaries = discover_mutually_exclusive_pairs(resolved_markets)

    merged_pairs: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for entry in [*seed_pairs, *mutual_exclusion_pairs]:
        key = _pair_key(entry)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_pairs.append(entry)

    summary = {
        "requested_input_root": str(input_root),
        "scan_root": str(scan_root),
        "metadata_path": str(metadata_path),
        "seed_config_path": str(seed_config_path),
        "source_file_count": len(source_files),
        "available_market_count": len(available_market_ids),
        "resolved_market_count": len(resolved_markets),
        "seed_pair_count": len(seed_pairs),
        "mutually_exclusive_event_count": len(event_summaries),
        "mutually_exclusive_pair_count": len(mutual_exclusion_pairs),
        "total_pair_count": len(merged_pairs),
        "events": event_summaries,
    }
    return merged_pairs, summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    pairs, summary = build_squeeze_pairs_config(args.input_root, args.metadata_path, args.seed_config)
    output_payload = {"pairs": pairs}
    _write_json(args.output_path, output_payload)

    summary_path = args.summary_path
    if summary_path is None:
        summary_path = args.output_path.with_name(f"{args.output_path.stem}.summary.json")
    _write_json(summary_path, summary)

    print(
        f"Wrote squeeze pair config to {args.output_path} | "
        f"pairs={summary['total_pair_count']} | mutual_exclusion_pairs={summary['mutually_exclusive_pair_count']} | "
        f"events={summary['mutually_exclusive_event_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
