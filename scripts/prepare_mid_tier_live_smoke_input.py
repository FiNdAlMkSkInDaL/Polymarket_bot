#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import polars as pl
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.build_l2_parquet_lake import (
    DEFAULT_BATCH_LINES,
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_FLUSH_ROWS,
    MarketMetadata,
    RunStats,
    process_market_day,
)


DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw_ticks"
DEFAULT_DAYS = ("2026-03-20", "2026-03-21", "2026-03-22")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "mid_tier_probability_compression_live_input"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
INTERMEDIATE_DIRNAME = "_intermediate_l2"
SUMMARY_NAME = "build_summary.json"


@dataclass(slots=True)
class GammaMarketRow:
    market_id: str
    event_id: str
    yes_token_id: str
    no_token_id: str
    question: str
    closed: bool
    resolution_timestamp: datetime | None
    final_resolution_value: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real Mid-Tier smoke-test input root from raw live ticks and Gamma metadata.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root containing raw_ticks/YYYY-MM-DD/*.jsonl partitions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination root for the final date-partitioned smoke input.",
    )
    parser.add_argument(
        "--day",
        action="append",
        dest="days",
        default=[],
        help="Optional YYYY-MM-DD day to include. Defaults to the March 20-22 live slice.",
    )
    parser.add_argument(
        "--gamma-batch-size",
        type=int,
        default=20,
        help="Number of condition_ids to request per Gamma API call.",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=DEFAULT_BATCH_LINES,
        help="Raw JSONL lines to parse per Polars batch during reconstruction.",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=DEFAULT_FLUSH_ROWS,
        help="Buffered rows threshold before an intermediate parquet flush.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help="Zstd compression level for intermediate and final parquet outputs.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Preserve the intermediate YES/NO reconstruction tree under the output root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow deleting an existing output root before rebuilding it.",
    )
    return parser.parse_args()


def _selected_days(raw_days: list[str]) -> tuple[str, ...]:
    if raw_days:
        return tuple(sorted({value.strip() for value in raw_days if value.strip()}))
    return DEFAULT_DAYS


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


def _ensure_clean_output_root(output_root: Path, *, force: bool) -> None:
    resolved_output = output_root.resolve()
    if resolved_output in {PROJECT_ROOT.resolve(), DEFAULT_RAW_ROOT.resolve()}:
        raise SystemExit(f"Refusing to use unsafe output root: {output_root}")
    if output_root.exists():
        if not force and any(output_root.iterdir()):
            raise SystemExit(
                f"Output root {output_root} is not empty. Use --force or choose a new directory."
            )
        if force:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def discover_condition_ids(raw_root: Path, days: tuple[str, ...]) -> list[str]:
    condition_ids: set[str] = set()
    for day in days:
        day_dir = raw_root / day
        if not day_dir.exists():
            raise FileNotFoundError(f"Missing raw day partition: {day_dir}")
        condition_ids.update(path.stem.lower() for path in day_dir.glob("0x*.jsonl"))
    if not condition_ids:
        raise ValueError(f"No hex market files found under {raw_root} for {days}")
    return sorted(condition_ids)


def fetch_gamma_market_rows(condition_ids: list[str], *, batch_size: int) -> dict[str, GammaMarketRow]:
    rows: dict[str, GammaMarketRow] = {}

    for offset in range(0, len(condition_ids), batch_size):
        batch = condition_ids[offset : offset + batch_size]
        response = requests.get(
            GAMMA_MARKETS_URL,
            params=[("condition_ids", condition_id) for condition_id in batch],
            timeout=30,
        )
        response.raise_for_status()

        for item in response.json():
            market_id = str(item.get("conditionId") or "").strip().lower()
            if not market_id:
                continue

            event_id = str(item.get("eventId") or "").strip()
            events = item.get("events") or []
            first_event = events[0] if isinstance(events, list) and events and isinstance(events[0], dict) else {}
            if not event_id and isinstance(events, list) and events:
                event_id = str(first_event.get("id") or "").strip()
            if not event_id:
                continue

            token_ids = [str(value).strip() for value in _parse_listish(item.get("clobTokenIds")) if str(value).strip()]
            outcomes = [str(value).strip().lower() for value in _parse_listish(item.get("outcomes"))]
            prices_raw = _parse_listish(item.get("outcomePrices"))
            if len(token_ids) < 2 or len(outcomes) < 2 or len(prices_raw) < 2:
                continue

            outcome_map = {outcome: index for index, outcome in enumerate(outcomes)}
            if "yes" not in outcome_map or "no" not in outcome_map:
                continue

            yes_index = outcome_map["yes"]
            no_index = outcome_map["no"]
            yes_token_id = token_ids[yes_index]
            no_token_id = token_ids[no_index]

            try:
                yes_price = float(prices_raw[yes_index])
            except (TypeError, ValueError):
                yes_price = None

            resolution_timestamp = _parse_timestamp(item.get("closedTime")) or _parse_timestamp(
                first_event.get("closedTime")
            )
            if resolution_timestamp is None:
                resolution_timestamp = _parse_timestamp(item.get("endDate")) or _parse_timestamp(
                    first_event.get("endDate")
                )

            final_resolution_value = yes_price if item.get("closed") and yes_price in {0.0, 1.0} else None
            rows[market_id] = GammaMarketRow(
                market_id=market_id,
                event_id=event_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                question=str(item.get("question") or "").strip(),
                closed=bool(item.get("closed")),
                resolution_timestamp=resolution_timestamp,
                final_resolution_value=final_resolution_value,
            )

    missing = [condition_id for condition_id in condition_ids if condition_id not in rows]
    if missing:
        raise ValueError(f"Gamma metadata lookup missed {len(missing)} condition_ids")
    return rows


def build_intermediate_lake(
    *,
    raw_root: Path,
    days: tuple[str, ...],
    gamma_rows: dict[str, GammaMarketRow],
    intermediate_root: Path,
    batch_lines: int,
    flush_rows: int,
    compression_level: int,
) -> RunStats:
    stats = RunStats(metadata_rows_loaded=len(gamma_rows))
    metadata_by_market = {
        market_id: MarketMetadata(
            market_id=market_id,
            event_id=row.event_id,
            yes_asset_id=row.yes_token_id,
            no_asset_id=row.no_token_id,
        )
        for market_id, row in gamma_rows.items()
    }

    for day in days:
        day_dir = raw_root / day
        if not day_dir.exists():
            stats.markets_skipped["missing_day_partition"] += 1
            continue

        stats.days_processed += 1
        available = {path.stem.lower() for path in day_dir.glob("*.jsonl")}
        for metadata in metadata_by_market.values():
            if metadata.market_id not in available:
                continue
            stats.markets_considered += 1
            process_market_day(
                day=day,
                day_dir=day_dir,
                metadata=metadata,
                output_root=intermediate_root,
                batch_lines=batch_lines,
                flush_rows=flush_rows,
                compression_level=compression_level,
                stats=stats,
            )

    return stats


def build_final_lake(
    *,
    intermediate_root: Path,
    output_root: Path,
    days: tuple[str, ...],
    gamma_rows: dict[str, GammaMarketRow],
    compression_level: int,
) -> dict[str, Any]:
    enrichment = pl.DataFrame(
        [
            {
                "market_id": row.market_id,
                "event_id": row.event_id,
                "yes_token_id": row.yes_token_id,
                "resolution_timestamp": row.resolution_timestamp,
                "final_resolution_value": row.final_resolution_value,
            }
            for row in gamma_rows.values()
        ],
        schema={
            "market_id": pl.String,
            "event_id": pl.String,
            "yes_token_id": pl.String,
            "resolution_timestamp": pl.Datetime("ms", "UTC"),
            "final_resolution_value": pl.Float64,
        },
    )

    day_row_counts: dict[str, int] = {}
    rows_with_resolution = 0
    rows_without_resolution = 0

    for day in days:
        day_partition = intermediate_root / "l2_book" / f"date={day}"
        parquet_files = list(day_partition.glob("**/*.parquet")) if day_partition.exists() else []
        if not parquet_files:
            raise ValueError(f"No intermediate parquet files were produced for {day}")

        frame = (
            pl.scan_parquet(str(day_partition / "**" / "*.parquet"), glob=True)
            .filter(pl.col("token_id") == "YES")
            .join(enrichment.lazy(), on=["market_id", "event_id"], how="left")
            .with_columns(pl.col("yes_token_id").alias("token_id"))
            .select(
                "timestamp",
                "market_id",
                "event_id",
                "token_id",
                "best_bid",
                "best_ask",
                "bid_depth",
                "ask_depth",
                "resolution_timestamp",
                "final_resolution_value",
            )
            .collect()
        )

        if frame.is_empty():
            raise ValueError(f"Final smoke input for {day} is empty after filtering YES rows")

        day_output_dir = output_root / day
        day_output_dir.mkdir(parents=True, exist_ok=True)
        day_output_path = day_output_dir / "live_ticks.parquet"
        frame.write_parquet(
            day_output_path,
            compression="zstd",
            compression_level=compression_level,
            use_pyarrow=False,
        )

        day_row_counts[day] = frame.height
        resolved_count = int(frame.get_column("final_resolution_value").is_not_null().sum())
        rows_with_resolution += resolved_count
        rows_without_resolution += frame.height - resolved_count

    return {
        "day_row_counts": day_row_counts,
        "rows_with_resolution": rows_with_resolution,
        "rows_without_resolution": rows_without_resolution,
    }


def main() -> int:
    args = parse_args()
    days = _selected_days(args.days)
    output_root = args.output_root
    intermediate_root = output_root / INTERMEDIATE_DIRNAME

    _ensure_clean_output_root(output_root, force=args.force)

    condition_ids = discover_condition_ids(args.raw_root, days)
    gamma_rows = fetch_gamma_market_rows(condition_ids, batch_size=args.gamma_batch_size)

    intermediate_stats = build_intermediate_lake(
        raw_root=args.raw_root,
        days=days,
        gamma_rows=gamma_rows,
        intermediate_root=intermediate_root,
        batch_lines=args.batch_lines,
        flush_rows=args.flush_rows,
        compression_level=args.compression_level,
    )

    final_stats = build_final_lake(
        intermediate_root=intermediate_root,
        output_root=output_root,
        days=days,
        gamma_rows=gamma_rows,
        compression_level=args.compression_level,
    )

    resolved_event_counts = Counter(
        row.event_id for row in gamma_rows.values() if row.final_resolution_value is not None
    )
    signalable_events = sorted(
        event_id for event_id, count in resolved_event_counts.items() if count >= 3
    )
    summary = {
        "raw_root": str(args.raw_root),
        "output_root": str(output_root),
        "days": list(days),
        "condition_ids_discovered": len(condition_ids),
        "gamma_markets_loaded": len(gamma_rows),
        "closed_markets": sum(1 for row in gamma_rows.values() if row.closed),
        "resolved_markets": sum(1 for row in gamma_rows.values() if row.final_resolution_value is not None),
        "signalable_resolved_event_ids": signalable_events,
        "signalable_resolved_event_count": len(signalable_events),
        "intermediate_stats": intermediate_stats.to_json(),
        **final_stats,
    }

    summary_path = output_root / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if not args.keep_intermediate:
        shutil.rmtree(intermediate_root)

    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())