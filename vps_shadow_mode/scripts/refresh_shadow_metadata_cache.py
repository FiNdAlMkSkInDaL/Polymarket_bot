#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

import httpx
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "shadow_mode" else PROJECT_ROOT

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

DEFAULT_LAKE_ROOT = REPO_ROOT / "data" / "l2_book_live"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "artifacts" / "clob_arb_baseline_metadata.json"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "shadow_metadata_preflight_summary.json"
DEFAULT_GAMMA_BATCH_SIZE = 20
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_COVERAGE_THRESHOLD_PCT = 95.0


@dataclass(slots=True)
class MarketFetchResult:
    rows_by_condition_id: dict[str, dict[str, Any]]
    failed_condition_ids: list[str]


@dataclass(slots=True)
class EventFetchResult:
    rows_by_event_id: dict[str, dict[str, Any]]
    failed_event_ids: list[str]


def _current_utc_date() -> date:
    return datetime.now(tz=UTC).date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the shadow metadata cache for today's active live-lake markets and emit a coverage alarm.",
    )
    parser.add_argument("--lake-root", type=Path, default=DEFAULT_LAKE_ROOT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--since-date",
        default=_current_utc_date().isoformat(),
        help="UTC date partition used to define the active live universe (YYYY-MM-DD).",
    )
    parser.add_argument("--coverage-threshold-pct", type=float, default=DEFAULT_COVERAGE_THRESHOLD_PCT)
    parser.add_argument("--gamma-batch-size", type=int, default=DEFAULT_GAMMA_BATCH_SIZE)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
    return parser.parse_args()


def _parse_listish(raw_value: Any) -> list[Any]:
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str) and raw_value:
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _request_json_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: list[tuple[str, str]] | dict[str, str] | None,
    max_retries: int,
    retry_backoff_seconds: float,
) -> Any:
    delay_seconds = max(retry_backoff_seconds, 0.0)
    last_error: Exception | None = None
    for attempt in range(max(1, max_retries)):
        try:
            response = client.get(url, params=params)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"transient Gamma error: {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == max(1, max_retries) - 1:
                break
            time.sleep(delay_seconds)
            delay_seconds = delay_seconds * 2.0 if delay_seconds > 0.0 else 1.0
    if last_error is None:
        raise RuntimeError("Gamma request failed without an exception")
    raise last_error


def _build_condition_row_map(markets_by_token: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows_by_condition_id: dict[str, dict[str, Any]] = {}
    for row in markets_by_token.values():
        if not isinstance(row, dict):
            continue
        condition_id = str(row.get("conditionId") or "").strip().lower()
        if not condition_id:
            continue
        rows_by_condition_id.setdefault(condition_id, row)
    return rows_by_condition_id


def _scan_active_market_catalog(lake_root: Path, *, since_date: date) -> pl.DataFrame:
    parquet_paths: list[Path] = []
    partition_root = lake_root / f"date={since_date.isoformat()}"
    if partition_root.is_dir():
        parquet_paths = sorted(partition_root.rglob("*.parquet"))
    elif lake_root.is_dir():
        parquet_paths = sorted(lake_root.rglob("*.parquet"))

    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {lake_root}")

    start_of_day = datetime.combine(since_date, datetime.min.time(), tzinfo=UTC)
    catalog = (
        pl.scan_parquet([str(path) for path in parquet_paths])
        .group_by("market_id")
        .agg(
            pl.col("timestamp").max().alias("last_timestamp"),
            pl.col("event_id").drop_nulls().last().alias("event_id"),
        )
        .collect()
        .with_columns(pl.col("market_id").cast(pl.Utf8).str.to_lowercase())
        .sort("market_id")
    )
    if not partition_root.is_dir():
        catalog = catalog.filter(pl.col("last_timestamp") >= pl.lit(start_of_day))
    return catalog


def fetch_gamma_market_rows(
    condition_ids: list[str],
    *,
    batch_size: int,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> MarketFetchResult:
    rows_by_condition_id: dict[str, dict[str, Any]] = {}
    failed_condition_ids: list[str] = []
    if not condition_ids:
        return MarketFetchResult(rows_by_condition_id={}, failed_condition_ids=[])

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for start in range(0, len(condition_ids), max(1, batch_size)):
            batch = condition_ids[start : start + max(1, batch_size)]
            try:
                payload = _request_json_with_retries(
                    client,
                    GAMMA_MARKETS_URL,
                    params=[("condition_ids", condition_id) for condition_id in batch],
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
            except Exception:
                failed_condition_ids.extend(batch)
                continue

            if not isinstance(payload, list):
                failed_condition_ids.extend(batch)
                continue

            batch_rows: dict[str, dict[str, Any]] = {}
            for item in payload:
                if not isinstance(item, dict):
                    continue
                condition_id = str(item.get("conditionId") or "").strip().lower()
                if not condition_id:
                    continue
                batch_rows[condition_id] = item
            rows_by_condition_id.update(batch_rows)
            failed_condition_ids.extend(sorted(set(batch) - set(batch_rows)))

    return MarketFetchResult(
        rows_by_condition_id=rows_by_condition_id,
        failed_condition_ids=sorted(set(failed_condition_ids)),
    )


def _event_ids_from_market_rows(rows_by_condition_id: dict[str, dict[str, Any]]) -> list[str]:
    event_ids: set[str] = set()
    for row in rows_by_condition_id.values():
        event_id = str(row.get("eventId") or "").strip()
        if event_id:
            event_ids.add(event_id)
        events = row.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("id") or "").strip()
            if event_id:
                event_ids.add(event_id)
    return sorted(event_ids)


def fetch_gamma_event_rows(
    event_ids: list[str],
    *,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> EventFetchResult:
    rows_by_event_id: dict[str, dict[str, Any]] = {}
    failed_event_ids: list[str] = []
    if not event_ids:
        return EventFetchResult(rows_by_event_id={}, failed_event_ids=[])

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for event_id in event_ids:
            payload: Any
            try:
                payload = _request_json_with_retries(
                    client,
                    f"{GAMMA_EVENTS_URL}/{event_id}",
                    params=None,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
            except Exception:
                try:
                    payload = _request_json_with_retries(
                        client,
                        GAMMA_EVENTS_URL,
                        params={"id": event_id},
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                except Exception:
                    failed_event_ids.append(event_id)
                    continue

            if isinstance(payload, list):
                payload = payload[0] if payload else None
            if not isinstance(payload, dict):
                failed_event_ids.append(event_id)
                continue
            rows_by_event_id[event_id] = payload

    return EventFetchResult(
        rows_by_event_id=rows_by_event_id,
        failed_event_ids=sorted(set(failed_event_ids)),
    )


def _seed_events_from_market_rows(
    rows_by_condition_id: dict[str, dict[str, Any]],
    *,
    existing_events_by_id: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    events_by_id: dict[str, dict[str, Any]] = {}
    for event_id, payload in existing_events_by_id.items():
        if isinstance(payload, dict):
            events_by_id[str(event_id)] = dict(payload)

    for row in rows_by_condition_id.values():
        events = row.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("id") or "").strip()
            if not event_id:
                continue
            existing = events_by_id.get(event_id, {})
            events_by_id[event_id] = {**event, **existing}
    return events_by_id


def _build_markets_by_token(rows_by_condition_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    markets_by_token: dict[str, dict[str, Any]] = {}
    for row in rows_by_condition_id.values():
        token_ids = _parse_listish(row.get("clobTokenIds"))
        for token_id in token_ids:
            token_text = str(token_id).strip()
            if token_text:
                markets_by_token[token_text] = row
    return markets_by_token


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def refresh_shadow_metadata_cache(
    lake_root: Path,
    *,
    cache_path: Path,
    summary_path: Path | None,
    since_date: date,
    coverage_threshold_pct: float,
    gamma_batch_size: int,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    market_fetcher: Callable[[list[str]], MarketFetchResult] | None = None,
    event_fetcher: Callable[[list[str]], EventFetchResult] | None = None,
) -> dict[str, Any]:
    active_catalog = _scan_active_market_catalog(lake_root, since_date=since_date)
    active_condition_ids = active_catalog.get_column("market_id").to_list()

    old_payload = _read_json_file(cache_path)
    old_markets_by_token = old_payload.get("markets_by_token") if isinstance(old_payload, dict) else None
    old_events_by_id = old_payload.get("events_by_id") if isinstance(old_payload, dict) else None
    old_markets_by_token = old_markets_by_token if isinstance(old_markets_by_token, dict) else {}
    old_events_by_id = old_events_by_id if isinstance(old_events_by_id, dict) else {}
    old_rows_by_condition_id = _build_condition_row_map(old_markets_by_token)

    market_fetch_result = (
        market_fetcher(active_condition_ids)
        if market_fetcher is not None
        else fetch_gamma_market_rows(
            active_condition_ids,
            batch_size=gamma_batch_size,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    )

    merged_rows_by_condition_id: dict[str, dict[str, Any]] = {}
    reused_condition_ids: list[str] = []
    missing_condition_ids: list[str] = []
    for condition_id in active_condition_ids:
        if condition_id in market_fetch_result.rows_by_condition_id:
            merged_rows_by_condition_id[condition_id] = market_fetch_result.rows_by_condition_id[condition_id]
            continue
        if condition_id in old_rows_by_condition_id:
            merged_rows_by_condition_id[condition_id] = old_rows_by_condition_id[condition_id]
            reused_condition_ids.append(condition_id)
            continue
        missing_condition_ids.append(condition_id)

    seeded_events_by_id = _seed_events_from_market_rows(
        merged_rows_by_condition_id,
        existing_events_by_id=old_events_by_id,
    )
    event_ids = _event_ids_from_market_rows(merged_rows_by_condition_id)
    event_fetch_result = (
        event_fetcher(event_ids)
        if event_fetcher is not None
        else fetch_gamma_event_rows(
            event_ids,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    )
    final_events_by_id: dict[str, dict[str, Any]] = {}
    for event_id in event_ids:
        merged_event = dict(seeded_events_by_id.get(event_id, {}))
        fresh_event = event_fetch_result.rows_by_event_id.get(event_id)
        if isinstance(fresh_event, dict):
            merged_event.update(fresh_event)
        if merged_event:
            final_events_by_id[event_id] = merged_event

    markets_by_token = _build_markets_by_token(merged_rows_by_condition_id)
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "generated_from": "shadow_metadata_preflight",
        "lake_root": str(lake_root),
        "since_date": since_date.isoformat(),
        "active_market_count": len(active_condition_ids),
        "markets_by_token": markets_by_token,
        "events_by_id": final_events_by_id,
    }
    _write_json_atomic(cache_path, payload)

    covered_market_count = len(merged_rows_by_condition_id)
    coverage_pct = 100.0 if not active_condition_ids else (covered_market_count / len(active_condition_ids)) * 100.0
    refresh_status = "success"
    if missing_condition_ids:
        refresh_status = "degraded"
    elif reused_condition_ids or market_fetch_result.failed_condition_ids or event_fetch_result.failed_event_ids:
        refresh_status = "fallback"

    summary = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "lake_root": str(lake_root),
        "cache_path": str(cache_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "since_date": since_date.isoformat(),
        "refresh_status": refresh_status,
        "active_market_count": len(active_condition_ids),
        "covered_market_count": covered_market_count,
        "coverage_pct": round(coverage_pct, 4),
        "coverage_threshold_pct": float(coverage_threshold_pct),
        "coverage_alarm": coverage_pct < float(coverage_threshold_pct),
        "fetched_market_count": len(market_fetch_result.rows_by_condition_id),
        "reused_market_count": len(reused_condition_ids),
        "missing_market_count": len(missing_condition_ids),
        "failed_market_fetch_count": len(market_fetch_result.failed_condition_ids),
        "failed_event_fetch_count": len(event_fetch_result.failed_event_ids),
        "missing_market_ids_sample": missing_condition_ids[:10],
        "failed_market_ids_sample": market_fetch_result.failed_condition_ids[:10],
        "event_count": len(final_events_by_id),
        "markets_by_token_entry_count": len(markets_by_token),
    }
    if summary_path is not None:
        _write_json_atomic(summary_path, summary)
    return summary


def _emit_summary_logs(summary: dict[str, Any]) -> None:
    print(
        "SHADOW_METADATA_PREFLIGHT "
        f"refresh_status={summary['refresh_status']} "
        f"active_markets={summary['active_market_count']} "
        f"covered_markets={summary['covered_market_count']} "
        f"coverage_pct={float(summary['coverage_pct']):.2f} "
        f"fetched_markets={summary['fetched_market_count']} "
        f"reused_markets={summary['reused_market_count']} "
        f"event_count={summary['event_count']} "
        f"cache_path={summary['cache_path']}"
    )
    if summary["refresh_status"] != "success":
        print(
            "WARNING SHADOW_METADATA_REFRESH_FALLBACK "
            f"refresh_status={summary['refresh_status']} "
            f"failed_market_fetch_count={summary['failed_market_fetch_count']} "
            f"failed_event_fetch_count={summary['failed_event_fetch_count']} "
            f"reused_market_count={summary['reused_market_count']} "
            f"missing_market_count={summary['missing_market_count']}"
        )
    if summary["coverage_alarm"]:
        missing_sample = ",".join(summary["missing_market_ids_sample"]) or "none"
        print(
            "WARNING SHADOW_METADATA_COVERAGE_LOW "
            f"coverage_pct={float(summary['coverage_pct']):.2f} "
            f"threshold_pct={float(summary['coverage_threshold_pct']):.2f} "
            f"active_markets={summary['active_market_count']} "
            f"covered_markets={summary['covered_market_count']} "
            f"missing_market_count={summary['missing_market_count']} "
            f"missing_market_ids_sample={missing_sample}"
        )
    else:
        print(
            "SHADOW_METADATA_COVERAGE_OK "
            f"coverage_pct={float(summary['coverage_pct']):.2f} "
            f"threshold_pct={float(summary['coverage_threshold_pct']):.2f} "
            f"missing_market_count={summary['missing_market_count']}"
        )


def main() -> int:
    args = parse_args()
    since_date = date.fromisoformat(str(args.since_date))
    try:
        summary = refresh_shadow_metadata_cache(
            args.lake_root,
            cache_path=args.cache_path,
            summary_path=args.summary_path,
            since_date=since_date,
            coverage_threshold_pct=float(args.coverage_threshold_pct),
            gamma_batch_size=int(args.gamma_batch_size),
            timeout_seconds=float(args.timeout_seconds),
            max_retries=int(args.max_retries),
            retry_backoff_seconds=float(args.retry_backoff_seconds),
        )
    except Exception as exc:
        cache_present = args.cache_path.is_file()
        summary = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "lake_root": str(args.lake_root),
            "cache_path": str(args.cache_path),
            "summary_path": str(args.summary_path),
            "since_date": since_date.isoformat(),
            "refresh_status": "cache_only" if cache_present else "unavailable",
            "active_market_count": 0,
            "covered_market_count": 0,
            "coverage_pct": 0.0,
            "coverage_threshold_pct": float(args.coverage_threshold_pct),
            "coverage_alarm": not cache_present,
            "fetched_market_count": 0,
            "reused_market_count": 0,
            "missing_market_count": 0,
            "failed_market_fetch_count": 0,
            "failed_event_fetch_count": 0,
            "missing_market_ids_sample": [],
            "failed_market_ids_sample": [],
            "event_count": 0,
            "markets_by_token_entry_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.summary_path is not None:
            _write_json_atomic(args.summary_path, summary)
        print(
            "WARNING SHADOW_METADATA_REFRESH_FAILED "
            f"cache_present={str(cache_present).lower()} "
            f"cache_path={args.cache_path} "
            f"error={type(exc).__name__}:{exc}"
        )
        if not cache_present:
            print("WARNING SHADOW_METADATA_CACHE_UNAVAILABLE cache_missing=true")
        return 0

    _emit_summary_logs(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())