#!/usr/bin/env python3
"""
backfill_data.py — Download historical Polymarket L2 deltas and trade ticks
for Walk-Forward Optimization (WFO).

Outputs JSONL files matching the schema produced by ``src/backtest/data_recorder.py``,
directly consumable by ``DataLoader.from_directory()`` and the ``BacktestEngine``.

Architecture
────────────
    Pluggable DataSourceAdapter ABC
        ├── PolymarketTradesAdapter   (trades — working, hits data-api.polymarket.com)
        ├── PMXTArchiveAdapter        (L2 deltas — skeleton, archive.pmxt.dev)
        └── TelonexAdapter            (L2 deltas — skeleton, telonex.io/datasets)

    AsyncOrchestrator
        └── Semaphore-bounded tasks per (date, market_id)
            └── Fetch → normalize → stream-write JSONL

    HealthAuditor
        └── Post-download integrity scan (gaps, monotonicity, coverage)

Output layout
─────────────
    data/vps_march2026/ticks/YYYY-MM-DD/<market_id>.jsonl

Each line:
    {"local_ts":1700000001.234,"source":"trade","asset_id":"0xabc...","payload":{...}}

Quick-start
───────────
    # Install deps
    pip install -r scripts/requirements-backfill.txt

    # Default 90-day backfill for all tracked markets
    python scripts/backfill_data.py

    # Custom date range
    python scripts/backfill_data.py --start-date 2025-12-06 --end-date 2026-03-04

    # Single day, low concurrency, force overwrite
    python scripts/backfill_data.py --lookback-days 1 --concurrency 2 --force

    # With optional Parquet conversion
    python scripts/backfill_data.py --parquet

    # Use a different source adapter (when available)
    python scripts/backfill_data.py --source pmxt
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import structlog

# ═══════════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════════

log = structlog.get_logger("backfill_data")

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

POLYMARKET_DATA_API = "https://data-api.polymarket.com"
DEFAULT_OUTPUT_DIR = Path("data/vps_march2026/ticks")
DEFAULT_MARKET_MAP = Path("data/market_map.json")
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_CONCURRENCY = 10
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  Market map
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MarketEntry:
    """A tracked market from market_map.json."""
    market_id: str
    yes_id: str
    no_id: str


def load_market_map(path: Path) -> list[MarketEntry]:
    """Load and parse market_map.json into MarketEntry objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = []
    for item in raw:
        entries.append(MarketEntry(
            market_id=item["market_id"],
            yes_id=item["yes_id"],
            no_id=item["no_id"],
        ))
    return entries


def build_token_to_market(entries: list[MarketEntry]) -> dict[str, str]:
    """Map token_id (yes_id / no_id) → parent market_id."""
    mapping: dict[str, str] = {}
    for e in entries:
        mapping[e.yes_id] = e.market_id
        mapping[e.no_id] = e.market_id
    return mapping


# ═══════════════════════════════════════════════════════════════════════════
#  Timestamp normalization (mirrors DataLoader convention)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_ts(raw: Any) -> float:
    """Normalize a timestamp to UTC seconds.

    Handles microseconds (>1e15), milliseconds (>1e12), and seconds.
    """
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if ts > 1e15:
        ts /= 1_000_000
    elif ts > 1e12:
        ts /= 1_000
    return ts


# ═══════════════════════════════════════════════════════════════════════════
#  DataSourceAdapter ABC
# ═══════════════════════════════════════════════════════════════════════════

class DataSourceAdapter(abc.ABC):
    """Abstract base for historical data source adapters.

    Each adapter normalizes source-specific formats into the canonical
    JSONL record schema used by ``DataRecorder`` / ``DataLoader``::

        {"local_ts": float, "source": "l2"|"trade", "asset_id": str, "payload": dict}
    """

    @abc.abstractmethod
    async def fetch_trades(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """Yield canonical trade records for a single market on a single day."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def fetch_l2(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """Yield canonical L2 delta records for a single market on a single day."""
        ...  # pragma: no cover

    async def fetch_all(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """Yield all records (L2 + trades) for a market on a day, sorted by timestamp."""
        records: list[dict] = []
        async for rec in self.fetch_l2(market, day, client):
            records.append(rec)
        async for rec in self.fetch_trades(market, day, client):
            records.append(rec)
        records.sort(key=lambda r: r.get("local_ts", 0.0))
        for rec in records:
            yield rec


# ═══════════════════════════════════════════════════════════════════════════
#  PolymarketTradesAdapter — working adapter
# ═══════════════════════════════════════════════════════════════════════════

class PolymarketTradesAdapter(DataSourceAdapter):
    """Fetches historical trades from the Polymarket Data API.

    Endpoint: GET https://data-api.polymarket.com/trades
    Pagination: cursor-based (``next_cursor`` in response).
    Filters by token ID (yes_id + no_id per market) and date range.

    L2 data is NOT available from this source — ``fetch_l2`` yields nothing.
    Use PMXTArchiveAdapter or TelonexAdapter for L2 deltas.
    """

    def __init__(self, rate_limit_per_sec: float = 5.0) -> None:
        self._rate_delay = 1.0 / max(rate_limit_per_sec, 0.1)

    async def fetch_trades(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """Fetch all trades for a market on a given day.

        Queries both YES and NO token IDs, merging results.
        """
        start_ts = int(datetime.combine(day, datetime.min.time(),
                                        tzinfo=timezone.utc).timestamp())
        end_ts = start_ts + 86400

        for token_id in (market.yes_id, market.no_id):
            async for record in self._paginate_trades(
                client, token_id, market.market_id, start_ts, end_ts
            ):
                yield record

    async def _paginate_trades(
        self,
        client: httpx.AsyncClient,
        token_id: str,
        market_id: str,
        start_ts: int,
        end_ts: int,
    ) -> AsyncIterator[dict]:
        """Paginate through /trades endpoint for a single token."""
        next_cursor: str | None = None
        page = 0
        max_pages = 500  # safety limit

        while page < max_pages:
            params: dict[str, Any] = {
                "asset_id": token_id,
                "after": start_ts,
                "before": end_ts,
                "limit": 500,
            }
            if next_cursor:
                params["next_cursor"] = next_cursor

            data = await self._request_with_retry(client, "/trades", params)
            if data is None:
                break

            trades = data if isinstance(data, list) else data.get("data", [])
            if not trades:
                break

            for trade in trades:
                ts = normalize_ts(
                    trade.get("match_time")
                    or trade.get("timestamp")
                    or trade.get("created_at")
                    or start_ts
                )
                # Build canonical record matching DataRecorder output
                yield {
                    "local_ts": ts,
                    "source": "trade",
                    "asset_id": market_id,
                    "payload": {
                        "price": float(trade.get("price", 0)),
                        "size": float(trade.get("size", trade.get("amount", 0))),
                        "side": trade.get("side", "unknown").lower(),
                        "timestamp": str(int(ts * 1000)),
                        "event_type": "last_trade_price",
                        "market": market_id,
                        "asset_id": token_id,
                        "trade_id": trade.get("id", ""),
                    },
                }

            # Advance cursor
            if isinstance(data, dict):
                next_cursor = data.get("next_cursor")
            else:
                # If response is a bare list, stop (no pagination metadata)
                break

            if not next_cursor:
                break

            page += 1
            await asyncio.sleep(self._rate_delay)

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        path: str,
        params: dict[str, Any],
    ) -> dict | list | None:
        """GET request with exponential backoff on transient errors."""
        url = f"{POLYMARKET_DATA_API}{path}"
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    log.warning("rate_limited", url=path, wait_s=wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                log.warning(
                    "http_error",
                    url=path,
                    status=exc.response.status_code,
                    attempt=attempt + 1,
                )
                if exc.response.status_code >= 500:
                    await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
                    continue
                return None
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                log.warning("network_error", url=path, error=str(exc), wait_s=wait)
                await asyncio.sleep(wait)
        log.error("request_failed", url=path, params=params)
        return None

    async def fetch_l2(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """L2 data not available from the Polymarket Data API."""
        return
        yield  # make this an async generator


# ═══════════════════════════════════════════════════════════════════════════
#  PMXTArchiveAdapter — skeleton
# ═══════════════════════════════════════════════════════════════════════════

class PMXTArchiveAdapter(DataSourceAdapter):
    """Skeleton adapter for the PMXT Open-Source Archive (archive.pmxt.dev).

    Expected workflow (to be implemented when API docs are available):
      1. GET archive.pmxt.dev/v1/l2/{asset_id}/{YYYY-MM-DD}.parquet
      2. Read Parquet into memory via pyarrow
      3. Convert each row to canonical JSONL record with source="l2"
         and payload matching the price_changes format:
         {
           "event_type": "price_change",
           "market": "<market_id>",
           "timestamp": "<ms>",
           "price_changes": [
             {"asset_id": "...", "price": "0.50", "size": "100",
              "side": "BUY", "best_bid": "0.49", "best_ask": "0.51"}
           ]
         }
    """

    def __init__(self, base_url: str = "https://archive.pmxt.dev") -> None:
        self._base_url = base_url

    async def fetch_trades(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """PMXT primarily provides L2 data; trades may not be available."""
        return
        yield

    async def fetch_l2(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """TODO: Implement when PMXT API docs are available.

        Expected steps:
          1. Download Parquet file for (market_id, date)
          2. Read with pyarrow: pf = pq.read_table(buffer)
          3. Iterate rows, building canonical records:
             {
               "local_ts": row["timestamp_s"],
               "source": "l2",
               "asset_id": market.market_id,
               "payload": { ... price_changes ... }
             }
        """
        raise NotImplementedError(
            "PMXTArchiveAdapter.fetch_l2() is a skeleton — "
            "implement once archive.pmxt.dev API docs are available. "
            "See class docstring for expected payload format."
        )
        yield  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════════
#  TelonexAdapter — skeleton
# ═══════════════════════════════════════════════════════════════════════════

class TelonexAdapter(DataSourceAdapter):
    """Skeleton adapter for Telonex Public Datasets (telonex.io/datasets).

    Expected workflow (to be implemented when API docs are available):
      1. GET telonex.io/datasets/polymarket/l2/{YYYY-MM-DD}/{asset_id}.parquet
      2. Read Parquet via pyarrow
      3. Convert rows to canonical JSONL records (same format as PMXTArchiveAdapter)

    Telonex datasets are typically distributed as daily Parquet files
    with columns: timestamp, asset_id, side, price, size, event_type.
    """

    def __init__(self, base_url: str = "https://telonex.io") -> None:
        self._base_url = base_url

    async def fetch_trades(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """TODO: Implement when Telonex API docs are available."""
        raise NotImplementedError(
            "TelonexAdapter.fetch_trades() is a skeleton — "
            "implement once telonex.io API docs are available."
        )
        yield  # pragma: no cover

    async def fetch_l2(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """TODO: Implement when Telonex API docs are available.

        Expected steps:
          1. Download Parquet: GET /datasets/polymarket/l2/{date}/{asset_id}.parquet
          2. Read with pyarrow
          3. Convert to canonical records with source="l2"
        """
        raise NotImplementedError(
            "TelonexAdapter.fetch_l2() is a skeleton — "
            "implement once telonex.io API docs are available."
        )
        yield  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════════
#  Adapter registry
# ═══════════════════════════════════════════════════════════════════════════

ADAPTERS: dict[str, type[DataSourceAdapter]] = {
    "polymarket": PolymarketTradesAdapter,
    "pmxt": PMXTArchiveAdapter,
    "telonex": TelonexAdapter,
}


# ═══════════════════════════════════════════════════════════════════════════
#  JSONL writer
# ═══════════════════════════════════════════════════════════════════════════

class JSONLWriter:
    """Stream-writes canonical records to JSONL files on disk.

    Creates the directory tree lazily (once per unique date directory).
    Idempotent: skips files that already exist unless force=True.
    """

    def __init__(self, output_dir: Path, *, force: bool = False) -> None:
        self._output_dir = output_dir
        self._force = force
        self._known_dirs: set[Path] = set()
        self._files_written: int = 0
        self._records_written: int = 0
        self._files_skipped: int = 0

    def should_skip(self, day: date, market_id: str) -> bool:
        """Return True if the file exists and is non-empty (and not --force)."""
        if self._force:
            return False
        path = self._file_path(day, market_id)
        return path.exists() and path.stat().st_size > 0

    def write_records(self, day: date, market_id: str, records: list[dict]) -> int:
        """Write a batch of records to JSONL. Returns number of records written."""
        if not records:
            return 0
        path = self._file_path(day, market_id)
        self._ensure_dir(path.parent)

        count = 0
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                line = json.dumps(rec, separators=(",", ":"), default=str)
                fh.write(line)
                fh.write("\n")
                count += 1

        self._files_written += 1
        self._records_written += count
        return count

    def _file_path(self, day: date, market_id: str) -> Path:
        safe_id = market_id.replace("/", "_").replace("\\", "_")
        return self._output_dir / day.isoformat() / f"{safe_id}.jsonl"

    def _ensure_dir(self, dirpath: Path) -> None:
        if dirpath not in self._known_dirs:
            dirpath.mkdir(parents=True, exist_ok=True)
            self._known_dirs.add(dirpath)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "files_written": self._files_written,
            "records_written": self._records_written,
            "files_skipped": self._files_skipped,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Health Auditor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DayMarketAudit:
    """Audit result for a single (date, market_id) file."""
    date: str
    market_id: str
    total_events: int = 0
    trade_events: int = 0
    l2_events: int = 0
    snapshot_events: int = 0
    ts_monotonic: bool = True
    max_gap_s: float = 0.0
    sequence_gaps: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Aggregate audit report across all downloaded data."""
    audits: list[DayMarketAudit] = field(default_factory=list)
    total_events: int = 0
    total_files: int = 0
    dates_with_warnings: int = 0
    markets_with_coverage_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_events": self.total_events,
                "total_files": self.total_files,
                "dates_with_warnings": self.dates_with_warnings,
                "markets_with_coverage_gaps": self.markets_with_coverage_gaps,
            },
            "per_file": [
                {
                    "date": a.date,
                    "market_id": a.market_id,
                    "total_events": a.total_events,
                    "trade_events": a.trade_events,
                    "l2_events": a.l2_events,
                    "snapshot_events": a.snapshot_events,
                    "ts_monotonic": a.ts_monotonic,
                    "max_gap_s": round(a.max_gap_s, 2),
                    "sequence_gaps": a.sequence_gaps,
                    "warnings": a.warnings,
                }
                for a in self.audits
            ],
        }


class HealthAuditor:
    """Post-download integrity scanner.

    Checks:
      - Coverage: flags dates where a market has 0 events
      - Timestamp monotonicity: verifies local_ts non-decreasing per file
      - L2 snapshot presence: flags days with no snapshot records
      - Sequence gap detection: counts discontinuities in sequence IDs
      - Time gap detection: flags intervals >gap_threshold_s with no events
    """

    def __init__(self, gap_threshold_s: float = 60.0) -> None:
        self._gap_threshold = gap_threshold_s

    def audit_file(self, path: Path) -> DayMarketAudit:
        """Audit a single JSONL file."""
        day_str = path.parent.name
        market_id = path.stem

        audit = DayMarketAudit(date=day_str, market_id=market_id)

        prev_ts = 0.0
        max_gap = 0.0
        seqs: list[int] = []

        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                audit.total_events += 1
                source = rec.get("source", "")
                payload = rec.get("payload", {})

                if source == "trade":
                    audit.trade_events += 1
                elif source in ("l2", "l2_delta"):
                    audit.l2_events += 1
                elif source in ("l2_snapshot", "book_snapshot", "snapshot"):
                    audit.snapshot_events += 1

                # Check payload event_type for snapshot detection
                if isinstance(payload, dict):
                    ptype = payload.get("event_type", "")
                    if ptype in ("book", "snapshot", "book_snapshot"):
                        audit.snapshot_events += 1

                ts = rec.get("local_ts", 0.0)
                try:
                    ts = float(ts)
                except (TypeError, ValueError):
                    continue

                if ts < prev_ts:
                    audit.ts_monotonic = False

                if prev_ts > 0:
                    gap = ts - prev_ts
                    if gap > max_gap:
                        max_gap = gap

                prev_ts = ts

                # Sequence tracking
                if isinstance(payload, dict):
                    seq = payload.get("seq") or payload.get("sequence") or payload.get("seq_num")
                    if seq is not None:
                        try:
                            seqs.append(int(seq))
                        except (TypeError, ValueError):
                            pass

        audit.max_gap_s = max_gap
        audit.sequence_gaps = self._count_gaps(seqs)

        # Generate warnings
        if audit.total_events == 0:
            audit.warnings.append("EMPTY: no events found")
        if not audit.ts_monotonic:
            audit.warnings.append("NON_MONOTONIC: timestamps not in order")
        if audit.max_gap_s > self._gap_threshold:
            audit.warnings.append(
                f"TIME_GAP: max gap {audit.max_gap_s:.1f}s > {self._gap_threshold}s threshold"
            )
        if audit.snapshot_events == 0 and audit.l2_events > 0:
            audit.warnings.append(
                "NO_SNAPSHOT: L2 deltas present but no baseline snapshot found"
            )

        return audit

    @staticmethod
    def _count_gaps(seqs: list[int]) -> int:
        """Count sequence ID discontinuities (mirrors ParquetConverter logic)."""
        if len(set(seqs)) < 2:
            return 0
        sorted_seqs = sorted(set(seqs))
        total = 0
        for i in range(1, len(sorted_seqs)):
            if sorted_seqs[i] - sorted_seqs[i - 1] > 1:
                total += 1
        return total

    def audit_directory(
        self,
        output_dir: Path,
        dates: list[date],
        market_ids: list[str],
    ) -> AuditReport:
        """Run health audit across all downloaded files."""
        report = AuditReport()

        # Track per-market coverage
        market_date_counts: dict[str, int] = {mid: 0 for mid in market_ids}
        warned_dates: set[str] = set()

        for day in dates:
            day_dir = output_dir / day.isoformat()
            if not day_dir.exists():
                # Flag all markets as missing for this date
                for mid in market_ids:
                    audit = DayMarketAudit(
                        date=day.isoformat(), market_id=mid,
                        warnings=["MISSING: date directory does not exist"],
                    )
                    report.audits.append(audit)
                    warned_dates.add(day.isoformat())
                continue

            for mid in market_ids:
                safe_id = mid.replace("/", "_").replace("\\", "_")
                fpath = day_dir / f"{safe_id}.jsonl"
                if not fpath.exists():
                    audit = DayMarketAudit(
                        date=day.isoformat(), market_id=mid,
                        warnings=["MISSING: file not found"],
                    )
                    report.audits.append(audit)
                    warned_dates.add(day.isoformat())
                    continue

                audit = self.audit_file(fpath)
                report.audits.append(audit)
                report.total_events += audit.total_events
                report.total_files += 1

                if audit.total_events > 0:
                    market_date_counts[mid] += 1

                if audit.warnings:
                    warned_dates.add(day.isoformat())

        report.dates_with_warnings = len(warned_dates)

        # Markets with <50% date coverage
        expected_days = len(dates)
        for mid, count in market_date_counts.items():
            if expected_days > 0 and count < expected_days * 0.5:
                report.markets_with_coverage_gaps.append(mid)

        return report


# ═══════════════════════════════════════════════════════════════════════════
#  Async download orchestrator
# ═══════════════════════════════════════════════════════════════════════════

async def download_market_day(
    adapter: DataSourceAdapter,
    market: MarketEntry,
    day: date,
    writer: JSONLWriter,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
) -> tuple[str, str, int]:
    """Download and write data for a single (market, day) pair.

    Returns (market_id, date_str, records_written).
    """
    async with semaphore:
        if writer.should_skip(day, market.market_id):
            log.debug("skipping_existing", market=market.market_id[:16], date=day.isoformat())
            return (market.market_id, day.isoformat(), 0)

        records: list[dict] = []
        try:
            async for rec in adapter.fetch_all(market, day, client):
                records.append(rec)
        except NotImplementedError as exc:
            log.warning("adapter_not_implemented", error=str(exc))
            return (market.market_id, day.isoformat(), 0)

        count = writer.write_records(day, market.market_id, records)
        if count > 0:
            log.info(
                "downloaded",
                market=market.market_id[:16],
                date=day.isoformat(),
                records=count,
            )
        return (market.market_id, day.isoformat(), count)


async def run_backfill(
    adapter: DataSourceAdapter,
    markets: list[MarketEntry],
    dates: list[date],
    output_dir: Path,
    concurrency: int,
    force: bool,
) -> JSONLWriter:
    """Orchestrate parallel downloads across all (market, date) pairs."""
    writer = JSONLWriter(output_dir, force=force)
    semaphore = asyncio.Semaphore(concurrency)

    total_pairs = len(markets) * len(dates)
    log.info(
        "backfill_start",
        markets=len(markets),
        dates=len(dates),
        total_pairs=total_pairs,
        concurrency=concurrency,
    )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        tasks = []
        for market in markets:
            for day in dates:
                task = download_market_day(
                    adapter, market, day, writer, semaphore, client,
                )
                tasks.append(task)

        # Process in batches to avoid overwhelming memory
        batch_size = concurrency * 10
        completed = 0
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    log.error("task_error", error=str(result))
                else:
                    mid, day_str, count = result
                    completed += 1
            if completed % 50 == 0 or completed == total_pairs:
                log.info("progress", completed=completed, total=total_pairs)

    return writer


# ═══════════════════════════════════════════════════════════════════════════
#  Optional Parquet conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_parquet(output_dir: Path) -> None:
    """Convert downloaded JSONL files to Parquet using the existing ParquetConverter."""
    try:
        from src.data.prep_data import ParquetConverter
    except ImportError:
        log.error(
            "parquet_import_error",
            msg="Could not import ParquetConverter. "
                "Install pyarrow and pandas: pip install pyarrow pandas",
        )
        return

    parquet_out = output_dir.parent / "processed"
    log.info("parquet_conversion_start", input=str(output_dir), output=str(parquet_out))

    converter = ParquetConverter()
    report = converter.convert([output_dir], parquet_out)
    log.info(
        "parquet_conversion_done",
        valid_rows=report.valid_rows,
        health_score=round(report.health_score, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_date_range(
    start_date: str | None,
    end_date: str | None,
    lookback_days: int,
) -> list[date]:
    """Build the list of dates to backfill."""
    today = date.today()

    if start_date:
        start = date.fromisoformat(start_date)
    else:
        start = today - timedelta(days=lookback_days)

    if end_date:
        end = date.fromisoformat(end_date)
    else:
        end = today - timedelta(days=1)

    if start > end:
        raise ValueError(f"Start date {start} is after end date {end}")

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical Polymarket L2 and trade data for WFO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 90-day backfill
  python scripts/backfill_data.py

  # Custom date range
  python scripts/backfill_data.py --start-date 2025-12-06 --end-date 2026-03-04

  # Force re-download, low concurrency
  python scripts/backfill_data.py --lookback-days 7 --concurrency 2 --force

  # With Parquet conversion
  python scripts/backfill_data.py --parquet
        """,
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date (YYYY-MM-DD). Defaults to today - lookback_days.",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help=f"Number of days to look back (default: {DEFAULT_LOOKBACK_DAYS}). "
             "Ignored if --start-date is provided.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max parallel download tasks (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--source", type=str, default="polymarket",
        choices=list(ADAPTERS.keys()),
        help="Data source adapter (default: polymarket).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for JSONL files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--market-map", type=str, default=str(DEFAULT_MARKET_MAP),
        help=f"Path to market_map.json (default: {DEFAULT_MARKET_MAP}).",
    )
    parser.add_argument(
        "--parquet", action="store_true",
        help="Convert downloaded JSONL to Parquet after download.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files instead of skipping.",
    )
    parser.add_argument(
        "--skip-audit", action="store_true",
        help="Skip the health audit step.",
    )
    parser.add_argument(
        "--gap-threshold", type=float, default=60.0,
        help="Time gap threshold (seconds) for health audit warnings (default: 60).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    market_map_path = Path(args.market_map)

    # ── Load markets ──────────────────────────────────────────────────
    if not market_map_path.exists():
        log.error("market_map_not_found", path=str(market_map_path))
        return 1

    markets = load_market_map(market_map_path)
    log.info("markets_loaded", count=len(markets))

    # ── Build date range ──────────────────────────────────────────────
    try:
        dates = build_date_range(args.start_date, args.end_date, args.lookback_days)
    except ValueError as exc:
        log.error("invalid_date_range", error=str(exc))
        return 1

    log.info(
        "date_range",
        start=dates[0].isoformat(),
        end=dates[-1].isoformat(),
        days=len(dates),
    )

    # ── Create adapter ────────────────────────────────────────────────
    adapter_cls = ADAPTERS[args.source]
    adapter = adapter_cls()

    # ── Run download ──────────────────────────────────────────────────
    writer = asyncio.run(
        run_backfill(adapter, markets, dates, output_dir, args.concurrency, args.force)
    )

    log.info("backfill_complete", **writer.stats)

    # ── Health audit ──────────────────────────────────────────────────
    if not args.skip_audit:
        log.info("health_audit_start")
        auditor = HealthAuditor(gap_threshold_s=args.gap_threshold)
        market_ids = [m.market_id for m in markets]
        report = auditor.audit_directory(output_dir, dates, market_ids)

        # Write audit JSON
        audit_path = output_dir / "backfill_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        log.info(
            "health_audit_complete",
            total_events=report.total_events,
            total_files=report.total_files,
            dates_with_warnings=report.dates_with_warnings,
            coverage_gaps=len(report.markets_with_coverage_gaps),
            audit_file=str(audit_path),
        )

        # Print summary
        if report.markets_with_coverage_gaps:
            log.warning(
                "coverage_gaps",
                markets=report.markets_with_coverage_gaps[:5],
                msg="Markets with <50% date coverage (first 5 shown)",
            )

        warned = [a for a in report.audits if a.warnings]
        if warned:
            log.warning(
                "audit_warnings",
                count=len(warned),
                sample=[
                    f"{a.date}/{a.market_id[:16]}: {', '.join(a.warnings)}"
                    for a in warned[:5]
                ],
            )

    # ── Optional Parquet conversion ───────────────────────────────────
    if args.parquet:
        convert_to_parquet(output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
