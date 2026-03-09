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
COALESCE_WINDOW_S = 0.050  # 50 ms bucket for delta coalescing


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
#  Delta coalescing (50 ms buckets)
# ═══════════════════════════════════════════════════════════════════════════

def coalesce_deltas(
    records: list[dict],
    window_s: float = COALESCE_WINDOW_S,
) -> list[dict]:
    """Merge L2 delta records within the same *window_s* time bucket.

    Snapshot records (``event_type`` in {"book", "snapshot", "book_snapshot"})
    are always emitted individually.  Delta records whose ``local_ts`` values
    fall within the same ``window_s`` bucket are merged into a single record
    with a combined ``changes`` array.  This preserves 50 ms fidelity while
    reducing JSONL line count (and disk I/O) by ~5-10×.

    The merged record inherits the *earliest* timestamp in the bucket
    so that downstream replay remains monotonic.
    """
    if not records:
        return []

    out: list[dict] = []
    bucket: list[dict] | None = None
    bucket_edge: float = 0.0

    for rec in records:
        payload = rec.get("payload", {})
        etype = payload.get("event_type", "")

        # Snapshots pass through un-merged
        if etype in ("book", "snapshot", "book_snapshot"):
            if bucket:
                out.append(_flush_bucket(bucket))
                bucket = None
            out.append(rec)
            continue

        ts = rec.get("local_ts", 0.0)

        if bucket is None or ts >= bucket_edge:
            # Start a new bucket
            if bucket:
                out.append(_flush_bucket(bucket))
            bucket = [rec]
            bucket_edge = ts + window_s
        else:
            bucket.append(rec)

    if bucket:
        out.append(_flush_bucket(bucket))

    return out


def _flush_bucket(bucket: list[dict]) -> dict:
    """Merge a bucket of delta records into a single record."""
    if len(bucket) == 1:
        return bucket[0]

    base = bucket[0]
    merged_changes: list[dict] = []
    for rec in bucket:
        changes = rec.get("payload", {}).get("changes", [])
        merged_changes.extend(changes)

    merged = {
        "local_ts": base["local_ts"],
        "source": base["source"],
        "asset_id": base["asset_id"],
        "payload": {
            **base["payload"],
            "changes": merged_changes,
        },
    }
    return merged


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
    """Adapter for the PMXT Open-Source Archive (archive.pmxt.dev).

    PMXT distributes hourly Parquet snapshots of *all* Polymarket L2 deltas
    via Cloudflare R2:

        https://r2.pmxt.dev/polymarket_orderbook_{YYYY-MM-DD}T{HH}.parquet

    Each file is 500-700 MB (26 M rows) covering every market for that hour.

    Parquet schema:
        timestamp_received   timestamp[ms, tz=UTC]   (when PMXT ingested)
        timestamp_created_at timestamp[ms, tz=UTC]   (exchange timestamp)
        market_id            string                  (hex condition ID, e.g. 0x06b0…)
        update_type          string                  (always "price_change")
        data                 string                  (JSON blob, see below)

    data JSON keys:
        update_type, market_id, token_id, side, best_bid, best_ask,
        timestamp, change_price, change_size, change_side

    Batch strategy:
        Because each hourly file contains ALL Polymarket markets, we use
        a batch-mode approach: download the file once per (day, hour) and
        extract records for all tracked markets in a single pass.  This
        avoids re-downloading ~500 MB per market.

        The orchestrator detects ``supports_batch = True`` and uses
        ``fetch_l2_day_batch()`` instead of per-market ``fetch_l2()``.
    """

    _R2_BASE = "https://r2.pmxt.dev"
    _FILE_TEMPLATE = "polymarket_orderbook_{date}T{hour:02d}.parquet"
    supports_batch: bool = True

    def __init__(
        self,
        r2_base: str | None = None,
        rate_limit_per_sec: float = 2.0,
    ) -> None:
        self._r2_base = (r2_base or self._R2_BASE).rstrip("/")
        self._rate_delay = 1.0 / max(rate_limit_per_sec, 0.1)

    # ── ABC implementation (per-market, used as fallback) ─────────────

    async def fetch_trades(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """PMXT provides L2 data only; trades are not available."""
        return
        yield

    async def fetch_l2(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict]:
        """Per-market fallback — prefer ``fetch_l2_day_batch()``."""
        result = await self._fetch_hour_batch(
            [market], day, 0, client,
        )
        for rec in result.get(market.market_id, []):
            yield rec
        return
        yield

    # ── Batch mode: one file read → all markets ──────────────────────

    async def fetch_l2_day_batch(
        self,
        markets: list[MarketEntry],
        day: date,
        client: httpx.AsyncClient,
    ) -> dict[str, list[dict]]:
        """Download all 24 hourly files for *day* and return L2 records
        grouped by market_id.

        Returns ``{market_id: [records]}`` for every market in *markets*
        that has data on this day.
        """
        all_records: dict[str, list[dict]] = {m.market_id: [] for m in markets}

        for hour in range(24):
            hour_result = await self._fetch_hour_batch(
                markets, day, hour, client,
            )
            for mid, recs in hour_result.items():
                all_records[mid].extend(recs)

        # Sort each market's records by timestamp, then coalesce 50 ms buckets
        for mid in all_records:
            all_records[mid].sort(key=lambda r: r.get("local_ts", 0.0))
            all_records[mid] = coalesce_deltas(all_records[mid])

        return all_records

    async def _fetch_hour_batch(
        self,
        markets: list[MarketEntry],
        day: date,
        hour: int,
        client: httpx.AsyncClient,
    ) -> dict[str, list[dict]]:
        """Read one hourly Parquet file and extract records for all *markets*."""
        filename = self._FILE_TEMPLATE.format(
            date=day.isoformat(), hour=hour,
        )
        url = f"{self._r2_base}/{filename}"

        try:
            import pyarrow.parquet as pq  # noqa: F811
            import fsspec  # noqa: F811
        except ImportError:
            log.error(
                "pmxt_missing_deps",
                msg="Install pyarrow and fsspec: pip install pyarrow fsspec",
            )
            return {}

        # Quick existence check before heavy I/O
        exists = await self._head_check(client, url)
        if not exists:
            log.debug("pmxt_hour_missing", file=filename)
            return {}

        # Build lookup structures
        market_id_set = {m.market_id for m in markets}
        token_map: dict[str, MarketEntry] = {}
        for m in markets:
            token_map[m.yes_id] = m
            token_map[m.no_id] = m

        # Run blocking Parquet I/O in a thread
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self._read_parquet_batch,
                url,
                market_id_set,
            )
        except Exception as exc:
            log.warning(
                "pmxt_read_error",
                file=filename,
                error=str(exc),
            )
            return {}

        log.debug(
            "pmxt_hour_done",
            file=filename,
            markets_found=len([m for m, r in result.items() if r]),
            total_records=sum(len(r) for r in result.values()),
        )

        await asyncio.sleep(self._rate_delay)
        return result

    # ── Synchronous Parquet I/O (runs in thread pool) ─────────────────

    def _read_parquet_batch(
        self,
        url: str,
        market_id_set: set[str],
    ) -> dict[str, list[dict]]:
        """Open remote Parquet via fsspec, filter by market_ids, convert.

        Reads only ``timestamp_received``, ``market_id``, and ``data``
        columns.  Skips row groups whose market_id statistics don't
        overlap with any tracked market.

        Emits a synthetic ``l2_snapshot`` as the first record per market
        from the BBO (best_bid / best_ask) of the earliest delta, then
        all subsequent deltas.  Downstream, ``DataLoader._parse_record``
        classifies ``event_type="book"`` as ``l2_snapshot``.
        """
        import pyarrow.parquet as pq
        import fsspec

        fs = fsspec.filesystem("https")
        result: dict[str, list[dict]] = {mid: [] for mid in market_id_set}
        snapshot_emitted: set[str] = set()  # track per-market snapshot emission

        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)

            for rg_idx in range(pf.metadata.num_row_groups):
                if self._can_skip_row_group_batch(pf, rg_idx, market_id_set):
                    continue

                table = pf.read_row_group(
                    rg_idx,
                    columns=["timestamp_received", "market_id", "data"],
                )

                mid_col = table.column("market_id").to_pylist()
                ts_col = table.column("timestamp_received").to_pylist()
                data_col = table.column("data").to_pylist()

                for i, mid in enumerate(mid_col):
                    if mid not in market_id_set:
                        continue

                    ts_recv = ts_col[i]
                    if hasattr(ts_recv, "timestamp"):
                        ts = ts_recv.timestamp()
                    else:
                        ts = normalize_ts(ts_recv)
                    if ts <= 0:
                        continue

                    # Try to emit a synthetic snapshot before the first delta
                    if mid not in snapshot_emitted:
                        snap = self._data_to_snapshot(data_col[i], ts, mid)
                        if snap is not None:
                            result[mid].append(snap)
                            snapshot_emitted.add(mid)

                    rec = self._data_to_record(data_col[i], ts, mid)
                    if rec is not None:
                        result[mid].append(rec)

        return result

    @staticmethod
    def _can_skip_row_group_batch(
        pf,
        rg_idx: int,
        market_id_set: set[str],
    ) -> bool:
        """Skip a row group if none of our market IDs fall within its
        min/max range for the market_id column."""
        try:
            rg = pf.metadata.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                if col.path_in_schema != "market_id":
                    continue
                if not col.statistics or not col.statistics.has_min_max:
                    return False
                smin = col.statistics.min
                smax = col.statistics.max
                # Check if ANY of our market_ids could be in [smin, smax]
                for mid in market_id_set:
                    if smin <= mid <= smax:
                        return False
                return True
        except Exception:
            pass
        return False

    def _data_to_record(
        self,
        data_raw: str,
        ts: float,
        market_id: str,
    ) -> dict | None:
        """Parse the JSON ``data`` column and build a canonical L2 record.

        PMXT data JSON:
            {
                "update_type": "price_change",
                "token_id": "<decimal>",
                "change_price": float,
                "change_size": int,
                "change_side": "BUY"|"SELL",
                "timestamp": float,
                ...
            }

        Mapped to canonical:
            source="l2", payload.event_type="price_change",
            payload.changes=[{side, price, size}]
        """
        try:
            d = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
        except (json.JSONDecodeError, TypeError):
            return None

        token_id = str(d.get("token_id", ""))
        change_side = str(d.get("change_side", "BUY")).upper()
        change_price = d.get("change_price", 0)
        change_size = d.get("change_size", 0)
        data_ts = d.get("timestamp", ts)

        # Use the more precise timestamp from the data payload
        precise_ts = normalize_ts(data_ts) if data_ts else ts

        return {
            "local_ts": precise_ts,
            "source": "l2",
            "asset_id": market_id,
            "payload": {
                "event_type": "price_change",
                "asset_id": token_id,
                "market": market_id,
                "timestamp": precise_ts,
                "changes": [
                    {
                        "side": change_side,
                        "price": str(change_price),
                        "size": str(change_size),
                    },
                ],
            },
        }

    def _data_to_snapshot(
        self,
        data_raw: str,
        ts: float,
        market_id: str,
    ) -> dict | None:
        """Synthesize an L2 snapshot record from the BBO fields in a PMXT row.

        Uses ``best_bid`` and ``best_ask`` from the data JSON to create a
        minimal snapshot that ``DataLoader._parse_record()`` will classify
        as ``l2_snapshot`` (via ``event_type="book"``).

        This anchors the delta stream for downstream ``L2OrderBook.load_snapshot()``.
        """
        try:
            d = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
        except (json.JSONDecodeError, TypeError):
            return None

        best_bid = d.get("best_bid")
        best_ask = d.get("best_ask")
        if best_bid is None or best_ask is None:
            return None

        try:
            bid_price = float(best_bid)
            ask_price = float(best_ask)
        except (TypeError, ValueError):
            return None

        if bid_price <= 0 and ask_price <= 0:
            return None

        token_id = str(d.get("token_id", ""))

        return {
            "local_ts": ts,
            "source": "l2",
            "asset_id": market_id,
            "payload": {
                "event_type": "book",
                "asset_id": token_id,
                "market": market_id,
                "timestamp": ts,
                "bids": [{"price": str(bid_price), "size": "1"}],
                "asks": [{"price": str(ask_price), "size": "1"}],
            },
        }

    # ── HTTP helper ───────────────────────────────────────────────────

    async def _head_check(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> bool:
        """Return True if the URL exists (HEAD 200)."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.head(url, follow_redirects=True)
                if resp.status_code == 200:
                    return True
                if resp.status_code in (404, 403):
                    return False
                if resp.status_code == 429:
                    await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
                    continue
                if resp.status_code >= 500:
                    await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
                    continue
                return False
            except (httpx.ConnectError, httpx.ReadTimeout):
                await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
        return False


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

    def open_stream(self, day: date, market_id: str):
        """Return an open file handle for streaming writes.

        Caller is responsible for closing the handle (use as context manager).
        """
        path = self._file_path(day, market_id)
        self._ensure_dir(path.parent)
        return open(path, "w", encoding="utf-8")

    def finalize_stream(self, count: int) -> None:
        """Update internal stats after a streaming write session."""
        if count > 0:
            self._files_written += 1
            self._records_written += count

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

    Uses streaming writes so that L2-heavy adapters don't need to
    buffer millions of records in memory before flushing to disk.

    Returns (market_id, date_str, records_written).
    """
    async with semaphore:
        if writer.should_skip(day, market.market_id):
            log.debug("skipping_existing", market=market.market_id[:16], date=day.isoformat())
            return (market.market_id, day.isoformat(), 0)

        count = 0
        try:
            with writer.open_stream(day, market.market_id) as fh:
                async for rec in adapter.fetch_all(market, day, client):
                    fh.write(json.dumps(rec, separators=(",", ":"), default=str))
                    fh.write("\n")
                    count += 1
        except NotImplementedError as exc:
            log.warning("adapter_not_implemented", error=str(exc))
            return (market.market_id, day.isoformat(), 0)

        writer.finalize_stream(count)
        if count > 0:
            log.info(
                "downloaded",
                market=market.market_id[:16],
                date=day.isoformat(),
                records=count,
            )
        return (market.market_id, day.isoformat(), count)


async def download_pmxt_day(
    adapter: PMXTArchiveAdapter,
    markets: list[MarketEntry],
    day: date,
    writer: JSONLWriter,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
) -> list[tuple[str, str, int]]:
    """Batch download: one pass over 24 hourly files → all markets.

    PMXT hourly files contain every market, so reading once and splitting
    locally is far more efficient than reading the same file N times.

    Returns a list of (market_id, date_str, records_written) tuples.
    """
    async with semaphore:
        # Determine which markets still need data
        needed = [m for m in markets if not writer.should_skip(day, m.market_id)]
        if not needed:
            return [
                (m.market_id, day.isoformat(), 0) for m in markets
            ]

        log.info(
            "pmxt_batch_start",
            date=day.isoformat(),
            markets=len(needed),
        )

        try:
            records_by_market = await adapter.fetch_l2_day_batch(
                needed, day, client,
            )
        except Exception as exc:
            log.error("pmxt_batch_error", date=day.isoformat(), error=str(exc))
            return [(m.market_id, day.isoformat(), 0) for m in needed]

        results: list[tuple[str, str, int]] = []
        for market in needed:
            recs = records_by_market.get(market.market_id, [])
            count = writer.write_records(day, market.market_id, recs)
            writer.finalize_stream(count)
            if count > 0:
                log.info(
                    "downloaded",
                    market=market.market_id[:16],
                    date=day.isoformat(),
                    records=count,
                )
            results.append((market.market_id, day.isoformat(), count))

        return results


async def run_backfill(
    adapter: DataSourceAdapter,
    markets: list[MarketEntry],
    dates: list[date],
    output_dir: Path,
    concurrency: int,
    force: bool,
) -> JSONLWriter:
    """Orchestrate parallel downloads across all (market, date) pairs.

    If the adapter declares ``supports_batch = True`` (e.g. PMXTArchiveAdapter),
    the orchestrator groups work by (day) instead of (market, day) to avoid
    re-downloading the same hourly files for every market.
    """
    writer = JSONLWriter(output_dir, force=force)
    semaphore = asyncio.Semaphore(concurrency)

    use_batch = getattr(adapter, "supports_batch", False)

    if use_batch:
        total_tasks = len(dates)
        log.info(
            "backfill_start",
            mode="batch",
            markets=len(markets),
            dates=len(dates),
            total_day_tasks=total_tasks,
            concurrency=concurrency,
        )
    else:
        total_tasks = len(markets) * len(dates)
        log.info(
            "backfill_start",
            markets=len(markets),
            dates=len(dates),
            total_pairs=total_tasks,
            concurrency=concurrency,
        )

    # Increase timeout for PMXT — remote Parquet reads are slow
    timeout = 600 if use_batch else REQUEST_TIMEOUT_S

    async with httpx.AsyncClient(timeout=timeout) as client:
        if use_batch:
            # ── Batch mode: one task per day, all markets per file ────
            completed = 0
            for day in dates:
                results = await download_pmxt_day(
                    adapter, markets, day, writer, semaphore, client,
                )
                for mid, day_str, count in results:
                    pass  # stats already updated via writer
                completed += 1
                log.info("progress", completed=completed, total=total_tasks)
        else:
            # ── Standard mode: one task per (market, day) ────────────
            tasks = []
            for market in markets:
                for day in dates:
                    task = download_market_day(
                        adapter, market, day, writer, semaphore, client,
                    )
                    tasks.append(task)

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
                if completed % 50 == 0 or completed == total_tasks:
                    log.info("progress", completed=completed, total=total_tasks)

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
