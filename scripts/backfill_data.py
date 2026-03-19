#!/usr/bin/env python3
"""Download PMXT historical order-book data into canonical replay Parquet.

The generated Parquet files are written under ``data/YYYY-MM-DD/general.parquet``
with the exact columns expected by ``src.backtest.data_loader.DataLoader``:
``local_ts``, ``msg_type``, ``asset_id``, ``payload``, ``exchange_ts``.

The script defaults to the March 12-18, 2026 top-25 Polymarket window requested
for pure market-making WFO. The legacy ``--source polymarket`` flag is kept as
an alias for PMXT so existing automation does not break.
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

log = structlog.get_logger("backfill_data")

DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_MARKET_MAP = Path("data/market_map_top25.json")
DEFAULT_START_DATE = date(2026, 3, 12)
DEFAULT_END_DATE = date(2026, 3, 18)
DEFAULT_SOURCE = "polymarket"
DEFAULT_HTTP_HEADERS = {
    "User-Agent": "polymarket-bot/pmxt-backfill",
    "Accept": "application/octet-stream,application/x-parquet,*/*",
}
REQUEST_TIMEOUT_S = 45.0
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
COALESCE_WINDOW_S = 0.050
OUTPUT_FILE_NAME = "general.parquet"
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("local_ts", pa.float64()),
        pa.field("msg_type", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("payload", pa.string()),
        pa.field("exchange_ts", pa.float64()),
    ]
)


@dataclass(frozen=True, slots=True)
class MarketEntry:
    market_id: str
    yes_id: str
    no_id: str


def load_market_map(path: Path) -> list[MarketEntry]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        MarketEntry(
            market_id=str(item["market_id"]),
            yes_id=str(item["yes_id"]),
            no_id=str(item["no_id"]),
        )
        for item in raw
    ]


def normalize_ts(raw: Any) -> float:
    try:
        if hasattr(raw, "timestamp"):
            return float(raw.timestamp())
        ts = float(raw)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if ts > 1e15:
        ts /= 1_000_000.0
    elif ts > 1e12:
        ts /= 1_000.0
    return ts


def build_date_range(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(days)]


def _flush_bucket(bucket: list[dict[str, Any]]) -> dict[str, Any]:
    if len(bucket) == 1:
        return bucket[0]
    base = bucket[0]
    merged_changes: list[dict[str, str]] = []
    for record in bucket:
        merged_changes.extend(record.get("payload", {}).get("changes", []))
    return {
        "local_ts": base["local_ts"],
        "source": base["source"],
        "asset_id": base["asset_id"],
        "payload": {
            **base["payload"],
            "changes": merged_changes,
        },
    }


def coalesce_deltas(records: list[dict[str, Any]], window_s: float = COALESCE_WINDOW_S) -> list[dict[str, Any]]:
    if not records:
        return []

    out: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] | None = None
    bucket_edge = 0.0

    for record in records:
        event_type = record.get("payload", {}).get("event_type", "")
        if event_type in {"book", "snapshot", "book_snapshot", "l2_snapshot"}:
            if bucket:
                out.append(_flush_bucket(bucket))
                bucket = None
            out.append(record)
            continue

        ts = float(record.get("local_ts", 0.0))
        if bucket is None or ts >= bucket_edge:
            if bucket:
                out.append(_flush_bucket(bucket))
            bucket = [record]
            bucket_edge = ts + window_s
        else:
            bucket.append(record)

    if bucket:
        out.append(_flush_bucket(bucket))
    return out


class DataSourceAdapter(abc.ABC):
    supports_batch = False

    @abc.abstractmethod
    async def fetch_trades(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict[str, Any]]:
        ...

    @abc.abstractmethod
    async def fetch_l2(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict[str, Any]]:
        ...


class PMXTArchiveAdapter(DataSourceAdapter):
    supports_batch = True
    _FILE_TEMPLATE = "polymarket_orderbook_{date}T{hour:02d}.parquet"
    _ARCHIVE_BASES = (
        "https://r2.pmxt.dev/Polymarket",
        "https://r2.pmxt.dev/data/Polymarket",
        "https://r2.pmxt.dev",
    )

    def __init__(self, r2_base: str | None = None, archive_bases: Iterable[str] | None = None) -> None:
        self._r2_base = (r2_base or "https://r2.pmxt.dev").rstrip("/")
        archive_candidates = [base.rstrip("/") for base in (archive_bases or self._ARCHIVE_BASES)]
        self._candidate_bases = list(dict.fromkeys([*archive_candidates, self._r2_base]))

    async def fetch_trades(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {"market": market.market_id, "day": day.isoformat(), "client": client}
        return

    async def fetch_l2(
        self,
        market: MarketEntry,
        day: date,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict[str, Any]]:
        day_records = await self.fetch_l2_day_batch([market], day, client)
        for record in day_records.get(market.market_id, []):
            yield record

    async def fetch_l2_day_batch(
        self,
        markets: list[MarketEntry],
        day: date,
        client: httpx.AsyncClient,
    ) -> dict[str, list[dict[str, Any]]]:
        merged: dict[str, list[dict[str, Any]]] = {market.market_id: [] for market in markets}
        for hour in range(24):
            hour_result = await self._fetch_hour_batch(markets, day, hour, client)
            for market_id, records in hour_result.items():
                merged.setdefault(market_id, []).extend(records)
        for market_id, records in merged.items():
            records.sort(key=lambda record: float(record.get("local_ts", 0.0)))
            merged[market_id] = coalesce_deltas(records)
        return merged

    async def _fetch_hour_batch(
        self,
        markets: list[MarketEntry],
        day: date,
        hour: int,
        client: httpx.AsyncClient,
    ) -> dict[str, list[dict[str, Any]]]:
        market_ids = {market.market_id for market in markets}
        filename = self._FILE_TEMPLATE.format(date=day.isoformat(), hour=hour)

        for base_url in self._candidate_bases:
            url = f"{base_url}/{filename}"
            if not await self._head_check(client, url):
                continue
            if not await self._looks_like_parquet(client, url):
                log.warning("pmxt_non_parquet_candidate", url=url)
                continue
            try:
                return self._read_parquet_batch(url, market_ids)
            except Exception as exc:
                log.warning("pmxt_hour_read_failed", url=url, error=str(exc))
        return {}

    async def _head_check(self, client: httpx.AsyncClient, url: str) -> bool:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.head(url)
            except httpx.HTTPError as exc:
                if attempt == MAX_RETRIES:
                    log.warning("pmxt_head_failed", url=url, error=str(exc))
                    return False
                await asyncio.sleep(RETRY_BACKOFF_BASE**attempt)
                continue

            if response.status_code == 200:
                return True
            if response.status_code == 404:
                return False
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt == MAX_RETRIES:
                    return False
                await asyncio.sleep(RETRY_BACKOFF_BASE**attempt)
                continue
            return False
        return False

    async def _looks_like_parquet(self, client: httpx.AsyncClient, url: str) -> bool:
        try:
            async with client.stream("GET", url, headers={"Range": "bytes=0-255"}) as response:
                if response.status_code not in {200, 206}:
                    return False
                prefix = await response.aread()
        except httpx.HTTPError:
            return False
        return prefix.startswith(b"PAR1")

    @staticmethod
    def _can_skip_row_group_batch(parquet_file: Any, row_group_index: int, market_ids: set[str]) -> bool:
        metadata = parquet_file.metadata.row_group(row_group_index)
        if metadata is None:
            return False
        for column_index in range(getattr(metadata, "num_columns", 0)):
            column = metadata.column(column_index)
            if getattr(column, "path_in_schema", "") != "market_id":
                continue
            stats = getattr(column, "statistics", None)
            if stats is None or getattr(stats, "has_min_max", False) is False:
                return False
            try:
                min_value = str(stats.min)
                max_value = str(stats.max)
            except Exception:
                return False
            if any(min_value <= market_id <= max_value for market_id in market_ids):
                return False
            return True
        return False

    def _read_parquet_batch(self, url: str, market_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
        import fsspec

        by_market: dict[str, list[dict[str, Any]]] = {market_id: [] for market_id in market_ids}
        fs = fsspec.filesystem("https")
        with fs.open(url, "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                if self._can_skip_row_group_batch(parquet_file, row_group_index, market_ids):
                    continue
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=["market_id", "timestamp_received", "data"],
                )
                market_column = table.column("market_id").to_pylist()
                ts_column = table.column("timestamp_received").to_pylist()
                data_column = table.column("data").to_pylist()

                emitted_snapshots: set[tuple[str, str]] = set()
                for market_id, raw_ts, raw_data in zip(market_column, ts_column, data_column, strict=False):
                    market_key = str(market_id)
                    if market_key not in market_ids:
                        continue
                    ts = normalize_ts(raw_ts)

                    snapshot = self._data_to_snapshot(raw_data, ts, market_key)
                    if snapshot is not None:
                        snapshot_key = (market_key, str(snapshot["payload"].get("asset_id", "")))
                        if snapshot_key not in emitted_snapshots:
                            by_market[market_key].append(snapshot)
                            emitted_snapshots.add(snapshot_key)

                    record = self._data_to_record(raw_data, ts, market_key)
                    if record is not None:
                        by_market[market_key].append(record)

        for market_id, records in by_market.items():
            records.sort(key=lambda record: float(record.get("local_ts", 0.0)))
        return by_market

    def _data_to_snapshot(self, raw_data: str | bytes, ts: float, market_id: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            return None

        token_id = str(payload.get("token_id", "")).strip()
        best_bid = payload.get("best_bid")
        best_ask = payload.get("best_ask")
        if not token_id or best_bid in (None, 0, "0") or best_ask in (None, 0, "0"):
            return None

        snapshot_size = self._snapshot_size(payload)
        return {
            "local_ts": ts,
            "source": "l2",
            "asset_id": market_id,
            "payload": {
                "event_type": "book",
                "market": market_id,
                "asset_id": token_id,
                "timestamp": normalize_ts(payload.get("timestamp", ts)),
                "bids": [{"price": self._fmt_price(best_bid), "size": snapshot_size}],
                "asks": [{"price": self._fmt_price(best_ask), "size": snapshot_size}],
            },
        }

    def _data_to_record(self, raw_data: str | bytes, ts: float, market_id: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            return None

        token_id = str(payload.get("token_id", "")).strip()
        change_price = payload.get("change_price")
        change_size = payload.get("change_size")
        change_side = str(payload.get("change_side", "")).upper()
        if not token_id or change_price in (None, "") or change_size in (None, "") or change_side not in {"BUY", "SELL"}:
            return None

        best_bid = payload.get("best_bid")
        best_ask = payload.get("best_ask")
        book_size = self._snapshot_size(payload)

        return {
            "local_ts": ts,
            "source": "l2",
            "asset_id": market_id,
            "payload": {
                "event_type": "price_change",
                "market": market_id,
                "asset_id": token_id,
                "timestamp": normalize_ts(payload.get("timestamp", ts)),
                "changes": [
                    {
                        "side": change_side,
                        "price": self._fmt_price(change_price),
                        "size": self._fmt_size(change_size),
                    }
                ],
                "bids": self._book_side(best_bid, book_size),
                "asks": self._book_side(best_ask, book_size),
            },
        }

    @staticmethod
    def _fmt_price(value: Any) -> str:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _fmt_size(value: Any) -> str:
        rendered = f"{max(float(value), 0.0):.6f}".rstrip("0").rstrip(".")
        return rendered or "0"

    def _snapshot_size(self, payload: dict[str, Any]) -> str:
        size = payload.get("change_size", 1.0)
        try:
            numeric_size = float(size)
        except (TypeError, ValueError):
            numeric_size = 1.0
        return self._fmt_size(numeric_size if numeric_size > 0 else 1.0)

    def _book_side(self, best_price: Any, size: str) -> list[dict[str, str]]:
        if best_price in (None, ""):
            return []
        return [{"price": self._fmt_price(best_price), "size": size}]


class PolymarketTradesAdapter(PMXTArchiveAdapter):
    """Legacy alias retained so old CLI invocations still work."""


ADAPTERS: dict[str, type[DataSourceAdapter]] = {
    "polymarket": PolymarketTradesAdapter,
    "pmxt": PMXTArchiveAdapter,
}


def _record_to_parquet_row(record: dict[str, Any]) -> dict[str, Any]:
    payload = record["payload"]
    event_type = payload.get("event_type", "")
    msg_type = "snapshot" if event_type in {"book", "snapshot", "book_snapshot", "l2_snapshot"} else "delta"
    asset_id = str(payload.get("asset_id") or record.get("asset_id") or "")
    return {
        "local_ts": float(record.get("local_ts", 0.0)),
        "msg_type": msg_type,
        "asset_id": asset_id,
        "payload": json.dumps(payload, separators=(",", ":"), sort_keys=True),
        "exchange_ts": float(payload.get("timestamp", record.get("local_ts", 0.0)) or 0.0),
    }


def _rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)


def _write_day_parquet(output_dir: Path, day: date, rows: list[dict[str, Any]], force: bool) -> Path | None:
    if not rows:
        return None
    day_dir = output_dir / day.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    file_path = day_dir / OUTPUT_FILE_NAME
    if file_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {file_path}")
    rows.sort(key=lambda row: (row["local_ts"], row["msg_type"], row["asset_id"]))
    pq.write_table(_rows_to_table(rows), file_path, compression="zstd")
    return file_path


async def _run_backfill_async(args: argparse.Namespace) -> dict[str, Any]:
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    days = build_date_range(start_date, end_date)
    output_dir = Path(args.output_dir)
    markets = load_market_map(Path(args.market_map))
    adapter_cls = ADAPTERS[args.source]
    adapter = adapter_cls()
    if args.source == "polymarket":
        log.warning("backfill_source_alias", requested="polymarket", actual="pmxt")

    written_files: list[str] = []
    total_rows = 0

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_S,
        follow_redirects=True,
        headers=DEFAULT_HTTP_HEADERS,
    ) as client:
        for day in days:
            log.info("pmxt_day_start", day=day.isoformat(), markets=len(markets))
            by_market = await adapter.fetch_l2_day_batch(markets, day, client)
            day_rows: list[dict[str, Any]] = []
            for market in markets:
                for record in by_market.get(market.market_id, []):
                    day_rows.append(_record_to_parquet_row(record))

            file_path = _write_day_parquet(output_dir, day, day_rows, force=args.force)
            if file_path is None:
                log.warning("pmxt_day_empty", day=day.isoformat())
                continue
            written_files.append(str(file_path))
            total_rows += len(day_rows)
            log.info("pmxt_day_written", day=day.isoformat(), rows=len(day_rows), file=str(file_path))

    return {
        "files": written_files,
        "total_rows": total_rows,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "market_count": len(markets),
    }


def run_backfill(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(_run_backfill_async(args))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill PMXT L2 data into canonical replay Parquet.")
    parser.add_argument("--source", choices=sorted(ADAPTERS), default=DEFAULT_SOURCE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--market-map", default=str(DEFAULT_MARKET_MAP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_backfill(args)
    log.info("pmxt_backfill_complete", **result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())