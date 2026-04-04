from __future__ import annotations

import json
from pathlib import Path

from scripts import build_l2_parquet_lake


def _snapshot_record(
    *,
    market_id: str,
    asset_id: str,
    timestamp_ms: int,
    bid_price: str,
    ask_price: str,
) -> dict[str, object]:
    return {
        "local_ts": timestamp_ms / 1000.0,
        "source": "snapshot",
        "asset_id": asset_id,
        "payload": {
            "market": market_id,
            "asset_id": asset_id,
            "timestamp": str(timestamp_ms),
            "hash": f"hash-{asset_id}",
            "bids": [{"price": bid_price, "size": "100"}],
            "asks": [{"price": ask_price, "size": "100"}],
            "tick_size": "0.01",
            "event_type": "book",
            "last_trade_price": "0.50",
        },
    }


def _delta_record(*, market_id: str, yes_asset_id: str, no_asset_id: str, timestamp_ms: int) -> dict[str, object]:
    return {
        "local_ts": timestamp_ms / 1000.0,
        "source": "delta",
        "asset_id": market_id,
        "payload": {
            "market": market_id,
            "price_changes": [
                {
                    "asset_id": yes_asset_id,
                    "price": "0.41",
                    "size": "120",
                    "side": "BUY",
                    "hash": "hash-yes",
                    "best_bid": "0.41",
                    "best_ask": "0.60",
                },
                {
                    "asset_id": no_asset_id,
                    "price": "0.31",
                    "size": "110",
                    "side": "BUY",
                    "hash": "hash-no",
                    "best_bid": "0.31",
                    "best_ask": "0.70",
                },
            ],
            "timestamp": str(timestamp_ms),
            "event_type": "price_change",
        },
    }


def _write_ndjson(path: Path, rows: list[dict[str, object]], *, malformed_lines: list[bytes] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(json.dumps(row).encode("utf-8") + b"\n")
        for line in malformed_lines or []:
            handle.write(line + b"\n")


def test_process_market_day_drops_malformed_ndjson_lines(tmp_path: Path) -> None:
    metadata = build_l2_parquet_lake.MarketMetadata(
        market_id="0xabc",
        event_id="evt-1",
        yes_asset_id="111",
        no_asset_id="222",
    )
    day = "2026-03-20"
    day_dir = tmp_path / "raw" / day

    _write_ndjson(
        day_dir / f"{metadata.yes_asset_id}.jsonl",
        [
            _snapshot_record(
                market_id=metadata.market_id,
                asset_id=metadata.yes_asset_id,
                timestamp_ms=1_710_000_000_000,
                bid_price="0.40",
                ask_price="0.60",
            )
        ],
        malformed_lines=[b'{"local_ts":1710000000.0,"source":"snapshot"'],
    )
    _write_ndjson(
        day_dir / f"{metadata.no_asset_id}.jsonl",
        [
            _snapshot_record(
                market_id=metadata.market_id,
                asset_id=metadata.no_asset_id,
                timestamp_ms=1_710_000_000_000,
                bid_price="0.30",
                ask_price="0.70",
            )
        ],
    )
    _write_ndjson(
        day_dir / f"{metadata.market_id}.jsonl",
        [
            _delta_record(
                market_id=metadata.market_id,
                yes_asset_id=metadata.yes_asset_id,
                no_asset_id=metadata.no_asset_id,
                timestamp_ms=1_710_000_001_000,
            )
        ],
        malformed_lines=[b'{"local_ts":1710000001.0,"source":"delta"'],
    )

    stats = build_l2_parquet_lake.RunStats()
    build_l2_parquet_lake.process_market_day(
        day=day,
        day_dir=day_dir,
        metadata=metadata,
        output_root=tmp_path / "out",
        batch_lines=100,
        flush_rows=10,
        compression_level=1,
        stats=stats,
    )

    assert stats.markets_completed == 1
    assert stats.output_rows == 4
    assert stats.rejected_rows == 1
    assert stats.raw_records_read["snapshot:yes"] == 2
    assert stats.raw_records_parsed["snapshot:yes"] == 1
    assert stats.raw_records_malformed["snapshot:yes"] == 1
    assert stats.raw_batches_salvaged["snapshot:yes"] == 1
    assert stats.raw_records_read["delta"] == 2
    assert stats.raw_records_parsed["delta"] == 1
    assert stats.raw_records_malformed["delta"] == 1
    assert stats.raw_batches_salvaged["delta"] == 1
    assert any((tmp_path / "out" / "l2_book").rglob("*.parquet"))