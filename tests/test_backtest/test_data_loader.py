"""
Tests for the DataLoader — chronological event replay from JSONL files.

Covers:
- Correct parsing of recorded JSONL lines
- Chronological merge across multiple files (heapq merge)
- Malformed / incomplete line handling
- Asset ID filtering
- Event type mapping (l2 → l2_delta, trade → trade, etc.)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.backtest.data_loader import DataLoader, MarketEvent


def _write_jsonl(lines: list[dict], path: Path) -> None:
    """Helper: write a list of dicts as JSONL to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")


class TestDataLoaderBasic:
    """Single-file loading and parsing."""

    def test_loads_events_in_order(self, tmp_path: Path):
        records = [
            {
                "local_ts": 1000.0,
                "source": "l2",
                "asset_id": "ABC",
                "payload": {"event_type": "price_change", "price": "0.50"},
            },
            {
                "local_ts": 1001.0,
                "source": "trade",
                "asset_id": "ABC",
                "payload": {"price": "0.51", "size": "10"},
            },
            {
                "local_ts": 1002.0,
                "source": "l2",
                "asset_id": "ABC",
                "payload": {"event_type": "snapshot", "bids": [], "asks": []},
            },
        ]
        fpath = tmp_path / "asset1.jsonl"
        _write_jsonl(records, fpath)

        loader = DataLoader.from_files(fpath)
        events = list(loader)

        assert len(events) == 3
        assert events[0].timestamp == 1000.0
        assert events[0].event_type == "l2_delta"
        assert events[1].timestamp == 1001.0
        assert events[1].event_type == "trade"
        assert events[2].timestamp == 1002.0
        assert events[2].event_type == "l2_snapshot"

    def test_preserves_payload(self, tmp_path: Path):
        payload = {"price": "0.55", "size": "42", "extra_field": True}
        records = [
            {
                "local_ts": 1000.0,
                "source": "trade",
                "asset_id": "XYZ",
                "payload": payload,
            }
        ]
        fpath = tmp_path / "asset.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert events[0].data == payload

    def test_asset_id_populated(self, tmp_path: Path):
        records = [
            {
                "local_ts": 1000.0,
                "source": "trade",
                "asset_id": "MY_ASSET",
                "payload": {"price": "0.5"},
            }
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert events[0].asset_id == "MY_ASSET"


class TestChronologicalMerge:
    """Multi-file merge via heapq."""

    def test_two_files_interleaved(self, tmp_path: Path):
        file_a = tmp_path / "a.jsonl"
        file_b = tmp_path / "b.jsonl"

        _write_jsonl(
            [
                {"local_ts": 1.0, "source": "l2", "asset_id": "A", "payload": {"x": 1}},
                {"local_ts": 3.0, "source": "l2", "asset_id": "A", "payload": {"x": 3}},
                {"local_ts": 5.0, "source": "l2", "asset_id": "A", "payload": {"x": 5}},
            ],
            file_a,
        )
        _write_jsonl(
            [
                {"local_ts": 2.0, "source": "trade", "asset_id": "B", "payload": {"y": 2}},
                {"local_ts": 4.0, "source": "trade", "asset_id": "B", "payload": {"y": 4}},
            ],
            file_b,
        )

        loader = DataLoader.from_files(file_a, file_b)
        events = list(loader)

        timestamps = [e.timestamp for e in events]
        assert timestamps == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_three_files_sorted(self, tmp_path: Path):
        """Three files, all events should come out sorted."""
        files = []
        for i, ts_list in enumerate([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]):
            fpath = tmp_path / f"f{i}.jsonl"
            _write_jsonl(
                [
                    {"local_ts": ts, "source": "trade", "asset_id": f"A{i}", "payload": {"v": ts}}
                    for ts in ts_list
                ],
                fpath,
            )
            files.append(fpath)

        loader = DataLoader.from_files(*files)
        events = list(loader)

        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)
        assert len(events) == 6


class TestMalformedLines:
    """Graceful handling of bad data."""

    def test_missing_local_ts(self, tmp_path: Path):
        records = [
            {"source": "l2", "asset_id": "A", "payload": {"x": 1}},  # no local_ts
            {"local_ts": 2.0, "source": "l2", "asset_id": "A", "payload": {"x": 2}},
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert len(events) == 1
        assert events[0].timestamp == 2.0

    def test_invalid_json(self, tmp_path: Path):
        fpath = tmp_path / "bad.jsonl"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "w") as fh:
            fh.write('{"local_ts": 1.0, "source": "trade", "asset_id": "A", "payload": {"p": 1}}\n')
            fh.write("NOT VALID JSON\n")
            fh.write('{"local_ts": 3.0, "source": "trade", "asset_id": "A", "payload": {"p": 3}}\n')

        events = list(DataLoader.from_files(fpath))
        assert len(events) == 2
        assert events[0].timestamp == 1.0
        assert events[1].timestamp == 3.0

    def test_missing_source(self, tmp_path: Path):
        records = [
            {"local_ts": 1.0, "asset_id": "A", "payload": {"x": 1}},  # no source
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert len(events) == 0

    def test_missing_payload(self, tmp_path: Path):
        records = [
            {"local_ts": 1.0, "source": "trade", "asset_id": "A"},  # no payload
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert len(events) == 0

    def test_empty_file(self, tmp_path: Path):
        fpath = tmp_path / "empty.jsonl"
        fpath.write_text("")
        events = list(DataLoader.from_files(fpath))
        assert events == []

    def test_blank_lines_ignored(self, tmp_path: Path):
        fpath = tmp_path / "blanks.jsonl"
        with open(fpath, "w") as fh:
            fh.write("\n\n")
            fh.write('{"local_ts": 1.0, "source": "l2", "asset_id": "A", "payload": {"x": 1}}\n')
            fh.write("\n")

        events = list(DataLoader.from_files(fpath))
        assert len(events) == 1


class TestAssetFiltering:
    """Asset ID filter parameter."""

    def test_filter_single_asset(self, tmp_path: Path):
        records = [
            {"local_ts": 1.0, "source": "trade", "asset_id": "KEEP", "payload": {"v": 1}},
            {"local_ts": 2.0, "source": "trade", "asset_id": "DROP", "payload": {"v": 2}},
            {"local_ts": 3.0, "source": "trade", "asset_id": "KEEP", "payload": {"v": 3}},
        ]
        fpath = tmp_path / "mixed.jsonl"
        _write_jsonl(records, fpath)

        loader = DataLoader.from_files(fpath, asset_ids={"KEEP"})
        events = list(loader)
        assert len(events) == 2
        assert all(e.asset_id == "KEEP" for e in events)

    def test_filter_multiple_assets(self, tmp_path: Path):
        records = [
            {"local_ts": 1.0, "source": "trade", "asset_id": "A", "payload": {"v": 1}},
            {"local_ts": 2.0, "source": "trade", "asset_id": "B", "payload": {"v": 2}},
            {"local_ts": 3.0, "source": "trade", "asset_id": "C", "payload": {"v": 3}},
        ]
        fpath = tmp_path / "multi.jsonl"
        _write_jsonl(records, fpath)

        loader = DataLoader.from_files(fpath, asset_ids={"A", "C"})
        events = list(loader)
        assert len(events) == 2
        assert {e.asset_id for e in events} == {"A", "C"}

    def test_no_filter_returns_all(self, tmp_path: Path):
        records = [
            {"local_ts": 1.0, "source": "trade", "asset_id": "X", "payload": {"v": 1}},
            {"local_ts": 2.0, "source": "trade", "asset_id": "Y", "payload": {"v": 2}},
        ]
        fpath = tmp_path / "all.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert len(events) == 2


class TestEventTypeMapping:
    """Source tag → event type mapping."""

    def test_l2_becomes_l2_delta(self, tmp_path: Path):
        records = [
            {"local_ts": 1.0, "source": "l2", "asset_id": "A",
             "payload": {"event_type": "price_change", "price": "0.5"}},
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert events[0].event_type == "l2_delta"

    def test_l2_snapshot_detection(self, tmp_path: Path):
        """If source=l2 but payload says snapshot, event_type = l2_snapshot."""
        records = [
            {"local_ts": 1.0, "source": "l2", "asset_id": "A",
             "payload": {"event_type": "snapshot", "bids": [], "asks": []}},
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert events[0].event_type == "l2_snapshot"

    def test_trade_maps_to_trade(self, tmp_path: Path):
        records = [
            {"local_ts": 1.0, "source": "trade", "asset_id": "A",
             "payload": {"price": "0.5", "size": "10"}},
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert events[0].event_type == "trade"

    def test_price_change_trade_source_becomes_l2_delta(self, tmp_path: Path):
        records = [
            {
                "local_ts": 1.0,
                "source": "trade",
                "asset_id": "0xmarket",
                "payload": {
                    "event_type": "price_change",
                    "price_changes": [
                        {"asset_id": "YES", "price": "0.45", "size": "100", "side": "BUY"},
                        {"asset_id": "NO", "price": "0.55", "size": "100", "side": "SELL"},
                    ],
                },
            },
        ]
        fpath = tmp_path / "price_change.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath, asset_ids={"YES"}))

        assert len(events) == 1
        assert events[0].event_type == "l2_delta"
        assert events[0].asset_id == "YES"
        assert events[0].data["price_changes"][0]["asset_id"] == "YES"


class TestFactoryMethods:
    """Factory constructors: from_directory, from_files."""

    def test_from_directory_finds_jsonl(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        _write_jsonl(
            [{"local_ts": 1.0, "source": "trade", "asset_id": "A", "payload": {"p": 1}}],
            data_dir / "asset1.jsonl",
        )
        _write_jsonl(
            [{"local_ts": 2.0, "source": "trade", "asset_id": "B", "payload": {"p": 2}}],
            data_dir / "asset2.jsonl",
        )

        loader = DataLoader.from_directory(data_dir)
        events = list(loader)
        assert len(events) == 2

    def test_from_directory_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            DataLoader.from_directory("/nonexistent/path/1234567890")

    def test_from_files_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DataLoader.from_files(tmp_path / "nope.jsonl")

    def test_stats_after_iteration(self, tmp_path: Path):
        fpath = tmp_path / "s.jsonl"
        _write_jsonl(
            [
                {"local_ts": 1.0, "source": "trade", "asset_id": "A", "payload": {"p": 1}},
                {"local_ts": 2.0, "source": "trade", "asset_id": "A", "payload": {"p": 2}},
            ],
            fpath,
        )
        loader = DataLoader.from_files(fpath)
        _ = list(loader)
        stats = loader.stats
        assert stats["total_events"] == 2
        assert stats["skipped_events"] == 0
        assert stats["file_count"] == 1


class TestServerTimestamp:
    """Server-side timestamp extraction."""

    def test_server_time_from_payload(self, tmp_path: Path):
        records = [
            {"local_ts": 1000.0, "source": "trade", "asset_id": "A",
             "payload": {"price": "0.5", "timestamp": 999.5}},
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert events[0].server_time == 999.5

    def test_server_time_millis_conversion(self, tmp_path: Path):
        """Timestamp in milliseconds should be auto-converted."""
        # 1_700_000_000_000 ms = 1_700_000_000 s (in millis range >1e12)
        records = [
            {"local_ts": 1000.0, "source": "trade", "asset_id": "A",
             "payload": {"price": "0.5", "timestamp": 1700000000000}},
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert abs(events[0].server_time - 1700000000.0) < 1.0

    def test_missing_server_time_defaults_zero(self, tmp_path: Path):
        records = [
            {"local_ts": 1000.0, "source": "trade", "asset_id": "A",
             "payload": {"price": "0.5"}},
        ]
        fpath = tmp_path / "a.jsonl"
        _write_jsonl(records, fpath)

        events = list(DataLoader.from_files(fpath))
        assert events[0].server_time == 0.0
