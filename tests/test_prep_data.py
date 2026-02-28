"""
Tests for the DataPrepPipeline (``src.data.prep_data``) and DataLoader
Parquet round-trip compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.backtest.data_loader import DataLoader, MarketEvent
from src.data.prep_data import HealthReport, ParquetConverter
from src.data.synthetic import SyntheticGenerator


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def raw_data_dir(tmp_path: Path) -> Path:
    """Generate a small synthetic dataset and return the raw_ticks dir."""
    gen = SyntheticGenerator(seed=42)
    return gen.generate(tmp_path / "raw", num_rows=2_000, duration_hours=1.0)


@pytest.fixture
def converter() -> ParquetConverter:
    return ParquetConverter()


# ── Helper ─────────────────────────────────────────────────────────────────

def _read_jsonl_records(raw_dir: Path) -> list[dict]:
    records = []
    for fp in sorted(raw_dir.rglob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  ParquetConverter tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParquetConverter:

    def test_happy_path_row_count(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Converted Parquet has the same valid-row count as source JSONL."""
        out_dir = tmp_path / "processed"
        report = converter.convert([raw_data_dir], out_dir)

        assert report.valid_rows == 2_000
        assert report.malformed_rows == 0
        assert report.dropped_rows == 0
        assert len(report.output_files) >= 1

    def test_parquet_schema(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Output Parquet files have the expected schema columns."""
        out_dir = tmp_path / "processed"
        report = converter.convert([raw_data_dir], out_dir)

        expected_cols = {
            "local_ts", "exchange_ts", "msg_type", "asset_id",
            "price", "size", "sequence_id", "side", "payload",
        }

        for fp in report.output_files:
            schema = pq.read_schema(fp)
            assert set(schema.names) == expected_cols

    def test_zstd_compression(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Parquet files use Zstd compression."""
        out_dir = tmp_path / "processed"
        report = converter.convert([raw_data_dir], out_dir)

        for fp in report.output_files:
            meta = pq.read_metadata(fp)
            # Check first row group's first column compression
            col_meta = meta.row_group(0).column(0)
            assert col_meta.compression == "ZSTD"

    def test_global_sort_order(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Rows in each Parquet file are sorted by local_ts."""
        out_dir = tmp_path / "processed"
        report = converter.convert([raw_data_dir], out_dir)

        for fp in report.output_files:
            table = pq.read_table(fp)
            ts_list = table.column("local_ts").to_pylist()
            for i in range(1, len(ts_list)):
                assert ts_list[i] >= ts_list[i - 1], (
                    f"Not sorted: row {i}: {ts_list[i]} < {ts_list[i-1]}"
                )

    def test_partitioned_directory_layout(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Output is partitioned as YYYY-MM-DD/<category>.parquet."""
        out_dir = tmp_path / "processed"
        converter.convert([raw_data_dir], out_dir)

        parquet_files = list(out_dir.rglob("*.parquet"))
        assert len(parquet_files) >= 1

        for fp in parquet_files:
            # Parent should be a date directory
            date_dir = fp.parent.name
            # Should look like YYYY-MM-DD
            parts = date_dir.split("-")
            assert len(parts) == 3
            assert len(parts[0]) == 4  # year

            # File should be <category>.parquet — default is "general"
            assert fp.stem == "general"

    def test_custom_category_map(self, tmp_path: Path) -> None:
        """Category map correctly partitions output."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(
            tmp_path / "raw", num_rows=500, duration_hours=0.5
        )

        # Get the asset IDs from the generated data
        records = _read_jsonl_records(raw_dir)
        asset_ids = list({r["asset_id"] for r in records})

        cat_map = {asset_ids[0]: "crypto"}
        converter = ParquetConverter(category_map=cat_map)

        out_dir = tmp_path / "processed"
        converter.convert([raw_dir], out_dir)

        parquet_files = list(out_dir.rglob("*.parquet"))
        stems = {fp.stem for fp in parquet_files}

        # First asset → "crypto", second (if exists) → "general"
        assert "crypto" in stems
        if len(asset_ids) > 1:
            assert "general" in stems

    def test_msg_type_values(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """msg_type column only contains valid values."""
        out_dir = tmp_path / "processed"
        converter.convert([raw_data_dir], out_dir)

        valid_types = {"delta", "snapshot", "trade"}
        for fp in (out_dir).rglob("*.parquet"):
            table = pq.read_table(fp)
            msg_types = set(table.column("msg_type").to_pylist())
            assert msg_types <= valid_types, f"unexpected msg_types: {msg_types}"


# ═══════════════════════════════════════════════════════════════════════════
#  Malformed / edge-case data tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDefensiveParsing:

    def test_malformed_json_lines(self, tmp_path: Path) -> None:
        """Corrupt JSON lines are counted and skipped."""
        jsonl_path = tmp_path / "bad.jsonl"
        good_record = {
            "local_ts": 1700000000.0,
            "source": "trade",
            "asset_id": "0xtest",
            "payload": {
                "price": "0.50",
                "size": "10",
                "side": "buy",
                "timestamp": 1699999999.9,
            },
        }
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(good_record) + "\n")
            fh.write("this is not json\n")
            fh.write("{broken json\n")
            fh.write(json.dumps(good_record) + "\n")

        converter = ParquetConverter()
        out_dir = tmp_path / "out"
        report = converter.convert([jsonl_path], out_dir)

        assert report.total_rows == 4
        assert report.malformed_rows == 2
        assert report.valid_rows == 2
        assert report.dropped_rows == 0

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        """Records missing local_ts or source are dropped."""
        records = [
            # Missing local_ts
            {"source": "trade", "asset_id": "0x1", "payload": {"price": "0.5"}},
            # Missing source
            {"local_ts": 1700000000.0, "asset_id": "0x1", "payload": {"price": "0.5"}},
            # Missing asset_id
            {"local_ts": 1700000000.0, "source": "trade", "payload": {"price": "0.5"}},
            # Missing payload
            {"local_ts": 1700000000.0, "source": "trade", "asset_id": "0x1"},
            # Bad source value
            {"local_ts": 1700000000.0, "source": "unknown_type", "asset_id": "0x1", "payload": {}},
            # Valid record
            {
                "local_ts": 1700000000.0,
                "source": "trade",
                "asset_id": "0x1",
                "payload": {"price": "0.5", "size": "10", "side": "buy", "timestamp": 1699999999.9},
            },
        ]
        jsonl_path = tmp_path / "partial.jsonl"
        _write_jsonl(jsonl_path, records)

        converter = ParquetConverter()
        out_dir = tmp_path / "out"
        report = converter.convert([jsonl_path], out_dir)

        assert report.valid_rows == 1
        assert report.dropped_rows == 5

    def test_sequence_gap_detection(self, tmp_path: Path) -> None:
        """Gaps in sequence IDs are counted correctly."""
        # Asset with seqs: 1, 2, 5, 6 → 1 gap (between 2 and 5)
        records = []
        base_ts = 1700000000.0
        for i, seq in enumerate([1, 2, 5, 6]):
            records.append({
                "local_ts": base_ts + i,
                "source": "l2",
                "asset_id": "0xA",
                "payload": {
                    "event_type": "price_change",
                    "seq": seq,
                    "timestamp": base_ts + i - 0.01,
                    "changes": [{"side": "BUY", "price": "0.50", "size": "10"}],
                },
            })

        jsonl_path = tmp_path / "gaps.jsonl"
        _write_jsonl(jsonl_path, records)

        converter = ParquetConverter()
        out_dir = tmp_path / "out"
        report = converter.convert([jsonl_path], out_dir)

        assert report.sequence_gaps == 1

    def test_no_sequence_gaps(self, tmp_path: Path) -> None:
        """Contiguous sequence IDs → 0 gaps."""
        records = []
        base_ts = 1700000000.0
        for i in range(5):
            records.append({
                "local_ts": base_ts + i,
                "source": "l2",
                "asset_id": "0xA",
                "payload": {
                    "event_type": "price_change",
                    "seq": i + 1,
                    "timestamp": base_ts + i - 0.01,
                    "changes": [{"side": "BUY", "price": "0.50", "size": "10"}],
                },
            })

        jsonl_path = tmp_path / "no_gaps.jsonl"
        _write_jsonl(jsonl_path, records)

        converter = ParquetConverter()
        out_dir = tmp_path / "out"
        report = converter.convert([jsonl_path], out_dir)

        assert report.sequence_gaps == 0

    def test_empty_input(self, tmp_path: Path) -> None:
        """No files → empty report without crash."""
        converter = ParquetConverter()
        report = converter.convert([], tmp_path / "out")
        assert report.total_rows == 0
        assert report.valid_rows == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Health Report
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthReport:

    def test_perfect_score(self) -> None:
        report = HealthReport(
            total_rows=100,
            valid_rows=100,
            malformed_rows=0,
            dropped_rows=0,
            sequence_gaps=0,
            avg_latency_ms=20.0,
        )
        assert report.health_score > 99.0

    def test_degraded_score(self) -> None:
        report = HealthReport(
            total_rows=100,
            valid_rows=90,
            malformed_rows=10,
            dropped_rows=0,
            sequence_gaps=5,
            avg_latency_ms=500.0,
        )
        assert 0 < report.health_score < 100

    def test_summary_string(self) -> None:
        report = HealthReport(
            total_rows=100,
            valid_rows=95,
            malformed_rows=3,
            dropped_rows=2,
            sequence_gaps=1,
            avg_latency_ms=15.0,
        )
        summary = report.summary()
        assert "Health" in summary or "HEALTH" in summary
        assert "100" in summary  # score denominator

    def test_percentages(self) -> None:
        report = HealthReport(
            total_rows=200,
            valid_rows=180,
            malformed_rows=10,
            dropped_rows=10,
            sequence_gaps=9,
        )
        assert report.malformed_pct == pytest.approx(5.0)
        assert report.dropped_pct == pytest.approx(5.0)
        assert report.sequence_gap_pct == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════════════
#  DataLoader round-trip (JSONL → Parquet → MarketEvent)
# ═══════════════════════════════════════════════════════════════════════════


class TestDataLoaderRoundTrip:

    def test_jsonl_to_parquet_to_events(self, tmp_path: Path) -> None:
        """JSONL → ParquetConverter → DataLoader produces identical events."""
        # Generate synthetic JSONL
        gen = SyntheticGenerator(seed=99)
        raw_dir = gen.generate(
            tmp_path / "raw", num_rows=500, duration_hours=0.5
        )

        # Load events from JSONL
        jsonl_loader = DataLoader.from_directory(raw_dir)
        jsonl_events = list(jsonl_loader)

        # Convert JSONL → Parquet
        converter = ParquetConverter()
        parquet_dir = tmp_path / "processed"
        report = converter.convert([raw_dir], parquet_dir)
        assert report.valid_rows == 500

        # Load events from Parquet
        parquet_loader = DataLoader.from_directory(parquet_dir)
        parquet_events = list(parquet_loader)

        # Same count
        assert len(parquet_events) == len(jsonl_events)

        # Same event content (in same order)
        for je, pe in zip(jsonl_events, parquet_events):
            assert je.timestamp == pytest.approx(pe.timestamp, abs=1e-6)
            assert je.event_type == pe.event_type
            assert je.asset_id == pe.asset_id
            assert je.data == pe.data
            assert je.server_time == pytest.approx(pe.server_time, abs=1e-3)

    def test_parquet_loader_asset_filter(self, tmp_path: Path) -> None:
        """DataLoader with asset_ids filter works for Parquet files."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(
            tmp_path / "raw", num_rows=1_000, duration_hours=0.5
        )

        # Convert
        converter = ParquetConverter()
        parquet_dir = tmp_path / "processed"
        converter.convert([raw_dir], parquet_dir)

        # Find one asset ID
        all_loader = DataLoader.from_directory(parquet_dir)
        all_events = list(all_loader)
        all_assets = {e.asset_id for e in all_events}
        first_asset = sorted(all_assets)[0]

        # Filter to single asset
        filtered_loader = DataLoader.from_directory(
            parquet_dir, asset_ids={first_asset}
        )
        filtered_events = list(filtered_loader)

        assert len(filtered_events) < len(all_events)
        assert all(e.asset_id == first_asset for e in filtered_events)

    def test_mixed_jsonl_and_parquet(self, tmp_path: Path) -> None:
        """DataLoader.from_directory can merge JSONL and Parquet files."""
        # Generate two separate batches
        gen1 = SyntheticGenerator(seed=1)
        raw_dir1 = gen1.generate(
            tmp_path / "raw1", num_rows=200, duration_hours=0.5
        )

        gen2 = SyntheticGenerator(seed=2)
        raw_dir2 = gen2.generate(
            tmp_path / "raw2", num_rows=200, duration_hours=0.5
        )

        # Convert second batch to Parquet
        converter = ParquetConverter()
        parquet_dir = tmp_path / "mixed"
        converter.convert([raw_dir2], parquet_dir)

        # Copy JSONL files into the same directory tree
        import shutil
        for jsonl_file in raw_dir1.rglob("*.jsonl"):
            dest = parquet_dir / jsonl_file.name
            shutil.copy2(jsonl_file, dest)

        # Load from mixed directory — should get all events
        loader = DataLoader.from_directory(parquet_dir)
        events = list(loader)

        assert len(events) == 400

        # Should be chronologically sorted
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp

# ═══════════════════════════════════════════════════════════════════════════
#  Audit File Persistence tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditFilePersistence:
    """Verify that health reports are persisted as JSON audit files."""

    def test_audit_files_created(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Audit JSON files are created alongside Parquet files."""
        out_dir = tmp_path / "processed"
        report = converter.convert([raw_data_dir], out_dir)

        # Should have Parquet files
        parquet_files = list(out_dir.rglob("*.parquet"))
        assert len(parquet_files) >= 1

        # Should have corresponding audit files
        audit_files = list(out_dir.rglob("batch_audit_*.json"))
        assert len(audit_files) >= 1

    def test_audit_file_structure(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Audit JSON contains all required health metrics."""
        out_dir = tmp_path / "processed"
        report = converter.convert([raw_data_dir], out_dir)

        audit_files = list(out_dir.rglob("batch_audit_*.json"))
        assert len(audit_files) >= 1

        for audit_file in audit_files:
            with open(audit_file, "r", encoding="utf-8") as fh:
                audit_data = json.load(fh)

            # Check required fields
            required_fields = {
                "date", "timestamp", "total_rows", "valid_rows",
                "malformed_rows", "dropped_rows", "sequence_gaps",
                "avg_latency_ms", "malformed_pct", "dropped_pct",
                "sequence_gap_pct", "health_score", "output_files",
            }
            assert set(audit_data.keys()) == required_fields

    def test_audit_file_metrics_match_report(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Audit JSON metrics match the returned HealthReport."""
        out_dir = tmp_path / "processed"
        report = converter.convert([raw_data_dir], out_dir)

        audit_files = list(out_dir.rglob("batch_audit_*.json"))
        assert len(audit_files) >= 1

        audit_file = audit_files[0]
        with open(audit_file, "r", encoding="utf-8") as fh:
            audit_data = json.load(fh)

        # Verify key metrics match
        assert audit_data["total_rows"] == report.total_rows
        assert audit_data["valid_rows"] == report.valid_rows
        assert audit_data["malformed_rows"] == report.malformed_rows
        assert audit_data["dropped_rows"] == report.dropped_rows
        assert audit_data["sequence_gaps"] == report.sequence_gaps
        assert abs(audit_data["avg_latency_ms"] - report.avg_latency_ms) < 0.1
        assert abs(audit_data["health_score"] - report.health_score) < 0.1

    def test_audit_file_per_date_partition(self, tmp_path: Path) -> None:
        """One audit file is created per date partition."""
        # Generate data spanning two days (via date partition)
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(
            tmp_path / "raw", num_rows=1_000, duration_hours=36.0
        )

        converter = ParquetConverter()
        out_dir = tmp_path / "processed"
        converter.convert([raw_dir], out_dir)

        # Should have audit files for each date
        parquet_dates = {
            fp.parent.name
            for fp in out_dir.rglob("*.parquet")
        }
        audit_files = list(out_dir.rglob("batch_audit_*.json"))

        # Extract dates from audit filenames
        audit_dates = {
            fp.stem.replace("batch_audit_", "")
            for fp in audit_files
        }

        # Should have one audit file per date partition
        assert len(audit_files) == len(parquet_dates)
        assert audit_dates == parquet_dates

    def test_audit_file_timestamp_is_recent(
        self, raw_data_dir: Path, converter: ParquetConverter, tmp_path: Path
    ) -> None:
        """Audit file timestamp is recent (current time)."""
        import time
        from datetime import datetime, timedelta, timezone

        before = datetime.now(tz=timezone.utc)
        time.sleep(0.1)

        out_dir = tmp_path / "processed"
        converter.convert([raw_data_dir], out_dir)

        time.sleep(0.1)
        after = datetime.now(tz=timezone.utc)

        audit_files = list(out_dir.rglob("batch_audit_*.json"))
        for audit_file in audit_files:
            with open(audit_file, "r", encoding="utf-8") as fh:
                audit_data = json.load(fh)

            # Parse ISO timestamp
            audit_ts = datetime.fromisoformat(
                audit_data["timestamp"].replace("Z", "+00:00")
            )

            # Should be within our test window
            assert before <= audit_ts <= after + timedelta(seconds=1)