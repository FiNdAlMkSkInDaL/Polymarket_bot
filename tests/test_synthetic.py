"""
Tests for the Synthetic Data Generator (``src.data.synthetic``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.synthetic import SyntheticGenerator


# ── Helpers ────────────────────────────────────────────────────────────────

def _read_all_records(raw_ticks_dir: Path) -> list[dict]:
    """Read every JSONL line from every file under *raw_ticks_dir*."""
    records: list[dict] = []
    for fp in sorted(raw_ticks_dir.rglob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ── Tests ──────────────────────────────────────────────────────────────────

class TestSyntheticGenerator:

    def test_generates_requested_row_count(self, tmp_path: Path) -> None:
        """Output contains exactly *num_rows* records."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=1_000, duration_hours=1.0)

        records = _read_all_records(raw_dir)
        assert len(records) == 1_000

    def test_output_directory_layout(self, tmp_path: Path) -> None:
        """Files are nested under raw_ticks/YYYY-MM-DD/<asset>.jsonl."""
        gen = SyntheticGenerator(seed=1)
        raw_dir = gen.generate(tmp_path, num_rows=500, duration_hours=1.0)

        assert raw_dir.exists()
        # Should have at least one date directory
        date_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        assert len(date_dirs) >= 1

        # Each date dir should have .jsonl files
        for dd in date_dirs:
            jsonl_files = list(dd.glob("*.jsonl"))
            assert len(jsonl_files) >= 1

    def test_record_schema(self, tmp_path: Path) -> None:
        """Every record has the required top-level keys."""
        gen = SyntheticGenerator(seed=7)
        raw_dir = gen.generate(tmp_path, num_rows=500, duration_hours=0.5)

        for rec in _read_all_records(raw_dir):
            assert "local_ts" in rec
            assert "source" in rec
            assert "asset_id" in rec
            assert "payload" in rec
            assert isinstance(rec["local_ts"], float)
            assert rec["source"] in ("l2", "trade")
            assert isinstance(rec["payload"], dict)

    def test_source_values(self, tmp_path: Path) -> None:
        """Source is either 'l2' or 'trade'."""
        gen = SyntheticGenerator(seed=99)
        raw_dir = gen.generate(tmp_path, num_rows=2_000, duration_hours=1.0)

        sources = {rec["source"] for rec in _read_all_records(raw_dir)}
        assert sources <= {"l2", "trade"}
        # With 2000 rows and p(trade)=0.25, both should appear
        assert "l2" in sources
        assert "trade" in sources

    def test_local_ts_monotonic_per_file(self, tmp_path: Path) -> None:
        """Within each file, local_ts is non-decreasing."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=2_000, duration_hours=2.0)

        for fp in raw_dir.rglob("*.jsonl"):
            prev = 0.0
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ts = rec["local_ts"]
                    assert ts >= prev, f"non-monotonic in {fp}: {ts} < {prev}"
                    prev = ts

    def test_sequence_ids_increasing(self, tmp_path: Path) -> None:
        """Sequence IDs (in payload.seq) increase per asset."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=2_000, duration_hours=1.0)

        records = _read_all_records(raw_dir)
        # Group by asset_id, check seq is increasing
        per_asset: dict[str, list[int]] = {}
        for rec in records:
            payload = rec["payload"]
            seq = payload.get("seq")
            if seq is not None:
                per_asset.setdefault(rec["asset_id"], []).append(seq)

        # L2 messages have seq; trades don't necessarily but our generator
        # increments seq for all events, so every payload with 'seq' should
        # be monotonically increasing per-asset when sorted by local_ts.
        for asset_id, seqs in per_asset.items():
            for i in range(1, len(seqs)):
                assert seqs[i] > seqs[i - 1], (
                    f"sequence not increasing for {asset_id}: "
                    f"seq[{i-1}]={seqs[i-1]}, seq[{i}]={seqs[i]}"
                )

    def test_prices_in_range(self, tmp_path: Path) -> None:
        """All prices stay within (0, 1)."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=3_000, duration_hours=2.0)

        for rec in _read_all_records(raw_dir):
            p = rec["payload"]
            if "price" in p:
                price = float(p["price"])
                assert 0 < price < 1, f"price out of range: {price}"
            for change in p.get("changes", []):
                price = float(change["price"])
                assert 0 < price < 1, f"change price out of range: {price}"
            for bid in p.get("bids", []):
                price = float(bid["price"])
                assert 0 < price < 1
            for ask in p.get("asks", []):
                price = float(ask["price"])
                assert 0 < price < 1

    def test_multiple_assets(self, tmp_path: Path) -> None:
        """Default generates events for 2 assets (YES + NO)."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=2_000, duration_hours=1.0)

        asset_ids = {rec["asset_id"] for rec in _read_all_records(raw_dir)}
        assert len(asset_ids) == 2

    def test_single_asset(self, tmp_path: Path) -> None:
        """num_assets=1 generates only one asset."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(
            tmp_path, num_rows=500, duration_hours=0.5, num_assets=1
        )

        asset_ids = {rec["asset_id"] for rec in _read_all_records(raw_dir)}
        assert len(asset_ids) == 1

    def test_trade_payload_fields(self, tmp_path: Path) -> None:
        """Trade payloads contain price, size, side, timestamp, asset_id."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=2_000, duration_hours=1.0)

        trades = [
            rec for rec in _read_all_records(raw_dir) if rec["source"] == "trade"
        ]
        assert len(trades) > 0

        for rec in trades:
            p = rec["payload"]
            assert "price" in p
            assert "size" in p
            assert "side" in p
            assert p["side"] in ("buy", "sell")
            assert "timestamp" in p
            assert "asset_id" in p

    def test_l2_snapshot_payload(self, tmp_path: Path) -> None:
        """L2 snapshot payloads contain bids, asks, seq, event_type='book'."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=5_000, duration_hours=2.0)

        records = _read_all_records(raw_dir)
        snapshots = [
            rec
            for rec in records
            if rec["source"] == "l2"
            and rec["payload"].get("event_type") == "book"
        ]
        assert len(snapshots) > 0

        for rec in snapshots:
            p = rec["payload"]
            assert isinstance(p["bids"], list)
            assert isinstance(p["asks"], list)
            assert "seq" in p

    def test_l2_delta_payload(self, tmp_path: Path) -> None:
        """L2 delta payloads contain changes array with side/price/size."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=3_000, duration_hours=1.0)

        records = _read_all_records(raw_dir)
        deltas = [
            rec
            for rec in records
            if rec["source"] == "l2"
            and rec["payload"].get("event_type") == "price_change"
        ]
        assert len(deltas) > 0

        for rec in deltas:
            p = rec["payload"]
            assert isinstance(p["changes"], list)
            assert len(p["changes"]) >= 1
            for ch in p["changes"]:
                assert "side" in ch
                assert "price" in ch
                assert "size" in ch

    def test_reproducibility(self, tmp_path: Path) -> None:
        """Same seed produces identical output."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        gen1 = SyntheticGenerator(seed=123)
        gen1.generate(dir1, num_rows=500, duration_hours=0.5)

        gen2 = SyntheticGenerator(seed=123)
        gen2.generate(dir2, num_rows=500, duration_hours=0.5)

        recs1 = _read_all_records(dir1 / "raw_ticks")
        recs2 = _read_all_records(dir2 / "raw_ticks")

        assert len(recs1) == len(recs2)
        for r1, r2 in zip(recs1, recs2):
            assert r1 == r2

    def test_latency_positive(self, tmp_path: Path) -> None:
        """local_ts > exchange_ts (positive latency)."""
        gen = SyntheticGenerator(seed=42)
        raw_dir = gen.generate(tmp_path, num_rows=1_000, duration_hours=0.5)

        for rec in _read_all_records(raw_dir):
            p = rec["payload"]
            exch_ts = p.get("timestamp")
            if exch_ts is not None:
                assert rec["local_ts"] > exch_ts
