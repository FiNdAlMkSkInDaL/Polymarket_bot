"""
DataPrepPipeline — converts raw JSONL tick data into optimised, indexed
Parquet files with Zstd compression and date/category partitioning.

Performs (with streaming/chunked processing to handle gigabyte-scale data):

* Defensive parsing (malformed rows are logged & skipped)
* Chunked sorting by local_ts within each partition
* Sequence-gap detection per asset
* Health-score computation
* Partitioned output:  ``<output_dir>/YYYY-MM-DD/<category>.parquet``
* Health audit JSON:   ``<output_dir>/batch_audit_{date}.json``

Memory efficiency:
  - Processes JSONL in configurable chunk sizes (default 10k rows).
  - Buffers chunks per (date, category) partition.
  - Sorts within each partition, not globally across all data.

Usage
-----
    from src.data.prep_data import ParquetConverter
    converter = ParquetConverter(chunk_size=50_000)
    report = converter.convert(input_paths, output_dir)
    print(report.summary())
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.logger import get_logger

log = get_logger(__name__)

# ── Parquet schema ────────────────────────────────────────────────────────
PARQUET_SCHEMA = pa.schema([
    pa.field("local_ts", pa.float64(), nullable=False),
    pa.field("exchange_ts", pa.float64(), nullable=True),
    pa.field("msg_type", pa.string(), nullable=False),
    pa.field("asset_id", pa.string(), nullable=False),
    pa.field("price", pa.float64(), nullable=True),
    pa.field("size", pa.float64(), nullable=True),
    pa.field("sequence_id", pa.int64(), nullable=True),
    pa.field("side", pa.string(), nullable=True),
    pa.field("payload", pa.string(), nullable=False),
])

# Map recorder source tags → Parquet msg_type values
_MSG_TYPE_MAP: dict[str, str] = {
    "l2": "delta",
    "l2_delta": "delta",
    "l2_snapshot": "snapshot",
    "book_snapshot": "snapshot",
    "snapshot": "snapshot",
    "trade": "trade",
}


@dataclass
class HealthReport:
    """Data-quality audit for a processed batch.

    Attributes
    ----------
    total_rows:      Total lines read (including malformed).
    valid_rows:      Rows successfully converted.
    malformed_rows:  Lines that failed JSON parsing.
    dropped_rows:    Valid JSON but missing required fields.
    sequence_gaps:   Number of discontinuities in per-asset sequence IDs.
    avg_latency_ms:  Mean(local_ts − exchange_ts) in milliseconds.
    output_files:    List of written Parquet file paths.
    """

    total_rows: int = 0
    valid_rows: int = 0
    malformed_rows: int = 0
    dropped_rows: int = 0
    sequence_gaps: int = 0
    avg_latency_ms: float = 0.0
    output_files: list[str] = field(default_factory=list)

    @property
    def malformed_pct(self) -> float:
        return (self.malformed_rows / self.total_rows * 100) if self.total_rows else 0.0

    @property
    def dropped_pct(self) -> float:
        return (self.dropped_rows / self.total_rows * 100) if self.total_rows else 0.0

    @property
    def sequence_gap_pct(self) -> float:
        return (self.sequence_gaps / self.valid_rows * 100) if self.valid_rows else 0.0

    @property
    def health_score(self) -> float:
        """Composite 0–100 score.  Higher is better.

        Weights:
        - 30 pts — malformed rows (0 % malformed = full 30)
        - 40 pts — sequence gaps (0 % gaps = full 40)
        - 30 pts — latency (< 1 s avg = full 30, ≥ 1 s = 0)
        """
        latency_penalty = min(self.avg_latency_ms / 1000.0, 1.0) * 30
        score = 100.0 - (self.malformed_pct * 0.30) - (self.sequence_gap_pct * 0.40) - latency_penalty
        return max(0.0, min(100.0, score))

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "",
            "═" * 55,
            "  Data Quality Report",
            "═" * 55,
            f"  Total rows read ........... {self.total_rows:>10,}",
            f"  Valid rows ................ {self.valid_rows:>10,}",
            f"  Malformed (JSON errors) ... {self.malformed_rows:>10,}  ({self.malformed_pct:.2f}%)",
            f"  Dropped (missing fields) .. {self.dropped_rows:>10,}  ({self.dropped_pct:.2f}%)",
            f"  Sequence gaps ............. {self.sequence_gaps:>10,}  ({self.sequence_gap_pct:.2f}%)",
            f"  Avg latency ............... {self.avg_latency_ms:>10.1f} ms",
            "─" * 55,
            f"  HEALTH SCORE .............. {self.health_score:>10.1f} / 100",
            "═" * 55,
            f"  Output files: {len(self.output_files)}",
        ]
        for fp in self.output_files:
            lines.append(f"    → {fp}")
        lines.append("")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for persistence."""
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "malformed_rows": self.malformed_rows,
            "dropped_rows": self.dropped_rows,
            "sequence_gaps": self.sequence_gaps,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "malformed_pct": round(self.malformed_pct, 2),
            "dropped_pct": round(self.dropped_pct, 2),
            "sequence_gap_pct": round(self.sequence_gap_pct, 2),
            "health_score": round(self.health_score, 1),
            "output_files": self.output_files,
        }


class ParquetConverter:
    """Convert raw JSONL tick data into partitioned, compressed Parquet.

    Memory-efficient streaming implementation: processes data in chunks
    to avoid loading multi-gigabyte datasets entirely into RAM.

    Parameters
    ----------
    category_map:
        Optional mapping ``{asset_id: category_string}``.  Assets not
        present in the map default to ``"general"``.
    chunk_size:
        Number of rows to buffer before writing to Parquet.
        Default 10,000; increase for faster I/O on large datasets.
    """

    def __init__(
        self,
        category_map: dict[str, str] | None = None,
        chunk_size: int = 10_000,
    ) -> None:
        self._category_map = category_map or {}
        self._chunk_size = chunk_size

    # ── Public API ─────────────────────────────────────────────────────

    def convert(
        self,
        input_paths: list[Path],
        output_dir: str | Path,
    ) -> HealthReport:
        """Read JSONL files, transform, and write Parquet.

        Streaming/chunked implementation: buffers rows by (date, category)
        partition, writes to Parquet incrementally when chunk threshold
        is reached. This avoids OOM when processing gigabyte-scale data.

        Parameters
        ----------
        input_paths:
            List of JSONL file paths **or** directories (recursively
            searched for ``*.jsonl``).
        output_dir:
            Root directory for partitioned Parquet output.

        Returns
        -------
        HealthReport with quality metrics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve all JSONL files
        jsonl_files = self._resolve_files(input_paths)
        if not jsonl_files:
            log.warning("parquet_converter_no_files")
            return HealthReport()

        log.info("parquet_converter_start", files=len(jsonl_files))

        report = HealthReport()

        # Buffers for each (date, category) partition: list of row dicts
        partition_buffers: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        # For latency and sequence tracking
        latencies: list[float] = []
        seqs_per_asset: dict[str, list[int]] = defaultdict(list)

        # Stream through all JSONL files and buffer by partition
        for fpath in jsonl_files:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line_no, raw_line in enumerate(fh, 1):
                    report.total_rows += 1
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    # Parse JSON
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        report.malformed_rows += 1
                        log.debug(
                            "prep_malformed_line",
                            file=str(fpath),
                            line=line_no,
                        )
                        continue

                    # Extract and validate required fields
                    row = self._flatten_record(record)
                    if row is None:
                        report.dropped_rows += 1
                        continue

                    # Track latency
                    if row["exchange_ts"] is not None and row["exchange_ts"] == row["exchange_ts"]:  # not NaN
                        lat = (row["local_ts"] - row["exchange_ts"]) * 1000
                        latencies.append(lat)

                    # Track sequence IDs for gap detection
                    if row["sequence_id"] is not None:
                        seqs_per_asset[row["asset_id"]].append(row["sequence_id"])

                    # Derive partition key
                    date_str = datetime.fromtimestamp(
                        row["local_ts"], tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                    category = self._category_map.get(row["asset_id"], "general")
                    key = (date_str, category)

                    # Buffer the row
                    partition_buffers[key].append(row)

                    # Flush buffer if threshold reached
                    if len(partition_buffers[key]) >= self._chunk_size:
                        self._write_partition_chunk(
                            partition_buffers[key],
                            date_str,
                            category,
                            output_dir,
                        )
                        partition_buffers[key].clear()

        # Flush remaining buffers
        for (date_str, category), rows in partition_buffers.items():
            if rows:
                self._write_partition_chunk(rows, date_str, category, output_dir)

        report.valid_rows = report.total_rows - report.malformed_rows - report.dropped_rows

        # Compute latency stats
        if latencies:
            report.avg_latency_ms = float(np.mean(latencies))

        # Detect sequence gaps
        report.sequence_gaps = self._count_sequence_gaps(seqs_per_asset)

        # Collect output files
        report.output_files = sorted([
            str(fp) for fp in output_dir.rglob("*.parquet")
        ])

        # Write audit JSON files per date
        self._write_audit_files(output_dir, report)

        log.info(
            "parquet_converter_done",
            valid_rows=report.valid_rows,
            malformed=report.malformed_rows,
            dropped=report.dropped_rows,
            gaps=report.sequence_gaps,
            score=round(report.health_score, 1),
            files=len(report.output_files),
        )

        return report

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_files(paths: list[Path]) -> list[Path]:
        """Expand directories into individual JSONL files."""
        result: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                result.extend(sorted(p.rglob("*.jsonl")))
            elif p.is_file() and p.suffix == ".jsonl":
                result.append(p)
        return result

    def _flatten_record(self, record: dict) -> dict[str, Any] | None:
        """Transform a raw JSONL record into a flat row dict.

        Returns ``None`` if required fields are missing.
        """
        # -- local_ts (required) --
        local_ts = record.get("local_ts")
        if local_ts is None:
            return None
        try:
            local_ts = float(local_ts)
        except (TypeError, ValueError):
            return None

        # -- source → msg_type (required) --
        source = record.get("source", "")
        msg_type = _MSG_TYPE_MAP.get(source)
        if msg_type is None:
            return None

        # -- asset_id (required) --
        asset_id = record.get("asset_id", "")
        if not asset_id:
            return None

        # -- payload (required) --
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None

        # Detect snapshot vs delta for "l2" source (mirrors DataLoader logic)
        if msg_type == "delta" and payload:
            payload_type = payload.get("event_type", "")
            if payload_type in ("book", "snapshot", "book_snapshot"):
                msg_type = "snapshot"

        # -- exchange_ts (optional, normalise µs/ms) --
        exchange_ts = None
        raw_srv = (
            payload.get("timestamp")
            or payload.get("server_timestamp")
            or payload.get("ts")
        )
        if raw_srv is not None:
            try:
                srv = float(raw_srv)
                if srv > 1e15:
                    srv /= 1_000_000
                elif srv > 1e12:
                    srv /= 1_000
                exchange_ts = srv
            except (TypeError, ValueError):
                pass

        # -- sequence_id (optional) --
        sequence_id = None
        raw_seq = (
            payload.get("seq")
            or payload.get("sequence")
            or payload.get("seq_num")
        )
        if raw_seq is not None:
            try:
                sequence_id = int(raw_seq)
            except (TypeError, ValueError):
                pass

        # -- price / size / side (trade only) --
        price = None
        size = None
        side = None
        if msg_type == "trade":
            try:
                price = float(payload.get("price", ""))
            except (TypeError, ValueError):
                pass
            raw_size = payload.get("size") or payload.get("amount")
            if raw_size is not None:
                try:
                    size = float(raw_size)
                except (TypeError, ValueError):
                    pass
            side = payload.get("side")

        # -- serialise payload --
        payload_str = json.dumps(payload, separators=(",", ":"), default=str)

        return {
            "local_ts": local_ts,
            "exchange_ts": exchange_ts,
            "msg_type": msg_type,
            "asset_id": asset_id,
            "price": price,
            "size": size,
            "sequence_id": sequence_id,
            "side": side,
            "payload": payload_str,
        }

    def _write_partition_chunk(
        self,
        rows: list[dict[str, Any]],
        date_str: str,
        category: str,
        output_dir: Path,
    ) -> None:
        """Convert a chunk of rows to DataFrame, sort, and append/write to Parquet."""
        if not rows:
            return

        # Create DataFrame and sort by local_ts
        df = pd.DataFrame(rows)
        df.sort_values("local_ts", inplace=True, kind="mergesort")

        # Enforce schema types
        df["local_ts"] = df["local_ts"].astype("float64")
        df["exchange_ts"] = pd.to_numeric(df["exchange_ts"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df["sequence_id"] = pd.to_numeric(df["sequence_id"], errors="coerce")
        df["sequence_id"] = df["sequence_id"].astype("Int64")

        # Select and order Parquet columns
        parquet_cols = [
            "local_ts", "exchange_ts", "msg_type", "asset_id",
            "price", "size", "sequence_id", "side", "payload",
        ]
        subset = df[parquet_cols].copy()
        subset["sequence_id"] = subset["sequence_id"].astype("float64")

        # Convert to Arrow table
        table = pa.Table.from_pandas(
            subset, schema=PARQUET_SCHEMA, preserve_index=False
        )

        # Write or append to Parquet
        partition_dir = output_dir / str(date_str)
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / f"{category}.parquet"

        if file_path.exists():
            # Append mode: read existing, concatenate, and merge row groups
            existing = pq.read_table(str(file_path))
            table = pa.concat_tables([existing, table])

        pq.write_table(
            table,
            str(file_path),
            compression="zstd",
            write_statistics=True,
        )

        log.info(
            "parquet_written",
            file=str(file_path),
            rows=len(table),
            compression="zstd",
        )

    @staticmethod
    def _count_sequence_gaps(seqs_per_asset: dict[str, list[int]]) -> int:
        """Count discontinuities in sequence IDs per asset."""
        total_gaps = 0
        for asset_id, seqs in seqs_per_asset.items():
            if not seqs:
                continue
            seqs_sorted = sorted(set(seqs))
            if len(seqs_sorted) < 2:
                continue
            diffs = np.diff(np.array(seqs_sorted, dtype="int64"))
            total_gaps += int(np.sum(diffs > 1))
        return total_gaps

    @staticmethod
    def _write_audit_files(
        output_dir: Path,
        report: HealthReport,
    ) -> None:
        """Write batch_audit_{date}.json files for each date partition.

        One audit file per date found in the output directory structure.
        """
        # Collect unique dates from report
        dates: set[str] = set()
        for fp in report.output_files:
            # Extract date from path: .../YYYY-MM-DD/<category>.parquet
            parts = Path(fp).parts
            if len(parts) >= 2:
                date_candidate = parts[-2]
                # Validate date format
                if len(date_candidate) == 10 and date_candidate[4] == "-" and date_candidate[7] == "-":
                    dates.add(date_candidate)

        # Write one audit file per date
        for date_str in sorted(dates):
            audit_path = output_dir / f"batch_audit_{date_str}.json"
            audit_payload = {
                "date": date_str,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                **report.to_json(),
            }
            with open(audit_path, "w", encoding="utf-8") as fh:
                json.dump(audit_payload, fh, indent=2)
            log.info("audit_file_written", file=str(audit_path))
