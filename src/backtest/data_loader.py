"""
Historical market data loader -- replays recorded JSONL tick files
(and optionally Parquet files from the DataPrepPipeline) in strict
chronological order for the backtesting engine.

Uses a ``heapq``-based merge across multiple files to produce events
in ascending ``local_ts`` order, ensuring correct interleaving of L2
deltas and trades from different assets.

Supported formats
─────────────────
* ``.jsonl`` — raw recorder output (one JSON object per line).
* ``.parquet`` — optimised columnar output from ``ParquetConverter``.
  Requires ``pyarrow`` (optional dependency: ``pip install pyarrow``).

Usage
─────
    loader = DataLoader.from_directory("data/raw_ticks/2026-02-25")
    for event in loader:
        engine.process(event)
"""

from __future__ import annotations

import datetime
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from src.core.logger import get_logger

log = get_logger(__name__)


# ── Event type literals ────────────────────────────────────────────────────
EventType = Literal["l2_delta", "l2_snapshot", "trade", "external_price"]

# Map the recorder's source tags → canonical event types
_SOURCE_MAP: dict[str, EventType] = {
    "l2": "l2_delta",
    "l2_delta": "l2_delta",
    "l2_snapshot": "l2_snapshot",
    "book_snapshot": "l2_snapshot",
    "snapshot": "l2_snapshot",
    "trade": "trade",
    "external_price": "external_price",
    "rpe_signal": "external_price",
}

# Reverse map: Parquet msg_type → canonical EventType
_PARQUET_MSG_MAP: dict[str, EventType] = {
    "delta": "l2_delta",
    "snapshot": "l2_snapshot",
    "trade": "trade",
    "external_price": "external_price",
    # Also accept canonical values directly
    "l2_delta": "l2_delta",
    "l2_snapshot": "l2_snapshot",
}


@dataclass(slots=True)
class MarketEvent:
    """A single timestamped market event for replay.

    Attributes
    ----------
    timestamp:
        Local receive-time (Unix epoch seconds) — the sort key.
    event_type:
        One of ``"l2_delta"``, ``"l2_snapshot"``, ``"trade"``.
    asset_id:
        The token / asset ID this event relates to.
    data:
        The raw payload dict (unmutated from what was recorded).
    server_time:
        Exchange-side timestamp (if available in the payload).
    """

    timestamp: float
    event_type: EventType
    asset_id: str
    data: dict
    server_time: float = 0.0


class DataLoader:
    """Chronological event stream from JSONL and/or Parquet tick files.

    Parameters
    ----------
    files:
        One or more ``Path`` objects to JSONL **or** Parquet files.
    asset_ids:
        If provided, only events for these asset IDs are emitted.
        ``None`` means all assets.
    """

    _SUPPORTED_SUFFIXES = {".jsonl", ".parquet"}

    def __init__(
        self,
        files: list[Path],
        *,
        asset_ids: set[str] | None = None,
    ) -> None:
        self._files = [f for f in files if f.suffix in self._SUPPORTED_SUFFIXES]
        self._asset_filter = asset_ids
        self._total_events: int = 0
        self._skipped_events: int = 0

    # ── Factory constructors ───────────────────────────────────────────

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        asset_ids: set[str] | None = None,
    ) -> DataLoader:
        """Load all JSONL and Parquet files from a directory tree.

        Parameters
        ----------
        directory:
            Path like ``data/raw_ticks/2026-02-25``, ``data/processed``,
            or any parent (will recurse into subdirs).
        """
        d = Path(directory)
        if not d.exists():
            raise FileNotFoundError(f"DataLoader: directory not found: {d}")

        jsonl_files = sorted(d.rglob("*.jsonl"))
        parquet_files = sorted(d.rglob("*.parquet"))
        files = sorted(jsonl_files + parquet_files, key=lambda p: p.name)

        if not files:
            raise FileNotFoundError(
                f"DataLoader: no .jsonl or .parquet files in {d}"
            )

        log.info(
            "dataloader_init",
            directory=str(d),
            jsonl=len(jsonl_files),
            parquet=len(parquet_files),
        )
        return cls(files, asset_ids=asset_ids)

    @classmethod
    def from_files(
        cls,
        *paths: str | Path,
        asset_ids: set[str] | None = None,
    ) -> DataLoader:
        """Load from explicit file paths."""
        files = [Path(p) for p in paths]
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"DataLoader: file not found: {f}")
        return cls(files, asset_ids=asset_ids)

    # ── Iterator ───────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[MarketEvent]:
        """Yield ``MarketEvent`` objects in strict chronological order.

        Uses a heap-merge across all files. Each file is assumed to contain
        events in **non-decreasing** ``local_ts`` order (as written by
        ``MarketDataRecorder`` or ``ParquetConverter``).
        """
        self._total_events = 0
        self._skipped_events = 0

        streams = []
        for file_idx, fpath in enumerate(self._files):
            if fpath.suffix == ".parquet":
                streams.append(self._parquet_stream(fpath, file_idx))
            else:
                streams.append(self._file_stream(fpath, file_idx))

        for event in heapq.merge(*streams, key=lambda e: e.timestamp):
            self._total_events += 1
            yield event

        log.info(
            "dataloader_done",
            total=self._total_events,
            skipped=self._skipped_events,
        )

    # Maximum span (seconds) that triggers synthetic timestamp spreading.
    # Backfilled data stored all trades at the backfill run time, so all
    # events in a single day file may be compressed into < 60 seconds.
    # When detected, we spread events evenly across the file's calendar date.
    _SYNTH_TS_THRESHOLD: float = 60.0

    @staticmethod
    def _day_start_from_path(fpath: Path) -> float | None:
        """Return midnight UTC (epoch seconds) for the date in the parent dir.

        Expects a path like ``ticks/2025-12-05/<market>.jsonl``.
        Returns ``None`` if the parent directory name is not a valid ISO date.
        """
        try:
            d = datetime.date.fromisoformat(fpath.parent.name)
            return datetime.datetime(d.year, d.month, d.day,
                                     tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            return None

    def _file_stream(self, fpath: Path, file_idx: int) -> Iterator[MarketEvent]:
        """Read a single JSONL file, yielding ``MarketEvent`` objects.

        When all timestamps in the file are compressed into a window smaller
        than ``_SYNTH_TS_THRESHOLD`` seconds (a known artifact of the
        Polymarket backfill script), timestamps are replaced with synthetic
        values spread evenly across the file's calendar day so that OHLCV
        bars can form correctly during backtest replay.
        """
        # --- Pass 1: buffer all valid records and check timestamp spread ---
        raw_records: list[tuple[dict, int]] = []   # (record, line_no)
        line_no = 0
        with open(fpath, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line_no += 1
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    log.warning("dataloader_bad_line", file=str(fpath), line=line_no)
                    self._skipped_events += 1
                    continue
                raw_records.append((record, line_no))

        if not raw_records:
            return

        # Determine if timestamps are collapsed (backfilled data artifact)
        ts_vals = [float(r.get("local_ts", 0)) for r, _ in raw_records if r.get("local_ts") is not None]
        ts_span = (max(ts_vals) - min(ts_vals)) if ts_vals else 0.0

        synth_day_start: float | None = None
        if ts_span < self._SYNTH_TS_THRESHOLD:
            synth_day_start = self._day_start_from_path(fpath)
            if synth_day_start is not None:
                # Only spread if the stored timestamps are real modern Unix
                # timestamps but mismatched from the file's calendar date
                # (backfill artifact: local_ts is the backfill run time, not
                # the historical date).  Test data typically uses small
                # synthetic values (e.g. 1.0, 2.0) that are valid timestamps
                # for near-epoch dates — we leave those unchanged.
                # Threshold: 2020-01-01 (1577836800) distinguishes real vs
                # synthetic test timestamps; 30-day tolerance for data
                # legitimately written in the same period as the file.
                _MIN_REAL_TS = 1_577_836_800.0   # 2020-01-01 UTC
                median_ts = sorted(ts_vals)[len(ts_vals) // 2] if ts_vals else 0.0
                ts_is_real = median_ts >= _MIN_REAL_TS
                ts_far_from_date = abs(median_ts - synth_day_start) > 30 * 86400
                if not (ts_is_real and ts_far_from_date):
                    synth_day_start = None  # timestamps are fine as-is
                else:
                    log.debug(
                        "dataloader_synth_ts",
                        file=str(fpath),
                        original_span_s=round(ts_span, 1),
                    )

        # --- Pass 2: emit events with corrected timestamps ---
        total = len(raw_records)
        prev_ts = 0.0
        for seq, (record, rec_line_no) in enumerate(raw_records):
            event = self._parse_record(record, fpath, rec_line_no)
            if event is None:
                self._skipped_events += 1
                continue

            # Replace collapsed timestamp with synthetic one spread across the day
            if synth_day_start is not None:
                event = MarketEvent(
                    timestamp=synth_day_start + seq * (86400.0 / max(total, 1)),
                    event_type=event.event_type,
                    asset_id=event.asset_id,
                    data=event.data,
                    server_time=event.server_time,
                )

            if event.timestamp < prev_ts:
                log.warning(
                    "dataloader_non_monotonic",
                    file=str(fpath),
                    line=rec_line_no,
                    ts=event.timestamp,
                    prev_ts=prev_ts,
                )
            prev_ts = event.timestamp
            yield event

    def _parse_record(
        self, record: dict, fpath: Path, line_no: int
    ) -> MarketEvent | None:
        """Parse a raw JSONL record into a ``MarketEvent``."""
        local_ts = record.get("local_ts")
        if local_ts is None:
            return None

        try:
            local_ts = float(local_ts)
        except (TypeError, ValueError):
            return None

        source = record.get("source", "")
        event_type = _SOURCE_MAP.get(source)
        if event_type is None:
            return None

        asset_id = record.get("asset_id", "")
        if not asset_id:
            return None

        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None

        # Prefer the per-token asset_id stored in the payload (decimal token
        # id) over the top-level market-id (hex condition_id).  The live
        # recorder stores the hex market_id at the top level but the
        # Polymarket API returns the decimal token id inside the payload.
        payload_token_id = payload.get("asset_id", "")
        if payload_token_id:
            asset_id = str(payload_token_id)

        # Apply asset filter (evaluated after payload id resolution)
        if self._asset_filter and asset_id not in self._asset_filter:
            return None

        # Reconcile event type using payload hints.
        # The recorder uses two top-level ``source`` tags ("l2" and
        # "trade") but the payload's ``event_type`` is the ground truth.
        if payload:
            payload_type = payload.get("event_type", "")

            if payload_type in ("book", "snapshot", "book_snapshot"):
                # Full L2 snapshot regardless of source tag
                event_type = "l2_snapshot"
            elif payload_type == "last_trade_price":
                # Public trade — contains price/size/side
                event_type = "trade"

        # Extract server timestamp if present
        server_time = 0.0
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
                server_time = srv
            except (TypeError, ValueError):
                pass

        return MarketEvent(
            timestamp=local_ts,
            event_type=event_type,
            asset_id=asset_id,
            data=payload,
            server_time=server_time,
        )

    # ── Parquet support ─────────────────────────────────────────────────

    def _parquet_stream(
        self, fpath: Path, file_idx: int
    ) -> Iterator[MarketEvent]:
        """Read a Parquet file (from ``ParquetConverter``) and yield
        ``MarketEvent`` objects.  The file is assumed to be pre-sorted
        by ``local_ts``."""
        try:
            import pyarrow.parquet as pq  # noqa: F811
        except ImportError:
            raise ImportError(
                "pyarrow is required to read .parquet files.  "
                "Install it with:  pip install 'polymarket-bot[data]'"
            ) from None

        table = pq.read_table(str(fpath))
        # Fast column access
        col_local_ts = table.column("local_ts").to_pylist()
        col_msg_type = table.column("msg_type").to_pylist()
        col_asset_id = table.column("asset_id").to_pylist()
        col_payload = table.column("payload").to_pylist()
        col_exchange_ts = table.column("exchange_ts").to_pylist()

        for idx in range(table.num_rows):
            local_ts = col_local_ts[idx]

            # Map msg_type back to canonical EventType
            msg_type = col_msg_type[idx]
            event_type = _PARQUET_MSG_MAP.get(msg_type)
            if event_type is None:
                self._skipped_events += 1
                continue

            asset_id = col_asset_id[idx]
            if not asset_id:
                self._skipped_events += 1
                continue

            # Apply asset filter
            if self._asset_filter and asset_id not in self._asset_filter:
                continue

            # Reconstruct payload dict from JSON string
            payload_str = col_payload[idx]
            try:
                data = json.loads(payload_str)
            except (json.JSONDecodeError, TypeError):
                self._skipped_events += 1
                continue

            server_time = col_exchange_ts[idx]
            if server_time is None or server_time != server_time:  # NaN check
                server_time = 0.0

            yield MarketEvent(
                timestamp=float(local_ts),
                event_type=event_type,
                asset_id=asset_id,
                data=data,
                server_time=float(server_time),
            )

    # ── Diagnostics ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_events": self._total_events,
            "skipped_events": self._skipped_events,
            "file_count": len(self._files),
        }
