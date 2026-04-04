from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import polars as pl


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WINDOW_DAYS = 3
SNAPSHOT_EVENT_TYPES = {"book", "snapshot", "book_snapshot", "l2_snapshot"}
REAL_SMOKE_OUTPUT_ROOT = Path("artifacts") / "scavenger_real_smoke_lake"

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ms", "UTC"),
    "market_id": pl.Utf8,
    "event_id": pl.Utf8,
    "token_id": pl.Utf8,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "bid_depth": pl.Float64,
    "ask_depth": pl.Float64,
    "resolution_timestamp": pl.Datetime("ms", "UTC"),
    "final_resolution_value": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class TokenSmokeMetadata:
    market_id: str
    event_id: str
    token_id: str
    resolution_timestamp: datetime
    final_resolution_value: float


@dataclass(frozen=True, slots=True)
class RealSmokeSlice:
    input_root: Path
    raw_root: Path
    start_date: date
    end_date: date
    candidate_market_count: int
    available_market_count: int
    row_count: int


def prepare_real_scavenger_smoke_lake(project_root: Path) -> RealSmokeSlice | None:
    metadata_path = project_root / "artifacts" / "clob_arb_baseline_metadata.json"
    if not metadata_path.exists():
        return None

    token_metadata, market_resolutions = _load_token_metadata(metadata_path)
    if not token_metadata:
        return None

    best_window = _select_best_window(project_root, market_resolutions)
    if best_window is None:
        return None

    raw_root, window_days, candidate_count, available_market_count = best_window
    output_root = (
        project_root
        / REAL_SMOKE_OUTPUT_ROOT
        / f"{window_days[0].isoformat()}_{window_days[-1].isoformat()}"
    )

    expected_outputs = [output_root / day.isoformat() / "real_smoke.parquet" for day in window_days]
    if all(path.exists() for path in expected_outputs):
        row_count = _count_rows(expected_outputs)
        return RealSmokeSlice(
            input_root=output_root,
            raw_root=raw_root,
            start_date=window_days[0],
            end_date=window_days[-1],
            candidate_market_count=candidate_count,
            available_market_count=available_market_count,
            row_count=row_count,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for window_day in window_days:
        day_dir = raw_root / window_day.isoformat()
        rows = _collect_snapshot_rows(day_dir, token_metadata)
        total_rows += len(rows)
        day_output_dir = output_root / window_day.isoformat()
        day_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = day_output_dir / "real_smoke.parquet"
        if rows:
            frame = pl.DataFrame(rows, schema=OUTPUT_SCHEMA).sort(["timestamp", "market_id", "token_id"])
        else:
            frame = pl.DataFrame(schema=OUTPUT_SCHEMA)
        frame.write_parquet(output_path, compression="zstd")

    return RealSmokeSlice(
        input_root=output_root,
        raw_root=raw_root,
        start_date=window_days[0],
        end_date=window_days[-1],
        candidate_market_count=candidate_count,
        available_market_count=available_market_count,
        row_count=total_rows,
    )


def _load_token_metadata(metadata_path: Path) -> tuple[dict[str, TokenSmokeMetadata], dict[str, datetime]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    markets_by_token = payload.get("markets_by_token")
    if not isinstance(markets_by_token, dict):
        return {}, {}

    token_metadata: dict[str, TokenSmokeMetadata] = {}
    market_resolutions: dict[str, datetime] = {}
    for row in markets_by_token.values():
        if not isinstance(row, dict):
            continue
        market_id = str(row.get("conditionId") or "").strip().lower()
        if not market_id:
            continue

        events = row.get("events")
        if not isinstance(events, list) or not events or not isinstance(events[0], dict):
            continue
        event = events[0]

        event_id = str(event.get("id") or row.get("eventId") or "").strip()
        token_ids = _parse_listish(row.get("clobTokenIds"))
        outcome_prices = _parse_listish(row.get("outcomePrices"))
        if not event_id or len(token_ids) != 2 or len(outcome_prices) != 2:
            continue

        resolution_timestamp = _parse_datetime(row.get("endDate") or event.get("endDate"))
        if resolution_timestamp is None:
            continue

        yes_token, no_token = (str(token_ids[0]).strip(), str(token_ids[1]).strip())
        if not yes_token or not no_token:
            continue

        try:
            yes_final = float(outcome_prices[0])
            no_final = float(outcome_prices[1])
        except (TypeError, ValueError):
            continue

        token_metadata[yes_token] = TokenSmokeMetadata(
            market_id=market_id,
            event_id=event_id,
            token_id="YES",
            resolution_timestamp=resolution_timestamp,
            final_resolution_value=yes_final,
        )
        token_metadata[no_token] = TokenSmokeMetadata(
            market_id=market_id,
            event_id=event_id,
            token_id="NO",
            resolution_timestamp=resolution_timestamp,
            final_resolution_value=no_final,
        )
        market_resolutions[market_id] = resolution_timestamp

    return token_metadata, market_resolutions


def _select_best_window(
    project_root: Path,
    market_resolutions: dict[str, datetime],
) -> tuple[Path, tuple[date, ...], int, int] | None:
    best_window: tuple[Path, tuple[date, ...], int, int] | None = None
    best_key: tuple[int, int, date] | None = None

    for raw_root in _candidate_raw_roots(project_root):
        available_days = _discover_available_days(raw_root)
        for window_days in _contiguous_windows(available_days, WINDOW_DAYS):
            available_market_ids = _available_market_ids(raw_root, window_days)
            if not available_market_ids:
                continue
            candidate_count = 0
            window_start = datetime.combine(window_days[0], datetime.min.time(), tzinfo=UTC)
            window_end = datetime.combine(window_days[-1], datetime.min.time(), tzinfo=UTC) + timedelta(hours=72)
            for market_id in available_market_ids:
                resolution_ts = market_resolutions.get(market_id)
                if resolution_ts is None:
                    continue
                if window_start <= resolution_ts <= window_end:
                    candidate_count += 1

            selection_key = (candidate_count, len(available_market_ids), window_days[-1])
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_window = (raw_root, window_days, candidate_count, len(available_market_ids))

    return best_window


def _candidate_raw_roots(project_root: Path) -> Iterable[Path]:
    for candidate in (
        project_root / "logs" / "local_snapshot" / "l2_data" / "data" / "raw_ticks",
        project_root / "data" / "raw_ticks",
    ):
        if candidate.exists():
            yield candidate


def _discover_available_days(raw_root: Path) -> list[date]:
    return sorted(
        date.fromisoformat(child.name)
        for child in raw_root.iterdir()
        if child.is_dir() and DATE_DIR_RE.match(child.name)
    )


def _contiguous_windows(days: list[date], window_days: int) -> list[tuple[date, ...]]:
    if len(days) < window_days:
        return []
    windows: list[tuple[date, ...]] = []
    for index in range(len(days) - window_days + 1):
        window = days[index : index + window_days]
        if all((window[offset] - window[offset - 1]).days == 1 for offset in range(1, len(window))):
            windows.append(tuple(window))
    return windows


def _available_market_ids(raw_root: Path, window_days: tuple[date, ...]) -> set[str]:
    market_ids: set[str] = set()
    for window_day in window_days:
        day_dir = raw_root / window_day.isoformat()
        if not day_dir.exists():
            continue
        for path in day_dir.glob("0x*.jsonl"):
            market_ids.add(path.stem.lower())
    return market_ids


def _collect_snapshot_rows(day_dir: Path, token_metadata: dict[str, TokenSmokeMetadata]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not day_dir.exists():
        return rows

    for path in sorted(day_dir.glob("*.jsonl")):
        metadata = token_metadata.get(path.stem)
        if metadata is None:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    continue
                event_type = str(payload.get("event_type") or "").strip().lower()
                if event_type not in SNAPSHOT_EVENT_TYPES:
                    continue

                bid_summary = _book_side_summary(payload.get("bids") or [], reverse=True)
                ask_summary = _book_side_summary(payload.get("asks") or [], reverse=False)
                if bid_summary is None or ask_summary is None:
                    continue

                timestamp = _timestamp_to_datetime(payload.get("timestamp") or record.get("local_ts"))
                if timestamp is None:
                    continue

                best_bid, bid_depth = bid_summary
                best_ask, ask_depth = ask_summary
                rows.append(
                    {
                        "timestamp": timestamp,
                        "market_id": metadata.market_id,
                        "event_id": metadata.event_id,
                        "token_id": metadata.token_id,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "bid_depth": bid_depth,
                        "ask_depth": ask_depth,
                        "resolution_timestamp": metadata.resolution_timestamp,
                        "final_resolution_value": metadata.final_resolution_value,
                    }
                )
    return rows


def _book_side_summary(levels: Iterable[Any], *, reverse: bool) -> tuple[float, float] | None:
    parsed: list[tuple[float, float]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        price = _safe_float(level.get("price"))
        size = _safe_float(level.get("size"))
        if price <= 0 or size <= 0:
            continue
        parsed.append((price, size))

    if not parsed:
        return None

    ordered = sorted(parsed, key=lambda item: item[0], reverse=reverse)
    best_price = ordered[0][0]
    depth = sum(price * size for price, size in ordered[:5])
    return best_price, depth


def _timestamp_to_datetime(raw_value: Any) -> datetime | None:
    if raw_value in (None, ""):
        return None
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None
    if numeric > 1e15:
        numeric /= 1_000.0
    elif numeric <= 1e12:
        numeric *= 1_000.0
    return datetime.fromtimestamp(numeric / 1_000.0, tz=UTC)


def _parse_datetime(raw_value: Any) -> datetime | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


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


def _safe_float(raw_value: Any) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0


def _count_rows(paths: Iterable[Path]) -> int:
    total_rows = 0
    for path in paths:
        total_rows += int(pl.scan_parquet(path).select(pl.len()).collect().item())
    return total_rows