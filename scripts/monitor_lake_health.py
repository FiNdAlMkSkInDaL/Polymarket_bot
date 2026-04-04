#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_STATE = PROJECT_ROOT / "artifacts" / "l2_parquet_lake_rolling" / "sync_state.json"
DEFAULT_MAX_AGE_MINUTES = 65.0


@dataclass(frozen=True, slots=True)
class HealthStatus:
    level: str
    exit_code: int
    message: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the rolling lake sync_state.json is fresh enough for live operation.",
    )
    parser.add_argument(
        "--sync-state",
        type=Path,
        default=DEFAULT_SYNC_STATE,
        help="Path to sync_state.json produced by scripts/sync_lake_from_vps.py.",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=DEFAULT_MAX_AGE_MINUTES,
        help="Maximum allowed age for the last successful sync before emitting an alert.",
    )
    return parser.parse_args()


def load_sync_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"ALERT: sync_state.json not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ALERT: sync_state.json at {path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"ALERT: sync_state.json at {path} does not contain a JSON object")
    return payload


def _parse_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"ALERT: sync_state.json is missing {field_name}")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"ALERT: sync_state.json field {field_name} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def evaluate_sync_state(payload: dict[str, Any], *, now: datetime | None = None, max_age_minutes: float = DEFAULT_MAX_AGE_MINUTES) -> HealthStatus:
    if max_age_minutes <= 0:
        raise ValueError("max_age_minutes must be positive")

    current_time = now.astimezone(UTC) if now is not None else datetime.now(tz=UTC)
    timestamp_field = "last_successful_sync_at" if payload.get("last_successful_sync_at") else "generated_at"
    last_success = _parse_timestamp(payload.get(timestamp_field), field_name=timestamp_field)
    age_minutes = (current_time - last_success).total_seconds() / 60.0
    latest_parquet_file = payload.get("latest_parquet_file") or "unknown"

    if age_minutes > max_age_minutes:
        return HealthStatus(
            level="ALERT",
            exit_code=1,
            message=(
                f"ALERT: rolling lake sync is stale. Last successful sync was {age_minutes:.1f} minutes ago "
                f"at {last_success.isoformat()} | latest_parquet_file={latest_parquet_file}"
            ),
        )

    return HealthStatus(
        level="OK",
        exit_code=0,
        message=(
            f"OK: rolling lake sync is healthy. Last successful sync was {age_minutes:.1f} minutes ago "
            f"at {last_success.isoformat()} | latest_parquet_file={latest_parquet_file}"
        ),
    )


def main() -> int:
    args = _parse_args()
    try:
        payload = load_sync_state(args.sync_state)
        status = evaluate_sync_state(payload, max_age_minutes=float(args.max_age_minutes))
    except Exception as exc:
        print(str(exc))
        return 2

    print(status.message)
    return status.exit_code


if __name__ == "__main__":
    raise SystemExit(main())