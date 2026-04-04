from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.monitor_lake_health import evaluate_sync_state, load_sync_state


def test_load_sync_state_returns_json_payload(tmp_path: Path) -> None:
    sync_state_path = tmp_path / "sync_state.json"
    sync_state_path.write_text(json.dumps({"schema": "rolling_lake_sync_state_v1"}), encoding="utf-8")

    payload = load_sync_state(sync_state_path)

    assert payload["schema"] == "rolling_lake_sync_state_v1"


def test_evaluate_sync_state_reports_ok_for_recent_success() -> None:
    recent = datetime.now(tz=UTC) - timedelta(minutes=12)
    payload = {
        "last_successful_sync_at": recent.isoformat(),
        "latest_parquet_file": "l2_book/date=2026-04-04/hour=10/l2_book_2026-04-04_10_000001.parquet",
    }

    status = evaluate_sync_state(payload, now=datetime.now(tz=UTC), max_age_minutes=65.0)

    assert status.exit_code == 0
    assert status.level == "OK"
    assert "12." in status.message or "11." in status.message or "13." in status.message


def test_evaluate_sync_state_reports_alert_for_stale_success() -> None:
    stale = datetime(2026, 4, 4, 9, 0, tzinfo=UTC)
    now = stale + timedelta(minutes=70)
    payload = {
        "last_successful_sync_at": stale.isoformat(),
        "latest_parquet_file": "l2_book/date=2026-04-04/hour=09/l2_book_2026-04-04_09_000014.parquet",
    }

    status = evaluate_sync_state(payload, now=now, max_age_minutes=65.0)

    assert status.exit_code == 1
    assert status.level == "ALERT"
    assert "70.0 minutes ago" in status.message


def test_evaluate_sync_state_falls_back_to_generated_at() -> None:
    generated = datetime(2026, 4, 4, 10, 0, tzinfo=UTC)
    now = generated + timedelta(minutes=20)
    payload = {
        "generated_at": generated.isoformat(),
        "latest_parquet_file": "l2_book/date=2026-04-04/hour=10/l2_book_2026-04-04_10_000001.parquet",
    }

    status = evaluate_sync_state(payload, now=now, max_age_minutes=65.0)

    assert status.exit_code == 0
    assert status.level == "OK"
    assert "20.0 minutes ago" in status.message