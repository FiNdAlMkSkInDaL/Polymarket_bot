from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import refresh_shadow_metadata_cache as refresh_script


def _strict_row(
    *,
    timestamp: datetime,
    event_id: str,
    market_id: str,
    token_id: str,
    best_bid: float,
    best_ask: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "event_id": event_id,
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": 100.0,
        "ask_depth": 100.0,
    }


def _write_today_fixture(tmp_path: Path) -> Path:
    hour_dir = tmp_path / "data" / "l2_book_live" / "date=2026-04-06" / "hour=10"
    hour_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            _strict_row(
                timestamp=datetime(2026, 4, 6, 10, 0, tzinfo=UTC),
                event_id="evt-a",
                market_id="mkt-a",
                token_id="YES",
                best_bid=0.91,
                best_ask=0.92,
            ),
            _strict_row(
                timestamp=datetime(2026, 4, 6, 10, 1, tzinfo=UTC),
                event_id="evt-b",
                market_id="mkt-b",
                token_id="YES",
                best_bid=0.12,
                best_ask=0.13,
            ),
        ]
    ).write_parquet(hour_dir / "part-000.parquet")
    return tmp_path / "data" / "l2_book_live"


def _market_row(condition_id: str, event_id: str, token_yes: str, token_no: str, end_date: str) -> dict[str, object]:
    return {
        "conditionId": condition_id,
        "clobTokenIds": [token_yes, token_no],
        "description": f"Description for {condition_id}",
        "endDate": end_date,
        "eventId": event_id,
        "events": [
            {
                "description": f"Event description for {event_id}",
                "endDate": end_date,
                "id": event_id,
                "slug": f"slug-{event_id}",
                "title": f"Event {event_id}",
            }
        ],
        "question": f"Question for {condition_id}",
    }


def test_refresh_shadow_metadata_cache_writes_full_cache_and_summary(tmp_path: Path) -> None:
    lake_root = _write_today_fixture(tmp_path)
    cache_path = tmp_path / "artifacts" / "clob_arb_baseline_metadata.json"
    summary_path = tmp_path / "artifacts" / "shadow_metadata_preflight_summary.json"

    def fake_market_fetcher(condition_ids: list[str]) -> refresh_script.MarketFetchResult:
        assert condition_ids == ["mkt-a", "mkt-b"]
        return refresh_script.MarketFetchResult(
            rows_by_condition_id={
                "mkt-a": _market_row("mkt-a", "evt-a", "tok-a-yes", "tok-a-no", "2026-04-08T12:00:00Z"),
                "mkt-b": _market_row("mkt-b", "evt-b", "tok-b-yes", "tok-b-no", "2026-04-09T12:00:00Z"),
            },
            failed_condition_ids=[],
        )

    def fake_event_fetcher(event_ids: list[str]) -> refresh_script.EventFetchResult:
        assert event_ids == ["evt-a", "evt-b"]
        return refresh_script.EventFetchResult(
            rows_by_event_id={
                "evt-a": {"id": "evt-a", "title": "Event evt-a", "type": "sports"},
                "evt-b": {"id": "evt-b", "title": "Event evt-b", "type": "politics"},
            },
            failed_event_ids=[],
        )

    summary = refresh_script.refresh_shadow_metadata_cache(
        lake_root,
        cache_path=cache_path,
        summary_path=summary_path,
        since_date=date(2026, 4, 6),
        coverage_threshold_pct=95.0,
        gamma_batch_size=20,
        timeout_seconds=30.0,
        max_retries=4,
        retry_backoff_seconds=1.0,
        market_fetcher=fake_market_fetcher,
        event_fetcher=fake_event_fetcher,
    )

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    summary_from_disk = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["refresh_status"] == "success"
    assert summary["active_market_count"] == 2
    assert summary["covered_market_count"] == 2
    assert summary["coverage_alarm"] is False
    assert payload["active_market_count"] == 2
    assert len(payload["markets_by_token"]) == 4
    assert payload["events_by_id"]["evt-a"]["type"] == "sports"
    assert summary_from_disk["coverage_pct"] == 100.0


def test_refresh_shadow_metadata_cache_falls_back_and_alarms_on_low_coverage(tmp_path: Path) -> None:
    lake_root = _write_today_fixture(tmp_path)
    cache_path = tmp_path / "artifacts" / "clob_arb_baseline_metadata.json"
    summary_path = tmp_path / "artifacts" / "shadow_metadata_preflight_summary.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "events_by_id": {
                    "evt-a": {"id": "evt-a", "title": "Old event a"},
                },
                "markets_by_token": {
                    "tok-a-yes": _market_row("mkt-a", "evt-a", "tok-a-yes", "tok-a-no", "2026-04-08T12:00:00Z"),
                    "tok-a-no": _market_row("mkt-a", "evt-a", "tok-a-yes", "tok-a-no", "2026-04-08T12:00:00Z"),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def fake_market_fetcher(condition_ids: list[str]) -> refresh_script.MarketFetchResult:
        assert condition_ids == ["mkt-a", "mkt-b"]
        return refresh_script.MarketFetchResult(
            rows_by_condition_id={},
            failed_condition_ids=["mkt-a", "mkt-b"],
        )

    def fake_event_fetcher(event_ids: list[str]) -> refresh_script.EventFetchResult:
        assert event_ids == ["evt-a"]
        return refresh_script.EventFetchResult(rows_by_event_id={}, failed_event_ids=["evt-a"])

    summary = refresh_script.refresh_shadow_metadata_cache(
        lake_root,
        cache_path=cache_path,
        summary_path=summary_path,
        since_date=date(2026, 4, 6),
        coverage_threshold_pct=95.0,
        gamma_batch_size=20,
        timeout_seconds=30.0,
        max_retries=4,
        retry_backoff_seconds=1.0,
        market_fetcher=fake_market_fetcher,
        event_fetcher=fake_event_fetcher,
    )

    payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert summary["refresh_status"] == "degraded"
    assert summary["covered_market_count"] == 1
    assert summary["missing_market_count"] == 1
    assert summary["missing_market_ids_sample"] == ["mkt-b"]
    assert summary["coverage_alarm"] is True
    assert len(payload["markets_by_token"]) == 2
    assert payload["active_market_count"] == 2