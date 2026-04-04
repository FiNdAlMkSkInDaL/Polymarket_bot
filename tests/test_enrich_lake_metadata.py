from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from scripts.enrich_lake_metadata import GammaMarketRow, run_enrichment


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


def test_run_enrichment_writes_enriched_manifest_with_market_status_counts(tmp_path: Path) -> None:
    hour_dir = tmp_path / "l2_book" / "date=2026-04-03" / "hour=00"
    hour_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            _strict_row(
                timestamp=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
                event_id="evt-1",
                market_id="mkt-open",
                token_id="tok-open",
                best_bid=0.10,
                best_ask=0.11,
            ),
            _strict_row(
                timestamp=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
                event_id="evt-2",
                market_id="mkt-resolved",
                token_id="tok-resolved",
                best_bid=0.20,
                best_ask=0.21,
            ),
        ]
    ).write_parquet(hour_dir / "part-000.parquet")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"days": ["2026-04-03"], "strict_schema": {"best_bid": "Float64"}}, indent=2),
        encoding="utf-8",
    )

    def fake_fetcher(market_ids: list[str]) -> dict[str, GammaMarketRow]:
        assert sorted(market_ids) == ["mkt-open", "mkt-resolved"]
        return {
            "mkt-open": GammaMarketRow(
                market_id="mkt-open",
                event_id="evt-1",
                question="Open market",
                gamma_closed=False,
                gamma_market_status="open",
                resolution_timestamp=None,
                final_resolution_value=None,
            ),
            "mkt-resolved": GammaMarketRow(
                market_id="mkt-resolved",
                event_id="evt-2",
                question="Resolved market",
                gamma_closed=True,
                gamma_market_status="resolved",
                resolution_timestamp=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
                final_resolution_value=1.0,
            ),
        }

    payload = run_enrichment(tmp_path, fetcher=fake_fetcher)
    manifest = json.loads((tmp_path / "enriched_manifest.json").read_text(encoding="utf-8"))

    assert payload["market_count"] == 2
    assert manifest["resolved_market_count"] == 1
    assert manifest["open_market_count"] == 1
    assert manifest["closed_unresolved_market_count"] == 0
    assert manifest["missing_gamma_market_count"] == 0
    status_by_market = {row["market_id"]: row["gamma_market_status"] for row in manifest["markets"]}
    assert status_by_market == {"mkt-open": "open", "mkt-resolved": "resolved"}