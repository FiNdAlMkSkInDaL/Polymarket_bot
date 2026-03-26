from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

import pytest

from src.data import universe_builder as universe_module
from src.data.universe_builder import (
    ClusterEvaluationReport,
    MarketCandidate,
    UniverseBuilder,
    UniverseBuilderConfig,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _snapshot_record(*, asset_id: str, timestamp: float, bid: float, ask: float) -> dict[str, object]:
    return {
        "local_ts": timestamp,
        "source": "l2",
        "asset_id": asset_id,
        "payload": {
            "event_type": "l2_snapshot",
            "asset_id": asset_id,
            "market": f"market-{asset_id}",
            "timestamp": timestamp,
            "bids": [{"price": f"{bid:.3f}", "size": "100"}],
            "asks": [{"price": f"{ask:.3f}", "size": "100"}],
        },
    }


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _market_map_payload() -> list[dict[str, object]]:
    return [
        {"market_id": "MKT_A", "yes_id": "YES_A", "no_id": "NO_A", "question": "A", "event_id": "evt", "tags": ["macro", "election"]},
        {"market_id": "MKT_B", "yes_id": "YES_B", "no_id": "NO_B", "question": "B", "event_id": "evt", "tags": ["macro", "election"]},
        {"market_id": "MKT_C", "yes_id": "YES_C", "no_id": "NO_C", "question": "C", "event_id": "evt", "tags": ["sports"]},
        {"market_id": "MKT_D", "yes_id": "YES_D", "no_id": "NO_D", "question": "D", "event_id": "evt", "tags": ["macro", "election"]},
        {"market_id": "MKT_EMPTY", "yes_id": "YES_EMPTY", "no_id": "NO_EMPTY", "question": "Empty", "event_id": "evt", "tags": ["macro"]},
    ]


def _build_archive(tmp_path: Path) -> tuple[Path, Path]:
    archive_path = tmp_path / "archive"
    market_map_path = tmp_path / "market_map.json"
    _write_json(market_map_path, _market_map_payload())

    day1 = archive_path / "raw_ticks" / "2026-03-01"
    day2 = archive_path / "raw_ticks" / "2026-03-02"
    market_patterns = {
        "YES_A": [0.50, 0.52, 0.54, 0.56, 0.58],
        "YES_B": [0.49, 0.50, 0.52, 0.54, 0.56],
        "YES_C": [0.70, 0.66, 0.69, 0.65, 0.68],
        "YES_D": [0.53, 0.55, 0.57, 0.59, 0.61],
    }
    market_times = {
        "YES_A": [0.0, 10.0, 20.0, 30.0, 40.0],
        "YES_B": [8.0, 18.0, 28.0, 38.0, 48.0],
        "YES_C": [0.0, 10.0, 20.0, 30.0, 40.0],
        "YES_D": [0.0, 8.0, 16.0, 24.0, 32.0],
    }

    for asset_id, prices in market_patterns.items():
        day1_records = []
        day2_records = []
        for index, price in enumerate(prices):
            ts1 = market_times[asset_id][index]
            ts2 = 86400.0 + market_times[asset_id][index]
            day1_records.append(_snapshot_record(asset_id=asset_id, timestamp=ts1, bid=price - 0.01, ask=price + 0.01))
            day2_records.append(_snapshot_record(asset_id=asset_id, timestamp=ts2, bid=price - 0.01, ask=price + 0.01))
        _write_records(day1 / f"{asset_id}.jsonl", day1_records)
        _write_records(day2 / f"{asset_id}.jsonl", day2_records)

    _write_records(day1 / "YES_SPARSE.jsonl", [_snapshot_record(asset_id="YES_SPARSE", timestamp=5.0, bid=0.40, ask=0.42)])
    return archive_path, market_map_path


def _candidate_set() -> list[MarketCandidate]:
    return [
        MarketCandidate("MKT_A", "A", frozenset({"macro", "election"}), "LEADER"),
        MarketCandidate("MKT_B", "B", frozenset({"macro", "election"}), "LAGGER"),
        MarketCandidate("MKT_C", "C", frozenset({"sports"}), "EITHER"),
        MarketCandidate("MKT_D", "D", frozenset({"macro", "election"}), "EITHER"),
        MarketCandidate("MKT_EMPTY", "Empty", frozenset({"macro"}), "EITHER"),
    ]


def _builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config_overrides: object) -> UniverseBuilder:
    archive_path, market_map_path = _build_archive(tmp_path)
    monkeypatch.setattr(universe_module, "DEFAULT_ARCHIVE_PATH", archive_path)
    monkeypatch.setattr(universe_module, "DEFAULT_MARKET_MAP_PATHS", (market_map_path,))
    config = UniverseBuilderConfig(
        min_correlation=Decimal("0.1"),
        min_events_per_day=3,
        min_archive_days=1,
        max_lagger_age_ms=15000,
        require_causal_ordering=False,
    )
    config = UniverseBuilderConfig(**{**asdict(config), **config_overrides})
    return UniverseBuilder(config)


def test_universe_builder_constructs_with_valid_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch)
    assert isinstance(builder.config.min_correlation, Decimal)


def test_universe_builder_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="min_correlation"):
        UniverseBuilder(
            UniverseBuilderConfig(
                min_correlation=Decimal("1.2"),
                min_events_per_day=1,
                min_archive_days=1,
                max_lagger_age_ms=1000,
                require_causal_ordering=False,
            )
        )


def test_build_cluster_recommends_leader_and_lagger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch)
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert isinstance(report, ClusterEvaluationReport)
    assert report.leader_market_id == "MKT_A"
    assert "MKT_B" in report.recommended_cluster


def test_build_cluster_filters_low_correlation_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch, min_correlation=Decimal("1.0"))
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert report.pairs_passing_correlation == 0
    assert report.leader_market_id is None


def test_build_cluster_filters_stale_laggers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch, max_lagger_age_ms=5000)
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert report.pairs_passing_correlation > report.pairs_passing_freshness
    assert report.rejection_reasons["MKT_B"] == "lagger_freshness_too_old"


def test_build_cluster_applies_causal_ordering_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch, require_causal_ordering=True)
    candidates = [
        MarketCandidate("MKT_A", "A", frozenset({"macro", "election"}), "LEADER"),
        MarketCandidate("MKT_D", "D", frozenset({"macro", "election"}), "LAGGER"),
    ]
    report = builder.build_cluster(candidates, "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert report.pairs_passing_freshness >= report.pairs_passing_causal_ordering
    assert report.rejection_reasons["MKT_D"] == "causal_ordering_reversed"


def test_build_cluster_reports_missing_archive_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch)
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert report.rejection_reasons["MKT_EMPTY"] == "no_archive_data"


def test_build_cluster_reports_empirical_correlations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch)
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert "MKT_A->MKT_B" in report.empirical_correlations
    assert report.empirical_correlations["MKT_A->MKT_B"] >= 0.0


def test_build_cluster_respects_expected_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch)
    candidates = [
        MarketCandidate("MKT_A", "A", frozenset({"macro"}), "LAGGER"),
        MarketCandidate("MKT_B", "B", frozenset({"macro"}), "LEADER"),
    ]
    report = builder.build_cluster(candidates, "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert report.leader_market_id == "MKT_B"


def test_build_cluster_rejects_insufficient_archive_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch, min_archive_days=3)
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert report.rejection_reasons["MKT_A"] == "insufficient_archive_days"


def test_build_cluster_rejects_insufficient_events_per_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch, min_events_per_day=6)
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    assert report.rejection_reasons["MKT_A"] == "insufficient_events_per_day"


def test_export_cluster_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder(tmp_path, monkeypatch)
    report = builder.build_cluster(_candidate_set(), "2026-03-01T00:00:00Z", "2026-03-02T00:00:59Z")
    output_path = tmp_path / "cluster.json"
    builder.export_cluster(report, str(output_path))
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["generated_at"] == report.generated_at
    assert payload["recommended_cluster"] == report.recommended_cluster
