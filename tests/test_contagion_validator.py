from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from src.data.archive_market_analyzer import build_yes_price_series, median_absolute_move_over_window
from src.signals.microstructure_utils import CausalLagConfig
from src.tools import contagion_validator as validator_module
from src.tools.contagion_validator import (
    ContagionValidationReport,
    ContagionValidator,
    ContagionValidatorConfig,
    build_freshness_sweep,
    recommend_freshness_setting,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _snapshot_record(*, asset_id: str, timestamp: float, bid: float, ask: float, ask_size: float = 100.0) -> dict[str, object]:
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
            "asks": [{"price": f"{ask:.3f}", "size": f"{ask_size:.1f}"}],
        },
    }


def _trade_record(*, asset_id: str, timestamp: float, price: float) -> dict[str, object]:
    return {
        "local_ts": timestamp,
        "source": "trade",
        "asset_id": asset_id,
        "payload": {
            "event_type": "last_trade_price",
            "asset_id": asset_id,
            "timestamp": timestamp,
            "price": price,
            "size": 25.0,
            "side": "BUY",
        },
    }


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _market_config_payload() -> list[dict[str, object]]:
    return [
        {
            "market_id": "MKT_LEADER",
            "yes_id": "YES_LEADER",
            "no_id": "NO_LEADER",
            "question": "Leader market",
            "event_id": "event-a",
            "tags": ["archive", "leader"],
        },
        {
            "market_id": "MKT_LAGGER",
            "yes_id": "YES_LAGGER",
            "no_id": "NO_LAGGER",
            "question": "Lagger market",
            "event_id": "event-a",
            "tags": ["archive", "lagger"],
        },
    ]


def _build_validator_fixture(tmp_path: Path) -> tuple[ContagionValidatorConfig, Path]:
    archive_path = tmp_path / "archive"
    raw_day_dir = archive_path / "raw_ticks" / "2026-03-03"
    market_map_path = tmp_path / "universe.json"
    _write_json(market_map_path, _market_config_payload())

    leader_yes_records: list[dict[str, object]] = []
    leader_no_records: list[dict[str, object]] = []
    lagger_yes_records: list[dict[str, object]] = []
    lagger_no_records: list[dict[str, object]] = []

    for index in range(6):
        ts = float(index * 10)
        leader_yes_records.append(_snapshot_record(asset_id="YES_LEADER", timestamp=ts, bid=0.49 + index * 0.001, ask=0.51 + index * 0.001))
        leader_no_records.append(_snapshot_record(asset_id="NO_LEADER", timestamp=ts, bid=0.49, ask=0.51))
        leader_yes_records.append(_trade_record(asset_id="YES_LEADER", timestamp=ts + 0.5, price=0.50 + index * 0.001))
        lagger_yes_records.append(_snapshot_record(asset_id="YES_LAGGER", timestamp=ts, bid=0.48 + index * 0.001, ask=0.50 + index * 0.001))
        lagger_no_records.append(_snapshot_record(asset_id="NO_LAGGER", timestamp=ts, bid=0.49, ask=0.51))
        lagger_yes_records.append(_trade_record(asset_id="YES_LAGGER", timestamp=ts + 0.5, price=0.49 + index * 0.001))

    leader_yes_records.append(_snapshot_record(asset_id="YES_LEADER", timestamp=70.0, bid=0.62, ask=0.64, ask_size=5.0))
    leader_no_records.append(_snapshot_record(asset_id="NO_LEADER", timestamp=70.0, bid=0.49, ask=0.51))
    leader_yes_records.append(_trade_record(asset_id="YES_LEADER", timestamp=70.5, price=0.63))
    lagger_yes_records[-2] = _snapshot_record(asset_id="YES_LAGGER", timestamp=50.0, bid=0.50, ask=0.52)
    lagger_no_records[-1] = _snapshot_record(asset_id="NO_LAGGER", timestamp=50.0, bid=0.49, ask=0.51)
    lagger_yes_records[-1] = _trade_record(asset_id="YES_LAGGER", timestamp=50.5, price=0.51)

    _write_records(raw_day_dir / "YES_LEADER.jsonl", leader_yes_records)
    _write_records(raw_day_dir / "NO_LEADER.jsonl", leader_no_records)
    _write_records(raw_day_dir / "YES_LAGGER.jsonl", lagger_yes_records)
    _write_records(raw_day_dir / "NO_LAGGER.jsonl", lagger_no_records)

    config = ContagionValidatorConfig(
        archive_path=str(archive_path),
        universe_path=str(market_map_path),
        max_events=None,
        emit_per_event_telemetry=False,
    )
    return config, market_map_path


@pytest.fixture
def easy_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        validator_module,
        "_load_default_champion_params",
        lambda: {
            "contagion_arb_min_correlation": 0.0,
            "contagion_arb_trigger_percentile": 0.5,
            "contagion_arb_min_history": 3,
            "contagion_arb_min_leader_shift": 0.005,
            "contagion_arb_min_residual_shift": 0.001,
            "contagion_arb_toxicity_impulse_scale": 0.2,
            "contagion_arb_cooldown_seconds": 0.0,
            "contagion_arb_max_pairs_per_leader": 2,
            "contagion_arb_max_lagging_spread_pct": 10.0,
            "contagion_arb_max_last_trade_age_s": 300.0,
            "kelly_fraction": 0.1,
            "max_trade_size_usd": 5.0,
        },
    )


def test_validator_constructs_with_valid_config(tmp_path: Path) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    validator = ContagionValidator(config)
    assert validator.config == config


def test_validator_rejects_missing_archive_path(tmp_path: Path) -> None:
    market_map_path = tmp_path / "universe.json"
    _write_json(market_map_path, _market_config_payload())
    config = ContagionValidatorConfig(
        archive_path=str(tmp_path / "missing"),
        universe_path=str(market_map_path),
        max_events=None,
        emit_per_event_telemetry=False,
    )
    with pytest.raises(ValueError, match="archive_path"):
        ContagionValidator(config)


def test_validator_run_writes_report_and_uses_config(tmp_path: Path, easy_params: None) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    validator = ContagionValidator(config)
    output_path = tmp_path / "validation.json"
    report = validator.run(
        replay_date="2026-03-03",
        causal_lag_config=CausalLagConfig(5000.0, 30000.0, 600000.0, False),
        output_path=str(output_path),
    )

    assert isinstance(report, ContagionValidationReport)
    assert report.replay_date == "2026-03-03"
    assert report.causal_lag_config.max_lagger_age_ms == 30000.0
    assert report.cross_market_pairs_evaluated > 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["report"]["replay_date"] == "2026-03-03"
    assert "diagnostics" in payload


def test_validator_run_emits_telemetry_when_enabled(tmp_path: Path, easy_params: None) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    config = ContagionValidatorConfig(**{**asdict(config), "emit_per_event_telemetry": True})
    validator = ContagionValidator(config)
    output_path = tmp_path / "validation_with_telemetry.json"
    validator.run(
        replay_date="2026-03-03",
        causal_lag_config=CausalLagConfig(5000.0, 30000.0, 600000.0, False),
        output_path=str(output_path),
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "per_event_telemetry" in payload
    assert isinstance(payload["per_event_telemetry"], list)


def test_validator_run_sweep_returns_expected_shape(tmp_path: Path, easy_params: None) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    validator = ContagionValidator(config)
    output_path = tmp_path / "sweep.json"
    reports = validator.run_sweep(
        replay_date="2026-03-03",
        sweep_param="max_lagger_age_ms",
        sweep_values=[10000, 30000],
        base_config=CausalLagConfig(5000.0, 30000.0, 600000.0, False),
        output_path=str(output_path),
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(reports) == 2
    assert len(payload) == 2
    assert payload[0]["report"]["causal_lag_config"]["max_lagger_age_ms"] == 10000.0
    assert payload[1]["report"]["causal_lag_config"]["max_lagger_age_ms"] == 30000.0


def test_validator_run_sweep_rejects_invalid_param(tmp_path: Path, easy_params: None) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    validator = ContagionValidator(config)
    with pytest.raises(ValueError, match="Unsupported sweep_param"):
        validator.run_sweep(
            replay_date="2026-03-03",
            sweep_param="not_real",  # type: ignore[arg-type]
            sweep_values=[1],
            base_config=CausalLagConfig(5000.0, 30000.0, 600000.0, False),
            output_path=str(tmp_path / "ignored.json"),
        )


def test_validator_report_exposes_lagger_age_distribution(tmp_path: Path, easy_params: None) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    validator = ContagionValidator(config)
    report = validator.run(
        replay_date="2026-03-03",
        causal_lag_config=CausalLagConfig(5000.0, 30000.0, 600000.0, False),
        output_path=str(tmp_path / "validation.json"),
    )
    assert report.median_lagger_age_ms >= 0.0
    assert report.p95_lagger_age_ms >= report.median_lagger_age_ms


def test_validator_cli_main_writes_output(tmp_path: Path, easy_params: None) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    output_path = tmp_path / "cli_output.json"
    exit_code = validator_module.main(
        [
            "--date",
            "2026-03-03",
            "--archive-path",
            config.archive_path,
            "--universe-path",
            config.universe_path,
            "--output",
            str(output_path),
            "--max-lagger-age",
            "30000",
        ]
    )
    assert exit_code == 0
    assert output_path.exists()


def test_build_freshness_sweep_produces_recommendation(tmp_path: Path, easy_params: None) -> None:
    config, _ = _build_validator_fixture(tmp_path)
    validator = ContagionValidator(config)
    payload = build_freshness_sweep(
        validator=validator,
        replay_date="2026-03-03",
        sweep_values=[10000, 30000],
        base_config=CausalLagConfig(5000.0, 30000.0, 600000.0, False),
    )
    assert len(payload["sweep"]) == 2
    assert "suggested_max_lagger_age_ms" in payload["recommendation"]


def test_recommendation_prefers_low_risk_knee() -> None:
    recommendation = recommend_freshness_setting(
        [
            {"max_lagger_age_ms": 30000, "pairs_passing": 10, "signals_fired": 0, "median_price_move_in_window": 0.001},
            {"max_lagger_age_ms": 60000, "pairs_passing": 12, "signals_fired": 0, "median_price_move_in_window": 0.002},
            {"max_lagger_age_ms": 120000, "pairs_passing": 13, "signals_fired": 0, "median_price_move_in_window": 0.006},
        ]
    )
    assert recommendation["suggested_max_lagger_age_ms"] == 60000


def test_archive_series_supports_median_move_calculation(tmp_path: Path) -> None:
    config, market_map_path = _build_validator_fixture(tmp_path)
    market_configs = validator_module.load_universe_market_configs(str(market_map_path))
    series = build_yes_price_series(config.archive_path, market_configs, ["2026-03-03"])
    median_move = median_absolute_move_over_window(series, 30000)
    assert series["MKT_LEADER"].event_count > 0
    assert median_move >= 0.0
