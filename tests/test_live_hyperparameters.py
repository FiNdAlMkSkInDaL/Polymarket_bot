from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.live_hyperparameters import (
    LiveHyperparameterValidationError,
    extract_params_payload,
    validate_strategy_param_overrides,
)


def test_validate_strategy_param_overrides_accepts_valid_wfo_knobs() -> None:
    validated = validate_strategy_param_overrides(
        {
            "ofi_tvi_kappa": 1.4,
            "contagion_arb_min_correlation": 0.42,
            "contagion_arb_trigger_percentile": 0.97,
            "si10_min_net_edge_usd": 0.15,
            "si9_latency_option_window_ms": 4500,
            "max_cross_book_desync_ms": 550,
        }
    )

    assert validated["ofi_tvi_kappa"] == pytest.approx(1.4)
    assert validated["contagion_arb_min_correlation"] == pytest.approx(0.42)
    assert validated["contagion_arb_trigger_percentile"] == pytest.approx(0.97)
    assert validated["si10_min_net_edge_usd"] == pytest.approx(0.15)
    assert validated["si9_latency_option_window_ms"] == 4500
    assert validated["max_cross_book_desync_ms"] == 550


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"ofi_tvi_kappa": None}, "cannot be None"),
        ({"ofi_tvi_kappa": float("nan")}, "cannot be NaN or infinite"),
        ({"ofi_tvi_kappa": float("inf")}, "cannot be NaN or infinite"),
        ({"si10_min_net_edge_usd": 0.0}, "strictly greater than 0"),
        ({"max_cross_book_desync_ms": 0.0}, "strictly greater than 0"),
        ({"contagion_arb_trigger_percentile": 1.2}, "between 0.0 and 1.0"),
        ({"contagion_arb_min_correlation": -0.1}, "between 0.0 and 1.0"),
    ],
)
def test_validate_strategy_param_overrides_rejects_invalid_values(
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(LiveHyperparameterValidationError, match=message):
        validate_strategy_param_overrides(params)


def test_extract_params_payload_handles_wfo_export_shape() -> None:
    payload = extract_params_payload(
        {
            "params": {"ofi_tvi_kappa": 1.8},
            "meta": {"champion_fold": 1},
        }
    )

    assert payload == {"ofi_tvi_kappa": 1.8}


def test_config_bootstrap_loads_live_hyperparameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    live_path = tmp_path / "live_hyperparameters.json"
    live_path.write_text(
        json.dumps(
            {
                "params": {
                    "ofi_tvi_kappa": 1.75,
                    "si10_min_net_edge_usd": 0.22,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("LIVE_HYPERPARAMETERS_PATH", str(live_path))

    import src.core.config as config_module

    reloaded = importlib.reload(config_module)
    try:
        assert reloaded.settings.strategy.ofi_tvi_kappa == pytest.approx(1.75)
        assert reloaded.settings.strategy.si10_min_net_edge_usd == pytest.approx(0.22)
    finally:
        monkeypatch.delenv("LIVE_HYPERPARAMETERS_PATH", raising=False)
        importlib.reload(config_module)


def test_inject_wfo_champions_merges_multiple_outputs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "live_hyperparameters.json"

    ofi_dir = tmp_path / "ofi_run"
    ofi_dir.mkdir()
    (ofi_dir / "champion_params.json").write_text(
        json.dumps({"params": {"ofi_tvi_kappa": 1.3}, "meta": {"champion_fold": 0}}),
        encoding="utf-8",
    )

    arb_dir = tmp_path / "arb_run"
    arb_dir.mkdir()
    (arb_dir / "champion_params.json").write_text(
        json.dumps(
            {
                "params": {
                    "contagion_arb_min_correlation": 0.44,
                    "si10_min_net_edge_usd": 0.19,
                },
                "meta": {"champion_fold": 2},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "inject_wfo_champions.py"),
            str(ofi_dir),
            str(arb_dir),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
    )

    assert completed.returncode == 0, completed.stderr

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["params"]["ofi_tvi_kappa"] == pytest.approx(1.3)
    assert written["params"]["contagion_arb_min_correlation"] == pytest.approx(0.44)
    assert written["params"]["si10_min_net_edge_usd"] == pytest.approx(0.19)
    assert written["meta"]["source_count"] == 2


def test_inject_wfo_champions_refuses_invalid_output(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "live_hyperparameters.json"
    output_path.write_text(
        json.dumps({"params": {"ofi_tvi_kappa": 1.1}}),
        encoding="utf-8",
    )

    bad_dir = tmp_path / "bad_run"
    bad_dir.mkdir()
    (bad_dir / "champion_params.json").write_text(
        json.dumps({"params": {"si10_min_net_edge_usd": 0.0}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "inject_wfo_champions.py"),
            str(bad_dir),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
    )

    assert completed.returncode == 2
    assert "CRITICAL" in completed.stderr
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["params"]["ofi_tvi_kappa"] == pytest.approx(1.1)
    assert "si10_min_net_edge_usd" not in written["params"]