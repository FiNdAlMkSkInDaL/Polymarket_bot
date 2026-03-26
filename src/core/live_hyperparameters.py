from __future__ import annotations

import json
import math
import os
import re
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_type_hints


class LiveHyperparameterValidationError(ValueError):
    """Raised when champion or live hyperparameter payloads are invalid."""


def default_live_hyperparameters_path() -> Path:
    configured = os.getenv("LIVE_HYPERPARAMETERS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "live_hyperparameters.json"


def extract_params_payload(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise LiveHyperparameterValidationError("hyperparameter payload must be a JSON object")

    params = raw_payload.get("params", raw_payload)
    if not isinstance(params, dict):
        raise LiveHyperparameterValidationError("hyperparameter params must be a JSON object")
    return dict(params)


def validate_strategy_param_overrides(raw_params: dict[str, Any]) -> dict[str, Any]:
    from src.core.config import StrategyParams

    param_fields = {field.name: field for field in fields(StrategyParams)}
    type_hints = get_type_hints(StrategyParams)
    validated: dict[str, Any] = {}

    for name, raw_value in raw_params.items():
        if name not in param_fields:
            raise LiveHyperparameterValidationError(f"unknown strategy parameter: {name}")
        validated[name] = _coerce_and_validate_value(
            name=name,
            raw_value=raw_value,
            expected_type=type_hints.get(name, Any),
        )

    return validated


def load_live_hyperparameters(path: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve() if path is not None else default_live_hyperparameters_path()
    if not resolved.exists():
        return {}

    raw_payload = json.loads(resolved.read_text(encoding="utf-8"))
    return validate_strategy_param_overrides(extract_params_payload(raw_payload))


def apply_live_hyperparameter_overrides(strategy_params: Any, path: str | Path | None = None):
    overrides = load_live_hyperparameters(path)
    if not overrides:
        return strategy_params
    return replace(strategy_params, **overrides)


def resolve_champion_params_path(path_str: str | Path) -> Path:
    candidate = Path(path_str).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "champion_params.json"
    if not candidate.exists():
        raise LiveHyperparameterValidationError(f"champion params file not found: {candidate}")
    return candidate


def build_live_hyperparameters_payload(
    *,
    existing_params: dict[str, Any],
    champion_sources: list[tuple[Path, dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    merged_params = dict(existing_params)
    sources_meta: list[dict[str, Any]] = []

    for source_path, params, meta in champion_sources:
        merged_params.update(params)
        sources_meta.append(
            {
                "path": str(source_path),
                "param_count": len(params),
                "params": sorted(params.keys()),
                "meta": meta,
            }
        )

    return {
        "params": merged_params,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_count": len(champion_sources),
            "sources": sources_meta,
        },
    }


def _coerce_and_validate_value(*, name: str, raw_value: Any, expected_type: Any) -> Any:
    if raw_value is None:
        raise LiveHyperparameterValidationError(f"{name} cannot be None")

    origin = expected_type
    if origin is bool:
        if not isinstance(raw_value, bool):
            raise LiveHyperparameterValidationError(f"{name} must be a boolean")
        return raw_value

    if origin is int:
        if isinstance(raw_value, bool):
            raise LiveHyperparameterValidationError(f"{name} must be an integer, not boolean")
        if isinstance(raw_value, float):
            if not math.isfinite(raw_value):
                raise LiveHyperparameterValidationError(f"{name} cannot be NaN or infinite")
            if not raw_value.is_integer():
                raise LiveHyperparameterValidationError(f"{name} must be an integer")
            value = int(raw_value)
        elif isinstance(raw_value, int):
            value = raw_value
        else:
            raise LiveHyperparameterValidationError(f"{name} must be an integer")
        _validate_numeric_constraints(name, float(value))
        return value

    if origin is float:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise LiveHyperparameterValidationError(f"{name} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise LiveHyperparameterValidationError(f"{name} cannot be NaN or infinite")
        _validate_numeric_constraints(name, value)
        return value

    if origin is str:
        if not isinstance(raw_value, str):
            raise LiveHyperparameterValidationError(f"{name} must be a string")
        return raw_value

    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if not math.isfinite(value):
            raise LiveHyperparameterValidationError(f"{name} cannot be NaN or infinite")
        _validate_numeric_constraints(name, value)
    return raw_value


def _validate_numeric_constraints(name: str, value: float) -> None:
    if _is_positive_edge_threshold(name) and value <= 0.0:
        raise LiveHyperparameterValidationError(f"{name} must be strictly greater than 0")
    if name == "max_cross_book_desync_ms" and value <= 0.0:
        raise LiveHyperparameterValidationError(f"{name} must be strictly greater than 0")
    if name in {
        "contagion_arb_max_leader_age_ms",
        "contagion_arb_max_lagger_age_ms",
        "contagion_arb_max_causal_lag_ms",
    } and value <= 0.0:
        raise LiveHyperparameterValidationError(f"{name} must be strictly greater than 0")
    if _is_zero_one_bounded(name) and not (0.0 <= value <= 1.0):
        raise LiveHyperparameterValidationError(f"{name} must be between 0.0 and 1.0")


def _is_positive_edge_threshold(name: str) -> bool:
    return bool(re.search(r"edge.*_(usd|cents)$", name))


def _is_zero_one_bounded(name: str) -> bool:
    if "percentile" in name:
        return True
    return name.endswith("_corr") or name.endswith("_correlation") or name.endswith("_min_correlation")