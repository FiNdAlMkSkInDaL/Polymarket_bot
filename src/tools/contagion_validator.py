from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Literal

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.strategy import ContagionReplayAdapter
from src.data.archive_market_analyzer import (
    build_yes_price_series,
    load_universe_market_configs,
    median_absolute_move_over_window,
    parse_iso_date,
    percentile,
)
from src.signals.contagion_arb import ContagionArbDetector
from src.signals.microstructure_utils import CausalLagAssessment, CausalLagConfig
from src.core.config import StrategyParams
from src.backtest.wfo_optimizer import _build_data_loader


DEFAULT_ARCHIVE_PATH = Path("data/vps_march2026")
DEFAULT_UNIVERSE_PATH = Path("data/domino_micro_fast_market_map.json")
DEFAULT_VALIDATION_OUTPUT = Path("contagion_validation_output.json")
DEFAULT_CHAMPION_CANDIDATES = (
    Path("data/wfo_contagion_arb_micro_2026_03_25_microscale_fast/champion_params.json"),
    Path("data/wfo_contagion_arb_micro_fast_2026_03_25/champion_params.json"),
    Path("data/wfo_contagion_arb_micro_2026_03_25/champion_params.json"),
)


@dataclass(frozen=True, slots=True)
class ContagionValidatorConfig:
    archive_path: str
    universe_path: str
    max_events: int | None
    emit_per_event_telemetry: bool


@dataclass(frozen=True, slots=True)
class ContagionValidationReport:
    replay_date: str
    causal_lag_config: CausalLagConfig
    leader_events_evaluated: int
    events_reaching_spike_check: int
    cross_market_pairs_evaluated: int
    causal_gate_pass_rate: float
    legacy_sync_pass_rate: float
    signals_fired: int
    fills_executed: int
    dominant_suppressor: str
    suppression_breakdown: dict[str, int]
    median_lagger_age_ms: float
    p95_lagger_age_ms: float
    generated_at: str


class _InstrumentedContagionDetector(ContagionArbDetector):
    def __init__(self, *args: Any, emit_per_event_telemetry: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._emit_per_event_telemetry = emit_per_event_telemetry
        self._pair_telemetry: list[dict[str, Any]] = []
        self._lagger_ages_ms: list[float] = []

    @property
    def pair_telemetry(self) -> list[dict[str, Any]]:
        return list(self._pair_telemetry)

    @property
    def lagger_ages_ms(self) -> list[float]:
        return list(self._lagger_ages_ms)

    def _assess_causal_pair(
        self,
        leader_snapshot: Any,
        lag_snapshot: Any,
        *,
        reference_timestamp: float,
    ) -> CausalLagAssessment:
        assessment = super()._assess_causal_pair(
            leader_snapshot,
            lag_snapshot,
            reference_timestamp=reference_timestamp,
        )
        if math.isfinite(assessment.lagger_age_ms):
            self._lagger_ages_ms.append(float(assessment.lagger_age_ms))
        if self._emit_per_event_telemetry:
            self._pair_telemetry.append(
                {
                    "reference_timestamp": round(reference_timestamp, 6),
                    "leader_market_id": getattr(leader_snapshot, "market_id", ""),
                    "lagger_market_id": getattr(lag_snapshot, "market_id", ""),
                    "leader_age_ms": round(float(assessment.leader_age_ms), 3),
                    "lagger_age_ms": round(float(assessment.lagger_age_ms), 3),
                    "causal_lag_ms": round(float(assessment.causal_lag_ms), 3),
                    "gate_result": assessment.gate_result,
                }
            )
        return assessment


class _InstrumentedContagionReplayAdapter(ContagionReplayAdapter):
    def __init__(self, *args: Any, emit_per_event_telemetry: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._contagion = _InstrumentedContagionDetector(
            self._pce,
            self._rpe,
            universe_size=self._params.max_active_l2_markets,
            min_correlation=self._params.contagion_arb_min_correlation,
            trigger_percentile=self._params.contagion_arb_trigger_percentile,
            min_history=self._params.contagion_arb_min_history,
            min_leader_shift=self._params.contagion_arb_min_leader_shift,
            min_residual_shift=self._params.contagion_arb_min_residual_shift,
            toxicity_impulse_scale=self._params.contagion_arb_toxicity_impulse_scale,
            cooldown_seconds=self._params.contagion_arb_cooldown_seconds,
            max_pairs_per_leader=self._params.contagion_arb_max_pairs_per_leader,
            shadow_mode=False,
            max_cross_book_desync_ms=self._params.max_cross_book_desync_ms,
            max_leader_age_ms=self._params.contagion_arb_max_leader_age_ms,
            max_lagger_age_ms=self._params.contagion_arb_max_lagger_age_ms,
            max_causal_lag_ms=self._params.contagion_arb_max_causal_lag_ms,
            allow_negative_lag=self._params.contagion_arb_allow_negative_lag,
            emit_per_event_telemetry=emit_per_event_telemetry,
        )

    def pair_telemetry(self) -> list[dict[str, Any]]:
        return self._contagion.pair_telemetry

    def lagger_ages_ms(self) -> list[float]:
        return self._contagion.lagger_ages_ms


class ContagionValidator:
    def __init__(self, config: ContagionValidatorConfig) -> None:
        self.config = config
        self._validate_config(config)

    def run(
        self,
        replay_date: str,
        causal_lag_config: CausalLagConfig,
        output_path: str,
    ) -> ContagionValidationReport:
        report, payload = self._run_validation(replay_date, causal_lag_config)
        self._write_json(output_path, payload)
        return report

    def run_sweep(
        self,
        replay_date: str,
        sweep_param: Literal["max_lagger_age_ms", "max_causal_lag_ms", "max_leader_age_ms"],
        sweep_values: list[int],
        base_config: CausalLagConfig,
        output_path: str,
    ) -> list[ContagionValidationReport]:
        parse_iso_date(replay_date)
        if not sweep_values:
            raise ValueError("sweep_values must not be empty")
        if sweep_param not in {"max_lagger_age_ms", "max_causal_lag_ms", "max_leader_age_ms"}:
            raise ValueError(f"Unsupported sweep_param: {sweep_param!r}")

        reports: list[ContagionValidationReport] = []
        serialized_payload: list[dict[str, Any]] = []
        for value in sweep_values:
            if int(value) < 0:
                raise ValueError("sweep_values must be >= 0")
            config = replace(base_config, **{sweep_param: float(value)})
            report, payload = self._run_validation(replay_date, config)
            reports.append(report)
            serialized_payload.append(payload)

        self._write_json(output_path, serialized_payload)
        return reports

    def _run_validation(
        self,
        replay_date: str,
        causal_lag_config: CausalLagConfig,
    ) -> tuple[ContagionValidationReport, dict[str, Any]]:
        parse_iso_date(replay_date)
        market_configs = load_universe_market_configs(self.config.universe_path)
        asset_ids = {
            str(config.get("yes_asset_id") or "")
            for config in market_configs
        } | {
            str(config.get("no_asset_id") or "")
            for config in market_configs
        }
        asset_ids.discard("")

        loader = _build_data_loader(
            self.config.archive_path,
            [replay_date],
            asset_ids=asset_ids,
        )
        if loader is None:
            raise RuntimeError(f"No archived data found for {replay_date!r}")
        if self.config.max_events is not None:
            loader = DataLoaderLimiter(loader, self.config.max_events)

        params = self._build_strategy_params(causal_lag_config)
        strategy = _InstrumentedContagionReplayAdapter(
            market_configs=market_configs,
            fee_enabled=True,
            initial_bankroll=1000.0,
            params=params,
            emit_per_event_telemetry=self.config.emit_per_event_telemetry,
        )
        engine = BacktestEngine(
            strategy=strategy,
            data_loader=loader,
            config=BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_max_pct=2.0, fee_enabled=True),
        )
        result = engine.run()
        diagnostics = strategy.detector_diagnostics()

        cross_market_pairs = int(diagnostics.get("cross_market_pairs_evaluated", 0) or 0)
        accepted_causal = int(diagnostics.get("accepted_causal_lag_count", 0) or 0)
        legacy_sync_passed = int(diagnostics.get("legacy_sync_pairs_passed", 0) or 0)
        suppressor_breakdown = _suppression_breakdown(diagnostics)
        dominant_suppressor = _dominant_suppressor(suppressor_breakdown)
        lagger_ages = strategy.lagger_ages_ms()
        report = ContagionValidationReport(
            replay_date=replay_date,
            causal_lag_config=causal_lag_config,
            leader_events_evaluated=int(diagnostics.get("evaluations_with_previous_snapshot", 0) or 0),
            events_reaching_spike_check=int(diagnostics.get("toxicity_spikes_detected", 0) or 0),
            cross_market_pairs_evaluated=cross_market_pairs,
            causal_gate_pass_rate=_safe_rate(accepted_causal, cross_market_pairs),
            legacy_sync_pass_rate=_safe_rate(legacy_sync_passed, cross_market_pairs),
            signals_fired=int(diagnostics.get("signals_emitted", 0) or 0),
            fills_executed=int(result.metrics.total_fills),
            dominant_suppressor=dominant_suppressor,
            suppression_breakdown=suppressor_breakdown,
            median_lagger_age_ms=_median(lagger_ages),
            p95_lagger_age_ms=percentile(lagger_ages, 95.0) if lagger_ages else 0.0,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        payload: dict[str, Any] = {
            "report": _serialize_report(report),
            "diagnostics": diagnostics,
        }
        if self.config.emit_per_event_telemetry:
            payload["per_event_telemetry"] = strategy.pair_telemetry()
        return report, payload

    @staticmethod
    def _validate_config(config: ContagionValidatorConfig) -> None:
        archive_path = Path(config.archive_path)
        universe_path = Path(config.universe_path)
        if not archive_path.exists():
            raise ValueError(f"archive_path does not exist: {config.archive_path!r}")
        if not universe_path.exists():
            raise ValueError(f"universe_path does not exist: {config.universe_path!r}")
        if config.max_events is not None and int(config.max_events) <= 0:
            raise ValueError("max_events must be None or > 0")

    @staticmethod
    def _write_json(output_path: str, payload: Any) -> None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _build_strategy_params(causal_lag_config: CausalLagConfig) -> StrategyParams:
        params = dict(_load_default_champion_params())
        params.update(
            {
                "contagion_arb_max_leader_age_ms": float(causal_lag_config.max_leader_age_ms),
                "contagion_arb_max_lagger_age_ms": float(causal_lag_config.max_lagger_age_ms),
                "contagion_arb_max_causal_lag_ms": float(causal_lag_config.max_causal_lag_ms),
                "contagion_arb_allow_negative_lag": bool(causal_lag_config.allow_negative_lag),
            }
        )
        return StrategyParams(**params)


class DataLoaderLimiter:
    def __init__(self, loader: Any, max_events: int) -> None:
        self._loader = loader
        self._max_events = max_events

    def __iter__(self):
        for index, event in enumerate(self._loader):
            if index >= self._max_events:
                break
            yield event


def _load_default_champion_params() -> dict[str, Any]:
    for candidate in DEFAULT_CHAMPION_CANDIDATES:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            params = payload.get("params", {})
            if isinstance(params, dict):
                return dict(params)

    for candidate in sorted(Path("data").glob("wfo_contagion_arb*/champion_params.json"), reverse=True):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        params = payload.get("params", {})
        if isinstance(params, dict):
            return dict(params)
    return {}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sorted(values)[len(values) // 2] if len(values) % 2 == 1 else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2.0)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _suppression_breakdown(diagnostics: dict[str, Any]) -> dict[str, int]:
    return {
        "no_toxicity_spike": int(diagnostics.get("reject_no_toxicity_spike", 0) or 0),
        "insufficient_leader_impulse": int(diagnostics.get("reject_insufficient_leader_impulse", 0) or 0),
        "lagger_snapshot_stale": int(diagnostics.get("reject_lagger_snapshot_stale", 0) or 0),
        "causal_lag_too_large": int(diagnostics.get("reject_causal_lag_too_large", 0) or 0),
        "lagger_newer_than_leader": int(diagnostics.get("reject_lagger_newer_than_leader", 0) or 0),
        "leader_snapshot_stale": int(diagnostics.get("reject_leader_snapshot_stale", 0) or 0),
        "correlation_too_low": int(diagnostics.get("reject_correlation_too_low", 0) or 0),
        "residual_shift_too_small": int(diagnostics.get("reject_residual_shift_too_small", 0) or 0),
        "missing_causal_timestamp": int(diagnostics.get("reject_missing_causal_timestamp", 0) or 0),
    }


def _dominant_suppressor(suppression_breakdown: dict[str, int]) -> str:
    if not suppression_breakdown:
        return "none"
    reason, count = max(suppression_breakdown.items(), key=lambda item: (item[1], item[0]))
    if count <= 0:
        return "none"
    return reason


def _serialize_report(report: ContagionValidationReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["causal_lag_config"] = asdict(report.causal_lag_config)
    return payload


def _build_validator_config(args: argparse.Namespace) -> ContagionValidatorConfig:
    return ContagionValidatorConfig(
        archive_path=str(args.archive_path),
        universe_path=str(args.universe_path),
        max_events=args.max_events,
        emit_per_event_telemetry=bool(args.emit_per_event_telemetry),
    )


def _base_causal_config(args: argparse.Namespace) -> CausalLagConfig:
    return CausalLagConfig(
        max_leader_age_ms=float(args.max_leader_age),
        max_lagger_age_ms=float(args.max_lagger_age),
        max_causal_lag_ms=float(args.max_causal_lag),
        allow_negative_lag=bool(args.allow_negative_lag),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate contagion causal gating on archived replay data.")
    parser.add_argument("--date", required=True, help="Replay date in YYYY-MM-DD format.")
    parser.add_argument("--archive-path", default=str(DEFAULT_ARCHIVE_PATH))
    parser.add_argument("--universe-path", default=str(DEFAULT_UNIVERSE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_VALIDATION_OUTPUT))
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--emit-per-event-telemetry", action="store_true")
    parser.add_argument("--max-leader-age", type=float, default=5000.0)
    parser.add_argument("--max-lagger-age", type=float, default=30000.0)
    parser.add_argument("--max-causal-lag", type=float, default=600000.0)
    parser.add_argument("--allow-negative-lag", action="store_true")
    parser.add_argument(
        "--sweep-param",
        choices=("max_lagger_age_ms", "max_causal_lag_ms", "max_leader_age_ms"),
        default=None,
    )
    parser.add_argument("--sweep-values", nargs="*", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validator = ContagionValidator(_build_validator_config(args))
    base_config = _base_causal_config(args)

    if args.sweep_param:
        if not args.sweep_values:
            parser.error("--sweep-values is required when --sweep-param is set")
        reports = validator.run_sweep(
            replay_date=args.date,
            sweep_param=args.sweep_param,
            sweep_values=list(args.sweep_values),
            base_config=base_config,
            output_path=args.output,
        )
        serialized = [_serialize_report(report) for report in reports]
        print(json.dumps(serialized, indent=2))
        return 0

    report = validator.run(
        replay_date=args.date,
        causal_lag_config=base_config,
        output_path=args.output,
    )
    print(json.dumps(_serialize_report(report), indent=2))
    return 0


def build_freshness_sweep(
    *,
    validator: ContagionValidator,
    replay_date: str,
    sweep_values: list[int],
    base_config: CausalLagConfig,
) -> dict[str, Any]:
    reports = validator.run_sweep(
        replay_date=replay_date,
        sweep_param="max_lagger_age_ms",
        sweep_values=sweep_values,
        base_config=base_config,
        output_path=str(Path("_tmp_contagion_validator_sweep.json")),
    )
    market_configs = load_universe_market_configs(validator.config.universe_path)
    price_series = build_yes_price_series(
        validator.config.archive_path,
        market_configs,
        [replay_date],
        max_events=validator.config.max_events,
    )

    sweep_rows: list[dict[str, Any]] = []
    for report, window_ms in zip(reports, sweep_values, strict=False):
        pairs_passing = int(round(report.causal_gate_pass_rate * float(report.cross_market_pairs_evaluated)))
        sweep_rows.append(
            {
                "max_lagger_age_ms": int(window_ms),
                "pairs_passing": pairs_passing,
                "pass_rate_pct": round(report.causal_gate_pass_rate * 100.0, 4),
                "signals_fired": int(report.signals_fired),
                "dominant_suppressor": report.dominant_suppressor,
                "median_price_move_in_window": round(median_absolute_move_over_window(price_series, int(window_ms)), 6),
            }
        )

    recommendation = recommend_freshness_setting(sweep_rows)
    return {
        "sweep": sweep_rows,
        "recommendation": recommendation,
    }


def recommend_freshness_setting(sweep_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not sweep_rows:
        raise ValueError("sweep_rows must not be empty")

    no_signals = all(int(row.get("signals_fired", 0) or 0) == 0 for row in sweep_rows)
    threshold = 0.005
    admissibility_values = [int(row.get("pairs_passing", 0) or 0) for row in sweep_rows]
    max_pairs = max(admissibility_values) if admissibility_values else 0
    best_index = 0
    for index, row in enumerate(sweep_rows):
        risk = float(row.get("median_price_move_in_window", 0.0) or 0.0)
        gain = int(row.get("pairs_passing", 0) or 0)
        next_gain = int(sweep_rows[index + 1].get("pairs_passing", 0) or 0) if index + 1 < len(sweep_rows) else gain
        incremental_gain = next_gain - gain
        if risk <= threshold and (incremental_gain <= max(1, int(max_pairs * 0.05)) or gain == max_pairs):
            best_index = index
            break
        if risk <= threshold:
            best_index = index

    chosen = sweep_rows[best_index]
    rationale = (
        f"Selected {int(chosen['max_lagger_age_ms'])} ms as the knee of the admissibility curve: "
        f"{int(chosen['pairs_passing'])} pairs pass with median archive price movement {float(chosen['median_price_move_in_window']):.4%}."
    )
    if no_signals:
        rationale += " No sweep value produced signals on this archive; recommendation maximizes admissibility while keeping median staleness risk at or below 0.5%."
    return {
        "suggested_max_lagger_age_ms": int(chosen["max_lagger_age_ms"]),
        "rationale": rationale,
    }


if __name__ == "__main__":
    raise SystemExit(main())
