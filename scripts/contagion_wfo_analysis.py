from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

from src.core.logger import setup_logging
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.strategy import ContagionReplayAdapter
from src.backtest.wfo_optimizer import (
    SEARCH_SPACE,
    WfoConfig,
    _build_data_loader,
    _load_market_configs,
    compute_wfo_score,
)
from src.core.config import StrategyParams
from src.trading.executor import OrderSide
from src.signals.contagion_arb import (
    ContagionArbDetector,
    ContagionArbSignal,
    ContagionSnapshot,
    _clamp_probability,
    _normalise_tags,
)


DEFAULT_ARTIFACT_DIR = Path("data/wfo_contagion_arb_micro_2026_03_25_microscale_fast")
DEFAULT_STORAGE = DEFAULT_ARTIFACT_DIR / "wfo_optuna.db"
DEFAULT_REPORT = DEFAULT_ARTIFACT_DIR / "wfo_report.json"
DEFAULT_CHAMPION = DEFAULT_ARTIFACT_DIR / "champion_params.json"
DEFAULT_MARKET_CONFIGS = Path("data/domino_micro_fast_market_map.json")
DEFAULT_DATA_DIR = Path("data/vps_march2026")
DEFAULT_TELEMETRY_DUMP = Path("contagion_telemetry_dump.json")

setup_logging(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso_utc_from_date(date_str: str, *, end_of_day: bool = False) -> str:
    suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
    return f"{date_str}{suffix}"


def _trial_duration_seconds(trial: optuna.trial.FrozenTrial) -> float | None:
    if trial.datetime_start is None or trial.datetime_complete is None:
        return None
    return (trial.datetime_complete - trial.datetime_start).total_seconds()


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _format_range(name: str) -> tuple[float | int | None, float | int | None, str]:
    spec = SEARCH_SPACE.get(name)
    if not spec or len(spec) < 3:
        return None, None, "unknown"
    return spec[1], spec[2], spec[0]


def _at_boundary(name: str, value: float) -> bool:
    lo, hi, _ = _format_range(name)
    if lo is None or hi is None:
        return False
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f <= lo_f:
        return False
    band = (hi_f - lo_f) * 0.05
    return value <= lo_f + band or value >= hi_f - band


@dataclass
class TrialOosResult:
    study_name: str
    trial_number: int
    params: dict[str, float]
    oos_metrics: dict[str, Any]
    oos_score: float


def _zero_backtest_metrics() -> dict[str, Any]:
    return {
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "total_fills": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "maker_fills": 0,
        "taker_fills": 0,
        "equity_curve": [],
    }


class TelemetryContagionArbDetector(ContagionArbDetector):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.telemetry_events: list[dict[str, Any]] = []
        self.leader_toxicity_events_detected = 0
        self._latest_event: dict[str, Any] | None = None

    @property
    def latest_event(self) -> dict[str, Any] | None:
        return self._latest_event

    def evaluate_market(
        self,
        *,
        market: Any,
        yes_price: float,
        yes_buy_toxicity: float,
        no_buy_toxicity: float,
        timestamp: float | None = None,
        universe: list[Any] | None = None,
        book_snapshots: tuple[Any, ...] | None = None,
    ) -> list[ContagionArbSignal]:
        self._diagnostics["evaluations_total"] += 1
        sync_assessment = None
        previous = self._snapshots.get(market.condition_id)
        if book_snapshots:
            sync_assessment = self._sync_gate.assess(book_snapshots)
            if not sync_assessment.is_synchronized:
                if previous is not None:
                    now = timestamp if timestamp is not None else sync_assessment.latest_timestamp
                    event = {
                        "event_timestamp_ms": int(round(now * 1000.0)),
                        "leader_market_id": market.condition_id,
                        "leader_price_shift": round(self._last_price_shift.get(market.condition_id, 0.0), 6),
                        "toxicity_impulse": 0.0,
                        "residual_at_signal": None,
                        "signal_fired": False,
                        "fill_executed": False,
                        "suppression_reason": "SYNC_GATE",
                    }
                    self.telemetry_events.append(event)
                    self._latest_event = event
                if self._on_sync_block is not None:
                    self._on_sync_block(sync_assessment)
                return []

        now = timestamp if timestamp is not None else 0.0
        if sync_assessment is not None and sync_assessment.latest_timestamp > 0.0:
            now = sync_assessment.latest_timestamp
        if now <= 0.0:
            now = timestamp or 0.0

        current_price = _clamp_probability(yes_price)
        tags = _normalise_tags(getattr(market, "tags", ""))
        leader_shift = 0.0 if previous is None else current_price - previous.yes_price
        self._last_price_shift[market.condition_id] = leader_shift
        leader_snapshot = ContagionSnapshot(
            market_id=market.condition_id,
            yes_price=current_price,
            yes_buy_toxicity=max(0.0, min(1.0, float(yes_buy_toxicity))),
            no_buy_toxicity=max(0.0, min(1.0, float(no_buy_toxicity))),
            timestamp=now,
            tags=tags,
        )
        self._snapshots[market.condition_id] = leader_snapshot

        spikes: list[tuple[str, float, float]] = []
        yes_threshold = self._current_percentile(f"{market.condition_id}:buy_yes")
        no_threshold = self._current_percentile(f"{market.condition_id}:buy_no")
        if yes_threshold is not None and yes_buy_toxicity >= yes_threshold:
            spikes.append(("buy_yes", float(yes_buy_toxicity), yes_threshold))
        if no_threshold is not None and no_buy_toxicity >= no_threshold:
            spikes.append(("buy_no", float(no_buy_toxicity), no_threshold))

        self._toxicity_history[f"{market.condition_id}:buy_yes"].append(float(yes_buy_toxicity))
        self._toxicity_history[f"{market.condition_id}:buy_no"].append(float(no_buy_toxicity))

        if previous is None:
            self._latest_event = None
            return []

        event = {
            "event_timestamp_ms": int(round(now * 1000.0)),
            "leader_market_id": market.condition_id,
            "leader_price_shift": round(leader_shift, 6),
            "toxicity_impulse": 0.0,
            "residual_at_signal": None,
            "signal_fired": False,
            "fill_executed": False,
            "suppression_reason": None,
        }

        if not spikes:
            self._diagnostics["reject_no_toxicity_spike"] += 1
            event["suppression_reason"] = "OTHER"
            self.telemetry_events.append(event)
            self._latest_event = event
            return []

        self.leader_toxicity_events_detected += 1
        event["toxicity_impulse"] = round(
            max(max(0.0, leader_toxicity - threshold) * self._toxicity_impulse_scale for _, leader_toxicity, threshold in spikes),
            6,
        )

        signals: list[ContagionArbSignal] = []
        candidate_universe = universe[: self._universe_size] if universe else list(self._markets.values())[: self._universe_size]
        gate_counter: Counter[str] = Counter()
        max_residual: float | None = None

        for direction, leader_toxicity, threshold in sorted(spikes, key=lambda row: row[1], reverse=True):
            direction_sign = 1.0 if direction == "buy_yes" else -1.0
            directional_leader_shift = direction_sign * leader_shift
            directional_move = max(0.0, directional_leader_shift)
            toxicity_impulse = max(0.0, leader_toxicity - threshold) * self._toxicity_impulse_scale
            leader_impulse = max(directional_move, toxicity_impulse)
            if leader_impulse < self._min_leader_shift:
                self._diagnostics["reject_insufficient_leader_impulse"] += 1
                gate_counter["LEADER_SHIFT"] += 1
                continue

            for lagger in candidate_universe:
                if lagger.condition_id == market.condition_id:
                    continue
                lag_snapshot = self._snapshots.get(lagger.condition_id)
                if lag_snapshot is None:
                    gate_counter["OTHER"] += 1
                    continue
                lag_sync_assessment = self._sync_gate.assess((leader_snapshot, lag_snapshot))
                if not lag_sync_assessment.is_synchronized:
                    if self._on_sync_block is not None:
                        self._on_sync_block(lag_sync_assessment)
                    gate_counter["SYNC_GATE"] += 1
                    continue
                correlation = self._pair_correlation(market.condition_id, lagger.condition_id)
                if correlation < self._min_corr:
                    self._diagnostics["reject_correlation_too_low"] += 1
                    gate_counter["CORRELATION_GATE"] += 1
                    continue
                thematic_group = self._shared_theme(market, lagger)
                if not thematic_group:
                    gate_counter["OTHER"] += 1
                    continue
                lag_directional_shift = max(0.0, direction_sign * self._last_price_shift.get(lagger.condition_id, 0.0))
                expected_shift = correlation * leader_impulse
                residual_shift = expected_shift - lag_directional_shift
                if max_residual is None or residual_shift > max_residual:
                    max_residual = residual_shift
                if residual_shift < self._min_residual_shift:
                    self._diagnostics["reject_residual_shift_too_small"] += 1
                    gate_counter["RESIDUAL_GATE"] += 1
                    continue
                if now - self._last_signal_at.get(lagger.condition_id, 0.0) < self._cooldown_seconds:
                    gate_counter["OTHER"] += 1
                    continue

                implied_probability = _clamp_probability(lag_snapshot.yes_price + direction_sign * residual_shift)
                confidence = min(
                    0.95,
                    max(
                        self._rpe.min_confidence,
                        0.30
                        + 0.35 * correlation
                        + 0.20 * leader_toxicity
                        + 0.15 * min(1.0, residual_shift / max(self._min_residual_shift, 1e-6)),
                    ),
                )
                dislocation = self._rpe.evaluate_probability_dislocation(
                    market_id=lagger.condition_id,
                    market_price=lag_snapshot.yes_price,
                    implied_probability=implied_probability,
                    confidence=confidence,
                    model_name="contagion_correlation",
                    shadow_mode=self._shadow,
                    model_metadata={
                        "leading_market_id": market.condition_id,
                        "leading_toxicity": round(leader_toxicity, 6),
                        "toxicity_percentile": round(threshold, 6),
                        "leader_price_shift": round(direction_sign * leader_shift, 6),
                        "expected_probability_shift": round(direction_sign * residual_shift, 6),
                        "correlation": round(correlation, 6),
                        "thematic_group": thematic_group,
                    },
                    signal_name="contagion_arb",
                )
                if dislocation is None:
                    gate_counter["OTHER"] += 1
                    continue

                lagging_asset_id = lagger.yes_token_id if direction == "buy_yes" else lagger.no_token_id
                self._last_signal_at[lagger.condition_id] = now
                self._diagnostics["signals_emitted"] += 1
                event["residual_at_signal"] = round(residual_shift, 6)
                signals.append(
                    ContagionArbSignal(
                        leading_market_id=market.condition_id,
                        lagging_market_id=lagger.condition_id,
                        lagging_asset_id=lagging_asset_id,
                        direction=str(dislocation.metadata.get("direction", direction)),
                        implied_probability=implied_probability,
                        lagging_market_price=lag_snapshot.yes_price,
                        confidence=confidence,
                        correlation=correlation,
                        thematic_group=thematic_group,
                        toxicity_percentile=threshold,
                        leader_toxicity=leader_toxicity,
                        leader_price_shift=direction_sign * leader_shift,
                        expected_probability_shift=direction_sign * residual_shift,
                        timestamp=now,
                        score=float(dislocation.score),
                        is_shadow=self._shadow,
                        metadata=dict(dislocation.metadata),
                    )
                )

        signals.sort(key=lambda item: (-item.score, -item.correlation, -item.leader_toxicity))
        if signals:
            self.telemetry_events.append(event)
            self._latest_event = event
            return signals[: self._max_pairs]

        if max_residual is not None:
            event["residual_at_signal"] = round(max_residual, 6)
        event["suppression_reason"] = max(
            gate_counter.items(),
            key=lambda item: (item[1], item[0]),
            default=("OTHER", 0),
        )[0]
        self.telemetry_events.append(event)
        self._latest_event = event
        return []


class TelemetryContagionReplayAdapter(ContagionReplayAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._entry_fill_order_ids: set[str] = set()
        self._active_event: dict[str, Any] | None = None
        self._contagion = TelemetryContagionArbDetector(
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
        )

    def on_fill(self, fill: Any) -> None:
        self._entry_fill_order_ids.add(fill.order_id)
        super().on_fill(fill)

    def _evaluate_contagion_market(self, market_id: str, timestamp: float) -> None:
        market = self._markets.get(market_id)
        if market is None or self.engine is None:
            return

        yes_book = self._books.get(market.yes_token_id)
        no_book = self._books.get(market.no_token_id)
        if yes_book is None or no_book is None or not yes_book.has_data or not no_book.has_data:
            return

        yes_snapshot = yes_book.snapshot()
        no_snapshot = no_book.snapshot()
        yes_bid = float(getattr(yes_snapshot, "best_bid", 0.0) or 0.0)
        yes_ask = float(getattr(yes_snapshot, "best_ask", 0.0) or 0.0)
        if yes_bid <= 0 or yes_ask <= 0 or yes_ask <= yes_bid:
            return

        self._pce.refresh_correlations()
        signals = self._contagion.evaluate_market(
            market=market,
            yes_price=(yes_bid + yes_ask) / 2.0,
            yes_buy_toxicity=yes_book.toxicity_index("BUY"),
            no_buy_toxicity=no_book.toxicity_index("BUY"),
            timestamp=timestamp,
            universe=list(self._markets.values()),
            book_snapshots=(yes_snapshot, no_snapshot),
        )
        self._active_event = self._contagion.latest_event
        for signal in signals:
            self._open_contagion_position(signal, timestamp)
        self._active_event = None

    def _open_contagion_position(self, signal: ContagionArbSignal, timestamp: float) -> None:
        event = self._active_event
        if self.engine is None:
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return

        market = self._markets.get(signal.lagging_market_id)
        if market is None or not market.accepting_orders:
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return
        if any(pos.get("market_id") == market.condition_id for pos in self._open_positions.values()):
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return

        asset_id = signal.lagging_asset_id
        book = self._books.get(asset_id)
        if book is None or not book.has_data or book.best_bid <= 0 or book.best_ask <= 0:
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return

        agg = self._yes_aggs.get(asset_id) or self._no_aggs.get(asset_id)
        if agg is None or agg.last_trade_time <= 0:
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return
        if (timestamp - agg.last_trade_time) > self._params.contagion_arb_max_last_trade_age_s:
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return

        spread_pct = ((book.best_ask - book.best_bid) / book.best_ask) * 100.0
        if spread_pct > self._params.contagion_arb_max_lagging_spread_pct:
            if event is not None:
                event["suppression_reason"] = "SPREAD_GATE"
            return

        entry_price = round(book.best_ask, 2)
        if entry_price <= 0:
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return

        trade_usd = min(self._initial_bankroll * self._params.kelly_fraction, self._params.max_trade_size_usd)
        size = trade_usd / entry_price if entry_price > 0 else 0.0
        if size < 1:
            if event is not None:
                event["suppression_reason"] = event["suppression_reason"] or "OTHER"
            return

        fair_value = signal.implied_probability
        if signal.direction == "buy_no":
            fair_value = max(0.01, min(0.99, 1.0 - signal.implied_probability))
        target_price = round(max(entry_price + 0.01, min(0.99, fair_value)), 2)
        stop_price = round(max(0.01, entry_price * (1.0 - self._params.stop_loss_pct)), 2)

        order = self.engine.submit_order(
            side=OrderSide.BUY,
            price=entry_price,
            size=size,
            order_type="limit",
            post_only=False,
            asset_id=asset_id,
        )
        if event is not None:
            event["signal_fired"] = True
            event["suppression_reason"] = None
        self._pending_entries[order.order_id] = {
            "asset_id": asset_id,
            "market_id": market.condition_id,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_seconds": max(60.0, self._params.contagion_arb_cooldown_seconds),
            "signal_source": signal.signal_source,
        }
        self._flush_pending_orders(asset_id, timestamp)
        if event is not None and order.order_id in self._entry_fill_order_ids:
            event["fill_executed"] = True


def _build_telemetry_adapter(market_configs: list[dict[str, Any]], params: dict[str, float]) -> TelemetryContagionReplayAdapter:
    strategy_params = StrategyParams(**params)
    return TelemetryContagionReplayAdapter(
        market_configs=market_configs,
        fee_enabled=True,
        initial_bankroll=1000.0,
        params=strategy_params,
    )


def _run_telemetry_replay(
    *,
    data_dir: Path,
    dates: list[str],
    market_configs: list[dict[str, Any]],
    params: dict[str, float],
) -> dict[str, Any]:
    loader = _build_data_loader(str(data_dir), dates, asset_ids={
        str(config.get("yes_asset_id") or "") for config in market_configs
    } | {
        str(config.get("no_asset_id") or "") for config in market_configs
    })
    if loader is None:
        return {
            "meta": {
                "total_leader_events": 0,
                "leader_toxicity_events_detected": 0,
                "total_signals_fired": 0,
                "total_fills_executed": 0,
                "suppression_breakdown": {
                    "LEADER_SHIFT": 0,
                    "RESIDUAL_GATE": 0,
                    "CORRELATION_GATE": 0,
                    "SYNC_GATE": 0,
                    "SPREAD_GATE": 0,
                    "OTHER": 0,
                },
                "backtest_metrics": _zero_backtest_metrics(),
                "contagion_detector_diagnostics": {},
            },
            "events": [],
        }

    strategy = _build_telemetry_adapter(market_configs, params)
    config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_max_pct=2.0, fee_enabled=True)
    engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
    result = engine.run()
    events = strategy._contagion.telemetry_events
    suppression_breakdown = Counter(
        str(event.get("suppression_reason") or "OTHER")
        for event in events
        if not bool(event.get("signal_fired"))
    )
    return {
        "meta": {
            "total_leader_events": len(events),
            "leader_toxicity_events_detected": int(strategy._contagion.leader_toxicity_events_detected),
            "total_signals_fired": sum(1 for event in events if event.get("signal_fired")),
            "total_fills_executed": sum(1 for event in events if event.get("fill_executed")),
            "suppression_breakdown": {
                "LEADER_SHIFT": int(suppression_breakdown.get("LEADER_SHIFT", 0)),
                "RESIDUAL_GATE": int(suppression_breakdown.get("RESIDUAL_GATE", 0)),
                "CORRELATION_GATE": int(suppression_breakdown.get("CORRELATION_GATE", 0)),
                "SYNC_GATE": int(suppression_breakdown.get("SYNC_GATE", 0)),
                "SPREAD_GATE": int(suppression_breakdown.get("SPREAD_GATE", 0)),
                "OTHER": int(suppression_breakdown.get("OTHER", 0)),
            },
            "backtest_metrics": result.metrics.to_dict(),
            "contagion_detector_diagnostics": strategy.detector_diagnostics(),
        },
        "events": events,
    }


def _compute_oos_score(metrics: dict[str, Any], cfg: WfoConfig) -> float:
    return compute_wfo_score(
        sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0) or 0.0),
        max_drawdown=float(metrics.get("max_drawdown", 0.0) or 0.0),
        max_acceptable_drawdown=cfg.max_acceptable_drawdown,
        sortino_ratio=float(metrics.get("sortino_ratio", 0.0) or 0.0),
        profit_factor=float(metrics.get("profit_factor", 0.0) or 0.0),
        total_fills=int(metrics.get("total_fills", 0) or 0),
        min_trades=1,
        sharpe_weight=cfg.sharpe_weight,
        sortino_weight=cfg.sortino_weight,
        profit_factor_weight=cfg.profit_factor_weight,
        trade_bonus_weight=cfg.trade_bonus_weight,
    )


def _analyse_trials(
    *,
    storage_url: str,
    report_data: dict[str, Any],
    market_configs: list[dict[str, Any]],
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[TrialOosResult]]:
    cfg = WfoConfig(strategy_adapter="contagion_arb", min_trades=1)
    study_names = optuna.study.get_all_study_names(storage=storage_url)
    fold_lookup = {f"polymarket_wfo_fold_{fold['fold_index']}": fold for fold in report_data["folds"]}
    per_study: list[dict[str, Any]] = []
    all_oos_results: list[TrialOosResult] = []

    for study_name in study_names:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        fold = fold_lookup.get(study_name)
        if fold is None:
            continue
        durations = [value for trial in study.trials if (value := _trial_duration_seconds(trial)) is not None]
        state_breakdown = Counter(trial.state.name for trial in study.trials)
        timed_out = sum(1 for trial in study.trials if bool(trial.user_attrs.get("timed_out")))
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            replay = _run_telemetry_replay(
                data_dir=data_dir,
                dates=list(fold["test_dates"]),
                market_configs=market_configs,
                params={str(key): float(value) for key, value in trial.params.items()},
            )
            oos_metrics = dict(replay.get("meta", {}).get("backtest_metrics", _zero_backtest_metrics()))
            oos_score = _compute_oos_score(oos_metrics, cfg)
            all_oos_results.append(
                TrialOosResult(
                    study_name=study_name,
                    trial_number=int(trial.number),
                    params={str(key): float(value) for key, value in trial.params.items()},
                    oos_metrics=oos_metrics,
                    oos_score=oos_score,
                )
            )
        per_study.append(
            {
                "study_name": study_name,
                "total_trials": len(study.trials),
                "state_breakdown": dict(state_breakdown),
                "duration_min_s": min(durations) if durations else None,
                "duration_median_s": _median(durations),
                "duration_max_s": max(durations) if durations else None,
                "timed_out_count": timed_out,
                "timed_out_pct": (timed_out / len(study.trials) * 100.0) if study.trials else 0.0,
            }
        )

    return per_study, all_oos_results


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        if str(value) == "-inf":
            return float("-inf")
        raise


def _build_analysis(args: argparse.Namespace) -> dict[str, Any]:
    report_data = _read_json(args.report_path)
    champion_data = _read_json(args.champion_path)
    market_configs = _load_market_configs(str(args.data_dir), str(args.market_configs_path))
    storage_url = f"sqlite:///{args.storage_path.as_posix()}"

    per_study, all_oos_results = _analyse_trials(
        storage_url=storage_url,
        report_data=report_data,
        market_configs=market_configs,
        data_dir=args.data_dir,
    )

    oos_scores = [result.oos_score for result in all_oos_results]
    positive_oos = sum(1 for value in oos_scores if value > 0)
    non_positive_oos = sum(1 for value in oos_scores if value <= 0)
    top10 = sorted(all_oos_results, key=lambda item: item.oos_score, reverse=True)[:10]
    top10_values_by_param: dict[str, list[float]] = {}
    for result in top10:
        for name, value in result.params.items():
            top10_values_by_param.setdefault(name, []).append(float(value))

    champion_params = {str(k): float(v) for k, v in champion_data["params"].items()}
    champion_table = []
    for name, value in champion_params.items():
        lo, hi, _ = _format_range(name)
        champion_table.append(
            {
                "parameter": name,
                "value": value,
                "search_range": [lo, hi],
                "at_boundary": _at_boundary(name, value),
                "top10_median": _median(top10_values_by_param.get(name, [])),
            }
        )

    champion_fold_index = int(champion_data["meta"]["champion_fold"])
    champion_fold = next(fold for fold in report_data["folds"] if int(fold["fold_index"]) == champion_fold_index)
    champion_telemetry = _run_telemetry_replay(
        data_dir=args.data_dir,
        dates=list(champion_fold["test_dates"]),
        market_configs=market_configs,
        params=champion_params,
    )
    champion_dump = {
        "meta": {
            "champion_fold": champion_fold_index,
            "backtest_window": {
                "start": _iso_utc_from_date(champion_fold["test_dates"][0]),
                "end": _iso_utc_from_date(champion_fold["test_dates"][-1], end_of_day=True),
            },
            "universe_size": len(market_configs),
            "total_leader_events": int(champion_telemetry["meta"]["total_leader_events"]),
            "total_signals_fired": int(champion_telemetry["meta"]["total_signals_fired"]),
            "total_fills_executed": int(champion_telemetry["meta"]["total_fills_executed"]),
            "suppression_breakdown": dict(champion_telemetry["meta"]["suppression_breakdown"]),
        },
        "events": champion_telemetry["events"],
    }
    args.telemetry_dump_path.write_text(json.dumps(champion_dump, indent=2), encoding="utf-8")
    json.loads(args.telemetry_dump_path.read_text(encoding="utf-8"))

    aggregate_oos_leader_events = 0
    aggregate_oos_toxicity_events = 0
    aggregate_oos_signals = 0
    aggregate_oos_fills = 0
    for fold in report_data["folds"]:
        fold_telemetry = _run_telemetry_replay(
            data_dir=args.data_dir,
            dates=list(fold["test_dates"]),
            market_configs=market_configs,
            params={str(k): float(v) for k, v in fold["best_params"].items()},
        )
        aggregate_oos_leader_events += int(fold_telemetry["meta"]["total_leader_events"])
        aggregate_oos_toxicity_events += int(fold_telemetry["meta"]["leader_toxicity_events_detected"])
        aggregate_oos_signals += int(fold_telemetry["meta"]["total_signals_fired"])
        aggregate_oos_fills += int(fold_telemetry["meta"]["total_fills_executed"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_analysis": per_study,
        "objective_metric": {
            "optimized_metric": "in_sample_composite_score",
            "formula": "compute_wfo_score(sharpe, sortino, profit_factor, sqrt(fills), max_drawdown_penalty)",
            "min_trades_gate": 1,
        },
        "oos_analysis": {
            "derived_metric": "oos_composite_score",
            "best_oos_value": max(oos_scores) if oos_scores else None,
            "all_complete_trial_oos_values": oos_scores,
            "positive_count": positive_oos,
            "non_positive_count": non_positive_oos,
            "all_non_positive": bool(oos_scores) and positive_oos == 0,
        },
        "champion_table": champion_table,
        "top10_oos_trials": [
            {
                "study_name": result.study_name,
                "trial_number": result.trial_number,
                "oos_score": result.oos_score,
                "params": result.params,
                "oos_total_fills": int(result.oos_metrics.get("total_fills", 0) or 0),
                "oos_sharpe": float(result.oos_metrics.get("sharpe_ratio", 0.0) or 0.0),
                "oos_total_pnl": float(result.oos_metrics.get("total_pnl", 0.0) or 0.0),
            }
            for result in top10
        ],
        "aggregate_oos_zero_fill_diagnosis": {
            "total_leader_events": aggregate_oos_leader_events,
            "leader_toxicity_events": aggregate_oos_toxicity_events,
            "signals_generated": aggregate_oos_signals,
            "fills_executed": aggregate_oos_fills,
            "zero_fills_means_zero_signals": aggregate_oos_signals == 0 and aggregate_oos_fills == 0,
            "fill_gate_too_tight": aggregate_oos_signals > 0 and aggregate_oos_fills == 0,
        },
        "champion_fold_window": {
            "fold_index": champion_fold_index,
            "train_dates": champion_fold["train_dates"],
            "test_dates": champion_fold["test_dates"],
        },
        "telemetry_dump_path": str(args.telemetry_dump_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse contagion WFO artifacts and emit a telemetry dump.")
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--champion-path", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--market-configs-path", type=Path, default=DEFAULT_MARKET_CONFIGS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--telemetry-dump-path", type=Path, default=DEFAULT_TELEMETRY_DUMP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = _build_analysis(args)
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())