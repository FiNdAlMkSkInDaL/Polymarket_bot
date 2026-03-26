from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.strategy import ContagionReplayAdapter
from src.backtest.wfo_optimizer import _build_data_loader, _load_market_configs
from src.core.config import StrategyParams
from src.core.logger import setup_logging
from src.signals.contagion_arb import (
    ContagionArbDetector,
    ContagionArbSignal,
    ContagionSnapshot,
    _clamp_probability,
    _normalise_tags,
)
from src.signals.microstructure_utils import snapshot_timestamp
from src.trading.executor import OrderSide

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "wfo_contagion_arb_micro_2026_03_25_microscale_fast"
DATA_DIR = ROOT / "data" / "vps_march2026"
MARKET_CONFIG_PATH = ROOT / "data" / "domino_micro_fast_market_map.json"
CLUSTER_PATH = ROOT / "data" / "si9_clusters_monday.json"
MARKET_MAP_PATH = ROOT / "data" / "market_map.json"
OUTPUT_PATH = ROOT / "_tmp_contagion_revision_analysis.json"

setup_logging(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    ordered = sorted(values)

    def pick(q: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return float(ordered[idx])

    return {
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def _pass_rates(values: list[float], thresholds: list[float]) -> dict[str, float]:
    if not values:
        return {str(int(threshold)): 0.0 for threshold in thresholds}
    total = len(values)
    return {str(int(threshold)): round(sum(1 for value in values if value <= threshold) / total, 6) for threshold in thresholds}


def _safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return float(cov / math.sqrt(var_x * var_y))


class RevisionContagionDetector(ContagionArbDetector):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.leader_intrabook_deltas_ms: list[float] = []
        self.leader_pair_deltas_ms: list[float] = []
        self.sync_samples_by_pair: dict[str, list[float]] = defaultdict(list)
        self.pair_corr_samples: dict[str, list[float]] = defaultdict(list)
        self.leader_evaluations_with_previous = 0
        self.leader_events_with_spike = 0
        self.leader_events_with_impulse = 0

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
            if math.isfinite(sync_assessment.delta_ms):
                self.leader_intrabook_deltas_ms.append(float(sync_assessment.delta_ms))
            if not sync_assessment.is_synchronized:
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
            return []

        self.leader_evaluations_with_previous += 1
        if not spikes:
            self._diagnostics["reject_no_toxicity_spike"] += 1
            return []

        self.leader_events_with_spike += 1
        signals: list[ContagionArbSignal] = []
        candidate_universe = universe[: self._universe_size] if universe else list(self._markets.values())[: self._universe_size]

        for direction, leader_toxicity, threshold in sorted(spikes, key=lambda row: row[1], reverse=True):
            direction_sign = 1.0 if direction == "buy_yes" else -1.0
            directional_leader_shift = direction_sign * leader_shift
            directional_move = max(0.0, directional_leader_shift)
            toxicity_impulse = max(0.0, leader_toxicity - threshold) * self._toxicity_impulse_scale
            leader_impulse = max(directional_move, toxicity_impulse)
            if leader_impulse < self._min_leader_shift:
                self._diagnostics["reject_insufficient_leader_impulse"] += 1
                continue

            self.leader_events_with_impulse += 1
            for lagger in candidate_universe:
                if lagger.condition_id == market.condition_id:
                    continue
                lag_snapshot = self._snapshots.get(lagger.condition_id)
                if lag_snapshot is None:
                    continue
                lag_sync_assessment = self._sync_gate.assess((leader_snapshot, lag_snapshot))
                pair_key = "|".join(sorted((market.condition_id, lagger.condition_id)))
                if math.isfinite(lag_sync_assessment.delta_ms):
                    delta_ms = float(lag_sync_assessment.delta_ms)
                    self.leader_pair_deltas_ms.append(delta_ms)
                    self.sync_samples_by_pair[pair_key].append(delta_ms)
                if not lag_sync_assessment.is_synchronized:
                    if self._on_sync_block is not None:
                        self._on_sync_block(lag_sync_assessment)
                    continue
                correlation = self._pair_correlation(market.condition_id, lagger.condition_id)
                self.pair_corr_samples[pair_key].append(float(correlation))
                if correlation < self._min_corr:
                    self._diagnostics["reject_correlation_too_low"] += 1
                    continue
                thematic_group = self._shared_theme(market, lagger)
                if not thematic_group:
                    continue
                lag_directional_shift = max(0.0, direction_sign * self._last_price_shift.get(lagger.condition_id, 0.0))
                expected_shift = correlation * leader_impulse
                residual_shift = expected_shift - lag_directional_shift
                if residual_shift < self._min_residual_shift:
                    self._diagnostics["reject_residual_shift_too_small"] += 1
                    continue
                if now - self._last_signal_at.get(lagger.condition_id, 0.0) < self._cooldown_seconds:
                    continue
                implied_probability = _clamp_probability(lag_snapshot.yes_price + direction_sign * residual_shift)
                confidence = min(
                    0.95,
                    max(
                        self._rpe.min_confidence,
                        0.30 + 0.35 * correlation + 0.20 * leader_toxicity + 0.15 * min(1.0, residual_shift / max(self._min_residual_shift, 1e-6)),
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
                    continue
                lagging_asset_id = lagger.yes_token_id if direction == "buy_yes" else lagger.no_token_id
                self._last_signal_at[lagger.condition_id] = now
                self._diagnostics["signals_emitted"] += 1
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
        return signals[: self._max_pairs]


class RevisionReplayAdapter(ContagionReplayAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.market_mid_history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._contagion = RevisionContagionDetector(
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
        mid = (yes_bid + yes_ask) / 2.0
        history = self.market_mid_history[market.condition_id]
        if not history or abs(history[-1][0] - timestamp) > 1e-9 or abs(history[-1][1] - mid) > 1e-9:
            history.append((float(timestamp), float(mid)))
        self._pce.refresh_correlations()
        signals = self._contagion.evaluate_market(
            market=market,
            yes_price=mid,
            yes_buy_toxicity=yes_book.toxicity_index("BUY"),
            no_buy_toxicity=no_book.toxicity_index("BUY"),
            timestamp=timestamp,
            universe=list(self._markets.values()),
            book_snapshots=(yes_snapshot, no_snapshot),
        )
        for signal in signals:
            self._open_contagion_position(signal, timestamp)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _match_deltas(series_a: list[tuple[float, float]], series_b: list[tuple[float, float]], tolerance_s: float = 5.0) -> list[tuple[float, float]]:
    if len(series_a) < 2 or len(series_b) < 2:
        return []
    deltas_a = [(series_a[index][0], series_a[index][1] - series_a[index - 1][1]) for index in range(1, len(series_a))]
    deltas_b = [(series_b[index][0], series_b[index][1] - series_b[index - 1][1]) for index in range(1, len(series_b))]
    matched: list[tuple[float, float]] = []
    j = 0
    for ts_a, delta_a in deltas_a:
        while j + 1 < len(deltas_b) and deltas_b[j + 1][0] <= ts_a:
            j += 1
        candidates: list[tuple[float, float]] = []
        for idx in (j, j + 1):
            if 0 <= idx < len(deltas_b):
                ts_b, delta_b = deltas_b[idx]
                if abs(ts_a - ts_b) <= tolerance_s:
                    candidates.append((abs(ts_a - ts_b), delta_b))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        matched.append((delta_a, candidates[0][1]))
    return matched


def _build_replay_analysis() -> dict[str, Any]:
    champion = _load_json(ARTIFACT_DIR / "champion_params.json")
    report = _load_json(ARTIFACT_DIR / "wfo_report.json")
    fold_zero = next(fold for fold in report["folds"] if int(fold["fold_index"]) == int(champion["meta"]["champion_fold"]))
    market_configs = _load_market_configs(str(DATA_DIR), str(MARKET_CONFIG_PATH))
    params = StrategyParams(**champion["params"])
    asset_ids = {str(config.get("yes_asset_id") or "") for config in market_configs} | {str(config.get("no_asset_id") or "") for config in market_configs}
    loader = _build_data_loader(str(DATA_DIR), list(fold_zero["test_dates"]), asset_ids=asset_ids)
    strategy = RevisionReplayAdapter(market_configs=market_configs, fee_enabled=True, initial_bankroll=1000.0, params=params)
    config = BacktestConfig(initial_cash=1000.0, latency_ms=0.0, fee_max_pct=2.0, fee_enabled=True)
    engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
    result = engine.run()
    detector = strategy._contagion

    leader_pair_values = detector.leader_pair_deltas_ms
    leader_book_values = detector.leader_intrabook_deltas_ms
    thresholds = [float(champion["params"]["max_cross_book_desync_ms"]), 2000.0, 5000.0]

    pair_summaries: list[dict[str, Any]] = []
    market_ids = sorted(strategy.market_mid_history)
    for index, market_a in enumerate(market_ids):
        for market_b in market_ids[index + 1 :]:
            matched = _match_deltas(strategy.market_mid_history[market_a], strategy.market_mid_history[market_b])
            if not matched:
                continue
            deltas_a = [item[0] for item in matched]
            deltas_b = [item[1] for item in matched]
            same_direction = [1 for delta_a, delta_b in matched if delta_a != 0.0 and delta_b != 0.0 and ((delta_a > 0) == (delta_b > 0))]
            nonzero = [1 for delta_a, delta_b in matched if delta_a != 0.0 and delta_b != 0.0]
            pair_key = "|".join(sorted((market_a, market_b)))
            corr_series = detector.pair_corr_samples.get(pair_key, [])
            sync_series = detector.sync_samples_by_pair.get(pair_key, [])
            pair_summaries.append(
                {
                    "pair": pair_key,
                    "matched_moves": len(matched),
                    "pearson_matched_returns": _safe_corr(deltas_a, deltas_b),
                    "same_direction_rate": None if not nonzero else round(sum(same_direction) / len(nonzero), 6),
                    "mean_abs_delta_a": round(statistics.fmean(abs(value) for value in deltas_a), 6),
                    "mean_abs_delta_b": round(statistics.fmean(abs(value) for value in deltas_b), 6),
                    "mean_runtime_corr": None if not corr_series else round(statistics.fmean(corr_series), 6),
                    "sync_p50_ms": None if not sync_series else round(_quantiles(sync_series)["p50"] or 0.0, 3),
                }
            )
    pair_summaries.sort(key=lambda item: ((item["pearson_matched_returns"] if item["pearson_matched_returns"] is not None else -2.0), item["matched_moves"]), reverse=True)

    corr_matrix_pairs = []
    for pair_a, pair_b in detector._pce.corr_matrix.all_pairs().keys():
        corr_matrix_pairs.append(
            {
                "pair": f"{pair_a}|{pair_b}",
                "corr": round(float(detector._pce.corr_matrix.get(pair_a, pair_b)), 6),
            }
        )
    corr_matrix_pairs.sort(key=lambda item: item["corr"], reverse=True)

    return {
        "champion_fold_dates": list(fold_zero["test_dates"]),
        "backtest_metrics": result.metrics.to_dict(),
        "detector_diagnostics": strategy.detector_diagnostics(),
        "leader_counts": {
            "evaluations_with_previous": detector.leader_evaluations_with_previous,
            "events_with_spike": detector.leader_events_with_spike,
            "events_with_impulse": detector.leader_events_with_impulse,
        },
        "sync_analysis": {
            "leader_yes_no": {
                "count": len(leader_book_values),
                "quantiles_ms": _quantiles(leader_book_values),
                "pass_rates": _pass_rates(leader_book_values, thresholds),
            },
            "leader_to_lagger": {
                "count": len(leader_pair_values),
                "quantiles_ms": _quantiles(leader_pair_values),
                "pass_rates": _pass_rates(leader_pair_values, thresholds),
            },
        },
        "pair_comovement": {
            "runtime_corr_matrix": corr_matrix_pairs,
            "matched_return_pairs": pair_summaries,
        },
    }


def _index_market_map(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        keys = [
            str(entry.get("condition_id") or "").strip(),
            str(entry.get("market_id") or "").strip(),
        ]
        for key in keys:
            if key:
                indexed[key] = entry
    return indexed


def _available_token_files(day_path: Path) -> set[str]:
    if not day_path.exists():
        return set()
    return {path.stem for path in day_path.glob("*.jsonl")}


def _cluster_candidates() -> dict[str, Any]:
    clusters = _load_json(CLUSTER_PATH)
    market_map = _load_json(MARKET_MAP_PATH)
    by_id = _index_market_map(market_map)
    day_files = {
        day: _available_token_files(DATA_DIR / "ticks" / day)
        for day in ("2026-03-02", "2026-03-03", "2026-03-04")
    }
    candidates: list[dict[str, Any]] = []
    for cluster in clusters:
        market_rows = []
        available_legs = 0
        for market in cluster.get("markets", []):
            condition_id = str(market.get("condition_id") or "").strip()
            mapped = by_id.get(condition_id, {})
            yes_id = str(mapped.get("yes_id") or mapped.get("yes_asset_id") or "").strip()
            no_id = str(mapped.get("no_id") or mapped.get("no_asset_id") or "").strip()
            coverage = {
                day: bool(yes_id and no_id and yes_id in files and no_id in files)
                for day, files in day_files.items()
            }
            if any(coverage.values()):
                available_legs += 1
            market_rows.append(
                {
                    "condition_id": condition_id,
                    "question": market.get("question"),
                    "group_item_title": market.get("group_item_title"),
                    "volume_24h": market.get("volume_24h"),
                    "liquidity": market.get("liquidity"),
                    "archive_coverage": coverage,
                }
            )
        leg_count = int(cluster.get("leg_count") or len(cluster.get("markets", [])) or 0)
        coverage_ratio = 0.0 if leg_count <= 0 else available_legs / leg_count
        candidate = {
            "event_id": cluster.get("event_id"),
            "title": cluster.get("title"),
            "leg_count": leg_count,
            "cluster_score": cluster.get("cluster_score"),
            "total_volume_24h": cluster.get("total_volume_24h"),
            "total_liquidity": cluster.get("total_liquidity"),
            "available_legs": available_legs,
            "coverage_ratio": round(coverage_ratio, 6),
            "markets": market_rows,
            "selection_score": round(float(cluster.get("cluster_score") or 0.0) * coverage_ratio * max(1, leg_count), 6),
        }
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item["coverage_ratio"], item["selection_score"], item["total_volume_24h"]), reverse=True)
    expanded = []
    market_total = 0
    for candidate in candidates:
        if candidate["coverage_ratio"] < 1.0:
            continue
        expanded.append(candidate)
        market_total += int(candidate["leg_count"])
        if market_total >= 15:
            break
    return {
        "top_candidates": candidates[:10],
        "recommended_expansion": expanded,
        "recommended_market_total": market_total,
    }


def main() -> None:
    output = {
        "replay_analysis": _build_replay_analysis(),
        "cluster_candidates": _cluster_candidates(),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
