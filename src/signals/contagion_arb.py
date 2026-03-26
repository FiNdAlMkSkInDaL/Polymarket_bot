"""L2 toxicity contagion arb detector for correlated Polymarket books.

Tracks the active L2 universe, groups markets by thematic tags, and converts
extreme directional toxicity on a leading market into an implied YES
probability for correlated lagging markets.  The final trigger uses the same
confidence-adjusted threshold math as the ResolutionProbabilityEngine.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.core.config import settings
from src.core.logger import get_logger
from src.signals.microstructure_utils import CausalLagAssessment, CausalLagGate, CrossBookSyncGate
from src.signals.resolution_probability import ResolutionProbabilityEngine

if TYPE_CHECKING:
    from src.data.market_discovery import MarketInfo
    from src.signals.signal_framework import SignalResult

log = get_logger(__name__)


def _normalise_tags(tags: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in (tags or "").split(","):
        tag = raw.strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        values.append(tag)
    return tuple(values)


def _clamp_probability(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


@dataclass(slots=True)
class ContagionSnapshot:
    market_id: str
    yes_price: float
    yes_buy_toxicity: float
    no_buy_toxicity: float
    timestamp: float
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ContagionArbSignal:
    leading_market_id: str
    lagging_market_id: str
    lagging_asset_id: str
    direction: str
    implied_probability: float
    lagging_market_price: float
    confidence: float
    correlation: float
    thematic_group: str
    toxicity_percentile: float
    leader_toxicity: float
    leader_price_shift: float
    expected_probability_shift: float
    timestamp: float
    score: float
    is_shadow: bool = False
    signal_source: str = "si10_contagion_arb"
    metadata: dict[str, Any] = field(default_factory=dict)


class ContagionArbDetector:
    """Tag-aware, toxicity-driven correlation arb detector.

    The detector is intentionally stateless across executions beyond a small
    rolling cache of toxicity samples and last-seen market prices.  Pairwise
    correlation estimates are delegated to the PortfolioCorrelationEngine.
    """

    _DEBUG_FORCE_MOVE_THRESHOLD = 0.01

    def __init__(
        self,
        pce: object,
        rpe: ResolutionProbabilityEngine,
        *,
        universe_size: int | None = None,
        min_correlation: float | None = None,
        trigger_percentile: float | None = None,
        min_history: int | None = None,
        min_leader_shift: float | None = None,
        min_residual_shift: float | None = None,
        toxicity_impulse_scale: float | None = None,
        cooldown_seconds: float | None = None,
        max_pairs_per_leader: int | None = None,
        shadow_mode: bool | None = None,
        max_cross_book_desync_ms: float | None = None,
        max_leader_age_ms: float | None = None,
        max_lagger_age_ms: float | None = None,
        max_causal_lag_ms: float | None = None,
        allow_negative_lag: bool | None = None,
        on_sync_block: Callable[[Any], None] | None = None,
    ) -> None:
        strat = settings.strategy
        self._pce = pce
        self._rpe = rpe
        self._universe_size = universe_size or strat.max_active_l2_markets
        self._min_corr = (
            min_correlation
            if min_correlation is not None
            else strat.contagion_arb_min_correlation
        )
        self._trigger_percentile = (
            trigger_percentile
            if trigger_percentile is not None
            else strat.contagion_arb_trigger_percentile
        )
        self._min_history = min_history or strat.contagion_arb_min_history
        self._min_leader_shift = (
            min_leader_shift
            if min_leader_shift is not None
            else strat.contagion_arb_min_leader_shift
        )
        self._min_residual_shift = (
            min_residual_shift
            if min_residual_shift is not None
            else strat.contagion_arb_min_residual_shift
        )
        self._toxicity_impulse_scale = (
            toxicity_impulse_scale
            if toxicity_impulse_scale is not None
            else strat.contagion_arb_toxicity_impulse_scale
        )
        self._cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else strat.contagion_arb_cooldown_seconds
        )
        self._max_pairs = max_pairs_per_leader or strat.contagion_arb_max_pairs_per_leader
        self._shadow = shadow_mode if shadow_mode is not None else strat.contagion_arb_shadow
        self._debug_force_signal = bool(strat.debug_force_contagion_signal)
        self._sync_gate = CrossBookSyncGate(max_cross_book_desync_ms)
        self._causal_lag_gate = CausalLagGate(
            max_leader_age_ms=(
                strat.contagion_arb_max_leader_age_ms
                if max_leader_age_ms is None
                else max_leader_age_ms
            ),
            max_lagger_age_ms=(
                strat.contagion_arb_max_lagger_age_ms
                if max_lagger_age_ms is None
                else max_lagger_age_ms
            ),
            max_causal_lag_ms=(
                strat.contagion_arb_max_causal_lag_ms
                if max_causal_lag_ms is None
                else max_causal_lag_ms
            ),
            allow_negative_lag=(
                strat.contagion_arb_allow_negative_lag
                if allow_negative_lag is None
                else allow_negative_lag
            ),
        )
        self._on_sync_block = on_sync_block

        self._markets: dict[str, MarketInfo] = {}
        self._snapshots: dict[str, ContagionSnapshot] = {}
        self._last_price_shift: dict[str, float] = {}
        self._toxicity_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=512))
        self._last_signal_at: dict[str, float] = {}
        self._diagnostics: dict[str, int] = {
            "evaluations_total": 0,
            "evaluations_with_previous_snapshot": 0,
            "toxicity_spikes_detected": 0,
            "reject_no_toxicity_spike": 0,
            "reject_insufficient_leader_impulse": 0,
            "reject_correlation_too_low": 0,
            "reject_residual_shift_too_small": 0,
            "cross_market_pairs_evaluated": 0,
            "legacy_sync_pairs_passed": 0,
            "accepted_causal_lag_count": 0,
            "reject_leader_snapshot_stale": 0,
            "reject_lagger_snapshot_stale": 0,
            "reject_lagger_newer_than_leader": 0,
            "reject_causal_lag_too_large": 0,
            "reject_missing_causal_timestamp": 0,
            "signals_emitted": 0,
            "forced_signals_emitted": 0,
        }
        self._top_leader_shift_samples: list[dict[str, Any]] = []
        self._top_toxicity_impulse_samples: list[dict[str, Any]] = []

    @property
    def shadow_mode(self) -> bool:
        return self._shadow

    def diagnostics_snapshot(self) -> dict[str, Any]:
        return {
            **self._diagnostics,
            "debug_force_contagion_signal": self._debug_force_signal,
            "debug_force_move_threshold": self._DEBUG_FORCE_MOVE_THRESHOLD,
            "causal_lag_config": {
                "max_leader_age_ms": self._causal_lag_gate.config.max_leader_age_ms,
                "max_lagger_age_ms": self._causal_lag_gate.config.max_lagger_age_ms,
                "max_causal_lag_ms": self._causal_lag_gate.config.max_causal_lag_ms,
                "allow_negative_lag": self._causal_lag_gate.config.allow_negative_lag,
            },
            "top_leader_shift_samples": [dict(sample) for sample in self._top_leader_shift_samples],
            "top_toxicity_impulse_samples": [dict(sample) for sample in self._top_toxicity_impulse_samples],
        }

    def register_market(self, market: MarketInfo) -> None:
        self._markets[market.condition_id] = market

    def unregister_market(self, condition_id: str) -> None:
        self._markets.pop(condition_id, None)
        self._snapshots.pop(condition_id, None)
        self._last_price_shift.pop(condition_id, None)
        self._last_signal_at.pop(condition_id, None)
        self._toxicity_history.pop(f"{condition_id}:buy_yes", None)
        self._toxicity_history.pop(f"{condition_id}:buy_no", None)

    def theme_groups(self, universe: list[MarketInfo] | None = None) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        candidates = universe[: self._universe_size] if universe else list(self._markets.values())[: self._universe_size]
        for market in candidates:
            for tag in _normalise_tags(getattr(market, "tags", "")):
                groups[tag].append(market.condition_id)
        return {tag: ids for tag, ids in groups.items() if len(ids) > 1}

    def evaluate_market(
        self,
        *,
        market: MarketInfo,
        yes_price: float,
        yes_buy_toxicity: float,
        no_buy_toxicity: float,
        timestamp: float | None = None,
        universe: list[MarketInfo] | None = None,
        book_snapshots: tuple[Any, ...] | None = None,
    ) -> list[ContagionArbSignal]:
        self._diagnostics["evaluations_total"] += 1
        sync_assessment = None
        if book_snapshots:
            sync_assessment = self._sync_gate.assess(book_snapshots)
            if not sync_assessment.is_synchronized:
                if self._on_sync_block is not None:
                    self._on_sync_block(sync_assessment)
                return []

        now = timestamp if timestamp is not None else time.time()
        if sync_assessment is not None and sync_assessment.latest_timestamp > 0.0:
            now = sync_assessment.latest_timestamp
        current_price = _clamp_probability(yes_price)
        tags = _normalise_tags(getattr(market, "tags", ""))

        previous = self._snapshots.get(market.condition_id)
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

        self._diagnostics["evaluations_with_previous_snapshot"] += 1

        if self._debug_force_signal:
            return self._evaluate_forced_signal(
                market=market,
                leader_snapshot=leader_snapshot,
                leader_shift=leader_shift,
                timestamp=now,
                universe=universe,
            )

        if not spikes:
            self._diagnostics["reject_no_toxicity_spike"] += 1
            return []
        self._diagnostics["toxicity_spikes_detected"] += 1

        signals: list[ContagionArbSignal] = []
        candidate_universe = universe[: self._universe_size] if universe else list(self._markets.values())[: self._universe_size]
        for direction, leader_toxicity, threshold in sorted(spikes, key=lambda row: row[1], reverse=True):
            direction_sign = 1.0 if direction == "buy_yes" else -1.0
            directional_leader_shift = direction_sign * leader_shift
            directional_move = max(0.0, directional_leader_shift)
            toxicity_impulse = max(0.0, leader_toxicity - threshold) * self._toxicity_impulse_scale
            self._record_top_sample(
                self._top_leader_shift_samples,
                directional_move,
                {
                    "market_id": market.condition_id,
                    "timestamp": round(now, 6),
                    "direction": direction,
                    "raw_leader_shift": round(leader_shift, 6),
                    "directional_leader_shift": round(directional_leader_shift, 6),
                    "leader_toxicity": round(leader_toxicity, 6),
                    "toxicity_threshold": round(threshold, 6),
                },
            )
            self._record_top_sample(
                self._top_toxicity_impulse_samples,
                toxicity_impulse,
                {
                    "market_id": market.condition_id,
                    "timestamp": round(now, 6),
                    "direction": direction,
                    "raw_leader_shift": round(leader_shift, 6),
                    "directional_leader_shift": round(directional_leader_shift, 6),
                    "leader_toxicity": round(leader_toxicity, 6),
                    "toxicity_threshold": round(threshold, 6),
                },
            )
            leader_impulse = max(directional_move, toxicity_impulse)
            if leader_impulse < self._min_leader_shift:
                self._diagnostics["reject_insufficient_leader_impulse"] += 1
                continue

            for lagger in candidate_universe:
                if lagger.condition_id == market.condition_id:
                    continue
                lag_snapshot = self._snapshots.get(lagger.condition_id)
                if lag_snapshot is None:
                    continue
                lag_causal_assessment = self._assess_causal_pair(
                    leader_snapshot,
                    lag_snapshot,
                    reference_timestamp=now,
                )
                if not lag_causal_assessment.is_valid:
                    continue

                correlation = self._pair_correlation(market.condition_id, lagger.condition_id)
                if correlation < self._min_corr:
                    self._diagnostics["reject_correlation_too_low"] += 1
                    continue

                thematic_group = self._shared_theme(market, lagger)
                if not thematic_group:
                    continue

                lag_directional_shift = max(
                    0.0,
                    direction_sign * self._last_price_shift.get(lagger.condition_id, 0.0),
                )
                expected_shift = correlation * leader_impulse
                residual_shift = expected_shift - lag_directional_shift
                if residual_shift < self._min_residual_shift:
                    self._diagnostics["reject_residual_shift_too_small"] += 1
                    continue

                if now - self._last_signal_at.get(lagger.condition_id, 0.0) < self._cooldown_seconds:
                    continue

                implied_probability = _clamp_probability(
                    lag_snapshot.yes_price + direction_sign * residual_shift
                )
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
                        "leader_age_ms": round(lag_causal_assessment.leader_age_ms, 3),
                        "lagger_age_ms": round(lag_causal_assessment.lagger_age_ms, 3),
                        "causal_lag_ms": round(lag_causal_assessment.causal_lag_ms, 3),
                        "causal_gate_result": lag_causal_assessment.gate_result,
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
                        metadata={
                            **dict(dislocation.metadata),
                            "leader_age_ms": round(lag_causal_assessment.leader_age_ms, 3),
                            "lagger_age_ms": round(lag_causal_assessment.lagger_age_ms, 3),
                            "causal_lag_ms": round(lag_causal_assessment.causal_lag_ms, 3),
                            "causal_gate_result": lag_causal_assessment.gate_result,
                        },
                    )
                )

        signals.sort(key=lambda item: (-item.score, -item.correlation, -item.leader_toxicity))
        return signals[: self._max_pairs]

    def _evaluate_forced_signal(
        self,
        *,
        market: MarketInfo,
        leader_snapshot: ContagionSnapshot,
        leader_shift: float,
        timestamp: float,
        universe: list[MarketInfo] | None,
    ) -> list[ContagionArbSignal]:
        if abs(leader_shift) < self._DEBUG_FORCE_MOVE_THRESHOLD:
            self._diagnostics["reject_insufficient_leader_impulse"] += 1
            return []

        direction = "buy_yes" if leader_shift > 0 else "buy_no"
        direction_sign = 1.0 if direction == "buy_yes" else -1.0
        leader_toxicity = (
            leader_snapshot.yes_buy_toxicity if direction == "buy_yes" else leader_snapshot.no_buy_toxicity
        )
        candidate_universe = universe[: self._universe_size] if universe else list(self._markets.values())[: self._universe_size]
        signals: list[ContagionArbSignal] = []
        for lagger in candidate_universe:
            if lagger.condition_id == market.condition_id:
                continue

            lag_snapshot = self._snapshots.get(lagger.condition_id)
            if lag_snapshot is None:
                continue

            lag_causal_assessment = self._assess_causal_pair(
                leader_snapshot,
                lag_snapshot,
                reference_timestamp=timestamp,
            )
            if not lag_causal_assessment.is_valid:
                continue

            thematic_group = self._shared_theme(market, lagger)
            if not thematic_group:
                continue

            if timestamp - self._last_signal_at.get(lagger.condition_id, 0.0) < self._cooldown_seconds:
                continue

            implied_probability = _clamp_probability(lag_snapshot.yes_price + leader_shift)
            lagging_asset_id = lagger.yes_token_id if direction == "buy_yes" else lagger.no_token_id
            correlation = self._pair_correlation(market.condition_id, lagger.condition_id)
            self._last_signal_at[lagger.condition_id] = timestamp
            self._diagnostics["signals_emitted"] += 1
            self._diagnostics["forced_signals_emitted"] += 1
            signals.append(
                ContagionArbSignal(
                    leading_market_id=market.condition_id,
                    lagging_market_id=lagger.condition_id,
                    lagging_asset_id=lagging_asset_id,
                    direction=direction,
                    implied_probability=implied_probability,
                    lagging_market_price=lag_snapshot.yes_price,
                    confidence=0.95,
                    correlation=correlation,
                    thematic_group=thematic_group,
                    toxicity_percentile=0.0,
                    leader_toxicity=leader_toxicity,
                    leader_price_shift=leader_shift,
                    expected_probability_shift=direction_sign * abs(leader_shift),
                    timestamp=timestamp,
                    score=1.0 + abs(leader_shift),
                    is_shadow=self._shadow,
                    signal_source="debug_force_contagion",
                    metadata={
                        "leading_market_id": market.condition_id,
                        "debug_forced": True,
                        "leader_price_shift": round(leader_shift, 6),
                        "thematic_group": thematic_group,
                        "correlation": round(correlation, 6),
                        "leader_age_ms": round(lag_causal_assessment.leader_age_ms, 3),
                        "lagger_age_ms": round(lag_causal_assessment.lagger_age_ms, 3),
                        "causal_lag_ms": round(lag_causal_assessment.causal_lag_ms, 3),
                        "causal_gate_result": lag_causal_assessment.gate_result,
                    },
                )
            )

        signals.sort(key=lambda item: (-item.score, -item.correlation, item.lagging_market_id))
        return signals[: self._max_pairs]

    def _candidate_laggers(
        self,
        leader: MarketInfo,
        universe: list[MarketInfo],
    ) -> list[MarketInfo]:
        ranked: list[tuple[float, MarketInfo]] = []
        for market in universe:
            if market.condition_id == leader.condition_id:
                continue
            corr = self._pair_correlation(leader.condition_id, market.condition_id)
            if corr < self._min_corr:
                continue
            if not self._shared_theme(leader, market):
                continue
            ranked.append((corr, market))
        ranked.sort(key=lambda row: (-row[0], row[1].condition_id))
        return [market for _, market in ranked[: self._max_pairs]]

    def _current_percentile(self, key: str) -> float | None:
        history = self._toxicity_history[key]
        if len(history) < self._min_history:
            return None
        ordered = sorted(history)
        idx = int(round((len(ordered) - 1) * self._trigger_percentile))
        idx = max(0, min(len(ordered) - 1, idx))
        return float(ordered[idx])

    def _pair_correlation(self, market_a: str, market_b: str) -> float:
        matrix = getattr(self._pce, "corr_matrix", None)
        if matrix is None or not hasattr(matrix, "get"):
            return 0.0
        return max(0.0, min(1.0, float(matrix.get(market_a, market_b))))

    def _assess_causal_pair(
        self,
        leader_snapshot: ContagionSnapshot,
        lag_snapshot: ContagionSnapshot,
        *,
        reference_timestamp: float,
    ) -> CausalLagAssessment:
        self._diagnostics["cross_market_pairs_evaluated"] += 1
        legacy_sync = self._sync_gate.assess((leader_snapshot, lag_snapshot))
        if legacy_sync.is_synchronized:
            self._diagnostics["legacy_sync_pairs_passed"] += 1

        assessment = self._causal_lag_gate.assess(
            leader_snapshot,
            lag_snapshot,
            reference_timestamp=reference_timestamp,
        )
        if assessment.is_valid:
            self._diagnostics["accepted_causal_lag_count"] += 1
            return assessment

        if assessment.gate_result == "leader_stale":
            self._diagnostics["reject_leader_snapshot_stale"] += 1
        elif assessment.gate_result == "lagger_stale":
            self._diagnostics["reject_lagger_snapshot_stale"] += 1
        elif assessment.gate_result == "lagger_newer_than_leader":
            self._diagnostics["reject_lagger_newer_than_leader"] += 1
        elif assessment.gate_result == "causal_lag_too_large":
            self._diagnostics["reject_causal_lag_too_large"] += 1
        else:
            self._diagnostics["reject_missing_causal_timestamp"] += 1
        return assessment

    def _record_top_sample(
        self,
        bucket: list[dict[str, Any]],
        magnitude: float,
        sample: dict[str, Any],
    ) -> None:
        candidate = {
            **sample,
            "observed_value": round(float(magnitude), 6),
        }
        bucket.append(candidate)
        bucket.sort(key=lambda item: float(item["observed_value"]), reverse=True)
        del bucket[5:]

    def _shared_theme(self, market_a: MarketInfo, market_b: MarketInfo) -> str:
        tags_a = set(_normalise_tags(getattr(market_a, "tags", "")))
        tags_b = set(_normalise_tags(getattr(market_b, "tags", "")))
        overlap = sorted(tags_a & tags_b)
        if overlap:
            return ",".join(overlap[:3])
        event_a = getattr(market_a, "event_id", "") or ""
        event_b = getattr(market_b, "event_id", "") or ""
        if event_a and event_a == event_b:
            return f"event:{event_a}"
        return ""