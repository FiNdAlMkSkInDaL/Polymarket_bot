from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

from src.data.archive_market_analyzer import (
    build_yes_price_series,
    compute_lagged_pair_metrics,
    date_range,
    events_per_day,
    load_market_map_entries,
    parse_iso_datetime,
)


DEFAULT_ARCHIVE_PATH = Path("data/vps_march2026")
DEFAULT_MARKET_MAP_PATHS = (
    Path("data/market_map.json"),
    Path("data/domino_micro_fast_market_map.json"),
    Path("data/domino_micro_market_map.json"),
)


@dataclass(frozen=True, slots=True)
class MarketCandidate:
    market_id: str
    question: str
    thematic_tags: frozenset[str]
    expected_role: Literal["LEADER", "LAGGER", "EITHER"]


@dataclass(frozen=True, slots=True)
class UniverseBuilderConfig:
    min_correlation: Decimal
    min_events_per_day: int
    min_archive_days: int
    max_lagger_age_ms: int
    require_causal_ordering: bool


@dataclass(frozen=True, slots=True)
class ClusterEvaluationReport:
    candidates_evaluated: int
    pairs_passing_correlation: int
    pairs_passing_freshness: int
    pairs_passing_causal_ordering: int
    recommended_cluster: list[str]
    leader_market_id: str | None
    rejection_reasons: dict[str, str]
    empirical_correlations: dict[str, float]
    generated_at: str


class UniverseBuilder:
    def __init__(self, config: UniverseBuilderConfig) -> None:
        self.config = config
        self._validate_config(config)
        self._archive_path = DEFAULT_ARCHIVE_PATH
        self._market_map_entries = self._load_market_map_entries()

    def build_cluster(
        self,
        candidate_markets: list[MarketCandidate],
        eval_window_start: str,
        eval_window_end: str,
    ) -> ClusterEvaluationReport:
        start = parse_iso_datetime(eval_window_start)
        end = parse_iso_datetime(eval_window_end)
        market_candidates = {candidate.market_id: candidate for candidate in candidate_markets}
        selected_entries = [entry for entry in self._market_map_entries if entry["market_id"] in market_candidates]
        series_by_market = build_yes_price_series(
            str(self._archive_path),
            selected_entries,
            date_range(start, end),
        )

        rejection_reasons: dict[str, str] = {}
        eligible_markets: dict[str, MarketCandidate] = {}
        for candidate in candidate_markets:
            series = series_by_market.get(candidate.market_id)
            if series is None or not series.observations:
                rejection_reasons[candidate.market_id] = "no_archive_data"
                continue
            if len(series.days_observed) < self.config.min_archive_days:
                rejection_reasons[candidate.market_id] = "insufficient_archive_days"
                continue
            if events_per_day(series) < float(self.config.min_events_per_day):
                rejection_reasons[candidate.market_id] = "insufficient_events_per_day"
                continue
            eligible_markets[candidate.market_id] = candidate

        pairs_passing_correlation = 0
        pairs_passing_freshness = 0
        pairs_passing_causal_ordering = 0
        empirical_correlations: dict[str, float] = {}
        adjacency: dict[str, set[str]] = {market_id: set() for market_id in eligible_markets}

        for leader_id, leader_candidate in eligible_markets.items():
            if leader_candidate.expected_role == "LAGGER":
                continue
            for lagger_id, lagger_candidate in eligible_markets.items():
                if leader_id == lagger_id:
                    continue
                if lagger_candidate.expected_role == "LEADER":
                    continue
                if not (leader_candidate.thematic_tags & lagger_candidate.thematic_tags):
                    rejection_reasons.setdefault(lagger_id, "no_thematic_overlap")
                    continue
                metrics = compute_lagged_pair_metrics(
                    series_by_market[leader_id],
                    series_by_market[lagger_id],
                    freshness_ms=self.config.max_lagger_age_ms,
                    response_window_ms=self.config.max_lagger_age_ms,
                )
                pair_key = f"{leader_id}->{lagger_id}"
                empirical_correlations[pair_key] = round(float(metrics["correlation"]), 6)
                if float(metrics["correlation"]) < float(self.config.min_correlation):
                    rejection_reasons.setdefault(lagger_id, "correlation_below_threshold")
                    continue
                pairs_passing_correlation += 1

                if float(metrics["median_lagger_age_ms"]) > float(self.config.max_lagger_age_ms):
                    rejection_reasons.setdefault(lagger_id, "lagger_freshness_too_old")
                    continue
                pairs_passing_freshness += 1

                if self.config.require_causal_ordering and (
                    float(metrics["leader_to_lagger_strength"]) < float(metrics["lagger_to_leader_strength"])
                ):
                    rejection_reasons.setdefault(lagger_id, "causal_ordering_reversed")
                    continue
                pairs_passing_causal_ordering += 1
                adjacency[leader_id].add(lagger_id)

        leader_market_id = _choose_leader(adjacency, series_by_market, eligible_markets)
        recommended_cluster = [leader_market_id] if leader_market_id else []
        if leader_market_id:
            recommended_cluster.extend(sorted(adjacency.get(leader_market_id, set())))

        for market_id in eligible_markets:
            if market_id in recommended_cluster:
                rejection_reasons.pop(market_id, None)
            elif market_id not in rejection_reasons:
                rejection_reasons[market_id] = "not_selected_for_cluster"

        return ClusterEvaluationReport(
            candidates_evaluated=len(candidate_markets),
            pairs_passing_correlation=pairs_passing_correlation,
            pairs_passing_freshness=pairs_passing_freshness,
            pairs_passing_causal_ordering=pairs_passing_causal_ordering,
            recommended_cluster=recommended_cluster,
            leader_market_id=leader_market_id,
            rejection_reasons=rejection_reasons,
            empirical_correlations=empirical_correlations,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def export_cluster(
        self,
        report: ClusterEvaluationReport,
        output_path: str,
    ) -> None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    @staticmethod
    def _validate_config(config: UniverseBuilderConfig) -> None:
        if config.min_correlation < Decimal("0") or config.min_correlation > Decimal("1"):
            raise ValueError("min_correlation must be in [0, 1]")
        if config.min_events_per_day <= 0:
            raise ValueError("min_events_per_day must be > 0")
        if config.min_archive_days <= 0:
            raise ValueError("min_archive_days must be > 0")
        if config.max_lagger_age_ms < 0:
            raise ValueError("max_lagger_age_ms must be >= 0")

    @staticmethod
    def _load_market_map_entries() -> list[dict[str, str]]:
        for candidate in DEFAULT_MARKET_MAP_PATHS:
            if candidate.exists():
                entries = load_market_map_entries(candidate)
                if entries:
                    return entries
        return []


def _choose_leader(
    adjacency: dict[str, set[str]],
    series_by_market: dict[str, Any],
    eligible_markets: dict[str, MarketCandidate],
) -> str | None:
    if not adjacency:
        return None
    ranked = sorted(
        adjacency,
        key=lambda market_id: (
            len(adjacency.get(market_id, set())),
            events_per_day(series_by_market[market_id]) if market_id in series_by_market else 0.0,
            1 if eligible_markets[market_id].expected_role == "LEADER" else 0,
            market_id,
        ),
    )
    ranked.reverse()
    if not ranked:
        return None
    best = ranked[0]
    if not adjacency.get(best):
        return None
    return best
