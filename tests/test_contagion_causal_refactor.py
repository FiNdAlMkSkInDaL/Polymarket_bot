from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.signals.contagion_arb import ContagionArbDetector
from src.signals.resolution_probability import ResolutionProbabilityEngine


@dataclass
class FakeMarketInfo:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    daily_volume_usd: float = 50_000.0
    end_date: datetime | None = None
    active: bool = True
    event_id: str = "EVT"
    liquidity_usd: float = 10_000.0
    score: float = 70.0
    accepting_orders: bool = True
    tags: str = "politics,elections"

    def __post_init__(self) -> None:
        if self.end_date is None:
            self.end_date = datetime.now(timezone.utc) + timedelta(days=30)


class FakeCorrelationMatrix:
    def __init__(self, corr: float) -> None:
        self._corr = corr

    def get(self, market_a: str, market_b: str) -> float:
        if market_a == market_b:
            return 1.0
        return self._corr


class FakePCE:
    def __init__(self, corr: float) -> None:
        self.corr_matrix = FakeCorrelationMatrix(corr)


class _Snapshot:
    def __init__(self, timestamp: float, server_time: float = 0.0) -> None:
        self.timestamp = timestamp
        self.server_time = server_time


def _seed_detector(detector: ContagionArbDetector, leader: FakeMarketInfo, lagger: FakeMarketInfo) -> None:
    for idx in range(30):
        base = 0.50 + idx * 0.0005
        detector.evaluate_market(
            market=leader,
            yes_price=base,
            yes_buy_toxicity=0.20 + idx * 0.002,
            no_buy_toxicity=0.10,
            timestamp=1_000.0 + idx,
            universe=[leader, lagger],
        )
        detector.evaluate_market(
            market=lagger,
            yes_price=0.50 + idx * 0.0002,
            yes_buy_toxicity=0.18,
            no_buy_toxicity=0.12,
            timestamp=1_000.5 + idx,
            universe=[leader, lagger],
        )


def _build_detector(*, on_sync_block=None, max_cross_book_desync_ms: float = 400.0) -> ContagionArbDetector:
    rpe = ResolutionProbabilityEngine(
        models=[],
        confidence_threshold=0.02,
        min_confidence=0.10,
        shadow_mode=False,
    )
    return ContagionArbDetector(
        FakePCE(0.82),
        rpe,
        min_correlation=0.50,
        trigger_percentile=0.95,
        min_history=20,
        min_leader_shift=0.01,
        min_residual_shift=0.01,
        toxicity_impulse_scale=0.05,
        cooldown_seconds=0.0,
        max_pairs_per_leader=2,
        shadow_mode=False,
        max_cross_book_desync_ms=max_cross_book_desync_ms,
        max_leader_age_ms=5000.0,
        max_lagger_age_ms=30000.0,
        max_causal_lag_ms=600000.0,
        allow_negative_lag=False,
        on_sync_block=on_sync_block,
    )


def test_contagion_cross_market_gate_accepts_causal_lag_without_simultaneity() -> None:
    detector = _build_detector()
    leader = FakeMarketInfo("LEAD", "Leader", "LEAD_YES", "LEAD_NO", tags="politics,elections,swing-state")
    lagger = FakeMarketInfo("LAG", "Lagger", "LAG_YES", "LAG_NO", tags="politics,elections,national")
    detector.register_market(leader)
    detector.register_market(lagger)
    _seed_detector(detector, leader, lagger)

    detector.evaluate_market(
        market=lagger,
        yes_price=0.506,
        yes_buy_toxicity=0.18,
        no_buy_toxicity=0.12,
        timestamp=1_980.0,
        universe=[leader, lagger],
        book_snapshots=(
            _Snapshot(1_980.0),
            _Snapshot(1_980.05),
        ),
    )

    signals = detector.evaluate_market(
        market=leader,
        yes_price=0.54,
        yes_buy_toxicity=0.97,
        no_buy_toxicity=0.08,
        timestamp=2_000.0,
        universe=[leader, lagger],
        book_snapshots=(
            _Snapshot(2_000.0),
            _Snapshot(2_000.05),
        ),
    )

    assert len(signals) == 1
    signal = signals[0]
    assert signal.metadata["causal_gate_result"] == "accepted"
    assert signal.metadata["lagger_age_ms"] == 20000.0
    assert signal.metadata["causal_lag_ms"] == 20000.0

    diagnostics = detector.diagnostics_snapshot()
    assert diagnostics["legacy_sync_pairs_passed"] == 0
    assert diagnostics["accepted_causal_lag_count"] >= 1


def test_contagion_same_market_sync_gate_still_blocks_desynced_books() -> None:
    sync_blocks: list[float] = []
    detector = _build_detector(on_sync_block=lambda assessment: sync_blocks.append(assessment.delta_ms))
    leader = FakeMarketInfo("LEAD", "Leader", "LEAD_YES", "LEAD_NO", tags="politics,elections")
    lagger = FakeMarketInfo("LAG", "Lagger", "LAG_YES", "LAG_NO", tags="politics,elections")
    detector.register_market(leader)
    detector.register_market(lagger)
    _seed_detector(detector, leader, lagger)

    signals = detector.evaluate_market(
        market=leader,
        yes_price=0.54,
        yes_buy_toxicity=0.97,
        no_buy_toxicity=0.08,
        timestamp=2_000.0,
        universe=[leader, lagger],
        book_snapshots=(
            _Snapshot(2_000.0),
            _Snapshot(2_001.0),
        ),
    )

    assert signals == []
    assert len(sync_blocks) == 1
    assert sync_blocks[0] == 1000.0