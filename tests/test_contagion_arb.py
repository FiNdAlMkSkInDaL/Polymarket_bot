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


def test_theme_groups_use_overlapping_tags() -> None:
    rpe = ResolutionProbabilityEngine(models=[], confidence_threshold=0.02, min_confidence=0.10)
    detector = ContagionArbDetector(FakePCE(0.8), rpe, shadow_mode=True)
    market_a = FakeMarketInfo("A", "Leader", "A_YES", "A_NO", tags="politics,elections,swing-state")
    market_b = FakeMarketInfo("B", "Lagger", "B_YES", "B_NO", tags="politics,elections,macro")
    detector.register_market(market_a)
    detector.register_market(market_b)

    groups = detector.theme_groups([market_a, market_b])

    assert groups["politics"] == ["A", "B"]
    assert groups["elections"] == ["A", "B"]


def test_contagion_signal_fires_on_buy_toxicity_spike() -> None:
    rpe = ResolutionProbabilityEngine(
        models=[],
        confidence_threshold=0.02,
        min_confidence=0.10,
        shadow_mode=False,
    )
    detector = ContagionArbDetector(
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
    )
    leader = FakeMarketInfo("LEAD", "Leader", "LEAD_YES", "LEAD_NO", tags="politics,elections,swing-state")
    lagger = FakeMarketInfo("LAG", "Lagger", "LAG_YES", "LAG_NO", tags="politics,elections,national")
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
    )

    assert len(signals) == 1
    signal = signals[0]
    assert signal.leading_market_id == "LEAD"
    assert signal.lagging_market_id == "LAG"
    assert signal.direction == "buy_yes"
    assert signal.implied_probability > signal.lagging_market_price
    assert signal.correlation > 0.5
    assert "politics" in signal.thematic_group


def test_contagion_signal_requires_shared_theme() -> None:
    rpe = ResolutionProbabilityEngine(models=[], confidence_threshold=0.02, min_confidence=0.10)
    detector = ContagionArbDetector(
        FakePCE(0.85),
        rpe,
        min_correlation=0.50,
        min_history=10,
        min_leader_shift=0.01,
        min_residual_shift=0.01,
        cooldown_seconds=0.0,
        shadow_mode=False,
    )
    leader = FakeMarketInfo("LEAD", "Leader", "LEAD_YES", "LEAD_NO", tags="politics,elections")
    lagger = FakeMarketInfo("LAG", "Lagger", "LAG_YES", "LAG_NO", event_id="OTHER", tags="crypto")
    detector.register_market(leader)
    detector.register_market(lagger)
    _seed_detector(detector, leader, lagger)

    signals = detector.evaluate_market(
        market=leader,
        yes_price=0.55,
        yes_buy_toxicity=0.98,
        no_buy_toxicity=0.05,
        timestamp=2_100.0,
        universe=[leader, lagger],
    )

    assert signals == []