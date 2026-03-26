from __future__ import annotations

from decimal import Decimal

import pytest

from src.detectors.si9_cluster_config import Si9ClusterConfig
from src.signals.si9_matrix_detector import (
    Si9MatrixDetector,
    Si9RawClusterSnapshot,
    Si9TradeableSnapshot,
)


def _config(**overrides: object) -> Si9ClusterConfig:
    values = {
        "target_yield": Decimal("0.02"),
        "taker_fee_per_leg": Decimal("0.002"),
        "slippage_budget": Decimal("0.001"),
        "ghost_town_floor": Decimal("0.85"),
        "implausible_edge_ceil": Decimal("0.15"),
        "max_ask_age_ms": 100,
        "min_cluster_size": 3,
        "tiebreak_policy": "lowest_market_id",
    }
    values.update(overrides)
    return Si9ClusterConfig(**values)


def _detector(
    config: Si9ClusterConfig | None = None,
    cluster_id: str = "cluster-1",
    market_ids: tuple[str, ...] = ("mkt-a", "mkt-b", "mkt-c"),
) -> Si9MatrixDetector:
    detector = Si9MatrixDetector(config or _config(min_cluster_size=len(market_ids)))
    detector.register_cluster(cluster_id, list(market_ids))
    return detector


def _update(
    detector: Si9MatrixDetector,
    market_id: str,
    ask_price: str,
    ask_size: str,
    timestamp_ms: int,
) -> None:
    detector.update_best_yes_ask(
        market_id,
        Decimal(ask_price),
        Decimal(ask_size),
        timestamp_ms,
    )


def _seed_cluster(
    detector: Si9MatrixDetector,
    prices: tuple[str, ...],
    sizes: tuple[str, ...],
    timestamp_ms: int,
    market_ids: tuple[str, ...] = ("mkt-a", "mkt-b", "mkt-c"),
) -> None:
    for market_id, ask_price, ask_size in zip(market_ids, prices, sizes, strict=True):
        _update(detector, market_id, ask_price, ask_size, timestamp_ms)


def test_evaluate_cluster_emits_signal_when_yes_asks_sum_to_0_96() -> None:
    detector = _detector()
    _seed_cluster(
        detector,
        prices=("0.31", "0.32", "0.33"),
        sizes=("1", "1", "1"),
        timestamp_ms=100,
    )

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is not None
    assert signal.cluster_id == "cluster-1"
    assert signal.market_ids == ("mkt-a", "mkt-b", "mkt-c")
    assert signal.total_yes_ask == Decimal("0.96")
    assert signal.gross_edge == Decimal("0.04")
    assert signal.required_share_counts == Decimal("1")


def test_evaluate_cluster_remains_silent_when_yes_asks_sum_to_1_02() -> None:
    detector = _detector()
    _seed_cluster(
        detector,
        prices=("0.34", "0.34", "0.34"),
        sizes=("1", "1", "1"),
        timestamp_ms=100,
    )

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is None


def test_si9_cluster_config_valid_construction_passes() -> None:
    config = _config()

    assert config.target_yield == Decimal("0.02")
    assert config.taker_fee_per_leg == Decimal("0.002")
    assert config.slippage_budget == Decimal("0.001")
    assert config.tiebreak_policy == "lowest_market_id"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_yield": Decimal("-0.01")}, "target_yield"),
        ({"target_yield": Decimal("1.00")}, "target_yield"),
        ({"taker_fee_per_leg": Decimal("0")}, "taker_fee_per_leg"),
        ({"slippage_budget": Decimal("-0.001")}, "slippage_budget"),
        ({"ghost_town_floor": Decimal("1.00")}, "ghost_town_floor"),
        ({"implausible_edge_ceil": Decimal("0")}, "implausible_edge_ceil"),
        ({"max_ask_age_ms": -1}, "max_ask_age_ms"),
        ({"min_cluster_size": 1}, "min_cluster_size"),
        ({"tiebreak_policy": "nondeterministic"}, "tiebreak_policy"),
    ],
)
def test_si9_cluster_config_invalid_fields_raise_value_error(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_fee_deduction_uses_net_edge_after_costs() -> None:
    detector = _detector()
    _seed_cluster(
        detector,
        prices=("0.31", "0.32", "0.33"),
        sizes=("2", "2", "2"),
        timestamp_ms=100,
    )

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is not None
    assert signal.net_edge.quantize(Decimal("0.000001")) == Decimal("0.031000")


def test_depth_bounded_sizing_uses_shallowest_leg() -> None:
    market_ids = ("mkt-1", "mkt-2", "mkt-3", "mkt-4", "mkt-5")
    detector = _detector(_config(min_cluster_size=5), market_ids=market_ids)
    _seed_cluster(
        detector,
        prices=("0.18", "0.19", "0.19", "0.19", "0.20"),
        sizes=("4", "5", "2.5", "6", "7"),
        timestamp_ms=100,
        market_ids=market_ids,
    )

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is not None
    assert signal.bottleneck_market_id == "mkt-3"
    assert signal.required_share_counts == Decimal("2.5")


def test_bottleneck_tie_break_prefers_higher_ask_price() -> None:
    market_ids = ("mkt-1", "mkt-2", "mkt-3", "mkt-4", "mkt-5")
    detector = _detector(_config(min_cluster_size=5), market_ids=market_ids)
    _seed_cluster(
        detector,
        prices=("0.18", "0.19", "0.18", "0.21", "0.19"),
        sizes=("4", "2", "5", "2", "6"),
        timestamp_ms=100,
        market_ids=market_ids,
    )

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is not None
    assert signal.bottleneck_market_id == "mkt-4"
    assert signal.required_share_counts == Decimal("2")


def test_tiebreak_policy_lowest_market_id_selects_lexicographic_leg() -> None:
    market_ids = ("mkt-b", "mkt-a", "mkt-c")
    detector = _detector(
        _config(min_cluster_size=3, tiebreak_policy="lowest_market_id"),
        market_ids=market_ids,
    )
    _seed_cluster(
        detector,
        prices=("0.30", "0.30", "0.30"),
        sizes=("1", "1", "2"),
        timestamp_ms=100,
        market_ids=market_ids,
    )

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is not None
    assert signal.bottleneck_market_id == "mkt-a"


def test_tiebreak_policy_stable_index_preserves_registration_order() -> None:
    market_ids = ("mkt-b", "mkt-a", "mkt-c")
    detector = _detector(
        _config(min_cluster_size=3, tiebreak_policy="stable_index"),
        market_ids=market_ids,
    )
    _seed_cluster(
        detector,
        prices=("0.30", "0.30", "0.30"),
        sizes=("1", "1", "2"),
        timestamp_ms=100,
        market_ids=market_ids,
    )

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is not None
    assert signal.bottleneck_market_id == "mkt-b"


def test_staleness_suppression_kills_entire_cluster_signal() -> None:
    detector = _detector(_config(max_ask_age_ms=100))
    _seed_cluster(
        detector,
        prices=("0.31", "0.32", "0.33"),
        sizes=("2", "2", "2"),
        timestamp_ms=0,
    )
    _update(detector, "mkt-b", "0.32", "2", 100)
    _update(detector, "mkt-c", "0.33", "2", 100)

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=101)

    assert signal is None


def test_staleness_boundary_allows_exact_limit_and_rejects_limit_plus_one() -> None:
    detector = _detector(_config(max_ask_age_ms=100))
    _seed_cluster(
        detector,
        prices=("0.31", "0.32", "0.33"),
        sizes=("2", "2", "2"),
        timestamp_ms=0,
    )

    boundary_signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)
    stale_signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=101)

    assert boundary_signal is not None
    assert stale_signal is None


def test_ghost_town_gate_fires_before_fee_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _detector(_config(ghost_town_floor=Decimal("0.90")))
    _seed_cluster(
        detector,
        prices=("0.29", "0.30", "0.30"),
        sizes=("2", "2", "2"),
        timestamp_ms=100,
    )

    def _explode(self: Si9MatrixDetector, total_yes_ask: Decimal, leg_count: int) -> Decimal:
        raise AssertionError("fee calculation should not run")

    monkeypatch.setattr(Si9MatrixDetector, "_compute_net_edge", _explode)

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is None


def test_implausible_edge_gate_fires_before_fee_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _detector(
        _config(
            ghost_town_floor=Decimal("0.80"),
            implausible_edge_ceil=Decimal("0.10"),
        )
    )
    _seed_cluster(
        detector,
        prices=("0.28", "0.28", "0.29"),
        sizes=("2", "2", "2"),
        timestamp_ms=100,
    )

    def _explode(self: Si9MatrixDetector, total_yes_ask: Decimal, leg_count: int) -> Decimal:
        raise AssertionError("fee calculation should not run")

    monkeypatch.setattr(Si9MatrixDetector, "_compute_net_edge", _explode)

    signal = detector.evaluate_cluster("cluster-1", eval_timestamp_ms=100)

    assert signal is None


def test_cluster_snapshot_returns_raw_snapshot_with_tradeable_metadata() -> None:
    market_ids = ("mkt-1", "mkt-2", "mkt-3", "mkt-4", "mkt-5")
    detector = _detector(_config(min_cluster_size=5), market_ids=market_ids)
    _seed_cluster(
        detector,
        prices=("0.18", "0.19", "0.19", "0.19", "0.20"),
        sizes=("4", "5", "2.5", "6", "7"),
        timestamp_ms=100,
        market_ids=market_ids,
    )

    snapshot = detector.cluster_snapshot("cluster-1")

    assert isinstance(snapshot, Si9RawClusterSnapshot)
    assert snapshot.cluster_id == "cluster-1"
    assert snapshot.cluster_ask_sum == Decimal("0.95")
    assert snapshot.net_edge == Decimal("0.035")
    assert snapshot.bottleneck_market_id == "mkt-3"
    assert snapshot.would_be_tradeable is True
    assert snapshot.suppression_reason is None
    assert snapshot.legs[2].market_id == "mkt-3"
    assert snapshot.legs[2].ask_size == Decimal("2.5")


def test_tradeable_snapshot_returns_valid_snapshot_on_clean_cluster() -> None:
    detector = _detector()
    _seed_cluster(
        detector,
        prices=("0.31", "0.32", "0.33"),
        sizes=("2", "2", "2"),
        timestamp_ms=100,
    )

    snapshot = detector.tradeable_snapshot("cluster-1", eval_timestamp_ms=100)

    assert isinstance(snapshot, Si9TradeableSnapshot)
    assert snapshot is not None
    assert snapshot.tradeable_at_ms == 100
    assert snapshot.required_share_counts == Decimal("2")


def test_tradeable_snapshot_returns_none_on_stale_cluster() -> None:
    detector = _detector(_config(max_ask_age_ms=50))
    _seed_cluster(
        detector,
        prices=("0.31", "0.32", "0.33"),
        sizes=("2", "2", "2"),
        timestamp_ms=0,
    )

    snapshot = detector.tradeable_snapshot("cluster-1", eval_timestamp_ms=51)
    assert snapshot is None


def test_cluster_snapshot_reports_stale_leg_suppression_reason() -> None:
    detector = _detector(_config(max_ask_age_ms=50))
    _seed_cluster(
        detector,
        prices=("0.31", "0.32", "0.33"),
        sizes=("2", "2", "2"),
        timestamp_ms=0,
    )
    _update(detector, "mkt-b", "0.32", "2", 51)
    _update(detector, "mkt-c", "0.33", "2", 51)

    snapshot = detector.cluster_snapshot("cluster-1")

    assert snapshot.would_be_tradeable is False
    assert snapshot.suppression_reason == "STALE_LEG"


def test_cluster_snapshot_reports_ghost_town_suppression_reason() -> None:
    detector = _detector(_config(ghost_town_floor=Decimal("0.90")))
    _seed_cluster(
        detector,
        prices=("0.29", "0.30", "0.30"),
        sizes=("2", "2", "2"),
        timestamp_ms=100,
    )

    snapshot = detector.cluster_snapshot("cluster-1")

    assert snapshot.would_be_tradeable is False
    assert snapshot.suppression_reason == "GHOST_TOWN"


def test_cluster_snapshot_reports_incomplete_cluster() -> None:
    detector = _detector()
    _update(detector, "mkt-a", "0.31", "2", 100)

    snapshot = detector.cluster_snapshot("cluster-1")

    assert snapshot.would_be_tradeable is False
    assert snapshot.suppression_reason == "INCOMPLETE_CLUSTER"
    assert snapshot.cluster_ask_sum is None


def test_invalid_ask_price_is_rejected_without_state_mutation() -> None:
    detector = _detector()
    _update(detector, "mkt-a", "0.31", "2", 100)
    initial_state = detector.top_of_book_by_market["mkt-a"]

    detector.update_best_yes_ask("mkt-a", Decimal("1.01"), Decimal("3"), 200)

    assert detector.top_of_book_by_market["mkt-a"] == initial_state


def test_zero_ask_size_is_rejected_without_state_mutation() -> None:
    detector = _detector()
    _update(detector, "mkt-a", "0.31", "2", 100)
    initial_state = detector.top_of_book_by_market["mkt-a"]

    detector.update_best_yes_ask("mkt-a", Decimal("0.32"), Decimal("0"), 200)

    assert detector.top_of_book_by_market["mkt-a"] == initial_state
