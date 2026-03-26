from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.ctf_unwind_manifest import CtfUnwindManifest
from src.execution.live_escalation_policy import LiveEscalationPolicy
from src.execution.live_unwind_cost_estimator import LiveUnwindCostEstimator
from src.execution.orphaned_leg_scanner import OrphanedLegRecoveryScanner, RecoveryOpenPosition
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindManifest


class _StubTracker:
    def __init__(self, *, asset_id: str, best_bid: float, timestamp: float = 1712345.678) -> None:
        self.asset_id = asset_id
        self.best_bid = best_bid
        self._timestamp = timestamp

    def snapshot(self):
        class _Snapshot:
            def __init__(self, timestamp: float) -> None:
                self.timestamp = timestamp

        return _Snapshot(self._timestamp)


def _config() -> Si9UnwindConfig:
    return Si9UnwindConfig(
        market_sell_threshold=Decimal("0.050000"),
        passive_unwind_threshold=Decimal("0.010000"),
        max_hold_recovery_ms=100,
        min_best_bid=Decimal("0.010000"),
    )


def _scanner() -> OrphanedLegRecoveryScanner:
    return OrphanedLegRecoveryScanner(
        _config(),
        si9_max_leg_fill_wait_ms=250,
        si9_cancel_on_stale_ms=500,
        ctf_cancel_on_stale_ms=125,
    )


def _providers(best_bids: dict[str, float]) -> dict[str, OrderbookBestBidProvider]:
    return {
        market_id: OrderbookBestBidProvider(_StubTracker(asset_id=market_id, best_bid=best_bid))
        for market_id, best_bid in best_bids.items()
    }


def _position(
    *,
    coordination_id: str,
    strategy_source: str,
    market_id: str,
    side: str,
    filled_size: str,
    filled_price: str,
    venue_timestamp_ms: int,
    expected_leg_count: int,
    leg_index: int,
) -> RecoveryOpenPosition:
    return RecoveryOpenPosition(
        coordination_id=coordination_id,
        strategy_source=strategy_source,  # type: ignore[arg-type]
        market_id=market_id,
        side=side,  # type: ignore[arg-type]
        filled_size=Decimal(filled_size),
        filled_price=Decimal(filled_price),
        venue_timestamp_ms=venue_timestamp_ms,
        expected_leg_count=expected_leg_count,
        leg_index=leg_index,
    )


def test_recovery_position_requires_non_empty_coordination_id() -> None:
    with pytest.raises(ValueError, match="coordination_id"):
        _position(
            coordination_id="",
            strategy_source="CTF",
            market_id="ctf-1",
            side="YES",
            filled_size="1",
            filled_price="0.40",
            venue_timestamp_ms=100,
            expected_leg_count=2,
            leg_index=0,
        )


def test_scan_ignores_balanced_ctf_pair() -> None:
    manifests = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="YES",
                filled_size="3.000000",
                filled_price="0.41",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            ),
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="NO",
                filled_size="3.000000",
                filled_price="0.39",
                venue_timestamp_ms=102,
                expected_leg_count=2,
                leg_index=1,
            ),
        ],
        150,
    )

    assert manifests == ()


def test_scan_ignores_balanced_si9_cluster() -> None:
    manifests = _scanner().scan(
        [
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-a",
                side="YES",
                filled_size="2.000000",
                filled_price="0.33",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            ),
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-b",
                side="YES",
                filled_size="2.000000",
                filled_price="0.31",
                venue_timestamp_ms=101,
                expected_leg_count=2,
                leg_index=1,
            ),
        ],
        150,
    )

    assert manifests == ()


def test_ctf_yes_only_orphan_produces_hold_recovery_manifest() -> None:
    manifests = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="YES",
                filled_size="3.000000",
                filled_price="0.41",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            )
        ],
        180,
    )

    assert len(manifests) == 1
    manifest = manifests[0]
    assert isinstance(manifest, CtfUnwindManifest)
    assert manifest.recommended_action == "HOLD_FOR_RECOVERY"
    assert manifest.hanging_legs[0].side == "YES"
    assert manifest.hanging_legs[0].filled_size == Decimal("3.000000")
    assert manifest.unwind_timestamp_ms == 100


def test_ctf_stale_orphan_seeds_market_sell() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="NO",
                filled_size="2.000000",
                filled_price="0.39",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=1,
            )
        ],
        201,
    )[0]

    assert isinstance(manifest, CtfUnwindManifest)
    assert manifest.recommended_action == "MARKET_SELL"


def test_ctf_asymmetric_pair_only_unwinds_residual_leg_size() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="YES",
                filled_size="5.000000",
                filled_price="0.42",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            ),
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="NO",
                filled_size="3.000000",
                filled_price="0.38",
                venue_timestamp_ms=110,
                expected_leg_count=2,
                leg_index=1,
            ),
        ],
        150,
    )[0]

    assert isinstance(manifest, CtfUnwindManifest)
    assert len(manifest.hanging_legs) == 1
    assert manifest.hanging_legs[0].side == "YES"
    assert manifest.hanging_legs[0].filled_size == Decimal("2.000000")


def test_ctf_aggregation_combines_duplicate_side_positions_with_weighted_price() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="YES",
                filled_size="1.500000",
                filled_price="0.40",
                venue_timestamp_ms=120,
                expected_leg_count=2,
                leg_index=0,
            ),
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="YES",
                filled_size="2.500000",
                filled_price="0.46",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            ),
        ],
        150,
    )[0]

    assert isinstance(manifest, CtfUnwindManifest)
    assert manifest.hanging_legs[0].filled_size == Decimal("4.000000")
    assert manifest.hanging_legs[0].filled_price == Decimal("0.437500")
    assert manifest.unwind_timestamp_ms == 100


def test_si9_missing_leg_unwinds_all_present_legs() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-a",
                side="YES",
                filled_size="2.000000",
                filled_price="0.33",
                venue_timestamp_ms=100,
                expected_leg_count=3,
                leg_index=0,
            ),
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-b",
                side="YES",
                filled_size="2.000000",
                filled_price="0.31",
                venue_timestamp_ms=110,
                expected_leg_count=3,
                leg_index=1,
            ),
        ],
        150,
    )[0]

    assert isinstance(manifest, Si9UnwindManifest)
    assert manifest.unwind_reason == "MANUAL_ABORT"
    assert [leg.market_id for leg in manifest.hanging_legs] == ["mkt-a", "mkt-b"]
    assert all(leg.filled_size == Decimal("2.000000") for leg in manifest.hanging_legs)


def test_si9_asymmetric_cluster_only_unwinds_residual_above_minimum_fill() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-a",
                side="YES",
                filled_size="5.000000",
                filled_price="0.33",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            ),
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-b",
                side="YES",
                filled_size="3.000000",
                filled_price="0.31",
                venue_timestamp_ms=105,
                expected_leg_count=2,
                leg_index=1,
            ),
        ],
        150,
    )[0]

    assert isinstance(manifest, Si9UnwindManifest)
    assert len(manifest.hanging_legs) == 1
    assert manifest.hanging_legs[0].market_id == "mkt-a"
    assert manifest.hanging_legs[0].filled_size == Decimal("2.000000")


def test_si9_aggregation_combines_duplicate_market_positions_weighted_by_notional() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-a",
                side="YES",
                filled_size="1.000000",
                filled_price="0.30",
                venue_timestamp_ms=120,
                expected_leg_count=3,
                leg_index=0,
            ),
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-a",
                side="YES",
                filled_size="2.000000",
                filled_price="0.36",
                venue_timestamp_ms=100,
                expected_leg_count=3,
                leg_index=0,
            ),
        ],
        150,
    )[0]

    assert isinstance(manifest, Si9UnwindManifest)
    assert manifest.hanging_legs[0].market_id == "mkt-a"
    assert manifest.hanging_legs[0].filled_size == Decimal("3.000000")
    assert manifest.hanging_legs[0].filled_price == Decimal("0.340000")
    assert manifest.unwind_timestamp_ms == 100


def test_scan_groups_multiple_coordination_ids_independently() -> None:
    manifests = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="YES",
                filled_size="1.000000",
                filled_price="0.40",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            ),
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-a",
                side="YES",
                filled_size="1.500000",
                filled_price="0.33",
                venue_timestamp_ms=101,
                expected_leg_count=2,
                leg_index=0,
            ),
        ],
        150,
    )

    assert len(manifests) == 2
    assert {manifest.cluster_id for manifest in manifests} == {"ctf-1", "cluster-1"}


def test_ctf_hanging_legs_are_sorted_by_leg_index() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="NO",
                filled_size="4.000000",
                filled_price="0.39",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=1,
            ),
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="YES",
                filled_size="6.000000",
                filled_price="0.41",
                venue_timestamp_ms=110,
                expected_leg_count=2,
                leg_index=0,
            ),
        ],
        150,
    )[0]

    assert isinstance(manifest, CtfUnwindManifest)
    assert [leg.leg_index for leg in manifest.hanging_legs] == [0]


def test_scanner_produced_si9_manifest_is_compatible_with_live_escalation_policy() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="cluster-1",
                strategy_source="SI9",
                market_id="mkt-a",
                side="YES",
                filled_size="2.000000",
                filled_price="0.40",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=0,
            )
        ],
        150,
    )[0]
    policy = LiveEscalationPolicy(
        LiveUnwindCostEstimator(_providers({"mkt-a": 0.399})),
        _config(),
    )

    escalated = policy.escalate_manifest(manifest, 150)

    assert isinstance(escalated, Si9UnwindManifest)
    assert escalated.recommended_action == "PASSIVE_UNWIND"


def test_scanner_produced_ctf_manifest_is_compatible_with_live_escalation_policy() -> None:
    manifest = _scanner().scan(
        [
            _position(
                coordination_id="ctf-1",
                strategy_source="CTF",
                market_id="ctf-1",
                side="NO",
                filled_size="3.000000",
                filled_price="0.39",
                venue_timestamp_ms=100,
                expected_leg_count=2,
                leg_index=1,
            )
        ],
        150,
    )[0]
    policy = LiveEscalationPolicy(
        LiveUnwindCostEstimator(_providers({"ctf-1": 0.389})),
        _config(),
    )

    escalated = policy.escalate_manifest(manifest, 150)

    assert isinstance(escalated, CtfUnwindManifest)
    assert escalated.recommended_action == "PASSIVE_UNWIND"