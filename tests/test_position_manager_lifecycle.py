from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.position_lifecycle_interface import PositionLifecycleInterface
from src.execution.position_manager_lifecycle import PositionManagerLifecycle
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest


class _FakePositionManager:
    def __init__(self, *, max_open: int = 3, open_positions: list[object] | None = None) -> None:
        self.max_open = max_open
        self._open_positions = list(open_positions or [])
        self.cleanup_calls = 0

    def get_open_positions(self) -> list[object]:
        return list(self._open_positions)

    def cleanup_closed(self) -> list[object]:
        self.cleanup_calls += 1
        return []


class _RaisingPositionManager(_FakePositionManager):
    def get_open_positions(self) -> list[object]:
        raise RuntimeError("boom")


class _CleanupRaisingPositionManager(_FakePositionManager):
    def cleanup_closed(self) -> list[object]:
        self.cleanup_calls += 1
        raise RuntimeError("boom")


def _manifest(cluster_id: str = "cluster-1") -> Si9ExecutionManifest:
    return Si9ExecutionManifest(
        cluster_id=cluster_id,
        legs=(
            Si9LegManifest("mkt-a", "YES", Decimal("0.31"), Decimal("2"), True, 0),
            Si9LegManifest("mkt-b", "YES", Decimal("0.32"), Decimal("2"), False, 1),
        ),
        net_edge=Decimal("0.02"),
        required_share_counts=Decimal("2"),
        bottleneck_market_id="mkt-a",
        manifest_timestamp_ms=100,
        max_leg_fill_wait_ms=200,
        cancel_on_stale_ms=300,
    )


def test_lifecycle_satisfies_position_lifecycle_interface_abc() -> None:
    assert isinstance(PositionManagerLifecycle(_FakePositionManager()), PositionLifecycleInterface)


def test_constructor_requires_position_manager() -> None:
    with pytest.raises(ValueError, match="position_manager"):
        PositionManagerLifecycle(None)  # type: ignore[arg-type]


def test_reserve_position_succeeds_under_capacity() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager(max_open=3, open_positions=[object()]))

    assert lifecycle.reserve_position("cluster-1", _manifest(), 1000) is True


def test_reserve_position_reuses_existing_cluster_reservation() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager(max_open=1))

    assert lifecycle.reserve_position("cluster-1", _manifest(), 1000) is True
    assert lifecycle.reserve_position("cluster-1", _manifest(), 1001) is True
    assert lifecycle.active_position_count == 1


def test_reserve_position_rejects_blank_cluster_id() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager())

    assert lifecycle.reserve_position("   ", _manifest(), 1000) is False


def test_reserve_position_rejects_when_manager_is_at_capacity() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager(max_open=2, open_positions=[object(), object()]))

    assert lifecycle.reserve_position("cluster-1", _manifest(), 1000) is False


def test_reserve_position_counts_local_reservations_conservatively() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager(max_open=2, open_positions=[object()]))

    assert lifecycle.reserve_position("cluster-1", _manifest("cluster-1"), 1000) is True
    assert lifecycle.reserve_position("cluster-2", _manifest("cluster-2"), 1001) is False


def test_reserve_position_rejects_when_manager_open_positions_cannot_be_read() -> None:
    lifecycle = PositionManagerLifecycle(_RaisingPositionManager(max_open=3))

    assert lifecycle.reserve_position("cluster-1", _manifest(), 1000) is False


def test_confirm_position_only_records_reserved_clusters() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager())
    reserved_manifest = _manifest("cluster-1")
    lifecycle.reserve_position("cluster-1", reserved_manifest, 1000)

    reserved_receipt = object()
    ignored_receipt = object()
    lifecycle.confirm_position("cluster-1", reserved_receipt, 1001)  # type: ignore[arg-type]
    lifecycle.confirm_position("cluster-2", ignored_receipt, 1002)  # type: ignore[arg-type]

    assert lifecycle._confirmed_receipts == {"cluster-1": reserved_receipt}


def test_confirm_position_overwrites_prior_receipt_for_same_cluster() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager())
    lifecycle.reserve_position("cluster-1", _manifest(), 1000)

    first_receipt = object()
    second_receipt = object()
    lifecycle.confirm_position("cluster-1", first_receipt, 1001)  # type: ignore[arg-type]
    lifecycle.confirm_position("cluster-1", second_receipt, 1002)  # type: ignore[arg-type]

    assert lifecycle._confirmed_receipts["cluster-1"] is second_receipt


def test_release_position_clears_local_state_and_triggers_cleanup() -> None:
    manager = _FakePositionManager()
    lifecycle = PositionManagerLifecycle(manager)
    lifecycle.reserve_position("cluster-1", _manifest(), 1000)
    lifecycle.confirm_position("cluster-1", object(), 1001)  # type: ignore[arg-type]

    lifecycle.release_position("cluster-1", object(), 1002)  # type: ignore[arg-type]

    assert lifecycle._reserved_manifests == {}
    assert lifecycle._confirmed_receipts == {}
    assert manager.cleanup_calls == 1


def test_release_position_swallows_cleanup_exceptions() -> None:
    manager = _CleanupRaisingPositionManager()
    lifecycle = PositionManagerLifecycle(manager)
    lifecycle.reserve_position("cluster-1", _manifest(), 1000)

    lifecycle.release_position("cluster-1", object(), 1002)  # type: ignore[arg-type]

    assert lifecycle._reserved_manifests == {}
    assert manager.cleanup_calls == 1


def test_active_position_count_combines_manager_and_local_reservations() -> None:
    lifecycle = PositionManagerLifecycle(_FakePositionManager(max_open=5, open_positions=[object(), object()]))
    lifecycle.reserve_position("cluster-1", _manifest("cluster-1"), 1000)
    lifecycle.reserve_position("cluster-2", _manifest("cluster-2"), 1001)

    assert lifecycle.active_position_count == 4


def test_active_position_count_falls_back_to_local_reservations_when_manager_read_fails() -> None:
    lifecycle = PositionManagerLifecycle(_RaisingPositionManager(max_open=5))
    lifecycle._reserved_manifests["cluster-1"] = _manifest("cluster-1")

    assert lifecycle.active_position_count == 1