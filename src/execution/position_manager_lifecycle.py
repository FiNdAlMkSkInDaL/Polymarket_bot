from __future__ import annotations

from typing import TYPE_CHECKING

from src.execution.position_lifecycle_interface import PositionLifecycleInterface
from src.execution.si9_execution_manifest import Si9ExecutionManifest
from src.execution.si9_paper_adapter import Si9PaperAdapterReceipt
from src.execution.unwind_executor_interface import UnwindExecutionReceipt

if TYPE_CHECKING:
    from src.trading.position_manager import PositionManager


class PositionManagerLifecycle(PositionLifecycleInterface):
    """Conservative orchestrator-side reservation bridge over PositionManager.

    PositionManager does not expose a native reserve/confirm/release contract,
    so this wrapper tracks cluster reservations locally while consulting the live
    manager's open-position count for capacity checks.
    """

    def __init__(self, position_manager: PositionManager):
        if position_manager is None:
            raise ValueError("position_manager is required")
        self._position_manager = position_manager
        self._reserved_manifests: dict[str, Si9ExecutionManifest] = {}
        self._confirmed_receipts: dict[str, Si9PaperAdapterReceipt] = {}

    def reserve_position(
        self,
        cluster_id: str,
        manifest: Si9ExecutionManifest,
        timestamp_ms: int,
    ) -> bool:
        _ = timestamp_ms
        cluster_key = str(cluster_id or "").strip()
        if not cluster_key:
            return False
        if cluster_key in self._reserved_manifests:
            return True

        max_open = self._max_open_positions()
        if max_open is None:
            return False

        active_count = self._manager_open_count()
        if active_count is None:
            return False

        if active_count + len(self._reserved_manifests) >= max_open:
            return False

        self._reserved_manifests[cluster_key] = manifest
        return True

    def confirm_position(
        self,
        cluster_id: str,
        receipt: Si9PaperAdapterReceipt,
        timestamp_ms: int,
    ) -> None:
        _ = timestamp_ms
        cluster_key = str(cluster_id or "").strip()
        if cluster_key in self._reserved_manifests:
            self._confirmed_receipts[cluster_key] = receipt

    def release_position(
        self,
        cluster_id: str,
        unwind_receipt: UnwindExecutionReceipt,
        timestamp_ms: int,
    ) -> None:
        _ = unwind_receipt
        _ = timestamp_ms
        cluster_key = str(cluster_id or "").strip()
        self._reserved_manifests.pop(cluster_key, None)
        self._confirmed_receipts.pop(cluster_key, None)
        try:
            self._position_manager.cleanup_closed()
        except Exception:
            pass

    @property
    def active_position_count(self) -> int:
        active_count = self._manager_open_count()
        if active_count is None:
            return len(self._reserved_manifests)
        return active_count + len(self._reserved_manifests)

    def _max_open_positions(self) -> int | None:
        try:
            max_open = int(getattr(self._position_manager, "max_open"))
        except Exception:
            return None
        return max_open if max_open > 0 else None

    def _manager_open_count(self) -> int | None:
        try:
            open_positions = self._position_manager.get_open_positions()
        except Exception:
            return None
        try:
            return len(open_positions)
        except Exception:
            return len(list(open_positions))