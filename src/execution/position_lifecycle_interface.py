from __future__ import annotations

from abc import ABC, abstractmethod

from src.execution.si9_execution_manifest import Si9ExecutionManifest
from src.execution.si9_paper_adapter import Si9PaperAdapterReceipt
from src.execution.unwind_executor_interface import UnwindExecutionReceipt


class PositionLifecycleInterface(ABC):
    @abstractmethod
    def reserve_position(
        self,
        cluster_id: str,
        manifest: Si9ExecutionManifest,
        timestamp_ms: int,
    ) -> bool:
        """Returns True if reservation succeeded, False if position cap reached."""

    @abstractmethod
    def confirm_position(
        self,
        cluster_id: str,
        receipt: Si9PaperAdapterReceipt,
        timestamp_ms: int,
    ) -> None:
        ...

    @abstractmethod
    def release_position(
        self,
        cluster_id: str,
        unwind_receipt: UnwindExecutionReceipt,
        timestamp_ms: int,
    ) -> None:
        ...


class PaperPositionLifecycle(PositionLifecycleInterface):
    """
    Paper-mode implementation. Tracks reservations and confirmations
    in O(k) state bounded by max_concurrent_clusters.
    No PositionManager wiring yet.
    """

    def __init__(self, max_concurrent_clusters: int):
        if not isinstance(max_concurrent_clusters, int) or max_concurrent_clusters <= 0:
            raise ValueError("max_concurrent_clusters must be a strictly positive int")
        self._max_concurrent_clusters = max_concurrent_clusters
        self._reserved_manifests: dict[str, Si9ExecutionManifest] = {}
        self._confirmed_receipts: dict[str, Si9PaperAdapterReceipt] = {}

    def reserve_position(
        self,
        cluster_id: str,
        manifest: Si9ExecutionManifest,
        timestamp_ms: int,
    ) -> bool:
        _ = timestamp_ms
        cluster_key = str(cluster_id).strip()
        if cluster_key in self._reserved_manifests:
            return True
        if len(self._reserved_manifests) >= self._max_concurrent_clusters:
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
        cluster_key = str(cluster_id).strip()
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
        cluster_key = str(cluster_id).strip()
        self._reserved_manifests.pop(cluster_key, None)
        self._confirmed_receipts.pop(cluster_key, None)

    @property
    def active_position_count(self) -> int:
        return len(self._reserved_manifests)