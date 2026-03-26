from __future__ import annotations

from abc import ABC, abstractmethod

from src.execution.si9_unwind_evaluator import Si9UnwindEvaluator
from src.execution.si9_unwind_manifest import Si9UnwindManifest
from src.execution.unwind_manifest import UnwindManifest


class EscalationPolicyInterface(ABC):
    @abstractmethod
    def should_escalate(
        self,
        manifest: UnwindManifest,
        current_timestamp_ms: int,
    ) -> bool:
        ...

    @abstractmethod
    def should_surrender(
        self,
        manifest: UnwindManifest,
        current_timestamp_ms: int,
    ) -> bool:
        """
        True when the position should be hard-flattened regardless of cost.
        Surrender supersedes all other recommendations.
        """
        ...


class PaperEscalationPolicy(EscalationPolicyInterface):
    """
    Paper-mode implementation.
    Escalates after max_hold_recovery_ms via unwind_evaluator.escalate().
    Surrenders never in paper mode - hard flatten is a live-only concept.
    """

    def __init__(
        self,
        evaluator: Si9UnwindEvaluator,
        surrender_after_ms: int,
    ):
        if not isinstance(surrender_after_ms, int) or surrender_after_ms <= 0:
            raise ValueError("surrender_after_ms must be a strictly positive int")
        self._evaluator = evaluator
        self._surrender_after_ms = surrender_after_ms

    def should_escalate(
        self,
        manifest: UnwindManifest,
        current_timestamp_ms: int,
    ) -> bool:
        if not isinstance(manifest, Si9UnwindManifest):
            return False
        return self._evaluator.escalate(manifest, int(current_timestamp_ms)) is not manifest

    def should_surrender(
        self,
        manifest: UnwindManifest,
        current_timestamp_ms: int,
    ) -> bool:
        _ = manifest
        _ = current_timestamp_ms
        return False

    def escalate_manifest(
        self,
        manifest: UnwindManifest,
        current_timestamp_ms: int,
    ) -> UnwindManifest:
        if not isinstance(manifest, Si9UnwindManifest):
            return manifest
        return self._evaluator.escalate(manifest, int(current_timestamp_ms))