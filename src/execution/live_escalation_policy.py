from __future__ import annotations

from dataclasses import replace

from src.execution.live_unwind_cost_estimator import LiveUnwindCostEstimator
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.unwind_manifest import RecommendedAction, UnwindManifest
from src.execution.escalation_policy_interface import EscalationPolicyInterface


class LiveEscalationPolicy(EscalationPolicyInterface):
    def __init__(
        self,
        cost_estimator: LiveUnwindCostEstimator,
        unwind_config: Si9UnwindConfig,
    ) -> None:
        self._cost_estimator = cost_estimator
        self._config = unwind_config

    def should_escalate(
        self,
        manifest: UnwindManifest,
        current_timestamp_ms: int,
    ) -> bool:
        return self.escalate_manifest(manifest, current_timestamp_ms) != manifest

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
    ):
        refreshed_manifest = self._cost_estimator.estimate_manifest(manifest)
        elapsed_ms = int(current_timestamp_ms) - refreshed_manifest.unwind_timestamp_ms
        recommended_action = self._recommended_action(refreshed_manifest.total_estimated_unwind_cost)
        if refreshed_manifest.recommended_action == "HOLD_FOR_RECOVERY" and elapsed_ms > self._config.max_hold_recovery_ms:
            recommended_action = "MARKET_SELL"
        return replace(refreshed_manifest, recommended_action=recommended_action)

    def _recommended_action(self, total_estimated_unwind_cost) -> RecommendedAction:
        if total_estimated_unwind_cost >= self._config.market_sell_threshold:
            return "MARKET_SELL"
        if total_estimated_unwind_cost <= self._config.passive_unwind_threshold:
            return "PASSIVE_UNWIND"
        return "HOLD_FOR_RECOVERY"