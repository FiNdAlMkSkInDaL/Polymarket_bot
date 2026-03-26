from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindManifest


class UnwindExecutor(ABC):
    @abstractmethod
    def execute_unwind(
        self,
        manifest: Si9UnwindManifest,
        current_timestamp_ms: int,
    ) -> "UnwindExecutionReceipt":
        ...


@dataclass(frozen=True, slots=True)
class UnwindExecutionReceipt:
    manifest: Si9UnwindManifest
    action_taken: Literal["MARKET_SELL", "PASSIVE_UNWIND", "HOLD", "SKIPPED"]
    legs_acted_on: tuple[str, ...]
    estimated_cost: Decimal
    execution_timestamp_ms: int
    notes: str | None


class PaperUnwindExecutor(UnwindExecutor):
    """
    Paper-mode implementation. Records recommendation, takes no real action.
    action_taken mirrors manifest.recommended_action.
    action_taken is SKIPPED when recommended_action is HOLD_FOR_RECOVERY
    and max_hold_recovery_ms has not elapsed.
    """

    def __init__(self, unwind_config: Si9UnwindConfig):
        self._config = unwind_config

    def execute_unwind(
        self,
        manifest: Si9UnwindManifest,
        current_timestamp_ms: int,
    ) -> UnwindExecutionReceipt:
        timestamp_ms = int(current_timestamp_ms)
        elapsed_ms = timestamp_ms - manifest.unwind_timestamp_ms
        market_ids = tuple(leg.market_id for leg in manifest.hanging_legs)

        if manifest.recommended_action == "MARKET_SELL":
            action_taken: Literal["MARKET_SELL", "PASSIVE_UNWIND", "HOLD", "SKIPPED"] = "MARKET_SELL"
            notes = "Paper unwind mirrors urgent market-sell recommendation"
        elif manifest.recommended_action == "PASSIVE_UNWIND":
            action_taken = "PASSIVE_UNWIND"
            notes = "Paper unwind mirrors passive recommendation"
        elif elapsed_ms < self._config.max_hold_recovery_ms:
            action_taken = "SKIPPED"
            market_ids = tuple()
            notes = "Paper unwind defers HOLD_FOR_RECOVERY before timeout"
        else:
            action_taken = "HOLD"
            notes = "Paper unwind records hold recommendation after timeout window"

        return UnwindExecutionReceipt(
            manifest=manifest,
            action_taken=action_taken,
            legs_acted_on=market_ids,
            estimated_cost=manifest.total_estimated_unwind_cost,
            execution_timestamp_ms=timestamp_ms,
            notes=notes,
        )