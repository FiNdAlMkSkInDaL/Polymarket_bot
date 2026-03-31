from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.multi_signal_orchestrator import OrchestratorConfig


@dataclass(frozen=True, slots=True)
class LiveOrchestratorConfig:
    orchestrator_config: OrchestratorConfig
    guard_config: DispatchGuardConfig
    deployment_phase: Literal["PAPER", "DRY_RUN", "LIVE"]
    session_id: str
    max_position_release_failures: int
    heartbeat_interval_ms: int

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.deployment_phase not in {"PAPER", "DRY_RUN", "LIVE"}:
            errors.append("deployment_phase must be one of 'PAPER', 'DRY_RUN', or 'LIVE'")
        if not str(self.session_id or "").strip():
            errors.append("session_id must be a non-empty string")
        if not isinstance(self.max_position_release_failures, int) or self.max_position_release_failures < 1:
            errors.append("max_position_release_failures must be >= 1")
        if not isinstance(self.heartbeat_interval_ms, int) or self.heartbeat_interval_ms <= 0:
            errors.append("heartbeat_interval_ms must be a strictly positive int")

        if errors:
            raise ValueError("; ".join(errors))