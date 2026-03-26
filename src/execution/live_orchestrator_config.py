from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.detectors.ctf_peg_config import CtfPegConfig
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.multi_signal_orchestrator import OrchestratorConfig
from src.execution.ofi_signal_bridge import OfiSignalBridgeConfig
from src.execution.si9_paper_adapter import Si9PaperAdapterConfig
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.signal_coordination_bus import CoordinationBusConfig


@dataclass(frozen=True, slots=True)
class LiveOrchestratorConfig:
    orchestrator_config: OrchestratorConfig
    bus_config: CoordinationBusConfig
    guard_config: DispatchGuardConfig
    ctf_adapter_config: CtfPaperAdapterConfig
    si9_adapter_config: Si9PaperAdapterConfig
    ofi_bridge_config: OfiSignalBridgeConfig
    ctf_peg_config: CtfPegConfig
    si9_cluster_configs: tuple[tuple[str, tuple[str, ...]], ...]
    unwind_config: Si9UnwindConfig
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

        if self.deployment_phase == "LIVE":
            for field_name in ("ctf_adapter_config", "si9_adapter_config", "ofi_bridge_config"):
                adapter_mode = getattr(getattr(self, field_name), "mode", None)
                if adapter_mode == "dry_run":
                    errors.append(
                        f"{field_name}.mode must not be 'dry_run' when deployment_phase='LIVE'"
                    )

        if errors:
            raise ValueError("; ".join(errors))