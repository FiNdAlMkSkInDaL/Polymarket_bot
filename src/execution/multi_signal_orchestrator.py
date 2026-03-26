from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from src.detectors.ctf_peg_config import CtfPegConfig
from src.detectors.ctf_peg_detector import CtfPegDetector
from src.detectors.si9_cluster_config import Si9ClusterConfig
from src.events.mev_events import CtfMergeSignal
from src.execution.ctf_paper_adapter import CtfPaperAdapter
from src.execution.ctf_paper_ledger import CtfLedgerSnapshot
from src.execution.dispatch_guard import DispatchGuard
from src.execution.escalation_policy_interface import EscalationPolicyInterface
from src.execution.guard_observability import GuardObservabilityPanel, ObservabilitySnapshot
from src.execution.live_book_interface import LiveBestBidProvider
from src.execution.live_wallet_balance import LiveWalletBalanceProvider
from src.execution.orchestrator_load_shedder import OrchestratorLoadShedder
from src.execution.ofi_exit_router import OfiExitRouter
from src.execution.ofi_local_exit_monitor import OfiExitDecision, OfiLocalExitMonitor
from src.execution.ofi_paper_ledger import OfiLedgerSnapshot
from src.execution.ofi_signal_bridge import OfiEntrySignal, OfiSignalBridge
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.execution.position_lifecycle_interface import PositionLifecycleInterface
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_paper_adapter import Si9PaperAdapter, Si9PaperAdapterReceipt
from src.execution.si9_paper_ledger import Si9LedgerSnapshot
from src.execution.si9_unwind_manifest import Si9UnwindManifest
from src.execution.signal_coordination_bus import CoordinationBusSnapshot, SignalCoordinationBus
from src.execution.unwind_executor_interface import UnwindExecutionReceipt, UnwindExecutor
from src.signals.si9_matrix_detector import Si9MatrixDetector, Si9MatrixSignal

if TYPE_CHECKING:
    from src.data.orderbook import OrderbookTracker


SignalSource = Literal["CTF", "SI9", "OFI", "SYSTEM"]
OrchestratorHealth = Literal["GREEN", "YELLOW", "RED"]
OrchestratorEventType = Literal[
    "CTF_DISPATCHED",
    "CTF_REJECTED",
    "OFI_DISPATCHED",
    "OFI_REJECTED",
    "SI9_DISPATCHED",
    "SI9_REJECTED",
    "UNWIND_INITIATED",
    "UNWIND_ESCALATED",
    "UNWIND_COMPLETE",
    "POSITION_RESERVED",
    "POSITION_RELEASED",
    "TICK_PROCESSED",
]


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    tick_interval_ms: int
    max_pending_unwinds: int
    max_concurrent_clusters: int
    signal_sources_enabled: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.tick_interval_ms, int) or self.tick_interval_ms <= 0:
            raise ValueError("tick_interval_ms must be a strictly positive int")
        if not isinstance(self.max_pending_unwinds, int) or self.max_pending_unwinds < 1:
            raise ValueError("max_pending_unwinds must be >= 1")
        if not isinstance(self.max_concurrent_clusters, int) or self.max_concurrent_clusters < 1:
            raise ValueError("max_concurrent_clusters must be >= 1")
        if not isinstance(self.signal_sources_enabled, frozenset):
            raise ValueError("signal_sources_enabled must be a frozenset")
        valid_sources = {"CTF", "SI9", "OFI"}
        if not self.signal_sources_enabled.issubset(valid_sources):
            raise ValueError("signal_sources_enabled contains unsupported source")


@dataclass(frozen=True, slots=True)
class OrchestratorEvent:
    event_type: OrchestratorEventType
    timestamp_ms: int
    source: SignalSource
    payload: dict
    orchestrator_health: OrchestratorHealth


@dataclass(frozen=True, slots=True)
class OrchestratorSnapshot:
    timestamp_ms: int
    pending_unwind_count: int
    active_position_count: int
    ctf_ledger: CtfLedgerSnapshot
    ofi_ledger: OfiLedgerSnapshot
    si9_ledger: Si9LedgerSnapshot
    bus_snapshot: CoordinationBusSnapshot
    observability: ObservabilitySnapshot
    health: OrchestratorHealth


class MultiSignalOrchestrator:
    def __init__(
        self,
        bus: SignalCoordinationBus,
        guard: DispatchGuard,
        dispatcher: PriorityDispatcher,
        ctf_adapter: CtfPaperAdapter,
        ofi_bridge: OfiSignalBridge,
        si9_adapter: Si9PaperAdapter,
        best_bid_provider: LiveBestBidProvider,
        unwind_executor: UnwindExecutor,
        position_lifecycle: PositionLifecycleInterface,
        escalation_policy: EscalationPolicyInterface,
        config: OrchestratorConfig,
        load_shedder: OrchestratorLoadShedder | None = None,
        wallet_balance_provider: LiveWalletBalanceProvider | None = None,
        ofi_exit_router: OfiExitRouter | None = None,
        ofi_exit_trackers: Mapping[str, OrderbookTracker] | None = None,
    ):
        self._bus = bus
        self._guard = guard
        self._dispatcher = dispatcher
        self._ctf_adapter = ctf_adapter
        self._ofi_bridge = ofi_bridge
        self._si9_adapter = si9_adapter
        self._best_bid_provider = best_bid_provider
        self._unwind_executor = unwind_executor
        self._position_lifecycle = position_lifecycle
        self._escalation_policy = escalation_policy
        self._config = config
        self._load_shedder = load_shedder
        self._wallet_balance_provider = wallet_balance_provider
        self._ofi_exit_router = ofi_exit_router
        self._ofi_exit_trackers = dict(ofi_exit_trackers or {})
        self._ofi_exit_monitors: dict[str, OfiLocalExitMonitor] = {}
        self._observability_panel = GuardObservabilityPanel({"SYSTEM": guard}, bus)
        self._pending_unwinds: dict[str, Si9UnwindManifest] = {}
        self._active_cluster_ids: set[str] = set()
        self._ctf_detector_config: CtfPegConfig | None = None
        self._ctf_detector_class: type[CtfPegDetector] = CtfPegDetector
        self._si9_detector: Si9MatrixDetector | None = None
        self._si9_cluster_config: Si9ClusterConfig | None = None

    @property
    def bus(self) -> SignalCoordinationBus:
        return self._bus

    @property
    def guard(self) -> DispatchGuard:
        return self._guard

    @property
    def dispatcher(self) -> PriorityDispatcher:
        return self._dispatcher

    @property
    def ctf_adapter(self) -> CtfPaperAdapter:
        return self._ctf_adapter

    @property
    def ofi_bridge(self) -> OfiSignalBridge:
        return self._ofi_bridge

    @property
    def si9_adapter(self) -> Si9PaperAdapter:
        return self._si9_adapter

    @property
    def position_lifecycle(self) -> PositionLifecycleInterface:
        return self._position_lifecycle

    @property
    def escalation_policy(self) -> EscalationPolicyInterface:
        return self._escalation_policy

    @property
    def best_bid_provider(self) -> LiveBestBidProvider:
        return self._best_bid_provider

    @property
    def si9_detector(self) -> Si9MatrixDetector | None:
        return self._si9_detector

    @property
    def ctf_detector_config(self) -> CtfPegConfig | None:
        return self._ctf_detector_config

    @property
    def si9_cluster_config(self) -> Si9ClusterConfig | None:
        return self._si9_cluster_config

    @property
    def load_shedder(self) -> OrchestratorLoadShedder | None:
        return self._load_shedder

    @property
    def wallet_balance_provider(self) -> LiveWalletBalanceProvider | None:
        return self._wallet_balance_provider

    @property
    def ofi_exit_router(self) -> OfiExitRouter | None:
        return self._ofi_exit_router

    def bind_detector_context(
        self,
        *,
        ctf_config: CtfPegConfig | None = None,
        si9_detector: Si9MatrixDetector | None = None,
        si9_cluster_config: Si9ClusterConfig | None = None,
    ) -> None:
        self._ctf_detector_config = ctf_config
        self._si9_detector = si9_detector
        self._si9_cluster_config = si9_cluster_config

    def on_best_yes_ask_update(
        self,
        market_id: str,
        ask_price: Decimal,
        ask_size: Decimal,
        timestamp_ms: int,
    ) -> bool:
        if not self._market_allowed(market_id):
            return False
        if self._si9_detector is None:
            return False
        self._si9_detector.update_best_yes_ask(market_id, ask_price, ask_size, int(timestamp_ms))
        return True

    def on_ctf_signal(
        self,
        signal: CtfMergeSignal,
        timestamp_ms: int,
    ) -> OrchestratorEvent:
        event_timestamp_ms = int(timestamp_ms)
        if not self._market_allowed(signal.market_id):
            return self._event(
                event_type="CTF_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="CTF",
                payload={"reason": "MARKET_NOT_ALLOWED", "market_id": signal.market_id},
            )
        if "CTF" not in self._config.signal_sources_enabled:
            return self._event(
                event_type="CTF_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="CTF",
                payload={"reason": "SOURCE_DISABLED", "market_id": signal.market_id},
            )

        receipt = self._ctf_adapter.on_signal(signal, event_timestamp_ms)
        if receipt.execution_outcome in {"FULL_FILL", "PARTIAL_FILL"}:
            return self._event(
                event_type="CTF_DISPATCHED",
                timestamp_ms=event_timestamp_ms,
                source="CTF",
                payload={
                    "market_id": signal.market_id,
                    "execution_outcome": receipt.execution_outcome,
                    "realized_net_edge": str(receipt.realized_net_edge),
                    "total_capital_deployed": str(receipt.total_capital_deployed),
                },
            )

        return self._event(
            event_type="CTF_REJECTED",
            timestamp_ms=event_timestamp_ms,
            source="CTF",
            payload={
                "market_id": signal.market_id,
                "execution_outcome": receipt.execution_outcome,
            },
        )

    def on_ofi_signal(
        self,
        signal: OfiEntrySignal,
        max_capital: Decimal,
        timestamp_ms: int,
    ) -> OrchestratorEvent:
        event_timestamp_ms = int(timestamp_ms)
        if not self._market_allowed(signal.market_id):
            return self._event(
                event_type="OFI_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="OFI",
                payload={"reason": "MARKET_NOT_ALLOWED", "market_id": signal.market_id, "side": signal.side},
            )
        if "OFI" not in self._config.signal_sources_enabled:
            return self._event(
                event_type="OFI_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="OFI",
                payload={"reason": "SOURCE_DISABLED", "market_id": signal.market_id, "side": signal.side},
            )

        receipt = self._ofi_bridge.on_signal(signal, max_capital, event_timestamp_ms)
        if receipt.bridge_outcome == "DISPATCHED":
            return self._event(
                event_type="OFI_DISPATCHED",
                timestamp_ms=event_timestamp_ms,
                source="OFI",
                payload={
                    "market_id": signal.market_id,
                    "side": signal.side,
                    "bridge_outcome": receipt.bridge_outcome,
                    "max_capital": str(min(max_capital, self._ofi_bridge._config.max_capital_per_signal)),
                    "dispatch_executed": None if receipt.dispatch_receipt is None else receipt.dispatch_receipt.executed,
                    "fill_status": None if receipt.dispatch_receipt is None else receipt.dispatch_receipt.fill_status,
                },
            )

        reason = receipt.bridge_outcome
        if receipt.guard_decision is not None:
            reason = receipt.guard_decision.reason
        elif receipt.yes_slot is not None and not receipt.yes_slot.granted:
            reason = receipt.yes_slot.reason
        elif receipt.no_slot is not None and not receipt.no_slot.granted:
            reason = receipt.no_slot.reason
        return self._event(
            event_type="OFI_REJECTED",
            timestamp_ms=event_timestamp_ms,
            source="OFI",
            payload={
                "market_id": signal.market_id,
                "side": signal.side,
                "bridge_outcome": receipt.bridge_outcome,
                "reason": reason,
            },
        )

    def on_si9_signal(
        self,
        signal: Si9MatrixSignal,
        timestamp_ms: int,
    ) -> OrchestratorEvent:
        event_timestamp_ms = int(timestamp_ms)
        blocked_market_ids = [market_id for market_id in signal.market_ids if not self._market_allowed(market_id)]
        if blocked_market_ids:
            return self._event(
                event_type="SI9_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="SI9",
                payload={
                    "reason": "MARKET_NOT_ALLOWED",
                    "cluster_id": signal.cluster_id,
                    "blocked_market_ids": blocked_market_ids,
                },
            )
        if "SI9" not in self._config.signal_sources_enabled:
            return self._event(
                event_type="SI9_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="SI9",
                payload={"reason": "SOURCE_DISABLED", "cluster_id": signal.cluster_id},
            )
        if len(self._pending_unwinds) >= self._config.max_pending_unwinds:
            return self._event(
                event_type="SI9_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="SI9",
                payload={"reason": "MAX_PENDING_UNWINDS", "cluster_id": signal.cluster_id},
            )

        manifest = self._build_si9_manifest(signal, event_timestamp_ms)
        reserved = self._position_lifecycle.reserve_position(signal.cluster_id, manifest, event_timestamp_ms)
        if not reserved:
            return self._event(
                event_type="SI9_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="SI9",
                payload={"reason": "POSITION_CAP", "cluster_id": signal.cluster_id},
            )

        receipt = self._si9_adapter.on_signal(signal, event_timestamp_ms)
        if receipt.cluster_outcome == "FULL_FILL":
            self._position_lifecycle.confirm_position(signal.cluster_id, receipt, event_timestamp_ms)
            self._active_cluster_ids.add(signal.cluster_id)
            return self._event(
                event_type="SI9_DISPATCHED",
                timestamp_ms=event_timestamp_ms,
                source="SI9",
                payload={
                    "cluster_id": signal.cluster_id,
                    "cluster_outcome": receipt.cluster_outcome,
                    "position_reserved": True,
                    "bottleneck_market_id": receipt.manifest.bottleneck_market_id,
                },
            )

        if receipt.cluster_outcome == "HANGING_LEG" and receipt.unwind_manifest is not None:
            self._position_lifecycle.confirm_position(signal.cluster_id, receipt, event_timestamp_ms)
            self._active_cluster_ids.add(signal.cluster_id)
            self._pending_unwinds[signal.cluster_id] = receipt.unwind_manifest
            return self._event(
                event_type="UNWIND_INITIATED",
                timestamp_ms=event_timestamp_ms,
                source="SI9",
                payload={
                    "cluster_id": signal.cluster_id,
                    "unwind_reason": receipt.unwind_manifest.unwind_reason,
                    "recommended_action": receipt.unwind_manifest.recommended_action,
                    "pending_unwind_count": len(self._pending_unwinds),
                },
            )

        self._release_cluster_without_unwind(signal.cluster_id, receipt.manifest, event_timestamp_ms, receipt.cluster_outcome)
        return self._event(
            event_type="SI9_REJECTED",
            timestamp_ms=event_timestamp_ms,
            source="SI9",
            payload={
                "cluster_id": signal.cluster_id,
                "cluster_outcome": receipt.cluster_outcome,
                "position_reserved": False,
            },
        )

    def on_tick(
        self,
        timestamp_ms: int,
    ) -> list[OrchestratorEvent]:
        event_timestamp_ms = int(timestamp_ms)
        events: list[OrchestratorEvent] = []
        resolved_clusters: list[str] = []
        for cluster_id, manifest in tuple(self._pending_unwinds.items()):
            if self._escalation_policy.should_surrender(manifest, event_timestamp_ms):
                forced_manifest = replace(manifest, recommended_action="MARKET_SELL")
                unwind_receipt = self._unwind_executor.execute_unwind(forced_manifest, event_timestamp_ms)
                self._position_lifecycle.release_position(cluster_id, unwind_receipt, event_timestamp_ms)
                self._active_cluster_ids.discard(cluster_id)
                resolved_clusters.append(cluster_id)
                events.append(
                    self._event(
                        event_type="UNWIND_COMPLETE",
                        timestamp_ms=event_timestamp_ms,
                        source="SYSTEM",
                        payload={
                            "cluster_id": cluster_id,
                            "action_taken": unwind_receipt.action_taken,
                            "estimated_cost": str(unwind_receipt.estimated_cost),
                            "forced": True,
                        },
                    )
                )
                continue

            if not self._escalation_policy.should_escalate(manifest, event_timestamp_ms):
                continue

            escalated_manifest = self._escalate_manifest(manifest, event_timestamp_ms)
            self._pending_unwinds[cluster_id] = escalated_manifest
            events.append(
                self._event(
                    event_type="UNWIND_ESCALATED",
                    timestamp_ms=event_timestamp_ms,
                    source="SYSTEM",
                    payload={
                        "cluster_id": cluster_id,
                        "recommended_action": escalated_manifest.recommended_action,
                        "unwind_reason": escalated_manifest.unwind_reason,
                    },
                )
            )
            if escalated_manifest.recommended_action == "MARKET_SELL":
                unwind_receipt = self._unwind_executor.execute_unwind(escalated_manifest, event_timestamp_ms)
                self._position_lifecycle.release_position(cluster_id, unwind_receipt, event_timestamp_ms)
                self._active_cluster_ids.discard(cluster_id)
                resolved_clusters.append(cluster_id)
                events.append(
                    self._event(
                        event_type="UNWIND_COMPLETE",
                        timestamp_ms=event_timestamp_ms,
                        source="SYSTEM",
                        payload={
                            "cluster_id": cluster_id,
                            "action_taken": unwind_receipt.action_taken,
                            "estimated_cost": str(unwind_receipt.estimated_cost),
                            "forced": False,
                        },
                    )
                )

        for cluster_id in resolved_clusters:
            self._pending_unwinds.pop(cluster_id, None)
        return events

    def evaluate_ofi_exit(
        self,
        *,
        asset_id: str,
        position_state: dict,
        current_timestamp_ms: int,
    ) -> OfiExitDecision:
        monitor = self._ofi_exit_monitor(asset_id)
        if monitor is None:
            return self._fallback_ofi_exit_decision(position_state, current_timestamp_ms)
        return monitor.evaluate_exit(position_state, current_timestamp_ms)

    def route_ofi_exit(
        self,
        position_state: dict,
        decision: OfiExitDecision,
    ):
        if self._ofi_exit_router is None:
            return None
        return self._ofi_exit_router.route_exit(position_state, decision)

    def evaluate_ofi_exit_promotion(
        self,
        *,
        position_id: str,
        current_timestamp_ms: int,
        current_bbo: dict,
    ):
        if self._ofi_exit_router is None:
            return None
        return self._ofi_exit_router.evaluate_passive_promotion(
            position_id,
            current_timestamp_ms,
            current_bbo,
        )

    def clear_ofi_exit(self, position_id: str) -> None:
        if self._ofi_exit_router is None:
            return
        self._ofi_exit_router.clear_exit(position_id)

    def orchestrator_snapshot(
        self,
        timestamp_ms: int,
    ) -> OrchestratorSnapshot:
        snapshot_timestamp_ms = int(timestamp_ms)
        observability = self._observability_panel.full_snapshot(snapshot_timestamp_ms)
        bus_snapshot = self._bus.bus_snapshot(snapshot_timestamp_ms)
        health = self._resolve_health(observability.system_health, len(self._pending_unwinds))
        return OrchestratorSnapshot(
            timestamp_ms=snapshot_timestamp_ms,
            pending_unwind_count=len(self._pending_unwinds),
            active_position_count=self._active_position_count(),
            ctf_ledger=self._ctf_adapter.ledger_snapshot(),
            ofi_ledger=self._ofi_bridge.ledger_snapshot(),
            si9_ledger=self._si9_adapter.ledger_snapshot(),
            bus_snapshot=bus_snapshot,
            observability=observability,
            health=health,
        )

    def _event(
        self,
        *,
        event_type: OrchestratorEventType,
        timestamp_ms: int,
        source: SignalSource,
        payload: dict,
    ) -> OrchestratorEvent:
        snapshot = self.orchestrator_snapshot(timestamp_ms)
        return OrchestratorEvent(
            event_type=event_type,
            timestamp_ms=int(timestamp_ms),
            source=source,
            payload=self._json_safe_dict(payload),
            orchestrator_health=snapshot.health,
        )

    def _build_si9_manifest(self, signal: Si9MatrixSignal, current_timestamp_ms: int) -> Si9ExecutionManifest:
        ordered_market_ids = [
            signal.bottleneck_market_id,
            *[market_id for market_id in signal.market_ids if market_id != signal.bottleneck_market_id],
        ]
        legs = tuple(
            Si9LegManifest(
                market_id=market_id,
                side="YES",
                target_price=signal.best_yes_asks[market_id],
                target_size=signal.required_share_counts,
                is_bottleneck=(market_id == signal.bottleneck_market_id),
                leg_index=index,
            )
            for index, market_id in enumerate(ordered_market_ids)
        )
        adapter_config = self._si9_adapter._config
        return Si9ExecutionManifest(
            cluster_id=signal.cluster_id,
            legs=legs,
            net_edge=signal.net_edge,
            required_share_counts=signal.required_share_counts,
            bottleneck_market_id=signal.bottleneck_market_id,
            manifest_timestamp_ms=int(current_timestamp_ms),
            max_leg_fill_wait_ms=adapter_config.max_leg_fill_wait_ms,
            cancel_on_stale_ms=adapter_config.cancel_on_stale_ms,
        )

    def _release_cluster_without_unwind(
        self,
        cluster_id: str,
        manifest: Si9ExecutionManifest,
        timestamp_ms: int,
        cluster_outcome: str,
    ) -> None:
        synthetic_unwind = UnwindExecutionReceipt(
            manifest=Si9UnwindManifest(
                cluster_id=cluster_id,
                hanging_legs=tuple(),
                unwind_reason="MANUAL_ABORT",
                original_manifest=manifest,
                unwind_timestamp_ms=int(timestamp_ms),
                total_estimated_unwind_cost=Decimal("0"),
                recommended_action="HOLD_FOR_RECOVERY",
            ),
            action_taken="SKIPPED",
            legs_acted_on=tuple(),
            estimated_cost=Decimal("0"),
            execution_timestamp_ms=int(timestamp_ms),
            notes=f"Released reservation after {cluster_outcome}",
        )
        self._position_lifecycle.release_position(cluster_id, synthetic_unwind, int(timestamp_ms))
        self._active_cluster_ids.discard(cluster_id)

    def _escalate_manifest(self, manifest: Si9UnwindManifest, current_timestamp_ms: int) -> Si9UnwindManifest:
        escalate_manifest = getattr(self._escalation_policy, "escalate_manifest", None)
        if callable(escalate_manifest):
            return escalate_manifest(manifest, int(current_timestamp_ms))
        return manifest

    def _resolve_health(
        self,
        observability_health: OrchestratorHealth,
        pending_unwind_count: int,
    ) -> OrchestratorHealth:
        if observability_health == "RED":
            return "RED"
        if observability_health == "YELLOW":
            return "YELLOW"
        if pending_unwind_count > (self._config.max_pending_unwinds / 2):
            return "YELLOW"
        return "GREEN"

    def _json_safe_dict(self, payload: dict) -> dict:
        return {str(key): self._json_safe_value(value) for key, value in payload.items()}

    def _active_position_count(self) -> int:
        lifecycle_count = getattr(self._position_lifecycle, "active_position_count", None)
        if isinstance(lifecycle_count, int):
            return lifecycle_count
        return len(self._active_cluster_ids)

    def _json_safe_value(self, value):
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._json_safe_value(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return self._json_safe_value(asdict(value))
        return str(value)

    def _market_allowed(self, market_id: str) -> bool:
        if self._load_shedder is None:
            return True
        return self._load_shedder.is_market_allowed(market_id)

    def _ofi_exit_monitor(self, asset_id: str) -> OfiLocalExitMonitor | None:
        asset_key = str(asset_id or "").strip()
        if not asset_key:
            return None
        monitor = self._ofi_exit_monitors.get(asset_key)
        if monitor is not None:
            return monitor
        tracker = self._ofi_exit_trackers.get(asset_key)
        if tracker is None:
            return None
        monitor = OfiLocalExitMonitor(OrderbookBestBidProvider(tracker))
        self._ofi_exit_monitors[asset_key] = monitor
        return monitor

    @staticmethod
    def _fallback_ofi_exit_decision(position_state: dict, current_timestamp_ms: int) -> OfiExitDecision:
        current_best_bid = MultiSignalOrchestrator._decimal_field(position_state, "current_best_bid")
        drawn_tp = MultiSignalOrchestrator._decimal_field(position_state, "drawn_tp")
        if drawn_tp > Decimal("0") and current_best_bid >= drawn_tp:
            return OfiExitDecision(action="TARGET_HIT", trigger_price=drawn_tp)
        drawn_stop = MultiSignalOrchestrator._decimal_field(position_state, "drawn_stop")
        if drawn_stop > Decimal("0") and current_best_bid > Decimal("0") and current_best_bid <= drawn_stop:
            return OfiExitDecision(action="STOP_HIT", trigger_price=drawn_stop)
        drawn_time_ms = int(position_state.get("drawn_time_ms", 0) or 0)
        if drawn_time_ms > 0 and int(current_timestamp_ms) > drawn_time_ms:
            return OfiExitDecision(action="TIME_STOP_TRIGGERED", trigger_price=current_best_bid)
        return OfiExitDecision(action="HOLD", trigger_price=current_best_bid)

    @staticmethod
    def _decimal_field(position_state: dict, field_name: str) -> Decimal:
        value = position_state.get(field_name, Decimal("0"))
        if isinstance(value, Decimal) and value.is_finite() and value >= Decimal("0"):
            return value
        return Decimal("0")