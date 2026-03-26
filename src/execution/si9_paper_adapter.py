from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from src.execution.alpha_adapters import ctf_to_context, si9_to_context
from src.execution.dispatch_guard import DispatchGuard
from src.execution.priority_dispatcher import DispatchReceipt, PriorityDispatcher
from src.execution.signal_coordination_bus import SignalCoordinationBus, SlotDecision
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_paper_ledger import Si9PaperLedger
from src.execution.si9_unwind_evaluator import Si9UnwindEvaluator
from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindManifest, UnwindReason
from src.signals.si9_matrix_detector import Si9MatrixSignal


ClusterOutcome = Literal["FULL_FILL", "HANGING_LEG", "CLUSTER_SUPPRESSED", "GUARD_REJECTED", "BUS_REJECTED"]


@dataclass(frozen=True, slots=True)
class Si9PaperAdapterConfig:
    max_expected_net_edge: Decimal
    max_capital_per_cluster: Decimal
    max_leg_fill_wait_ms: int
    cancel_on_stale_ms: int
    mode: Literal["paper", "dry_run"]
    unwind_config: Si9UnwindConfig = field(
        default_factory=lambda: Si9UnwindConfig(
            market_sell_threshold=Decimal("0.040"),
            passive_unwind_threshold=Decimal("0.010"),
            max_hold_recovery_ms=250,
            min_best_bid=Decimal("0.010"),
        )
    )
    bus: SignalCoordinationBus | None = None

    def __post_init__(self) -> None:
        if self.max_expected_net_edge <= Decimal("0"):
            raise ValueError("max_expected_net_edge must be strictly positive")
        if self.max_capital_per_cluster <= Decimal("0"):
            raise ValueError("max_capital_per_cluster must be strictly positive")
        if not isinstance(self.max_leg_fill_wait_ms, int) or self.max_leg_fill_wait_ms <= 0:
            raise ValueError("max_leg_fill_wait_ms must be a strictly positive int")
        if not isinstance(self.cancel_on_stale_ms, int) or self.cancel_on_stale_ms <= 0:
            raise ValueError("cancel_on_stale_ms must be a strictly positive int")
        if self.mode not in {"paper", "dry_run"}:
            raise ValueError("mode must be 'paper' or 'dry_run'")


@dataclass(frozen=True, slots=True)
class Si9PaperAdapterReceipt:
    per_leg_receipts: tuple[DispatchReceipt, ...]
    manifest: Si9ExecutionManifest
    cluster_outcome: ClusterOutcome
    unwind_manifest: Si9UnwindManifest | None = None
    slot_decisions: tuple[SlotDecision, ...] = tuple()


class Si9PaperAdapter:
    def __init__(
        self,
        dispatcher: PriorityDispatcher,
        guard: DispatchGuard,
        ledger: Si9PaperLedger,
        config: Si9PaperAdapterConfig,
    ):
        self._dispatcher = dispatcher
        self._guard = guard
        self._ledger = ledger
        self._config = config
        self._bus = config.bus
        self._unwind_evaluator = Si9UnwindEvaluator(config.unwind_config)

    @property
    def dispatcher(self) -> PriorityDispatcher:
        return self._dispatcher

    @property
    def guard(self) -> DispatchGuard:
        return self._guard

    @property
    def bus(self) -> SignalCoordinationBus | None:
        return self._bus

    def ledger_snapshot(self):
        return self._ledger.snapshot()

    def on_signal(
        self,
        signal: Si9MatrixSignal,
        current_timestamp_ms: int,
    ) -> Si9PaperAdapterReceipt:
        manifest = self._build_manifest(signal, int(current_timestamp_ms))
        if self._config.mode == "dry_run":
            self._ledger.record_cluster_suppressed(manifest, int(current_timestamp_ms))
            return Si9PaperAdapterReceipt(
                per_leg_receipts=tuple(),
                manifest=manifest,
                cluster_outcome="CLUSTER_SUPPRESSED",
            )

        conviction_scalar = self._conviction_scalar(signal.net_edge)
        receipts: list[DispatchReceipt] = []
        slot_decisions: list[SlotDecision] = []
        acquired_contexts: list[tuple[str, str, str]] = []
        for leg in manifest.legs:
            context = self._context_for_leg(leg, conviction_scalar)
            if self._bus is not None:
                slot_decision = self._bus.request_slot(
                    context.market_id,
                    context.side,
                    context.signal_source,
                    int(current_timestamp_ms),
                )
                slot_decisions.append(slot_decision)
                if not slot_decision.granted:
                    self._release_slots(acquired_contexts, int(current_timestamp_ms))
                    if receipts:
                        self._ledger.record_hanging_leg(manifest, tuple(receipts), int(current_timestamp_ms))
                        unwind_manifest = self._build_unwind_manifest(
                            signal=signal,
                            manifest=manifest,
                            receipts=tuple(receipts),
                            unwind_reason="BUS_EVICTED",
                            current_timestamp_ms=int(current_timestamp_ms),
                        )
                        if unwind_manifest is not None:
                            self._ledger.record_unwind_manifest(unwind_manifest)
                        return Si9PaperAdapterReceipt(
                            per_leg_receipts=tuple(receipts),
                            manifest=manifest,
                            cluster_outcome="HANGING_LEG",
                            unwind_manifest=unwind_manifest,
                            slot_decisions=tuple(slot_decisions),
                        )
                    self._ledger.record_cluster_suppressed(manifest, int(current_timestamp_ms))
                    return Si9PaperAdapterReceipt(
                        per_leg_receipts=tuple(receipts),
                        manifest=manifest,
                        cluster_outcome="BUS_REJECTED",
                        slot_decisions=tuple(slot_decisions),
                    )
                acquired_contexts.append((context.market_id, context.side, context.signal_source))

            decision = self._guard.check(context, int(current_timestamp_ms))
            if not decision.allowed:
                self._guard.record_suppression(context.signal_source)
                self._release_slots(acquired_contexts, int(current_timestamp_ms))
                outcome: ClusterOutcome = "GUARD_REJECTED" if leg.is_bottleneck else "HANGING_LEG"
                unwind_manifest = None
                if outcome == "GUARD_REJECTED":
                    self._ledger.record_guard_rejected(manifest, tuple(receipts), int(current_timestamp_ms))
                else:
                    self._ledger.record_hanging_leg(manifest, tuple(receipts), int(current_timestamp_ms))
                    unwind_manifest = self._build_unwind_manifest(
                        signal=signal,
                        manifest=manifest,
                        receipts=tuple(receipts),
                        unwind_reason=self._resolve_unwind_reason(leg, len(receipts), "guard"),
                        current_timestamp_ms=int(current_timestamp_ms),
                    )
                    if unwind_manifest is not None:
                        self._ledger.record_unwind_manifest(unwind_manifest)
                return Si9PaperAdapterReceipt(
                    per_leg_receipts=tuple(receipts),
                    manifest=manifest,
                    cluster_outcome=outcome,
                    unwind_manifest=unwind_manifest,
                    slot_decisions=tuple(slot_decisions),
                )

            receipt = self._dispatcher.dispatch(context, dispatch_timestamp_ms=int(current_timestamp_ms))
            if not receipt.executed:
                self._release_slots(acquired_contexts, int(current_timestamp_ms))
                outcome = "GUARD_REJECTED" if leg.is_bottleneck else "HANGING_LEG"
                unwind_manifest = None
                if outcome == "GUARD_REJECTED":
                    self._ledger.record_guard_rejected(manifest, tuple(receipts), int(current_timestamp_ms))
                else:
                    self._ledger.record_hanging_leg(manifest, tuple(receipts), int(current_timestamp_ms))
                    unwind_manifest = self._build_unwind_manifest(
                        signal=signal,
                        manifest=manifest,
                        receipts=tuple(receipts),
                        unwind_reason=self._resolve_unwind_reason(leg, len(receipts), "dispatch"),
                        current_timestamp_ms=int(current_timestamp_ms),
                    )
                    if unwind_manifest is not None:
                        self._ledger.record_unwind_manifest(unwind_manifest)
                return Si9PaperAdapterReceipt(
                    per_leg_receipts=tuple(receipts),
                    manifest=manifest,
                    cluster_outcome=outcome,
                    unwind_manifest=unwind_manifest,
                    slot_decisions=tuple(slot_decisions),
                )

            self._guard.record_dispatch(context, int(current_timestamp_ms))
            receipts.append(receipt)

        self._release_slots(acquired_contexts, int(current_timestamp_ms))
        self._ledger.record_full_fill(manifest, tuple(receipts), int(current_timestamp_ms))
        return Si9PaperAdapterReceipt(
            per_leg_receipts=tuple(receipts),
            manifest=manifest,
            cluster_outcome="FULL_FILL",
            slot_decisions=tuple(slot_decisions),
        )

    def _build_manifest(
        self,
        signal: Si9MatrixSignal,
        current_timestamp_ms: int,
    ) -> Si9ExecutionManifest:
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
        return Si9ExecutionManifest(
            cluster_id=signal.cluster_id,
            legs=legs,
            net_edge=signal.net_edge,
            required_share_counts=signal.required_share_counts,
            bottleneck_market_id=signal.bottleneck_market_id,
            manifest_timestamp_ms=current_timestamp_ms,
            max_leg_fill_wait_ms=self._config.max_leg_fill_wait_ms,
            cancel_on_stale_ms=self._config.cancel_on_stale_ms,
        )

    def _context_for_leg(self, leg: Si9LegManifest, conviction_scalar: Decimal):
        if leg.is_bottleneck:
            return ctf_to_context(
                market_id=leg.market_id,
                side="YES",
                target_price=leg.target_price,
                anchor_volume=leg.target_size,
                max_capital=self._config.max_capital_per_cluster,
                conviction_scalar=conviction_scalar,
            )
        return si9_to_context(
            market_id=leg.market_id,
            side="YES",
            target_price=leg.target_price,
            anchor_volume=leg.target_size,
            max_capital=self._config.max_capital_per_cluster,
            conviction_scalar=conviction_scalar,
        )

    def _conviction_scalar(self, net_edge: Decimal) -> Decimal:
        if net_edge <= Decimal("0"):
            return Decimal("0")
        conviction = net_edge / self._config.max_expected_net_edge
        if conviction >= Decimal("1"):
            return Decimal("1")
        return conviction

    def _release_slots(
        self,
        acquired_contexts: list[tuple[str, str, str]],
        current_timestamp_ms: int,
    ) -> None:
        if self._bus is None:
            return
        for market_id, side, signal_source in acquired_contexts:
            self._bus.release_slot(market_id, side, signal_source, int(current_timestamp_ms))

    def _build_unwind_manifest(
        self,
        signal: Si9MatrixSignal,
        manifest: Si9ExecutionManifest,
        receipts: tuple[DispatchReceipt, ...],
        unwind_reason: UnwindReason,
        current_timestamp_ms: int,
    ) -> Si9UnwindManifest | None:
        hanging_legs = [
            (receipt.context.market_id, receipt.fill_size, receipt.fill_price)
            for receipt in receipts
            if receipt.executed and receipt.fill_size is not None and receipt.fill_price is not None
        ]
        if not hanging_legs:
            return None

        # Paper-mode approximation: live best-bid state is not wired yet, so the
        # visible signal ask is used as the unwind bid proxy for deterministic replay.
        current_bids = {
            market_id: signal.best_yes_asks[market_id]
            for market_id, _, _ in hanging_legs
        }
        return self._unwind_evaluator.evaluate(
            cluster_id=manifest.cluster_id,
            hanging_legs=hanging_legs,
            current_bids=current_bids,
            unwind_reason=unwind_reason,
            original_manifest=manifest,
            unwind_timestamp_ms=int(current_timestamp_ms),
        )

    def _resolve_unwind_reason(
        self,
        failed_leg: Si9LegManifest,
        executed_leg_count: int,
        abort_kind: Literal["guard", "dispatch"],
    ) -> UnwindReason:
        if executed_leg_count > 0 and not failed_leg.is_bottleneck:
            return "SECOND_LEG_REJECTED"
        if abort_kind == "guard":
            return "GUARD_CIRCUIT_OPEN"
        return "MANUAL_ABORT"
