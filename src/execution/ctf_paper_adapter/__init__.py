from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Literal, Protocol

from src.events.mev_events import CtfMergeSignal
from src.execution.alpha_adapters import ctf_to_context
from src.execution.ctf_execution_manifest import (
    CtfExecutionManifest,
    CtfExecutionReceipt,
    CtfLegManifest,
    CtfLegReceipt,
    build_ctf_execution_manifest,
)
from src.execution.ctf_paper_ledger import CtfLedgerSnapshot, CtfPaperLedger
from src.execution.dispatch_guard import DispatchGuard
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.signal_coordination_bus import SignalCoordinationBus


_DECIMAL_TICK = Decimal("0.000001")


class CtfDispatcher(Protocol):
    def dispatch(self, context: PriorityOrderContext, dispatch_timestamp_ms: int) -> DispatchReceipt:
        ...


@dataclass(frozen=True, slots=True)
class CtfPaperAdapterConfig:
    max_expected_net_edge: Decimal
    max_capital_per_signal: Decimal
    default_anchor_volume: Decimal
    taker_fee_yes: Decimal
    taker_fee_no: Decimal
    cancel_on_stale_ms: int
    max_size_per_leg: Decimal
    mode: Literal["paper"]
    bus: SignalCoordinationBus | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_expected_net_edge",
            "max_capital_per_signal",
            "default_anchor_volume",
            "max_size_per_leg",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= Decimal("0"):
                raise ValueError(f"{field_name} must be a strictly positive Decimal")

        for field_name in ("taker_fee_yes", "taker_fee_no"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
                raise ValueError(f"{field_name} must be a non-negative Decimal")

        if self.max_expected_net_edge >= Decimal("1"):
            raise ValueError("max_expected_net_edge must be strictly less than Decimal('1')")
        if not isinstance(self.cancel_on_stale_ms, int) or self.cancel_on_stale_ms <= 0:
            raise ValueError("cancel_on_stale_ms must be a strictly positive int")
        if self.mode != "paper":
            raise ValueError("Unsupported CTF paper adapter mode")


class CtfPaperAdapter:
    def __init__(
        self,
        dispatcher: CtfDispatcher,
        guard: DispatchGuard,
        config: CtfPaperAdapterConfig,
        ledger: CtfPaperLedger | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._guard = guard
        self._config = config
        self._bus = config.bus
        self._ledger = CtfPaperLedger() if ledger is None else ledger

    @property
    def dispatcher(self) -> CtfDispatcher:
        return self._dispatcher

    @property
    def guard(self) -> DispatchGuard:
        return self._guard

    @property
    def bus(self) -> SignalCoordinationBus | None:
        return self._bus

    def on_signal(self, signal: CtfMergeSignal, current_timestamp_ms: int) -> CtfExecutionReceipt:
        timestamp_ms = int(current_timestamp_ms)
        conviction_scalar = self._conviction_scalar(signal.net_edge)
        manifest = build_ctf_execution_manifest(
            market_id=signal.market_id,
            yes_price=signal.yes_ask,
            no_price=signal.no_ask,
            net_edge=signal.net_edge,
            gas_estimate=signal.gas_estimate,
            default_anchor_volume=self._config.default_anchor_volume,
            max_capital_per_signal=self._config.max_capital_per_signal,
            max_size_per_leg=self._config.max_size_per_leg,
            taker_fee_yes=self._config.taker_fee_yes,
            taker_fee_no=self._config.taker_fee_no,
            manifest_timestamp_ms=timestamp_ms,
            cancel_on_stale_ms=self._config.cancel_on_stale_ms,
        )
        yes_context = self._context_for_leg(manifest.yes_leg, conviction_scalar)
        no_context = self._context_for_leg(manifest.no_leg, conviction_scalar)
        acquired_slots: list[tuple[str, str, str]] = []

        try:
            if self._bus is not None:
                for context in (yes_context, no_context):
                    decision = self._bus.request_slot(
                        context.market_id,
                        context.side,
                        context.signal_source,
                        timestamp_ms,
                    )
                    if not decision.granted:
                        receipt = self._build_suppressed_receipt(
                            manifest=manifest,
                            yes_context=yes_context,
                            no_context=no_context,
                            reason="BUS_REJECTED",
                            execution_outcome="BUS_REJECTED",
                            timestamp_ms=timestamp_ms,
                        )
                        self._ledger.record(receipt, conviction_scalar)
                        return receipt
                    acquired_slots.append((context.market_id, context.side, context.signal_source))

            anchor_leg, anchor_context, second_leg, second_context = self._ordered_legs(
                manifest,
                yes_context,
                no_context,
            )

            guard_decision = self._guard.check(anchor_context, timestamp_ms)
            if not guard_decision.allowed:
                self._guard.record_suppression(anchor_context.signal_source)
                receipt = self._build_suppressed_receipt(
                    manifest=manifest,
                    yes_context=yes_context,
                    no_context=no_context,
                    reason=guard_decision.reason,
                    execution_outcome="GUARD_REJECTED",
                    timestamp_ms=timestamp_ms,
                )
                self._ledger.record(receipt, conviction_scalar)
                return receipt

            anchor_dispatch = self._dispatcher.dispatch(anchor_context, timestamp_ms)
            anchor_leg_receipt = self._leg_receipt(anchor_leg, anchor_dispatch, timestamp_ms)
            if not anchor_dispatch.executed:
                receipt = self._build_anchor_rejected_receipt(
                    manifest=manifest,
                    anchor_leg_receipt=anchor_leg_receipt,
                    second_leg=second_leg,
                    second_context=second_context,
                    timestamp_ms=timestamp_ms,
                )
                self._ledger.record(receipt, conviction_scalar)
                return receipt

            self._guard.record_dispatch(anchor_context, timestamp_ms)

            # Once the anchor leg fills, any later gate failure leaves the cluster exposed.
            # Classify that path as a second-leg rejection rather than a pre-dispatch bus rejection.
            if self._bus is not None and not self._bus.owns_slot(
                second_context.market_id,
                second_context.side,
                second_context.signal_source,
                timestamp_ms,
            ):
                second_leg_receipt = self._rejected_leg_receipt(
                    second_leg,
                    second_context,
                    reason="BUS_SLOT_LOST",
                    timestamp_ms=timestamp_ms,
                )
                receipt = self._build_second_leg_rejected_receipt(
                    manifest=manifest,
                    anchor_leg_receipt=anchor_leg_receipt,
                    second_leg_receipt=second_leg_receipt,
                    timestamp_ms=timestamp_ms,
                )
                self._ledger.record(receipt, conviction_scalar)
                return receipt

            second_guard_decision = self._guard.check(second_context, timestamp_ms)
            if not second_guard_decision.allowed:
                self._guard.record_suppression(second_context.signal_source)
                second_leg_receipt = self._rejected_leg_receipt(
                    second_leg,
                    second_context,
                    reason=second_guard_decision.reason,
                    timestamp_ms=timestamp_ms,
                )
                receipt = self._build_second_leg_rejected_receipt(
                    manifest=manifest,
                    anchor_leg_receipt=anchor_leg_receipt,
                    second_leg_receipt=second_leg_receipt,
                    timestamp_ms=timestamp_ms,
                )
                self._ledger.record(receipt, conviction_scalar)
                return receipt

            second_dispatch = self._dispatcher.dispatch(second_context, timestamp_ms)
            second_leg_receipt = self._leg_receipt(second_leg, second_dispatch, timestamp_ms)
            if not second_dispatch.executed:
                receipt = self._build_second_leg_rejected_receipt(
                    manifest=manifest,
                    anchor_leg_receipt=anchor_leg_receipt,
                    second_leg_receipt=second_leg_receipt,
                    timestamp_ms=timestamp_ms,
                )
                self._ledger.record(receipt, conviction_scalar)
                return receipt

            yes_receipt = anchor_leg_receipt if anchor_leg.leg_role == "YES_LEG" else second_leg_receipt
            no_receipt = second_leg_receipt if anchor_leg.leg_role == "YES_LEG" else anchor_leg_receipt
            receipt = self._build_fill_receipt(
                manifest=manifest,
                yes_receipt=yes_receipt,
                no_receipt=no_receipt,
                timestamp_ms=timestamp_ms,
            )
            self._ledger.record(receipt, conviction_scalar)
            return receipt
        finally:
            self._release_slots(acquired_slots, timestamp_ms)

    def ledger_snapshot(self) -> CtfLedgerSnapshot:
        return self._ledger.ledger_snapshot()

    def reset_ledger(self) -> None:
        self._ledger.reset()

    def coordination_snapshot(self, current_timestamp_ms: int) -> dict[str, object]:
        if self._config.bus is None:
            return {
                "snapshot_timestamp_ms": int(current_timestamp_ms),
                "total_active_slots": 0,
                "slots_by_source": {},
                "slots_by_market": {},
                "expired_reclaimed_count": 0,
            }
        return asdict(self._bus.bus_snapshot(int(current_timestamp_ms)))

    def _context_for_leg(self, leg: CtfLegManifest, conviction_scalar: Decimal) -> PriorityOrderContext:
        return ctf_to_context(
            market_id=leg.market_id,
            side=leg.side,
            target_price=leg.target_price,
            anchor_volume=leg.target_size,
            max_capital=self._config.max_capital_per_signal,
            conviction_scalar=conviction_scalar,
            leg_role=leg.leg_role,
        )

    @staticmethod
    def _ordered_legs(
        manifest: CtfExecutionManifest,
        yes_context: PriorityOrderContext,
        no_context: PriorityOrderContext,
    ) -> tuple[CtfLegManifest, PriorityOrderContext, CtfLegManifest, PriorityOrderContext]:
        if manifest.anchor_leg == "YES_LEG":
            return manifest.yes_leg, yes_context, manifest.no_leg, no_context
        return manifest.no_leg, no_context, manifest.yes_leg, yes_context

    def _build_suppressed_receipt(
        self,
        *,
        manifest: CtfExecutionManifest,
        yes_context: PriorityOrderContext,
        no_context: PriorityOrderContext,
        reason: str,
        execution_outcome: Literal["BUS_REJECTED", "GUARD_REJECTED"],
        timestamp_ms: int,
    ) -> CtfExecutionReceipt:
        yes_receipt = self._suppressed_leg_receipt(manifest.yes_leg, yes_context, reason, timestamp_ms)
        no_receipt = self._suppressed_leg_receipt(manifest.no_leg, no_context, reason, timestamp_ms)
        return CtfExecutionReceipt(
            manifest=manifest,
            yes_receipt=yes_receipt,
            no_receipt=no_receipt,
            execution_outcome=execution_outcome,
            realized_net_edge=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_capital_deployed=Decimal("0"),
            execution_timestamp_ms=timestamp_ms,
        )

    def _build_anchor_rejected_receipt(
        self,
        *,
        manifest: CtfExecutionManifest,
        anchor_leg_receipt: CtfLegReceipt,
        second_leg: CtfLegManifest,
        second_context: PriorityOrderContext,
        timestamp_ms: int,
    ) -> CtfExecutionReceipt:
        second_leg_receipt = self._suppressed_leg_receipt(second_leg, second_context, "ANCHOR_REJECTED", timestamp_ms)
        yes_receipt = anchor_leg_receipt if anchor_leg_receipt.leg_manifest.leg_role == "YES_LEG" else second_leg_receipt
        no_receipt = second_leg_receipt if anchor_leg_receipt.leg_manifest.leg_role == "YES_LEG" else anchor_leg_receipt
        return CtfExecutionReceipt(
            manifest=manifest,
            yes_receipt=yes_receipt,
            no_receipt=no_receipt,
            execution_outcome="ANCHOR_REJECTED",
            realized_net_edge=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_capital_deployed=Decimal("0"),
            execution_timestamp_ms=timestamp_ms,
        )

    def _build_second_leg_rejected_receipt(
        self,
        *,
        manifest: CtfExecutionManifest,
        anchor_leg_receipt: CtfLegReceipt,
        second_leg_receipt: CtfLegReceipt,
        timestamp_ms: int,
    ) -> CtfExecutionReceipt:
        yes_receipt = anchor_leg_receipt if anchor_leg_receipt.leg_manifest.leg_role == "YES_LEG" else second_leg_receipt
        no_receipt = second_leg_receipt if anchor_leg_receipt.leg_manifest.leg_role == "YES_LEG" else anchor_leg_receipt
        return CtfExecutionReceipt(
            manifest=manifest,
            yes_receipt=yes_receipt,
            no_receipt=no_receipt,
            execution_outcome="SECOND_LEG_REJECTED",
            realized_net_edge=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_capital_deployed=self._total_capital_deployed((anchor_leg_receipt,)),
            execution_timestamp_ms=timestamp_ms,
        )

    def _build_fill_receipt(
        self,
        *,
        manifest: CtfExecutionManifest,
        yes_receipt: CtfLegReceipt,
        no_receipt: CtfLegReceipt,
        timestamp_ms: int,
    ) -> CtfExecutionReceipt:
        filled_cluster_size = min(yes_receipt.filled_size, no_receipt.filled_size)
        realized_net_edge = (
            Decimal("1")
            - self._filled_price(yes_receipt)
            - self._filled_price(no_receipt)
            - manifest.yes_leg.taker_fee
            - manifest.no_leg.taker_fee
            - manifest.gas_estimate
        ).quantize(_DECIMAL_TICK)
        realized_pnl = (realized_net_edge * filled_cluster_size).quantize(_DECIMAL_TICK)
        execution_outcome: Literal["FULL_FILL", "PARTIAL_FILL"] = "FULL_FILL"
        if yes_receipt.fill_status == "PARTIAL" or no_receipt.fill_status == "PARTIAL":
            execution_outcome = "PARTIAL_FILL"
        return CtfExecutionReceipt(
            manifest=manifest,
            yes_receipt=yes_receipt,
            no_receipt=no_receipt,
            execution_outcome=execution_outcome,
            realized_net_edge=realized_net_edge,
            realized_pnl=realized_pnl,
            total_capital_deployed=self._total_capital_deployed((yes_receipt, no_receipt)),
            execution_timestamp_ms=timestamp_ms,
        )

    @staticmethod
    def _leg_receipt(leg: CtfLegManifest, dispatch_receipt: DispatchReceipt, timestamp_ms: int) -> CtfLegReceipt:
        if not dispatch_receipt.executed:
            return CtfLegReceipt(
                leg_manifest=leg,
                dispatch_receipt=dispatch_receipt,
                fill_status="REJECTED",
                filled_size=Decimal("0"),
                filled_price=None,
                fill_timestamp_ms=None,
            )
        if dispatch_receipt.fill_status == "PARTIAL":
            assert dispatch_receipt.partial_fill_size is not None
            assert dispatch_receipt.partial_fill_price is not None
            return CtfLegReceipt(
                leg_manifest=leg,
                dispatch_receipt=dispatch_receipt,
                fill_status="PARTIAL",
                filled_size=dispatch_receipt.partial_fill_size.quantize(_DECIMAL_TICK),
                filled_price=dispatch_receipt.partial_fill_price.quantize(_DECIMAL_TICK),
                fill_timestamp_ms=timestamp_ms,
            )
        assert dispatch_receipt.fill_size is not None
        assert dispatch_receipt.fill_price is not None
        return CtfLegReceipt(
            leg_manifest=leg,
            dispatch_receipt=dispatch_receipt,
            fill_status="FILLED",
            filled_size=dispatch_receipt.fill_size.quantize(_DECIMAL_TICK),
            filled_price=dispatch_receipt.fill_price.quantize(_DECIMAL_TICK),
            fill_timestamp_ms=timestamp_ms,
        )

    def _suppressed_leg_receipt(
        self,
        leg: CtfLegManifest,
        context: PriorityOrderContext,
        reason: str,
        timestamp_ms: int,
    ) -> CtfLegReceipt:
        return CtfLegReceipt(
            leg_manifest=leg,
            dispatch_receipt=DispatchReceipt(
                context=context,
                mode=self._config.mode,
                executed=False,
                fill_price=None,
                fill_size=None,
                serialized_envelope="{}",
                dispatch_timestamp_ms=timestamp_ms,
                guard_reason=reason,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="NONE",
            ),
            fill_status="SUPPRESSED",
            filled_size=Decimal("0"),
            filled_price=None,
            fill_timestamp_ms=None,
        )

    def _rejected_leg_receipt(
        self,
        leg: CtfLegManifest,
        context: PriorityOrderContext,
        reason: str,
        timestamp_ms: int,
    ) -> CtfLegReceipt:
        return self._leg_receipt(
            leg,
            DispatchReceipt(
                context=context,
                mode=self._config.mode,
                executed=False,
                fill_price=None,
                fill_size=None,
                serialized_envelope="{}",
                dispatch_timestamp_ms=timestamp_ms,
                guard_reason=reason,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="NONE",
            ),
            timestamp_ms,
        )

    def _release_slots(self, acquired_slots: list[tuple[str, str, str]], current_timestamp_ms: int) -> None:
        if self._bus is None:
            return
        for market_id, side, signal_source in acquired_slots:
            self._bus.release_slot(market_id, side, signal_source, current_timestamp_ms)

    def _conviction_scalar(self, net_edge: Decimal) -> Decimal:
        if net_edge <= Decimal("0"):
            return Decimal("0")
        conviction_scalar = net_edge / self._config.max_expected_net_edge
        if conviction_scalar >= Decimal("1"):
            return Decimal("1")
        return conviction_scalar.normalize()

    @staticmethod
    def _filled_price(receipt: CtfLegReceipt) -> Decimal:
        assert receipt.filled_price is not None
        return receipt.filled_price

    @staticmethod
    def _total_capital_deployed(receipts: tuple[CtfLegReceipt, ...]) -> Decimal:
        total = Decimal("0")
        for receipt in receipts:
            if receipt.filled_price is None:
                continue
            total += receipt.filled_price * receipt.filled_size
        return total.quantize(_DECIMAL_TICK)
