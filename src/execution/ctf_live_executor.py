from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Literal

from src.execution.alpha_adapters import ctf_to_context
from src.execution.ctf_execution_manifest import (
    CtfExecutionManifest,
    CtfExecutionReceipt,
    CtfLegManifest,
    CtfLegReceipt,
)
from src.execution.ctf_unwind_manifest import CtfUnwindLeg, CtfUnwindManifest
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.unwind_executor_interface import UnwindExecutionReceipt, UnwindExecutor
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


_DECIMAL_TICK = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class CtfLiveExecutionResult:
    execution_receipt: CtfExecutionReceipt
    unwind_manifest: CtfUnwindManifest | None = None
    unwind_execution_receipt: UnwindExecutionReceipt | None = None


@dataclass(frozen=True, slots=True)
class _LegDispatchOutcome:
    dispatch_receipt: DispatchReceipt
    fill_timestamp_ms: int | None


class CtfLiveExecutor:
    def __init__(
        self,
        venue_adapter: VenueAdapter,
        unwind_executor: UnwindExecutor,
        *,
        poll_interval_ms: int = 50,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if not isinstance(poll_interval_ms, int) or poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be a strictly positive int")
        self._venue_adapter = venue_adapter
        self._unwind_executor = unwind_executor
        self._poll_interval_ms = poll_interval_ms
        self._sleep_fn = time.sleep if sleep_fn is None else sleep_fn
        self._sequence = 0

    def execute(
        self,
        manifest: CtfExecutionManifest,
        current_timestamp_ms: int,
    ) -> CtfLiveExecutionResult:
        timestamp_ms = int(current_timestamp_ms)
        execution_sequence = self._next_sequence()
        anchor_leg, second_leg = self._ordered_legs(manifest)

        anchor_outcome = self._submit_and_confirm_leg(
            manifest=manifest,
            leg=anchor_leg,
            submitted_size=anchor_leg.target_size,
            dispatch_timestamp_ms=timestamp_ms,
            execution_sequence=execution_sequence,
        )
        anchor_leg_receipt = self._leg_receipt(anchor_leg, anchor_outcome)
        if not anchor_outcome.dispatch_receipt.executed:
            receipt = self._build_anchor_rejected_receipt(
                manifest=manifest,
                anchor_leg_receipt=anchor_leg_receipt,
                second_leg=second_leg,
                timestamp_ms=timestamp_ms,
            )
            return CtfLiveExecutionResult(execution_receipt=receipt)

        anchor_filled_size = anchor_leg_receipt.filled_size
        second_outcome = self._submit_and_confirm_leg(
            manifest=manifest,
            leg=second_leg,
            submitted_size=anchor_filled_size,
            dispatch_timestamp_ms=timestamp_ms,
            execution_sequence=execution_sequence,
        )
        second_leg_receipt = self._leg_receipt(second_leg, second_outcome)

        yes_receipt = anchor_leg_receipt if anchor_leg.leg_role == "YES_LEG" else second_leg_receipt
        no_receipt = second_leg_receipt if anchor_leg.leg_role == "YES_LEG" else anchor_leg_receipt

        if second_leg_receipt.filled_size < anchor_filled_size:
            matched_cluster_size = min(anchor_leg_receipt.filled_size, second_leg_receipt.filled_size)
            unwind_manifest = self._build_unwind_manifest(
                manifest=manifest,
                anchor_leg_receipt=anchor_leg_receipt,
                second_leg_receipt=second_leg_receipt,
                timestamp_ms=timestamp_ms,
            )
            unwind_receipt = self._unwind_executor.execute_unwind(unwind_manifest, timestamp_ms)
            if matched_cluster_size > Decimal("0"):
                receipt = self._build_fill_receipt(
                    manifest=manifest,
                    yes_receipt=yes_receipt,
                    no_receipt=no_receipt,
                    timestamp_ms=timestamp_ms,
                )
            else:
                receipt = self._build_second_leg_rejected_receipt(
                    manifest=manifest,
                    anchor_leg_receipt=anchor_leg_receipt,
                    second_leg_receipt=second_leg_receipt,
                    timestamp_ms=timestamp_ms,
                )
            return CtfLiveExecutionResult(
                execution_receipt=receipt,
                unwind_manifest=unwind_manifest,
                unwind_execution_receipt=unwind_receipt,
            )

        receipt = self._build_fill_receipt(
            manifest=manifest,
            yes_receipt=yes_receipt,
            no_receipt=no_receipt,
            timestamp_ms=timestamp_ms,
        )
        return CtfLiveExecutionResult(execution_receipt=receipt)

    def _submit_and_confirm_leg(
        self,
        *,
        manifest: CtfExecutionManifest,
        leg: CtfLegManifest,
        submitted_size: Decimal,
        dispatch_timestamp_ms: int,
        execution_sequence: int,
    ) -> _LegDispatchOutcome:
        client_order_id = self._build_client_order_id(execution_sequence, leg)
        context = self._context_for_leg(leg, submitted_size)
        submit_response = self._venue_adapter.submit_order(
            market_id=leg.market_id,
            side=leg.side,
            price=leg.target_price,
            size=submitted_size,
            order_type="LIMIT",
            client_order_id=client_order_id,
        )
        if submit_response.status == "REJECTED":
            return _LegDispatchOutcome(
                dispatch_receipt=self._rejected_dispatch_receipt(
                    context=context,
                    submitted_size=submitted_size,
                    dispatch_timestamp_ms=dispatch_timestamp_ms,
                    submit_response=submit_response,
                    rejection_reason=submit_response.rejection_reason or "VENUE_REJECTED",
                ),
                fill_timestamp_ms=None,
            )

        elapsed_ms = 0
        while True:
            observed_timestamp_ms = dispatch_timestamp_ms + elapsed_ms
            order_status = self._venue_adapter.get_order_status(client_order_id)
            mapped_receipt = self._map_status_receipt(
                context=context,
                submit_response=submit_response,
                order_status=order_status,
                submitted_size=submitted_size,
                dispatch_timestamp_ms=dispatch_timestamp_ms,
            )
            if mapped_receipt is not None:
                if mapped_receipt.fill_status == "PARTIAL" and order_status.remaining_size > Decimal("0"):
                    cancel_response = self._venue_adapter.cancel_order(client_order_id, leg.market_id)
                    final_status = self._venue_adapter.get_order_status(client_order_id)
                    remapped_receipt = self._map_status_receipt(
                        context=context,
                        submit_response=submit_response,
                        order_status=final_status,
                        submitted_size=submitted_size,
                        dispatch_timestamp_ms=dispatch_timestamp_ms,
                    )
                    if remapped_receipt is not None:
                        return _LegDispatchOutcome(
                            dispatch_receipt=remapped_receipt,
                            fill_timestamp_ms=observed_timestamp_ms,
                        )
                    return _LegDispatchOutcome(
                        dispatch_receipt=mapped_receipt,
                        fill_timestamp_ms=observed_timestamp_ms,
                    )
                return _LegDispatchOutcome(
                    dispatch_receipt=mapped_receipt,
                    fill_timestamp_ms=observed_timestamp_ms,
                )

            if elapsed_ms >= manifest.cancel_on_stale_ms:
                break
            self._sleep_fn(self._poll_interval_ms / 1000)
            elapsed_ms = min(manifest.cancel_on_stale_ms, elapsed_ms + self._poll_interval_ms)

        cancel_response = self._venue_adapter.cancel_order(client_order_id, leg.market_id)
        final_status = self._venue_adapter.get_order_status(client_order_id)
        observed_timestamp_ms = dispatch_timestamp_ms + manifest.cancel_on_stale_ms
        mapped_receipt = self._map_status_receipt(
            context=context,
            submit_response=submit_response,
            order_status=final_status,
            submitted_size=submitted_size,
            dispatch_timestamp_ms=dispatch_timestamp_ms,
        )
        if mapped_receipt is not None:
            return _LegDispatchOutcome(
                dispatch_receipt=mapped_receipt,
                fill_timestamp_ms=observed_timestamp_ms,
            )

        rejection_reason = cancel_response.rejection_reason or "FILL_TIMEOUT"
        return _LegDispatchOutcome(
            dispatch_receipt=self._rejected_dispatch_receipt(
                context=context,
                submitted_size=submitted_size,
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                submit_response=submit_response,
                rejection_reason=rejection_reason,
                order_status=final_status,
                cancel_response=cancel_response,
            ),
            fill_timestamp_ms=None,
        )

    @staticmethod
    def _ordered_legs(manifest: CtfExecutionManifest) -> tuple[CtfLegManifest, CtfLegManifest]:
        if manifest.anchor_leg == "YES_LEG":
            return manifest.yes_leg, manifest.no_leg
        return manifest.no_leg, manifest.yes_leg

    @staticmethod
    def _context_for_leg(leg: CtfLegManifest, submitted_size: Decimal) -> PriorityOrderContext:
        return ctf_to_context(
            market_id=leg.market_id,
            side=leg.side,
            target_price=leg.target_price,
            anchor_volume=submitted_size,
            max_capital=(leg.target_price * submitted_size).quantize(_DECIMAL_TICK),
            conviction_scalar=Decimal("1"),
            leg_role=leg.leg_role,
        )

    def _build_client_order_id(self, execution_sequence: int, leg: CtfLegManifest) -> str:
        return f"CTF-{execution_sequence:06d}-{leg.leg_index}"

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    @staticmethod
    def _map_status_receipt(
        *,
        context: PriorityOrderContext,
        submit_response: VenueOrderResponse,
        order_status: VenueOrderStatus,
        submitted_size: Decimal,
        dispatch_timestamp_ms: int,
    ) -> DispatchReceipt | None:
        if order_status.filled_size >= submitted_size and order_status.average_fill_price is not None:
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=True,
                fill_price=order_status.average_fill_price.quantize(_DECIMAL_TICK),
                fill_size=submitted_size.quantize(_DECIMAL_TICK),
                serialized_envelope="{}",
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="FULL",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=Decimal("0"),
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        if order_status.filled_size > Decimal("0") and order_status.average_fill_price is not None:
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=True,
                fill_price=None,
                fill_size=submitted_size.quantize(_DECIMAL_TICK),
                serialized_envelope="{}",
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                partial_fill_size=order_status.filled_size.quantize(_DECIMAL_TICK),
                partial_fill_price=order_status.average_fill_price.quantize(_DECIMAL_TICK),
                fill_status="PARTIAL",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=order_status.remaining_size.quantize(_DECIMAL_TICK),
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        return None

    @staticmethod
    def _rejected_dispatch_receipt(
        *,
        context: PriorityOrderContext,
        submitted_size: Decimal,
        dispatch_timestamp_ms: int,
        submit_response: VenueOrderResponse,
        rejection_reason: str,
        order_status: VenueOrderStatus | None = None,
        cancel_response: VenueCancelResponse | None = None,
    ) -> DispatchReceipt:
        venue_timestamp_ms = submit_response.venue_timestamp_ms
        if cancel_response is not None and cancel_response.venue_timestamp_ms is not None:
            venue_timestamp_ms = cancel_response.venue_timestamp_ms
        remaining_size = submitted_size
        if order_status is not None:
            remaining_size = order_status.remaining_size.quantize(_DECIMAL_TICK)
        return DispatchReceipt(
            context=context,
            mode="live",
            executed=False,
            fill_price=None,
            fill_size=None,
            serialized_envelope="{}",
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            guard_reason=rejection_reason,
            partial_fill_size=None,
            partial_fill_price=None,
            fill_status="NONE",
            order_id=submit_response.client_order_id,
            execution_id=submit_response.client_order_id,
            remaining_size=remaining_size,
            venue_timestamp_ms=venue_timestamp_ms,
            latency_ms=submit_response.latency_ms,
        )

    @staticmethod
    def _leg_receipt(leg: CtfLegManifest, outcome: _LegDispatchOutcome) -> CtfLegReceipt:
        dispatch_receipt = outcome.dispatch_receipt
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
                fill_timestamp_ms=outcome.fill_timestamp_ms,
            )
        assert dispatch_receipt.fill_price is not None
        assert dispatch_receipt.fill_size is not None
        return CtfLegReceipt(
            leg_manifest=leg,
            dispatch_receipt=dispatch_receipt,
            fill_status="FILLED",
            filled_size=dispatch_receipt.fill_size.quantize(_DECIMAL_TICK),
            filled_price=dispatch_receipt.fill_price.quantize(_DECIMAL_TICK),
            fill_timestamp_ms=outcome.fill_timestamp_ms,
        )

    @staticmethod
    def _suppressed_leg_receipt(leg: CtfLegManifest, timestamp_ms: int, reason: str) -> CtfLegReceipt:
        context = ctf_to_context(
            market_id=leg.market_id,
            side=leg.side,
            target_price=leg.target_price,
            anchor_volume=leg.target_size,
            max_capital=(leg.target_price * leg.target_size).quantize(_DECIMAL_TICK),
            conviction_scalar=Decimal("1"),
            leg_role=leg.leg_role,
        )
        return CtfLegReceipt(
            leg_manifest=leg,
            dispatch_receipt=DispatchReceipt(
                context=context,
                mode="live",
                executed=False,
                fill_price=None,
                fill_size=None,
                serialized_envelope="{}",
                dispatch_timestamp_ms=timestamp_ms,
                guard_reason=reason,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="NONE",
                order_id=None,
                execution_id=None,
                remaining_size=None,
                venue_timestamp_ms=None,
                latency_ms=None,
            ),
            fill_status="SUPPRESSED",
            filled_size=Decimal("0"),
            filled_price=None,
            fill_timestamp_ms=None,
        )

    def _build_anchor_rejected_receipt(
        self,
        *,
        manifest: CtfExecutionManifest,
        anchor_leg_receipt: CtfLegReceipt,
        second_leg: CtfLegManifest,
        timestamp_ms: int,
    ) -> CtfExecutionReceipt:
        second_leg_receipt = self._suppressed_leg_receipt(second_leg, timestamp_ms, "ANCHOR_REJECTED")
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

    @staticmethod
    def _build_second_leg_rejected_receipt(
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
            total_capital_deployed=CtfLiveExecutor._total_capital_deployed((anchor_leg_receipt, second_leg_receipt)),
            execution_timestamp_ms=timestamp_ms,
        )

    @staticmethod
    def _build_fill_receipt(
        *,
        manifest: CtfExecutionManifest,
        yes_receipt: CtfLegReceipt,
        no_receipt: CtfLegReceipt,
        timestamp_ms: int,
    ) -> CtfExecutionReceipt:
        filled_cluster_size = min(yes_receipt.filled_size, no_receipt.filled_size)
        realized_net_edge = (
            Decimal("1")
            - CtfLiveExecutor._filled_price(yes_receipt)
            - CtfLiveExecutor._filled_price(no_receipt)
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
            total_capital_deployed=CtfLiveExecutor._total_capital_deployed((yes_receipt, no_receipt)),
            execution_timestamp_ms=timestamp_ms,
        )

    @staticmethod
    def _build_unwind_manifest(
        *,
        manifest: CtfExecutionManifest,
        anchor_leg_receipt: CtfLegReceipt,
        second_leg_receipt: CtfLegReceipt,
        timestamp_ms: int,
    ) -> CtfUnwindManifest:
        residual_size = (anchor_leg_receipt.filled_size - second_leg_receipt.filled_size).quantize(_DECIMAL_TICK)
        if residual_size <= Decimal("0"):
            raise ValueError("unwind manifest requires positive residual size")
        assert anchor_leg_receipt.filled_price is not None
        return CtfUnwindManifest(
            cluster_id=manifest.market_id,
            hanging_legs=(
                CtfUnwindLeg(
                    market_id=anchor_leg_receipt.leg_manifest.market_id,
                    side=anchor_leg_receipt.leg_manifest.side,
                    filled_size=residual_size,
                    filled_price=anchor_leg_receipt.filled_price,
                    current_best_bid=anchor_leg_receipt.filled_price,
                    estimated_unwind_cost=Decimal("0"),
                    leg_index=anchor_leg_receipt.leg_manifest.leg_index,
                ),
            ),
            unwind_reason="SECOND_LEG_REJECTED",
            original_manifest=manifest,
            unwind_timestamp_ms=timestamp_ms,
            total_estimated_unwind_cost=Decimal("0"),
            recommended_action="MARKET_SELL",
        )

    @staticmethod
    def _filled_price(receipt: CtfLegReceipt) -> Decimal:
        assert receipt.filled_price is not None
        return receipt.filled_price

    @staticmethod
    def _total_capital_deployed(receipts: tuple[CtfLegReceipt, ...]) -> Decimal:
        total = Decimal("0")
        for receipt in receipts:
            if receipt.filled_price is None or receipt.filled_size == Decimal("0"):
                continue
            total += receipt.filled_price * receipt.filled_size
        return total.quantize(_DECIMAL_TICK)