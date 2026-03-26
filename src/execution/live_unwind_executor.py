from __future__ import annotations

from decimal import Decimal

from src.execution.alpha_adapters import ctf_to_context, si9_to_context
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.ctf_execution_manifest import CtfExecutionManifest
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.si9_execution_manifest import Si9ExecutionManifest
from src.execution.unwind_executor_interface import UnwindExecutionReceipt, UnwindExecutor
from src.execution.unwind_manifest import UnwindActionTaken, UnwindManifest
from src.execution.venue_adapter_interface import VenueAdapter, VenueOrderResponse, VenueOrderStatus


_DECIMAL_TICK = Decimal("0.000001")


class LiveUnwindExecutor(UnwindExecutor):
    def __init__(
        self,
        venue_adapter: VenueAdapter,
        client_order_id_generator: ClientOrderIdGenerator,
    ) -> None:
        self._venue_adapter = venue_adapter
        self._client_order_id_generator = client_order_id_generator
        self._active_unwinds: dict[str, UnwindExecutionReceipt] = {}

    @property
    def active_unwind_count(self) -> int:
        return len(self._active_unwinds)

    def execute_unwind(
        self,
        manifest: UnwindManifest,
        current_timestamp_ms: int,
    ) -> UnwindExecutionReceipt:
        timestamp_ms = int(current_timestamp_ms)
        cluster_id = str(manifest.cluster_id).strip()
        if cluster_id in self._active_unwinds:
            return self._active_unwinds[cluster_id]

        if manifest.recommended_action == "HOLD_FOR_RECOVERY":
            return UnwindExecutionReceipt(
                manifest=manifest,
                action_taken="SKIPPED",
                legs_acted_on=tuple(),
                estimated_cost=manifest.total_estimated_unwind_cost,
                execution_timestamp_ms=timestamp_ms,
                notes="Live unwind deferred by HOLD_FOR_RECOVERY recommendation",
                per_leg_receipts=tuple(),
            )

        order_type = "MARKET" if manifest.recommended_action == "MARKET_SELL" else "LIMIT"
        dispatch_receipts: list[DispatchReceipt] = []
        acted_on: list[str] = []
        for leg in sorted(manifest.hanging_legs, key=lambda unwind_leg: unwind_leg.leg_index):
            client_order_id = self._client_order_id_generator.generate(
                market_id=leg.market_id,
                side=leg.side,
                timestamp_ms=timestamp_ms + leg.leg_index,
            )
            submit_response = self._venue_adapter.submit_order(
                market_id=leg.market_id,
                side=leg.side,
                price=leg.current_best_bid.quantize(_DECIMAL_TICK),
                size=leg.filled_size.quantize(_DECIMAL_TICK),
                order_type=order_type,
                client_order_id=client_order_id,
            )
            order_status = self._venue_adapter.get_order_status(client_order_id)
            dispatch_receipts.append(
                self._map_dispatch_receipt(
                    manifest=manifest,
                    market_id=leg.market_id,
                    side=leg.side,
                    submitted_size=leg.filled_size,
                    submitted_price=leg.current_best_bid,
                    dispatch_timestamp_ms=timestamp_ms,
                    submit_response=submit_response,
                    order_status=order_status,
                )
            )
            acted_on.append(leg.market_id)

        execution_timestamp_ms = max(
            [receipt.venue_timestamp_ms for receipt in dispatch_receipts if receipt.venue_timestamp_ms is not None]
            or [timestamp_ms]
        )
        action_taken: UnwindActionTaken = "MARKET_SELL"
        notes = "Live unwind submitted defensive unwind orders"
        if manifest.recommended_action == "PASSIVE_UNWIND":
            action_taken = "PASSIVE_UNWIND"
            notes = "Live unwind submitted passive unwind orders"
        receipt = UnwindExecutionReceipt(
            manifest=manifest,
            action_taken=action_taken,
            legs_acted_on=tuple(acted_on),
            estimated_cost=manifest.total_estimated_unwind_cost,
            execution_timestamp_ms=execution_timestamp_ms,
            notes=notes,
            per_leg_receipts=tuple(dispatch_receipts),
        )
        self._active_unwinds[cluster_id] = receipt
        return receipt

    def clear_unwind(self, cluster_id: str) -> None:
        self._active_unwinds.pop(str(cluster_id).strip(), None)

    def _map_dispatch_receipt(
        self,
        *,
        manifest: UnwindManifest,
        market_id: str,
        side: str,
        submitted_size: Decimal,
        submitted_price: Decimal,
        dispatch_timestamp_ms: int,
        submit_response: VenueOrderResponse,
        order_status: VenueOrderStatus,
    ) -> DispatchReceipt:
        context = self._build_context(
            manifest=manifest,
            market_id=market_id,
            side=side,
            submitted_size=submitted_size,
            submitted_price=submitted_price,
        )
        if submit_response.status == "REJECTED":
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=False,
                fill_price=None,
                fill_size=None,
                serialized_envelope="{}",
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                guard_reason=submit_response.rejection_reason,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="NONE",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=order_status.remaining_size.quantize(_DECIMAL_TICK),
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        if order_status.fill_status == "FILLED":
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=True,
                fill_price=order_status.average_fill_price,
                fill_size=order_status.filled_size.quantize(_DECIMAL_TICK),
                serialized_envelope="{}",
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="FULL",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=order_status.remaining_size.quantize(_DECIMAL_TICK),
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        if order_status.fill_status == "PARTIAL":
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=True,
                fill_price=None,
                fill_size=submitted_size.quantize(_DECIMAL_TICK),
                serialized_envelope="{}",
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                partial_fill_size=order_status.filled_size.quantize(_DECIMAL_TICK),
                partial_fill_price=order_status.average_fill_price,
                fill_status="PARTIAL",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=order_status.remaining_size.quantize(_DECIMAL_TICK),
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        return DispatchReceipt(
            context=context,
            mode="live",
            executed=True,
            fill_price=None,
            fill_size=None,
            serialized_envelope="{}",
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            partial_fill_size=None,
            partial_fill_price=None,
            fill_status="NONE",
            order_id=submit_response.client_order_id,
            execution_id=submit_response.client_order_id,
            remaining_size=order_status.remaining_size.quantize(_DECIMAL_TICK),
            venue_timestamp_ms=submit_response.venue_timestamp_ms,
            latency_ms=submit_response.latency_ms,
        )

    @staticmethod
    def _build_context(
        *,
        manifest: UnwindManifest,
        market_id: str,
        side: str,
        submitted_size: Decimal,
        submitted_price: Decimal,
    ):
        if isinstance(manifest.original_manifest, CtfExecutionManifest):
            return ctf_to_context(
                market_id=market_id,
                side=side,  # type: ignore[arg-type]
                target_price=submitted_price.quantize(_DECIMAL_TICK),
                anchor_volume=submitted_size.quantize(_DECIMAL_TICK),
                max_capital=(submitted_size * submitted_price).quantize(_DECIMAL_TICK),
                conviction_scalar=Decimal("1"),
                leg_role=None,
            )
        if isinstance(manifest.original_manifest, Si9ExecutionManifest):
            return si9_to_context(
                market_id=market_id,
                side="YES",
                target_price=submitted_price.quantize(_DECIMAL_TICK),
                anchor_volume=submitted_size.quantize(_DECIMAL_TICK),
                max_capital=(submitted_size * submitted_price).quantize(_DECIMAL_TICK),
                conviction_scalar=Decimal("1"),
            )
        raise ValueError("Unsupported original_manifest for unwind execution")