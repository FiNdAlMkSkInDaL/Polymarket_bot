from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol

from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.dispatch_guard import DispatchGuard
from src.execution.mev_router import MevExecutionBatch
from src.execution.mev_serializer import deserialize_envelope, serialize_mev_execution_batch
from src.execution.priority_context import PriorityOrderContext

if TYPE_CHECKING:
    from src.execution.venue_adapter_interface import VenueAdapter
else:
    VenueAdapter = Any


_logger = logging.getLogger(__name__)


class MevRouter(Protocol):
    def plan_priority_sequence(self, context: PriorityOrderContext) -> MevExecutionBatch:
        ...


@dataclass(frozen=True, slots=True)
class DispatchReceipt:
    context: PriorityOrderContext
    mode: Literal["paper", "dry_run", "live"]
    executed: bool
    fill_price: Decimal | None
    fill_size: Decimal | None
    serialized_envelope: str
    dispatch_timestamp_ms: int
    guard_reason: str | None = None
    partial_fill_size: Decimal | None = None
    partial_fill_price: Decimal | None = None
    fill_status: Literal["FULL", "PARTIAL", "NONE"] = "NONE"
    order_id: str | None = None
    execution_id: str | None = None
    remaining_size: Decimal | None = None
    venue_timestamp_ms: int | None = None
    latency_ms: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"paper", "dry_run", "live"}:
            raise ValueError(f"Unsupported dispatch mode: {self.mode!r}")
        derived_fill_status = self.fill_status
        if derived_fill_status == "NONE" and self.executed and self.fill_price is not None and self.fill_size is not None:
            derived_fill_status = "FULL"
            object.__setattr__(self, "fill_status", derived_fill_status)
        if derived_fill_status not in {"FULL", "PARTIAL", "NONE"}:
            raise ValueError(f"Unsupported fill_status: {derived_fill_status!r}")

        if derived_fill_status == "FULL":
            if self.fill_price is None or self.fill_size is None:
                raise ValueError("fill_status='FULL' requires fill_price and fill_size")
            if self.partial_fill_size is not None or self.partial_fill_price is not None:
                raise ValueError("fill_status='FULL' requires partial fill fields to be unset")
        elif derived_fill_status == "PARTIAL":
            if self.fill_size is None:
                raise ValueError("fill_status='PARTIAL' requires fill_size")
            if self.partial_fill_size is None or self.partial_fill_price is None:
                raise ValueError("fill_status='PARTIAL' requires partial fill fields")
            if self.partial_fill_size >= self.fill_size:
                raise ValueError("partial_fill_size must be strictly less than fill_size")
        else:
            if self.fill_price is not None or self.fill_size is not None or self.partial_fill_size is not None:
                raise ValueError("fill_status='NONE' requires fill_price, fill_size, and partial_fill_size to be unset")
            if self.partial_fill_price is not None:
                raise ValueError("fill_status='NONE' requires partial_fill_price to be unset")

        if self.executed:
            if self.mode in {"paper", "dry_run"} and derived_fill_status not in {"FULL", "PARTIAL"}:
                raise ValueError("executed receipts must use fill_status 'FULL' or 'PARTIAL'")
            if self.guard_reason is not None:
                raise ValueError("executed receipts must not set guard_reason")
        elif derived_fill_status != "NONE":
            raise ValueError("unexecuted receipts must use fill_status 'NONE'")
        if not isinstance(self.dispatch_timestamp_ms, int):
            raise ValueError("dispatch_timestamp_ms must be an int")

        if self.mode == "live":
            guard_blocked_before_venue = (
                not self.executed
                and self.guard_reason is not None
                and self.execution_id is None
                and self.order_id is None
                and self.remaining_size is None
                and self.venue_timestamp_ms is None
                and self.latency_ms is None
            )
            if not guard_blocked_before_venue:
                if self.execution_id is None or not str(self.execution_id).strip():
                    raise ValueError("live receipts must set execution_id")
                if self.remaining_size is None or not isinstance(self.remaining_size, Decimal) or not self.remaining_size.is_finite() or self.remaining_size < Decimal("0"):
                    raise ValueError("live receipts must set remaining_size to a non-negative Decimal")
                if self.latency_ms is None or not isinstance(self.latency_ms, int) or self.latency_ms < 0:
                    raise ValueError("live receipts must set latency_ms to a non-negative int")
            if self.venue_timestamp_ms is not None and not isinstance(self.venue_timestamp_ms, int):
                raise ValueError("venue_timestamp_ms must be an int or None")
            if self.order_id is not None and not str(self.order_id).strip():
                raise ValueError("order_id must be a non-empty string when set")
            if derived_fill_status == "FULL" and self.remaining_size != Decimal("0"):
                raise ValueError("live FULL receipts must set remaining_size to 0")
        else:
            if any(value is not None for value in (self.order_id, self.execution_id, self.remaining_size, self.venue_timestamp_ms, self.latency_ms)):
                raise ValueError("paper and dry_run receipts must not set live-only venue fields")


class PriorityDispatcher:
    def __init__(
        self,
        router: MevRouter,
        mode: Literal["paper", "dry_run", "live"],
        guard: DispatchGuard | None = None,
        guard_enabled: bool = True,
        venue_adapter: VenueAdapter | None = None,
        client_order_id_generator: ClientOrderIdGenerator | None = None,
    ):
        if mode not in {"paper", "dry_run", "live"}:
            raise ValueError(f"Unsupported priority dispatch mode: {mode!r}")
        if mode == "live" and venue_adapter is None:
            raise ValueError("live mode requires venue_adapter")
        if mode == "live" and client_order_id_generator is None:
            raise ValueError("live mode requires client_order_id_generator")
        self._router = router
        self._mode = mode
        self._guard = guard
        self._guard_enabled = bool(guard_enabled)
        self._venue_adapter = venue_adapter
        self._client_order_id_generator = client_order_id_generator

    @property
    def guard(self) -> DispatchGuard | None:
        return self._guard

    @property
    def guard_enabled(self) -> bool:
        return self._guard_enabled

    def dispatch(
        self,
        context: PriorityOrderContext,
        dispatch_timestamp_ms: int,
    ) -> DispatchReceipt:
        if self._guard is not None and self._guard_enabled:
            decision = self._guard.check(context, dispatch_timestamp_ms)
            if not decision.allowed:
                self._guard.record_suppression(context.signal_source)
                receipt = DispatchReceipt(
                    context=context,
                    mode=self._mode,
                    executed=False,
                    fill_price=None,
                    fill_size=None,
                    serialized_envelope="",
                    dispatch_timestamp_ms=dispatch_timestamp_ms,
                    guard_reason=decision.reason,
                    partial_fill_size=None,
                    partial_fill_price=None,
                    fill_status="NONE",
                )
                self._log_dispatch(receipt)
                return receipt

        batch = self._router.plan_priority_sequence(context)
        serialized_envelope = serialize_mev_execution_batch(batch)
        envelope = deserialize_envelope(serialized_envelope)

        fill_price: Decimal | None = None
        fill_size: Decimal | None = None
        executed = False
        first_payload = envelope["payloads"][0]

        if self._mode == "paper":
            fill_price = Decimal(first_payload["price"])
            fill_size = Decimal(first_payload["metadata"]["effective_size"])
            executed = True

        if self._mode == "live":
            receipt = self._dispatch_live_order(
                context=context,
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                serialized_envelope=serialized_envelope,
                first_payload=first_payload,
            )
            if self._guard is not None and self._guard_enabled and receipt.executed:
                self._guard.record_dispatch(context, dispatch_timestamp_ms)
            self._log_dispatch(receipt)
            return receipt

        receipt = DispatchReceipt(
            context=context,
            mode=self._mode,
            executed=executed,
            fill_price=fill_price,
            fill_size=fill_size,
            serialized_envelope=serialized_envelope,
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            partial_fill_size=None,
            partial_fill_price=None,
            fill_status="FULL" if executed else "NONE",
        )
        if self._guard is not None and self._guard_enabled:
            self._guard.record_dispatch(context, dispatch_timestamp_ms)
        self._log_dispatch(receipt)
        return receipt

    def _dispatch_live_order(
        self,
        *,
        context: PriorityOrderContext,
        dispatch_timestamp_ms: int,
        serialized_envelope: str,
        first_payload: dict,
    ) -> DispatchReceipt:
        if self._venue_adapter is None:
            raise ValueError("live mode requires venue_adapter")
        if self._client_order_id_generator is None:
            raise ValueError("live mode requires client_order_id_generator")

        signal_source = context.signal_source
        generator = self._client_order_id_generator
        if signal_source in {"OFI", "SI9", "CTF", "CONTAGION", "MANUAL"}:
            generator = generator.for_signal_source(signal_source)

        client_order_id = generator.generate(
            market_id=context.market_id,
            side=context.side,
            timestamp_ms=dispatch_timestamp_ms,
        )
        order_price = Decimal(first_payload["price"])
        effective_size = Decimal(first_payload["metadata"]["effective_size"])

        submit_response = self._venue_adapter.submit_order(
            market_id=context.market_id,
            side=context.side,
            price=order_price,
            size=effective_size,
            order_type="LIMIT",
            client_order_id=client_order_id,
        )
        order_status = self._venue_adapter.get_order_status(submit_response.client_order_id)

        if submit_response.status == "REJECTED":
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=False,
                fill_price=None,
                fill_size=None,
                serialized_envelope=serialized_envelope,
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                guard_reason=submit_response.rejection_reason,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="NONE",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=order_status.remaining_size,
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        if order_status.fill_status == "FILLED":
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=True,
                fill_price=order_status.average_fill_price,
                fill_size=order_status.filled_size,
                serialized_envelope=serialized_envelope,
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="FULL",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=order_status.remaining_size,
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        if order_status.fill_status == "PARTIAL":
            return DispatchReceipt(
                context=context,
                mode="live",
                executed=True,
                fill_price=None,
                fill_size=effective_size,
                serialized_envelope=serialized_envelope,
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                partial_fill_size=order_status.filled_size,
                partial_fill_price=order_status.average_fill_price,
                fill_status="PARTIAL",
                order_id=submit_response.client_order_id,
                execution_id=submit_response.client_order_id,
                remaining_size=order_status.remaining_size,
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        return DispatchReceipt(
            context=context,
            mode="live",
            executed=True,
            fill_price=None,
            fill_size=None,
            serialized_envelope=serialized_envelope,
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            partial_fill_size=None,
            partial_fill_price=None,
            fill_status="NONE",
            order_id=submit_response.client_order_id,
            execution_id=submit_response.client_order_id,
            remaining_size=order_status.remaining_size,
            venue_timestamp_ms=submit_response.venue_timestamp_ms,
            latency_ms=submit_response.latency_ms,
        )

    def _log_dispatch(self, receipt: DispatchReceipt) -> None:
        log_payload = {
            "mode": receipt.mode,
            "signal_source": receipt.context.signal_source,
            "market_id": receipt.context.market_id,
            "side": receipt.context.side,
            "conviction_scalar": format(receipt.context.conviction_scalar, ".6f"),
            "fill_price": None if receipt.fill_price is None else format(receipt.fill_price, ".6f"),
            "fill_size": None if receipt.fill_size is None else format(receipt.fill_size, ".6f"),
            "executed": receipt.executed,
            "fill_status": receipt.fill_status,
            "guard_reason": receipt.guard_reason,
            "order_id": receipt.order_id,
            "execution_id": receipt.execution_id,
            "remaining_size": None if receipt.remaining_size is None else format(receipt.remaining_size, ".6f"),
            "venue_timestamp_ms": receipt.venue_timestamp_ms,
            "latency_ms": receipt.latency_ms,
        }
        level = logging.INFO if receipt.executed else logging.DEBUG
        _logger.log(level, json.dumps(log_payload, sort_keys=True))