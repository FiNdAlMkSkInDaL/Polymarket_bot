from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from src.execution.alpha_adapters import ofi_to_context
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.ofi_local_exit_monitor import OfiExitDecision
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


_DECIMAL_TICK = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class _ActiveExitState:
    position_id: str
    market_id: str
    side: Literal["YES", "NO"]
    size: Decimal
    stage: Literal["TAKER_SUBMITTED", "PASSIVE_SUBMITTED", "PROMOTED_TO_TAKER"]
    submitted_client_order_id: str
    submitted_price: Decimal
    dispatch_timestamp_ms: int
    receipt: DispatchReceipt


class OfiExitRouter:
    def __init__(
        self,
        venue_adapter: VenueAdapter,
        client_order_id_generator: ClientOrderIdGenerator,
        *,
        passive_wait_ms: int = 15_000,
        max_slippage_ticks: int = 10,
        price_tick_size: Decimal = Decimal("0.0001"),
    ) -> None:
        if venue_adapter is None:
            raise ValueError("venue_adapter is required")
        if client_order_id_generator is None:
            raise ValueError("client_order_id_generator is required")
        if not isinstance(passive_wait_ms, int) or passive_wait_ms <= 0:
            raise ValueError("passive_wait_ms must be a strictly positive int")
        if not isinstance(max_slippage_ticks, int) or max_slippage_ticks < 0:
            raise ValueError("max_slippage_ticks must be a non-negative int")
        if not isinstance(price_tick_size, Decimal) or not price_tick_size.is_finite() or price_tick_size <= Decimal("0"):
            raise ValueError("price_tick_size must be a strictly positive Decimal")
        self._venue_adapter = venue_adapter
        self._client_order_id_generator = client_order_id_generator
        self._passive_wait_ms = passive_wait_ms
        self._max_slippage_ticks = max_slippage_ticks
        self._price_tick_size = price_tick_size
        self._active_exit_ledger: dict[str, _ActiveExitState] = {}

    @property
    def active_exit_count(self) -> int:
        return len(self._active_exit_ledger)

    def route_exit(
        self,
        position_state: dict,
        decision: OfiExitDecision,
    ) -> DispatchReceipt | None:
        position_id = self._position_id(position_state)
        if decision.action == "SUPPRESSED_BY_VACUUM":
            return None
        if position_id in self._active_exit_ledger:
            return None

        if decision.action in {"TARGET_HIT", "STOP_HIT"}:
            receipt = self._submit_exit_order(
                position_state=position_state,
                order_type="MARKET",
                price=self._market_price(position_state, decision),
            )
            self._active_exit_ledger[position_id] = _ActiveExitState(
                position_id=position_id,
                market_id=self._market_id(position_state),
                side=self._side(position_state),
                size=self._size(position_state),
                stage="TAKER_SUBMITTED",
                submitted_client_order_id=receipt.order_id or receipt.execution_id or "",
                submitted_price=self._market_price(position_state, decision),
                dispatch_timestamp_ms=self._timestamp_ms(position_state),
                receipt=receipt,
            )
            return receipt

        if decision.action == "TIME_STOP_TRIGGERED":
            passive_price = self._passive_price(position_state, decision)
            receipt = self._submit_exit_order(
                position_state=position_state,
                order_type="LIMIT",
                price=passive_price,
            )
            self._active_exit_ledger[position_id] = _ActiveExitState(
                position_id=position_id,
                market_id=self._market_id(position_state),
                side=self._side(position_state),
                size=self._size(position_state),
                stage="PASSIVE_SUBMITTED",
                submitted_client_order_id=receipt.order_id or receipt.execution_id or "",
                submitted_price=passive_price,
                dispatch_timestamp_ms=self._timestamp_ms(position_state),
                receipt=receipt,
            )
            return receipt

        return None

    def evaluate_passive_promotion(
        self,
        position_id: str,
        current_timestamp_ms: int,
        current_bbo: dict,
    ) -> DispatchReceipt | None:
        ledger_key = str(position_id or "").strip()
        active_exit = self._active_exit_ledger.get(ledger_key)
        if active_exit is None or active_exit.stage != "PASSIVE_SUBMITTED":
            return None

        elapsed_ms = int(current_timestamp_ms) - active_exit.dispatch_timestamp_ms
        if elapsed_ms <= self._passive_wait_ms and not self._breached_slippage(active_exit, current_bbo):
            return None

        cancel_response = self._venue_adapter.cancel_order(
            client_order_id=active_exit.submitted_client_order_id,
            market_id=active_exit.market_id,
        )
        if not cancel_response.cancelled:
            return None

        promotion_state = {
            "position_id": active_exit.position_id,
            "market_id": active_exit.market_id,
            "side": active_exit.side,
            "size": active_exit.size,
            "current_timestamp_ms": int(current_timestamp_ms),
            "current_best_bid": self._bbo_decimal(current_bbo, "best_bid"),
            "current_best_ask": self._bbo_decimal(current_bbo, "best_ask"),
        }
        promoted_receipt = self._submit_exit_order(
            position_state=promotion_state,
            order_type="MARKET",
            price=self._market_price(
                promotion_state,
                OfiExitDecision(action="TIME_STOP_TRIGGERED", trigger_price=self._bbo_decimal(current_bbo, "best_bid")),
            ),
        )
        self._active_exit_ledger[ledger_key] = replace(
            active_exit,
            stage="PROMOTED_TO_TAKER",
            submitted_client_order_id=promoted_receipt.order_id or promoted_receipt.execution_id or "",
            submitted_price=self._market_price(
                promotion_state,
                OfiExitDecision(action="TIME_STOP_TRIGGERED", trigger_price=self._bbo_decimal(current_bbo, "best_bid")),
            ),
            dispatch_timestamp_ms=int(current_timestamp_ms),
            receipt=promoted_receipt,
        )
        return promoted_receipt

    def clear_exit(self, position_id: str) -> None:
        self._active_exit_ledger.pop(str(position_id or "").strip(), None)

    def _submit_exit_order(
        self,
        *,
        position_state: dict,
        order_type: Literal["LIMIT", "MARKET"],
        price: Decimal,
    ) -> DispatchReceipt:
        market_id = self._market_id(position_state)
        side = self._side(position_state)
        size = self._size(position_state)
        dispatch_timestamp_ms = self._timestamp_ms(position_state)
        client_order_id = self._client_order_id_generator.generate(
            market_id=market_id,
            side=side,
            timestamp_ms=dispatch_timestamp_ms,
        )
        submit_response = self._venue_adapter.submit_order(
            market_id=market_id,
            side=side,
            price=price.quantize(_DECIMAL_TICK),
            size=size.quantize(_DECIMAL_TICK),
            order_type=order_type,
            client_order_id=client_order_id,
        )
        order_status = self._venue_adapter.get_order_status(client_order_id)
        return self._map_dispatch_receipt(
            position_state=position_state,
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            submit_response=submit_response,
            order_status=order_status,
            submitted_price=price,
            submitted_size=size,
        )

    def _map_dispatch_receipt(
        self,
        *,
        position_state: dict,
        dispatch_timestamp_ms: int,
        submit_response: VenueOrderResponse,
        order_status: VenueOrderStatus,
        submitted_price: Decimal,
        submitted_size: Decimal,
    ) -> DispatchReceipt:
        context = ofi_to_context(
            market_id=self._market_id(position_state),
            side=self._side(position_state),
            target_price=submitted_price.quantize(_DECIMAL_TICK),
            anchor_volume=submitted_size.quantize(_DECIMAL_TICK),
            max_capital=(submitted_size * submitted_price).quantize(_DECIMAL_TICK),
            conviction_scalar=Decimal("1"),
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

    def _market_price(self, position_state: dict, decision: OfiExitDecision) -> Decimal:
        best_bid = self._optional_decimal(position_state, "current_best_bid")
        if best_bid is not None and best_bid > Decimal("0"):
            return best_bid
        return decision.trigger_price

    def _passive_price(self, position_state: dict, decision: OfiExitDecision) -> Decimal:
        best_ask = self._optional_decimal(position_state, "current_best_ask")
        if best_ask is not None and best_ask > Decimal("0"):
            return best_ask
        return decision.trigger_price

    def _breached_slippage(self, active_exit: _ActiveExitState, current_bbo: dict) -> bool:
        if self._max_slippage_ticks == 0:
            return False
        current_best_bid = self._bbo_decimal(current_bbo, "best_bid")
        threshold = self._price_tick_size * Decimal(self._max_slippage_ticks)
        return active_exit.submitted_price - current_best_bid > threshold

    @staticmethod
    def _position_id(position_state: dict) -> str:
        position_id = str(position_state.get("position_id", "") or "").strip()
        if not position_id:
            raise ValueError("position_state.position_id must be a non-empty string")
        return position_id

    @staticmethod
    def _market_id(position_state: dict) -> str:
        market_id = str(position_state.get("market_id", "") or "").strip()
        if not market_id:
            raise ValueError("position_state.market_id must be a non-empty string")
        return market_id

    @staticmethod
    def _side(position_state: dict) -> Literal["YES", "NO"]:
        side = position_state.get("side")
        if side not in {"YES", "NO"}:
            raise ValueError("position_state.side must be 'YES' or 'NO'")
        return side

    @staticmethod
    def _size(position_state: dict) -> Decimal:
        size = position_state.get("size")
        if not isinstance(size, Decimal) or not size.is_finite() or size <= Decimal("0"):
            raise ValueError("position_state.size must be a strictly positive finite Decimal")
        return size

    @staticmethod
    def _timestamp_ms(position_state: dict) -> int:
        timestamp_ms = position_state.get("current_timestamp_ms")
        if not isinstance(timestamp_ms, int) or timestamp_ms <= 0:
            raise ValueError("position_state.current_timestamp_ms must be a strictly positive int")
        return timestamp_ms

    @staticmethod
    def _optional_decimal(position_state: dict, key: str) -> Decimal | None:
        value = position_state.get(key)
        if value is None:
            return None
        if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
            raise ValueError(f"position_state.{key} must be a non-negative finite Decimal when set")
        return value

    @staticmethod
    def _bbo_decimal(current_bbo: dict, key: str) -> Decimal:
        value = current_bbo.get(key)
        if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
            raise ValueError(f"current_bbo.{key} must be a non-negative finite Decimal")
        return value
