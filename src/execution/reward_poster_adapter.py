from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.venue_adapter_interface import VenueOrderStatus
from src.rewards.models import RewardPosterIntent


RewardQuoteStatus = Literal["REJECTED", "WORKING", "PARTIAL", "FILLED", "CANCELLED"]


@dataclass(frozen=True, slots=True)
class RewardQuoteState:
    quote_id: str
    market_id: str
    asset_id: str
    side: Literal["YES", "NO"]
    target_price: Decimal
    target_size: Decimal
    max_capital: Decimal
    status: RewardQuoteStatus
    order_id: str | None = None
    filled_size: Decimal = Decimal("0")
    remaining_size: Decimal = Decimal("0")
    filled_price: Decimal | None = None
    guard_reason: str | None = None
    last_update_ms: int = 0
    extra_payload: dict[str, object] | None = None


class RewardPosterAdapter:
    def __init__(self, dispatcher: PriorityDispatcher) -> None:
        self._dispatcher = dispatcher

    def submit_intent(self, intent: RewardPosterIntent, timestamp_ms: int) -> RewardQuoteState:
        context = intent.to_priority_context()
        decision = self._dispatcher.evaluate_intent(context, timestamp_ms, enforce_guard=True)
        if not decision.allowed:
            return self._state_from_intent(
                intent,
                status="REJECTED",
                guard_reason=decision.reason,
                last_update_ms=timestamp_ms,
            )

        receipt = self._dispatcher.dispatch(context, timestamp_ms, enforce_guard=True)
        if not receipt.executed:
            return self._state_from_intent(
                intent,
                status="REJECTED",
                order_id=receipt.order_id,
                guard_reason=receipt.guard_reason,
                last_update_ms=timestamp_ms,
            )

        if receipt.fill_status == "FULL":
            return self._state_from_intent(
                intent,
                status="FILLED",
                order_id=receipt.order_id,
                filled_size=receipt.fill_size or intent.target_size,
                remaining_size=Decimal("0"),
                filled_price=receipt.fill_price,
                last_update_ms=timestamp_ms,
            )

        if receipt.fill_status == "PARTIAL":
            partial_size = receipt.partial_fill_size or Decimal("0")
            remaining_size = receipt.remaining_size if receipt.remaining_size is not None else max(intent.target_size - partial_size, Decimal("0"))
            return self._state_from_intent(
                intent,
                status="PARTIAL",
                order_id=receipt.order_id,
                filled_size=partial_size,
                remaining_size=remaining_size,
                filled_price=receipt.partial_fill_price,
                last_update_ms=timestamp_ms,
            )

        remaining_size = receipt.remaining_size if receipt.remaining_size is not None else intent.target_size
        return self._state_from_intent(
            intent,
            status="WORKING",
            order_id=receipt.order_id,
            remaining_size=remaining_size,
            last_update_ms=timestamp_ms,
        )

    def sync_quote(self, state: RewardQuoteState, timestamp_ms: int) -> RewardQuoteState:
        if state.order_id is None or self._dispatcher.venue_adapter is None:
            return state
        order_status = self._dispatcher.venue_adapter.get_order_status(state.order_id)
        return self._state_from_order_status(state, order_status, timestamp_ms)

    def cancel_quote(self, state: RewardQuoteState, timestamp_ms: int) -> RewardQuoteState:
        if state.order_id is None or self._dispatcher.venue_adapter is None:
            return replace(state, status="CANCELLED", remaining_size=Decimal("0"), last_update_ms=timestamp_ms)
        response = self._dispatcher.venue_adapter.cancel_order(state.order_id, state.market_id)
        if response.cancelled:
            return replace(state, status="CANCELLED", remaining_size=Decimal("0"), last_update_ms=timestamp_ms)
        return replace(state, guard_reason=response.rejection_reason, last_update_ms=timestamp_ms)

    def replace_quote(self, state: RewardQuoteState, intent: RewardPosterIntent, timestamp_ms: int) -> RewardQuoteState:
        cancelled_state = self.cancel_quote(state, timestamp_ms)
        if cancelled_state.status != "CANCELLED":
            return cancelled_state
        return self.submit_intent(intent, timestamp_ms)

    @staticmethod
    def _state_from_intent(
        intent: RewardPosterIntent,
        *,
        status: RewardQuoteStatus,
        order_id: str | None = None,
        filled_size: Decimal = Decimal("0"),
        remaining_size: Decimal = Decimal("0"),
        filled_price: Decimal | None = None,
        guard_reason: str | None = None,
        last_update_ms: int,
    ) -> RewardQuoteState:
        return RewardQuoteState(
            quote_id=intent.quote_id,
            market_id=intent.market_id,
            asset_id=intent.asset_id,
            side=intent.side,
            target_price=intent.target_price,
            target_size=intent.target_size,
            max_capital=intent.max_capital,
            status=status,
            order_id=order_id,
            filled_size=filled_size,
            remaining_size=remaining_size,
            filled_price=filled_price,
            guard_reason=guard_reason,
            last_update_ms=last_update_ms,
            extra_payload=intent.as_signal_metadata(),
        )

    @staticmethod
    def _state_from_order_status(
        state: RewardQuoteState,
        order_status: VenueOrderStatus,
        timestamp_ms: int,
    ) -> RewardQuoteState:
        if order_status.fill_status == "FILLED":
            return replace(
                state,
                status="FILLED",
                filled_size=order_status.filled_size,
                remaining_size=Decimal("0"),
                filled_price=order_status.average_fill_price,
                last_update_ms=timestamp_ms,
            )
        if order_status.fill_status == "PARTIAL":
            return replace(
                state,
                status="PARTIAL",
                filled_size=order_status.filled_size,
                remaining_size=order_status.remaining_size,
                filled_price=order_status.average_fill_price,
                last_update_ms=timestamp_ms,
            )
        if order_status.fill_status == "CANCELLED":
            return replace(state, status="CANCELLED", remaining_size=Decimal("0"), last_update_ms=timestamp_ms)
        return replace(state, status="WORKING", remaining_size=order_status.remaining_size, last_update_ms=timestamp_ms)