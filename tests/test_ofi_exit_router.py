from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.ofi_exit_router import OfiExitRouter
from src.execution.ofi_local_exit_monitor import OfiExitDecision
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


def _submit_response(
    *,
    status: str = "ACCEPTED",
    venue_order_id: str | None = "VENUE-1",
    rejection_reason: str | None = None,
    venue_timestamp_ms: int = 1000,
    latency_ms: int = 4,
) -> VenueOrderResponse:
    return VenueOrderResponse(
        client_order_id="template",
        venue_order_id=venue_order_id,
        status=status,  # type: ignore[arg-type]
        rejection_reason=rejection_reason,
        venue_timestamp_ms=venue_timestamp_ms,
        latency_ms=latency_ms,
    )


def _cancel_response(*, cancelled: bool = True, rejection_reason: str | None = None) -> VenueCancelResponse:
    return VenueCancelResponse(
        client_order_id="template",
        cancelled=cancelled,
        rejection_reason=rejection_reason,
        venue_timestamp_ms=1001,
    )


def _status(
    fill_status: str,
    *,
    filled_size: Decimal,
    remaining_size: Decimal,
    average_fill_price: Decimal | None,
    venue_order_id: str | None = "VENUE-1",
) -> VenueOrderStatus:
    return VenueOrderStatus(
        client_order_id="template",
        venue_order_id=venue_order_id,
        fill_status=fill_status,  # type: ignore[arg-type]
        filled_size=filled_size,
        remaining_size=remaining_size,
        average_fill_price=average_fill_price,
    )


class _ScriptedVenueAdapter(VenueAdapter):
    def __init__(
        self,
        *,
        submit_responses: list[VenueOrderResponse] | None = None,
        status_responses: list[VenueOrderStatus] | None = None,
        cancel_responses: list[VenueCancelResponse] | None = None,
    ) -> None:
        self._submit_responses = list(submit_responses or [])
        self._status_responses = list(status_responses or [])
        self._cancel_responses = list(cancel_responses or [])
        self.submit_calls: list[dict[str, object]] = []
        self.status_calls: list[str] = []
        self.cancel_calls: list[dict[str, object]] = []

    def submit_order(
        self,
        market_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        order_type: str,
        client_order_id: str,
    ) -> VenueOrderResponse:
        self.submit_calls.append(
            {
                "market_id": market_id,
                "side": side,
                "price": price,
                "size": size,
                "order_type": order_type,
                "client_order_id": client_order_id,
            }
        )
        index = len(self.submit_calls) - 1
        template = self._submit_responses[index] if index < len(self._submit_responses) else _submit_response()
        return replace(template, client_order_id=client_order_id)

    def cancel_order(self, client_order_id: str, market_id: str) -> VenueCancelResponse:
        self.cancel_calls.append({"client_order_id": client_order_id, "market_id": market_id})
        index = len(self.cancel_calls) - 1
        template = self._cancel_responses[index] if index < len(self._cancel_responses) else _cancel_response()
        return replace(template, client_order_id=client_order_id)

    def get_order_status(self, client_order_id: str) -> VenueOrderStatus:
        self.status_calls.append(client_order_id)
        index = len(self.status_calls) - 1
        template = self._status_responses[index] if index < len(self._status_responses) else _status(
            "OPEN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("3.000000"),
            average_fill_price=None,
        )
        return replace(template, client_order_id=client_order_id)

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        _ = asset_symbol
        return Decimal("100.000000")


def _router(
    *,
    submit_responses: list[VenueOrderResponse] | None = None,
    status_responses: list[VenueOrderStatus] | None = None,
    cancel_responses: list[VenueCancelResponse] | None = None,
    passive_wait_ms: int = 100,
    max_slippage_ticks: int = 10,
) -> tuple[OfiExitRouter, _ScriptedVenueAdapter]:
    adapter = _ScriptedVenueAdapter(
        submit_responses=submit_responses,
        status_responses=status_responses,
        cancel_responses=cancel_responses,
    )
    router = OfiExitRouter(
        adapter,
        ClientOrderIdGenerator("OFI", "abc12345-session"),
        passive_wait_ms=passive_wait_ms,
        max_slippage_ticks=max_slippage_ticks,
    )
    return router, adapter


def _position_state(**overrides: object) -> dict:
    state = {
        "position_id": "POS-1",
        "market_id": "MKT_OFI",
        "side": "NO",
        "size": Decimal("3.000000"),
        "current_timestamp_ms": 700,
        "current_best_bid": Decimal("0.390000"),
        "current_best_ask": Decimal("0.400000"),
    }
    state.update(overrides)
    return state


def _decision(action: str, trigger_price: str = "0.390000") -> OfiExitDecision:
    return OfiExitDecision(action=action, trigger_price=Decimal(trigger_price))  # type: ignore[arg-type]


def test_suppressed_by_vacuum_does_not_submit_order() -> None:
    router, adapter = _router()

    receipt = router.route_exit(_position_state(), _decision("SUPPRESSED_BY_VACUUM"))

    assert receipt is None
    assert adapter.submit_calls == []
    assert router.active_exit_count == 0


def test_target_hit_submits_market_order() -> None:
    router, adapter = _router()

    receipt = router.route_exit(_position_state(), _decision("TARGET_HIT", "0.410000"))

    assert isinstance(receipt, DispatchReceipt)
    assert adapter.submit_calls[0]["order_type"] == "MARKET"
    assert adapter.submit_calls[0]["price"] == Decimal("0.390000")
    assert receipt.context.signal_source == "OFI"


def test_stop_hit_submits_market_order() -> None:
    router, adapter = _router()

    receipt = router.route_exit(_position_state(), _decision("STOP_HIT", "0.380000"))

    assert isinstance(receipt, DispatchReceipt)
    assert adapter.submit_calls[0]["order_type"] == "MARKET"


def test_target_hit_duplicate_is_rejected_for_same_position() -> None:
    router, adapter = _router()

    first = router.route_exit(_position_state(), _decision("TARGET_HIT"))
    second = router.route_exit(_position_state(current_timestamp_ms=701), _decision("TARGET_HIT"))

    assert isinstance(first, DispatchReceipt)
    assert second is None
    assert len(adapter.submit_calls) == 1
    assert router.active_exit_count == 1


def test_duplicate_guard_is_scoped_per_position_id() -> None:
    router, adapter = _router()

    first = router.route_exit(_position_state(position_id="POS-1"), _decision("TARGET_HIT"))
    second = router.route_exit(_position_state(position_id="POS-2", current_timestamp_ms=701), _decision("TARGET_HIT"))

    assert isinstance(first, DispatchReceipt)
    assert isinstance(second, DispatchReceipt)
    assert len(adapter.submit_calls) == 2


def test_time_stop_submits_passive_limit_order_at_best_ask() -> None:
    router, adapter = _router()

    receipt = router.route_exit(_position_state(current_best_ask=Decimal("0.405000")), _decision("TIME_STOP_TRIGGERED"))

    assert isinstance(receipt, DispatchReceipt)
    assert adapter.submit_calls[0]["order_type"] == "LIMIT"
    assert adapter.submit_calls[0]["price"] == Decimal("0.405000")


def test_time_stop_uses_trigger_price_when_best_ask_missing() -> None:
    router, adapter = _router()

    router.route_exit(_position_state(current_best_ask=None), _decision("TIME_STOP_TRIGGERED", "0.392000"))

    assert adapter.submit_calls[0]["price"] == Decimal("0.392000")


def test_market_exit_uses_trigger_price_when_best_bid_missing() -> None:
    router, adapter = _router()

    router.route_exit(_position_state(current_best_bid=None), _decision("STOP_HIT", "0.381000"))

    assert adapter.submit_calls[0]["price"] == Decimal("0.381000")


def test_passive_promotion_returns_none_when_position_not_in_ledger() -> None:
    router, _ = _router()

    receipt = router.evaluate_passive_promotion(
        "POS-404",
        900,
        {"best_bid": Decimal("0.380000"), "best_ask": Decimal("0.390000")},
    )

    assert receipt is None


def test_passive_promotion_waits_below_threshold_and_without_slippage_breach() -> None:
    router, adapter = _router(passive_wait_ms=100, max_slippage_ticks=10)
    router.route_exit(_position_state(current_timestamp_ms=700), _decision("TIME_STOP_TRIGGERED"))

    receipt = router.evaluate_passive_promotion(
        "POS-1",
        750,
        {"best_bid": Decimal("0.399500"), "best_ask": Decimal("0.405000")},
    )

    assert receipt is None
    assert adapter.cancel_calls == []
    assert len(adapter.submit_calls) == 1


def test_passive_promotion_cancels_and_replaces_with_market_order_after_wait_breach() -> None:
    router, adapter = _router(passive_wait_ms=100)
    router.route_exit(_position_state(current_timestamp_ms=700, current_best_ask=Decimal("0.405000")), _decision("TIME_STOP_TRIGGERED"))

    receipt = router.evaluate_passive_promotion(
        "POS-1",
        801,
        {"best_bid": Decimal("0.390000"), "best_ask": Decimal("0.401000")},
    )

    assert isinstance(receipt, DispatchReceipt)
    assert adapter.cancel_calls == [{"client_order_id": "OFI-abc12345-MKT_OFI-N-700", "market_id": "MKT_OFI"}]
    assert len(adapter.submit_calls) == 2
    assert adapter.submit_calls[1]["order_type"] == "MARKET"
    assert adapter.submit_calls[1]["price"] == Decimal("0.390000")


def test_passive_promotion_cancels_and_replaces_when_slippage_threshold_is_breached() -> None:
    router, adapter = _router(passive_wait_ms=1_000, max_slippage_ticks=10)
    router.route_exit(_position_state(current_timestamp_ms=700, current_best_ask=Decimal("0.405000")), _decision("TIME_STOP_TRIGGERED"))

    receipt = router.evaluate_passive_promotion(
        "POS-1",
        750,
        {"best_bid": Decimal("0.390000"), "best_ask": Decimal("0.406000")},
    )

    assert isinstance(receipt, DispatchReceipt)
    assert len(adapter.cancel_calls) == 1
    assert len(adapter.submit_calls) == 2
    assert adapter.submit_calls[1]["order_type"] == "MARKET"


def test_passive_promotion_does_not_replace_when_cancel_fails() -> None:
    router, adapter = _router(
        passive_wait_ms=100,
        cancel_responses=[_cancel_response(cancelled=False, rejection_reason="already_filled")],
    )
    router.route_exit(_position_state(current_timestamp_ms=700), _decision("TIME_STOP_TRIGGERED"))

    receipt = router.evaluate_passive_promotion(
        "POS-1",
        801,
        {"best_bid": Decimal("0.390000"), "best_ask": Decimal("0.400000")},
    )

    assert receipt is None
    assert len(adapter.submit_calls) == 1
    assert len(adapter.cancel_calls) == 1


def test_route_exit_maps_full_fill_status() -> None:
    router, _ = _router(
        status_responses=[
            _status("FILLED", filled_size=Decimal("3.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.390000"))
        ]
    )

    receipt = router.route_exit(_position_state(), _decision("TARGET_HIT"))

    assert receipt is not None
    assert receipt.fill_status == "FULL"
    assert receipt.fill_price == Decimal("0.390000")


def test_route_exit_maps_partial_fill_status() -> None:
    router, _ = _router(
        status_responses=[
            _status("PARTIAL", filled_size=Decimal("1.500000"), remaining_size=Decimal("1.500000"), average_fill_price=Decimal("0.390000"))
        ]
    )

    receipt = router.route_exit(_position_state(), _decision("TARGET_HIT"))

    assert receipt is not None
    assert receipt.fill_status == "PARTIAL"
    assert receipt.partial_fill_size == Decimal("1.500000")


def test_route_exit_maps_rejected_submission_without_execution() -> None:
    router, _ = _router(
        submit_responses=[
            _submit_response(status="REJECTED", venue_order_id=None, rejection_reason="venue_reject")
        ],
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("3.000000"), average_fill_price=None, venue_order_id=None)
        ],
    )

    receipt = router.route_exit(_position_state(), _decision("TARGET_HIT"))

    assert receipt is not None
    assert receipt.executed is False
    assert receipt.guard_reason == "venue_reject"


def test_client_order_id_is_deterministic_for_initial_exit_submission() -> None:
    router, adapter = _router()

    router.route_exit(_position_state(current_timestamp_ms=700), _decision("TARGET_HIT"))

    assert adapter.submit_calls[0]["client_order_id"] == "OFI-abc12345-MKT_OFI-N-700"


def test_client_order_id_is_deterministic_for_promotion_submission() -> None:
    router, adapter = _router(passive_wait_ms=100)
    router.route_exit(_position_state(current_timestamp_ms=700), _decision("TIME_STOP_TRIGGERED"))

    router.evaluate_passive_promotion(
        "POS-1",
        801,
        {"best_bid": Decimal("0.389000"), "best_ask": Decimal("0.401000")},
    )

    assert adapter.submit_calls[1]["client_order_id"] == "OFI-abc12345-MKT_OFI-N-801"


def test_clear_exit_releases_position_for_future_submission() -> None:
    router, adapter = _router()
    router.route_exit(_position_state(position_id="POS-1"), _decision("TARGET_HIT"))

    router.clear_exit("POS-1")
    receipt = router.route_exit(_position_state(position_id="POS-1", current_timestamp_ms=701), _decision("TARGET_HIT"))

    assert isinstance(receipt, DispatchReceipt)
    assert len(adapter.submit_calls) == 2


def test_constructor_rejects_invalid_decimal_tick_size() -> None:
    with pytest.raises(ValueError, match="price_tick_size"):
        OfiExitRouter(
            _ScriptedVenueAdapter(),
            ClientOrderIdGenerator("OFI", "abc12345-session"),
            price_tick_size=Decimal("0"),
        )