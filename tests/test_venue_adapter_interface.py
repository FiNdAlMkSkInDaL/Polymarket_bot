from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from src.execution.alpha_adapters import ofi_to_context
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.live_wallet_balance import LiveWalletBalanceProvider
from src.execution.mev_router import MevExecutionRouter
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.venue_adapter_interface import (
    VenueAdapter,
    VenueCancelResponse,
    VenueOrderResponse,
    VenueOrderStatus,
)


def _make_router() -> MevExecutionRouter:
    return MevExecutionRouter(
        lambda market_id: {
            "yes_bid": 0.45,
            "yes_ask": 0.55,
            "no_bid": 0.45,
            "no_ask": 0.55,
        }
    )


def _make_context():
    return ofi_to_context(
        market_id="MKT_PRIORITY",
        side="YES",
        target_price=Decimal("0.640000"),
        anchor_volume=Decimal("50.000000"),
        max_capital=Decimal("100.000000"),
        conviction_scalar=Decimal("0.850000"),
    )


def _make_guard() -> DispatchGuard:
    return DispatchGuard(
        DispatchGuardConfig(
            dedup_window_ms=100,
            max_dispatches_per_source_per_window=2,
            rate_window_ms=200,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_ms=300,
            max_open_positions_per_market=3,
        )
    )


def _make_client_order_id_generator(session_id: str = "a3f9b2c1-session") -> ClientOrderIdGenerator:
    return ClientOrderIdGenerator("OFI", session_id)


def _expected_client_order_id(timestamp_ms: int, market_id: str = "MKT_PRIORITY", session_id: str = "a3f9b2c1-session") -> str:
    return f"OFI-{session_id[:8]}-{market_id[:8]}-Y-{timestamp_ms}"


def _make_live_dispatcher(
    adapter: MockVenueAdapter,
    *,
    guard: DispatchGuard | None = None,
    session_id: str = "a3f9b2c1-session",
) -> PriorityDispatcher:
    return PriorityDispatcher(
        _make_router(),
        "live",
        guard=guard,
        venue_adapter=adapter,
        client_order_id_generator=_make_client_order_id_generator(session_id),
        wallet_balance_provider=LiveWalletBalanceProvider(
            adapter,
            tracked_assets=["USDC"],
            initial_balances={"USDC": Decimal("100.000000")},
        ),
    )


class MockVenueAdapter(VenueAdapter):
    def __init__(
        self,
        *,
        submit_response: VenueOrderResponse | None = None,
        status_response: VenueOrderStatus | None = None,
        cancel_response: VenueCancelResponse | None = None,
        wallet_balance: Decimal = Decimal("100.000000"),
    ) -> None:
        self._submit_response = submit_response or VenueOrderResponse(
            client_order_id="ROUTE-1-1",
            venue_order_id="VENUE-1",
            status="ACCEPTED",
            rejection_reason=None,
            venue_timestamp_ms=1234,
            latency_ms=12,
        )
        self._status_response = status_response or VenueOrderStatus(
            client_order_id="ROUTE-1-1",
            venue_order_id="VENUE-1",
            fill_status="OPEN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("42.500000"),
            average_fill_price=None,
        )
        self._cancel_response = cancel_response or VenueCancelResponse(
            client_order_id="ROUTE-1-1",
            cancelled=True,
            rejection_reason=None,
            venue_timestamp_ms=1235,
        )
        self.submit_calls: list[dict[str, object]] = []
        self.cancel_calls: list[dict[str, object]] = []
        self.status_calls: list[str] = []
        self.balance_calls: list[str] = []
        self.wallet_balance = wallet_balance

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
        return replace(self._submit_response, client_order_id=client_order_id)

    def cancel_order(self, client_order_id: str, market_id: str) -> VenueCancelResponse:
        self.cancel_calls.append({"client_order_id": client_order_id, "market_id": market_id})
        return replace(self._cancel_response, client_order_id=client_order_id)

    def get_order_status(self, client_order_id: str) -> VenueOrderStatus:
        self.status_calls.append(client_order_id)
        return replace(self._status_response, client_order_id=client_order_id)

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        self.balance_calls.append(asset_symbol)
        return self.wallet_balance


def test_mock_venue_adapter_satisfies_abc() -> None:
    assert isinstance(MockVenueAdapter(), VenueAdapter)


def test_mock_venue_adapter_returns_configured_wallet_balance() -> None:
    adapter = MockVenueAdapter(wallet_balance=Decimal("42.500000"))

    balance = adapter.get_wallet_balance("USDC")

    assert balance == Decimal("42.500000")
    assert adapter.balance_calls == ["USDC"]


def test_venue_order_response_is_frozen() -> None:
    response = VenueOrderResponse(
        client_order_id="CID-1",
        venue_order_id="VID-1",
        status="ACCEPTED",
        rejection_reason=None,
        venue_timestamp_ms=1000,
        latency_ms=5,
    )

    with pytest.raises(FrozenInstanceError):
        response.status = "REJECTED"  # type: ignore[misc]


def test_venue_order_response_rejected_requires_reason() -> None:
    with pytest.raises(ValueError, match="rejection_reason"):
        VenueOrderResponse(
            client_order_id="CID-1",
            venue_order_id=None,
            status="REJECTED",
            rejection_reason=None,
            venue_timestamp_ms=1000,
            latency_ms=5,
        )


def test_venue_order_response_non_rejected_cannot_set_reason() -> None:
    with pytest.raises(ValueError, match="rejection_reason"):
        VenueOrderResponse(
            client_order_id="CID-1",
            venue_order_id="VID-1",
            status="ACCEPTED",
            rejection_reason="venue said no",
            venue_timestamp_ms=1000,
            latency_ms=5,
        )


def test_venue_cancel_response_failed_cancel_requires_reason() -> None:
    with pytest.raises(ValueError, match="rejection_reason"):
        VenueCancelResponse(
            client_order_id="CID-1",
            cancelled=False,
            rejection_reason=None,
            venue_timestamp_ms=1000,
        )


def test_venue_order_status_filled_requires_average_fill_price() -> None:
    with pytest.raises(ValueError, match="average_fill_price"):
        VenueOrderStatus(
            client_order_id="CID-1",
            venue_order_id="VID-1",
            fill_status="FILLED",
            filled_size=Decimal("10"),
            remaining_size=Decimal("0"),
            average_fill_price=None,
        )


def test_venue_order_status_negative_remaining_size_raises() -> None:
    with pytest.raises(ValueError, match="remaining_size"):
        VenueOrderStatus(
            client_order_id="CID-1",
            venue_order_id="VID-1",
            fill_status="OPEN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("-1"),
            average_fill_price=None,
        )


def test_venue_order_status_filled_requires_zero_remaining_size() -> None:
    with pytest.raises(ValueError, match="remaining_size"):
        VenueOrderStatus(
            client_order_id="CID-1",
            venue_order_id="VID-1",
            fill_status="FILLED",
            filled_size=Decimal("10"),
            remaining_size=Decimal("1"),
            average_fill_price=Decimal("0.45"),
        )


def test_live_dispatcher_requires_venue_adapter() -> None:
    with pytest.raises(ValueError, match="venue_adapter"):
        PriorityDispatcher(_make_router(), "live")  # type: ignore[arg-type]


def test_paper_receipt_live_fields_are_none() -> None:
    receipt = PriorityDispatcher(_make_router(), "paper").dispatch(_make_context(), 10)

    assert receipt.order_id is None
    assert receipt.execution_id is None
    assert receipt.remaining_size is None
    assert receipt.venue_timestamp_ms is None
    assert receipt.latency_ms is None


def test_dry_run_receipt_live_fields_are_none() -> None:
    receipt = PriorityDispatcher(_make_router(), "dry_run").dispatch(_make_context(), 10)

    assert receipt.order_id is None
    assert receipt.execution_id is None
    assert receipt.remaining_size is None
    assert receipt.venue_timestamp_ms is None
    assert receipt.latency_ms is None


def test_live_dispatcher_submits_limit_order_with_router_price_and_effective_size() -> None:
    adapter = MockVenueAdapter()
    dispatcher = _make_live_dispatcher(adapter)

    dispatcher.dispatch(_make_context(), 10)

    assert adapter.submit_calls == [
        {
            "market_id": "MKT_PRIORITY",
            "side": "YES",
            "price": Decimal("0.640001"),
            "size": Decimal("42.500000"),
            "order_type": "LIMIT",
            "client_order_id": _expected_client_order_id(10),
        }
    ]


def test_live_dispatcher_uses_deterministic_client_order_id() -> None:
    adapter = MockVenueAdapter()
    dispatcher = _make_live_dispatcher(adapter)

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.execution_id == _expected_client_order_id(10)
    assert adapter.status_calls == [_expected_client_order_id(10)]


def test_live_dispatcher_populates_open_receipt_fields_from_venue() -> None:
    adapter = MockVenueAdapter(
        submit_response=VenueOrderResponse(
            client_order_id="ROUTE-1-1",
            venue_order_id="VENUE-OPEN",
            status="ACCEPTED",
            rejection_reason=None,
            venue_timestamp_ms=2222,
            latency_ms=17,
        ),
        status_response=VenueOrderStatus(
            client_order_id="ROUTE-1-1",
            venue_order_id="VENUE-OPEN",
            fill_status="OPEN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("42.500000"),
            average_fill_price=None,
        ),
    )
    dispatcher = _make_live_dispatcher(adapter)

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.executed is True
    assert receipt.fill_status == "NONE"
    assert receipt.order_id == _expected_client_order_id(10)
    assert receipt.execution_id == _expected_client_order_id(10)
    assert receipt.remaining_size == Decimal("42.500000")
    assert receipt.venue_timestamp_ms == 2222
    assert receipt.latency_ms == 17


def test_live_dispatcher_populates_full_fill_receipt_fields_from_status() -> None:
    adapter = MockVenueAdapter(
        status_response=VenueOrderStatus(
            client_order_id="ROUTE-1-1",
            venue_order_id="VENUE-FULL",
            fill_status="FILLED",
            filled_size=Decimal("42.500000"),
            remaining_size=Decimal("0"),
            average_fill_price=Decimal("0.640001"),
        )
    )
    dispatcher = _make_live_dispatcher(adapter)

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.executed is True
    assert receipt.fill_status == "FULL"
    assert receipt.fill_price == Decimal("0.640001")
    assert receipt.fill_size == Decimal("42.500000")
    assert receipt.remaining_size == Decimal("0")


def test_live_dispatcher_maps_partial_fill_to_partial_receipt_shape() -> None:
    adapter = MockVenueAdapter(
        status_response=VenueOrderStatus(
            client_order_id="ROUTE-1-1",
            venue_order_id="VENUE-PARTIAL",
            fill_status="PARTIAL",
            filled_size=Decimal("10.000000"),
            remaining_size=Decimal("32.500000"),
            average_fill_price=Decimal("0.630000"),
        )
    )
    dispatcher = _make_live_dispatcher(adapter)

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.executed is True
    assert receipt.fill_status == "PARTIAL"
    assert receipt.fill_size == Decimal("42.500000")
    assert receipt.partial_fill_size == Decimal("10.000000")
    assert receipt.partial_fill_price == Decimal("0.630000")
    assert receipt.remaining_size == Decimal("32.500000")


def test_live_dispatcher_maps_pending_submission_to_executed_open_receipt() -> None:
    adapter = MockVenueAdapter(
        submit_response=VenueOrderResponse(
            client_order_id="ROUTE-1-1",
            venue_order_id="VENUE-PENDING",
            status="PENDING",
            rejection_reason=None,
            venue_timestamp_ms=3000,
            latency_ms=21,
        )
    )
    dispatcher = _make_live_dispatcher(adapter)

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.executed is True
    assert receipt.fill_status == "NONE"
    assert receipt.order_id == _expected_client_order_id(10)
    assert receipt.latency_ms == 21


def test_live_dispatcher_maps_rejection_reason_into_unexecuted_receipt() -> None:
    adapter = MockVenueAdapter(
        submit_response=VenueOrderResponse(
            client_order_id="ROUTE-1-1",
            venue_order_id=None,
            status="REJECTED",
            rejection_reason="PRICE_OUT_OF_BOUNDS",
            venue_timestamp_ms=4444,
            latency_ms=30,
        ),
        status_response=VenueOrderStatus(
            client_order_id="ROUTE-1-1",
            venue_order_id=None,
            fill_status="UNKNOWN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("42.500000"),
            average_fill_price=None,
        ),
    )
    dispatcher = _make_live_dispatcher(adapter)

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.executed is False
    assert receipt.fill_status == "NONE"
    assert receipt.guard_reason == "PRICE_OUT_OF_BOUNDS"
    assert receipt.order_id == _expected_client_order_id(10)
    assert receipt.remaining_size == Decimal("42.500000")


def test_live_dispatcher_guard_rejection_skips_submit() -> None:
    adapter = MockVenueAdapter()
    dispatcher = _make_live_dispatcher(adapter, guard=_make_guard())
    context = _make_context()

    dispatcher.dispatch(context, 10)
    rejected = dispatcher.dispatch(context, 20)

    assert rejected.executed is False
    assert rejected.guard_reason == "DUPLICATE"
    assert adapter.submit_calls == [
        {
            "market_id": "MKT_PRIORITY",
            "side": "YES",
            "price": Decimal("0.640001"),
            "size": Decimal("42.500000"),
            "order_type": "LIMIT",
            "client_order_id": _expected_client_order_id(10),
        }
    ]


def test_live_dispatcher_uses_router_effective_size_not_anchor_volume() -> None:
    adapter = MockVenueAdapter()
    dispatcher = _make_live_dispatcher(adapter)

    dispatcher.dispatch(_make_context(), 10)

    assert adapter.submit_calls[0]["size"] != Decimal("50.000000")
    assert adapter.submit_calls[0]["size"] == Decimal("42.500000")


def test_mock_venue_adapter_cancel_returns_deterministic_response() -> None:
    adapter = MockVenueAdapter()

    response = adapter.cancel_order("ROUTE-1-1", "MKT_PRIORITY")

    assert response.cancelled is True
    assert adapter.cancel_calls == [{"client_order_id": "ROUTE-1-1", "market_id": "MKT_PRIORITY"}]