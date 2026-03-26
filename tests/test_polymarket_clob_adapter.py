from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

import pytest
from py_clob_client.config import get_contract_config

from src.execution.alpha_adapters import ofi_to_context
from src.execution.clob_signer import ClobSigner
from src.execution.clob_transport import ClobTransportHttpError
from src.execution.nonce_manager import ClobNonceManager
from src.execution.polymarket_clob_adapter import PolymarketClobAdapter
from src.execution.polymarket_clob_translator import (
    ClobOrderIntent,
    ClobPayloadBuilder,
    ClobReceiptParser,
    ClobTimeInForce,
    VenueRejectionReason,
)
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderStatus


def _condition_id() -> str:
    return "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _intent(
    *,
    condition_id: str | None = None,
    token_id: str = "123456789",
    time_in_force: ClobTimeInForce = ClobTimeInForce.GTC,
    post_only: bool = False,
) -> ClobOrderIntent:
    return ClobOrderIntent(
        condition_id=_condition_id() if condition_id is None else condition_id,
        token_id=token_id,
        outcome="YES",
        action="BUY",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        time_in_force=time_in_force,
        client_order_id="OFI-a3f9b2c1-deadbeef-Y-1711234567890",
        post_only=post_only,
        fee_rate_bps=0,
        nonce=7,
        expiration=0,
    )


def _context():
    return ofi_to_context(
        market_id="MKT_PRIORITY",
        side="YES",
        target_price=Decimal("0.640000"),
        anchor_volume=Decimal("50.000000"),
        max_capital=Decimal("100.000000"),
        conviction_scalar=Decimal("0.850000"),
    )


class MockClobClient:
    def __init__(self) -> None:
        self.owner_id = "api-key-1"
        self.lookup_payloads: list[Mapping[str, str]] = []
        self.post_payloads: list[Mapping[str, Any]] = []
        self.cancel_payloads: list[Mapping[str, str]] = []
        self.status_payloads: list[Mapping[str, str]] = []
        self.lookup_response: Mapping[str, Any] = {"token_id": "123456789"}
        self.post_response: Mapping[str, Any] = {
            "status": "ACCEPTED",
            "orderID": "venue-1",
            "clientOrderId": "OFI-a3f9b2c1-deadbeef-Y-1711234567890",
            "timestampMs": 1711234567890,
            "latencyMs": 17,
        }
        self.cancel_response: Mapping[str, Any] = {
            "status": "CANCELLED",
            "clientOrderId": "OFI-a3f9b2c1-deadbeef-Y-1711234567890",
            "timestampMs": 1711234567900,
        }
        self.status_response: Mapping[str, Any] = {
            "status": "OPEN",
            "orderID": "venue-1",
            "clientOrderId": "OFI-a3f9b2c1-deadbeef-Y-1711234567890",
            "filledSize": "0",
            "remainingSize": "42.500000",
            "averagePrice": None,
        }

    def resolve_market_token(self, payload: Mapping[str, str]) -> Mapping[str, Any]:
        self.lookup_payloads.append(payload)
        return self.lookup_response


class MockClobTransport:
    def __init__(self) -> None:
        self.post_payloads: list[Mapping[str, Any]] = []
        self.cancel_payloads: list[Mapping[str, Any]] = []
        self.order_requests: list[str] = []
        self.expected_nonce_payloads: list[Mapping[str, Any]] = []
        self.post_response: Mapping[str, Any] = {
            "status": "ACCEPTED",
            "orderID": "venue-1",
            "clientOrderId": "OFI-a3f9b2c1-deadbeef-Y-1711234567890",
            "timestampMs": 1711234567890,
            "latencyMs": 17,
        }
        self.cancel_response: Mapping[str, Any] = {
            "status": "CANCELLED",
            "clientOrderId": "OFI-a3f9b2c1-deadbeef-Y-1711234567890",
            "timestampMs": 1711234567900,
        }
        self.order_response: Mapping[str, Any] = {
            "status": "OPEN",
            "orderID": "venue-1",
            "clientOrderId": "OFI-a3f9b2c1-deadbeef-Y-1711234567890",
            "filledSize": "0",
            "remainingSize": "42.500000",
            "averagePrice": None,
        }
        self.wallet_balance = Decimal("100.000000")
        self.expected_nonce = 19
        self.raise_on_post: Exception | None = None
        self.balance_requests: list[str] = []

    async def post_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | str:
        self.post_payloads.append(payload)
        if self.raise_on_post is not None:
            raise self.raise_on_post
        return self.post_response

    async def cancel_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | str:
        self.cancel_payloads.append(payload)
        return self.cancel_response

    async def get_order(self, order_id: str) -> Mapping[str, Any] | str:
        self.order_requests.append(order_id)
        return self.order_response

    async def get_expected_nonce(self, payload: Mapping[str, Any]) -> int:
        self.expected_nonce_payloads.append(payload)
        return self.expected_nonce

    async def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        self.balance_requests.append(asset_symbol)
        return self.wallet_balance


def _assert_no_floats(value: Any) -> None:
    if isinstance(value, float):
        raise AssertionError("float leaked into payload")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_floats(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_floats(nested)


def test_payload_builder_maps_decimal_fields_to_strings_without_float_leakage() -> None:
    payload = ClobPayloadBuilder().build_create_order_payload(_intent())

    assert payload["price"] == "0.640001"
    assert payload["size"] == "42.500000"
    _assert_no_floats(payload)


def test_payload_builder_maps_condition_id_into_lookup_payload() -> None:
    payload = ClobPayloadBuilder().build_token_lookup_payload(condition_id=_condition_id(), outcome="YES")

    assert payload == {"conditionId": _condition_id(), "outcome": "YES"}


def test_payload_builder_rejects_non_hex_condition_id() -> None:
    with pytest.raises(ValueError, match="condition_id"):
        ClobPayloadBuilder().build_token_lookup_payload(condition_id="not-hex", outcome="YES")


def test_payload_builder_translates_ioc_to_fak_wrapper_value() -> None:
    payload = ClobPayloadBuilder().build_post_order_payload(
        signed_order={"tokenId": "123456789"},
        owner_id="api-key-1",
        time_in_force=ClobTimeInForce.IOC,
        post_only=False,
    )

    assert payload["orderType"] == "FAK"
    assert payload["postOnly"] is False
    _assert_no_floats(payload)


def test_payload_builder_preserves_gtc_wrapper_value() -> None:
    payload = ClobPayloadBuilder().build_post_order_payload(
        signed_order={"tokenId": "123456789"},
        owner_id="api-key-1",
        time_in_force=ClobTimeInForce.GTC,
        post_only=False,
    )

    assert payload["orderType"] == "GTC"


def test_payload_builder_rejects_post_only_ioc() -> None:
    with pytest.raises(ValueError, match="post_only"):
        ClobPayloadBuilder().build_post_order_payload(
            signed_order={"tokenId": "123456789"},
            owner_id="api-key-1",
            time_in_force=ClobTimeInForce.IOC,
            post_only=True,
        )


def test_payload_builder_cancel_payload_maps_condition_id_and_client_order_id() -> None:
    payload = ClobPayloadBuilder().build_cancel_payload(
        client_order_id="CID-1",
        condition_id=_condition_id(),
    )

    assert payload == {"clientOrderId": "CID-1", "conditionId": _condition_id()}


def test_receipt_parser_maps_insufficient_balance_to_typed_rejection_reason() -> None:
    parser = ClobReceiptParser()

    response = parser.parse_submit_response(
        {
            "status": "REJECTED",
            "error": "not enough balance / allowance",
            "clientOrderId": "CID-1",
            "timestampMs": 100,
            "latencyMs": 5,
        },
        expected_client_order_id="CID-1",
    )

    assert response.status == "REJECTED"
    assert response.rejection_reason == VenueRejectionReason.INSUFFICIENT_BALANCE


def test_receipt_parser_maps_stale_nonce_to_typed_rejection_reason() -> None:
    parser = ClobReceiptParser()

    response = parser.parse_submit_response(
        {
            "status": "failed",
            "message": "stale nonce for account",
            "clientOrderId": "CID-1",
            "timestampMs": 100,
            "latencyMs": 5,
        },
        expected_client_order_id="CID-1",
    )

    assert response.rejection_reason == VenueRejectionReason.STALE_NONCE


def test_receipt_parser_maps_order_not_found_to_typed_rejection_reason() -> None:
    parser = ClobReceiptParser()

    response = parser.parse_submit_response(
        {
            "status": "error",
            "reason": "order not found",
            "clientOrderId": "CID-1",
            "timestampMs": 100,
            "latencyMs": 5,
        },
        expected_client_order_id="CID-1",
    )

    assert response.rejection_reason == VenueRejectionReason.ORDER_NOT_FOUND


def _signer() -> ClobSigner:
    return ClobSigner(
        private_key="0x59c6995e998f97a5a0044966f0945382dbf59596e17f7b7b6b6d3d4d6f8d2f4c",
        chain_id=137,
        exchange_address=get_contract_config(137).exchange,
    )


def _adapter(client: MockClobClient, *, starting_nonce: int = 7, default_expiration: int = 0) -> PolymarketClobAdapter:
    transport = MockClobTransport()
    return PolymarketClobAdapter(
        client,
        transport,
        ClobPayloadBuilder(),
        ClobReceiptParser(),
        ClobNonceManager(starting_nonce),
        _signer(),
        default_expiration=default_expiration,
    )


def _adapter_with_transport(
    client: MockClobClient,
    transport: MockClobTransport,
    *,
    starting_nonce: int = 7,
    default_expiration: int = 0,
) -> tuple[PolymarketClobAdapter, ClobNonceManager]:
    nonce_manager = ClobNonceManager(starting_nonce)
    return (
        PolymarketClobAdapter(
            client,
            transport,
            ClobPayloadBuilder(),
            ClobReceiptParser(),
            nonce_manager,
            _signer(),
            default_expiration=default_expiration,
        ),
        nonce_manager,
    )


def test_receipt_parser_rejects_float_numerics_to_prevent_precision_leakage() -> None:
    parser = ClobReceiptParser()

    with pytest.raises(ValueError, match="filledSize"):
        parser.parse_order_status(
            {
                "status": "PARTIAL",
                "clientOrderId": "CID-1",
                "filledSize": 1.25,
                "remainingSize": "2.75",
                "averagePrice": "0.55",
            },
            expected_client_order_id="CID-1",
        )


def test_receipt_parser_maps_open_status_with_decimal_strings() -> None:
    parser = ClobReceiptParser()

    status = parser.parse_order_status(
        {
            "status": "OPEN",
            "clientOrderId": "CID-1",
            "orderID": "venue-1",
            "filledSize": "0",
            "remainingSize": "42.500000",
            "averagePrice": None,
        },
        expected_client_order_id="CID-1",
    )

    assert status.fill_status == "OPEN"
    assert status.remaining_size == Decimal("42.500000")


def test_receipt_parser_builds_rejected_dispatch_receipt() -> None:
    parser = ClobReceiptParser()

    receipt = parser.parse_dispatch_receipt(
        context=_context(),
        serialized_envelope="{}",
        dispatch_timestamp_ms=10,
        effective_size=Decimal("42.500000"),
        submit_response_raw={
            "status": "REJECTED",
            "error": "order not found",
            "clientOrderId": "CID-1",
            "timestampMs": 11,
            "latencyMs": 7,
        },
        expected_client_order_id="CID-1",
    )

    assert receipt.executed is False
    assert receipt.order_id == "CID-1"
    assert receipt.execution_id == "CID-1"
    assert receipt.guard_reason == VenueRejectionReason.ORDER_NOT_FOUND


def test_adapter_satisfies_venue_adapter_interface() -> None:
    adapter = _adapter(MockClobClient())

    assert isinstance(adapter, VenueAdapter)


def test_adapter_submit_order_translates_condition_id_and_decimal_payloads() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport)

    response = adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-1",
    )

    assert response.status == "ACCEPTED"
    assert client.lookup_payloads == [{"conditionId": _condition_id(), "outcome": "YES"}]
    assert transport.post_payloads[0]["order"]["makerAmount"] == "27200042"
    assert transport.post_payloads[0]["order"]["takerAmount"] == "42500000"
    assert transport.post_payloads[0]["orderType"] == "GTC"
    _assert_no_floats(transport.post_payloads[0])


def test_adapter_market_order_uses_ioc_translation() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport)

    adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="MARKET",
        client_order_id="CID-1",
    )

    assert transport.post_payloads[0]["orderType"] == "FAK"


def test_adapter_cancel_order_maps_payload_and_response() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport)

    response = adapter.cancel_order("CID-1", _condition_id())

    assert response == VenueCancelResponse(
        client_order_id="OFI-a3f9b2c1-deadbeef-Y-1711234567890",
        cancelled=True,
        rejection_reason=None,
        venue_timestamp_ms=1711234567900,
    )
    assert transport.cancel_payloads == [{"clientOrderId": "CID-1", "conditionId": _condition_id()}]


def test_adapter_get_order_status_maps_decimal_strings() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport)

    status = adapter.get_order_status("CID-1")

    assert status == VenueOrderStatus(
        client_order_id="OFI-a3f9b2c1-deadbeef-Y-1711234567890",
        venue_order_id="venue-1",
        fill_status="OPEN",
        filled_size=Decimal("0"),
        remaining_size=Decimal("42.500000"),
        average_fill_price=None,
    )
    assert transport.order_requests == ["CID-1"]


def test_adapter_submit_order_maps_rejection_response() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    transport.post_response = {
        "status": "REJECTED",
        "error": "price out of bounds",
        "clientOrderId": "CID-1",
        "timestampMs": 1711234567890,
        "latencyMs": 9,
    }
    adapter, _ = _adapter_with_transport(client, transport)

    response = adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-1",
    )

    assert response.status == "REJECTED"
    assert response.rejection_reason == VenueRejectionReason.PRICE_OUT_OF_BOUNDS


def test_adapter_get_wallet_balance_uses_transport_decimal_boundary() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    transport.wallet_balance = Decimal("88.125000")
    adapter, _ = _adapter_with_transport(client, transport)

    balance = adapter.get_wallet_balance("USDC")

    assert balance == Decimal("88.125000")
    assert transport.balance_requests == ["USDC"]


def test_adapter_requires_token_lookup_response_to_include_token_id() -> None:
    client = MockClobClient()
    client.lookup_response = {"conditionId": _condition_id()}
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport)

    with pytest.raises(ValueError, match="token_id"):
        adapter.submit_order(
            market_id=_condition_id(),
            side="YES",
            price=Decimal("0.640001"),
            size=Decimal("42.500000"),
            order_type="LIMIT",
            client_order_id="CID-1",
        )


def test_adapter_preserves_signed_order_payload_without_float_leakage() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport)

    adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-1",
    )

    assert transport.post_payloads[0]["order"]["makerAmount"] == "27200042"
    assert transport.post_payloads[0]["order"]["nonce"] == "7"
    assert transport.post_payloads[0]["order"]["signature"].startswith("0x")
    _assert_no_floats(transport.post_payloads[0])


def test_adapter_reserves_incrementing_nonces_across_submissions() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport, starting_nonce=11)

    adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-1",
    )
    adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-2",
    )

    assert transport.post_payloads[0]["order"]["nonce"] == "11"
    assert transport.post_payloads[1]["order"]["nonce"] == "12"


def test_adapter_includes_injected_expiration_in_signed_order() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    adapter, _ = _adapter_with_transport(client, transport, default_expiration=1700000000)

    adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-1",
    )

    assert transport.post_payloads[0]["order"]["expiration"] == "1700000000"
    _assert_no_floats(transport.post_payloads[0])


def test_adapter_syncs_nonce_from_http_400_stale_nonce_payload() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    transport.raise_on_post = ClobTransportHttpError(
        status_code=400,
        payload={
            "status": "REJECTED",
            "error": "stale nonce",
            "expectedNonce": 23,
            "clientOrderId": "CID-1",
            "timestampMs": 1711234567890,
            "latencyMs": 9,
        },
    )
    adapter, nonce_manager = _adapter_with_transport(client, transport, starting_nonce=7)

    response = adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-1",
    )

    assert response.status == "REJECTED"
    assert response.rejection_reason == VenueRejectionReason.STALE_NONCE
    assert nonce_manager.next_nonce == 24
    assert transport.expected_nonce_payloads == []


def test_adapter_fetches_expected_nonce_when_stale_nonce_payload_has_no_nonce_field() -> None:
    client = MockClobClient()
    transport = MockClobTransport()
    transport.raise_on_post = ClobTransportHttpError(
        status_code=400,
        payload={
            "status": "REJECTED",
            "error": "nonce too low",
            "clientOrderId": "CID-1",
            "timestampMs": 1711234567890,
            "latencyMs": 9,
        },
    )
    transport.expected_nonce = 31
    adapter, nonce_manager = _adapter_with_transport(client, transport, starting_nonce=7)

    response = adapter.submit_order(
        market_id=_condition_id(),
        side="YES",
        price=Decimal("0.640001"),
        size=Decimal("42.500000"),
        order_type="LIMIT",
        client_order_id="CID-1",
    )

    assert response.status == "REJECTED"
    assert response.rejection_reason == VenueRejectionReason.STALE_NONCE
    assert nonce_manager.next_nonce == 32
    assert transport.expected_nonce_payloads == [
        {
            "address": _signer().signer_address,
            "owner": "api-key-1",
            "market": _condition_id(),
            "clientOrderId": "CID-1",
        }
    ]
