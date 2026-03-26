from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol

from src.execution.clob_signer import ClobSigner
from src.execution.clob_transport import ClobTransportHttpError
from src.execution.nonce_manager import ClobNonceManager
from src.execution.polymarket_clob_translator import (
    ClobOrderIntent,
    ClobPayloadBuilder,
    ClobReceiptParser,
    ClobTimeInForce,
    VenueRejectionReason,
)
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


class PolymarketClobClient(Protocol):
    owner_id: str

    def resolve_market_token(self, payload: Mapping[str, str]) -> Mapping[str, Any]:
        ...


class PolymarketClobTransport(Protocol):
    async def post_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | str:
        ...

    async def cancel_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | str:
        ...

    async def get_order(self, order_id: str) -> Mapping[str, Any] | str:
        ...

    async def get_expected_nonce(self, payload: Mapping[str, Any]) -> int:
        ...

    async def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        ...


def _run_async_transport(coro: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("PolymarketClobAdapter transport runner cannot use asyncio.run inside an active event loop")


class PolymarketClobAdapter(VenueAdapter):
    def __init__(
        self,
        client: PolymarketClobClient,
        transport: PolymarketClobTransport,
        payload_builder: ClobPayloadBuilder,
        receipt_parser: ClobReceiptParser,
        nonce_manager: ClobNonceManager,
        signer: ClobSigner,
        default_expiration: int = 0,
        transport_runner: Callable[[Awaitable[Any]], Any] | None = None,
    ) -> None:
        self._client = client
        self._transport = transport
        self._payload_builder = payload_builder
        self._receipt_parser = receipt_parser
        self._nonce_manager = nonce_manager
        self._signer = signer
        self._transport_runner = transport_runner or _run_async_transport
        if not isinstance(default_expiration, int) or default_expiration < 0:
            raise ValueError("default_expiration must be a non-negative int")
        self._default_expiration = default_expiration

    def submit_order(
        self,
        market_id: str,
        side: Literal["YES", "NO"],
        price: Decimal,
        size: Decimal,
        order_type: Literal["LIMIT", "MARKET"],
        client_order_id: str,
    ) -> VenueOrderResponse:
        time_in_force = self._map_order_type(order_type)
        token_lookup_payload = self._payload_builder.build_token_lookup_payload(
            condition_id=market_id,
            outcome=side,
        )
        token_lookup_response = self._client.resolve_market_token(token_lookup_payload)
        token_id = self._extract_token_id(token_lookup_response)
        nonce = self._nonce_manager.reserve_nonces(1)[0]
        intent = ClobOrderIntent(
            condition_id=market_id,
            token_id=token_id,
            outcome=side,
            action="BUY",
            price=price,
            size=size,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            post_only=False,
            nonce=nonce,
            expiration=self._default_expiration,
        )
        create_order_payload = self._payload_builder.build_create_order_payload(intent)
        signed_order = self._signer.sign_create_order_payload(create_order_payload)
        post_order_payload = self._payload_builder.build_post_order_payload(
            signed_order=signed_order,
            owner_id=self._client.owner_id,
            time_in_force=time_in_force,
            post_only=intent.post_only,
        )
        try:
            raw_response = self._transport_runner(self._transport.post_order(post_order_payload))
        except ClobTransportHttpError as exc:
            raw_response = exc.payload or {}
            if exc.status_code == 400 and self._is_stale_nonce_payload(raw_response):
                self._sync_nonce_from_payload_or_transport(
                    raw_response,
                    market_id=market_id,
                    client_order_id=client_order_id,
                )
        return self._receipt_parser.parse_submit_response(
            raw_response,
            expected_client_order_id=client_order_id,
        )

    def cancel_order(
        self,
        client_order_id: str,
        market_id: str,
    ) -> VenueCancelResponse:
        payload = self._payload_builder.build_cancel_payload(
            client_order_id=client_order_id,
            condition_id=market_id,
        )
        raw_response = self._transport_runner(self._transport.cancel_order(payload))
        return self._receipt_parser.parse_cancel_response(
            raw_response,
            expected_client_order_id=client_order_id,
        )

    def get_order_status(
        self,
        client_order_id: str,
    ) -> VenueOrderStatus:
        raw_response = self._transport_runner(self._transport.get_order(client_order_id))
        return self._receipt_parser.parse_order_status(
            raw_response,
            expected_client_order_id=client_order_id,
        )

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        balance = self._transport_runner(self._transport.get_wallet_balance(asset_symbol))
        if not isinstance(balance, Decimal) or not balance.is_finite():
            raise ValueError("wallet balance must be a finite Decimal")
        if balance < Decimal("0"):
            raise ValueError("wallet balance must be greater than or equal to 0")
        return balance

    def _sync_nonce_from_payload_or_transport(
        self,
        raw_response: Mapping[str, Any] | str,
        *,
        market_id: str,
        client_order_id: str,
    ) -> None:
        expected_nonce = self._extract_expected_nonce(raw_response)
        if expected_nonce is None:
            expected_nonce = self._transport_runner(
                self._transport.get_expected_nonce(
                    {
                        "address": self._signer.signer_address,
                        "owner": self._client.owner_id,
                        "market": market_id,
                        "clientOrderId": client_order_id,
                    }
                )
            )
        self._nonce_manager.sync_nonce(int(expected_nonce))

    def _is_stale_nonce_payload(self, raw_response: Mapping[str, Any] | str) -> bool:
        if not isinstance(raw_response, Mapping):
            return False
        return self._receipt_parser._rejection_reason(raw_response) == VenueRejectionReason.STALE_NONCE

    @staticmethod
    def _extract_expected_nonce(raw_response: Mapping[str, Any] | str) -> int | None:
        if not isinstance(raw_response, Mapping):
            return None
        for key in ("expectedNonce", "expected_nonce", "nextNonce", "next_nonce", "nonce"):
            value = raw_response.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        details = raw_response.get("details")
        if isinstance(details, Mapping):
            return PolymarketClobAdapter._extract_expected_nonce(details)
        return None

    @staticmethod
    def _map_order_type(order_type: Literal["LIMIT", "MARKET"]) -> ClobTimeInForce:
        if order_type == "LIMIT":
            return ClobTimeInForce.GTC
        if order_type == "MARKET":
            return ClobTimeInForce.IOC
        raise ValueError(f"Unsupported order_type: {order_type!r}")

    @staticmethod
    def _extract_token_id(response: Mapping[str, Any]) -> str:
        token_id = response.get("token_id") or response.get("tokenId") or response.get("asset_id")
        if token_id is None or not str(token_id).strip():
            raise ValueError("token lookup response must include token_id")
        return str(token_id)
