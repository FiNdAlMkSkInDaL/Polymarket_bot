from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import aiohttp
import pytest

from src.execution.clob_transport import (
    AiohttpClobTransport,
    ClobTransportCircuitOpenError,
    ClobTransportHttpError,
    ClobTransportRateLimitError,
    ClobTransportTimeoutError,
)


class _FakeResponse:
    def __init__(self, *, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self._response = response

    async def __aenter__(self):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self._responses = list(responses)
        self.request_calls: list[dict[str, object]] = []
        self.closed = False

    def request(self, method: str, url: str, json=None):
        self.request_calls.append({"method": method, "url": url, "json": json})
        if not self._responses:
            raise AssertionError("no fake responses left")
        return _FakeRequestContext(self._responses.pop(0))

    async def close(self) -> None:
        self.closed = True


def _now_factory(start: int = 1_000):
    state = {"value": start}

    def _now() -> int:
        return state["value"]

    def _advance(delta: int) -> None:
        state["value"] += delta

    return _now, _advance


def _transport(*, responses: list[_FakeResponse | Exception], now_ms=None, backoff_ms: int = 2_000, nonce_sync_path: str | None = "/nonce"):
    session = _FakeSession(responses)
    now, advance = _now_factory() if now_ms is None else (now_ms, lambda delta: None)
    transport = AiohttpClobTransport(
        base_url="https://clob.polymarket.test",
        now_ms=now,
        rate_limit_backoff_ms=backoff_ms,
        nonce_sync_path=nonce_sync_path,
        session_factory=lambda **kwargs: session,
    )
    return transport, session, advance


@pytest.mark.asyncio
async def test_post_order_returns_decimal_parsed_payload() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text='{"price": 0.64, "nonce": 7}')])

    payload = await transport.post_order({"order": {"id": "1"}})

    assert payload == {"price": Decimal("0.64"), "nonce": 7}


@pytest.mark.asyncio
async def test_cancel_order_uses_delete_method() -> None:
    transport, session, _ = _transport(responses=[_FakeResponse(status=200, text='{"status": "ok"}')])

    await transport.cancel_order({"orderID": "1"})

    assert session.request_calls[0]["method"] == "DELETE"


@pytest.mark.asyncio
async def test_get_order_uses_expected_url_path() -> None:
    transport, session, _ = _transport(responses=[_FakeResponse(status=200, text='{"id": "venue-1"}')])

    await transport.get_order("venue-1")

    assert session.request_calls[0]["url"] == "https://clob.polymarket.test/data/order/venue-1"


@pytest.mark.asyncio
async def test_get_expected_nonce_reads_integer_field() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text='{"expectedNonce": 12}')])

    expected_nonce = await transport.get_expected_nonce({"address": "0xabc"})

    assert expected_nonce == 12


@pytest.mark.asyncio
async def test_get_expected_nonce_reads_string_field() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text='{"next_nonce": "13"}')])

    expected_nonce = await transport.get_expected_nonce({"address": "0xabc"})

    assert expected_nonce == 13


@pytest.mark.asyncio
async def test_get_wallet_balance_reads_decimal_string_field() -> None:
    transport, session, _ = _transport(responses=[_FakeResponse(status=200, text='{"balance": "42.500000"}')])

    balance = await transport.get_wallet_balance("USDC")

    assert balance == Decimal("42.500000")
    assert session.request_calls[0]["url"] == "https://clob.polymarket.test/balance-allowance?asset_type=COLLATERAL"


@pytest.mark.asyncio
async def test_get_wallet_balance_reads_decimal_response_value() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text='{"availableBalance": 17.25}')])

    balance = await transport.get_wallet_balance("USDC")

    assert balance == Decimal("17.25")


@pytest.mark.asyncio
async def test_get_wallet_balance_rejects_unsupported_symbol() -> None:
    transport, _, _ = _transport(responses=[])

    with pytest.raises(ValueError, match="Unsupported asset_symbol"):
        await transport.get_wallet_balance("BTC")


@pytest.mark.asyncio
async def test_get_wallet_balance_requires_balance_field_in_response() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text='{"ok": true}')])

    with pytest.raises(ValueError, match="balance field"):
        await transport.get_wallet_balance("USDC")


@pytest.mark.asyncio
async def test_empty_body_decodes_to_empty_mapping() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text="")])

    payload = await transport.post_order({"order": {"id": "1"}})

    assert payload == {}


@pytest.mark.asyncio
async def test_non_json_body_returns_raw_text() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text="accepted")])

    payload = await transport.post_order({"order": {"id": "1"}})

    assert payload == "accepted"


@pytest.mark.asyncio
async def test_http_500_raises_http_error_with_payload() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=500, text='{"error": "boom"}')])

    with pytest.raises(ClobTransportHttpError) as exc_info:
        await transport.post_order({"order": {"id": "1"}})

    assert exc_info.value.status_code == 500
    assert exc_info.value.payload == {"error": "boom"}


@pytest.mark.asyncio
async def test_timeout_raises_timeout_error() -> None:
    transport, _, _ = _transport(responses=[TimeoutError()])

    with pytest.raises(ClobTransportTimeoutError):
        await transport.post_order({"order": {"id": "1"}})


@pytest.mark.asyncio
async def test_aiohttp_client_error_raises_transport_error() -> None:
    transport, _, _ = _transport(responses=[aiohttp.ClientConnectionError("down")])

    with pytest.raises(Exception, match="transport request failed"):
        await transport.post_order({"order": {"id": "1"}})


@pytest.mark.asyncio
async def test_http_429_trips_rate_limit_circuit() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=429, text='{"error": "rate limited"}')], backoff_ms=3_000)

    with pytest.raises(ClobTransportRateLimitError) as exc_info:
        await transport.post_order({"order": {"id": "1"}})

    assert exc_info.value.backoff_until_ms == 4_000
    assert transport.rate_limit_open_until_ms == 4_000


@pytest.mark.asyncio
async def test_rate_limit_circuit_rejects_subsequent_requests_locally() -> None:
    transport, session, _ = _transport(
        responses=[_FakeResponse(status=429, text='{"error": "rate limited"}'), _FakeResponse(status=200, text='{"ok": true}')],
        backoff_ms=3_000,
    )

    with pytest.raises(ClobTransportRateLimitError):
        await transport.post_order({"order": {"id": "1"}})
    with pytest.raises(ClobTransportCircuitOpenError):
        await transport.post_order({"order": {"id": "2"}})

    assert len(session.request_calls) == 1


@pytest.mark.asyncio
async def test_rate_limit_circuit_resets_after_backoff_window() -> None:
    transport, session, advance = _transport(
        responses=[_FakeResponse(status=429, text='{"error": "rate limited"}'), _FakeResponse(status=200, text='{"ok": true}')],
        backoff_ms=2_000,
    )

    with pytest.raises(ClobTransportRateLimitError):
        await transport.post_order({"order": {"id": "1"}})
    advance(2_001)

    payload = await transport.post_order({"order": {"id": "2"}})

    assert payload == {"ok": True}
    assert len(session.request_calls) == 2


@pytest.mark.asyncio
async def test_close_closes_underlying_session() -> None:
    transport, session, _ = _transport(responses=[])
    await transport._ensure_session()

    await transport.close()

    assert session.closed is True


@pytest.mark.asyncio
async def test_context_manager_closes_session() -> None:
    transport, session, _ = _transport(responses=[])

    async with transport:
        pass

    assert session.closed is True


@pytest.mark.asyncio
async def test_query_params_are_encoded_for_nonce_sync() -> None:
    transport, session, _ = _transport(responses=[_FakeResponse(status=200, text='{"nonce": 8}')])

    await transport.get_expected_nonce({"address": "0xabc", "owner": "api-key-1"})

    assert session.request_calls[0]["url"] == "https://clob.polymarket.test/nonce?address=0xabc&owner=api-key-1"


@pytest.mark.asyncio
async def test_nonce_sync_requires_configured_path() -> None:
    transport, _, _ = _transport(responses=[], nonce_sync_path=None)

    with pytest.raises(ValueError, match="nonce_sync_path"):
        await transport.get_expected_nonce({"address": "0xabc"})


@pytest.mark.asyncio
async def test_nonce_sync_requires_nonce_field_in_response() -> None:
    transport, _, _ = _transport(responses=[_FakeResponse(status=200, text='{"ok": true}')])

    with pytest.raises(ValueError, match="nonce field"):
        await transport.get_expected_nonce({"address": "0xabc"})