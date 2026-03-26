from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

import aiohttp

from src.core.config import settings


class ClobTransportError(Exception):
    pass


class ClobTransportTimeoutError(ClobTransportError):
    pass


class ClobTransportRateLimitError(ClobTransportError):
    def __init__(self, message: str, *, backoff_until_ms: int) -> None:
        super().__init__(message)
        self.backoff_until_ms = backoff_until_ms


class ClobTransportCircuitOpenError(ClobTransportError):
    def __init__(self, *, backoff_until_ms: int) -> None:
        super().__init__("rate limit circuit is open")
        self.backoff_until_ms = backoff_until_ms


class ClobTransportHttpError(ClobTransportError):
    def __init__(
        self,
        *,
        status_code: int,
        payload: Mapping[str, Any] | str | None,
        backoff_until_ms: int | None = None,
    ) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.payload = payload
        self.backoff_until_ms = backoff_until_ms


class AiohttpClobTransport:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_ms: int | None = None,
        rate_limit_backoff_ms: int = 30_000,
        now_ms: Callable[[], int],
        order_path: str = "/order",
        cancel_path: str = "/order",
        get_order_path_prefix: str = "/data/order/",
        balance_path: str = "/balance-allowance",
        nonce_sync_path: str | None = None,
        balance_signature_type: int | None = None,
        session_factory: Callable[..., aiohttp.ClientSession] | None = None,
    ) -> None:
        normalized_base_url = str(base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must be a non-empty string")
        effective_timeout_ms = settings.strategy.latency_block_ms if timeout_ms is None else int(timeout_ms)
        if effective_timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive int")
        if not isinstance(rate_limit_backoff_ms, int) or rate_limit_backoff_ms <= 0:
            raise ValueError("rate_limit_backoff_ms must be a positive int")

        self._base_url = normalized_base_url
        self._timeout_ms = min(effective_timeout_ms, int(settings.strategy.latency_block_ms))
        self._rate_limit_backoff_ms = rate_limit_backoff_ms
        self._now_ms = now_ms
        self._order_path = order_path
        self._cancel_path = cancel_path
        self._get_order_path_prefix = get_order_path_prefix
        self._balance_path = balance_path
        self._nonce_sync_path = nonce_sync_path
        self._balance_signature_type = balance_signature_type
        self._session_factory = session_factory or aiohttp.ClientSession
        self._session: aiohttp.ClientSession | None = None
        self._rate_limit_open_until_ms = 0

    @property
    def rate_limit_open_until_ms(self) -> int:
        return self._rate_limit_open_until_ms

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "AiohttpClobTransport":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def post_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | str:
        return await self._request_json("POST", self._order_path, json_body=dict(payload))

    async def cancel_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | str:
        return await self._request_json("DELETE", self._cancel_path, json_body=dict(payload))

    async def get_order(self, order_id: str) -> Mapping[str, Any] | str:
        normalized_order_id = str(order_id or "").strip()
        if not normalized_order_id:
            raise ValueError("order_id must be a non-empty string")
        return await self._request_json("GET", f"{self._get_order_path_prefix}{normalized_order_id}")

    async def get_expected_nonce(self, payload: Mapping[str, Any]) -> int:
        if not self._nonce_sync_path:
            raise ValueError("nonce_sync_path is not configured")
        response = await self._request_json(
            "GET",
            self._nonce_sync_path,
            query_params={str(key): str(value) for key, value in payload.items() if value is not None},
        )
        if not isinstance(response, Mapping):
            raise ValueError("expected nonce response must be a mapping")
        for key in ("expectedNonce", "expected_nonce", "nextNonce", "next_nonce", "nonce"):
            value = response.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        raise ValueError("expected nonce response must include an integer nonce field")

    async def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        query_params = {"asset_type": self._asset_type_for_symbol(asset_symbol)}
        if self._balance_signature_type is not None:
            query_params["signature_type"] = str(self._balance_signature_type)
        response = await self._request_json("GET", self._balance_path, query_params=query_params)
        return self._extract_balance_value(response)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout_ms / 1000)
            self._session = self._session_factory(timeout=timeout)
        return self._session

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        query_params: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any] | str:
        self._assert_rate_limit_allows_request()
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"
        if query_params:
            url = f"{url}?{urlencode(query_params)}"
        try:
            async with session.request(method, url, json=json_body) as response:
                text = await response.text()
                payload = self._decode_response_payload(text)
                if response.status == 429:
                    backoff_until_ms = self._trip_rate_limit_circuit()
                    raise ClobTransportRateLimitError(
                        "transport rate limited by venue",
                        backoff_until_ms=backoff_until_ms,
                    )
                if response.status != 200:
                    raise ClobTransportHttpError(
                        status_code=response.status,
                        payload=payload,
                    )
                return payload
        except TimeoutError as exc:
            raise ClobTransportTimeoutError("transport request timed out") from exc
        except aiohttp.ClientError as exc:
            raise ClobTransportError(f"transport request failed: {exc}") from exc

    def _assert_rate_limit_allows_request(self) -> None:
        now_ms = int(self._now_ms())
        if now_ms < self._rate_limit_open_until_ms:
            raise ClobTransportCircuitOpenError(backoff_until_ms=self._rate_limit_open_until_ms)

    def _trip_rate_limit_circuit(self) -> int:
        backoff_until_ms = int(self._now_ms()) + self._rate_limit_backoff_ms
        self._rate_limit_open_until_ms = backoff_until_ms
        return backoff_until_ms

    @staticmethod
    def _asset_type_for_symbol(asset_symbol: str) -> str:
        normalized = str(asset_symbol or "").strip().upper()
        if normalized in {"USDC", "USD", "COLLATERAL"}:
            return "COLLATERAL"
        if normalized == "CONDITIONAL":
            return "CONDITIONAL"
        raise ValueError(f"Unsupported asset_symbol: {asset_symbol!r}")

    @staticmethod
    def _extract_balance_value(payload: Mapping[str, Any] | str) -> Decimal:
        if not isinstance(payload, Mapping):
            raise ValueError("balance response must be a mapping")
        for key in ("availableBalance", "available_balance", "balance", "available", "amount"):
            value = payload.get(key)
            if value is not None:
                return AiohttpClobTransport._coerce_decimal_balance(value)
        details = payload.get("details")
        if isinstance(details, Mapping):
            return AiohttpClobTransport._extract_balance_value(details)
        raise ValueError("balance response must include a Decimal-compatible balance field")

    @staticmethod
    def _coerce_decimal_balance(value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("balance response contained an empty balance string")
            return Decimal(normalized)
        raise ValueError("balance response must not contain float balance values")

    @staticmethod
    def _decode_response_payload(text: str) -> Mapping[str, Any] | str:
        stripped = text.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped, parse_float=Decimal)
        except json.JSONDecodeError:
            return text