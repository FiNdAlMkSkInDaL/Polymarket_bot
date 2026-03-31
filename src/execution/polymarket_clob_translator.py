from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Mapping

from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.venue_adapter_interface import VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def _require_non_empty_string(name: str, value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _require_condition_id(value: str) -> str:
    normalized = _require_non_empty_string("condition_id", value)
    if not normalized.startswith("0x"):
        raise ValueError("condition_id must start with '0x'")
    hex_part = normalized[2:]
    if not hex_part or any(ch not in "0123456789abcdefABCDEF" for ch in hex_part):
        raise ValueError("condition_id must be a hex string")
    return normalized.lower()


def _require_mapping(name: str, value: Mapping[str, Any] | dict[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
            return int(normalized)
    raise ValueError(f"{name} must be an int")


def _require_decimal(name: str, value: Any) -> Decimal:
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, str):
        decimal_value = Decimal(value.strip())
    else:
        raise ValueError(f"{name} must be provided as Decimal or decimal string")
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def _decimal_to_str(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("Decimal value must be finite")
    return format(value, "f")


class VenueRejectionReason(str, Enum):
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    STALE_NONCE = "STALE_NONCE"
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    WOULD_CROSS = "WOULD_CROSS"
    PRICE_OUT_OF_BOUNDS = "PRICE_OUT_OF_BOUNDS"
    UNKNOWN = "UNKNOWN"


class ClobTimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class ClobApiOrderType(str, Enum):
    GTC = "GTC"
    FAK = "FAK"
    FOK = "FOK"


@dataclass(frozen=True, slots=True)
class ClobOrderIntent:
    condition_id: str
    token_id: str
    outcome: Literal["YES", "NO"]
    action: Literal["BUY", "SELL"]
    price: Decimal
    size: Decimal
    time_in_force: ClobTimeInForce
    client_order_id: str
    post_only: bool = False
    fee_rate_bps: int = 0
    nonce: int = 0
    expiration: int = 0
    taker: str = _ZERO_ADDRESS

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", _require_condition_id(self.condition_id))
        object.__setattr__(self, "token_id", _require_non_empty_string("token_id", self.token_id))
        object.__setattr__(self, "client_order_id", _require_non_empty_string("client_order_id", self.client_order_id))
        object.__setattr__(self, "taker", _require_non_empty_string("taker", self.taker))
        if self.outcome not in {"YES", "NO"}:
            raise ValueError(f"Unsupported outcome: {self.outcome!r}")
        if self.action not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported action: {self.action!r}")
        if self.price <= Decimal("0"):
            raise ValueError("price must be strictly positive")
        if self.size <= Decimal("0"):
            raise ValueError("size must be strictly positive")
        if not isinstance(self.fee_rate_bps, int) or self.fee_rate_bps < 0:
            raise ValueError("fee_rate_bps must be a non-negative int")
        if not isinstance(self.nonce, int) or self.nonce < 0:
            raise ValueError("nonce must be a non-negative int")
        if not isinstance(self.expiration, int) or self.expiration < 0:
            raise ValueError("expiration must be a non-negative int")
        if self.post_only and self.time_in_force != ClobTimeInForce.GTC:
            raise ValueError("post_only orders must use GTC")


class ClobPayloadBuilder:
    def build_token_lookup_payload(
        self,
        *,
        condition_id: str,
        outcome: Literal["YES", "NO"],
    ) -> dict[str, str]:
        if outcome not in {"YES", "NO"}:
            raise ValueError(f"Unsupported outcome: {outcome!r}")
        return {
            "conditionId": _require_condition_id(condition_id),
            "outcome": outcome,
        }

    def build_create_order_payload(self, intent: ClobOrderIntent) -> dict[str, str | bool]:
        return {
            "conditionId": intent.condition_id,
            "token_id": intent.token_id,
            "client_order_id": intent.client_order_id,
            "price": _decimal_to_str(intent.price),
            "size": _decimal_to_str(intent.size),
            "side": intent.action,
            "fee_rate_bps": str(intent.fee_rate_bps),
            "nonce": str(intent.nonce),
            "expiration": str(intent.expiration),
            "taker": intent.taker,
            "post_only": intent.post_only,
        }

    def build_post_order_payload(
        self,
        *,
        signed_order: Mapping[str, Any],
        owner_id: str,
        time_in_force: ClobTimeInForce,
        post_only: bool,
    ) -> dict[str, Any]:
        if post_only and time_in_force != ClobTimeInForce.GTC:
            raise ValueError("post_only orders must use GTC")
        return {
            "order": dict(_require_mapping("signed_order", signed_order)),
            "owner": _require_non_empty_string("owner_id", owner_id),
            "orderType": self.translate_time_in_force(time_in_force).value,
            "postOnly": bool(post_only),
        }

    def build_cancel_payload(self, *, client_order_id: str, condition_id: str) -> dict[str, str]:
        return {
            "clientOrderId": _require_non_empty_string("client_order_id", client_order_id),
            "conditionId": _require_condition_id(condition_id),
        }

    def build_order_status_payload(self, *, client_order_id: str) -> dict[str, str]:
        return {"clientOrderId": _require_non_empty_string("client_order_id", client_order_id)}

    @staticmethod
    def translate_time_in_force(time_in_force: ClobTimeInForce) -> ClobApiOrderType:
        if time_in_force == ClobTimeInForce.GTC:
            return ClobApiOrderType.GTC
        if time_in_force == ClobTimeInForce.IOC:
            return ClobApiOrderType.FAK
        if time_in_force == ClobTimeInForce.FOK:
            return ClobApiOrderType.FOK
        raise ValueError(f"Unsupported time_in_force: {time_in_force!r}")


class ClobReceiptParser:
    def parse_submit_response(
        self,
        raw_response: Mapping[str, Any],
        *,
        expected_client_order_id: str,
    ) -> VenueOrderResponse:
        raw = _require_mapping("raw_response", raw_response)
        status_token = self._normalized_status(raw)
        if status_token in {"REJECTED", "ERROR", "FAILED"}:
            return VenueOrderResponse(
                client_order_id=self._client_order_id(raw, expected_client_order_id),
                venue_order_id=self._venue_order_id(raw),
                status="REJECTED",
                rejection_reason=self._rejection_reason(raw),
                venue_timestamp_ms=self._venue_timestamp_ms(raw),
                latency_ms=self._latency_ms(raw),
            )
        if status_token in {"PENDING", "LIVE", "OPEN", "PROCESSING"}:
            mapped_status: Literal["ACCEPTED", "REJECTED", "PENDING"] = "PENDING"
        else:
            mapped_status = "ACCEPTED"
        return VenueOrderResponse(
            client_order_id=self._client_order_id(raw, expected_client_order_id),
            venue_order_id=self._venue_order_id(raw),
            status=mapped_status,
            rejection_reason=None,
            venue_timestamp_ms=self._venue_timestamp_ms(raw),
            latency_ms=self._latency_ms(raw),
        )

    def parse_cancel_response(
        self,
        raw_response: Mapping[str, Any],
        *,
        expected_client_order_id: str,
    ) -> VenueCancelResponse:
        raw = _require_mapping("raw_response", raw_response)
        status_token = self._normalized_status(raw)
        cancelled = status_token in {"CANCELLED", "CANCELED", "OK", "SUCCESS"}
        return VenueCancelResponse(
            client_order_id=self._client_order_id(raw, expected_client_order_id),
            cancelled=cancelled,
            rejection_reason=None if cancelled else self._rejection_reason(raw),
            venue_timestamp_ms=self._venue_timestamp_ms(raw),
        )

    def parse_order_status(
        self,
        raw_response: Mapping[str, Any],
        *,
        expected_client_order_id: str,
    ) -> VenueOrderStatus:
        raw = _require_mapping("raw_response", raw_response)
        fill_status = self._map_fill_status(raw)
        filled_size = self._decimal_from_raw(raw, "filled_size", "filledSize", default=Decimal("0"))
        remaining_size = self._decimal_from_raw(raw, "remaining_size", "remainingSize", default=Decimal("0"))
        average_fill_price = self._optional_decimal_from_raw(raw, "average_fill_price", "averagePrice")
        return VenueOrderStatus(
            client_order_id=self._client_order_id(raw, expected_client_order_id),
            venue_order_id=self._venue_order_id(raw),
            fill_status=fill_status,
            filled_size=filled_size,
            remaining_size=remaining_size,
            average_fill_price=average_fill_price,
        )

    def parse_dispatch_receipt(
        self,
        *,
        context: PriorityOrderContext,
        serialized_envelope: str,
        dispatch_timestamp_ms: int,
        effective_size: Decimal,
        submit_response_raw: Mapping[str, Any],
        order_status_raw: Mapping[str, Any] | None = None,
        expected_client_order_id: str,
    ) -> DispatchReceipt:
        submit_response = self.parse_submit_response(
            submit_response_raw,
            expected_client_order_id=expected_client_order_id,
        )
        if submit_response.status == "REJECTED":
            remaining_size = effective_size
            if order_status_raw is not None:
                remaining_size = self.parse_order_status(
                    order_status_raw,
                    expected_client_order_id=submit_response.client_order_id,
                ).remaining_size
            # Rejections still carry the client order id for deterministic tracing.
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
                remaining_size=remaining_size,
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        if order_status_raw is None:
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
                remaining_size=effective_size,
                venue_timestamp_ms=submit_response.venue_timestamp_ms,
                latency_ms=submit_response.latency_ms,
            )

        order_status = self.parse_order_status(
            order_status_raw,
            expected_client_order_id=submit_response.client_order_id,
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

    @staticmethod
    def _client_order_id(raw: Mapping[str, Any], expected_client_order_id: str) -> str:
        return _require_non_empty_string(
            "client_order_id",
            str(raw.get("clientOrderId") or raw.get("client_order_id") or expected_client_order_id),
        )

    @staticmethod
    def _venue_order_id(raw: Mapping[str, Any]) -> str | None:
        candidate = raw.get("orderID") or raw.get("orderId") or raw.get("id")
        if candidate is None:
            return None
        return _require_non_empty_string("venue_order_id", str(candidate))

    def _venue_timestamp_ms(self, raw: Mapping[str, Any]) -> int | None:
        for key in ("timestampMs", "timestamp_ms", "createdAtMs", "created_at_ms"):
            value = raw.get(key)
            if value is not None:
                return _require_int("venue_timestamp_ms", value)
        return None

    def _latency_ms(self, raw: Mapping[str, Any]) -> int:
        value = raw.get("latencyMs")
        if value is None:
            value = raw.get("latency_ms", 0)
        latency_ms = _require_int("latency_ms", value)
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        return latency_ms

    def _map_fill_status(self, raw: Mapping[str, Any]) -> Literal["OPEN", "PARTIAL", "FILLED", "CANCELLED", "UNKNOWN"]:
        status_token = self._normalized_status(raw)
        if status_token in {"OPEN", "LIVE", "PENDING"}:
            return "OPEN"
        if status_token in {"PARTIAL", "PARTIALLY_FILLED"}:
            return "PARTIAL"
        if status_token in {"FILLED", "MATCHED"}:
            return "FILLED"
        if status_token in {"CANCELLED", "CANCELED", "EXPIRED"}:
            return "CANCELLED"
        return "UNKNOWN"

    @staticmethod
    def _normalized_status(raw: Mapping[str, Any]) -> str:
        value = raw.get("status") or raw.get("state") or raw.get("order_status") or ""
        return str(value).strip().upper()

    def _rejection_reason(self, raw: Mapping[str, Any]) -> VenueRejectionReason:
        payload = " ".join(
            str(raw.get(key) or "")
            for key in ("reason", "error", "message", "status")
        ).upper()
        if "BALANCE" in payload or "ALLOWANCE" in payload:
            return VenueRejectionReason.INSUFFICIENT_BALANCE
        if "NONCE" in payload:
            return VenueRejectionReason.STALE_NONCE
        if "NOT FOUND" in payload or "ORDER_NOT_FOUND" in payload:
            return VenueRejectionReason.ORDER_NOT_FOUND
        if "WOULD_CROSS" in payload or "WOULD CROSS" in payload:
            return VenueRejectionReason.WOULD_CROSS
        if "PRICE_OUT_OF_BOUNDS" in payload or "PRICE OUT OF BOUNDS" in payload:
            return VenueRejectionReason.PRICE_OUT_OF_BOUNDS
        return VenueRejectionReason.UNKNOWN

    def _decimal_from_raw(self, raw: Mapping[str, Any], *keys: str, default: Decimal) -> Decimal:
        for key in keys:
            if key in raw and raw.get(key) is not None:
                return _require_decimal(key, raw[key])
        return default

    def _optional_decimal_from_raw(self, raw: Mapping[str, Any], *keys: str) -> Decimal | None:
        for key in keys:
            if key in raw and raw.get(key) is not None:
                return _require_decimal(key, raw[key])
        return None
