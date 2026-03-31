from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


def _require_non_empty_string(name: str, value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _require_non_negative_decimal(name: str, value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if value < Decimal("0"):
        raise ValueError(f"{name} must be greater than or equal to 0")
    return value


class VenueAdapter(ABC):
    @abstractmethod
    def submit_order(
        self,
        market_id: str,
        side: Literal["YES", "NO"],
        price: Decimal,
        size: Decimal,
        order_type: Literal["LIMIT", "MARKET"],
        client_order_id: str,
        *,
        time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC",
        post_only: bool = False,
    ) -> "VenueOrderResponse":
        ...

    @abstractmethod
    def cancel_order(
        self,
        client_order_id: str,
        market_id: str,
    ) -> "VenueCancelResponse":
        ...

    @abstractmethod
    def get_order_status(
        self,
        client_order_id: str,
    ) -> "VenueOrderStatus":
        ...

    @abstractmethod
    def get_wallet_balance(
        self,
        asset_symbol: str,
    ) -> Decimal:
        ...


@dataclass(frozen=True, slots=True)
class VenueOrderResponse:
    client_order_id: str
    venue_order_id: str | None
    status: Literal["ACCEPTED", "REJECTED", "PENDING"]
    rejection_reason: str | None
    venue_timestamp_ms: int | None
    latency_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", _require_non_empty_string("client_order_id", self.client_order_id))
        if self.venue_order_id is not None:
            object.__setattr__(self, "venue_order_id", _require_non_empty_string("venue_order_id", self.venue_order_id))
        if self.status not in {"ACCEPTED", "REJECTED", "PENDING"}:
            raise ValueError(f"Unsupported venue order status: {self.status!r}")
        if self.status == "REJECTED":
            if not self.rejection_reason:
                raise ValueError("rejection_reason is required when status='REJECTED'")
        elif self.rejection_reason is not None:
            raise ValueError("rejection_reason must be unset unless status='REJECTED'")
        if self.venue_timestamp_ms is not None and not isinstance(self.venue_timestamp_ms, int):
            raise ValueError("venue_timestamp_ms must be an int or None")
        if not isinstance(self.latency_ms, int) or self.latency_ms < 0:
            raise ValueError("latency_ms must be a non-negative int")


@dataclass(frozen=True, slots=True)
class VenueCancelResponse:
    client_order_id: str
    cancelled: bool
    rejection_reason: str | None
    venue_timestamp_ms: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", _require_non_empty_string("client_order_id", self.client_order_id))
        if self.cancelled:
            if self.rejection_reason is not None:
                raise ValueError("rejection_reason must be unset when cancelled is True")
        elif not self.rejection_reason:
            raise ValueError("rejection_reason is required when cancelled is False")
        if self.venue_timestamp_ms is not None and not isinstance(self.venue_timestamp_ms, int):
            raise ValueError("venue_timestamp_ms must be an int or None")


@dataclass(frozen=True, slots=True)
class VenueOrderStatus:
    client_order_id: str
    venue_order_id: str | None
    fill_status: Literal["OPEN", "PARTIAL", "FILLED", "CANCELLED", "UNKNOWN"]
    filled_size: Decimal
    remaining_size: Decimal
    average_fill_price: Decimal | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", _require_non_empty_string("client_order_id", self.client_order_id))
        if self.venue_order_id is not None:
            object.__setattr__(self, "venue_order_id", _require_non_empty_string("venue_order_id", self.venue_order_id))
        if self.fill_status not in {"OPEN", "PARTIAL", "FILLED", "CANCELLED", "UNKNOWN"}:
            raise ValueError(f"Unsupported fill_status: {self.fill_status!r}")

        filled_size = _require_non_negative_decimal("filled_size", self.filled_size)
        remaining_size = _require_non_negative_decimal("remaining_size", self.remaining_size)
        if self.average_fill_price is not None:
            if not isinstance(self.average_fill_price, Decimal) or not self.average_fill_price.is_finite():
                raise ValueError("average_fill_price must be a finite Decimal or None")
            if self.average_fill_price <= Decimal("0"):
                raise ValueError("average_fill_price must be strictly positive when set")

        if self.fill_status in {"PARTIAL", "FILLED"} and filled_size == Decimal("0"):
            raise ValueError(f"{self.fill_status} status requires filled_size greater than 0")
        if filled_size > Decimal("0") and self.average_fill_price is None:
            raise ValueError("average_fill_price is required when filled_size is greater than 0")
        if filled_size == Decimal("0") and self.average_fill_price is not None:
            raise ValueError("average_fill_price must be unset when filled_size is 0")
        if self.fill_status == "FILLED" and remaining_size != Decimal("0"):
            raise ValueError("FILLED status requires remaining_size == 0")