from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.execution.priority_dispatcher import DispatchReceipt


CtfLegRole = Literal["YES_LEG", "NO_LEG"]
CtfLegFillStatus = Literal["FILLED", "REJECTED", "SUPPRESSED", "PARTIAL"]
CtfExecutionOutcome = Literal[
    "FULL_FILL",
    "ANCHOR_REJECTED",
    "SECOND_LEG_REJECTED",
    "PARTIAL_FILL",
    "GUARD_REJECTED",
    "BUS_REJECTED",
]


def _require_decimal(field_name: str, value: Decimal) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class CtfLegManifest:
    market_id: str
    side: Literal["YES", "NO"]
    target_price: Decimal
    target_size: Decimal
    taker_fee: Decimal
    leg_role: CtfLegRole
    leg_index: int

    def __post_init__(self) -> None:
        market_id = str(self.market_id or "").strip()
        if not market_id:
            raise ValueError("market_id must be a non-empty string")
        object.__setattr__(self, "market_id", market_id)

        if self.side not in {"YES", "NO"}:
            raise ValueError(f"side must be 'YES' or 'NO'; got {self.side!r}")
        if self.leg_role not in {"YES_LEG", "NO_LEG"}:
            raise ValueError(f"leg_role must be 'YES_LEG' or 'NO_LEG'; got {self.leg_role!r}")
        if self.leg_index not in {0, 1}:
            raise ValueError(f"leg_index must be 0 or 1; got {self.leg_index!r}")

        target_price = _require_decimal("target_price", self.target_price)
        if target_price <= Decimal("0"):
            raise ValueError("target_price must be strictly positive")

        target_size = _require_decimal("target_size", self.target_size)
        if target_size <= Decimal("0"):
            raise ValueError("target_size must be strictly positive")

        taker_fee = _require_decimal("taker_fee", self.taker_fee)
        if taker_fee < Decimal("0"):
            raise ValueError("taker_fee cannot be negative")


@dataclass(frozen=True, slots=True)
class CtfExecutionManifest:
    market_id: str
    yes_leg: CtfLegManifest
    no_leg: CtfLegManifest
    net_edge: Decimal
    gas_estimate: Decimal
    required_size: Decimal
    anchor_leg: CtfLegRole
    manifest_timestamp_ms: int
    cancel_on_stale_ms: int

    def __post_init__(self) -> None:
        market_id = str(self.market_id or "").strip()
        if not market_id:
            raise ValueError("market_id must be a non-empty string")
        object.__setattr__(self, "market_id", market_id)

        if self.yes_leg.leg_role != "YES_LEG" or self.yes_leg.side != "YES":
            raise ValueError("yes_leg must carry side='YES' and leg_role='YES_LEG'")
        if self.no_leg.leg_role != "NO_LEG" or self.no_leg.side != "NO":
            raise ValueError("no_leg must carry side='NO' and leg_role='NO_LEG'")
        if self.yes_leg.market_id != market_id or self.no_leg.market_id != market_id:
            raise ValueError("both legs must share the manifest market_id")

        required_size = _require_decimal("required_size", self.required_size)
        if required_size <= Decimal("0"):
            raise ValueError("required_size must be strictly positive")
        if self.yes_leg.target_size != required_size or self.no_leg.target_size != required_size:
            raise ValueError("both legs must share the manifest required_size")

        _require_decimal("net_edge", self.net_edge)
        gas_estimate = _require_decimal("gas_estimate", self.gas_estimate)
        if gas_estimate < Decimal("0"):
            raise ValueError("gas_estimate cannot be negative")

        expected_anchor = "YES_LEG" if self.yes_leg.target_price <= self.no_leg.target_price else "NO_LEG"
        if self.anchor_leg != expected_anchor:
            raise ValueError("anchor_leg must reference the cheaper leg; equal-price ties default to YES_LEG")

        if not isinstance(self.manifest_timestamp_ms, int):
            raise ValueError("manifest_timestamp_ms must be an int")
        if not isinstance(self.cancel_on_stale_ms, int) or self.cancel_on_stale_ms <= 0:
            raise ValueError("cancel_on_stale_ms must be a strictly positive int")


@dataclass(frozen=True, slots=True)
class CtfLegReceipt:
    leg_manifest: CtfLegManifest
    dispatch_receipt: DispatchReceipt
    fill_status: CtfLegFillStatus
    filled_size: Decimal
    filled_price: Decimal | None
    fill_timestamp_ms: int | None

    def __post_init__(self) -> None:
        filled_size = _require_decimal("filled_size", self.filled_size)
        if filled_size < Decimal("0"):
            raise ValueError("filled_size cannot be negative")
        if self.fill_status not in {"FILLED", "REJECTED", "SUPPRESSED", "PARTIAL"}:
            raise ValueError(f"Unsupported fill_status: {self.fill_status!r}")
        if self.fill_status in {"FILLED", "PARTIAL"}:
            if self.filled_price is None or filled_size <= Decimal("0"):
                raise ValueError(f"{self.fill_status} leg receipts require filled_price and positive filled_size")
            if self.fill_timestamp_ms is None:
                raise ValueError(f"{self.fill_status} leg receipts require fill_timestamp_ms")
            return
        if filled_size != Decimal("0"):
            raise ValueError("Unfilled leg receipts must set filled_size to Decimal('0')")
        if self.filled_price is not None:
            raise ValueError("Unfilled leg receipts must set filled_price to None")
        if self.fill_timestamp_ms is not None:
            raise ValueError("Unfilled leg receipts must set fill_timestamp_ms to None")


@dataclass(frozen=True, slots=True)
class CtfExecutionReceipt:
    manifest: CtfExecutionManifest
    yes_receipt: CtfLegReceipt
    no_receipt: CtfLegReceipt
    execution_outcome: CtfExecutionOutcome
    realized_net_edge: Decimal
    realized_pnl: Decimal
    total_capital_deployed: Decimal
    execution_timestamp_ms: int

    def __post_init__(self) -> None:
        realized_net_edge = _require_decimal("realized_net_edge", self.realized_net_edge)
        realized_pnl = _require_decimal("realized_pnl", self.realized_pnl)
        total_capital_deployed = _require_decimal("total_capital_deployed", self.total_capital_deployed)
        if total_capital_deployed < Decimal("0"):
            raise ValueError("total_capital_deployed cannot be negative")
        if self.execution_outcome not in {
            "FULL_FILL",
            "ANCHOR_REJECTED",
            "SECOND_LEG_REJECTED",
            "PARTIAL_FILL",
            "GUARD_REJECTED",
            "BUS_REJECTED",
        }:
            raise ValueError(f"Unsupported execution_outcome: {self.execution_outcome!r}")
        if self.execution_outcome not in {"FULL_FILL", "PARTIAL_FILL"}:
            if realized_net_edge != Decimal("0") or realized_pnl != Decimal("0"):
                raise ValueError(
                    "realized_net_edge and realized_pnl must be Decimal('0') when the execution outcome is not a fill"
                )
        if not isinstance(self.execution_timestamp_ms, int):
            raise ValueError("execution_timestamp_ms must be an int")

    @property
    def executed(self) -> bool:
        return self.execution_outcome in {"FULL_FILL", "PARTIAL_FILL"}


def build_ctf_execution_manifest(
    *,
    market_id: str,
    yes_price: Decimal,
    no_price: Decimal,
    net_edge: Decimal,
    gas_estimate: Decimal,
    default_anchor_volume: Decimal,
    max_capital_per_signal: Decimal,
    max_size_per_leg: Decimal,
    taker_fee_yes: Decimal,
    taker_fee_no: Decimal,
    manifest_timestamp_ms: int,
    cancel_on_stale_ms: int,
) -> CtfExecutionManifest:
    capital_bound = _require_decimal("max_capital_per_signal", max_capital_per_signal) / (
        _require_decimal("yes_price", yes_price) + _require_decimal("no_price", no_price)
    )
    required_size = min(
        _require_decimal("default_anchor_volume", default_anchor_volume),
        _require_decimal("max_size_per_leg", max_size_per_leg),
        capital_bound,
    )
    yes_leg = CtfLegManifest(
        market_id=market_id,
        side="YES",
        target_price=yes_price,
        target_size=required_size,
        taker_fee=taker_fee_yes,
        leg_role="YES_LEG",
        leg_index=0 if yes_price <= no_price else 1,
    )
    no_leg = CtfLegManifest(
        market_id=market_id,
        side="NO",
        target_price=no_price,
        target_size=required_size,
        taker_fee=taker_fee_no,
        leg_role="NO_LEG",
        leg_index=0 if no_price < yes_price else 1,
    )
    anchor_leg: CtfLegRole = "YES_LEG" if yes_price <= no_price else "NO_LEG"
    return CtfExecutionManifest(
        market_id=market_id,
        yes_leg=yes_leg,
        no_leg=no_leg,
        net_edge=net_edge,
        gas_estimate=gas_estimate,
        required_size=required_size,
        anchor_leg=anchor_leg,
        manifest_timestamp_ms=manifest_timestamp_ms,
        cancel_on_stale_ms=cancel_on_stale_ms,
    )