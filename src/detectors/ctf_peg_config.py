from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


def _validate_strictly_positive_decimal(field_name: str, value: Decimal) -> None:
    if value <= Decimal("0"):
        raise ValueError(f"{field_name} must be strictly positive")


def _validate_non_negative_decimal(field_name: str, value: Decimal) -> None:
    if value < Decimal("0"):
        raise ValueError(f"{field_name} must be greater than or equal to zero")


@dataclass(frozen=True, slots=True)
class CtfPegConfig:
    min_yield: Decimal
    taker_fee_yes: Decimal
    taker_fee_no: Decimal
    slippage_budget: Decimal
    gas_ewma_alpha: Decimal
    max_desync_ms: int

    def __post_init__(self) -> None:
        _validate_strictly_positive_decimal("min_yield", self.min_yield)
        _validate_strictly_positive_decimal("taker_fee_yes", self.taker_fee_yes)
        _validate_strictly_positive_decimal("taker_fee_no", self.taker_fee_no)
        _validate_non_negative_decimal("slippage_budget", self.slippage_budget)

        if self.gas_ewma_alpha <= Decimal("0") or self.gas_ewma_alpha > Decimal("1"):
            raise ValueError("gas_ewma_alpha must be > 0 and <= 1")

        if not isinstance(self.max_desync_ms, int) or self.max_desync_ms <= 0:
            raise ValueError("max_desync_ms must be a strictly positive integer")