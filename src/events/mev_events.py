"""Strict event contracts for isolated MEV execution flows."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


def _normalize_direction(direction: str) -> str:
    value = str(direction or "").strip().upper()
    if value not in {"YES", "NO"}:
        raise ValueError(f"Unsupported MEV direction: {direction!r}")
    return value


def _require_positive(name: str, value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return numeric


def _require_non_negative(name: str, value: float) -> float:
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return numeric


@dataclass(frozen=True, slots=True)
class ShadowSweepSignal:
    target_market_id: str
    direction: str
    max_capital: float
    premium_pct: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _normalize_direction(self.direction))
        object.__setattr__(self, "max_capital", _require_positive("max_capital", self.max_capital))
        object.__setattr__(self, "premium_pct", _require_non_negative("premium_pct", self.premium_pct))


@dataclass(frozen=True, slots=True)
class MMPredationSignal:
    target_market_id: str
    correlated_market_id: str
    v_attack: float
    trap_direction: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trap_direction", _normalize_direction(self.trap_direction))
        object.__setattr__(self, "v_attack", _require_positive("v_attack", self.v_attack))


@dataclass(frozen=True, slots=True)
class DisputeArbitrageSignal:
    market_id: str
    panic_direction: str
    limit_price: float
    max_capital: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "panic_direction", _normalize_direction(self.panic_direction))
        object.__setattr__(self, "limit_price", _require_positive("limit_price", self.limit_price))
        object.__setattr__(self, "max_capital", _require_positive("max_capital", self.max_capital))


@dataclass(frozen=True, slots=True)
class CtfMergeSignal:
    market_id: str
    yes_ask: Decimal
    no_ask: Decimal
    gas_estimate: Decimal
    net_edge: Decimal

    def __post_init__(self) -> None:
        if self.yes_ask <= Decimal("0"):
            raise ValueError("yes_ask must be strictly positive")
        if self.no_ask <= Decimal("0"):
            raise ValueError("no_ask must be strictly positive")
        if self.gas_estimate < Decimal("0"):
            raise ValueError("gas_estimate cannot be negative")