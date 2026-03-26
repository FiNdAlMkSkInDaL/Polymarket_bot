from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


PrioritySide = Literal["YES", "NO"]
PrioritySignalSource = Literal["OFI", "SI9", "SI10", "CONTAGION", "MANUAL", "CTF"]


def _require_decimal(name: str, value: object) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{name} must be a Decimal; got {type(value).__name__}")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite; got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class PriorityOrderContext:
    market_id: str
    side: PrioritySide
    signal_source: PrioritySignalSource
    conviction_scalar: Decimal
    target_price: Decimal
    anchor_volume: Decimal
    max_capital: Decimal
    leg_role: str | None = None

    def __post_init__(self) -> None:
        market_id = str(self.market_id or "").strip()
        if not market_id:
            raise ValueError("market_id must be a non-empty string")
        object.__setattr__(self, "market_id", market_id)

        if self.side not in {"YES", "NO"}:
            raise ValueError(f"side must be 'YES' or 'NO'; got {self.side!r}")
        if self.signal_source not in {"OFI", "SI9", "SI10", "CONTAGION", "MANUAL", "CTF"}:
            raise ValueError(
                "signal_source must be one of 'OFI', 'SI9', 'SI10', 'CONTAGION', 'MANUAL', or 'CTF'; "
                f"got {self.signal_source!r}"
            )
        if self.leg_role is not None and self.leg_role not in {"YES_LEG", "NO_LEG"}:
            raise ValueError(f"leg_role must be 'YES_LEG', 'NO_LEG', or None; got {self.leg_role!r}")

        conviction_scalar = _require_decimal("conviction_scalar", self.conviction_scalar)
        if conviction_scalar < Decimal("0.0") or conviction_scalar > Decimal("1.0"):
            raise ValueError(
                f"conviction_scalar must be within [0.0, 1.0]; got {conviction_scalar!r}"
            )

        for field_name in ("target_price", "anchor_volume", "max_capital"):
            value = _require_decimal(field_name, getattr(self, field_name))
            if value <= Decimal("0"):
                raise ValueError(f"{field_name} must be strictly positive; got {value!r}")