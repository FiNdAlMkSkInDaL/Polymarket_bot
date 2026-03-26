from __future__ import annotations

from decimal import Decimal
from typing import Literal

from src.execution.priority_context import PriorityOrderContext


def ofi_to_context(
    market_id: str,
    side: Literal["YES", "NO"],
    target_price: Decimal,
    anchor_volume: Decimal,
    max_capital: Decimal,
    conviction_scalar: Decimal,
) -> PriorityOrderContext:
    return PriorityOrderContext(
        market_id=market_id,
        side=side,
        signal_source="OFI",
        conviction_scalar=conviction_scalar,
        target_price=target_price,
        anchor_volume=anchor_volume,
        max_capital=max_capital,
    )


def si9_to_context(
    market_id: str,
    side: Literal["YES"],
    target_price: Decimal,
    anchor_volume: Decimal,
    max_capital: Decimal,
    conviction_scalar: Decimal,
) -> PriorityOrderContext:
    if side != "YES":
        raise ValueError(f"SI-9 adapter requires side='YES'; got {side!r}")
    return PriorityOrderContext(
        market_id=market_id,
        side=side,
        signal_source="SI9",
        conviction_scalar=conviction_scalar,
        target_price=target_price,
        anchor_volume=anchor_volume,
        max_capital=max_capital,
    )


def ctf_to_context(
    market_id: str,
    side: Literal["YES", "NO"],
    target_price: Decimal,
    anchor_volume: Decimal,
    max_capital: Decimal,
    conviction_scalar: Decimal,
    leg_role: Literal["YES_LEG", "NO_LEG"] | None = None,
) -> PriorityOrderContext:
    return PriorityOrderContext(
        market_id=market_id,
        side=side,
        signal_source="CTF",
        conviction_scalar=conviction_scalar,
        target_price=target_price,
        anchor_volume=anchor_volume,
        max_capital=max_capital,
        leg_role=leg_role,
    )