from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


_ZERO = Decimal("0")
_ONE = Decimal("1")
_HALF = Decimal("0.5")
_MIN_PRICE = Decimal("0.001")
_MAX_PRICE = Decimal("0.999")
_DEFAULT_EXIT_URGENCY = Decimal("0.85")


def _as_decimal(value: Any, *, name: str) -> Decimal:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def _clamp(value: Decimal, lower: Decimal, upper: Decimal) -> Decimal:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _sign(value: Decimal) -> Decimal:
    if value > _ZERO:
        return _ONE
    if value < _ZERO:
        return Decimal("-1")
    return _ZERO


@dataclass(frozen=True, slots=True)
class InventorySkewInputs:
    current_inventory_usd: Decimal
    max_inventory_usd: Decimal
    base_spread: Decimal
    mid_price: Decimal
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    price_floor: Decimal = _MIN_PRICE
    price_ceiling: Decimal = _MAX_PRICE
    exit_urgency_threshold: Decimal = _DEFAULT_EXIT_URGENCY


@dataclass(frozen=True, slots=True)
class InventorySkewQuote:
    inventory_ratio: Decimal
    urgency: Decimal
    center_shift: Decimal
    adjusted_half_spread: Decimal
    bid_price: Decimal
    ask_price: Decimal
    aggressive_exit: bool
    aggressive_side: str | None


def compute_inventory_skew(
    current_inventory_usd: Decimal | float | int | str,
    max_inventory_usd: Decimal | float | int | str,
    base_spread: Decimal | float | int | str,
) -> dict[str, Decimal]:
    current_inventory = _as_decimal(current_inventory_usd, name="current_inventory_usd")
    max_inventory = _as_decimal(max_inventory_usd, name="max_inventory_usd")
    spread = _as_decimal(base_spread, name="base_spread")
    if max_inventory <= _ZERO:
        raise ValueError("max_inventory_usd must be strictly positive")
    if spread <= _ZERO:
        raise ValueError("base_spread must be strictly positive")

    inventory_ratio = _clamp(current_inventory / max_inventory, Decimal("-1"), _ONE)
    risk_ratio = abs(inventory_ratio)

    # Cubic pressure keeps the curve quiet near flat and steep near the cap.
    urgency = risk_ratio ** 3

    # The shift is directed opposite the inventory sign.
    # At the cap this reaches 1.25x the base spread, enough to cross a
    # symmetric market and prioritize flattening over earning maker edge.
    shift_magnitude = spread * (urgency + (Decimal("0.25") * (urgency ** 2)))
    center_shift = -_sign(inventory_ratio) * shift_magnitude

    # Keep some quoted width, but widen it slightly under stress to avoid
    # reloading risk on the wrong side while the center is being skewed.
    adjusted_half_spread = (spread * _HALF) * (Decimal("1") + (Decimal("0.25") * urgency))

    return {
        "inventory_ratio": inventory_ratio,
        "urgency": urgency,
        "center_shift": center_shift,
        "adjusted_half_spread": adjusted_half_spread,
    }


def compute_inventory_skew_quotes(inputs: InventorySkewInputs) -> InventorySkewQuote:
    current_inventory = _as_decimal(inputs.current_inventory_usd, name="current_inventory_usd")
    max_inventory = _as_decimal(inputs.max_inventory_usd, name="max_inventory_usd")
    spread = _as_decimal(inputs.base_spread, name="base_spread")
    mid_price = _as_decimal(inputs.mid_price, name="mid_price")
    price_floor = _as_decimal(inputs.price_floor, name="price_floor")
    price_ceiling = _as_decimal(inputs.price_ceiling, name="price_ceiling")
    exit_urgency_threshold = _as_decimal(inputs.exit_urgency_threshold, name="exit_urgency_threshold")
    if not (price_floor < price_ceiling):
        raise ValueError("price_floor must be strictly less than price_ceiling")

    skew = compute_inventory_skew(
        current_inventory_usd=current_inventory,
        max_inventory_usd=max_inventory,
        base_spread=spread,
    )
    inventory_ratio = skew["inventory_ratio"]
    urgency = skew["urgency"]
    center_shift = skew["center_shift"]
    adjusted_half_spread = skew["adjusted_half_spread"]

    bid_price = _clamp(mid_price - adjusted_half_spread + center_shift, price_floor, price_ceiling)
    ask_price = _clamp(mid_price + adjusted_half_spread + center_shift, price_floor, price_ceiling)

    aggressive_exit = False
    aggressive_side: str | None = None
    best_bid = None if inputs.best_bid is None else _as_decimal(inputs.best_bid, name="best_bid")
    best_ask = None if inputs.best_ask is None else _as_decimal(inputs.best_ask, name="best_ask")

    if urgency >= exit_urgency_threshold and inventory_ratio > _ZERO:
        aggressive_exit = True
        aggressive_side = "SELL"
        if best_bid is not None:
            ask_price = min(ask_price, best_bid)
    elif urgency >= exit_urgency_threshold and inventory_ratio < _ZERO:
        aggressive_exit = True
        aggressive_side = "BUY"
        if best_ask is not None:
            bid_price = max(bid_price, best_ask)

    bid_price = _clamp(bid_price, price_floor, price_ceiling)
    ask_price = _clamp(ask_price, price_floor, price_ceiling)
    if bid_price > ask_price:
        pivot = _clamp((bid_price + ask_price) * _HALF, price_floor, price_ceiling)
        bid_price = pivot
        ask_price = pivot

    return InventorySkewQuote(
        inventory_ratio=inventory_ratio,
        urgency=urgency,
        center_shift=center_shift,
        adjusted_half_spread=adjusted_half_spread,
        bid_price=bid_price,
        ask_price=ask_price,
        aggressive_exit=aggressive_exit,
        aggressive_side=aggressive_side,
    )