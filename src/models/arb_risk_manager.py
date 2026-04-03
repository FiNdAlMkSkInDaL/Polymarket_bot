from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Mapping

from src.models.amm_pricing import binary_cpmm_marginal_price, binary_lmsr_marginal_price


_ZERO = Decimal("0")
_ONE = Decimal("1")

AmmModel = Literal["CPMM", "LMSR"]
Outcome = Literal["YES", "NO"]


def _as_decimal(value: Any, *, name: str) -> Decimal:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def _require_positive(value: Decimal, *, name: str) -> Decimal:
    if value <= _ZERO:
        raise ValueError(f"{name} must be strictly positive")
    return value


@dataclass(frozen=True, slots=True)
class ArbSizingResult:
    size_shares: Decimal
    size_usd: Decimal
    reference_price: Decimal
    q_capital: Decimal
    q_inventory: Decimal
    q_venue_depth_safe: Decimal
    capped_by: str


def calculate_safe_arb_size(
    *,
    clob_available_volume_at_bbo: Decimal | float | int | str,
    amm_reserves: Mapping[str, Any],
    capital_cap_usd: Decimal | float | int | str,
    max_trade_size_usd: Decimal | float | int | str = Decimal("50"),
    clob_target_price: Decimal | float | int | str | None = None,
    amm_model: AmmModel = "CPMM",
    outcome: Outcome = "YES",
    max_amm_reserve_fraction: Decimal | float | int | str = Decimal("0.05"),
) -> ArbSizingResult:
    """Compute a safe taker size for hybrid arbitrage.

    The sizing rule is the memo's conservative constraint intersection:

        q* = min(q_capital, q_inventory, q_venue_depth_safe)

    Definitions:

    - q_capital: shares affordable under the tighter of the global capital cap
      and the per-trade hard USD cap.
    - q_inventory: AMM-capacity-safe shares, derived from a haircut against the
      relevant CPMM reserve or LMSR liquidity parameter.
    - q_venue_depth_safe: the exact displayed shares resting at the CLOB BBO,
      ensuring an IOC taker order cannot walk past the target level.

    `clob_target_price` is optional; when omitted, the function uses the AMM
    marginal price as a conservative reference price for USD-to-share
    conversion.
    """

    bbo_volume = _require_positive(
        _as_decimal(clob_available_volume_at_bbo, name="clob_available_volume_at_bbo"),
        name="clob_available_volume_at_bbo",
    )
    global_cap = _require_positive(_as_decimal(capital_cap_usd, name="capital_cap_usd"), name="capital_cap_usd")
    hard_cap = _require_positive(_as_decimal(max_trade_size_usd, name="max_trade_size_usd"), name="max_trade_size_usd")
    reserve_fraction = _as_decimal(max_amm_reserve_fraction, name="max_amm_reserve_fraction")
    if reserve_fraction <= _ZERO or reserve_fraction > _ONE:
        raise ValueError("max_amm_reserve_fraction must lie in (0, 1]")

    normalized_model = str(amm_model or "").strip().upper()
    if normalized_model not in {"CPMM", "LMSR"}:
        raise ValueError("amm_model must be 'CPMM' or 'LMSR'")
    normalized_outcome = str(outcome or "").strip().upper()
    if normalized_outcome not in {"YES", "NO"}:
        raise ValueError("outcome must be 'YES' or 'NO'")

    amm_snapshot = _coerce_mapping(amm_reserves)
    amm_marginal_price = _infer_amm_marginal_price(
        amm_snapshot=amm_snapshot,
        amm_model=normalized_model,
        outcome=normalized_outcome,
    )
    if clob_target_price is None:
        reference_price = amm_marginal_price
    else:
        reference_price = _require_positive(_as_decimal(clob_target_price, name="clob_target_price"), name="clob_target_price")

    q_capital = min(global_cap, hard_cap) / reference_price
    q_inventory = _compute_amm_inventory_capacity(
        amm_snapshot=amm_snapshot,
        amm_model=normalized_model,
        outcome=normalized_outcome,
        reserve_fraction=reserve_fraction,
    )
    q_venue_depth_safe = bbo_volume

    size_shares = min(q_capital, q_inventory, q_venue_depth_safe)
    size_usd = size_shares * reference_price

    if size_shares == q_capital:
        capped_by = "capital"
    elif size_shares == q_inventory:
        capped_by = "inventory"
    else:
        capped_by = "venue_depth_safe"

    return ArbSizingResult(
        size_shares=size_shares,
        size_usd=size_usd,
        reference_price=reference_price,
        q_capital=q_capital,
        q_inventory=q_inventory,
        q_venue_depth_safe=q_venue_depth_safe,
        capped_by=capped_by,
    )


def _infer_amm_marginal_price(
    *,
    amm_snapshot: Mapping[str, Decimal],
    amm_model: AmmModel,
    outcome: Outcome,
) -> Decimal:
    if amm_model == "CPMM":
        return binary_cpmm_marginal_price(
            yes_reserve=amm_snapshot["yes_reserve"],
            no_reserve=amm_snapshot["no_reserve"],
            outcome=outcome,
        )
    return binary_lmsr_marginal_price(
        yes_inventory=amm_snapshot["yes_inventory"],
        no_inventory=amm_snapshot["no_inventory"],
        liquidity=amm_snapshot["liquidity"],
        outcome=outcome,
    )


def _compute_amm_inventory_capacity(
    *,
    amm_snapshot: Mapping[str, Decimal],
    amm_model: AmmModel,
    outcome: Outcome,
    reserve_fraction: Decimal,
) -> Decimal:
    if amm_model == "CPMM":
        reserve_key = "yes_reserve" if outcome == "YES" else "no_reserve"
        return amm_snapshot[reserve_key] * reserve_fraction
    return amm_snapshot["liquidity"] * reserve_fraction


def _coerce_mapping(value: Mapping[str, Any]) -> dict[str, Decimal]:
    if not isinstance(value, Mapping):
        raise ValueError("amm_reserves must be a mapping")
    return {
        str(key): _as_decimal(raw_value, name=str(key))
        for key, raw_value in value.items()
    }