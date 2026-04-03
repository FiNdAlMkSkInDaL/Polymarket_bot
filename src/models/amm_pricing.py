from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Any, Literal


getcontext().prec = 42


_ZERO = Decimal("0")
_ONE = Decimal("1")
_ONE_HUNDRED = Decimal("100")

DEFAULT_GAS_AND_FEE_BUFFER_CENTS = Decimal("1.5")

Outcome = Literal["YES", "NO"]
TradeSide = Literal["BUY", "SELL"]


def _as_decimal(value: Any, *, name: str) -> Decimal:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def _require_positive(value: Decimal, *, name: str) -> Decimal:
    if value <= _ZERO:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _require_probability(value: Decimal, *, name: str) -> Decimal:
    if value <= _ZERO or value >= _ONE:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return value


@dataclass(frozen=True, slots=True)
class BinaryCpmPool:
    yes_reserve: Decimal
    no_reserve: Decimal


@dataclass(frozen=True, slots=True)
class BinaryLmsrState:
    yes_inventory: Decimal
    no_inventory: Decimal
    liquidity: Decimal


@dataclass(frozen=True, slots=True)
class AmmExecutionQuote:
    model: str
    outcome: Outcome
    side: TradeSide
    shares: Decimal
    collateral_amount: Decimal
    average_price: Decimal
    marginal_price_before: Decimal
    marginal_price_after: Decimal


@dataclass(frozen=True, slots=True)
class ArbitrageSpread:
    direction: Literal["CLOB_TO_AMM", "AMM_TO_CLOB"]
    order_size_shares: Decimal
    gross_spread_cents: Decimal
    net_spread_cents: Decimal
    gross_total_cents: Decimal
    net_total_cents: Decimal
    gas_and_fee_buffer_cents: Decimal
    is_arbitrage_present: bool


def binary_cpmm_marginal_price(
    yes_reserve: Decimal | float | int | str,
    no_reserve: Decimal | float | int | str,
    *,
    outcome: Outcome = "YES",
) -> Decimal:
    """Return the current marginal probability under a binary fixed-product AMM.

    For a binary fixed-product market maker with reserve balances `y_yes` and
    `y_no`, the marginal price of YES is:

        p_yes = y_no / (y_yes + y_no)

    and the marginal price of NO is its complement.
    """

    pool = _normalize_cpmm_pool(yes_reserve=yes_reserve, no_reserve=no_reserve)
    total_reserves = pool.yes_reserve + pool.no_reserve
    if outcome == "YES":
        return pool.no_reserve / total_reserves
    if outcome == "NO":
        return pool.yes_reserve / total_reserves
    raise ValueError(f"Unsupported outcome: {outcome!r}")


def quote_binary_cpmm_trade(
    *,
    yes_reserve: Decimal | float | int | str,
    no_reserve: Decimal | float | int | str,
    outcome: Outcome,
    side: TradeSide,
    shares: Decimal | float | int | str,
) -> AmmExecutionQuote:
    """Quote an exact binary fixed-product AMM trade in collateral terms.

    This models the binary FPMM-style constant-product mechanism where a
    collateral deposit first mints complete YES/NO sets and then rebalances the
    pool while preserving the product invariant.

    Let `s` be the reserve of the selected outcome and `o` the reserve of the
    opposite outcome.

    Buying `q` shares of the selected outcome requires collateral `c` solving:

        (s + c - q)(o + c) = s o

    which yields the positive root:

        c = (-(s + o - q) + sqrt((s + o - q)^2 + 4 q o)) / 2

    Selling `q` shares of the selected outcome returns collateral `c` solving:

        (s + q - c)(o - c) = s o

    which yields the smaller positive root:

        c = ((s + o + q) - sqrt((s + o + q)^2 - 4 q o)) / 2
    """

    pool = _normalize_cpmm_pool(yes_reserve=yes_reserve, no_reserve=no_reserve)
    q = _require_positive(_as_decimal(shares, name="shares"), name="shares")

    selected_reserve, opposite_reserve = _select_reserves(pool=pool, outcome=outcome)
    marginal_before = binary_cpmm_marginal_price(pool.yes_reserve, pool.no_reserve, outcome=outcome)

    if side == "BUY":
        if q >= selected_reserve:
            raise ValueError("shares is too large relative to selected reserve")
        collateral_amount = _cpmm_buy_cost(
            selected_reserve=selected_reserve,
            opposite_reserve=opposite_reserve,
            shares=q,
        )
        selected_after = selected_reserve + collateral_amount - q
        opposite_after = opposite_reserve + collateral_amount
    elif side == "SELL":
        collateral_amount = _cpmm_sell_proceeds(
            selected_reserve=selected_reserve,
            opposite_reserve=opposite_reserve,
            shares=q,
        )
        selected_after = selected_reserve + q - collateral_amount
        opposite_after = opposite_reserve - collateral_amount
    else:
        raise ValueError(f"Unsupported side: {side!r}")

    yes_after, no_after = _combine_reserves(
        outcome=outcome,
        selected_reserve=selected_after,
        opposite_reserve=opposite_after,
    )
    marginal_after = binary_cpmm_marginal_price(yes_after, no_after, outcome=outcome)

    return AmmExecutionQuote(
        model="CPMM",
        outcome=outcome,
        side=side,
        shares=q,
        collateral_amount=collateral_amount,
        average_price=collateral_amount / q,
        marginal_price_before=marginal_before,
        marginal_price_after=marginal_after,
    )


def binary_lmsr_marginal_price(
    yes_inventory: Decimal | float | int | str,
    no_inventory: Decimal | float | int | str,
    liquidity: Decimal | float | int | str,
    *,
    outcome: Outcome = "YES",
) -> Decimal:
    """Return the current LMSR marginal price for a binary market."""

    state = _normalize_lmsr_state(
        yes_inventory=yes_inventory,
        no_inventory=no_inventory,
        liquidity=liquidity,
    )
    yes_probability = _lmsr_probability(
        selected_inventory=state.yes_inventory,
        opposite_inventory=state.no_inventory,
        liquidity=state.liquidity,
    )
    if outcome == "YES":
        return yes_probability
    if outcome == "NO":
        return _ONE - yes_probability
    raise ValueError(f"Unsupported outcome: {outcome!r}")


def quote_binary_lmsr_trade(
    *,
    yes_inventory: Decimal | float | int | str,
    no_inventory: Decimal | float | int | str,
    liquidity: Decimal | float | int | str,
    outcome: Outcome,
    side: TradeSide,
    shares: Decimal | float | int | str,
) -> AmmExecutionQuote:
    """Quote an exact binary LMSR trade via the cost-function difference.

    For the binary LMSR cost function

        C(q_yes, q_no) = b log(exp(q_yes / b) + exp(q_no / b))

    the exact cost to buy `q` shares of an outcome is:

        C(Q + q e_i) - C(Q)

    and the exact proceeds to sell `q` shares are:

        C(Q) - C(Q - q e_i)
    """

    state = _normalize_lmsr_state(
        yes_inventory=yes_inventory,
        no_inventory=no_inventory,
        liquidity=liquidity,
    )
    q = _require_positive(_as_decimal(shares, name="shares"), name="shares")
    marginal_before = binary_lmsr_marginal_price(
        state.yes_inventory,
        state.no_inventory,
        state.liquidity,
        outcome=outcome,
    )

    if outcome == "YES":
        yes_after, no_after = _apply_lmsr_trade(state=state, outcome=outcome, side=side, shares=q)
    elif outcome == "NO":
        yes_after, no_after = _apply_lmsr_trade(state=state, outcome=outcome, side=side, shares=q)
    else:
        raise ValueError(f"Unsupported outcome: {outcome!r}")

    if side == "BUY":
        collateral_amount = _lmsr_cost(yes_after, no_after, state.liquidity) - _lmsr_cost(
            state.yes_inventory,
            state.no_inventory,
            state.liquidity,
        )
    elif side == "SELL":
        collateral_amount = _lmsr_cost(state.yes_inventory, state.no_inventory, state.liquidity) - _lmsr_cost(
            yes_after,
            no_after,
            state.liquidity,
        )
    else:
        raise ValueError(f"Unsupported side: {side!r}")

    marginal_after = binary_lmsr_marginal_price(
        yes_after,
        no_after,
        state.liquidity,
        outcome=outcome,
    )
    return AmmExecutionQuote(
        model="LMSR",
        outcome=outcome,
        side=side,
        shares=q,
        collateral_amount=collateral_amount,
        average_price=collateral_amount / q,
        marginal_price_before=marginal_before,
        marginal_price_after=marginal_after,
    )


def compute_delta_1(
    *,
    amm_sell_price: Decimal | float | int | str,
    clob_best_ask: Decimal | float | int | str,
    order_size_shares: Decimal | float | int | str,
    gas_and_fee_buffer_cents: Decimal | float | int | str = DEFAULT_GAS_AND_FEE_BUFFER_CENTS,
) -> ArbitrageSpread:
    """Compute Delta_1(q) = P_amm_sell(q) - P_clob_buy(q) in cents.

    This is the direction where we lift the CLOB ask and immediately offset by
    selling the same shares to the AMM.
    """

    amm_sell = _require_probability(_as_decimal(amm_sell_price, name="amm_sell_price"), name="amm_sell_price")
    clob_ask = _require_probability(_as_decimal(clob_best_ask, name="clob_best_ask"), name="clob_best_ask")
    shares = _require_positive(_as_decimal(order_size_shares, name="order_size_shares"), name="order_size_shares")
    buffer = _require_positive(_as_decimal(gas_and_fee_buffer_cents, name="gas_and_fee_buffer_cents"), name="gas_and_fee_buffer_cents")

    gross_spread_cents = (amm_sell - clob_ask) * _ONE_HUNDRED
    net_spread_cents = gross_spread_cents - buffer
    return ArbitrageSpread(
        direction="CLOB_TO_AMM",
        order_size_shares=shares,
        gross_spread_cents=gross_spread_cents,
        net_spread_cents=net_spread_cents,
        gross_total_cents=gross_spread_cents * shares,
        net_total_cents=net_spread_cents * shares,
        gas_and_fee_buffer_cents=buffer,
        is_arbitrage_present=net_spread_cents > _ZERO,
    )


def compute_delta_2(
    *,
    amm_buy_price: Decimal | float | int | str,
    clob_best_bid: Decimal | float | int | str,
    order_size_shares: Decimal | float | int | str,
    gas_and_fee_buffer_cents: Decimal | float | int | str = DEFAULT_GAS_AND_FEE_BUFFER_CENTS,
) -> ArbitrageSpread:
    """Compute Delta_2(q) = P_clob_sell(q) - P_amm_buy(q) in cents.

    This is the direction where we buy from the AMM and immediately hit the
    CLOB bid.
    """

    amm_buy = _require_probability(_as_decimal(amm_buy_price, name="amm_buy_price"), name="amm_buy_price")
    clob_bid = _require_probability(_as_decimal(clob_best_bid, name="clob_best_bid"), name="clob_best_bid")
    shares = _require_positive(_as_decimal(order_size_shares, name="order_size_shares"), name="order_size_shares")
    buffer = _require_positive(_as_decimal(gas_and_fee_buffer_cents, name="gas_and_fee_buffer_cents"), name="gas_and_fee_buffer_cents")

    gross_spread_cents = (clob_bid - amm_buy) * _ONE_HUNDRED
    net_spread_cents = gross_spread_cents - buffer
    return ArbitrageSpread(
        direction="AMM_TO_CLOB",
        order_size_shares=shares,
        gross_spread_cents=gross_spread_cents,
        net_spread_cents=net_spread_cents,
        gross_total_cents=gross_spread_cents * shares,
        net_total_cents=net_spread_cents * shares,
        gas_and_fee_buffer_cents=buffer,
        is_arbitrage_present=net_spread_cents > _ZERO,
    )


def evaluate_dislocation_against_bbo(
    *,
    amm_buy_price: Decimal | float | int | str,
    amm_sell_price: Decimal | float | int | str,
    clob_best_bid: Decimal | float | int | str,
    clob_best_ask: Decimal | float | int | str,
    order_size_shares: Decimal | float | int | str,
    gas_and_fee_buffer_cents: Decimal | float | int | str = DEFAULT_GAS_AND_FEE_BUFFER_CENTS,
) -> tuple[ArbitrageSpread, ArbitrageSpread]:
    return (
        compute_delta_1(
            amm_sell_price=amm_sell_price,
            clob_best_ask=clob_best_ask,
            order_size_shares=order_size_shares,
            gas_and_fee_buffer_cents=gas_and_fee_buffer_cents,
        ),
        compute_delta_2(
            amm_buy_price=amm_buy_price,
            clob_best_bid=clob_best_bid,
            order_size_shares=order_size_shares,
            gas_and_fee_buffer_cents=gas_and_fee_buffer_cents,
        ),
    )


def _normalize_cpmm_pool(
    *,
    yes_reserve: Decimal | float | int | str,
    no_reserve: Decimal | float | int | str,
) -> BinaryCpmPool:
    yes = _require_positive(_as_decimal(yes_reserve, name="yes_reserve"), name="yes_reserve")
    no = _require_positive(_as_decimal(no_reserve, name="no_reserve"), name="no_reserve")
    return BinaryCpmPool(yes_reserve=yes, no_reserve=no)


def _normalize_lmsr_state(
    *,
    yes_inventory: Decimal | float | int | str,
    no_inventory: Decimal | float | int | str,
    liquidity: Decimal | float | int | str,
) -> BinaryLmsrState:
    yes = _as_decimal(yes_inventory, name="yes_inventory")
    no = _as_decimal(no_inventory, name="no_inventory")
    b = _require_positive(_as_decimal(liquidity, name="liquidity"), name="liquidity")
    return BinaryLmsrState(yes_inventory=yes, no_inventory=no, liquidity=b)


def _select_reserves(*, pool: BinaryCpmPool, outcome: Outcome) -> tuple[Decimal, Decimal]:
    if outcome == "YES":
        return pool.yes_reserve, pool.no_reserve
    if outcome == "NO":
        return pool.no_reserve, pool.yes_reserve
    raise ValueError(f"Unsupported outcome: {outcome!r}")


def _combine_reserves(
    *,
    outcome: Outcome,
    selected_reserve: Decimal,
    opposite_reserve: Decimal,
) -> tuple[Decimal, Decimal]:
    if outcome == "YES":
        return selected_reserve, opposite_reserve
    if outcome == "NO":
        return opposite_reserve, selected_reserve
    raise ValueError(f"Unsupported outcome: {outcome!r}")


def _cpmm_buy_cost(*, selected_reserve: Decimal, opposite_reserve: Decimal, shares: Decimal) -> Decimal:
    root_term = (selected_reserve + opposite_reserve - shares) ** 2 + (Decimal("4") * shares * opposite_reserve)
    return (-(selected_reserve + opposite_reserve - shares) + root_term.sqrt()) / Decimal("2")


def _cpmm_sell_proceeds(*, selected_reserve: Decimal, opposite_reserve: Decimal, shares: Decimal) -> Decimal:
    root_term = (selected_reserve + opposite_reserve + shares) ** 2 - (Decimal("4") * shares * opposite_reserve)
    if root_term < _ZERO:
        raise ValueError("shares is too large for a valid CPMM sell quote")
    return ((selected_reserve + opposite_reserve + shares) - root_term.sqrt()) / Decimal("2")


def _lmsr_cost(yes_inventory: Decimal, no_inventory: Decimal, liquidity: Decimal) -> Decimal:
    scaled_yes = yes_inventory / liquidity
    scaled_no = no_inventory / liquidity
    if scaled_yes >= scaled_no:
        anchor = scaled_yes
        return liquidity * (anchor + (Decimal("1") + (scaled_no - anchor).exp()).ln())
    anchor = scaled_no
    return liquidity * (anchor + (Decimal("1") + (scaled_yes - anchor).exp()).ln())


def _lmsr_probability(*, selected_inventory: Decimal, opposite_inventory: Decimal, liquidity: Decimal) -> Decimal:
    spread = (opposite_inventory - selected_inventory) / liquidity
    return _ONE / (_ONE + spread.exp())


def _apply_lmsr_trade(
    *,
    state: BinaryLmsrState,
    outcome: Outcome,
    side: TradeSide,
    shares: Decimal,
) -> tuple[Decimal, Decimal]:
    yes_after = state.yes_inventory
    no_after = state.no_inventory
    delta = shares if side == "BUY" else -shares
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported side: {side!r}")
    if outcome == "YES":
        yes_after += delta
    elif outcome == "NO":
        no_after += delta
    else:
        raise ValueError(f"Unsupported outcome: {outcome!r}")
    return yes_after, no_after