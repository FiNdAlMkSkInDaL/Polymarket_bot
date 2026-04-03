from __future__ import annotations

from decimal import Decimal

import pytest

from src.models.amm_pricing import (
    DEFAULT_GAS_AND_FEE_BUFFER_CENTS,
    compute_delta_1,
    compute_delta_2,
    quote_binary_cpmm_trade,
    quote_binary_lmsr_trade,
)


@pytest.mark.parametrize(
    ("yes_reserve", "no_reserve", "shares"),
    [
        (Decimal("100"), Decimal("100"), Decimal("5")),
        (Decimal("250"), Decimal("150"), Decimal("12.5")),
        (Decimal("1000"), Decimal("400"), Decimal("25")),
    ],
)
def test_cpmm_buy_average_price_is_worse_than_pre_trade_marginal_price(
    yes_reserve: Decimal,
    no_reserve: Decimal,
    shares: Decimal,
) -> None:
    quote = quote_binary_cpmm_trade(
        yes_reserve=yes_reserve,
        no_reserve=no_reserve,
        outcome="YES",
        side="BUY",
        shares=shares,
    )

    assert quote.collateral_amount > Decimal("0")
    assert quote.average_price > quote.marginal_price_before
    assert quote.marginal_price_after > quote.marginal_price_before
    assert quote.marginal_price_after > quote.average_price


@pytest.mark.parametrize(
    ("yes_inventory", "no_inventory", "liquidity", "shares"),
    [
        (Decimal("0"), Decimal("0"), Decimal("100"), Decimal("5")),
        (Decimal("20"), Decimal("-10"), Decimal("80"), Decimal("7.5")),
        (Decimal("120"), Decimal("40"), Decimal("250"), Decimal("30")),
    ],
)
def test_lmsr_buy_average_price_is_worse_than_pre_trade_marginal_price(
    yes_inventory: Decimal,
    no_inventory: Decimal,
    liquidity: Decimal,
    shares: Decimal,
) -> None:
    quote = quote_binary_lmsr_trade(
        yes_inventory=yes_inventory,
        no_inventory=no_inventory,
        liquidity=liquidity,
        outcome="YES",
        side="BUY",
        shares=shares,
    )

    assert quote.collateral_amount > Decimal("0")
    assert quote.average_price > quote.marginal_price_before
    assert quote.marginal_price_after > quote.marginal_price_before
    assert quote.marginal_price_after > quote.average_price


def test_compute_delta_1_returns_exact_cents_and_total_spread() -> None:
    spread = compute_delta_1(
        amm_sell_price=Decimal("0.56"),
        clob_best_ask=Decimal("0.52"),
        order_size_shares=Decimal("10"),
    )

    assert spread.direction == "CLOB_TO_AMM"
    assert spread.gas_and_fee_buffer_cents == DEFAULT_GAS_AND_FEE_BUFFER_CENTS
    assert spread.gross_spread_cents == Decimal("4.00")
    assert spread.net_spread_cents == Decimal("2.50")
    assert spread.gross_total_cents == Decimal("40.00")
    assert spread.net_total_cents == Decimal("25.00")
    assert spread.is_arbitrage_present is True


def test_compute_delta_2_returns_exact_cents_and_total_spread() -> None:
    spread = compute_delta_2(
        amm_buy_price=Decimal("0.48"),
        clob_best_bid=Decimal("0.53"),
        order_size_shares=Decimal("10"),
    )

    assert spread.direction == "AMM_TO_CLOB"
    assert spread.gas_and_fee_buffer_cents == DEFAULT_GAS_AND_FEE_BUFFER_CENTS
    assert spread.gross_spread_cents == Decimal("5.00")
    assert spread.net_spread_cents == Decimal("3.50")
    assert spread.gross_total_cents == Decimal("50.00")
    assert spread.net_total_cents == Decimal("35.00")
    assert spread.is_arbitrage_present is True


def test_compute_delta_2_rejects_one_cent_spread_that_fails_buffer() -> None:
    spread = compute_delta_2(
        amm_buy_price=Decimal("0.50"),
        clob_best_bid=Decimal("0.51"),
        order_size_shares=Decimal("20"),
    )

    assert spread.gross_spread_cents == Decimal("1.00")
    assert spread.net_spread_cents == Decimal("-0.50")
    assert spread.is_arbitrage_present is False


def test_compute_delta_2_accepts_five_cent_spread_that_clears_buffer() -> None:
    spread = compute_delta_2(
        amm_buy_price=Decimal("0.47"),
        clob_best_bid=Decimal("0.52"),
        order_size_shares=Decimal("20"),
    )

    assert spread.gross_spread_cents == Decimal("5.00")
    assert spread.net_spread_cents == Decimal("3.50")
    assert spread.net_total_cents == Decimal("70.00")
    assert spread.is_arbitrage_present is True