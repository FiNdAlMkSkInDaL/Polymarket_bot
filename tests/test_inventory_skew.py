from __future__ import annotations

from decimal import Decimal

from src.models.inventory_skew import InventorySkewInputs, compute_inventory_skew, compute_inventory_skew_quotes


def test_inventory_skew_is_exactly_zero_when_inventory_is_flat() -> None:
    skew = compute_inventory_skew(
        current_inventory_usd=Decimal("0"),
        max_inventory_usd=Decimal("1000"),
        base_spread=Decimal("0.04"),
    )

    assert skew["inventory_ratio"] == Decimal("0")
    assert skew["urgency"] == Decimal("0")
    assert skew["center_shift"] == Decimal("0")
    assert skew["adjusted_half_spread"] == Decimal("0.02")


def test_inventory_skew_at_half_inventory_has_small_cubic_shift() -> None:
    base_spread = Decimal("0.04")
    skew = compute_inventory_skew(
        current_inventory_usd=Decimal("500"),
        max_inventory_usd=Decimal("1000"),
        base_spread=base_spread,
    )

    expected_urgency = Decimal("0.125")
    expected_shift = -(base_spread * (expected_urgency + (Decimal("0.25") * (expected_urgency ** 2))))

    assert skew["inventory_ratio"] == Decimal("0.5")
    assert skew["urgency"] == expected_urgency
    assert skew["center_shift"] == expected_shift
    assert abs(skew["center_shift"]) > Decimal("0")
    assert abs(skew["center_shift"]) < (base_spread * Decimal("0.2"))


def test_inventory_skew_at_inventory_cap_reaches_one_point_two_five_times_base_spread() -> None:
    base_spread = Decimal("0.04")
    skew = compute_inventory_skew(
        current_inventory_usd=Decimal("1000"),
        max_inventory_usd=Decimal("1000"),
        base_spread=base_spread,
    )

    assert skew["inventory_ratio"] == Decimal("1")
    assert skew["urgency"] == Decimal("1")
    assert skew["center_shift"] == -(base_spread * Decimal("1.25"))
    assert abs(skew["center_shift"]) == (base_spread * Decimal("1.25"))


def test_long_inventory_lowers_ask_relative_to_flat_quote() -> None:
    flat_quote = compute_inventory_skew_quotes(
        InventorySkewInputs(
            current_inventory_usd=Decimal("0"),
            max_inventory_usd=Decimal("1000"),
            base_spread=Decimal("0.04"),
            mid_price=Decimal("0.50"),
        )
    )
    long_quote = compute_inventory_skew_quotes(
        InventorySkewInputs(
            current_inventory_usd=Decimal("1000"),
            max_inventory_usd=Decimal("1000"),
            base_spread=Decimal("0.04"),
            mid_price=Decimal("0.50"),
            best_bid=Decimal("0.47"),
        )
    )

    assert flat_quote.bid_price == Decimal("0.48")
    assert flat_quote.ask_price == Decimal("0.52")
    assert long_quote.ask_price < flat_quote.ask_price
    assert long_quote.bid_price < flat_quote.bid_price
    assert long_quote.aggressive_exit is True
    assert long_quote.aggressive_side == "SELL"


def test_short_inventory_raises_bid_relative_to_flat_quote() -> None:
    flat_quote = compute_inventory_skew_quotes(
        InventorySkewInputs(
            current_inventory_usd=Decimal("0"),
            max_inventory_usd=Decimal("1000"),
            base_spread=Decimal("0.04"),
            mid_price=Decimal("0.50"),
        )
    )
    short_quote = compute_inventory_skew_quotes(
        InventorySkewInputs(
            current_inventory_usd=Decimal("-1000"),
            max_inventory_usd=Decimal("1000"),
            base_spread=Decimal("0.04"),
            mid_price=Decimal("0.50"),
            best_ask=Decimal("0.53"),
        )
    )

    assert short_quote.bid_price > flat_quote.bid_price
    assert short_quote.ask_price > flat_quote.ask_price
    assert short_quote.aggressive_exit is True
    assert short_quote.aggressive_side == "BUY"