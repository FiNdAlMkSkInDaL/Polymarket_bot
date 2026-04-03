from __future__ import annotations

from decimal import Decimal

from src.models.arb_risk_manager import calculate_safe_arb_size


def test_safe_arb_size_is_clamped_by_hard_trade_cap_under_deep_liquidity() -> None:
    result = calculate_safe_arb_size(
        clob_available_volume_at_bbo=Decimal("1000"),
        amm_reserves={
            "yes_reserve": Decimal("10000"),
            "no_reserve": Decimal("10000"),
        },
        capital_cap_usd=Decimal("5000"),
        max_trade_size_usd=Decimal("50"),
        clob_target_price=Decimal("0.50"),
        amm_model="CPMM",
        outcome="YES",
    )

    assert result.reference_price == Decimal("0.50")
    assert result.q_capital == Decimal("100")
    assert result.q_inventory == Decimal("500.00")
    assert result.q_venue_depth_safe == Decimal("1000")
    assert result.size_shares == Decimal("100")
    assert result.size_usd == Decimal("50.00")
    assert result.capped_by == "capital"


def test_safe_arb_size_is_exactly_clamped_by_clob_venue_depth() -> None:
    result = calculate_safe_arb_size(
        clob_available_volume_at_bbo=Decimal("14"),
        amm_reserves={
            "yes_reserve": Decimal("10000"),
            "no_reserve": Decimal("10000"),
        },
        capital_cap_usd=Decimal("5000"),
        max_trade_size_usd=Decimal("5000"),
        clob_target_price=Decimal("0.50"),
        amm_model="CPMM",
        outcome="YES",
    )

    assert result.q_capital == Decimal("10000")
    assert result.q_inventory == Decimal("500.00")
    assert result.q_venue_depth_safe == Decimal("14")
    assert result.size_shares == Decimal("14")
    assert result.size_usd == Decimal("7.00")
    assert result.capped_by == "venue_depth_safe"


def test_safe_arb_size_is_clamped_by_amm_inventory_logic_when_reserves_are_shallow() -> None:
    result = calculate_safe_arb_size(
        clob_available_volume_at_bbo=Decimal("1000"),
        amm_reserves={
            "yes_reserve": Decimal("40"),
            "no_reserve": Decimal("60"),
        },
        capital_cap_usd=Decimal("5000"),
        max_trade_size_usd=Decimal("5000"),
        clob_target_price=Decimal("0.50"),
        amm_model="CPMM",
        outcome="YES",
    )

    assert result.q_capital == Decimal("10000")
    assert result.q_inventory == Decimal("2.00")
    assert result.q_venue_depth_safe == Decimal("1000")
    assert result.size_shares == Decimal("2.00")
    assert result.size_usd == Decimal("1.0000")
    assert result.capped_by == "inventory"