from __future__ import annotations

from src.trading.ensemble_risk import EnsembleRiskManager


def test_blocks_same_direction_across_strategies_but_allows_hedge() -> None:
    risk = EnsembleRiskManager()
    risk.register_position(
        position_id="POS-1",
        market_id="MKT-1",
        strategy_source="ofi_momentum",
        direction="YES",
    )

    allowed_yes, yes_reason = risk.can_enter(
        market_id="MKT-1",
        strategy_source="si10_contagion_arb",
        direction="YES",
    )
    allowed_no, no_reason = risk.can_enter(
        market_id="MKT-1",
        strategy_source="si10_contagion_arb",
        direction="NO",
    )

    assert allowed_yes is False
    assert yes_reason == {
        "market_id": "MKT-1",
        "direction": "YES",
        "strategy_source": "si10_contagion_arb",
        "blocking_strategy": "ofi_momentum",
    }
    assert allowed_no is True
    assert no_reason is None


def test_release_restores_capacity() -> None:
    risk = EnsembleRiskManager()
    risk.register_position(
        position_id="POS-1",
        market_id="MKT-1",
        strategy_source="ofi_momentum",
        direction="NO",
    )
    risk.release_position("POS-1")

    allowed, reason = risk.can_enter(
        market_id="MKT-1",
        strategy_source="si10_contagion_arb",
        direction="NO",
    )

    assert allowed is True
    assert reason is None
    assert risk.exposure_snapshot("MKT-1") == {"YES": {}, "NO": {}}