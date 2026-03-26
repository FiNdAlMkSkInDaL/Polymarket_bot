from __future__ import annotations

import pytest

from src.signals.mm_tracker import MarketMakerTracker
from src.signals.mm_predation_detector import (
    CorrelatedOrderBookState,
    MMPredationDetector,
)


def test_evaluate_market_tick_emits_signal_when_vulnerability_and_book_gate_pass() -> None:
    detector = MMPredationDetector(
        target_spread_delta=0.02,
        max_capital=10.0,
        min_correlated_liquidity=5.0,
        max_correlated_spread=0.03,
    )
    detector.set_order_book(
        "corr-1",
        CorrelatedOrderBookState(
            yes_bid=0.48,
            yes_ask=0.49,
            no_bid=0.51,
            no_ask=0.52,
            yes_liquidity=25.0,
            no_liquidity=25.0,
        ),
    )

    signal = None
    for _ in range(5):
        signal = detector.evaluate_market_tick(
            target_market_id="target-1",
            maker_address="maker-a",
            price_delta=0.05,
            taker_volume=10.0,
            correlated_market_id="corr-1",
        )

    assert signal is not None
    assert signal.target_market_id == "target-1"
    assert signal.correlated_market_id == "corr-1"
    assert signal.maker_address == "maker-a"
    assert signal.trap_direction == "YES"
    assert signal.hedge_direction == "NO"
    assert signal.v_attack == pytest.approx(5.949547834364587)
    assert signal.estimated_kappa == pytest.approx(0.0033616)
    assert signal.correlated_spread == pytest.approx(0.01)
    assert signal.correlated_available_liquidity == pytest.approx(25.0)


def test_evaluate_market_tick_rejects_signal_when_correlated_book_cannot_absorb_trap() -> None:
    detector = MMPredationDetector(
        tracker=MarketMakerTracker(alpha=1.0),
        target_spread_delta=0.02,
        max_capital=10.0,
        min_correlated_liquidity=5.0,
        max_correlated_spread=0.01,
    )
    detector.set_order_book(
        "corr-1",
        CorrelatedOrderBookState(
            yes_bid=0.48,
            yes_ask=0.50,
            no_bid=0.50,
            no_ask=0.53,
            yes_liquidity=3.0,
            no_liquidity=3.0,
        ),
    )

    signal = detector.evaluate_market_tick(
        target_market_id="target-1",
        maker_address="maker-a",
        price_delta=0.05,
        taker_volume=10.0,
        correlated_market_id="corr-1",
    )

    assert signal is None