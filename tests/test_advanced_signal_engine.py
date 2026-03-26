from __future__ import annotations

import pytest

from src.risk.entropy_circuit_breaker import EntropyCircuitBreaker
from src.signals.advanced_signal_engine import AdvancedMarketState, AdvancedSignalEngine
from src.signals.mm_predation_detector import CorrelatedOrderBookState, MMPredationDetector
from src.signals.mm_tracker import MarketMakerTracker
from src.signals.toxic_wake_detector import ToxicWakeDetector


def test_trade_tick_updates_toxic_wake_and_mm_predation_state_together() -> None:
    engine = AdvancedSignalEngine(
        mm_predation_detector=MMPredationDetector(
            tracker=MarketMakerTracker(alpha=1.0),
            target_spread_delta=0.02,
            max_capital=10.0,
            min_correlated_liquidity=5.0,
            max_correlated_spread=0.03,
        ),
        entropy_circuit_breaker=EntropyCircuitBreaker(),
        toxic_wake_detector=ToxicWakeDetector(
            contagion_intensity_threshold=5.0,
        ),
    )
    engine.register_correlated_market("mkt-1", "corr-1")
    engine.set_correlated_order_book(
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

    result = engine.on_trade_tick(
        market_id="mkt-1",
        maker_address="mkt-1",
        price_delta=0.05,
        trade_volume=10.0,
        timestamp=1.0,
    )

    assert result.market_id == "mkt-1"
    assert result.wake_intensity == pytest.approx(10.0)
    assert result.wake_toxic is True
    assert result.mm_predation_signal is not None
    assert result.mm_predation_signal.target_market_id == "mkt-1"
    assert result.mm_predation_signal.correlated_market_id == "corr-1"
    assert result.mm_predation_signal.v_attack == pytest.approx(4.0)
    assert engine.toxic_wake_detector.wake_states["mkt-1"].excitation_state == pytest.approx(10.0)
    assert engine.mm_predation_detector.tracker.fingerprints["mkt-1"].kappa_ewma == pytest.approx(0.005)


def test_orderbook_update_routes_to_entropy_breaker() -> None:
    engine = AdvancedSignalEngine()

    is_safe = engine.on_orderbook_update(
        market_id="mkt-1",
        top_bid_vol=1.0,
        top_ask_vol=1.0,
    )

    assert is_safe is False
    assert engine.last_make_safety_by_market["mkt-1"] is False
    assert engine.entropy_circuit_breaker.last_entropy_by_market["mkt-1"] == pytest.approx(1.0)


def test_get_market_state_returns_unified_snapshot_with_zero_active_traps() -> None:
    engine = AdvancedSignalEngine(
        mm_predation_detector=MMPredationDetector(
            tracker=MarketMakerTracker(alpha=1.0),
            target_spread_delta=0.02,
            max_capital=10.0,
            min_correlated_liquidity=5.0,
            max_correlated_spread=0.03,
        ),
        entropy_circuit_breaker=EntropyCircuitBreaker(critical_entropy_threshold=0.95),
        toxic_wake_detector=ToxicWakeDetector(contagion_intensity_threshold=5.0),
    )
    engine.register_correlated_market("mkt-2", "corr-2")
    engine.set_correlated_order_book(
        "corr-2",
        CorrelatedOrderBookState(
            yes_bid=0.48,
            yes_ask=0.60,
            no_bid=0.40,
            no_ask=0.60,
            yes_liquidity=25.0,
            no_liquidity=25.0,
        ),
    )

    engine.on_trade_tick(
        market_id="mkt-2",
        maker_address="maker-a",
        price_delta=0.05,
        trade_volume=10.0,
        timestamp=1.0,
    )
    engine.on_trade_tick(
        market_id="mkt-2",
        maker_address="maker-a",
        price_delta=0.05,
        trade_volume=10.0,
        timestamp=2.0,
    )
    engine.on_orderbook_update(
        market_id="mkt-2",
        top_bid_vol=9.0,
        top_ask_vol=1.0,
    )

    state = engine.get_market_state("mkt-2", current_timestamp=2.0)

    assert isinstance(state, AdvancedMarketState)
    assert state.market_id == "mkt-2"
    assert state.is_toxic_wake is True
    assert state.is_safe_entropy is True
    assert state.active_mm_traps == []