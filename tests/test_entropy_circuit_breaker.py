from __future__ import annotations

import pytest

from src.risk.entropy_circuit_breaker import EntropyCircuitBreaker


def test_balanced_thin_touch_volumes_trip_entropy_breaker() -> None:
    breaker = EntropyCircuitBreaker(critical_entropy_threshold=0.95)

    is_safe = breaker.is_safe_to_make(
        market_id="mkt-balanced",
        top_bid_vol=1.0,
        top_ask_vol=1.0,
    )

    assert is_safe is False
    assert breaker.last_entropy_by_market["mkt-balanced"] == pytest.approx(1.0)


def test_skewed_touch_volumes_leave_entropy_breaker_open() -> None:
    breaker = EntropyCircuitBreaker(critical_entropy_threshold=0.95)

    is_safe = breaker.is_safe_to_make(
        market_id="mkt-skewed",
        top_bid_vol=90.0,
        top_ask_vol=10.0,
    )

    assert is_safe is True
    assert breaker.last_entropy_by_market["mkt-skewed"] < 0.95


def test_zero_touch_volume_defaults_to_safe_state() -> None:
    breaker = EntropyCircuitBreaker(critical_entropy_threshold=0.95)

    is_safe = breaker.is_safe_to_make(
        market_id="mkt-empty",
        top_bid_vol=0.0,
        top_ask_vol=0.0,
    )

    assert is_safe is True
    assert breaker.last_entropy_by_market["mkt-empty"] == pytest.approx(0.0)