from __future__ import annotations

import math

import pytest

from src.signals.mm_tracker import MarketMakerTracker


def test_process_fill_event_tracks_multiple_makers_without_cross_contamination() -> None:
    tracker = MarketMakerTracker(alpha=0.5)

    tracker.process_fill_event("maker_a", price_delta=0.02, taker_volume=10.0)
    tracker.process_fill_event("maker_b", price_delta=0.06, taker_volume=10.0)
    tracker.process_fill_event("maker_a", price_delta=0.04, taker_volume=10.0)

    maker_a = tracker.fingerprints["maker_a"]
    maker_b = tracker.fingerprints["maker_b"]

    assert maker_a.kappa_ewma == pytest.approx(0.0025)
    assert maker_a.observations == 2
    assert maker_b.kappa_ewma == pytest.approx(0.003)
    assert maker_b.observations == 1


def test_get_vulnerable_makers_filters_by_capital_and_infinite_attack_volume() -> None:
    tracker = MarketMakerTracker(alpha=1.0)

    tracker.process_fill_event("maker_fast", price_delta=0.05, taker_volume=10.0)
    tracker.process_fill_event("maker_slow", price_delta=0.01, taker_volume=20.0)
    tracker.get_or_create_fingerprint("maker_idle")

    vulnerable = tracker.get_vulnerable_makers(target_spread_delta=0.02, max_capital=10.0)

    assert vulnerable == [
        ("maker_fast", pytest.approx(4.0)),
    ]
    assert math.isinf(
        tracker.fingerprints["maker_idle"].calculate_attack_volume(0.02)
    )


def test_get_or_create_fingerprint_reuses_existing_instance() -> None:
    tracker = MarketMakerTracker(alpha=0.3)

    first = tracker.get_or_create_fingerprint("maker_a")
    second = tracker.get_or_create_fingerprint("maker_a")

    assert first is second
    assert first.alpha == pytest.approx(0.3)


def test_process_fill_event_rejects_invalid_volume() -> None:
    tracker = MarketMakerTracker()

    with pytest.raises(ValueError, match="strictly greater than 0"):
        tracker.process_fill_event("maker_a", price_delta=0.01, taker_volume=0.0)