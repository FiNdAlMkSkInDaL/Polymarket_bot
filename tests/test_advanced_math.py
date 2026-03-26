from __future__ import annotations

import math

import pytest

from src.signals.advanced_math import (
    BinaryTouchEntropyMethod,
    BoundaryAvellanedaStoikovMethod,
    HawkesToxicWakeState,
)


def test_hawkes_toxic_wake_decay_over_time_steps() -> None:
    state = HawkesToxicWakeState(decay_rate=0.5, impulse_scale=1.0)

    intensity_at_start = state.update(timestamp=100.0, trade_volume=4.0)
    assert intensity_at_start == pytest.approx(4.0)

    intensity_after_two_seconds = state.get_intensity(102.0)
    assert intensity_after_two_seconds == pytest.approx(4.0 * math.exp(-1.0), rel=1e-9)

    updated_intensity = state.update(timestamp=102.0, trade_volume=1.5)
    expected = 4.0 * math.exp(-1.0) + 1.5
    assert updated_intensity == pytest.approx(expected, rel=1e-9)


def test_hawkes_update_ignores_negative_elapsed_time() -> None:
    state = HawkesToxicWakeState(decay_rate=1.0, impulse_scale=2.0)

    state.update(timestamp=10.0, trade_volume=1.0)
    intensity = state.update(timestamp=9.0, trade_volume=3.0)

    assert intensity == pytest.approx(8.0, rel=1e-9)
    assert state.last_update_time == pytest.approx(10.0)


def test_binary_touch_entropy_peaks_at_balanced_flow() -> None:
    balanced = BinaryTouchEntropyMethod.calculate_entropy(50.0, 50.0)
    imbalanced = BinaryTouchEntropyMethod.calculate_entropy(90.0, 10.0)

    assert balanced == pytest.approx(1.0, rel=1e-9)
    assert balanced > imbalanced


def test_binary_touch_entropy_handles_zero_volume_edges() -> None:
    assert BinaryTouchEntropyMethod.calculate_entropy(0.0, 0.0) == pytest.approx(0.0)
    assert BinaryTouchEntropyMethod.calculate_entropy(100.0, 0.0) == pytest.approx(0.0)
    assert BinaryTouchEntropyMethod.calculate_entropy(0.0, 100.0) == pytest.approx(0.0)


def test_boundary_avellaneda_stoikov_reservation_price_is_clamped_to_unit_interval() -> None:
    low = BoundaryAvellanedaStoikovMethod.get_reservation_price(
        mid_price=0.02,
        inventory_qty=100.0,
        risk_aversion_gamma=5.0,
        raw_variance=8.0,
        time_to_resolution=10.0,
    )
    high = BoundaryAvellanedaStoikovMethod.get_reservation_price(
        mid_price=0.98,
        inventory_qty=-100.0,
        risk_aversion_gamma=5.0,
        raw_variance=8.0,
        time_to_resolution=10.0,
    )

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)


def test_boundary_avellaneda_stoikov_uses_binary_bounded_variance() -> None:
    reservation_price = BoundaryAvellanedaStoikovMethod.get_reservation_price(
        mid_price=0.60,
        inventory_qty=2.0,
        risk_aversion_gamma=0.5,
        raw_variance=0.20,
        time_to_resolution=3.0,
    )
    expected_variance = 0.20 * 0.60 * 0.40
    expected = 0.60 - 2.0 * 0.5 * expected_variance * 3.0

    assert reservation_price == pytest.approx(expected, rel=1e-9)