from __future__ import annotations

import pytest

from src.signals.toxic_wake_detector import ToxicWakeDetector


def test_rapid_sequential_large_trades_trip_toxic_wake() -> None:
    detector = ToxicWakeDetector(
        decay_rate=1.0,
        impulse_scale=1.0,
        contagion_intensity_threshold=100.0,
    )

    detector.process_trade_tick("mkt-1", timestamp=1.0, trade_volume=60.0)
    detector.process_trade_tick("mkt-1", timestamp=1.1, trade_volume=60.0)

    assert detector.is_wake_toxic("mkt-1", current_timestamp=1.1) is True
    assert detector.wake_states["mkt-1"].excitation_state == pytest.approx(114.290245)


def test_toxic_wake_decays_back_to_safe_after_enough_time() -> None:
    detector = ToxicWakeDetector(
        decay_rate=1.0,
        impulse_scale=1.0,
        contagion_intensity_threshold=100.0,
    )

    detector.process_trade_tick("mkt-1", timestamp=1.0, trade_volume=60.0)
    detector.process_trade_tick("mkt-1", timestamp=1.1, trade_volume=60.0)

    assert detector.is_wake_toxic("mkt-1", current_timestamp=1.1) is True
    assert detector.is_wake_toxic("mkt-1", current_timestamp=4.5) is False


def test_toxic_wake_state_is_isolated_per_market() -> None:
    detector = ToxicWakeDetector(
        decay_rate=1.0,
        impulse_scale=1.0,
        contagion_intensity_threshold=100.0,
    )

    detector.process_trade_tick("mkt-a", timestamp=1.0, trade_volume=120.0)
    detector.process_trade_tick("mkt-b", timestamp=1.0, trade_volume=20.0)

    assert detector.is_wake_toxic("mkt-a", current_timestamp=1.0) is True
    assert detector.is_wake_toxic("mkt-b", current_timestamp=1.0) is False