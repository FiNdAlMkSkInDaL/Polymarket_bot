from __future__ import annotations

import pytest

from src.execution.volatility_monitor import RollingMidPriceVolatilityMonitor


def test_rolling_mid_price_volatility_monitor_computes_sigma_in_cents() -> None:
    monitor = RollingMidPriceVolatilityMonitor(window_ms=300_000, max_safe_volatility_cents=5.0)

    for timestamp_ms, mid_price in [
        (1_000, 0.50),
        (2_000, 0.52),
        (3_000, 0.48),
        (4_000, 0.50),
    ]:
        status = monitor.record_mid_price("asset-1", mid_price, timestamp_ms)

    assert status.asset_id == "asset-1"
    assert status.sample_count == 4
    assert status.sigma_cents == pytest.approx(1.4142, rel=1e-3)
    assert status.is_breached is False


def test_rolling_mid_price_volatility_monitor_prunes_old_points_outside_window() -> None:
    monitor = RollingMidPriceVolatilityMonitor(window_ms=5_000, max_safe_volatility_cents=2.0)

    monitor.record_mid_price("asset-1", 0.50, 1_000)
    monitor.record_mid_price("asset-1", 0.70, 2_000)
    status = monitor.record_mid_price("asset-1", 0.50, 7_000)

    assert status.sample_count == 2
    assert status.sigma_cents == pytest.approx(10.0, rel=1e-3)
    assert status.is_breached is True