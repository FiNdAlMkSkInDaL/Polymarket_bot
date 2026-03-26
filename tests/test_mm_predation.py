from __future__ import annotations

import math

import pytest

from src.signals.mm_predation import MarketMakerFingerprint


def test_update_sensitivity_uses_recursive_ewma_smoothing() -> None:
    fingerprint = MarketMakerFingerprint("0xmaker", alpha=0.5)

    first = fingerprint.update_sensitivity(price_delta=0.02, taker_volume=10.0)
    second = fingerprint.update_sensitivity(price_delta=0.03, taker_volume=10.0)
    third = fingerprint.update_sensitivity(price_delta=0.01, taker_volume=10.0)

    assert first == pytest.approx(0.001)
    assert second == pytest.approx(0.002)
    assert third == pytest.approx(0.0015)
    assert fingerprint.kappa_current == pytest.approx(0.001)
    assert fingerprint.kappa_ewma == pytest.approx(0.0015)
    assert fingerprint.observations == 3


def test_calculate_attack_volume_uses_smoothed_sensitivity() -> None:
    fingerprint = MarketMakerFingerprint("0xmaker", alpha=0.25)
    fingerprint.update_sensitivity(price_delta=0.04, taker_volume=20.0)
    fingerprint.update_sensitivity(price_delta=0.08, taker_volume=20.0)

    assert fingerprint.kappa_ewma == pytest.approx(0.001375)
    assert fingerprint.calculate_attack_volume(0.02) == pytest.approx(14.545454545454545)


def test_calculate_attack_volume_avoids_division_by_zero() -> None:
    fingerprint = MarketMakerFingerprint("0xmaker", alpha=0.3)

    assert math.isinf(fingerprint.calculate_attack_volume(0.01))
    assert fingerprint.calculate_attack_volume(0.0) == 0.0


def test_update_sensitivity_rejects_non_positive_volume() -> None:
    fingerprint = MarketMakerFingerprint("0xmaker")

    with pytest.raises(ValueError, match="strictly greater than 0"):
        fingerprint.update_sensitivity(price_delta=0.01, taker_volume=0.0)


def test_alpha_must_be_in_open_closed_unit_interval() -> None:
    with pytest.raises(ValueError, match="alpha"):
        MarketMakerFingerprint("0xmaker", alpha=0.0)

    with pytest.raises(ValueError, match="alpha"):
        MarketMakerFingerprint("0xmaker", alpha=1.1)