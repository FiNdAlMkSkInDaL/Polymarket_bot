from __future__ import annotations

from src.signals.microstructure_utils import CausalLagGate


class _Snapshot:
    def __init__(self, timestamp: float, server_time: float = 0.0) -> None:
        self.timestamp = timestamp
        self.server_time = server_time


def test_causal_lag_gate_accepts_fresh_ordered_pair() -> None:
    gate = CausalLagGate(
        max_leader_age_ms=5000.0,
        max_lagger_age_ms=30000.0,
        max_causal_lag_ms=600000.0,
        allow_negative_lag=False,
    )

    assessment = gate.assess(
        _Snapshot(200.0),
        _Snapshot(185.0),
        reference_timestamp=200.0,
    )

    assert assessment.is_valid is True
    assert assessment.gate_result == "accepted"
    assert assessment.leader_age_ms == 0.0
    assert assessment.lagger_age_ms == 15000.0
    assert assessment.causal_lag_ms == 15000.0


def test_causal_lag_gate_rejects_lagger_newer_than_leader_when_negative_lag_disabled() -> None:
    gate = CausalLagGate(
        max_leader_age_ms=5000.0,
        max_lagger_age_ms=30000.0,
        max_causal_lag_ms=600000.0,
        allow_negative_lag=False,
    )

    assessment = gate.assess(
        _Snapshot(200.0),
        _Snapshot(201.0),
        reference_timestamp=201.0,
    )

    assert assessment.is_valid is False
    assert assessment.gate_result == "lagger_newer_than_leader"
    assert assessment.causal_lag_ms == -1000.0


def test_causal_lag_gate_rejects_stale_leader_before_other_checks() -> None:
    gate = CausalLagGate(
        max_leader_age_ms=5000.0,
        max_lagger_age_ms=30000.0,
        max_causal_lag_ms=600000.0,
        allow_negative_lag=False,
    )

    assessment = gate.assess(
        _Snapshot(190.0),
        _Snapshot(189.0),
        reference_timestamp=200.0,
    )

    assert assessment.is_valid is False
    assert assessment.gate_result == "leader_stale"
    assert assessment.leader_age_ms == 10000.0


def test_causal_lag_gate_rejects_causal_lag_beyond_boundary() -> None:
    gate = CausalLagGate(
        max_leader_age_ms=5000.0,
        max_lagger_age_ms=700000.0,
        max_causal_lag_ms=600000.0,
        allow_negative_lag=False,
    )

    assessment = gate.assess(
        _Snapshot(1000.0),
        _Snapshot(399.0),
        reference_timestamp=1000.0,
    )

    assert assessment.is_valid is False
    assert assessment.gate_result == "causal_lag_too_large"
    assert assessment.causal_lag_ms == 601000.0


def test_causal_lag_gate_accepts_exact_threshold_boundary() -> None:
    gate = CausalLagGate(
        max_leader_age_ms=5000.0,
        max_lagger_age_ms=30000.0,
        max_causal_lag_ms=30000.0,
        allow_negative_lag=False,
    )

    assessment = gate.assess(
        _Snapshot(200.0),
        _Snapshot(170.0),
        reference_timestamp=200.0,
    )

    assert assessment.is_valid is True
    assert assessment.gate_result == "accepted"
    assert assessment.lagger_age_ms == 30000.0
    assert assessment.causal_lag_ms == 30000.0