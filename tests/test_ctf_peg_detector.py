from __future__ import annotations

from decimal import Decimal

import pytest

from src.detectors import CtfPegConfig
from src.events.mev_events import CtfMergeSignal
from src.signals.ctf_peg_detector import CtfPegDetector


def _config(**overrides) -> CtfPegConfig:
    values = {
        "min_yield": Decimal("0.050000"),
        "taker_fee_yes": Decimal("0.010000"),
        "taker_fee_no": Decimal("0.010000"),
        "slippage_budget": Decimal("0.005000"),
        "gas_ewma_alpha": Decimal("0.500000"),
        "max_desync_ms": 400,
    }
    values.update(overrides)
    return CtfPegConfig(**values)


def test_ctf_peg_detector_emits_signal_when_merge_edge_exceeds_min_yield() -> None:
    detector = CtfPegDetector("MKT_MERGE", _config())

    detector.record_base_fee(Decimal("0.010000"))
    signal = detector.evaluate(
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1200,
    )

    assert isinstance(signal, CtfMergeSignal)
    assert signal.market_id == "MKT_MERGE"
    assert signal.yes_ask == Decimal("0.380000")
    assert signal.no_ask == Decimal("0.400000")
    assert signal.gas_estimate == Decimal("0.010000")
    assert detector.state.net_edge == Decimal("0.185000")


def test_ctf_peg_detector_returns_none_when_edge_does_not_clear_floor() -> None:
    detector = CtfPegDetector("MKT_MERGE", _config(min_yield=Decimal("0.060000")))

    signal = detector.evaluate(
        yes_ask=Decimal("0.450000"),
        no_ask=Decimal("0.470000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1100,
        base_fee=Decimal("0.010000"),
    )

    assert signal is None
    assert detector.state.gas_estimate == Decimal("0.010000")
    assert detector.state.net_edge == Decimal("0.045000")


def test_ctf_peg_detector_uses_rolling_base_fee_in_o1_state() -> None:
    detector = CtfPegDetector("MKT_MERGE", _config(gas_ewma_alpha=Decimal("0.250000")))

    detector.record_base_fee(Decimal("0.020000"))
    detector.record_base_fee(Decimal("0.060000"))
    signal = detector.evaluate(
        yes_ask=Decimal("0.300000"),
        no_ask=Decimal("0.320000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1000,
    )

    assert signal is not None
    assert signal.gas_estimate == Decimal("0.03000000")
    assert detector.state.net_edge == Decimal("0.32500000")


def test_ctf_peg_config_accepts_valid_values() -> None:
    config = _config()

    assert config.min_yield == Decimal("0.050000")
    assert config.max_desync_ms == 400


@pytest.mark.parametrize(
    ("field_name", "override", "message"),
    [
        ("min_yield", Decimal("0"), "min_yield must be strictly positive"),
        ("taker_fee_yes", Decimal("0"), "taker_fee_yes must be strictly positive"),
        ("taker_fee_no", Decimal("-0.1"), "taker_fee_no must be strictly positive"),
        ("slippage_budget", Decimal("-0.1"), "slippage_budget must be greater than or equal to zero"),
        ("gas_ewma_alpha", Decimal("0"), "gas_ewma_alpha must be > 0 and <= 1"),
        ("max_desync_ms", 0, "max_desync_ms must be a strictly positive integer"),
    ],
)
def test_ctf_peg_config_invalid_fields_raise_descriptive_value_error(field_name: str, override, message: str) -> None:
    kwargs = {field_name: override}
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_fee_deduction_matches_manual_net_edge_to_six_decimal_places() -> None:
    config = _config(
        min_yield=Decimal("0.100000"),
        taker_fee_yes=Decimal("0.011111"),
        taker_fee_no=Decimal("0.022222"),
        slippage_budget=Decimal("0.003333"),
    )
    detector = CtfPegDetector("MKT_MERGE", config)

    signal = detector.evaluate(
        yes_ask=Decimal("0.300000"),
        no_ask=Decimal("0.400000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1000,
        base_fee=Decimal("0.010000"),
    )

    assert signal is not None
    manual_net_edge = Decimal("1") - Decimal("0.300000") - Decimal("0.400000") - Decimal("0.010000") - Decimal("0.011111") - Decimal("0.022222") - Decimal("0.003333")
    assert signal.net_edge.quantize(Decimal("0.000001")) == manual_net_edge.quantize(Decimal("0.000001"))


def test_slippage_budget_zero_is_valid_boundary() -> None:
    config = _config(slippage_budget=Decimal("0"))
    detector = CtfPegDetector("MKT_MERGE", config)

    signal = detector.evaluate(
        yes_ask=Decimal("0.400000"),
        no_ask=Decimal("0.430000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1000,
        base_fee=Decimal("0.010000"),
    )

    assert signal is not None
    assert signal.net_edge == Decimal("0.140000")


def test_gas_ewma_alpha_of_exactly_one_is_valid_and_uses_latest_value() -> None:
    config = _config(gas_ewma_alpha=Decimal("1"))
    detector = CtfPegDetector("MKT_MERGE", config)

    detector.record_base_fee(Decimal("0.020000"))
    detector.record_base_fee(Decimal("0.090000"))

    assert detector.state.gas_estimate == Decimal("0.090000")


def test_gas_ewma_alpha_of_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="gas_ewma_alpha"):
        _config(gas_ewma_alpha=Decimal("0.0"))


def test_desync_suppression_rejects_signal_and_preserves_state() -> None:
    detector = CtfPegDetector("MKT_MERGE", _config(max_desync_ms=100))
    baseline = detector.evaluate(
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1050,
        base_fee=Decimal("0.010000"),
    )
    state_before = detector.state

    suppressed = detector.evaluate(
        yes_ask=Decimal("0.100000"),
        no_ask=Decimal("0.100000"),
        yes_timestamp_ms=2000,
        no_timestamp_ms=2500,
        base_fee=Decimal("0.001000"),
    )

    assert baseline is not None
    assert suppressed is None
    assert detector.state == state_before


def test_state_immutability_for_identical_inputs() -> None:
    detector = CtfPegDetector("MKT_MERGE", _config(gas_ewma_alpha=Decimal("1")))
    first = detector.evaluate(
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1000,
        base_fee=Decimal("0.010000"),
    )
    second = detector.evaluate(
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1000,
        base_fee=Decimal("0.010000"),
    )

    assert first == second
    assert detector.state.net_edge == Decimal("0.185000")


def test_invalid_asks_return_none_without_initializing_gas_when_no_state_should_mutate() -> None:
    detector = CtfPegDetector("MKT_MERGE", _config())

    signal = detector.evaluate(
        yes_ask=Decimal("0"),
        no_ask=Decimal("0.400000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1000,
        base_fee=Decimal("0.010000"),
    )

    assert signal is None
    assert detector.state.gas_estimate == Decimal("0")
    assert detector.state.net_edge == Decimal("0")


def test_signal_carries_authoritative_post_cost_net_edge() -> None:
    detector = CtfPegDetector("MKT_MERGE", _config())

    signal = detector.evaluate(
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        yes_timestamp_ms=1000,
        no_timestamp_ms=1000,
        base_fee=Decimal("0.010000"),
    )

    assert signal is not None
    assert signal.net_edge == detector.state.net_edge