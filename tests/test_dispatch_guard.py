from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.execution.dispatch_guard import DispatchGuard, GuardDecision
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.priority_dispatcher import DispatchReceipt, PriorityDispatcher
from src.execution.priority_context import PriorityOrderContext
from src.execution.mev_router import MevExecutionRouter


def _make_config(**overrides: int) -> DispatchGuardConfig:
    values = {
        "dedup_window_ms": 100,
        "max_dispatches_per_source_per_window": 2,
        "rate_window_ms": 200,
        "circuit_breaker_threshold": 2,
        "circuit_breaker_reset_ms": 300,
        "max_open_positions_per_market": 2,
    }
    values.update(overrides)
    return DispatchGuardConfig(**values)


def _make_context(
    market_id: str = "MKT_A",
    side: str = "YES",
    signal_source: str = "OFI",
) -> PriorityOrderContext:
    return PriorityOrderContext(
        market_id=market_id,
        side=side,  # type: ignore[arg-type]
        signal_source=signal_source,  # type: ignore[arg-type]
        conviction_scalar=Decimal("0.5"),
        target_price=Decimal("0.64"),
        anchor_volume=Decimal("10"),
        max_capital=Decimal("20"),
    )


def _make_router() -> MevExecutionRouter:
    return MevExecutionRouter(
        lambda market_id: {
            "yes_bid": 0.45,
            "yes_ask": 0.55,
            "no_bid": 0.45,
            "no_ask": 0.55,
        }
    )


def test_valid_config_construction_passes() -> None:
    config = _make_config()
    assert config.circuit_breaker_threshold == 2


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("dedup_window_ms", 0, "dedup_window_ms"),
        ("max_dispatches_per_source_per_window", 0, "max_dispatches_per_source_per_window"),
        ("rate_window_ms", 0, "rate_window_ms"),
        ("circuit_breaker_reset_ms", 0, "circuit_breaker_reset_ms"),
        ("max_open_positions_per_market", 0, "max_open_positions_per_market"),
    ],
)
def test_invalid_positive_config_fields_raise_value_error(field_name: str, field_value: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _make_config(**{field_name: field_value})


def test_circuit_breaker_threshold_below_two_raises_value_error() -> None:
    with pytest.raises(ValueError, match="circuit_breaker_threshold"):
        _make_config(circuit_breaker_threshold=1)


def test_clean_guard_allows_dispatch() -> None:
    guard = DispatchGuard(_make_config())
    decision = guard.check(_make_context(), current_timestamp_ms=100)
    assert decision == GuardDecision(allowed=True, reason="OK")


def test_duplicate_within_window_returns_duplicate() -> None:
    guard = DispatchGuard(_make_config())
    context = _make_context()
    guard.check(context, 100)
    guard.record_dispatch(context, 100)

    decision = guard.check(context, 150)
    assert decision == GuardDecision(allowed=False, reason="DUPLICATE")


def test_duplicate_after_window_expires_is_allowed() -> None:
    guard = DispatchGuard(_make_config())
    context = _make_context()
    guard.check(context, 100)
    guard.record_dispatch(context, 100)

    decision = guard.check(context, 201)
    assert decision == GuardDecision(allowed=True, reason="OK")


def test_rate_gate_exceeds_per_source_but_other_source_unaffected() -> None:
    guard = DispatchGuard(_make_config(max_dispatches_per_source_per_window=2, rate_window_ms=100))
    context_a = _make_context(market_id="MKT_A1", signal_source="OFI")
    context_a_second = _make_context(market_id="MKT_A2", signal_source="OFI")
    context_a_third = _make_context(market_id="MKT_A3", signal_source="OFI")
    context_b = _make_context(signal_source="SI10")

    guard.check(context_a, 10)
    guard.record_dispatch(context_a, 10)
    guard.check(context_a_second, 20)
    guard.record_dispatch(context_a_second, 20)

    assert guard.check(context_a_third, 30) == GuardDecision(allowed=False, reason="RATE_EXCEEDED")
    assert guard.check(context_b, 30) == GuardDecision(allowed=True, reason="OK")


def test_rate_window_rolls_and_count_resets() -> None:
    guard = DispatchGuard(_make_config(max_dispatches_per_source_per_window=1, rate_window_ms=50))
    context = _make_context(market_id="MKT_RATE_1", signal_source="OFI")
    second_context = _make_context(market_id="MKT_RATE_2", signal_source="OFI")

    guard.check(context, 10)
    guard.record_dispatch(context, 10)

    assert guard.check(second_context, 20) == GuardDecision(allowed=False, reason="RATE_EXCEEDED")
    assert guard.check(second_context, 61) == GuardDecision(allowed=True, reason="OK")


def test_position_cap_rejects_same_market_after_limit() -> None:
    guard = DispatchGuard(_make_config(max_open_positions_per_market=2))
    context = _make_context(market_id="MKT_CAP", side="YES", signal_source="OFI")
    second_context = _make_context(market_id="MKT_CAP", side="NO", signal_source="OFI")
    third_context = _make_context(market_id="MKT_CAP", side="YES", signal_source="SI10")

    guard.check(context, 10)
    guard.record_dispatch(context, 10)
    guard.check(second_context, 20)
    guard.record_dispatch(second_context, 20)

    assert guard.check(third_context, 30) == GuardDecision(allowed=False, reason="POSITION_CAP")


def test_circuit_opens_after_exact_threshold_suppressions() -> None:
    guard = DispatchGuard(_make_config(circuit_breaker_threshold=2))

    guard.record_suppression("OFI")
    assert guard.guard_snapshot()["circuit_state"] == "CLOSED"
    guard.check(_make_context(), 100)
    guard.record_suppression("OFI")

    snapshot = guard.guard_snapshot()
    assert snapshot["circuit_state"] == "OPEN"
    assert snapshot["consecutive_suppressions"] == 2


def test_circuit_blocks_dispatches_while_open() -> None:
    guard = DispatchGuard(_make_config(circuit_breaker_threshold=2, circuit_breaker_reset_ms=100))
    guard.check(_make_context(), 10)
    guard.record_suppression("OFI")
    guard.record_suppression("OFI")

    assert guard.check(_make_context(), 50) == GuardDecision(allowed=False, reason="CIRCUIT_OPEN")


def test_circuit_transitions_to_half_open_after_reset_window() -> None:
    guard = DispatchGuard(_make_config(circuit_breaker_threshold=2, circuit_breaker_reset_ms=100))
    guard.check(_make_context(), 10)
    guard.record_suppression("OFI")
    guard.record_suppression("OFI")

    decision = guard.check(_make_context(), 110)
    assert decision == GuardDecision(allowed=True, reason="OK")
    assert guard.guard_snapshot()["circuit_state"] == "HALF_OPEN"


def test_successful_dispatch_during_half_open_closes_circuit_and_resets_counter() -> None:
    guard = DispatchGuard(_make_config(circuit_breaker_threshold=2, circuit_breaker_reset_ms=100))
    context = _make_context()
    guard.check(context, 10)
    guard.record_suppression("OFI")
    guard.record_suppression("OFI")
    guard.check(context, 110)

    guard.record_dispatch(context, 110)

    snapshot = guard.guard_snapshot()
    assert snapshot["circuit_state"] == "CLOSED"
    assert snapshot["consecutive_suppressions"] == 0
    assert snapshot["circuit_opened_at_ms"] is None


def test_suppression_during_half_open_reopens_circuit() -> None:
    guard = DispatchGuard(_make_config(circuit_breaker_threshold=2, circuit_breaker_reset_ms=100))
    context = _make_context()
    guard.check(context, 10)
    guard.record_suppression("OFI")
    guard.record_suppression("OFI")
    guard.check(context, 110)

    guard.record_suppression("OFI")

    snapshot = guard.guard_snapshot()
    assert snapshot["circuit_state"] == "OPEN"
    assert snapshot["circuit_opened_at_ms"] == 110


def test_guard_snapshot_reports_all_three_circuit_states() -> None:
    guard = DispatchGuard(_make_config(circuit_breaker_threshold=2, circuit_breaker_reset_ms=100))
    context = _make_context()

    assert guard.guard_snapshot()["circuit_state"] == "CLOSED"
    guard.check(context, 10)
    guard.record_suppression("OFI")
    guard.record_suppression("OFI")
    assert guard.guard_snapshot()["circuit_state"] == "OPEN"
    guard.check(context, 110)
    assert guard.guard_snapshot()["circuit_state"] == "HALF_OPEN"


def test_guard_snapshot_evicts_expired_dedup_entries() -> None:
    guard = DispatchGuard(_make_config(dedup_window_ms=50))
    context = _make_context()
    guard.check(context, 10)
    guard.record_dispatch(context, 10)
    assert guard.guard_snapshot()["active_dedup_keys"] == 1

    guard.check(context, 61)
    assert guard.guard_snapshot()["active_dedup_keys"] == 0


def test_dispatch_receipt_guard_reason_none_when_guard_allows() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "dry_run", guard=DispatchGuard(_make_config()))

    receipt = dispatcher.dispatch(_make_context(), 10)
    assert receipt.guard_reason is None


def test_dispatch_receipt_guard_reason_populated_when_guard_rejects() -> None:
    guard = DispatchGuard(_make_config())
    dispatcher = PriorityDispatcher(_make_router(), "dry_run", guard=guard)
    context = _make_context()

    dispatcher.dispatch(context, 10)
    receipt = dispatcher.dispatch(context, 20)

    assert receipt.executed is False
    assert receipt.guard_reason == "DUPLICATE"


def test_executed_receipt_requires_guard_reason_none() -> None:
    with pytest.raises(ValueError, match="guard_reason"):
        DispatchReceipt(
            context=_make_context(),
            mode="paper",
            executed=True,
            fill_price=Decimal("0.64"),
            fill_size=Decimal("1.0"),
            serialized_envelope="{}",
            dispatch_timestamp_ms=1,
            guard_reason="OK",
        )


def test_guard_absent_dispatcher_matches_existing_behavior() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "paper")

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.executed is True
    assert receipt.fill_price == Decimal("0.640001")
    assert receipt.fill_size == Decimal("5.000000")
    assert receipt.guard_reason is None


def test_guard_decision_is_frozen() -> None:
    decision = GuardDecision(allowed=True, reason="OK")

    with pytest.raises(FrozenInstanceError):
        decision.allowed = False  # type: ignore[misc]