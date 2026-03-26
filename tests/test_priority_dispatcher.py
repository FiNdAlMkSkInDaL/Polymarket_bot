from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from decimal import Decimal
import logging

import pytest

from src.execution.alpha_adapters import ctf_to_context, ofi_to_context, si9_to_context
from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.mev_router import MevExecutionRouter
from src.execution.mev_serializer import deserialize_conviction_scalar, deserialize_envelope
from src.execution.priority_dispatcher import DispatchReceipt, PriorityDispatcher
from src.execution.priority_context import PriorityOrderContext


def _make_router() -> MevExecutionRouter:
    return MevExecutionRouter(
        lambda market_id: {
            "yes_bid": 0.45,
            "yes_ask": 0.55,
            "no_bid": 0.45,
            "no_ask": 0.55,
        }
    )


def _make_context() -> PriorityOrderContext:
    return ofi_to_context(
        market_id="MKT_PRIORITY",
        side="YES",
        target_price=Decimal("0.640000"),
        anchor_volume=Decimal("50.000000"),
        max_capital=Decimal("100.000000"),
        conviction_scalar=Decimal("0.850000"),
    )


def test_dry_run_dispatch_returns_unexecuted_receipt_with_null_fills() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "dry_run")

    receipt = dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=123456)

    assert receipt.executed is False
    assert receipt.fill_price is None
    assert receipt.fill_size is None
    assert receipt.dispatch_timestamp_ms == 123456
    assert receipt.guard_reason is None


def test_dry_run_dispatch_returns_non_empty_serialized_envelope() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "dry_run")

    receipt = dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=7)
    envelope = deserialize_envelope(receipt.serialized_envelope)

    assert receipt.serialized_envelope
    assert envelope["payload_count"] == 2


def test_paper_dispatch_returns_executed_receipt_with_fill_details() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "paper")

    receipt = dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=1000)

    assert receipt.executed is True
    assert receipt.fill_price == Decimal("0.640001")
    assert receipt.fill_size == Decimal("42.500000")
    assert receipt.dispatch_timestamp_ms == 1000
    assert receipt.guard_reason is None


def test_paper_fill_price_is_exact_target_plus_epsilon() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "paper")
    context = _make_context()

    receipt = dispatcher.dispatch(context, dispatch_timestamp_ms=10)

    assert receipt.fill_price == (context.target_price + Decimal("0.000001")).quantize(Decimal("0.000001"))


def test_paper_fill_size_matches_effective_size_formula() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "paper")
    context = _make_context()

    receipt = dispatcher.dispatch(context, dispatch_timestamp_ms=10)
    fill_price = Decimal("0.640001")
    expected_size = (min(context.anchor_volume, context.max_capital / fill_price) * context.conviction_scalar).quantize(Decimal("0.000001"))

    assert receipt.fill_size == expected_size


def test_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported priority dispatch mode"):
        PriorityDispatcher(_make_router(), "sandbox")  # type: ignore[arg-type]


def test_dispatch_receipt_is_immutable() -> None:
    receipt = DispatchReceipt(
        context=_make_context(),
        mode="dry_run",
        executed=False,
        fill_price=None,
        fill_size=None,
        serialized_envelope="{}",
        dispatch_timestamp_ms=1,
    )

    with pytest.raises(FrozenInstanceError):
        receipt.executed = True  # type: ignore[misc]


def test_ofi_adapter_produces_ofi_signal_source() -> None:
    context = ofi_to_context(
        market_id="MKT_OFI",
        side="NO",
        target_price=Decimal("0.5"),
        anchor_volume=Decimal("5"),
        max_capital=Decimal("10"),
        conviction_scalar=Decimal("0.4"),
    )

    assert context.signal_source == "OFI"


def test_si9_adapter_rejects_no_side_explicitly() -> None:
    with pytest.raises(ValueError, match="SI-9 adapter requires side='YES'"):
        si9_to_context(
            market_id="MKT_SI9",
            side="NO",  # type: ignore[arg-type]
            target_price=Decimal("0.5"),
            anchor_volume=Decimal("5"),
            max_capital=Decimal("10"),
            conviction_scalar=Decimal("0.4"),
        )


def test_ctf_adapter_produces_ctf_signal_source() -> None:
    context = ctf_to_context(
        market_id="MKT_CTF",
        side="YES",
        target_price=Decimal("0.55"),
        anchor_volume=Decimal("8"),
        max_capital=Decimal("12"),
        conviction_scalar=Decimal("0.9"),
    )

    assert context.signal_source == "CTF"


def test_deserialize_conviction_scalar_round_trips_fixed_precision_strings() -> None:
    expected_values = [
        Decimal("0.000000"),
        Decimal("0.500000"),
        Decimal("1.000000"),
    ]

    assert [deserialize_conviction_scalar(format(value, ".6f")) for value in expected_values] == expected_values


def test_deserialize_conviction_scalar_rejects_out_of_range_values() -> None:
    for raw in ("1.000001", "-0.000001"):
        with pytest.raises(ValueError, match="conviction_scalar"):
            deserialize_conviction_scalar(raw)


def test_deserialize_envelope_returns_decimal_conviction_scalar() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "dry_run")

    receipt = dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=1)
    envelope = deserialize_envelope(receipt.serialized_envelope)

    assert envelope["payloads"][0]["context"]["conviction_scalar"] == Decimal("0.850000")
    assert isinstance(envelope["payloads"][0]["context"]["conviction_scalar"], Decimal)


def test_dispatch_log_line_is_valid_machine_parseable_json(caplog: pytest.LogCaptureFixture) -> None:
    dispatcher = PriorityDispatcher(_make_router(), "dry_run")

    with caplog.at_level(logging.DEBUG):
        dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=1)

    parsed = json.loads(caplog.records[-1].getMessage())
    assert parsed["mode"] == "dry_run"
    assert parsed["signal_source"] == "OFI"
    assert parsed["market_id"] == "MKT_PRIORITY"
    assert parsed["executed"] is False


def test_decimal_fields_in_receipt_stringify_without_scientific_notation() -> None:
    dispatcher = PriorityDispatcher(_make_router(), "paper")

    receipt = dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=5)

    assert "e" not in str(receipt.fill_price).lower()
    assert "e" not in str(receipt.fill_size).lower()


def test_guard_present_dry_run_allowed_records_clean_receipt() -> None:
    guard = DispatchGuard(
        DispatchGuardConfig(
            dedup_window_ms=100,
            max_dispatches_per_source_per_window=2,
            rate_window_ms=200,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_ms=300,
            max_open_positions_per_market=3,
        )
    )
    dispatcher = PriorityDispatcher(_make_router(), "dry_run", guard=guard)

    receipt = dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=10)

    assert receipt.executed is False
    assert receipt.guard_reason is None
    assert receipt.serialized_envelope


def test_guard_present_paper_allowed_records_clean_receipt() -> None:
    guard = DispatchGuard(
        DispatchGuardConfig(
            dedup_window_ms=100,
            max_dispatches_per_source_per_window=2,
            rate_window_ms=200,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_ms=300,
            max_open_positions_per_market=3,
        )
    )
    dispatcher = PriorityDispatcher(_make_router(), "paper", guard=guard)

    receipt = dispatcher.dispatch(_make_context(), dispatch_timestamp_ms=10)

    assert receipt.executed is True
    assert receipt.guard_reason is None
    assert receipt.fill_price == Decimal("0.640001")


def test_guard_present_rejection_returns_correct_receipt_shape() -> None:
    guard = DispatchGuard(
        DispatchGuardConfig(
            dedup_window_ms=100,
            max_dispatches_per_source_per_window=2,
            rate_window_ms=200,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_ms=300,
            max_open_positions_per_market=3,
        )
    )
    dispatcher = PriorityDispatcher(_make_router(), "dry_run", guard=guard)
    context = _make_context()

    first = dispatcher.dispatch(context, dispatch_timestamp_ms=10)
    second = dispatcher.dispatch(context, dispatch_timestamp_ms=20)

    assert first.guard_reason is None
    assert second.executed is False
    assert second.fill_price is None
    assert second.fill_size is None
    assert second.serialized_envelope == ""
    assert second.guard_reason == "DUPLICATE"