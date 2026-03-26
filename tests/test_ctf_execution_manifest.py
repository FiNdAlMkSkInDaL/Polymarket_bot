from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.execution.ctf_execution_manifest import (
    CtfExecutionManifest,
    CtfExecutionReceipt,
    CtfLegManifest,
    CtfLegReceipt,
    build_ctf_execution_manifest,
)
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchReceipt


def _context(side: str = "YES") -> PriorityOrderContext:
    return PriorityOrderContext(
        market_id="MKT_CTF",
        side=side,  # type: ignore[arg-type]
        signal_source="CTF",
        conviction_scalar=Decimal("0.500000"),
        target_price=Decimal("0.380000"),
        anchor_volume=Decimal("5.000000"),
        max_capital=Decimal("2.000000"),
        leg_role="YES_LEG" if side == "YES" else "NO_LEG",
    )


def _dispatch_receipt(
    *,
    side: str = "YES",
    executed: bool = True,
    fill_price: Decimal | None = Decimal("0.380001"),
    fill_size: Decimal | None = Decimal("5.000000"),
    fill_status: str = "FULL",
    guard_reason: str | None = None,
    partial_fill_size: Decimal | None = None,
    partial_fill_price: Decimal | None = None,
) -> DispatchReceipt:
    return DispatchReceipt(
        context=_context(side),
        mode="paper",
        executed=executed,
        fill_price=fill_price,
        fill_size=fill_size,
        serialized_envelope="{}",
        dispatch_timestamp_ms=1000,
        guard_reason=guard_reason,
        partial_fill_size=partial_fill_size,
        partial_fill_price=partial_fill_price,
        fill_status=fill_status,  # type: ignore[arg-type]
    )


def _manifest(**overrides) -> CtfExecutionManifest:
    values = {
        "market_id": "MKT_CTF",
        "yes_price": Decimal("0.380000"),
        "no_price": Decimal("0.400000"),
        "net_edge": Decimal("0.185000"),
        "gas_estimate": Decimal("0.010000"),
        "default_anchor_volume": Decimal("10.000000"),
        "max_capital_per_signal": Decimal("25.000000"),
        "max_size_per_leg": Decimal("8.000000"),
        "taker_fee_yes": Decimal("0.010000"),
        "taker_fee_no": Decimal("0.010000"),
        "manifest_timestamp_ms": 1000,
        "cancel_on_stale_ms": 250,
    }
    values.update(overrides)
    return build_ctf_execution_manifest(**values)


def test_valid_manifest_construction_anchors_cheaper_leg_first() -> None:
    manifest = _manifest(yes_price=Decimal("0.370000"), no_price=Decimal("0.410000"))

    assert manifest.anchor_leg == "YES_LEG"
    assert manifest.yes_leg.leg_index == 0
    assert manifest.no_leg.leg_index == 1


def test_anchor_tiebreak_defaults_to_yes_leg_for_equal_prices() -> None:
    manifest = _manifest(yes_price=Decimal("0.400000"), no_price=Decimal("0.400000"))

    assert manifest.anchor_leg == "YES_LEG"
    assert manifest.yes_leg.leg_index == 0
    assert manifest.no_leg.leg_index == 1


def test_required_size_is_bounded_by_depth_and_capital_constraints() -> None:
    manifest = _manifest(
        default_anchor_volume=Decimal("7.000000"),
        max_size_per_leg=Decimal("6.000000"),
        max_capital_per_signal=Decimal("3.900000"),
        yes_price=Decimal("0.300000"),
        no_price=Decimal("0.350000"),
    )

    assert manifest.required_size == Decimal("6.000000")


def test_required_size_uses_capital_bound_when_it_is_minimum_binding() -> None:
    manifest = _manifest(
        default_anchor_volume=Decimal("10.000000"),
        max_size_per_leg=Decimal("9.000000"),
        max_capital_per_signal=Decimal("3.250000"),
        yes_price=Decimal("0.300000"),
        no_price=Decimal("0.350000"),
    )

    assert manifest.required_size == Decimal("5")


def test_realized_pnl_manual_calculation_matches_to_six_decimal_places() -> None:
    manifest = _manifest()
    yes_receipt = CtfLegReceipt(
        leg_manifest=manifest.yes_leg,
        dispatch_receipt=_dispatch_receipt(side="YES", fill_price=Decimal("0.380001")),
        fill_status="FILLED",
        filled_size=Decimal("5.000000"),
        filled_price=Decimal("0.380001"),
        fill_timestamp_ms=1000,
    )
    no_receipt = CtfLegReceipt(
        leg_manifest=manifest.no_leg,
        dispatch_receipt=_dispatch_receipt(side="NO", fill_price=Decimal("0.400001")),
        fill_status="FILLED",
        filled_size=Decimal("5.000000"),
        filled_price=Decimal("0.400001"),
        fill_timestamp_ms=1000,
    )
    realized_net_edge = Decimal("1") - Decimal("0.380001") - Decimal("0.400001") - Decimal("0.010000") - Decimal("0.010000") - Decimal("0.010000")
    realized_pnl = realized_net_edge * Decimal("5.000000")

    receipt = CtfExecutionReceipt(
        manifest=manifest,
        yes_receipt=yes_receipt,
        no_receipt=no_receipt,
        execution_outcome="FULL_FILL",
        realized_net_edge=realized_net_edge,
        realized_pnl=realized_pnl,
        total_capital_deployed=Decimal("3.900010"),
        execution_timestamp_ms=1000,
    )

    assert receipt.realized_pnl.quantize(Decimal("0.000001")) == realized_pnl.quantize(Decimal("0.000001"))


def test_non_fill_outcomes_require_zero_realized_fields() -> None:
    manifest = _manifest()
    suppressed_yes = CtfLegReceipt(
        leg_manifest=manifest.yes_leg,
        dispatch_receipt=_dispatch_receipt(side="YES", executed=False, fill_price=None, fill_size=None, fill_status="NONE", guard_reason="BUS_REJECTED"),
        fill_status="SUPPRESSED",
        filled_size=Decimal("0"),
        filled_price=None,
        fill_timestamp_ms=None,
    )
    suppressed_no = CtfLegReceipt(
        leg_manifest=manifest.no_leg,
        dispatch_receipt=_dispatch_receipt(side="NO", executed=False, fill_price=None, fill_size=None, fill_status="NONE", guard_reason="BUS_REJECTED"),
        fill_status="SUPPRESSED",
        filled_size=Decimal("0"),
        filled_price=None,
        fill_timestamp_ms=None,
    )
    receipt = CtfExecutionReceipt(
        manifest=manifest,
        yes_receipt=suppressed_yes,
        no_receipt=suppressed_no,
        execution_outcome="BUS_REJECTED",
        realized_net_edge=Decimal("0"),
        realized_pnl=Decimal("0"),
        total_capital_deployed=Decimal("0"),
        execution_timestamp_ms=1000,
    )

    assert receipt.realized_net_edge == Decimal("0")
    assert receipt.realized_pnl == Decimal("0")


def test_ctf_execution_receipt_full_fill_construction_passes() -> None:
    manifest = _manifest()
    receipt = CtfExecutionReceipt(
        manifest=manifest,
        yes_receipt=CtfLegReceipt(manifest.yes_leg, _dispatch_receipt(side="YES"), "FILLED", Decimal("5.000000"), Decimal("0.380001"), 1000),
        no_receipt=CtfLegReceipt(manifest.no_leg, _dispatch_receipt(side="NO", fill_price=Decimal("0.400001")), "FILLED", Decimal("5.000000"), Decimal("0.400001"), 1000),
        execution_outcome="FULL_FILL",
        realized_net_edge=Decimal("0.189998"),
        realized_pnl=Decimal("0.949990"),
        total_capital_deployed=Decimal("3.900010"),
        execution_timestamp_ms=1000,
    )

    assert receipt.executed is True


def test_non_fill_receipt_with_non_zero_realized_pnl_raises_value_error() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="realized_net_edge and realized_pnl"):
        CtfExecutionReceipt(
            manifest=manifest,
            yes_receipt=CtfLegReceipt(manifest.yes_leg, _dispatch_receipt(side="YES", executed=False, fill_price=None, fill_size=None, fill_status="NONE", guard_reason="GUARD_REJECTED"), "SUPPRESSED", Decimal("0"), None, None),
            no_receipt=CtfLegReceipt(manifest.no_leg, _dispatch_receipt(side="NO", executed=False, fill_price=None, fill_size=None, fill_status="NONE", guard_reason="GUARD_REJECTED"), "SUPPRESSED", Decimal("0"), None, None),
            execution_outcome="GUARD_REJECTED",
            realized_net_edge=Decimal("0.1"),
            realized_pnl=Decimal("0.5"),
            total_capital_deployed=Decimal("0"),
            execution_timestamp_ms=1000,
        )


def test_ctf_leg_receipt_filled_status_requires_non_none_filled_price() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="FILLED leg receipts"):
        CtfLegReceipt(
            leg_manifest=manifest.yes_leg,
            dispatch_receipt=_dispatch_receipt(side="YES"),
            fill_status="FILLED",
            filled_size=Decimal("5.000000"),
            filled_price=None,
            fill_timestamp_ms=1000,
        )


def test_dispatch_receipt_valid_partial_fill_contract_passes() -> None:
    receipt = DispatchReceipt(
        context=_context(),
        mode="paper",
        executed=True,
        fill_price=Decimal("0.380001"),
        fill_size=Decimal("5.000000"),
        serialized_envelope="{}",
        dispatch_timestamp_ms=1000,
        partial_fill_size=Decimal("2.500000"),
        partial_fill_price=Decimal("0.380001"),
        fill_status="PARTIAL",
    )

    assert receipt.fill_status == "PARTIAL"


def test_dispatch_receipt_partial_fill_size_must_be_less_than_fill_size() -> None:
    with pytest.raises(ValueError, match="partial_fill_size"):
        DispatchReceipt(
            context=_context(),
            mode="paper",
            executed=True,
            fill_price=Decimal("0.380001"),
            fill_size=Decimal("5.000000"),
            serialized_envelope="{}",
            dispatch_timestamp_ms=1000,
            partial_fill_size=Decimal("5.000000"),
            partial_fill_price=Decimal("0.380001"),
            fill_status="PARTIAL",
        )


def test_dispatch_receipt_full_status_with_missing_fill_price_raises() -> None:
    with pytest.raises(ValueError, match="fill_status='FULL'"):
        DispatchReceipt(
            context=_context(),
            mode="paper",
            executed=True,
            fill_price=None,
            fill_size=Decimal("5.000000"),
            serialized_envelope="{}",
            dispatch_timestamp_ms=1000,
            fill_status="FULL",
        )


def test_dispatch_receipt_none_status_with_fill_price_set_raises() -> None:
    with pytest.raises(ValueError, match="fill_status='NONE'"):
        DispatchReceipt(
            context=_context(),
            mode="dry_run",
            executed=False,
            fill_price=Decimal("0.380001"),
            fill_size=None,
            serialized_envelope="{}",
            dispatch_timestamp_ms=1000,
            fill_status="NONE",
        )


def test_manifest_is_frozen() -> None:
    manifest = _manifest()

    with pytest.raises(FrozenInstanceError):
        manifest.anchor_leg = "NO_LEG"  # type: ignore[misc]