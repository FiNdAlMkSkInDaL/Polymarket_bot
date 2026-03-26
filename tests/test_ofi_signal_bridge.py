from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.execution.dispatch_guard import DispatchGuard, GuardDecision
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.mev_router import MevExecutionRouter
from src.execution.ofi_paper_ledger import OfiPaperLedger
from src.execution.ofi_signal_bridge import OfiBridgeReceipt, OfiEntrySignal, OfiSignalBridge, OfiSignalBridgeConfig
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus


class RejectMarketGuard(DispatchGuard):
    def __init__(self, config: DispatchGuardConfig, rejected_market_ids: set[str] | None = None):
        super().__init__(config)
        self._rejected_market_ids = rejected_market_ids or set()

    def check(self, context, current_timestamp_ms: int):  # type: ignore[override]
        if context.market_id in self._rejected_market_ids:
            return GuardDecision(allowed=False, reason="RATE_EXCEEDED")
        return super().check(context, current_timestamp_ms)


def _snapshot_provider(market_id: str) -> dict[str, float]:
    _ = market_id
    return {
        "yes_bid": 0.44,
        "yes_ask": 0.46,
        "no_bid": 0.54,
        "no_ask": 0.56,
    }


def _bus_config() -> CoordinationBusConfig:
    return CoordinationBusConfig(
        slot_lease_ms=500,
        max_slots_per_source=10,
        max_total_slots=10,
        allow_same_source_reentry=False,
    )


def _guard_config() -> DispatchGuardConfig:
    return DispatchGuardConfig(
        dedup_window_ms=100,
        max_dispatches_per_source_per_window=10,
        rate_window_ms=200,
        circuit_breaker_threshold=2,
        circuit_breaker_reset_ms=300,
        max_open_positions_per_market=10,
    )


def _signal(*, market_id: str = "ofi-1", side: str = "NO", conviction_scalar: Decimal = Decimal("0.8")) -> OfiEntrySignal:
    return OfiEntrySignal(
        market_id=market_id,
        side=side,  # type: ignore[arg-type]
        target_price=Decimal("0.42"),
        anchor_volume=Decimal("10"),
        conviction_scalar=conviction_scalar,
        signal_timestamp_ms=1000,
        tvi_kappa=Decimal("1.25"),
        ofi_window_ms=2000,
    )


def _bridge(
    *,
    mode: str = "paper",
    slot_side_lock: bool = True,
    source_enabled: bool = True,
    guard: DispatchGuard | None = None,
) -> tuple[OfiSignalBridge, SignalCoordinationBus]:
    bus = SignalCoordinationBus(_bus_config())
    bridge = OfiSignalBridge(
        dispatcher=PriorityDispatcher(MevExecutionRouter(_snapshot_provider), mode),
        guard=guard or DispatchGuard(_guard_config()),
        bus=bus,
        ledger=OfiPaperLedger(),
        config=OfiSignalBridgeConfig(
            max_capital_per_signal=Decimal("15"),
            mode=mode,  # type: ignore[arg-type]
            slot_side_lock=slot_side_lock,
            source_enabled=source_enabled,
        ),
    )
    return bridge, bus


def test_valid_bridge_construction() -> None:
    bridge, _ = _bridge()

    assert isinstance(bridge, OfiSignalBridge)


def test_clean_dispatch_in_paper_mode() -> None:
    bridge, _ = _bridge(mode="paper")

    receipt = bridge.on_signal(_signal(), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "DISPATCHED"
    assert receipt.dispatch_receipt is not None
    assert receipt.dispatch_receipt.executed is True


def test_clean_dispatch_in_dry_run_mode() -> None:
    bridge, _ = _bridge(mode="dry_run")

    receipt = bridge.on_signal(_signal(), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "DISPATCHED"
    assert receipt.dispatch_receipt is not None
    assert receipt.dispatch_receipt.executed is False


def test_slot_side_lock_true_acquires_both_sides() -> None:
    bridge, bus = _bridge(slot_side_lock=True)

    bridge.on_signal(_signal(), Decimal("12"), 1000)

    assert bus.bus_snapshot(1000).slots_by_market["ofi-1"] == ["NO", "YES"]


def test_slot_side_lock_false_acquires_only_traded_side() -> None:
    bridge, bus = _bridge(slot_side_lock=False)

    bridge.on_signal(_signal(), Decimal("12"), 1000)

    assert bus.bus_snapshot(1000).slots_by_market["ofi-1"] == ["NO"]


def test_guard_rejection_returns_correct_bridge_outcome() -> None:
    bridge, bus = _bridge(guard=RejectMarketGuard(_guard_config(), {"ofi-1"}))

    receipt = bridge.on_signal(_signal(), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "GUARD_REJECTED"
    assert receipt.guard_decision is not None
    assert bus.bus_snapshot(1000).total_active_slots == 0


def test_bus_rejection_on_yes_slot_returns_bus_rejected() -> None:
    bridge, bus = _bridge(slot_side_lock=False)
    bus.request_slot("ofi-1", "NO", "CTF", 999)

    receipt = bridge.on_signal(_signal(), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "BUS_REJECTED"


def test_bus_rejection_on_no_slot_after_yes_granted_releases_yes_slot() -> None:
    bridge, bus = _bridge(slot_side_lock=True)
    bus.request_slot("ofi-1", "NO", "CTF", 999)

    receipt = bridge.on_signal(_signal(side="YES"), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "BUS_REJECTED"
    assert bus.owns_slot("ofi-1", "YES", "OFI", 1000) is False
    assert bus.bus_snapshot(1000).slots_by_market["ofi-1"] == ["NO"]


def test_source_disabled_gate_via_config() -> None:
    bridge, bus = _bridge(source_enabled=False)

    receipt = bridge.on_signal(_signal(), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "SOURCE_DISABLED"
    assert bus.bus_snapshot(1000).total_active_slots == 0


def test_ofi_bridge_receipt_is_frozen() -> None:
    receipt = OfiBridgeReceipt(
        signal=_signal(),
        dispatch_receipt=None,
        bridge_outcome="CAPITAL_ZERO",
        yes_slot=None,
        no_slot=None,
        guard_decision=None,
        timestamp_ms=1000,
    )

    with pytest.raises(FrozenInstanceError):
        receipt.bridge_outcome = "DISPATCHED"  # type: ignore[misc]


def test_dispatch_rate_is_zero_when_no_signals_received() -> None:
    bridge, _ = _bridge()

    assert bridge.ledger_snapshot().dispatch_rate == Decimal("0")


def test_mean_conviction_scalar_is_zero_when_no_dispatches() -> None:
    bridge, _ = _bridge()

    assert bridge.ledger_snapshot().mean_conviction_scalar == Decimal("0")


def test_conviction_scalar_zero_dispatches_with_zero_effective_size() -> None:
    bridge, _ = _bridge()

    receipt = bridge.on_signal(_signal(conviction_scalar=Decimal("0")), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "DISPATCHED"
    assert receipt.dispatch_receipt is not None
    assert receipt.dispatch_receipt.fill_size == Decimal("0.000000")


def test_conviction_scalar_one_dispatches_with_full_base_size() -> None:
    bridge, _ = _bridge()

    receipt = bridge.on_signal(_signal(conviction_scalar=Decimal("1")), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "DISPATCHED"
    assert receipt.dispatch_receipt is not None
    assert receipt.dispatch_receipt.fill_size == Decimal("10.000000")


def test_capital_zero_returns_capital_zero_outcome() -> None:
    bridge, bus = _bridge()

    receipt = bridge.on_signal(_signal(), Decimal("0"), 1000)

    assert receipt.bridge_outcome == "CAPITAL_ZERO"
    assert bus.bus_snapshot(1000).total_active_slots == 0


def test_slot_side_lock_releases_both_slots_on_guard_reject() -> None:
    bridge, bus = _bridge(slot_side_lock=True, guard=RejectMarketGuard(_guard_config(), {"ofi-1"}))

    receipt = bridge.on_signal(_signal(side="YES"), Decimal("12"), 1000)

    assert receipt.bridge_outcome == "GUARD_REJECTED"
    assert bus.bus_snapshot(1000).total_active_slots == 0


def test_dispatched_payload_is_json_serializable() -> None:
    bridge, _ = _bridge()

    receipt = bridge.on_signal(_signal(), Decimal("12"), 1000)

    json.dumps({
        "bridge_outcome": receipt.bridge_outcome,
        "signal": receipt.signal.market_id,
        "guard_reason": None if receipt.guard_decision is None else receipt.guard_decision.reason,
    })