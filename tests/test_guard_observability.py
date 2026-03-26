from __future__ import annotations

from decimal import Decimal

from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.guard_observability import GuardObservabilityPanel
from src.execution.priority_context import PriorityOrderContext
from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus


def _guard(threshold: int = 4) -> DispatchGuard:
    return DispatchGuard(
        DispatchGuardConfig(
            dedup_window_ms=100,
            max_dispatches_per_source_per_window=2,
            rate_window_ms=200,
            circuit_breaker_threshold=threshold,
            circuit_breaker_reset_ms=300,
            max_open_positions_per_market=2,
        )
    )


def _context(source: str = "OFI") -> PriorityOrderContext:
    return PriorityOrderContext(
        market_id="MKT_A",
        side="YES",
        signal_source=source,
        conviction_scalar=Decimal("0.5"),
        target_price=Decimal("0.64"),
        anchor_volume=Decimal("10"),
        max_capital=Decimal("20"),
    )


def _bus() -> SignalCoordinationBus:
    return SignalCoordinationBus(
        CoordinationBusConfig(
            slot_lease_ms=100,
            max_slots_per_source=3,
            max_total_slots=5,
            allow_same_source_reentry=True,
        )
    )


def test_full_snapshot_green_when_all_guards_clean() -> None:
    panel = GuardObservabilityPanel({"OFI": _guard()}, _bus())
    snapshot = panel.full_snapshot(100)
    assert snapshot.system_health == "GREEN"


def test_system_health_red_when_any_guard_open() -> None:
    guard = _guard(threshold=2)
    guard.check(_context("OFI"), 10)
    guard.record_suppression("OFI")
    guard.record_suppression("OFI")
    panel = GuardObservabilityPanel({"OFI": guard}, _bus())
    snapshot = panel.full_snapshot(10)
    assert snapshot.system_health == "RED"


def test_system_health_yellow_when_half_threshold_crossed() -> None:
    guard = _guard(threshold=4)
    guard.record_suppression("OFI")
    guard.record_suppression("OFI")
    panel = GuardObservabilityPanel({"OFI": guard}, _bus())
    snapshot = panel.full_snapshot(10)
    assert snapshot.system_health == "YELLOW"


def test_highest_suppression_source_identifies_correct_source() -> None:
    ofi = _guard()
    ctf = _guard()
    ofi.record_suppression("OFI")
    ctf.record_suppression("CTF")
    ctf.record_suppression("CTF")
    panel = GuardObservabilityPanel({"OFI": ofi, "CTF": ctf}, _bus())
    snapshot = panel.full_snapshot(10)
    assert snapshot.highest_suppression_source == "CTF"


def test_suppression_report_sorted_descending() -> None:
    ofi = _guard()
    ctf = _guard()
    si9 = _guard()
    ctf.record_suppression("CTF")
    ctf.record_suppression("CTF")
    ofi.record_suppression("OFI")
    panel = GuardObservabilityPanel({"OFI": ofi, "CTF": ctf, "SI9": si9}, _bus())
    report = panel.suppression_report(10)
    assert [entry.signal_source for entry in report] == ["CTF", "OFI", "SI9"]


def test_total_circuit_opens_counts_correctly() -> None:
    ofi = _guard(threshold=2)
    ctf = _guard(threshold=2)
    ofi.check(_context("OFI"), 10)
    ofi.record_suppression("OFI")
    ofi.record_suppression("OFI")
    panel = GuardObservabilityPanel({"OFI": ofi, "CTF": ctf}, _bus())
    assert panel.full_snapshot(10).total_circuit_opens == 1


def test_total_active_slots_matches_bus_snapshot() -> None:
    bus = _bus()
    bus.request_slot("MKT_A", "YES", "OFI", 10)
    bus.request_slot("MKT_B", "NO", "CTF", 10)
    panel = GuardObservabilityPanel({"OFI": _guard()}, bus)
    snapshot = panel.full_snapshot(10)
    assert snapshot.total_active_slots == snapshot.bus_snapshot.total_active_slots == 2


def test_panel_handles_single_and_three_guard_configs() -> None:
    single = GuardObservabilityPanel({"OFI": _guard()}, _bus()).full_snapshot(10)
    multi = GuardObservabilityPanel(
        {"OFI": _guard(), "CTF": _guard(), "SI9": _guard()},
        _bus(),
    ).full_snapshot(10)
    assert set(single.per_source.keys()) == {"OFI"}
    assert set(multi.per_source.keys()) == {"OFI", "CTF", "SI9"}
