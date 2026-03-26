from __future__ import annotations

import pytest

from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus, SlotDecision


def _config(**overrides) -> CoordinationBusConfig:
    values = {
        "slot_lease_ms": 100,
        "max_slots_per_source": 2,
        "max_total_slots": 4,
        "allow_same_source_reentry": True,
    }
    values.update(overrides)
    return CoordinationBusConfig(**values)


def test_valid_config_construction_passes() -> None:
    config = _config()
    assert config.slot_lease_ms == 100


def test_max_total_slots_less_than_max_slots_per_source_raises_value_error() -> None:
    with pytest.raises(ValueError, match="max_total_slots"):
        _config(max_slots_per_source=3, max_total_slots=2)


def test_clean_bus_grants_slot() -> None:
    bus = SignalCoordinationBus(_config())
    decision = bus.request_slot("MKT_A", "YES", "OFI", 100)
    assert decision == SlotDecision(True, "MKT_A", "YES", "OFI", "GRANTED", None, 200)


def test_same_source_reentry_true_refreshes_slot() -> None:
    bus = SignalCoordinationBus(_config(allow_same_source_reentry=True))
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    decision = bus.request_slot("MKT_A", "YES", "OFI", 150)
    assert decision.granted is True
    assert decision.reason == "GRANTED"
    assert decision.lease_expires_ms == 250


def test_same_source_reentry_false_returns_slot_owned() -> None:
    bus = SignalCoordinationBus(_config(allow_same_source_reentry=False))
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    decision = bus.request_slot("MKT_A", "YES", "OFI", 150)
    assert decision.granted is False
    assert decision.reason == "SLOT_OWNED"
    assert decision.owner == "OFI"


def test_different_source_requests_owned_slot_within_ttl() -> None:
    bus = SignalCoordinationBus(_config())
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    decision = bus.request_slot("MKT_A", "YES", "CTF", 150)
    assert decision.granted is False
    assert decision.reason == "SLOT_OWNED"
    assert decision.owner == "OFI"


def test_different_source_requests_expired_slot_reclaims_it() -> None:
    bus = SignalCoordinationBus(_config(slot_lease_ms=100))
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    decision = bus.request_slot("MKT_A", "YES", "CTF", 201)
    assert decision.granted is True
    assert decision.reason == "EXPIRED_RECLAIMED"


def test_source_cap_denies_additional_slot() -> None:
    bus = SignalCoordinationBus(_config(max_slots_per_source=1, max_total_slots=4))
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    decision = bus.request_slot("MKT_B", "YES", "OFI", 110)
    assert decision.granted is False
    assert decision.reason == "SOURCE_CAP"


def test_bus_cap_denies_new_slot() -> None:
    bus = SignalCoordinationBus(_config(max_slots_per_source=2, max_total_slots=2))
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    bus.request_slot("MKT_B", "YES", "CTF", 110)
    decision = bus.request_slot("MKT_C", "YES", "SI9", 120)
    assert decision.granted is False
    assert decision.reason == "BUS_CAP"


def test_release_slot_frees_slot_immediately() -> None:
    bus = SignalCoordinationBus(_config())
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    bus.release_slot("MKT_A", "YES", "OFI", 120)
    decision = bus.request_slot("MKT_A", "YES", "CTF", 121)
    assert decision.granted is True


def test_bus_snapshot_evicts_expired_slots() -> None:
    bus = SignalCoordinationBus(_config(slot_lease_ms=100))
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    snapshot = bus.bus_snapshot(201)
    assert snapshot.total_active_slots == 0


def test_expired_reclaimed_count_is_monotonic() -> None:
    bus = SignalCoordinationBus(_config(slot_lease_ms=10))
    bus.request_slot("MKT_A", "YES", "OFI", 0)
    bus.request_slot("MKT_A", "YES", "CTF", 11)
    bus.request_slot("MKT_B", "YES", "CTF", 12)
    bus.request_slot("MKT_B", "YES", "OFI", 23)
    snapshot = bus.bus_snapshot(23)
    assert snapshot.expired_reclaimed_count == 2


def test_slot_ttl_boundary_exactly_valid_then_plus_one_expired() -> None:
    bus = SignalCoordinationBus(_config(slot_lease_ms=100))
    bus.request_slot("MKT_A", "YES", "OFI", 100)
    assert bus.request_slot("MKT_A", "YES", "CTF", 200).reason == "SLOT_OWNED"
    assert bus.request_slot("MKT_A", "YES", "CTF", 201).reason == "EXPIRED_RECLAIMED"


def test_two_sources_on_same_market_different_sides_both_granted() -> None:
    bus = SignalCoordinationBus(_config())
    first = bus.request_slot("MKT_A", "YES", "OFI", 100)
    second = bus.request_slot("MKT_A", "NO", "CTF", 100)
    assert first.granted is True
    assert second.granted is True


def test_snapshot_counts_consistent_after_mixed_sequence() -> None:
    bus = SignalCoordinationBus(_config(slot_lease_ms=50))
    bus.request_slot("MKT_A", "YES", "OFI", 0)
    bus.request_slot("MKT_B", "NO", "CTF", 10)
    bus.release_slot("MKT_B", "NO", "CTF", 20)
    snapshot = bus.bus_snapshot(60)
    assert snapshot.total_active_slots == 0
    assert snapshot.slots_by_source == {}
    assert snapshot.slots_by_market == {}
