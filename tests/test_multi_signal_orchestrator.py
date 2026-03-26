from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.detectors.ctf_peg_config import CtfPegConfig
from src.events.mev_events import CtfMergeSignal
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.dispatch_guard import GuardDecision
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_signal_bridge import OfiSignalBridgeConfig
from src.execution.orchestrator_factory import build_paper_orchestrator
from src.execution.si9_paper_adapter import Si9PaperAdapterConfig
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.signal_coordination_bus import CoordinationBusConfig
from src.signals.si9_matrix_detector import Si9MatrixSignal


def _ctf_config() -> CtfPegConfig:
    return CtfPegConfig(
        min_yield=Decimal("0.050000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        slippage_budget=Decimal("0.005000"),
        gas_ewma_alpha=Decimal("0.500000"),
        max_desync_ms=400,
    )


def _ctf_adapter_config() -> CtfPaperAdapterConfig:
    return CtfPaperAdapterConfig(
        max_expected_net_edge=Decimal("0.250000"),
        max_capital_per_signal=Decimal("25.000000"),
        default_anchor_volume=Decimal("10.000000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        cancel_on_stale_ms=250,
        max_size_per_leg=Decimal("8.000000"),
        mode="paper",
        bus=None,
    )


def _si9_adapter_config() -> Si9PaperAdapterConfig:
    return Si9PaperAdapterConfig(
        max_expected_net_edge=Decimal("0.050000"),
        max_capital_per_cluster=Decimal("20.000000"),
        max_leg_fill_wait_ms=100,
        cancel_on_stale_ms=50,
        mode="paper",
        unwind_config=Si9UnwindConfig(
            market_sell_threshold=Decimal("0.040000"),
            passive_unwind_threshold=Decimal("0.010000"),
            max_hold_recovery_ms=100,
            min_best_bid=Decimal("0.010000"),
        ),
        bus=None,
    )


def _ofi_bridge_config() -> OfiSignalBridgeConfig:
    return OfiSignalBridgeConfig(
        max_capital_per_signal=Decimal("15.000000"),
        mode="paper",
        slot_side_lock=True,
        source_enabled=True,
    )


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


def _orchestrator_config(enabled_sources: frozenset[str] | None = None) -> OrchestratorConfig:
    return OrchestratorConfig(
        tick_interval_ms=50,
        max_pending_unwinds=4,
        max_concurrent_clusters=4,
        signal_sources_enabled=enabled_sources or frozenset({"CTF", "SI9"}),
    )


def _build_orchestrator(ask_proxy: dict[str, Decimal] | None = None, enabled_sources: frozenset[str] | None = None) -> MultiSignalOrchestrator:
    return build_paper_orchestrator(
        ctf_config=_ctf_config(),
        ctf_adapter_config=_ctf_adapter_config(),
        si9_cluster_configs=[("cluster-1", ["mkt-a", "mkt-b", "mkt-c"])],
        si9_adapter_config=_si9_adapter_config(),
        ofi_bridge_config=_ofi_bridge_config(),
        bus_config=_bus_config(),
        guard_config=_guard_config(),
        orchestrator_config=_orchestrator_config(enabled_sources),
        ask_proxy={} if ask_proxy is None else dict(ask_proxy),
    )


def _ctf_signal(market_id: str = "mkt-a") -> CtfMergeSignal:
    return CtfMergeSignal(
        market_id=market_id,
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        gas_estimate=Decimal("0.010000"),
        net_edge=Decimal("0.185000"),
    )


def _si9_signal(required_share_counts: Decimal = Decimal("2.500000")) -> Si9MatrixSignal:
    return Si9MatrixSignal(
        cluster_id="cluster-1",
        market_ids=("mkt-a", "mkt-b", "mkt-c"),
        best_yes_asks={"mkt-a": Decimal("0.180000"), "mkt-b": Decimal("0.190000"), "mkt-c": Decimal("0.200000")},
        ask_sizes={"mkt-a": required_share_counts, "mkt-b": required_share_counts, "mkt-c": required_share_counts},
        total_yes_ask=Decimal("0.570000"),
        gross_edge=Decimal("0.430000"),
        net_edge=Decimal("0.030000"),
        target_yield=Decimal("0.020000"),
        bottleneck_market_id="mkt-a",
        required_share_counts=required_share_counts,
    )


def test_factory_builds_runtime_orchestrator() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")})

    assert isinstance(orchestrator, MultiSignalOrchestrator)
    assert orchestrator.si9_detector is not None


def test_ctf_signal_dispatches_cleanly() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")})

    event = orchestrator.on_ctf_signal(_ctf_signal(), 1000)

    assert event.event_type == "CTF_DISPATCHED"
    assert event.payload["market_id"] == "mkt-a"


def test_ctf_source_disabled_rejects() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")}, frozenset({"SI9"}))

    event = orchestrator.on_ctf_signal(_ctf_signal(), 1000)

    assert event.event_type == "CTF_REJECTED"
    assert event.payload["reason"] == "SOURCE_DISABLED"


def test_si9_signal_dispatches_and_updates_snapshot() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19"), "mkt-b": Decimal("0.19"), "mkt-c": Decimal("0.19")})

    event = orchestrator.on_si9_signal(_si9_signal(), 1000)
    snapshot = orchestrator.orchestrator_snapshot(1001)

    assert event.event_type == "SI9_DISPATCHED"
    assert snapshot.active_position_count == 1


def test_hanging_leg_initiates_unwind_and_payload_serializes() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.18"), "mkt-b": Decimal("0.19"), "mkt-c": Decimal("0.20")})

    original_check = orchestrator.guard.check

    def guarded_check(context, current_timestamp_ms: int):
        if current_timestamp_ms == 1000 and context.market_id == "mkt-b":
            return GuardDecision(allowed=False, reason="RATE_EXCEEDED")
        return original_check(context, current_timestamp_ms)

    orchestrator.guard.check = guarded_check  # type: ignore[method-assign]

    event = orchestrator.on_si9_signal(_si9_signal(Decimal("30000.000000")), 1000)

    assert event.event_type == "UNWIND_INITIATED"
    json.dumps(event.payload)


def test_event_and_snapshot_contracts_are_frozen_and_on_tick_uses_injected_time() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")})
    event = orchestrator.on_ctf_signal(_ctf_signal(), 1000)
    snapshot = orchestrator.orchestrator_snapshot(1001)

    with pytest.raises(FrozenInstanceError):
        event.event_type = "CTF_REJECTED"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.health = "RED"  # type: ignore[misc]

    source = inspect.getsource(MultiSignalOrchestrator.on_tick)
    assert "time.time" not in source


