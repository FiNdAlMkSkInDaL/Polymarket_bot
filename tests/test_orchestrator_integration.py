from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.detectors.ctf_peg_config import CtfPegConfig
from src.events.mev_events import CtfMergeSignal
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard import GuardDecision
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.multi_signal_orchestrator import OrchestratorConfig
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
        max_capital_per_signal=Decimal("5000.000000"),
        default_anchor_volume=Decimal("100.000000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        cancel_on_stale_ms=250,
        max_size_per_leg=Decimal("100.000000"),
        mode="paper",
        bus=None,
    )


def _si9_adapter_config() -> Si9PaperAdapterConfig:
    return Si9PaperAdapterConfig(
        max_expected_net_edge=Decimal("0.050000"),
        max_capital_per_cluster=Decimal("5000.000000"),
        max_leg_fill_wait_ms=100,
        cancel_on_stale_ms=50,
        mode="paper",
        unwind_config=Si9UnwindConfig(
            market_sell_threshold=Decimal("0.040000"),
            passive_unwind_threshold=Decimal("0.010000"),
            max_hold_recovery_ms=500,
            min_best_bid=Decimal("0.010000"),
        ),
        bus=None,
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
        dedup_window_ms=10,
        max_dispatches_per_source_per_window=100,
        rate_window_ms=100,
        circuit_breaker_threshold=2,
        circuit_breaker_reset_ms=300,
        max_open_positions_per_market=100,
    )


def _ofi_bridge_config() -> OfiSignalBridgeConfig:
    return OfiSignalBridgeConfig(
        max_capital_per_signal=Decimal("15.000000"),
        mode="paper",
        slot_side_lock=True,
        source_enabled=True,
    )


def _orchestrator_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        tick_interval_ms=100,
        max_pending_unwinds=4,
        max_concurrent_clusters=20,
        signal_sources_enabled=frozenset({"CTF", "SI9"}),
    )


def _ctf_signal(market_id: str) -> CtfMergeSignal:
    return CtfMergeSignal(
        market_id=market_id,
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        gas_estimate=Decimal("0.010000"),
        net_edge=Decimal("0.185000"),
    )


def _si9_signal(
    *,
    cluster_id: str,
    market_ids: tuple[str, str, str],
    bottleneck_market_id: str,
    required_share_counts: Decimal = Decimal("2.500000"),
) -> Si9MatrixSignal:
    return Si9MatrixSignal(
        cluster_id=cluster_id,
        market_ids=market_ids,
        best_yes_asks={
            market_ids[0]: Decimal("0.180000"),
            market_ids[1]: Decimal("0.190000"),
            market_ids[2]: Decimal("0.200000"),
        },
        ask_sizes={market_id: required_share_counts for market_id in market_ids},
        total_yes_ask=Decimal("0.570000"),
        gross_edge=Decimal("0.430000"),
        net_edge=Decimal("0.030000"),
        target_yield=Decimal("0.020000"),
        bottleneck_market_id=bottleneck_market_id,
        required_share_counts=required_share_counts,
    )


@pytest.fixture
def replay_results():
    ask_proxy = {
        **{f"ctf-{tick}": Decimal("0.380000") for tick in range(6, 11)},
        **{f"rctf-{tick}": Decimal("0.380000") for tick in range(26, 28)},
        **{f"m{tick}{suffix}": Decimal("0.19") for tick in range(11, 16) for suffix in ("a", "b", "c")},
        **{f"r{tick}{suffix}": Decimal("0.19") for tick in range(28, 31) for suffix in ("a", "b", "c")},
        "hang-a": Decimal("0.18"),
        "hang-b": Decimal("0.19"),
        "hang-c": Decimal("0.20"),
        "shared-market": Decimal("0.18"),
        "shared-b": Decimal("0.19"),
        "shared-c": Decimal("0.20"),
    }
    orchestrator = build_paper_orchestrator(
        ctf_config=_ctf_config(),
        ctf_adapter_config=_ctf_adapter_config(),
        si9_cluster_configs=[
            ("cluster-1", ["m11a", "m11b", "m11c"]),
            ("cluster-2", ["m12a", "m12b", "m12c"]),
        ],
        si9_adapter_config=_si9_adapter_config(),
        ofi_bridge_config=_ofi_bridge_config(),
        bus_config=_bus_config(),
        guard_config=_guard_config(),
        orchestrator_config=_orchestrator_config(),
        ask_proxy=ask_proxy,
    )
    original_check = orchestrator.guard.check

    def guarded_check(context, current_timestamp_ms: int):
        if current_timestamp_ms == 1600 and context.market_id == "hang-b":
            return GuardDecision(allowed=False, reason="RATE_EXCEEDED")
        return original_check(context, current_timestamp_ms)

    orchestrator.guard.check = guarded_check  # type: ignore[method-assign]
    events_by_tick: dict[int, list] = {}
    all_events: list = []

    for tick in range(1, 6):
        tick_events = orchestrator.on_tick(tick * 100)
        events_by_tick[tick] = tick_events
        all_events.extend(tick_events)

    for tick in range(6, 11):
        event = orchestrator.on_ctf_signal(_ctf_signal(f"ctf-{tick}"), tick * 100)
        events_by_tick[tick] = [event]
        all_events.append(event)

    for tick in range(11, 16):
        signal = _si9_signal(
            cluster_id=f"cluster-{tick}",
            market_ids=(f"m{tick}a", f"m{tick}b", f"m{tick}c"),
            bottleneck_market_id=f"m{tick}a",
        )
        event = orchestrator.on_si9_signal(signal, tick * 100)
        events_by_tick[tick] = [event]
        all_events.append(event)

    hanging_event = orchestrator.on_si9_signal(
        _si9_signal(
            cluster_id="cluster-hang",
            market_ids=("hang-a", "hang-b", "hang-c"),
            bottleneck_market_id="hang-a",
            required_share_counts=Decimal("30000.000000"),
        ),
        1600,
    )
    events_by_tick[16] = [hanging_event]
    all_events.append(hanging_event)

    for tick in range(17, 21):
        tick_events = orchestrator.on_tick(tick * 100)
        events_by_tick[tick] = tick_events
        all_events.extend(tick_events)

    tick_21_events = orchestrator.on_tick(2101)
    events_by_tick[21] = tick_21_events
    all_events.extend(tick_21_events)

    for tick in range(22, 26):
        timestamp_ms = tick * 100
        orchestrator.bus.request_slot("shared-market", "YES", "CTF", timestamp_ms)
        event = orchestrator.on_si9_signal(
            _si9_signal(
                cluster_id=f"cluster-shared-{tick}",
                market_ids=("shared-market", "shared-b", "shared-c"),
                bottleneck_market_id="shared-market",
            ),
            timestamp_ms,
        )
        orchestrator.bus.release_slot("shared-market", "YES", "CTF", timestamp_ms)
        events_by_tick[tick] = [event]
        all_events.append(event)

    for tick in range(26, 28):
        event = orchestrator.on_ctf_signal(_ctf_signal(f"rctf-{tick}"), tick * 100)
        events_by_tick[tick] = [event]
        all_events.append(event)

    for tick in range(28, 31):
        signal = _si9_signal(
            cluster_id=f"cluster-recovery-{tick}",
            market_ids=(f"r{tick}a", f"r{tick}b", f"r{tick}c"),
            bottleneck_market_id=f"r{tick}a",
        )
        event = orchestrator.on_si9_signal(signal, tick * 100)
        events_by_tick[tick] = [event]
        all_events.append(event)

    return {
        "orchestrator": orchestrator,
        "events_by_tick": events_by_tick,
        "all_events": all_events,
        "final_snapshot": orchestrator.orchestrator_snapshot(3000),
    }


def test_warmup_ticks_return_empty_lists(replay_results) -> None:
    for tick in range(1, 6):
        assert replay_results["events_by_tick"][tick] == []


def test_ctf_dislocation_window_records_expected_dispatched_count(replay_results) -> None:
    ctf_dispatched = [
        event
        for tick in range(6, 11)
        for event in replay_results["events_by_tick"][tick]
        if event.event_type == "CTF_DISPATCHED"
    ]

    assert len(ctf_dispatched) == 5


def test_si9_cluster_compression_window_records_expected_dispatches(replay_results) -> None:
    si9_dispatched = [
        event
        for tick in range(11, 16)
        for event in replay_results["events_by_tick"][tick]
        if event.event_type == "SI9_DISPATCHED"
    ]

    assert len(si9_dispatched) == 5


def test_tick_16_records_exactly_one_unwind_initiated(replay_results) -> None:
    events = replay_results["events_by_tick"][16]

    assert len([event for event in events if event.event_type == "UNWIND_INITIATED"]) == 1


def test_ticks_17_to_20_do_not_escalate_before_threshold(replay_results) -> None:
    assert all(
        all(event.event_type != "UNWIND_ESCALATED" for event in replay_results["events_by_tick"][tick])
        for tick in range(17, 21)
    )


def test_tick_21_emits_unwind_escalated_once(replay_results) -> None:
    events = replay_results["events_by_tick"][21]

    assert len([event for event in events if event.event_type == "UNWIND_ESCALATED"]) == 1


def test_ticks_22_to_25_bus_block_second_signal(replay_results) -> None:
    blocked = [
        event
        for tick in range(22, 26)
        for event in replay_results["events_by_tick"][tick]
        if event.event_type == "SI9_REJECTED" and event.payload.get("cluster_outcome") == "BUS_REJECTED"
    ]

    assert len(blocked) == 4


def test_final_counts_match_expected_totals(replay_results) -> None:
    ctf_dispatched = [event for event in replay_results["all_events"] if event.event_type == "CTF_DISPATCHED"]
    si9_dispatched = [event for event in replay_results["all_events"] if event.event_type == "SI9_DISPATCHED"]
    unwind_initiated = [event for event in replay_results["all_events"] if event.event_type == "UNWIND_INITIATED"]
    unwind_escalated = [event for event in replay_results["all_events"] if event.event_type == "UNWIND_ESCALATED"]

    assert len(ctf_dispatched) == 7
    assert len(si9_dispatched) == 8
    assert len(unwind_initiated) == 1
    assert len(unwind_escalated) == 1


def test_final_snapshot_clears_pending_unwind_and_reports_green_health(replay_results) -> None:
    snapshot = replay_results["final_snapshot"]

    assert snapshot.pending_unwind_count == 0
    assert snapshot.health == "GREEN"


def test_final_ledgers_capture_positive_edge_and_all_events_serialize(replay_results) -> None:
    snapshot = replay_results["final_snapshot"]

    assert snapshot.ctf_ledger.gross_edge_captured > Decimal("0")
    assert snapshot.si9_ledger.gross_edge_captured > Decimal("0")
    for event in replay_results["all_events"]:
        json.dumps(event.payload)