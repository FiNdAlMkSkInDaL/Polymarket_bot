from __future__ import annotations

from decimal import Decimal
import inspect

from src.detectors import CtfPegConfig
from src.execution.ctf_paper_adapter import CtfPaperAdapter, CtfPaperAdapterConfig
from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.priority_context import PriorityOrderContext
from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus
from src.signals.ctf_peg_detector import CtfPegDetector


def _config() -> CtfPegConfig:
    return CtfPegConfig(
        min_yield=Decimal("0.050000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        slippage_budget=Decimal("0.005000"),
        gas_ewma_alpha=Decimal("0.500000"),
        max_desync_ms=100,
    )


def _build_ticks() -> list[dict[str, object]]:
    ticks: list[dict[str, object]] = []

    for index in range(8):
        ticks.append({
            "phase": "warmup",
            "yes_ask": Decimal("0.480000"),
            "no_ask": Decimal("0.470000"),
            "yes_timestamp_ms": index * 100,
            "no_timestamp_ms": index * 100 + 20,
            "base_fee": Decimal("0.030000"),
        })

    for index in range(8, 18):
        ticks.append({
            "phase": "dislocation",
            "yes_ask": Decimal("0.380000"),
            "no_ask": Decimal("0.400000"),
            "yes_timestamp_ms": index * 100,
            "no_timestamp_ms": index * 100 + 10,
            "base_fee": Decimal("0.010000"),
        })

    for index in range(18, 24):
        ticks.append({
            "phase": "desync",
            "yes_ask": Decimal("0.380000"),
            "no_ask": Decimal("0.400000"),
            "yes_timestamp_ms": index * 100,
            "no_timestamp_ms": index * 100 + 250,
            "base_fee": Decimal("0.010000"),
        })

    for index in range(24, 32):
        ticks.append({
            "phase": "gas_spike",
            "yes_ask": Decimal("0.380000"),
            "no_ask": Decimal("0.400000"),
            "yes_timestamp_ms": index * 100,
            "no_timestamp_ms": index * 100 + 5,
            "base_fee": Decimal("0.320000"),
        })

    for index in range(32, 40):
        ticks.append({
            "phase": "recovery",
            "yes_ask": Decimal("0.470000"),
            "no_ask": Decimal("0.470000"),
            "yes_timestamp_ms": index * 100,
            "no_timestamp_ms": index * 100 + 10,
            "base_fee": Decimal("0.020000"),
        })

    return ticks


def _run_replay() -> tuple[dict[str, int], list[object], CtfPegDetector]:
    detector = CtfPegDetector("MKT_REPLAY", _config())
    counts = {phase: 0 for phase in ("warmup", "dislocation", "desync", "gas_spike", "recovery")}
    signals: list[object] = []

    for tick in _build_ticks():
        signal = detector.evaluate(
            yes_ask=tick["yes_ask"],
            no_ask=tick["no_ask"],
            yes_timestamp_ms=tick["yes_timestamp_ms"],
            no_timestamp_ms=tick["no_timestamp_ms"],
            base_fee=tick["base_fee"],
        )
        if signal is not None:
            counts[tick["phase"]] += 1
            signals.append(signal)

    return counts, signals, detector


def _adapter_config() -> CtfPaperAdapterConfig:
    return CtfPaperAdapterConfig(
        max_expected_net_edge=Decimal("0.250000"),
        max_capital_per_signal=Decimal("25.000000"),
        default_anchor_volume=Decimal("10.000000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        cancel_on_stale_ms=250,
        max_size_per_leg=Decimal("8.000000"),
        mode="paper",
        bus=SignalCoordinationBus(_bus_config()),
    )


def _guard_config() -> DispatchGuardConfig:
    return DispatchGuardConfig(
        dedup_window_ms=100,
        max_dispatches_per_source_per_window=10,
        rate_window_ms=1000,
        circuit_breaker_threshold=2,
        circuit_breaker_reset_ms=300,
        max_open_positions_per_market=50,
    )


class _ReplayDispatcher:
    def __init__(self, reject_second_leg_at_call: int | None = None):
        self._mode = "paper"
        self._guard = None
        self._reject_second_leg_at_call = reject_second_leg_at_call
        self.calls: list[PriorityOrderContext] = []

    def dispatch(self, context: PriorityOrderContext, dispatch_timestamp_ms: int) -> DispatchReceipt:
        self.calls.append(context)
        call_index = len(self.calls)
        if self._reject_second_leg_at_call is not None and call_index == self._reject_second_leg_at_call:
            return DispatchReceipt(
                context=context,
                mode="paper",
                executed=False,
                fill_price=None,
                fill_size=None,
                serialized_envelope="{}",
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                fill_status="NONE",
            )
        fill_price = (context.target_price + Decimal("0.000001")).quantize(Decimal("0.000001"))
        fill_size = (context.anchor_volume * context.conviction_scalar).quantize(Decimal("0.000001"))
        return DispatchReceipt(
            context=context,
            mode="paper",
            executed=True,
            fill_price=fill_price,
            fill_size=fill_size,
            serialized_envelope="{}",
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            fill_status="FULL",
        )


def _bus_config() -> CoordinationBusConfig:
    return CoordinationBusConfig(
        slot_lease_ms=500,
        max_slots_per_source=4,
        max_total_slots=8,
        allow_same_source_reentry=False,
    )


def _build_adapter_replay_ticks() -> list[dict[str, object]]:
    ticks: list[dict[str, object]] = []

    for index in range(5):
        ticks.append({
            "yes_ask": Decimal("0.480000"),
            "no_ask": Decimal("0.470000"),
            "yes_timestamp_ms": index * 50,
            "no_timestamp_ms": index * 50 + 10,
            "base_fee": Decimal("0.030000"),
        })

    signal_times = [300, 340, 380, 520, 700, 740, 880, 1020]
    for timestamp_ms in signal_times:
        ticks.append({
            "yes_ask": Decimal("0.380000"),
            "no_ask": Decimal("0.400000"),
            "yes_timestamp_ms": timestamp_ms,
            "no_timestamp_ms": timestamp_ms + 10,
            "base_fee": Decimal("0.010000"),
        })

    for index in range(7):
        base_timestamp = 1200 + index * 60
        ticks.append({
            "yes_ask": Decimal("0.470000"),
            "no_ask": Decimal("0.470000"),
            "yes_timestamp_ms": base_timestamp,
            "no_timestamp_ms": base_timestamp + 10,
            "base_fee": Decimal("0.020000"),
        })

    return ticks


def _run_adapter_wired_replay() -> tuple[int, int, CtfPaperAdapter, list[object]]:
    detector = CtfPegDetector("MKT_ADAPTER_REPLAY", _config())
    dispatcher = _ReplayDispatcher()
    adapter = CtfPaperAdapter(dispatcher, DispatchGuard(_guard_config()), _adapter_config())
    emitted_signals = 0
    adapter_receipts: list[object] = []

    for tick in _build_adapter_replay_ticks():
        signal = detector.evaluate(
            yes_ask=tick["yes_ask"],
            no_ask=tick["no_ask"],
            yes_timestamp_ms=tick["yes_timestamp_ms"],
            no_timestamp_ms=tick["no_timestamp_ms"],
            base_fee=tick["base_fee"],
        )
        if signal is None:
            continue
        emitted_signals += 1
        adapter_receipts.append(adapter.on_signal(signal, current_timestamp_ms=tick["yes_timestamp_ms"]))

    return emitted_signals, adapter.ledger_snapshot().total_dispatched, adapter, adapter_receipts


def _build_two_leg_replay_ticks() -> list[dict[str, object]]:
    ticks: list[dict[str, object]] = []

    for index in range(4):
        ticks.append({
            "yes_ask": Decimal("0.490000"),
            "no_ask": Decimal("0.470000"),
            "yes_timestamp_ms": index * 100,
            "no_timestamp_ms": index * 100 + 10,
            "base_fee": Decimal("0.030000"),
        })

    for index, timestamp_ms in enumerate(range(400, 1000, 100), start=1):
        ticks.append({
            "yes_ask": Decimal("0.380000"),
            "no_ask": Decimal("0.400000"),
            "yes_timestamp_ms": timestamp_ms,
            "no_timestamp_ms": timestamp_ms + 10,
            "base_fee": Decimal("0.010000"),
            "force_second_leg_reject": timestamp_ms == 700,
        })

    for index in range(5):
        ticks.append({
            "yes_ask": Decimal("0.470000"),
            "no_ask": Decimal("0.470000"),
            "yes_timestamp_ms": 1100 + index * 100,
            "no_timestamp_ms": 1110 + index * 100,
            "base_fee": Decimal("0.020000"),
        })

    return ticks


def _run_two_leg_replay() -> tuple[CtfPaperAdapter, list[object], int]:
    detector = CtfPegDetector("MKT_TWO_LEG_REPLAY", _config())
    dispatcher = _ReplayDispatcher(reject_second_leg_at_call=8)
    adapter = CtfPaperAdapter(dispatcher, DispatchGuard(_guard_config()), _adapter_config())
    emitted_signals = 0
    receipts: list[object] = []

    for tick in _build_two_leg_replay_ticks():
        signal = detector.evaluate(
            yes_ask=tick["yes_ask"],
            no_ask=tick["no_ask"],
            yes_timestamp_ms=tick["yes_timestamp_ms"],
            no_timestamp_ms=tick["no_timestamp_ms"],
            base_fee=tick["base_fee"],
        )
        if signal is None:
            continue
        emitted_signals += 1
        receipts.append(adapter.on_signal(signal, current_timestamp_ms=tick["yes_timestamp_ms"]))

    return adapter, receipts, emitted_signals


def test_replay_total_tick_count_is_deterministic() -> None:
    assert len(_build_ticks()) == 40


def test_replay_warmup_phase_emits_zero_signals() -> None:
    counts, _, _ = _run_replay()
    assert counts["warmup"] == 0


def test_replay_dislocation_phase_emits_exact_signal_count() -> None:
    counts, _, _ = _run_replay()
    assert counts["dislocation"] == 10


def test_replay_desync_phase_suppresses_signals_exactly() -> None:
    counts, _, _ = _run_replay()
    assert counts["desync"] == 0


def test_replay_gas_spike_phase_suppresses_signals_exactly() -> None:
    counts, _, _ = _run_replay()
    assert counts["gas_spike"] == 0


def test_replay_recovery_phase_emits_zero_signals() -> None:
    counts, _, _ = _run_replay()
    assert counts["recovery"] == 0


def test_replay_per_phase_counts_match_expected_map() -> None:
    counts, _, _ = _run_replay()
    assert counts == {
        "warmup": 0,
        "dislocation": 10,
        "desync": 0,
        "gas_spike": 0,
        "recovery": 0,
    }


def test_replay_final_detector_state_remains_post_spike_and_recovered() -> None:
    _, signals, detector = _run_replay()
    assert len(signals) == 10
    assert detector.state.yes_ask == Decimal("0.470000")
    assert detector.state.no_ask == Decimal("0.470000")
    assert detector.state.gas_estimate > Decimal("0.020000")


def test_adapter_replay_tick_count_is_at_least_twenty() -> None:
    assert len(_build_adapter_replay_ticks()) == 20


def test_adapter_replay_emits_expected_detector_signal_count() -> None:
    emitted_signals, _, _, _ = _run_adapter_wired_replay()
    assert emitted_signals == 8


def test_adapter_replay_records_expected_dispatch_count_after_dedup_suppression() -> None:
    _, dispatch_count, adapter, _ = _run_adapter_wired_replay()
    assert dispatch_count == 4
    assert adapter.ledger_snapshot().total_suppressed == 4


def test_adapter_replay_gross_edge_captured_is_strictly_positive() -> None:
    _, _, adapter, _ = _run_adapter_wired_replay()
    assert adapter.ledger_snapshot().gross_edge_captured > Decimal("0")


def test_adapter_replay_uses_only_synthetic_timestamps_without_time_time_calls() -> None:
    assert "time.time" not in inspect.getsource(_run_adapter_wired_replay)


def test_two_leg_replay_tick_count_is_at_least_fifteen() -> None:
    assert len(_build_two_leg_replay_ticks()) == 15


def test_two_leg_replay_full_fills_record_across_dislocation_window() -> None:
    adapter, receipts, emitted_signals = _run_two_leg_replay()
    full_fill_count = sum(1 for receipt in receipts if receipt.execution_outcome == "FULL_FILL")

    assert emitted_signals == 6
    assert full_fill_count == 5
    assert adapter.ledger_snapshot().total_executed == 5


def test_two_leg_replay_second_leg_rejection_makes_hanging_leg_rate_non_zero() -> None:
    adapter, receipts, _ = _run_two_leg_replay()

    assert any(receipt.execution_outcome == "SECOND_LEG_REJECTED" for receipt in receipts)
    assert adapter.ledger_snapshot().second_leg_rejection_rate > Decimal("0")


def test_two_leg_replay_gross_realized_pnl_is_strictly_positive() -> None:
    adapter, _, _ = _run_two_leg_replay()
    assert adapter.ledger_snapshot().gross_realized_pnl > Decimal("0")


def test_two_leg_replay_bus_slots_release_after_fill_and_allow_reentry() -> None:
    adapter, receipts, _ = _run_two_leg_replay()

    assert receipts[0].execution_outcome == "FULL_FILL"
    assert adapter.coordination_snapshot(2000)["total_active_slots"] == 0