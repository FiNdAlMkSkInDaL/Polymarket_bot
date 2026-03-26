from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import Callable

import pytest

from src.events.mev_events import CtfMergeSignal
from src.execution.ctf_execution_manifest import CtfExecutionReceipt
from src.execution.ctf_paper_adapter import CtfPaperAdapter, CtfPaperAdapterConfig
from src.execution.dispatch_guard import DispatchGuard
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchReceipt
from src.execution.signal_coordination_bus import CoordinationBusConfig, SignalCoordinationBus


def _signal(
    *,
    market_id: str = "MKT_CTF",
    yes_ask: Decimal = Decimal("0.380000"),
    no_ask: Decimal = Decimal("0.400000"),
    gas_estimate: Decimal = Decimal("0.010000"),
    net_edge: Decimal = Decimal("0.185000"),
) -> CtfMergeSignal:
    return CtfMergeSignal(
        market_id=market_id,
        yes_ask=yes_ask,
        no_ask=no_ask,
        gas_estimate=gas_estimate,
        net_edge=net_edge,
    )


def _bus_config(**overrides) -> CoordinationBusConfig:
    values = {
        "slot_lease_ms": 500,
        "max_slots_per_source": 4,
        "max_total_slots": 8,
        "allow_same_source_reentry": False,
    }
    values.update(overrides)
    return CoordinationBusConfig(**values)


def _adapter_config(**overrides) -> CtfPaperAdapterConfig:
    values = {
        "max_expected_net_edge": Decimal("0.250000"),
        "max_capital_per_signal": Decimal("25.000000"),
        "default_anchor_volume": Decimal("10.000000"),
        "taker_fee_yes": Decimal("0.010000"),
        "taker_fee_no": Decimal("0.010000"),
        "cancel_on_stale_ms": 250,
        "max_size_per_leg": Decimal("8.000000"),
        "mode": "paper",
        "bus": SignalCoordinationBus(_bus_config()),
    }
    values.update(overrides)
    return CtfPaperAdapterConfig(**values)


def _guard_config(**overrides: int) -> DispatchGuardConfig:
    values = {
        "dedup_window_ms": 100,
        "max_dispatches_per_source_per_window": 2,
        "rate_window_ms": 200,
        "circuit_breaker_threshold": 2,
        "circuit_breaker_reset_ms": 300,
        "max_open_positions_per_market": 10,
    }
    values.update(overrides)
    return DispatchGuardConfig(**values)


class _RecordingGuard(DispatchGuard):
    def __init__(self, config: DispatchGuardConfig):
        super().__init__(config)
        self.check_calls = 0
        self.record_dispatch_calls = 0
        self.recorded_dispatch_contexts: list[PriorityOrderContext] = []

    def check(self, context: PriorityOrderContext, current_timestamp_ms: int):
        self.check_calls += 1
        return super().check(context, current_timestamp_ms)

    def record_dispatch(self, context: PriorityOrderContext, current_timestamp_ms: int) -> None:
        self.record_dispatch_calls += 1
        self.recorded_dispatch_contexts.append(context)
        super().record_dispatch(context, current_timestamp_ms)


class _ScriptedDispatcher:
    def __init__(
        self,
        outcomes: list[str] | None = None,
        after_dispatch_hooks: dict[int, Callable[[PriorityOrderContext, int], None]] | None = None,
    ):
        self.calls: list[PriorityOrderContext] = []
        self._outcomes = list(outcomes or [])
        self._after_dispatch_hooks = dict(after_dispatch_hooks or {})

    def dispatch(self, context: PriorityOrderContext, dispatch_timestamp_ms: int) -> DispatchReceipt:
        self.calls.append(context)
        hook = self._after_dispatch_hooks.get(len(self.calls))
        if hook is not None:
            hook(context, dispatch_timestamp_ms)
        outcome = self._outcomes.pop(0) if self._outcomes else "FULL"
        fill_price = (context.target_price + Decimal("0.000001")).quantize(Decimal("0.000001"))
        fill_size = (context.anchor_volume * context.conviction_scalar).quantize(Decimal("0.000001"))

        if outcome == "REJECT":
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
        if outcome == "PARTIAL":
            partial_fill_size = (fill_size / Decimal("2")).quantize(Decimal("0.000001"))
            return DispatchReceipt(
                context=context,
                mode="paper",
                executed=True,
                fill_price=fill_price,
                fill_size=fill_size,
                serialized_envelope="{}",
                dispatch_timestamp_ms=dispatch_timestamp_ms,
                partial_fill_size=partial_fill_size,
                partial_fill_price=fill_price,
                fill_status="PARTIAL",
            )
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


def _make_adapter(
    *,
    guard_config: DispatchGuardConfig | None = None,
    adapter_config: CtfPaperAdapterConfig | None = None,
    outcomes: list[str] | None = None,
    after_dispatch_hooks: dict[int, Callable[[PriorityOrderContext, int], None]] | None = None,
) -> tuple[CtfPaperAdapter, _RecordingGuard, SignalCoordinationBus | None, _ScriptedDispatcher]:
    config = adapter_config or _adapter_config()
    dispatcher = _ScriptedDispatcher(outcomes, after_dispatch_hooks=after_dispatch_hooks)
    guard = _RecordingGuard(guard_config or _guard_config())
    return CtfPaperAdapter(dispatcher, guard, config), guard, config.bus, dispatcher


def _manual_required_size(signal: CtfMergeSignal, config: CtfPaperAdapterConfig) -> Decimal:
    return min(
        config.default_anchor_volume,
        config.max_size_per_leg,
        config.max_capital_per_signal / (signal.yes_ask + signal.no_ask),
    )


def _manual_fill_size(signal: CtfMergeSignal, config: CtfPaperAdapterConfig) -> Decimal:
    conviction_scalar = min(Decimal("1"), signal.net_edge / config.max_expected_net_edge)
    return (_manual_required_size(signal, config) * conviction_scalar).quantize(Decimal("0.000001"))


def _manual_realized_net_edge(signal: CtfMergeSignal, config: CtfPaperAdapterConfig) -> Decimal:
    return (
        Decimal("1")
        - (signal.yes_ask + Decimal("0.000001"))
        - (signal.no_ask + Decimal("0.000001"))
        - config.taker_fee_yes
        - config.taker_fee_no
        - signal.gas_estimate
    ).quantize(Decimal("0.000001"))


def test_valid_ctf_paper_adapter_construction_passes() -> None:
    adapter, _, _, _ = _make_adapter()

    assert adapter.ledger_snapshot().total_dispatched == 0


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("max_expected_net_edge", Decimal("0"), "max_expected_net_edge"),
        ("max_capital_per_signal", Decimal("0"), "max_capital_per_signal"),
        ("default_anchor_volume", Decimal("0"), "default_anchor_volume"),
        ("taker_fee_yes", Decimal("-0.000001"), "taker_fee_yes"),
        ("taker_fee_no", Decimal("-0.000001"), "taker_fee_no"),
        ("cancel_on_stale_ms", 0, "cancel_on_stale_ms"),
        ("max_size_per_leg", Decimal("0"), "max_size_per_leg"),
    ],
)
def test_invalid_ctf_paper_adapter_config_fields_raise_value_error(field_name: str, field_value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _adapter_config(**{field_name: field_value})


def test_invalid_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported CTF paper adapter mode"):
        _adapter_config(mode="live")


def test_ctf_paper_adapter_config_is_frozen() -> None:
    config = _adapter_config()

    with pytest.raises(FrozenInstanceError):
        config.mode = "dry_run"  # type: ignore[misc]


def test_two_leg_dispatch_both_legs_fill_with_full_fill_outcome() -> None:
    adapter, _, _, dispatcher = _make_adapter()

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert isinstance(receipt, CtfExecutionReceipt)
    assert receipt.execution_outcome == "FULL_FILL"
    assert receipt.executed is True
    assert receipt.yes_receipt.fill_status == "FILLED"
    assert receipt.no_receipt.fill_status == "FILLED"
    assert len(dispatcher.calls) == 2


def test_anchor_leg_dispatches_first_when_yes_is_cheaper() -> None:
    adapter, _, _, dispatcher = _make_adapter()

    adapter.on_signal(_signal(yes_ask=Decimal("0.380000"), no_ask=Decimal("0.400000")), current_timestamp_ms=1000)

    assert dispatcher.calls[0].side == "YES"
    assert dispatcher.calls[0].leg_role == "YES_LEG"


def test_anchor_leg_dispatches_first_when_no_is_cheaper() -> None:
    adapter, _, _, dispatcher = _make_adapter()

    adapter.on_signal(_signal(yes_ask=Decimal("0.420000"), no_ask=Decimal("0.390000")), current_timestamp_ms=1000)

    assert dispatcher.calls[0].side == "NO"
    assert dispatcher.calls[0].leg_role == "NO_LEG"


def test_conviction_scalar_is_normalized_from_net_edge_ratio() -> None:
    config = _adapter_config(max_expected_net_edge=Decimal("0.250000"))
    adapter, _, _, dispatcher = _make_adapter(adapter_config=config)

    adapter.on_signal(_signal(net_edge=Decimal("0.125000")), current_timestamp_ms=1000)

    assert dispatcher.calls[0].conviction_scalar == Decimal("0.5")


def test_conviction_scalar_clamps_to_exactly_one_when_edge_exceeds_ceiling() -> None:
    adapter, _, _, dispatcher = _make_adapter(adapter_config=_adapter_config(max_expected_net_edge=Decimal("0.100000")))

    adapter.on_signal(_signal(net_edge=Decimal("0.185000")), current_timestamp_ms=1000)

    assert dispatcher.calls[0].conviction_scalar == Decimal("1")


def test_bus_rejection_on_yes_slot_returns_bus_rejected_and_guard_never_called() -> None:
    adapter, guard, bus, _ = _make_adapter()
    assert bus is not None
    bus.request_slot("MKT_CTF", "YES", "CTF", 900)

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.execution_outcome == "BUS_REJECTED"
    assert guard.check_calls == 0
    assert adapter.ledger_snapshot().total_bus_rejected == 1


def test_bus_rejection_on_no_slot_releases_granted_yes_slot_and_skips_guard() -> None:
    bus = SignalCoordinationBus(_bus_config())
    config = _adapter_config(bus=bus)
    adapter, guard, _, _ = _make_adapter(adapter_config=config)
    bus.request_slot("MKT_CTF", "NO", "CTF", 900)

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.execution_outcome == "BUS_REJECTED"
    assert guard.check_calls == 0
    assert bus.bus_snapshot(1000).total_active_slots == 1
    assert bus.bus_snapshot(1600).total_active_slots == 0


def test_guard_rejection_returns_guard_rejected_and_releases_slots() -> None:
    adapter, guard, _, _ = _make_adapter()
    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1050)

    assert receipt.execution_outcome == "GUARD_REJECTED"
    assert guard.check_calls == 3
    assert adapter.coordination_snapshot(1050)["total_active_slots"] == 0


def test_anchor_leg_rejection_returns_anchor_rejected_and_skips_second_dispatch() -> None:
    adapter, _, _, dispatcher = _make_adapter(outcomes=["REJECT"])

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.execution_outcome == "ANCHOR_REJECTED"
    assert len(dispatcher.calls) == 1
    assert adapter.ledger_snapshot().total_anchor_rejected == 1


def test_second_leg_rejection_returns_second_leg_rejected_and_records_hanging_leg() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["FULL", "REJECT"])

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.execution_outcome == "SECOND_LEG_REJECTED"
    assert adapter.ledger_snapshot().total_second_leg_rejected == 1
    assert adapter.coordination_snapshot(1000)["total_active_slots"] == 0


def test_anchor_fill_followed_by_second_leg_guard_rejection_records_single_anchor_dispatch() -> None:
    adapter, guard, _, _ = _make_adapter(guard_config=_guard_config(max_open_positions_per_market=1))

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.execution_outcome == "SECOND_LEG_REJECTED"
    assert receipt.yes_receipt.fill_status == "FILLED"
    assert receipt.no_receipt.fill_status == "REJECTED"
    assert guard.record_dispatch_calls == 1
    assert guard.recorded_dispatch_contexts[0].side == "YES"
    assert adapter.coordination_snapshot(1000)["total_active_slots"] == 0
    snapshot = adapter.ledger_snapshot()
    assert snapshot.total_second_leg_rejected == 1
    assert snapshot.total_anchor_rejected == 0


def test_anchor_fill_with_midflight_second_leg_bus_expiry_is_classified_as_second_leg_rejection() -> None:
    bus = SignalCoordinationBus(_bus_config())

    def _expire_second_leg_slot(_: PriorityOrderContext, dispatch_timestamp_ms: int) -> None:
        slot = bus._slot_map[("MKT_CTF", "NO")]
        slot.lease_expires_ms = dispatch_timestamp_ms - 1

    adapter, _, _, _ = _make_adapter(
        adapter_config=_adapter_config(bus=bus),
        after_dispatch_hooks={1: _expire_second_leg_slot},
    )

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    # The bus race is resolved after anchor exposure, so lost second-leg ownership
    # is treated as SECOND_LEG_REJECTED rather than a pre-dispatch BUS_REJECTED.
    assert receipt.execution_outcome == "SECOND_LEG_REJECTED"
    assert receipt.no_receipt.dispatch_receipt.guard_reason == "BUS_SLOT_LOST"
    assert adapter.coordination_snapshot(1000)["total_active_slots"] == 0


def test_sequential_cluster_attempts_hold_dedup_at_cluster_level_and_release_bus_slots_between_attempts() -> None:
    adapter, _, _, _ = _make_adapter(guard_config=_guard_config(dedup_window_ms=100))

    first_receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)
    second_receipt = adapter.on_signal(_signal(), current_timestamp_ms=1050)
    snapshot_after_second = adapter.ledger_snapshot()
    third_receipt = adapter.on_signal(_signal(), current_timestamp_ms=1201)

    assert first_receipt.execution_outcome == "FULL_FILL"
    assert adapter.coordination_snapshot(1000)["total_active_slots"] == 0
    assert second_receipt.execution_outcome == "GUARD_REJECTED"
    assert snapshot_after_second.total_dispatched == 1
    assert third_receipt.execution_outcome == "FULL_FILL"
    assert adapter.ledger_snapshot().total_dispatched == 2
    assert adapter.coordination_snapshot(1201)["total_active_slots"] == 0


def test_negative_realized_pnl_is_recorded_without_blocking_full_fill_execution() -> None:
    adapter, _, _, _ = _make_adapter()
    signal = _signal(
        yes_ask=Decimal("0.520000"),
        no_ask=Decimal("0.500000"),
        gas_estimate=Decimal("0.020000"),
        net_edge=Decimal("0.030000"),
    )

    receipt = adapter.on_signal(signal, current_timestamp_ms=1000)

    assert receipt.execution_outcome == "FULL_FILL"
    assert receipt.realized_pnl < Decimal("0")
    assert adapter.ledger_snapshot().gross_realized_pnl == receipt.realized_pnl


def test_zero_gas_estimate_keeps_manifest_and_realized_pnl_calculation_valid() -> None:
    config = _adapter_config()
    adapter, _, _, _ = _make_adapter(adapter_config=config)
    signal = _signal(gas_estimate=Decimal("0"))

    receipt = adapter.on_signal(signal, current_timestamp_ms=1000)
    expected_pnl = (_manual_realized_net_edge(signal, config) * _manual_fill_size(signal, config)).quantize(Decimal("0.000001"))

    assert receipt.manifest.gas_estimate == Decimal("0")
    assert receipt.realized_net_edge == _manual_realized_net_edge(signal, config)
    assert receipt.realized_pnl == expected_pnl


def test_full_fill_releases_bus_slots_immediately() -> None:
    adapter, _, _, _ = _make_adapter()

    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert adapter.coordination_snapshot(1000)["total_active_slots"] == 0


def test_second_leg_rejection_releases_bus_slots_immediately() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["FULL", "REJECT"])

    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert adapter.coordination_snapshot(1000)["total_active_slots"] == 0


def test_full_fill_realized_net_edge_matches_manual_formula() -> None:
    config = _adapter_config()
    adapter, _, _, _ = _make_adapter(adapter_config=config)
    signal = _signal()

    receipt = adapter.on_signal(signal, current_timestamp_ms=1000)

    assert receipt.realized_net_edge == _manual_realized_net_edge(signal, config)


def test_full_fill_realized_pnl_matches_manual_formula() -> None:
    config = _adapter_config()
    adapter, _, _, _ = _make_adapter(adapter_config=config)
    signal = _signal()

    receipt = adapter.on_signal(signal, current_timestamp_ms=1000)
    expected = (_manual_realized_net_edge(signal, config) * _manual_fill_size(signal, config)).quantize(Decimal("0.000001"))

    assert receipt.realized_pnl == expected


def test_full_fill_total_capital_deployed_matches_manual_formula() -> None:
    config = _adapter_config()
    adapter, _, _, _ = _make_adapter(adapter_config=config)
    signal = _signal()

    receipt = adapter.on_signal(signal, current_timestamp_ms=1000)
    fill_size = _manual_fill_size(signal, config)
    expected = (
        (signal.yes_ask + Decimal("0.000001")) * fill_size
        + (signal.no_ask + Decimal("0.000001")) * fill_size
    ).quantize(Decimal("0.000001"))

    assert receipt.total_capital_deployed == expected


def test_ledger_records_dispatched_and_executed_once_per_full_fill() -> None:
    adapter, _, _, _ = _make_adapter()

    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    snapshot = adapter.ledger_snapshot()
    assert snapshot.total_clusters_attempted == 1
    assert snapshot.total_dispatched == 1
    assert snapshot.total_executed == 1


def test_ledger_records_suppressed_count_on_guard_rejection() -> None:
    adapter, _, _, _ = _make_adapter()
    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    adapter.on_signal(_signal(), current_timestamp_ms=1050)

    assert adapter.ledger_snapshot().total_suppressed == 1


def test_ledger_records_bus_rejected_count() -> None:
    adapter, _, bus, _ = _make_adapter()
    assert bus is not None
    bus.request_slot("MKT_CTF", "YES", "CTF", 900)

    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert adapter.ledger_snapshot().total_bus_rejected == 1


def test_ledger_second_leg_rejection_rate_uses_attempt_denominator() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["FULL", "REJECT", "FULL", "FULL"])

    adapter.on_signal(_signal(market_id="MKT_1"), current_timestamp_ms=1000)
    adapter.on_signal(_signal(market_id="MKT_2"), current_timestamp_ms=2000)

    assert adapter.ledger_snapshot().second_leg_rejection_rate == Decimal("0.5")


def test_ledger_second_leg_rejection_rate_is_zero_with_no_attempts() -> None:
    adapter, _, _, _ = _make_adapter()

    assert adapter.ledger_snapshot().second_leg_rejection_rate == Decimal("0")


def test_ledger_second_leg_rejection_rate_is_one_with_single_rejection() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["FULL", "REJECT"])

    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert adapter.ledger_snapshot().second_leg_rejection_rate == Decimal("1")


def test_ledger_second_leg_rejection_rate_supports_fractional_attempts_to_six_decimals() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["FULL", "REJECT", "FULL", "FULL", "FULL", "FULL"])

    adapter.on_signal(_signal(market_id="MKT_1"), current_timestamp_ms=1000)
    adapter.on_signal(_signal(market_id="MKT_2"), current_timestamp_ms=2000)
    adapter.on_signal(_signal(market_id="MKT_3"), current_timestamp_ms=3000)

    assert adapter.ledger_snapshot().second_leg_rejection_rate.quantize(Decimal("0.000001")) == Decimal("0.333333")


def test_ledger_reset_clears_accumulators() -> None:
    adapter, _, _, _ = _make_adapter()
    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    adapter.reset_ledger()

    snapshot = adapter.ledger_snapshot()
    assert snapshot.total_clusters_attempted == 0
    assert snapshot.gross_realized_pnl == Decimal("0")


def test_partial_fill_outcome_is_recorded_when_any_leg_partially_fills() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["PARTIAL", "FULL"])

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.execution_outcome == "PARTIAL_FILL"
    assert adapter.ledger_snapshot().total_partial_fills == 1


def test_partial_fill_uses_partial_fill_size_as_actual_leg_size() -> None:
    config = _adapter_config()
    adapter, _, _, _ = _make_adapter(adapter_config=config, outcomes=["PARTIAL", "FULL"])

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.yes_receipt.filled_size == (_manual_fill_size(_signal(), config) / Decimal("2")).quantize(Decimal("0.000001"))


def test_partial_fill_realized_pnl_uses_minimum_completed_cluster_size() -> None:
    config = _adapter_config()
    adapter, _, _, _ = _make_adapter(adapter_config=config, outcomes=["PARTIAL", "FULL"])
    signal = _signal()

    receipt = adapter.on_signal(signal, current_timestamp_ms=1000)
    expected_cluster_size = (_manual_fill_size(signal, config) / Decimal("2")).quantize(Decimal("0.000001"))

    assert receipt.realized_pnl == (_manual_realized_net_edge(signal, config) * expected_cluster_size).quantize(Decimal("0.000001"))


def test_coordination_snapshot_without_bus_reports_empty_state() -> None:
    adapter, _, _, _ = _make_adapter(adapter_config=_adapter_config(bus=None))

    snapshot = adapter.coordination_snapshot(1000)

    assert snapshot["total_active_slots"] == 0
    assert snapshot["slots_by_source"] == {}


def test_full_fill_leg_statuses_are_filled() -> None:
    adapter, _, _, _ = _make_adapter()

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.yes_receipt.fill_status == "FILLED"
    assert receipt.no_receipt.fill_status == "FILLED"


def test_anchor_rejected_marks_second_leg_suppressed() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["REJECT"])

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.execution_outcome == "ANCHOR_REJECTED"
    assert receipt.no_receipt.fill_status == "SUPPRESSED"


def test_second_leg_rejected_marks_anchor_filled_and_second_rejected() -> None:
    adapter, _, _, _ = _make_adapter(outcomes=["FULL", "REJECT"])

    receipt = adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert receipt.yes_receipt.fill_status == "FILLED"
    assert receipt.no_receipt.fill_status == "REJECTED"


def test_guard_is_checked_once_per_signal() -> None:
    adapter, guard, _, _ = _make_adapter()

    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert guard.check_calls == 2


def test_dispatcher_contexts_carry_leg_role_tags() -> None:
    adapter, _, _, dispatcher = _make_adapter()

    adapter.on_signal(_signal(), current_timestamp_ms=1000)

    assert {call.leg_role for call in dispatcher.calls} == {"YES_LEG", "NO_LEG"}
