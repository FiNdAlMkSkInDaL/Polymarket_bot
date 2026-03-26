from __future__ import annotations

from decimal import Decimal

import pytest

from src.detectors.si9_cluster_config import Si9ClusterConfig
from src.execution.dispatch_guard import DispatchGuard, GuardDecision
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.mev_router import MevExecutionRouter, MevMarketSnapshot
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.si9_paper_adapter import Si9PaperAdapter, Si9PaperAdapterConfig
from src.execution.si9_paper_ledger import Si9PaperLedger
from src.execution.si9_unwind_evaluator import Si9UnwindEvaluator
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.signals.si9_matrix_detector import Si9MatrixDetector, Si9MatrixSignal


MARKET_IDS = ("m1", "m2", "m3", "m4", "m5")
CLUSTER_ID = "cluster-replay"


class ReplayRejectGuard(DispatchGuard):
    def __init__(self, config: DispatchGuardConfig, *, reject_at_ms: int | None = None, reject_market_id: str | None = None):
        super().__init__(config)
        self._reject_at_ms = reject_at_ms
        self._reject_market_id = reject_market_id

    def check(self, context, current_timestamp_ms: int):  # type: ignore[override]
        if self._reject_at_ms == current_timestamp_ms and self._reject_market_id == context.market_id:
            return GuardDecision(allowed=False, reason="RATE_EXCEEDED")
        return super().check(context, current_timestamp_ms)


def _replay_config() -> Si9ClusterConfig:
    return Si9ClusterConfig(
        target_yield=Decimal("0.02"),
        taker_fee_per_leg=Decimal("0.002"),
        slippage_budget=Decimal("0.001"),
        ghost_town_floor=Decimal("0.82"),
        implausible_edge_ceil=Decimal("0.15"),
        max_ask_age_ms=100,
        min_cluster_size=5,
    )


def _phase_quotes(tick: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if 1 <= tick <= 10:
        return (
            ("0.205", "0.205", "0.205", "0.205", "0.205"),
            ("10", "10", "10", "10", "10"),
        )
    if 11 <= tick <= 20:
        return (
            ("0.18", "0.19", "0.19", "0.19", "0.20"),
            ("10", "9", "8", "7", "6"),
        )
    if 21 <= tick <= 25:
        return (
            ("0.18", "0.19", "0.19", "0.19", "0.20"),
            ("10", "9", "0.5", "7", "6"),
        )
    if 26 <= tick <= 30:
        return (
            ("0.18", "0.19", "0.19", "0.19", "0.20"),
            ("10", "9", "0.5", "7", "6"),
        )
    if 31 <= tick <= 35:
        return (
            ("0.18", "0.19", "0.19", "0.19", "0.20"),
            ("10", "9", "4", "7", "6"),
        )
    if 36 <= tick <= 40:
        return (
            ("0.15", "0.16", "0.16", "0.17", "0.17"),
            ("10", "9", "4", "7", "6"),
        )
    if 41 <= tick <= 45:
        return (
            ("0.16", "0.17", "0.17", "0.17", "0.17"),
            ("10", "9", "4", "7", "6"),
        )
    return (
        ("0.19", "0.20", "0.20", "0.20", "0.20"),
        ("10", "10", "10", "10", "10"),
    )


def _market_snapshot(_: str) -> MevMarketSnapshot:
    return MevMarketSnapshot(
        yes_bid=0.45,
        yes_ask=0.55,
        no_bid=0.45,
        no_ask=0.55,
    )


@pytest.fixture(scope="module")
def replay_results() -> list[dict]:
    detector = Si9MatrixDetector(_replay_config())
    detector.register_cluster(CLUSTER_ID, list(MARKET_IDS))
    results: list[dict] = []

    for tick in range(1, 51):
        timestamp_ms = tick * 25
        prices, sizes = _phase_quotes(tick)
        for market_id, ask_price, ask_size in zip(MARKET_IDS, prices, sizes, strict=True):
            if 26 <= tick <= 30 and market_id == "m3":
                continue
            detector.update_best_yes_ask(
                market_id,
                Decimal(ask_price),
                Decimal(ask_size),
                timestamp_ms,
            )

        signal = detector.evaluate_cluster(CLUSTER_ID, eval_timestamp_ms=timestamp_ms)
        results.append(
            {
                "tick": tick,
                "timestamp_ms": timestamp_ms,
                "signal": signal,
                "snapshot": detector.cluster_snapshot(CLUSTER_ID),
            }
        )

    return results


@pytest.fixture(scope="module")
def adapter_replay_results() -> tuple[list[dict], Si9PaperLedger]:
    detector = Si9MatrixDetector(_replay_config())
    detector.register_cluster(CLUSTER_ID, list(MARKET_IDS))
    dispatcher = PriorityDispatcher(MevExecutionRouter(_market_snapshot), "paper")
    guard = ReplayRejectGuard(
        DispatchGuardConfig(
            dedup_window_ms=1,
            max_dispatches_per_source_per_window=10,
            rate_window_ms=1,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_ms=100,
            max_open_positions_per_market=100,
        ),
        reject_at_ms=500,
        reject_market_id="m2",
    )
    ledger = Si9PaperLedger()
    adapter = Si9PaperAdapter(
        dispatcher=dispatcher,
        guard=guard,
        ledger=ledger,
        config=Si9PaperAdapterConfig(
            max_expected_net_edge=Decimal("0.05"),
            max_capital_per_cluster=Decimal("1000"),
            max_leg_fill_wait_ms=250,
            cancel_on_stale_ms=500,
            mode="paper",
            unwind_config=Si9UnwindConfig(
                market_sell_threshold=Decimal("10.0"),
                passive_unwind_threshold=Decimal("0.10"),
                max_hold_recovery_ms=50,
                min_best_bid=Decimal("0.010"),
            ),
        ),
    )
    results: list[dict] = []

    for tick in range(11, 26):
        timestamp_ms = tick * 25
        prices, sizes = _phase_quotes(tick)
        for market_id, ask_price, ask_size in zip(MARKET_IDS, prices, sizes, strict=True):
            detector.update_best_yes_ask(
                market_id,
                Decimal(ask_price),
                Decimal(ask_size),
                timestamp_ms,
            )
        signal = detector.evaluate_cluster(CLUSTER_ID, eval_timestamp_ms=timestamp_ms)
        if signal is None:
            continue
        receipt = adapter.on_signal(signal, current_timestamp_ms=timestamp_ms)
        results.append({
            "tick": tick,
            "receipt": receipt,
            "signal": signal,
        })

    return results, ledger


def _signals(results: list[dict], start_tick: int, end_tick: int) -> list[Si9MatrixSignal]:
    return [
        item["signal"]
        for item in results
        if start_tick <= item["tick"] <= end_tick and item["signal"] is not None
    ]


def test_replay_harness_produces_50_ticks(replay_results: list[dict]) -> None:
    assert len(replay_results) == 50


def test_replay_warm_up_phase_emits_no_signals(replay_results: list[dict]) -> None:
    assert _signals(replay_results, 1, 10) == []


def test_replay_dislocation_window_emits_ten_signals(replay_results: list[dict]) -> None:
    signals = _signals(replay_results, 11, 20)

    assert len(signals) == 10
    assert {signal.bottleneck_market_id for signal in signals} == {"m5"}


def test_replay_dislocation_window_uses_expected_net_edge(replay_results: list[dict]) -> None:
    signal = _signals(replay_results, 11, 20)[0]

    assert signal.net_edge == Decimal("0.035")


def test_replay_depth_shock_reprices_uniform_share_counts(replay_results: list[dict]) -> None:
    signals = _signals(replay_results, 21, 25)

    assert len(signals) == 5
    assert {signal.bottleneck_market_id for signal in signals} == {"m3"}
    assert all(signal.required_share_counts == Decimal("0.5") for signal in signals)


def test_replay_staleness_boundary_is_still_valid_at_tick_29(replay_results: list[dict]) -> None:
    tick_29 = replay_results[28]["signal"]

    assert tick_29 is not None
    assert tick_29.bottleneck_market_id == "m3"


def test_replay_staleness_suppresses_tick_30(replay_results: list[dict]) -> None:
    assert replay_results[29]["signal"] is None


def test_replay_recovery_resumes_cluster_signals(replay_results: list[dict]) -> None:
    signals = _signals(replay_results, 31, 35)

    assert len(signals) == 5
    assert {signal.bottleneck_market_id for signal in signals} == {"m3"}


def test_replay_ghost_town_phase_suppresses_all_signals(replay_results: list[dict]) -> None:
    assert _signals(replay_results, 36, 40) == []


def test_replay_implausible_edge_phase_suppresses_all_signals(replay_results: list[dict]) -> None:
    assert _signals(replay_results, 41, 45) == []


def test_replay_total_signal_count_is_exactly_24(replay_results: list[dict]) -> None:
    assert len([item for item in replay_results if item["signal"] is not None]) == 24


def test_adapter_replay_processes_signals_end_to_end(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    results, _ = adapter_replay_results

    assert len(results) == 15
    assert results[0]["receipt"].cluster_outcome == "FULL_FILL"


def test_adapter_replay_records_one_hanging_leg_scenario(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    results, ledger = adapter_replay_results
    hanging = [item for item in results if item["receipt"].cluster_outcome == "HANGING_LEG"]

    assert len(hanging) == 1
    assert hanging[0]["tick"] == 20
    assert ledger.snapshot().total_hanging_leg_events == 1


def test_adapter_replay_final_ledger_snapshot_has_positive_full_fills_and_edge(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    _, ledger = adapter_replay_results
    snapshot = ledger.snapshot()

    assert snapshot.total_full_fills > 0
    assert snapshot.hanging_leg_rate > Decimal("0")
    assert snapshot.gross_edge_captured > Decimal("0")


def test_adapter_replay_hanging_receipt_stops_after_rejected_leg(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    results, _ = adapter_replay_results
    hanging = next(item for item in results if item["receipt"].cluster_outcome == "HANGING_LEG")

    assert len(hanging["receipt"].per_leg_receipts) == 2
    assert [receipt.context.market_id for receipt in hanging["receipt"].per_leg_receipts] == ["m5", "m1"]


def test_adapter_replay_full_fill_receipts_start_with_bottleneck(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    results, _ = adapter_replay_results
    full_fill = next(item for item in results if item["receipt"].cluster_outcome == "FULL_FILL")

    assert full_fill["receipt"].per_leg_receipts[0].context.market_id == full_fill["signal"].bottleneck_market_id


def test_adapter_replay_hanging_leg_triggers_unwind_evaluator(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    results, _ = adapter_replay_results
    hanging = next(item for item in results if item["receipt"].cluster_outcome == "HANGING_LEG")

    assert hanging["receipt"].unwind_manifest is not None
    assert hanging["receipt"].unwind_manifest.total_estimated_unwind_cost > Decimal("0")


def test_adapter_replay_unwind_escalation_promotes_hold_to_market_sell(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    results, _ = adapter_replay_results
    hanging = next(item for item in results if item["receipt"].cluster_outcome == "HANGING_LEG")
    receipt = hanging["receipt"]
    signal = hanging["signal"]

    assert receipt.unwind_manifest is not None

    evaluator = Si9UnwindEvaluator(
        Si9UnwindConfig(
            market_sell_threshold=Decimal("10.0"),
            passive_unwind_threshold=Decimal("0.000001"),
            max_hold_recovery_ms=50,
            min_best_bid=Decimal("0.010"),
        )
    )
    hold_manifest = evaluator.evaluate(
        cluster_id=receipt.manifest.cluster_id,
        hanging_legs=[
            (executed.context.market_id, executed.fill_size, executed.fill_price)
            for executed in receipt.per_leg_receipts
            if executed.fill_size is not None and executed.fill_price is not None
        ],
        current_bids={
            executed.context.market_id: signal.best_yes_asks[executed.context.market_id]
            for executed in receipt.per_leg_receipts
        },
        unwind_reason="MANUAL_ABORT",
        original_manifest=receipt.manifest,
        unwind_timestamp_ms=receipt.manifest.manifest_timestamp_ms,
    )
    assert hold_manifest.recommended_action == "HOLD_FOR_RECOVERY"

    escalated = evaluator.escalate(hold_manifest, current_timestamp_ms=hold_manifest.unwind_timestamp_ms + 51)

    assert escalated.recommended_action == "MARKET_SELL"
    assert hold_manifest.recommended_action == "HOLD_FOR_RECOVERY"


def test_adapter_replay_ledger_tracks_unwind_counts_and_cost(adapter_replay_results: tuple[list[dict], Si9PaperLedger]) -> None:
    _, ledger = adapter_replay_results
    snapshot = ledger.snapshot()

    assert snapshot.total_unwind_manifests > 0
    assert snapshot.gross_estimated_unwind_cost > Decimal("0")
