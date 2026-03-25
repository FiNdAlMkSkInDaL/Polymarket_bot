from __future__ import annotations

import json

import pytest

from src.events.mev_events import (
    DisputeArbitrageSignal,
    MMPredationSignal,
    ShadowSweepSignal,
)
from src.execution.mev_dispatcher import MevDispatcher
from src.execution.mev_router import MevExecutionRouter, MevMarketSnapshot
from src.execution.mev_serializer import serialize_mev_execution_batch


@pytest.fixture
def snapshot_provider():
    snapshots = {
        "MKT_SHADOW": MevMarketSnapshot(
            yes_bid=0.45,
            yes_ask=0.55,
            no_bid=0.43,
            no_ask=0.57,
        ),
        "MKT_TARGET": MevMarketSnapshot(
            yes_bid=0.47,
            yes_ask=0.48,
            no_bid=0.50,
            no_ask=0.52,
        ),
        "MKT_CORR": MevMarketSnapshot(
            yes_bid=0.44,
            yes_ask=0.46,
            no_bid=0.51,
            no_ask=0.55,
        ),
        "MKT_CLAMP": MevMarketSnapshot(
            yes_bid=0.48,
            yes_ask=0.50,
            no_bid=0.50,
            no_ask=0.52,
        ),
    }

    return lambda market_id: snapshots[market_id]


def test_shadow_sweep_builds_ioc_then_follow_up_passive_order(snapshot_provider) -> None:
    router = MevExecutionRouter(snapshot_provider)

    batch = router.execute_shadow_sweep(
        market_id="MKT_SHADOW",
        direction="YES",
        max_capital=55.0,
        premium_pct=0.03,
    )

    assert batch.playbook == "shadow_sweep"
    assert len(batch.payloads) == 2
    assert [payload.sequence for payload in batch.payloads] == [1, 2]

    taker_payload, maker_payload = batch.payloads
    assert taker_payload.market_id == "MKT_SHADOW"
    assert taker_payload.direction == "YES"
    assert taker_payload.liquidity_intent == "TAKER"
    assert taker_payload.time_in_force == "IOC"
    assert taker_payload.post_only is False
    assert taker_payload.price == pytest.approx(0.55)
    assert taker_payload.size == pytest.approx(100.0)

    assert maker_payload.market_id == "MKT_SHADOW"
    assert maker_payload.direction == "YES"
    assert maker_payload.liquidity_intent == "MAKER"
    assert maker_payload.time_in_force == "GTC"
    assert maker_payload.post_only is True
    assert maker_payload.price == pytest.approx(0.53)
    assert maker_payload.metadata["rationale"] == "capture_post_sweep_repricing"
    assert router.sent_payloads == list(batch.payloads)
    assert [response["sequence"] for response in batch.responses] == [1, 2]


def test_shadow_sweep_clamps_passive_price_below_ask(snapshot_provider) -> None:
    router = MevExecutionRouter(snapshot_provider)

    batch = router.execute_shadow_sweep(
        market_id="MKT_CLAMP",
        direction="YES",
        max_capital=50.0,
        premium_pct=0.03,
    )

    maker_payload = batch.payloads[1]
    assert maker_payload.price == pytest.approx(0.49)
    assert maker_payload.post_only is True


def test_mm_trap_builds_correlated_maker_then_exact_attack_ping(snapshot_provider) -> None:
    router = MevExecutionRouter(snapshot_provider)

    batch = router.execute_mm_trap(
        target_market_id="MKT_TARGET",
        correlated_market_id="MKT_CORR",
        v_attack=40.0,
        trap_direction="YES",
    )

    assert batch.playbook == "mm_trap"
    assert len(batch.payloads) == 2
    assert [payload.market_id for payload in batch.payloads] == ["MKT_CORR", "MKT_TARGET"]

    maker_payload, taker_payload = batch.payloads
    assert maker_payload.direction == "NO"
    assert maker_payload.liquidity_intent == "MAKER"
    assert maker_payload.post_only is True
    assert maker_payload.time_in_force == "GTC"
    assert maker_payload.price == pytest.approx(0.53)
    assert maker_payload.size == pytest.approx(36.2264)
    assert maker_payload.metadata["trap_direction"] == "YES"

    assert taker_payload.direction == "YES"
    assert taker_payload.liquidity_intent == "TAKER"
    assert taker_payload.post_only is False
    assert taker_payload.time_in_force == "IOC"
    assert taker_payload.price == pytest.approx(0.48)
    assert taker_payload.size == pytest.approx(40.0)
    assert taker_payload.metadata["v_attack"] == pytest.approx(40.0)
    assert router.sent_payloads == list(batch.payloads)


def test_mev_router_rejects_unknown_direction(snapshot_provider) -> None:
    router = MevExecutionRouter(snapshot_provider)

    with pytest.raises(ValueError, match="Unsupported MEV direction"):
        router.execute_shadow_sweep(
            market_id="MKT_SHADOW",
            direction="BUY",
            max_capital=10.0,
            premium_pct=0.01,
        )


def test_d3_panic_absorption_builds_passive_grid(snapshot_provider) -> None:
    router = MevExecutionRouter(snapshot_provider)

    batch = router.execute_d3_panic_absorption(
        market_id="MKT_CLAMP",
        panic_direction="YES",
        limit_price=0.49,
        max_capital=49.0,
    )

    assert batch.playbook == "d3_panic_absorption"
    assert len(batch.payloads) == 3
    assert [payload.sequence for payload in batch.payloads] == [1, 2, 3]
    assert [payload.price for payload in batch.payloads] == pytest.approx([0.49, 0.48, 0.48])
    assert [payload.size for payload in batch.payloads] == pytest.approx([33.3333, 33.3334, 33.3333])
    assert all(payload.post_only for payload in batch.payloads)
    assert all(payload.liquidity_intent == "MAKER" for payload in batch.payloads)
    assert batch.payloads[0].metadata["rationale"] == "absorb_retail_panic_cascade"
    assert router.sent_payloads == list(batch.payloads)


def test_priority_sequence_builds_anchor_exit_with_fixed_precision_json(snapshot_provider) -> None:
    router = MevExecutionRouter(snapshot_provider)

    batch = router.execute_priority_sequence(
        target_price=0.64,
        anchor_volume=50.0,
        max_capital=100.0,
    )

    assert batch.playbook == "priority_sequence"
    assert len(batch.payloads) == 2
    assert [payload.sequence for payload in batch.payloads] == [1, 2]

    entry_payload, exit_payload = batch.payloads
    assert entry_payload.side == "BUY"
    assert entry_payload.liquidity_intent == "PRIORITY"
    assert entry_payload.price == pytest.approx(0.640001)
    assert entry_payload.size == pytest.approx(50.0)
    assert entry_payload.metadata["optimized_price"] == "0.640001"
    assert entry_payload.metadata["priority_epsilon"] == "0.000001"

    assert exit_payload.side == "SELL"
    assert exit_payload.liquidity_intent == "CONDITIONAL"
    assert exit_payload.price == pytest.approx(0.64)
    assert exit_payload.size == pytest.approx(50.0)
    assert exit_payload.metadata["conditional_order_type"] == "STOP_LIMIT"
    assert exit_payload.metadata["stop_price"] == "0.640000"
    assert exit_payload.metadata["limit_price"] == "0.640000"

    serialized = serialize_mev_execution_batch(batch)
    parsed = json.loads(serialized)

    assert parsed["payloads"][0]["price"] == "0.640001"
    assert isinstance(parsed["payloads"][0]["price"], str)
    assert parsed["payloads"][1]["price"] == "0.640000"
    assert parsed["payloads"][1]["metadata"]["stop_price"] == "0.640000"
    assert parsed["payloads"][1]["metadata"]["limit_price"] == "0.640000"
    assert "e-" not in serialized.lower()


class _RecordingRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute_shadow_sweep(self, **kwargs):
        self.calls.append(("shadow", kwargs))
        return "shadow-batch"

    def execute_mm_trap(self, **kwargs):
        self.calls.append(("trap", kwargs))
        return "trap-batch"

    def execute_d3_panic_absorption(self, **kwargs):
        self.calls.append(("d3", kwargs))
        return "d3-batch"


def test_dispatcher_routes_mempool_event_to_shadow_sweep() -> None:
    router = _RecordingRouter()
    dispatcher = MevDispatcher(router)  # type: ignore[arg-type]

    result = dispatcher.on_mempool_whale_detected(
        ShadowSweepSignal(
            target_market_id="MKT_SHADOW",
            direction="YES",
            max_capital=55.0,
            premium_pct=0.03,
        )
    )

    assert result == "shadow-batch"
    assert router.calls == [(
        "shadow",
        {
            "market_id": "MKT_SHADOW",
            "direction": "YES",
            "max_capital": 55.0,
            "premium_pct": 0.03,
        },
    )]


def test_dispatcher_routes_mm_event_to_trap() -> None:
    router = _RecordingRouter()
    dispatcher = MevDispatcher(router)  # type: ignore[arg-type]

    result = dispatcher.on_mm_vulnerability_detected(
        MMPredationSignal(
            target_market_id="MKT_TARGET",
            correlated_market_id="MKT_CORR",
            v_attack=40.0,
            trap_direction="YES",
        )
    )

    assert result == "trap-batch"
    assert router.calls == [(
        "trap",
        {
            "target_market_id": "MKT_TARGET",
            "correlated_market_id": "MKT_CORR",
            "v_attack": 40.0,
            "trap_direction": "YES",
        },
    )]


def test_dispatcher_routes_uma_event_to_d3_absorption() -> None:
    router = _RecordingRouter()
    dispatcher = MevDispatcher(router)  # type: ignore[arg-type]

    result = dispatcher.on_uma_dispute_panic(
        DisputeArbitrageSignal(
            market_id="MKT_CLAMP",
            panic_direction="YES",
            limit_price=0.49,
            max_capital=49.0,
        )
    )

    assert result == "d3-batch"
    assert router.calls == [(
        "d3",
        {
            "market_id": "MKT_CLAMP",
            "panic_direction": "YES",
            "limit_price": 0.49,
            "max_capital": 49.0,
        },
    )]


def test_serialize_mev_execution_batch_outputs_clean_json(snapshot_provider) -> None:
    router = MevExecutionRouter(snapshot_provider)
    batch = router.execute_shadow_sweep(
        market_id="MKT_SHADOW",
        direction="YES",
        max_capital=55.0,
        premium_pct=0.03,
    )

    serialized = serialize_mev_execution_batch(batch)
    parsed = json.loads(serialized)

    assert parsed["route_id"] == batch.route_id
    assert parsed["playbook"] == "shadow_sweep"
    assert parsed["payload_count"] == 2
    assert parsed["payloads"][0]["sequence"] == 1
    assert parsed["payloads"][0]["liquidity_intent"] == "TAKER"
    assert parsed["payloads"][0]["price"] == "0.550000"
    assert parsed["payloads"][1]["post_only"] is True
    assert parsed["payloads"][1]["price"] == "0.530000"
    assert parsed["responses"][0]["mock"] is True