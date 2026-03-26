from __future__ import annotations

import pytest

from src.events.mev_events import ShadowSweepSignal
from src.execution.mempool_monitor import MempoolMonitor, POLYMARKET_CTF_CONTRACT
from src.execution.mev_paper_adapter import MevPaperAdapter
from src.signals.dispute_arbitrage_detector import DisputeArbitrageDetector
from src.signals.shadow_sweep_detector import ShadowSweepDetector
from src.signals.uma_dispute_tracker import UmaConditionState, UmaDisputeTracker, UmaMarketState


CONDITION_ID = "0x" + "42" * 32


class _FakeUmaStateClient:
    def __init__(self, responses: list[UmaConditionState]) -> None:
        self._responses = list(responses)

    async def get_condition_state(self, condition_id: str) -> UmaConditionState:
        response = self._responses.pop(0)
        assert response.condition_id == condition_id
        return response


def _encode_uint256(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _encode_address(address: str) -> bytes:
    raw = bytes.fromhex(address.lower().replace("0x", ""))
    return b"\x00" * 12 + raw


def _make_split_position_input(collateral_token: str, condition_id_hex: str, amount_raw: int) -> str:
    selector = bytes.fromhex("c9ff79aa")
    parent_collection_id = b"\x00" * 32
    condition_id = bytes.fromhex(condition_id_hex.lower().replace("0x", ""))
    partition_offset = _encode_uint256(32 * 5)
    amount = _encode_uint256(amount_raw)
    partition = _encode_uint256(2) + _encode_uint256(1) + _encode_uint256(2)
    return "0x" + (
        selector
        + _encode_address(collateral_token)
        + parent_collection_id
        + condition_id
        + partition_offset
        + amount
        + partition
    ).hex()


class _StubPendingTxClient:
    async def subscribe_pending_transactions(self):
        if False:
            yield ""

    async def get_transaction_by_hash(self, tx_hash: str):
        return None

    async def close(self) -> None:
        return None


class _StubShadowDetector:
    def __init__(self) -> None:
        self.callback = None

    def register_callback(self, callback):
        self.callback = callback


class _RecordingDispatcher:
    def __init__(self) -> None:
        self.shadow_calls: list[object] = []
        self.d3_calls: list[object] = []

    def on_mempool_whale_detected(self, event):
        self.shadow_calls.append(event)
        return {"playbook": "shadow_sweep", "market_id": event.target_market_id}

    def on_uma_dispute_panic(self, event):
        self.d3_calls.append(event)
        return {"playbook": "d3_panic_absorption", "market_id": event.market_id}


def test_mev_paper_adapter_forwards_shadow_sweep_signal_into_dispatcher() -> None:
    monitor = MempoolMonitor(
        _StubPendingTxClient(),
        volume_threshold=100_000.0,
        clock=lambda: 1000.0,
    )
    shadow_detector = ShadowSweepDetector(
        monitor,
        direction_resolver=lambda match: "NO",
        max_capital=75.0,
        premium_pct=0.04,
    )
    tracker = UmaDisputeTracker(_FakeUmaStateClient([]), condition_ids=[CONDITION_ID])
    d3_detector = DisputeArbitrageDetector(tracker)
    dispatcher = _RecordingDispatcher()
    adapter = MevPaperAdapter(dispatcher)  # type: ignore[arg-type]
    adapter.register_detectors(shadow_detector, d3_detector)

    signal = shadow_detector.ingest_transaction(
        {
            "hash": "0xshadow",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": POLYMARKET_CTF_CONTRACT,
            "input": _make_split_position_input(
                "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                CONDITION_ID,
                150_000_000_000,
            ),
        },
        seen_at=1000.0,
    )

    assert isinstance(signal, ShadowSweepSignal)
    assert signal.target_market_id == CONDITION_ID
    assert signal.direction == "NO"
    assert signal.max_capital == 75.0
    assert signal.premium_pct == 0.04
    assert len(dispatcher.shadow_calls) == 1
    forwarded = dispatcher.shadow_calls[0]
    assert isinstance(forwarded, ShadowSweepSignal)
    assert forwarded == signal
    assert adapter.shadow_dispatch_results == [{"playbook": "shadow_sweep", "market_id": CONDITION_ID}]


@pytest.mark.asyncio
async def test_mev_paper_adapter_forwards_d3_signal_into_dispatcher() -> None:
    tracker = UmaDisputeTracker(
        _FakeUmaStateClient([
            UmaConditionState(CONDITION_ID, UmaMarketState.PROPOSED, 2, 1.0),
            UmaConditionState(CONDITION_ID, UmaMarketState.DISPUTED, 3, 2.0),
        ]),
        condition_ids=[CONDITION_ID],
        panic_discount_ewma_alpha=0.5,
    )
    detector = DisputeArbitrageDetector(
        tracker,
        max_capital=50.0,
        min_panic_discount=0.12,
        panic_direction="YES",
    )
    dispatcher = _RecordingDispatcher()
    adapter = MevPaperAdapter(dispatcher)  # type: ignore[arg-type]
    adapter.register_detectors(_StubShadowDetector(), detector)  # type: ignore[arg-type]

    tracker.record_market_price(CONDITION_ID, 0.94)
    await detector.poll_once({CONDITION_ID: 0.94})
    result = await detector.poll_once({CONDITION_ID: 0.88})

    assert len(result) == 1
    assert len(dispatcher.d3_calls) == 1
    forwarded = dispatcher.d3_calls[0]
    assert forwarded.market_id == CONDITION_ID
    assert forwarded.panic_direction == "YES"
    assert forwarded.limit_price == 0.88
    assert forwarded.max_capital == 50.0
    assert adapter.d3_dispatch_results == [{"playbook": "d3_panic_absorption", "market_id": CONDITION_ID}]


def test_mev_paper_adapter_registers_shadow_callback() -> None:
    dispatcher = _RecordingDispatcher()
    adapter = MevPaperAdapter(dispatcher)  # type: ignore[arg-type]
    shadow_detector = _StubShadowDetector()

    tracker = UmaDisputeTracker(_FakeUmaStateClient([]), condition_ids=[CONDITION_ID])
    d3_detector = DisputeArbitrageDetector(tracker)
    adapter.register_detectors(shadow_detector, d3_detector)  # type: ignore[arg-type]

    assert shadow_detector.callback is not None