from __future__ import annotations

import pytest

from src.signals.uma_dispute_tracker import (
    DisputeArbitrageSignal,
    EthCallUmaStateClient,
    JsonRpcEthCallClient,
    UmaConditionState,
    UmaDisputeTracker,
    UmaMarketState,
)


CONDITION_ID = "0x" + "42" * 32


class _FakeUmaStateClient:
    def __init__(self, responses: list[UmaConditionState]) -> None:
        self._responses = list(responses)

    async def get_condition_state(self, condition_id: str) -> UmaConditionState:
        response = self._responses.pop(0)
        assert response.condition_id == condition_id
        return response


class _StubJsonRpcEthCallClient(JsonRpcEthCallClient):
    def __init__(self, raw_results: list[str]) -> None:
        super().__init__("http://stub.invalid")
        self._raw_results = list(raw_results)
        self.calls: list[tuple[str, str]] = []

    async def eth_call(self, *, to: str, data: str, block: str = "latest") -> str:
        self.calls.append((to, data))
        return self._raw_results.pop(0)


class TestEthCallUmaStateClient:
    @pytest.mark.asyncio
    async def test_decodes_uint_state_from_eth_call(self):
        rpc_client = _StubJsonRpcEthCallClient(["0x" + "0" * 63 + "3"])
        client = EthCallUmaStateClient(
            rpc_client,
            oracle_contract="0x9999000000000000000000000000000000000001",
            state_selector="0x12345678",
        )

        state = await client.get_condition_state(CONDITION_ID)

        assert state.state == UmaMarketState.DISPUTED
        assert state.raw_value == 3
        assert rpc_client.calls[0][1].startswith("0x12345678")


class TestUmaDisputeTracker:
    @pytest.mark.asyncio
    async def test_emits_signal_on_transition_to_disputed_below_trigger(self):
        active = UmaConditionState(
            condition_id=CONDITION_ID,
            state=UmaMarketState.PROPOSED,
            raw_value=2,
            timestamp=1.0,
        )
        disputed = UmaConditionState(
            condition_id=CONDITION_ID,
            state=UmaMarketState.DISPUTED,
            raw_value=3,
            timestamp=2.0,
        )
        tracker = UmaDisputeTracker(
            _FakeUmaStateClient([active, disputed]),
            condition_ids=[CONDITION_ID],
            panic_discount_ewma_alpha=0.5,
        )

        tracker.record_market_price(CONDITION_ID, 0.94)
        signals = await tracker.poll_once({CONDITION_ID: 0.94})
        assert signals == []

        signals = await tracker.poll_once({CONDITION_ID: 0.88})

        assert len(signals) == 1
        signal = signals[0]
        assert isinstance(signal, DisputeArbitrageSignal)
        assert signal.oracle_state == "DISPUTED"
        assert signal.current_price == 0.88
        assert pytest.approx(signal.trigger_price) == 0.94
        assert pytest.approx(signal.panic_discount_ewma) == 0.06

    @pytest.mark.asyncio
    async def test_no_signal_without_disputed_transition(self):
        tracker = UmaDisputeTracker(
            _FakeUmaStateClient([
                UmaConditionState(CONDITION_ID, UmaMarketState.DISPUTED, 3, 1.0),
                UmaConditionState(CONDITION_ID, UmaMarketState.DISPUTED, 3, 2.0),
            ]),
            condition_ids=[CONDITION_ID],
        )

        tracker.record_market_price(CONDITION_ID, 0.85)
        first = await tracker.poll_once({CONDITION_ID: 0.85})
        second = await tracker.poll_once({CONDITION_ID: 0.80})

        assert len(first) == 1
        assert second == []

    @pytest.mark.asyncio
    async def test_no_signal_when_price_is_above_trigger(self):
        tracker = UmaDisputeTracker(
            _FakeUmaStateClient([
                UmaConditionState(CONDITION_ID, UmaMarketState.PROPOSED, 2, 1.0),
                UmaConditionState(CONDITION_ID, UmaMarketState.DISPUTED, 3, 2.0),
            ]),
            condition_ids=[CONDITION_ID],
        )

        tracker.record_market_price(CONDITION_ID, 0.90)
        await tracker.poll_once({CONDITION_ID: 0.90})
        signals = await tracker.poll_once({CONDITION_ID: 0.95})

        assert signals == []

    def test_record_market_price_updates_discount_ewma(self):
        tracker = UmaDisputeTracker(
            _FakeUmaStateClient([]),
            condition_ids=[CONDITION_ID],
            panic_discount_ewma_alpha=0.25,
        )

        first = tracker.record_market_price(CONDITION_ID, 0.92)
        second = tracker.record_market_price(CONDITION_ID, 0.80)

        assert pytest.approx(first) == 0.08
        assert pytest.approx(second) == 0.11
