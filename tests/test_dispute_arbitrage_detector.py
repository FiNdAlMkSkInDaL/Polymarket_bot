from __future__ import annotations

import pytest

from src.events.mev_events import DisputeArbitrageSignal
from src.signals.dispute_arbitrage_detector import DisputeArbitrageDetector
from src.signals.uma_dispute_tracker import UmaConditionState, UmaDisputeTracker, UmaMarketState


CONDITION_ID = "0x" + "42" * 32


class _FakeUmaStateClient:
    def __init__(self, responses: list[UmaConditionState]) -> None:
        self._responses = list(responses)

    async def get_condition_state(self, condition_id: str) -> UmaConditionState:
        response = self._responses.pop(0)
        assert response.condition_id == condition_id
        return response


@pytest.mark.asyncio
async def test_detector_emits_strict_dispute_event_on_disputed_transition_and_panic_drop() -> None:
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
        max_capital=75.0,
        min_panic_discount=0.12,
        panic_direction="YES",
    )

    tracker.record_market_price(CONDITION_ID, 0.94)
    first = await detector.poll_once({CONDITION_ID: 0.94})
    second = await detector.poll_once({CONDITION_ID: 0.88})

    assert first == []
    assert len(second) == 1
    signal = second[0]
    assert isinstance(signal, DisputeArbitrageSignal)
    assert signal.market_id == CONDITION_ID
    assert signal.panic_direction == "YES"
    assert signal.limit_price == 0.88
    assert signal.max_capital == 75.0


@pytest.mark.asyncio
async def test_detector_blocks_tracker_signal_when_configured_panic_threshold_not_met() -> None:
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
        min_panic_discount=0.15,
    )

    tracker.record_market_price(CONDITION_ID, 0.94)
    await detector.poll_once({CONDITION_ID: 0.94})
    result = await detector.poll_once({CONDITION_ID: 0.88})

    assert result == []


def test_translate_signals_skips_missing_market_ids() -> None:
    tracker = UmaDisputeTracker(_FakeUmaStateClient([]), condition_ids=[CONDITION_ID])
    detector = DisputeArbitrageDetector(tracker)

    class _IncompleteSignal:
        market_id = ""
        current_price = 0.80

    assert detector.translate_signals([_IncompleteSignal()]) == []
