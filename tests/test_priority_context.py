from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.priority_context import PriorityOrderContext, RewardExecutionHints


def _reward_execution_hints(**overrides: object) -> RewardExecutionHints:
    payload = {
        "post_only": True,
        "time_in_force": "GTC",
        "liquidity_intent": "MAKER_REWARD",
        "allow_taker_escalation": False,
        "quote_id": "quote-123",
        "tick_size": Decimal("0.01"),
        "cancel_on_stale_ms": 500,
        "replace_only_if_price_moves_ticks": 2,
        "metadata": {"origin": "reward-sidecar"},
    }
    payload.update(overrides)
    return RewardExecutionHints(**payload)


def test_reward_execution_hints_validate_fail_closed_contract() -> None:
    hints = _reward_execution_hints()

    assert hints.post_only is True
    assert hints.time_in_force == "GTC"
    assert hints.liquidity_intent == "MAKER_REWARD"
    assert hints.allow_taker_escalation is False
    assert hints.quote_id == "quote-123"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"post_only": False}, "post_only"),
        ({"time_in_force": "IOC"}, "time_in_force"),
        ({"liquidity_intent": "MAKER"}, "liquidity_intent"),
        ({"allow_taker_escalation": True}, "allow_taker_escalation"),
        ({"quote_id": ""}, "quote_id"),
        ({"tick_size": Decimal("0")}, "tick_size"),
        ({"cancel_on_stale_ms": 0}, "cancel_on_stale_ms"),
        ({"replace_only_if_price_moves_ticks": 0}, "replace_only_if_price_moves_ticks"),
    ],
)
def test_reward_execution_hints_reject_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _reward_execution_hints(**overrides)


def test_reward_context_accepts_valid_contract_shape() -> None:
    context = PriorityOrderContext(
        market_id="MKT_REWARD",
        side="YES",
        signal_source="REWARD",
        conviction_scalar=Decimal("1"),
        target_price=Decimal("0.25"),
        anchor_volume=Decimal("4"),
        max_capital=Decimal("1.0000"),
        execution_hints=_reward_execution_hints(),
        signal_metadata={"campaign": "march"},
    )

    assert context.execution_hints is not None
    assert dict(context.signal_metadata) == {"campaign": "march"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"execution_hints": None}, "execution_hints"),
        ({"conviction_scalar": Decimal("0.9")}, "conviction_scalar"),
        ({"leg_role": "YES_LEG"}, "leg_role"),
        ({"max_capital": Decimal("1.0100")}, "max_capital"),
    ],
)
def test_reward_context_rejects_malformed_reward_contract(
    overrides: dict[str, object],
    message: str,
) -> None:
    payload = {
        "market_id": "MKT_REWARD",
        "side": "YES",
        "signal_source": "REWARD",
        "conviction_scalar": Decimal("1"),
        "target_price": Decimal("0.25"),
        "anchor_volume": Decimal("4"),
        "max_capital": Decimal("1.0000"),
        "execution_hints": _reward_execution_hints(),
        "signal_metadata": {},
    }
    payload.update(overrides)

    with pytest.raises(ValueError, match=message):
        PriorityOrderContext(**payload)


def test_non_reward_context_still_accepts_legacy_shape_without_reward_fields() -> None:
    context = PriorityOrderContext(
        market_id="MKT_OFI",
        side="NO",
        signal_source="OFI",
        conviction_scalar=Decimal("0.5"),
        target_price=Decimal("0.45"),
        anchor_volume=Decimal("10"),
        max_capital=Decimal("4.5"),
    )

    assert context.execution_hints is None
    assert dict(context.signal_metadata) == {}