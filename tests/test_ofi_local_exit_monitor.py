from __future__ import annotations

import dis
from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.execution.ofi_local_exit_monitor import OfiExitDecision, OfiLocalExitMonitor
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider


class _StubTracker:
    def __init__(
        self,
        *,
        asset_id: str = "asset-1",
        best_bid: float = 0.50,
        best_ask: float = 0.52,
        bid_depth_usd: float = 120.0,
        ask_depth_usd: float = 150.0,
        bid_depth_ewma: float = 200.0,
        ask_depth_ewma: float = 220.0,
    ) -> None:
        self.asset_id = asset_id
        self.best_bid = best_bid
        self.best_ask = best_ask
        self._bid_depth_usd = bid_depth_usd
        self._ask_depth_usd = ask_depth_usd
        self._bid_depth_ewma = bid_depth_ewma
        self._ask_depth_ewma = ask_depth_ewma

    def snapshot(self):
        return SimpleNamespace(
            timestamp=1712345.678,
            best_ask=self.best_ask,
            bid_depth_usd=self._bid_depth_usd,
            ask_depth_usd=self._ask_depth_usd,
        )

    def top_depths_usd(self) -> tuple[float, float]:
        return self._bid_depth_usd, self._ask_depth_usd

    def top_depth_ewma(self, side: str) -> float:
        return self._bid_depth_ewma if side == "bid" else self._ask_depth_ewma


def _monitor(**tracker_kwargs: float) -> OfiLocalExitMonitor:
    return OfiLocalExitMonitor(OrderbookBestBidProvider(_StubTracker(**tracker_kwargs)))


def _state(**overrides: object) -> dict:
    state = {
        "market_id": "asset-1",
        "drawn_tp": Decimal("0.55"),
        "drawn_stop": Decimal("0.48"),
        "drawn_time_ms": 1_000,
        "baseline_spread": Decimal("0.02"),
    }
    state.update(overrides)
    return state


def test_exit_decision_is_frozen() -> None:
    decision = OfiExitDecision(action="HOLD", trigger_price=Decimal("0.50"))

    with pytest.raises(FrozenInstanceError):
        decision.action = "STOP_HIT"  # type: ignore[misc]


def test_constructor_requires_provider() -> None:
    with pytest.raises(ValueError, match="best_bid_provider"):
        OfiLocalExitMonitor(None)  # type: ignore[arg-type]


def test_target_hit_returns_drawn_target_price() -> None:
    decision = _monitor(best_bid=0.56).evaluate_exit(_state(), 900)

    assert decision.action == "TARGET_HIT"
    assert decision.trigger_price == Decimal("0.55")


def test_target_precedes_stop_when_thresholds_overlap() -> None:
    decision = _monitor(best_bid=0.50).evaluate_exit(
        _state(drawn_tp=Decimal("0.50"), drawn_stop=Decimal("0.50")),
        900,
    )

    assert decision.action == "TARGET_HIT"


def test_stop_hit_returns_drawn_stop_price() -> None:
    decision = _monitor(best_bid=0.47).evaluate_exit(_state(), 900)

    assert decision.action == "STOP_HIT"
    assert decision.trigger_price == Decimal("0.48")


def test_stop_precedes_time_stop_after_deadline() -> None:
    decision = _monitor(best_bid=0.47).evaluate_exit(_state(), 1_500)

    assert decision.action == "STOP_HIT"


def test_target_precedes_time_stop_after_deadline() -> None:
    decision = _monitor(best_bid=0.56).evaluate_exit(_state(), 1_500)

    assert decision.action == "TARGET_HIT"


def test_hold_before_deadline_when_no_price_boundary_reached() -> None:
    decision = _monitor(best_bid=0.50).evaluate_exit(_state(), 999)

    assert decision.action == "HOLD"
    assert decision.trigger_price == Decimal("0.5")


def test_exact_time_boundary_does_not_trigger_time_stop() -> None:
    decision = _monitor(best_bid=0.50).evaluate_exit(_state(), 1_000)

    assert decision.action == "HOLD"


def test_time_stop_triggers_after_deadline_when_book_is_healthy() -> None:
    decision = _monitor(best_bid=0.50, best_ask=0.52, bid_depth_usd=180.0, ask_depth_usd=190.0).evaluate_exit(
        _state(),
        1_001,
    )

    assert decision.action == "TIME_STOP_TRIGGERED"
    assert decision.trigger_price == Decimal("0.5")


def test_time_stop_is_suppressed_when_bid_depth_collapses_and_spread_blows_out() -> None:
    decision = _monitor(
        best_bid=0.50,
        best_ask=0.55,
        bid_depth_usd=40.0,
        ask_depth_usd=190.0,
        bid_depth_ewma=200.0,
        ask_depth_ewma=220.0,
    ).evaluate_exit(_state(baseline_spread=Decimal("0.02")), 1_001)

    assert decision.action == "SUPPRESSED_BY_VACUUM"


def test_time_stop_is_suppressed_when_ask_depth_collapses_and_spread_blows_out() -> None:
    decision = _monitor(
        best_bid=0.50,
        best_ask=0.55,
        bid_depth_usd=180.0,
        ask_depth_usd=30.0,
        bid_depth_ewma=200.0,
        ask_depth_ewma=220.0,
    ).evaluate_exit(_state(baseline_spread=Decimal("0.02")), 1_001)

    assert decision.action == "SUPPRESSED_BY_VACUUM"


def test_time_stop_is_not_suppressed_when_spread_is_not_blown_out() -> None:
    decision = _monitor(
        best_bid=0.50,
        best_ask=0.53,
        bid_depth_usd=40.0,
        bid_depth_ewma=200.0,
    ).evaluate_exit(_state(baseline_spread=Decimal("0.02")), 1_001)

    assert decision.action == "TIME_STOP_TRIGGERED"


def test_time_stop_is_not_suppressed_when_depth_is_healthy_despite_wide_spread() -> None:
    decision = _monitor(
        best_bid=0.50,
        best_ask=0.56,
        bid_depth_usd=120.0,
        ask_depth_usd=150.0,
        bid_depth_ewma=200.0,
        ask_depth_ewma=220.0,
    ).evaluate_exit(_state(baseline_spread=Decimal("0.02")), 1_001)

    assert decision.action == "TIME_STOP_TRIGGERED"


def test_missing_ewma_baselines_fall_back_to_current_depth_and_do_not_suppress() -> None:
    decision = _monitor(
        best_bid=0.50,
        best_ask=0.56,
        bid_depth_usd=40.0,
        ask_depth_usd=50.0,
        bid_depth_ewma=0.0,
        ask_depth_ewma=0.0,
    ).evaluate_exit(_state(baseline_spread=Decimal("0.02")), 1_001)

    assert decision.action == "TIME_STOP_TRIGGERED"


def test_missing_best_bid_before_deadline_holds_with_zero_trigger_price() -> None:
    decision = _monitor(best_bid=0.0).evaluate_exit(_state(), 999)

    assert decision.action == "HOLD"
    assert decision.trigger_price == Decimal("0")


def test_time_stop_without_book_signal_still_triggers_after_deadline() -> None:
    decision = _monitor(best_bid=0.0, best_ask=0.0, bid_depth_usd=0.0, ask_depth_usd=0.0).evaluate_exit(_state(), 1_001)

    assert decision.action == "TIME_STOP_TRIGGERED"
    assert decision.trigger_price == Decimal("0")


def test_evaluate_exit_requires_absolute_deadline_ms() -> None:
    with pytest.raises(ValueError, match="drawn_time_ms"):
        _monitor().evaluate_exit(_state(drawn_time_ms=0), 1_000)


def test_evaluate_exit_requires_decimal_thresholds() -> None:
    with pytest.raises(ValueError, match="drawn_tp"):
        _monitor().evaluate_exit(_state(drawn_tp=0.55), 900)


def test_monitor_uses_no_list_building_opcodes_in_hot_path() -> None:
    evaluate_opnames = {instruction.opname for instruction in dis.get_instructions(OfiLocalExitMonitor.evaluate_exit)}
    time_stop_opnames = {instruction.opname for instruction in dis.get_instructions(OfiLocalExitMonitor._evaluate_time_stop)}

    assert "BUILD_LIST" not in evaluate_opnames
    assert "BUILD_LIST" not in time_stop_opnames
    assert "LIST_APPEND" not in evaluate_opnames
    assert "LIST_APPEND" not in time_stop_opnames