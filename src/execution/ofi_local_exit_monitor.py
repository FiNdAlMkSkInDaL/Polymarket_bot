from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider


OfiExitAction = Literal[
    "HOLD",
    "TARGET_HIT",
    "STOP_HIT",
    "TIME_STOP_TRIGGERED",
    "SUPPRESSED_BY_VACUUM",
]


@dataclass(frozen=True, slots=True)
class OfiExitDecision:
    action: OfiExitAction
    trigger_price: Decimal

    def __post_init__(self) -> None:
        if self.action not in {
            "HOLD",
            "TARGET_HIT",
            "STOP_HIT",
            "TIME_STOP_TRIGGERED",
            "SUPPRESSED_BY_VACUUM",
        }:
            raise ValueError(f"Unsupported action: {self.action!r}")
        if not isinstance(self.trigger_price, Decimal) or not self.trigger_price.is_finite():
            raise ValueError("trigger_price must be a finite Decimal")
        if self.trigger_price < Decimal("0"):
            raise ValueError("trigger_price cannot be negative")


class OfiLocalExitMonitor:
    def __init__(
        self,
        best_bid_provider: OrderbookBestBidProvider,
        *,
        vacuum_ratio: Decimal = Decimal("0.35"),
        spread_multiple: Decimal = Decimal("1.75"),
    ) -> None:
        if best_bid_provider is None:
            raise ValueError("best_bid_provider is required")
        if not isinstance(vacuum_ratio, Decimal) or not vacuum_ratio.is_finite() or vacuum_ratio <= Decimal("0"):
            raise ValueError("vacuum_ratio must be a strictly positive Decimal")
        if not isinstance(spread_multiple, Decimal) or not spread_multiple.is_finite() or spread_multiple <= Decimal("0"):
            raise ValueError("spread_multiple must be a strictly positive Decimal")
        self._best_bid_provider = best_bid_provider
        self._vacuum_ratio = vacuum_ratio
        self._spread_multiple = spread_multiple

    def evaluate_exit(self, position_state: dict, current_timestamp_ms: int) -> OfiExitDecision:
        market_id = self._market_id(position_state)
        current_best_bid = self._best_bid_provider.get_best_bid(market_id) or Decimal("0")
        drawn_tp = self._decimal_field(position_state, "drawn_tp")
        if drawn_tp > Decimal("0") and current_best_bid > Decimal("0") and current_best_bid >= drawn_tp:
            return OfiExitDecision(action="TARGET_HIT", trigger_price=drawn_tp)
        drawn_stop = self._decimal_field(position_state, "drawn_stop")
        if drawn_stop > Decimal("0") and current_best_bid > Decimal("0") and current_best_bid <= drawn_stop:
            return OfiExitDecision(action="STOP_HIT", trigger_price=drawn_stop)
        drawn_time_ms = self._deadline_ms(position_state)
        if int(current_timestamp_ms) <= drawn_time_ms:
            return OfiExitDecision(action="HOLD", trigger_price=current_best_bid)
        return self._evaluate_time_stop(position_state, market_id, current_best_bid)

    def _evaluate_time_stop(
        self,
        position_state: dict,
        market_id: str,
        current_best_bid: Decimal,
    ) -> OfiExitDecision:
        current_bid_depth = self._best_bid_provider.get_top_depth(market_id, "bid") or Decimal("0")
        current_ask_depth = self._best_bid_provider.get_top_depth(market_id, "ask") or Decimal("0")
        bid_depth_baseline = self._best_bid_provider.get_top_depth_ewma(market_id, "bid") or current_bid_depth
        ask_depth_baseline = self._best_bid_provider.get_top_depth_ewma(market_id, "ask") or current_ask_depth
        current_spread = self._best_bid_provider.get_spread(market_id) or Decimal("0")
        baseline_spread = self._baseline_spread(position_state)
        bid_vacuum = (
            current_bid_depth > Decimal("0")
            and bid_depth_baseline > Decimal("0")
            and current_bid_depth < bid_depth_baseline * self._vacuum_ratio
        )
        ask_vacuum = (
            current_ask_depth > Decimal("0")
            and ask_depth_baseline > Decimal("0")
            and current_ask_depth < ask_depth_baseline * self._vacuum_ratio
        )
        spread_blown_out = current_spread > baseline_spread * self._spread_multiple
        if (bid_vacuum or ask_vacuum) and spread_blown_out:
            return OfiExitDecision(action="SUPPRESSED_BY_VACUUM", trigger_price=current_best_bid)
        return OfiExitDecision(action="TIME_STOP_TRIGGERED", trigger_price=current_best_bid)

    @staticmethod
    def _market_id(position_state: dict) -> str:
        market_id = str(position_state.get("market_id", "") or "").strip()
        if not market_id:
            raise ValueError("position_state.market_id must be a non-empty string")
        return market_id

    @staticmethod
    def _decimal_field(position_state: dict, field_name: str) -> Decimal:
        value = position_state.get(field_name, Decimal("0"))
        if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
            raise ValueError(f"position_state.{field_name} must be a non-negative finite Decimal")
        return value

    def _baseline_spread(self, position_state: dict) -> Decimal:
        baseline_spread = position_state.get("baseline_spread")
        if baseline_spread is not None:
            if not isinstance(baseline_spread, Decimal) or not baseline_spread.is_finite() or baseline_spread < Decimal("0"):
                raise ValueError("position_state.baseline_spread must be a non-negative finite Decimal")
            return max(Decimal("0.01"), baseline_spread)
        return Decimal("0.01")

    @staticmethod
    def _deadline_ms(position_state: dict) -> int:
        drawn_time_ms = position_state.get("drawn_time_ms")
        if not isinstance(drawn_time_ms, int) or drawn_time_ms <= 0:
            raise ValueError("position_state.drawn_time_ms must be a strictly positive int")
        return drawn_time_ms