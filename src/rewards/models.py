from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from src.execution.priority_context import PriorityOrderContext, RewardExecutionHints


_CAPITAL_QUANTUM = Decimal("0.0001")
_TICK_SIZE = Decimal("0.01")


def _to_decimal(name: str, value: object) -> Decimal:
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


@dataclass(frozen=True, slots=True)
class RewardPosterIntent:
    market_id: str
    asset_id: str
    side: str
    reference_mid_price: Decimal
    target_price: Decimal
    target_size: Decimal
    max_capital: Decimal
    quote_id: str
    reward_program: str
    reward_daily_rate_usd: Decimal
    reward_to_competition: Decimal
    competition_score: Decimal
    reward_max_spread_cents: Decimal
    cancel_on_stale_ms: int
    replace_only_if_price_moves_ticks: int
    extra_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("market_id", "asset_id", "quote_id", "reward_program"):
            normalized = str(getattr(self, field_name) or "").strip()
            if not normalized:
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, normalized)

        if self.side not in {"YES", "NO"}:
            raise ValueError("side must be 'YES' or 'NO'")

        for field_name in (
            "reference_mid_price",
            "target_price",
            "target_size",
            "max_capital",
            "reward_daily_rate_usd",
            "reward_to_competition",
            "competition_score",
            "reward_max_spread_cents",
        ):
            decimal_value = _to_decimal(field_name, getattr(self, field_name))
            if field_name in {"reference_mid_price", "target_price", "target_size", "max_capital"}:
                if decimal_value <= Decimal("0"):
                    raise ValueError(f"{field_name} must be strictly positive")
            elif decimal_value < Decimal("0"):
                raise ValueError(f"{field_name} must be greater than or equal to 0")
            object.__setattr__(self, field_name, decimal_value)

        expected_capital = (self.target_price * self.target_size).quantize(_CAPITAL_QUANTUM, rounding=ROUND_HALF_UP)
        actual_capital = self.max_capital.quantize(_CAPITAL_QUANTUM, rounding=ROUND_HALF_UP)
        if actual_capital != expected_capital:
            raise ValueError("max_capital must equal target_price * target_size to 4 decimal places")

        if not isinstance(self.cancel_on_stale_ms, int) or self.cancel_on_stale_ms <= 0:
            raise ValueError("cancel_on_stale_ms must be a strictly positive int")
        if not isinstance(self.replace_only_if_price_moves_ticks, int) or self.replace_only_if_price_moves_ticks < 1:
            raise ValueError("replace_only_if_price_moves_ticks must be >= 1")

        object.__setattr__(self, "extra_payload", dict(self.extra_payload))

    def build_execution_hints(self) -> RewardExecutionHints:
        return RewardExecutionHints(
            post_only=True,
            time_in_force="GTC",
            liquidity_intent="MAKER_REWARD",
            allow_taker_escalation=False,
            quote_id=self.quote_id,
            tick_size=_TICK_SIZE,
            cancel_on_stale_ms=self.cancel_on_stale_ms,
            replace_only_if_price_moves_ticks=self.replace_only_if_price_moves_ticks,
            metadata=self.as_signal_metadata(),
        )

    def as_signal_metadata(self) -> dict[str, Any]:
        return {
            "signal_source": "reward_sidecar",
            "quote_id": self.quote_id,
            "asset_id": self.asset_id,
            "reference_mid_price": str(self.reference_mid_price),
            "reward_program": self.reward_program,
            "reward_daily_rate_usd": str(self.reward_daily_rate_usd),
            "reward_to_competition": str(self.reward_to_competition),
            "competition_score": str(self.competition_score),
            "reward_max_spread_cents": str(self.reward_max_spread_cents),
            "extra_payload": dict(self.extra_payload),
        }

    def to_priority_context(self) -> PriorityOrderContext:
        return PriorityOrderContext(
            market_id=self.market_id,
            side=self.side,
            signal_source="REWARD",
            conviction_scalar=Decimal("1"),
            target_price=self.target_price,
            anchor_volume=self.target_size,
            max_capital=self.max_capital,
            execution_hints=self.build_execution_hints(),
            signal_metadata=self.as_signal_metadata(),
        )