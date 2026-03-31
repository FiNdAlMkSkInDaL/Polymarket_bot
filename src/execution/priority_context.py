from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Mapping


PrioritySide = Literal["YES", "NO"]
PrioritySignalSource = Literal["OFI", "SI9", "SI10", "CONTAGION", "MANUAL", "CTF", "REWARD"]


_REWARD_CAPITAL_QUANTUM = Decimal("0.0001")


def _require_decimal(name: str, value: object) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{name} must be a Decimal; got {type(value).__name__}")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite; got {value!r}")
    return value


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive; got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class RewardExecutionHints:
    post_only: bool
    time_in_force: Literal["GTC", "IOC", "FOK"]
    liquidity_intent: str
    allow_taker_escalation: bool
    quote_id: str
    tick_size: Decimal
    cancel_on_stale_ms: int
    replace_only_if_price_moves_ticks: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.post_only is not True:
            raise ValueError("post_only must be True")
        if self.time_in_force != "GTC":
            raise ValueError("time_in_force must be 'GTC'")
        if self.liquidity_intent != "MAKER_REWARD":
            raise ValueError("liquidity_intent must be 'MAKER_REWARD'")
        if self.allow_taker_escalation is not False:
            raise ValueError("allow_taker_escalation must be False")

        quote_id = str(self.quote_id or "").strip()
        if not quote_id:
            raise ValueError("quote_id must be a non-empty string")
        object.__setattr__(self, "quote_id", quote_id)

        tick_size = _require_decimal("tick_size", self.tick_size)
        if tick_size <= Decimal("0"):
            raise ValueError(f"tick_size must be strictly positive; got {tick_size!r}")

        _require_positive_int("cancel_on_stale_ms", self.cancel_on_stale_ms)
        replace_ticks = _require_positive_int(
            "replace_only_if_price_moves_ticks",
            self.replace_only_if_price_moves_ticks,
        )
        if replace_ticks < 1:
            raise ValueError("replace_only_if_price_moves_ticks must be greater than or equal to 1")

        metadata = dict(_require_mapping("metadata", self.metadata))
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class PriorityOrderContext:
    market_id: str
    side: PrioritySide
    signal_source: PrioritySignalSource
    conviction_scalar: Decimal
    target_price: Decimal
    anchor_volume: Decimal
    max_capital: Decimal
    leg_role: str | None = None
    execution_hints: RewardExecutionHints | None = None
    signal_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        market_id = str(self.market_id or "").strip()
        if not market_id:
            raise ValueError("market_id must be a non-empty string")
        object.__setattr__(self, "market_id", market_id)

        if self.side not in {"YES", "NO"}:
            raise ValueError(f"side must be 'YES' or 'NO'; got {self.side!r}")
        if self.signal_source not in {"OFI", "SI9", "SI10", "CONTAGION", "MANUAL", "CTF", "REWARD"}:
            raise ValueError(
                "signal_source must be one of 'OFI', 'SI9', 'SI10', 'CONTAGION', 'MANUAL', 'CTF', or 'REWARD'; "
                f"got {self.signal_source!r}"
            )
        if self.leg_role is not None and self.leg_role not in {"YES_LEG", "NO_LEG"}:
            raise ValueError(f"leg_role must be 'YES_LEG', 'NO_LEG', or None; got {self.leg_role!r}")

        conviction_scalar = _require_decimal("conviction_scalar", self.conviction_scalar)
        if conviction_scalar < Decimal("0.0") or conviction_scalar > Decimal("1.0"):
            raise ValueError(
                f"conviction_scalar must be within [0.0, 1.0]; got {conviction_scalar!r}"
            )

        for field_name in ("target_price", "anchor_volume", "max_capital"):
            value = _require_decimal(field_name, getattr(self, field_name))
            if value <= Decimal("0"):
                raise ValueError(f"{field_name} must be strictly positive; got {value!r}")

        signal_metadata = dict(_require_mapping("signal_metadata", self.signal_metadata))
        object.__setattr__(self, "signal_metadata", signal_metadata)

        execution_hints = self.execution_hints
        if execution_hints is not None and not isinstance(execution_hints, RewardExecutionHints):
            raise ValueError("execution_hints must be a RewardExecutionHints instance or None")

        if self.signal_source == "REWARD":
            if execution_hints is None:
                raise ValueError("execution_hints is required when signal_source='REWARD'")
            if conviction_scalar != Decimal("1"):
                raise ValueError("REWARD contexts require conviction_scalar == Decimal('1')")
            if self.leg_role is not None:
                raise ValueError("REWARD contexts require leg_role to be None")

            expected_max_capital = (self.target_price * self.anchor_volume).quantize(_REWARD_CAPITAL_QUANTUM)
            actual_max_capital = self.max_capital.quantize(_REWARD_CAPITAL_QUANTUM)
            if actual_max_capital != expected_max_capital:
                raise ValueError(
                    "REWARD contexts require max_capital == target_price * anchor_volume to 4 decimal places"
                )