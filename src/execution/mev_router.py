"""Isolated order-generation router for crypto-native MEV-style plays."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import count
from typing import Any, Callable, Mapping

from src.execution.priority_context import PriorityOrderContext


_PRIORITY_PRICE_EPSILON = Decimal("0.000001")
_NATIVE_PRICE_QUANTUM = Decimal("0.000001")
_NATIVE_PRICE_MIN = Decimal("0.01")
_NATIVE_PRICE_MAX = Decimal("0.99")


def _normalize_direction(direction: str) -> str:
    value = str(direction or "").strip().upper()
    if value not in {"YES", "NO"}:
        raise ValueError(f"Unsupported MEV direction: {direction!r}")
    return value


def _to_decimal(value: float | str | Decimal) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Unsupported decimal value: {value!r}") from exc


def _quantize_native_price(price: float | str | Decimal) -> Decimal:
    return _to_decimal(price).quantize(_NATIVE_PRICE_QUANTUM)


def _normalize_native_price(price: float | str | Decimal) -> Decimal:
    bounded = min(_NATIVE_PRICE_MAX, max(_NATIVE_PRICE_MIN, _to_decimal(price)))
    return _quantize_native_price(bounded)


def format_native_price(price: float | str | Decimal) -> str:
    return format(_quantize_native_price(price), ".6f")


def _round_price(price: float) -> float:
    return round(min(0.99, max(0.01, float(price))), 2)


def _round_size(size: float) -> float:
    rounded = round(float(size), 4)
    if rounded <= 0.0:
        raise ValueError("Order size must be strictly positive")
    return rounded


def _round_size_allow_zero(size: float) -> float:
    rounded = round(float(size), 4)
    if rounded < 0.0:
        raise ValueError("Order size cannot be negative")
    return rounded


@dataclass(frozen=True, slots=True)
class MevMarketSnapshot:
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    def best_bid(self, direction: str) -> float:
        normalized = _normalize_direction(direction)
        return float(self.yes_bid if normalized == "YES" else self.no_bid)

    def best_ask(self, direction: str) -> float:
        normalized = _normalize_direction(direction)
        return float(self.yes_ask if normalized == "YES" else self.no_ask)

    def mid_price(self, direction: str) -> float:
        normalized = _normalize_direction(direction)
        best_bid = self.best_bid(normalized)
        best_ask = self.best_ask(normalized)
        if best_bid <= 0.0 or best_ask <= 0.0:
            raise ValueError(f"Missing two-sided market for {normalized}")
        return (best_bid + best_ask) / 2.0


@dataclass(frozen=True, slots=True)
class MevOrderPayload:
    route_id: str
    sequence: int
    playbook: str
    market_id: str
    direction: str
    side: str
    price: float
    size: float
    liquidity_intent: str
    time_in_force: str
    post_only: bool
    context: PriorityOrderContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MevExecutionBatch:
    route_id: str
    playbook: str
    payloads: tuple[MevOrderPayload, ...]
    responses: tuple[dict[str, Any], ...]


class MevExecutionRouter:
    """Generate and mock-submit sequenced MEV execution payloads.

    The router is intentionally isolated from the live order stack. It accepts a
    market snapshot provider, builds deterministic payloads, and records mock
    submissions in sequence for verification.
    """

    def __init__(
        self,
        market_snapshot_provider: Callable[[str], MevMarketSnapshot | Mapping[str, float]],
        *,
        tick_size: float = 0.01,
        shadow_sweep_tif: str = "IOC",
        submitter: Callable[[MevOrderPayload], dict[str, Any]] | None = None,
    ) -> None:
        self._market_snapshot_provider = market_snapshot_provider
        self._tick_size = _round_price(tick_size)
        self._shadow_sweep_tif = str(shadow_sweep_tif or "IOC").strip().upper()
        self._submitter = submitter or self._mock_submit
        self._route_counter = count(1)
        self.sent_payloads: list[MevOrderPayload] = []
        self.sent_responses: list[dict[str, Any]] = []

    def execute_shadow_sweep(
        self,
        market_id: str,
        direction: str,
        max_capital: float,
        premium_pct: float,
    ) -> MevExecutionBatch:
        normalized_direction = _normalize_direction(direction)
        if max_capital <= 0.0:
            raise ValueError("max_capital must be strictly positive")
        if premium_pct < 0.0:
            raise ValueError("premium_pct cannot be negative")

        route_id = self._next_route_id("SHADOW")
        snapshot = self._get_snapshot(market_id)
        taker_price = _round_price(snapshot.best_ask(normalized_direction))
        taker_size = _round_size(max_capital / taker_price)

        maker_target = snapshot.mid_price(normalized_direction) + float(premium_pct)
        maker_price = self._passive_buy_price(snapshot, normalized_direction, maker_target)
        maker_size = _round_size(max_capital / maker_price)

        payloads = (
            MevOrderPayload(
                route_id=route_id,
                sequence=1,
                playbook="shadow_sweep",
                market_id=market_id,
                direction=normalized_direction,
                side="BUY",
                price=taker_price,
                size=taker_size,
                liquidity_intent="TAKER",
                time_in_force=self._shadow_sweep_tif,
                post_only=False,
                metadata={
                    "capital_limit": round(float(max_capital), 4),
                    "premium_pct": round(float(premium_pct), 6),
                    "rationale": "aggressive_sweep_before_whale_flow",
                },
            ),
            MevOrderPayload(
                route_id=route_id,
                sequence=2,
                playbook="shadow_sweep",
                market_id=market_id,
                direction=normalized_direction,
                side="BUY",
                price=maker_price,
                size=maker_size,
                liquidity_intent="MAKER",
                time_in_force="GTC",
                post_only=True,
                metadata={
                    "capital_limit": round(float(max_capital), 4),
                    "premium_pct": round(float(premium_pct), 6),
                    "rationale": "capture_post_sweep_repricing",
                },
            ),
        )
        return self._dispatch_batch(route_id, "shadow_sweep", payloads)

    def execute_mm_trap(
        self,
        target_market_id: str,
        correlated_market_id: str,
        v_attack: float,
        trap_direction: str,
    ) -> MevExecutionBatch:
        normalized_direction = _normalize_direction(trap_direction)
        if v_attack <= 0.0:
            raise ValueError("v_attack must be strictly positive")

        route_id = self._next_route_id("TRAP")
        target_snapshot = self._get_snapshot(target_market_id)
        correlated_snapshot = self._get_snapshot(correlated_market_id)

        attack_price = _round_price(target_snapshot.best_ask(normalized_direction))
        attack_size = _round_size(v_attack)
        attack_notional = attack_price * attack_size

        hedge_direction = self._opposite_direction(normalized_direction)
        hedge_price = self._passive_buy_price(
            correlated_snapshot,
            hedge_direction,
            correlated_snapshot.mid_price(hedge_direction),
        )
        hedge_size = _round_size(attack_notional / hedge_price)

        payloads = (
            MevOrderPayload(
                route_id=route_id,
                sequence=1,
                playbook="mm_trap",
                market_id=correlated_market_id,
                direction=hedge_direction,
                side="BUY",
                price=hedge_price,
                size=hedge_size,
                liquidity_intent="MAKER",
                time_in_force="GTC",
                post_only=True,
                metadata={
                    "target_market_id": target_market_id,
                    "trap_direction": normalized_direction,
                    "attack_notional": round(attack_notional, 4),
                    "rationale": "resting_contra_skew_before_ping",
                },
            ),
            MevOrderPayload(
                route_id=route_id,
                sequence=2,
                playbook="mm_trap",
                market_id=target_market_id,
                direction=normalized_direction,
                side="BUY",
                price=attack_price,
                size=attack_size,
                liquidity_intent="TAKER",
                time_in_force="IOC",
                post_only=False,
                metadata={
                    "correlated_market_id": correlated_market_id,
                    "v_attack": round(float(v_attack), 4),
                    "rationale": "force_mm_repricing_ping",
                },
            ),
        )
        return self._dispatch_batch(route_id, "mm_trap", payloads)

    def execute_d3_panic_absorption(
        self,
        market_id: str,
        panic_direction: str,
        limit_price: float,
        max_capital: float,
    ) -> MevExecutionBatch:
        normalized_direction = _normalize_direction(panic_direction)
        if max_capital <= 0.0:
            raise ValueError("max_capital must be strictly positive")

        route_id = self._next_route_id("D3PANIC")
        snapshot = self._get_snapshot(market_id)
        bounded_limit_price = _round_price(limit_price)
        total_shares = _round_size(max_capital / bounded_limit_price)
        level_targets = self._panic_absorption_levels(bounded_limit_price)

        remaining_shares = total_shares
        payloads: list[MevOrderPayload] = []
        for index, target_price in enumerate(level_targets, start=1):
            levels_remaining = len(level_targets) - index + 1
            if levels_remaining <= 1:
                level_size = _round_size(remaining_shares)
            else:
                level_size = _round_size(remaining_shares / levels_remaining)
            remaining_shares = max(0.0, round(remaining_shares - level_size, 4))
            maker_price = self._passive_buy_price(snapshot, normalized_direction, target_price)
            payloads.append(
                MevOrderPayload(
                    route_id=route_id,
                    sequence=index,
                    playbook="d3_panic_absorption",
                    market_id=market_id,
                    direction=normalized_direction,
                    side="BUY",
                    price=maker_price,
                    size=level_size,
                    liquidity_intent="MAKER",
                    time_in_force="GTC",
                    post_only=True,
                    metadata={
                        "limit_price": bounded_limit_price,
                        "level_target": target_price,
                        "grid_level": index,
                        "grid_levels": len(level_targets),
                        "capital_limit": round(float(max_capital), 4),
                        "rationale": "absorb_retail_panic_cascade",
                    },
                )
            )

        return self._dispatch_batch(route_id, "d3_panic_absorption", tuple(payloads))

    def execute_priority_sequence(
        self,
        context: PriorityOrderContext,
    ) -> MevExecutionBatch:
        route_id = self._next_route_id("PRIORITY")
        playbook, payloads = self._plan_priority_payloads(route_id, context)
        return self._dispatch_batch(route_id, playbook, payloads)

    def plan_priority_sequence(
        self,
        context: PriorityOrderContext,
    ) -> MevExecutionBatch:
        # Dry-run planning intentionally still advances the route-id counter.
        # Live client_order_id generation no longer depends on this counter;
        # PriorityDispatcher uses ClientOrderIdGenerator for live submissions,
        # while route_id remains a deterministic planning/debug envelope field.
        route_id = self._next_route_id("PRIORITY")
        playbook, payloads = self._plan_priority_payloads(route_id, context)
        return MevExecutionBatch(
            route_id=route_id,
            playbook=playbook,
            payloads=payloads,
            responses=(),
        )

    def _plan_priority_payloads(
        self,
        route_id: str,
        context: PriorityOrderContext,
    ) -> tuple[str, tuple[MevOrderPayload, ...]]:
        if context.signal_source == "REWARD":
            return "reward_post", self._build_reward_post_payloads(route_id, context)
        return "priority_sequence", self._build_priority_sequence_payloads(route_id, context)

    def _build_priority_sequence_payloads(
        self,
        route_id: str,
        context: PriorityOrderContext,
    ) -> tuple[MevOrderPayload, MevOrderPayload]:
        target_price_decimal = _normalize_native_price(context.target_price)
        optimized_price_decimal = _normalize_native_price(target_price_decimal + _PRIORITY_PRICE_EPSILON)

        optimized_price = float(optimized_price_decimal)
        anchor_price = float(target_price_decimal)
        affordable_size_decimal = context.max_capital / optimized_price_decimal
        base_size_decimal = min(context.anchor_volume, affordable_size_decimal)
        effective_size_decimal = base_size_decimal * context.conviction_scalar
        sequence_size = _round_size_allow_zero(float(effective_size_decimal))

        target_price_str = format_native_price(target_price_decimal)
        optimized_price_str = format_native_price(optimized_price_decimal)
        epsilon_str = format_native_price(_PRIORITY_PRICE_EPSILON)

        payloads = (
            MevOrderPayload(
                route_id=route_id,
                sequence=1,
                playbook="priority_sequence",
                market_id=context.market_id,
                direction=context.side,
                side="BUY",
                price=optimized_price,
                size=sequence_size,
                liquidity_intent="PRIORITY",
                time_in_force="GTC",
                post_only=False,
                context=context,
                metadata={
                    "capital_limit": format_native_price(context.max_capital),
                    "anchor_volume": format_native_price(context.anchor_volume),
                    "target_price": target_price_str,
                    "optimized_price": optimized_price_str,
                    "priority_epsilon": epsilon_str,
                    "base_size": format_native_price(base_size_decimal),
                    "effective_size": format_native_price(effective_size_decimal),
                    "rationale": "queue_priority_entry_at_native_precision",
                },
            ),
            MevOrderPayload(
                route_id=route_id,
                sequence=2,
                playbook="priority_sequence",
                market_id=context.market_id,
                direction=context.side,
                side="SELL",
                price=anchor_price,
                size=sequence_size,
                liquidity_intent="CONDITIONAL",
                time_in_force="GTC",
                post_only=False,
                context=context,
                metadata={
                    "conditional_order_type": "STOP_LIMIT",
                    "anchor_volume": format_native_price(context.anchor_volume),
                    "linked_entry_sequence": 1,
                    "trigger_price": target_price_str,
                    "stop_price": target_price_str,
                    "limit_price": target_price_str,
                    "exit_anchor_price": target_price_str,
                    "effective_size": format_native_price(effective_size_decimal),
                    "rationale": "predefined_exit_against_anchor_node",
                },
            ),
        )
        return payloads

    def _build_reward_post_payloads(
        self,
        route_id: str,
        context: PriorityOrderContext,
    ) -> tuple[MevOrderPayload, ...]:
        execution_hints = context.execution_hints
        if execution_hints is None:
            raise ValueError("REWARD contexts require execution_hints")

        reward_price_decimal = _normalize_native_price(context.target_price)
        reward_metadata = dict(execution_hints.metadata)
        reward_metadata.update(context.signal_metadata)

        return (
            MevOrderPayload(
                route_id=route_id,
                sequence=1,
                playbook="reward_post",
                market_id=context.market_id,
                direction=context.side,
                side="BUY",
                price=float(reward_price_decimal),
                size=_round_size(float(context.anchor_volume)),
                liquidity_intent="MAKER_REWARD",
                time_in_force=execution_hints.time_in_force,
                post_only=execution_hints.post_only,
                context=context,
                metadata=reward_metadata,
            ),
        )

    def _dispatch_batch(
        self,
        route_id: str,
        playbook: str,
        payloads: tuple[MevOrderPayload, ...],
    ) -> MevExecutionBatch:
        responses = tuple(self._submit(payload) for payload in payloads)
        return MevExecutionBatch(
            route_id=route_id,
            playbook=playbook,
            payloads=payloads,
            responses=responses,
        )

    def _submit(self, payload: MevOrderPayload) -> dict[str, Any]:
        response = dict(self._submitter(payload))
        self.sent_payloads.append(payload)
        self.sent_responses.append(response)
        return response

    def _mock_submit(self, payload: MevOrderPayload) -> dict[str, Any]:
        return {
            "accepted": True,
            "mock": True,
            "route_id": payload.route_id,
            "sequence": payload.sequence,
            "client_order_id": f"{payload.route_id}-{payload.sequence}",
        }

    def _get_snapshot(self, market_id: str) -> MevMarketSnapshot:
        raw_snapshot = self._market_snapshot_provider(market_id)
        if isinstance(raw_snapshot, MevMarketSnapshot):
            return raw_snapshot
        if isinstance(raw_snapshot, Mapping):
            return MevMarketSnapshot(
                yes_bid=float(raw_snapshot["yes_bid"]),
                yes_ask=float(raw_snapshot["yes_ask"]),
                no_bid=float(raw_snapshot["no_bid"]),
                no_ask=float(raw_snapshot["no_ask"]),
            )
        raise TypeError(f"Unsupported snapshot payload for {market_id}: {type(raw_snapshot)!r}")

    def _passive_buy_price(
        self,
        snapshot: MevMarketSnapshot,
        direction: str,
        target_price: float,
    ) -> float:
        normalized = _normalize_direction(direction)
        best_bid = snapshot.best_bid(normalized)
        best_ask = snapshot.best_ask(normalized)
        if best_bid <= 0.0 or best_ask <= 0.0:
            raise ValueError(f"Missing two-sided market for {normalized}")

        capped_target = _round_price(target_price)
        passive_ceiling = _round_price(best_ask - self._tick_size)
        if passive_ceiling < best_bid:
            return _round_price(best_bid)
        return _round_price(min(passive_ceiling, max(best_bid, capped_target)))

    def _next_route_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._route_counter):06d}"

    def _panic_absorption_levels(self, limit_price: float) -> tuple[float, ...]:
        top = _round_price(limit_price)
        second = _round_price(top - self._tick_size)
        third = _round_price(top - 2.0 * self._tick_size)
        levels = []
        for candidate in (top, second, third):
            if candidate not in levels:
                levels.append(candidate)
        return tuple(levels)

    @staticmethod
    def _opposite_direction(direction: str) -> str:
        return "NO" if _normalize_direction(direction) == "YES" else "YES"