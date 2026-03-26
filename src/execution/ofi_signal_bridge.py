from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol

from src.execution.alpha_adapters import ofi_to_context
from src.execution.dispatch_guard import DispatchGuard, GuardDecision
from src.execution.ofi_paper_ledger import OfiPaperLedger, OfiLedgerSnapshot
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchReceipt, PriorityDispatcher
from src.execution.signal_coordination_bus import SignalCoordinationBus, SlotDecision


OfiBridgeOutcome = Literal[
    "DISPATCHED",
    "GUARD_REJECTED",
    "BUS_REJECTED",
    "SOURCE_DISABLED",
    "CAPITAL_ZERO",
]


@dataclass(frozen=True, slots=True)
class OfiEntrySignal:
    market_id: str
    side: Literal["YES", "NO"]
    target_price: Decimal
    anchor_volume: Decimal
    conviction_scalar: Decimal
    signal_timestamp_ms: int
    tvi_kappa: Decimal
    ofi_window_ms: int

    def __post_init__(self) -> None:
        market_id = str(self.market_id or "").strip()
        if not market_id:
            raise ValueError("market_id must be a non-empty string")
        object.__setattr__(self, "market_id", market_id)
        if self.side not in {"YES", "NO"}:
            raise ValueError("side must be 'YES' or 'NO'")
        for field_name in ("target_price", "anchor_volume", "conviction_scalar", "tvi_kappa"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{field_name} must be a finite Decimal")
        if self.target_price <= Decimal("0"):
            raise ValueError("target_price must be strictly positive")
        if self.anchor_volume <= Decimal("0"):
            raise ValueError("anchor_volume must be strictly positive")
        if self.conviction_scalar < Decimal("0") or self.conviction_scalar > Decimal("1"):
            raise ValueError("conviction_scalar must be within [0, 1]")
        if not isinstance(self.signal_timestamp_ms, int):
            raise ValueError("signal_timestamp_ms must be an int")
        if not isinstance(self.ofi_window_ms, int) or self.ofi_window_ms <= 0:
            raise ValueError("ofi_window_ms must be a strictly positive int")


@dataclass(frozen=True, slots=True)
class OfiSignalBridgeConfig:
    max_capital_per_signal: Decimal
    mode: Literal["paper", "dry_run"]
    slot_side_lock: bool = True
    source_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_capital_per_signal, Decimal) or not self.max_capital_per_signal.is_finite() or self.max_capital_per_signal <= Decimal("0"):
            raise ValueError("max_capital_per_signal must be a strictly positive Decimal")
        if self.mode not in {"paper", "dry_run"}:
            raise ValueError("mode must be 'paper' or 'dry_run'")
        if not isinstance(self.slot_side_lock, bool):
            raise ValueError("slot_side_lock must be a bool")
        if not isinstance(self.source_enabled, bool):
            raise ValueError("source_enabled must be a bool")


@dataclass(frozen=True, slots=True)
class OfiBridgeReceipt:
    signal: OfiEntrySignal
    dispatch_receipt: DispatchReceipt | None
    bridge_outcome: OfiBridgeOutcome
    yes_slot: SlotDecision | None
    no_slot: SlotDecision | None
    guard_decision: GuardDecision | None
    timestamp_ms: int


class OfiSignalBridge:
    def __init__(
        self,
        dispatcher: PriorityDispatcher,
        guard: DispatchGuard,
        bus: SignalCoordinationBus,
        ledger: OfiPaperLedger,
        config: OfiSignalBridgeConfig,
    ):
        self._dispatcher = dispatcher
        self._guard = guard
        self._bus = bus
        self._ledger = ledger
        self._config = config

    @property
    def dispatcher(self) -> PriorityDispatcher:
        return self._dispatcher

    @property
    def guard(self) -> DispatchGuard:
        return self._guard

    @property
    def bus(self) -> SignalCoordinationBus:
        return self._bus

    def ledger_snapshot(self) -> OfiLedgerSnapshot:
        return self._ledger.snapshot()

    def on_signal(
        self,
        signal: OfiEntrySignal,
        max_capital: Decimal,
        timestamp_ms: int,
    ) -> OfiBridgeReceipt:
        event_timestamp_ms = int(timestamp_ms)
        if not self._config.source_enabled:
            receipt = OfiBridgeReceipt(
                signal=signal,
                dispatch_receipt=None,
                bridge_outcome="SOURCE_DISABLED",
                yes_slot=None,
                no_slot=None,
                guard_decision=None,
                timestamp_ms=event_timestamp_ms,
            )
            self._ledger.record_signal(
                outcome=receipt.bridge_outcome,
                conviction_scalar=signal.conviction_scalar,
                timestamp_ms=event_timestamp_ms,
            )
            return receipt

        allowed_capital = min(self._config.max_capital_per_signal, max_capital)
        if allowed_capital <= Decimal("0"):
            receipt = OfiBridgeReceipt(
                signal=signal,
                dispatch_receipt=None,
                bridge_outcome="CAPITAL_ZERO",
                yes_slot=None,
                no_slot=None,
                guard_decision=None,
                timestamp_ms=event_timestamp_ms,
            )
            self._ledger.record_signal(
                outcome=receipt.bridge_outcome,
                conviction_scalar=signal.conviction_scalar,
                timestamp_ms=event_timestamp_ms,
            )
            return receipt

        context = ofi_to_context(
            market_id=signal.market_id,
            side=signal.side,
            target_price=signal.target_price,
            anchor_volume=signal.anchor_volume,
            max_capital=allowed_capital,
            conviction_scalar=signal.conviction_scalar,
        )

        yes_slot, no_slot, acquired_slots = self._acquire_slots(signal, event_timestamp_ms)
        if self._slot_rejected(yes_slot, no_slot):
            self._release_slots(acquired_slots, event_timestamp_ms)
            receipt = OfiBridgeReceipt(
                signal=signal,
                dispatch_receipt=None,
                bridge_outcome="BUS_REJECTED",
                yes_slot=yes_slot,
                no_slot=no_slot,
                guard_decision=None,
                timestamp_ms=event_timestamp_ms,
            )
            self._ledger.record_signal(
                outcome=receipt.bridge_outcome,
                conviction_scalar=signal.conviction_scalar,
                timestamp_ms=event_timestamp_ms,
            )
            return receipt

        guard_decision = self._guard.check(context, event_timestamp_ms)
        if not guard_decision.allowed:
            self._guard.record_suppression(context.signal_source)
            self._release_slots(acquired_slots, event_timestamp_ms)
            receipt = OfiBridgeReceipt(
                signal=signal,
                dispatch_receipt=None,
                bridge_outcome="GUARD_REJECTED",
                yes_slot=yes_slot,
                no_slot=no_slot,
                guard_decision=guard_decision,
                timestamp_ms=event_timestamp_ms,
            )
            self._ledger.record_signal(
                outcome=receipt.bridge_outcome,
                conviction_scalar=signal.conviction_scalar,
                timestamp_ms=event_timestamp_ms,
            )
            return receipt

        dispatch_receipt = self._dispatcher.dispatch(context, event_timestamp_ms)
        if dispatch_receipt.guard_reason is not None:
            self._release_slots(acquired_slots, event_timestamp_ms)
            receipt = OfiBridgeReceipt(
                signal=signal,
                dispatch_receipt=dispatch_receipt,
                bridge_outcome="GUARD_REJECTED",
                yes_slot=yes_slot,
                no_slot=no_slot,
                guard_decision=GuardDecision(allowed=False, reason=dispatch_receipt.guard_reason),
                timestamp_ms=event_timestamp_ms,
            )
            self._ledger.record_signal(
                outcome=receipt.bridge_outcome,
                conviction_scalar=signal.conviction_scalar,
                timestamp_ms=event_timestamp_ms,
            )
            return receipt

        self._guard.record_dispatch(context, event_timestamp_ms)
        deployed_capital = Decimal("0")
        if dispatch_receipt.fill_price is not None and dispatch_receipt.fill_size is not None:
            deployed_capital = dispatch_receipt.fill_price * dispatch_receipt.fill_size
        receipt = OfiBridgeReceipt(
            signal=signal,
            dispatch_receipt=dispatch_receipt,
            bridge_outcome="DISPATCHED",
            yes_slot=yes_slot,
            no_slot=no_slot,
            guard_decision=guard_decision,
            timestamp_ms=event_timestamp_ms,
        )
        self._ledger.record_signal(
            outcome=receipt.bridge_outcome,
            conviction_scalar=signal.conviction_scalar,
            timestamp_ms=event_timestamp_ms,
            deployed_capital=deployed_capital,
        )
        return receipt

    def _acquire_slots(
        self,
        signal: OfiEntrySignal,
        timestamp_ms: int,
    ) -> tuple[SlotDecision | None, SlotDecision | None, list[tuple[str, str, str]]]:
        if self._config.slot_side_lock:
            requested_sides = [signal.side, "NO" if signal.side == "YES" else "YES"]
        else:
            requested_sides = [signal.side]

        decisions: dict[str, SlotDecision] = {}
        acquired_slots: list[tuple[str, str, str]] = []
        for side in requested_sides:
            decision = self._bus.request_slot(signal.market_id, side, "OFI", int(timestamp_ms))
            decisions[side] = decision
            if not decision.granted:
                break
            acquired_slots.append((signal.market_id, side, "OFI"))
        return decisions.get("YES"), decisions.get("NO"), acquired_slots

    def _release_slots(
        self,
        acquired_slots: list[tuple[str, str, str]],
        timestamp_ms: int,
    ) -> None:
        for market_id, side, signal_source in acquired_slots:
            self._bus.release_slot(market_id, side, signal_source, int(timestamp_ms))

    @staticmethod
    def _slot_rejected(yes_slot: SlotDecision | None, no_slot: SlotDecision | None) -> bool:
        for decision in (yes_slot, no_slot):
            if decision is not None and not decision.granted:
                return True
        return False