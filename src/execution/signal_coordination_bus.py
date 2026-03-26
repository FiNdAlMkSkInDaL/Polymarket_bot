"""Sole authorized coordination bus implementation for execution adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SlotSide = Literal["YES", "NO"]
SlotSource = Literal["OFI", "SI9", "CTF", "CONTAGION", "MANUAL"]
SlotReason = Literal["GRANTED", "SLOT_OWNED", "SOURCE_CAP", "BUS_CAP", "EXPIRED_RECLAIMED"]


@dataclass(frozen=True, slots=True)
class CoordinationBusConfig:
    slot_lease_ms: int
    max_slots_per_source: int
    max_total_slots: int
    allow_same_source_reentry: bool

    def __post_init__(self) -> None:
        for field_name in ("slot_lease_ms", "max_slots_per_source", "max_total_slots"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a strictly positive int")
        if self.max_total_slots < self.max_slots_per_source:
            raise ValueError("max_total_slots must be >= max_slots_per_source")


@dataclass(frozen=True, slots=True)
class SlotDecision:
    granted: bool
    market_id: str
    side: str
    signal_source: str
    reason: SlotReason
    owner: str | None
    lease_expires_ms: int | None


@dataclass(frozen=True, slots=True)
class CoordinationBusSnapshot:
    snapshot_timestamp_ms: int
    total_active_slots: int
    slots_by_source: dict[str, int]
    slots_by_market: dict[str, list[str]]
    expired_reclaimed_count: int


@dataclass(slots=True)
class _SlotLease:
    signal_source: str
    lease_expires_ms: int


class SignalCoordinationBus:
    def __init__(self, config: CoordinationBusConfig):
        self._config = config
        self._slot_map: dict[tuple[str, str], _SlotLease] = {}
        self._slots_by_source: dict[str, int] = {}
        self._expired_reclaimed_count = 0

    def request_slot(
        self,
        market_id: str,
        side: SlotSide,
        signal_source: SlotSource,
        request_timestamp_ms: int,
    ) -> SlotDecision:
        timestamp_ms = int(request_timestamp_ms)
        slot_market_id = str(market_id).strip()
        slot_side = str(side).strip()
        slot_source = str(signal_source).strip()
        key = (slot_market_id, slot_side)

        expired_owner: str | None = None
        existing = self._slot_map.get(key)
        if existing is not None and timestamp_ms > existing.lease_expires_ms:
            expired_owner = existing.signal_source
            self._release_key(key)

        self._evict_expired(timestamp_ms)
        existing = self._slot_map.get(key)
        if existing is not None:
            if existing.signal_source == slot_source and self._config.allow_same_source_reentry:
                existing.lease_expires_ms = timestamp_ms + self._config.slot_lease_ms
                return SlotDecision(
                    granted=True,
                    market_id=slot_market_id,
                    side=slot_side,
                    signal_source=slot_source,
                    reason="GRANTED",
                    owner=None,
                    lease_expires_ms=existing.lease_expires_ms,
                )
            return SlotDecision(
                granted=False,
                market_id=slot_market_id,
                side=slot_side,
                signal_source=slot_source,
                reason="SLOT_OWNED",
                owner=existing.signal_source,
                lease_expires_ms=existing.lease_expires_ms,
            )

        if self._slots_by_source.get(slot_source, 0) >= self._config.max_slots_per_source:
            return SlotDecision(
                granted=False,
                market_id=slot_market_id,
                side=slot_side,
                signal_source=slot_source,
                reason="SOURCE_CAP",
                owner=None,
                lease_expires_ms=None,
            )
        if len(self._slot_map) >= self._config.max_total_slots:
            return SlotDecision(
                granted=False,
                market_id=slot_market_id,
                side=slot_side,
                signal_source=slot_source,
                reason="BUS_CAP",
                owner=None,
                lease_expires_ms=None,
            )

        lease_expires_ms = timestamp_ms + self._config.slot_lease_ms
        self._slot_map[key] = _SlotLease(
            signal_source=slot_source,
            lease_expires_ms=lease_expires_ms,
        )
        self._slots_by_source[slot_source] = self._slots_by_source.get(slot_source, 0) + 1
        if expired_owner is not None and expired_owner != slot_source:
            self._expired_reclaimed_count += 1
        return SlotDecision(
            granted=True,
            market_id=slot_market_id,
            side=slot_side,
            signal_source=slot_source,
            reason="EXPIRED_RECLAIMED" if expired_owner is not None and expired_owner != slot_source else "GRANTED",
            owner=None,
            lease_expires_ms=lease_expires_ms,
        )

    def release_slot(
        self,
        market_id: str,
        side: SlotSide,
        signal_source: str,
        release_timestamp_ms: int,
    ) -> None:
        self._evict_expired(int(release_timestamp_ms))
        key = (str(market_id).strip(), str(side).strip())
        slot = self._slot_map.get(key)
        if slot is None or slot.signal_source != str(signal_source).strip():
            return
        self._release_key(key)

    def owns_slot(
        self,
        market_id: str,
        side: SlotSide,
        signal_source: str,
        current_timestamp_ms: int,
    ) -> bool:
        self._evict_expired(int(current_timestamp_ms))
        key = (str(market_id).strip(), str(side).strip())
        slot = self._slot_map.get(key)
        return slot is not None and slot.signal_source == str(signal_source).strip()

    def bus_snapshot(self, current_timestamp_ms: int) -> CoordinationBusSnapshot:
        timestamp_ms = int(current_timestamp_ms)
        self._evict_expired(timestamp_ms)
        slots_by_market: dict[str, list[str]] = {}
        for market_id, side in self._slot_map:
            slots_by_market.setdefault(market_id, []).append(side)
        for market_id in tuple(slots_by_market.keys()):
            slots_by_market[market_id] = sorted(slots_by_market[market_id])
        return CoordinationBusSnapshot(
            snapshot_timestamp_ms=timestamp_ms,
            total_active_slots=len(self._slot_map),
            slots_by_source=dict(self._slots_by_source),
            slots_by_market=slots_by_market,
            expired_reclaimed_count=self._expired_reclaimed_count,
        )

    def _evict_expired(self, current_timestamp_ms: int) -> None:
        expired_keys = [
            key for key, slot in self._slot_map.items() if current_timestamp_ms > slot.lease_expires_ms
        ]
        for key in expired_keys:
            self._release_key(key)

    def _release_key(self, key: tuple[str, str]) -> None:
        slot = self._slot_map.pop(key, None)
        if slot is None:
            return
        next_count = self._slots_by_source.get(slot.signal_source, 0) - 1
        if next_count <= 0:
            self._slots_by_source.pop(slot.signal_source, None)
        else:
            self._slots_by_source[slot.signal_source] = next_count