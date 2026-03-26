"""UMA dispute tracker for isolated D3 panic-absorption research.

The tracker stays unwired from the live bot. It polls an injected UMA oracle
state client, maintains an O(1) per-market EWMA panic-discount baseline, and
emits a signal only when a market transitions into DISPUTED and the current
price trades through that historical discount threshold.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import httpx

from src.core.logger import get_logger
from src.signals.signal_framework import BaseSignal

log = get_logger(__name__)

DEFAULT_DISPUTE_SCAN_INTERVAL_S = 5.0
DEFAULT_PANIC_DISCOUNT_EWMA_ALPHA = 0.2


def _normalize_hex_bytes32(value: str) -> str:
    value = (value or "").lower()
    if not value.startswith("0x"):
        value = "0x" + value
    body = value[2:]
    if len(body) > 64:
        raise ValueError(f"bytes32 value too long: {value}")
    return "0x" + body.rjust(64, "0")


def _decode_uint256_hex(payload: str) -> int:
    payload = payload or "0x0"
    if payload.startswith("0x"):
        payload = payload[2:]
    if not payload:
        return 0
    return int(payload, 16)


class UmaMarketState(enum.IntEnum):
    """Minimal oracle-state enum for dispute detection."""

    UNKNOWN = 0
    REQUESTED = 1
    PROPOSED = 2
    DISPUTED = 3
    RESOLVED = 4
    SETTLED = 5

    @classmethod
    def from_raw(cls, raw_value: int) -> "UmaMarketState":
        try:
            return cls(raw_value)
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class UmaConditionState:
    """Decoded oracle state for a tracked Polymarket condition."""

    condition_id: str
    state: UmaMarketState
    raw_value: int
    timestamp: float


@dataclass
class DisputeArbitrageSignal(BaseSignal):
    """Signal emitted when a disputed market trades through panic discount."""

    oracle_state: str = ""
    current_price: float = 0.0
    panic_discount_ewma: float = 0.0
    trigger_price: float = 0.0
    discount: float = 0.0
    transition_timestamp: float = 0.0


class UmaStateClient(Protocol):
    """Transport contract for fetching UMA oracle state."""

    async def get_condition_state(self, condition_id: str) -> UmaConditionState:
        ...


class JsonRpcEthCallClient:
    """Minimal async JSON-RPC client for isolated contract polling."""

    def __init__(self, rpc_url: str, *, timeout_s: float = 5.0) -> None:
        self._rpc_url = rpc_url
        self._timeout_s = timeout_s
        self._next_id = 1

    async def eth_call(self, *, to: str, data: str, block: str = "latest") -> str:
        request_id = self._next_id
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "eth_call",
            "params": [{"to": to, "data": data}, block],
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(self._rpc_url, json=payload)
            response.raise_for_status()
            body = response.json()

        if "error" in body:
            raise RuntimeError(body["error"])
        result = body.get("result")
        if not isinstance(result, str):
            raise TypeError(f"unexpected eth_call result: {result!r}")
        return result


class EthCallUmaStateClient:
    """Fetch UMA question-state words via ``eth_call``.

    ``state_selector`` should be a 4-byte method selector hex string for a
    bytes32-taking view that returns a uint-like dispute state.
    """

    def __init__(
        self,
        rpc_client: JsonRpcEthCallClient,
        *,
        oracle_contract: str,
        state_selector: str,
    ) -> None:
        selector = state_selector.lower()
        if selector.startswith("0x"):
            selector = selector[2:]
        if len(selector) != 8:
            raise ValueError("state_selector must be exactly 4 bytes / 8 hex chars")

        self._rpc_client = rpc_client
        self._oracle_contract = oracle_contract
        self._state_selector = selector

    async def get_condition_state(self, condition_id: str) -> UmaConditionState:
        normalized_condition_id = _normalize_hex_bytes32(condition_id)
        calldata = "0x" + self._state_selector + normalized_condition_id[2:]
        raw_result = await self._rpc_client.eth_call(
            to=self._oracle_contract,
            data=calldata,
        )
        raw_value = _decode_uint256_hex(raw_result)
        return UmaConditionState(
            condition_id=normalized_condition_id,
            state=UmaMarketState.from_raw(raw_value),
            raw_value=raw_value,
            timestamp=time.time(),
        )


class UmaDisputeTracker:
    """Poll UMA state and emit isolated dispute-arbitrage signals."""

    def __init__(
        self,
        state_client: UmaStateClient,
        *,
        condition_ids: Sequence[str],
        panic_discount_ewma_alpha: float = DEFAULT_PANIC_DISCOUNT_EWMA_ALPHA,
        scan_interval_s: float = DEFAULT_DISPUTE_SCAN_INTERVAL_S,
    ) -> None:
        self._state_client = state_client
        self._condition_ids = [_normalize_hex_bytes32(condition_id) for condition_id in condition_ids]
        self._scan_interval_s = max(0.0, float(scan_interval_s))
        self._alpha = min(1.0, max(0.0, float(panic_discount_ewma_alpha)))
        self._market_state: dict[str, UmaMarketState] = {
            condition_id: UmaMarketState.UNKNOWN for condition_id in self._condition_ids
        }
        self._discount_ewma: dict[str, float] = {}
        self._last_price: dict[str, float] = {}
        self._running = False

    @property
    def tracked_condition_ids(self) -> list[str]:
        return list(self._condition_ids)

    def get_discount_ewma(self, condition_id: str) -> float:
        return self._discount_ewma.get(_normalize_hex_bytes32(condition_id), 0.0)

    def record_market_price(self, condition_id: str, price: float) -> float:
        normalized_condition_id = _normalize_hex_bytes32(condition_id)
        bounded_price = min(1.0, max(0.0, float(price)))
        discount = 1.0 - bounded_price

        previous = self._discount_ewma.get(normalized_condition_id)
        if previous is None:
            updated = discount
        else:
            updated = self._alpha * discount + (1.0 - self._alpha) * previous

        self._discount_ewma[normalized_condition_id] = updated
        self._last_price[normalized_condition_id] = bounded_price
        return updated

    async def poll_once(
        self,
        current_prices: Mapping[str, float],
    ) -> list[DisputeArbitrageSignal]:
        signals: list[DisputeArbitrageSignal] = []

        for condition_id in self._condition_ids:
            state = await self._state_client.get_condition_state(condition_id)
            previous_state = self._market_state.get(condition_id, UmaMarketState.UNKNOWN)
            self._market_state[condition_id] = state.state

            current_price = current_prices.get(condition_id)
            if current_price is None:
                current_price = self._last_price.get(condition_id)
            if current_price is None:
                continue

            bounded_price = min(1.0, max(0.0, float(current_price)))
            baseline_discount = self._discount_ewma.get(condition_id, 0.0)
            trigger_price = max(0.0, min(1.0, 1.0 - baseline_discount))

            if (
                state.state == UmaMarketState.DISPUTED
                and previous_state != UmaMarketState.DISPUTED
                and bounded_price <= trigger_price
            ):
                signal = DisputeArbitrageSignal(
                    market_id=condition_id,
                    no_best_ask=bounded_price,
                    signal_source="D3_DisputeArb",
                    oracle_state=state.state.name,
                    current_price=bounded_price,
                    panic_discount_ewma=baseline_discount,
                    trigger_price=trigger_price,
                    discount=1.0 - bounded_price,
                    transition_timestamp=state.timestamp,
                )
                signals.append(signal)
                log.info(
                    "uma_dispute_signal_fired",
                    condition_id=condition_id,
                    current_price=bounded_price,
                    trigger_price=trigger_price,
                    panic_discount_ewma=baseline_discount,
                )

            self.record_market_price(condition_id, bounded_price)

        return signals

    async def start(self, price_feed: Mapping[str, float] | None = None) -> None:
        self._running = True
        current_prices = dict(price_feed or {})
        while self._running:
            await self.poll_once(current_prices)
            await asyncio.sleep(self._scan_interval_s)

    async def stop(self) -> None:
        self._running = False
