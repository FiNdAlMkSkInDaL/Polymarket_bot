"""Polygon mempool monitor for CTF-facing pending transactions.

This module stays isolated from the live bot and exposes two layers:

1. A lightweight async JSON-RPC websocket client for pending-transaction
   subscription and transaction lookup.
2. A pure O(1) parser/state machine that tracks pending CTF-facing volume and
   flips a whale flag when the configured threshold is exceeded.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Mapping, Protocol

import websockets

from src.core.logger import get_logger

log = get_logger(__name__)

POLYMARKET_CTF_CONTRACT = "0x4d9702590a32765052304e32e116992d00a71943"
POLYGON_USDC_CONTRACTS = {
    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",  # USDC.e
    "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",  # native USDC
}
USDC_DECIMALS = 6
DEFAULT_PENDING_VOLUME_THRESHOLD = 100_000.0
DEFAULT_PENDING_TTL_S = 90.0

_APPROVE_SELECTOR = bytes.fromhex("095ea7b3")
_SPLIT_POSITION_SELECTOR = bytes.fromhex("c9ff79aa")


def _normalize_address(value: str | None) -> str:
    if not value:
        return ""
    return value.lower()


def _hex_to_bytes(payload: str) -> bytes:
    payload = payload or ""
    if payload.startswith("0x"):
        payload = payload[2:]
    if len(payload) % 2 == 1:
        payload = "0" + payload
    return bytes.fromhex(payload)


def _decode_uint256(word: bytes) -> int:
    if len(word) != 32:
        raise ValueError(f"expected 32-byte word, got {len(word)}")
    return int.from_bytes(word, "big")


def _decode_address(word: bytes) -> str:
    if len(word) != 32:
        raise ValueError(f"expected 32-byte word, got {len(word)}")
    return "0x" + word[-20:].hex()


def _decode_bytes32(word: bytes) -> str:
    if len(word) != 32:
        raise ValueError(f"expected 32-byte word, got {len(word)}")
    return "0x" + word.hex()


def _chunks(payload: bytes) -> list[bytes]:
    if len(payload) % 32 != 0:
        raise ValueError("ABI payload length must be a multiple of 32 bytes")
    return [payload[index:index + 32] for index in range(0, len(payload), 32)]


@dataclass(frozen=True)
class PendingTransactionMatch:
    """Decoded pending transaction relevant to Polymarket CTF flow."""

    tx_hash: str
    sender: str
    contract_address: str
    method_name: str
    raw_amount: int
    volume: float
    seen_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PendingTxRpcClient(Protocol):
    """Transport contract for pending-transaction monitoring."""

    async def subscribe_pending_transactions(self) -> AsyncIterator[str]:
        ...

    async def get_transaction_by_hash(self, tx_hash: str) -> Mapping[str, Any] | None:
        ...

    async def close(self) -> None:
        ...


class WebSocketPendingTxRpcClient:
    """Minimal JSON-RPC websocket client for Polygon pending tx monitoring."""

    def __init__(self, rpc_wss_url: str, *, request_timeout_s: float = 5.0) -> None:
        self._rpc_wss_url = rpc_wss_url
        self._request_timeout_s = request_timeout_s
        self._next_id = 1
        self._pending_calls: dict[int, asyncio.Future] = {}
        self._notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._ws: Any = None

    async def connect(self) -> None:
        if self._ws is not None:
            return
        self._ws = await websockets.connect(self._rpc_wss_url)
        self._reader_task = asyncio.create_task(self._reader_loop(), name="mempool_rpc_reader")

    async def close(self) -> None:
        reader_task = self._reader_task
        self._reader_task = None
        if reader_task is not None:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

        ws = self._ws
        self._ws = None
        if ws is not None:
            await ws.close()

        for future in self._pending_calls.values():
            if not future.done():
                future.cancel()
        self._pending_calls.clear()

    async def subscribe_pending_transactions(self) -> AsyncIterator[str]:
        await self.connect()
        subscription_id = await self._rpc_request("eth_subscribe", ["newPendingTransactions"])
        try:
            while True:
                message = await self._notifications.get()
                params = message.get("params") or {}
                if params.get("subscription") != subscription_id:
                    continue
                tx_hash = params.get("result")
                if isinstance(tx_hash, str) and tx_hash:
                    yield tx_hash
        finally:
            try:
                await self._rpc_request("eth_unsubscribe", [subscription_id])
            except Exception:
                log.debug("mempool_unsubscribe_failed", exc_info=True)

    async def get_transaction_by_hash(self, tx_hash: str) -> Mapping[str, Any] | None:
        result = await self._rpc_request("eth_getTransactionByHash", [tx_hash])
        if result is None:
            return None
        if not isinstance(result, Mapping):
            raise TypeError(f"unexpected transaction payload: {type(result)!r}")
        return result

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw_message in self._ws:
                message = json.loads(raw_message)
                if "id" in message:
                    request_id = int(message["id"])
                    future = self._pending_calls.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_result(message)
                    continue

                if message.get("method") == "eth_subscription":
                    await self._notifications.put(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            for future in self._pending_calls.values():
                if not future.done():
                    future.set_exception(RuntimeError("mempool websocket reader failed"))
            self._pending_calls.clear()
            raise

    async def _rpc_request(self, method: str, params: list[Any]) -> Any:
        await self.connect()
        assert self._ws is not None

        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_calls[request_id] = future

        await self._ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }))

        try:
            response = await asyncio.wait_for(future, timeout=self._request_timeout_s)
        except Exception:
            self._pending_calls.pop(request_id, None)
            raise

        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")


class PendingVolumeStateMachine:
    """O(1) pending-volume tracker with TTL-based eviction."""

    def __init__(
        self,
        *,
        volume_threshold: float = DEFAULT_PENDING_VOLUME_THRESHOLD,
        ttl_s: float = DEFAULT_PENDING_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._volume_threshold = max(0.0, float(volume_threshold))
        self._ttl_s = max(0.0, float(ttl_s))
        self._clock = clock
        self._pending_volume = 0.0
        self._tracked: dict[str, tuple[float, float]] = {}
        self._expiry_queue: deque[tuple[float, str]] = deque()

    @property
    def pending_volume(self) -> float:
        return self._pending_volume

    @property
    def is_whale_incoming(self) -> bool:
        return self._pending_volume >= self._volume_threshold

    def observe(self, match: PendingTransactionMatch, *, now: float | None = None) -> None:
        current_time = self._clock() if now is None else float(now)
        self.sweep_expired(now=current_time)

        previous = self._tracked.get(match.tx_hash)
        if previous is not None:
            self._pending_volume -= previous[0]

        expiry_time = current_time + self._ttl_s
        self._tracked[match.tx_hash] = (match.volume, expiry_time)
        self._expiry_queue.append((expiry_time, match.tx_hash))
        self._pending_volume += match.volume

    def remove(self, tx_hash: str, *, now: float | None = None) -> None:
        current_time = self._clock() if now is None else float(now)
        self.sweep_expired(now=current_time)
        previous = self._tracked.pop(tx_hash, None)
        if previous is not None:
            self._pending_volume -= previous[0]

    def sweep_expired(self, *, now: float | None = None) -> None:
        current_time = self._clock() if now is None else float(now)
        while self._expiry_queue and self._expiry_queue[0][0] <= current_time:
            expiry_time, tx_hash = self._expiry_queue.popleft()
            tracked = self._tracked.get(tx_hash)
            if tracked is None:
                continue
            tracked_volume, tracked_expiry = tracked
            if tracked_expiry != expiry_time:
                continue
            self._tracked.pop(tx_hash, None)
            self._pending_volume -= tracked_volume


class MempoolMonitor:
    """Async Polygon mempool listener filtered to Polymarket-relevant flows."""

    def __init__(
        self,
        rpc_client: PendingTxRpcClient,
        *,
        ctf_contract: str = POLYMARKET_CTF_CONTRACT,
        usdc_contracts: set[str] | None = None,
        collateral_decimals: int = USDC_DECIMALS,
        volume_threshold: float = DEFAULT_PENDING_VOLUME_THRESHOLD,
        pending_ttl_s: float = DEFAULT_PENDING_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._rpc_client = rpc_client
        self._ctf_contract = _normalize_address(ctf_contract)
        self._usdc_contracts = {
            _normalize_address(address) for address in (usdc_contracts or POLYGON_USDC_CONTRACTS)
        }
        self._amount_scale = float(10 ** max(0, collateral_decimals))
        self._state = PendingVolumeStateMachine(
            volume_threshold=volume_threshold,
            ttl_s=pending_ttl_s,
            clock=clock,
        )
        self._running = False
        self._recent_matches: deque[PendingTransactionMatch] = deque(maxlen=256)

    @property
    def pending_volume(self) -> float:
        self._state.sweep_expired()
        return self._state.pending_volume

    @property
    def is_whale_incoming(self) -> bool:
        self._state.sweep_expired()
        return self._state.is_whale_incoming

    @property
    def recent_matches(self) -> list[PendingTransactionMatch]:
        self._state.sweep_expired()
        return list(self._recent_matches)

    async def start(self) -> None:
        self._running = True
        async for tx_hash in self._rpc_client.subscribe_pending_transactions():
            if not self._running:
                break
            await self.process_pending_hash(tx_hash)

    async def stop(self) -> None:
        self._running = False
        await self._rpc_client.close()

    async def process_pending_hash(self, tx_hash: str) -> PendingTransactionMatch | None:
        transaction = await self._rpc_client.get_transaction_by_hash(tx_hash)
        return self.ingest_transaction(transaction)

    def ingest_transaction(
        self,
        transaction: Mapping[str, Any] | None,
        *,
        seen_at: float | None = None,
    ) -> PendingTransactionMatch | None:
        self._state.sweep_expired(now=seen_at)
        match = self._match_transaction(transaction, seen_at=seen_at)
        if match is None:
            return None

        self._state.observe(match, now=match.seen_at)
        self._recent_matches.append(match)
        return match

    def remove_pending(self, tx_hash: str, *, now: float | None = None) -> None:
        self._state.remove(tx_hash, now=now)

    def _match_transaction(
        self,
        transaction: Mapping[str, Any] | None,
        *,
        seen_at: float | None = None,
    ) -> PendingTransactionMatch | None:
        if not transaction:
            return None

        tx_to = _normalize_address(str(transaction.get("to") or ""))
        tx_input = str(transaction.get("input") or transaction.get("data") or "")
        tx_hash = str(transaction.get("hash") or "")
        sender = _normalize_address(str(transaction.get("from") or ""))

        if not tx_to or not tx_input or len(tx_input) < 10:
            return None

        input_bytes = _hex_to_bytes(tx_input)
        selector = input_bytes[:4]
        payload = input_bytes[4:]
        current_seen_at = time.time() if seen_at is None else float(seen_at)

        if tx_to == self._ctf_contract and selector == _SPLIT_POSITION_SELECTOR:
            split_metadata = self._parse_split_position(payload)
            raw_amount = split_metadata["raw_amount"]
            volume = raw_amount / self._amount_scale
            return PendingTransactionMatch(
                tx_hash=tx_hash,
                sender=sender,
                contract_address=tx_to,
                method_name="splitPosition",
                raw_amount=raw_amount,
                volume=volume,
                seen_at=current_seen_at,
                metadata=split_metadata,
            )

        if tx_to in self._usdc_contracts and selector == _APPROVE_SELECTOR:
            spender, raw_amount = self._parse_approve(payload)
            if spender != self._ctf_contract:
                return None
            volume = raw_amount / self._amount_scale
            return PendingTransactionMatch(
                tx_hash=tx_hash,
                sender=sender,
                contract_address=tx_to,
                method_name="approve",
                raw_amount=raw_amount,
                volume=volume,
                seen_at=current_seen_at,
                metadata={"spender": spender, "target": "usdc"},
            )

        return None

    def _parse_split_position(self, payload: bytes) -> dict[str, Any]:
        words = _chunks(payload)
        if len(words) < 5:
            raise ValueError("splitPosition payload too short")
        return {
            "target": "ctf",
            "collateral_token": _decode_address(words[0]),
            "parent_collection_id": _decode_bytes32(words[1]),
            "condition_id": _decode_bytes32(words[2]),
            "partition_offset": _decode_uint256(words[3]),
            "raw_amount": _decode_uint256(words[4]),
        }

    def _parse_approve(self, payload: bytes) -> tuple[str, int]:
        words = _chunks(payload)
        if len(words) < 2:
            raise ValueError("approve payload too short")
        spender = _decode_address(words[0])
        amount = _decode_uint256(words[1])
        return spender, amount
