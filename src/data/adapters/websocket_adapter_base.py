from __future__ import annotations

import asyncio
import inspect
import json
import random
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import websockets

from src.core.logger import get_logger
from src.data.oracle_adapter import OffChainOracleAdapter, OracleMarketConfig, OracleSnapshot

log = get_logger(__name__)


class WebSocketOracleAdapter(OffChainOracleAdapter):
    """Reusable async WebSocket client for standalone oracle streams."""

    _QUEUE_SENTINEL = object()

    def __init__(
        self,
        market_config: OracleMarketConfig,
        *,
        websocket_url: str,
        api_key: str = "",
        headers: Mapping[str, str] | None = None,
        connect_kwargs: Mapping[str, Any] | None = None,
        heartbeat_interval_s: float = 10.0,
        heartbeat_timeout_s: float = 10.0,
        reconnect_base_s: float = 0.5,
        reconnect_max_s: float = 30.0,
        inbound_queue_size: int = 1024,
        on_trip=None,
    ) -> None:
        super().__init__(market_config, on_trip=on_trip)
        self._websocket_url = websocket_url
        self._api_key = api_key
        self._headers = dict(headers or {})
        self._connect_kwargs = dict(connect_kwargs or {})
        self._heartbeat_interval_s = heartbeat_interval_s
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._reconnect_base_s = reconnect_base_s
        self._reconnect_max_s = reconnect_max_s
        self._inbound_queue_size = max(1, inbound_queue_size)
        self._latest_snapshot: OracleSnapshot | None = None
        self._stop_event = asyncio.Event()

    async def start(self, queue: asyncio.Queue) -> None:
        self._running = True
        self._stop_event.clear()
        log.info(
            "oracle_adapter_started",
            adapter=self.name,
            market_id=self._config.market_id,
        )
        try:
            async for snapshot in self.stream_snapshots():
                queue.put_nowait(snapshot)
        except asyncio.CancelledError:
            raise
        finally:
            log.info(
                "oracle_adapter_stopped",
                adapter=self.name,
                market_id=self._config.market_id,
            )

    async def poll(self) -> OracleSnapshot:
        if self._latest_snapshot is not None:
            return self._latest_snapshot

        async for snapshot in self.stream_snapshots():
            return snapshot

        raise RuntimeError(f"{self.name} stream ended before the first snapshot")

    async def stream_snapshots(self) -> AsyncIterator[OracleSnapshot]:
        self._running = True
        self._stop_event.clear()
        attempt = 0

        while self._running and not self._stop_event.is_set():
            try:
                connect_headers = self._build_connect_headers()
                async with websockets.connect(
                    self._websocket_url,
                    ping_interval=None,
                    **self._connect_header_kwargs(connect_headers),
                    **self._connect_kwargs,
                ) as websocket:
                    attempt = 0
                    log.info(
                        "oracle_websocket_connected",
                        adapter=self.name,
                        market_id=self._config.market_id,
                        websocket_url=self._websocket_url,
                    )

                    await self._send_subscriptions(websocket)
                    inbound_queue: asyncio.Queue[Any] = asyncio.Queue(
                        maxsize=self._inbound_queue_size,
                    )
                    reader_task = asyncio.create_task(
                        self._reader_loop(websocket, inbound_queue),
                    )
                    heartbeat_task = asyncio.create_task(
                        self._heartbeat_loop(websocket, inbound_queue),
                    )

                    try:
                        while self._running and not self._stop_event.is_set():
                            payload = await inbound_queue.get()
                            if payload is self._QUEUE_SENTINEL:
                                raise ConnectionError("websocket transport closed")

                            try:
                                snapshot = await self._payload_to_snapshot(websocket, payload)
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                log.error(
                                    "oracle_websocket_payload_parse_error",
                                    adapter=self.name,
                                    market_id=self._config.market_id,
                                    exc_info=True,
                                )
                                continue

                            if snapshot is None:
                                continue

                            snapshot.timestamp = time.monotonic()
                            self._latest_snapshot = snapshot
                            self._last_phase = snapshot.event_phase
                            yield snapshot
                    finally:
                        await self._cancel_task(reader_task)
                        await self._cancel_task(heartbeat_task)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.error(
                    "oracle_websocket_connection_error",
                    adapter=self.name,
                    market_id=self._config.market_id,
                    websocket_url=self._websocket_url,
                    exc_info=True,
                )

                if self._breaker.record():
                    log.critical(
                        "oracle_websocket_breaker_tripped",
                        adapter=self.name,
                        market_id=self._config.market_id,
                        errors_in_window=self._breaker.recent_errors,
                    )
                    if self._on_trip is not None:
                        try:
                            await self._on_trip()
                        except Exception:
                            log.error("oracle_websocket_on_trip_error", exc_info=True)
                    self._running = False
                    return

                delay_s = self._compute_backoff_seconds(attempt)
                attempt += 1
                await asyncio.sleep(delay_s)

        log.info(
            "oracle_websocket_stopped",
            adapter=self.name,
            market_id=self._config.market_id,
        )

    def stop(self) -> None:
        self._stop_event.set()
        super().stop()

    def _build_connect_headers(self) -> dict[str, str]:
        headers = dict(self._headers)
        if self._api_key and "Authorization" not in headers and "authorization" not in headers:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _connect_header_kwargs(headers: Mapping[str, str]) -> dict[str, Any]:
        if not headers:
            return {}
        params = inspect.signature(websockets.connect).parameters
        if "additional_headers" in params:
            return {"additional_headers": headers}
        if "extra_headers" in params:
            return {"extra_headers": headers}
        return {}

    def _compute_backoff_seconds(self, attempt: int) -> float:
        base = min(self._reconnect_max_s, self._reconnect_base_s * (2 ** attempt))
        jitter = min(self._reconnect_base_s, base / 4.0)
        return min(self._reconnect_max_s, base + random.uniform(0.0, jitter))

    async def _send_subscriptions(self, websocket: Any) -> None:
        for message in self._subscription_messages():
            if isinstance(message, (dict, list)):
                encoded = json.dumps(message)
            else:
                encoded = message
            await websocket.send(encoded)

    async def _reader_loop(self, websocket: Any, inbound_queue: asyncio.Queue[Any]) -> None:
        try:
            async for payload in websocket:
                self._put_nowait_drop_oldest(inbound_queue, payload)
        finally:
            self._put_nowait_drop_oldest(inbound_queue, self._QUEUE_SENTINEL)

    async def _heartbeat_loop(self, websocket: Any, inbound_queue: asyncio.Queue[Any]) -> None:
        try:
            while self._running and not self._stop_event.is_set():
                await asyncio.sleep(self._heartbeat_interval_s)
                pong_waiter = await websocket.ping()
                await asyncio.wait_for(pong_waiter, timeout=self._heartbeat_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._put_nowait_drop_oldest(inbound_queue, self._QUEUE_SENTINEL)
            raise

    @staticmethod
    async def _cancel_task(task: asyncio.Task[Any]) -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    @staticmethod
    def _put_nowait_drop_oldest(queue: asyncio.Queue[Any], item: Any) -> None:
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass

        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    @staticmethod
    def _decode_payload(payload: Any) -> Any:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return payload
        return payload

    def _subscription_messages(self) -> Sequence[str | bytes | dict[str, Any] | list[Any]]:
        return ()

    @abstractmethod
    async def _payload_to_snapshot(self, websocket: Any, payload: Any) -> OracleSnapshot | None:
        raise NotImplementedError