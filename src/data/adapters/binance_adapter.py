"""Binance aggregate-trade WebSocket adapter for SI-8 crypto oracle feeds."""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import aiohttp

from src.core.config import settings
from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
from src.core.logger import get_logger
from src.data.oracle_adapter import OffChainOracleAdapter, OracleMarketConfig, OracleSnapshot

log = get_logger(__name__)


class BinanceWebSocketAdapter(OffChainOracleAdapter):
    """Streams Binance aggTrade ticks and emits coalesced OracleSnapshot updates."""

    def __init__(
        self,
        market_config: OracleMarketConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(market_config, **kwargs)
        raw_symbol = (
            market_config.external_id
            or str(market_config.oracle_params.get("symbol", ""))
            or "btcusdt"
        )
        self._symbol = raw_symbol.lower().strip()
        self._ws_url = f"wss://stream.binance.com:9443/ws/{self._symbol}@aggTrade"
        self._session: aiohttp.ClientSession | None = None
        self._breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._last_emit_time: float = 0.0
        self._last_threshold_state: bool | None = None
        self._emit_interval_s: float = settings.strategy.oracle_critical_poll_ms / 1000.0

    @property
    def name(self) -> str:
        return "crypto"

    async def poll(self) -> OracleSnapshot:
        raise NotImplementedError("BinanceWebSocketAdapter is stream-driven")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _opposing_outcome(self, outcome: str) -> str | None:
        up = outcome.strip().upper()
        if up == "YES":
            return "NO"
        if up == "NO":
            return "YES"
        if up == "ABOVE":
            return "BELOW"
        if up == "BELOW":
            return "ABOVE"
        return None

    def _resolve_outcome(self, price: float) -> tuple[str | None, bool | None]:
        if self._config.market_type != "threshold":
            return None, None

        target = (self._config.target_outcome or "").strip()
        crossed = price >= float(self._config.goal_line)

        if crossed:
            return (target or None), crossed

        opposing = self._opposing_outcome(target)
        return opposing, crossed

    async def start(self, queue: asyncio.Queue) -> None:
        self._running = True
        backoff_s = 1.0
        max_backoff_s = 30.0

        log.info(
            "binance_adapter_started",
            market_id=self._config.market_id,
            symbol=self._symbol,
            ws_url=self._ws_url,
            emit_interval_s=self._emit_interval_s,
        )

        try:
            while self._running:
                try:
                    session = await self._ensure_session()
                    async with session.ws_connect(
                        self._ws_url,
                        heartbeat=20,
                        autoping=True,
                        timeout=15,
                    ) as ws:
                        backoff_s = 1.0
                        log.info(
                            "binance_ws_connected",
                            market_id=self._config.market_id,
                            symbol=self._symbol,
                        )

                        async for msg in ws:
                            if not self._running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                payload = json.loads(msg.data)
                                price = float(payload.get("p", 0.0) or 0.0)
                                qty = float(payload.get("q", 0.0) or 0.0)
                                if price <= 0.0:
                                    continue

                                now = time.monotonic()
                                resolved_outcome, threshold_state = self._resolve_outcome(price)
                                state_changed = (
                                    threshold_state is not None
                                    and self._last_threshold_state is not None
                                    and threshold_state != self._last_threshold_state
                                )

                                if not state_changed and (now - self._last_emit_time) < self._emit_interval_s:
                                    self._last_threshold_state = threshold_state
                                    continue

                                snapshot = OracleSnapshot(
                                    adapter_name=self.name,
                                    market_id=self._config.market_id,
                                    raw_state={
                                        "symbol": self._symbol,
                                        "price": price,
                                        "quantity": qty,
                                        "event_time_ms": payload.get("E"),
                                        "trade_time_ms": payload.get("T"),
                                        "agg_trade_id": payload.get("a"),
                                        "goal_line": self._config.goal_line,
                                        "market_type": self._config.market_type,
                                    },
                                    resolved_outcome=resolved_outcome,
                                    confidence=1.0,
                                    event_phase="critical",
                                    timestamp=now,
                                )

                                try:
                                    queue.put_nowait(snapshot)
                                except asyncio.QueueFull:
                                    log.warning(
                                        "binance_queue_full_drop",
                                        market_id=self._config.market_id,
                                        symbol=self._symbol,
                                    )

                                self._last_emit_time = now
                                self._last_threshold_state = threshold_state

                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                raise ConnectionError(f"Binance WS message type: {msg.type!s}")

                except asyncio.CancelledError:
                    raise
                except Exception:
                    tripped = self._breaker.record()
                    log.error(
                        "binance_ws_error_reconnecting",
                        market_id=self._config.market_id,
                        symbol=self._symbol,
                        backoff_s=backoff_s,
                        errors_in_window=self._breaker.recent_errors,
                        exc_info=True,
                    )
                    if tripped:
                        log.critical(
                            "binance_ws_breaker_tripped",
                            market_id=self._config.market_id,
                            symbol=self._symbol,
                            errors_in_window=self._breaker.recent_errors,
                        )
                        self._running = False
                        break

                    jitter_s = random.uniform(0.0, 0.25 * backoff_s)
                    await asyncio.sleep(backoff_s + jitter_s)
                    backoff_s = min(max_backoff_s, backoff_s * 2.0)
        finally:
            if self._session is not None and not self._session.closed:
                await self._session.close()
            log.info(
                "binance_adapter_stopped",
                market_id=self._config.market_id,
                symbol=self._symbol,
            )
