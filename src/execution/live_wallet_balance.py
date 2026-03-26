from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Callable, Iterable, Protocol

from src.core.logger import get_logger
from src.execution.clob_transport import (
    ClobTransportCircuitOpenError,
    ClobTransportRateLimitError,
    ClobTransportTimeoutError,
)
from src.execution.venue_adapter_interface import VenueAdapter


class _BalanceLogger(Protocol):
    def warning(self, event: str, **kwargs: object) -> object:
        ...


class LiveWalletBalanceProvider:
    def __init__(
        self,
        venue_adapter: VenueAdapter,
        *,
        tracked_assets: Iterable[str],
        initial_balances: dict[str, Decimal] | None = None,
        now_ms: Callable[[], int] | None = None,
        sleep_fn: Callable[[float], asyncio.Future[object] | object] | None = None,
        logger: _BalanceLogger | None = None,
    ) -> None:
        self._venue_adapter = venue_adapter
        self._tracked_assets = tuple(self._normalize_asset_symbol(asset) for asset in tracked_assets)
        self._balances: dict[str, Decimal] = {}
        self._last_updated_ms: dict[str, int] = {}
        self._now_ms = now_ms
        self._sleep_fn = sleep_fn or asyncio.sleep
        self._logger = logger or get_logger(__name__)

        if initial_balances:
            for asset_symbol, balance in initial_balances.items():
                self._set_cached_balance(self._normalize_asset_symbol(asset_symbol), balance, None)

    def get_available_margin(self, asset_symbol: str) -> Decimal:
        return self._balances.get(self._normalize_asset_symbol(asset_symbol), Decimal("0"))

    def get_last_updated_ms(self, asset_symbol: str) -> int | None:
        return self._last_updated_ms.get(self._normalize_asset_symbol(asset_symbol))

    async def poll_balance_loop(self, interval_ms: int) -> None:
        if not isinstance(interval_ms, int) or interval_ms <= 0:
            raise ValueError("interval_ms must be a positive int")
        sleep_seconds = interval_ms / 1000
        while True:
            try:
                for asset_symbol in self._tracked_assets:
                    balance = self._venue_adapter.get_wallet_balance(asset_symbol)
                    self._set_cached_balance(asset_symbol, balance, self._current_timestamp_ms())
            except asyncio.CancelledError:
                raise
            except (ClobTransportCircuitOpenError, ClobTransportRateLimitError, ClobTransportTimeoutError) as exc:
                self._logger.warning(
                    "wallet_balance_poll_failed",
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )
            await self._sleep(sleep_seconds)

    async def _sleep(self, seconds: float) -> None:
        result = self._sleep_fn(seconds)
        if asyncio.iscoroutine(result):
            await result

    def _set_cached_balance(self, asset_symbol: str, balance: Decimal, updated_ms: int | None) -> None:
        if not isinstance(balance, Decimal) or not balance.is_finite():
            raise ValueError("balance must be a finite Decimal")
        if balance < Decimal("0"):
            raise ValueError("balance must be greater than or equal to 0")
        self._balances[asset_symbol] = balance
        if updated_ms is not None:
            self._last_updated_ms[asset_symbol] = updated_ms

    def _current_timestamp_ms(self) -> int | None:
        if self._now_ms is None:
            return None
        return int(self._now_ms())

    @staticmethod
    def _normalize_asset_symbol(asset_symbol: str) -> str:
        normalized = str(asset_symbol or "").strip().upper()
        if not normalized:
            raise ValueError("asset_symbol must be a non-empty string")
        return normalized