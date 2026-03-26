"""Shadow sweep detector that wraps the isolated Polygon mempool monitor."""

from __future__ import annotations

from typing import Callable

from src.core.config import settings
from src.events.mev_events import ShadowSweepSignal
from src.execution.mempool_monitor import MempoolMonitor, PendingTransactionMatch


def _normalize_direction(direction: str) -> str:
    value = str(direction or "").strip().upper()
    if value not in {"YES", "NO"}:
        raise ValueError(f"Unsupported shadow sweep direction: {direction!r}")
    return value


def _default_market_id(match: PendingTransactionMatch) -> str:
    metadata = match.metadata or {}
    return str(metadata.get("market_id") or metadata.get("condition_id") or "").strip()


def _default_direction(_: PendingTransactionMatch) -> str:
    return "YES"


class ShadowSweepDetector:
    """Translate mempool whale detection into a shadow-sweep signal."""

    def __init__(
        self,
        mempool_monitor: MempoolMonitor,
        *,
        market_id_resolver: Callable[[PendingTransactionMatch], str] | None = None,
        direction_resolver: Callable[[PendingTransactionMatch], str] | None = None,
        max_capital: float | None = None,
        premium_pct: float | None = None,
    ) -> None:
        self._mempool_monitor = mempool_monitor
        self._market_id_resolver = market_id_resolver or _default_market_id
        self._direction_resolver = direction_resolver or _default_direction
        self._max_capital = float(
            settings.strategy.max_trade_size_usd if max_capital is None else max_capital
        )
        self._premium_pct = float(
            settings.mev_shadow_sweep_premium_pct if premium_pct is None else premium_pct
        )
        self._last_emitted_tx_hash: str = ""
        self._callbacks: list[Callable[[ShadowSweepSignal], object]] = []

    @property
    def is_whale_incoming(self) -> bool:
        return self._mempool_monitor.is_whale_incoming

    def register_callback(self, callback: Callable[[ShadowSweepSignal], object]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def evaluate(self) -> ShadowSweepSignal | None:
        if not self._mempool_monitor.is_whale_incoming:
            return None

        matches = self._mempool_monitor.recent_matches
        if not matches:
            return None

        trigger_match = matches[-1]
        if trigger_match.tx_hash == self._last_emitted_tx_hash:
            return None

        target_market_id = str(self._market_id_resolver(trigger_match) or "").strip()
        if not target_market_id:
            return None

        direction = _normalize_direction(self._direction_resolver(trigger_match))
        signal = ShadowSweepSignal(
            target_market_id=target_market_id,
            direction=direction,
            max_capital=self._max_capital,
            premium_pct=self._premium_pct,
        )
        self._last_emitted_tx_hash = trigger_match.tx_hash
        for callback in self._callbacks:
            callback(signal)
        return signal

    async def process_pending_hash(self, tx_hash: str) -> ShadowSweepSignal | None:
        await self._mempool_monitor.process_pending_hash(tx_hash)
        return self.evaluate()

    def ingest_transaction(self, transaction: dict | None, *, seen_at: float | None = None) -> ShadowSweepSignal | None:
        self._mempool_monitor.ingest_transaction(transaction, seen_at=seen_at)
        return self.evaluate()
