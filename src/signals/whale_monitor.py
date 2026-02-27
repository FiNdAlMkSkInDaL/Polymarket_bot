"""
Whale wallet monitor — polls Polygonscan for CTF token transfers by
known profitable wallets and provides a confluence check.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# ── Known whale addresses (public leaderboard / Dune sources) ──────────
# Populate with actual addresses from Polymarket leaderboards.
DEFAULT_WHALE_WALLETS: list[str] = [
    # Example — replace with real addresses:
    # "0xABC123...",
]

# Polymarket CTF (Conditional Token Framework) contract on Polygon
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"


@dataclass
class WhaleActivity:
    """A detected whale transaction on a specific market."""

    wallet: str
    market_token_id: str
    direction: str          # "buy_no" | "buy_yes" | "sell_no" | "sell_yes"
    amount: float
    timestamp: float
    tx_hash: str


class WhaleMonitor:
    """Periodically polls Polygonscan for whale CTF token movements.

    The polling interval adapts to current market volatility: during panic
    events (high Z-score) we poll aggressively (every ~2 s) to catch whale
    entries, then decay back to the baseline as volatility subsides.
    """

    # ── Adaptive polling bounds ────────────────────────────────────────────
    _BASELINE_MIN: float = 15.0     # slowest normal poll (seconds)
    _BASELINE_MAX: float = 30.0     # default normal poll (seconds)
    _PANIC_INTERVAL: float = 2.0    # fastest poll when Z > threshold
    _DECAY_RATE: float = 0.10       # how fast the interval recovers per tick

    def __init__(
        self,
        whale_wallets: list[str] | None = None,
        poll_interval: int | None = None,
        zscore_fn: Callable[[], float] | None = None,
        zscore_threshold: float | None = None,
    ):
        self.wallets = [w.lower() for w in (whale_wallets or DEFAULT_WHALE_WALLETS)]
        self.baseline_interval = float(poll_interval or settings.whale_poll_interval_seconds)
        self._current_interval: float = self.baseline_interval
        self._recent: list[WhaleActivity] = []
        self._running = False
        self._last_block: int = 0

        # ── Fix 5: Adaptive polling via panic Z-score callback ─────────────
        self._zscore_fn = zscore_fn
        self._zscore_threshold = zscore_threshold or settings.strategy.zscore_threshold

    # ── adaptive interval ──────────────────────────────────────────────────
    @property
    def poll_interval(self) -> float:
        return self._current_interval

    def _update_interval(self) -> None:
        """Adjust polling speed based on the latest Z-score."""
        if self._zscore_fn is None:
            return
        zscore = self._zscore_fn()
        if zscore >= self._zscore_threshold:
            # Panic regime — poll as fast as possible
            self._current_interval = self._PANIC_INTERVAL
            log.debug(
                "whale_poll_adaptive",
                zscore=round(zscore, 2),
                interval=self._current_interval,
            )
        else:
            # Decay back towards baseline
            self._current_interval = min(
                self.baseline_interval,
                self._current_interval + self._DECAY_RATE * (
                    self.baseline_interval - self._current_interval
                ) + 0.5,   # +0.5 s per tick ensures convergence
            )

    # ── public API ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Begin background polling loop."""
        self._running = True
        log.info("whale_monitor_started", wallets=len(self.wallets))
        while self._running:
            try:
                self._update_interval()
                await self._poll()
            except Exception as exc:
                log.warning("whale_poll_error", error=str(exc))
            await asyncio.sleep(self._current_interval)

    async def stop(self) -> None:
        self._running = False

    def has_confluence(
        self,
        no_token_id: str,
        lookback_seconds: int | None = None,
    ) -> bool:
        """Return True if any whale bought the NO token within the lookback window."""
        lookback = lookback_seconds or settings.whale_lookback_seconds
        cutoff = time.time() - lookback
        return any(
            a.market_token_id.lower() == no_token_id.lower()
            and a.direction == "buy_no"
            and a.timestamp >= cutoff
            for a in self._recent
        )

    @property
    def recent_activity(self) -> list[WhaleActivity]:
        return list(self._recent)

    # ── internals ───────────────────────────────────────────────────────────
    async def _poll(self) -> None:
        """Fetch recent ERC-1155 transfer events from Polygonscan."""
        if not self.wallets:
            return
        if not settings.polygonscan_api_key:
            return

        api_url = "https://api.polygonscan.com/api"

        async with httpx.AsyncClient(timeout=20) as client:
            for wallet in self.wallets:
                params = {
                    "module": "account",
                    "action": "token1155tx",
                    "address": wallet,
                    "startblock": max(self._last_block - 10, 0),
                    "endblock": 99999999,
                    "sort": "desc",
                    "apikey": settings.polygonscan_api_key,
                }
                try:
                    resp = await client.get(api_url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    txs = data.get("result", [])
                    if not isinstance(txs, list):
                        continue

                    for tx in txs[:20]:  # Only recent
                        self._process_tx(tx, wallet)

                except Exception as exc:
                    log.debug("whale_poll_wallet_error", wallet=wallet[:10], error=str(exc))

    def _process_tx(self, tx: dict, wallet: str) -> None:
        """Parse a Polygonscan ERC-1155 transfer into WhaleActivity."""
        try:
            token_id = tx.get("tokenID", "")
            from_addr = tx.get("from", "").lower()
            to_addr = tx.get("to", "").lower()
            ts = float(tx.get("timeStamp", 0))
            tx_hash = tx.get("hash", "")

            # Update block tracker
            block = int(tx.get("blockNumber", 0))
            if block > self._last_block:
                self._last_block = block

            # Deduplicate
            if any(a.tx_hash == tx_hash for a in self._recent):
                return

            # Determine direction
            if to_addr == wallet:
                direction = "buy_no"  # simplified — would need token mapping
            elif from_addr == wallet:
                direction = "sell_no"
            else:
                return

            amount = float(tx.get("tokenValue", 0))

            activity = WhaleActivity(
                wallet=wallet,
                market_token_id=token_id,
                direction=direction,
                amount=amount,
                timestamp=ts,
                tx_hash=tx_hash,
            )

            self._recent.append(activity)

            # Trim old entries (keep last hour)
            cutoff = time.time() - 3600
            self._recent = [a for a in self._recent if a.timestamp >= cutoff]

            log.info(
                "whale_activity",
                wallet=wallet[:10],
                direction=direction,
                token=token_id[:16],
                amount=amount,
            )

        except (ValueError, TypeError, KeyError) as exc:
            log.debug("whale_tx_parse_error", error=str(exc))
