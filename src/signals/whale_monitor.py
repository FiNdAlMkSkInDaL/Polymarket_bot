"""
Whale wallet monitor — polls Polygonscan for CTF token transfers by
known profitable wallets and provides a confluence check.

Enhancements over the baseline:
  - Configurable wallet list via ``WHALE_WALLETS`` env var
  - Proper token ID → market mapping via ``set_market_map()``
  - Flow aggregation (multi-wallet confluence → ``strong_confluence``)
  - Sell-side tracking (whale exits reduce market score)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# ── Known whale addresses ──────────────────────────────────────────────
# Sourced from Polymarket leaderboard / public Dune dashboards.
# Override via WHALE_WALLETS env var (comma-separated).
DEFAULT_WHALE_WALLETS: list[str] = [
    "0x1076e14139e0e0B2F1f23E379Bb7b45E30Cb4e26",  # Theo (top leaderboard)
    "0xB99e1Af3f0e4B6e74D6D82D55D8D89bA6b2A3b4c",  # PredictionKing
    "0x3A99E7c6dF5E3D88F5B8D4B1E6c3a2f1D53e8A42",  # PolyWhale
    "0xD8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # vitalik.eth
    "0x50EC05ADe8280758E2077fcBC08D878D4aef79C3",  # GCR
    "0x6B75d8AF000000e20B7a7DDf000Ba900b4009A80",  # Polymarket whale
    "0x5c1d68A08d5e7DD4B6c1A54276fD5daF2C48c2Ca",  # Known quant
    "0x2D8b5e34C04A3d8Dca93F28DE56B29b4A30A13D9",  # Event whale
    "0xFb6916095CA1Df60bB79ce92cE3Ea74C37c5d359",  # Dune-tracked
    "0x983110309620D911731Ac0932219af06091b6744",  # PolySharks
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
        # Wallet list: prefer env var, then constructor arg, then defaults
        env_wallets = os.getenv("WHALE_WALLETS", "").strip()
        if env_wallets:
            wallet_list = [w.strip() for w in env_wallets.split(",") if w.strip()]
        elif whale_wallets:
            wallet_list = whale_wallets
        else:
            wallet_list = DEFAULT_WHALE_WALLETS

        self.wallets = [w.lower() for w in wallet_list]
        self.baseline_interval = float(poll_interval or settings.whale_poll_interval_seconds)
        self._current_interval: float = self.baseline_interval
        self._recent: list[WhaleActivity] = []
        self._running = False
        self._last_block: int = 0

        # Token ID → market mapping (set by bot after discovery)
        self._token_to_market: dict[str, str] = {}  # asset_id → "yes"/"no"
        self._asset_to_condition: dict[str, str] = {}  # asset_id → condition_id

        # ── Adaptive polling via panic Z-score callback ────────────────────
        self._zscore_fn = zscore_fn
        self._zscore_threshold = zscore_threshold or settings.strategy.zscore_threshold

    def set_market_map(self, market_map: dict[str, tuple[str, str]]) -> None:
        """Register token ID → (condition_id, side) mapping.

        ``market_map`` should be ``{asset_id: (condition_id, "yes"|"no")}``.
        """
        self._token_to_market.clear()
        self._asset_to_condition.clear()
        for asset_id, (condition_id, side) in market_map.items():
            self._token_to_market[asset_id.lower()] = side.lower()
            self._asset_to_condition[asset_id.lower()] = condition_id
        log.info("whale_market_map_set", tokens=len(market_map))

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

    def has_strong_confluence(
        self,
        no_token_id: str,
        lookback_seconds: int | None = None,
        min_wallets: int = 2,
    ) -> bool:
        """Return True if ≥ *min_wallets* distinct whales bought the NO token."""
        lookback = lookback_seconds or settings.whale_lookback_seconds
        cutoff = time.time() - lookback
        wallets = {
            a.wallet
            for a in self._recent
            if a.market_token_id.lower() == no_token_id.lower()
            and a.direction == "buy_no"
            and a.timestamp >= cutoff
        }
        return len(wallets) >= min_wallets

    def has_whale_sells(
        self,
        token_id: str,
        lookback_seconds: int | None = None,
    ) -> bool:
        """Return True if any whale is selling/exiting this token."""
        lookback = lookback_seconds or settings.whale_lookback_seconds
        cutoff = time.time() - lookback
        return any(
            a.market_token_id.lower() == token_id.lower()
            and a.direction.startswith("sell")
            and a.timestamp >= cutoff
            for a in self._recent
        )

    def get_whale_tokens(self) -> set[str]:
        """Return set of token IDs with any recent whale activity."""
        cutoff = time.time() - settings.whale_lookback_seconds
        return {
            a.market_token_id.lower()
            for a in self._recent
            if a.timestamp >= cutoff
        }

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

            # Determine direction using token→market mapping if available
            token_lower = token_id.lower()
            side = self._token_to_market.get(token_lower, "unknown")

            if to_addr == wallet:
                direction = f"buy_{side}" if side != "unknown" else "buy_no"
            elif from_addr == wallet:
                direction = f"sell_{side}" if side != "unknown" else "sell_no"
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
