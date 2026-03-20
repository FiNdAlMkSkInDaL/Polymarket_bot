"""
Whale wallet monitor — real-time WebSocket-based CTF transfer detection
on Polygon via ``eth_subscribe`` (logs).

Replaces the previous REST polling approach with a persistent WSS
connection to a Polygon RPC node, reducing whale detection latency
from seconds to milliseconds.

Key features:
  - WebSocket subscription to TransferSingle / TransferBatch events on
    the Polymarket CTF contract (``0x4D9702590A32765052304E32E116992d00a71943``).
  - Automatic reconnection with exponential backoff + jitter.
  - Heartbeat watchdog — forces reconnect if the stream goes silent.
  - Configurable whale_threshold_shares for anonymous large-transfer detection.
  - Wallet-cluster deduplication (funded-from-same-CEX grouping).
  - Full interface compatibility: downstream modules (PanicDetector,
    PositionManager, EdgeFilter) continue using has_confluence() etc.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import httpx
import websockets
import websockets.exceptions

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
CTF_CONTRACT = "0x4D9702590A32765052304E32E116992d00a71943"

# ── ERC-1155 event signatures (keccak256 hashes) ──────────────────────
# TransferSingle(address operator, address from, address to, uint256 id, uint256 value)
TRANSFER_SINGLE_TOPIC = (
    "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
)
# TransferBatch(address operator, address from, address to, uint256[] ids, uint256[] values)
TRANSFER_BATCH_TOPIC = (
    "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"
)


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
    """Real-time WebSocket-based CTF transfer monitor on Polygon.

    Subscribes to ``eth_subscribe("logs", ...)`` on the CTF contract to
    receive TransferSingle and TransferBatch events with sub-second latency.

    Falls back to REST polling via Polygonscan if no WSS URL is configured.
    """

    # ── Exponential backoff constants ──────────────────────────────────────
    _BACKOFF_BASE: float = 1.0
    _BACKOFF_MAX: float = 60.0

    def __init__(
        self,
        whale_wallets: list[str] | None = None,
        poll_interval: int | None = None,
        zscore_fn: Callable[[], float] | None = None,
        zscore_threshold: float | None = None,
        *,
        wss_url: str | None = None,
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
        self._wallet_set: set[str] = set(self.wallets)
        self.baseline_interval = float(poll_interval or settings.whale_poll_interval_seconds)
        self._recent: list[WhaleActivity] = []
        self._recent_tx_hashes: set[str] = set()  # O(1) dedup for tx hashes
        self._running = False
        self._last_block: int = 0

        # WebSocket URL (prefer explicit, then config, then env)
        self._wss_url: str = (
            wss_url
            or settings.polygon_rpc_wss_url
            or os.getenv("POLYGON_RPC_WSS_URL", "")
        )
        self._whale_threshold: float = settings.whale_threshold_shares
        self._heartbeat_timeout: float = settings.whale_ws_heartbeat_s
        self._last_message_time: float = 0.0
        self.reconnect_count: int = 0

        # Token ID → market mapping (set by bot after discovery)
        self._token_to_market: dict[str, str] = {}  # asset_id → "yes"/"no"
        self._asset_to_condition: dict[str, str] = {}  # asset_id → condition_id

        # ── Adaptive polling via panic Z-score callback ────────────────────
        self._zscore_fn = zscore_fn
        self._zscore_threshold = (
            zscore_threshold
            or settings.strategy.whale_zscore_threshold
        )

        # ── Wallet cluster detection (funded-from-same-CEX grouping) ───────
        self._clusters: dict[str, str] = {}  # wallet → cluster_id (funder)
        self._cluster_map_built_at: float = 0.0

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

    # ── public API (unchanged interface) ────────────────────────────────────
    async def start(self) -> None:
        """Begin monitoring.  Uses WSS streaming if URL is configured,
        otherwise falls back to REST polling."""
        self._running = True

        # Build initial cluster map
        try:
            await self._maybe_rebuild_clusters()
        except Exception:
            log.warning("whale_initial_cluster_build_error", exc_info=True)

        if self._wss_url:
            log.info(
                "whale_monitor_started_wss",
                wallets=len(self.wallets),
                threshold=self._whale_threshold,
            )
            await self._run_wss_loop()
        else:
            log.info("whale_monitor_started_poll", wallets=len(self.wallets))
            await self._run_poll_loop()

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
        """Return True if ≥ *min_wallets* distinct **entities** bought the NO token.

        Uses wallet-cluster deduplication: wallets funded from the same CEX
        deposit address are treated as a single entity.
        """
        lookback = lookback_seconds or settings.whale_lookback_seconds
        cutoff = time.time() - lookback
        wallets = {
            a.wallet
            for a in self._recent
            if a.market_token_id.lower() == no_token_id.lower()
            and a.direction == "buy_no"
            and a.timestamp >= cutoff
        }
        # Deduplicate by cluster
        entities = {self._clusters.get(w, w) for w in wallets}
        return len(entities) >= min_wallets

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

    def get_unique_entity_count(
        self,
        token_id: str,
        lookback_seconds: int | None = None,
    ) -> int:
        """Return the number of unique whale entities active on *token_id*.

        Entities are determined by wallet-cluster deduplication.
        """
        lookback = lookback_seconds or settings.whale_lookback_seconds
        cutoff = time.time() - lookback
        wallets = {
            a.wallet
            for a in self._recent
            if a.market_token_id.lower() == token_id.lower()
            and a.timestamp >= cutoff
        }
        entities = {self._clusters.get(w, w) for w in wallets}
        return len(entities)

    @property
    def recent_activity(self) -> list[WhaleActivity]:
        return list(self._recent)

    # ── WSS streaming internals ─────────────────────────────────────────────

    async def _run_wss_loop(self) -> None:
        """Outer reconnection loop with exponential backoff."""
        attempt = 0
        # Schedule periodic cluster rebuild as a sibling task
        cluster_task = asyncio.create_task(
            self._cluster_rebuild_loop(), name="whale_cluster_loop"
        )
        try:
            while self._running:
                try:
                    await self._connect_and_stream()
                    attempt = 0  # clean disconnect resets
                except (
                    websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.InvalidStatus,
                    ConnectionError,
                    OSError,
                    asyncio.TimeoutError,
                ) as exc:
                    attempt += 1
                    self.reconnect_count += 1
                    sleep_time = min(
                        self._BACKOFF_MAX,
                        self._BACKOFF_BASE * (2 ** attempt),
                    ) + random.uniform(0, 1)
                    log.warning(
                        "whale_wss_disconnected",
                        error=str(exc),
                        retry_in=round(sleep_time, 2),
                        attempt=attempt,
                        reconnect_count=self.reconnect_count,
                    )
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    self._running = False
                    break
        finally:
            cluster_task.cancel()
            try:
                await cluster_task
            except asyncio.CancelledError:
                pass

    async def _connect_and_stream(self) -> None:
        """Open WSS, subscribe to CTF logs, and consume events."""
        async with websockets.connect(
            self._wss_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # Subscribe to CTF contract logs (TransferSingle + TransferBatch)
            subscribe_msg = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": [
                    "logs",
                    {
                        "address": CTF_CONTRACT,
                        "topics": [[TRANSFER_SINGLE_TOPIC, TRANSFER_BATCH_TOPIC]],
                    },
                ],
            })
            await ws.send(subscribe_msg)

            # Read subscription confirmation
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            resp = json.loads(resp_raw)
            sub_id = resp.get("result")
            if not sub_id:
                log.error("whale_wss_subscribe_failed", response=resp)
                return
            log.info("whale_wss_subscribed", subscription_id=sub_id)

            self._last_message_time = time.time()

            # Consume event stream with heartbeat watchdog
            while self._running:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=self._heartbeat_timeout
                    )
                except asyncio.TimeoutError:
                    # No event in heartbeat window; ping to verify liveness
                    # before deciding to reconnect (avoids churn on quiet periods)
                    try:
                        pong_waiter = await ws.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10.0)
                        self._last_message_time = time.time()
                        continue
                    except Exception:
                        pass
                    elapsed = time.time() - self._last_message_time
                    log.warning(
                        "whale_wss_heartbeat_stale",
                        silence_s=round(elapsed, 1),
                        threshold_s=self._heartbeat_timeout,
                    )
                    # Force reconnect by breaking out
                    return

                self._last_message_time = time.time()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                # eth_subscribe notifications have method == "eth_subscription"
                if msg.get("method") != "eth_subscription":
                    continue

                params = msg.get("params", {})
                log_entry = params.get("result", {})
                if log_entry:
                    self._process_log(log_entry)

    def _process_log(self, log_entry: dict) -> None:
        """Decode a raw EVM log into WhaleActivity records."""
        try:
            tx_hash = log_entry.get("transactionHash", "")
            if tx_hash in self._recent_tx_hashes:
                return

            topics = log_entry.get("topics", [])
            if not topics:
                return

            data_hex = log_entry.get("data", "0x")
            event_sig = topics[0].lower()

            if event_sig == TRANSFER_SINGLE_TOPIC.lower():
                self._decode_transfer_single(topics, data_hex, tx_hash)
            elif event_sig == TRANSFER_BATCH_TOPIC.lower():
                self._decode_transfer_batch(topics, data_hex, tx_hash)

            # Trim stale entries (keep last hour)
            self._trim_stale()

        except Exception as exc:
            log.debug("whale_log_decode_error", error=str(exc), exc_info=True)

    def _decode_transfer_single(
        self, topics: list[str], data_hex: str, tx_hash: str,
    ) -> None:
        """Decode TransferSingle(operator, from, to, id, value).

        Topics layout:
          [0] event signature
          [1] operator (indexed)
          [2] from     (indexed)
          [3] to       (indexed)

        Data layout (non-indexed):
          bytes  0..31  = id     (uint256)
          bytes 32..63  = value  (uint256)
        """
        if len(topics) < 4:
            return

        from_addr = _topic_to_address(topics[2])
        to_addr = _topic_to_address(topics[3])

        data_bytes = bytes.fromhex(data_hex[2:]) if data_hex.startswith("0x") else b""
        if len(data_bytes) < 64:
            return

        token_id = int.from_bytes(data_bytes[0:32], "big")
        value = int.from_bytes(data_bytes[32:64], "big")

        self._emit_activity(from_addr, to_addr, str(token_id), value, tx_hash)

    def _decode_transfer_batch(
        self, topics: list[str], data_hex: str, tx_hash: str,
    ) -> None:
        """Decode TransferBatch(operator, from, to, ids[], values[]).

        Topics: same as TransferSingle (operator, from, to indexed).
        Data: ABI-encoded (uint256[], uint256[]).
        """
        if len(topics) < 4:
            return

        from_addr = _topic_to_address(topics[2])
        to_addr = _topic_to_address(topics[3])

        data_bytes = bytes.fromhex(data_hex[2:]) if data_hex.startswith("0x") else b""
        if len(data_bytes) < 128:
            return

        try:
            # ABI dynamic array: offset_ids (32) | offset_vals (32) |
            #   len_ids (32) | ids... | len_vals (32) | vals...
            offset_ids = int.from_bytes(data_bytes[0:32], "big")
            offset_vals = int.from_bytes(data_bytes[32:64], "big")

            # Read ids array
            ids_len = int.from_bytes(
                data_bytes[offset_ids : offset_ids + 32], "big"
            )
            ids: list[int] = []
            for i in range(ids_len):
                start = offset_ids + 32 + i * 32
                ids.append(int.from_bytes(data_bytes[start : start + 32], "big"))

            # Read values array
            vals_len = int.from_bytes(
                data_bytes[offset_vals : offset_vals + 32], "big"
            )
            vals: list[int] = []
            for i in range(vals_len):
                start = offset_vals + 32 + i * 32
                vals.append(int.from_bytes(data_bytes[start : start + 32], "big"))

            for token_id, value in zip(ids, vals):
                self._emit_activity(
                    from_addr, to_addr, str(token_id), value, tx_hash,
                )
        except Exception as exc:
            log.debug("whale_batch_decode_error", error=str(exc))

    def _emit_activity(
        self,
        from_addr: str,
        to_addr: str,
        token_id: str,
        value: int,
        tx_hash: str,
    ) -> None:
        """Create WhaleActivity if the transfer involves a tracked wallet
        or exceeds the whale_threshold_shares."""
        from_lower = from_addr.lower()
        to_lower = to_addr.lower()

        is_tracked_from = from_lower in self._wallet_set
        is_tracked_to = to_lower in self._wallet_set
        is_large = value >= self._whale_threshold

        if not (is_tracked_from or is_tracked_to or is_large):
            return

        # Determine direction
        token_lower = token_id.lower()
        side = self._token_to_market.get(token_lower, "unknown")

        if is_tracked_to or (is_large and not is_tracked_from):
            wallet = to_lower if is_tracked_to else to_lower
            direction = f"buy_{side}" if side != "unknown" else "buy_no"
        elif is_tracked_from:
            wallet = from_lower
            direction = f"sell_{side}" if side != "unknown" else "sell_no"
        else:
            return

        # Deduplicate by (tx_hash, token_id, wallet) to handle batch logs
        dedup_key = f"{tx_hash}:{token_id}:{wallet}"
        if dedup_key in self._recent_tx_hashes:
            return

        activity = WhaleActivity(
            wallet=wallet,
            market_token_id=token_id,
            direction=direction,
            amount=float(value),
            timestamp=time.time(),
            tx_hash=tx_hash,
        )

        self._recent.append(activity)
        self._recent_tx_hashes.add(dedup_key)

        log.info(
            "whale_activity",
            wallet=wallet[:10],
            direction=direction,
            token=token_id[:16],
            amount=value,
            source="wss",
        )

    # ── REST polling fallback ───────────────────────────────────────────────

    async def _run_poll_loop(self) -> None:
        """Legacy REST polling loop (used when no WSS URL configured)."""
        while self._running:
            try:
                await self._poll()
                await self._maybe_rebuild_clusters()
            except Exception as exc:
                log.warning("whale_poll_error", error=str(exc))
            await asyncio.sleep(self.baseline_interval)

    async def _poll(self) -> None:
        """Fetch recent ERC-1155 transfer events from Polygonscan."""
        if not self.wallets:
            return
        if not settings.polygonscan_api_key:
            return

        self._trim_stale()

        api_url = "https://api.polygonscan.com/api"

        async with httpx.AsyncClient(timeout=20) as client:
            async def _fetch_wallet(wallet: str) -> list[tuple[dict, str]]:
                """Fetch recent txs for a single wallet."""
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
                        return []
                    return [(tx, wallet) for tx in txs[:20]]
                except Exception as exc:
                    log.debug("whale_poll_wallet_error", wallet=wallet[:10], error=str(exc))
                    return []

            results = await asyncio.gather(*[_fetch_wallet(w) for w in self.wallets])
            for wallet_txs in results:
                for tx, wallet in wallet_txs:
                    self._process_tx(tx, wallet)

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

            # Deduplicate (O(1) via set)
            if tx_hash in self._recent_tx_hashes:
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
            self._recent_tx_hashes.add(tx_hash)
            self._trim_stale()

            log.info(
                "whale_activity",
                wallet=wallet[:10],
                direction=direction,
                token=token_id[:16],
                amount=amount,
                source="poll",
            )

        except (ValueError, TypeError, KeyError) as exc:
            log.debug("whale_tx_parse_error", error=str(exc))

    # ── Shared helpers ──────────────────────────────────────────────────────

    def _trim_stale(self) -> None:
        """Remove entries older than 1 hour."""
        cutoff = time.time() - 3600
        stale = [a for a in self._recent if a.timestamp < cutoff]
        if not stale:
            return
        stale_hashes = set()
        for a in stale:
            stale_hashes.add(a.tx_hash)
            # Also remove dedup keys (WSS format: tx:token:wallet)
            stale_hashes.add(f"{a.tx_hash}:{a.market_token_id}:{a.wallet}")
        self._recent_tx_hashes -= stale_hashes
        self._recent = [a for a in self._recent if a.timestamp >= cutoff]

    # ── Wallet cluster detection ───────────────────────────────────────────

    async def _cluster_rebuild_loop(self) -> None:
        """Periodically rebuild clusters as a background task."""
        while self._running:
            try:
                await self._maybe_rebuild_clusters()
            except Exception:
                log.debug("whale_cluster_rebuild_error", exc_info=True)
            await asyncio.sleep(600)  # check every 10 min

    async def _maybe_rebuild_clusters(self) -> None:
        """Rebuild the wallet cluster map if stale (every N hours)."""
        refresh_s = settings.strategy.whale_cluster_refresh_hours * 3600.0
        if (time.time() - self._cluster_map_built_at) < refresh_s:
            return
        await self._build_cluster_map()

    async def _build_cluster_map(self) -> None:
        """Query Polygonscan for the first funder of each whale wallet.

        Wallets funded from the same address are grouped into a cluster.
        Chain funding (A → B → C) is resolved transitively.
        """
        if not settings.polygonscan_api_key:
            self._cluster_map_built_at = time.time()
            return

        api_url = "https://api.polygonscan.com/api"
        funder_of: dict[str, str] = {}  # wallet → first funder address
        wallet_set = set(self.wallets)

        async with httpx.AsyncClient(timeout=20) as client:
            for wallet in self.wallets:
                try:
                    params = {
                        "module": "account",
                        "action": "txlist",
                        "address": wallet,
                        "startblock": 0,
                        "endblock": 99999999,
                        "sort": "asc",
                        "page": 1,
                        "offset": 20,
                        "apikey": settings.polygonscan_api_key,
                    }
                    resp = await client.get(api_url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    txs = data.get("result", [])
                    if not isinstance(txs, list):
                        continue

                    # Find the first incoming ETH/MATIC transfer
                    for tx in txs:
                        to_addr = tx.get("to", "").lower()
                        from_addr = tx.get("from", "").lower()
                        value = int(tx.get("value", "0"))
                        if to_addr == wallet and value > 0 and from_addr:
                            funder_of[wallet] = from_addr
                            break

                except Exception as exc:
                    log.debug(
                        "cluster_lookup_error",
                        wallet=wallet[:10],
                        error=str(exc),
                    )
                # Rate-limit: small delay between lookups
                await asyncio.sleep(0.25)

        # Resolve transitive chains:  if funder is also a tracked wallet,
        # walk up the chain until we find a non-tracked address.
        def _resolve(w: str, depth: int = 0) -> str:
            if depth > 10:  # prevent infinite loops
                return w
            funder = funder_of.get(w)
            if not funder:
                return w
            if funder in wallet_set:
                return _resolve(funder, depth + 1)
            return funder

        self._clusters = {w: _resolve(w) for w in self.wallets}
        self._cluster_map_built_at = time.time()

        # Log cluster summary
        cluster_sizes = Counter(self._clusters.values())
        multi = {cid: cnt for cid, cnt in cluster_sizes.items() if cnt > 1}
        log.info(
            "whale_clusters_built",
            total_wallets=len(self.wallets),
            unique_entities=len(cluster_sizes),
            multi_wallet_clusters=len(multi),
        )


# ── Module-level helpers ────────────────────────────────────────────────
def _topic_to_address(topic_hex: str) -> str:
    """Extract a 20-byte address from a 32-byte ABI-encoded topic."""
    # Topics are 0x-prefixed 64-char hex (32 bytes), address is last 20 bytes
    raw = topic_hex[-40:]
    return f"0x{raw}"
