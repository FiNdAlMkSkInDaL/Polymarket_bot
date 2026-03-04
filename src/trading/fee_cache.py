"""
Fee-rate cache — queries ``GET /fee-rate`` for each token and caches the
result with a configurable TTL.

Polymarket's dynamic taker-fee curve peaks near the 50% probability mark
(up to ~156 bps for Crypto 5m/15m markets).  Makers pay 0 bps.

Usage::

    cache = FeeCache()
    bps = await cache.get_fee_rate(token_id)   # e.g. 156
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


class FeeCache:
    """Per-token fee-rate cache backed by the CLOB ``/fee-rate`` endpoint.

    Parameters
    ----------
    ttl_s:
        Cache entry time-to-live in seconds.
    default_bps:
        Conservative fallback when the endpoint is unreachable.
    base_url:
        CLOB HTTP root (defaults to ``settings.clob_http_url``).
    """

    def __init__(
        self,
        *,
        ttl_s: int | None = None,
        default_bps: int | None = None,
        base_url: str | None = None,
        max_size: int = 500,
    ):
        strat = settings.strategy
        self._ttl = ttl_s if ttl_s is not None else strat.fee_cache_ttl_s
        self._default_bps = default_bps if default_bps is not None else strat.fee_default_bps
        self._base_url = (base_url or settings.clob_http_url).rstrip("/")
        # {token_id: (fee_rate_bps, fetched_at)}
        self._cache: dict[str, tuple[int, float]] = {}
        self._max_size = max_size
        self._locks: dict[str, asyncio.Lock] = {}  # Per-token locks for concurrent fetches
        self._global_lock = asyncio.Lock()  # Protects _locks dict creation
        self._http_client: httpx.AsyncClient | None = None  # Persistent connection pool

    # ── Public API ──────────────────────────────────────────────────────────

    async def get_fee_rate(self, token_id: str) -> int:
        """Return the taker fee for *token_id* in basis points.

        Uses a cached value if fresh; otherwise fetches from the CLOB.
        Task-safe via per-token locks to avoid serialising unrelated tokens.
        """
        now = time.time()
        cached = self._cache.get(token_id)
        if cached is not None:
            bps, ts = cached
            if now - ts < self._ttl:
                return bps

        # Get or create a per-token lock (lightweight; protects only this token)
        async with self._global_lock:
            if token_id not in self._locks:
                self._locks[token_id] = asyncio.Lock()
            token_lock = self._locks[token_id]

        async with token_lock:
            # Double-check after acquiring lock
            cached = self._cache.get(token_id)
            if cached is not None and now - cached[1] < self._ttl:
                return cached[0]

            bps = await self._fetch(token_id)
            self._cache[token_id] = (bps, now)
            self._evict_if_over_limit()
            return bps

    def _evict_if_over_limit(self) -> None:
        """Evict stale and then oldest entries if cache exceeds max_size."""
        if len(self._cache) <= self._max_size:
            return
        now = time.time()
        # First pass: drop expired entries
        stale = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for k in stale:
            del self._cache[k]
        # Second pass: LRU evict oldest if still over limit
        if len(self._cache) > self._max_size:
            oldest = sorted(self._cache, key=lambda k: self._cache[k][1])
            for k in oldest[: len(self._cache) - self._max_size]:
                del self._cache[k]

    def get_fee_rate_sync(self, token_id: str) -> int:
        """Non-async accessor — returns cached value or the default.

        Useful in hot paths where an ``await`` is undesirable.
        """
        cached = self._cache.get(token_id)
        if cached is not None:
            return cached[0]
        return self._default_bps

    async def prefetch(self, token_ids: list[str]) -> None:
        """Bulk-warm the cache for a set of tokens."""
        tasks = [self.get_fee_rate(tid) for tid in token_ids]
        await asyncio.gather(*tasks, return_exceptions=True)

    def maker_fee_bps(self) -> int:
        """Maker fee is always 0 on Polymarket."""
        return 0

    # ── Fee math helpers ────────────────────────────────────────────────────

    @staticmethod
    def fee_cents(price: float, fee_bps: int) -> float:
        """Convert a fee rate + price to a cost in cents.

        ``fee_cents(0.50, 156) → 0.78``
        """
        return price * fee_bps / 10_000 * 100

    def compute_fee_floor_cents(
        self,
        entry_price: float,
        target_price: float,
        entry_fee_bps: int,
        exit_fee_bps: int,
        desired_margin_cents: float | None = None,
    ) -> float:
        """Minimum spread (in cents) that covers both legs + margin.

        .. math::

            \\Delta_{floor} = \\frac{F_{entry}}{10000} P_{entry} \\times 100
                            + \\frac{F_{exit}}{10000} P_{target} \\times 100
                            + M
        """
        margin = (
            desired_margin_cents
            if desired_margin_cents is not None
            else settings.strategy.desired_margin_cents
        )
        entry_cost = self.fee_cents(entry_price, entry_fee_bps)
        exit_cost = self.fee_cents(target_price, exit_fee_bps)
        return round(entry_cost + exit_cost + margin, 4)

    # ── Internal ────────────────────────────────────────────────────────────

    async def _fetch(self, token_id: str) -> int:
        """Query the CLOB for the current taker fee rate."""
        url = f"{self._base_url}/fee-rate"
        try:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=5.0)
            resp = await self._http_client.get(url, params={"tokenID": token_id})
            resp.raise_for_status()
            data = resp.json()
            bps = int(float(data.get("feeRateBps", self._default_bps)))
            log.debug("fee_rate_fetched", token_id=token_id[:16], bps=bps)
            return bps
        except Exception as exc:
            log.warning(
                "fee_rate_fetch_failed",
                token_id=token_id[:16],
                error=str(exc),
                fallback_bps=self._default_bps,
            )
            return self._default_bps

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None


# ── Startup validation ─────────────────────────────────────────────────────

async def validate_fee_model(
    token_ids: list[str],
    mid_prices: list[float],
    *,
    tolerance_bps: int = 5,
    base_url: str | None = None,
) -> bool:
    """Compare the local fee formula against the CLOB REST endpoint.

    Probes up to ``len(token_ids)`` tokens.  For each, fetches the
    exchange's fee rate and compares it to our parabolic model
    ``Fee(p) = f_max · 4·p·(1-p)``.  If any diverge by more than
    *tolerance_bps*, a warning is logged and the function returns
    ``False``.

    Returns ``True`` (model validated) or ``False`` (divergence found
    or endpoint unreachable).  Never raises — safe to call on startup.
    """
    from src.trading.fees import get_fee_rate as local_fee_rate

    if not token_ids:
        return True

    cache = FeeCache(base_url=base_url)
    ok = True

    for token_id, mid in zip(token_ids, mid_prices):
        try:
            remote_bps = await cache.get_fee_rate(token_id)
            local_bps = round(local_fee_rate(mid) * 10_000)
            delta = abs(remote_bps - local_bps)
            if delta > tolerance_bps:
                log.warning(
                    "fee_model_divergence",
                    token_id=token_id[:16],
                    mid_price=round(mid, 4),
                    local_bps=local_bps,
                    remote_bps=remote_bps,
                    delta_bps=delta,
                    tolerance_bps=tolerance_bps,
                )
                ok = False
            else:
                log.info(
                    "fee_model_validated",
                    token_id=token_id[:16],
                    local_bps=local_bps,
                    remote_bps=remote_bps,
                )
        except Exception as exc:
            log.warning(
                "fee_model_validation_error",
                token_id=token_id[:16],
                error=str(exc),
            )
            ok = False

    return ok
