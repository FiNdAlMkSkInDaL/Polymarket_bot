"""
Tests for the FeeCache — per-token fee-rate caching with TTL.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.trading.fee_cache import FeeCache


class TestFeeCacheSync:
    """Synchronous helper tests (no HTTP mocking needed)."""

    def test_fee_cents_calculation(self):
        """fee_cents(price=0.50, bps=156) → 0.78 cents."""
        assert FeeCache.fee_cents(0.50, 156) == pytest.approx(0.78, abs=0.001)

    def test_fee_cents_zero_bps(self):
        assert FeeCache.fee_cents(0.50, 0) == 0.0

    def test_fee_cents_full_price(self):
        """fee_cents(1.0, 200) → 2.0 cents."""
        assert FeeCache.fee_cents(1.0, 200) == pytest.approx(2.0, abs=0.001)

    def test_maker_fee_always_zero(self):
        cache = FeeCache()
        assert cache.maker_fee_bps() == 0

    def test_get_fee_rate_sync_returns_default_when_empty(self):
        cache = FeeCache(default_bps=200)
        assert cache.get_fee_rate_sync("UNKNOWN_TOKEN") == 200

    def test_get_fee_rate_sync_returns_cached(self):
        cache = FeeCache(default_bps=200)
        # Manually seed the cache
        cache._cache["TOKEN_A"] = (156, time.time())
        assert cache.get_fee_rate_sync("TOKEN_A") == 156

    def test_compute_fee_floor_cents(self):
        cache = FeeCache(default_bps=200)
        # entry_fee = 0.47 * 100 / 10000 * 100 = 0.47 cents
        # exit_fee  = 0.56 * 100 / 10000 * 100 = 0.56 cents
        # margin = 1.0
        # total = 0.47 + 0.56 + 1.0 = 2.03 cents
        floor = cache.compute_fee_floor_cents(
            entry_price=0.47,
            target_price=0.56,
            entry_fee_bps=100,
            exit_fee_bps=100,
            desired_margin_cents=1.0,
        )
        assert floor == pytest.approx(2.03, abs=0.01)


class TestFeeCacheAsync:
    """Async tests with HTTP mocking via monkeypatch."""

    @pytest.mark.asyncio
    async def test_get_fee_rate_fallback_on_error(self):
        """When fetch fails, returns default_bps."""
        cache = FeeCache(default_bps=200, base_url="http://localhost:1")
        bps = await cache.get_fee_rate("BAD_TOKEN")
        assert bps == 200

    @pytest.mark.asyncio
    async def test_cache_respects_ttl(self):
        """Cached value should be returned within TTL."""
        cache = FeeCache(ttl_s=300, default_bps=200, base_url="http://localhost:1")
        # Seed cache manually
        cache._cache["TOKEN_X"] = (156, time.time())
        bps = await cache.get_fee_rate("TOKEN_X")
        assert bps == 156

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self):
        """Expired entries should be refetched (falls back to default)."""
        cache = FeeCache(ttl_s=1, default_bps=200, base_url="http://localhost:1")
        # Seed cache with an old timestamp
        cache._cache["TOKEN_Y"] = (100, time.time() - 10)
        bps = await cache.get_fee_rate("TOKEN_Y")
        # Should have tried to re-fetch, failed, returned default
        assert bps == 200

    @pytest.mark.asyncio
    async def test_prefetch_does_not_raise(self):
        """Prefetch should gracefully handle errors."""
        cache = FeeCache(default_bps=200, base_url="http://localhost:1")
        await cache.prefetch(["T1", "T2", "T3"])
        # All should default
        assert cache.get_fee_rate_sync("T1") == 200
        assert cache.get_fee_rate_sync("T2") == 200
