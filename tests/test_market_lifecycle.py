"""Tests for src.data.market_lifecycle — three-tier state machine."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import settings
from src.data.market_discovery import MarketInfo
from src.data.market_lifecycle import (
    ActiveMarket,
    DrainingMarket,
    MarketLifecycleManager,
    ObservingMarket,
)
from src.data.market_scorer import ScoreBreakdown, compute_score


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_market(
    cid: str = "CID_A",
    question: str = "Test market?",
    volume: float = 50_000,
    liquidity: float = 30_000,
    end_days: int = 30,
) -> MarketInfo:
    return MarketInfo(
        condition_id=cid,
        question=question,
        yes_token_id=f"YES_{cid}",
        no_token_id=f"NO_{cid}",
        daily_volume_usd=volume,
        end_date=datetime.now(timezone.utc) + timedelta(days=end_days),
        active=True,
        event_id=f"EVT_{cid}",
        liquidity_usd=liquidity,
        score=0.0,
        accepting_orders=True,
    )


def _high_score_market(cid: str = "CID_HI") -> MarketInfo:
    return _make_market(cid=cid, volume=200_000, liquidity=100_000, end_days=30)


def _low_score_market(cid: str = "CID_LO") -> MarketInfo:
    return _make_market(cid=cid, volume=10, liquidity=10, end_days=365)


# ── Bootstrap tests ──────────────────────────────────────────────────────

class TestInitialDiscovery:
    @pytest.mark.asyncio
    async def test_high_score_goes_to_active(self):
        lm = MarketLifecycleManager()
        high = _high_score_market()
        with patch("src.data.market_lifecycle.fetch_active_markets", new_callable=AsyncMock) as mock:
            mock.return_value = [high]
            result = await lm.initial_discovery()

        assert len(result) == 1
        assert high.condition_id in lm.active
        assert high.condition_id not in lm.observing

    @pytest.mark.asyncio
    async def test_low_score_goes_to_observing(self):
        lm = MarketLifecycleManager()
        low = _low_score_market()
        with patch("src.data.market_lifecycle.fetch_active_markets", new_callable=AsyncMock) as mock:
            mock.return_value = [low]
            result = await lm.initial_discovery()

        assert len(result) == 0
        assert low.condition_id in lm.observing

    @pytest.mark.asyncio
    async def test_empty_discovery(self):
        lm = MarketLifecycleManager()
        with patch("src.data.market_lifecycle.fetch_active_markets", new_callable=AsyncMock) as mock:
            mock.return_value = []
            result = await lm.initial_discovery()

        assert result == []


# ── Query tests ───────────────────────────────────────────────────────────

class TestQueries:
    def test_is_tradeable(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        score = compute_score(daily_volume_usd=200_000, liquidity_usd=100_000)
        lm.active["CID_HI"] = ActiveMarket(info=m, score=score)

        assert lm.is_tradeable("CID_HI")
        assert not lm.is_tradeable("CID_MISSING")

    def test_is_tracked(self):
        lm = MarketLifecycleManager()
        m = _make_market("CID_OBS")
        lm.observing["CID_OBS"] = ObservingMarket(info=m)

        assert lm.is_tracked("CID_OBS")
        assert not lm.is_tracked("CID_MISSING")

    def test_get_active_markets(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        score = compute_score(daily_volume_usd=200_000, liquidity_usd=100_000)
        lm.active["CID_HI"] = ActiveMarket(info=m, score=score)

        result = lm.get_active_markets()
        assert len(result) == 1
        assert result[0].condition_id == "CID_HI"


# ── Signal cooldown ──────────────────────────────────────────────────────

class TestCooldown:
    def test_fresh_market_is_cooled(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        lm.active["CID_HI"] = ActiveMarket(info=m, last_signal_time=0)
        assert lm.is_cooled_down("CID_HI")

    def test_recent_signal_not_cooled(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        lm.active["CID_HI"] = ActiveMarket(info=m, last_signal_time=time.time())
        assert not lm.is_cooled_down("CID_HI")

    def test_record_signal_updates_time(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        lm.active["CID_HI"] = ActiveMarket(info=m, last_signal_time=0)
        lm.record_signal("CID_HI")
        assert lm.active["CID_HI"].last_signal_time > 0

    def test_non_active_not_cooled(self):
        lm = MarketLifecycleManager()
        assert not lm.is_cooled_down("CID_MISSING")


# ── Score queries ─────────────────────────────────────────────────────────

class TestScoreQueries:
    def test_get_score_active(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        bd = compute_score(daily_volume_usd=200_000, liquidity_usd=100_000)
        lm.active["CID_HI"] = ActiveMarket(info=m, score=bd)
        assert lm.get_score("CID_HI") == bd.total

    def test_get_score_missing(self):
        lm = MarketLifecycleManager()
        assert lm.get_score("CID_MISSING") == 0.0

    def test_get_breakdown(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        bd = compute_score(daily_volume_usd=200_000, liquidity_usd=100_000)
        lm.active["CID_HI"] = ActiveMarket(info=m, score=bd)
        result = lm.get_score_breakdown("CID_HI")
        assert result is not None
        assert result.total == bd.total


# ── Draining & eviction ──────────────────────────────────────────────────

class TestDraining:
    def test_drain_active_market(self):
        lm = MarketLifecycleManager()
        m = _high_score_market()
        lm.active["CID_HI"] = ActiveMarket(info=m)
        lm.drain_market("CID_HI", reason="test")

        assert "CID_HI" not in lm.active
        assert "CID_HI" in lm.draining
        assert lm.draining["CID_HI"].reason == "test"

    def test_evict_removes_from_all_tiers(self):
        lm = MarketLifecycleManager()
        m = _make_market("CID_D")
        lm.draining["CID_D"] = DrainingMarket(info=m, reason="resolved")
        lm._evict("CID_D")

        assert "CID_D" not in lm.draining
        assert not lm.is_tracked("CID_D")


# ── Refresh ───────────────────────────────────────────────────────────────

class TestRefresh:
    @pytest.mark.asyncio
    async def test_resolution_detection(self):
        """Markets absent from fresh discovery are detected as resolved."""
        lm = MarketLifecycleManager()
        m = _high_score_market("CID_OLD")
        lm.active["CID_OLD"] = ActiveMarket(info=m)

        # Fresh discovery returns empty → CID_OLD is resolved
        with patch("src.data.market_lifecycle.fetch_active_markets", new_callable=AsyncMock) as mock:
            mock.return_value = []
            _, evicted = await lm.refresh()

        assert "CID_OLD" in evicted

    @pytest.mark.asyncio
    async def test_resolved_with_position_goes_to_draining(self):
        lm = MarketLifecycleManager()
        m = _high_score_market("CID_POS")
        lm.active["CID_POS"] = ActiveMarket(info=m)

        with patch("src.data.market_lifecycle.fetch_active_markets", new_callable=AsyncMock) as mock:
            mock.return_value = []
            _, evicted = await lm.refresh(
                open_position_markets={"CID_POS"}
            )

        assert "CID_POS" not in evicted
        assert "CID_POS" in lm.draining

    @pytest.mark.asyncio
    async def test_new_markets_go_to_observing(self):
        lm = MarketLifecycleManager()
        new_m = _make_market("CID_NEW")

        with patch("src.data.market_lifecycle.fetch_active_markets", new_callable=AsyncMock) as mock:
            mock.return_value = [new_m]
            newly_added, _ = await lm.refresh()

        assert len(newly_added) == 1
        assert "CID_NEW" in lm.observing


# ── Promotion ─────────────────────────────────────────────────────────────

class TestPromotion:
    def test_promote_after_observation_period(self):
        lm = MarketLifecycleManager()
        m = _high_score_market("CID_PROM")
        # Put in observing with entered_at far in the past
        lm.observing["CID_PROM"] = ObservingMarket(
            info=m,
            entered_at=time.time() - 9999,
        )
        lm._promote_ready()
        assert "CID_PROM" in lm.active
        assert "CID_PROM" not in lm.observing

    def test_no_promote_if_too_recent(self):
        lm = MarketLifecycleManager()
        m = _high_score_market("CID_RECENT")
        lm.observing["CID_RECENT"] = ObservingMarket(
            info=m,
            entered_at=time.time(),  # just now
        )
        lm._promote_ready()
        assert "CID_RECENT" in lm.observing
        assert "CID_RECENT" not in lm.active

    def test_no_promote_if_low_score(self):
        lm = MarketLifecycleManager()
        m = _low_score_market("CID_LOW")
        lm.observing["CID_LOW"] = ObservingMarket(
            info=m,
            entered_at=time.time() - 9999,
        )
        lm._promote_ready()
        assert "CID_LOW" in lm.observing
