from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import settings
from src.data import market_discovery


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _event_payload() -> list[dict]:
    return [
        {
            "id": "evt-1",
            "markets": [
                {
                    "conditionId": "cid-low",
                    "question": "Low volume leg",
                    "active": True,
                    "closed": False,
                    "acceptingOrders": True,
                    "negRisk": True,
                    "liquidity": 1000,
                    "volume24hr": 2000,
                    "endDate": "2099-12-31T00:00:00Z",
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["yes-low", "no-low"]',
                },
                {
                    "conditionId": "cid-high",
                    "question": "High volume leg",
                    "active": True,
                    "closed": False,
                    "acceptingOrders": True,
                    "negRisk": True,
                    "liquidity": 1000,
                    "volume24hr": 3000,
                    "endDate": "2099-12-31T00:00:00Z",
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["yes-high", "no-high"]',
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_fetch_gamma_events_returns_all_event_legs_when_si9_enabled() -> None:
    old_one_per_event = settings.strategy.one_market_per_event
    old_si9_enabled = settings.strategy.si9_arb_enabled

    object.__setattr__(settings.strategy, "one_market_per_event", True)
    object.__setattr__(settings.strategy, "si9_arb_enabled", True)

    try:
        with patch("src.data.market_discovery.httpx.AsyncClient"), patch(
            "src.data.market_discovery._rate_limited_get",
            new=AsyncMock(return_value=_DummyResponse(_event_payload())),
        ):
            markets = await market_discovery._fetch_gamma_events(
                min_volume=1000,
                min_days_to_resolution=1,
                limit=10,
            )
    finally:
        object.__setattr__(settings.strategy, "one_market_per_event", old_one_per_event)
        object.__setattr__(settings.strategy, "si9_arb_enabled", old_si9_enabled)

    assert [market.condition_id for market in markets] == ["cid-low", "cid-high"]
    assert all(market.event_id == "evt-1" for market in markets)
    assert all(market.neg_risk for market in markets)


@pytest.mark.asyncio
async def test_fetch_gamma_events_keeps_best_leg_only_when_si9_disabled() -> None:
    old_one_per_event = settings.strategy.one_market_per_event
    old_si9_enabled = settings.strategy.si9_arb_enabled

    object.__setattr__(settings.strategy, "one_market_per_event", True)
    object.__setattr__(settings.strategy, "si9_arb_enabled", False)

    try:
        with patch("src.data.market_discovery.httpx.AsyncClient"), patch(
            "src.data.market_discovery._rate_limited_get",
            new=AsyncMock(return_value=_DummyResponse(_event_payload())),
        ):
            markets = await market_discovery._fetch_gamma_events(
                min_volume=1000,
                min_days_to_resolution=1,
                limit=10,
            )
    finally:
        object.__setattr__(settings.strategy, "one_market_per_event", old_one_per_event)
        object.__setattr__(settings.strategy, "si9_arb_enabled", old_si9_enabled)

    assert [market.condition_id for market in markets] == ["cid-high"]