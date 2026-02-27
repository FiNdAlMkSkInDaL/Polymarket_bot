"""
Market discovery — enumerate Polymarket CLOB markets, filter for
high-volume geopolitical binary events, and return tradeable asset ids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class MarketInfo:
    """Lightweight descriptor for a tradeable market."""

    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    daily_volume_usd: float
    end_date: datetime | None
    active: bool


async def fetch_active_markets(
    min_volume: float | None = None,
    min_days_to_resolution: int | None = None,
    limit: int = 50,
) -> list[MarketInfo]:
    """Query the Polymarket CLOB REST API for eligible markets.

    Filters:
      - Binary outcomes only (exactly 2 tokens).
      - Active / open for trading.
      - Daily volume >= *min_volume*.
      - Resolution date >= *min_days_to_resolution* from now.
    """
    min_volume = min_volume or settings.strategy.min_daily_volume_usd
    min_days_to_resolution = min_days_to_resolution or settings.strategy.min_days_to_resolution

    url = f"{settings.clob_http_url}/markets"
    markets: list[MarketInfo] = []

    async with httpx.AsyncClient(timeout=30) as client:
        # The CLOB API paginates; fetch up to `limit` valid markets.
        next_cursor = ""
        while len(markets) < limit:
            params: dict = {"limit": 100}
            if next_cursor:
                params["next_cursor"] = next_cursor

            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

            items = payload if isinstance(payload, list) else payload.get("data", [])
            if not items:
                break

            for m in items:
                info = _parse_market(m, min_volume, min_days_to_resolution)
                if info:
                    markets.append(info)

            next_cursor = payload.get("next_cursor", "") if isinstance(payload, dict) else ""
            if not next_cursor:
                break

    log.info("markets_fetched", count=len(markets))
    return markets[:limit]


def _parse_market(
    raw: dict,
    min_volume: float,
    min_days_to_resolution: int,
) -> MarketInfo | None:
    """Validate a single market dict against our filters."""
    try:
        # Must be active
        if not raw.get("active", False):
            return None
        if raw.get("closed", True):
            return None

        # Must be binary (2 tokens)
        tokens = raw.get("tokens", [])
        if len(tokens) != 2:
            return None

        # Volume gate
        volume = float(raw.get("volume_num_24hr") or raw.get("volume", 0))
        if volume < min_volume:
            return None

        # Resolution date gate
        end_raw = raw.get("end_date_iso") or raw.get("end_date", "")
        end_date = None
        if end_raw:
            try:
                end_date = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
            except ValueError:
                end_date = None

        if end_date:
            days_left = (end_date - datetime.now(timezone.utc)).days
            if days_left < min_days_to_resolution:
                return None

        # Identify YES / NO tokens
        yes_token = None
        no_token = None
        for tok in tokens:
            outcome = (tok.get("outcome") or "").upper()
            if outcome == "YES":
                yes_token = tok
            elif outcome == "NO":
                no_token = tok

        if not yes_token or not no_token:
            # Fall back to positional convention  (index 0=YES, 1=NO)
            yes_token = tokens[0]
            no_token = tokens[1]

        return MarketInfo(
            condition_id=raw.get("condition_id", ""),
            question=raw.get("question", ""),
            yes_token_id=yes_token.get("token_id", ""),
            no_token_id=no_token.get("token_id", ""),
            daily_volume_usd=volume,
            end_date=end_date,
            active=True,
        )

    except (KeyError, TypeError, ValueError) as exc:
        log.debug("market_parse_skip", error=str(exc))
        return None
