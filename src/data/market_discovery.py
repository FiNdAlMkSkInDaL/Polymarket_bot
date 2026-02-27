"""
Market discovery — enumerate Polymarket CLOB markets, filter for
high-volume binary events, and return tradeable asset ids.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# Module-level rate limiter (shared across all discovery functions)
_rate_sem: asyncio.Semaphore | None = None


def _get_rate_sem() -> asyncio.Semaphore:
    global _rate_sem
    if _rate_sem is None:
        _rate_sem = asyncio.Semaphore(settings.strategy.api_rate_limit_per_sec)
    return _rate_sem


async def _rate_limited_get(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> httpx.Response:
    """GET with semaphore-based rate limiting."""
    sem = _get_rate_sem()
    async with sem:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        await asyncio.sleep(1.0 / max(settings.strategy.api_rate_limit_per_sec, 1))
        return resp


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

    # Extended fields for scoring & lifecycle
    event_id: str = ""
    liquidity_usd: float = 0.0
    score: float = 0.0
    accepting_orders: bool = True


async def fetch_active_markets(
    min_volume: float | None = None,
    min_days_to_resolution: int | None = None,
    limit: int = 200,
) -> list[MarketInfo]:
    """Discover tradeable markets using a tiered strategy.

    Discovery order:
      1. CLOB ``/markets`` (fastest, but returns stale data sometimes).
      2. Gamma ``/events`` per tag (``politics``, ``crypto``, …) — best for
         finding high-quality standalone binary markets.
      3. Gamma ``/markets`` — last-resort catch-all.

    Filters applied by ``_parse_market``:
      - Active + open for trading + ``acceptingOrders``.
      - Not a ``negRisk`` bracket market.
      - Binary outcomes (exactly 2 tokens).
      - Daily volume >= *min_volume*.
      - Resolution date >= *min_days_to_resolution* from now.
    """
    min_volume = min_volume or settings.strategy.min_daily_volume_usd
    min_days_to_resolution = min_days_to_resolution or settings.strategy.min_days_to_resolution

    # Tier 1 — CLOB
    markets = await _fetch_clob_markets(min_volume, min_days_to_resolution, limit)
    if markets:
        log.info("markets_fetched", count=len(markets), source="clob")
        return markets[:limit]

    # Tier 2 — Gamma /events (tag-based, best quality)
    log.info("clob_returned_zero_trying_gamma_events")
    markets = await _fetch_gamma_events(
        min_volume, min_days_to_resolution, limit,
    )
    if markets:
        log.info("markets_fetched", count=len(markets), source="gamma_events")
        return markets[:limit]

    # Tier 3 — Gamma /markets (catch-all fallback)
    log.info("gamma_events_returned_zero_trying_gamma_markets")
    markets = await _fetch_gamma_markets(min_volume, min_days_to_resolution, limit)
    log.info("markets_fetched", count=len(markets), source="gamma_markets")
    return markets[:limit]


async def _fetch_clob_markets(
    min_volume: float,
    min_days_to_resolution: int,
    limit: int,
) -> list[MarketInfo]:
    """Paginate through the CLOB /markets endpoint."""
    url = f"{settings.clob_http_url}/markets"
    markets: list[MarketInfo] = []
    rejections: list[dict] = []

    _END_CURSORS = {"", "LTE=", "LTE%3D"}
    _MAX_PAGES = 20  # cap pagination to avoid scanning 1000+ pages

    async with httpx.AsyncClient(timeout=30) as client:
        next_cursor = ""
        page = 0
        while len(markets) < limit and page < _MAX_PAGES:
            page += 1
            # Filter server-side so we never page through years of closed markets
            params: dict = {"limit": 100, "active": "true", "closed": "false"}
            if next_cursor:
                params["next_cursor"] = next_cursor

            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                log.warning(
                    "market_fetch_http_error",
                    status=exc.response.status_code,
                    cursor=next_cursor,
                    markets_so_far=len(markets),
                )
                break

            payload = resp.json()
            items = payload if isinstance(payload, list) else payload.get("data", [])
            if not items:
                break

            for m in items:
                info, reason = _parse_market(m, min_volume, min_days_to_resolution)
                if info:
                    markets.append(info)
                elif reason and len(rejections) < 10:
                    rejections.append(reason)

            next_cursor = payload.get("next_cursor", "") if isinstance(payload, dict) else ""
            if next_cursor in _END_CURSORS:
                break

        if page >= _MAX_PAGES and len(markets) < limit:
            log.warning(
                "clob_pagination_capped",
                pages=page,
                markets_found=len(markets),
                limit=limit,
            )

    if not markets:
        _log_rejections(rejections, source="clob")
    return markets


async def _fetch_gamma_markets(
    min_volume: float,
    min_days_to_resolution: int,
    limit: int,
) -> list[MarketInfo]:
    """Fetch markets from the Gamma API as a fallback (with pagination)."""
    gamma_url = "https://gamma-api.polymarket.com/markets"
    markets: list[MarketInfo] = []
    rejections: list[dict] = []

    _PAGE_SIZE = 100
    _MAX_PAGES = 10

    async with httpx.AsyncClient(timeout=30) as client:
        offset = 0
        for _page in range(_MAX_PAGES):
            if len(markets) >= limit:
                break
            try:
                resp = await _rate_limited_get(
                    client,
                    gamma_url,
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": _PAGE_SIZE,
                        "offset": offset,
                    },
                )
            except httpx.HTTPStatusError as exc:
                log.warning("gamma_fetch_http_error", status=exc.response.status_code)
                break

            items = resp.json()
            if not isinstance(items, list):
                items = items.get("data", []) if isinstance(items, dict) else []

            if not items:
                break

            for m in items:
                normalised = _normalize_gamma_market(m)
                info, reason = _parse_market(normalised, min_volume, min_days_to_resolution)
                if info:
                    markets.append(info)
                elif reason and len(rejections) < 10:
                    rejections.append(reason)

            offset += _PAGE_SIZE
            if len(items) < _PAGE_SIZE:
                break

    if not markets:
        _log_rejections(rejections, source="gamma_markets")
    return markets[:limit]


async def _fetch_gamma_events(
    min_volume: float,
    min_days_to_resolution: int,
    limit: int,
) -> list[MarketInfo]:
    """Fetch markets via the Gamma /events endpoint.

    If ``discovery_tags`` is empty, fetches ALL categories (no tag filter).
    Supports pagination via ``offset`` to walk the full event universe.

    Optionally enforces one-market-per-event (the highest-volume market)
    to prevent correlated sub-questions from flooding the watchlist.
    """
    tags_raw = settings.strategy.discovery_tags.strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    one_per_event = settings.strategy.one_market_per_event
    gamma_url = "https://gamma-api.polymarket.com/events"
    markets: list[MarketInfo] = []
    rejections: list[dict] = []
    seen_events: set[str] = set()

    _PAGE_SIZE = 100
    _MAX_PAGES = 10  # safety cap per tag (or total if no tags)

    async with httpx.AsyncClient(timeout=30) as client:
        tag_list = tags if tags else [""]  # empty string = no tag filter
        for tag in tag_list:
            if len(markets) >= limit:
                break

            offset = 0
            for _page in range(_MAX_PAGES):
                if len(markets) >= limit:
                    break

                params: dict = {
                    "active": "true",
                    "closed": "false",
                    "limit": _PAGE_SIZE,
                    "offset": offset,
                }
                if tag:
                    params["tag"] = tag

                try:
                    resp = await _rate_limited_get(client, gamma_url, params)
                except httpx.HTTPStatusError as exc:
                    log.warning(
                        "gamma_events_http_error",
                        status=exc.response.status_code,
                        tag=tag or "all",
                    )
                    break

                events = resp.json()
                if not isinstance(events, list):
                    events = events.get("data", []) if isinstance(events, dict) else []

                if not events:
                    break  # no more pages

                for event in events:
                    event_id = str(event.get("id", ""))
                    if one_per_event and event_id in seen_events:
                        continue

                    sub_markets = event.get("markets", [])
                    if not isinstance(sub_markets, list):
                        continue

                    # Normalise, parse, and rank by 24h volume
                    candidates: list[tuple[MarketInfo, float]] = []
                    for m in sub_markets:
                        normalised = _normalize_gamma_market(m)
                        # Inject event_id
                        normalised["event_id"] = event_id
                        info, reason = _parse_market(
                            normalised, min_volume, min_days_to_resolution,
                        )
                        if info:
                            candidates.append((info, info.daily_volume_usd))
                        elif reason and len(rejections) < 10:
                            rejections.append(reason)

                    if not candidates:
                        continue

                    if one_per_event:
                        best = max(candidates, key=lambda c: c[1])
                        markets.append(best[0])
                        seen_events.add(event_id)
                    else:
                        markets.extend(c[0] for c in candidates)

                offset += _PAGE_SIZE
                if len(events) < _PAGE_SIZE:
                    break  # last page

            log.info(
                "gamma_events_tag_done",
                tag=tag or "all",
                markets_so_far=len(markets),
            )

    if not markets:
        _log_rejections(rejections, source="gamma_events")
    return markets[:limit]


def _normalize_gamma_market(m: dict) -> dict:
    """Translate a Gamma API market dict into the CLOB schema used by _parse_market.

    Key differences:
      Gamma field          CLOB equivalent
      ────────────         ───────────────
      clobTokenIds         tokens[*].token_id
      outcomes             tokens[*].outcome
      volume24hr           volume_num_24hr  (lifetime ``volume`` is NOT used)
      endDate              end_date_iso
      conditionId          condition_id
      negRisk              passed through for filtering
    """
    clob_ids_raw = m.get("clobTokenIds") or []
    outcomes_raw = m.get("outcomes") or []

    # Gamma sometimes returns these as JSON-encoded strings instead of lists
    if isinstance(clob_ids_raw, str):
        try:
            clob_ids_raw = json.loads(clob_ids_raw)
        except (json.JSONDecodeError, TypeError):
            clob_ids_raw = []
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except (json.JSONDecodeError, TypeError):
            outcomes_raw = []

    clob_ids: list[str] = clob_ids_raw if isinstance(clob_ids_raw, list) else []
    outcomes: list[str] = outcomes_raw if isinstance(outcomes_raw, list) else []
    tokens = [
        {"token_id": tid, "outcome": outcome}
        for tid, outcome in zip(clob_ids, outcomes)
    ]
    # ── Volume: use 24-hour fields, NOT lifetime cumulative ──
    raw_volume = (
        m.get("volume24hr")
        or m.get("volume24hrClob")
        or m.get("volumeNum24hr")
        or 0
    )
    try:
        volume_float = float(raw_volume)
    except (TypeError, ValueError):
        volume_float = 0.0

    # ── Liquidity ──
    raw_liq = m.get("liquidityClob") or m.get("liquidity") or 0
    try:
        liquidity_float = float(raw_liq)
    except (TypeError, ValueError):
        liquidity_float = 0.0

    return {
        "condition_id": m.get("conditionId", ""),
        "question": m.get("question", ""),
        "active": m.get("active", False),
        "closed": m.get("closed", True),
        "tokens": tokens,
        "volume_num_24hr": volume_float,
        "end_date_iso": m.get("endDate", ""),
        # Pass through Gamma-specific fields for _parse_market filters
        "negRisk": m.get("negRisk", False),
        "acceptingOrders": m.get("acceptingOrders", True),
        "enableOrderBook": m.get("enableOrderBook", True),
        "groupItemTitle": m.get("groupItemTitle", ""),
        "liquidity": liquidity_float,
        "event_id": str(m.get("eventId") or m.get("event_id", "")),
    }


def _log_rejections(rejections: list[dict], *, source: str) -> None:
    """Emit detailed debug logs for the first N rejected markets."""
    if not rejections:
        log.debug("no_rejection_details_captured", source=source)
        return
    log.warning(
        "no_eligible_markets_debug",
        source=source,
        rejected_sample_count=len(rejections),
    )
    for r in rejections:
        log.warning("market_rejected", **r)


def _parse_market(
    raw: dict,
    min_volume: float,
    min_days_to_resolution: int,
) -> tuple[MarketInfo | None, dict | None]:
    """Validate a single market dict against our filters.

    Returns ``(MarketInfo, None)`` on success or ``(None, rejection_dict)``
    on failure so the caller can aggregate debug info.
    """
    question = raw.get("question", "<unknown>")[:80]

    try:
        # Must be active
        if not raw.get("active", False):
            return None, {"reason": "not_active", "question": question}
        if raw.get("closed", True):
            return None, {"reason": "closed", "question": question}

        # Must be accepting orders (catches frozen / about-to-resolve markets)
        if not raw.get("acceptingOrders", True):
            return None, {"reason": "not_accepting_orders", "question": question}

        # Reject negRisk bracket markets — correlated pricing breaks
        # the mean-reversion assumption
        if settings.strategy.reject_neg_risk and raw.get("negRisk", False):
            return None, {
                "reason": "neg_risk_group",
                "group_title": raw.get("groupItemTitle", "")[:40],
                "question": question,
            }

        # Token count check — must be exactly 2 (true binary)
        tokens = raw.get("tokens", [])
        if len(tokens) < 2:
            return None, {
                "reason": "too_few_tokens",
                "token_count": len(tokens),
                "question": question,
            }
        if len(tokens) > 2:
            return None, {
                "reason": "too_many_tokens",
                "token_count": len(tokens),
                "question": question,
            }

        # Volume gate (24-hour)
        volume = float(raw.get("volume_num_24hr") or raw.get("volume", 0))
        if volume < min_volume:
            return None, {
                "reason": "volume_too_low",
                "value": round(volume, 2),
                "threshold": min_volume,
                "question": question,
            }

        # Liquidity gate (optional)
        min_liq = settings.strategy.min_liquidity_usd
        if min_liq > 0:
            liq = float(raw.get("liquidity", 0))
            if liq < min_liq:
                return None, {
                    "reason": "liquidity_too_low",
                    "value": round(liq, 2),
                    "threshold": min_liq,
                    "question": question,
                }

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
                return None, {
                    "reason": "resolves_too_soon",
                    "days_left": days_left,
                    "threshold": min_days_to_resolution,
                    "question": question,
                }

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
            event_id=str(raw.get("event_id", "")),
            liquidity_usd=float(raw.get("liquidity", 0)),
            accepting_orders=bool(raw.get("acceptingOrders", True)),
        ), None

    except (KeyError, TypeError, ValueError) as exc:
        log.debug("market_parse_skip", error=str(exc))
        return None, {"reason": "parse_error", "error": str(exc), "question": question}
