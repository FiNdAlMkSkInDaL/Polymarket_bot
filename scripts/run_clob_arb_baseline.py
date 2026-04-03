from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.config import EXCHANGE_MIN_SHARES, EXCHANGE_MIN_USD
from src.data.orderbook import OrderbookTracker


DEFAULT_START_DATE = "2026-03-15"
DEFAULT_END_DATE = "2026-03-19"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "logs" / "local_snapshot" / "l2_data"
DEFAULT_METADATA_CACHE = PROJECT_ROOT / "artifacts" / "clob_arb_baseline_metadata.json"
DEFAULT_PROGRESS_EVERY_RECORDS = 200_000
DEFAULT_LATENCY_SENSITIVITY_MS = 1_000
DEFAULT_WARMUP_DAYS = 2
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONCURRENCY = 12
SCAN_INTERVAL_MS = 300_000
SIZE_QUANT = Decimal("0.000001")
USD_CENT = Decimal("0.01")
SAFE_NOTIONAL_QUANT = Decimal("1")
PRICE_PRECISION = Decimal("0.0001")
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
THRESHOLD_PATTERN = re.compile(
    r"(at least|at most|or more|or less|more than|less than|above|below|under|over|>=|<=|>|<|\b\d+(?:\.\d+)?\+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DayInputs:
    day: str
    market_ids: set[str]
    token_paths: dict[str, Path]


@dataclass(frozen=True, slots=True)
class ResolvedMarket:
    event_id: str
    market_id: str
    condition_id: str
    question: str
    outcome_label: str
    yes_token_id: str
    no_token_id: str
    accepting_orders: bool
    enable_order_book: bool
    neg_risk: bool
    active: bool
    closed: bool
    created_at: datetime | None
    accepting_orders_at: datetime | None
    end_date: datetime | None


@dataclass(frozen=True, slots=True)
class ResolvedEvent:
    event_id: str
    title: str
    slug: str
    enable_neg_risk: bool
    markets: tuple[ResolvedMarket, ...]


@dataclass(frozen=True, slots=True)
class HistoricalLeg:
    condition_id: str
    market_id: str
    question: str
    outcome_label: str
    yes_token_id: str
    no_token_id: str
    best_bid: Decimal
    best_bid_size_shares: Decimal
    best_bid_notional_usd: Decimal
    best_ask: Decimal
    best_ask_size_shares: Decimal
    best_ask_notional_usd: Decimal


@dataclass(frozen=True, slots=True)
class HistoricalStripTarget:
    event_id: str
    event_title: str
    event_slug: str
    recommended_action: str
    execution_price_sum: Decimal
    min_leg_depth_usd_observed: Decimal
    strip_max_size_shares_at_bbo: Decimal
    legs: tuple[HistoricalLeg, ...]


@dataclass(frozen=True, slots=True)
class HistoricalOrderPlan:
    event_id: str
    event_title: str
    action: str
    side: str
    strip_notional_cap_usd: Decimal
    strip_shares: Decimal
    total_notional_usd: Decimal
    total_execution_price: Decimal
    recommended_action: str
    legs: tuple[HistoricalLeg, ...]


@dataclass(frozen=True, slots=True)
class PendingLatencyAttempt:
    execute_at_ms: int
    plan: HistoricalOrderPlan


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    status: str
    filled_legs: int
    total_legs: int


@dataclass(frozen=True, slots=True)
class BaselineDayResult:
    day: str
    trigger_count_per_day: int
    total_simulated_fills: int
    pnl_usd: Decimal
    strict_partial_fills: int
    strict_dead_strips: int
    latency_full_fills: int
    latency_legging_events: int
    latency_dead_strips: int
    launcher_rejections: int
    complete_event_groups: int
    candidate_event_groups: int
    scan_rejections: dict[str, int]
    coverage_rejections: dict[str, int]
    launcher_rejection_reasons: dict[str, int]


@dataclass(order=True)
class QueuedRecord:
    timestamp_ms: int
    token_id: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    iterator: Iterator[tuple[int, dict[str, Any]]] = field(compare=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict historical Sword baseline on archived local L2 data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Replay source root containing the archived raw tick folder.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--metadata-cache", type=Path, default=DEFAULT_METADATA_CACHE, help="Disk cache for Gamma market and event lookups.")
    parser.add_argument("--fee-buffer", type=Decimal, default=Decimal("0.02"), help="Sword fee buffer applied around fair value.")
    parser.add_argument("--min-leg-depth-usd", type=Decimal, default=Decimal("10"), help="Minimum displayed depth per leg required by the live scanner.")
    parser.add_argument(
        "--latency-sensitivity-ms",
        type=int,
        default=DEFAULT_LATENCY_SENSITIVITY_MS,
        help="Frozen-plan delay used for the legging sensitivity note. Set to 0 to disable the delayed recheck.",
    )
    parser.add_argument(
        "--progress-every-records",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_RECORDS,
        help="Heartbeat interval for long per-day book replay loops.",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=DEFAULT_WARMUP_DAYS,
        help="Calendar days of archive warmup to replay before each scored day so unchanged books can carry forward.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout used for Gamma metadata calls.")
    parser.add_argument("--http-concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent Gamma metadata requests.")
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _parse_datetime(value: Any) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _timestamp_ms_from_record(raw: dict[str, Any], payload: dict[str, Any]) -> int:
    for value in (payload.get("timestamp"), raw.get("local_ts")):
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 1e12:
            return int(numeric)
        return int(numeric * 1000)
    return 0


def _round_decimal(value: Decimal, quantum: Decimal = PRICE_PRECISION) -> Decimal:
    return value.quantize(quantum)


def _quantize_shares(value: Decimal) -> Decimal:
    return value.quantize(SIZE_QUANT, rounding=ROUND_DOWN)


def _round_down_whole_dollars(value: Decimal) -> Decimal:
    if value <= Decimal("0"):
        return Decimal("0")
    return value.quantize(SAFE_NOTIONAL_QUANT, rounding=ROUND_DOWN)


def _usd(value: Decimal) -> Decimal:
    return value.quantize(USD_CENT, rounding=ROUND_DOWN)


def _iter_dates(start_date: str, end_date: str) -> list[str]:
    current = date.fromisoformat(start_date)
    final = date.fromisoformat(end_date)
    if final < current:
        raise ValueError("end_date must be on or after start_date")
    days: list[str] = []
    while current <= final:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _default_markdown_output_path(start_date: str, end_date: str) -> Path:
    return PROJECT_ROOT / f"clob_arb_baseline_{start_date}_{end_date}.md"


def _default_json_output_path(start_date: str, end_date: str) -> Path:
    return PROJECT_ROOT / "artifacts" / f"clob_arb_baseline_{start_date}_{end_date}.json"


def _raw_ticks_root(input_dir: Path) -> Path:
    return input_dir / "data" / "raw_ticks"


def _load_day_inputs(raw_ticks_root: Path, day: str) -> DayInputs:
    day_dir = raw_ticks_root / day
    if not day_dir.exists():
        raise FileNotFoundError(f"Missing raw tick directory for {day}: {day_dir}")
    market_ids: set[str] = set()
    token_paths: dict[str, Path] = {}
    for path in sorted(day_dir.glob("*.jsonl")):
        stem = path.stem.strip()
        if not stem:
            continue
        if stem.startswith("0x"):
            market_ids.add(stem.lower())
            continue
        if stem.isdigit():
            token_paths[stem] = path
    return DayInputs(day=day, market_ids=market_ids, token_paths=token_paths)


def _date_from_day(day: str) -> date:
    return date.fromisoformat(day)


def _iter_calendar_dates(start_day: str, end_day: str) -> list[str]:
    return _iter_dates(start_day, end_day)


def _window_days_for_target(
    *,
    target_day: str,
    available_days: dict[str, DayInputs],
    warmup_days: int,
) -> list[DayInputs]:
    target_date = _date_from_day(target_day)
    earliest_date = target_date - timedelta(days=max(0, warmup_days))
    return [
        available_days[day]
        for day in sorted(available_days)
        if earliest_date <= _date_from_day(day) <= target_date
    ]


def _market_event_id(row: dict[str, Any]) -> str:
    events = row.get("events") or []
    if isinstance(events, list) and events:
        event_id = _clean_text(events[0].get("id"))
        if event_id:
            return event_id
    return _clean_text(row.get("eventId") or row.get("event_id"))


def _market_from_payload(event_id: str, payload: dict[str, Any]) -> ResolvedMarket | None:
    token_ids = _parse_listish(payload.get("clobTokenIds"))
    if len(token_ids) != 2:
        return None
    condition_id = _clean_text(payload.get("conditionId") or payload.get("condition_id")).lower()
    market_id = _clean_text(payload.get("id")).lower()
    if not condition_id or not market_id:
        return None
    return ResolvedMarket(
        event_id=event_id,
        market_id=market_id,
        condition_id=condition_id,
        question=_clean_text(payload.get("question")),
        outcome_label=_clean_text(payload.get("groupItemTitle") or payload.get("question")),
        yes_token_id=_clean_text(token_ids[0]),
        no_token_id=_clean_text(token_ids[1]),
        accepting_orders=_safe_bool(payload.get("acceptingOrders"), True),
        enable_order_book=_safe_bool(payload.get("enableOrderBook"), True),
        neg_risk=_safe_bool(payload.get("negRisk"), False),
        active=_safe_bool(payload.get("active"), False),
        closed=_safe_bool(payload.get("closed"), False),
        created_at=_parse_datetime(payload.get("createdAt") or payload.get("startDate")),
        accepting_orders_at=_parse_datetime(payload.get("acceptingOrdersTimestamp")),
        end_date=_parse_datetime(payload.get("endDate") or payload.get("endDateIso")),
    )


class GammaMetadataCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.markets_by_token: dict[str, dict[str, Any]] = {}
        self.events_by_id: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        raw_markets = payload.get("markets_by_token") if isinstance(payload, dict) else None
        raw_events = payload.get("events_by_id") if isinstance(payload, dict) else None
        if isinstance(raw_markets, dict):
            self.markets_by_token = {str(key): value for key, value in raw_markets.items() if isinstance(value, dict)}
        if isinstance(raw_events, dict):
            self.events_by_id = {str(key): value for key, value in raw_events.items() if isinstance(value, dict)}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "markets_by_token": self.markets_by_token,
            "events_by_id": self.events_by_id,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class GammaResolver:
    def __init__(self, *, cache: GammaMetadataCache, timeout: float, concurrency: int) -> None:
        self._cache = cache
        self._timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))
        self._market_sem = asyncio.Semaphore(max(1, concurrency))
        self._event_sem = asyncio.Semaphore(max(1, max(4, concurrency // 2)))
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GammaResolver:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        if self._client is None:
            raise RuntimeError("GammaResolver client is not initialized")
        delay_seconds = 1.0
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = await self._client.get(url, params=params)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"transient Gamma error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 4:
                    break
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2.0
        if last_error is None:
            raise RuntimeError("Gamma metadata request failed without an exception")
        raise last_error

    async def market_for_token(self, token_id: str) -> dict[str, Any]:
        cached = self._cache.markets_by_token.get(token_id)
        if cached is not None:
            return cached
        async with self._market_sem:
            cached = self._cache.markets_by_token.get(token_id)
            if cached is not None:
                return cached
            payload = await self._get_json(GAMMA_MARKETS_URL, params={"clob_token_ids": token_id})
            if not isinstance(payload, list) or not payload:
                raise RuntimeError(f"Gamma markets lookup returned no rows for token {token_id}")
            market_row = next(
                (
                    item
                    for item in payload
                    if isinstance(item, dict) and token_id in {_clean_text(value) for value in _parse_listish(item.get("clobTokenIds"))}
                ),
                None,
            )
            if market_row is None:
                raise RuntimeError(f"Gamma markets lookup did not include token {token_id} in clobTokenIds")
            self._cache.markets_by_token[token_id] = market_row
            return market_row

    async def event_for_id(self, event_id: str) -> dict[str, Any]:
        cached = self._cache.events_by_id.get(event_id)
        if cached is not None:
            return cached
        async with self._event_sem:
            cached = self._cache.events_by_id.get(event_id)
            if cached is not None:
                return cached
            payload = await self._get_json(f"{GAMMA_EVENTS_URL}/{event_id}")
            if not isinstance(payload, dict):
                fallback = await self._get_json(GAMMA_EVENTS_URL, params={"id": event_id})
                if isinstance(fallback, list) and fallback:
                    payload = fallback[0]
            if not isinstance(payload, dict):
                raise RuntimeError(f"Gamma events lookup returned an invalid payload for event {event_id}")
            self._cache.events_by_id[event_id] = payload
            return payload


def _materialize_events(event_payloads: Iterable[dict[str, Any]]) -> dict[str, ResolvedEvent]:
    resolved: dict[str, ResolvedEvent] = {}
    for payload in event_payloads:
        event_id = _clean_text(payload.get("id"))
        if not event_id:
            continue
        raw_markets = payload.get("markets") or []
        markets: list[ResolvedMarket] = []
        if isinstance(raw_markets, list):
            for item in raw_markets:
                if not isinstance(item, dict):
                    continue
                market = _market_from_payload(event_id, item)
                if market is not None:
                    markets.append(market)
        if not markets:
            continue
        resolved[event_id] = ResolvedEvent(
            event_id=event_id,
            title=_clean_text(payload.get("title")),
            slug=_clean_text(payload.get("slug")),
            enable_neg_risk=_safe_bool(payload.get("enableNegRisk"), False),
            markets=tuple(sorted(markets, key=lambda item: item.condition_id)),
        )
    return resolved


def _event_market_expected_on_day(market: ResolvedMarket, *, day_start: datetime, day_end: datetime) -> bool:
    if not market.market_id or not market.condition_id or not market.yes_token_id or not market.no_token_id:
        return False
    if not market.enable_order_book:
        return False
    if market.created_at is not None and market.created_at > day_end:
        return False
    if market.accepting_orders_at is not None and market.accepting_orders_at > day_end:
        return False
    if market.end_date is not None and market.end_date < day_start:
        return False
    return True


def _is_cumulative_threshold_group(markets: Iterable[ResolvedMarket]) -> bool:
    threshold_hits = 0
    for market in markets:
        combined = f"{market.question} {market.outcome_label}".strip()
        if combined and THRESHOLD_PATTERN.search(combined):
            threshold_hits += 1
    return threshold_hits >= 2


def _best_level(tracker: OrderbookTracker, side: str) -> tuple[Decimal, Decimal]:
    levels = tracker.levels(side, n=1)
    if not levels:
        return Decimal("0"), Decimal("0")
    return Decimal(str(levels[0].price)), Decimal(str(levels[0].size))


def _snapshot_leg(market: ResolvedMarket, tracker: OrderbookTracker) -> HistoricalLeg | None:
    best_bid, best_bid_size = _best_level(tracker, "bid")
    best_ask, best_ask_size = _best_level(tracker, "ask")
    if best_bid <= Decimal("0") and best_ask <= Decimal("0"):
        return None
    best_bid_notional = _round_decimal(best_bid * best_bid_size)
    best_ask_notional = _round_decimal(best_ask * best_ask_size)
    return HistoricalLeg(
        condition_id=market.condition_id,
        market_id=market.market_id,
        question=market.question,
        outcome_label=market.outcome_label,
        yes_token_id=market.yes_token_id,
        no_token_id=market.no_token_id,
        best_bid=_round_decimal(best_bid),
        best_bid_size_shares=_round_decimal(best_bid_size),
        best_bid_notional_usd=best_bid_notional,
        best_ask=_round_decimal(best_ask),
        best_ask_size_shares=_round_decimal(best_ask_size),
        best_ask_notional_usd=best_ask_notional,
    )


def _evaluate_event_snapshot(
    event: ResolvedEvent,
    event_markets: tuple[ResolvedMarket, ...],
    trackers: dict[str, OrderbookTracker],
    *,
    fee_buffer: Decimal,
    min_leg_depth_usd: Decimal,
) -> tuple[list[HistoricalStripTarget], str | None]:
    if not event.enable_neg_risk:
        return [], "event_not_negrisk"
    if not all(market.neg_risk for market in event_markets):
        return [], "market_not_negrisk"
    if _is_cumulative_threshold_group(event_markets):
        return [], "cumulative_threshold_ladder"

    legs: list[HistoricalLeg] = []
    for market in event_markets:
        tracker = trackers.get(market.yes_token_id)
        if tracker is None:
            return [], "missing_tracker"
        leg = _snapshot_leg(market, tracker)
        if leg is None:
            return [], "missing_bbo"
        legs.append(leg)

    if len(legs) < 3:
        return [], "too_few_live_legs"

    results: list[HistoricalStripTarget] = []
    ask_sum = _round_decimal(sum((leg.best_ask for leg in legs), start=Decimal("0")))
    ask_threshold = Decimal("1") - fee_buffer
    ask_depths = [leg.best_ask_notional_usd for leg in legs]
    if ask_sum < ask_threshold and min(ask_depths) >= min_leg_depth_usd:
        results.append(
            HistoricalStripTarget(
                event_id=event.event_id,
                event_title=event.title,
                event_slug=event.slug,
                recommended_action="BUY_YES_STRIP",
                execution_price_sum=ask_sum,
                min_leg_depth_usd_observed=_round_decimal(min(ask_depths)),
                strip_max_size_shares_at_bbo=_round_decimal(min(leg.best_ask_size_shares for leg in legs)),
                legs=tuple(sorted(legs, key=lambda item: item.best_ask, reverse=True)),
            )
        )

    bid_sum = _round_decimal(sum((leg.best_bid for leg in legs), start=Decimal("0")))
    bid_threshold = Decimal("1") + fee_buffer
    bid_depths = [leg.best_bid_notional_usd for leg in legs]
    if bid_sum > bid_threshold and min(bid_depths) >= min_leg_depth_usd:
        results.append(
            HistoricalStripTarget(
                event_id=event.event_id,
                event_title=event.title,
                event_slug=event.slug,
                recommended_action="SELL_NO_STRIP",
                execution_price_sum=bid_sum,
                min_leg_depth_usd_observed=_round_decimal(min(bid_depths)),
                strip_max_size_shares_at_bbo=_round_decimal(min(leg.best_bid_size_shares for leg in legs)),
                legs=tuple(sorted(legs, key=lambda item: item.best_bid, reverse=True)),
            )
        )

    if results:
        return results, None
    if ask_sum < ask_threshold or bid_sum > bid_threshold:
        return [], "insufficient_leg_depth"
    return [], "no_executable_edge"


def _build_order_plan(target: HistoricalStripTarget) -> HistoricalOrderPlan:
    safe_notional_cap = _round_down_whole_dollars(target.min_leg_depth_usd_observed)
    if safe_notional_cap < Decimal(str(EXCHANGE_MIN_USD)):
        raise RuntimeError(
            f"{target.event_title}: dynamic strip cap ${safe_notional_cap} is below exchange minimum ${EXCHANGE_MIN_USD}"
        )
    if target.execution_price_sum <= Decimal("0"):
        raise RuntimeError(f"{target.event_title}: execution_price_sum must be positive")

    shares_from_cap = safe_notional_cap / target.execution_price_sum
    strip_shares = _quantize_shares(min(shares_from_cap, target.strip_max_size_shares_at_bbo))
    if strip_shares < Decimal(str(EXCHANGE_MIN_SHARES)):
        raise RuntimeError(
            f"{target.event_title}: computed strip size {strip_shares} is below exchange minimum shares {EXCHANGE_MIN_SHARES}"
        )

    total_notional = _usd(strip_shares * target.execution_price_sum)
    if total_notional < Decimal(str(EXCHANGE_MIN_USD)):
        raise RuntimeError(
            f"{target.event_title}: computed strip notional ${total_notional} is below exchange minimum ${EXCHANGE_MIN_USD}"
        )

    if target.recommended_action == "BUY_YES_STRIP":
        action = "BUY"
        side = "YES"
    else:
        action = "SELL"
        side = "NO"

    return HistoricalOrderPlan(
        event_id=target.event_id,
        event_title=target.event_title,
        action=action,
        side=side,
        strip_notional_cap_usd=safe_notional_cap,
        strip_shares=strip_shares,
        total_notional_usd=total_notional,
        total_execution_price=target.execution_price_sum,
        recommended_action=target.recommended_action,
        legs=target.legs,
    )


def _available_size_at_limit(tracker: OrderbookTracker, *, side: str, limit_price: Decimal) -> Decimal:
    available = Decimal("0")
    if side == "BUY":
        for level in tracker.levels("ask", n=10):
            level_price = Decimal(str(level.price))
            if level_price <= limit_price:
                available += Decimal(str(level.size))
    else:
        for level in tracker.levels("bid", n=10):
            level_price = Decimal(str(level.price))
            if level_price >= limit_price:
                available += Decimal(str(level.size))
    return _quantize_shares(available)


def _simulate_execution(plan: HistoricalOrderPlan, trackers: dict[str, OrderbookTracker]) -> ExecutionOutcome:
    filled_legs = 0
    for leg in plan.legs:
        tracker = trackers.get(leg.yes_token_id)
        if tracker is None:
            continue
        limit_price = leg.best_ask if plan.recommended_action == "BUY_YES_STRIP" else leg.best_bid
        side = "BUY" if plan.recommended_action == "BUY_YES_STRIP" else "SELL"
        available = _available_size_at_limit(tracker, side=side, limit_price=limit_price)
        if available >= plan.strip_shares:
            filled_legs += 1
    if filled_legs == len(plan.legs):
        return ExecutionOutcome(status="full_fill", filled_legs=filled_legs, total_legs=len(plan.legs))
    if filled_legs > 0:
        return ExecutionOutcome(status="legging", filled_legs=filled_legs, total_legs=len(plan.legs))
    return ExecutionOutcome(status="dead", filled_legs=0, total_legs=len(plan.legs))


def _gross_pnl(plan: HistoricalOrderPlan) -> Decimal:
    if plan.recommended_action == "BUY_YES_STRIP":
        edge = Decimal("1") - plan.total_execution_price
    else:
        edge = plan.total_execution_price - Decimal("1")
    if edge <= Decimal("0"):
        return Decimal("0.00")
    return _usd(edge * plan.strip_shares)


def _iter_token_file_records(paths: Iterable[Path]) -> Iterator[tuple[int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    continue
                payload = raw.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                timestamp_ms = _timestamp_ms_from_record(raw, payload)
                if timestamp_ms <= 0:
                    continue
                yield timestamp_ms, payload


def _process_payload(tracker: OrderbookTracker, payload: dict[str, Any]) -> None:
    event_type = _clean_text(payload.get("event_type")).lower()
    if event_type == "book":
        tracker.on_book_snapshot(payload)
        return
    if event_type == "price_change":
        tracker.on_price_change(payload)


def _day_window(day: str) -> tuple[datetime, datetime, int, int]:
    day_start = datetime.fromisoformat(day).replace(tzinfo=UTC)
    day_end = day_start + timedelta(days=1) - timedelta(milliseconds=1)
    return day_start, day_end, int(day_start.timestamp() * 1000), int(day_end.timestamp() * 1000)


def _select_complete_events_for_day(
    *,
    current_day: str,
    window_inputs: list[DayInputs],
    resolved_events: dict[str, ResolvedEvent],
    token_market_rows: dict[str, dict[str, Any]],
    observed_yes_token_paths: dict[str, tuple[Path, ...]],
) -> tuple[dict[str, tuple[ResolvedEvent, tuple[ResolvedMarket, ...]]], Counter[str], set[str]]:
    day_start, day_end, _, _ = _day_window(current_day)
    observed_condition_ids = set().union(*(day.market_ids for day in window_inputs)) if window_inputs else set()
    observed_yes_tokens = set(observed_yes_token_paths)
    candidate_event_ids = {
        event_id
        for token_id in observed_yes_tokens
        if (event_id := _market_event_id(token_market_rows.get(token_id, {})))
    }

    coverage_rejections: Counter[str] = Counter()
    complete_events: dict[str, tuple[ResolvedEvent, tuple[ResolvedMarket, ...]]] = {}
    for event_id in sorted(candidate_event_ids):
        event = resolved_events.get(event_id)
        if event is None:
            coverage_rejections["missing_event_metadata"] += 1
            continue
        full_expected_markets = tuple(
            market
            for market in event.markets
            if _event_market_expected_on_day(market, day_start=day_start, day_end=day_end)
        )
        if len(full_expected_markets) < 3:
            coverage_rejections["too_few_expected_outcomes"] += 1
            continue
        if any(market.condition_id not in observed_condition_ids or market.yes_token_id not in observed_yes_tokens for market in full_expected_markets):
            coverage_rejections["missing_full_event_coverage"] += 1
            continue
        complete_events[event_id] = (event, full_expected_markets)
    return complete_events, coverage_rejections, candidate_event_ids


def _push_initial_records(
    relevant_token_paths: dict[str, tuple[Path, ...]],
    trackers: dict[str, OrderbookTracker],
) -> list[QueuedRecord]:
    queue: list[QueuedRecord] = []
    for token_id, paths in relevant_token_paths.items():
        tracker = trackers.get(token_id)
        if tracker is None:
            continue
        iterator = _iter_token_file_records(paths)
        try:
            timestamp_ms, payload = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(queue, QueuedRecord(timestamp_ms=timestamp_ms, token_id=token_id, payload=payload, iterator=iterator))
    return queue


def _advance_trackers_to(
    target_ms: int,
    *,
    queue: list[QueuedRecord],
    trackers: dict[str, OrderbookTracker],
    processed_records: list[int],
    progress_every_records: int,
    day: str,
) -> None:
    while queue and queue[0].timestamp_ms <= target_ms:
        record = heapq.heappop(queue)
        tracker = trackers.get(record.token_id)
        if tracker is not None:
            _process_payload(tracker, record.payload)
        processed_records[0] += 1
        if progress_every_records > 0 and processed_records[0] % progress_every_records == 0:
            print(
                f"baseline_day_progress date={day} processed_records={processed_records[0]} last_timestamp_ms={record.timestamp_ms}",
                flush=True,
            )
        try:
            next_timestamp_ms, next_payload = next(record.iterator)
        except StopIteration:
            continue
        heapq.heappush(
            queue,
            QueuedRecord(
                timestamp_ms=next_timestamp_ms,
                token_id=record.token_id,
                payload=next_payload,
                iterator=record.iterator,
            ),
        )


def _render_markdown(
    results: list[BaselineDayResult],
    *,
    start_date: str,
    end_date: str,
    input_dir: Path,
    latency_sensitivity_ms: int,
    warmup_days: int,
) -> str:
    total_triggers = sum(result.trigger_count_per_day for result in results)
    total_fills = sum(result.total_simulated_fills for result in results)
    total_pnl = sum((result.pnl_usd for result in results), start=Decimal("0.00"))
    total_strict_partials = sum(result.strict_partial_fills for result in results)
    total_strict_dead = sum(result.strict_dead_strips for result in results)
    total_latency_full = sum(result.latency_full_fills for result in results)
    total_latency_legging = sum(result.latency_legging_events for result in results)
    total_latency_dead = sum(result.latency_dead_strips for result in results)
    total_launcher_rejections = sum(result.launcher_rejections for result in results)
    total_complete_events = sum(result.complete_event_groups for result in results)
    total_candidate_events = sum(result.candidate_event_groups for result in results)
    coverage_columns = sorted({key for result in results for key in result.coverage_rejections})

    lines = [
        f"# CLOB Arbitrage Strict Baseline ({start_date} to {end_date})",
        "",
        "## Setup",
        "",
        "- Strategy path: `scripts/live_bbo_arb_scanner.py` plus `scripts/launch_clob_arb.py`.",
        f"- Replay source: `{_raw_ticks_root(input_dir)}`.",
        "- Scan cadence: strict 300-second Sword scheduler cadence.",
        f"- Warmup carry-forward: {warmup_days} calendar archive day(s) replayed before each scored day so unchanged books can persist when the archive does not emit a same-day update.",
        "- Event grouping: Gamma events endpoint with full market reconstruction per event.",
        "- Strict coverage rule: a day is only eligible if every historically eligible outcome for that event is present somewhere in the warmup-plus-current archive window with a replayable YES-token book file.",
        "- Primary fill model: same-snapshot frozen-plan FOK-at-BBO upper bound using YES-book asks for `BUY_YES_STRIP` and YES-book bids for `SELL_NO_STRIP`.",
        f"- Legging sensitivity: the original strip size and prices are frozen at detection time, then rechecked at +{latency_sensitivity_ms} ms.",
        "",
        "## Baseline Table",
        "",
        "| Date | trigger_count_per_day | Total Simulated Fills | PnL (USD) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.day} | {result.trigger_count_per_day} | {result.total_simulated_fills} | {result.pnl_usd:.2f} |"
        )
    lines.extend(
        [
            f"| Total | {total_triggers} | {total_fills} | {total_pnl:.2f} |",
            "",
            "## Execution Risk Notes",
            "",
            "| Date | Strict Partial Fills | Strict Dead Strips | +Delay Full Fills | +Delay Legging Events | +Delay Dead Strips | Launcher Rejects | Complete Event Groups | Candidate Event Groups |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        lines.append(
            "| {day} | {strict_partial} | {strict_dead} | {latency_full} | {latency_legging} | {latency_dead} | {launcher_rejections} | {complete_events} | {candidate_events} |".format(
                day=result.day,
                strict_partial=result.strict_partial_fills,
                strict_dead=result.strict_dead_strips,
                latency_full=result.latency_full_fills,
                latency_legging=result.latency_legging_events,
                latency_dead=result.latency_dead_strips,
                launcher_rejections=result.launcher_rejections,
                complete_events=result.complete_event_groups,
                candidate_events=result.candidate_event_groups,
            )
        )
    lines.extend(
        [
            "| Total | {strict_partial} | {strict_dead} | {latency_full} | {latency_legging} | {latency_dead} | {launcher_rejections} | {complete_events} | {candidate_events} |".format(
                strict_partial=total_strict_partials,
                strict_dead=total_strict_dead,
                latency_full=total_latency_full,
                latency_legging=total_latency_legging,
                latency_dead=total_latency_dead,
                launcher_rejections=total_launcher_rejections,
                complete_events=total_complete_events,
                candidate_events=total_candidate_events,
            ),
            "",
            "## Coverage Rejections",
            "",
        ]
    )
    if coverage_columns:
        header = "| Date | " + " | ".join(column for column in coverage_columns) + " |"
        separator = "| --- | " + " | ".join("---:" for _ in coverage_columns) + " |"
        lines.extend([header, separator])
        for result in results:
            lines.append(
                "| {day} | {values} |".format(
                    day=result.day,
                    values=" | ".join(str(result.coverage_rejections.get(column, 0)) for column in coverage_columns),
                )
            )
        lines.append(
            "| Total | {values} |".format(
                values=" | ".join(str(sum(result.coverage_rejections.get(column, 0) for result in results)) for column in coverage_columns)
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `Total Simulated Fills` and `PnL (USD)` use the strict same-snapshot fill model, so they should be treated as an execution upper bound rather than a latency-adjusted realized estimate.",
            "- `+Delay Legging Events` counts strip attempts where some legs still satisfied the frozen plan after the delay but at least one leg did not.",
            "- `missing_full_event_coverage` means the archive never exposed every historically eligible outcome for that event within the warmup-plus-current replay window, so the group was excluded rather than priced as a false partial cluster.",
            "- `Launcher Rejects` are strips the live scanner would have emitted but `launch_clob_arb.py` would refuse because the whole-dollar notional cap or exchange minimum share rules collapse the executable size.",
            "",
            "## Verdict",
            "",
            "Across the measured range, the strict Sword replay generated `{triggers}` total triggers, `{fills}` strict same-snapshot strip fills, and `${pnl:.2f}` of gross strip edge capture. Under the +{delay} ms frozen-plan sensitivity, `{latency_full}` strips still cleared in full, `{latency_legging}` degraded into legging events, and `{latency_dead}` lost all executable legs.".format(
                triggers=total_triggers,
                fills=total_fills,
                pnl=total_pnl,
                delay=latency_sensitivity_ms,
                latency_full=total_latency_full,
                latency_legging=total_latency_legging,
                latency_dead=total_latency_dead,
            ),
        ]
    )
    return "\n".join(lines) + "\n"


async def _resolve_metadata(
    days: list[DayInputs],
    *,
    metadata_cache: GammaMetadataCache,
    timeout: float,
    concurrency: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, ResolvedEvent]]:
    unique_tokens = sorted({token_id for day in days for token_id in day.token_paths})
    print(f"metadata_resolve_start unique_tokens={len(unique_tokens)}", flush=True)
    async with GammaResolver(cache=metadata_cache, timeout=timeout, concurrency=concurrency) as resolver:
        market_rows = await asyncio.gather(*(resolver.market_for_token(token_id) for token_id in unique_tokens))
        token_market_rows = {token_id: row for token_id, row in zip(unique_tokens, market_rows, strict=True)}
        event_ids = sorted({_market_event_id(row) for row in token_market_rows.values() if _market_event_id(row)})
        print(f"metadata_resolve_progress unique_events={len(event_ids)}", flush=True)
        event_payloads = await asyncio.gather(*(resolver.event_for_id(event_id) for event_id in event_ids))
    resolved_events = _materialize_events(event_payloads)
    metadata_cache.save()
    print(
        f"metadata_resolve_complete cached_markets={len(metadata_cache.markets_by_token)} cached_events={len(metadata_cache.events_by_id)} resolved_events={len(resolved_events)}",
        flush=True,
    )
    return token_market_rows, resolved_events


async def _run_day(
    current_day: str,
    *,
    window_inputs: list[DayInputs],
    resolved_events: dict[str, ResolvedEvent],
    token_market_rows: dict[str, dict[str, Any]],
    fee_buffer: Decimal,
    min_leg_depth_usd: Decimal,
    latency_sensitivity_ms: int,
    progress_every_records: int,
) -> BaselineDayResult:
    print(f"baseline_day_start date={current_day} phase=setup", flush=True)
    observed_yes_token_paths_by_day: dict[str, list[Path]] = {}
    for day_input in window_inputs:
        for token_id, path in day_input.token_paths.items():
            market_row = token_market_rows.get(token_id)
            if market_row is None:
                continue
            token_ids = _parse_listish(market_row.get("clobTokenIds"))
            if len(token_ids) != 2:
                continue
            yes_token_id = _clean_text(token_ids[0])
            if token_id != yes_token_id:
                continue
            observed_yes_token_paths_by_day.setdefault(token_id, []).append(path)

    observed_yes_token_paths = {
        token_id: tuple(sorted(paths, key=lambda item: item.parent.name))
        for token_id, paths in observed_yes_token_paths_by_day.items()
    }
    complete_events, coverage_rejections, candidate_event_ids = _select_complete_events_for_day(
        current_day=current_day,
        window_inputs=window_inputs,
        resolved_events=resolved_events,
        token_market_rows=token_market_rows,
        observed_yes_token_paths=observed_yes_token_paths,
    )
    print(
        f"baseline_day_setup date={current_day} warmup_days={[item.day for item in window_inputs if item.day != current_day]} candidate_events={len(candidate_event_ids)} complete_events={len(complete_events)} coverage_rejections={dict(coverage_rejections)}",
        flush=True,
    )

    relevant_token_paths: dict[str, tuple[Path, ...]] = {}
    trackers: dict[str, OrderbookTracker] = {}
    for _, (_, event_markets) in sorted(complete_events.items()):
        for market in event_markets:
            if market.yes_token_id in relevant_token_paths:
                continue
            paths = observed_yes_token_paths.get(market.yes_token_id)
            if paths is None:
                continue
            relevant_token_paths[market.yes_token_id] = paths
            trackers[market.yes_token_id] = OrderbookTracker(market.yes_token_id)

    queue = _push_initial_records(relevant_token_paths, trackers)
    day_start, day_end, day_start_ms, day_end_ms = _day_window(current_day)
    del day_start, day_end
    scan_times = list(range(day_start_ms, day_end_ms + 1, SCAN_INTERVAL_MS))
    pending_latency_attempts: list[tuple[int, int, PendingLatencyAttempt]] = []
    processed_records = [0]
    attempt_sequence = 0

    trigger_count = 0
    strict_fills = 0
    strict_partials = 0
    strict_dead = 0
    strict_pnl = Decimal("0.00")
    latency_full_fills = 0
    latency_legging_events = 0
    latency_dead_strips = 0
    launcher_rejections = 0
    scan_rejections: Counter[str] = Counter()
    launcher_rejection_reasons: Counter[str] = Counter()

    scan_index = 0
    while scan_index < len(scan_times) or pending_latency_attempts:
        next_scan_ms = scan_times[scan_index] if scan_index < len(scan_times) else None
        next_latency_ms = pending_latency_attempts[0][0] if pending_latency_attempts else None
        if next_scan_ms is None:
            checkpoint_ms = next_latency_ms
        elif next_latency_ms is None:
            checkpoint_ms = next_scan_ms
        else:
            checkpoint_ms = min(next_scan_ms, next_latency_ms)
        if checkpoint_ms is None:
            break

        _advance_trackers_to(
            checkpoint_ms,
            queue=queue,
            trackers=trackers,
            processed_records=processed_records,
            progress_every_records=progress_every_records,
            day=current_day,
        )

        while pending_latency_attempts and pending_latency_attempts[0][0] == checkpoint_ms:
            _, _, pending_attempt = heapq.heappop(pending_latency_attempts)
            outcome = _simulate_execution(pending_attempt.plan, trackers)
            if outcome.status == "full_fill":
                latency_full_fills += 1
            elif outcome.status == "legging":
                latency_legging_events += 1
            else:
                latency_dead_strips += 1

        if next_scan_ms is not None and checkpoint_ms == next_scan_ms:
            for event_id in sorted(complete_events):
                event, event_markets = complete_events[event_id]
                strip_targets, rejection_reason = _evaluate_event_snapshot(
                    event,
                    event_markets,
                    trackers,
                    fee_buffer=fee_buffer,
                    min_leg_depth_usd=min_leg_depth_usd,
                )
                if rejection_reason is not None:
                    scan_rejections[rejection_reason] += 1
                    continue
                for target in strip_targets:
                    trigger_count += 1
                    try:
                        plan = _build_order_plan(target)
                    except RuntimeError as exc:
                        launcher_rejections += 1
                        launcher_rejection_reasons[str(exc)] += 1
                        continue
                    strict_outcome = _simulate_execution(plan, trackers)
                    if strict_outcome.status == "full_fill":
                        strict_fills += 1
                        strict_pnl += _gross_pnl(plan)
                    elif strict_outcome.status == "legging":
                        strict_partials += 1
                    else:
                        strict_dead += 1
                    if latency_sensitivity_ms > 0:
                        pending_attempt = PendingLatencyAttempt(
                            execute_at_ms=checkpoint_ms + latency_sensitivity_ms,
                            plan=plan,
                        )
                        attempt_sequence += 1
                        heapq.heappush(
                            pending_latency_attempts,
                            (pending_attempt.execute_at_ms, attempt_sequence, pending_attempt),
                        )
            scan_index += 1

    print(
        "baseline_day_complete date={day} triggers={triggers} strict_fills={fills} strict_pnl_usd={pnl:.2f} strict_partials={strict_partials} latency_full={latency_full} latency_legging={latency_legging} latency_dead={latency_dead} launcher_rejections={launcher_rejections} processed_records={records}".format(
            day=current_day,
            triggers=trigger_count,
            fills=strict_fills,
            pnl=strict_pnl,
            strict_partials=strict_partials,
            latency_full=latency_full_fills,
            latency_legging=latency_legging_events,
            latency_dead=latency_dead_strips,
            launcher_rejections=launcher_rejections,
            records=processed_records[0],
        ),
        flush=True,
    )
    condensed_launcher_reasons = Counter()
    for message, count in launcher_rejection_reasons.items():
        if "below exchange minimum shares" in message:
            condensed_launcher_reasons["below_exchange_minimum_shares"] += count
        elif "below exchange minimum" in message:
            condensed_launcher_reasons["below_exchange_minimum_notional"] += count
        else:
            condensed_launcher_reasons[message] += count

    return BaselineDayResult(
        day=current_day,
        trigger_count_per_day=trigger_count,
        total_simulated_fills=strict_fills,
        pnl_usd=_usd(strict_pnl),
        strict_partial_fills=strict_partials,
        strict_dead_strips=strict_dead,
        latency_full_fills=latency_full_fills,
        latency_legging_events=latency_legging_events,
        latency_dead_strips=latency_dead_strips,
        launcher_rejections=launcher_rejections,
        complete_event_groups=len(complete_events),
        candidate_event_groups=len(candidate_event_ids),
        scan_rejections=dict(scan_rejections),
        coverage_rejections=dict(coverage_rejections),
        launcher_rejection_reasons=dict(condensed_launcher_reasons),
    )


async def _main() -> None:
    args = parse_args()
    raw_ticks_root = _raw_ticks_root(args.input_dir)
    start_date = _date_from_day(args.start_date)
    warmup_start = (start_date - timedelta(days=max(0, args.warmup_days))).isoformat()
    all_days: dict[str, DayInputs] = {}
    for day in _iter_calendar_dates(warmup_start, args.end_date):
        day_dir = raw_ticks_root / day
        if not day_dir.exists():
            continue
        all_days[day] = _load_day_inputs(raw_ticks_root, day)
    scored_days = _iter_dates(args.start_date, args.end_date)
    markdown_output = args.output or _default_markdown_output_path(args.start_date, args.end_date)
    json_output = args.json_output or _default_json_output_path(args.start_date, args.end_date)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    metadata_cache = GammaMetadataCache(args.metadata_cache)
    token_market_rows, resolved_events = await _resolve_metadata(
        list(all_days.values()),
        metadata_cache=metadata_cache,
        timeout=args.timeout,
        concurrency=args.http_concurrency,
    )

    results: list[BaselineDayResult] = []
    for current_day in scored_days:
        window_inputs = _window_days_for_target(
            target_day=current_day,
            available_days=all_days,
            warmup_days=args.warmup_days,
        )
        if not window_inputs:
            raise RuntimeError(f"No archive days found for scored day {current_day}")
        results.append(
            await _run_day(
                current_day,
                window_inputs=window_inputs,
                resolved_events=resolved_events,
                token_market_rows=token_market_rows,
                fee_buffer=args.fee_buffer,
                min_leg_depth_usd=args.min_leg_depth_usd,
                latency_sensitivity_ms=args.latency_sensitivity_ms,
                progress_every_records=args.progress_every_records,
            )
        )

    markdown = _render_markdown(
        results,
        start_date=args.start_date,
        end_date=args.end_date,
        input_dir=args.input_dir,
        latency_sensitivity_ms=args.latency_sensitivity_ms,
        warmup_days=args.warmup_days,
    )
    markdown_output.write_text(markdown, encoding="utf-8")

    json_payload = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "input_dir": str(args.input_dir),
        "latency_sensitivity_ms": args.latency_sensitivity_ms,
        "warmup_days": args.warmup_days,
        "fee_buffer": str(args.fee_buffer),
        "min_leg_depth_usd": str(args.min_leg_depth_usd),
        "results": [asdict(result) for result in results],
    }
    json_output.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")

    print("\n---MARKDOWN---\n", flush=True)
    print(markdown, flush=True)
    print(f"baseline_output_markdown={markdown_output}", flush=True)
    print(f"baseline_output_json={json_output}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())