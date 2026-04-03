from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import heapq
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Sequence

from src.core.config import settings
from src.core.logger import get_logger
from src.data.market_discovery import MarketInfo
from src.data.orderbook import OrderbookTracker
from src.execution.orchestrator_health_monitor import HealthReport
from src.execution.reward_poster_adapter import RewardQuoteState
from src.monitoring.trade_store import TradeStore
from src.rewards.models import RewardPosterIntent
from src.rewards.reward_poster_sidecar import RewardPosterSidecar
from src.rewards.reward_selector import RewardSelector, RewardSelectorConfig
from src.rewards.reward_shadow_metrics import build_shadow_extra_payload


log = get_logger(__name__)

_MONTH_INDEX = {
    month.lower(): index
    for index, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    timestamp_ms: int
    event_type: str
    asset_id: str
    market_id: str
    payload: dict[str, Any]
    dedupe_key: tuple[Any, ...]
    trade_price: Decimal | None = None
    trade_size: Decimal | None = None
    trade_side: str | None = None


@dataclass(slots=True)
class ReplaySummary:
    total_events: int = 0
    book_events: int = 0
    trade_events: int = 0
    matched_fills: int = 0
    persisted_shadow_rows: int = 0
    first_timestamp_ms: int = 0
    last_timestamp_ms: int = 0


@dataclass(slots=True)
class ReplayConfig:
    input_dir: Path
    db_path: Path
    reward_universe_path: Path
    market_map_path: Path | None = None
    activation_latency_ms: int = 50
    max_events: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    condition_ids: frozenset[str] = field(default_factory=frozenset)
    selector_config: RewardSelectorConfig = field(default_factory=RewardSelectorConfig)
    reward_market_cap: int | None = None
    reward_quote_cap: int | None = None
    reward_quote_notional_cap: float | None = None
    reward_inventory_cap: float | None = None
    reward_cancel_on_stale_ms: int | None = None
    reward_replace_only_if_price_moves_ticks: int | None = None
    reward_refresh_interval_ms: int | None = None
    markout_horizons_seconds: tuple[int, ...] = (5, 15, 60)


@dataclass(slots=True)
class _SimulatedOrderRecord:
    order_id: str
    quote_id: str
    asset_id: str
    market_id: str
    target_price: Decimal
    target_size: Decimal
    activation_ms: int
    filled_size: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None
    cancelled: bool = False
    last_update_ms: int = 0

    @property
    def remaining_size(self) -> Decimal:
        return max(self.target_size - self.filled_size, Decimal("0"))


@dataclass(slots=True)
class _PendingMarkoutPersist:
    payload: dict[str, object]
    asset_id: str
    direction: str
    reference_price: Decimal
    realized_notional_usd: Decimal
    deadlines_ms: dict[int, int]
    mark_prices: dict[int, Decimal] = field(default_factory=dict)


class ReplayOrderbookTracker(OrderbookTracker):
    def on_price_change(self, data: dict) -> None:
        super().on_price_change(data)
        self._last_update = _timestamp_seconds(data)

    def on_book_snapshot(self, data: dict) -> None:
        super().on_book_snapshot(data)
        self._last_update = _timestamp_seconds(data)


class ReplayRewardAdapter:
    def __init__(self, *, activation_latency_ms: int) -> None:
        self._activation_latency_ms = max(int(activation_latency_ms), 0)
        self._orders: dict[str, _SimulatedOrderRecord] = {}
        self._next_order_id = 0

    def submit_intent(self, intent: RewardPosterIntent, timestamp_ms: int) -> RewardQuoteState:
        self._next_order_id += 1
        order_id = f"replay-order-{self._next_order_id}"
        self._orders[order_id] = _SimulatedOrderRecord(
            order_id=order_id,
            quote_id=intent.quote_id,
            asset_id=intent.asset_id,
            market_id=intent.market_id,
            target_price=intent.target_price,
            target_size=intent.target_size,
            activation_ms=timestamp_ms + self._activation_latency_ms,
            last_update_ms=timestamp_ms,
        )
        return RewardQuoteState(
            quote_id=intent.quote_id,
            market_id=intent.market_id,
            asset_id=intent.asset_id,
            side=intent.side,
            target_price=intent.target_price,
            target_size=intent.target_size,
            max_capital=intent.max_capital,
            status="WORKING",
            order_id=order_id,
            remaining_size=intent.target_size,
            last_update_ms=timestamp_ms,
            extra_payload=intent.as_signal_metadata(),
        )

    def sync_quote(self, state: RewardQuoteState, timestamp_ms: int) -> RewardQuoteState:
        if state.order_id is None:
            return state
        record = self._orders.get(state.order_id)
        if record is None:
            return state
        record.last_update_ms = timestamp_ms
        if record.cancelled:
            return RewardQuoteState(
                quote_id=state.quote_id,
                market_id=state.market_id,
                asset_id=state.asset_id,
                side=state.side,
                target_price=state.target_price,
                target_size=state.target_size,
                max_capital=state.max_capital,
                status="CANCELLED",
                order_id=state.order_id,
                filled_size=record.filled_size,
                remaining_size=Decimal("0"),
                filled_price=record.average_fill_price,
                guard_reason=state.guard_reason,
                last_update_ms=timestamp_ms,
                extra_payload=state.extra_payload,
            )
        if record.filled_size >= record.target_size:
            status = "FILLED"
            remaining_size = Decimal("0")
        elif record.filled_size > Decimal("0"):
            status = "PARTIAL"
            remaining_size = record.remaining_size
        else:
            status = "WORKING"
            remaining_size = record.remaining_size
        return RewardQuoteState(
            quote_id=state.quote_id,
            market_id=state.market_id,
            asset_id=state.asset_id,
            side=state.side,
            target_price=state.target_price,
            target_size=state.target_size,
            max_capital=state.max_capital,
            status=status,
            order_id=state.order_id,
            filled_size=record.filled_size,
            remaining_size=remaining_size,
            filled_price=record.average_fill_price,
            guard_reason=state.guard_reason,
            last_update_ms=timestamp_ms,
            extra_payload=state.extra_payload,
        )

    def cancel_quote(self, state: RewardQuoteState, timestamp_ms: int) -> RewardQuoteState:
        if state.order_id is not None:
            record = self._orders.get(state.order_id)
            if record is not None:
                record.cancelled = True
                record.last_update_ms = timestamp_ms
        return RewardQuoteState(
            quote_id=state.quote_id,
            market_id=state.market_id,
            asset_id=state.asset_id,
            side=state.side,
            target_price=state.target_price,
            target_size=state.target_size,
            max_capital=state.max_capital,
            status="CANCELLED",
            order_id=state.order_id,
            filled_size=state.filled_size,
            remaining_size=Decimal("0"),
            filled_price=state.filled_price,
            guard_reason=state.guard_reason,
            last_update_ms=timestamp_ms,
            extra_payload=state.extra_payload,
        )

    def is_active_for_matching(self, order_id: str, current_timestamp_ms: int) -> bool:
        record = self._orders.get(order_id)
        if record is None:
            return False
        if record.cancelled or record.remaining_size <= Decimal("0"):
            return False
        return current_timestamp_ms >= record.activation_ms

    def record_fill(
        self,
        order_id: str,
        *,
        fill_size: Decimal,
        fill_price: Decimal,
        timestamp_ms: int,
    ) -> SimpleNamespace | None:
        record = self._orders.get(order_id)
        if record is None or record.cancelled or fill_size <= Decimal("0"):
            return None
        applied_size = min(fill_size, record.remaining_size)
        if applied_size <= Decimal("0"):
            return None
        if record.average_fill_price is None or record.filled_size <= Decimal("0"):
            next_average = fill_price
        else:
            weighted_notional = (record.average_fill_price * record.filled_size) + (fill_price * applied_size)
            next_average = weighted_notional / (record.filled_size + applied_size)
        record.filled_size += applied_size
        record.average_fill_price = next_average
        record.last_update_ms = timestamp_ms
        return SimpleNamespace(
            order_id=order_id,
            filled_size=float(record.filled_size),
            filled_avg_price=float(record.average_fill_price),
        )


class RewardReplayEngine:
    def __init__(self, config: ReplayConfig, markets: Sequence[MarketInfo]) -> None:
        self._config = config
        self._markets = list(markets)
        self._market_by_asset: dict[str, MarketInfo] = {}
        self._trackers: dict[str, ReplayOrderbookTracker] = {}
        for market in self._markets:
            self._market_by_asset[market.yes_token_id] = market
            self._market_by_asset[market.no_token_id] = market
            self._trackers[market.yes_token_id] = ReplayOrderbookTracker(market.yes_token_id)
            self._trackers[market.no_token_id] = ReplayOrderbookTracker(market.no_token_id)

        self._now_ms = 0
        self._summary = ReplaySummary()
        self._trade_store = TradeStore(config.db_path)
        self._pending_persist: list[dict[str, object]] = []
        self._pending_markout_persists: list[_PendingMarkoutPersist] = []
        self._adapter = ReplayRewardAdapter(activation_latency_ms=config.activation_latency_ms)
        self._orchestrator = SimpleNamespace(reward_poster_adapter=self._adapter)
        self._sidecar = RewardPosterSidecar(
            orchestrator=self._orchestrator,
            selector=RewardSelector(config.selector_config),
            markets_provider=lambda: list(self._markets),
            market_by_asset_provider=lambda asset_id: self._market_by_asset.get(asset_id),
            book_provider=lambda asset_id: self._trackers.get(asset_id),
            health_report_provider=self._green_health_report,
            maker_monitor=None,
            now_ms=lambda: self._now_ms,
            shadow_persist_callback=self._queue_shadow_persist,
        )

    async def run(self) -> ReplaySummary:
        with self._override_strategy_settings():
            tracked_ids = tracked_file_ids(self._markets)
            events = iter_replay_events(
                self._config.input_dir,
                tracked_ids=tracked_ids,
                start_date=self._config.start_date,
                end_date=self._config.end_date,
            )
            for event in events:
                self._now_ms = event.timestamp_ms
                self._release_replay_inventory(self._now_ms)
                if self._summary.first_timestamp_ms <= 0:
                    self._summary.first_timestamp_ms = event.timestamp_ms
                self._summary.last_timestamp_ms = event.timestamp_ms
                self._summary.total_events += 1
                self._sidecar.on_tick(self._now_ms)
                if event.event_type in {"BOOK", "PRICE_CHANGE"}:
                    tracker = self._trackers.get(event.asset_id)
                    if tracker is not None:
                        if event.event_type == "BOOK":
                            tracker.on_book_snapshot(event.payload)
                        else:
                            tracker.on_price_change(event.payload)
                        self._sidecar.on_book_update(event.asset_id, current_timestamp_ms=self._now_ms)
                        self._capture_markouts(event.asset_id, self._now_ms)
                        self._summary.book_events += 1
                elif event.event_type == "TRADE":
                    self._summary.trade_events += 1
                    self._summary.matched_fills += self._match_trade(event)
                await self._flush_persists()
                if self._config.max_events is not None and self._summary.total_events >= self._config.max_events:
                    break

            if self._summary.last_timestamp_ms > 0:
                stale_window_ms = int(self._effective_strategy_value("reward_cancel_on_stale_ms"))
                self._now_ms = self._summary.last_timestamp_ms + stale_window_ms + 1
                self._release_replay_inventory(self._now_ms)
                self._sidecar.on_tick(self._now_ms)
                self._finalize_markout_persists(force=True)
                await self._flush_persists()
        await self._trade_store.close()
        return self._summary

    def _queue_shadow_persist(self, payload: dict[str, object]) -> None:
        queued_payload = dict(payload)
        extra_payload = queued_payload.get("extra_payload")
        normalized_extra = dict(extra_payload) if isinstance(extra_payload, dict) else {}
        queued_payload["extra_payload"] = normalized_extra
        if normalized_extra.get("fill_occurred") is True:
            pending = self._build_pending_markout_persist(queued_payload, normalized_extra)
            if pending is not None:
                self._pending_markout_persists.append(pending)
                return
        self._pending_persist.append(queued_payload)

    async def _flush_persists(self) -> None:
        if not self._pending_persist:
            return
        pending = self._pending_persist
        self._pending_persist = []
        for payload in pending:
            await self._trade_store.record_shadow_trade(**payload)
            self._summary.persisted_shadow_rows += 1

    def _match_trade(self, event: ReplayEvent) -> int:
        if event.trade_price is None or event.trade_size is None:
            return 0
        for quote in list(self._sidecar.working_quotes.values()):
            if quote.asset_id != event.asset_id:
                continue
            if quote.status not in {"WORKING", "PARTIAL"}:
                continue
            if quote.order_id is None:
                continue
            if not self._adapter.is_active_for_matching(quote.order_id, self._now_ms):
                continue
            if event.trade_price > quote.target_price:
                continue
            fill_order = self._adapter.record_fill(
                quote.order_id,
                fill_size=event.trade_size,
                fill_price=event.trade_price,
                timestamp_ms=self._now_ms,
            )
            if fill_order is None:
                return 0
            return 1 if self._sidecar.on_fill(fill_order, current_timestamp_ms=self._now_ms) else 0
        return 0

    def _release_replay_inventory(self, current_timestamp_ms: int) -> None:
        for inventory in self._sidecar._inventory.values():
            if inventory.filled_notional <= Decimal("0"):
                continue
            if inventory.flatten_due_ms <= 0 or current_timestamp_ms < inventory.flatten_due_ms:
                continue
            inventory.filled_size = Decimal("0")
            inventory.filled_notional = Decimal("0")
            inventory.flatten_due_ms = 0
            inventory.flatten_escalated = False

    def _build_pending_markout_persist(
        self,
        payload: dict[str, object],
        extra_payload: dict[str, object],
    ) -> _PendingMarkoutPersist | None:
        asset_id = str(payload.get("asset_id") or "").strip()
        if not asset_id:
            return None
        exit_time = float(payload.get("exit_time") or 0.0)
        if exit_time <= 0.0:
            return None
        reference_price = Decimal(str(payload.get("entry_price") or payload.get("reference_price") or 0.0))
        if reference_price <= Decimal("0"):
            return None
        realized_notional_usd = Decimal(str(payload.get("entry_price") or 0.0)) * Decimal(str(payload.get("entry_size") or 0.0))
        fill_timestamp_ms = int(round(exit_time * 1000))
        deadlines_ms = {
            int(horizon_seconds): fill_timestamp_ms + int(horizon_seconds) * 1000
            for horizon_seconds in self._config.markout_horizons_seconds
            if int(horizon_seconds) > 0
        }
        if not deadlines_ms:
            return None
        return _PendingMarkoutPersist(
            payload=payload,
            asset_id=asset_id,
            direction=str(payload.get("direction") or "YES"),
            reference_price=reference_price,
            realized_notional_usd=realized_notional_usd,
            deadlines_ms=deadlines_ms,
        )

    def _capture_markouts(self, asset_id: str, current_timestamp_ms: int) -> None:
        tracker = self._trackers.get(asset_id)
        if tracker is None:
            return
        snapshot = tracker.snapshot()
        mid_price = Decimal(str(getattr(snapshot, "mid_price", 0.0) or 0.0))
        if mid_price <= Decimal("0"):
            return
        for pending in self._pending_markout_persists:
            if pending.asset_id != asset_id:
                continue
            for horizon_seconds, deadline_ms in pending.deadlines_ms.items():
                if horizon_seconds in pending.mark_prices:
                    continue
                if current_timestamp_ms >= deadline_ms:
                    pending.mark_prices[horizon_seconds] = mid_price
        self._finalize_markout_persists(force=False)

    def _finalize_markout_persists(self, *, force: bool) -> None:
        remaining: list[_PendingMarkoutPersist] = []
        expected_horizons = {int(horizon) for horizon in self._config.markout_horizons_seconds if int(horizon) > 0}
        for pending in self._pending_markout_persists:
            if not force and not expected_horizons.issubset(set(pending.mark_prices)):
                remaining.append(pending)
                continue
            payload = dict(pending.payload)
            extra_payload = dict(payload.get("extra_payload") or {})
            updated_extra = build_shadow_extra_payload(
                reward_daily_usd=extra_payload.get("reward_daily_usd"),
                reward_min_size=extra_payload.get("reward_min_size"),
                reward_max_spread_cents=extra_payload.get("reward_max_spread_cents"),
                competition_usd=extra_payload.get("competition_usd"),
                reward_to_competition=extra_payload.get("reward_to_competition"),
                queue_depth_ahead_usd=extra_payload.get("queue_depth_ahead_usd"),
                quoted_at=extra_payload.get("quoted_at"),
                terminal_at=extra_payload.get("terminal_at"),
                queue_residency_seconds=extra_payload.get("queue_residency_seconds"),
                fill_occurred=True,
                fill_latency_ms=extra_payload.get("fill_latency_ms"),
                reference_price=float(pending.reference_price),
                direction=pending.direction,
                mark_price_5s=float(pending.mark_prices[5]) if 5 in pending.mark_prices else None,
                mark_price_15s=float(pending.mark_prices[15]) if 15 in pending.mark_prices else None,
                mark_price_60s=float(pending.mark_prices[60]) if 60 in pending.mark_prices else None,
                estimated_reward_capture_usd=extra_payload.get("estimated_reward_capture_usd"),
                quote_size_usd=float(pending.realized_notional_usd),
                fees_paid_usd=0.0,
                quote_id=extra_payload.get("quote_id"),
                quote_reason=extra_payload.get("quote_reason"),
                emergency_flatten=extra_payload.get("emergency_flatten"),
            )
            extra_payload.update(updated_extra)
            payload["extra_payload"] = extra_payload
            self._pending_persist.append(payload)
        self._pending_markout_persists = remaining

    def _green_health_report(self, current_timestamp_ms: int) -> HealthReport:
        return HealthReport(
            timestamp_ms=current_timestamp_ms,
            orchestrator_health="GREEN",
            is_safe_to_trade=True,
            consecutive_release_failures=0,
            last_snapshot_age_ms=0,
            heartbeat_ok=True,
            halt_reason=None,
        )

    def _effective_strategy_value(self, field_name: str) -> object:
        override_value = getattr(self._config, field_name, None)
        if override_value is not None:
            return override_value
        selector_value = getattr(self._config.selector_config, field_name, None)
        if selector_value is not None:
            return selector_value
        return getattr(settings.strategy, field_name)

    @contextmanager
    def _override_strategy_settings(self) -> Iterator[None]:
        strategy_updates: dict[str, object] = {}
        for field_name in (
            "reward_market_cap",
            "reward_quote_cap",
            "reward_quote_notional_cap",
            "reward_inventory_cap",
            "reward_cancel_on_stale_ms",
            "reward_replace_only_if_price_moves_ticks",
            "reward_refresh_interval_ms",
        ):
            override_value = getattr(self._config, field_name)
            if override_value is not None:
                strategy_updates[field_name] = override_value
        if not strategy_updates:
            yield
            return

        original_strategy = settings.strategy
        object.__setattr__(settings, "strategy", replace(original_strategy, **strategy_updates))
        try:
            yield
        finally:
            object.__setattr__(settings, "strategy", original_strategy)


def tracked_file_ids(markets: Sequence[MarketInfo]) -> frozenset[str]:
    ids: set[str] = set()
    for market in markets:
        ids.add(str(market.condition_id))
        ids.add(str(market.yes_token_id))
        ids.add(str(market.no_token_id))
    return frozenset(ids)


def iter_replay_events(
    input_dir: Path,
    *,
    tracked_ids: frozenset[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[ReplayEvent]:
    recent_keys: OrderedDict[tuple[Any, ...], None] = OrderedDict()
    for date_dir in _iter_date_dirs(input_dir, start_date=start_date, end_date=end_date):
        file_paths = sorted(
            path
            for path in date_dir.glob("*.jsonl")
            if path.stem in tracked_ids
        )
        if not file_paths:
            continue
        heap: list[tuple[int, int, ReplayEvent, Iterator[ReplayEvent]]] = []
        for index, path in enumerate(file_paths):
            iterator = _iter_file_events(path, tracked_ids=tracked_ids)
            first_event = next(iterator, None)
            if first_event is not None:
                heapq.heappush(heap, (first_event.timestamp_ms, index, first_event, iterator))
        while heap:
            _, index, event, iterator = heapq.heappop(heap)
            if event.dedupe_key not in recent_keys:
                recent_keys[event.dedupe_key] = None
                if len(recent_keys) > 8192:
                    recent_keys.popitem(last=False)
                yield event
            next_event = next(iterator, None)
            if next_event is not None:
                heapq.heappush(heap, (next_event.timestamp_ms, index, next_event, iterator))


def load_reward_markets(
    reward_universe_path: Path,
    *,
    replay_anchor_ms: int,
    market_map_path: Path | None = None,
    condition_ids: frozenset[str] = frozenset(),
) -> list[MarketInfo]:
    market_map = _load_market_map(market_map_path)
    raw_markets = json.loads(reward_universe_path.read_text(encoding="utf-8"))
    anchor_dt = datetime.fromtimestamp(replay_anchor_ms / 1000.0, tz=timezone.utc)
    loaded: list[MarketInfo] = []
    for row in raw_markets:
        condition_id = str(row.get("condition_id") or row.get("market_id") or "").strip()
        if not condition_id:
            continue
        if condition_ids and condition_id not in condition_ids:
            continue
        yes_token_id, no_token_id = _extract_token_ids(row, market_map.get(condition_id))
        if not yes_token_id or not no_token_id:
            continue
        question = str(row.get("question") or row.get("title") or condition_id)
        reward_daily_rate = float(row.get("daily_reward_usd") or row.get("rewards_daily_rate") or 0.0)
        reward_min_size = float(row.get("reward_min_size") or 0.0)
        reward_max_spread = float(row.get("reward_max_spread_cents") or 0.0)
        reward_to_competition = float(row.get("reward_to_competition") or 0.0)
        if reward_to_competition > 0.0 and reward_daily_rate > 0.0:
            competition_score = reward_daily_rate / reward_to_competition
        else:
            competition_score = float(row.get("competition_usd") or 0.0)
        if competition_score <= 0.0:
            competition_score = 1.0 if reward_daily_rate > 0.0 else 0.0
        end_date = _infer_end_date(question, anchor_dt)
        if end_date is None:
            end_date = anchor_dt + timedelta(days=14)
        loaded.append(
            MarketInfo(
                condition_id=condition_id,
                question=question,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                daily_volume_usd=float(row.get("volume_24h") or row.get("daily_volume_usd") or 0.0),
                end_date=end_date,
                active=True,
                event_id=str(row.get("market_id") or ""),
                liquidity_usd=float(row.get("liquidity") or 0.0),
                accepting_orders=True,
                tags=str(row.get("tags") or "reward"),
                neg_risk=False,
                reward_program_active=reward_daily_rate > 0.0,
                reward_daily_rate_usd=reward_daily_rate,
                reward_min_size=reward_min_size,
                reward_max_spread_cents=reward_max_spread,
                reward_competition_score=competition_score,
            )
        )
    return loaded


def replay_anchor_ms_for_range(input_dir: Path, start_date: str | None = None) -> int:
    date_dirs = list(_iter_date_dirs(input_dir, start_date=start_date, end_date=start_date))
    if not date_dirs:
        raise FileNotFoundError(f"No raw tick directories found under {input_dir}")
    return int(datetime.fromisoformat(date_dirs[0].name).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _iter_date_dirs(input_dir: Path, *, start_date: str | None, end_date: str | None) -> Iterator[Path]:
    lower = start_date or ""
    upper = end_date or "9999-12-31"
    for path in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if path.name < lower or path.name > upper:
            continue
        yield path


def _iter_file_events(path: Path, *, tracked_ids: frozenset[str]) -> Iterator[ReplayEvent]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                log.warning("reward_replay_bad_jsonl_row", path=str(path))
                continue
            yield from _normalize_raw_record(raw, tracked_ids=tracked_ids)


def _normalize_raw_record(raw: dict[str, Any], *, tracked_ids: frozenset[str]) -> Iterator[ReplayEvent]:
    payload = raw.get("payload") or {}
    if not isinstance(payload, dict):
        return
    event_type = str(payload.get("event_type") or "").strip().lower()
    timestamp_ms = _timestamp_ms(raw, payload)
    if timestamp_ms <= 0:
        return

    if event_type == "book":
        asset_id = str(payload.get("asset_id") or "").strip()
        market_id = str(payload.get("market") or raw.get("asset_id") or "").strip()
        if asset_id and asset_id in tracked_ids:
            dedupe_key = ("BOOK", asset_id, timestamp_ms, str(payload.get("hash") or ""))
            yield ReplayEvent(
                timestamp_ms=timestamp_ms,
                event_type="BOOK",
                asset_id=asset_id,
                market_id=market_id,
                payload=dict(payload),
                dedupe_key=dedupe_key,
            )
        return

    if event_type == "price_change":
        market_id = str(payload.get("market") or raw.get("asset_id") or "").strip()
        source = str(raw.get("source") or "").strip().lower()
        changes = payload.get("price_changes") or []
        for change in changes:
            asset_id = str(change.get("asset_id") or "").strip()
            if asset_id not in tracked_ids:
                continue
            dedupe_key = (
                "PRICE_CHANGE",
                asset_id,
                timestamp_ms,
                str(change.get("hash") or ""),
                str(change.get("price") or ""),
                str(change.get("size") or ""),
                str(change.get("side") or ""),
            )
            yield ReplayEvent(
                timestamp_ms=timestamp_ms,
                event_type="PRICE_CHANGE",
                asset_id=asset_id,
                market_id=market_id,
                payload={
                    "asset_id": asset_id,
                    "timestamp": str(timestamp_ms),
                    "changes": [
                        {
                            "price": change.get("price"),
                            "size": change.get("size"),
                            "side": change.get("side"),
                        }
                    ],
                },
                dedupe_key=dedupe_key,
            )
            if source == "trade":
                yield ReplayEvent(
                    timestamp_ms=timestamp_ms,
                    event_type="TRADE",
                    asset_id=asset_id,
                    market_id=market_id,
                    payload={
                        "market": market_id,
                        "asset_id": asset_id,
                        "timestamp": str(timestamp_ms),
                        "price": change.get("price"),
                        "size": change.get("size"),
                        "side": change.get("side"),
                        "hash": change.get("hash"),
                    },
                    dedupe_key=(
                        "TRADE_PRICE_CHANGE",
                        asset_id,
                        timestamp_ms,
                        str(change.get("hash") or ""),
                    ),
                    trade_price=Decimal(str(change.get("price") or 0)),
                    trade_size=Decimal(str(change.get("size") or 0)),
                    trade_side=str(change.get("side") or "").upper(),
                )
        return

    if event_type == "last_trade_price":
        asset_id = str(payload.get("asset_id") or "").strip()
        market_id = str(payload.get("market") or "").strip()
        if asset_id and asset_id in tracked_ids:
            yield ReplayEvent(
                timestamp_ms=timestamp_ms,
                event_type="TRADE",
                asset_id=asset_id,
                market_id=market_id,
                payload=dict(payload),
                dedupe_key=(
                    "TRADE",
                    asset_id,
                    timestamp_ms,
                    str(payload.get("transaction_hash") or payload.get("hash") or ""),
                ),
                trade_price=Decimal(str(payload.get("price") or 0)),
                trade_size=Decimal(str(payload.get("size") or 0)),
                trade_side=str(payload.get("side") or "").upper(),
            )


def _timestamp_ms(raw: dict[str, Any], payload: dict[str, Any]) -> int:
    value = payload.get("timestamp")
    if value not in (None, ""):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    local_ts = raw.get("local_ts")
    if local_ts not in (None, ""):
        try:
            return int(float(local_ts) * 1000)
        except (TypeError, ValueError):
            return 0
    return 0


def _timestamp_seconds(payload: dict[str, Any]) -> float:
    try:
        return _timestamp_ms({}, payload) / 1000.0
    except Exception:
        return 0.0


def _extract_token_ids(row: dict[str, Any], market_map_row: dict[str, str] | None) -> tuple[str, str]:
    yes_token_id = ""
    no_token_id = ""
    for token_row in row.get("token_audit") or []:
        outcome = str(token_row.get("outcome") or "").strip().upper()
        token_id = str(token_row.get("token_id") or "").strip()
        if outcome == "YES" and token_id:
            yes_token_id = token_id
        elif outcome == "NO" and token_id:
            no_token_id = token_id
    if market_map_row is not None:
        yes_token_id = yes_token_id or market_map_row.get("yes_id", "")
        no_token_id = no_token_id or market_map_row.get("no_id", "")
    return yes_token_id, no_token_id


def _load_market_map(market_map_path: Path | None) -> dict[str, dict[str, str]]:
    if market_map_path is None or not market_map_path.exists():
        return {}
    rows = json.loads(market_map_path.read_text(encoding="utf-8"))
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        market_id = str(row.get("market_id") or "").strip()
        if market_id:
            indexed[market_id] = {
                "yes_id": str(row.get("yes_id") or "").strip(),
                "no_id": str(row.get("no_id") or "").strip(),
            }
    return indexed


def _infer_end_date(question: str, anchor_dt: datetime) -> datetime | None:
    normalized = str(question or "").strip()
    if not normalized:
        return None

    before_year = re.search(r"before\s+(\d{4})", normalized, flags=re.IGNORECASE)
    if before_year:
        year = int(before_year.group(1))
        return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    end_of_year = re.search(r"by\s+end\s+of\s+(\d{4})", normalized, flags=re.IGNORECASE)
    if end_of_year:
        year = int(end_of_year.group(1))
        return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    month_day = re.search(
        r"(?:by|before|out by)\s+([A-Za-z]+)\s+(\d{1,2})(?:,?\s*(\d{4}))?",
        normalized,
        flags=re.IGNORECASE,
    )
    if month_day:
        month_name = month_day.group(1).lower()
        month = _MONTH_INDEX.get(month_name)
        if month is not None:
            day = int(month_day.group(2))
            year = int(month_day.group(3) or anchor_dt.year)
            return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)

    return None
