from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import heapq
import importlib
import inspect
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.orderbook import OrderbookTracker
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchIntentDecision, DispatchReceipt, PriorityDispatcher
from src.monitoring.trade_store import TradeStore
from src.signals.base_strategy import BaseStrategy
from src.backtest.multi_market_streamer import iter_multiplexed_market_records


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    timestamp_ms: int
    event_type: str
    market_id: str
    asset_id: str
    payload: dict[str, Any]
    dedupe_key: tuple[Any, ...]
    trade_price: Decimal | None = None
    trade_size: Decimal | None = None
    trade_side: str | None = None


@dataclass(frozen=True, slots=True)
class MarketTokens:
    market_id: str
    yes_asset_id: str
    no_asset_id: str


@dataclass(slots=True)
class SimulatedOrder:
    order_id: str
    context: PriorityOrderContext
    asset_id: str
    price: Decimal
    requested_size: Decimal
    remaining_size: Decimal
    submitted_at_ms: int
    liquidity_mode: str
    quote_side: str = "BID"
    quote_id: str | None = None
    fill_count: int = 0


@dataclass(slots=True)
class ReplaySummary:
    total_events: int = 0
    book_events: int = 0
    trade_events: int = 0
    dispatches: int = 0
    rejections: int = 0
    taker_fills: int = 0
    maker_fills: int = 0
    maker_bid_fills: int = 0
    maker_ask_fills: int = 0
    persisted_shadow_rows: int = 0
    open_orders: int = 0
    cancel_requests: int = 0
    cancelled_orders: int = 0
    forced_day_boundary_cancels: int = 0
    forced_run_end_cancels: int = 0


class MarketCatalog:
    def __init__(self, mappings: Sequence[MarketTokens] = ()) -> None:
        self._markets: dict[str, MarketTokens] = {entry.market_id: entry for entry in mappings}
        self._assets: dict[str, tuple[str, str]] = {}
        for entry in mappings:
            self._assets[entry.yes_asset_id] = (entry.market_id, "YES")
            self._assets[entry.no_asset_id] = (entry.market_id, "NO")

    def register(self, market_id: str, *, yes_asset_id: str, no_asset_id: str) -> None:
        entry = MarketTokens(market_id=market_id, yes_asset_id=yes_asset_id, no_asset_id=no_asset_id)
        self._markets[market_id] = entry
        self._assets[yes_asset_id] = (market_id, "YES")
        self._assets[no_asset_id] = (market_id, "NO")

    def asset_id_for_side(self, market_id: str, side: str) -> str | None:
        entry = self._markets.get(str(market_id))
        if entry is None:
            return str(market_id) if market_id else None
        if str(side).upper() == "YES":
            return entry.yes_asset_id
        if str(side).upper() == "NO":
            return entry.no_asset_id
        return None

    def market_for_asset(self, asset_id: str) -> tuple[str, str] | None:
        return self._assets.get(str(asset_id))

    def has_market(self, market_id: str) -> bool:
        return str(market_id) in self._markets


class SimulatedMatchingEngine:
    def __init__(self, *, market_catalog: MarketCatalog, order_id_prefix: str = "universal-order") -> None:
        self._market_catalog = market_catalog
        self._trackers: dict[str, OrderbookTracker] = {}
        self._working_orders: dict[str, SimulatedOrder] = {}
        self._quote_index: dict[str, str] = {}
        self._now_ms = 0
        self._next_order_id = 0
        self._order_id_prefix = str(order_id_prefix or "universal-order").strip() or "universal-order"
        self._persist_queue: list[dict[str, Any]] = []
        self._rejection_reasons: dict[str, int] = {}
        self._rejection_samples: list[dict[str, Any]] = []
        self.summary = ReplaySummary()

    def set_now(self, timestamp_ms: int) -> None:
        self._now_ms = int(timestamp_ms)

    def tracker(self, asset_id: str) -> OrderbookTracker:
        asset_id = str(asset_id)
        tracker = self._trackers.get(asset_id)
        if tracker is None:
            tracker = OrderbookTracker(asset_id)
            self._trackers[asset_id] = tracker
        return tracker

    def update_book(self, asset_id: str, event_type: str, payload: dict[str, Any]) -> None:
        tracker = self.tracker(asset_id)
        if event_type == "BOOK":
            tracker.on_book_snapshot(payload)
        else:
            tracker.on_price_change(payload)
        self.summary.book_events += 1

    def top_levels(self, asset_id: str, *, levels: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tracker = self.tracker(asset_id)
        bids = [
            {"asset_id": asset_id, "price": level.price, "size": level.size}
            for level in tracker.levels("bid", n=levels)
        ]
        asks = [
            {"asset_id": asset_id, "price": level.price, "size": level.size}
            for level in tracker.levels("ask", n=levels)
        ]
        return bids, asks

    def dispatch(self, context: PriorityOrderContext, dispatch_timestamp_ms: int) -> DispatchReceipt:
        self.summary.dispatches += 1
        self.set_now(dispatch_timestamp_ms)
        metadata = dict(context.signal_metadata)
        if str(metadata.get("action") or "").strip().upper() == "CANCEL_ALL":
            return self._cancel_matching_orders(context, dispatch_timestamp_ms)

        asset_id = self._market_catalog.asset_id_for_side(context.market_id, context.side)
        if not asset_id:
            self.summary.rejections += 1
            return self._rejected_receipt(context, dispatch_timestamp_ms, "UNKNOWN_MARKET_SIDE")

        tracker = self._trackers.get(asset_id)
        if tracker is None or not tracker.has_data:
            self.summary.rejections += 1
            return self._rejected_receipt(context, dispatch_timestamp_ms, "BOOK_UNAVAILABLE")

        order_size = self._context_order_size(context)
        if order_size <= Decimal("0"):
            self.summary.rejections += 1
            return self._rejected_receipt(context, dispatch_timestamp_ms, "SIZE_ZERO")

        liquidity_mode = self._liquidity_mode(context)
        price = context.target_price
        if liquidity_mode == "TAKER":
            return self._execute_taker(context, tracker=tracker, asset_id=asset_id, order_size=order_size, timestamp_ms=dispatch_timestamp_ms)

        quote_side = self._quote_side(context)
        best_bid = Decimal(str(tracker.best_bid or 0.0))
        best_ask = Decimal(str(tracker.best_ask or 0.0))
        if quote_side == "ASK":
            if best_bid > Decimal("0") and price <= best_bid:
                self.summary.rejections += 1
                return self._rejected_receipt(context, dispatch_timestamp_ms, "POST_ONLY_WOULD_CROSS")
        elif best_ask > Decimal("0") and price >= best_ask:
            self.summary.rejections += 1
            return self._rejected_receipt(context, dispatch_timestamp_ms, "POST_ONLY_WOULD_CROSS")

        quote_id = self._quote_id(context)
        if quote_id:
            existing_order_id = self._quote_index.get(quote_id)
            if existing_order_id is not None:
                existing_order = self._working_orders.get(existing_order_id)
                if existing_order is not None:
                    if (
                        existing_order.asset_id == asset_id
                        and existing_order.quote_side == quote_side
                        and existing_order.price == price
                        and existing_order.remaining_size > Decimal("0")
                    ):
                        return DispatchReceipt(
                            context=context,
                            mode="paper",
                            executed=False,
                            fill_price=None,
                            fill_size=None,
                            serialized_envelope="",
                            dispatch_timestamp_ms=dispatch_timestamp_ms,
                            partial_fill_size=None,
                            partial_fill_price=None,
                            fill_status="NONE",
                        )
                    self._remove_order(existing_order_id)

        self._next_order_id += 1
        order = SimulatedOrder(
            order_id=f"{self._order_id_prefix}-{self._next_order_id}",
            context=context,
            asset_id=asset_id,
            price=price,
            requested_size=order_size,
            remaining_size=order_size,
            submitted_at_ms=dispatch_timestamp_ms,
            liquidity_mode=liquidity_mode,
            quote_side=quote_side,
            quote_id=quote_id,
        )
        self._working_orders[order.order_id] = order
        if quote_id:
            self._quote_index[quote_id] = order.order_id
        self.summary.open_orders = len(self._working_orders)
        return DispatchReceipt(
            context=context,
            mode="paper",
            executed=False,
            fill_price=None,
            fill_size=None,
            serialized_envelope="",
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            partial_fill_size=None,
            partial_fill_price=None,
            fill_status="NONE",
        )

    def on_trade(self, event: ReplayEvent) -> None:
        self.summary.trade_events += 1
        if event.trade_price is None or event.trade_size is None or event.trade_size <= Decimal("0"):
            return
        completed: list[str] = []
        for order_id, order in list(self._working_orders.items()):
            if order.asset_id != event.asset_id:
                continue
            trade_side = (event.trade_side or "").upper()
            if order.quote_side == "ASK":
                if trade_side and trade_side != "BUY":
                    continue
                if event.trade_price < order.price:
                    continue
            else:
                if trade_side and trade_side != "SELL":
                    continue
                if event.trade_price > order.price:
                    continue
            fill_size = min(order.remaining_size, event.trade_size)
            if fill_size <= Decimal("0"):
                continue
            order.fill_count += 1
            order.remaining_size -= fill_size
            self.summary.maker_fills += 1
            if order.quote_side == "ASK":
                self.summary.maker_ask_fills += 1
            else:
                self.summary.maker_bid_fills += 1
            self._queue_persist(
                order=order,
                fill_price=event.trade_price,
                fill_size=fill_size,
                timestamp_ms=event.timestamp_ms,
                exit_reason="MAKER_FILL",
                liquidity_mode="MAKER",
            )
            if order.remaining_size <= Decimal("0"):
                completed.append(order_id)
        for order_id in completed:
            self._remove_order(order_id)
        self.summary.open_orders = len(self._working_orders)

    def drain_persist_queue(self) -> list[dict[str, Any]]:
        queued = self._persist_queue
        self._persist_queue = []
        return queued

    def rejection_reason_counts(self) -> dict[str, int]:
        return dict(self._rejection_reasons)

    def rejection_samples(self) -> list[dict[str, Any]]:
        return list(self._rejection_samples)

    def _execute_taker(
        self,
        context: PriorityOrderContext,
        *,
        tracker: OrderbookTracker,
        asset_id: str,
        order_size: Decimal,
        timestamp_ms: int,
    ) -> DispatchReceipt:
        if str(context.side).upper() == "NO":
            bids = tracker.levels("bid", n=1)
            if not bids:
                self.summary.rejections += 1
                return self._rejected_receipt(context, timestamp_ms, "BID_LIQUIDITY_UNAVAILABLE")
            top_bid = bids[0]
            fill_price = Decimal("1") - Decimal(str(top_bid.price))
            available_size = Decimal(str(top_bid.size))
        else:
            asks = tracker.levels("ask", n=1)
            if not asks:
                self.summary.rejections += 1
                return self._rejected_receipt(context, timestamp_ms, "ASK_LIQUIDITY_UNAVAILABLE")
            top_ask = asks[0]
            fill_price = Decimal(str(top_ask.price))
            available_size = Decimal(str(top_ask.size))

        if fill_price <= Decimal("0"):
            self.summary.rejections += 1
            return self._rejected_receipt(context, timestamp_ms, "INVALID_FILL_PRICE")
        fill_size = min(order_size, available_size)
        if fill_size <= Decimal("0"):
            self.summary.rejections += 1
            return self._rejected_receipt(context, timestamp_ms, "ASK_LIQUIDITY_UNAVAILABLE")

        self._next_order_id += 1
        order = SimulatedOrder(
            order_id=f"{self._order_id_prefix}-{self._next_order_id}",
            context=context,
            asset_id=asset_id,
            price=context.target_price,
            requested_size=order_size,
            remaining_size=max(order_size - fill_size, Decimal("0")),
            submitted_at_ms=timestamp_ms,
            liquidity_mode="TAKER",
            fill_count=1,
        )
        self.summary.taker_fills += 1
        self._queue_persist(
            order=order,
            fill_price=fill_price,
            fill_size=fill_size,
            timestamp_ms=timestamp_ms,
            exit_reason="TAKER_FILL",
            liquidity_mode="TAKER",
        )

        if fill_size == order_size:
            return DispatchReceipt(
                context=context,
                mode="paper",
                executed=True,
                fill_price=fill_price,
                fill_size=fill_size,
                serialized_envelope="",
                dispatch_timestamp_ms=timestamp_ms,
                partial_fill_size=None,
                partial_fill_price=None,
                fill_status="FULL",
            )
        return DispatchReceipt(
            context=context,
            mode="paper",
            executed=True,
            fill_price=fill_price,
            fill_size=order_size,
            serialized_envelope="",
            dispatch_timestamp_ms=timestamp_ms,
            partial_fill_size=fill_size,
            partial_fill_price=fill_price,
            fill_status="PARTIAL",
        )

    def _queue_persist(
        self,
        *,
        order: SimulatedOrder,
        fill_price: Decimal,
        fill_size: Decimal,
        timestamp_ms: int,
        exit_reason: str,
        liquidity_mode: str,
    ) -> None:
        fill_timestamp = timestamp_ms / 1000.0
        trade_id = f"{order.order_id}-fill-{order.fill_count}"
        self._persist_queue.append(
            {
                "trade_id": trade_id,
                "signal_source": order.context.signal_source,
                "market_id": order.context.market_id,
                "asset_id": order.asset_id,
                "direction": order.context.side,
                "reference_price": float(order.context.target_price),
                "reference_price_band": f"{liquidity_mode}:{order.quote_side}",
                "entry_price": float(fill_price),
                "entry_size": float(fill_size),
                "entry_time": fill_timestamp,
                "target_price": float(order.context.target_price),
                "stop_price": 0.0,
                "exit_price": float(fill_price),
                "exit_time": fill_timestamp,
                "exit_reason": exit_reason,
                "pnl_cents": 0.0,
                "entry_fee_bps": 0,
                "exit_fee_bps": 0,
                "zscore": float(order.context.conviction_scalar),
                "confidence": float(order.context.conviction_scalar),
                "toxicity_index": 0.0,
                "extra_payload": {
                    "order_id": order.order_id,
                    "quote_id": order.quote_id,
                    "quote_side": order.quote_side,
                    "requested_size": str(order.requested_size),
                    "fill_size": str(fill_size),
                    "remaining_size": str(order.remaining_size),
                    "liquidity_mode": liquidity_mode,
                    "signal_metadata": dict(order.context.signal_metadata),
                },
            }
        )

    @staticmethod
    def _context_order_size(context: PriorityOrderContext) -> Decimal:
        if context.target_price <= Decimal("0"):
            return Decimal("0")
        max_size = context.max_capital / context.target_price
        return min(context.anchor_volume, max_size)

    @staticmethod
    def _liquidity_mode(context: PriorityOrderContext) -> str:
        hints = context.execution_hints
        if hints is not None and hints.post_only:
            return "MAKER"
        metadata = dict(context.signal_metadata)
        liquidity_intent = str(metadata.get("liquidity_intent") or metadata.get("execution_mode") or "").strip().upper()
        if liquidity_intent in {"MAKER", "PASSIVE", "POST_ONLY", "MAKER_REWARD"}:
            return "MAKER"
        if bool(metadata.get("post_only")):
            return "MAKER"
        return "TAKER"

    @staticmethod
    def _quote_side(context: PriorityOrderContext) -> str:
        metadata = dict(context.signal_metadata)
        quote_side = str(metadata.get("quote_side") or metadata.get("maker_side") or "BID").strip().upper()
        if quote_side in {"ASK", "SELL"}:
            return "ASK"
        return "BID"

    @staticmethod
    def _quote_id(context: PriorityOrderContext) -> str | None:
        metadata = dict(context.signal_metadata)
        quote_id = str(metadata.get("quote_id") or "").strip()
        return quote_id or None

    def _cancel_matching_orders(self, context: PriorityOrderContext, dispatch_timestamp_ms: int) -> DispatchReceipt:
        self.summary.cancel_requests += 1
        metadata = dict(context.signal_metadata)
        strategy_name = str(metadata.get("strategy") or "").strip()
        quote_scope = str(metadata.get("quote_scope") or "").strip().upper()
        cancelled = 0
        for order_id, order in list(self._working_orders.items()):
            order_metadata = dict(order.context.signal_metadata)
            if context.market_id and order.context.market_id != context.market_id:
                continue
            if strategy_name and str(order_metadata.get("strategy") or "").strip() != strategy_name:
                continue
            if quote_scope and order.quote_side != quote_scope:
                continue
            self._remove_order(order_id)
            cancelled += 1
        self.summary.cancelled_orders += cancelled
        self.summary.open_orders = len(self._working_orders)
        return DispatchReceipt(
            context=context,
            mode="paper",
            executed=False,
            fill_price=None,
            fill_size=None,
            serialized_envelope="",
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            partial_fill_size=None,
            partial_fill_price=None,
            fill_status="NONE",
        )

    def _remove_order(self, order_id: str) -> None:
        order = self._working_orders.pop(order_id, None)
        if order is None:
            return
        if order.quote_id:
            indexed_order_id = self._quote_index.get(order.quote_id)
            if indexed_order_id == order_id:
                self._quote_index.pop(order.quote_id, None)

    def cancel_all_open_orders(self) -> int:
        cancelled = len(self._working_orders)
        if cancelled == 0:
            return 0
        for order_id in list(self._working_orders):
            self._remove_order(order_id)
        self.summary.open_orders = 0
        return cancelled

    def _record_rejection(self, context: PriorityOrderContext, reason: str) -> None:
        normalized_reason = str(reason or "UNKNOWN").strip() or "UNKNOWN"
        self._rejection_reasons[normalized_reason] = self._rejection_reasons.get(normalized_reason, 0) + 1
        if len(self._rejection_samples) >= 5:
            return
        metadata = dict(context.signal_metadata)
        self._rejection_samples.append(
            {
                "reason": normalized_reason,
                "market_id": context.market_id,
                "side": context.side,
                "target_price": str(context.target_price),
                "quote_side": str(metadata.get("quote_side") or ""),
                "quote_id": str(metadata.get("quote_id") or ""),
            }
        )

    def _rejected_receipt(self, context: PriorityOrderContext, dispatch_timestamp_ms: int, reason: str) -> DispatchReceipt:
        self._record_rejection(context, reason)
        return DispatchReceipt(
            context=context,
            mode="paper",
            executed=False,
            fill_price=None,
            fill_size=None,
            serialized_envelope="",
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            guard_reason=reason,
            partial_fill_size=None,
            partial_fill_price=None,
            fill_status="NONE",
        )


class MockPriorityDispatcher(PriorityDispatcher):
    def __init__(self, engine: SimulatedMatchingEngine):
        self._engine = engine

    def dispatch(
        self,
        context: PriorityOrderContext,
        dispatch_timestamp_ms: int,
        *,
        enforce_guard: bool | None = None,
    ) -> DispatchReceipt:
        del enforce_guard
        return self._engine.dispatch(context, dispatch_timestamp_ms)

    def evaluate_intent(
        self,
        context: PriorityOrderContext,
        dispatch_timestamp_ms: int,
        *,
        enforce_guard: bool | None = None,
    ) -> DispatchIntentDecision:
        del dispatch_timestamp_ms, enforce_guard
        return DispatchIntentDecision(allowed=self._engine._market_catalog.asset_id_for_side(context.market_id, context.side) is not None)

    def record_external_dispatch(
        self,
        context: PriorityOrderContext,
        dispatch_timestamp_ms: int,
        *,
        enforce_guard: bool | None = None,
    ) -> None:
        del context, dispatch_timestamp_ms, enforce_guard


class UniversalReplayEngine:
    def __init__(
        self,
        *,
        input_dir: Path,
        db_path: Path,
        strategy_path: str,
        market_catalog: MarketCatalog,
        strategy_config: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        max_events: int | None = None,
        secondary_data_path: Path | None = None,
        order_id_prefix: str = "universal-order",
        progress_every_events: int | None = None,
        progress_label: str | None = None,
    ) -> None:
        self._secondary_data_path = secondary_data_path
        self._input_dir = resolve_replay_input(input_dir) if secondary_data_path is None else input_dir
        self._db_path = db_path
        self._market_catalog = market_catalog
        self._max_events = max_events
        self._start_date = start_date
        self._end_date = end_date
        if progress_every_events is not None and progress_every_events <= 0:
            raise ValueError("progress_every_events must be strictly positive when provided")
        self._progress_every_events = progress_every_events
        self._next_progress_event = progress_every_events
        self._progress_label = str(progress_label or strategy_path).strip() or strategy_path
        self._matcher = SimulatedMatchingEngine(market_catalog=market_catalog, order_id_prefix=order_id_prefix)
        self._dispatcher = MockPriorityDispatcher(self._matcher)
        self._trade_store = TradeStore(db_path)
        self._now_ms = 0
        self._strategy = load_strategy(
            strategy_path,
            dispatcher=self._dispatcher,
            market_catalog=market_catalog,
            strategy_config=strategy_config or {},
            clock=lambda: self._now_ms,
        )

    async def run(self) -> ReplaySummary:
        self._emit_run_banner("start")
        active_day: str | None = None
        for event in self._event_stream():
            self._now_ms = event.timestamp_ms
            self._matcher.summary.total_events += 1
            self._matcher.set_now(event.timestamp_ms)
            event_day = datetime.fromtimestamp(event.timestamp_ms / 1000.0, tz=timezone.utc).date().isoformat()
            if active_day is None:
                active_day = event_day
            elif event_day != active_day:
                cancelled = self._matcher.cancel_all_open_orders()
                self._matcher.summary.forced_day_boundary_cancels += cancelled
                active_day = event_day
            strategy_market_id = event.market_id if self._market_catalog.has_market(event.market_id) else event.asset_id

            if event.event_type in {"BOOK", "PRICE_CHANGE"}:
                self._matcher.update_book(event.asset_id, event.event_type, event.payload)
                bids, asks = self._matcher.top_levels(event.asset_id, levels=3)
                self._strategy.on_bbo_update(strategy_market_id, bids, asks)
            elif event.event_type == "TRADE":
                trade_data = {
                    "asset_id": event.asset_id,
                    "market_id": strategy_market_id,
                    "price": float(event.trade_price or 0),
                    "size": float(event.trade_size or 0),
                    "side": event.trade_side or "",
                    "timestamp_ms": event.timestamp_ms,
                }
                self._strategy.on_trade(strategy_market_id, trade_data)
                self._matcher.on_trade(event)

            self._strategy.on_tick()
            await self._flush_persists()
            self._emit_progress(event_day=event_day)

            if self._max_events is not None and self._matcher.summary.total_events >= self._max_events:
                break

        cancelled = self._matcher.cancel_all_open_orders()
        self._matcher.summary.forced_run_end_cancels += cancelled
        await self._flush_persists()
        await self._trade_store.close()
        self._emit_run_banner("complete")
        return self._matcher.summary

    def _emit_progress(self, *, event_day: str) -> None:
        if self._progress_every_events is None or self._next_progress_event is None:
            return
        total_events = self._matcher.summary.total_events
        if total_events < self._next_progress_event:
            return
        print(
            "replay_progress label={label} events={events} book={book} trades={trades} dispatches={dispatches} maker_fills={maker_fills} persisted_shadow_rows={rows} day={day}".format(
                label=self._progress_label,
                events=total_events,
                book=self._matcher.summary.book_events,
                trades=self._matcher.summary.trade_events,
                dispatches=self._matcher.summary.dispatches,
                maker_fills=self._matcher.summary.maker_fills,
                rows=self._matcher.summary.persisted_shadow_rows,
                day=event_day,
            ),
            flush=True,
        )
        self._next_progress_event += self._progress_every_events

    def _emit_run_banner(self, phase: str) -> None:
        if self._progress_every_events is None:
            return
        print(
            "replay_{phase} label={label} input={input_dir} start_date={start_date} end_date={end_date} total_events={events} book={book} trades={trades} dispatches={dispatches} maker_fills={maker_fills} persisted_shadow_rows={rows}".format(
                phase=phase,
                label=self._progress_label,
                input_dir=self._input_dir,
                start_date=self._start_date,
                end_date=self._end_date,
                events=self._matcher.summary.total_events,
                book=self._matcher.summary.book_events,
                trades=self._matcher.summary.trade_events,
                dispatches=self._matcher.summary.dispatches,
                maker_fills=self._matcher.summary.maker_fills,
                rows=self._matcher.summary.persisted_shadow_rows,
            ),
            flush=True,
        )

    def strategy_diagnostics(self) -> dict[str, Any]:
        snapshot = getattr(self._strategy, "diagnostics_snapshot", None)
        if not callable(snapshot):
            return {}
        diagnostics = snapshot()
        return diagnostics if isinstance(diagnostics, dict) else {}

    def dispatch_diagnostics(self) -> dict[str, Any]:
        return {
            "rejection_reasons": self._matcher.rejection_reason_counts(),
            "rejection_samples": self._matcher.rejection_samples(),
        }

    def _event_stream(self) -> Iterator[ReplayEvent]:
        if self._secondary_data_path is None:
            if self._input_dir.is_file():
                return iter_file_events(self._input_dir)
            return iter_replay_events(self._input_dir, start_date=self._start_date, end_date=self._end_date)
        return iter_replay_events_from_raw_records(
            iter_multiplexed_market_records(self._input_dir, self._secondary_data_path)
        )

    async def _flush_persists(self) -> None:
        for payload in self._matcher.drain_persist_queue():
            await self._trade_store.record_shadow_trade(**payload)
            self._matcher.summary.persisted_shadow_rows += 1


def load_strategy(
    strategy_path: str,
    *,
    dispatcher: MockPriorityDispatcher,
    market_catalog: MarketCatalog,
    strategy_config: dict[str, Any],
    clock,
) -> BaseStrategy:
    module_name, class_name = strategy_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)
    if not inspect.isclass(strategy_cls) or not issubclass(strategy_cls, BaseStrategy):
        raise TypeError(f"{strategy_path} must inherit from BaseStrategy")

    init_signature = inspect.signature(strategy_cls)
    init_parameters = init_signature.parameters
    kwargs: dict[str, Any] = {}
    for name, value in (
        ("dispatcher", dispatcher),
        ("market_catalog", market_catalog),
        ("strategy_config", strategy_config),
        ("config", strategy_config),
        ("clock", clock),
    ):
        if name in init_parameters:
            kwargs[name] = value
    for name, value in strategy_config.items():
        if name in init_parameters and name not in kwargs:
            kwargs[name] = value
    strategy = strategy_cls(**kwargs)
    strategy.bind_dispatcher(dispatcher)
    strategy.bind_market_catalog(market_catalog)
    strategy.bind_clock(clock)
    return strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal historical replay runner for dispatcher-driven strategies.")
    parser.add_argument(
        "--input-dir",
        default="logs/local_snapshot/l2_data",
        help="Replay root containing data/raw_ticks or raw_ticks date partitions. When --secondary-data is set, this is the primary raw tick JSONL file.",
    )
    parser.add_argument(
        "--secondary-data",
        default=None,
        help="Optional secondary raw tick JSONL file to multiplex with the primary feed in strict timestamp order.",
    )
    parser.add_argument("--db", default="logs/universal_backtest.db", help="SQLite output path.")
    parser.add_argument("--strategy", required=True, help="Import path to strategy class, e.g. src.signals.obi_scalper.ObiScalper")
    parser.add_argument(
        "--market-map",
        default="data/market_map.json",
        help="Optional market map used to resolve YES/NO token ids for matching.",
    )
    parser.add_argument("--strategy-config", default=None, help="Inline JSON object or path to a JSON file passed to the strategy constructor.")
    parser.add_argument("--start-date", default=None, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on normalized replay events.")
    parser.add_argument("--progress-every-events", type=int, default=None, help="Optional heartbeat interval for long replay runs.")
    return parser.parse_args()


def resolve_replay_input(input_dir: Path) -> Path:
    if input_dir.exists() and input_dir.is_file():
        return input_dir
    candidates = [
        input_dir / "data" / "raw_ticks",
        input_dir / "raw_ticks",
        input_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find replay input under {input_dir}")


def resolve_tick_root(input_dir: Path) -> Path:
    resolved = resolve_replay_input(input_dir)
    if resolved.is_file():
        raise FileNotFoundError(f"Expected raw_ticks directory but got file {resolved}")
    return resolved


def load_market_catalog(market_map_path: Path | None) -> MarketCatalog:
    if market_map_path is None or not market_map_path.exists():
        return MarketCatalog()
    raw_rows = json.loads(market_map_path.read_text(encoding="utf-8"))
    entries: list[MarketTokens] = []
    for row in raw_rows:
        market_id = str(row.get("market_id") or "").strip()
        yes_asset_id = str(row.get("yes_id") or "").strip()
        no_asset_id = str(row.get("no_id") or "").strip()
        if market_id and yes_asset_id and no_asset_id:
            entries.append(MarketTokens(market_id=market_id, yes_asset_id=yes_asset_id, no_asset_id=no_asset_id))
    return MarketCatalog(entries)


def load_strategy_config(raw_config: str | None) -> dict[str, Any]:
    if raw_config is None:
        return {}
    raw_config = str(raw_config).strip()
    if not raw_config:
        return {}
    config_path = Path(raw_config)
    if config_path.exists() and config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        loaded = json.loads(raw_config)
    if not isinstance(loaded, dict):
        raise ValueError("Strategy config must be a JSON object")
    return loaded


def iter_replay_events(
    input_dir: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[ReplayEvent]:
    raw_records = (
        raw_record
        for date_dir in iter_date_dirs(input_dir, start_date=start_date, end_date=end_date)
        for path in sorted(path for path in date_dir.glob("*.jsonl") if path.is_file())
        for raw_record in iter_raw_file_records(path)
    )
    yield from iter_replay_events_from_raw_records(raw_records)


def iter_replay_events_from_raw_records(raw_records: Iterable[dict[str, Any] | tuple[str, dict[str, Any]]]) -> Iterator[ReplayEvent]:
    recent_keys: OrderedDict[tuple[Any, ...], None] = OrderedDict()
    for raw_record in raw_records:
        raw = raw_record[1] if isinstance(raw_record, tuple) else raw_record
        for event in normalize_raw_record(raw):
            if event.dedupe_key in recent_keys:
                continue
            recent_keys[event.dedupe_key] = None
            if len(recent_keys) > 8192:
                recent_keys.popitem(last=False)
            yield event


def iter_date_dirs(input_dir: Path, *, start_date: str | None, end_date: str | None) -> Iterator[Path]:
    lower = start_date or ""
    upper = end_date or "9999-12-31"
    for path in sorted(candidate for candidate in input_dir.iterdir() if candidate.is_dir()):
        if path.name < lower or path.name > upper:
            continue
        yield path


def iter_file_events(path: Path) -> Iterator[ReplayEvent]:
    for raw in iter_raw_file_records(path):
        yield from normalize_raw_record(raw)


def iter_raw_file_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def normalize_raw_record(raw: dict[str, Any]) -> Iterator[ReplayEvent]:
    payload = raw.get("payload") or {}
    if not isinstance(payload, dict):
        return
    event_type = str(payload.get("event_type") or "").strip().lower()
    timestamp_ms = timestamp_ms_from_record(raw, payload)
    if timestamp_ms <= 0:
        return

    if event_type == "book":
        asset_id = str(payload.get("asset_id") or "").strip()
        market_id = str(payload.get("market") or raw.get("asset_id") or "").strip()
        if market_id and asset_id:
            yield ReplayEvent(
                timestamp_ms=timestamp_ms,
                event_type="BOOK",
                market_id=market_id,
                asset_id=asset_id,
                payload=dict(payload),
                dedupe_key=("BOOK", market_id, asset_id, timestamp_ms, str(payload.get("hash") or "")),
            )
        return

    if event_type == "price_change":
        market_id = str(payload.get("market") or raw.get("asset_id") or "").strip()
        price_changes = payload.get("price_changes") or []
        for change in price_changes:
            asset_id = str(change.get("asset_id") or "").strip()
            if not market_id or not asset_id:
                continue
            yield ReplayEvent(
                timestamp_ms=timestamp_ms,
                event_type="PRICE_CHANGE",
                market_id=market_id,
                asset_id=asset_id,
                payload={
                    "asset_id": asset_id,
                    "timestamp": timestamp_ms,
                    "changes": [
                        {
                            "price": change.get("price"),
                            "size": change.get("size"),
                            "side": change.get("side"),
                        }
                    ],
                },
                dedupe_key=(
                    "PRICE_CHANGE",
                    market_id,
                    asset_id,
                    timestamp_ms,
                    str(change.get("hash") or ""),
                    str(change.get("price") or ""),
                    str(change.get("size") or ""),
                ),
            )
        return

    if event_type == "last_trade_price":
        market_id = str(payload.get("market") or raw.get("asset_id") or "").strip()
        asset_id = str(payload.get("asset_id") or "").strip()
        if market_id and asset_id:
            trade_price = to_decimal(payload.get("price"))
            trade_size = to_decimal(payload.get("size"))
            if trade_price is None or trade_size is None:
                return
            yield ReplayEvent(
                timestamp_ms=timestamp_ms,
                event_type="TRADE",
                market_id=market_id,
                asset_id=asset_id,
                payload=dict(payload),
                dedupe_key=(
                    "TRADE",
                    market_id,
                    asset_id,
                    timestamp_ms,
                    str(payload.get("transaction_hash") or payload.get("hash") or ""),
                ),
                trade_price=trade_price,
                trade_size=trade_size,
                trade_side=str(payload.get("side") or "").upper(),
            )


def timestamp_ms_from_record(raw: dict[str, Any], payload: dict[str, Any]) -> int:
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


def to_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


async def _main() -> None:
    args = parse_args()
    market_catalog = load_market_catalog(Path(args.market_map) if args.market_map else None)
    secondary_data_path = Path(args.secondary_data) if args.secondary_data else None
    engine = UniversalReplayEngine(
        input_dir=Path(args.input_dir),
        db_path=Path(args.db),
        strategy_path=args.strategy,
        market_catalog=market_catalog,
        strategy_config=load_strategy_config(args.strategy_config),
        start_date=args.start_date,
        end_date=args.end_date,
        max_events=args.max_events,
        secondary_data_path=secondary_data_path,
        progress_every_events=args.progress_every_events,
        progress_label=args.strategy,
    )
    summary = await engine.run()
    print(
        "events={events} book={book} trades={trades} dispatches={dispatches} rejections={rejections} taker_fills={taker} maker_fills={maker} persisted_shadow_rows={rows} open_orders={open_orders} db={db}".format(
            events=summary.total_events,
            book=summary.book_events,
            trades=summary.trade_events,
            dispatches=summary.dispatches,
            rejections=summary.rejections,
            taker=summary.taker_fills,
            maker=summary.maker_fills,
            rows=summary.persisted_shadow_rows,
            open_orders=summary.open_orders,
            db=Path(args.db),
        )
    )
    diagnostics = engine.strategy_diagnostics()
    if diagnostics:
        print("strategy_diagnostics " + " ".join(f"{key}={value}" for key, value in diagnostics.items()))
        quotes_attempted = int(diagnostics.get("quotes_attempted") or 0)
        if quotes_attempted > 0 and summary.maker_fills == 0:
            dispatch_diagnostics = engine.dispatch_diagnostics()
            rejection_reasons = dispatch_diagnostics.get("rejection_reasons") or {}
            rejection_samples = dispatch_diagnostics.get("rejection_samples") or []
            print(
                "dispatcher_debug rejection_reasons="
                + ",".join(f"{key}:{value}" for key, value in sorted(rejection_reasons.items()))
            )
            for sample in rejection_samples:
                print(
                    "dispatcher_debug_sample reason={reason} market_id={market_id} side={side} quote_side={quote_side} target_price={target_price} quote_id={quote_id}".format(
                        reason=sample.get("reason"),
                        market_id=sample.get("market_id"),
                        side=sample.get("side"),
                        quote_side=sample.get("quote_side"),
                        target_price=sample.get("target_price"),
                        quote_id=sample.get("quote_id"),
                    )
                )


if __name__ == "__main__":
    asyncio.run(_main())