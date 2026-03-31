from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from src.core.config import settings
from src.core.logger import get_logger
from src.data.market_discovery import MarketInfo
from src.execution.reward_poster_adapter import RewardPosterAdapter, RewardQuoteState
from src.rewards.reward_selector import RewardSelector
from src.rewards.reward_shadow_metrics import build_shadow_extra_payload


_FLATTEN_DELAY_MS = 30_000
_TERMINAL_QUOTE_STATES = {"REJECTED", "FILLED", "CANCELLED"}


log = get_logger(__name__)


@dataclass(slots=True)
class RewardInventory:
    market_id: str
    asset_id: str
    side: str
    filled_size: Decimal = Decimal("0")
    filled_notional: Decimal = Decimal("0")
    flatten_due_ms: int = 0
    flatten_escalated: bool = False


class RewardPosterSidecar:
    def __init__(
        self,
        *,
        orchestrator: Any,
        selector: RewardSelector,
        markets_provider: Callable[[], list[MarketInfo]],
        market_by_asset_provider: Callable[[str], MarketInfo | None],
        book_provider: Callable[[str], Any | None],
        health_report_provider: Callable[[int], Any],
        maker_monitor: Any | None = None,
        now_ms: Callable[[], int] | None = None,
        shadow_persist_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._adapter: RewardPosterAdapter = orchestrator.reward_poster_adapter
        self._selector = selector
        self._markets_provider = markets_provider
        self._market_by_asset_provider = market_by_asset_provider
        self._book_provider = book_provider
        self._health_report_provider = health_report_provider
        self._maker_monitor = maker_monitor
        self._now_ms = now_ms or (lambda: 0)
        self._shadow_persist_callback = shadow_persist_callback

        self._admitted_markets: dict[str, MarketInfo] = {}
        self._working_quotes: dict[str, RewardQuoteState] = {}
        self._quotes_by_order_id: dict[str, str] = {}
        self._inventory: dict[str, RewardInventory] = {}
        self._active_shadow_trade_ids: dict[str, str] = {}
        self._quote_opened_ms: dict[str, int] = {}
        self._persisted_shadow_trade_ids: set[str] = set()
        self._mid_history: dict[str, deque[tuple[int, Decimal]]] = defaultdict(lambda: deque(maxlen=16))
        self._last_refresh_ms: int = 0

    @property
    def working_quotes(self) -> dict[str, RewardQuoteState]:
        return dict(self._working_quotes)

    @property
    def inventory(self) -> dict[str, RewardInventory]:
        return dict(self._inventory)

    def replace_market_universe(self, markets: list[MarketInfo], current_timestamp_ms: int) -> None:
        del markets
        self._refresh_admitted_markets(current_timestamp_ms, force=True)

    def on_book_update(self, asset_id: str, *, current_timestamp_ms: int | None = None) -> None:
        timestamp_ms = self._resolve_timestamp(current_timestamp_ms)
        market = self._market_by_asset_provider(asset_id)
        if market is None:
            return
        self._record_mid(asset_id, timestamp_ms)
        if self._refresh_due(timestamp_ms):
            self._refresh_admitted_markets(timestamp_ms)
        self._reconcile_market_quotes(market, timestamp_ms)

    def on_tick(self, current_timestamp_ms: int | None = None) -> None:
        timestamp_ms = self._resolve_timestamp(current_timestamp_ms)
        if self._refresh_due(timestamp_ms):
            self._refresh_admitted_markets(timestamp_ms)
        self._sync_working_quotes(timestamp_ms)
        self._cancel_stale_or_unsafe_quotes(timestamp_ms)
        self._handle_flatten_timers(timestamp_ms)
        for market in list(self._admitted_markets.values()):
            self._reconcile_market_quotes(market, timestamp_ms)

    def on_fill(self, order: Any, *, current_timestamp_ms: int | None = None) -> bool:
        order_id = str(getattr(order, "order_id", "") or "").strip()
        if not order_id:
            return False
        quote_id = self._quotes_by_order_id.get(order_id)
        if quote_id is None:
            return False
        quote = self._working_quotes.get(quote_id)
        if quote is None:
            return False

        timestamp_ms = self._resolve_timestamp(current_timestamp_ms)
        filled_size = Decimal(str(getattr(order, "filled_size", 0.0) or 0.0))
        filled_price_raw = getattr(order, "filled_avg_price", 0.0) or 0.0
        filled_price = Decimal(str(filled_price_raw)) if filled_price_raw else quote.target_price
        remaining_size = max(quote.target_size - filled_size, Decimal("0"))
        status = "FILLED" if remaining_size <= Decimal("0") else "PARTIAL"
        updated = RewardQuoteState(
            quote_id=quote.quote_id,
            market_id=quote.market_id,
            asset_id=quote.asset_id,
            side=quote.side,
            target_price=quote.target_price,
            target_size=quote.target_size,
            max_capital=quote.max_capital,
            status=status,
            order_id=quote.order_id,
            filled_size=filled_size,
            remaining_size=remaining_size,
            filled_price=filled_price,
            guard_reason=quote.guard_reason,
            last_update_ms=timestamp_ms,
            extra_payload=quote.extra_payload,
        )
        self._store_quote(updated, quote_reason="FILLED" if status == "FILLED" else None)
        self._apply_fill_side_effects(updated, timestamp_ms)
        return True

    def _resolve_timestamp(self, current_timestamp_ms: int | None) -> int:
        return int(self._now_ms() if current_timestamp_ms is None else current_timestamp_ms)

    def _refresh_due(self, timestamp_ms: int) -> bool:
        return timestamp_ms - self._last_refresh_ms >= int(settings.strategy.reward_refresh_interval_ms)

    def _refresh_admitted_markets(self, timestamp_ms: int, *, force: bool = False) -> None:
        if not force and not self._refresh_due(timestamp_ms):
            return
        candidates = self._selector.static_candidates(self._markets_provider(), timestamp_ms)
        admitted = {market.condition_id: market for market in candidates[: max(settings.strategy.reward_market_cap, 0)]}
        removed_market_ids = set(self._admitted_markets).difference(admitted)
        previous_market_ids = set(self._admitted_markets)
        self._admitted_markets = admitted
        self._last_refresh_ms = timestamp_ms
        if force or previous_market_ids != set(admitted):
            log.info(
                "reward_sidecar_universe_refreshed",
                candidate_count=len(candidates),
                admitted_count=len(admitted),
                removed_count=len(removed_market_ids),
                admitted_market_ids=list(admitted)[:8],
            )
        for quote in list(self._working_quotes.values()):
            if quote.market_id in removed_market_ids and quote.status in {"WORKING", "PARTIAL"}:
                self._store_quote(self._adapter.cancel_quote(quote, timestamp_ms), quote_reason="MARKET_REMOVED")

    def _sync_working_quotes(self, timestamp_ms: int) -> None:
        for quote in list(self._working_quotes.values()):
            if quote.status not in {"WORKING", "PARTIAL"}:
                continue
            updated = self._adapter.sync_quote(quote, timestamp_ms)
            quote_reason = None
            if updated.status == "FILLED":
                quote_reason = "FILLED"
            elif updated.status == "CANCELLED":
                quote_reason = "VENUE_CANCELLED"
            self._store_quote(updated, quote_reason=quote_reason)
            if updated.status in {"FILLED", "PARTIAL"} and updated != quote:
                self._apply_fill_side_effects(updated, timestamp_ms)

    def _cancel_stale_or_unsafe_quotes(self, timestamp_ms: int) -> None:
        health_report = self._health_report_provider(timestamp_ms)
        for quote in list(self._working_quotes.values()):
            if quote.status not in {"WORKING", "PARTIAL"}:
                continue
            if timestamp_ms - quote.last_update_ms > settings.strategy.reward_cancel_on_stale_ms:
                self._store_quote(self._adapter.cancel_quote(quote, timestamp_ms), quote_reason="STALE_BOOK")
                continue
            if health_report is None or health_report.orchestrator_health != "GREEN":
                self._store_quote(self._adapter.cancel_quote(quote, timestamp_ms), quote_reason="HEALTH_NOT_GREEN")

    def _handle_flatten_timers(self, timestamp_ms: int) -> None:
        for inventory in self._inventory.values():
            if inventory.filled_notional <= Decimal("0") or inventory.flatten_escalated:
                continue
            if inventory.flatten_due_ms > 0 and timestamp_ms >= inventory.flatten_due_ms:
                inventory.flatten_escalated = True

    def _reconcile_market_quotes(self, market: MarketInfo, timestamp_ms: int) -> None:
        if market.condition_id not in self._admitted_markets:
            return
        if self._inventory_blocks_market(market.condition_id):
            return
        for asset_id, side in ((market.yes_token_id, "YES"), (market.no_token_id, "NO")):
            quote_id = self._quote_id(market.condition_id, side)
            existing = self._working_quotes.get(quote_id)
            if existing is not None and existing.status not in {"WORKING", "PARTIAL"}:
                existing = None

            selection = self._selector.select_intent(
                market,
                asset_id=asset_id,
                side=side,
                quote_id=quote_id,
                book=self._book_provider(asset_id),
                current_timestamp_ms=timestamp_ms,
                health_report=self._health_report_provider(timestamp_ms),
                maker_monitor=self._maker_monitor,
                mid_history=tuple(self._mid_history.get(asset_id, ())),
            )
            if not selection.admitted or selection.intent is None:
                if existing is not None:
                    self._store_quote(
                        self._adapter.cancel_quote(existing, timestamp_ms),
                        quote_reason=selection.reason or "SELECTION_REJECTED",
                    )
                continue

            if existing is not None and not self._should_replace(existing, selection.intent):
                continue
            if existing is None and not self._can_open_new_quote():
                continue
            if not self._quote_notional_room_allows(selection.intent.max_capital):
                continue

            if existing is not None:
                self._store_quote(self._adapter.cancel_quote(existing, timestamp_ms), quote_reason="REPRICED")
            quote_state = self._adapter.submit_intent(selection.intent, timestamp_ms)
            quote_reason = None
            if quote_state.status == "REJECTED":
                quote_reason = quote_state.guard_reason or "REJECTED"
            self._store_quote(quote_state, quote_reason=quote_reason)

    def _record_mid(self, asset_id: str, timestamp_ms: int) -> None:
        book = self._book_provider(asset_id)
        if book is None or not callable(getattr(book, "snapshot", None)):
            return
        snapshot = book.snapshot()
        mid_value = getattr(snapshot, "mid_price", 0.0) or 0.0
        mid_decimal = Decimal(str(mid_value))
        if mid_decimal <= Decimal("0"):
            return
        self._mid_history[asset_id].append((timestamp_ms, mid_decimal))

    def _can_open_new_quote(self) -> bool:
        working_count = sum(1 for quote in self._working_quotes.values() if quote.status in {"WORKING", "PARTIAL"})
        return working_count < max(settings.strategy.reward_quote_cap, 0)

    def _quote_notional_room_allows(self, next_notional: Decimal) -> bool:
        open_notional = sum(
            quote.target_price * quote.remaining_size
            for quote in self._working_quotes.values()
            if quote.status in {"WORKING", "PARTIAL"}
        )
        return open_notional + next_notional <= Decimal(str(settings.strategy.reward_quote_notional_cap))

    def _inventory_blocks_market(self, market_id: str) -> bool:
        return any(inventory.market_id == market_id and inventory.filled_notional > Decimal("0") for inventory in self._inventory.values())

    def _apply_fill_side_effects(self, quote: RewardQuoteState, timestamp_ms: int) -> None:
        sibling_id = self._quote_id(quote.market_id, "NO" if quote.side == "YES" else "YES")
        sibling = self._working_quotes.get(sibling_id)
        if sibling is not None and sibling.status in {"WORKING", "PARTIAL"}:
            self._store_quote(self._adapter.cancel_quote(sibling, timestamp_ms), quote_reason="ONE_SIDED_FILL")

        filled_size = quote.filled_size if quote.filled_size > Decimal("0") else max(quote.target_size - quote.remaining_size, Decimal("0"))
        filled_price = quote.filled_price if quote.filled_price is not None else quote.target_price
        inventory_key = f"{quote.market_id}:{quote.side}"
        inventory = self._inventory.get(inventory_key)
        if inventory is None:
            inventory = RewardInventory(market_id=quote.market_id, asset_id=quote.asset_id, side=quote.side)
            self._inventory[inventory_key] = inventory
        inventory.filled_size += filled_size
        inventory.filled_notional += filled_size * filled_price
        inventory.flatten_due_ms = timestamp_ms + _FLATTEN_DELAY_MS
        if self._total_inventory_notional() >= Decimal(str(settings.strategy.reward_inventory_cap)):
            inventory.flatten_due_ms = timestamp_ms

    def _store_quote(self, quote_state: RewardQuoteState, *, quote_reason: str | None = None) -> None:
        previous = self._working_quotes.get(quote_state.quote_id)
        was_active = previous is not None and previous.status in {"WORKING", "PARTIAL"}
        if quote_state.status in {"WORKING", "PARTIAL"} and not was_active:
            opened_ms = int(quote_state.last_update_ms)
            self._quote_opened_ms[quote_state.quote_id] = opened_ms
            self._active_shadow_trade_ids[quote_state.quote_id] = self._build_shadow_trade_id(quote_state, opened_ms)

        self._working_quotes[quote_state.quote_id] = quote_state
        if quote_state.order_id is not None:
            self._quotes_by_order_id[quote_state.order_id] = quote_state.quote_id

        if quote_state.status in _TERMINAL_QUOTE_STATES:
            self._persist_shadow_trade(quote_state, quote_reason=quote_reason)

    def _persist_shadow_trade(self, quote_state: RewardQuoteState, *, quote_reason: str | None) -> None:
        if self._shadow_persist_callback is None:
            return

        opened_ms = self._quote_opened_ms.pop(quote_state.quote_id, int(quote_state.last_update_ms))
        trade_id = self._active_shadow_trade_ids.pop(
            quote_state.quote_id,
            self._build_shadow_trade_id(quote_state, opened_ms),
        )
        if trade_id in self._persisted_shadow_trade_ids:
            return
        self._persisted_shadow_trade_ids.add(trade_id)

        metadata = dict(quote_state.extra_payload or {})
        detail_payload = metadata.get("extra_payload")
        detail_payload = dict(detail_payload) if isinstance(detail_payload, dict) else {}
        reference_price = self._decimal_from_metadata(metadata, "reference_mid_price", quote_state.target_price)
        fill_occurred = quote_state.filled_size > Decimal("0")
        fill_price = quote_state.filled_price if quote_state.filled_price is not None else quote_state.target_price
        entry_price = fill_price if fill_occurred else quote_state.target_price
        exit_price = reference_price if fill_occurred else quote_state.target_price
        realized_size = quote_state.filled_size if fill_occurred else quote_state.target_size
        pnl_cents = float((exit_price - entry_price) * realized_size * Decimal("100")) if fill_occurred else 0.0
        inventory = self._inventory.get(f"{quote_state.market_id}:{quote_state.side}")
        emergency_flatten = bool(
            inventory is not None
            and inventory.filled_notional > Decimal("0")
            and (inventory.flatten_escalated or inventory.flatten_due_ms <= int(quote_state.last_update_ms))
        )
        terminal_reason = quote_reason or quote_state.guard_reason or quote_state.status
        extra_payload = build_shadow_extra_payload(
            reward_daily_usd=self._float_from_metadata(metadata, "reward_daily_rate_usd"),
            reward_min_size=float(quote_state.target_size),
            reward_max_spread_cents=self._float_from_metadata(metadata, "reward_max_spread_cents"),
            reward_to_competition=self._float_from_metadata(metadata, "reward_to_competition"),
            queue_depth_ahead_usd=self._float_from_mapping(detail_payload, "bid_depth_usd"),
            quoted_at=opened_ms / 1000.0,
            terminal_at=quote_state.last_update_ms / 1000.0,
            fill_occurred=fill_occurred,
            fill_latency_ms=(quote_state.last_update_ms - opened_ms) if fill_occurred else None,
            reference_price=float(reference_price),
            direction=quote_state.side,
            quote_size_usd=float(quote_state.max_capital),
            quote_id=quote_state.quote_id,
            quote_reason=terminal_reason,
            emergency_flatten=emergency_flatten,
        )
        self._shadow_persist_callback(
            {
                "trade_id": trade_id,
                "signal_source": self._shadow_signal_source(quote_state),
                "market_id": quote_state.market_id,
                "asset_id": quote_state.asset_id,
                "direction": quote_state.side,
                "reference_price": float(reference_price),
                "reference_price_band": "reward_sidecar",
                "entry_price": float(entry_price),
                "entry_size": float(realized_size),
                "entry_time": opened_ms / 1000.0,
                "target_price": float(reference_price),
                "stop_price": float(quote_state.target_price),
                "exit_price": float(exit_price),
                "exit_time": quote_state.last_update_ms / 1000.0,
                "exit_reason": terminal_reason,
                "pnl_cents": pnl_cents,
                "entry_fee_bps": 0,
                "exit_fee_bps": 0,
                "extra_payload": extra_payload,
            }
        )

    def _should_replace(self, existing: RewardQuoteState, intent: Any) -> bool:
        tick_move = abs(intent.target_price - existing.target_price) / Decimal("0.01")
        return int(tick_move) >= settings.strategy.reward_replace_only_if_price_moves_ticks

    def _total_inventory_notional(self) -> Decimal:
        return sum(inventory.filled_notional for inventory in self._inventory.values())

    @staticmethod
    def _build_shadow_trade_id(quote_state: RewardQuoteState, opened_ms: int) -> str:
        order_key = str(quote_state.order_id or "").strip()
        if order_key:
            return f"REWARD-SHADOW:{order_key}"
        return f"REWARD-SHADOW:{quote_state.quote_id}:{opened_ms}"

    @staticmethod
    def _decimal_from_metadata(metadata: dict[str, object], key: str, fallback: Decimal) -> Decimal:
        raw = metadata.get(key)
        if raw in (None, ""):
            return fallback
        try:
            return Decimal(str(raw))
        except Exception:
            return fallback

    @staticmethod
    def _float_from_metadata(metadata: dict[str, object], key: str) -> float | None:
        raw = metadata.get(key)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except Exception:
            return None

    @staticmethod
    def _float_from_mapping(payload: dict[str, object], key: str) -> float | None:
        raw = payload.get(key)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except Exception:
            return None

    @staticmethod
    def _shadow_signal_source(quote_state: RewardQuoteState) -> str:
        if Decimal("0") < quote_state.filled_size < quote_state.target_size:
            return "REWARD_PARTIAL"
        if quote_state.filled_size >= quote_state.target_size and quote_state.target_size > Decimal("0"):
            return "REWARD_FILLED"
        return "REWARD_SHADOW"

    @staticmethod
    def _quote_id(market_id: str, side: str) -> str:
        return f"REWARD:{market_id}:{side}"