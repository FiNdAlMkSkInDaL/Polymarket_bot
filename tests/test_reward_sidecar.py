from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

from src.data.market_discovery import MarketInfo
from src.execution.orchestrator_health_monitor import HealthReport
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import DispatchIntentDecision, DispatchReceipt
from src.execution.reward_poster_adapter import RewardPosterAdapter, RewardQuoteState
from src.rewards.models import RewardPosterIntent
from src.rewards.reward_poster_sidecar import RewardPosterSidecar
from src.rewards.reward_selector import RewardSelector


class _StubBook:
    def __init__(
        self,
        *,
        best_bid: float = 0.47,
        best_ask: float = 0.49,
        mid_price: float = 0.48,
        spread: float = 0.02,
        timestamp: float = 1.0,
        fresh: bool = True,
    ) -> None:
        self._snapshot = SimpleNamespace(
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            bid_depth_usd=100.0,
            ask_depth_usd=100.0,
            timestamp=timestamp,
            fresh=fresh,
        )

    def snapshot(self):
        return self._snapshot


class _StubMakerMonitor:
    def __init__(self, *, allowed: bool = True) -> None:
        self._allowed = allowed

    def is_maker_allowed(self, market_id: str) -> bool:
        _ = market_id
        return self._allowed


class _StubDispatcher:
    def __init__(self) -> None:
        self.venue_adapter = SimpleNamespace(
            get_order_status=lambda order_id: SimpleNamespace(
                client_order_id=order_id,
                venue_order_id=f"venue-{order_id}",
                fill_status="OPEN",
                filled_size=Decimal("0"),
                remaining_size=Decimal("5"),
                average_fill_price=None,
            ),
            cancel_order=lambda order_id, market_id: SimpleNamespace(
                client_order_id=order_id,
                cancelled=True,
                rejection_reason=None,
                venue_timestamp_ms=1001,
            ),
        )

    def evaluate_intent(self, context: PriorityOrderContext, dispatch_timestamp_ms: int, *, enforce_guard: bool | None = None) -> DispatchIntentDecision:
        _ = (context, dispatch_timestamp_ms, enforce_guard)
        return DispatchIntentDecision(allowed=True)

    def dispatch(self, context: PriorityOrderContext, dispatch_timestamp_ms: int, *, enforce_guard: bool | None = None) -> DispatchReceipt:
        _ = enforce_guard
        return DispatchReceipt(
            context=context,
            mode="live",
            executed=True,
            fill_price=None,
            fill_size=None,
            serialized_envelope="{}",
            dispatch_timestamp_ms=dispatch_timestamp_ms,
            fill_status="NONE",
            order_id=f"order-{context.signal_metadata['quote_id']}",
            execution_id=f"exec-{context.signal_metadata['quote_id']}",
            remaining_size=context.anchor_volume,
            venue_timestamp_ms=dispatch_timestamp_ms,
            latency_ms=2,
        )


class _RejectingDispatcher(_StubDispatcher):
    def evaluate_intent(self, context: PriorityOrderContext, dispatch_timestamp_ms: int, *, enforce_guard: bool | None = None) -> DispatchIntentDecision:
        _ = (context, dispatch_timestamp_ms, enforce_guard)
        return DispatchIntentDecision(allowed=False, reason="DEGRADED_RISK_ENTRY_BLOCKED")


class _StubRewardAdapter:
    def __init__(self) -> None:
        self.submit_calls: list[RewardPosterIntent] = []
        self.cancel_calls: list[str] = []
        self.sync_calls: list[str] = []

    def submit_intent(self, intent: RewardPosterIntent, timestamp_ms: int) -> RewardQuoteState:
        self.submit_calls.append(intent)
        return RewardQuoteState(
            quote_id=intent.quote_id,
            market_id=intent.market_id,
            asset_id=intent.asset_id,
            side=intent.side,
            target_price=intent.target_price,
            target_size=intent.target_size,
            max_capital=intent.max_capital,
            status="WORKING",
            order_id=f"order-{intent.quote_id}",
            remaining_size=intent.target_size,
            last_update_ms=timestamp_ms,
            extra_payload=intent.as_signal_metadata(),
        )

    def sync_quote(self, state: RewardQuoteState, timestamp_ms: int) -> RewardQuoteState:
        _ = timestamp_ms
        self.sync_calls.append(state.quote_id)
        return state

    def cancel_quote(self, state: RewardQuoteState, timestamp_ms: int) -> RewardQuoteState:
        self.cancel_calls.append(state.quote_id)
        return replace(state, status="CANCELLED", remaining_size=Decimal("0"), last_update_ms=timestamp_ms)

    def replace_quote(self, state: RewardQuoteState, intent: RewardPosterIntent, timestamp_ms: int) -> RewardQuoteState:
        self.cancel_calls.append(state.quote_id)
        return self.submit_intent(intent, timestamp_ms)


class _StubOrchestrator:
    def __init__(self, adapter: _StubRewardAdapter) -> None:
        self.reward_poster_adapter = adapter


def _health_report(state: str = "GREEN") -> HealthReport:
    return HealthReport(
        timestamp_ms=1000,
        orchestrator_health=state,
        is_safe_to_trade=(state == "GREEN"),
        consecutive_release_failures=0,
        last_snapshot_age_ms=0,
        heartbeat_ok=True,
        halt_reason=None if state == "GREEN" else "DEGRADED",
    )


def _market() -> MarketInfo:
    return MarketInfo(
        condition_id="mkt-a",
        question="Will BTC stay in range next week?",
        yes_token_id="yes-a",
        no_token_id="no-a",
        daily_volume_usd=50_000.0,
        end_date=datetime.now(timezone.utc) + timedelta(days=7),
        active=True,
        event_id="evt-a",
        liquidity_usd=20_000.0,
        accepting_orders=True,
        tags="crypto range",
        neg_risk=False,
        reward_program_active=True,
        reward_daily_rate_usd=40.0,
        reward_min_size=5.0,
        reward_max_spread_cents=3.0,
        reward_competition_score=4.0,
    )


def _intent(*, quote_id: str = "quote-a", side: str = "YES") -> RewardPosterIntent:
    return RewardPosterIntent(
        market_id="mkt-a",
        asset_id="yes-a" if side == "YES" else "no-a",
        side=side,
        reference_mid_price=Decimal("0.48"),
        target_price=Decimal("0.48"),
        target_size=Decimal("5"),
        max_capital=Decimal("2.4000"),
        quote_id=quote_id,
        reward_program="mid_tier_reward_v1",
        reward_daily_rate_usd=Decimal("40"),
        reward_to_competition=Decimal("10"),
        competition_score=Decimal("4"),
        reward_max_spread_cents=Decimal("3"),
        cancel_on_stale_ms=15_000,
        replace_only_if_price_moves_ticks=1,
    )


def test_reward_poster_intent_builds_reward_priority_context() -> None:
    intent = _intent()

    context = intent.to_priority_context()

    assert context.signal_source == "REWARD"
    assert context.execution_hints is not None
    assert context.execution_hints.quote_id == "quote-a"
    assert context.max_capital == Decimal("2.4000")


def test_reward_selector_admits_green_fresh_reward_market() -> None:
    selector = RewardSelector()
    market = _market()
    book = _StubBook(timestamp=1.0)

    result = selector.select_intent(
        market,
        asset_id=market.yes_token_id,
        side="YES",
        quote_id="quote-a",
        book=book,
        current_timestamp_ms=1_500,
        health_report=_health_report("GREEN"),
        maker_monitor=_StubMakerMonitor(allowed=True),
        mid_history=((1_000, Decimal("0.48")), (1_500, Decimal("0.481"))),
    )

    assert result.admitted is True
    assert result.intent is not None
    assert result.intent.side == "YES"


def test_reward_selector_rejects_recent_jump_risk() -> None:
    selector = RewardSelector()
    market = _market()
    book = _StubBook(timestamp=1.0)

    result = selector.select_intent(
        market,
        asset_id=market.yes_token_id,
        side="YES",
        quote_id="quote-a",
        book=book,
        current_timestamp_ms=1_500,
        health_report=_health_report("GREEN"),
        maker_monitor=_StubMakerMonitor(allowed=True),
        mid_history=((1_000, Decimal("0.40")), (1_500, Decimal("0.50"))),
    )

    assert result.admitted is False
    assert result.reason == "RECENT_MOVE_VETO"


def test_reward_poster_adapter_maps_live_working_quote() -> None:
    adapter = RewardPosterAdapter(_StubDispatcher())

    state = adapter.submit_intent(_intent(), 1_000)

    assert state.status == "WORKING"
    assert state.order_id == "order-quote-a"
    assert state.remaining_size == Decimal("5")


def test_reward_poster_adapter_rejects_guard_blocked_intent() -> None:
    adapter = RewardPosterAdapter(_RejectingDispatcher())

    state = adapter.submit_intent(_intent(), 1_000)

    assert state.status == "REJECTED"
    assert state.guard_reason == "DEGRADED_RISK_ENTRY_BLOCKED"


def test_reward_sidecar_cancels_stale_quotes() -> None:
    adapter = _StubRewardAdapter()
    persisted: list[dict[str, object]] = []
    sidecar = RewardPosterSidecar(
        orchestrator=_StubOrchestrator(adapter),
        selector=RewardSelector(),
        markets_provider=lambda: [_market()],
        market_by_asset_provider=lambda asset_id: _market() if asset_id in {"yes-a", "no-a"} else None,
        book_provider=lambda asset_id: _StubBook(timestamp=1.0),
        health_report_provider=lambda current_timestamp_ms: _health_report("GREEN"),
        maker_monitor=_StubMakerMonitor(allowed=True),
        now_ms=lambda: 20_000,
        shadow_persist_callback=persisted.append,
    )
    quote = adapter.submit_intent(_intent(), 1_000)
    sidecar._store_quote(quote)

    sidecar.on_tick(20_000)

    assert "quote-a" in adapter.cancel_calls
    assert persisted[0]["signal_source"] == "REWARD_SHADOW"
    assert persisted[0]["exit_reason"] == "STALE_BOOK"
    assert persisted[0]["extra_payload"]["fill_occurred"] is False


def test_reward_sidecar_one_sided_fill_cancels_sibling_and_tracks_inventory() -> None:
    adapter = _StubRewardAdapter()
    market = _market()
    persisted: list[dict[str, object]] = []
    sidecar = RewardPosterSidecar(
        orchestrator=_StubOrchestrator(adapter),
        selector=RewardSelector(),
        markets_provider=lambda: [market],
        market_by_asset_provider=lambda asset_id: market if asset_id in {"yes-a", "no-a"} else None,
        book_provider=lambda asset_id: _StubBook(timestamp=1.0),
        health_report_provider=lambda current_timestamp_ms: _health_report("GREEN"),
        maker_monitor=_StubMakerMonitor(allowed=True),
        now_ms=lambda: 2_000,
        shadow_persist_callback=persisted.append,
    )
    yes_quote = adapter.submit_intent(_intent(quote_id="REWARD:mkt-a:YES", side="YES"), 1_000)
    no_quote = adapter.submit_intent(_intent(quote_id="REWARD:mkt-a:NO", side="NO"), 1_000)
    sidecar._store_quote(yes_quote)
    sidecar._store_quote(no_quote)

    filled_order = SimpleNamespace(
        order_id=yes_quote.order_id,
        filled_size=5.0,
        filled_avg_price=0.48,
    )

    handled = sidecar.on_fill(filled_order, current_timestamp_ms=2_000)

    assert handled is True
    assert no_quote.quote_id in adapter.cancel_calls
    assert sidecar.inventory["mkt-a:YES"].filled_notional == Decimal("2.40")
    sibling_records = [payload for payload in persisted if payload["exit_reason"] == "ONE_SIDED_FILL"]
    assert sibling_records[0]["signal_source"] == "REWARD_SHADOW"


def test_reward_sidecar_partial_fill_persists_reward_partial_shadow_record() -> None:
    adapter = _StubRewardAdapter()
    market = _market()
    persisted: list[dict[str, object]] = []
    sidecar = RewardPosterSidecar(
        orchestrator=_StubOrchestrator(adapter),
        selector=RewardSelector(),
        markets_provider=lambda: [market],
        market_by_asset_provider=lambda asset_id: market if asset_id in {"yes-a", "no-a"} else None,
        book_provider=lambda asset_id: _StubBook(timestamp=1.0),
        health_report_provider=lambda current_timestamp_ms: _health_report("GREEN"),
        maker_monitor=_StubMakerMonitor(allowed=True),
        now_ms=lambda: 20_000,
        shadow_persist_callback=persisted.append,
    )
    yes_quote = adapter.submit_intent(_intent(quote_id="REWARD:mkt-a:YES", side="YES"), 1_000)
    no_quote = adapter.submit_intent(_intent(quote_id="REWARD:mkt-a:NO", side="NO"), 1_000)
    sidecar._store_quote(yes_quote)
    sidecar._store_quote(no_quote)

    handled = sidecar.on_fill(
        SimpleNamespace(
            order_id=yes_quote.order_id,
            filled_size=2.0,
            filled_avg_price=0.48,
        ),
        current_timestamp_ms=2_000,
    )

    assert handled is True

    sidecar.on_tick(20_000)

    partial_records = [
        payload for payload in persisted if payload["trade_id"] == "REWARD-SHADOW:order-REWARD:mkt-a:YES"
    ]
    assert partial_records[-1]["signal_source"] == "REWARD_PARTIAL"
    assert partial_records[-1]["extra_payload"]["fill_occurred"] is True
    assert partial_records[-1]["exit_reason"] == "STALE_BOOK"


def test_reward_sidecar_inventory_cap_blocks_new_quotes() -> None:
    adapter = _StubRewardAdapter()
    market = _market()
    sidecar = RewardPosterSidecar(
        orchestrator=_StubOrchestrator(adapter),
        selector=RewardSelector(),
        markets_provider=lambda: [market],
        market_by_asset_provider=lambda asset_id: market if asset_id in {"yes-a", "no-a"} else None,
        book_provider=lambda asset_id: _StubBook(timestamp=1.0),
        health_report_provider=lambda current_timestamp_ms: _health_report("GREEN"),
        maker_monitor=_StubMakerMonitor(allowed=True),
        now_ms=lambda: 2_000,
    )
    sidecar._inventory["mkt-a:YES"] = SimpleNamespace(
        market_id="mkt-a",
        asset_id="yes-a",
        side="YES",
        filled_size=Decimal("5"),
        filled_notional=Decimal("3.0"),
        flatten_due_ms=0,
        flatten_escalated=False,
    )

    sidecar.on_book_update("yes-a", current_timestamp_ms=2_000)

    assert adapter.submit_calls == []