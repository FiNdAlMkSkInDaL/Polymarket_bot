from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from statistics import pstdev
from typing import Any

from src.core.config import settings
from src.data.market_discovery import MarketInfo
from src.execution.orchestrator_health_monitor import HealthReport
from src.rewards.models import RewardPosterIntent


_TICK_SIZE = Decimal("0.01")
_MID_TIER_MIN_DAILY_VOLUME_USD = 10_000.0
_MID_TIER_MAX_DAILY_VOLUME_USD = 250_000.0
_MIN_REWARD_DAYS_TO_RESOLUTION = 2
_MAX_REWARD_DAYS_TO_RESOLUTION = 45
_VENUE_SPREAD_LIMIT_CENTS = 3.0
_JUMP_EVENT_KEYWORDS = (
    "earnings",
    "cpi",
    "inflation",
    "fomc",
    "fed",
    "nfp",
    "payroll",
    "rate decision",
    "debate",
    "election night",
)


def _d(value: object) -> Decimal:
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class RewardBookState:
    asset_id: str
    best_bid: Decimal
    best_ask: Decimal
    mid_price: Decimal
    spread_cents: Decimal
    bid_depth_usd: Decimal
    ask_depth_usd: Decimal
    fresh: bool
    book_age_ms: int


@dataclass(frozen=True, slots=True)
class RewardSelectionResult:
    admitted: bool
    reason: str | None
    intent: RewardPosterIntent | None = None
    reward_to_competition: Decimal = Decimal("0")


class RewardSelector:
    def static_candidates(self, markets: Sequence[MarketInfo], current_timestamp_ms: int) -> list[MarketInfo]:
        _ = current_timestamp_ms
        candidates: list[MarketInfo] = []
        for market in markets:
            if not market.reward_program_active:
                continue
            if market.reward_daily_rate_usd < settings.strategy.reward_daily_reward_floor:
                continue
            if market.daily_volume_usd < _MID_TIER_MIN_DAILY_VOLUME_USD or market.daily_volume_usd > _MID_TIER_MAX_DAILY_VOLUME_USD:
                continue
            days_to_resolution = self._days_to_resolution(market)
            if days_to_resolution < _MIN_REWARD_DAYS_TO_RESOLUTION or days_to_resolution > _MAX_REWARD_DAYS_TO_RESOLUTION:
                continue
            if self._looks_like_jump_event(market):
                continue
            candidates.append(market)

        candidates.sort(
            key=lambda market: (
                -market.reward_daily_rate_usd,
                market.reward_competition_score,
                market.daily_volume_usd,
            )
        )
        return candidates

    def select_intent(
        self,
        market: MarketInfo,
        *,
        asset_id: str,
        side: str,
        quote_id: str,
        book: Any,
        current_timestamp_ms: int,
        health_report: HealthReport | None,
        maker_monitor: Any | None = None,
        mid_history: Sequence[tuple[int, Decimal]] = (),
    ) -> RewardSelectionResult:
        if health_report is None or health_report.orchestrator_health != "GREEN":
            return RewardSelectionResult(False, "HEALTH_NOT_GREEN")

        book_state = self._book_state(book, asset_id=asset_id, current_timestamp_ms=current_timestamp_ms)
        if book_state is None:
            return RewardSelectionResult(False, "BOOK_UNAVAILABLE")
        if not book_state.fresh or book_state.book_age_ms > settings.strategy.reward_cancel_on_stale_ms:
            return RewardSelectionResult(False, "STALE_BOOK")
        if book_state.best_bid <= Decimal("0") or book_state.best_ask <= Decimal("0") or book_state.best_ask <= book_state.best_bid:
            return RewardSelectionResult(False, "ONE_SIDED_BOOK")

        spread_limit = Decimal(str(_VENUE_SPREAD_LIMIT_CENTS))
        reward_limit = _d(market.reward_max_spread_cents) if market.reward_max_spread_cents > 0 else spread_limit
        if book_state.spread_cents > min(spread_limit, reward_limit):
            return RewardSelectionResult(False, "SPREAD_TOO_WIDE")

        min_mid = _d(settings.strategy.reward_safe_mid_min_price)
        max_mid = _d(settings.strategy.reward_safe_mid_max_price)
        if not (min_mid < book_state.mid_price < max_mid):
            return RewardSelectionResult(False, "MID_OUT_OF_BAND")

        if self._recent_move_exceeds_threshold(mid_history):
            return RewardSelectionResult(False, "RECENT_MOVE_VETO")
        if self._volatility_jump_exceeds_threshold(mid_history):
            return RewardSelectionResult(False, "VOLATILITY_JUMP_VETO")

        reward_to_competition = self._reward_to_competition(market)
        if reward_to_competition < _d(settings.strategy.reward_to_competition_floor):
            return RewardSelectionResult(False, "REWARD_TO_COMPETITION_TOO_LOW", reward_to_competition=reward_to_competition)

        if maker_monitor is not None and not maker_monitor.is_maker_allowed(market.condition_id):
            return RewardSelectionResult(False, "MAKER_ADVERSE_SELECTION_VETO", reward_to_competition=reward_to_competition)

        target_price = self._passive_buy_price(book_state.best_bid, book_state.best_ask)
        if target_price <= Decimal("0") or target_price >= book_state.best_ask:
            return RewardSelectionResult(False, "UNSAFE_TARGET_PRICE", reward_to_competition=reward_to_competition)

        reward_min_size = max(_d(market.reward_min_size or 0), Decimal("1"))
        quote_notional_cap = _d(settings.strategy.reward_quote_notional_cap)
        required_notional = (target_price * reward_min_size).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        if required_notional <= Decimal("0") or required_notional > quote_notional_cap:
            return RewardSelectionResult(False, "QUOTE_NOTIONAL_CAP_EXCEEDED", reward_to_competition=reward_to_competition)

        intent = RewardPosterIntent(
            market_id=market.condition_id,
            asset_id=asset_id,
            side=side,
            reference_mid_price=book_state.mid_price,
            target_price=target_price,
            target_size=reward_min_size,
            max_capital=required_notional,
            quote_id=quote_id,
            reward_program="mid_tier_reward_v1",
            reward_daily_rate_usd=_d(market.reward_daily_rate_usd),
            reward_to_competition=reward_to_competition,
            competition_score=_d(max(market.reward_competition_score, 0.0)),
            reward_max_spread_cents=_d(max(market.reward_max_spread_cents, 0.0)),
            cancel_on_stale_ms=settings.strategy.reward_cancel_on_stale_ms,
            replace_only_if_price_moves_ticks=settings.strategy.reward_replace_only_if_price_moves_ticks,
            extra_payload={
                "book_age_ms": book_state.book_age_ms,
                "bid_depth_usd": str(book_state.bid_depth_usd),
                "ask_depth_usd": str(book_state.ask_depth_usd),
                "mid_price": str(book_state.mid_price),
            },
        )
        return RewardSelectionResult(True, None, intent=intent, reward_to_competition=reward_to_competition)

    @staticmethod
    def _days_to_resolution(market: MarketInfo) -> int:
        if market.end_date is None:
            return -1
        return max(0, (market.end_date - datetime.now(timezone.utc)).days)

    @staticmethod
    def _looks_like_jump_event(market: MarketInfo) -> bool:
        haystack = f"{market.question} {market.tags}".lower()
        return any(keyword in haystack for keyword in _JUMP_EVENT_KEYWORDS)

    @staticmethod
    def _book_state(book: Any, *, asset_id: str, current_timestamp_ms: int) -> RewardBookState | None:
        if book is None:
            return None
        snapshot_fn = getattr(book, "snapshot", None)
        if not callable(snapshot_fn):
            return None
        snapshot = snapshot_fn()
        best_bid = _d(getattr(snapshot, "best_bid", 0.0) or 0.0)
        best_ask = _d(getattr(snapshot, "best_ask", 0.0) or 0.0)
        default_mid = ((best_bid + best_ask) / Decimal("2")) if best_bid > 0 and best_ask > 0 else Decimal("0")
        mid_price = _d(getattr(snapshot, "mid_price", default_mid) or default_mid)
        raw_spread = getattr(snapshot, "spread", float(best_ask - best_bid)) or 0.0
        spread_cents = _d(raw_spread) * Decimal("100")
        bid_depth_usd = _d(getattr(snapshot, "bid_depth_usd", 0.0) or 0.0)
        ask_depth_usd = _d(getattr(snapshot, "ask_depth_usd", 0.0) or 0.0)
        timestamp = float(getattr(snapshot, "timestamp", 0.0) or 0.0)
        book_age_ms = max(0, int(current_timestamp_ms - int(timestamp * 1000))) if timestamp > 0 else settings.strategy.reward_cancel_on_stale_ms + 1
        fresh = bool(getattr(snapshot, "fresh", True))
        return RewardBookState(
            asset_id=asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread_cents=spread_cents,
            bid_depth_usd=bid_depth_usd,
            ask_depth_usd=ask_depth_usd,
            fresh=fresh,
            book_age_ms=book_age_ms,
        )

    @staticmethod
    def _reward_to_competition(market: MarketInfo) -> Decimal:
        competition = Decimal(str(max(market.reward_competition_score, 0.0)))
        denominator = competition if competition > Decimal("0") else Decimal("1")
        return (_d(market.reward_daily_rate_usd) / denominator).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)

    @staticmethod
    def _passive_buy_price(best_bid: Decimal, best_ask: Decimal) -> Decimal:
        if best_bid <= Decimal("0") or best_ask <= Decimal("0") or best_ask <= best_bid:
            return Decimal("0")
        candidate = best_bid + _TICK_SIZE
        if candidate >= best_ask:
            candidate = best_ask - _TICK_SIZE
        if candidate <= Decimal("0"):
            return Decimal("0")
        return candidate.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    @staticmethod
    def _recent_move_exceeds_threshold(mid_history: Sequence[tuple[int, Decimal]]) -> bool:
        if len(mid_history) < 2:
            return False
        first = mid_history[0][1]
        last = mid_history[-1][1]
        if first <= Decimal("0"):
            return False
        return abs(last - first) / first >= _d(settings.strategy.reward_jump_risk_move_pct)

    @staticmethod
    def _volatility_jump_exceeds_threshold(mid_history: Sequence[tuple[int, Decimal]]) -> bool:
        if len(mid_history) < 3:
            return False
        returns: list[float] = []
        for (_, prev_price), (_, next_price) in zip(mid_history[:-1], mid_history[1:]):
            if prev_price <= Decimal("0") or next_price <= Decimal("0"):
                continue
            returns.append(math.log(float(next_price / prev_price)))
        if len(returns) < 2:
            return False
        return pstdev(returns) >= float(settings.strategy.reward_jump_risk_volatility_pct)