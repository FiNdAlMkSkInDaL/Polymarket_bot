"""
Market lifecycle manager — three-tier state machine that governs which
markets the bot actively trades.

Tiers:
  **observing**  — freshly discovered; subscribed to WS for data collection
                   but NOT eligible for trading.  Promoted after the
                   observation period if score ≥ threshold.
  **active**     — fully scored and tradeable.  Re-scored every refresh.
                   Demoted if score drops below threshold for N consecutive
                   cycles.
  **draining**   — pending eviction but the bot still has an open position.
                   No new trades.  Evicted once the position closes.

Resolution detection:  each refresh cycle polls Gamma for ``active``/
``closed`` status.  Resolved markets are immediately drained or evicted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from src.core.config import settings
from src.core.logger import get_logger
from src.data.market_discovery import MarketInfo, fetch_active_markets
from src.data.market_scorer import ScoreBreakdown, compute_score
from src.data.orderbook import OrderbookTracker

log = get_logger(__name__)


# ── Internal tier structs ──────────────────────────────────────────────────

@dataclass
class ObservingMarket:
    """Market in the observation tier."""
    info: MarketInfo
    entered_at: float = field(default_factory=time.time)
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)


@dataclass
class ActiveMarket:
    """Market in the active (tradeable) tier."""
    info: MarketInfo
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    low_score_cycles: int = 0
    last_signal_time: float = 0.0  # for signal cooldown


@dataclass
class DrainingMarket:
    """Market pending eviction — has an open position."""
    info: MarketInfo
    reason: str = "resolved"


@dataclass
class EmergencyDrainMarket:
    """Market in emergency drain due to ghost liquidity detection.

    The bot immediately cancels entry orders and force-exits open
    positions.  Recovery requires depth returning to ≥50% of the
    baseline for ``ghost_recovery_s`` continuous seconds.
    """
    info: MarketInfo
    reason: str = "ghost_liquidity"
    triggered_at: float = 0.0
    baseline_depth: float = 0.0
    recovery_start: float = 0.0     # when depth first recovered


class MarketLifecycleManager:
    """Manages the three-tier market universe.

    Public interface consumed by the bot:
      - ``initial_discovery()`` — bootstrap at startup
      - ``refresh()``          — periodic re-evaluation
      - ``is_tradeable(condition_id)`` — can the bot open a position?
      - ``record_signal(condition_id)`` — mark last signal time (cooldown)
      - ``drain_market(condition_id, reason)`` — move to draining
      - ``evict_market(condition_id)`` — full removal (returns cleanup info)
      - ``promote_ready()`` — promote observed → active where ready
    """

    def __init__(self) -> None:
        self.observing: dict[str, ObservingMarket] = {}
        self.active: dict[str, ActiveMarket] = {}
        self.draining: dict[str, DrainingMarket] = {}
        self.emergency: dict[str, EmergencyDrainMarket] = {}

    # ────────────────────────── Bootstrap ──────────────────────────────────

    async def initial_discovery(self) -> list[MarketInfo]:
        """Fetch the initial market universe.

        All markets that pass filters start in the **observing** tier.
        Those with enough Gamma data to score above threshold are
        immediately promoted to **active**.
        """
        raw_markets = await fetch_active_markets()

        for m in raw_markets:
            score = compute_score(
                daily_volume_usd=m.daily_volume_usd,
                liquidity_usd=m.liquidity_usd,
                end_date=m.end_date,
            )
            m.score = score.total

            if score.total >= settings.strategy.min_market_score:
                self.active[m.condition_id] = ActiveMarket(info=m, score=score)
            else:
                self.observing[m.condition_id] = ObservingMarket(info=m, score=score)

        log.info(
            "lifecycle_bootstrap",
            active=len(self.active),
            observing=len(self.observing),
            total_discovered=len(raw_markets),
        )

        return [am.info for am in self.active.values()]

    # ────────────────────────── Refresh ────────────────────────────────────

    async def refresh(
        self,
        orderbook_trackers: dict[str, OrderbookTracker] | None = None,
        trade_counts: dict[str, float] | None = None,
        whale_tokens: set[str] | None = None,
        open_position_markets: set[str] | None = None,
        taker_counts: dict[str, int] | None = None,
        total_counts: dict[str, int] | None = None,
    ) -> tuple[list[MarketInfo], list[str]]:
        """Re-discover, re-score, promote/demote/evict.

        Returns ``(newly_added, evicted_condition_ids)``.
        """
        orderbook_trackers = orderbook_trackers or {}
        trade_counts = trade_counts or {}
        whale_tokens = whale_tokens or set()
        open_position_markets = open_position_markets or set()
        taker_counts = taker_counts or {}
        total_counts = total_counts or {}

        # 1. Fetch fresh universe
        fresh = await fetch_active_markets()
        fresh_by_id = {m.condition_id: m for m in fresh}

        # 2. Check for resolved markets among active + observing
        all_tracked = set(self.active) | set(self.observing) | set(self.draining)
        resolved_ids = await self._check_resolved(all_tracked, fresh_by_id)

        # 3. Handle resolved markets
        evicted: list[str] = []
        for cid in resolved_ids:
            if cid in open_position_markets:
                self._move_to_draining(cid, reason="resolved")
            else:
                self._evict(cid)
                evicted.append(cid)

        # 4. Re-score active markets
        min_score = settings.strategy.min_market_score
        max_demotion = settings.strategy.demotion_cycles_before_evict

        for cid in list(self.active):
            am = self.active[cid]

            # Update info if Gamma returned fresh data
            if cid in fresh_by_id:
                fm = fresh_by_id[cid]
                am.info.daily_volume_usd = fm.daily_volume_usd
                am.info.liquidity_usd = fm.liquidity_usd
                am.info.end_date = fm.end_date
                am.info.accepting_orders = fm.accepting_orders

            # Build score with live data if available
            spread = 0.0
            mid_price = 0.5
            live_spread_score = None
            book_tracker = orderbook_trackers.get(am.info.yes_token_id)
            if book_tracker and book_tracker.has_data:
                spread = book_tracker.spread_cents
                snap = book_tracker.snapshot()
                mid_price = snap.mid_price or 0.5
                if snap.spread_score > 0:
                    live_spread_score = snap.spread_score

            tpm = trade_counts.get(am.info.yes_token_id, 0.0)
            has_whale = (
                am.info.yes_token_id in whale_tokens
                or am.info.no_token_id in whale_tokens
            )

            # Aggregate taker/total counts across both tokens
            mkt_taker = (
                taker_counts.get(am.info.yes_token_id, 0)
                + taker_counts.get(am.info.no_token_id, 0)
            )
            mkt_total = (
                total_counts.get(am.info.yes_token_id, 0)
                + total_counts.get(am.info.no_token_id, 0)
            )

            score = compute_score(
                daily_volume_usd=am.info.daily_volume_usd,
                liquidity_usd=am.info.liquidity_usd,
                spread_cents=spread,
                end_date=am.info.end_date,
                mid_price=mid_price,
                trades_per_minute=tpm,
                has_whale_activity=has_whale,
                taker_count=mkt_taker,
                total_count=mkt_total,
                live_spread_score=live_spread_score,
            )
            am.score = score
            am.info.score = score.total

            # Not accepting orders → drain immediately
            if not am.info.accepting_orders:
                if cid in open_position_markets:
                    self._move_to_draining(cid, reason="not_accepting_orders")
                else:
                    self._evict(cid)
                    evicted.append(cid)
                continue

            # Demotion check
            if score.total < min_score:
                am.low_score_cycles += 1
                if am.low_score_cycles >= max_demotion:
                    if cid in open_position_markets:
                        self._move_to_draining(cid, reason="low_score")
                    else:
                        self._evict(cid)
                        evicted.append(cid)
                    log.info(
                        "market_demoted",
                        condition_id=cid,
                        score=round(score.total, 1),
                        cycles=am.low_score_cycles,
                    )
            else:
                am.low_score_cycles = 0

        # 5. Promote observation-tier markets
        self._promote_ready(
            orderbook_trackers, trade_counts, whale_tokens,
            taker_counts, total_counts,
        )

        # 6. Add genuinely new markets to observation
        existing = set(self.active) | set(self.observing) | set(self.draining)
        newly_added: list[MarketInfo] = []

        for m in fresh:
            if m.condition_id not in existing:
                score = compute_score(
                    daily_volume_usd=m.daily_volume_usd,
                    liquidity_usd=m.liquidity_usd,
                    end_date=m.end_date,
                )
                m.score = score.total
                self.observing[m.condition_id] = ObservingMarket(info=m, score=score)
                newly_added.append(m)

        # 7. Check draining markets — evict if no longer in open positions
        for cid in list(self.draining):
            if cid not in open_position_markets:
                self._evict(cid)
                evicted.append(cid)

        log.info(
            "lifecycle_refresh",
            active=len(self.active),
            observing=len(self.observing),
            draining=len(self.draining),
            new=len(newly_added),
            evicted=len(evicted),
        )

        return newly_added, evicted

    # ────────────────────────── Queries ────────────────────────────────────

    def is_tradeable(self, condition_id: str) -> bool:
        """Can the bot open a NEW position on this market?"""
        return condition_id in self.active and condition_id not in self.emergency

    def is_tracked(self, condition_id: str) -> bool:
        """Is this market in any tier?"""
        return (
            condition_id in self.active
            or condition_id in self.observing
            or condition_id in self.draining
            or condition_id in self.emergency
        )

    def get_active_markets(self) -> list[MarketInfo]:
        """Return all currently tradeable markets."""
        return [am.info for am in self.active.values()]

    def get_all_tracked(self) -> list[MarketInfo]:
        """All markets across all tiers."""
        result: list[MarketInfo] = []
        for am in self.active.values():
            result.append(am.info)
        for om in self.observing.values():
            result.append(om.info)
        for dm in self.draining.values():
            result.append(dm.info)
        for em in self.emergency.values():
            result.append(em.info)
        return result

    def record_signal(self, condition_id: str) -> None:
        """Mark that a panic signal just fired on this market (cooldown)."""
        am = self.active.get(condition_id)
        if am:
            am.last_signal_time = time.time()

    def is_cooled_down(self, condition_id: str) -> bool:
        """Has enough time passed since the last signal on this market?"""
        am = self.active.get(condition_id)
        if not am:
            return False
        cooldown_sec = settings.strategy.signal_cooldown_minutes * 60
        return (time.time() - am.last_signal_time) >= cooldown_sec

    def get_score(self, condition_id: str) -> float:
        """Get current score (0 if not tracked)."""
        am = self.active.get(condition_id)
        if am:
            return am.score.total
        om = self.observing.get(condition_id)
        if om:
            return om.score.total
        return 0.0

    def get_score_breakdown(self, condition_id: str) -> ScoreBreakdown | None:
        am = self.active.get(condition_id)
        if am:
            return am.score
        om = self.observing.get(condition_id)
        if om:
            return om.score
        return None

    # ────────────────────────── Mutations ──────────────────────────────────

    def drain_market(self, condition_id: str, reason: str = "manual") -> None:
        """Move a market to the draining tier."""
        self._move_to_draining(condition_id, reason)

    def emergency_drain(
        self, condition_id: str, baseline_depth: float, reason: str = "ghost_liquidity"
    ) -> None:
        """Move a market to the EMERGENCY_DRAIN state.

        Triggered by the Ghost Liquidity Circuit Breaker when depth drops
        > 50% in < 2s without matching trades.  The bot immediately cancels
        entry orders and force-exits open positions.
        """
        info = None
        if condition_id in self.active:
            info = self.active.pop(condition_id).info
        elif condition_id in self.observing:
            info = self.observing.pop(condition_id).info

        if info:
            self.emergency[condition_id] = EmergencyDrainMarket(
                info=info,
                reason=reason,
                triggered_at=time.time(),
                baseline_depth=baseline_depth,
            )
            log.warning(
                "emergency_drain_triggered",
                condition_id=condition_id,
                reason=reason,
                baseline_depth=round(baseline_depth, 2),
            )

    def check_emergency_recovery(
        self,
        orderbook_trackers: dict[str, OrderbookTracker],
    ) -> list[str]:
        """Check if any emergency-drained markets have recovered.

        Recovery requires depth to be ≥ 50% of the pre-ghost baseline
        for ``ghost_recovery_s`` continuous seconds.

        Returns list of condition_ids that recovered back to active.
        """
        recovery_s = settings.strategy.ghost_recovery_s
        recovered: list[str] = []
        now = time.time()

        for cid in list(self.emergency):
            em = self.emergency[cid]

            # Find the right book tracker
            tracker = (
                orderbook_trackers.get(em.info.yes_token_id)
                or orderbook_trackers.get(em.info.no_token_id)
            )
            if not tracker or not tracker.has_data:
                em.recovery_start = 0.0
                continue

            current_depth = tracker.current_total_depth()
            threshold = em.baseline_depth * 0.50

            if current_depth >= threshold:
                if em.recovery_start <= 0:
                    em.recovery_start = now
                elif (now - em.recovery_start) >= recovery_s:
                    # Depth has been stable for recovery_s — promote back
                    self.active[cid] = ActiveMarket(info=em.info)
                    del self.emergency[cid]
                    recovered.append(cid)
                    log.info(
                        "emergency_drain_recovered",
                        condition_id=cid,
                        depth=round(current_depth, 2),
                        baseline=round(em.baseline_depth, 2),
                    )
            else:
                em.recovery_start = 0.0  # reset recovery timer

        return recovered

    def is_emergency_drained(self, condition_id: str) -> bool:
        """Is this market in emergency drain?"""
        return condition_id in self.emergency

    # ────────────────────── Stale-trade eviction ──────────────────────────

    def check_stale_markets(
        self,
        yes_aggs: dict[str, "OHLCVAggregator"],
        open_position_markets: set[str],
        *,
        stale_threshold_s: float = 900.0,
    ) -> list[str]:
        """Evict active markets with no trades for ``stale_threshold_s`` seconds.

        Markets with open positions are drained instead of evicted so the
        position manager can close them gracefully.

        Parameters
        ----------
        yes_aggs:
            Mapping of ``yes_token_id → OHLCVAggregator``.
        open_position_markets:
            Set of ``condition_id``s that have open positions.
        stale_threshold_s:
            Duration without trades before a market is considered dead.
            Default 900s (15 minutes).

        Returns
        -------
        list[str]
            Condition IDs that were evicted or moved to draining.
        """
        evicted: list[str] = []
        now = time.time()

        for cid in list(self.active):
            am = self.active[cid]
            agg = yes_aggs.get(am.info.yes_token_id)
            if not agg:
                continue

            # If we have never received a trade at all, use discovery time
            last = agg.last_trade_time
            if last <= 0:
                continue  # no trades yet — too early to judge

            age = now - last
            if age > stale_threshold_s:
                log.info(
                    "stale_trade_eviction",
                    condition_id=cid,
                    last_trade_age_s=round(age, 1),
                    threshold_s=stale_threshold_s,
                    question=am.info.question[:60],
                )
                if cid in open_position_markets:
                    self._move_to_draining(cid, reason="stale_trades")
                else:
                    self._evict(cid)
                evicted.append(cid)

        return evicted

    # ────────────────────────── Internals ──────────────────────────────────

    def _promote_ready(
        self,
        orderbook_trackers: dict[str, OrderbookTracker] | None = None,
        trade_counts: dict[str, float] | None = None,
        whale_tokens: set[str] | None = None,
        taker_counts: dict[str, int] | None = None,
        total_counts: dict[str, int] | None = None,
    ) -> None:
        """Promote observed markets to active if period elapsed + score OK."""
        orderbook_trackers = orderbook_trackers or {}
        trade_counts = trade_counts or {}
        whale_tokens = whale_tokens or set()
        taker_counts = taker_counts or {}
        total_counts = total_counts or {}

        obs_period = settings.strategy.observation_period_minutes * 60
        min_score = settings.strategy.min_market_score
        now = time.time()

        for cid in list(self.observing):
            om = self.observing[cid]
            if (now - om.entered_at) < obs_period:
                continue

            # Re-score with any live data
            spread = 0.0
            mid_price = 0.5
            live_spread_score = None
            bt = orderbook_trackers.get(om.info.yes_token_id)
            if bt and bt.has_data:
                spread = bt.spread_cents
                snap = bt.snapshot()
                mid_price = snap.mid_price or 0.5
                if snap.spread_score > 0:
                    live_spread_score = snap.spread_score

            tpm = trade_counts.get(om.info.yes_token_id, 0.0)
            has_whale = (
                om.info.yes_token_id in whale_tokens
                or om.info.no_token_id in whale_tokens
            )

            # Aggregate taker/total counts for MTI penalty
            mkt_taker = (
                taker_counts.get(om.info.yes_token_id, 0)
                + taker_counts.get(om.info.no_token_id, 0)
            )
            mkt_total = (
                total_counts.get(om.info.yes_token_id, 0)
                + total_counts.get(om.info.no_token_id, 0)
            )

            score = compute_score(
                daily_volume_usd=om.info.daily_volume_usd,
                liquidity_usd=om.info.liquidity_usd,
                spread_cents=spread,
                end_date=om.info.end_date,
                mid_price=mid_price,
                trades_per_minute=tpm,
                has_whale_activity=has_whale,
                taker_count=mkt_taker,
                total_count=mkt_total,
                live_spread_score=live_spread_score,
            )

            if score.total >= min_score:
                om.info.score = score.total
                self.active[cid] = ActiveMarket(info=om.info, score=score)
                del self.observing[cid]
                log.info(
                    "market_promoted",
                    condition_id=cid,
                    question=om.info.question[:60],
                    score=round(score.total, 1),
                )
            else:
                om.score = score

    def _move_to_draining(self, condition_id: str, reason: str) -> None:
        """Move from active/observing to draining."""
        info = None
        if condition_id in self.active:
            info = self.active.pop(condition_id).info
        elif condition_id in self.observing:
            info = self.observing.pop(condition_id).info
        if info:
            self.draining[condition_id] = DrainingMarket(info=info, reason=reason)
            log.info("market_draining", condition_id=condition_id, reason=reason)

    def _evict(self, condition_id: str) -> None:
        """Remove from all tiers."""
        self.active.pop(condition_id, None)
        self.observing.pop(condition_id, None)
        self.draining.pop(condition_id, None)
        self.emergency.pop(condition_id, None)
        log.info("market_evicted", condition_id=condition_id)

    async def _check_resolved(
        self,
        tracked_ids: set[str],
        fresh_by_id: dict[str, MarketInfo],
    ) -> set[str]:
        """Identify tracked markets that are no longer in the fresh set.

        If a market disappears from the fresh discovery results (because it's
        now ``active=false`` or ``closed=true`` on Gamma), it's considered
        resolved.  We also do a targeted Gamma poll for markets still
        appearing but with stale data.
        """
        resolved: set[str] = set()

        for cid in tracked_ids:
            # If the market is NOT in the fresh discovery set, it's probably
            # resolved / closed / deactivated.
            if cid not in fresh_by_id:
                resolved.add(cid)

        if resolved:
            log.info(
                "resolved_detected",
                count=len(resolved),
                sample=list(resolved)[:5],
            )

        return resolved
