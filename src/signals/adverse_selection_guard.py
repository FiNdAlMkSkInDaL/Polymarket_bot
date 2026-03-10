"""
Anti-adverse-selection module -- the **"Fast-Kill"** mechanism (v2).

Detects toxic flow using four **intra-Polymarket** microstructure signals
derived entirely from data the bot already collects (trade events, L2
book snapshots, per-asset taker counts).  No external WebSocket
connections or price feeds are needed.

Kill condition
--------------
A fast-kill fires when **any two of the four signals** trigger
simultaneously (2-of-4 rule).  Single-signal triggers are logged but
do not cancel orders -- this dramatically reduces false positives from
idiosyncratic market noise while still catching genuine information events.

Detection mechanisms
--------------------
1. **Cross-market flow coherence** -- simultaneous directional taker flow
   across 3+ unrelated markets indicates a platform-wide information
   event.  Measured via windowed MTI (maker/taker imbalance) deltas.

2. **Book depth evaporation** -- market makers pull quotes before price
   moves when informed traders arrive.  Detects >60%% depth-near-mid
   withdrawal within 2 seconds on positioned assets.

3. **Spread blow-out** -- bid-ask spread widening to 3x the 5-minute
   rolling average indicates defensive MM widening due to perceived
   information asymmetry.

4. **Velocity anomaly** -- 5x spike in trade arrival rate over a
   10-minute baseline on a positioned market indicates a burst of
   informed activity targeting that specific market.

Usage (wired into ``TradingBot._run()``)::

    guard = AdverseSelectionGuard(
        executor, book_trackers, fast_kill_event,
        taker_counts=bot._taker_counts,
        total_counts=bot._total_counts,
        trade_counts=bot._trade_counts,
        get_position_assets=bot._positioned_asset_ids,
        telegram=bot.telegram,
    )
    tasks.append(asyncio.create_task(guard.start()))
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

from src.core.config import settings
from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
from src.core.logger import get_logger

if TYPE_CHECKING:
    from src.data.orderbook import OrderbookTracker
    from src.trading.executor import OrderExecutor

log = get_logger(__name__)


def _no_positions() -> set[str]:
    """Default callback when no position manager is wired."""
    return set()


class AdverseSelectionGuard:
    """Intrinsic predictive cancellation engine driven by intra-Polymarket
    microstructure signals.

    Runs a single fast-polling decision loop (default 50 ms cadence) that
    evaluates four detection mechanisms each cycle.  All computations are
    pure arithmetic on shared mutable references -- no blocking I/O in the
    hot loop.

    Parameters
    ----------
    executor:
        Shared ``OrderExecutor`` for cancellation.
    book_trackers:
        Dict of ``asset_id -> OrderbookTracker`` -- shared mutable reference
        populated by ``TradingBot._wire_market()``.
    fast_kill_event:
        Shared ``asyncio.Event`` -- *cleared* during a kill cooldown so
        all ``OrderChaser`` instances pause.
    taker_counts:
        Dict of ``asset_id -> int`` -- cumulative taker-initiated trades,
        populated by ``TradingBot._process_trades()``.  Halved each
        market refresh cycle.
    total_counts:
        Dict of ``asset_id -> int`` -- cumulative classified trades.
    trade_counts:
        Dict of ``asset_id -> float`` -- trades/min, decayed each refresh.
    get_position_assets:
        Callable returning the set of ``asset_id`` strings that currently
        have open positions.  Depth evaporation, spread blow-out, and
        velocity signals only evaluate positioned assets.
    telegram:
        Optional ``TelegramAlerter`` for kill notifications.
    """

    def __init__(
        self,
        executor: OrderExecutor,
        book_trackers: dict[str, OrderbookTracker],
        fast_kill_event: asyncio.Event,
        *,
        taker_counts: dict[str, int] | None = None,
        total_counts: dict[str, int] | None = None,
        trade_counts: dict[str, float] | None = None,
        get_position_assets: Callable[[], set[str]] | None = None,
        telegram: object | None = None,
        on_shutdown: Callable[[], Any] | None = None,
    ):
        strat = settings.strategy
        self._executor = executor
        self._books = book_trackers
        self._kill_event = fast_kill_event

        # Shared mutable references (populated by TradingBot._process_trades)
        self._taker_counts = taker_counts if taker_counts is not None else {}
        self._total_counts = total_counts if total_counts is not None else {}
        self._trade_counts = trade_counts if trade_counts is not None else {}
        self._get_position_assets = get_position_assets or _no_positions
        self._telegram = telegram
        self._on_shutdown = on_shutdown

        # Exception circuit breaker for the decision loop
        self._loop_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)

        # Core lifecycle config (preserved from v1)
        self._enabled = strat.adverse_sel_enabled
        self._cooldown_s = strat.adverse_sel_cooldown_s
        self._poll_s = strat.adverse_sel_poll_ms / 1000.0

        # Signal 1: Cross-market flow coherence
        self._mti_threshold = strat.adverse_sel_mti_threshold
        self._mti_min_markets = strat.adverse_sel_mti_min_markets
        self._mti_window_s = strat.adverse_sel_mti_window_s

        # Signal 2: Book depth evaporation
        self._depth_drop_pct = strat.adverse_sel_depth_drop_pct
        self._depth_window_s = strat.adverse_sel_depth_window_s
        self._depth_near_mid_cents = strat.adverse_sel_depth_near_mid_cents

        # Signal 3: Spread blow-out
        self._spread_blowout_mult = strat.adverse_sel_spread_blowout_mult
        self._spread_avg_window_s = strat.adverse_sel_spread_avg_window_s

        # Signal 4: Velocity anomaly
        self._velocity_mult = strat.adverse_sel_velocity_mult
        self._velocity_window_s = strat.adverse_sel_velocity_window_s
        self._high_freq_baseline = strat.adverse_sel_high_freq_baseline
        self._high_freq_mult_boost = strat.adverse_sel_high_freq_mult_boost

        # Kill outcome retrospective analysis
        self._outcome_delay_s = strat.adverse_sel_outcome_delay_s
        self._tp_threshold_cents = strat.adverse_sel_tp_threshold_cents
        self._outcome_file = "logs/adverse_sel_outcomes.jsonl"

        # -- Internal state --------------------------------------------------

        # Signal 1: MTI snapshots -- (timestamp, {asset_id: (taker, total)})
        max_mti_samples = max(10, int(self._mti_window_s / self._poll_s) + 5)
        self._mti_snapshots: deque[tuple[float, dict[str, tuple[int, int]]]] = deque(
            maxlen=max_mti_samples
        )

        # Signal 2: depth-near-mid history -- per asset
        max_depth_samples = max(10, int(self._depth_window_s / self._poll_s) + 5)
        self._depth_near_mid_history: dict[str, deque[tuple[float, float]]] = {}
        self._depth_history_maxlen = max_depth_samples

        # Signal 3: spread history (timestamped) -- per asset
        max_spread_samples = max(60, int(self._spread_avg_window_s / self._poll_s) + 5)
        max_spread_samples = min(max_spread_samples, 10_000)
        self._spread_ts_history: dict[str, deque[tuple[float, float]]] = {}
        self._spread_history_maxlen = max_spread_samples

        # Signal 4: trade count snapshots -- (timestamp, {asset_id: count})
        max_velocity_samples = max(10, int(self._velocity_window_s / self._poll_s) + 5)
        max_velocity_samples = min(max_velocity_samples, 15_000)
        self._trade_rate_snapshots: deque[tuple[float, dict[str, float]]] = deque(
            maxlen=max_velocity_samples
        )

        # Lifecycle
        self._cooldown_until: float = 0.0
        self._running = False
        self._cancel_count: int = 0

        # Confirmation persistence — require 2-of-4 condition to hold
        # for N consecutive poll cycles before firing a kill.  This
        # filters transient microstructure noise (single-cycle spikes)
        # that would be false positives.
        # At 50ms poll cadence, 4 cycles = 200ms confirmation delay.
        self._confirmation_required: int = strat.adverse_sel_confirmation_cycles
        self._consecutive_trigger_count: int = 0

    # === Lifecycle ==========================================================

    async def start(self) -> None:
        """Launch the decision loop and run until stopped."""
        if not self._enabled:
            log.info("adverse_sel_disabled")
            return

        self._running = True
        self._kill_event.set()  # start with chasers allowed to proceed

        try:
            await self._decision_loop()
        except asyncio.CancelledError:
            self._running = False

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self._kill_event.set()

    # === Signal 1: Cross-market flow coherence ==============================

    def _snapshot_mti(self) -> None:
        """Record a point-in-time snapshot of per-asset taker/total counts.

        Called every poll cycle.  The windowed MTI is computed as the
        *delta* between the oldest and newest snapshots in the window,
        which handles the halving decay in ``_market_refresh_loop``
        gracefully (a decay epoch produces zero or negative delta ->
        treated as no data).
        """
        now = time.time()
        snap: dict[str, tuple[int, int]] = {}
        for asset_id in self._total_counts:
            taker = self._taker_counts.get(asset_id, 0)
            total = self._total_counts.get(asset_id, 0)
            snap[asset_id] = (taker, total)
        self._mti_snapshots.append((now, snap))

    def _check_flow_coherence(self) -> tuple[bool, dict]:
        """Evaluate Signal 1: cross-market flow coherence.

        Computes windowed MTI per asset over ``mti_window_s`` by taking
        the delta between cumulative counts at window boundaries.  If
        the number of tracked markets where windowed MTI exceeds the
        threshold is >= ``mti_min_markets``, the signal fires.

        Returns
        -------
        (fired, diagnostics)
        """
        if len(self._mti_snapshots) < 2:
            return False, {}

        now = time.time()
        cutoff = now - self._mti_window_s

        # Find oldest snapshot within window
        oldest_snap: dict[str, tuple[int, int]] | None = None
        for ts, snap in self._mti_snapshots:
            if ts >= cutoff:
                oldest_snap = snap
                break
        if oldest_snap is None:
            oldest_snap = self._mti_snapshots[0][1]

        newest_snap = self._mti_snapshots[-1][1]

        hot_markets: list[str] = []
        all_assets = set(newest_snap.keys()) | set(oldest_snap.keys())
        for asset_id in all_assets:
            new_taker, new_total = newest_snap.get(asset_id, (0, 0))
            old_taker, old_total = oldest_snap.get(asset_id, (0, 0))
            delta_total = new_total - old_total
            delta_taker = new_taker - old_taker

            # Skip if negative delta (halving decay boundary) or insufficient data
            if delta_total < 3:
                continue

            mti = delta_taker / delta_total
            if mti >= self._mti_threshold:
                hot_markets.append(asset_id)

        fired = len(hot_markets) >= self._mti_min_markets
        diag = {
            "signal": "flow_coherence",
            "hot_markets": len(hot_markets),
            "min_required": self._mti_min_markets,
            "threshold": self._mti_threshold,
            "hot_asset_ids": hot_markets[:5],
        }
        return fired, diag

    # === Signal 2: Book depth evaporation ===================================

    def _depth_near_mid(self, tracker: Any) -> float:
        """Compute total resting depth (USD) within ``depth_near_mid_cents``
        of mid-price for a given orderbook tracker.

        Prefers the allocation-free ``depth_near_mid_usd()`` method when
        available (L2OrderBook / L2OrderBookAdapter).  Falls back to
        ``tracker.levels()`` for legacy trackers.
        """
        # Fast path: use the allocation-free method if available
        if hasattr(tracker, 'depth_near_mid_usd'):
            return tracker.depth_near_mid_usd(self._depth_near_mid_cents, 50)

        best_bid = tracker.best_bid
        best_ask = tracker.best_ask
        if best_bid <= 0 or best_ask <= 0:
            return 0.0

        mid = (best_bid + best_ask) / 2.0
        threshold = self._depth_near_mid_cents / 100.0  # convert cents -> price

        depth = 0.0
        for level in tracker.levels("bid", 50):
            if abs(level.price - mid) <= threshold:
                depth += level.price * level.size
        for level in tracker.levels("ask", 50):
            if abs(level.price - mid) <= threshold:
                depth += level.price * level.size

        return depth

    def _snapshot_depth(self) -> None:
        """Record depth-near-mid for all positioned assets."""
        now = time.time()
        position_assets = self._get_position_assets()

        for asset_id in position_assets:
            tracker = self._books.get(asset_id)
            if not tracker or not tracker.has_data:
                continue
            depth = self._depth_near_mid(tracker)

            if asset_id not in self._depth_near_mid_history:
                self._depth_near_mid_history[asset_id] = deque(
                    maxlen=self._depth_history_maxlen
                )
            self._depth_near_mid_history[asset_id].append((now, depth))

        # Prune assets no longer positioned
        for aid in list(self._depth_near_mid_history):
            if aid not in position_assets:
                del self._depth_near_mid_history[aid]

    def _check_depth_evaporation(self) -> tuple[bool, dict]:
        """Evaluate Signal 2: book depth evaporation on positioned assets.

        For each positioned asset, compares current depth-near-mid to the
        depth recorded ``depth_window_s`` ago.  If the drop exceeds
        ``depth_drop_pct``, the signal fires.

        Returns
        -------
        (fired, diagnostics)
        """
        now = time.time()
        cutoff = now - self._depth_window_s
        triggered_assets: list[str] = []
        worst_drop = 0.0

        for asset_id, history in self._depth_near_mid_history.items():
            if len(history) < 2:
                continue

            # Find depth at window boundary
            past_depth: float | None = None
            for ts, depth in history:
                if ts <= cutoff:
                    past_depth = depth
                else:
                    break

            if past_depth is None or past_depth <= 0:
                continue

            current_depth = history[-1][1]
            drop_frac = (past_depth - current_depth) / past_depth

            if drop_frac >= self._depth_drop_pct:
                triggered_assets.append(asset_id)
                worst_drop = max(worst_drop, drop_frac)

        fired = len(triggered_assets) > 0
        diag = {
            "signal": "depth_evaporation",
            "triggered_assets": triggered_assets[:5],
            "worst_drop_pct": round(worst_drop * 100, 1),
            "threshold_pct": round(self._depth_drop_pct * 100, 1),
        }
        return fired, diag

    # === Signal 3: Spread blow-out ==========================================

    def _snapshot_spreads(self) -> None:
        """Record current spread for all positioned assets."""
        now = time.time()
        position_assets = self._get_position_assets()

        for asset_id in position_assets:
            tracker = self._books.get(asset_id)
            if not tracker or not tracker.has_data:
                continue
            spread = tracker.spread_cents

            if asset_id not in self._spread_ts_history:
                self._spread_ts_history[asset_id] = deque(
                    maxlen=self._spread_history_maxlen
                )
            self._spread_ts_history[asset_id].append((now, spread))

        # Prune assets no longer positioned
        for aid in list(self._spread_ts_history):
            if aid not in position_assets:
                del self._spread_ts_history[aid]

    def _check_spread_blowout(self) -> tuple[bool, dict]:
        """Evaluate Signal 3: spread blow-out on positioned assets.

        For each positioned asset, compares current spread to the rolling
        average over ``spread_avg_window_s``.  If the ratio exceeds
        ``spread_blowout_mult``, the signal fires.

        Returns
        -------
        (fired, diagnostics)
        """
        now = time.time()
        cutoff = now - self._spread_avg_window_s
        triggered_assets: list[str] = []
        worst_mult = 0.0

        for asset_id, history in self._spread_ts_history.items():
            if len(history) < 10:  # minimum history for meaningful average
                continue

            # Only include samples within the configured time window
            windowed = [(ts, s) for ts, s in history if ts >= cutoff]
            if len(windowed) < 10:
                continue

            current_spread = windowed[-1][1]
            # Exclude the current sample from the average to prevent
            # a single blowout from inflating the baseline it's compared to.
            if len(windowed) < 11:  # need >=10 historical + 1 current
                continue
            historical = windowed[:-1]
            avg_spread = sum(s for _, s in historical) / len(historical)
            if avg_spread <= 0:
                continue

            ratio = current_spread / avg_spread

            if ratio >= self._spread_blowout_mult:
                triggered_assets.append(asset_id)
                worst_mult = max(worst_mult, ratio)

        fired = len(triggered_assets) > 0
        diag = {
            "signal": "spread_blowout",
            "triggered_assets": triggered_assets[:5],
            "worst_mult": round(worst_mult, 2),
            "threshold_mult": self._spread_blowout_mult,
        }
        return fired, diag

    # === Signal 4: Velocity anomaly on positioned assets ====================

    def _snapshot_trade_rates(self) -> None:
        """Record a point-in-time snapshot of per-asset trade counts."""
        now = time.time()
        snap: dict[str, float] = {}
        for asset_id, count in self._trade_counts.items():
            snap[asset_id] = count
        self._trade_rate_snapshots.append((now, snap))

    def _check_velocity_anomaly(self) -> tuple[bool, dict]:
        """Evaluate Signal 4: velocity anomaly on positioned assets.

        Computes the trade arrival rate over the last 10 seconds and
        compares it to the rate over the full ``velocity_window_s``.
        If the short-term rate is >= ``velocity_mult`` x long-term rate
        on any positioned asset, the signal fires.

        We use a short window (10s) for the burst rate since prediction
        market information events produce sharp spikes, not gradual
        ramps.  The 10-minute baseline captures the market's normal
        activity level.

        Returns
        -------
        (fired, diagnostics)
        """
        if len(self._trade_rate_snapshots) < 2:
            return False, {}

        now = time.time()
        newest_ts, newest_snap = self._trade_rate_snapshots[-1]

        # Find snapshot closest to velocity_window_s ago (long baseline)
        long_cutoff = now - self._velocity_window_s
        long_snap: dict[str, float] | None = None
        long_ts: float = 0.0
        for ts, snap in self._trade_rate_snapshots:
            if ts <= long_cutoff:
                long_snap = snap
                long_ts = ts
            else:
                break

        # Find snapshot closest to 10s ago (short burst window)
        short_window = 10.0
        short_cutoff = now - short_window
        short_snap: dict[str, float] | None = None
        short_ts: float = 0.0
        for ts, snap in self._trade_rate_snapshots:
            if ts <= short_cutoff:
                short_snap = snap
                short_ts = ts
            else:
                break

        if long_snap is None or short_snap is None:
            return False, {}

        long_elapsed = newest_ts - long_ts
        short_elapsed = newest_ts - short_ts
        if long_elapsed <= 0 or short_elapsed <= 0:
            return False, {}

        position_assets = self._get_position_assets()
        triggered_assets: list[str] = []
        worst_mult = 0.0

        for asset_id in position_assets:
            new_count = newest_snap.get(asset_id, 0.0)
            long_count = long_snap.get(asset_id, 0.0)
            short_count = short_snap.get(asset_id, 0.0)

            long_delta = new_count - long_count
            if long_delta <= 0:
                continue
            long_rate = long_delta / long_elapsed

            short_delta = new_count - short_count
            if short_delta <= 0:
                continue
            short_rate = short_delta / short_elapsed

            if long_rate <= 0:
                continue

            # Adaptive multiplier: high-frequency markets need a bigger
            # spike to trigger, preventing false positives on naturally
            # active markets.  Baseline is trades per minute.
            long_rate_per_min = long_rate * 60.0
            effective_mult = self._velocity_mult
            if long_rate_per_min >= self._high_freq_baseline:
                effective_mult = self._velocity_mult * self._high_freq_mult_boost

            ratio = short_rate / long_rate
            if ratio >= effective_mult:
                triggered_assets.append(asset_id)
                worst_mult = max(worst_mult, ratio)

        fired = len(triggered_assets) > 0
        diag = {
            "signal": "velocity_anomaly",
            "triggered_assets": triggered_assets[:5],
            "worst_mult": round(worst_mult, 2),
            "threshold_mult": self._velocity_mult,
        }
        return fired, diag

    # === Decision loop ======================================================

    async def _decision_loop(self) -> None:
        """Fast-polling loop (default 50ms) evaluating all four signals.

        Each cycle:
        1. Snapshot all data sources (pure computation, no I/O)
        2. Evaluate all four detectors
        3. If >= 2 fire AND open orders exist -> execute fast kill
        """
        while self._running:
            await asyncio.sleep(self._poll_s)
            try:
                now = time.time()

                # Respect cooldown
                if now < self._cooldown_until:
                    continue

                # -- Snapshot phase (no I/O) ------------------------------------
                self._snapshot_mti()
                self._snapshot_depth()
                self._snapshot_spreads()
                self._snapshot_trade_rates()

                # -- Evaluation phase -------------------------------------------
                signals_fired: list[dict] = []

                flow_fired, flow_diag = self._check_flow_coherence()
                if flow_fired:
                    signals_fired.append(flow_diag)

                depth_fired, depth_diag = self._check_depth_evaporation()
                if depth_fired:
                    signals_fired.append(depth_diag)

                spread_fired, spread_diag = self._check_spread_blowout()
                if spread_fired:
                    signals_fired.append(spread_diag)

                velocity_fired, velocity_diag = self._check_velocity_anomaly()
                if velocity_fired:
                    signals_fired.append(velocity_diag)

                # -- 2-of-4 composite trigger -----------------------------------
                if len(signals_fired) < 2:
                    self._consecutive_trigger_count = 0  # reset confirmation
                    if signals_fired:
                        log.debug(
                            "adverse_sel_single_signal",
                            signal=signals_fired[0].get("signal"),
                        )
                    continue

                # -- Confirmation persistence gate (OE-3) -----------------------
                # Require the trigger condition to persist for N consecutive
                # poll cycles before actually firing the kill.
                self._consecutive_trigger_count += 1
                if self._consecutive_trigger_count < self._confirmation_required:
                    log.debug(
                        "adverse_sel_confirming",
                        count=self._consecutive_trigger_count,
                        required=self._confirmation_required,
                        signals=[d.get("signal") for d in signals_fired],
                    )
                    continue

                # Reset counter — kill will fire
                self._consecutive_trigger_count = 0

                # Skip no-op kills when there are no resting orders
                open_count = self._executor.open_order_count
                if open_count == 0:
                    log.debug(
                        "adverse_sel_skip",
                        reason="no_open_orders",
                        signals=[d.get("signal") for d in signals_fired],
                    )
                    self._cooldown_until = time.time() + self._cooldown_s
                    continue

                signal_names = [d.get("signal") for d in signals_fired]
                log.warning(
                    "adverse_sel_trigger",
                    signals=signal_names,
                    signals_fired=len(signals_fired),
                    open_orders=open_count,
                    diagnostics=signals_fired,
                )
                await self._execute_fast_kill(signals_fired)
            except asyncio.CancelledError:
                raise
            except (KeyError, AttributeError):
                # Stale/missing book or asset data — non-fatal, skip cycle
                log.warning("adverse_sel_stale_data", exc_info=True)
            except Exception:
                log.error("adverse_sel_loop_error", exc_info=True)
                if self._loop_breaker.record():
                    log.critical(
                        "adverse_sel_circuit_breaker_tripped",
                        errors_in_window=self._loop_breaker.recent_errors,
                        msg="Too many unexpected errors in decision loop — initiating shutdown",
                    )
                    if self._telegram and hasattr(self._telegram, "send"):
                        try:
                            await self._telegram.send(
                                "\U0001f534 <b>CIRCUIT BREAKER</b>: adverse_sel_loop tripped "
                                "(5 unexpected errors in 60s) — shutting down."
                            )
                        except Exception:
                            pass
                    self._running = False
                    if self._on_shutdown is not None:
                        asyncio.ensure_future(self._on_shutdown())
                    return

    # === Kill execution =====================================================

    async def _execute_fast_kill(self, signals: list[dict] | None = None) -> None:
        """Cancel all resting orders and enter cooldown.

        Parameters
        ----------
        signals:
            List of diagnostic dicts from triggered signals, included in
            logs and Telegram alert for post-mortem analysis.
        """
        t0 = time.perf_counter()
        self._kill_event.clear()  # pause all chasers

        count = await self._executor.cancel_all()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._cooldown_until = time.time() + self._cooldown_s
        self._cancel_count += 1

        signal_names = [d.get("signal", "unknown") for d in (signals or [])]

        log.warning(
            "fast_kill_executed",
            cancelled=count,
            elapsed_ms=round(elapsed_ms, 1),
            cooldown_s=self._cooldown_s,
            total_kills=self._cancel_count,
            signals=signal_names,
        )

        # Send Telegram alert with signal breakdown (fire-and-forget to avoid
        # delaying the kill path if Telegram is slow/unreachable)
        if self._telegram and hasattr(self._telegram, "send"):
            async def _send_kill_alert() -> None:
                try:
                    signal_details = "\n".join(
                        f"  - {d.get('signal', '?')}: {d}"
                        for d in (signals or [])
                    )
                    await self._telegram.send(
                        f"<b>Adverse Selection Kill #{self._cancel_count}</b>\n"
                        f"Cancelled {count} orders in {elapsed_ms:.0f}ms\n"
                        f"Signals ({len(signal_names)}): {', '.join(signal_names)}\n"
                        f"Cooldown: {self._cooldown_s}s\n"
                        f"<pre>{signal_details[:1000]}</pre>"
                    )
                except Exception:
                    pass
            asyncio.ensure_future(_send_kill_alert())

        # Schedule cooldown release
        loop = asyncio.get_running_loop()
        loop.call_later(self._cooldown_s, self._kill_event.set)

        # Schedule retrospective outcome analysis
        self._schedule_outcome_check(signal_names)

    # === Kill outcome retrospective analysis ================================

    def _get_mid_prices(self) -> dict[str, float]:
        """Return {asset_id: mid_price} for all positioned assets with data."""
        mids: dict[str, float] = {}
        position_assets = self._get_position_assets()
        for asset_id in position_assets:
            tracker = self._books.get(asset_id)
            if not tracker or not tracker.has_data:
                continue
            bid = tracker.best_bid
            ask = tracker.best_ask
            if bid > 0 and ask > 0:
                mids[asset_id] = (bid + ask) / 2.0
        return mids

    def _schedule_outcome_check(self, signal_names: list[str]) -> None:
        """Snapshot mid-prices now and schedule a delayed re-read.

        After ``outcome_delay_s`` seconds, re-read mid-prices, compute
        per-asset deltas, and classify the kill as true positive (at
        least one positioned asset moved adversely >= threshold) or
        false positive.
        """
        kill_time = time.time()
        kill_number = self._cancel_count
        mids_at_kill = self._get_mid_prices()

        if not mids_at_kill:
            log.info("outcome_skip_no_mids", kill=kill_number)
            return

        async def _evaluate() -> None:
            await asyncio.sleep(self._outcome_delay_s)
            if not self._running:
                return
            mids_after = self._get_mid_prices()
            self._record_kill_outcome(
                kill_number, kill_time, signal_names,
                mids_at_kill, mids_after,
            )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_evaluate())
        except RuntimeError:
            pass  # no running loop (e.g. during shutdown)

    def _record_kill_outcome(
        self,
        kill_number: int,
        kill_time: float,
        signal_names: list[str],
        mids_at_kill: dict[str, float],
        mids_after: dict[str, float],
    ) -> None:
        """Classify a kill as TP or FP and persist the result.

        True positive: at least one positioned asset moved adversely
        (price moved away from our position direction) by >= threshold.
        Since we don't know position direction here, we check for
        **any** absolute price movement >= threshold.

        Parameters
        ----------
        kill_number: sequential kill counter
        kill_time: epoch timestamp of the kill
        signal_names: which signals triggered
        mids_at_kill: {asset_id: mid_price} at kill time
        mids_after: {asset_id: mid_price} after delay
        """
        threshold = self._tp_threshold_cents / 100.0  # cents -> price units
        deltas: dict[str, float] = {}
        max_adverse_move = 0.0

        for asset_id, mid_before in mids_at_kill.items():
            mid_now = mids_after.get(asset_id)
            if mid_now is None:
                continue
            delta_cents = abs(mid_now - mid_before) * 100.0
            deltas[asset_id] = round(delta_cents, 2)
            if abs(mid_now - mid_before) >= threshold:
                max_adverse_move = max(max_adverse_move, delta_cents)

        is_tp = max_adverse_move >= self._tp_threshold_cents
        classification = "true_positive" if is_tp else "false_positive"

        outcome = {
            "kill_number": kill_number,
            "kill_time": kill_time,
            "outcome_time": time.time(),
            "delay_s": self._outcome_delay_s,
            "signals": signal_names,
            "classification": classification,
            "deltas_cents": deltas,
            "max_move_cents": round(max_adverse_move, 2),
            "threshold_cents": self._tp_threshold_cents,
        }

        log.info(
            "kill_outcome",
            classification=classification,
            kill=kill_number,
            max_move_cents=round(max_adverse_move, 2),
            deltas=deltas,
            signals=signal_names,
        )

        # Persist to JSONL
        try:
            os.makedirs(os.path.dirname(self._outcome_file) or ".", exist_ok=True)
            with open(self._outcome_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome) + "\n")
        except OSError as exc:
            log.warning("outcome_write_failed", error=str(exc))

        # Telegram summary
        if self._telegram and hasattr(self._telegram, "send"):
            delta_lines = "\n".join(
                f"  {aid}: {d:+.1f}¢" for aid, d in deltas.items()
            )
            msg = (
                f"<b>Kill #{kill_number} Outcome: {classification.upper()}</b>\n"
                f"Delay: {self._outcome_delay_s}s\n"
                f"Max move: {max_adverse_move:.1f}¢ "
                f"(threshold: {self._tp_threshold_cents}¢)\n"
                f"Signals: {', '.join(signal_names)}\n"
                f"<pre>{delta_lines[:800]}</pre>"
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._telegram.send(msg))
            except RuntimeError:
                pass