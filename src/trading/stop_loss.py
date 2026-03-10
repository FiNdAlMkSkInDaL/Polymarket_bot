"""
Event-driven stop-loss engine — evaluates stop-loss conditions only when
the Best Bid or Offer (BBO) changes on a tracked asset's order book.

**Zero REST polling.**  The monitor subscribes to BBO-change callbacks
from the local L2 or WS order book.  When a BBO delta arrives for an
asset that has an open EXIT_PENDING position, the engine immediately
evaluates the trailing / fixed stop-loss threshold and fires
``force_stop_loss()`` if breached.

This replaces the previous 500 ms REST-polling loop, eliminating
HTTP 429 rate-limit risk entirely.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from src.core.config import settings
from src.core.logger import get_logger
from src.trading.position_manager import PositionState

if TYPE_CHECKING:
    from src.data.ohlcv import OHLCVAggregator
    from src.data.orderbook import OrderbookTracker
    from src.monitoring.telegram import TelegramAlerter
    from src.monitoring.trade_store import TradeStore
    from src.trading.position_manager import PositionManager

log = get_logger(__name__)


class StopLossMonitor:
    """Event-driven stop-loss monitor — zero polling.

    Instead of a periodic REST poll, the monitor exposes
    :meth:`on_bbo_update` which is invoked by BBO-change callbacks from
    the order-book infrastructure (both ``L2OrderBook`` and
    ``OrderbookTracker``).  Only positions whose ``no_asset_id`` matches
    the changed asset are evaluated, giving O(1) lookup for markets
    without open positions.

    The monitor also supports an optional ``trailing_offset_cents``
    parameter: if the mid-price rallies after entry, the stop-loss
    ratchets upward (never back down), converting a fixed stop into a
    trailing stop.
    """

    def __init__(
        self,
        position_manager: "PositionManager",
        no_aggs: dict[str, "OHLCVAggregator"],
        book_trackers: dict[str, "OrderbookTracker"],
        trade_store: "TradeStore",
        telegram: "TelegramAlerter",
        *,
        trailing_offset_cents: float | None = None,
        on_probe_breakeven: "Any | None" = None,
    ):
        self._pm = position_manager
        self._no_aggs = no_aggs
        self._books = book_trackers
        self._store = trade_store
        self._telegram = telegram
        self._trailing_offset = (
            trailing_offset_cents
            if trailing_offset_cents is not None
            else settings.strategy.trailing_stop_offset_cents
        )
        self._stop_loss_cents = settings.strategy.stop_loss_cents
        self._running = False
        # Trailing high-water marks: pos.id → highest mid-price seen since entry
        self._hwm: dict[str, float] = {}
        # V4: Callback for probe positions that reach breakeven activation
        self._on_probe_breakeven = on_probe_breakeven
        # V4: Track which probes have already emitted breakeven callback
        self._probe_breakeven_emitted: set[str] = set()
        # Breakeven activation: trailing stop only engages after position
        # has been profitable by at least this many cents.  Prevents
        # the trailing stop from ratcheting during the initial adverse
        # move and cutting winners too early.
        self._be_activation_cents = max(
            1.0, self._trailing_offset * 0.5
        ) if self._trailing_offset > 0 else 0.0

    def start(self) -> None:
        """Mark the monitor as active (called once during bot startup)."""
        if self._stop_loss_cents <= 0:
            log.info("stop_loss_monitor_disabled", reason="stop_loss_cents=0")
            return
        self._running = True
        log.info(
            "stop_loss_monitor_started",
            mode="event_driven",
            trailing_offset=self._trailing_offset,
        )

    def stop(self) -> None:
        self._running = False
        log.info("stop_loss_monitor_stopped")

    # ═══════════════════════════════════════════════════════════════════════
    #  Event entry-point — called by BBO-change callbacks
    # ═══════════════════════════════════════════════════════════════════════
    async def on_bbo_update(self, asset_id: str) -> None:
        """Called when the BBO changes for *asset_id*.

        Only evaluates positions whose ``no_asset_id`` matches the
        changed asset — typically 0 or 1 positions, so this is
        extremely cheap when there are no open positions on the asset.
        """
        if not self._running:
            return

        # Fast-path: find EXIT_PENDING positions for this asset
        # Match on trade_asset_id (supports both YES and NO side entries)
        positions_for_asset = [
            p for p in self._pm.get_open_positions()
            if ((p.trade_asset_id or p.no_asset_id) == asset_id or p.no_asset_id == asset_id)
            and p.state == PositionState.EXIT_PENDING
        ]
        if not positions_for_asset:
            return

        # Clean stale HWM entries
        open_ids = {p.id for p in self._pm.get_open_positions()}
        stale = [pid for pid in self._hwm if pid not in open_ids]
        for pid in stale:
            del self._hwm[pid]

        for pos in positions_for_asset:
            await self._evaluate_position(pos)

    async def _evaluate_position(self, pos) -> None:
        """Evaluate stop-loss for a single position."""
        eval_asset = pos.trade_asset_id or pos.no_asset_id
        mid = self._get_mid_price(eval_asset)
        if mid <= 0:
            return

        # Update high-water mark for trailing stop
        hwm = self._hwm.get(pos.id, pos.entry_price)
        if mid > hwm:
            hwm = mid
            self._hwm[pos.id] = hwm

        # ── V4: Probe breakeven callback ──────────────────────────────
        # When a probe position reaches the breakeven activation point,
        # emit a callback so the bot can evaluate scaling into a full
        # position.  Only fires once per probe.
        if (
            getattr(pos, "is_probe", False)
            and self._on_probe_breakeven is not None
            and pos.id not in self._probe_breakeven_emitted
        ):
            profit_cents = (mid - pos.entry_price) * 100
            be_threshold = self._be_activation_cents or 1.0
            if profit_cents >= be_threshold:
                self._probe_breakeven_emitted.add(pos.id)
                log.info(
                    "probe_breakeven_reached",
                    pos_id=pos.id,
                    profit_cents=round(profit_cents, 2),
                    threshold=be_threshold,
                )
                try:
                    await self._on_probe_breakeven(pos)
                except Exception:
                    log.error("probe_breakeven_callback_error", pos_id=pos.id, exc_info=True)

        # ── Pillar 11.3: Preemptive Liquidity Drain ──────────────────
        # If the support side of the book has evaporated relative to
        # resistance AND the position is underwater, exit preemptively
        # before slippage makes a standard stop-loss too expensive.
        strat = settings.strategy
        obi_threshold = strat.sl_preemptive_obi_threshold
        if obi_threshold > 0:
            unrealised_pnl = (mid - pos.entry_price) * 100  # cents
            if unrealised_pnl < 0:
                book = self._books.get(eval_asset)
                if book and book.has_data:
                    bdr = book.book_depth_ratio  # bid/ask
                    # For BUY_NO positions our exit is a SELL whose
                    # fill quality depends on bid depth.
                    # bdr = bid/ask: when bdr → 0 bid support vanishes
                    # and our exit liquidity is draining.
                    our_support_ratio = bdr
                    if our_support_ratio < obi_threshold:
                        log.warning(
                            "preemptive_liquidity_drain",
                            pos_id=pos.id,
                            support_ratio=round(our_support_ratio, 4),
                            threshold=obi_threshold,
                            unrealised_pnl=round(unrealised_pnl, 2),
                        )
                        await self._trigger_stop(
                            pos, mid, 0.0,
                            reason="preemptive_liquidity_drain",
                        )
                        return

        # Determine effective stop-loss threshold
        base_sl = (
            pos.sl_trigger_cents
            if pos.sl_trigger_cents > 0
            else self._stop_loss_cents
        )

        # ── Pillar 11.3: Time-Decay Stop Tightening ──────────────────
        # If the trade has been open longer than sl_decay_start_minutes,
        # exponentially decay the vol multiplier back toward 1.0,
        # tightening the stop to exit stale positions.
        decay_start = strat.sl_decay_start_minutes
        decay_hl = strat.sl_decay_half_life_minutes
        sl_vol_mult = getattr(pos, "sl_vol_multiplier", 1.0)
        if sl_vol_mult > 1.0 and decay_start > 0 and decay_hl > 0:
            elapsed_mins = (time.time() - pos.entry_time) / 60.0
            if elapsed_mins > decay_start:
                decay_factor = math.exp(
                    -math.log(2) * (elapsed_mins - decay_start) / decay_hl
                )
                current_mult = 1.0 + (sl_vol_mult - 1.0) * decay_factor
                # Fee drag is the difference between the stretched stop
                # and the stored trigger (constant over the trade's life).
                fee_drag = self._stop_loss_cents * sl_vol_mult - base_sl
                base_sl = max(
                    1.0,
                    self._stop_loss_cents * current_mult - fee_drag,
                )

        # Use per-position adaptive trailing offset if set, else fall back
        # to the monitor-level default (from config or constructor arg).
        effective_trailing = (
            pos.trailing_offset_cents
            if getattr(pos, "trailing_offset_cents", 0.0) > 0
            else self._trailing_offset
        )

        if effective_trailing > 0 and hwm > pos.entry_price:
            # Trailing stop: only activate after the position has been
            # profitable by at least _be_activation_cents.  This prevents
            # ratcheting during the initial adverse move from cutting
            # winners before they reach target.
            profit_from_entry = (hwm - pos.entry_price) * 100
            if profit_from_entry >= self._be_activation_cents:
                trail_floor_price = hwm - effective_trailing / 100.0
                trail_loss = (trail_floor_price - mid) * 100
                entry_loss = (pos.entry_price - mid) * 100
                unrealised_loss = max(trail_loss, entry_loss)
            else:
                # Not yet profitable enough to trail — use fixed stop only
                unrealised_loss = (pos.entry_price - mid) * 100
        else:
            unrealised_loss = (pos.entry_price - mid) * 100

        if unrealised_loss >= base_sl:
            await self._trigger_stop(pos, mid, base_sl, reason="stop_loss")

    # ═══════════════════════════════════════════════════════════════════════
    #  Backward-compatible batch check (used by tests)
    # ═══════════════════════════════════════════════════════════════════════
    async def _check_once(self, stop_loss_cents: float) -> None:
        """Evaluate all EXIT_PENDING positions.

        Retained for test backward-compatibility.  The live path uses
        :meth:`on_bbo_update` which is driven by BBO callbacks.
        """

        # Temporarily override the configured SL for this invocation
        saved = self._stop_loss_cents
        self._stop_loss_cents = stop_loss_cents
        saved_running = self._running
        self._running = True

        try:
            open_positions = self._pm.get_open_positions()

            # Clean stale HWM entries
            open_ids = {p.id for p in open_positions}
            stale = [pid for pid in self._hwm if pid not in open_ids]
            for pid in stale:
                del self._hwm[pid]

            for pos in open_positions:
                if pos.state != PositionState.EXIT_PENDING:
                    continue
                await self._evaluate_position(pos)
        finally:
            self._stop_loss_cents = saved
            self._running = saved_running

    # ═══════════════════════════════════════════════════════════════════════
    #  Internals
    # ═══════════════════════════════════════════════════════════════════════
    def _get_mid_price(self, asset_id: str) -> float:
        """Best source of current price: orderbook mid → last trade."""
        book = self._books.get(asset_id)
        if book and book.has_data:
            bb = book.best_bid
            ba = book.best_ask
            if bb > 0 and ba > 0:
                return (bb + ba) / 2.0

        agg = self._no_aggs.get(asset_id)
        if agg and agg.current_price > 0:
            return agg.current_price

        return 0.0

    async def _trigger_stop(self, pos, mid: float, threshold: float, *, reason: str = "stop_loss") -> None:
        if pos.state != PositionState.EXIT_PENDING:
            return  # may have been closed between check and trigger

        log.warning(
            "active_stop_loss_triggered",
            pos_id=pos.id,
            entry=pos.entry_price,
            mid=mid,
            threshold=threshold,
            hwm=self._hwm.get(pos.id, 0.0),
            reason=reason,
        )

        await self._pm.force_stop_loss(pos, reason=reason)
        await self._store.record(pos)
        await self._telegram.notify_exit(
            pos.id, pos.entry_price, pos.exit_price,
            pos.pnl_cents, reason,
        )
