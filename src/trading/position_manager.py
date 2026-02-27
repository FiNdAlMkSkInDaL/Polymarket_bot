"""
Position manager — tracks open positions, orchestrates the entry → exit
lifecycle, and enforces risk limits.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.config import settings
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator
from src.signals.panic_detector import PanicSignal
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.take_profit import TakeProfitResult, compute_take_profit

log = get_logger(__name__)


class PositionState(str, Enum):
    ENTRY_PENDING = "ENTRY_PENDING"
    ENTRY_FILLED = "ENTRY_FILLED"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class Position:
    """Tracks a single round-trip trade (entry + exit)."""

    id: str
    market_id: str
    no_asset_id: str

    state: PositionState = PositionState.ENTRY_PENDING

    # Entry
    entry_order: Order | None = None
    entry_price: float = 0.0
    entry_size: float = 0.0
    entry_time: float = 0.0

    # Exit
    exit_order: Order | None = None
    target_price: float = 0.0
    tp_result: TakeProfitResult | None = None
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""

    # Signal metadata
    signal: PanicSignal | None = None

    # PnL
    pnl_cents: float = 0.0

    created_at: float = field(default_factory=time.time)


class PositionManager:
    """Orchestrates the full lifecycle of mean-reversion trades.

    Workflow:
      1. A PanicSignal fires → open_position() is called.
      2. A GTC limit BUY for NO is placed at a discount.
      3. When the entry fills → compute take-profit → place limit SELL.
      4. When the exit fills or times out → position is closed.
    """

    def __init__(
        self,
        executor: OrderExecutor,
        *,
        max_open_positions: int = 3,
    ):
        self.executor = executor
        self.max_open = max_open_positions
        self._positions: dict[str, Position] = {}
        self._next_id = 1
        self._wallet_balance_usd: float = 0.0  # Updated externally

    # ── Wallet balance (simplified — would query on-chain in production) ───
    def set_wallet_balance(self, usd: float) -> None:
        self._wallet_balance_usd = usd

    # ── Open a new position on signal ──────────────────────────────────────
    async def open_position(
        self,
        signal: PanicSignal,
        no_aggregator: OHLCVAggregator,
    ) -> Position | None:
        """Attempt to open a mean-reversion position on a panic signal."""

        # Risk gate: max open positions
        open_count = sum(
            1 for p in self._positions.values() if p.state in (
                PositionState.ENTRY_PENDING, PositionState.EXIT_PENDING,
            )
        )
        if open_count >= self.max_open:
            log.warning("max_positions_reached", open=open_count)
            return None

        # Risk gate: capital allocation
        max_trade = min(
            settings.strategy.max_trade_size_usd,
            self._wallet_balance_usd * settings.strategy.max_wallet_risk_pct / 100.0,
        )
        if max_trade <= 0:
            log.warning("insufficient_balance", balance=self._wallet_balance_usd)
            return None

        # Entry price: undercut the best ask by 1¢ to ensure maker
        entry_price = round(signal.no_best_ask - 0.01, 2)
        if entry_price <= 0:
            log.warning("entry_price_invalid", ask=signal.no_best_ask)
            return None

        # Size: how many shares can we buy?
        entry_size = round(max_trade / entry_price, 2)
        if entry_size < 1:
            entry_size = 1.0  # minimum 1 share

        # Pre-check: will the take-profit be viable?
        tp = compute_take_profit(
            entry_price=entry_price,
            no_vwap=no_aggregator.rolling_vwap,
            realised_vol=no_aggregator.rolling_volatility,
            whale_confluence=signal.whale_confluence,
        )
        if not tp.viable:
            log.info(
                "skip_entry_low_spread",
                spread=tp.spread_cents,
                min_required=settings.strategy.min_spread_cents,
            )
            return None

        # Place entry order
        pos_id = f"POS-{self._next_id}"
        self._next_id += 1

        order = await self.executor.place_limit_order(
            market_id=signal.market_id,
            asset_id=signal.no_asset_id,
            side=OrderSide.BUY,
            price=entry_price,
            size=entry_size,
        )

        pos = Position(
            id=pos_id,
            market_id=signal.market_id,
            no_asset_id=signal.no_asset_id,
            state=PositionState.ENTRY_PENDING,
            entry_order=order,
            entry_price=entry_price,
            entry_size=entry_size,
            entry_time=time.time(),
            signal=signal,
            tp_result=tp,
            target_price=tp.target_price,
        )

        self._positions[pos.id] = pos

        log.info(
            "position_opened",
            pos_id=pos.id,
            market=signal.market_id,
            entry=entry_price,
            size=entry_size,
            target=tp.target_price,
            alpha=tp.alpha,
        )

        return pos

    # ── Handle entry fill ──────────────────────────────────────────────────
    async def on_entry_filled(self, pos: Position) -> None:
        """Called when the entry buy order is filled.  Places the exit sell."""
        pos.state = PositionState.ENTRY_FILLED
        pos.entry_price = pos.entry_order.filled_avg_price or pos.entry_price

        # Place exit limit sell at the computed target
        exit_order = await self.executor.place_limit_order(
            market_id=pos.market_id,
            asset_id=pos.no_asset_id,
            side=OrderSide.SELL,
            price=pos.target_price,
            size=pos.entry_size,
        )

        pos.exit_order = exit_order
        pos.state = PositionState.EXIT_PENDING

        log.info(
            "exit_order_placed",
            pos_id=pos.id,
            target=pos.target_price,
            size=pos.entry_size,
        )

    # ── Handle exit fill ───────────────────────────────────────────────────
    def on_exit_filled(self, pos: Position, reason: str = "target") -> None:
        """Close the position after exit fill or forced liquidation."""
        pos.state = PositionState.CLOSED
        pos.exit_price = (
            pos.exit_order.filled_avg_price if pos.exit_order else pos.target_price
        )
        pos.exit_time = time.time()
        pos.exit_reason = reason
        pos.pnl_cents = round((pos.exit_price - pos.entry_price) * pos.entry_size * 100, 2)

        log.info(
            "position_closed",
            pos_id=pos.id,
            entry=pos.entry_price,
            exit=pos.exit_price,
            pnl_cents=pos.pnl_cents,
            reason=reason,
            hold_seconds=round(pos.exit_time - pos.entry_time, 1),
        )

    # ── Timeout enforcement ────────────────────────────────────────────────
    async def check_timeouts(self) -> None:
        """Cancel stale entry orders and force-exit stale positions."""
        now = time.time()

        for pos in list(self._positions.values()):
            # Entry timeout
            if pos.state == PositionState.ENTRY_PENDING:
                elapsed = now - pos.entry_time
                if elapsed > settings.strategy.entry_timeout_seconds:
                    if pos.entry_order:
                        await self.executor.cancel_order(pos.entry_order)
                    pos.state = PositionState.CANCELLED
                    pos.exit_reason = "entry_timeout"
                    log.info("entry_timeout", pos_id=pos.id, elapsed_s=round(elapsed))

            # Exit timeout — force market sell
            elif pos.state == PositionState.EXIT_PENDING:
                elapsed = now - pos.entry_time
                if elapsed > settings.strategy.exit_timeout_seconds:
                    # Cancel the existing limit sell
                    if pos.exit_order:
                        await self.executor.cancel_order(pos.exit_order)

                    # Place a market sell (IOC at best bid — simulated by
                    # placing at a very low price in paper mode)
                    exit_order = await self.executor.place_limit_order(
                        market_id=pos.market_id,
                        asset_id=pos.no_asset_id,
                        side=OrderSide.SELL,
                        price=0.01,  # effectively market sell
                        size=pos.entry_size,
                    )
                    pos.exit_order = exit_order

                    # In paper mode, this will fill instantly on next tick
                    # In live mode, the CLOB IOC logic handles it
                    self.on_exit_filled(pos, reason="timeout")

    # ── Query helpers ───────────────────────────────────────────────────────
    def get_open_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.state in (
                PositionState.ENTRY_PENDING,
                PositionState.ENTRY_FILLED,
                PositionState.EXIT_PENDING,
            )
        ]

    def get_all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_closed_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.state == PositionState.CLOSED]
