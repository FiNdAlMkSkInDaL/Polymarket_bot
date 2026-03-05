"""
Stealth Executor — time-sliced order splitting to reduce market footprint.

Problem
───────
Placing a single large limit order (~$15) on a thin Polymarket book
signals intent and invites front-running.  Informed participants can
observe the resting order and trade ahead.

Solution
────────
Split the parent order into N child slices with:
  - Randomised inter-slice delay (uniform within a configurable range)
  - Slightly varied slice sizes (±jitter_pct around the mean)
  - Optional price improvement on later slices if the book moves

The ``StealthExecutor`` wraps the existing ``OrderExecutor`` and is
designed to be a drop-in replacement for ``place_limit_order()`` when
stealth behaviour is desired (orders above ``stealth_min_size_usd``).

Smaller orders (below the minimum) pass through to the underlying
executor unchanged.

Thread-safety: single-threaded asyncio — no locks needed.

Usage (in bot.py or position_manager.py)::

    stealth = StealthExecutor(executor)
    orders = await stealth.place_stealth_order(
        market_id, asset_id, side, price, total_size,
    )
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable

from src.core.config import settings
from src.core.logger import get_logger
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus

log = get_logger(__name__)


@dataclass(slots=True)
class StealthPlan:
    """Summary of a stealth execution plan (for logging/telemetry)."""

    parent_size: float
    num_slices: int
    slice_sizes: list[float]
    delays_ms: list[float]
    price: float


class StealthExecutor:
    """Wraps ``OrderExecutor`` with time-sliced order splitting.

    Parameters
    ----------
    executor:
        The underlying ``OrderExecutor`` for actual order placement.
    min_size_usd:
        Minimum parent order size (in $) to trigger slicing.  Orders
        below this threshold pass through un-split.
    max_slices:
        Maximum number of child slices.
    min_delay_ms:
        Minimum inter-slice delay in milliseconds.
    max_delay_ms:
        Maximum inter-slice delay in milliseconds.
    size_jitter_pct:
        Maximum fractional randomisation of slice sizes (e.g. 0.15 =
        ±15%).  The total always sums to the parent size.
    """

    def __init__(
        self,
        executor: OrderExecutor,
        *,
        min_size_usd: float = 0.0,
        max_slices: int = 0,
        min_delay_ms: float = 0.0,
        max_delay_ms: float = 0.0,
        size_jitter_pct: float = 0.0,
    ):
        strat = settings.strategy
        self._executor = executor
        self._min_size = min_size_usd or getattr(strat, "stealth_min_size_usd", 5.0)
        self._max_slices = max_slices or getattr(strat, "stealth_max_slices", 4)
        self._min_delay = min_delay_ms or getattr(strat, "stealth_min_delay_ms", 200.0)
        self._max_delay = max_delay_ms or getattr(strat, "stealth_max_delay_ms", 1500.0)
        self._size_jitter = size_jitter_pct or getattr(strat, "stealth_size_jitter_pct", 0.15)
        # Abandonment controls
        self._abandon_drift_cents = getattr(strat, "stealth_abandon_drift_cents", 2.0)
        self._abandon_fill_pct = getattr(strat, "stealth_abandon_fill_pct", 0.75)

    # ── Public API ─────────────────────────────────────────────────────────

    async def place_stealth_order(
        self,
        market_id: str,
        asset_id: str,
        side: OrderSide,
        price: float,
        total_size: float,
        *,
        post_only: bool = False,
        fee_rate_bps: int = 0,
        get_mid_fn: Callable[[], float | None] | None = None,
    ) -> list[Order]:
        """Place a parent order as time-sliced child orders.

        If ``total_size * price`` is below ``min_size_usd``, the order
        is placed as a single un-split order.

        Parameters
        ----------
        get_mid_fn:
            Optional callable returning the current mid-price.  When
            supplied, the executor checks for adverse drift between
            slices and abandons remaining slices if the mid has moved
            more than ``stealth_abandon_drift_cents`` from *price*.

        Returns the list of all child Order objects.
        """
        notional = total_size * price
        if notional < self._min_size:
            order = await self._executor.place_limit_order(
                market_id, asset_id, side, price, total_size,
                post_only=post_only, fee_rate_bps=fee_rate_bps,
            )
            return [order]

        plan = self._build_plan(total_size, price)

        log.info(
            "stealth_exec_start",
            asset_id=asset_id[:16],
            side=side.value,
            price=price,
            total_size=total_size,
            slices=plan.num_slices,
            notional=round(notional, 2),
        )

        orders: list[Order] = []
        filled_size = 0.0
        for i, (slice_size, delay_ms) in enumerate(
            zip(plan.slice_sizes, plan.delays_ms)
        ):
            order = await self._executor.place_limit_order(
                market_id, asset_id, side, price, slice_size,
                post_only=post_only, fee_rate_bps=fee_rate_bps,
            )
            orders.append(order)

            if order.status == OrderStatus.CANCELLED:
                log.warning(
                    "stealth_slice_rejected",
                    slice_idx=i,
                    reason=order.rejection_reason,
                )
                break

            # Track filled volume for abandonment check
            if order.status == OrderStatus.FILLED:
                filled_size += order.size

            # Delay before next slice (skip delay after last slice)
            if i < plan.num_slices - 1:
                # ── Abandonment check: filled-pct ─────────────────
                if (
                    filled_size > 0
                    and total_size > 0
                    and filled_size / total_size >= self._abandon_fill_pct
                ):
                    log.info(
                        "stealth_abandoned_fill_pct",
                        asset_id=asset_id[:16],
                        filled_pct=round(filled_size / total_size, 3),
                        threshold=self._abandon_fill_pct,
                        slices_done=i + 1,
                        slices_total=plan.num_slices,
                    )
                    break

                await asyncio.sleep(delay_ms / 1000.0)

                # ── Abandonment check: mid-price drift ────────────
                if get_mid_fn is not None:
                    mid = get_mid_fn()
                    if mid is not None and mid > 0:
                        if side == OrderSide.BUY:
                            drift_cents = (mid - price) * 100.0
                        else:
                            drift_cents = (price - mid) * 100.0
                        if drift_cents > self._abandon_drift_cents:
                            log.info(
                                "stealth_abandoned_drift",
                                asset_id=asset_id[:16],
                                drift_cents=round(drift_cents, 2),
                                threshold=self._abandon_drift_cents,
                                mid=round(mid, 4),
                                anchor=round(price, 4),
                                slices_done=i + 1,
                                slices_total=plan.num_slices,
                            )
                            break

        placed = sum(1 for o in orders if o.status == OrderStatus.LIVE)
        log.info(
            "stealth_exec_done",
            asset_id=asset_id[:16],
            placed=placed,
            total_slices=plan.num_slices,
            total_size=round(sum(o.size for o in orders), 4),
        )

        return orders

    def _build_plan(self, total_size: float, price: float) -> StealthPlan:
        """Compute slice sizes and delays for a stealth execution plan."""
        # Determine number of slices (proportional to notional)
        notional = total_size * price
        # 1 slice per $3, capped at max_slices, min 2
        n = max(2, min(self._max_slices, int(notional / 3.0) + 1))

        # Build slice sizes with jitter
        base_size = total_size / n
        raw_sizes: list[float] = []
        for _ in range(n):
            jitter = random.uniform(-self._size_jitter, self._size_jitter)
            raw_sizes.append(base_size * (1.0 + jitter))

        # Normalise so they sum exactly to total_size
        raw_sum = sum(raw_sizes)
        slice_sizes = [s * (total_size / raw_sum) for s in raw_sizes]

        # Ensure min size per slice ($0.10 to avoid dust orders)
        min_slice = 0.10 / max(price, 0.01)
        slice_sizes = [max(s, min_slice) for s in slice_sizes]
        # Re-normalise after floor
        raw_sum = sum(slice_sizes)
        slice_sizes = [round(s * (total_size / raw_sum), 2) for s in slice_sizes]

        # Fix rounding residual on the last slice
        placed_sum = sum(slice_sizes[:-1])
        slice_sizes[-1] = round(total_size - placed_sum, 2)

        # Build delays
        delays_ms = [
            round(random.uniform(self._min_delay, self._max_delay), 0)
            for _ in range(n)
        ]

        return StealthPlan(
            parent_size=total_size,
            num_slices=n,
            slice_sizes=slice_sizes,
            delays_ms=delays_ms,
            price=price,
        )

    @property
    def executor(self) -> OrderExecutor:
        """Access the underlying executor."""
        return self._executor
