"""Aggressive execution adapter for fleeting OFI momentum signals."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.core.logger import get_logger
from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus

log = get_logger(__name__)

DEFAULT_MOMENTUM_TP_PCT = 0.03
DEFAULT_MOMENTUM_SL_PCT = 0.015
DEFAULT_MOMENTUM_MAX_HOLD_SECONDS = 300
_MOMENTUM_TP_LOW_SCALE = 0.60
_MOMENTUM_TP_HIGH_SCALE = 1.50
_MOMENTUM_SL_LOW_SCALE = 0.70
_MOMENTUM_SL_HIGH_SCALE = 1.40
_MOMENTUM_HOLD_LOW_SCALE = 0.50
_MOMENTUM_HOLD_HIGH_SCALE = 1.50


@dataclass(frozen=True)
class MomentumBracket:
    """Hard bailout brackets for OFI momentum entries."""

    take_profit_pct: float = DEFAULT_MOMENTUM_TP_PCT
    stop_loss_pct: float = DEFAULT_MOMENTUM_SL_PCT
    max_hold_seconds: int = DEFAULT_MOMENTUM_MAX_HOLD_SECONDS

    def target_price(self, entry_price: float) -> float:
        return round(min(0.99, entry_price * (1.0 + self.take_profit_pct)), 2)

    def stop_price(self, entry_price: float) -> float:
        return round(max(0.01, entry_price * (1.0 - self.stop_loss_pct)), 4)

    def stop_loss_cents(self, entry_price: float) -> float:
        return round((entry_price - self.stop_price(entry_price)) * 100.0, 4)


@dataclass(frozen=True)
class DrawnMomentumBracket:
    """Randomized OFI bracket kept private to the strategy runtime."""

    take_profit_pct: float
    stop_loss_pct: float
    max_hold_seconds: float

    def target_price(self, entry_price: float) -> float:
        target = min(0.99, entry_price * (1.0 + self.take_profit_pct))
        return round(max(entry_price + 0.01, target), 2)

    def stop_price(self, entry_price: float) -> float:
        return round(max(0.01, entry_price * (1.0 - self.stop_loss_pct)), 4)

    def stop_loss_cents(self, entry_price: float) -> float:
        return round((entry_price - self.stop_price(entry_price)) * 100.0, 4)


def _draw_hazard_bounded(
    rng: random.Random,
    *,
    mean_value: float,
    low_scale: float,
    high_scale: float,
    minimum_value: float,
) -> float:
    mean_value = max(minimum_value, float(mean_value))
    low_value = max(minimum_value, mean_value * low_scale)
    high_value = max(low_value, mean_value * high_scale)
    if high_value <= low_value:
        return round(low_value, 6)

    residual_mean = max(mean_value - low_value, 1e-9)
    draw = low_value + min(rng.expovariate(1.0 / residual_mean), high_value - low_value)
    return round(min(high_value, max(low_value, draw)), 6)


def draw_stochastic_momentum_bracket(
    *,
    mean_take_profit_pct: float = DEFAULT_MOMENTUM_TP_PCT,
    mean_stop_loss_pct: float = DEFAULT_MOMENTUM_SL_PCT,
    mean_max_hold_seconds: float = DEFAULT_MOMENTUM_MAX_HOLD_SECONDS,
    rng: random.Random | None = None,
) -> DrawnMomentumBracket:
    """Draw a bounded hazard-style OFI bracket around the configured means."""
    rng = rng or random.Random()
    return DrawnMomentumBracket(
        take_profit_pct=_draw_hazard_bounded(
            rng,
            mean_value=mean_take_profit_pct,
            low_scale=_MOMENTUM_TP_LOW_SCALE,
            high_scale=_MOMENTUM_TP_HIGH_SCALE,
            minimum_value=0.005,
        ),
        stop_loss_pct=_draw_hazard_bounded(
            rng,
            mean_value=mean_stop_loss_pct,
            low_scale=_MOMENTUM_SL_LOW_SCALE,
            high_scale=_MOMENTUM_SL_HIGH_SCALE,
            minimum_value=0.005,
        ),
        max_hold_seconds=_draw_hazard_bounded(
            rng,
            mean_value=max(30.0, mean_max_hold_seconds),
            low_scale=_MOMENTUM_HOLD_LOW_SCALE,
            high_scale=_MOMENTUM_HOLD_HIGH_SCALE,
            minimum_value=30.0,
        ),
    )


class MomentumTakerExecutor:
    """Places immediate spread-crossing entry orders for momentum signals."""

    def __init__(self, executor: OrderExecutor):
        self._executor = executor

    async def place_buy(
        self,
        *,
        market_id: str,
        asset_id: str,
        size: float,
        best_ask: float,
        signal_fired_at: float | None = None,
    ) -> tuple[Order, float]:
        entry_price = round(best_ask, 2)
        if entry_price <= 0.0 or entry_price >= 1.0:
            raise ValueError(f"invalid momentum taker entry price: {best_ask!r}")

        order = await self._executor.place_limit_order(
            market_id=market_id,
            asset_id=asset_id,
            side=OrderSide.BUY,
            price=entry_price,
            size=size,
            signal_fired_at=signal_fired_at,
        )

        if self._executor.paper_mode and order.status == OrderStatus.LIVE:
            order.status = OrderStatus.FILLED
            order.filled_size = size
            order.filled_avg_price = entry_price
            order.updated_at = time.time()
            self._executor._open_count = max(0, self._executor._open_count - 1)
            log.info(
                "momentum_taker_paper_fill",
                order_id=order.order_id,
                market_id=market_id,
                asset_id=asset_id,
                entry_price=entry_price,
                size=size,
            )

        return order, entry_price