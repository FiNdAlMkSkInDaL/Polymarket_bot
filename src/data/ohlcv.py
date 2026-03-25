"""
OHLCV bar aggregator — converts raw TradeEvents from the WebSocket into
1-minute OHLCV bars and maintains rolling statistics (VWAP, σ) needed by
the panic-spike detector.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

from src.core.config import settings
from src.core.logger import get_logger
from src.data.websocket_client import TradeEvent

log = get_logger(__name__)

BAR_INTERVAL = 60  # seconds


@dataclass
class OHLCVBar:
    """One completed 1-minute bar."""

    open_time: float   # unix ts at bar open
    open: float
    high: float
    low: float
    close: float
    volume: float      # total share volume
    vwap: float        # volume-weighted avg price for this bar
    trade_count: int


@dataclass
class _BarBuilder:
    """Accumulates ticks within the current bar interval."""

    open_time: float = 0.0
    _true_open: float = 0.0       # preserved across truncation
    _total_volume: float = 0.0
    _weighted_notional: float = 0.0
    _high: float = 0.0
    _low: float = 0.0
    _last_price: float = 0.0
    _trade_count: int = 0

    def add(self, price: float, size: float, ts: float) -> None:
        if self._trade_count == 0:
            self.open_time = ts
            self._true_open = price
            self._total_volume = 0.0
            self._weighted_notional = 0.0
            self._high = price
            self._low = price
        else:
            self._high = max(self._high, price)
            self._low = min(self._low, price)

        self._total_volume += size
        self._weighted_notional += price * size
        self._last_price = price
        self._trade_count += 1

    def finalise(self) -> OHLCVBar | None:
        if self._trade_count == 0:
            return None
        vwap = (
            self._weighted_notional / self._total_volume
            if self._total_volume > 0
            else self._last_price
        )
        return OHLCVBar(
            open_time=self.open_time,
            open=self._true_open if self._true_open > 0 else self._last_price,
            high=self._high,
            low=self._low,
            close=self._last_price,
            volume=self._total_volume,
            vwap=vwap,
            trade_count=self._trade_count,
        )

    def reset(self) -> None:
        self.open_time = 0.0
        self._true_open = 0.0
        self._total_volume = 0.0
        self._weighted_notional = 0.0
        self._high = 0.0
        self._low = 0.0
        self._last_price = 0.0
        self._trade_count = 0


class OHLCVAggregator:
    """Per-market aggregator that produces rolling bars and stats.

    Call :meth:`on_trade` for every incoming TradeEvent with matching
    ``asset_id``.  When a bar closes, it is appended to the history deque
    and the rolling VWAP & volatility are updated.
    """

    def __init__(
        self,
        asset_id: str,
        lookback_minutes: int | None = None,
    ):
        self.asset_id = asset_id
        self.lookback = lookback_minutes or 60

        # Rolling bar history (bounded by lookback window)
        self.bars: Deque[OHLCVBar] = deque(maxlen=self.lookback + 5)

        # Current in-progress bar
        self._builder = _BarBuilder()
        self._bar_start: float = 0.0

        # Pre-computed stats (updated on each bar close)
        self.rolling_vwap: float = 0.0
        self.rolling_volatility: float = 0.0
        self.rolling_volatility_30m: float = 0.0  # 30-min window for TP rescaling
        self.rolling_volatility_ewma: float = 0.0  # EWMA σ (fast-reacting)
        self._ewma_variance: float = 0.0  # internal EWMA variance state
        self._ewma_initialised: bool = False
        self.rolling_downside_vol_ewma: float = 0.0  # Downside semi-variance EWMA σ
        self._downside_ewma_variance: float = 0.0
        self._downside_ewma_initialised: bool = False
        self.avg_bar_volume: float = 0.0
        self._stat_bars: Deque[OHLCVBar] = deque()
        self._rolling_volume_sum: float = 0.0
        self._rolling_vwap_volume_sum: float = 0.0
        self._rolling_vwap_sum: float = 0.0
        self._close_window: Deque[float] = deque()
        self._return_window: Deque[float] = deque()
        self._return_sum: float = 0.0
        self._return_sq_sum: float = 0.0
        self._close_30_window: Deque[float] = deque()
        self._return_30_window: Deque[float] = deque()
        self._return_30_sum: float = 0.0
        self._return_30_sq_sum: float = 0.0
        self._taker_flow_window: Deque[tuple[int, float, float]] = deque()
        self._taker_buy_volume: float = 0.0
        self._taker_sell_volume: float = 0.0

        # Most recent trade timestamp — used for data freshness checks.
        # Unlike bar open_time, this is updated on EVERY trade.
        self.last_trade_time: float = 0.0

    # ── public API ──────────────────────────────────────────────────────────
    def on_trade(self, event: TradeEvent) -> OHLCVBar | None:
        """Ingest a trade tick.  Returns a completed bar if one just closed."""
        now = event.timestamp
        self.last_trade_time = now
        self._record_taker_flow(event)

        # Initialise bar window on first trade
        if self._bar_start == 0.0:
            self._bar_start = now

        # Check if the current bar interval has elapsed
        if now - self._bar_start >= BAR_INTERVAL:
            closed_bar = self._close_bar()
            self._bar_start = now
            self._builder.add(event.price, event.size, now)
            return closed_bar

        self._builder.add(event.price, event.size, now)
        return None

    def flush_stale_bar(self, now: float | None = None) -> OHLCVBar | None:
        """Close the current bar if it has been open longer than BAR_INTERVAL.

        Call this periodically (e.g. every 30s) to ensure bars close even
        when no trades arrive.  On low-volume markets this prevents VWAP
        and σ from becoming frozen indefinitely.  Returns the closed bar,
        or ``None`` if the bar is still within its interval or empty.
        """
        if now is None:
            now = time.time()
        if self._bar_start <= 0:
            return None
        if self._builder._trade_count == 0:
            # No ticks accumulated — nothing to close.  Advance the
            # bar window so the next trade starts a fresh interval.
            if now - self._bar_start >= BAR_INTERVAL:
                self._bar_start = now
            return None
        if now - self._bar_start < BAR_INTERVAL:
            return None

        closed = self._close_bar()
        self._bar_start = now
        return closed

    @property
    def current_bar_volume(self) -> float:
        """Volume accumulated in the current (not yet closed) bar."""
        return self._builder._total_volume

    @property
    def current_price(self) -> float:
        """Last traded price seen by this aggregator."""
        if self._builder._trade_count > 0:
            return self._builder._last_price
        if self.bars:
            return self.bars[-1].close
        return 0.0

    def trade_flow_moments(
        self,
        window_ms: int,
        *,
        current_time_ms: int | None = None,
    ) -> tuple[float, float]:
        """Return rolling taker buy/sell volume over the requested window."""
        if window_ms <= 0:
            return 0.0, 0.0
        now_ms = self._coerce_time_ms(current_time_ms)
        self._prune_taker_flow(now_ms=now_ms, window_ms=window_ms)
        return self._taker_buy_volume, self._taker_sell_volume

    def trade_flow_imbalance(
        self,
        window_ms: int,
        *,
        current_time_ms: int | None = None,
    ) -> float:
        """Return rolling taker-flow imbalance in [-1, 1]."""
        buy_volume, sell_volume = self.trade_flow_moments(
            window_ms,
            current_time_ms=current_time_ms,
        )
        total = buy_volume + sell_volume
        if total <= 0:
            return 0.0
        return (buy_volume - sell_volume) / total

    # ── internals ───────────────────────────────────────────────────────────
    def _close_bar(self) -> OHLCVBar | None:
        bar = self._builder.finalise()
        self._builder.reset()
        if bar is None:
            return None

        self.bars.append(bar)
        self._append_bar(bar)
        self._recompute_stats()
        return bar

    @staticmethod
    def _cap_log_return(prev_close: float, close: float) -> float:
        prev = max(prev_close, 1e-8)
        curr = max(close, 1e-8)
        return max(-1.1, min(1.1, math.log(curr) - math.log(prev)))

    @staticmethod
    def _std_from_moments(sum_value: float, sq_sum_value: float, count: int) -> float:
        if count <= 1:
            return 0.0
        mean = sum_value / count
        variance = max(0.0, (sq_sum_value / count) - (mean * mean))
        return variance ** 0.5

    @staticmethod
    def _coerce_time_ms(timestamp_ms: int | None) -> int:
        if timestamp_ms is not None:
            return int(timestamp_ms)
        return int(time.time() * 1000)

    def _record_taker_flow(self, event: TradeEvent) -> None:
        if not getattr(event, "is_taker", False):
            return

        side = str(getattr(event, "side", "") or "").lower()
        buy_volume = 0.0
        sell_volume = 0.0
        if side == "buy":
            buy_volume = float(event.size or 0.0)
        elif side == "sell":
            sell_volume = float(event.size or 0.0)
        else:
            return

        now_ms = int(float(event.timestamp) * 1000)
        self._taker_flow_window.append((now_ms, buy_volume, sell_volume))
        self._taker_buy_volume += buy_volume
        self._taker_sell_volume += sell_volume

    def _prune_taker_flow(self, *, now_ms: int, window_ms: int) -> None:
        cutoff = now_ms - window_ms
        while self._taker_flow_window and self._taker_flow_window[0][0] < cutoff:
            _, expired_buy, expired_sell = self._taker_flow_window.popleft()
            self._taker_buy_volume -= expired_buy
            self._taker_sell_volume -= expired_sell

    def _append_bar(self, bar: OHLCVBar) -> None:
        self._stat_bars.append(bar)
        self._rolling_volume_sum += bar.volume
        self._rolling_vwap_volume_sum += bar.vwap * bar.volume
        self._rolling_vwap_sum += bar.vwap
        if len(self._stat_bars) > self.lookback:
            expired = self._stat_bars.popleft()
            self._rolling_volume_sum -= expired.volume
            self._rolling_vwap_volume_sum -= expired.vwap * expired.volume
            self._rolling_vwap_sum -= expired.vwap

        self._append_close(
            bar.close,
            close_window=self._close_window,
            close_limit=self.lookback,
            return_window=self._return_window,
            sum_attr="_return_sum",
            sq_sum_attr="_return_sq_sum",
        )
        self._append_close(
            bar.close,
            close_window=self._close_30_window,
            close_limit=30,
            return_window=self._return_30_window,
            sum_attr="_return_30_sum",
            sq_sum_attr="_return_30_sq_sum",
        )

    def _append_close(
        self,
        close: float,
        *,
        close_window: Deque[float],
        close_limit: int,
        return_window: Deque[float],
        sum_attr: str,
        sq_sum_attr: str,
    ) -> None:
        if close_window:
            log_return = self._cap_log_return(close_window[-1], close)
            return_window.append(log_return)
            setattr(self, sum_attr, getattr(self, sum_attr) + log_return)
            setattr(self, sq_sum_attr, getattr(self, sq_sum_attr) + (log_return * log_return))

        close_window.append(close)
        if len(close_window) > close_limit:
            close_window.popleft()
            if return_window:
                expired = return_window.popleft()
                setattr(self, sum_attr, getattr(self, sum_attr) - expired)
                setattr(self, sq_sum_attr, getattr(self, sq_sum_attr) - (expired * expired))

    def _recompute_stats(self) -> None:
        """Recompute rolling VWAP, σ and avg volume over the lookback window."""
        if len(self._stat_bars) < 2:
            return

        total_vol = self._rolling_volume_sum

        if total_vol > 0:
            self.rolling_vwap = self._rolling_vwap_volume_sum / total_vol
        else:
            self.rolling_vwap = self._rolling_vwap_sum / len(self._stat_bars)

        # Avg bar volume
        self.avg_bar_volume = total_vol / len(self._stat_bars)

        # Realised volatility: std dev of 1-min log returns
        self.rolling_volatility = self._std_from_moments(
            self._return_sum,
            self._return_sq_sum,
            len(self._return_window),
        )

        # ── EWMA volatility (RiskMetrics λ=0.94) ────────────────────────
        #   σ²_t = λ · σ²_{t-1} + (1 - λ) · r²_t
        # Reacts to regime changes in ~6 bars instead of ~30.
        lam = settings.strategy.volatility_ewma_lambda
        if self._return_window:
            latest_return = self._return_window[-1]
            if not self._ewma_initialised:
                # Seed with equal-weight variance for stability
                self._ewma_variance = self._return_sq_sum / len(self._return_window)
                self._ewma_initialised = True
            else:
                self._ewma_variance = (
                    lam * self._ewma_variance
                    + (1.0 - lam) * latest_return * latest_return
                )
            self.rolling_volatility_ewma = self._ewma_variance ** 0.5

            # ── Downside semi-variance EWMA (adverse returns only) ─────
            #   downside_r = min(r, 0.0)  — only negative moves count
            #   σ²_down_t = λ · σ²_down_{t-1} + (1-λ) · downside_r²
            downside_r = min(latest_return, 0.0)
            if not self._downside_ewma_initialised:
                downside_sq_sum = sum(value * value for value in self._return_window if value < 0.0)
                self._downside_ewma_variance = downside_sq_sum / len(self._return_window)
                self._downside_ewma_initialised = True
            else:
                self._downside_ewma_variance = (
                    lam * self._downside_ewma_variance
                    + (1.0 - lam) * downside_r * downside_r
                )
            self.rolling_downside_vol_ewma = self._downside_ewma_variance ** 0.5

        # ── 30-minute short-window volatility for TP rescaling ─────────
        if len(self._close_30_window) >= 3:
            self.rolling_volatility_30m = self._std_from_moments(
                self._return_30_sum,
                self._return_30_sq_sum,
                len(self._return_30_window),
            )
        else:
            self.rolling_volatility_30m = self.rolling_volatility
