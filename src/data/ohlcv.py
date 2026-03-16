"""
OHLCV bar aggregator — converts raw TradeEvents from the WebSocket into
1-minute OHLCV bars and maintains rolling statistics (VWAP, σ) needed by
the panic-spike detector.
"""

from __future__ import annotations

import time
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

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
    prices: list[float] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)
    _true_open: float = 0.0       # preserved across truncation
    _total_volume: float = 0.0    # running volume total (survives truncation)

    _MAX_TICKS = 10_000  # cap per bar to prevent unbounded growth

    def add(self, price: float, size: float, ts: float) -> None:
        if not self.prices:
            self.open_time = ts
            self._true_open = price
            self._total_volume = 0.0
        self._total_volume += size
        if len(self.prices) >= self._MAX_TICKS:
            # Keep most recent half to maintain statistical validity
            half = self._MAX_TICKS // 2
            self.prices = self.prices[half:]
            self.sizes = self.sizes[half:]
        self.prices.append(price)
        self.sizes.append(size)

    def finalise(self) -> OHLCVBar | None:
        if not self.prices:
            return None
        prices = np.array(self.prices)
        sizes = np.array(self.sizes)
        vwap = float(np.average(prices, weights=sizes)) if sizes.sum() > 0 else float(prices.mean())
        return OHLCVBar(
            open_time=self.open_time,
            open=self._true_open if self._true_open > 0 else self.prices[0],
            high=float(prices.max()),
            low=float(prices.min()),
            close=self.prices[-1],
            volume=self._total_volume if self._total_volume > 0 else float(sizes.sum()),
            vwap=vwap,
            trade_count=len(self.prices),
        )

    def reset(self) -> None:
        self.open_time = 0.0
        self.prices.clear()
        self.sizes.clear()
        self._true_open = 0.0
        self._total_volume = 0.0


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
        self.lookback = lookback_minutes or settings.strategy.lookback_minutes

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

        # Most recent trade timestamp — used for data freshness checks.
        # Unlike bar open_time, this is updated on EVERY trade.
        self.last_trade_time: float = 0.0

    # ── public API ──────────────────────────────────────────────────────────
    def on_trade(self, event: TradeEvent) -> OHLCVBar | None:
        """Ingest a trade tick.  Returns a completed bar if one just closed."""
        now = event.timestamp
        self.last_trade_time = now

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
        if not self._builder.prices:
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
        if self._builder.prices:
            return self._builder.prices[-1]
        if self.bars:
            return self.bars[-1].close
        return 0.0

    # ── internals ───────────────────────────────────────────────────────────
    def _close_bar(self) -> OHLCVBar | None:
        bar = self._builder.finalise()
        self._builder.reset()
        if bar is None:
            return None

        log.info(
            "bar_closed",
            asset_id=self.asset_id,
            open_time=bar.open_time,
            close=bar.close,
            volume=bar.volume,
            trade_count=bar.trade_count,
        )

        self.bars.append(bar)
        self._recompute_stats()
        return bar

    def _recompute_stats(self) -> None:
        """Recompute rolling VWAP, σ and avg volume over the lookback window."""
        n = len(self.bars)
        skip = max(0, n - self.lookback)
        window = list(itertools.islice(self.bars, skip, n))
        if len(window) < 2:
            return

        # Rolling VWAP: weighted by bar volume
        vwaps = np.array([b.vwap for b in window])
        volumes = np.array([b.volume for b in window])
        total_vol = volumes.sum()

        if total_vol > 0:
            self.rolling_vwap = float(np.average(vwaps, weights=volumes))
        else:
            self.rolling_vwap = float(vwaps.mean())

        # Avg bar volume
        self.avg_bar_volume = float(volumes.mean())

        # Realised volatility: std dev of 1-min log returns
        closes = np.array([b.close for b in window])
        # Guard against zero prices
        closes = np.maximum(closes, 1e-8)
        log_returns = np.diff(np.log(closes))
        # Cap extreme per-bar log-returns to prevent outlier bars (e.g.
        # backfill artefacts with prices near 0 or 1) from inflating σ.
        # ±ln(3) ≈ ±1.1 allows a price to triple/third per bar — well
        # beyond normal binary-market dynamics.
        _LR_CAP = 1.1
        log_returns = np.clip(log_returns, -_LR_CAP, _LR_CAP)
        self.rolling_volatility = float(log_returns.std()) if len(log_returns) > 1 else 0.0

        # ── EWMA volatility (RiskMetrics λ=0.94) ────────────────────────
        #   σ²_t = λ · σ²_{t-1} + (1 - λ) · r²_t
        # Reacts to regime changes in ~6 bars instead of ~30.
        lam = settings.strategy.volatility_ewma_lambda
        if len(log_returns) > 0:
            latest_return = float(log_returns[-1])
            if not self._ewma_initialised:
                # Seed with equal-weight variance for stability
                self._ewma_variance = float((log_returns ** 2).mean())
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
                ds_returns = np.minimum(log_returns, 0.0)
                self._downside_ewma_variance = float((ds_returns ** 2).mean())
                self._downside_ewma_initialised = True
            else:
                self._downside_ewma_variance = (
                    lam * self._downside_ewma_variance
                    + (1.0 - lam) * downside_r * downside_r
                )
            self.rolling_downside_vol_ewma = self._downside_ewma_variance ** 0.5

        # ── 30-minute short-window volatility for TP rescaling ─────────
        n30 = len(self.bars)
        skip30 = max(0, n30 - 30)
        window_30 = list(itertools.islice(self.bars, skip30, n30))
        if len(window_30) >= 3:
            closes_30 = np.array([b.close for b in window_30])
            closes_30 = np.maximum(closes_30, 1e-8)
            lr_30 = np.diff(np.log(closes_30))
            self.rolling_volatility_30m = float(lr_30.std()) if len(lr_30) > 1 else 0.0
        else:
            self.rolling_volatility_30m = self.rolling_volatility
