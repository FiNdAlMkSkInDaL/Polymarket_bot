"""OFI momentum signal built from rolling top-of-book volume imbalance.

This module tracks the top-level resting size imbalance:

    VI = (Q_bid - Q_ask) / (Q_bid + Q_ask)

over a rolling millisecond window and emits directional momentum signals
when the rolling average crosses a configurable threshold.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from src.signals.signal_framework import BaseSignal, SignalGenerator, SignalResult


@dataclass
class OFIMomentumSignal(BaseSignal):
    """Executable OFI momentum payload for downstream routing.

    Fields extend :class:`BaseSignal` with the current VI state and a
    directional instruction based on rolling top-of-book imbalance.
    """

    ofi: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    direction: str = "BUY"
    current_vi: float = 0.0
    rolling_vi: float = 0.0
    raw_rolling_vi: float = 0.0
    trade_flow_imbalance: float = 0.0
    tvi_multiplier: float = 1.0
    top_bid_size: float = 0.0
    top_ask_size: float = 0.0
    threshold: float = 0.0
    window_ms: int = 0
    timestamp_ms: int = 0
    max_hold_seconds: int = 300
    toxicity_index: float = 0.0
    toxicity_depth_evaporation: float = 0.0
    toxicity_sweep_ratio: float = 0.0


def compute_toxicity_size_multiplier(
    toxicity_index: float,
    *,
    elevated_threshold: float,
    max_multiplier: float,
) -> float:
    """Map a normalised toxicity reading into a bounded size multiplier."""
    idx = max(0.0, min(1.0, float(toxicity_index or 0.0)))
    threshold = max(0.0, min(0.999999, float(elevated_threshold or 0.0)))
    ceiling = max(1.0, float(max_multiplier or 1.0))
    if idx <= threshold or ceiling <= 1.0:
        return 1.0

    progress = min(1.0, (idx - threshold) / max(1e-9, 1.0 - threshold))
    return 1.0 + progress * (ceiling - 1.0)


class OFIMomentumDetector(SignalGenerator):
    """Rolling top-of-book volume-imbalance momentum detector.

    Parameters
    ----------
    market_id:
        Market identifier carried into emitted signals.
    no_asset_id:
        Optional NO token identifier for execution-layer compatibility.
    window_ms:
        Rolling lookback window in milliseconds.
    threshold:
        Absolute rolling-VI threshold required to trigger.
    """

    def __init__(
        self,
        market_id: str,
        *,
        no_asset_id: str = "",
        window_ms: int = 2000,
        threshold: float = 0.85,
        tvi_kappa: float = 1.0,
    ) -> None:
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be between 0 and 1")
        if tvi_kappa < 0:
            raise ValueError("tvi_kappa must be non-negative")

        self.market_id = market_id
        self.no_asset_id = no_asset_id
        self.window_ms = int(window_ms)
        self.threshold = float(threshold)
        self.tvi_kappa = float(tvi_kappa)
        self._vi_window: deque[tuple[int, float]] = deque()
        self._rolling_sum = 0.0
        self._last_vi = 0.0

    @property
    def name(self) -> str:
        return "ofi_momentum"

    @property
    def rolling_vi(self) -> float:
        if not self._vi_window:
            return 0.0
        return self._rolling_sum / len(self._vi_window)

    @property
    def current_vi(self) -> float:
        return self._last_vi

    def record_top_of_book(
        self,
        bid_size: float,
        ask_size: float,
        *,
        timestamp_ms: int | None = None,
    ) -> float:
        """Record a top-of-book observation and return the rolling VI."""
        vi = self._compute_vi(bid_size, ask_size)
        if vi is None:
            return self.rolling_vi

        now_ms = self._coerce_timestamp_ms(timestamp_ms)
        self._prune(now_ms)
        self._vi_window.append((now_ms, vi))
        self._rolling_sum += vi
        self._last_vi = vi
        self._prune(now_ms)
        return self.rolling_vi

    def evaluate(self, **kwargs: Any) -> SignalResult | None:
        """Evaluate current book state and emit a framework-level signal."""
        sample = self._build_sample(**kwargs)
        if sample is None:
            return None

        bid_size, ask_size, best_bid, best_ask, timestamp_ms = sample
        rolling_vi = self.record_top_of_book(
            bid_size,
            ask_size,
            timestamp_ms=timestamp_ms,
        )
        trade_aggregator = kwargs.get("trade_aggregator") or kwargs.get("no_aggregator")
        trade_flow_imbalance, tvi_multiplier, adjusted_vi = self._trade_verified_signal(
            rolling_vi,
            trade_aggregator=trade_aggregator,
            timestamp_ms=timestamp_ms,
        )
        direction = self._direction_for(adjusted_vi)
        if direction is None:
            return None
        toxicity = self._toxicity_from_book(kwargs.get("no_book") or kwargs.get("book"), direction)

        return SignalResult(
            name=self.name,
            market_id=self.market_id,
            score=self._score_from_rolling_vi(adjusted_vi),
            metadata={
                "direction": direction,
                "current_vi": round(self.current_vi, 6),
                "rolling_vi": round(adjusted_vi, 6),
                "raw_rolling_vi": round(rolling_vi, 6),
                "trade_flow_imbalance": round(trade_flow_imbalance, 6),
                "tvi_multiplier": round(tvi_multiplier, 6),
                "top_bid_size": round(bid_size, 6),
                "top_ask_size": round(ask_size, 6),
                "best_bid": round(best_bid, 6),
                "window_ms": self.window_ms,
                "threshold": self.threshold,
                "no_best_ask": round(best_ask, 6),
                "timestamp_ms": timestamp_ms,
                "toxicity_index": toxicity["toxicity_index"],
                "toxicity_depth_evaporation": toxicity["toxicity_depth_evaporation"],
                "toxicity_sweep_ratio": toxicity["toxicity_sweep_ratio"],
            },
            timestamp=timestamp_ms / 1000.0,
        )

    def generate_signal(self, **kwargs: Any) -> OFIMomentumSignal | None:
        """Evaluate current state and return an execution-layer signal."""
        sample = self._build_sample(**kwargs)
        if sample is None:
            return None

        bid_size, ask_size, best_bid, best_ask, timestamp_ms = sample
        rolling_vi = self.record_top_of_book(
            bid_size,
            ask_size,
            timestamp_ms=timestamp_ms,
        )
        trade_aggregator = kwargs.get("trade_aggregator") or kwargs.get("no_aggregator")
        trade_flow_imbalance, tvi_multiplier, adjusted_vi = self._trade_verified_signal(
            rolling_vi,
            trade_aggregator=trade_aggregator,
            timestamp_ms=timestamp_ms,
        )
        direction = self._direction_for(adjusted_vi)
        if direction is None:
            return None
        toxicity = self._toxicity_from_book(kwargs.get("no_book") or kwargs.get("book"), direction)

        return OFIMomentumSignal(
            market_id=self.market_id,
            no_asset_id=self.no_asset_id,
            no_best_ask=best_ask,
            ofi=round(adjusted_vi, 6),
            best_bid=round(best_bid, 6),
            best_ask=round(best_ask, 6),
            signal_source=self.name,
            direction=direction,
            current_vi=round(self.current_vi, 6),
            rolling_vi=round(adjusted_vi, 6),
            raw_rolling_vi=round(rolling_vi, 6),
            trade_flow_imbalance=round(trade_flow_imbalance, 6),
            tvi_multiplier=round(tvi_multiplier, 6),
            top_bid_size=round(bid_size, 6),
            top_ask_size=round(ask_size, 6),
            threshold=self.threshold,
            window_ms=self.window_ms,
            timestamp_ms=timestamp_ms,
            toxicity_index=toxicity["toxicity_index"],
            toxicity_depth_evaporation=toxicity["toxicity_depth_evaporation"],
            toxicity_sweep_ratio=toxicity["toxicity_sweep_ratio"],
        )

    def _build_sample(
        self,
        **kwargs: Any,
    ) -> tuple[float, float, float, float, int] | None:
        book = kwargs.get("no_book") or kwargs.get("book")
        if book is None:
            bid_size = kwargs.get("bid_size")
            ask_size = kwargs.get("ask_size")
            if bid_size is None or ask_size is None:
                return None
            best_bid = float(kwargs.get("best_bid", 0.0) or 0.0)
            best_ask = float(kwargs.get("no_best_ask", kwargs.get("best_ask", 0.0)) or 0.0)
            timestamp_ms = self._timestamp_ms_from_input(kwargs.get("timestamp_ms"))
            return float(bid_size), float(ask_size), best_bid, best_ask, timestamp_ms

        bid_size, ask_size = self._extract_top_sizes(book)
        if bid_size is None or ask_size is None:
            return None

        snapshot = book.snapshot() if hasattr(book, "snapshot") else None
        best_bid = float(getattr(snapshot, "best_bid", 0.0) or 0.0)
        best_ask = float(getattr(snapshot, "best_ask", 0.0) or 0.0)
        if "timestamp_ms" in kwargs and kwargs.get("timestamp_ms") is not None:
            timestamp_ms = self._timestamp_ms_from_input(kwargs.get("timestamp_ms"))
        else:
            timestamp_ms = self._timestamp_ms_from_snapshot(getattr(snapshot, "timestamp", None))
        return bid_size, ask_size, best_bid, best_ask, timestamp_ms

    def _extract_top_sizes(self, book: Any) -> tuple[float | None, float | None]:
        if not hasattr(book, "levels"):
            return None, None

        bid_levels = book.levels("bid", 1)
        ask_levels = book.levels("ask", 1)
        if not bid_levels or not ask_levels:
            return None, None

        top_bid = bid_levels[0]
        top_ask = ask_levels[0]
        bid_size = float(getattr(top_bid, "size", 0.0) or 0.0)
        ask_size = float(getattr(top_ask, "size", 0.0) or 0.0)
        if bid_size < 0 or ask_size < 0:
            return None, None
        return bid_size, ask_size

    def _prune(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_ms
        while self._vi_window and self._vi_window[0][0] < cutoff:
            _, expired_vi = self._vi_window.popleft()
            self._rolling_sum -= expired_vi

    @staticmethod
    def _compute_vi(bid_size: float, ask_size: float) -> float | None:
        bid = float(bid_size)
        ask = float(ask_size)
        total = bid + ask
        if bid < 0 or ask < 0 or total <= 0:
            return None
        return (bid - ask) / total

    @staticmethod
    def _coerce_timestamp_ms(timestamp: Any) -> int:
        if timestamp is None:
            return int(time.time() * 1000)
        return int(float(timestamp))

    def _timestamp_ms_from_input(self, timestamp_ms: Any) -> int:
        return self._coerce_timestamp_ms(timestamp_ms)

    def _timestamp_ms_from_snapshot(self, timestamp: Any) -> int:
        if timestamp is None:
            return int(time.time() * 1000)
        ts = float(timestamp)
        if ts > 10_000_000_000:
            return int(ts)
        return int(ts * 1000)

    def _direction_for(self, rolling_vi: float) -> str | None:
        if rolling_vi > self.threshold:
            return "BUY"
        if rolling_vi < -self.threshold:
            return "SELL"
        return None

    def _trade_verified_signal(
        self,
        rolling_vi: float,
        *,
        trade_aggregator: Any,
        timestamp_ms: int,
    ) -> tuple[float, float, float]:
        if trade_aggregator is None:
            return 0.0, 1.0, rolling_vi

        moments_fn = getattr(trade_aggregator, "trade_flow_moments", None)
        if callable(moments_fn):
            moments = moments_fn(
                self.window_ms,
                current_time_ms=timestamp_ms,
            )
            if isinstance(moments, tuple) and len(moments) == 2:
                buy_volume, sell_volume = moments
                if float(buy_volume or 0.0) + float(sell_volume or 0.0) <= 0:
                    return 0.0, 1.0, rolling_vi

        imbalance_fn = getattr(trade_aggregator, "trade_flow_imbalance", None)
        if not callable(imbalance_fn):
            return 0.0, 1.0, rolling_vi

        trade_flow_imbalance = float(
            imbalance_fn(
                self.window_ms,
                current_time_ms=timestamp_ms,
            )
            or 0.0
        )
        penalty = max(0.0, 1.0 - self.tvi_kappa * abs(rolling_vi - trade_flow_imbalance))
        return trade_flow_imbalance, penalty, rolling_vi * penalty

    def _toxicity_from_book(self, book: Any, direction: str) -> dict[str, float]:
        default_metrics = {
            "toxicity_index": 0.0,
            "toxicity_depth_evaporation": 0.0,
            "toxicity_sweep_ratio": 0.0,
        }
        if book is None:
            return default_metrics

        metrics_fn = getattr(book, "toxicity_metrics", None)
        if callable(metrics_fn):
            metrics = metrics_fn(direction)
            if isinstance(metrics, dict):
                return {
                    "toxicity_index": round(float(metrics.get("toxicity_index", 0.0) or 0.0), 6),
                    "toxicity_depth_evaporation": round(float(metrics.get("toxicity_depth_evaporation", 0.0) or 0.0), 6),
                    "toxicity_sweep_ratio": round(float(metrics.get("toxicity_sweep_ratio", 0.0) or 0.0), 6),
                }

        toxicity_fn = getattr(book, "toxicity_index", None)
        if callable(toxicity_fn):
            return {
                "toxicity_index": round(float(toxicity_fn(direction) or 0.0), 6),
                "toxicity_depth_evaporation": 0.0,
                "toxicity_sweep_ratio": 0.0,
            }
        return default_metrics

    def _score_from_rolling_vi(self, rolling_vi: float) -> float:
        excess = max(0.0, abs(rolling_vi) - self.threshold)
        denominator = 1.0 - self.threshold
        if denominator <= 0:
            return 1.0
        return min(1.0, excess / denominator)