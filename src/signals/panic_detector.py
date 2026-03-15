"""
Panic-spike detector — fires a signal when a YES token shows a sudden
price/volume spike consistent with retail panic buying.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import settings
from src.core.logger import get_logger
from src.data.ohlcv import OHLCVAggregator, OHLCVBar
from src.signals.signal_framework import BaseSignal

log = get_logger(__name__)


@dataclass
class PanicSignal(BaseSignal):
    """Emitted when all panic-spike entry conditions are met.

    Inherits from :class:`~src.signals.signal_framework.BaseSignal`:
    ``market_id``, ``no_asset_id``, ``no_best_ask``.
    """

    # YES-side diagnostics (panic-specific)
    yes_asset_id: str = ""
    yes_price: float = 0.0      # current YES price (spike)
    yes_vwap: float = 0.0       # rolling VWAP for YES
    zscore: float = 0.0         # Z-score of the move
    volume_ratio: float = 0.0   # current bar volume / avg bar volume
    whale_confluence: bool = False  # whether a whale bought NO recently


class PanicDetector:
    """Evaluates each new YES-side OHLCV bar against spike criteria.

    This detector operates on the *YES* token aggregator but the resulting
    signal is about buying the *NO* token at a discount.
    """

    def __init__(
        self,
        market_id: str,
        yes_asset_id: str,
        no_asset_id: str,
        yes_aggregator: OHLCVAggregator,
        no_aggregator: OHLCVAggregator,
        *,
        zscore_threshold: float | None = None,
        volume_ratio_threshold: float | None = None,
        trend_guard_pct: float | None = None,
        trend_guard_bars: int | None = None,
        ofi_veto_threshold: float = 50.0,
    ):
        self.market_id = market_id
        self.yes_asset_id = yes_asset_id
        self.no_asset_id = no_asset_id
        self.yes_agg = yes_aggregator
        self.no_agg = no_aggregator

        self.z_thresh = zscore_threshold if zscore_threshold is not None else settings.strategy.zscore_threshold
        self.v_thresh = volume_ratio_threshold if volume_ratio_threshold is not None else settings.strategy.volume_ratio_threshold
        self._trend_guard_pct = trend_guard_pct if trend_guard_pct is not None else settings.strategy.trend_guard_pct
        self._trend_guard_bars = trend_guard_bars if trend_guard_bars is not None else settings.strategy.trend_guard_bars
        self._ofi_veto_threshold = ofi_veto_threshold

    # ── public ──────────────────────────────────────────────────────────────
    def evaluate(
        self,
        bar: OHLCVBar,
        no_best_ask: float,
        whale_confluence: bool = False,
        yes_ofi: float = 0.0,
    ) -> PanicSignal | None:
        """Check a newly closed YES bar.  Returns a signal or None.

        Parameters
        ----------
        yes_ofi:
            SI-5 Order Flow Imbalance on the YES token L2 book.
            Positive → net bid-side accumulation.  If strongly positive
            while we're about to BUY_NO (sell YES), the spike is likely
            institutional momentum, not retail panic — veto the entry.
        """

        # Need enough history to compute meaningful stats.
        # With < 5 bars, rolling VWAP/σ are dominated by noise.
        if len(self.yes_agg.bars) < 5:
            return None

        vwap = self.yes_agg.rolling_vwap
        sigma = self.yes_agg.rolling_volatility

        if sigma <= 0 or vwap <= 0:
            return None

        # Z-score of the current bar close vs rolling VWAP.
        # Normalise delta_p by vwap so numerator (relative price deviation)
        # and denominator (log-return σ) share the same dimensionless scale.
        delta_p = (bar.close - vwap) / vwap
        zscore = delta_p / sigma

        # ── Intra-bar momentum confirmation ────────────────────────────
        # A spike where the bar closes at/near its high is more likely
        # to be genuine retail panic than one where price reverted
        # within the bar.  Discount the z-score by intra-bar retracement.
        bar_range = bar.high - bar.low
        if bar_range > 0 and delta_p > 0:
            # close_position: 1.0 = closed at high, 0.0 = closed at low
            close_position = (bar.close - bar.low) / bar_range
            # Only discount when close is in the bottom half of the bar
            # (price already started reverting within the bar → weaker signal)
            if close_position < 0.5:
                retracement_factor = 0.7 + 0.3 * close_position  # range [0.7, 0.85)
                zscore *= retracement_factor

        # Volume ratio
        avg_vol = self.yes_agg.avg_bar_volume
        v_ratio = (bar.volume / avg_vol) if avg_vol > 0 else 0.0

        # ── Gate conditions ────────────────────────────────────────────────
        if zscore < self.z_thresh:
            # Near-miss diagnostic: log when z-score is within 25% of
            # threshold to help calibrate whether the threshold is too
            # high for actual market volatility.
            miss_delta = self.z_thresh - zscore
            if miss_delta <= self.z_thresh * 0.25:
                log.info(
                    "zscore_near_miss",
                    market=self.market_id,
                    zscore=round(zscore, 3),
                    threshold=self.z_thresh,
                    gap=round(miss_delta, 3),
                    v_ratio=round(v_ratio, 2),
                )
            else:
                log.debug(
                    "spike_check_fail_zscore",
                    market=self.market_id,
                    zscore=round(zscore, 3),
                    threshold=self.z_thresh,
                )
            return None

        if v_ratio < self.v_thresh:
            log.debug(
                "spike_check_fail_volume",
                market=self.market_id,
                v_ratio=round(v_ratio, 2),
                threshold=self.v_thresh,
            )
            return None

        # NO must actually be discounted vs its VWAP
        no_vwap = self.no_agg.rolling_vwap
        if no_vwap <= 0:
            log.debug(
                "spike_check_fail_no_vwap_missing",
                market=self.market_id,
            )
            return None
        if no_best_ask >= no_vwap * settings.strategy.no_discount_factor:
            log.debug(
                "spike_check_fail_no_not_discounted",
                market=self.market_id,
                no_ask=no_best_ask,
                no_vwap=round(no_vwap, 4),
                discount_factor=settings.strategy.no_discount_factor,
            )
            return None

        # ── Trend regime guard ─────────────────────────────────────────────
        # Suppress signals when the YES token has been trending steadily
        # upward — this is a regime shift, not a panic spike.
        trend_bars = self._trend_guard_bars
        trend_pct = self._trend_guard_pct
        if len(self.yes_agg.bars) >= trend_bars:
            bars_list = self.yes_agg.bars
            recent_close = bars_list[-1].close
            anchor_close = bars_list[-trend_bars].close
            if anchor_close > 0:
                move = (recent_close - anchor_close) / anchor_close
                if move >= trend_pct:
                    log.info(
                        "trend_guard_suppressed",
                        market=self.market_id,
                        trend_pct=round(move, 4),
                        threshold=trend_pct,
                        window_bars=trend_bars,
                        zscore=round(zscore, 3),
                    )
                    return None

        # ── SI-5: Order Flow Imbalance veto ────────────────────────────────
        # A BUY_NO signal fires on a YES spike.  If the YES-token OFI is
        # strongly positive (institutional bid accumulation), the spike is
        # momentum — not panic — and the NO discount will evaporate.
        if yes_ofi > self._ofi_veto_threshold:
            log.info(
                "ofi_veto_institutional_momentum",
                market=self.market_id,
                yes_ofi=round(yes_ofi, 2),
                threshold=self._ofi_veto_threshold,
                zscore=round(zscore, 3),
            )
            return None

        # All conditions met
        signal = PanicSignal(
            market_id=self.market_id,
            yes_asset_id=self.yes_asset_id,
            no_asset_id=self.no_asset_id,
            yes_price=bar.close,
            yes_vwap=vwap,
            zscore=zscore,
            volume_ratio=v_ratio,
            no_best_ask=no_best_ask,
            whale_confluence=whale_confluence,
            signal_source="PRIMARY_Panic",
        )

        log.info(
            "panic_signal_fired",
            market=self.market_id,
            zscore=round(zscore, 3),
            v_ratio=round(v_ratio, 2),
            yes_price=bar.close,
            no_ask=no_best_ask,
            whale=whale_confluence,
        )

        return signal
