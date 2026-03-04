"""
Low-volatility mean-reversion drift signal — ``MeanReversionDrift``.

Captures slow-bleed price displacement in sideways, low-volatility
markets where the ``PanicDetector`` is silent.  Instead of detecting
single-bar z-score spikes, this signal measures **cumulative drift**
over ``drift_lookback_bars`` bars: the sustained displacement of the
NO-token price from its rolling VWAP, normalised by σ.

Firing conditions (all must hold simultaneously):
  1. ``|cumulative_displacement| ≥ drift_z_threshold``
  2. ``RegimeDetector.is_mean_revert == True``
  3. ``EWMA σ < drift_vol_ceiling`` (low-volatility environment)
  4. ``L2OrderBook.is_reliable == True``
  5. No bar in the window had volume_ratio > 1.5 (ensures
     uncorrelated with PanicDetector)

The signal is intentionally **orthogonal** to PanicDetector — it only
fires when price movement is gradual and the market is quiet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class DriftSignal:
    """Result from a drift evaluation."""

    market_id: str
    displacement: float      #: cumulative z-score displacement
    score: float             #: normalised signal quality [0, 1]
    direction: str           #: "BUY_NO" (NO is cheap) or "SELL_NO"
    ewma_vol: float          #: current EWMA volatility
    lookback_bars: int       #: number of bars in the window
    timestamp: float = 0.0


class MeanReversionDrift:
    """Detects slow-bleed drift away from VWAP in low-vol regimes.

    Parameters
    ----------
    market_id:
        The condition ID for this market.
    lookback_bars:
        Number of recent bars to measure cumulative displacement.
    z_threshold:
        Minimum |displacement| to fire.
    vol_ceiling:
        Maximum EWMA σ allowed (ensures low-vol environment).
    max_bar_volume_ratio:
        Bars with volume_ratio above this are excluded (ensures
        uncorrelated with PanicDetector).
    """

    def __init__(
        self,
        market_id: str,
        *,
        lookback_bars: int | None = None,
        z_threshold: float | None = None,
        vol_ceiling: float | None = None,
        max_bar_volume_ratio: float = 1.5,
    ):
        strat = settings.strategy
        self.market_id = market_id
        self.lookback_bars = lookback_bars or strat.drift_lookback_bars
        self.z_threshold = z_threshold or strat.drift_z_threshold
        self.vol_ceiling = vol_ceiling or strat.drift_vol_ceiling
        self.max_bar_volume_ratio = max_bar_volume_ratio

    def evaluate(
        self,
        no_aggregator,
        *,
        regime_is_mean_revert: bool = False,
        l2_reliable: bool = True,
    ) -> DriftSignal | None:
        """Evaluate the drift signal.

        Parameters
        ----------
        no_aggregator:
            The ``OHLCVAggregator`` for the NO token.
        regime_is_mean_revert:
            Whether the ``RegimeDetector`` classifies the current regime
            as mean-reverting.
        l2_reliable:
            Whether the L2 book is reliable.

        Returns
        -------
        DriftSignal or None
            A signal if all conditions are met, else ``None``.
        """
        # Gate 1: Regime must be mean-reverting
        if not regime_is_mean_revert:
            return None

        # Gate 2: L2 book must be reliable
        if not l2_reliable:
            return None

        # Gate 3: Need enough bar history
        bars = list(no_aggregator.bars)
        if len(bars) < self.lookback_bars:
            return None

        window = bars[-self.lookback_bars:]

        # Gate 4: EWMA volatility must be below ceiling (low-vol only)
        ewma_vol = no_aggregator.rolling_volatility_ewma or 0.0
        if ewma_vol <= 0 or ewma_vol >= self.vol_ceiling:
            return None

        # Gate 5: No bar in the window should have high volume ratio
        # (ensures this signal is uncorrelated with PanicDetector)
        avg_vol = no_aggregator.avg_bar_volume
        if avg_vol > 0:
            for bar in window:
                bar_vol_ratio = bar.volume / avg_vol
                if bar_vol_ratio > self.max_bar_volume_ratio:
                    return None

        # Compute cumulative displacement: how far is current price
        # from rolling VWAP, normalised by σ
        vwap = no_aggregator.rolling_vwap
        sigma = no_aggregator.rolling_volatility
        if sigma <= 0 or vwap <= 0:
            return None

        current_price = window[-1].close
        displacement = (current_price - vwap) / sigma

        # Gate 6: Displacement must exceed threshold
        if abs(displacement) < self.z_threshold:
            return None

        # Direction: negative displacement = NO token is cheap → buy NO
        direction = "BUY_NO" if displacement < 0 else "SELL_NO"

        # Score: normalised quality, saturates at 2× threshold
        score = min(1.0, abs(displacement) / (2.0 * self.z_threshold))

        signal = DriftSignal(
            market_id=self.market_id,
            displacement=round(displacement, 4),
            score=round(score, 4),
            direction=direction,
            ewma_vol=round(ewma_vol, 6),
            lookback_bars=self.lookback_bars,
            timestamp=time.time(),
        )

        log.info(
            "drift_signal_fired",
            market_id=self.market_id[:16],
            displacement=signal.displacement,
            score=signal.score,
            direction=direction,
            ewma_vol=signal.ewma_vol,
        )

        return signal
