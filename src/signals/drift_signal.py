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
from src.signals.signal_framework import BaseSignal

log = get_logger(__name__)

# Cold-start volatility floor to prevent unstable displacement from tiny σ.
MIN_VOLATILITY = 0.0001


@dataclass
class DriftSignal(BaseSignal):
    """Result from a drift evaluation.

    Inherits from :class:`~src.signals.signal_framework.BaseSignal`:
    ``market_id``, ``no_asset_id``, ``no_best_ask``.

    The ``no_asset_id`` and ``no_best_ask`` fields are populated by the
    caller (``_on_yes_bar_closed`` in ``bot.py``) when invoking
    ``MeanReversionDrift.evaluate()`` with the market-specific values.
    """

    displacement: float = 0.0   #: cumulative z-score displacement
    score: float = 0.0          #: normalised signal quality [0, 1]
    direction: str = "BUY_NO"   #: "BUY_NO" (NO is cheap) or "SELL_NO"
    ewma_vol: float = 0.0       #: current EWMA volatility
    lookback_bars: int = 0      #: number of bars in the window
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
        no_asset_id: str = "",
        no_best_ask: float = 0.0,
        regime_is_mean_revert: bool = False,
        l2_reliable: bool = True,
    ) -> DriftSignal | None:
        """Evaluate the drift signal.

        Parameters
        ----------
        no_aggregator:
            The ``OHLCVAggregator`` for the NO token.
        no_asset_id:
            Token ID for the NO outcome token.  Forwarded into the
            returned :class:`DriftSignal` so the execution layer can
            route the order without needing to re-look it up.
        no_best_ask:
            Current best-ask price for the NO token.  Used as the
            entry-price reference inside the returned signal.
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
        def _reject(gate: str, **kw):
            """Log gate rejection and return None."""
            log.debug(
                "drift_eval",
                market_id=self.market_id[:16],
                gate=gate,
                passed=False,
                **kw,
            )
            return None

        # Gate 1: Regime must be mean-reverting
        if not regime_is_mean_revert:
            return _reject("regime", regime_is_mean_revert=False)

        # Gate 2: L2 book must be reliable
        if not l2_reliable:
            return _reject("l2_unreliable", l2_reliable=False)

        # Gate 3: Need enough bar history
        bars = list(no_aggregator.bars)
        if len(bars) < self.lookback_bars:
            return _reject(
                "insufficient_history",
                bars_available=len(bars),
                lookback_required=self.lookback_bars,
            )

        window = bars[-self.lookback_bars:]

        # Gate 4: EWMA volatility must be below ceiling (low-vol only)
        ewma_vol = no_aggregator.rolling_volatility_ewma or 0.0
        if ewma_vol < MIN_VOLATILITY or ewma_vol >= self.vol_ceiling:
            return _reject(
                "ewma_vol_bounds",
                ewma_vol=round(ewma_vol, 8),
                vol_ceiling=self.vol_ceiling,
                vol_too_low=(ewma_vol < MIN_VOLATILITY),
            )

        # Gate 5: No bar in the window should have high volume ratio
        # (ensures this signal is uncorrelated with PanicDetector)
        avg_vol = no_aggregator.avg_bar_volume
        if avg_vol > 0:
            for bar in window:
                bar_vol_ratio = bar.volume / avg_vol
                if bar_vol_ratio > self.max_bar_volume_ratio:
                    return _reject(
                        "high_volume_bar",
                        bar_vol_ratio=round(bar_vol_ratio, 4),
                        max_bar_volume_ratio=self.max_bar_volume_ratio,
                    )

        # Compute cumulative displacement: how far is current price
        # from rolling VWAP, normalised by σ
        vwap = no_aggregator.rolling_vwap
        sigma = no_aggregator.rolling_volatility
        if sigma < MIN_VOLATILITY or vwap <= 0:
            return _reject(
                "sigma_vwap_invalid",
                sigma=round(sigma, 8) if sigma else 0.0,
                vwap=round(vwap, 8) if vwap else 0.0,
            )

        current_price = window[-1].close
        displacement = (current_price - vwap) / sigma

        # Gate 6: Displacement must exceed threshold
        if abs(displacement) < self.z_threshold:
            return _reject(
                "displacement_below_threshold",
                displacement=round(displacement, 4),
                z_threshold=self.z_threshold,
            )

        # Direction: negative displacement = NO token is cheap → buy NO
        direction = "BUY_NO" if displacement < 0 else "SELL_NO"

        # Score: normalised quality, saturates at 2× threshold
        score = min(1.0, abs(displacement) / (2.0 * self.z_threshold))

        signal = DriftSignal(
            market_id=self.market_id,
            no_asset_id=no_asset_id,
            no_best_ask=no_best_ask,
            displacement=round(displacement, 4),
            score=round(score, 4),
            direction=direction,
            ewma_vol=round(ewma_vol, 6),
            lookback_bars=self.lookback_bars,
            timestamp=time.time(),
            signal_source="V3_MeanRevDrift",
        )

        log.debug(
            "drift_eval",
            market_id=self.market_id[:16],
            gate="all_passed",
            passed=True,
            displacement=signal.displacement,
            ewma_vol=signal.ewma_vol,
            direction=direction,
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
