"""
Regime Detector — lightweight Hidden-Markov-inspired two-state model
that classifies the current market regime as **MEAN_REVERT** (noise-
dominated, favourable for the bot) or **TRENDING** (information-driven,
risky for mean-reversion entries).

The model maintains a continuous *regime score* ∈ [0, 1] where
0 → pure trending and 1 → pure mean-reversion.  The score is an
exponentially-weighted blend of three micro-features computed from the
per-market OHLCV aggregator:

  1. **Return autocorrelation** — rolling Lag-1 autocorrelation of 1-min
     log returns.  Negative autocorrelation (bouncing) → mean-reversion.
     Positive (momentum) → trending.

  2. **Volatility ratio** — EWMA σ / equal-weight σ.  When EWMA ≫ EW,
     recent vol is elevated (regime change); when EWMA ≈ EW the regime
     is stable.

  3. **Directional persistence** — fraction of the last N bars whose
     close-to-close return has the same sign.  High persistence →
     trending.

The three features are independently mapped to [0, 1] via sigmoid-like
transforms and then geometrically averaged (like EQS) so that any
single feature at zero dominates the score.

The detector is **per-market** — each market has its own regime state.

Thread-safety: single-threaded asyncio — no locks needed.

Usage in the bot:
  - ``_wire_market()`` instantiates a ``RegimeDetector`` per market.
  - On each OHLCV bar close, call ``detector.update(aggregator)``.
  - Before entry, read ``detector.regime_score`` and pass to EQS or
    use ``detector.is_mean_revert`` to gate entries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class RegimeState:
    """Snapshot of the regime detector's current assessment."""

    score: float = 0.50           # 0 = trending, 1 = mean-revert
    autocorr_factor: float = 0.50
    vol_ratio_factor: float = 0.50
    persistence_factor: float = 0.50
    is_mean_revert: bool = True
    bar_count: int = 0


class RegimeDetector:
    """Per-market regime classifier.

    Parameters
    ----------
    market_id:
        Condition ID for logging.
    ewma_lambda:
        Smoothing factor for the regime score EMA (default 0.90 → ~10-bar
        half-life at 1-min cadence).
    autocorr_window:
        Number of past returns to use for Lag-1 autocorrelation.
    persistence_window:
        Number of past returns for directional-persistence measurement.
    threshold:
        Regime score below which the market is classified as TRENDING.
    """

    def __init__(
        self,
        market_id: str,
        *,
        ewma_lambda: float = 0.0,
        autocorr_window: int = 0,
        persistence_window: int = 0,
        threshold: float = 0.0,
    ):
        strat = settings.strategy
        self.market_id = market_id
        self._lambda = ewma_lambda or getattr(strat, "regime_ewma_lambda", 0.90)
        self._ac_window = autocorr_window or getattr(strat, "regime_autocorr_window", 20)
        self._pers_window = persistence_window or getattr(strat, "regime_persistence_window", 15)
        self._threshold = threshold or getattr(strat, "regime_threshold", 0.40)

        # Internal state
        self._returns: list[float] = []  # rolling log-returns
        self._max_window = max(self._ac_window, self._pers_window) + 5
        self._score: float = 0.50  # agnostic at start
        self._bar_count: int = 0
        self._last_ewma_vol: float = 0.0
        self._last_ew_vol: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def regime_score(self) -> float:
        """Current regime score ∈ [0, 1].  Higher = more mean-revertible."""
        return self._score

    @property
    def is_mean_revert(self) -> bool:
        """True if the current regime favours mean-reversion strategies."""
        return self._score >= self._threshold

    def state(self) -> RegimeState:
        """Return a frozen snapshot of the detector."""
        ac = self._autocorrelation_factor()
        vr = self._vol_ratio_factor(self._last_ewma_vol, self._last_ew_vol)
        pf = self._persistence_factor()
        return RegimeState(
            score=self._score,
            autocorr_factor=ac,
            vol_ratio_factor=vr,
            persistence_factor=pf,
            is_mean_revert=self.is_mean_revert,
            bar_count=self._bar_count,
        )

    def update(self, *, log_return: float, ewma_vol: float, ew_vol: float) -> float:
        """Ingest a new 1-min bar and return the updated regime score.

        Parameters
        ----------
        log_return:
            The latest 1-minute log return.
        ewma_vol:
            Current EWMA σ from the aggregator (OE-2).
        ew_vol:
            Current equal-weight rolling σ from the aggregator.
        """
        self._returns.append(log_return)
        if len(self._returns) > self._max_window:
            self._returns = self._returns[-self._max_window:]

        self._bar_count += 1
        self._last_ewma_vol = ewma_vol
        self._last_ew_vol = ew_vol

        # Compute the three regime factors
        ac = self._autocorrelation_factor()
        vr = self._vol_ratio_factor(ewma_vol, ew_vol)
        pf = self._persistence_factor()

        # Geometric mean — any factor near 0 kills the score
        raw = (ac * vr * pf) ** (1.0 / 3.0)

        # Exponential smoothing
        self._score = self._lambda * self._score + (1.0 - self._lambda) * raw

        return self._score

    # ── Feature extractors ─────────────────────────────────────────────────

    def _autocorrelation_factor(self) -> float:
        """Lag-1 autocorrelation → [0, 1].

        Negative autocorrelation (bouncing) → high MR score.
        Positive (momentum) → low MR score.
        """
        rets = self._returns[-self._ac_window:]
        n = len(rets)
        if n < 5:
            return 0.50  # insufficient data → agnostic

        mean_r = sum(rets) / n
        var = sum((r - mean_r) ** 2 for r in rets)
        if var == 0:
            return 0.50

        cov = sum(
            (rets[i] - mean_r) * (rets[i - 1] - mean_r)
            for i in range(1, n)
        )
        ac1 = cov / var  # ∈ [-1, 1] roughly

        # Map: ac1 = -1 → 1.0 (strong MR), ac1 = 0 → 0.5, ac1 = +1 → 0.0
        return _sigmoid(-ac1 * 3.0)

    def _vol_ratio_factor(self, ewma_vol: float = 0.0, ew_vol: float = 0.0) -> float:
        """EWMA σ / EW σ → [0, 1].

        Ratio ≈ 1 → stable, high MR score.
        Ratio ≫ 1 → vol expanding, low MR score.
        """
        if ew_vol <= 0 or ewma_vol <= 0:
            return 0.50  # no data

        ratio = ewma_vol / ew_vol
        # ratio = 1.0 → 0.73 (slightly favourable)
        # ratio = 2.0 → 0.12 (regime change)
        # ratio = 0.5 → 0.95 (calm → very favourable)
        return _sigmoid(-(ratio - 1.0) * 3.0 + 1.0)

    def _persistence_factor(self) -> float:
        """Directional persistence → [0, 1].

        High fraction of same-sign returns → trending → low MR score.
        """
        rets = self._returns[-self._pers_window:]
        n = len(rets)
        if n < 5:
            return 0.50

        signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in rets]
        # Count consecutive same-sign runs
        same = sum(1 for i in range(1, n) if signs[i] == signs[i - 1] and signs[i] != 0)
        frac = same / max(1, n - 1)  # ∈ [0, 1]

        # frac ≈ 0.5 (random) → MR friendly.  frac > 0.7 → trending.
        return _sigmoid(-(frac - 0.5) * 6.0)


def _sigmoid(x: float) -> float:
    """Standard sigmoid clamped for numerical safety."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))
