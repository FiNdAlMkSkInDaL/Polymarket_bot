"""
Cross-Market Signal Generator — leverages correlated price movements
across monitored markets for **offensive** alpha generation.

The existing PCE (Portfolio Correlation Engine) uses cross-market
correlations **defensively** — to gate entries via portfolio VaR.
This module flips the lens:

  - When two markets have high empirical correlation and one market
    moves while the other lags, that *lag* is a mean-reversion entry
    opportunity on the lagging market.

This is a classic statistical-arbitrage / pairs-trading pattern adapted
for Polymarket prediction markets.

Signal flow
───────────
  1. Each 1-min bar close, the signal generator receives returns for all
     tracked markets.
  2. For every pair (A, B) with empirical |ρ| > ``min_correlation``:
     - Compute the return spread:  z = (r_A − ρ * r_B) / σ_spread
     - If |z| exceeds ``z_entry``, emit a ``CrossMarketSignal`` on the
       lagging market (direction: toward convergence).
  3. The bot can use the signal to boost EQS, trigger RPE re-evaluation,
     or directly open a position.

Design decisions
────────────────
  - Uses the same per-market ``OHLCVAggregator`` bar returns already
    computed for the PanicDetector.
  - Purely stateless per-cycle — no position tracking.  The
    ``PositionManager`` already handles position limits.
  - Shadow-mode capable: set ``cross_market_shadow=True`` in config.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class CrossMarketSignal:
    """Emitted when a cross-market lag divergence is detected."""

    lagging_market_id: str    # condition_id of the lagging market
    leading_market_id: str    # condition_id of the leading market
    direction: str            # "YES" or "NO" — which side to enter on the lagger
    z_score: float            # standardised spread
    correlation: float        # pair ρ used
    leading_return: float     # 1-bar return of the leader
    lagging_return: float     # 1-bar return of the lagger
    spread_vol: float         # σ of the spread series
    confidence: float         # 0-1
    timestamp: float


class CrossMarketSignalGenerator:
    """Scans for exploitable inter-market divergences.

    Parameters
    ----------
    pce:
        Reference to the ``PortfolioCorrelationEngine`` for live
        pairwise correlations.
    min_correlation:
        Minimum |ρ| to consider a pair for cross-market signals.
    z_entry:
        Z-score threshold for the return-spread to emit a signal.
    spread_ewma_lambda:
        Smoothing factor for the running σ of the return spread.
    shadow_mode:
        When True, signals are logged but not acted on.
    """

    def __init__(
        self,
        pce: object,
        *,
        min_correlation: float = 0.0,
        z_entry: float = 0.0,
        spread_ewma_lambda: float = 0.0,
        shadow_mode: bool | None = None,
    ):
        strat = settings.strategy
        self._pce = pce
        self._min_corr = min_correlation or getattr(strat, "cross_mkt_min_correlation", 0.50)
        self._z_entry = z_entry or getattr(strat, "cross_mkt_z_entry", 2.0)
        self._spread_lambda = spread_ewma_lambda or getattr(strat, "cross_mkt_spread_ewma_lambda", 0.94)
        self._shadow = shadow_mode if shadow_mode is not None else getattr(strat, "cross_mkt_shadow", True)

        # Per-pair EWMA spread variance:  (mkt_a, mkt_b) → variance
        self._spread_var: dict[tuple[str, str], float] = {}
        self._spread_initialised: dict[tuple[str, str], bool] = {}

        # Last bar returns per market_id
        self._last_returns: dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def record_return(self, market_id: str, log_return: float) -> None:
        """Record the latest 1-bar log return for a market."""
        self._last_returns[market_id] = log_return

    def scan(self) -> list[CrossMarketSignal]:
        """Scan all high-ρ pairs for divergence signals.

        Should be called after all per-market ``record_return()`` calls
        for the current bar cycle.

        Returns a list of signals (may be empty).
        """
        if not hasattr(self._pce, "corr_matrix"):
            return []

        pairs = self._pce.corr_matrix.all_pairs()
        if not pairs:
            return []

        signals: list[CrossMarketSignal] = []
        now = time.time()

        for (mkt_a, mkt_b), est in pairs.items():
            rho = est.blended
            if abs(rho) < self._min_corr:
                continue

            ret_a = self._last_returns.get(mkt_a)
            ret_b = self._last_returns.get(mkt_b)
            if ret_a is None or ret_b is None:
                continue

            # Compute return spread relative to expected co-movement
            spread = ret_a - rho * ret_b
            pair_key = (mkt_a, mkt_b)

            # Update EWMA variance of the spread
            spread_var = self._spread_var.get(pair_key, 0.0)
            lam = self._spread_lambda
            if not self._spread_initialised.get(pair_key, False):
                spread_var = spread ** 2 if spread != 0 else 1e-8
                self._spread_initialised[pair_key] = True
            else:
                spread_var = lam * spread_var + (1.0 - lam) * spread ** 2

            self._spread_var[pair_key] = spread_var
            spread_vol = math.sqrt(max(spread_var, 1e-12))

            z = spread / spread_vol

            if abs(z) < self._z_entry:
                continue

            # Determine which market is lagging
            if z > 0:
                # A moved more than expected given B's move
                # A is the leader, B is lagging (should move up)
                lagging_id = mkt_b
                leading_id = mkt_a
                direction = "YES"  # buy the lagger
            else:
                # B moved more than expected
                lagging_id = mkt_a
                leading_id = mkt_b
                direction = "YES"

            confidence = min(1.0, (abs(z) - self._z_entry) / self._z_entry + 0.5)

            sig = CrossMarketSignal(
                lagging_market_id=lagging_id,
                leading_market_id=leading_id,
                direction=direction,
                z_score=round(z, 3),
                correlation=round(rho, 4),
                leading_return=round(ret_a if leading_id == mkt_a else ret_b, 6),
                lagging_return=round(ret_a if lagging_id == mkt_a else ret_b, 6),
                spread_vol=round(spread_vol, 6),
                confidence=round(confidence, 3),
                timestamp=now,
            )
            signals.append(sig)

            prefix = "[SHADOW] " if self._shadow else ""
            log.info(
                f"{prefix}cross_market_signal",
                lagging=lagging_id[:16],
                leading=leading_id[:16],
                direction=direction,
                z=round(z, 3),
                rho=round(rho, 4),
                confidence=round(confidence, 3),
            )

        # Clear returns for next cycle
        self._last_returns.clear()

        return signals

    @property
    def is_shadow(self) -> bool:
        return self._shadow

    def reset(self) -> None:
        """Clear all state."""
        self._spread_var.clear()
        self._spread_initialised.clear()
        self._last_returns.clear()
