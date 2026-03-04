"""
Multi-signal framework — abstract base class for signal generators and
a composite evaluator that combines multiple signal sources into a
unified scoring system.

New signals:
  - **OrderbookImbalance**: fires when bid-side depth overwhelms asks
    on the NO token, suggesting accumulation.
  - **SpreadCompression**: fires when the NO spread tightens below
    historical norms, indicating imminent price discovery.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Abstract base
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SignalResult:
    """Unified output from any signal generator."""

    name: str           # e.g. "panic", "imbalance", "spread_compression"
    market_id: str
    score: float        # 0.0 – 1.0 normalised strength
    metadata: dict      # signal-specific diagnostics
    timestamp: float


class SignalGenerator(ABC):
    """Base class for all signal generators.

    Subclasses implement :meth:`evaluate` and return ``None`` (no signal)
    or a :class:`SignalResult` with a normalised score.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> SignalResult | None:
        """Evaluate the signal.  Return ``None`` if conditions are not met."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  Concrete signals
# ═══════════════════════════════════════════════════════════════════════════


class OrderbookImbalanceSignal(SignalGenerator):
    """Fires when bid-side depth on the NO token materially exceeds ask depth.

    A strong bid imbalance suggests accumulation by informed traders and
    supports a mean-reversion entry.

    Parameters
    ----------
    imbalance_threshold:
        Minimum bid/ask depth ratio to fire (default 2.0 = 2:1 bids vs asks).
    min_depth_usd:
        Minimum total depth on both sides (filters low-liquidity books).
    """

    def __init__(
        self,
        market_id: str,
        *,
        imbalance_threshold: float | None = None,
        min_depth_usd: float = 50.0,
    ):
        self.market_id = market_id
        self.threshold = imbalance_threshold if imbalance_threshold is not None else settings.strategy.imbalance_threshold
        self.min_depth_usd = min_depth_usd

    @property
    def name(self) -> str:
        return "imbalance"

    def evaluate(self, **kwargs: Any) -> SignalResult | None:
        """Expects kwargs: ``no_book`` (:class:`OrderbookTracker`)."""
        no_book = kwargs.get("no_book")
        if no_book is None or not no_book.has_data:
            return None

        snap = no_book.snapshot()
        bid_depth = snap.bid_depth_usd
        ask_depth = snap.ask_depth_usd

        if ask_depth <= 0 or bid_depth <= 0:
            return None

        total_depth = bid_depth + ask_depth
        if total_depth < self.min_depth_usd:
            return None

        ratio = bid_depth / ask_depth
        if ratio < self.threshold:
            return None

        # Normalise score: 0 at threshold, 1.0 at 2x threshold
        score = min(1.0, (ratio - self.threshold) / self.threshold)

        return SignalResult(
            name=self.name,
            market_id=self.market_id,
            score=score,
            metadata={
                "bid_depth": round(bid_depth, 2),
                "ask_depth": round(ask_depth, 2),
                "imbalance_ratio": round(ratio, 3),
            },
            timestamp=time.time(),
        )


class SpreadCompressionSignal(SignalGenerator):
    """Fires when the NO spread compresses below its rolling average.

    Tight spreads typically precede price moves; when the spread is
    significantly below normal, it suggests strong two-sided interest
    and imminent price discovery — a favourable entry condition.

    Parameters
    ----------
    compression_pct:
        Fire when current spread < ``compression_pct`` × rolling average
        spread (default 0.5 = spread is half normal).
    min_history:
        Need at least this many spread samples before evaluating.
    """

    def __init__(
        self,
        market_id: str,
        *,
        compression_pct: float | None = None,
        min_history: int = 10,
    ):
        self.market_id = market_id
        self.compression_pct = compression_pct if compression_pct is not None else settings.strategy.spread_compression_pct
        self.min_history = min_history
        self._spread_history: deque[float] = deque(maxlen=120)
        self._running_sum: float = 0.0

    @property
    def name(self) -> str:
        return "spread_compression"

    def record_spread(self, spread: float) -> None:
        """Record a spread observation for rolling average computation."""
        # If deque is full, the leftmost element will be evicted on append
        if len(self._spread_history) == self._spread_history.maxlen:
            self._running_sum -= self._spread_history[0]
        self._spread_history.append(spread)
        self._running_sum += spread

    def evaluate(self, **kwargs: Any) -> SignalResult | None:
        """Expects kwargs: ``no_book`` (:class:`OrderbookTracker`)."""
        no_book = kwargs.get("no_book")
        if no_book is None or not no_book.has_data:
            return None

        snap = no_book.snapshot()
        if snap.best_bid <= 0 or snap.best_ask <= 0:
            return None

        current_spread = snap.best_ask - snap.best_bid

        # Record for history
        self.record_spread(current_spread)

        if len(self._spread_history) < self.min_history:
            return None

        avg_spread = self._running_sum / len(self._spread_history)
        if avg_spread <= 0:
            return None

        ratio = current_spread / avg_spread
        if ratio >= self.compression_pct:
            return None

        # Normalise: 0 at compression_pct, 1.0 at 0 spread
        score = min(1.0, (self.compression_pct - ratio) / self.compression_pct)

        return SignalResult(
            name=self.name,
            market_id=self.market_id,
            score=score,
            metadata={
                "current_spread": round(current_spread, 4),
                "avg_spread": round(avg_spread, 4),
                "compression_ratio": round(ratio, 3),
            },
            timestamp=time.time(),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Composite signal evaluator
# ═══════════════════════════════════════════════════════════════════════════


class CompositeSignalEvaluator:
    """Combines scores from multiple :class:`SignalGenerator` instances.

    Each generator is assigned a weight.  The composite score is the
    weighted average of all fired signals.  A minimum composite score
    threshold determines whether the combined signal is actionable.

    Also computes a dynamic **uncertainty penalty** (0.0–1.0) from:
    - Spread instability: current spread vs rolling average.
    - Signal conviction: inverse of composite score.

    The uncertainty penalty is injected into each fired ``SignalResult``'s
    ``metadata`` dict under the key ``"uncertainty_penalty"`` so that
    downstream consumers (Kelly sizer) can use it for edge discounting.

    Parameters
    ----------
    min_composite_score:
        Minimum weighted-average score to emit a composite signal.
    """

    def __init__(
        self,
        generators: list[tuple[SignalGenerator, float]],
        *,
        min_composite_score: float = 0.3,
    ):
        """
        Parameters
        ----------
        generators:
            List of (generator, weight) tuples.  Weights are normalised
            internally so they don't need to sum to 1.
        """
        total_w = sum(w for _, w in generators) or 1.0
        self._generators = [(g, w / total_w) for g, w in generators]
        self._min_score = min_composite_score

    def evaluate(self, **kwargs) -> tuple[float, list[SignalResult]]:
        """Run all generators and return (composite_score, fired_signals).

        Returns
        -------
        composite_score:
            Weighted average score (0 if no signals fired).
        fired:
            List of :class:`SignalResult` from generators that produced
            a non-``None`` result.  Each result's ``metadata`` is
            augmented with ``uncertainty_penalty``.
        """
        fired: list[SignalResult] = []
        weighted_sum = 0.0
        weight_sum = 0.0

        for gen, weight in self._generators:
            result = gen.evaluate(**kwargs)
            if result is not None:
                fired.append(result)
                weighted_sum += result.score * weight
                weight_sum += weight

        composite = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # ── Compute dynamic uncertainty penalty ────────────────────────────
        uncertainty = self._compute_uncertainty(
            composite, kwargs.get("no_book")
        )
        for sig in fired:
            sig.metadata["uncertainty_penalty"] = round(uncertainty, 4)

        return composite, fired

    def is_actionable(self, **kwargs) -> tuple[bool, float, list[SignalResult]]:
        """Convenience: evaluate and check against threshold.

        Returns (actionable, composite_score, fired_signals).
        """
        score, fired = self.evaluate(**kwargs)
        return score >= self._min_score, score, fired

    # ── Uncertainty computation ──────────────────────────────────────────
    def _compute_uncertainty(
        self,
        composite_score: float,
        no_book: Any | None,
    ) -> float:
        """Compute a dynamic uncertainty penalty \u2208 [0.0, 1.0].

        Blends two real-time proxies:

        - **Spread instability** ($U_{spread}$): how wide the current
          spread is relative to its rolling average. A wider-than-normal
          spread indicates low liquidity and high uncertainty.
        - **Signal conviction** ($U_{conf}$): ``1 - composite_score``.
          A weaker signal implies higher uncertainty in the edge.

        The final penalty is:

        .. math::
            U = \\min(1.0,\\; w_1 \\cdot U_{spread} + w_2 \\cdot U_{conf})

        with configurable weights (default: 0.6 / 0.4).
        """
        w_spread = settings.strategy.uncertainty_spread_weight
        w_conf = settings.strategy.uncertainty_conf_weight

        # U_conf: inverse of signal conviction
        u_conf = 1.0 - composite_score

        # U_spread: derive from the SpreadCompressionSignal's history
        u_spread = 0.5  # default when no spread data available
        if no_book is not None and hasattr(no_book, "has_data") and no_book.has_data:
            snap = no_book.snapshot()
            current_spread = snap.best_ask - snap.best_bid if (snap.best_ask > 0 and snap.best_bid > 0) else 0.0

            # Look for a SpreadCompressionSignal among our generators to
            # get the rolling average spread (use _running_sum for O(1))
            avg_spread = 0.0
            for gen, _ in self._generators:
                if isinstance(gen, SpreadCompressionSignal) and gen._spread_history:
                    avg_spread = gen._running_sum / len(gen._spread_history)
                    break

            if avg_spread > 0 and current_spread > 0:
                # Ratio > 1.0 means spread is wider than normal → high uncertainty
                u_spread = max(0.0, min(1.0, (current_spread / avg_spread) - 1.0))
            elif current_spread <= 0:
                u_spread = 0.0  # zero spread = very tight = low uncertainty

        return min(1.0, w_spread * u_spread + w_conf * u_conf)
