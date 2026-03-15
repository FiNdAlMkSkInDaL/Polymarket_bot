"""
Oracle Signal Engine — maps off-chain API state changes into probability
signals for the SI-8 Generalised Oracle Latency Arbitrage system.

Consumes ``OracleSnapshot`` objects from off-chain adapters, derives a
probability signal from the resolved outcome and graduated confidence,
and emits a ``SignalResult`` compatible with the existing RPE execution
funnel (``PositionManager.open_rpe_position()``).

Design decisions
────────────────
* **Adaptive threshold** — identical to the RPE engine:
  ``effective_threshold = base × (1 + (1 − confidence))``.  High-confidence
  oracle results (0.97+) use a near-base threshold; uncertain results
  (0.85) face a wider gate.

* **Monotonic state latch** — prevents repeated fires on the same
  ``resolved_outcome`` per market.  Once a market has been signalled
  as ``"YES"`` (or ``"NO"``), the engine will not re-fire until the
  outcome changes.  This is the primary anti-spam layer before the
  per-market cooldown in ``bot.py``.

* **Spread-width rejection** — if CLOB spread exceeds
  ``oracle_max_spread_cents``, the market is likely already pricing in
  the event or is illiquid; the signal is suppressed.

* **No signal on ``resolved_outcome=None``** — indeterminate events
  produce no signal.  The oracle system only fires on definitive
  (or highly probable) state changes.
"""

from __future__ import annotations

import time

from src.core.config import settings
from src.core.logger import get_logger
from src.data.oracle_adapter import OracleSnapshot
from src.signals.signal_framework import SignalResult

log = get_logger(__name__)


class OracleSignalEngine:
    """Converts ``OracleSnapshot`` objects into actionable ``SignalResult`` signals.

    Parameters
    ----------
    confidence_threshold:
        Base divergence threshold.  Defaults to
        ``settings.strategy.oracle_confidence_threshold`` (0.06).
    min_confidence:
        Minimum oracle confidence required to emit a signal.
        Defaults to ``settings.strategy.oracle_min_confidence`` (0.80).
    max_spread_cents:
        Maximum CLOB spread (in cents) before suppressing the signal.
        Defaults to ``settings.strategy.oracle_max_spread_cents`` (15.0).
    """

    def __init__(
        self,
        *,
        confidence_threshold: float | None = None,
        min_confidence: float | None = None,
        max_spread_cents: float | None = None,
    ) -> None:
        strat = settings.strategy
        self._confidence_threshold = (
            confidence_threshold if confidence_threshold is not None
            else strat.oracle_confidence_threshold
        )
        self._min_confidence = (
            min_confidence if min_confidence is not None
            else strat.oracle_min_confidence
        )
        self._max_spread_cents = (
            max_spread_cents if max_spread_cents is not None
            else strat.oracle_max_spread_cents
        )
        # Monotonic state latch: market_id → last resolved_outcome
        self._last_outcome: dict[str, str] = {}

    def evaluate(
        self,
        snapshot: OracleSnapshot,
        market_price: float,
        *,
        spread_cents: float = 0.0,
    ) -> SignalResult | None:
        """Evaluate an oracle snapshot and optionally emit a signal.

        Parameters
        ----------
        snapshot:
            The latest state from an off-chain adapter.
        market_price:
            Current YES price on the Polymarket CLOB.
        spread_cents:
            Current CLOB spread in cents (for spread-width gate).

        Returns
        -------
        SignalResult or None
            A signal if divergence exceeds the adaptive threshold,
            otherwise ``None``.
        """
        # Gate 1: No signal if outcome is indeterminate
        if snapshot.resolved_outcome is None:
            return None

        # Gate 2: Minimum confidence floor
        if snapshot.confidence < self._min_confidence:
            log.debug(
                "oracle_signal_low_confidence",
                market_id=snapshot.market_id,
                confidence=round(snapshot.confidence, 3),
                min_required=self._min_confidence,
            )
            return None

        # Gate 3: Monotonic state latch — suppress if outcome unchanged
        prev = self._last_outcome.get(snapshot.market_id)
        if prev == snapshot.resolved_outcome:
            return None

        # Gate 4: Spread-width rejection
        if spread_cents > self._max_spread_cents:
            log.info(
                "oracle_signal_spread_too_wide",
                market_id=snapshot.market_id,
                spread_cents=round(spread_cents, 2),
                max_allowed=self._max_spread_cents,
            )
            return None

        # ── Derive model probability from outcome + confidence ─────
        if snapshot.resolved_outcome == "YES":
            model_probability = snapshot.confidence
        else:
            model_probability = 1.0 - snapshot.confidence

        # ── Divergence and adaptive threshold ──────────────────────
        divergence = market_price - model_probability
        abs_div = abs(divergence)

        effective_threshold = self._confidence_threshold * (
            1.0 + (1.0 - snapshot.confidence)
        )

        if abs_div <= effective_threshold:
            log.debug(
                "oracle_signal_below_threshold",
                market_id=snapshot.market_id,
                divergence=round(divergence, 4),
                threshold=round(effective_threshold, 4),
            )
            return None

        # ── Direction determination ────────────────────────────────
        direction = "buy_no" if divergence > 0 else "buy_yes"

        # ── Latch the outcome to prevent re-firing ─────────────────
        self._last_outcome[snapshot.market_id] = snapshot.resolved_outcome

        # Normalised score ∈ [0, 1]: how far above threshold we are
        score = min(1.0, abs_div / max(effective_threshold, 0.01))

        log.info(
            "oracle_signal_fired",
            market_id=snapshot.market_id,
            adapter=snapshot.adapter_name,
            direction=direction,
            model_probability=round(model_probability, 4),
            market_price=round(market_price, 4),
            divergence=round(divergence, 4),
            confidence=round(snapshot.confidence, 3),
            event_phase=snapshot.event_phase,
        )

        return SignalResult(
            name=f"oracle_{snapshot.adapter_name}",
            market_id=snapshot.market_id,
            score=score,
            metadata={
                "model_probability": model_probability,
                "confidence": snapshot.confidence,
                "divergence": divergence,
                "direction": direction,
                "model_name": f"oracle_{snapshot.adapter_name}",
                "oracle_event_phase": snapshot.event_phase,
                "oracle_raw_state": snapshot.raw_state,
                "resolved_outcome": snapshot.resolved_outcome,
                "effective_threshold": effective_threshold,
            },
            timestamp=time.time(),
        )

    def reset_latch(self, market_id: str) -> None:
        """Clear the monotonic latch for a market (e.g., after position exit)."""
        self._last_outcome.pop(market_id, None)
