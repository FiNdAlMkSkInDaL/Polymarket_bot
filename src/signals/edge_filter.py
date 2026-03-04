"""
Pre-trade edge quality filter — information-theoretic entry gate.

Replaces the naive min/max price-range clamp with a principled expected-
value gate that naturally penalises:

  - Tail probabilities (low entropy → moves are information, not panic)
  - Fee-heavy price regions (quadratic drag peaks at p = 0.50)
  - Trades where net spread can't resolve to positive discrete ticks

The **Edge Quality Score (EQS)** is a weighted geometric mean of four
normalised factors, each in [0, 1]::

    EQS = 100 × regime^0.35 × fee_eff^0.30 × tick_viab^0.20 × signal_q^0.15

Geometric-mean properties:
  - Any factor at **zero** zeros the *entire* score — impossible trades
    are hard-rejected regardless of how strong other factors are.
  - All factors must be acceptable to produce a high score — no single
    factor can rescue a terrible trade.
  - The score is continuous and differentiable (no cliff edges).

Factors
-------
1. **Regime quality** — binary entropy ``H(p)``.  Markets near 50/50 have
   maximum uncertainty (``H = 1``); panic-driven moves there are most likely
   noise and thus mean-revertible.  Near the tails (``p → 0`` or ``p → 1``),
   ``H → 0``; moves are information, not panic.

2. **Fee efficiency** — fraction of the expected α-spread that survives
   round-trip fee drag:  ``FER = max(0, 1 − fees / gross)``.  Uses the
   quadratic fee curve ``Fee(p) = f_max · 4p(1−p)``.

3. **Tick viability** — can the net expected profit resolve to at least one
   positive 1¢ tick on Polymarket's discrete price grid?  Normalised so
   that ≥ 3¢ of net profit saturates at 1.0.

4. **Signal quality** — excess z-score and volume ratio above PanicDetector
   thresholds, plus whale-confluence bonus.  Baseline 0.5 (barely passed
   detector), scaling to 1.0 for strong excess.

A trade with ``EQS < min_edge_score`` (default 40) is rejected.

Theory
------
The information-theoretic foundation comes from the observation that a
mean-reversion strategy is implicitly a bet that the observed price move is
*noise* (temporary) rather than *signal* (permanent information arrival).
Binary entropy ``H(p)`` quantifies the market's residual uncertainty — at
``p = 0.03`` the market is 97% resolved and any further move is almost
certainly real information, while at ``p = 0.50`` maximum uncertainty means
sentiment swings dominate.  Weighting EQS by ``H(p)^0.35`` encodes this
directly into the entry decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.config import settings
from src.core.logger import get_logger
from src.trading.fees import get_fee_rate

log = get_logger(__name__)

# ── Component weights (must sum to 1.0) ────────────────────────────────────
W_REGIME = 0.35
W_FEE = 0.30
W_TICK = 0.20
W_SIGNAL = 0.15


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def binary_entropy(p: float) -> float:
    """Normalised binary entropy H(p) ∈ [0, 1].

    .. math::

        H(p) = -\\,p \\log_2 p  -  (1-p) \\log_2(1-p)

    Maximum at *p = 0.50* → 1.0.  Zero at *p ∈ {0, 1}*.
    Symmetric: ``H(p) == H(1 − p)``.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


# ═══════════════════════════════════════════════════════════════════════════
#  Data class
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class EdgeAssessment:
    """Pre-trade edge quality assessment with full diagnostics."""

    score: float                  #: 0–100 composite EQS
    regime_quality: float         #: binary entropy H(p), 0–1
    fee_efficiency: float         #: fraction of spread surviving fees, 0–1
    tick_margin: int              #: net profitable ticks after fees
    tick_viability: float         #: normalised tick factor, 0–1
    signal_quality: float         #: normalised signal strength, 0–1
    expected_gross_cents: float   #: α × (VWAP − entry) × 100
    expected_fee_cents: float     #: round-trip fee drag in cents
    expected_net_cents: float     #: gross − fees
    viable: bool                  #: score ≥ threshold
    rejection_reason: str = ""    #: "" if viable


# ═══════════════════════════════════════════════════════════════════════════
#  Main scorer
# ═══════════════════════════════════════════════════════════════════════════

def compute_edge_score(
    entry_price: float,
    no_vwap: float,
    zscore: float,
    volume_ratio: float,
    *,
    whale_confluence: bool = False,
    fee_enabled: bool = True,
    alpha: float | None = None,
    zscore_threshold: float | None = None,
    volume_ratio_threshold: float | None = None,
    min_score: float | None = None,
    tick_size: float = 0.01,
    model_confidence: float | None = None,
) -> EdgeAssessment:
    """Compute the Edge Quality Score for a proposed trade.

    Parameters
    ----------
    entry_price:
        Proposed NO entry price (probability), e.g. 0.47.
    no_vwap:
        Rolling VWAP for the NO token — the mean-reversion anchor.
    zscore:
        Z-score of the YES price spike from ``PanicDetector``.
    volume_ratio:
        Volume ratio of the triggering bar.
    whale_confluence:
        Whether a whale bought NO recently.
    fee_enabled:
        Whether this market charges dynamic fees.
    alpha:
        Expected mean-reversion fraction α ∈ [0, 1].
        Defaults to ``settings.strategy.alpha_default``.
    zscore_threshold:
        PanicDetector z-score threshold.  Defaults to config.
    volume_ratio_threshold:
        PanicDetector volume threshold.  Defaults to config.
    min_score:
        Minimum EQS for viability.  Defaults to ``min_edge_score``.
    tick_size:
        Price grid resolution (0.01 for Polymarket).
    model_confidence:
        When provided (e.g. from RPE signals), scales the signal_quality
        factor in place of the zscore/volume_ratio formula.  This allows
        non-panic signals to pass through the EQS gate with appropriate
        quality scoring based on model confidence.

    Returns
    -------
    EdgeAssessment
        Full diagnostic breakdown of edge quality.
    """
    strat = settings.strategy
    _alpha = alpha if alpha is not None else strat.alpha_default
    z_thresh = (
        zscore_threshold
        if zscore_threshold is not None
        else strat.zscore_threshold
    )
    v_thresh = (
        volume_ratio_threshold
        if volume_ratio_threshold is not None
        else strat.volume_ratio_threshold
    )
    threshold = min_score if min_score is not None else strat.min_edge_score

    tick_cents = tick_size * 100.0  # 1.0

    # ── Factor 1: Regime quality (binary entropy) ──────────────────────
    regime = binary_entropy(entry_price)

    # ── Expected gross spread ──────────────────────────────────────────
    raw_gross = _alpha * max(0.0, no_vwap - entry_price)
    gross_cents = raw_gross * 100.0

    # ── Fee drag (quadratic curve) ─────────────────────────────────────
    entry_fee_frac = get_fee_rate(entry_price, fee_enabled=fee_enabled)
    exit_est = entry_price + raw_gross
    exit_fee_frac = get_fee_rate(exit_est, fee_enabled=fee_enabled)
    fee_cents = (entry_fee_frac + exit_fee_frac) * 100.0

    net_cents = gross_cents - fee_cents

    # ── Factor 2: Fee efficiency ───────────────────────────────────────
    # Floor at eqs_fee_efficiency_floor (default 0.10) so that fees
    # cannot zero the entire geometric-mean EQS.  Trades with poor fee
    # economics are still heavily penalised but not hard-rejected,
    # allowing strong regime/signal/tick factors to rescue them.
    fee_floor = strat.eqs_fee_efficiency_floor
    fee_eff = (
        max(fee_floor, 1.0 - fee_cents / gross_cents)
        if gross_cents > 0
        else 0.0
    )

    # ── Factor 3: Tick viability ───────────────────────────────────────
    # How many discrete ticks can the expected move span?
    gross_ticks = int(gross_cents / tick_cents)  # floor

    if gross_ticks < 1:
        # Expected move can't even span one tick on the discrete grid.
        tick_margin = 0
        tick_viab = 0.0
    else:
        # What does the net P&L look like at the discretised target?
        discretised_gross_cents = gross_ticks * tick_cents
        discretised_net = discretised_gross_cents - fee_cents
        if discretised_net <= 0:
            tick_margin = 0
            tick_viab = 0.0
        else:
            tick_margin = int(discretised_net / tick_cents)
            # Normalise: ≥ 3¢ of net profit → 1.0
            tick_viab = min(1.0, discretised_net / (3.0 * tick_cents))

    # ── Factor 4: Signal quality ───────────────────────────────────────
    if model_confidence is not None:
        # RPE / model-driven signal: confidence directly drives quality
        signal_q = max(0.1, min(1.0, model_confidence))
    else:
        z_excess = (
            max(0.0, (zscore - z_thresh) / z_thresh)
            if z_thresh > 0
            else 0.0
        )
        v_excess = (
            max(0.0, (volume_ratio - v_thresh) / v_thresh)
            if v_thresh > 0
            else 0.0
        )
        # Baseline 0.5 (just cleared PanicDetector), rises with excess
        signal_q = min(1.0, 0.5 + 0.25 * z_excess + 0.25 * v_excess)
    if whale_confluence:
        signal_q = min(1.0, signal_q + 0.15)

    # ── Weighted geometric mean ────────────────────────────────────────
    # score = 100 × ∏ f_i^w_i
    # If ANY factor is ≤ 0 the product is 0 (hard reject).
    if regime <= 0 or fee_eff <= 0 or tick_viab <= 0 or signal_q <= 0:
        score = 0.0
    else:
        score = 100.0 * (
            regime ** W_REGIME
            * fee_eff ** W_FEE
            * tick_viab ** W_TICK
            * signal_q ** W_SIGNAL
        )

    viable = score >= threshold

    # ── Rejection reason taxonomy ──────────────────────────────────────
    reason = ""
    if not viable:
        if gross_cents <= 0:
            reason = "no_mean_reversion_target"
        elif gross_ticks < 1:
            reason = "sub_tick_spread"
        elif tick_viab <= 0:
            reason = "fees_exceed_discretised_spread"
        elif fee_eff <= 0:
            reason = "fees_exceed_spread"
        elif regime < 0.25:
            reason = "low_regime_entropy"
        else:
            reason = "score_below_threshold"

    assessment = EdgeAssessment(
        score=round(score, 2),
        regime_quality=round(regime, 4),
        fee_efficiency=round(fee_eff, 4),
        tick_margin=tick_margin,
        tick_viability=round(tick_viab, 4),
        signal_quality=round(signal_q, 4),
        expected_gross_cents=round(gross_cents, 4),
        expected_fee_cents=round(fee_cents, 4),
        expected_net_cents=round(net_cents, 4),
        viable=viable,
        rejection_reason=reason,
    )

    log.info(
        "edge_assessment",
        entry=entry_price,
        vwap=no_vwap,
        score=assessment.score,
        regime=assessment.regime_quality,
        fee_eff=assessment.fee_efficiency,
        tick_margin=assessment.tick_margin,
        tick_viab=assessment.tick_viability,
        signal_q=assessment.signal_quality,
        gross_cents=assessment.expected_gross_cents,
        fee_cents=assessment.expected_fee_cents,
        net_cents=assessment.expected_net_cents,
        viable=assessment.viable,
        reason=assessment.rejection_reason,
    )

    return assessment
