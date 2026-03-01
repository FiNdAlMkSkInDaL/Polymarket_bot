"""
Resolution Probability Engine (RPE) — independent model-based alpha.

Maintains a calibrated probability estimate for each active market and
fires a ``SignalResult`` when the market price diverges significantly
from the model estimate.  This provides a structurally uncorrelated
second alpha complementing the existing panic-spike mean-reversion
strategy.

Architecture
────────────
    ProbabilityModel (ABC)
        ├── CryptoPriceModel      — log-normal for "Will BTC exceed $X by Y"
        └── GenericBayesianModel   — Beta prior + market-price observations

    ResolutionProbabilityEngine(SignalGenerator)
        Owns a prioritised list of ProbabilityModel instances.
        Routes each market to the first model that can handle it.
        Produces SignalResult with direction metadata (buy_yes / buy_no).

Design decisions
────────────────
* **Risk-neutral drift (μ = 0) for crypto** — Prediction market prices
  are themselves risk-neutral instruments.  Embedding a bullish or
  bearish prior would create systematic bias.  Empirically, μ = 0 is
  a conservative choice: it means we only profit when the crypto model
  is better-calibrated than the market, not when BTC goes up.

* **Beta(2, 2) prior for generic model** — Slight regularization
  prevents a single observation from producing extreme (0.01 or 0.99)
  estimates.  Beta(1, 1) (uniform) would give the market price too
  much authority relative to our uncertainty.

* **Confidence gates size to zero, not to minimum** — Kelly with zero
  edge naturally gives zero size.  Forcing a minimum position when
  confidence is low would violate the Kelly criterion and risk capital
  on low-conviction trades.  The sizer already handles this:
  ``kelly_no_edge`` rejection path is the correct gate.

* **Shadow mode default ON** — A new, unproven alpha must demonstrate
  calibration on live data before committing real capital.  The switch
  to live is a single env-var flip (RPE_SHADOW_MODE=false).
"""

from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from src.core.config import settings
from src.core.logger import get_logger
from src.signals.signal_framework import SignalGenerator, SignalResult

if TYPE_CHECKING:
    from src.data.market_discovery import MarketInfo

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Model Estimate
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelEstimate:
    """A probability estimate from a ProbabilityModel.

    Every estimate MUST carry an explicit ``confidence`` that gates
    downstream trade sizing via the Kelly sizer.  When confidence is
    low (< rpe_min_confidence), the RPE will not fire a signal at all —
    the system knows what it doesn't know.

    Attributes
    ----------
    probability:
        Estimated probability that the YES outcome resolves true.
        Range: [0, 1].
    confidence:
        How much the model trusts its own estimate.  Range: (0, 1].
        Flows into the Kelly sizer as ``1 - uncertainty_penalty``.
    model_name:
        Identifier for the model that produced this estimate (e.g.
        ``"crypto_lognormal"``, ``"generic_bayesian"``).
    metadata:
        Model-specific diagnostics (strike, spot, vol, etc.) for
        logging and backtesting.
    """

    probability: float
    confidence: float
    model_name: str
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  ProbabilityModel ABC
# ═══════════════════════════════════════════════════════════════════════════

class ProbabilityModel(ABC):
    """Base class for category-specific probability models.

    To add a new model (e.g. political, sports), subclass this and
    implement ``can_handle``, ``estimate``, and the ``name`` property.
    Then register the model instance with ``ResolutionProbabilityEngine``
    — no core logic changes required.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this model (e.g. "crypto_lognormal")."""
        ...

    @abstractmethod
    def can_handle(self, market: "MarketInfo") -> bool:
        """Return True if this model should be used for the given market.

        Models are checked in priority order (first match wins).
        """
        ...

    @abstractmethod
    def estimate(
        self,
        market: "MarketInfo",
        market_price: float,
        **kwargs: Any,
    ) -> ModelEstimate | None:
        """Produce a probability estimate for this market.

        Parameters
        ----------
        market:
            Full market descriptor (question, end_date, tags, etc.).
        market_price:
            Current YES token price on Polymarket (0–1).
        **kwargs:
            Model-specific inputs (e.g. ``days_to_resolution``).

        Returns
        -------
        ModelEstimate or None
            None if the model cannot produce a reliable estimate
            (e.g. missing external data).  This is the "no hallucination"
            contract — never fabricate data.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  Crypto question parser
# ═══════════════════════════════════════════════════════════════════════════

# Matches questions like:
#   "Will Bitcoin exceed $100,000 by March 31, 2026?"
#   "Will BTC reach $150k by end of 2026?"
#   "Will ETH be above $5,000 on December 31, 2026?"
_CRYPTO_QUESTION_RE = re.compile(
    r"(?:Will|will)\s+"
    r"(?:Bitcoin|BTC|Ethereum|ETH|bitcoin|ethereum|btc|eth)"
    r".*?"
    r"(?:exceed|reach|above|hit|surpass|over|top)\s+"
    r"\$?([\d,]+(?:\.\d+)?[kKmM]?)",
    re.IGNORECASE,
)

_CRYPTO_TICKER_RE = re.compile(
    r"\b(BTC|Bitcoin|ETH|Ethereum)\b", re.IGNORECASE
)

_TICKER_MAP = {
    "btc": "BTC",
    "bitcoin": "BTC",
    "eth": "ETH",
    "ethereum": "ETH",
}


def parse_crypto_question(question: str) -> tuple[str, float] | None:
    """Extract (ticker, strike_price) from a crypto prediction market question.

    Returns None if the question doesn't match the expected pattern.
    This is intentionally conservative — it's better to fall through
    to the GenericBayesianModel than to misparse a non-crypto market.

    Examples
    --------
    >>> parse_crypto_question("Will Bitcoin exceed $100,000 by March 31?")
    ('BTC', 100000.0)
    >>> parse_crypto_question("Will ETH reach $5k by end of 2026?")
    ('ETH', 5000.0)
    >>> parse_crypto_question("Will the US enter a recession?")
    # None
    """
    price_match = _CRYPTO_QUESTION_RE.search(question)
    if not price_match:
        return None

    ticker_match = _CRYPTO_TICKER_RE.search(question)
    if not ticker_match:
        return None

    ticker = _TICKER_MAP.get(ticker_match.group(1).lower())
    if not ticker:
        return None

    raw_price = price_match.group(1).replace(",", "")
    # Handle k/K suffix (e.g. "150k" → 150000)
    multiplier = 1.0
    if raw_price[-1].lower() == "k":
        multiplier = 1_000
        raw_price = raw_price[:-1]
    elif raw_price[-1].lower() == "m":
        multiplier = 1_000_000
        raw_price = raw_price[:-1]

    try:
        strike = float(raw_price) * multiplier
    except ValueError:
        return None

    if strike <= 0:
        return None

    return (ticker, strike)


# ═══════════════════════════════════════════════════════════════════════════
#  CryptoPriceModel
# ═══════════════════════════════════════════════════════════════════════════

def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the complementary error function.

    Uses ``math.erfc`` which is available in the stdlib — no numpy
    dependency needed for this critical-path computation.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2))


class CryptoPriceModel(ProbabilityModel):
    r"""Log-normal probability model for crypto price threshold markets.

    Estimates P(YES resolves) for markets like "Will BTC exceed $X by Y"
    using the Black-Scholes-style formula:

    .. math::
        P(S_T > K) = \Phi(d_2)

    where:

    .. math::
        d_2 = \frac{\ln(S/K) + (\mu - \sigma^2/2) T}{\sigma \sqrt{T}}

    Design decision — **μ = 0 (risk-neutral drift)**:
        Crypto assets have historically exhibited positive drift, but
        prediction markets are themselves risk-neutral pricing instruments.
        Using μ > 0 would embed a systematic bullish bias, causing the
        model to consistently overestimate YES probabilities.  With μ = 0,
        we only profit when we are better-calibrated than the market on
        volatility and time-to-expiry — a sustainable edge.  If the
        market is pricing in an implicit drift and we are not, our signal
        will correctly identify the divergence without taking a directional
        crypto bet.

    Parameters
    ----------
    price_fn:
        Callable returning the latest spot price (e.g. BTC/USDC) or
        None if unavailable.  This should be wired to the adverse
        selection guard's existing Binance WS connection to avoid
        opening a second socket.
    vol_override:
        Optional fixed annualized volatility.  Used in backtesting
        when historical vol is known.  If None, falls back to
        ``settings.strategy.rpe_crypto_vol_default``.
    """

    def __init__(
        self,
        price_fn: Callable[[], float | None],
        *,
        vol_override: float | None = None,
    ) -> None:
        self._price_fn = price_fn
        self._vol_override = vol_override

    @property
    def name(self) -> str:
        return "crypto_lognormal"

    def can_handle(self, market: "MarketInfo") -> bool:
        """True if the market question matches a crypto price pattern.

        Checks both the question text (regex) and the tags field.
        The regex is conservative: only fires for well-structured
        "Will X exceed $Y by Z" patterns.
        """
        parsed = parse_crypto_question(market.question)
        if parsed is not None:
            return True
        tags = (getattr(market, "tags", "") or "").lower()
        if "crypto" in tags and any(
            kw in market.question.lower()
            for kw in ("price", "exceed", "above", "reach", "hit", "below")
        ):
            return True
        return False

    def estimate(
        self,
        market: "MarketInfo",
        market_price: float,
        **kwargs: Any,
    ) -> ModelEstimate | None:
        """Compute P(S_T > K) using the log-normal model.

        Returns None (not a guess) when:
        - External price feed is unavailable
        - Question cannot be parsed for ticker/strike
        - Market has no end_date (needed for T)
        """
        parsed = parse_crypto_question(market.question)
        if parsed is None:
            return None

        ticker, strike = parsed

        spot = self._price_fn()
        if spot is None or spot <= 0:
            log.debug("rpe_crypto_no_spot", market=market.condition_id, ticker=ticker)
            return None

        # Time to expiry in years
        if market.end_date is None:
            return None

        now = datetime.now(timezone.utc)
        days_left = (market.end_date - now).total_seconds() / 86400.0
        if days_left <= 0:
            # Market already expired — probability collapses to 0 or 1
            prob = 1.0 if spot > strike else 0.0
            return ModelEstimate(
                probability=prob,
                confidence=0.95,
                model_name=self.name,
                metadata={
                    "ticker": ticker,
                    "spot": spot,
                    "strike": strike,
                    "days_left": 0,
                    "reason": "expired",
                },
            )

        T = days_left / 365.0

        # Annualized volatility
        sigma = self._vol_override or settings.strategy.rpe_crypto_vol_default

        # Black-Scholes d2 with μ = 0 (risk-neutral)
        mu = 0.0
        d2 = (math.log(spot / strike) + (mu - 0.5 * sigma**2) * T) / (
            sigma * math.sqrt(T)
        )
        prob = _normal_cdf(d2)

        # Clamp to avoid exact 0/1 (which would give infinite Kelly edge)
        prob = max(0.01, min(0.99, prob))

        # ── Confidence estimation ────────────────────────────────────────
        # Confidence is higher when:
        #  1. More time to expiry (vol estimate more reliable, less regime change)
        #  2. Price is far from strike (model is more certain)
        #  3. Implied vol is low (less noise)
        #
        # The confidence formula is:
        #   base = sigmoid(|d2|) — further from ATM = more certain
        #   time_factor = min(1, sqrt(days_left / 30))
        #   conf = base * time_factor
        #
        # Design decision — time_factor increases with sqrt(days) because
        # the log-normal model is more reliable over longer horizons
        # (central limit theorem on log-returns) but saturates because
        # regime change risk caps the benefit of long horizons.
        d2_abs = abs(d2)
        base_conf = 2.0 / (1.0 + math.exp(-1.5 * d2_abs)) - 1.0  # sigmoid [0, 1)
        time_factor = min(1.0, math.sqrt(days_left / 30.0))
        confidence = max(0.05, min(0.95, base_conf * time_factor))

        return ModelEstimate(
            probability=prob,
            confidence=confidence,
            model_name=self.name,
            metadata={
                "ticker": ticker,
                "spot": spot,
                "strike": strike,
                "sigma": sigma,
                "T_years": round(T, 4),
                "days_left": round(days_left, 1),
                "d2": round(d2, 4),
                "mu": mu,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
#  GenericBayesianModel
# ═══════════════════════════════════════════════════════════════════════════

class GenericBayesianModel(ProbabilityModel):
    r"""Bayesian prior model for non-crypto markets.

    Combines three information sources:

    1. **Prior**: ``Beta(α₀, β₀)`` with ``α₀ = β₀ = 2`` (mildly
       uninformative, peaked at 0.5).  Why not ``Beta(1,1)`` (uniform)?
       Because a single noisy market-price observation would dominate
       the posterior, producing extreme estimates (0.05 or 0.95) with
       unwarranted confidence.  The ``Beta(2,2)`` regularization pulls
       the estimate toward 0.5 absent strong evidence — this is a
       deliberate conservatism choice.

    2. **Market price as noisy signal**: The current YES price ``p`` is
       treated as a draw from the market's collective wisdom.  We
       update the prior with observation weight ``n`` (configurable via
       ``RPE_BAYESIAN_OBS_WEIGHT``):

       .. math::
           \alpha' = \alpha_0 + n \cdot p \\
           \beta' = \beta_0 + n \cdot (1 - p)

       Higher ``n`` = more trust in the market price.  The default
       ``n = 5`` reflects that liquid Polymarket prices are informative
       but not perfectly efficient (thin books, retail noise).

    3. **Time decay**: As ``days_to_resolution`` shrinks, the market
       price becomes a stronger signal (more information has been
       revealed).  Counter-intuitively, confidence *increases* as
       resolution approaches, because:
       - Fewer unknown future events can shift the probability
       - Market participants have more information
       - Price discovery converges

       This is implemented as:
       ``confidence *= min(1.0, 1.0 - (days / 90) * 0.3)``
       At 90+ days: up to 30% confidence discount; at 0 days: no discount.

    Parameters
    ----------
    alpha0:
        Prior α parameter for Beta distribution.
    beta0:
        Prior β parameter for Beta distribution.
    obs_weight:
        How many pseudo-observations the market price contributes.
        Default from ``settings.strategy.rpe_bayesian_obs_weight``.
    prior_k:
        Beta prior concentration parameter.  When provided, the prior
        is anchored to market price: α₀ = k * p, β₀ = k * (1 - p).
        This eliminates the persistent divergence on tail markets
        caused by the old fixed Beta(2,2) prior pulling toward 50%.
        Default from ``settings.strategy.rpe_prior_k``.
    """

    def __init__(
        self,
        *,
        alpha0: float = 2.0,
        beta0: float = 2.0,
        obs_weight: float | None = None,
        prior_k: float | None = None,
    ) -> None:
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._obs_weight = obs_weight
        self._prior_k = prior_k

    @property
    def _n(self) -> float:
        """Observation weight (lazy — reads config at call time for testability)."""
        if self._obs_weight is not None:
            return self._obs_weight
        return settings.strategy.rpe_bayesian_obs_weight

    @property
    def _k(self) -> float:
        """Beta prior concentration (lazy — reads config at call time)."""
        if self._prior_k is not None:
            return self._prior_k
        return settings.strategy.rpe_prior_k

    @property
    def name(self) -> str:
        return "generic_bayesian"

    def can_handle(self, market: "MarketInfo") -> bool:
        """Always returns True — this is the fallback model.

        The RPE checks models in priority order.  This model sits at
        the end of the list and handles every market that no
        category-specific model claimed.
        """
        return True

    def estimate(
        self,
        market: "MarketInfo",
        market_price: float,
        **kwargs: Any,
    ) -> ModelEstimate | None:
        """Produce a Bayesian posterior estimate.

        Parameters
        ----------
        market_price:
            Current YES token price on Polymarket (0–1).
        **kwargs:
            Must include ``days_to_resolution: int``.

        Returns None if market_price is outside (0, 1) — this would
        indicate corrupt data, not a legitimate trading opportunity.
        """
        if market_price <= 0 or market_price >= 1:
            return None

        days = kwargs.get("days_to_resolution", 30)
        n = self._n
        k = self._k

        # Adaptive prior: anchor α₀, β₀ to market price so the prior
        # does not create artificial divergence on tail markets.
        # When k=0, fall back to the legacy fixed alpha0/beta0 constructor args.
        if k > 0:
            alpha0 = k * market_price
            beta0 = k * (1.0 - market_price)
        else:
            alpha0 = self._alpha0
            beta0 = self._beta0

        # Observations are market-price pseudo-counts.  Combined with the
        # market-anchored prior this yields posterior = market_price (divergence
        # = 0).  That is the CORRECT behaviour for a model with no independent
        # external signal: "I have no information beyond the market price, so
        # my best estimate IS the market price."  The GenericBayesianModel
        # therefore never fires RPE signals on its own — alpha comes only from
        # models with real external data (e.g. CryptoPriceModel with BTC spot).
        alpha_post = alpha0 + n * market_price
        beta_post = beta0 + n * (1.0 - market_price)

        # Posterior mean
        prob = alpha_post / (alpha_post + beta_post)

        # Clamp
        prob = max(0.01, min(0.99, prob))

        # ── Confidence estimation ────────────────────────────────────────
        # Three factors:
        #
        # 1. Posterior concentration — higher α+β = tighter posterior.
        #    We normalize to [0, 1] using a saturation function.
        total = alpha_post + beta_post
        concentration_conf = 1.0 - 1.0 / (1.0 + total / 10.0)

        # 2. Entropy penalty — at p=0.5 the model is maximally uncertain
        #    about the binary outcome.  Use binary entropy H(p):
        #    H(0.5) = 1 (max uncertainty), H(p→0 or 1) → 0.
        if 0 < prob < 1:
            H = -(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))
        else:
            H = 0.0
        entropy_penalty = H  # ∈ [0, 1]

        # 3. Time factor — closer to resolution → more confident in
        #    the market price signal.
        if days >= 90:
            time_factor = 0.7
        elif days <= 0:
            time_factor = 1.0
        else:
            time_factor = 1.0 - (days / 90.0) * 0.3

        confidence = concentration_conf * (1.0 - 0.5 * entropy_penalty) * time_factor
        confidence = max(0.05, min(0.95, confidence))

        return ModelEstimate(
            probability=prob,
            confidence=confidence,
            model_name=self.name,
            metadata={
                "alpha_post": round(alpha_post, 3),
                "beta_post": round(beta_post, 3),
                "market_price": market_price,
                "days_to_resolution": days,
                "concentration_conf": round(concentration_conf, 3),
                "entropy_penalty": round(entropy_penalty, 3),
                "time_factor": round(time_factor, 3),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
#  RPE Signal Data (metadata convention)
# ═══════════════════════════════════════════════════════════════════════════

# The RPE stores these keys in SignalResult.metadata:
#   "model_probability": float     — model's estimated YES probability
#   "confidence": float            — model's self-assessed confidence
#   "divergence": float            — market_price - model_probability
#   "direction": str               — "buy_yes" or "buy_no"
#   "model_name": str              — which model produced the estimate
#   "shadow_mode": bool            — True if running in shadow mode
#   "model_metadata": dict         — model-specific diagnostics


# ═══════════════════════════════════════════════════════════════════════════
#  ResolutionProbabilityEngine (SignalGenerator)
# ═══════════════════════════════════════════════════════════════════════════

class ResolutionProbabilityEngine(SignalGenerator):
    """Main RPE orchestrator — routes markets to models and fires signals.

    Implements the ``SignalGenerator`` ABC from ``signal_framework.py``.
    Produces a ``SignalResult`` when the market price diverges from the
    model estimate by more than a confidence-adjusted threshold.

    Signal firing rule:

        |market_price − model_estimate| > confidence_threshold × (1 − confidence)

    The threshold *widens* when confidence is low, preventing the RPE
    from firing on markets it doesn't understand.  When confidence is
    high (e.g. crypto model with fresh Binance data), the threshold
    tightens, allowing the RPE to capture smaller mispricings.

    Parameters
    ----------
    models:
        Priority-ordered list of ``ProbabilityModel`` instances.
        The first model whose ``can_handle()`` returns True is used.
    confidence_threshold:
        Base threshold for divergence detection (default from config).
    shadow_mode:
        If True, signals are logged but do not trigger orders.
    min_confidence:
        Minimum model confidence to produce a signal.  Below this,
        the RPE stays silent (knows what it doesn't know).
    """

    def __init__(
        self,
        models: list[ProbabilityModel],
        *,
        confidence_threshold: float | None = None,
        shadow_mode: bool | None = None,
        min_confidence: float | None = None,
        generic_enabled: bool | None = None,
    ) -> None:
        self._models = models
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.strategy.rpe_confidence_threshold
        )
        self._shadow_mode = (
            shadow_mode if shadow_mode is not None
            else settings.strategy.rpe_shadow_mode
        )
        self._min_confidence = (
            min_confidence if min_confidence is not None
            else settings.strategy.rpe_min_confidence
        )
        self._generic_enabled = (
            generic_enabled if generic_enabled is not None
            else settings.strategy.rpe_generic_enabled
        )

        # Cache: market_id → latest ModelEstimate
        self._estimates: dict[str, ModelEstimate] = {}

    @property
    def name(self) -> str:
        return "resolution_probability_engine"

    @property
    def shadow_mode(self) -> bool:
        return self._shadow_mode

    def get_estimate(self, market_id: str) -> ModelEstimate | None:
        """Return the latest model estimate for a market, or None."""
        return self._estimates.get(market_id)

    def clear_market(self, condition_id: str) -> None:
        """Remove all cached state for a market that has been evicted.

        Called from TradingBot._unwire_market() when a market leaves the
        active market set. Clears the estimate cache so stale data from
        the evicted market doesn't persist in memory.
        """
        self._estimates.pop(condition_id, None)
        log.info("rpe_market_cleared", condition_id=condition_id)

    def evaluate(self, **kwargs: Any) -> SignalResult | None:
        """Evaluate the RPE signal for a single market.

        Expected keyword arguments
        --------------------------
        market : MarketInfo
            Full market descriptor.
        market_price : float
            Current YES token price (0–1).
        days_to_resolution : int
            Days until market resolution.

        Returns
        -------
        SignalResult or None
            None if no model can handle the market, confidence is too
            low, or divergence is within threshold.
        """
        market: MarketInfo | None = kwargs.get("market")
        market_price: float = kwargs.get("market_price", 0.0)
        days_to_resolution: int = kwargs.get("days_to_resolution", 30)

        if market is None or market_price <= 0 or market_price >= 1:
            return None

        # Route to first matching model
        estimate: ModelEstimate | None = None
        for model in self._models:
            if model.can_handle(market):
                estimate = model.estimate(
                    market,
                    market_price,
                    days_to_resolution=days_to_resolution,
                )
                if estimate is not None:
                    break

        if estimate is None:
            return None

        # Cache the estimate (always — even for gated models, so
        # calibration data is recorded and available via get_estimate())
        self._estimates[market.condition_id] = estimate

        # ── Generic model gate ──────────────────────────────────────────
        # When rpe_generic_enabled is False, the generic Bayesian model
        # still runs and its estimates are cached/logged for calibration,
        # but it does NOT produce live trading signals.  This prevents
        # the Bayesian shrinkage artifact from generating false edge
        # against extreme market prices.
        if (
            estimate.model_name == "generic_bayesian"
            and not self._generic_enabled
        ):
            log.debug(
                "rpe_generic_gated",
                market=market.condition_id,
                model_prob=round(estimate.probability, 4),
                market_price=round(market_price, 4),
            )
            return None

        # ── Confidence gate ─────────────────────────────────────────────
        if estimate.confidence < self._min_confidence:
            log.debug(
                "rpe_low_confidence",
                market=market.condition_id,
                confidence=round(estimate.confidence, 3),
                min_required=self._min_confidence,
                model=estimate.model_name,
            )
            return None

        # ── Divergence check ────────────────────────────────────────────
        divergence = market_price - estimate.probability
        abs_div = abs(divergence)

        # Adaptive threshold: widens with uncertainty (1 - confidence)
        effective_threshold = self._confidence_threshold * (
            1.0 + (1.0 - estimate.confidence)
        )

        if abs_div <= effective_threshold:
            return None

        # ── Direction ───────────────────────────────────────────────────
        # If market_price > model_estimate: market overvalues YES → buy NO
        # If market_price < model_estimate: market undervalues YES → buy YES
        direction = "buy_no" if divergence > 0 else "buy_yes"

        # ── Score (0–1, saturates at 20¢ divergence) ────────────────────
        score = min(1.0, abs_div / 0.20)

        log.info(
            "rpe_signal_fired",
            market=market.condition_id,
            model=estimate.model_name,
            model_prob=round(estimate.probability, 4),
            market_price=round(market_price, 4),
            divergence=round(divergence, 4),
            confidence=round(estimate.confidence, 3),
            direction=direction,
            score=round(score, 3),
            shadow=self._shadow_mode,
            threshold=round(effective_threshold, 4),
        )

        return SignalResult(
            name=self.name,
            market_id=market.condition_id,
            score=score,
            metadata={
                "model_probability": estimate.probability,
                "confidence": estimate.confidence,
                "divergence": divergence,
                "direction": direction,
                "model_name": estimate.model_name,
                "shadow_mode": self._shadow_mode,
                "model_metadata": estimate.metadata,
                "effective_threshold": effective_threshold,
                # Uncertainty penalty for Kelly sizer integration:
                # higher confidence → lower penalty → larger positions
                "uncertainty_penalty": 1.0 - estimate.confidence,
            },
            timestamp=time.time(),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  RPE Calibration Tracker
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _CalibrationEntry:
    """Single RPE signal record for calibration scoring."""

    market_id: str
    model_prob: float
    market_price: float
    direction: str
    timestamp: float
    shadow: bool = True
    resolution_price: float | None = None  # 0.0 or 1.0 once resolved


class RPECalibrationTracker:
    """Ring-buffer tracker that scores RPE signal calibration.

    Records every fired RPE signal (shadow or live) and, once a market
    resolves, computes Brier score, log-loss, and direction accuracy
    across resolved entries.

    Thread-safe via Python's GIL for single-writer patterns.

    Parameters
    ----------
    max_entries:
        Maximum entries in the ring buffer.  Oldest are evicted first.
    """

    def __init__(self, *, max_entries: int = 500) -> None:
        self._entries: deque[_CalibrationEntry] = deque(maxlen=max_entries)
        self._by_market: dict[str, list[_CalibrationEntry]] = {}

    # ── Recording ──────────────────────────────────────────────────────

    def record_signal(
        self,
        market_id: str,
        model_prob: float,
        market_price: float,
        direction: str,
        timestamp: float,
        *,
        shadow: bool = True,
    ) -> None:
        """Record a fired RPE signal for later calibration."""
        entry = _CalibrationEntry(
            market_id=market_id,
            model_prob=model_prob,
            market_price=market_price,
            direction=direction,
            timestamp=timestamp,
            shadow=shadow,
        )
        self._entries.append(entry)
        self._by_market.setdefault(market_id, []).append(entry)

    def on_market_resolved(self, market_id: str, resolution_price: float) -> None:
        """Mark all entries for *market_id* as resolved.

        Parameters
        ----------
        resolution_price:
            0.0 (NO wins) or 1.0 (YES wins).
        """
        for e in self._by_market.get(market_id, []):
            e.resolution_price = resolution_price

    # ── Scoring ────────────────────────────────────────────────────────

    def _resolved(self) -> list[_CalibrationEntry]:
        return [e for e in self._entries if e.resolution_price is not None]

    @property
    def total_signals(self) -> int:
        return len(self._entries)

    @property
    def live_signals(self) -> int:
        return sum(1 for e in self._entries if not e.shadow)

    @property
    def shadow_signals(self) -> int:
        return sum(1 for e in self._entries if e.shadow)

    @property
    def resolved_count(self) -> int:
        return len(self._resolved())

    def compute_brier_score(self) -> float | None:
        """Mean Brier score across resolved signals.  Lower is better.

        Returns None if no resolved signals exist.
        """
        resolved = self._resolved()
        if not resolved:
            return None
        total = 0.0
        for e in resolved:
            assert e.resolution_price is not None
            total += (e.model_prob - e.resolution_price) ** 2
        return total / len(resolved)

    def compute_log_loss(self) -> float | None:
        """Mean log-loss across resolved signals.  Lower is better.

        Returns None if no resolved signals exist.
        """
        resolved = self._resolved()
        if not resolved:
            return None
        eps = 1e-15
        total = 0.0
        for e in resolved:
            assert e.resolution_price is not None
            p = max(eps, min(1 - eps, e.model_prob))
            y = e.resolution_price
            total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total / len(resolved)

    def compute_direction_accuracy(self) -> float | None:
        """Fraction of resolved signals where model direction matched resolution.

        Returns None if no resolved signals exist.
        """
        resolved = self._resolved()
        if not resolved:
            return None
        correct = 0
        for e in resolved:
            assert e.resolution_price is not None
            if e.direction == "buy_yes" and e.resolution_price == 1.0:
                correct += 1
            elif e.direction == "buy_no" and e.resolution_price == 0.0:
                correct += 1
        return correct / len(resolved)

    def calibration_summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging / Telegram."""
        resolved_n = self.resolved_count
        summary: dict[str, Any] = {
            "total_signals": self.total_signals,
            "live_signals": self.live_signals,
            "shadow_signals": self.shadow_signals,
            "resolved": resolved_n,
        }
        if resolved_n > 0:
            brier = self.compute_brier_score()
            logloss = self.compute_log_loss()
            accuracy = self.compute_direction_accuracy()
            summary["brier_score"] = round(brier, 4) if brier is not None else None
            summary["log_loss"] = round(logloss, 4) if logloss is not None else None
            summary["direction_accuracy"] = (
                round(accuracy, 4) if accuracy is not None else None
            )
        return summary
