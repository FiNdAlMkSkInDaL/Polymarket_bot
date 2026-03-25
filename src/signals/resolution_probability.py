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
from src.signals.tag_prior_registry import TagPriorRegistry

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

        # ── Per-market adaptive σ via EWMA (OE-5) ─────────────────────
        # Instead of a single global constant, track an EWMA of
        # log-return² from the spot price feed.  Updated on each
        # evaluate() call (every 5s via _rpe_crypto_retrigger_loop).
        self._prev_spot: float | None = None
        self._ewma_var: float = 0.0
        self._ewma_initialised: bool = False
        self._EWMA_LAMBDA: float = 0.94
        self._MIN_ANNUALIZED_VOL: float = 0.30
        self._MAX_ANNUALIZED_VOL: float = 1.50
        # Constants for annualisation: ~525960 minutes/year, spot
        # sampled every ~5 seconds (12/min), so scale sqrt(525960*12).
        self._ANNUALISE_FACTOR: float = (525960.0 * 12.0) ** 0.5

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

        # ── Adaptive annualized volatility (OE-5) ─────────────────────
        # 1) Explicit override (backtesting) takes priority.
        # 2) EWMA from live spot feed (if enough samples).
        # 3) Fall back to the global config constant.
        if self._vol_override:
            sigma = self._vol_override
        elif self._ewma_initialised and self._ewma_var > 0:
            # Annualise the per-sample EWMA std-dev
            raw_annual = (self._ewma_var ** 0.5) * self._ANNUALISE_FACTOR
            sigma = max(self._MIN_ANNUALIZED_VOL,
                        min(self._MAX_ANNUALIZED_VOL, raw_annual))
        else:
            sigma = settings.strategy.rpe_crypto_vol_default

        # Update EWMA variance from the latest spot observation
        if self._prev_spot is not None and self._prev_spot > 0 and spot > 0:
            log_ret = math.log(spot / self._prev_spot)
            if not self._ewma_initialised:
                self._ewma_var = log_ret * log_ret
                self._ewma_initialised = True
            else:
                self._ewma_var = (
                    self._EWMA_LAMBDA * self._ewma_var
                    + (1.0 - self._EWMA_LAMBDA) * log_ret * log_ret
                )
        self._prev_spot = spot

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
#  GenericBayesianModel — Dynamic Prior Generation Engine
# ═══════════════════════════════════════════════════════════════════════════

class GenericBayesianModel(ProbabilityModel):
    r"""Bayesian prior model for non-crypto markets.

    Upgraded with the Dynamic Prior Generation Engine (DPGE):

    1. **Tag-based empirical priors**: Routes market tags to calibrated
       Beta(α₀, β₀) distributions via ``TagPriorRegistry``.  Different
       market categories receive historically accurate priors instead of
       a static Beta(2,2).

    2. **L2 order-book imbalance as continuous observation**: Uses the
       ``book_depth_ratio`` (bid/ask volume imbalance) as a continuous
       signal.  The log-ratio is converted to additive pseudo-counts:

       .. math::
           \alpha' = \alpha_0 + n_{\text{eff}} \cdot p
                     + \kappa_{\text{eff}} \cdot \max(0, \ln r)
           \\
           \beta'  = \beta_0  + n_{\text{eff}} \cdot (1-p)
                     + \kappa_{\text{eff}} \cdot \max(0, \ln(1/r))

       where r = book_depth_ratio, κ = ``rpe_l2_kappa``.

    3. **Sigmoid time-decay (theta calibration)**: As T → 0 the model
       becomes less responsive to prior beliefs and L2 noise, heavily
       weighting the passage of time (market price convergence).

       The prior/observation balance follows a sigmoid:

       .. math::
           w_{\text{prior}}(t) = \frac{1}{1 + e^{-\gamma(t - t_{\text{half}})}}

       where t = days_to_resolution / max(total_market_duration, 1),
       γ = ``rpe_theta_gamma``, t_half = ``rpe_theta_half``.

       Near expiry (t → 0):  w_prior → 0, n_eff → ∞, κ_eff → 0.
       Early in market (t → 1): w_prior → 1, prior dominates.

    Backward compatibility
    ──────────────────────
    When ``rpe_dynamic_prior_enabled = False`` or ``rpe_prior_k > 0``
    and the tag registry returns no match, the model falls back to
    market-anchored adaptive priors (posterior = market price, zero
    divergence).

    Parameters
    ----------
    alpha0:
        Legacy prior α parameter (used when dynamic priors disabled).
    beta0:
        Legacy prior β parameter.
    obs_weight:
        Base pseudo-observation count for market-price updates.
    prior_k:
        Market-anchored prior concentration (legacy fallback).
    tag_registry:
        Injected TagPriorRegistry (default: singleton instance).
    """

    def __init__(
        self,
        *,
        alpha0: float = 2.0,
        beta0: float = 2.0,
        obs_weight: float | None = None,
        prior_k: float | None = None,
        tag_registry: TagPriorRegistry | None = None,
    ) -> None:
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._obs_weight = obs_weight
        self._prior_k = prior_k
        self._tag_registry = tag_registry or TagPriorRegistry()

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

    # ── Time-decay kernel ────────────────────────────────────────────

    @staticmethod
    def _sigmoid_time_weight(
        days_to_resolution: float,
        total_duration_days: float,
        gamma: float,
        t_half: float,
    ) -> float:
        """Compute w_prior(t) — prior weight via sigmoid time-decay.

        Returns a value in (0, 1):
            - Near 1 when plenty of time remains (prior dominates).
            - Near 0 when market is about to resolve (market price dominates).
        """
        if total_duration_days <= 0:
            return 0.0  # market already expired → trust market price
        t = max(0.0, min(1.0, days_to_resolution / total_duration_days))
        # Sigmoid: 1 / (1 + exp(-γ(t - t_half)))
        exponent = -gamma * (t - t_half)
        # Clamp exponent to prevent overflow
        exponent = max(-20.0, min(20.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

    # ── L2 imbalance pseudo-counts ───────────────────────────────────

    @staticmethod
    def _l2_pseudo_counts(
        book_depth_ratio: float,
        kappa_eff: float,
    ) -> tuple[float, float]:
        """Convert book_depth_ratio to (α_delta, β_delta) pseudo-counts.

        Uses ln(r) clamped to [-2, 2] to prevent outlier book states
        from dominating.  The log-transform ensures symmetric treatment:
        ratio 0.5 and 2.0 contribute equal-magnitude counts in opposite
        directions.

        Returns (alpha_delta, beta_delta) — both non-negative.
        """
        if book_depth_ratio <= 0 or kappa_eff <= 0:
            return 0.0, 0.0

        ln_r = math.log(book_depth_ratio)
        ln_r = max(-2.0, min(2.0, ln_r))  # clamp

        alpha_delta = kappa_eff * max(0.0, ln_r)   # r > 1 → YES support
        beta_delta = kappa_eff * max(0.0, -ln_r)    # r < 1 → NO support
        return alpha_delta, beta_delta

    def estimate(
        self,
        market: "MarketInfo",
        market_price: float,
        **kwargs: Any,
    ) -> ModelEstimate | None:
        """Produce a Bayesian posterior estimate with dynamic priors.

        Parameters
        ----------
        market_price:
            Current YES token price on Polymarket (0–1).
        **kwargs:
            days_to_resolution : int
                Days until market resolution.
            total_duration_days : float
                Total market lifespan in days (for theta normalisation).
                Defaults to 90 if not provided.
            book_depth_ratio : float | None
                L2 bid/ask depth ratio.  None disables L2 update.
            l2_reliable : bool
                Whether the L2 book is in SYNCED state.

        Returns None if market_price is outside (0, 1).
        """
        if market_price <= 0 or market_price >= 1:
            return None

        days = kwargs.get("days_to_resolution", 30)
        total_duration = kwargs.get("total_duration_days", 90.0)
        book_depth_ratio = kwargs.get("book_depth_ratio", None)
        l2_reliable = kwargs.get("l2_reliable", False)

        n = self._n
        k = self._k
        strat = settings.strategy

        dynamic_enabled = strat.rpe_dynamic_prior_enabled
        l2_kappa = strat.rpe_l2_kappa
        theta_gamma = strat.rpe_theta_gamma
        theta_half = strat.rpe_theta_half

        # ── Step 1: Resolve prior ────────────────────────────────────
        prior_source = "market_anchored"
        if dynamic_enabled:
            tag_alpha, tag_beta, prior_source = self._tag_registry.get_prior(
                getattr(market, "tags", ""),
            )
            alpha0 = tag_alpha
            beta0 = tag_beta
        elif k > 0:
            # Legacy market-anchored adaptive prior
            alpha0 = k * market_price
            beta0 = k * (1.0 - market_price)
            prior_source = "market_anchored"
        else:
            alpha0 = self._alpha0
            beta0 = self._beta0
            prior_source = "static_legacy"

        # ── Step 2: Compute time-decay kernel ────────────────────────
        w_prior = self._sigmoid_time_weight(
            days_to_resolution=float(days),
            total_duration_days=float(total_duration),
            gamma=theta_gamma,
            t_half=theta_half,
        )
        w_obs = 1.0 - w_prior
        eps = 1e-6

        # Scale observation weight: as w_obs → 1 (near expiry),
        # n_eff → large, market price dominates posterior.
        n_eff = n * (w_obs / (w_prior + eps))
        # Cap to prevent numerical explosion in terminal hours
        n_eff = min(n_eff, 200.0)

        # Scale L2 kappa by w_prior so book noise is suppressed
        # near expiry (when the market price is most informative).
        kappa_eff = l2_kappa * w_prior

        # ── Step 3: L2 imbalance pseudo-counts ───────────────────────
        alpha_l2, beta_l2 = 0.0, 0.0
        l2_active = False
        if book_depth_ratio is not None and l2_reliable and kappa_eff > 0.01:
            alpha_l2, beta_l2 = self._l2_pseudo_counts(book_depth_ratio, kappa_eff)
            l2_active = True

        # ── Step 4: Bayesian update ──────────────────────────────────
        alpha_post = alpha0 + n_eff * market_price + alpha_l2
        beta_post = beta0 + n_eff * (1.0 - market_price) + beta_l2

        # Posterior mean
        prob = alpha_post / (alpha_post + beta_post)

        # Clamp
        prob = max(0.01, min(0.99, prob))

        # ── Step 5: Confidence estimation ────────────────────────────
        # Factor 1: Posterior concentration (tighter → more confident)
        total = alpha_post + beta_post
        concentration_conf = 1.0 - 1.0 / (1.0 + total / 10.0)

        # Factor 2: Entropy penalty — at p=0.5 model is maximally
        # uncertain about the binary outcome.
        if 0 < prob < 1:
            H = -(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))
        else:
            H = 0.0
        entropy_penalty = H  # ∈ [0, 1]

        # Factor 3: Time factor — use sigmoid w_prior directly.
        # High w_prior (early market) → high confidence in our prior.
        # Low w_prior (near expiry) → low confidence (model defers
        # to market, should NOT fire signals).
        time_factor = max(0.05, w_prior)

        confidence = concentration_conf * (1.0 - 0.5 * entropy_penalty) * time_factor
        confidence = max(0.05, min(0.95, confidence))

        return ModelEstimate(
            probability=prob,
            confidence=confidence,
            model_name=self.name,
            metadata={
                "alpha_post": round(alpha_post, 3),
                "beta_post": round(beta_post, 3),
                "alpha0": round(alpha0, 3),
                "beta0": round(beta0, 3),
                "market_price": market_price,
                "days_to_resolution": days,
                "total_duration_days": round(total_duration, 1),
                "concentration_conf": round(concentration_conf, 3),
                "entropy_penalty": round(entropy_penalty, 3),
                "time_factor": round(time_factor, 3),
                "w_prior": round(w_prior, 4),
                "n_eff": round(n_eff, 2),
                "kappa_eff": round(kappa_eff, 3),
                "prior_source": prior_source,
                "l2_active": l2_active,
                "book_depth_ratio": round(book_depth_ratio, 3) if book_depth_ratio is not None else None,
                "dynamic_prior_enabled": dynamic_enabled,
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

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

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

    def evaluate_probability_dislocation(
        self,
        *,
        market_id: str,
        market_price: float,
        implied_probability: float,
        confidence: float,
        model_name: str,
        shadow_mode: bool | None = None,
        model_metadata: dict[str, Any] | None = None,
        signal_name: str | None = None,
    ) -> SignalResult | None:
        """Apply the standard RPE threshold math to an external fair value.

        Used by secondary alpha engines that derive a fair YES probability
        outside the built-in model stack but still want identical divergence,
        confidence, and direction logic before routing into the RPE execution
        funnel.
        """
        if market_price <= 0 or market_price >= 1:
            return None

        prob = max(0.01, min(0.99, float(implied_probability)))
        conf = max(0.01, min(0.99, float(confidence)))
        shadow = self._shadow_mode if shadow_mode is None else shadow_mode

        if conf < self._min_confidence:
            log.debug(
                "rpe_low_confidence",
                market=market_id,
                confidence=round(conf, 3),
                min_required=self._min_confidence,
                model=model_name,
            )
            return None

        divergence = market_price - prob
        abs_div = abs(divergence)
        effective_threshold = self._confidence_threshold * (1.0 + (1.0 - conf))
        if abs_div <= effective_threshold:
            return None

        direction = "buy_no" if divergence > 0 else "buy_yes"
        score = min(1.0, abs_div / 0.20)

        log.info(
            "rpe_signal_fired",
            market=market_id,
            model=model_name,
            model_prob=round(prob, 4),
            market_price=round(market_price, 4),
            divergence=round(divergence, 4),
            confidence=round(conf, 3),
            direction=direction,
            score=round(score, 3),
            shadow=shadow,
            threshold=round(effective_threshold, 4),
        )

        return SignalResult(
            name=signal_name or self.name,
            market_id=market_id,
            score=score,
            metadata={
                "model_probability": prob,
                "confidence": conf,
                "divergence": divergence,
                "direction": direction,
                "model_name": model_name,
                "shadow_mode": shadow,
                "model_metadata": model_metadata or {},
                "effective_threshold": effective_threshold,
                "uncertainty_penalty": 1.0 - conf,
            },
            timestamp=time.time(),
        )

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

        return self.evaluate_probability_dislocation(
            market_id=market.condition_id,
            market_price=market_price,
            implied_probability=estimate.probability,
            confidence=estimate.confidence,
            model_name=estimate.model_name,
            shadow_mode=self._shadow_mode,
            model_metadata=estimate.metadata,
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
    # Dynamic Prior Engine tracking fields
    prior_source: str = ""          # tag category label (e.g. "politics", "default_fallback")
    l2_active: bool = False         # whether L2 order book data contributed
    theta_w_prior: float = 0.0     # time-decay prior weight at signal time


class RPECalibrationTracker:
    """Ring-buffer tracker that scores RPE signal calibration.

    Records every fired RPE signal (shadow or live) and, once a market
    resolves, computes Brier score, log-loss, and direction accuracy
    across resolved entries.

    Supports per-tag Brier score slicing to identify miscalibrated priors
    in the Dynamic Prior Generation Engine.

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
        prior_source: str = "",
        l2_active: bool = False,
        theta_w_prior: float = 0.0,
    ) -> None:
        """Record a fired RPE signal for later calibration."""
        entry = _CalibrationEntry(
            market_id=market_id,
            model_prob=model_prob,
            market_price=market_price,
            direction=direction,
            timestamp=timestamp,
            shadow=shadow,
            prior_source=prior_source,
            l2_active=l2_active,
            theta_w_prior=theta_w_prior,
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
            # Per-tag Brier breakdown (Dynamic Prior Engine diagnostics)
            summary["per_tag_brier"] = self.compute_per_tag_brier()
        return summary

    def compute_per_tag_brier(self) -> dict[str, dict[str, Any]]:
        """Per-tag Brier score breakdown for prior calibration diagnostics.

        Returns a dict keyed by prior_source label with:
            count: number of resolved signals
            brier: mean Brier score for that tag
            direction_accuracy: fraction of correct direction calls
        """
        resolved = self._resolved()
        if not resolved:
            return {}

        from collections import defaultdict
        by_tag: dict[str, list[_CalibrationEntry]] = defaultdict(list)
        for e in resolved:
            tag = e.prior_source or "unknown"
            by_tag[tag].append(e)

        result: dict[str, dict[str, Any]] = {}
        for tag, entries in by_tag.items():
            brier_sum = 0.0
            correct = 0
            for e in entries:
                assert e.resolution_price is not None
                brier_sum += (e.model_prob - e.resolution_price) ** 2
                if e.direction == "buy_yes" and e.resolution_price == 1.0:
                    correct += 1
                elif e.direction == "buy_no" and e.resolution_price == 0.0:
                    correct += 1
            n = len(entries)
            result[tag] = {
                "count": n,
                "brier": round(brier_sum / n, 4),
                "direction_accuracy": round(correct / n, 4),
            }
        return result
