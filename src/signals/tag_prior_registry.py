"""
Tag-Based Empirical Prior Registry — Dynamic Prior Generation Engine.

Maps Polymarket Gamma API tags to calibrated Beta(α, β) priors based on
historical resolution base rates per market category.

Architecture
────────────
    TagPriorRegistry  (singleton, stateless)
        ├── _PRIOR_TABLE   — tag-pattern → (α, β) routing matrix
        ├── get_prior()    — resolve MarketInfo.tags → (α₀, β₀)
        └── prior_source() — human-readable label for calibration tracking

The registry is a pure function: no state, no side effects, trivially
testable and backtestable.  Tag matching is case-insensitive, first-match-wins
against a priority-ordered list.

Calibration
───────────
Initial α/β values are empirical estimates.  To recalibrate:

1. Pull resolved markets from Gamma API:
       GET /markets?closed=true&limit=1000
   Fields: condition_id, tags, outcome (YES/NO), end_date_iso, question.

2. Per-tag base rate:
       p̂_tag = YES_resolutions / total_resolutions

3. Set α = c · p̂, β = c · (1 − p̂) where c is the pseudo-count
   concentration (c = 5 for moderate regularization).

4. Minimum viable: ≥ 200 resolved markets per tag for SE < 0.05.
   Tags below this threshold use the default fallback prior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


# ═══════════════════════════════════════════════════════════════════════════
#  Prior specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TagPrior:
    """A calibrated Beta prior linked to a market-category tag pattern.

    Attributes
    ----------
    pattern:
        Regex pattern matched case-insensitively against the normalised
        comma-separated tag string.  First match in priority order wins.
    alpha:
        Beta distribution α parameter (pseudo-count for YES).
    beta:
        Beta distribution β parameter (pseudo-count for NO).
    label:
        Human-readable category name for logging and calibration slicing.
    """

    pattern: str
    alpha: float
    beta: float
    label: str


# ═══════════════════════════════════════════════════════════════════════════
#  Prior routing matrix — priority-ordered (first match wins)
# ═══════════════════════════════════════════════════════════════════════════
#
# Rationale for each prior is documented inline.  Values are initial
# empirical estimates; recalibrate quarterly using the procedure above.

_PRIOR_TABLE: tuple[TagPrior, ...] = (
    # ── Crypto ──────────────────────────────────────────────────────────
    # Should never reach GenericBayesian (CryptoPriceModel claims first).
    # Included as a safety-net with symmetric prior.
    TagPrior(
        pattern=r"\bcrypto\b",
        alpha=2.0,
        beta=2.0,
        label="crypto_fallback",
    ),
    # ── Politics / Elections ────────────────────────────────────────────
    # Incumbents and front-runners historically win ~60%.  Higher α
    # reflects this YES-bias for "Will X win?" framing.
    TagPrior(
        pattern=r"\b(?:politics|elections?|political|government)\b",
        alpha=3.0,
        beta=2.0,
        label="politics",
    ),
    # ── Legal / Supreme Court ───────────────────────────────────────────
    # Government legal challenges (cert grants, overturns) succeed ~40%.
    # Slight NO-bias reflects institutional inertia.
    TagPrior(
        pattern=r"\b(?:supreme\s*court|legal|court|judicial|scotus)\b",
        alpha=2.0,
        beta=3.0,
        label="legal",
    ),
    # ── Sports ──────────────────────────────────────────────────────────
    # Roughly symmetric across all sport markets when aggregated.
    # Favourite/underdog balance washes out at portfolio level.
    TagPrior(
        pattern=r"\b(?:sports|nfl|nba|mlb|nhl|soccer|football|tennis|mma|ufc|boxing|f1|formula)\b",
        alpha=2.5,
        beta=2.5,
        label="sports",
    ),
    # ── Economy / Financial ─────────────────────────────────────────────
    # "Will X exceed threshold?" framing has slight YES-bias (~55%)
    # due to inflationary / growth-biased question construction.
    TagPrior(
        pattern=r"\b(?:economy|economic|financial|fed|inflation|gdp|jobs|unemployment|interest\s*rate)\b",
        alpha=2.5,
        beta=2.0,
        label="economy",
    ),
    # ── Geopolitics / Conflict ──────────────────────────────────────────
    # Escalation markets ("Will war break out?", "Will sanctions be
    # imposed?") resolve NO ~55%.  Slight NO-bias.
    TagPrior(
        pattern=r"\b(?:geopolitic|conflict|war|military|sanctions|nato|china|russia|iran)\b",
        alpha=2.0,
        beta=2.5,
        label="geopolitics",
    ),
    # ── Pop Culture / Entertainment ─────────────────────────────────────
    # High-entropy category.  Symmetric prior — insufficient historical
    # data for directional calibration.
    TagPrior(
        pattern=r"\b(?:pop\s*culture|entertainment|celebrity|tv|movie|music|award|oscar|grammy|emmy)\b",
        alpha=2.0,
        beta=2.0,
        label="pop_culture",
    ),
    # ── Science / Technology ────────────────────────────────────────────
    # Symmetric.  Milestone questions ("Will X launch by Y?") have
    # mixed base rates.
    TagPrior(
        pattern=r"\b(?:science|tech|technology|ai|space|nasa|climate)\b",
        alpha=2.0,
        beta=2.0,
        label="science_tech",
    ),
    # ── Weather ─────────────────────────────────────────────────────────
    # "Will it be the hottest X on record?" — slight YES-bias due to
    # climate trend, but conservative pending calibration.
    TagPrior(
        pattern=r"\b(?:weather|temperature|hurricane|storm|climate)\b",
        alpha=2.2,
        beta=2.0,
        label="weather",
    ),
)

# Default fallback when no tag matches.  Slight NO-lean (α=2, β=3)
# reflects conservative "extraordinary claims require extraordinary
# evidence" stance.  Polymarket's overall YES resolution rate is
# ~55%, but for unrecognised categories we prefer conservatism.
_DEFAULT_PRIOR = TagPrior(
    pattern="",
    alpha=2.0,
    beta=3.0,
    label="default_fallback",
)

# Pre-compile regexes at module load (once)
_COMPILED_TABLE: tuple[tuple[re.Pattern[str], TagPrior], ...] = tuple(
    (re.compile(tp.pattern, re.IGNORECASE), tp) for tp in _PRIOR_TABLE
)


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

class TagPriorRegistry:
    """Stateless router: MarketInfo.tags → calibrated Beta(α, β) prior.

    Usage
    -----
    >>> reg = TagPriorRegistry()
    >>> alpha, beta, label = reg.get_prior("Politics, Elections")
    (3.0, 2.0, "politics")
    >>> alpha, beta, label = reg.get_prior("")
    (2.0, 3.0, "default_fallback")
    """

    def __init__(
        self,
        *,
        prior_table: Sequence[tuple[re.Pattern[str], TagPrior]] | None = None,
        default_prior: TagPrior | None = None,
    ) -> None:
        """Optionally override the routing table for testing."""
        self._table = prior_table if prior_table is not None else _COMPILED_TABLE
        self._default = default_prior if default_prior is not None else _DEFAULT_PRIOR

    def get_prior(self, tags: str) -> tuple[float, float, str]:
        """Resolve a comma-separated tag string to (α₀, β₀, label).

        Parameters
        ----------
        tags:
            Comma-separated tag string from ``MarketInfo.tags``.
            Case-insensitive.  Empty or None → default fallback.

        Returns
        -------
        (alpha, beta, label):
            Calibrated Beta prior parameters and the matched category
            label for calibration tracking.
        """
        if not tags or not tags.strip():
            return self._default.alpha, self._default.beta, self._default.label

        normalised = tags.lower().strip()

        for compiled_re, prior in self._table:
            if compiled_re.search(normalised):
                return prior.alpha, prior.beta, prior.label

        return self._default.alpha, self._default.beta, self._default.label

    def list_priors(self) -> list[dict]:
        """Return all priors in priority order (for diagnostics/logging)."""
        result = []
        for _, prior in self._table:
            result.append({
                "pattern": prior.pattern,
                "alpha": prior.alpha,
                "beta": prior.beta,
                "label": prior.label,
            })
        result.append({
            "pattern": "(default)",
            "alpha": self._default.alpha,
            "beta": self._default.beta,
            "label": self._default.label,
        })
        return result
