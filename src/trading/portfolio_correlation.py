"""
Portfolio Correlation Engine (PCE) — estimates pairwise market correlations
and gates new entries via parametric Value-at-Risk.

Components
──────────
1. **CorrelationMatrix** — Bayesian blend of structural priors and empirical
   Pearson correlations from rolling 1-minute OHLCV bar returns.
2. **VaRCalculator** — closed-form parametric VaR: ``VaR = z × √(w'Σw)``.
3. **PortfolioCorrelationEngine** — top-level orchestrator that wires
   market aggregators, updates correlations on each refresh cycle, and
   provides the risk-gate and concentration-haircut APIs consumed by
   ``PositionManager``.

Constraints
───────────
- Pure Python (``math``, ``json``, ``time``, ``os``).  No numpy/scipy.
- VaR computation must complete in <10 ms for 20 positions.
- Correlation matrix serialisable to JSON for restart persistence.
- Shadow mode (``PCE_SHADOW_MODE``) logs rejections without blocking.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.core.config import settings
from src.core.logger import get_logger

if TYPE_CHECKING:
    from src.data.ohlcv import OHLCVAggregator

log = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Pure-Python math helpers
# ═══════════════════════════════════════════════════════════════════════════

def _log_returns(closes: list[float]) -> list[float]:
    """Compute 1-period log returns, guarding against zero/negative prices."""
    out: list[float] = []
    for i in range(1, len(closes)):
        prev = max(closes[i - 1], 1e-9)
        curr = max(closes[i], 1e-9)
        out.append(math.log(curr / prev))
    return out


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient (pure Python).

    Returns 0.0 for degenerate inputs (fewer than 2 points, constant series).
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    xs, ys = xs[:n], ys[:n]

    mx = sum(xs) / n
    my = sum(ys) / n

    cov = 0.0
    vx = 0.0
    vy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy

    denom = math.sqrt(vx * vy)
    if denom < 1e-15:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def _mat_vec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    """Multiply N×N matrix by N-vector."""
    return [_dot(row, vec) for row in matrix]


def _std_dev(xs: list[float]) -> float:
    """Population standard deviation (pure Python)."""
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


# ═══════════════════════════════════════════════════════════════════════════
#  Correlation estimate for a single pair
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CorrelationEstimate:
    """Stores both structural prior and empirical estimate for one pair."""
    empirical_corr: float = 0.0
    structural_corr: float = 0.05   # baseline market-wide factor
    overlap_bars: int = 0           # number of concurrent bars used
    last_updated: float = 0.0       # unix timestamp

    @property
    def blended(self) -> float:
        """Bayesian blend: structural prior has weight = prior_weight bars."""
        pw = settings.strategy.pce_structural_prior_weight
        n = self.overlap_bars
        if pw + n <= 0:
            return self.structural_corr
        return (pw * self.structural_corr + n * self.empirical_corr) / (pw + n)

    def blended_with_weight(self, prior_weight: int) -> float:
        """Bayesian blend with explicit prior weight (for backtest)."""
        n = self.overlap_bars
        if prior_weight + n <= 0:
            return self.structural_corr
        return (prior_weight * self.structural_corr + n * self.empirical_corr) / (prior_weight + n)

    def to_dict(self) -> dict:
        return {
            "empirical_corr": round(self.empirical_corr, 6),
            "structural_corr": round(self.structural_corr, 6),
            "overlap_bars": self.overlap_bars,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CorrelationEstimate:
        return cls(
            empirical_corr=d.get("empirical_corr", 0.0),
            structural_corr=d.get("structural_corr", 0.05),
            overlap_bars=d.get("overlap_bars", 0),
            last_updated=d.get("last_updated", 0.0),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════════

def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Canonical ordering for a market pair."""
    return (min(a, b), max(a, b))


class CorrelationMatrix:
    """N×N correlation matrix with Bayesian blending of structural priors
    and empirical Pearson estimates.

    Keys are market IDs (``condition_id``).  The matrix is symmetric;
    internally stored as ``dict[tuple[str, str], CorrelationEstimate]``
    with canonical key ordering.
    """

    def __init__(self) -> None:
        self._pairs: dict[tuple[str, str], CorrelationEstimate] = {}
        self._market_metadata: dict[str, dict] = {}  # market_id → {event_id, tags}
        self.prior_weight_override: int | None = None  # set by PCE to override global prior weight

    # ── Structural priors ──────────────────────────────────────────────────

    def set_structural(
        self,
        market_a: str,
        market_b: str,
        event_id_a: str,
        event_id_b: str,
        tags_a: str,
        tags_b: str,
    ) -> None:
        """Set the structural correlation prior based on metadata overlap."""
        strat = settings.strategy
        if event_id_a and event_id_b and event_id_a == event_id_b:
            structural = strat.pce_structural_same_event_corr
        elif _tags_overlap(tags_a, tags_b):
            structural = strat.pce_structural_same_tag_corr
        else:
            structural = strat.pce_structural_baseline_corr

        key = _pair_key(market_a, market_b)
        if key not in self._pairs:
            self._pairs[key] = CorrelationEstimate(structural_corr=structural)
        else:
            self._pairs[key].structural_corr = structural

    def set_structural_with_values(
        self,
        market_a: str,
        market_b: str,
        event_id_a: str,
        event_id_b: str,
        tags_a: str,
        tags_b: str,
        same_event_corr: float,
        same_tag_corr: float,
        baseline_corr: float,
    ) -> None:
        """Set structural prior with explicit correlation values (for backtest)."""
        if event_id_a and event_id_b and event_id_a == event_id_b:
            structural = same_event_corr
        elif _tags_overlap(tags_a, tags_b):
            structural = same_tag_corr
        else:
            structural = baseline_corr

        key = _pair_key(market_a, market_b)
        if key not in self._pairs:
            self._pairs[key] = CorrelationEstimate(structural_corr=structural)
        else:
            self._pairs[key].structural_corr = structural

    # ── Empirical update ───────────────────────────────────────────────────

    def update_empirical(
        self,
        market_a: str,
        market_b: str,
        bars_a: list[Any],
        bars_b: list[Any],
        min_overlap: int = 2,
        near_extreme_threshold: float | None = None,
        near_extreme_overlap_multiplier: int | None = None,
    ) -> float | None:
        """Compute empirical Pearson correlation from aligned bar close prices.

        Bars are aligned by ``open_time`` (rounded to BAR_INTERVAL = 60s).
        Returns the raw empirical correlation, or ``None`` if insufficient
        overlap (fewer than ``min_overlap`` concurrent bars).
        """
        # Build time → close maps
        map_a: dict[int, float] = {}
        for b in bars_a:
            bucket = int(b.open_time // 60)
            map_a[bucket] = b.close

        map_b: dict[int, float] = {}
        for b in bars_b:
            bucket = int(b.open_time // 60)
            map_b[bucket] = b.close

        # Find overlapping timestamps
        common = sorted(set(map_a.keys()) & set(map_b.keys()))
        if len(common) < min_overlap:
            return None

        closes_a = [map_a[t] for t in common]
        closes_b = [map_b[t] for t in common]

        # Near-extreme price gate: when both series hover near 0 or 1,
        # log returns have poor signal-to-noise ratio.  Require more
        # overlap bars before trusting the empirical correlation.
        strat = settings.strategy
        mean_a = sum(closes_a) / len(closes_a)
        mean_b = sum(closes_b) / len(closes_b)
        near_extreme_thresh = near_extreme_threshold if near_extreme_threshold is not None else strat.pce_near_extreme_threshold
        if (
            (mean_a > near_extreme_thresh or mean_a < (1.0 - near_extreme_thresh))
            and (mean_b > near_extreme_thresh or mean_b < (1.0 - near_extreme_thresh))
        ):
            _ne_mult = near_extreme_overlap_multiplier if near_extreme_overlap_multiplier is not None else strat.pce_near_extreme_overlap_multiplier
            elevated_overlap = min_overlap * _ne_mult
            if len(common) < elevated_overlap:
                log.info(
                    "pce_near_extreme_overlap_insufficient",
                    market_a=market_a,
                    market_b=market_b,
                    mean_a=round(mean_a, 3),
                    mean_b=round(mean_b, 3),
                    overlap=len(common),
                    required=elevated_overlap,
                )
                return None

        rets_a = _log_returns(closes_a)
        rets_b = _log_returns(closes_b)

        if len(rets_a) < 1:
            return None

        corr = pearson_correlation(rets_a, rets_b)

        key = _pair_key(market_a, market_b)
        if key not in self._pairs:
            self._pairs[key] = CorrelationEstimate()
        est = self._pairs[key]
        est.empirical_corr = corr
        est.overlap_bars = len(common)
        est.last_updated = time.time()

        return corr

    # ── Query ──────────────────────────────────────────────────────────────

    def get(self, market_a: str, market_b: str) -> float:
        """Return Bayesian-blended correlation between two markets.

        If ``prior_weight_override`` is set on this matrix, uses
        ``blended_with_weight()`` instead of the global-setting ``blended``.
        """
        if market_a == market_b:
            return 1.0
        key = _pair_key(market_a, market_b)
        est = self._pairs.get(key)
        if est is None:
            if self.prior_weight_override is not None:
                return settings.strategy.pce_structural_baseline_corr
            return settings.strategy.pce_structural_baseline_corr
        if self.prior_weight_override is not None:
            return est.blended_with_weight(self.prior_weight_override)
        return est.blended

    def get_with_weight(self, market_a: str, market_b: str, prior_weight: int) -> float:
        """Return blended correlation with explicit prior weight (for backtest)."""
        if market_a == market_b:
            return 1.0
        key = _pair_key(market_a, market_b)
        est = self._pairs.get(key)
        if est is None:
            return 0.05  # baseline default
        return est.blended_with_weight(prior_weight)

    def get_estimate(self, market_a: str, market_b: str) -> CorrelationEstimate | None:
        """Return raw estimate object (for diagnostics)."""
        if market_a == market_b:
            return None
        return self._pairs.get(_pair_key(market_a, market_b))

    def get_matrix(self, market_ids: list[str]) -> list[list[float]]:
        """Return N×N correlation matrix for the given market set."""
        n = len(market_ids)
        mat: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 1.0
            for j in range(i + 1, n):
                c = self.get(market_ids[i], market_ids[j])
                mat[i][j] = c
                mat[j][i] = c
        return mat

    def all_pairs(self) -> dict[tuple[str, str], CorrelationEstimate]:
        """Return all pair estimates (for dashboard / diagnostics)."""
        return dict(self._pairs)

    # ── Staleness decay ────────────────────────────────────────────────────

    def decay_confidence(self, hours_stale: float, halflife_hours: float | None = None) -> None:
        """Decay overlap_bars by a staleness factor.

        After ``halflife_hours`` of no new data, overlap counts are halved,
        causing structural priors to re-emerge.
        """
        hl = halflife_hours if halflife_hours is not None else settings.strategy.pce_staleness_halflife_hours
        if hl <= 0:
            return
        factor = 0.5 ** (hours_stale / hl)
        for est in self._pairs.values():
            est.overlap_bars = max(0, int(est.overlap_bars * factor))

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_json(self) -> dict:
        """Serialise to a JSON-safe dict."""
        pairs_out: dict[str, dict] = {}
        for (a, b), est in self._pairs.items():
            pairs_out[f"{a}|{b}"] = est.to_dict()
        d: dict[str, Any] = {
            "version": 1,
            "saved_at": time.time(),
            "pairs": pairs_out,
            "market_metadata": self._market_metadata,
        }
        if self.prior_weight_override is not None:
            d["prior_weight_override"] = self.prior_weight_override
        return d

    @classmethod
    def from_json(cls, data: dict) -> CorrelationMatrix:
        """Deserialise from a JSON dict."""
        cm = cls()
        for key_str, est_dict in data.get("pairs", {}).items():
            parts = key_str.split("|", 1)
            if len(parts) != 2:
                continue
            pair_key = (parts[0], parts[1])
            cm._pairs[pair_key] = CorrelationEstimate.from_dict(est_dict)
        cm._market_metadata = data.get("market_metadata", {})
        if "prior_weight_override" in data:
            cm.prior_weight_override = int(data["prior_weight_override"])
        return cm

    def save(self, path: str | Path) -> None:
        """Save correlation matrix to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        # Atomic rename (best-effort on Windows)
        tmp.replace(path)
        log.info("pce_matrix_saved", path=str(path), pairs=len(self._pairs))

    @classmethod
    def load(cls, path: str | Path) -> CorrelationMatrix:
        """Load from JSON file.  Returns empty matrix if file missing/corrupt."""
        path = Path(path)
        if not path.exists():
            log.info("pce_matrix_not_found", path=str(path))
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            cm = cls.from_json(data)
            log.info("pce_matrix_loaded", path=str(path), pairs=len(cm._pairs))
            return cm
        except Exception as exc:
            log.warning("pce_matrix_load_error", error=str(exc))
            return cls()


def _tags_overlap(tags_a: str, tags_b: str) -> bool:
    """Check if two comma-separated tag strings share any tag."""
    if not tags_a or not tags_b:
        return False
    set_a = {t.strip().lower() for t in tags_a.split(",") if t.strip()}
    set_b = {t.strip().lower() for t in tags_b.split(",") if t.strip()}
    return bool(set_a & set_b)


# ═══════════════════════════════════════════════════════════════════════════
#  VaR Calculator
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VaRResult:
    """Output of the portfolio VaR computation."""

    portfolio_var_usd: float = 0.0
    marginal_var_usd: float = 0.0       # contribution of the proposed position
    diversification_benefit: float = 0.0  # gross VaR - net VaR
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    position_decomposition: list[dict] = field(default_factory=list)
    exceeds_threshold: bool = False

    def to_dict(self) -> dict:
        return {
            "portfolio_var_usd": round(self.portfolio_var_usd, 4),
            "marginal_var_usd": round(self.marginal_var_usd, 4),
            "diversification_benefit": round(self.diversification_benefit, 4),
            "gross_exposure": round(self.gross_exposure, 4),
            "net_exposure": round(self.net_exposure, 4),
            "exceeds_threshold": self.exceeds_threshold,
            "positions": self.position_decomposition,
        }


@dataclass
class VaRSizingResult:
    """Output of the VaR-aware sizing cap computation.

    Returned by ``compute_var_sizing_cap()`` to provide both the
    cap and diagnostics for logging / dashboard.
    """
    cap_usd: float = 0.0
    current_var: float = 0.0
    headroom: float = 0.0
    marginal_per_dollar: float = 0.0
    bisect_iterations: int = 0

    def to_dict(self) -> dict:
        return {
            "cap_usd": round(self.cap_usd, 4),
            "current_var": round(self.current_var, 4),
            "headroom": round(self.headroom, 4),
            "marginal_per_dollar": round(self.marginal_per_dollar, 6),
            "bisect_iterations": self.bisect_iterations,
        }


class VaRCalculator:
    """Parametric VaR calculator using the closed-form formula.

    ``VaR = z × √(w' Σ w)`` where:
      - ``w`` = position-size vector (USD exposure per market)
      - ``Σ`` = covariance matrix (``σ_i × σ_j × ρ_ij``)
      - ``z`` = confidence z-score (1.645 for 95th percentile)

    Performance: O(N²) for N positions. For N=20, the inner loop is
    400 multiplications — sub-millisecond in pure Python.
    """

    def __init__(self, z_score: float | None = None) -> None:
        self.z = z_score if z_score is not None else settings.strategy.pce_var_confidence_z
        # Cached covariance matrix
        self._cov_cache: list[list[float]] | None = None
        self._cov_market_ids: list[str] | None = None
        self._cov_dirty: bool = True

    def invalidate_cache(self) -> None:
        """Mark the covariance cache as stale (call when correlations change)."""
        self._cov_dirty = True

    def _build_covariance(
        self,
        market_ids: list[str],
        corr_matrix: CorrelationMatrix,
        volatilities: dict[str, float],
    ) -> list[list[float]]:
        """Build the covariance matrix Σ where Σ_ij = σ_i × σ_j × ρ_ij."""
        n = len(market_ids)
        cov: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            si = volatilities.get(market_ids[i], 0.0)
            cov[i][i] = si * si
            for j in range(i + 1, n):
                sj = volatilities.get(market_ids[j], 0.0)
                rho = corr_matrix.get(market_ids[i], market_ids[j])
                val = si * sj * rho
                cov[i][j] = val
                cov[j][i] = val
        return cov

    def compute_portfolio_var(
        self,
        positions: list[dict],
        proposed: dict | None,
        corr_matrix: CorrelationMatrix,
        volatilities: dict[str, float],
        threshold: float | None = None,
    ) -> VaRResult:
        """Compute parametric VaR for the current portfolio + proposed position.

        Parameters
        ----------
        positions:
            List of ``{"market_id": str, "exposure_usd": float}`` for
            each existing open position.
        proposed:
            ``{"market_id": str, "exposure_usd": float}`` for the new
            position being considered, or ``None`` for portfolio-only VaR.
        corr_matrix:
            The current ``CorrelationMatrix``.
        volatilities:
            ``{market_id: rolling_sigma}`` for each market.
        threshold:
            Max portfolio VaR in USD.  The result's ``exceeds_threshold``
            flag will be set if VaR exceeds this.

        Returns
        -------
        VaRResult with full decomposition.
        """
        if threshold is None:
            threshold = settings.strategy.pce_max_portfolio_var_usd

        # Aggregate exposures by market (one entry per unique market)
        exposure_map: dict[str, float] = {}
        for p in positions:
            mid = p["market_id"]
            exposure_map[mid] = exposure_map.get(mid, 0.0) + p["exposure_usd"]

        # Compute VaR *without* the proposed position first
        market_ids_existing = list(exposure_map.keys())
        weights_existing = [exposure_map[m] for m in market_ids_existing]
        var_existing = self._compute_var_for_weights(
            market_ids_existing, weights_existing, corr_matrix, volatilities
        )

        # Add proposed position
        if proposed is not None:
            mid = proposed["market_id"]
            exposure_map_with = dict(exposure_map)
            exposure_map_with[mid] = exposure_map_with.get(mid, 0.0) + proposed["exposure_usd"]
        else:
            exposure_map_with = dict(exposure_map)

        market_ids_all = list(exposure_map_with.keys())
        weights_all = [exposure_map_with[m] for m in market_ids_all]

        # Pre-build the covariance matrix once for reuse
        cov_all = self._build_covariance(market_ids_all, corr_matrix, volatilities)

        var_total = self._compute_var_for_weights(
            market_ids_all, weights_all, corr_matrix, volatilities,
            _prebuilt_cov=cov_all,
        )

        # Compute gross VaR (assuming all correlations = 1)
        gross_var = self.z * sum(
            abs(w) * volatilities.get(m, 0.0) for m, w in zip(market_ids_all, weights_all)
        )

        # Marginal VaR = VaR(with proposed) - VaR(without)
        marginal = var_total - var_existing

        # Diversification benefit = gross - net
        div_benefit = gross_var - var_total

        # Per-position marginal VaR decomposition
        decomp: list[dict] = []
        for mid, w in zip(market_ids_all, weights_all):
            sigma = volatilities.get(mid, 0.0)
            individual_var = self.z * abs(w) * sigma
            decomp.append({
                "market_id": mid,
                "exposure_usd": round(w, 4),
                "volatility": round(sigma, 6),
                "individual_var": round(individual_var, 4),
            })

        gross_exp = sum(abs(w) for w in weights_all)
        net_exp = abs(sum(weights_all))

        result = VaRResult(
            portfolio_var_usd=round(var_total, 4),
            marginal_var_usd=round(marginal, 4),
            diversification_benefit=round(div_benefit, 4),
            gross_exposure=round(gross_exp, 4),
            net_exposure=round(net_exp, 4),
            position_decomposition=decomp,
            exceeds_threshold=var_total > threshold,
        )
        return result

    def _compute_var_for_weights(
        self,
        market_ids: list[str],
        weights: list[float],
        corr_matrix: CorrelationMatrix,
        volatilities: dict[str, float],
        *,
        _prebuilt_cov: list[list[float]] | None = None,
    ) -> float:
        """VaR = z × √(w' Σ w)."""
        n = len(market_ids)
        if n == 0:
            return 0.0

        # Reuse pre-built covariance matrix when available (avoids O(n²) rebuild)
        cov = _prebuilt_cov if _prebuilt_cov is not None else self._build_covariance(market_ids, corr_matrix, volatilities)

        # w' Σ w = Σ_i Σ_j w_i * cov_ij * w_j
        portfolio_variance = 0.0
        for i in range(n):
            for j in range(n):
                portfolio_variance += weights[i] * cov[i][j] * weights[j]

        # Guard against numerical noise producing negative variance
        portfolio_variance = max(0.0, portfolio_variance)

        return self.z * math.sqrt(portfolio_variance)


# ═══════════════════════════════════════════════════════════════════════════
#  Portfolio Correlation Engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _MarketRegistration:
    """Internal record of a registered market."""
    market_id: str
    event_id: str
    tags: str
    aggregator: Any  # OHLCVAggregator — typed as Any to avoid import cycle


class PortfolioCorrelationEngine:
    """Top-level orchestrator for portfolio correlation analysis.

    Integrates with the live bot via:
      - ``register_market()`` / ``unregister_market()`` (from ``_wire_market``)
      - ``refresh_correlations()`` (from ``_market_refresh_loop``)
      - ``check_var_gate()`` (from ``_check_risk_gates``)
      - ``compute_concentration_haircut()`` (from ``open_position``)
      - ``get_dashboard_data()`` (for Telegram dashboard)
    """

    def __init__(
        self,
        data_dir: str = "data",
        *,
        shadow_mode: bool | None = None,
        max_portfolio_var_usd: float | None = None,
        haircut_threshold: float | None = None,
        structural_same_event: float | None = None,
        structural_same_tag: float | None = None,
        structural_baseline: float | None = None,
        structural_prior_weight: int | None = None,
        min_overlap_bars: int | None = None,
        var_z: float | None = None,
        staleness_halflife_hours: float | None = None,
        holding_period_minutes: int | None = None,
        var_soft_cap: bool | None = None,
        var_bisect_iterations: int | None = None,
        near_extreme_threshold: float | None = None,
        near_extreme_overlap_multiplier: int | None = None,
    ):
        strat = settings.strategy
        self.data_dir = data_dir
        self.shadow_mode = shadow_mode if shadow_mode is not None else strat.pce_shadow_mode
        self.max_portfolio_var_usd = max_portfolio_var_usd if max_portfolio_var_usd is not None else strat.pce_max_portfolio_var_usd
        self.haircut_threshold = haircut_threshold if haircut_threshold is not None else strat.pce_correlation_haircut_threshold
        self.structural_same_event = structural_same_event if structural_same_event is not None else strat.pce_structural_same_event_corr
        self.structural_same_tag = structural_same_tag if structural_same_tag is not None else strat.pce_structural_same_tag_corr
        self.structural_baseline = structural_baseline if structural_baseline is not None else strat.pce_structural_baseline_corr
        self.structural_prior_weight = structural_prior_weight if structural_prior_weight is not None else strat.pce_structural_prior_weight
        self.min_overlap_bars = min_overlap_bars if min_overlap_bars is not None else strat.pce_min_overlap_bars
        self.staleness_halflife_hours = staleness_halflife_hours if staleness_halflife_hours is not None else strat.pce_staleness_halflife_hours

        self.holding_period_minutes: int = holding_period_minutes if holding_period_minutes is not None else strat.pce_holding_period_minutes
        self.var_soft_cap: bool = var_soft_cap if var_soft_cap is not None else strat.pce_var_soft_cap
        self.var_bisect_iterations: int = var_bisect_iterations if var_bisect_iterations is not None else strat.pce_var_bisect_iterations

        self.near_extreme_threshold: float | None = near_extreme_threshold  # None → use global
        self.near_extreme_overlap_multiplier: int | None = near_extreme_overlap_multiplier  # None → use global

        self.corr_matrix = CorrelationMatrix()
        self.corr_matrix.prior_weight_override = self.structural_prior_weight
        self.var_calc = VaRCalculator(z_score=var_z)
        self._markets: dict[str, _MarketRegistration] = {}
        self._persistence_path = Path(data_dir) / "pce_correlation.json"

        vol_scale = math.sqrt(max(1, self.holding_period_minutes))
        log.info(
            "pce_engine_init",
            shadow_mode=self.shadow_mode,
            holding_period_minutes=self.holding_period_minutes,
            vol_scaling_factor=round(vol_scale, 4),
            var_soft_cap=self.var_soft_cap,
        )

    # ── Market registration ────────────────────────────────────────────────

    def register_market(
        self,
        market_id: str,
        event_id: str,
        tags: str,
        aggregator: Any,
    ) -> None:
        """Register a market for correlation tracking.

        Called from ``TradingBot._wire_market()``.
        """
        self._markets[market_id] = _MarketRegistration(
            market_id=market_id,
            event_id=event_id,
            tags=tags,
            aggregator=aggregator,
        )
        self.corr_matrix._market_metadata[market_id] = {
            "event_id": event_id,
            "tags": tags,
        }
        # Set structural priors with all existing markets
        for other_id, other_reg in self._markets.items():
            if other_id == market_id:
                continue
            self.corr_matrix.set_structural_with_values(
                market_id, other_id,
                event_id, other_reg.event_id,
                tags, other_reg.tags,
                same_event_corr=self.structural_same_event,
                same_tag_corr=self.structural_same_tag,
                baseline_corr=self.structural_baseline,
            )
        log.info("pce_market_registered", market=market_id, event_id=event_id)

    def unregister_market(self, market_id: str) -> None:
        """Remove a market from correlation tracking."""
        self._markets.pop(market_id, None)
        # Don't remove pair data — it may be useful for historical context
        log.info("pce_market_unregistered", market=market_id)

    # ── Correlation refresh ────────────────────────────────────────────────

    def refresh_correlations(self) -> None:
        """Recompute empirical correlations from current aggregator bar data.

        Called every market refresh cycle (~30 minutes).
        """
        market_ids = list(self._markets.keys())
        updated = 0

        for i in range(len(market_ids)):
            for j in range(i + 1, len(market_ids)):
                mid_a = market_ids[i]
                mid_b = market_ids[j]
                reg_a = self._markets[mid_a]
                reg_b = self._markets[mid_b]

                bars_a = list(reg_a.aggregator.bars)
                bars_b = list(reg_b.aggregator.bars)

                if len(bars_a) < 2 or len(bars_b) < 2:
                    continue

                corr = self.corr_matrix.update_empirical(
                    mid_a, mid_b, bars_a, bars_b,
                    min_overlap=self.min_overlap_bars,
                    near_extreme_threshold=self.near_extreme_threshold,
                    near_extreme_overlap_multiplier=self.near_extreme_overlap_multiplier,
                )
                if corr is not None:
                    updated += 1

        # Invalidate VaR cache after correlation update
        self.var_calc.invalidate_cache()

        log.info(
            "pce_correlation_refresh",
            markets=len(market_ids),
            pairs_updated=updated,
        )

    # ── VaR gate ───────────────────────────────────────────────────────────

    def check_var_gate(
        self,
        open_positions: list[Any],
        proposed_market_id: str,
        proposed_size_usd: float,
        proposed_direction: str = "NO",
    ) -> tuple[bool, VaRResult]:
        """Check if adding a new position keeps portfolio VaR within limits.

        Parameters
        ----------
        open_positions:
            List of ``Position`` objects (from ``get_open_positions()``).
        proposed_market_id:
            The ``condition_id`` / ``market_id`` of the proposed trade.
        proposed_size_usd:
            Dollar exposure of the proposed position (``entry_price × size``).
        proposed_direction:
            ``"YES"`` or ``"NO"`` (for logging).

        Returns
        -------
        ``(allowed, var_result)`` — ``allowed`` is always True in shadow mode.
        """
        # Build existing portfolio exposure vector with directional signs
        existing = self._build_exposure_list(open_positions)

        # Apply direction sign to proposed position
        dir_sign = 1 if proposed_direction == "YES" else -1
        proposed = {
            "market_id": proposed_market_id,
            "exposure_usd": dir_sign * proposed_size_usd,
        }

        # Get volatilities from registered aggregators
        volatilities = self._get_volatilities()

        result = self.var_calc.compute_portfolio_var(
            positions=existing,
            proposed=proposed,
            corr_matrix=self.corr_matrix,
            volatilities=volatilities,
            threshold=self.max_portfolio_var_usd,
        )

        # Log full decomposition
        log.info(
            "pce_var_gate",
            portfolio_var=result.portfolio_var_usd,
            marginal_var=result.marginal_var_usd,
            diversification_benefit=result.diversification_benefit,
            gross_exposure=result.gross_exposure,
            net_exposure=result.net_exposure,
            exceeds_threshold=result.exceeds_threshold,
            proposed_market=proposed_market_id,
            proposed_size=proposed_size_usd,
            proposed_direction=proposed_direction,
            threshold=self.max_portfolio_var_usd,
            shadow_mode=self.shadow_mode,
        )

        if result.exceeds_threshold:
            action = "shadow_log" if self.shadow_mode else "rejected"
            log.warning(
                "pce_var_gate_exceeded",
                action=action,
                portfolio_var=result.portfolio_var_usd,
                threshold=self.max_portfolio_var_usd,
                proposed_market=proposed_market_id,
            )
            if self.shadow_mode:
                return True, result
            return False, result

        return True, result

    # ── Concentration haircut ──────────────────────────────────────────────

    def compute_concentration_haircut(
        self,
        proposed_market_id: str,
        open_positions: list[Any],
    ) -> float:
        """Compute the sizing haircut for a correlated addition.

        Returns a multiplier in (0, 1]. Uses **exposure-weighted** average
        correlation: a $50 peer with ρ=0.8 contributes more than a $5 peer
        with ρ=0.1.  Falls back to unweighted average if total peer
        exposure is zero (defensive).
        """
        if not open_positions:
            return 1.0

        # Collect peer positions (excluding proposed market)
        peer_exposures: list[tuple[str, float]] = []  # (market_id, exposure_usd)
        for pos in open_positions:
            if pos.market_id != proposed_market_id:
                exposure = pos.entry_price * pos.entry_size
                peer_exposures.append((pos.market_id, exposure))

        if not peer_exposures:
            return 1.0

        # Compute exposure-weighted average correlation
        total_weight = sum(exp for _, exp in peer_exposures)
        if total_weight > 0:
            weighted_corr = sum(
                exp * self.corr_matrix.get(proposed_market_id, mid)
                for mid, exp in peer_exposures
            ) / total_weight
        else:
            # Fallback: unweighted average
            unique_markets = {mid for mid, _ in peer_exposures}
            weighted_corr = sum(
                self.corr_matrix.get(proposed_market_id, mid)
                for mid in unique_markets
            ) / len(unique_markets)

        if weighted_corr > self.haircut_threshold:
            haircut = max(0.05, 1.0 - weighted_corr)  # floor at 5% to avoid zero sizing
            log.info(
                "pce_concentration_haircut",
                proposed_market=proposed_market_id,
                weighted_avg_correlation=round(weighted_corr, 4),
                haircut=round(haircut, 4),
                threshold=self.haircut_threshold,
            )
            return haircut

        return 1.0

    # ── Marginal-VaR sizing cap ────────────────────────────────────────────

    def compute_var_sizing_cap(
        self,
        open_positions: list[Any],
        proposed_market_id: str,
        proposed_size_usd: float,
        proposed_direction: str = "NO",
    ) -> VaRSizingResult:
        """Compute max allowable position size given VaR headroom.

        Uses **iterative bisection** to find the exact position size
        where VaR(portfolio + size) = threshold.  This correctly handles
        the convexity of marginal VaR for correlated portfolios.

        Returns
        -------
        VaRSizingResult with ``cap_usd``, ``current_var``, ``headroom``,
        ``marginal_per_dollar``, and ``bisect_iterations``.
        """
        existing = self._build_exposure_list(open_positions)
        volatilities = self._get_volatilities()
        dir_sign = 1 if proposed_direction == "YES" else -1

        # Current VaR without proposed position
        current = self.var_calc.compute_portfolio_var(
            positions=existing,
            proposed=None,
            corr_matrix=self.corr_matrix,
            volatilities=volatilities,
        )

        current_var = current.portfolio_var_usd
        headroom = max(0.0, self.max_portfolio_var_usd - current_var)

        if headroom <= 0:
            log.info(
                "pce_var_sizing_cap",
                headroom=0.0,
                cap=0.0,
                proposed=round(proposed_size_usd, 2),
            )
            return VaRSizingResult(
                cap_usd=0.0,
                current_var=current_var,
                headroom=0.0,
            )

        # Check if full size is within threshold (fast path)
        full_proposed = {
            "market_id": proposed_market_id,
            "exposure_usd": dir_sign * proposed_size_usd,
        }
        full_result = self.var_calc.compute_portfolio_var(
            positions=existing,
            proposed=full_proposed,
            corr_matrix=self.corr_matrix,
            volatilities=volatilities,
            threshold=self.max_portfolio_var_usd,
        )
        marginal_full = full_result.marginal_var_usd

        if not full_result.exceeds_threshold:
            mpd = marginal_full / proposed_size_usd if proposed_size_usd > 0 else 0.0
            log.info(
                "pce_var_sizing_cap",
                headroom=round(headroom, 4),
                marginal_per_dollar=round(mpd, 6),
                cap=round(proposed_size_usd, 2),
                proposed=round(proposed_size_usd, 2),
                bisect_iterations=0,
            )
            return VaRSizingResult(
                cap_usd=proposed_size_usd,
                current_var=current_var,
                headroom=headroom,
                marginal_per_dollar=mpd,
                bisect_iterations=0,
            )

        # Check if even $1 exceeds threshold
        min_proposed = {
            "market_id": proposed_market_id,
            "exposure_usd": dir_sign * 1.0,
        }
        min_result = self.var_calc.compute_portfolio_var(
            positions=existing,
            proposed=min_proposed,
            corr_matrix=self.corr_matrix,
            volatilities=volatilities,
            threshold=self.max_portfolio_var_usd,
        )
        if min_result.exceeds_threshold:
            log.info(
                "pce_var_sizing_cap",
                headroom=round(headroom, 4),
                cap=0.0,
                proposed=round(proposed_size_usd, 2),
                bisect_iterations=0,
                reason="even_min_exceeds",
            )
            return VaRSizingResult(
                cap_usd=0.0,
                current_var=current_var,
                headroom=headroom,
                bisect_iterations=0,
            )

        # Binary search in [1.0, proposed_size_usd]
        lo = 1.0
        hi = proposed_size_usd
        max_iter = self.var_bisect_iterations
        iterations = 0

        for iterations in range(1, max_iter + 1):
            if hi - lo < 0.50:
                break
            mid_size = (lo + hi) / 2.0
            mid_proposed = {
                "market_id": proposed_market_id,
                "exposure_usd": dir_sign * mid_size,
            }
            mid_result = self.var_calc.compute_portfolio_var(
                positions=existing,
                proposed=mid_proposed,
                corr_matrix=self.corr_matrix,
                volatilities=volatilities,
                threshold=self.max_portfolio_var_usd,
            )
            if mid_result.exceeds_threshold:
                hi = mid_size
            else:
                lo = mid_size

        # Return lo (conservative — guaranteed within threshold)
        cap = lo
        mpd = marginal_full / proposed_size_usd if proposed_size_usd > 0 else 0.0

        log.info(
            "pce_var_sizing_cap",
            headroom=round(headroom, 4),
            marginal_per_dollar=round(mpd, 6),
            cap=round(cap, 2),
            proposed=round(proposed_size_usd, 2),
            bisect_iterations=iterations,
        )
        return VaRSizingResult(
            cap_usd=cap,
            current_var=current_var,
            headroom=headroom,
            marginal_per_dollar=mpd,
            bisect_iterations=iterations,
        )

    # ── Dashboard data ─────────────────────────────────────────────────────

    def get_dashboard_data(self, open_positions: list[Any] | None = None) -> dict:
        """Compile dashboard summary for Telegram notification.

        Returns dict with keys: portfolio_var, top_correlated_pairs,
        open_positions_count, gross_exposure, net_exposure.
        """
        # Compute current portfolio VaR (without proposed position)
        existing: list[dict] = []
        if open_positions:
            existing = self._build_exposure_list(open_positions)

        volatilities = self._get_volatilities()

        var_result = self.var_calc.compute_portfolio_var(
            positions=existing,
            proposed=None,
            corr_matrix=self.corr_matrix,
            volatilities=volatilities,
        )

        # Find top-3 most correlated pairs among registered markets
        top_pairs: list[tuple[str, str, float]] = []
        pairs = self.corr_matrix.all_pairs()
        for (a, b), est in pairs.items():
            top_pairs.append((a, b, est.blended))
        top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_3 = top_pairs[:3]

        return {
            "portfolio_var": var_result.portfolio_var_usd,
            "threshold": self.max_portfolio_var_usd,
            "top_correlated_pairs": [
                {"market_a": a, "market_b": b, "correlation": round(c, 4)}
                for a, b, c in top_3
            ],
            "open_positions": len(existing),
            "gross_exposure": var_result.gross_exposure,
            "net_exposure": var_result.net_exposure,
            "total_pairs_tracked": len(pairs),
            "shadow_mode": self.shadow_mode,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist the correlation matrix to disk."""
        self.corr_matrix.save(self._persistence_path)

    def load_state(self) -> None:
        """Load correlation matrix from disk with staleness decay.

        If the loaded matrix is empty (no prior state), attempts to
        bootstrap from historical tick data.
        """
        self.corr_matrix = CorrelationMatrix.load(self._persistence_path)

        # Re-apply instance-level prior weight override after loading
        self.corr_matrix.prior_weight_override = self.structural_prior_weight

        # Apply staleness decay based on time since last save
        if self.corr_matrix._pairs:
            # Find the most recent update timestamp
            latest = max(
                (est.last_updated for est in self.corr_matrix._pairs.values()),
                default=0.0,
            )
            if latest > 0:
                hours_stale = (time.time() - latest) / 3600.0
                if hours_stale > 0:
                    self.corr_matrix.decay_confidence(
                        hours_stale, self.staleness_halflife_hours
                    )
                    log.info(
                        "pce_staleness_decay_applied",
                        hours_stale=round(hours_stale, 1),
                        halflife=self.staleness_halflife_hours,
                    )
        else:
            # Empty matrix — attempt bootstrap from historical data
            pairs_seeded = self.bootstrap_from_ticks()
            log.info(
                "pce_bootstrap_on_load",
                attempted=True,
                pairs_seeded=pairs_seeded,
            )

    def bootstrap_from_ticks(self, data_dir: str | None = None) -> int:
        """Bootstrap correlation matrix from historical raw tick data.

        Reads ``raw_ticks/YYYY-MM-DD/<asset_id>.jsonl`` files produced by
        ``MarketDataRecorder``, reconstructs 1-minute bars, and computes
        initial pairwise correlations.

        Returns the number of pairs bootstrapped.
        """
        base = Path(data_dir or self.data_dir)
        raw_dir = base / "raw_ticks"
        if not raw_dir.exists():
            log.info("pce_bootstrap_no_data", path=str(raw_dir))
            return 0

        # Collect all asset bar histories from tick files
        asset_bars: dict[str, list[dict]] = {}  # asset_id → list of {open_time, close}

        dates = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())
        for date_dir_name in dates:
            date_dir = raw_dir / date_dir_name
            for tick_file in date_dir.glob("*.jsonl"):
                asset_id = tick_file.stem
                bars = self._reconstruct_bars_from_ticks(tick_file)
                if bars:
                    if asset_id not in asset_bars:
                        asset_bars[asset_id] = []
                    asset_bars[asset_id].extend(bars)

        # Build pairwise correlations
        asset_ids = list(asset_bars.keys())
        pairs_done = 0
        for i in range(len(asset_ids)):
            for j in range(i + 1, len(asset_ids)):
                bars_a = asset_bars[asset_ids[i]]
                bars_b = asset_bars[asset_ids[j]]
                corr = self.corr_matrix.update_empirical(
                    asset_ids[i], asset_ids[j],
                    [_BarProxy(b["open_time"], b["close"]) for b in bars_a],
                    [_BarProxy(b["open_time"], b["close"]) for b in bars_b],
                    near_extreme_threshold=self.near_extreme_threshold,
                    near_extreme_overlap_multiplier=self.near_extreme_overlap_multiplier,
                )
                if corr is not None:
                    pairs_done += 1

        log.info(
            "pce_bootstrap_complete",
            assets=len(asset_ids),
            pairs=pairs_done,
        )
        return pairs_done

    def _reconstruct_bars_from_ticks(self, tick_file: Path) -> list[dict]:
        """Reconstruct 1-minute bars from a JSONL tick file."""
        # Group ticks into 60-second buckets
        buckets: dict[int, list[tuple[float, float]]] = {}  # bucket → [(ts, price)]

        try:
            with open(tick_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if record.get("source") != "trade":
                        continue

                    ts = record.get("local_ts", 0.0)
                    payload = record.get("payload", {})
                    price = float(payload.get("price", 0))
                    if price <= 0 or ts <= 0:
                        continue

                    bucket = int(ts // 60)
                    if bucket not in buckets:
                        buckets[bucket] = []
                    buckets[bucket].append((ts, price))
        except Exception:
            return []

        # Convert buckets to bars
        bars: list[dict] = []
        for bucket in sorted(buckets.keys()):
            ticks = buckets[bucket]
            if not ticks:
                continue
            prices = [p for _, p in ticks]
            bars.append({
                "open_time": bucket * 60.0,
                "close": prices[-1],
            })

        return bars

    # ── Prior validation (OE-7) ──────────────────────────────────────────

    def validate_structural_priors(self) -> dict:
        """Log structural-vs-empirical divergence for every tracked pair.

        Useful as a periodic health-check (e.g. every 30 min alongside
        ``refresh_correlations``).  Returns a summary dict for telemetry.
        """
        pairs = self.corr_matrix.all_pairs()
        if not pairs:
            log.info("pce_validate_priors_empty")
            return {"pairs": 0}

        same_event_devs: list[float] = []
        same_tag_devs: list[float] = []
        baseline_devs: list[float] = []

        for (a, b), est in pairs.items():
            if est.overlap_bars < 10:
                continue  # not enough empirical data to judge

            dev = est.empirical_corr - est.structural_corr
            structural = est.structural_corr

            if abs(structural - self.structural_same_event) < 0.01:
                same_event_devs.append(dev)
            elif abs(structural - self.structural_same_tag) < 0.01:
                same_tag_devs.append(dev)
            else:
                baseline_devs.append(dev)

        def _stats(devs: list[float]) -> dict:
            if not devs:
                return {"n": 0}
            mean_d = sum(devs) / len(devs)
            mad = sum(abs(d) for d in devs) / len(devs)
            return {"n": len(devs), "mean_dev": round(mean_d, 4),
                    "mad": round(mad, 4)}

        summary = {
            "pairs": len(pairs),
            "same_event": _stats(same_event_devs),
            "same_tag": _stats(same_tag_devs),
            "baseline": _stats(baseline_devs),
        }

        # Flag large mean deviations
        for tier in ("same_event", "same_tag", "baseline"):
            s = summary[tier]
            if s["n"] > 0 and abs(s.get("mean_dev", 0)) > 0.15:
                log.warning(
                    "pce_prior_misalignment",
                    tier=tier,
                    mean_deviation=s["mean_dev"],
                    n_pairs=s["n"],
                )

        log.info("pce_validate_priors", **summary)
        return summary

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_exposure_list(self, open_positions: list[Any]) -> list[dict]:
        """Build directionally-signed exposure list from open positions.

        YES positions get positive exposure (profit when event resolves YES).
        NO positions get negative exposure (profit when event resolves NO).
        Legacy positions without ``trade_side`` default to NO.
        """
        existing: list[dict] = []
        for pos in open_positions:
            trade_side = getattr(pos, "trade_side", "NO")
            direction_sign = 1 if trade_side == "YES" else -1
            exposure = direction_sign * pos.entry_price * pos.entry_size
            existing.append({
                "market_id": pos.market_id,
                "exposure_usd": exposure,
            })
        return existing

    def _get_volatilities(self) -> dict[str, float]:
        """Extract rolling volatilities from registered aggregators.

        Scales 1-minute bar volatility to the configured holding period:
        ``σ_hold = σ_1m × √(holding_period_minutes)``.
        """
        scale = math.sqrt(max(1, self.holding_period_minutes))
        vols: dict[str, float] = {}
        for mid, reg in self._markets.items():
            vol = getattr(reg.aggregator, "rolling_volatility", 0.0)
            raw = vol if vol > 0 else 0.01  # floor at 1% to avoid zero
            vols[mid] = raw * scale
        return vols


@dataclass
class _BarProxy:
    """Minimal bar-like object for bootstrap correlation computation."""
    open_time: float
    close: float
