"""
Adverse Selection Monitor for Maker Fills.

Tracks mark-to-market P&L at T+5, T+15, and T+60 seconds after each
maker-mode fill.  Uses a one-sided Welch t-test on the T+15 PnL window
to detect markets where our limit orders are consistently "picked off" by
informed flow.  Auto-suspends maker routing per-market with exponential
backoff when statistical significance is reached.

Integration
-----------
1. Call ``record_maker_fill()`` in the ``on_fill`` callback for any order
   executed with ``execution_mode="maker"``.
2. Call ``tick()`` on every BBO update (or at least every ~1s) to schedule
   the T+5/15/60 marks.
3. Check ``is_maker_allowed(market_id)`` before placing a POST_ONLY order;
   route to taker if False.

Statistical Design
------------------
* Window: rolling 30 maker fills per market.
* Test: one-sided Welch t-test, H₀: μ(PnL_t15) ≥ 0, H₁: μ(PnL_t15) < 0.
* Trigger: p-value < α (default 0.05) with n ≥ 10 fills.
* Backoff: 15 min → 30 min → 60 min → 120 min → 240 min (max).
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Awaitable

from src.core.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MakerFillRecord:
    """Single maker-mode fill, annotated with post-fill mid prices."""

    market_id: str
    asset_id: str
    fill_price: float             # probability scale (0–1)
    fill_side: str                # "BUY" | "SELL"
    fill_time: float              # unix timestamp
    size_usd: float

    # Populated by tick() as time elapses
    mark_t5:  float | None = None
    mark_t15: float | None = None
    mark_t60: float | None = None

    pnl_t5:   float | None = None   # cents × size_usd
    pnl_t15:  float | None = None
    pnl_t60:  float | None = None

    # True when pnl_t15 < -ADVERSE_CENTS_FLOOR
    is_adverse: bool = False


@dataclass
class MarketAdverseStats:
    """Rolling adverse-selection statistics per market."""

    market_id: str
    n_fills:   int = 0
    n_adverse: int = 0

    avg_pnl_t5:  float = 0.0
    avg_pnl_t15: float = 0.0
    avg_pnl_t60: float = 0.0

    last_t_stat: float = 0.0
    last_p_value: float = 1.0

    # Suspension state
    suspension_until:  float = 0.0   # unix timestamp; 0 = not suspended
    suspension_count:  int = 0       # increments each suspension (backoff)
    last_suspended_at: float = 0.0

    @property
    def adverse_rate(self) -> float:
        return self.n_adverse / self.n_fills if self.n_fills else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  t-distribution CDF (pure Python — no scipy dependency)
#  Abramowitz & Stegun §26.7, regularised incomplete beta function.
# ═══════════════════════════════════════════════════════════════════════════

def _t_cdf_lower_tail(t: float, df: int) -> float:
    """One-sided lower-tail CDF P(T ≤ t) for Student's t with *df* degrees of freedom.

    Accurate to ~4 decimal places for df ≥ 5.  Pure Python, no numpy.
    """
    # Use symmetry: for t < 0 we need the left tail which is .5 * I(...).
    # For t > 0 the left tail is 1 - .5 * I(...).
    x = df / (df + t * t)
    # Regularised incomplete beta I(x; a=df/2, b=0.5) via continued fraction
    # Lentz algorithm
    a = df / 2.0
    b = 0.5

    def _beta_cf(x: float, a: float, b: float) -> float:
        """Continued-fraction expansion of I_x(a, b) — Lentz algorithm."""
        TINY = 1e-300
        fpmin = TINY
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        ITMAX = 200
        EPS = 3e-7
        for m in range(1, ITMAX + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < EPS:
                break
        return h

    def _log_gamma(z: float) -> float:
        """Lanczos approximation ln Γ(z)."""
        c = [76.18009172947146, -86.50532032941677,
             24.01409824083091, -1.231739572450155,
             0.1208650973866179e-2, -0.5395239384953e-5]
        y = x_z = z
        tmp = x_z + 5.5
        tmp -= (x_z + 0.5) * math.log(tmp)
        ser = 1.000000000190015
        for cj in c:
            y += 1.0
            ser += cj / y
        return -tmp + math.log(2.5066282746310005 * ser / x_z)

    # betacf numerically approximates I_x(a, b)
    # Use symmetry relation for numerical stability
    if x <= 0.0:
        ibeta = 0.0
    elif x >= 1.0:
        ibeta = 1.0
    elif (x < (a + 1.0) / (a + b + 2.0)):
        betacf = _beta_cf(x, a, b)
        lbeta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
        ibeta = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) * betacf / a
    else:
        betacf = _beta_cf(1.0 - x, b, a)
        lbeta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
        ibeta = 1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x) - lbeta) * betacf / b

    # I_x(df/2, 0.5) is the two-tailed probability of exceeding |t|
    two_tailed_above = ibeta
    if t < 0:
        return two_tailed_above / 2.0
    else:
        return 1.0 - two_tailed_above / 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  Core monitor
# ═══════════════════════════════════════════════════════════════════════════

#: Minimum PnL loss below which a T+15 outcome is classified as "adverse"
_ADVERSE_CENTS_FLOOR: float = -0.20   # 0.2¢ per USD size

#: Mid-price provider type alias (async callable: asset_id → float | None)
MidPriceProvider = Callable[[str], Awaitable[float | None]]

#: Volatility provider type alias (sync callable: market_id → float | None)
#: Returns the current EWMA volatility for the market.  Used to scale
#: the t-test significance threshold dynamically.
VolProvider = Callable[[str], float | None]


class AdverseSelectionMonitor:
    """Monitor for maker-fill adverse selection, with per-market suspension.

    Parameters
    ----------
    mid_price_fn:
        Async callable ``(asset_id: str) → float | None`` that returns the
        current mid price (0–1 probability scale).  Should use the local
        L2 book, falling back to the OHLCV aggregator.
    alpha:
        One-sided significance level for suspension (default 0.05).
        When *vol_provider* is supplied, this serves as ``alpha_base``
        for the dynamic scaling formula.
    min_fills_to_suspend:
        Minimum T+15-marked fills before the t-test can trigger (default 10).
    window:
        Rolling fill window per market (default 30).
    suspension_base_s:
        Base suspension duration in seconds (default 900 = 15 min).
    suspension_max_s:
        Maximum suspension duration in seconds (default 14 400 = 4 hr).
    vol_provider:
        Optional sync callable ``(market_id: str) → float | None`` that
        returns the current EWMA volatility for the market.  When supplied,
        alpha is scaled dynamically::

            α_dynamic = α_base × (σ_rolling / σ_ref)^γ

        clamped to ``[alpha_min, alpha_max]``.
    vol_ref:
        Reference EWMA volatility for alpha scaling (default 0.01).
    alpha_gamma:
        Exponent for the vol-scaling curve (default 0.5, square-root).
    alpha_min:
        Floor for dynamic alpha (default 0.01).
    alpha_max:
        Ceiling for dynamic alpha (default 0.15).
    """

    _HORIZONS = {"t5": 5.0, "t15": 15.0, "t60": 60.0}

    def __init__(
        self,
        mid_price_fn: MidPriceProvider,
        *,
        alpha: float = 0.05,
        min_fills_to_suspend: int = 10,
        window: int = 30,
        suspension_base_s: float = 900.0,
        suspension_max_s: float = 14_400.0,
        vol_provider: VolProvider | None = None,
        vol_ref: float = 0.01,
        alpha_gamma: float = 0.5,
        alpha_min: float = 0.01,
        alpha_max: float = 0.15,
    ) -> None:
        self._mid_price_fn = mid_price_fn
        self._alpha = alpha
        self._min_fills = min_fills_to_suspend
        self._window = window
        self._suspension_base_s = suspension_base_s
        self._suspension_max_s = suspension_max_s

        # Dynamic alpha scaling
        self._vol_provider = vol_provider
        self._vol_ref = vol_ref
        self._alpha_gamma = alpha_gamma
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max

        # market_id → deque of MakerFillRecord
        self._fills: dict[str, deque[MakerFillRecord]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        # market_id → MarketAdverseStats
        self._stats: dict[str, MarketAdverseStats] = {}
        # Records waiting for their T+5/15/60 marks
        self._pending: list[MakerFillRecord] = []

    # ── Public API ────────────────────────────────────────────────────────

    def record_maker_fill(self, record: MakerFillRecord) -> None:
        """Register a confirmed maker fill for post-fill P&L tracking.

        Call this from the ``on_fill`` callback immediately after a
        POST_ONLY order is confirmed filled.
        """
        self._fills[record.market_id].append(record)
        self._pending.append(record)
        log.info(
            "maker_fill_recorded",
            market_id=record.market_id,
            asset_id=record.asset_id,
            fill_price=round(record.fill_price, 4),
            side=record.fill_side,
            size_usd=round(record.size_usd, 2),
        )

    async def tick(self, now: float | None = None) -> None:
        """Mark pending fills at T+5, T+15, T+60 seconds.

        Must be called regularly (every BBO update or at minimum every 1s).
        Fills leave ``_pending`` only after all three horizons are marked.
        """
        now = now if now is not None else time.time()
        still_pending: list[MakerFillRecord] = []

        for record in self._pending:
            elapsed = now - record.fill_time

            if elapsed >= 5.0 and record.mark_t5 is None:
                mid = await self._mid_price_fn(record.asset_id)
                if mid is not None:
                    record.mark_t5 = mid
                    record.pnl_t5 = _signed_pnl(record, mid)

            if elapsed >= 15.0 and record.mark_t15 is None:
                mid = await self._mid_price_fn(record.asset_id)
                if mid is not None:
                    record.mark_t15 = mid
                    record.pnl_t15 = _signed_pnl(record, mid)
                    record.is_adverse = record.pnl_t15 < _ADVERSE_CENTS_FLOOR
                    self._recompute_stats(record.market_id)

            if elapsed >= 60.0 and record.mark_t60 is None:
                mid = await self._mid_price_fn(record.asset_id)
                if mid is not None:
                    record.mark_t60 = mid
                    record.pnl_t60 = _signed_pnl(record, mid)

            # Keep in pending until all three horizons are filled
            if record.mark_t60 is None:
                still_pending.append(record)

        self._pending = still_pending

    def is_maker_allowed(self, market_id: str, now: float | None = None) -> bool:
        """Return True if maker routing is currently allowed for *market_id*.

        Returns False while the market is within its suspension window.
        """
        now = now if now is not None else time.time()
        stats = self._stats.get(market_id)
        if stats is None:
            return True
        if stats.suspension_until > now:
            log.debug(
                "maker_suspended",
                market_id=market_id,
                remaining_s=round(stats.suspension_until - now),
            )
            return False
        return True

    def get_stats(self, market_id: str) -> MarketAdverseStats | None:
        """Return the rolling stats for *market_id*, or None if no data."""
        return self._stats.get(market_id)

    def get_all_stats(self) -> dict[str, MarketAdverseStats]:
        """Return a snapshot of all market stats (for health reporting)."""
        return dict(self._stats)

    def get_pending_count(self) -> int:
        """Number of fills still awaiting their T+60 mark."""
        return len(self._pending)

    # ── Dynamic alpha ─────────────────────────────────────────────────────

    def _dynamic_alpha(self, market_id: str) -> float:
        """Compute volatility-scaled significance threshold.

        Formula::

            α_dynamic = α_base × (σ_rolling / σ_ref)^γ

        Clamped to ``[α_min, α_max]``.  Falls back to fixed ``α_base``
        when no *vol_provider* is configured or returns ``None``.

        Rationale
        ---------
        - **Low vol** (σ = 0.005):  α ≈ 0.035 — more conservative;
          small price noise produces inflated t-stats, so raising the
          bar prevents false suspensions.
        - **Normal vol** (σ = σ_ref = 0.01):  α = α_base = 0.05.
        - **High vol** (σ = 0.02):  α ≈ 0.071 — more aggressive;
          genuine adverse selection is masked by wide swings, so
          loosening the threshold accelerates suspension.
        """
        if self._vol_provider is None:
            return self._alpha

        vol = self._vol_provider(market_id)
        if vol is None or vol <= 0 or self._vol_ref <= 0:
            return self._alpha

        ratio = vol / self._vol_ref
        scaled = self._alpha * (ratio ** self._alpha_gamma)
        clamped = max(self._alpha_min, min(self._alpha_max, scaled))

        log.debug(
            "dynamic_alpha_computed",
            market_id=market_id,
            vol=round(vol, 6),
            vol_ref=self._vol_ref,
            ratio=round(ratio, 4),
            alpha_base=self._alpha,
            alpha_dynamic=round(clamped, 6),
        )
        return clamped

    # ── Statistical engine ────────────────────────────────────────────────

    def _recompute_stats(self, market_id: str) -> None:
        """Recompute stats after a new T+15 mark; suspend if significant."""
        fills_with_t15 = [
            f for f in self._fills[market_id]
            if f.pnl_t15 is not None
        ]
        n = len(fills_with_t15)

        stats = self._stats.setdefault(
            market_id, MarketAdverseStats(market_id=market_id)
        )
        stats.n_fills = n
        stats.n_adverse = sum(1 for f in fills_with_t15 if f.is_adverse)

        if n == 0:
            return

        pnls = [f.pnl_t15 for f in fills_with_t15]  # type: ignore[misc]
        stats.avg_pnl_t15 = statistics.mean(pnls)

        # Compute T+5 and T+60 averages if data available
        t5s = [f.pnl_t5 for f in fills_with_t15 if f.pnl_t5 is not None]
        t60s = [f.pnl_t60 for f in fills_with_t15 if f.pnl_t60 is not None]
        if t5s:
            stats.avg_pnl_t5 = statistics.mean(t5s)
        if t60s:
            stats.avg_pnl_t60 = statistics.mean(t60s)

        # One-sample t-test: H₀: μ ≥ 0, H₁: μ < 0 (lower-tail)
        if n < self._min_fills:
            return
        if n < 2:
            return

        try:
            stdev = statistics.stdev(pnls)
        except statistics.StatisticsError:
            return
        if stdev == 0.0:
            return

        t_stat = stats.avg_pnl_t15 / (stdev / math.sqrt(n))
        p_value = _t_cdf_lower_tail(t_stat, df=n - 1)

        stats.last_t_stat = round(t_stat, 4)
        stats.last_p_value = round(p_value, 6)

        # Use volatility-scaled alpha when a vol_provider is configured;
        # otherwise falls back to the fixed self._alpha.
        effective_alpha = self._dynamic_alpha(market_id)

        log.info(
            "adverse_sel_stats",
            market_id=market_id,
            n_fills=n,
            avg_pnl_t15=round(stats.avg_pnl_t15, 4),
            t_stat=stats.last_t_stat,
            p_value=stats.last_p_value,
            adverse_rate=round(stats.adverse_rate, 3),
            effective_alpha=round(effective_alpha, 6),
        )

        if p_value < effective_alpha and not self._is_suspended(market_id):
            self._suspend(market_id, stats)

    def _is_suspended(self, market_id: str) -> bool:
        stats = self._stats.get(market_id)
        if stats is None:
            return False
        return stats.suspension_until > time.time()

    def _suspend(self, market_id: str, stats: MarketAdverseStats) -> None:
        stats.suspension_count += 1
        backoff_s = min(
            self._suspension_base_s * (2 ** (stats.suspension_count - 1)),
            self._suspension_max_s,
        )
        stats.suspension_until = time.time() + backoff_s
        stats.last_suspended_at = time.time()
        log.warning(
            "maker_routing_suspended",
            market_id=market_id,
            backoff_s=backoff_s,
            suspension_count=stats.suspension_count,
            avg_pnl_t15=round(stats.avg_pnl_t15, 4),
            t_stat=stats.last_t_stat,
            p_value=stats.last_p_value,
            n_fills=stats.n_fills,
            adverse_rate=round(stats.adverse_rate, 3),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _signed_pnl(record: MakerFillRecord, mark_price: float) -> float:
    """Compute signed P&L: cents × USD size.

    Positive = price moved in our favour after fill.
    Uses probability-scale prices (0–1); cents = Δ × 100.
    """
    direction = 1.0 if record.fill_side == "BUY" else -1.0
    return direction * (mark_price - record.fill_price) * 100.0 * record.size_usd


def make_fill_record(
    market_id: str,
    asset_id: str,
    fill_price: float,
    fill_side: str,
    size_usd: float,
    fill_time: float | None = None,
) -> MakerFillRecord:
    """Convenience constructor for a :class:`MakerFillRecord`."""
    return MakerFillRecord(
        market_id=market_id,
        asset_id=asset_id,
        fill_price=fill_price,
        fill_side=fill_side,
        fill_time=fill_time if fill_time is not None else time.time(),
        size_usd=size_usd,
    )
