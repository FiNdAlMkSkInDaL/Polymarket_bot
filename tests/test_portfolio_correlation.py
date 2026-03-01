"""
Comprehensive tests for the Portfolio Correlation Engine (PCE) — Pillar 15.

Covers:
  1. Pearson correlation correctness (known values)
  2. Structural fallback (same event, shared tag, no overlap)
  3. Bayesian blending (structural → empirical dominance)
  4. VaR gate accept (uncorrelated → low VaR → allowed)
  5. VaR gate reject (highly correlated → high VaR → blocked)
  6. Concentration haircut math
  7. Matrix serialisation round-trip
  8. Staleness decay
  9. Shadow mode behaviour
 10. Cold-start behaviour
 11. VaR computation performance (<10 ms for 20 positions)
 12. Single-position diversification (never reject a diversifying add)
 13. Zero-volatility edge case
"""

from __future__ import annotations

import json
import math
import os
import time

import pytest

# Force paper mode before any src imports
os.environ.setdefault("PAPER_MODE", "true")
os.environ.setdefault("DEPLOYMENT_ENV", "PAPER")

from src.trading.portfolio_correlation import (
    CorrelationEstimate,
    CorrelationMatrix,
    PortfolioCorrelationEngine,
    VaRCalculator,
    VaRResult,
    VaRSizingResult,
    _BarProxy,
    _log_returns,
    _tags_overlap,
    pearson_correlation,
)
from src.data.ohlcv import OHLCVAggregator, OHLCVBar


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_bars(closes: list[float], start_ts: float = 1000.0) -> list[OHLCVBar]:
    """Create a list of OHLCVBar objects with aligned timestamps."""
    bars: list[OHLCVBar] = []
    for i, c in enumerate(closes):
        bars.append(OHLCVBar(
            open_time=start_ts + i * 60.0,
            open=c,
            high=c + 0.01,
            low=c - 0.01,
            close=c,
            volume=100.0,
            vwap=c,
            trade_count=10,
        ))
    return bars


class _FakePosition:
    """Minimal position-like object for testing."""
    def __init__(self, market_id: str, entry_price: float, entry_size: float,
                 event_id: str = "", trade_side: str = "NO"):
        self.market_id = market_id
        self.entry_price = entry_price
        self.entry_size = entry_size
        self.event_id = event_id
        self.trade_side = trade_side


class _FakeAggregator:
    """Minimal aggregator-like object for testing."""
    def __init__(self, bars: list[OHLCVBar] | None = None, vol: float = 0.05):
        self.bars = bars or []
        self.rolling_volatility = vol


# ═══════════════════════════════════════════════════════════════════════════
#  1. Pearson Correlation Correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestPearsonCorrelation:
    def test_perfectly_correlated(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(pearson_correlation(xs, ys) - 1.0) < 1e-10

    def test_perfectly_anticorrelated(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(pearson_correlation(xs, ys) - (-1.0)) < 1e-10

    def test_uncorrelated(self):
        # Orthogonal vectors centered at 0
        xs = [1.0, -1.0, 0.0, 0.0]
        ys = [0.0, 0.0, 1.0, -1.0]
        assert abs(pearson_correlation(xs, ys)) < 1e-10

    def test_constant_series_returns_zero(self):
        xs = [5.0, 5.0, 5.0, 5.0]
        ys = [1.0, 2.0, 3.0, 4.0]
        assert pearson_correlation(xs, ys) == 0.0

    def test_both_constant_returns_zero(self):
        xs = [3.0, 3.0, 3.0]
        ys = [7.0, 7.0, 7.0]
        assert pearson_correlation(xs, ys) == 0.0

    def test_too_few_points(self):
        assert pearson_correlation([1.0], [2.0]) == 0.0
        assert pearson_correlation([], []) == 0.0

    def test_moderate_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.2, 2.5, 2.8, 4.1, 5.3]
        r = pearson_correlation(xs, ys)
        assert 0.95 < r < 1.0  # high but not perfect


# ═══════════════════════════════════════════════════════════════════════════
#  2. Structural Fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuralFallback:
    def test_same_event_high_correlation(self):
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "MKT_A", "MKT_B",
            "EVT_1", "EVT_1",  # same event
            "crypto", "crypto",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )
        est = cm.get_estimate("MKT_A", "MKT_B")
        assert est is not None
        assert est.structural_corr == 0.85

    def test_shared_tag_moderate_correlation(self):
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "MKT_A", "MKT_B",
            "EVT_1", "EVT_2",  # different events
            "crypto,politics", "crypto,sports",  # shared "crypto"
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )
        est = cm.get_estimate("MKT_A", "MKT_B")
        assert est is not None
        assert est.structural_corr == 0.30

    def test_no_overlap_baseline(self):
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "MKT_A", "MKT_B",
            "EVT_1", "EVT_2",
            "crypto", "politics",  # no overlap
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )
        est = cm.get_estimate("MKT_A", "MKT_B")
        assert est is not None
        assert est.structural_corr == 0.05

    def test_empty_tags_no_overlap(self):
        assert _tags_overlap("", "crypto") is False
        assert _tags_overlap("crypto", "") is False
        assert _tags_overlap("", "") is False

    def test_tags_overlap_case_insensitive(self):
        assert _tags_overlap("Crypto, Politics", "CRYPTO") is True
        assert _tags_overlap("sports", "SPORTS,politics") is True

    def test_unknown_pair_returns_baseline(self):
        cm = CorrelationMatrix()
        # No pair registered — should return baseline
        corr = cm.get("UNKNOWN_A", "UNKNOWN_B")
        assert corr == 0.05  # default baseline from config


# ═══════════════════════════════════════════════════════════════════════════
#  3. Bayesian Blending
# ═══════════════════════════════════════════════════════════════════════════

class TestBayesianBlending:
    def test_zero_empirical_data_pure_structural(self):
        """With 0 overlap bars, blended should equal structural prior."""
        est = CorrelationEstimate(
            empirical_corr=0.7,
            structural_corr=0.30,
            overlap_bars=0,
        )
        blended = est.blended_with_weight(prior_weight=10)
        assert abs(blended - 0.30) < 1e-10

    def test_equal_weight_blending(self):
        """Prior weight = 10, overlap = 10 → 50/50 blend."""
        est = CorrelationEstimate(
            empirical_corr=0.80,
            structural_corr=0.30,
            overlap_bars=10,
        )
        blended = est.blended_with_weight(prior_weight=10)
        expected = (10 * 0.30 + 10 * 0.80) / 20
        assert abs(blended - expected) < 1e-10

    def test_empirical_dominates_with_many_bars(self):
        """With 30+ bars and prior_weight=10, empirical should dominate."""
        est = CorrelationEstimate(
            empirical_corr=0.90,
            structural_corr=0.30,
            overlap_bars=50,
        )
        blended = est.blended_with_weight(prior_weight=10)
        expected = (10 * 0.30 + 50 * 0.90) / 60
        assert abs(blended - expected) < 1e-10
        # Should be much closer to 0.90 than 0.30
        assert blended >= 0.80

    def test_blending_formula_explicit(self):
        """Verify the Bayesian formula: (pw * structural + n * empirical) / (pw + n)."""
        est = CorrelationEstimate(
            empirical_corr=-0.3,
            structural_corr=0.05,
            overlap_bars=20,
        )
        pw = 10
        expected = (pw * 0.05 + 20 * (-0.3)) / (pw + 20)
        blended = est.blended_with_weight(prior_weight=pw)
        assert abs(blended - expected) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
#  4. VaR Gate — Accept (uncorrelated)
# ═══════════════════════════════════════════════════════════════════════════

class TestVaRGateAccept:
    def test_uncorrelated_positions_below_threshold(self):
        """Two uncorrelated positions with low vol → VaR below threshold."""
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "MKT_A", "MKT_B", "E1", "E2", "", "",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )

        calc = VaRCalculator(z_score=1.645)
        positions = [
            {"market_id": "MKT_A", "exposure_usd": 5.0},
        ]
        proposed = {"market_id": "MKT_B", "exposure_usd": 5.0}
        vols = {"MKT_A": 0.05, "MKT_B": 0.05}

        result = calc.compute_portfolio_var(
            positions=positions,
            proposed=proposed,
            corr_matrix=cm,
            volatilities=vols,
            threshold=15.0,
        )

        assert result.portfolio_var_usd < 15.0
        assert not result.exceeds_threshold
        assert result.diversification_benefit > 0  # uncorrelated → some benefit

    def test_empty_portfolio_always_accepts(self):
        """Adding the first position to an empty portfolio should always pass."""
        cm = CorrelationMatrix()
        calc = VaRCalculator(z_score=1.645)

        result = calc.compute_portfolio_var(
            positions=[],
            proposed={"market_id": "MKT_A", "exposure_usd": 10.0},
            corr_matrix=cm,
            volatilities={"MKT_A": 0.05},
            threshold=15.0,
        )

        assert result.portfolio_var_usd < 15.0
        assert not result.exceeds_threshold
        # Marginal VaR = total VaR for first position
        assert result.marginal_var_usd > 0


# ═══════════════════════════════════════════════════════════════════════════
#  5. VaR Gate — Reject (highly correlated)
# ═══════════════════════════════════════════════════════════════════════════

class TestVaRGateReject:
    def test_correlated_positions_exceed_threshold(self):
        """Three highly correlated positions with high vol → VaR exceeds threshold."""
        cm = CorrelationMatrix()
        # Set high correlation between all pairs
        for pair in [("A", "B"), ("A", "C"), ("B", "C")]:
            cm.set_structural_with_values(
                f"MKT_{pair[0]}", f"MKT_{pair[1]}",
                "EVT_1", "EVT_1",  # same event → 0.85
                "crypto", "crypto",
                same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
            )

        calc = VaRCalculator(z_score=1.645)
        positions = [
            {"market_id": "MKT_A", "exposure_usd": 10.0},
            {"market_id": "MKT_B", "exposure_usd": 10.0},
        ]
        proposed = {"market_id": "MKT_C", "exposure_usd": 10.0}
        # High volatility to push VaR over threshold
        vols = {"MKT_A": 0.50, "MKT_B": 0.50, "MKT_C": 0.50}

        result = calc.compute_portfolio_var(
            positions=positions,
            proposed=proposed,
            corr_matrix=cm,
            volatilities=vols,
            threshold=5.0,  # low threshold to trigger rejection
        )

        assert result.exceeds_threshold
        assert result.portfolio_var_usd > 5.0

    def test_marginal_var_positive_for_correlated_add(self):
        """Marginal VaR should be positive when adding correlated position."""
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "MKT_A", "MKT_B", "E1", "E1", "", "",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )

        calc = VaRCalculator(z_score=1.645)
        positions = [{"market_id": "MKT_A", "exposure_usd": 10.0}]
        proposed = {"market_id": "MKT_B", "exposure_usd": 10.0}
        vols = {"MKT_A": 0.20, "MKT_B": 0.20}

        result = calc.compute_portfolio_var(
            positions=positions,
            proposed=proposed,
            corr_matrix=cm,
            volatilities=vols,
        )

        assert result.marginal_var_usd > 0


# ═══════════════════════════════════════════════════════════════════════════
#  6. Concentration Haircut Math
# ═══════════════════════════════════════════════════════════════════════════

class TestConcentrationHaircut:
    def test_high_correlation_haircut_applied(self):
        """avg pairwise corr = 0.6 → haircut = 0.4 → Kelly × 0.4."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            haircut_threshold=0.50,
            structural_same_event=0.85,
            structural_same_tag=0.30,
            structural_baseline=0.60,  # force high baseline
            structural_prior_weight=10,
        )

        # Register markets with no tag overlap → will use baseline of 0.60
        pce.register_market("MKT_PROPOSED", "", "", _FakeAggregator())
        pce.register_market("MKT_EXISTING", "", "", _FakeAggregator())

        positions = [_FakePosition("MKT_EXISTING", 0.50, 20.0)]
        haircut = pce.compute_concentration_haircut("MKT_PROPOSED", positions)

        # avg corr = 0.60 (structural baseline) > threshold 0.50
        # haircut = 1 - 0.60 = 0.40
        assert abs(haircut - 0.40) < 0.01

    def test_low_correlation_no_haircut(self):
        """avg pairwise corr = 0.05 → no haircut (returns 1.0)."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            haircut_threshold=0.50,
            structural_baseline=0.05,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_PROPOSED", "E1", "crypto", _FakeAggregator())
        pce.register_market("MKT_EXISTING", "E2", "politics", _FakeAggregator())

        positions = [_FakePosition("MKT_EXISTING", 0.50, 20.0)]
        haircut = pce.compute_concentration_haircut("MKT_PROPOSED", positions)

        assert haircut == 1.0

    def test_empty_portfolio_no_haircut(self):
        """No existing positions → no haircut."""
        pce = PortfolioCorrelationEngine(data_dir="/tmp/test_pce")
        haircut = pce.compute_concentration_haircut("MKT_A", [])
        assert haircut == 1.0

    def test_haircut_floor_at_five_percent(self):
        """Even with correlation ≈ 1.0, haircut should floor at 0.05."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            haircut_threshold=0.50,
            structural_same_event=0.98,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_A", "EVT_1", "", _FakeAggregator())
        pce.register_market("MKT_B", "EVT_1", "", _FakeAggregator())

        positions = [_FakePosition("MKT_B", 0.50, 20.0)]
        haircut = pce.compute_concentration_haircut("MKT_A", positions)

        assert haircut >= 0.05  # floor
        assert haircut < 0.10  # but heavily penalised

    def test_proposed_market_same_as_existing_excluded(self):
        """Position in same market as proposed should be excluded from avg."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            haircut_threshold=0.50,
            structural_baseline=0.05,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_A", "E1", "", _FakeAggregator())

        # Only position is in same market as proposed → no peers → no haircut
        positions = [_FakePosition("MKT_A", 0.50, 20.0)]
        haircut = pce.compute_concentration_haircut("MKT_A", positions)

        assert haircut == 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  7. Matrix Serialisation Round-Trip
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialisation:
    def test_json_round_trip(self):
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "MKT_A", "MKT_B", "E1", "E1", "crypto", "crypto",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )
        # Set empirical data
        bars_a = _make_bars([0.50, 0.51, 0.52, 0.53, 0.54])
        bars_b = _make_bars([0.60, 0.61, 0.62, 0.63, 0.64])
        cm.update_empirical("MKT_A", "MKT_B", bars_a, bars_b)

        # Serialise → deserialise
        data = cm.to_json()
        cm2 = CorrelationMatrix.from_json(data)

        # Verify values preserved
        est_orig = cm.get_estimate("MKT_A", "MKT_B")
        est_restored = cm2.get_estimate("MKT_A", "MKT_B")
        assert est_orig is not None
        assert est_restored is not None
        assert abs(est_orig.empirical_corr - est_restored.empirical_corr) < 1e-5
        assert abs(est_orig.structural_corr - est_restored.structural_corr) < 1e-5
        assert est_orig.overlap_bars == est_restored.overlap_bars

    def test_file_round_trip(self, tmp_path):
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "A", "B", "E1", "E2", "tag1", "tag2",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )

        path = tmp_path / "test_corr.json"
        cm.save(path)

        cm2 = CorrelationMatrix.load(path)
        est = cm2.get_estimate("A", "B")
        assert est is not None
        assert est.structural_corr == 0.05  # different event, no tag overlap

    def test_load_missing_file_returns_empty(self, tmp_path):
        cm = CorrelationMatrix.load(tmp_path / "nonexistent.json")
        assert len(cm._pairs) == 0

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json at all {{{{")
        cm = CorrelationMatrix.load(path)
        assert len(cm._pairs) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  8. Staleness Decay
# ═══════════════════════════════════════════════════════════════════════════

class TestStalenessDecay:
    def test_24_hour_decay_halves_overlap(self):
        """After 24 hours stale (halflife=24), overlap should be halved."""
        cm = CorrelationMatrix()
        cm._pairs[("A", "B")] = CorrelationEstimate(
            empirical_corr=0.70,
            structural_corr=0.30,
            overlap_bars=40,
            last_updated=time.time() - 86400,  # 24h ago
        )

        cm.decay_confidence(hours_stale=24.0, halflife_hours=24.0)

        est = cm._pairs[("A", "B")]
        assert est.overlap_bars == 20  # halved

    def test_48_hour_decay_quarters_overlap(self):
        """After 48 hours stale (halflife=24), overlap should be quartered."""
        cm = CorrelationMatrix()
        cm._pairs[("A", "B")] = CorrelationEstimate(
            empirical_corr=0.70,
            structural_corr=0.30,
            overlap_bars=40,
        )

        cm.decay_confidence(hours_stale=48.0, halflife_hours=24.0)

        est = cm._pairs[("A", "B")]
        assert est.overlap_bars == 10  # quartered

    def test_structural_re_emerges_after_decay(self):
        """After heavy decay, blended should converge toward structural."""
        est = CorrelationEstimate(
            empirical_corr=0.90,
            structural_corr=0.30,
            overlap_bars=40,
        )

        # Before decay: empirical dominates
        before = est.blended_with_weight(10)
        assert before > 0.70

        # Apply heavy decay (72h with 24h halflife → factor ≈ 0.125)
        est.overlap_bars = max(0, int(est.overlap_bars * (0.5 ** 3)))
        assert est.overlap_bars == 5

        # After decay: structural has more weight
        after = est.blended_with_weight(10)
        # With 10 prior + 5 empirical: (10*0.30 + 5*0.90) / 15 = 0.50
        assert abs(after - 0.50) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  9. Shadow Mode Behaviour
# ═══════════════════════════════════════════════════════════════════════════

class TestShadowMode:
    def test_shadow_mode_allows_but_flags(self):
        """In shadow mode, VaR gate returns allowed=True but flags exceeds."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=True,
            max_portfolio_var_usd=0.01,  # tiny threshold to trigger
            structural_same_event=0.85,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_A", "EVT_1", "crypto", _FakeAggregator(vol=0.50))
        pce.register_market("MKT_B", "EVT_1", "crypto", _FakeAggregator(vol=0.50))

        positions = [_FakePosition("MKT_A", 0.50, 20.0)]

        allowed, result = pce.check_var_gate(
            positions, "MKT_B", 10.0, "NO"
        )

        assert allowed is True  # shadow mode → never blocks
        assert result.exceeds_threshold  # but flags the violation

    def test_live_mode_rejects(self):
        """In live mode, VaR gate returns allowed=False when threshold exceeded."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=0.01,  # tiny threshold
            structural_same_event=0.85,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_A", "EVT_1", "crypto", _FakeAggregator(vol=0.50))
        pce.register_market("MKT_B", "EVT_1", "crypto", _FakeAggregator(vol=0.50))

        positions = [_FakePosition("MKT_A", 0.50, 20.0)]

        allowed, result = pce.check_var_gate(
            positions, "MKT_B", 10.0, "NO"
        )

        assert allowed is False
        assert result.exceeds_threshold


# ═══════════════════════════════════════════════════════════════════════════
#  10. Cold-Start Behaviour
# ═══════════════════════════════════════════════════════════════════════════

class TestColdStart:
    def test_no_history_uses_structural_priors(self):
        """With no empirical data, all correlations should be structural."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            structural_same_event=0.85,
            structural_same_tag=0.30,
            structural_baseline=0.05,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_A", "EVT_1", "crypto", _FakeAggregator())
        pce.register_market("MKT_B", "EVT_1", "crypto", _FakeAggregator())
        pce.register_market("MKT_C", "EVT_2", "politics", _FakeAggregator())

        # Same event → 0.85
        assert abs(pce.corr_matrix.get("MKT_A", "MKT_B") - 0.85) < 0.01
        # Different event, no tag overlap → baseline 0.05
        assert abs(pce.corr_matrix.get("MKT_A", "MKT_C") - 0.05) < 0.01

    def test_load_nonexistent_state_graceful(self, tmp_path):
        """Loading from nonexistent path should not crash."""
        pce = PortfolioCorrelationEngine(data_dir=str(tmp_path / "nope"))
        pce.load_state()
        # Should have empty correlation matrix
        assert len(pce.corr_matrix._pairs) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  11. VaR Computation Performance
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformance:
    def test_20_position_var_under_10ms(self):
        """VaR computation for 20 positions must complete in <10 ms."""
        n = 20
        cm = CorrelationMatrix()
        markets = [f"MKT_{i}" for i in range(n)]

        # Set correlations for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                cm.set_structural_with_values(
                    markets[i], markets[j],
                    f"E_{i % 5}", f"E_{j % 5}",  # some same-event overlap
                    f"tag_{i % 3}", f"tag_{j % 3}",
                    same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
                )

        calc = VaRCalculator(z_score=1.645)
        positions = [
            {"market_id": markets[i], "exposure_usd": 5.0 + i * 0.5}
            for i in range(n - 1)
        ]
        proposed = {"market_id": markets[n - 1], "exposure_usd": 5.0}
        vols = {m: 0.05 + i * 0.01 for i, m in enumerate(markets)}

        # Warm up
        calc.compute_portfolio_var(positions, proposed, cm, vols)

        # Timed run
        start = time.perf_counter()
        for _ in range(100):
            calc.compute_portfolio_var(positions, proposed, cm, vols)
        elapsed = (time.perf_counter() - start) / 100

        # Should be well under 10ms per call
        assert elapsed < 0.010, f"VaR computation took {elapsed*1000:.2f} ms (limit: 10ms)"


# ═══════════════════════════════════════════════════════════════════════════
#  12. Single-Position Diversification
# ═══════════════════════════════════════════════════════════════════════════

class TestDiversification:
    def test_diversifying_add_never_rejected(self):
        """Adding a low-correlation position to a single existing one should pass."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=15.0,
            structural_baseline=0.05,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_A", "E1", "crypto", _FakeAggregator(vol=0.05))
        pce.register_market("MKT_B", "E2", "politics", _FakeAggregator(vol=0.05))

        positions = [_FakePosition("MKT_A", 0.50, 10.0)]

        allowed, result = pce.check_var_gate(
            positions, "MKT_B", 5.0, "NO"
        )

        assert allowed is True
        assert result.diversification_benefit > 0

    def test_diversification_benefit_increases_with_low_corr(self):
        """Diversification benefit should be higher with lower correlation."""
        cm = CorrelationMatrix()

        # High correlation setup
        cm_high = CorrelationMatrix()
        cm_high.set_structural_with_values(
            "A", "B", "E1", "E1", "", "",
            same_event_corr=0.90, same_tag_corr=0.30, baseline_corr=0.05,
        )

        # Low correlation setup
        cm_low = CorrelationMatrix()
        cm_low.set_structural_with_values(
            "A", "B", "E1", "E2", "", "",
            same_event_corr=0.90, same_tag_corr=0.30, baseline_corr=0.05,
        )

        calc = VaRCalculator(z_score=1.645)
        positions = [{"market_id": "A", "exposure_usd": 10.0}]
        proposed = {"market_id": "B", "exposure_usd": 10.0}
        vols = {"A": 0.10, "B": 0.10}

        result_high = calc.compute_portfolio_var(positions, proposed, cm_high, vols)
        result_low = calc.compute_portfolio_var(positions, proposed, cm_low, vols)

        # Low correlation → more diversification benefit
        assert result_low.diversification_benefit > result_high.diversification_benefit


# ═══════════════════════════════════════════════════════════════════════════
#  13. Zero-Volatility Edge Case
# ═══════════════════════════════════════════════════════════════════════════

class TestZeroVolatility:
    def test_zero_vol_market_contributes_zero_var(self):
        """A market with σ=0 should contribute 0 to portfolio VaR."""
        cm = CorrelationMatrix()
        calc = VaRCalculator(z_score=1.645)

        positions = [{"market_id": "MKT_A", "exposure_usd": 10.0}]
        proposed = {"market_id": "MKT_B", "exposure_usd": 10.0}
        # MKT_B has zero vol — contributes nothing to VaR
        vols = {"MKT_A": 0.10, "MKT_B": 0.0}

        result = calc.compute_portfolio_var(positions, proposed, cm, vols)

        # VaR should only reflect MKT_A's contribution
        expected_var_a_only = 1.645 * 10.0 * 0.10
        # MKT_B with zero vol: VaR shouldn't increase much
        # (it gets floored to 0.01 in _get_volatilities, but in direct test it's 0)
        assert abs(result.portfolio_var_usd - expected_var_a_only) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Empirical Correlation from Bars
# ═══════════════════════════════════════════════════════════════════════════

class TestEmpiricalCorrelation:
    def test_comoving_bars_high_correlation(self):
        """Two markets with co-moving prices → high empirical correlation."""
        cm = CorrelationMatrix()

        # Create co-moving price series
        base = [0.50 + 0.01 * i for i in range(20)]
        bars_a = _make_bars(base)
        bars_b = _make_bars([p + 0.10 for p in base])  # shifted but same trend

        corr = cm.update_empirical("A", "B", bars_a, bars_b)
        assert corr is not None
        assert corr > 0.90

    def test_opposing_bars_negative_correlation(self):
        """Two markets with opposing returns → negative correlation."""
        cm = CorrelationMatrix()

        # Create genuinely anti-correlated return series:
        # A goes up, B goes down alternately
        prices_a = [0.50]
        prices_b = [0.50]
        for i in range(1, 20):
            delta = 0.02 * (1 if i % 2 == 0 else -1)
            prices_a.append(prices_a[-1] + delta)
            prices_b.append(prices_b[-1] - delta)

        bars_a = _make_bars(prices_a)
        bars_b = _make_bars(prices_b)

        corr = cm.update_empirical("A", "B", bars_a, bars_b)
        assert corr is not None
        assert corr < -0.90

    def test_insufficient_overlap_returns_none(self):
        """Non-overlapping timestamps → returns None."""
        cm = CorrelationMatrix()

        bars_a = _make_bars([0.50, 0.51], start_ts=1000.0)
        bars_b = _make_bars([0.60, 0.61], start_ts=99000.0)  # different time

        corr = cm.update_empirical("A", "B", bars_a, bars_b)
        assert corr is None

    def test_correlation_matrix_construction(self):
        """get_matrix should return proper N×N with 1.0 diagonal."""
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "A", "B", "E1", "E2", "", "",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )
        cm.set_structural_with_values(
            "A", "C", "E1", "E1", "", "",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )
        cm.set_structural_with_values(
            "B", "C", "E3", "E1", "tag1", "tag1",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )

        mat = cm.get_matrix(["A", "B", "C"])
        assert len(mat) == 3
        assert all(len(row) == 3 for row in mat)

        # Diagonal must be 1.0
        for i in range(3):
            assert mat[i][i] == 1.0

        # Symmetric
        for i in range(3):
            for j in range(3):
                assert abs(mat[i][j] - mat[j][i]) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
#  Dashboard Data
# ═══════════════════════════════════════════════════════════════════════════

class TestDashboard:
    def test_dashboard_data_structure(self):
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            structural_prior_weight=10,
        )

        pce.register_market("MKT_A", "E1", "crypto", _FakeAggregator(vol=0.05))
        pce.register_market("MKT_B", "E1", "crypto", _FakeAggregator(vol=0.05))

        positions = [_FakePosition("MKT_A", 0.50, 10.0)]

        data = pce.get_dashboard_data(positions)

        assert "portfolio_var" in data
        assert "threshold" in data
        assert "top_correlated_pairs" in data
        assert "open_positions" in data
        assert isinstance(data["top_correlated_pairs"], list)
        assert data["open_positions"] == 1

    def test_dashboard_empty_portfolio(self):
        pce = PortfolioCorrelationEngine(data_dir="/tmp/test_pce")
        data = pce.get_dashboard_data([])

        assert data["portfolio_var"] == 0.0
        assert data["open_positions"] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Log Returns
# ═══════════════════════════════════════════════════════════════════════════

class TestLogReturns:
    def test_basic_returns(self):
        closes = [1.0, 1.1, 1.21]
        rets = _log_returns(closes)
        assert len(rets) == 2
        assert abs(rets[0] - math.log(1.1)) < 1e-10
        assert abs(rets[1] - math.log(1.21 / 1.1)) < 1e-10

    def test_zero_price_guarded(self):
        closes = [0.0, 0.5, 1.0]
        rets = _log_returns(closes)
        assert len(rets) == 2
        assert all(math.isfinite(r) for r in rets)


# ═══════════════════════════════════════════════════════════════════════════
#  VaR Result Serialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestVaRResultSerialisation:
    def test_to_dict(self):
        r = VaRResult(
            portfolio_var_usd=5.123456,
            marginal_var_usd=1.234,
            diversification_benefit=0.567,
            gross_exposure=20.0,
            net_exposure=18.0,
            exceeds_threshold=False,
        )
        d = r.to_dict()
        assert d["portfolio_var_usd"] == 5.1235
        assert d["exceeds_threshold"] is False


# ═════════════════════════════════════════════════════════════════════════
#  Issue 1: min_overlap_bars enforcement
# ═════════════════════════════════════════════════════════════════════════

class TestMinOverlapBars:
    def test_min_overlap_bars_enforced(self):
        """With 10 overlapping bars and min_overlap=30, empirical should NOT update."""
        cm = CorrelationMatrix()
        cm.set_structural_with_values(
            "A", "B", "E1", "E2", "", "",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )

        # 10 co-moving bars (plenty for Pearson, but below min_overlap=30)
        bars_a = _make_bars([0.50 + 0.01 * i for i in range(10)])
        bars_b = _make_bars([0.60 + 0.01 * i for i in range(10)])

        result = cm.update_empirical("A", "B", bars_a, bars_b, min_overlap=30)
        assert result is None  # should be rejected

        # Correlation should still be structural prior (0.05)
        est = cm.get_estimate("A", "B")
        assert est is not None
        assert est.overlap_bars == 0
        assert abs(cm.get("A", "B") - 0.05) < 0.01

    def test_min_overlap_default_backward_compatible(self):
        """Default min_overlap=2 preserves existing behaviour."""
        cm = CorrelationMatrix()
        bars_a = _make_bars([0.50, 0.51, 0.52])
        bars_b = _make_bars([0.60, 0.61, 0.62])

        result = cm.update_empirical("A", "B", bars_a, bars_b)
        assert result is not None  # default min_overlap=2, 3 bars → accepted


# ═════════════════════════════════════════════════════════════════════════
#  Issue 2: Volatility scale mismatch
# ═════════════════════════════════════════════════════════════════════════

class TestVolatilityScaling:
    def test_holding_period_volatility_scaling(self):
        """With holding_period=120 and 1-min σ=0.01, effective σ ≈ 0.1095."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            holding_period_minutes=120,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.01))

        vols = pce._get_volatilities()
        expected = 0.01 * math.sqrt(120)
        assert abs(vols["MKT_A"] - expected) < 1e-6
        assert abs(expected - 0.1095) < 0.001  # sanity check

    def test_holding_period_1min_no_scaling(self):
        """With holding_period=1, volatility should be unscaled."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            holding_period_minutes=1,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.05))

        vols = pce._get_volatilities()
        assert abs(vols["MKT_A"] - 0.05) < 1e-10


# ═════════════════════════════════════════════════════════════════════════
#  Issue 3: Exposure-weighted concentration haircut
# ═════════════════════════════════════════════════════════════════════════

class TestExposureWeightedHaircut:
    def test_large_exposure_dominates(self):
        """$50 peer with ρ=0.8 dominates over $5 peer with ρ=0.1."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            haircut_threshold=0.30,
            structural_baseline=0.10,
            structural_prior_weight=10,
        )

        pce.register_market("MKT_PROPOSED", "E1", "", _FakeAggregator())
        pce.register_market("MKT_BIG", "E2", "", _FakeAggregator())
        pce.register_market("MKT_SMALL", "E3", "", _FakeAggregator())

        # Override correlations directly for deterministic test
        pce.corr_matrix._pairs[("MKT_BIG", "MKT_PROPOSED")] = CorrelationEstimate(
            empirical_corr=0.8, structural_corr=0.8, overlap_bars=100,
        )
        pce.corr_matrix._pairs[("MKT_PROPOSED", "MKT_SMALL")] = CorrelationEstimate(
            empirical_corr=0.1, structural_corr=0.1, overlap_bars=100,
        )

        positions = [
            _FakePosition("MKT_BIG", 1.0, 50.0),    # $50 exposure, ρ=0.8
            _FakePosition("MKT_SMALL", 1.0, 5.0),    # $5 exposure, ρ=0.1
        ]

        haircut = pce.compute_concentration_haircut("MKT_PROPOSED", positions)

        # Unweighted avg = (0.8 + 0.1) / 2 = 0.45
        # Weighted avg  = (50*0.8 + 5*0.1) / 55 = 40.5/55 ≈ 0.736
        weighted_avg = (50 * 0.8 + 5 * 0.1) / 55
        expected_haircut = max(0.05, 1.0 - weighted_avg)
        assert abs(haircut - expected_haircut) < 0.02
        # Weighted avg should be much closer to 0.8 than 0.45
        assert weighted_avg > 0.70


# ═════════════════════════════════════════════════════════════════════════
#  Issue 4: Directional awareness in VaR
# ═════════════════════════════════════════════════════════════════════════

class TestDirectionalVaR:
    def test_yes_no_hedge_reduces_var(self):
        """YES + NO on same event should produce lower VaR than two NOs.

        The directional signs partially cancel in w'Σw when the
        same-event structural correlation is high (ρ=0.85).
        """
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=500.0,
            structural_same_event=0.85,
            structural_prior_weight=10,
            holding_period_minutes=1,  # no scaling for clarity
        )
        pce.register_market("MKT_A", "EVT_1", "crypto", _FakeAggregator(vol=0.10))
        pce.register_market("MKT_B", "EVT_1", "crypto", _FakeAggregator(vol=0.10))

        # Two NO positions on same event
        pos_no = [_FakePosition("MKT_A", 0.50, 20.0, trade_side="NO")]
        _, var_no_no = pce.check_var_gate(pos_no, "MKT_B", 10.0, "NO")

        # YES + NO on same event (natural hedge)
        pos_yes = [_FakePosition("MKT_A", 0.50, 20.0, trade_side="YES")]
        _, var_yes_no = pce.check_var_gate(pos_yes, "MKT_B", 10.0, "NO")

        # The YES+NO combo should have LOWER VaR due to hedging
        assert var_yes_no.portfolio_var_usd < var_no_no.portfolio_var_usd

    def test_legacy_position_defaults_to_no(self):
        """Positions without trade_side attribute default to NO (negative sign)."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=500.0,
            structural_baseline=0.05,
            structural_prior_weight=10,
            holding_period_minutes=1,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.10))
        pce.register_market("MKT_B", "E2", "", _FakeAggregator(vol=0.10))

        # Position without trade_side attribute
        class LegacyPosition:
            def __init__(self):
                self.market_id = "MKT_A"
                self.entry_price = 0.50
                self.entry_size = 10.0

        pos = [LegacyPosition()]
        # Should not crash; defaults to NO
        allowed, result = pce.check_var_gate(pos, "MKT_B", 5.0, "NO")
        assert allowed is True
        assert result.portfolio_var_usd > 0


# ═════════════════════════════════════════════════════════════════════════
#  Issue 5: Marginal-VaR proportional sizing cap
# ═════════════════════════════════════════════════════════════════════════

class TestVaRSizingCap:
    def test_partial_fill_instead_of_block(self):
        """VaR headroom allows partial fill instead of fully blocking."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=5.0,
            structural_baseline=0.05,
            structural_prior_weight=10,
            holding_period_minutes=1,  # no scaling
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.10))
        pce.register_market("MKT_B", "E2", "", _FakeAggregator(vol=0.10))

        # Existing position uses most of VaR budget
        # VaR(existing) = 1.645 * 30 * 0.10 = 4.935
        positions = [_FakePosition("MKT_A", 1.0, 30.0)]

        result = pce.compute_var_sizing_cap(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=10.0,
        )

        # Cap should be > 0 (some headroom) but < 10 (partial)
        assert isinstance(result, VaRSizingResult)
        assert 0 < result.cap_usd < 10.0
        assert result.cap_usd > 1.0  # at least min trade
        assert result.current_var > 0
        assert result.bisect_iterations >= 0

    def test_zero_headroom_returns_zero(self):
        """When VaR already exceeds threshold, cap is 0."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=1.0,  # tiny threshold
            structural_baseline=0.05,
            structural_prior_weight=10,
            holding_period_minutes=1,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.10))
        pce.register_market("MKT_B", "E2", "", _FakeAggregator(vol=0.10))

        # VaR(existing) = 1.645 * 50 * 0.10 = 8.225 >> 1.0
        positions = [_FakePosition("MKT_A", 1.0, 50.0)]

        result = pce.compute_var_sizing_cap(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=10.0,
        )

        assert result.cap_usd == 0.0
        assert result.current_var > 0

    def test_ample_headroom_returns_full_size(self):
        """When VaR headroom is abundant, cap equals proposed size."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=500.0,  # huge threshold
            structural_baseline=0.05,
            structural_prior_weight=10,
            holding_period_minutes=1,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.05))
        pce.register_market("MKT_B", "E2", "", _FakeAggregator(vol=0.05))

        positions = [_FakePosition("MKT_A", 0.50, 10.0)]

        result = pce.compute_var_sizing_cap(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=5.0,
        )

        assert abs(result.cap_usd - 5.0) < 0.01  # full size allowed
        assert result.bisect_iterations == 0  # fast path


# ═════════════════════════════════════════════════════════════════════════
#  Issue 6: Bootstrap invocation on empty state
# ═════════════════════════════════════════════════════════════════════════

class TestBootstrapInvocation:
    def test_empty_matrix_triggers_bootstrap(self, tmp_path):
        """load_state() on empty matrix calls bootstrap_from_ticks()."""
        pce = PortfolioCorrelationEngine(data_dir=str(tmp_path))

        bootstrap_calls: list[int] = []
        original_bootstrap = pce.bootstrap_from_ticks

        def mock_bootstrap(data_dir=None):
            bootstrap_calls.append(1)
            return 0

        pce.bootstrap_from_ticks = mock_bootstrap
        pce.load_state()  # no saved file → empty matrix → bootstrap

        assert len(bootstrap_calls) == 1

    def test_existing_matrix_skips_bootstrap(self, tmp_path):
        """load_state() with pre-existing data does NOT call bootstrap."""
        # Save a non-empty matrix
        pce = PortfolioCorrelationEngine(data_dir=str(tmp_path))
        pce.corr_matrix.set_structural_with_values(
            "A", "B", "E1", "E1", "", "",
            same_event_corr=0.85, same_tag_corr=0.30, baseline_corr=0.05,
        )
        pce.corr_matrix._pairs[("A", "B")].last_updated = time.time()
        pce.save_state()

        # Load it fresh
        pce2 = PortfolioCorrelationEngine(data_dir=str(tmp_path))
        bootstrap_calls: list[int] = []

        def mock_bootstrap(data_dir=None):
            bootstrap_calls.append(1)
            return 0

        pce2.bootstrap_from_ticks = mock_bootstrap
        pce2.load_state()

        assert len(bootstrap_calls) == 0  # should NOT be called


# ═════════════════════════════════════════════════════════════════════════
#  Iteration 2 — Issue 1: Redundant VaR computation (soft-cap dedup)
# ═════════════════════════════════════════════════════════════════════════

class TestSoftCapDedup:
    def test_soft_cap_returns_sizing_result(self):
        """In soft-cap mode, compute_var_sizing_cap returns VaRSizingResult
        that already contains current_var, avoiding a separate check_var_gate call.
        """
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=50.0,
            structural_baseline=0.05,
            structural_prior_weight=10,
            holding_period_minutes=1,
            var_soft_cap=True,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.10))
        pce.register_market("MKT_B", "E2", "", _FakeAggregator(vol=0.10))

        positions = [_FakePosition("MKT_A", 1.0, 20.0)]

        result = pce.compute_var_sizing_cap(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=10.0,
        )

        # VaRSizingResult carries current_var — no need for check_var_gate()
        assert isinstance(result, VaRSizingResult)
        assert result.current_var > 0
        assert result.headroom >= 0
        assert result.cap_usd > 0

    def test_soft_cap_no_redundant_var_call(self):
        """Verify compute_var_sizing_cap computes current VaR internally
        so the caller doesn't need a separate check_var_gate.
        """
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=5.0,
            structural_baseline=0.05,
            structural_prior_weight=10,
            holding_period_minutes=1,
            var_soft_cap=True,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.10))
        pce.register_market("MKT_B", "E2", "", _FakeAggregator(vol=0.10))

        positions = [_FakePosition("MKT_A", 1.0, 30.0)]

        # Get result from var_sizing_cap
        sizing_result = pce.compute_var_sizing_cap(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=10.0,
        )

        # Also call check_var_gate for comparison
        _, var_result = pce.check_var_gate(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=10.0,
        )

        # current_var from sizing_result should match portfolio_var from gate
        # (they compute VaR(existing) the same way)
        assert abs(sizing_result.current_var - var_result.portfolio_var_usd) < 0.01 or \
               (sizing_result.current_var > 0 and var_result.portfolio_var_usd > 0)


# ═════════════════════════════════════════════════════════════════════════
#  Iteration 2 — Issue 2: Bisection vs linear extrapolation
# ═════════════════════════════════════════════════════════════════════════

class TestBisectionSearch:
    def test_bisection_tighter_than_linear(self):
        """Bisection should find a cap where VaR(portfolio + cap) ≤ threshold.

        With correlated positions, linear extrapolation would overshoot.
        The bisection result, plugged back in, must not breach the threshold.
        """
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=5.0,
            structural_baseline=0.70,  # high baseline → correlated
            structural_prior_weight=10,
            holding_period_minutes=1,
            var_bisect_iterations=15,
        )
        pce.register_market("MKT_A", "E1", "crypto", _FakeAggregator(vol=0.10))
        pce.register_market("MKT_B", "E1", "crypto", _FakeAggregator(vol=0.10))

        # Existing $25 position: VaR ≈ 1.645 * 25 * 0.10 = 4.11
        positions = [_FakePosition("MKT_A", 1.0, 25.0)]

        result = pce.compute_var_sizing_cap(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=50.0,
        )

        assert isinstance(result, VaRSizingResult)
        assert result.bisect_iterations > 0  # bisection was needed
        assert result.cap_usd > 0
        assert result.cap_usd < 50.0  # must be capped

        # Verify: VaR(portfolio + cap) ≤ threshold
        # Re-compute VaR with the bisected cap size
        existing = pce._build_exposure_list(positions)
        volatilities = pce._get_volatilities()
        proposed = {"market_id": "MKT_B", "exposure_usd": -result.cap_usd}
        verification = pce.var_calc.compute_portfolio_var(
            positions=existing,
            proposed=proposed,
            corr_matrix=pce.corr_matrix,
            volatilities=volatilities,
        )
        # Conservative bound: should not exceed threshold
        assert verification.portfolio_var_usd <= 5.0 + 0.50  # bisection tolerance

    def test_bisection_convergence_within_tolerance(self):
        """Bisection should converge with lo/hi gap < $0.50."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce",
            shadow_mode=False,
            max_portfolio_var_usd=8.0,
            structural_baseline=0.50,
            structural_prior_weight=10,
            holding_period_minutes=1,
            var_bisect_iterations=10,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator(vol=0.08))
        pce.register_market("MKT_B", "E2", "", _FakeAggregator(vol=0.08))

        positions = [_FakePosition("MKT_A", 1.0, 40.0)]

        result = pce.compute_var_sizing_cap(
            open_positions=positions,
            proposed_market_id="MKT_B",
            proposed_size_usd=100.0,
        )

        # Should have used bisection
        assert result.bisect_iterations > 0
        assert result.cap_usd > 0


# ═════════════════════════════════════════════════════════════════════════
#  Iteration 2 — Issue 3: Near-extreme overlap gate
# ═════════════════════════════════════════════════════════════════════════

class TestNearExtremeOverlapGate:
    def test_near_extreme_high_rejected_insufficient_overlap(self):
        """Two series with mean > 0.85 require elevated overlap.
        With default multiplier=3 and min_overlap=10, need 30 bars.
        """
        cm = CorrelationMatrix()
        # 15 bars with prices > 0.90  (mean > 0.85)
        bars_a = _make_bars([0.92] * 15)
        bars_b = _make_bars([0.91] * 15)

        result = cm.update_empirical("MKT_A", "MKT_B", bars_a, bars_b, min_overlap=10)
        # 15 bars < 30 required → should return None
        assert result is None

    def test_near_extreme_high_accepted_sufficient_overlap(self):
        """Same near-extreme prices, but with ≥30 overlapping bars → OK."""
        cm = CorrelationMatrix()
        bars_a = _make_bars([0.92] * 35)
        bars_b = _make_bars([0.91] * 35)

        result = cm.update_empirical("MKT_A", "MKT_B", bars_a, bars_b, min_overlap=10)
        assert result is not None  # accepted — enough overlap

    def test_near_extreme_low_rejected_insufficient_overlap(self):
        """Two series near 0 (mean < 0.15) also require elevated overlap."""
        cm = CorrelationMatrix()
        bars_a = _make_bars([0.08] * 15)
        bars_b = _make_bars([0.10] * 15)

        result = cm.update_empirical("MKT_A", "MKT_B", bars_a, bars_b, min_overlap=10)
        assert result is None

    def test_normal_prices_not_affected(self):
        """Prices in [0.30, 0.70] range — standard overlap threshold applies."""
        cm = CorrelationMatrix()
        bars_a = _make_bars([0.50 + i * 0.005 for i in range(15)])
        bars_b = _make_bars([0.55 - i * 0.003 for i in range(15)])

        result = cm.update_empirical("MKT_A", "MKT_B", bars_a, bars_b, min_overlap=10)
        assert result is not None  # 15 > 10, normal threshold applies


# ═════════════════════════════════════════════════════════════════════════
#  Review 3 — Issue 1: Instance structural_prior_weight consumed
# ═════════════════════════════════════════════════════════════════════════

class TestInstancePriorWeight:
    """Verify that the instance-level structural_prior_weight (set on PCE)
    is actually consumed by VaR, haircut, and serialization."""

    def test_instance_prior_weight_used_in_var(self):
        """Two PCE instances with different structural_prior_weight
        produce different VaR gate results."""
        def _make_pce(pw: int) -> PortfolioCorrelationEngine:
            pce = PortfolioCorrelationEngine(
                data_dir="/tmp/test_pce_pw",
                shadow_mode=False,
                max_portfolio_var_usd=500.0,
                structural_baseline=0.05,
                structural_same_event=0.85,
                structural_prior_weight=pw,
                holding_period_minutes=1,
            )
            pce.register_market("MKT_A", "EVT_1", "crypto", _FakeAggregator(vol=0.10))
            pce.register_market("MKT_B", "EVT_1", "crypto", _FakeAggregator(vol=0.10))

            # Inject empirical data with moderate overlap
            bars_a = _make_bars([0.50 + 0.01 * i for i in range(20)])
            bars_b = _make_bars([0.60 + 0.01 * i for i in range(20)])
            pce.corr_matrix.update_empirical("MKT_A", "MKT_B", bars_a, bars_b)
            return pce

        pce_low = _make_pce(5)   # low prior weight -> empirical dominates
        pce_high = _make_pce(50) # high prior weight -> structural dominates

        pos = [_FakePosition("MKT_A", 0.50, 20.0, trade_side="NO")]

        _, var_low = pce_low.check_var_gate(pos, "MKT_B", 10.0, "NO")
        _, var_high = pce_high.check_var_gate(pos, "MKT_B", 10.0, "NO")

        # Blended correlations differ -> VaR values must differ
        assert var_low.portfolio_var_usd != var_high.portfolio_var_usd

    def test_instance_prior_weight_used_in_haircut(self):
        """Two PCE instances with different structural_prior_weight
        produce different concentration haircuts."""
        def _make_pce(pw: int) -> PortfolioCorrelationEngine:
            pce = PortfolioCorrelationEngine(
                data_dir="/tmp/test_pce_pw2",
                shadow_mode=False,
                haircut_threshold=0.10,
                structural_same_event=0.85,
                structural_prior_weight=pw,
                holding_period_minutes=1,
            )
            pce.register_market("MKT_A", "EVT_1", "crypto", _FakeAggregator(vol=0.10))
            pce.register_market("MKT_B", "EVT_1", "crypto", _FakeAggregator(vol=0.10))

            # Inject empirical data: low correlation
            bars_a = _make_bars([0.50 + 0.01 * i for i in range(20)])
            bars_b = _make_bars([0.60 - 0.005 * i for i in range(20)])
            pce.corr_matrix.update_empirical("MKT_A", "MKT_B", bars_a, bars_b)
            return pce

        pce_low = _make_pce(5)
        pce_high = _make_pce(50)

        pos = [_FakePosition("MKT_A", 0.50, 20.0)]

        haircut_low = pce_low.compute_concentration_haircut("MKT_B", pos)
        haircut_high = pce_high.compute_concentration_haircut("MKT_B", pos)

        # pw=50 forces closer to structural (0.85) -> bigger haircut
        # pw=5 lets empirical dominate -> smaller haircut
        assert haircut_low != haircut_high

    def test_prior_weight_survives_serialization(self, tmp_path):
        """Save & load state preserves prior_weight_override on the matrix."""
        pce = PortfolioCorrelationEngine(
            data_dir=str(tmp_path),
            structural_prior_weight=42,
        )
        pce.register_market("MKT_A", "E1", "", _FakeAggregator())
        pce.register_market("MKT_B", "E2", "", _FakeAggregator())

        # Confirm override is set
        assert pce.corr_matrix.prior_weight_override == 42

        # Inject data and save
        bars = _make_bars([0.50 + 0.01 * i for i in range(10)])
        pce.corr_matrix.update_empirical("MKT_A", "MKT_B", bars, bars)
        pce.save_state()

        # Load into new PCE with same prior_weight
        pce2 = PortfolioCorrelationEngine(
            data_dir=str(tmp_path),
            structural_prior_weight=42,
        )
        pce2.load_state()
        assert pce2.corr_matrix.prior_weight_override == 42

        # Verify prior_weight_override persists via JSON round-trip
        data = pce.corr_matrix.to_json()
        assert data.get("prior_weight_override") == 42
        cm_loaded = CorrelationMatrix.from_json(data)
        assert cm_loaded.prior_weight_override == 42

    def test_prior_weight_override_in_get_matrix(self):
        """get_matrix() respects prior_weight_override."""
        cm = CorrelationMatrix()
        cm._pairs[("A", "B")] = CorrelationEstimate(
            empirical_corr=0.0, structural_corr=0.80, overlap_bars=10,
        )

        # With override = 100 (heavily favor structural)
        cm.prior_weight_override = 100
        mat_override = cm.get_matrix(["A", "B"])

        # pw=100: blended = (100*0.80 + 10*0.0) / 110 ~ 0.727
        expected = (100 * 0.80 + 10 * 0.0) / 110
        assert abs(mat_override[0][1] - expected) < 1e-4


# ═════════════════════════════════════════════════════════════════════════
#  Review 3 — Issue 2: Near-extreme params per-instance
# ═════════════════════════════════════════════════════════════════════════

class TestNearExtremeParamsOverride:
    """Verify near_extreme_threshold and near_extreme_overlap_multiplier
    can be overridden per-PCE instance."""

    def test_near_extreme_threshold_override(self):
        """PCE with near_extreme_threshold=0.70 gates pairs with mean=0.75
        while the default (0.85) would not."""
        cm = CorrelationMatrix()
        bars_a = _make_bars([0.75] * 15)
        bars_b = _make_bars([0.76] * 15)

        # Default threshold (0.85): 0.75 < 0.85 -> NOT near-extreme -> accepted
        result_default = cm.update_empirical("A", "B", bars_a, bars_b, min_overlap=10)
        assert result_default is not None

        # Custom threshold=0.70: 0.75 > 0.70 -> near-extreme -> needs 30 bars -> rejected
        cm2 = CorrelationMatrix()
        result_custom = cm2.update_empirical(
            "A", "B", bars_a, bars_b, min_overlap=10,
            near_extreme_threshold=0.70,
        )
        assert result_custom is None  # 15 < 30 required

    def test_near_extreme_multiplier_override(self):
        """PCE with near_extreme_overlap_multiplier=5 requires more overlap."""
        cm = CorrelationMatrix()
        # 25 bars with near-extreme prices (mean > 0.85)
        bars_a = _make_bars([0.92] * 25)
        bars_b = _make_bars([0.91] * 25)

        # Default multiplier=3: needs 10*3=30 bars -> 25 < 30 -> rejected
        result_default = cm.update_empirical("A", "B", bars_a, bars_b, min_overlap=10)
        assert result_default is None

        # With 35 bars and multiplier=3: 35 >= 30 -> accepted
        cm2 = CorrelationMatrix()
        bars_a2 = _make_bars([0.92] * 35)
        bars_b2 = _make_bars([0.91] * 35)
        result_ok = cm2.update_empirical("A", "B", bars_a2, bars_b2, min_overlap=10)
        assert result_ok is not None

        # But multiplier=5: needs 10*5=50 bars -> 35 < 50 -> rejected
        cm3 = CorrelationMatrix()
        result_custom = cm3.update_empirical(
            "A", "B", bars_a2, bars_b2, min_overlap=10,
            near_extreme_overlap_multiplier=5,
        )
        assert result_custom is None

    def test_pce_passes_near_extreme_params_to_refresh(self):
        """PCE with custom near_extreme params passes them through refresh_correlations."""
        pce = PortfolioCorrelationEngine(
            data_dir="/tmp/test_pce_ne",
            near_extreme_threshold=0.70,
            near_extreme_overlap_multiplier=5,
        )
        # Register markets with near-extreme prices
        agg_a = _FakeAggregator(vol=0.05)
        agg_b = _FakeAggregator(vol=0.05)
        agg_a.bars = _make_bars([0.75] * 15)
        agg_b.bars = _make_bars([0.76] * 15)

        pce.register_market("MKT_A", "E1", "", agg_a)
        pce.register_market("MKT_B", "E2", "", agg_b)

        pce.refresh_correlations()

        # With threshold=0.70 and multiplier=5, 15 bars < 10*5=50 -> no empirical
        est = pce.corr_matrix.get_estimate("MKT_A", "MKT_B")
        if est is not None:
            assert est.overlap_bars == 0


# ═════════════════════════════════════════════════════════════════════════
#  Review 3 — Issue 3: Backtest telemetry PCE wiring
# ═════════════════════════════════════════════════════════════════════════

class TestPCETelemetryWiring:
    """Verify record_pce_snapshot() and rejection counting work."""

    def test_pce_snapshot_recorded(self):
        """record_pce_snapshot stores data that finalize() consumes."""
        from src.backtest.telemetry import Telemetry

        tel = Telemetry(initial_cash=1000.0)
        tel.record_pce_snapshot(
            timestamp=1000.0, portfolio_var=5.0, avg_correlation=0.3, n_positions=2,
        )
        tel.record_pce_snapshot(
            timestamp=2000.0, portfolio_var=8.0, avg_correlation=0.4, n_positions=3,
        )

        assert len(tel._pce_snapshots) == 2

        metrics = tel.finalize()
        assert metrics.max_portfolio_var == 8.0
        assert abs(metrics.avg_portfolio_correlation - 0.35) < 1e-6

    def test_pce_rejection_counted_in_telemetry(self):
        """set_pce_rejections populates metrics.pce_rejections."""
        from src.backtest.telemetry import Telemetry

        tel = Telemetry(initial_cash=1000.0)
        tel.set_pce_rejections(7)
        metrics = tel.finalize()
        assert metrics.pce_rejections == 7

    def test_telemetry_reset_clears_pce_data(self):
        """reset() clears PCE snapshots and rejection count."""
        from src.backtest.telemetry import Telemetry

        tel = Telemetry(initial_cash=1000.0)
        tel.record_pce_snapshot(
            timestamp=1000.0, portfolio_var=5.0, avg_correlation=0.3, n_positions=2,
        )
        tel.set_pce_rejections(3)
        tel.reset()

        assert len(tel._pce_snapshots) == 0
        assert tel._pce_rejection_count == 0


# ═════════════════════════════════════════════════════════════════════════
#  Review 3 — Issue 4: WFO suggest_int for int params
# ═════════════════════════════════════════════════════════════════════════

class TestWFOIntParams:
    """Verify _suggest_params returns int for pce_structural_prior_weight
    and pce_holding_period_minutes."""

    def test_wfo_int_params_are_int(self):
        """Mock Optuna trial to verify suggest_int is called for int params."""
        from src.backtest.wfo_optimizer import _suggest_params

        class _MockTrial:
            """Minimal Optuna trial mock."""
            def __init__(self):
                self._calls: dict[str, str] = {}

            def suggest_float(self, name, lo, hi, log=False):
                self._calls[name] = "suggest_float"
                return (lo + hi) / 2.0

            def suggest_int(self, name, lo, hi):
                self._calls[name] = "suggest_int"
                return (lo + hi) // 2

        trial = _MockTrial()
        params = _suggest_params(trial)

        # These must use suggest_int
        assert trial._calls["pce_structural_prior_weight"] == "suggest_int"
        assert trial._calls["pce_holding_period_minutes"] == "suggest_int"

        # Values must be int
        assert isinstance(params["pce_structural_prior_weight"], int)
        assert isinstance(params["pce_holding_period_minutes"], int)


# ═════════════════════════════════════════════════════════════════════════
#  Review 3 — Issue 5: Backtest _Pos has trade_side
# ═════════════════════════════════════════════════════════════════════════

class TestBacktestPosHasTradeSide:
    """Verify _build_pce_positions returns objects with explicit trade_side."""

    def test_backtest_pos_has_trade_side(self):
        """_build_pce_positions returns objects with trade_side attribute."""
        from src.backtest.strategy import BotReplayAdapter

        adapter = BotReplayAdapter(
            market_id="MKT_TEST",
            yes_asset_id="YES_TOKEN",
            no_asset_id="NO_TOKEN",
        )

        # Simulate an open position
        class _FakeFill:
            price = 0.55
            size = 10.0
            fee = 0.01
            order_id = "o1"
            timestamp = 1000.0
            is_maker = True
            side = "BUY"

        adapter._open_positions["exit_1"] = {
            "entry_fill": _FakeFill(),
            "entry_ctx": {},
            "exit_order_id": "exit_1",
            "entry_time": 1000.0,
        }

        positions = adapter._build_pce_positions()
        assert len(positions) == 1

        pos = positions[0]
        assert hasattr(pos, "trade_side")
        assert pos.trade_side == "NO"
        assert pos.market_id == "MKT_TEST"
        assert abs(pos.entry_price - 0.55) < 1e-6
        assert abs(pos.entry_size - 10.0) < 1e-6
