"""
Tests for SI-6: Meta-Strategy Hybrid Controller.
"""

from __future__ import annotations

import pytest

from src.signals.signal_framework import MetaDecision, MetaStrategyController


class TestMetaStrategyController:
    """Verify regime-weighted signal scaling and vetoing."""

    def setup_method(self):
        self.ctrl = MetaStrategyController()

    # ── Deep mean-reversion regime (score > 0.8) ─────────────────────

    def test_deep_mr_panic_boost(self):
        d = self.ctrl.evaluate("panic", 0.9)
        assert d.weight == 1.5
        assert not d.vetoed

    def test_deep_mr_drift_boost(self):
        d = self.ctrl.evaluate("drift", 0.85)
        assert d.weight == 1.5
        assert not d.vetoed

    def test_deep_mr_rpe_halved(self):
        d = self.ctrl.evaluate("rpe", 0.9)
        assert d.weight == 0.5
        assert not d.vetoed

    # ── Trending regime (score < 0.3) ────────────────────────────────

    def test_trending_panic_vetoed(self):
        d = self.ctrl.evaluate("panic", 0.1)
        assert d.vetoed
        assert d.veto_reason == "regime_trend_veto"
        assert d.weight == 0.0

    def test_trending_drift_vetoed(self):
        d = self.ctrl.evaluate("drift", 0.2)
        assert d.vetoed
        assert d.veto_reason == "regime_trend_veto"

    def test_trending_rpe_full(self):
        d = self.ctrl.evaluate("rpe", 0.1)
        assert d.weight == 1.0
        assert not d.vetoed

    # ── Neutral regime (0.3 ≤ score ≤ 0.8) ──────────────────────────

    def test_neutral_panic_passthrough(self):
        d = self.ctrl.evaluate("panic", 0.5)
        assert d.weight == 1.0
        assert not d.vetoed

    def test_neutral_rpe_passthrough(self):
        d = self.ctrl.evaluate("rpe", 0.5)
        assert d.weight == 1.0
        assert not d.vetoed

    # ── Boundary values ──────────────────────────────────────────────

    def test_boundary_exactly_0_3_is_neutral(self):
        d = self.ctrl.evaluate("panic", 0.3)
        assert not d.vetoed
        assert d.weight == 1.0

    def test_boundary_exactly_0_8_is_deep_mr(self):
        d = self.ctrl.evaluate("panic", 0.8)
        assert d.weight == 1.5

    def test_regime_score_preserved(self):
        d = self.ctrl.evaluate("panic", 0.42)
        assert d.regime_score == 0.42
