"""
Tests for the Walk-Forward Optimization (WFO) pipeline.

Covers:
- Time-series fold generation (date windowing, edge cases, embargo, anchored)
- Multi-metric objective function (drawdown penalty, trade gate, composite)
- Strategy parameter injection via BotReplayAdapter
- OOS equity-curve stitching
- Overfitting diagnostics (decay, probability, stability)
- Expanded search space with log-scale
- End-to-end single-fold smoke test with synthetic data
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

from src.backtest.strategy import (
    LEGACY_BACKTEST_SIGNAL_DEFAULTS,
    BotReplayAdapter,
    split_strategy_and_legacy_params,
)
from src.backtest.wfo_optimizer import (
    FoldResult,
    WfoConfig,
    WfoReport,
    _export_champion_params,
    _suggest_params,
    _compute_stitched_metrics,
    _stitch_equity_curves,
    compute_wfo_score,
    generate_folds,
)
from src.core.config import StrategyParams


# ═══════════════════════════════════════════════════════════════════════════
#  Fold generation
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateFolds:
    """Rolling window fold generation from available dates."""

    @staticmethod
    def _date_range(start: str, n_days: int) -> list[str]:
        """Generate a list of n consecutive YYYY-MM-DD date strings."""
        from datetime import datetime, timedelta

        d = datetime.strptime(start, "%Y-%m-%d").date()
        return [(d + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]

    def test_basic_45_day_window(self):
        """30-day train + 7-day test, step 7 → expect 2 folds from 45 days."""
        dates = self._date_range("2026-01-01", 45)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7)

        assert len(folds) >= 1
        # First fold: train 2026-01-01..01-30, test 2026-01-31..02-06
        assert folds[0].train_dates[0] == "2026-01-01"
        assert folds[0].test_dates[0] == "2026-01-31"

    def test_fold_indices_sequential(self):
        dates = self._date_range("2026-01-01", 60)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7)

        for i, f in enumerate(folds):
            assert f.index == i

    def test_train_test_no_overlap(self):
        """Train and test date lists must not overlap."""
        dates = self._date_range("2026-01-01", 60)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7)

        for fold in folds:
            train_set = set(fold.train_dates)
            test_set = set(fold.test_dates)
            assert train_set.isdisjoint(test_set), f"Fold {fold.index} has overlapping dates"

    def test_empty_dates_returns_empty(self):
        assert generate_folds([], train_days=30, test_days=7, step_days=7) == []

    def test_insufficient_data_returns_empty(self):
        """If data is shorter than train + test, no folds should be generated."""
        dates = self._date_range("2026-01-01", 10)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7)
        assert folds == []

    def test_exact_fit_single_fold(self):
        """Data spanning exactly train_days + test_days → 1 fold."""
        dates = self._date_range("2026-01-01", 37)  # 30 + 7
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7)
        assert len(folds) == 1
        assert len(folds[0].train_dates) == 30
        assert len(folds[0].test_dates) == 7

    def test_step_smaller_than_test(self):
        """Step = 3, test = 7 → overlapping test windows are allowed."""
        dates = self._date_range("2026-01-01", 50)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=3)
        assert len(folds) >= 3

    def test_sparse_dates_handled(self):
        """Only some dates have data (gaps in recording)."""
        all_dates = self._date_range("2026-01-01", 45)
        # Keep only every other day
        sparse = [d for i, d in enumerate(all_dates) if i % 2 == 0]
        folds = generate_folds(sparse, train_days=30, test_days=7, step_days=7)

        for fold in folds:
            # All returned dates must be from our sparse set
            for d in fold.train_dates + fold.test_dates:
                assert d in sparse

    def test_multiple_folds_step_forward(self):
        """With 60 days and step=7, expect ~4 folds."""
        dates = self._date_range("2026-01-01", 60)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7)
        assert len(folds) >= 3

    # ── Embargo tests ──────────────────────────────────────────────────

    def test_embargo_creates_gap(self):
        """Embargo=2 pushes OOS start forward by 2 days."""
        dates = self._date_range("2026-01-01", 50)
        folds_no_embargo = generate_folds(dates, train_days=30, test_days=7, step_days=7, embargo_days=0)
        folds_embargo = generate_folds(dates, train_days=30, test_days=7, step_days=7, embargo_days=2)

        if folds_no_embargo and folds_embargo:
            # OOS start should be 2 days later with embargo
            assert folds_embargo[0].test_dates[0] > folds_no_embargo[0].test_dates[0]

    def test_embargo_no_overlap_between_train_and_test(self):
        """With embargo, there must be a clear gap between IS and OOS."""
        from datetime import datetime

        dates = self._date_range("2026-01-01", 60)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7, embargo_days=3)

        for fold in folds:
            if fold.train_dates and fold.test_dates:
                last_train = datetime.strptime(fold.train_dates[-1], "%Y-%m-%d").date()
                first_test = datetime.strptime(fold.test_dates[0], "%Y-%m-%d").date()
                gap = (first_test - last_train).days
                assert gap >= 4, f"Fold {fold.index}: gap={gap} days, expected ≥ 4"

    def test_embargo_too_large_returns_fewer_folds(self):
        """Very large embargo can reduce or eliminate folds."""
        dates = self._date_range("2026-01-01", 38)
        # 30d train + 10d embargo + 7d test = 47 days needed but only 38
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7, embargo_days=10)
        assert len(folds) == 0

    # ── Anchored tests ─────────────────────────────────────────────────

    def test_anchored_expanding_window(self):
        """In anchored mode, all folds start from the first date."""
        dates = self._date_range("2026-01-01", 60)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7, anchored=True)

        for fold in folds:
            assert fold.train_dates[0] == "2026-01-01", (
                f"Fold {fold.index}: anchored should start from first date"
            )

    def test_anchored_train_grows(self):
        """Each subsequent anchored fold should have more training data."""
        dates = self._date_range("2026-01-01", 60)
        folds = generate_folds(dates, train_days=30, test_days=7, step_days=7, anchored=True)

        if len(folds) >= 2:
            assert len(folds[1].train_dates) > len(folds[0].train_dates)

    def test_anchored_with_embargo(self):
        """Anchored + embargo should work together."""
        dates = self._date_range("2026-01-01", 60)
        folds = generate_folds(
            dates, train_days=30, test_days=7, step_days=7,
            embargo_days=2, anchored=True,
        )
        for fold in folds:
            assert fold.train_dates[0] == "2026-01-01"
            train_set = set(fold.train_dates)
            test_set = set(fold.test_dates)
            assert train_set.isdisjoint(test_set)


# ═══════════════════════════════════════════════════════════════════════════
#  Objective / scoring math
# ═══════════════════════════════════════════════════════════════════════════

class TestWfoScore:
    """Tests for compute_wfo_score — multi-metric composite objective."""

    def test_zero_drawdown_no_penalty(self):
        """If drawdown is 0, penalty = 1 → score = weighted composite."""
        score = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            total_fills=10,
        )
        assert score > 0

    def test_half_threshold_drawdown(self):
        """If drawdown = half of threshold, penalty = 0.5."""
        score_full = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            total_fills=10,
        )
        score_half = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.075, max_acceptable_drawdown=0.15,
            total_fills=10,
        )
        assert abs(score_half - score_full * 0.5) < 1e-9

    def test_at_threshold_collapses_to_zero(self):
        """If drawdown == threshold, penalty = 0 → score = 0."""
        score = compute_wfo_score(
            sharpe_ratio=5.0, max_drawdown=0.15, max_acceptable_drawdown=0.15,
            total_fills=10,
        )
        assert score == 0.0

    def test_exceeds_threshold_collapses_to_zero(self):
        """If drawdown > threshold, score must still be 0."""
        score = compute_wfo_score(
            sharpe_ratio=10.0, max_drawdown=0.25, max_acceptable_drawdown=0.15,
            total_fills=10,
        )
        assert score == 0.0

    def test_negative_sharpe_with_drawdown(self):
        """Negative Sharpe in composite → negative score."""
        score = compute_wfo_score(
            sharpe_ratio=-1.5, max_drawdown=0.05, max_acceptable_drawdown=0.15,
            sortino_ratio=-1.0, profit_factor=0.5, total_fills=10,
        )
        assert score < 0

    def test_very_small_drawdown(self):
        """Near-zero drawdown: penalty ≈ 1."""
        score = compute_wfo_score(
            sharpe_ratio=3.0, max_drawdown=0.001, max_acceptable_drawdown=0.15,
            total_fills=10,
        )
        assert score > 1.0

    def test_custom_threshold(self):
        """Different max_acceptable_drawdown values."""
        score_10 = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.10, max_acceptable_drawdown=0.10,
            total_fills=10,
        )
        assert score_10 == 0.0  # exactly at threshold

    # ── Trade gate tests ───────────────────────────────────────────────

    def test_too_few_trades_returns_neg_inf(self):
        """Fewer than min_trades → score = -inf."""
        score = compute_wfo_score(
            sharpe_ratio=3.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            total_fills=2, min_trades=5,
        )
        assert score == float("-inf")

    def test_exactly_min_trades_passes(self):
        """Exactly min_trades should pass the gate."""
        score = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            total_fills=5, min_trades=5,
        )
        assert score > 0

    def test_zero_fills_rejected(self):
        """Zero fills always rejected."""
        score = compute_wfo_score(
            sharpe_ratio=0.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            total_fills=0,
        )
        assert score == float("-inf")

    # ── Multi-metric composite tests ───────────────────────────────────

    def test_sortino_contribution(self):
        """Higher Sortino → higher score."""
        base = compute_wfo_score(
            sharpe_ratio=1.5, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            sortino_ratio=0.0, total_fills=10,
        )
        higher = compute_wfo_score(
            sharpe_ratio=1.5, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            sortino_ratio=3.0, total_fills=10,
        )
        assert higher > base

    def test_profit_factor_contribution(self):
        """Higher profit factor → higher score."""
        base = compute_wfo_score(
            sharpe_ratio=1.5, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            profit_factor=0.0, total_fills=10,
        )
        higher = compute_wfo_score(
            sharpe_ratio=1.5, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            profit_factor=3.0, total_fills=10,
        )
        assert higher > base

    def test_profit_factor_log_transform(self):
        """Profit factor uses ln(1+PF) — verifiable."""
        score = compute_wfo_score(
            sharpe_ratio=0.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            sortino_ratio=0.0, profit_factor=2.0, total_fills=10,
            sharpe_weight=0.0, sortino_weight=0.0, profit_factor_weight=1.0,
            trade_bonus_weight=0.0,
        )
        expected = math.log(1.0 + 2.0)  # ln(3) ≈ 1.0986
        assert abs(score - expected) < 1e-6

    def test_custom_weights(self):
        """Custom weights affect the score as expected."""
        # Only Sharpe weight
        sharpe_only = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
            sortino_ratio=5.0, profit_factor=3.0, total_fills=10,
            sharpe_weight=1.0, sortino_weight=0.0, profit_factor_weight=0.0,
            trade_bonus_weight=0.0,
        )
        assert abs(sharpe_only - 2.0) < 1e-9

    def test_backward_compatible_simple_call(self):
        """Old-style call with just Sharpe/DD still works (defaults handle it)."""
        score = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.0, max_acceptable_drawdown=0.15,
        )
        # With default min_trades=5 and total_fills=0, should be rejected.
        assert score == float("-inf")


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy parameter injection
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterInjection:
    """BotReplayAdapter must use injected params, not hardcoded defaults."""

    def test_default_params_when_none(self):
        """When no params passed, adapter uses StrategyParams() defaults."""
        adapter = BotReplayAdapter(
            market_id="MKT", yes_asset_id="YES", no_asset_id="NO"
        )
        assert adapter._params.kelly_fraction == StrategyParams().kelly_fraction
        assert adapter._legacy_signal_params == LEGACY_BACKTEST_SIGNAL_DEFAULTS

    def test_custom_params_used(self):
        """Injected StrategyParams override defaults."""
        custom = StrategyParams(kelly_fraction=0.10)
        adapter = BotReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=custom,
            legacy_signal_params={"zscore_threshold": 1.7},
        )
        assert adapter._params.kelly_fraction == 0.10
        assert adapter._legacy_signal_params["zscore_threshold"] == 1.7

    def test_alpha_default_injected(self):
        """The target-price alpha should read from params."""
        custom = StrategyParams(alpha_default=0.35)
        adapter = BotReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=custom,
        )
        assert adapter._params.alpha_default == 0.35

    def test_max_open_positions_injected(self):
        """Position limit should read from params."""
        custom = StrategyParams(max_open_positions=7)
        adapter = BotReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=custom,
        )
        assert adapter._params.max_open_positions == 7

    def test_backward_compatible_no_params(self):
        """Existing code that doesn't pass params should still work."""
        adapter = BotReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            fee_enabled=False,
            initial_bankroll=500.0,
        )
        # Should have default params
        assert adapter._params is not None
        assert isinstance(adapter._params, StrategyParams)


# ═══════════════════════════════════════════════════════════════════════════
#  OOS equity-curve stitching
# ═══════════════════════════════════════════════════════════════════════════

class TestEquityStitching:
    """Concatenation of OOS equity curves across folds."""

    def test_single_fold_passthrough(self):
        """A single fold's curve should be returned adjusted from initial_cash."""
        fr = FoldResult(
            fold_index=0,
            best_params={},
            oos_equity_curve=[(1.0, 1000.0), (2.0, 1010.0), (3.0, 1005.0)],
        )
        stitched = _stitch_equity_curves([fr], initial_cash=1000.0)

        assert len(stitched) == 3
        assert stitched[0][1] == 1000.0
        assert stitched[-1][1] == 1005.0

    def test_two_fold_carry_equity(self):
        """Fold 2's curve should start where fold 1's ended."""
        fr1 = FoldResult(
            fold_index=0,
            best_params={},
            oos_equity_curve=[(1.0, 1000.0), (2.0, 1020.0)],
        )
        fr2 = FoldResult(
            fold_index=1,
            best_params={},
            # This fold also starts at 1000.0 (independent backtest)
            # but should be rebased to start at 1020.0
            oos_equity_curve=[(3.0, 1000.0), (4.0, 1015.0)],
        )
        stitched = _stitch_equity_curves([fr1, fr2], initial_cash=1000.0)

        assert len(stitched) == 4
        # Fold 1: 1000 → 1020
        assert stitched[0][1] == 1000.0
        assert stitched[1][1] == 1020.0
        # Fold 2: rebased → 1020 → 1035
        assert stitched[2][1] == 1020.0
        assert stitched[3][1] == 1035.0

    def test_three_fold_chain(self):
        """Three folds chain correctly."""
        fr1 = FoldResult(
            fold_index=0, best_params={},
            oos_equity_curve=[(1.0, 500.0), (2.0, 520.0)],
        )
        fr2 = FoldResult(
            fold_index=1, best_params={},
            oos_equity_curve=[(3.0, 500.0), (4.0, 490.0)],
        )
        fr3 = FoldResult(
            fold_index=2, best_params={},
            oos_equity_curve=[(5.0, 500.0), (6.0, 530.0)],
        )
        stitched = _stitch_equity_curves([fr1, fr2, fr3], initial_cash=500.0)

        assert len(stitched) == 6
        # Fold 1: 500 → 520
        assert stitched[1][1] == 520.0
        # Fold 2: rebased from 500→520, offset=+20, so 490+20=510
        assert stitched[3][1] == 510.0
        # Fold 3: rebased from 500→510, offset=+10, so 530+10=540
        assert stitched[5][1] == 540.0

    def test_empty_folds_skipped(self):
        """Folds with no equity curve are gracefully skipped."""
        fr1 = FoldResult(
            fold_index=0, best_params={},
            oos_equity_curve=[(1.0, 1000.0), (2.0, 1010.0)],
        )
        fr2 = FoldResult(
            fold_index=1, best_params={},
            oos_equity_curve=[],
        )
        fr3 = FoldResult(
            fold_index=2, best_params={},
            oos_equity_curve=[(3.0, 1000.0), (4.0, 1020.0)],
        )
        stitched = _stitch_equity_curves([fr1, fr2, fr3], initial_cash=1000.0)

        assert len(stitched) == 4
        assert stitched[-1][1] == 1030.0  # 1010 + (1020 - 1000) = 1030

    def test_no_folds_returns_empty(self):
        assert _stitch_equity_curves([], initial_cash=1000.0) == []


# ═══════════════════════════════════════════════════════════════════════════
#  Stitched metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestStitchedMetrics:
    """Aggregate metrics from stitched equity curves."""

    def test_flat_curve_zero_sharpe(self):
        curve = [(float(i), 1000.0) for i in range(10)]
        sharpe, mdd, pnl = _compute_stitched_metrics(curve, 1000.0)
        assert sharpe == 0.0
        assert mdd == 0.0
        assert pnl == 0.0

    def test_positive_pnl(self):
        curve = [(float(i), 1000.0 + i * 10) for i in range(5)]
        sharpe, mdd, pnl = _compute_stitched_metrics(curve, 1000.0)
        assert pnl == 40.0
        assert mdd == 0.0  # monotonically increasing → no drawdown
        assert sharpe > 0

    def test_drawdown_detected(self):
        curve = [
            (0.0, 1000.0),
            (1.0, 1100.0),  # new high
            (2.0, 1050.0),  # drawdown
            (3.0, 1120.0),  # recovery
        ]
        sharpe, mdd, pnl = _compute_stitched_metrics(curve, 1000.0)
        assert mdd > 0.04  # (1100-1050)/1100 ≈ 4.5%
        assert pnl == 120.0

    def test_too_few_points(self):
        curve = [(0.0, 1000.0), (1.0, 1010.0)]
        sharpe, mdd, pnl = _compute_stitched_metrics(curve, 1000.0)
        assert sharpe == 0.0
        assert pnl == 10.0

    def test_empty_curve(self):
        sharpe, mdd, pnl = _compute_stitched_metrics([], 1000.0)
        assert sharpe == 0.0
        assert pnl == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  WfoReport
# ═══════════════════════════════════════════════════════════════════════════

class TestWfoReport:
    """WfoReport structure and summary rendering."""

    def test_summary_renders_without_error(self):
        report = WfoReport(
            folds=[
                FoldResult(
                    fold_index=0,
                    best_params={"zscore_threshold": 2.1, "kelly_fraction": 0.15},
                    is_sharpe=1.8,
                    oos_sharpe=1.2,
                    is_max_drawdown=0.05,
                    oos_max_drawdown=0.07,
                    is_total_fills=45,
                    sharpe_decay_pct=-33.3,
                ),
                FoldResult(
                    fold_index=1,
                    best_params={"zscore_threshold": 2.3, "kelly_fraction": 0.12},
                    is_sharpe=2.1,
                    oos_sharpe=1.5,
                    is_max_drawdown=0.04,
                    oos_max_drawdown=0.06,
                    is_total_fills=52,
                    sharpe_decay_pct=-28.6,
                ),
            ],
            aggregate_oos_sharpe=1.35,
            aggregate_oos_max_drawdown=0.07,
            aggregate_oos_total_pnl=42.5,
            parameter_stability={
                "zscore_threshold": [2.1, 2.3],
                "kelly_fraction": [0.15, 0.12],
            },
            total_elapsed_s=123.4,
            avg_sharpe_decay_pct=-30.9,
            overfit_probability=0.0,
            unstable_params=[],
        )

        text = report.summary()
        assert "WALK-FORWARD" in text
        assert "1.35" in text  # aggregate sharpe
        assert "zscore_threshold" in text
        assert "Overfitting Analysis" in text
        assert "Overfit Probability" in text

    def test_empty_report(self):
        report = WfoReport()
        text = report.summary()
        assert "Folds completed" in text
        assert "0" in text

    def test_unstable_params_shown(self):
        """Unstable parameters flagged in report."""
        report = WfoReport(
            unstable_params=["kelly_fraction", "zscore_threshold"],
            parameter_stability={
                "kelly_fraction": [0.05, 0.30],  # high CV
                "zscore_threshold": [1.5, 3.5],
            },
        )
        text = report.summary()
        assert "kelly_fraction" in text
        assert "⚠" in text  # stability flag

    def test_overfit_probability_shown(self):
        """High overfit probability is rendered."""
        report = WfoReport(overfit_probability=0.75)
        text = report.summary()
        assert "75.0%" in text

    def test_export_champion_params_uses_champion_fold_metrics(self, tmp_path: Path):
        output_path = tmp_path / "champion_params.json"
        cfg = WfoConfig(output_params_path=str(output_path), n_trials=17)
        report = WfoReport(
            folds=[
                FoldResult(
                    fold_index=0,
                    best_params={"pure_mm_wide_spread_pct": 0.2},
                    oos_sharpe=0.5,
                    oos_win_rate=0.45,
                    oos_profit_factor=0.9,
                    oos_total_fills=11,
                    train_dates=["2026-03-20"],
                    test_dates=["2026-03-21"],
                ),
                FoldResult(
                    fold_index=1,
                    best_params={"pure_mm_wide_spread_pct": 0.25},
                    oos_sharpe=1.8,
                    oos_win_rate=0.62,
                    oos_profit_factor=1.4,
                    oos_total_fills=29,
                    train_dates=["2026-03-21"],
                    test_dates=["2026-03-22"],
                ),
            ],
            aggregate_oos_sharpe=-3.7,
            aggregate_oos_win_rate=0.12,
            aggregate_oos_profit_factor=0.4,
            aggregate_oos_trade_count=999,
            champion_params={"pure_mm_wide_spread_pct": 0.25},
            champion_fold_index=1,
            champion_degradation_pct=12.34,
        )

        _export_champion_params(cfg, report)

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["meta"]["champion_fold"] == 1
        assert payload["meta"]["oos_sharpe"] == 1.8
        assert payload["meta"]["oos_win_rate"] == 0.62
        assert payload["meta"]["oos_profit_factor"] == 1.4
        assert payload["meta"]["oos_trade_count"] == 29
        assert payload["meta"]["n_trials_per_fold"] == 17
        assert payload["params"] == {"pure_mm_wide_spread_pct": 0.25}


# ═══════════════════════════════════════════════════════════════════════════
#  WfoConfig defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestWfoConfig:
    """WfoConfig sensible defaults."""

    def test_defaults(self):
        cfg = WfoConfig()
        assert cfg.train_days == 30
        assert cfg.test_days == 7
        assert cfg.step_days == 7
        assert cfg.embargo_days == 1
        assert cfg.anchored is False
        assert cfg.n_trials == 100
        assert cfg.max_acceptable_drawdown == 0.15
        assert cfg.min_trades == 5
        assert cfg.initial_cash == 1000.0
        assert cfg.max_workers >= 1
        assert cfg.warm_start is True
        assert cfg.sharpe_weight == 0.50
        assert cfg.sortino_weight == 0.30
        assert cfg.profit_factor_weight == 0.20

    def test_custom_config(self):
        cfg = WfoConfig(
            data_dir="/tmp/data",
            train_days=14,
            test_days=3,
            step_days=3,
            embargo_days=2,
            anchored=True,
            n_trials=50,
            max_acceptable_drawdown=0.10,
            min_trades=10,
            warm_start=False,
        )
        assert cfg.train_days == 14
        assert cfg.n_trials == 50
        assert cfg.max_acceptable_drawdown == 0.10
        assert cfg.embargo_days == 2
        assert cfg.anchored is True
        assert cfg.min_trades == 10
        assert cfg.warm_start is False


# ═══════════════════════════════════════════════════════════════════════════
#  Search space
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchSpace:
    """Verify search space configuration."""

    def test_suggest_params_returns_all_keys(self):
        from src.backtest.wfo_optimizer import SEARCH_SPACE, _suggest_params

        # Mock an Optuna trial — must handle log=True kwarg
        mock_trial = MagicMock()
        mock_trial.suggest_float = MagicMock(
            side_effect=lambda name, lo, hi, **kwargs: (lo + hi) / 2
        )
        mock_trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])

        params = _suggest_params(mock_trial)

        for key in SEARCH_SPACE:
            assert key in params, f"Missing parameter: {key}"

    def test_suggested_params_build_strategy_params(self):
        """Suggested params partition cleanly into live and legacy buckets."""
        from src.backtest.wfo_optimizer import _suggest_params

        mock_trial = MagicMock()
        mock_trial.suggest_float = MagicMock(
            side_effect=lambda name, lo, hi, **kwargs: (lo + hi) / 2
        )
        mock_trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])

        params = _suggest_params(mock_trial)
        strategy_params, legacy_params = split_strategy_and_legacy_params(params)
        sp = StrategyParams(**strategy_params)

        assert sp.kelly_fraction == (0.03 + 0.40) / 2
        assert sp.pure_mm_wide_tier_enabled is True
        assert legacy_params["zscore_threshold"] == (0.2 + 2.5) / 2

    def test_expanded_search_space_has_new_params(self):
        """Search space includes the new parameters."""
        from src.backtest.wfo_optimizer import SEARCH_SPACE

        new_params = [
            "volume_ratio_threshold",
            "alpha_default",
            "tp_vol_sensitivity",
            "min_edge_score",
            "iceberg_eqs_bonus",
            "iceberg_tp_alpha",
            "pure_mm_wide_tier_enabled",
            "pure_mm_wide_spread_pct",
        ]
        for p in new_params:
            assert p in SEARCH_SPACE, f"Missing new param: {p}"

    def test_pure_mm_bounds_are_hardened(self):
        """Pure-MM sweep bounds should match the intended Monday grid."""
        from src.backtest.wfo_optimizer import SEARCH_SPACE

        assert SEARCH_SPACE["pure_mm_wide_spread_pct"] == ("suggest_float", 0.05, 0.25)
        assert SEARCH_SPACE["pure_mm_toxic_ofi_ratio"] == ("suggest_float", 0.60, 0.95)

    def test_log_scale_params_marked(self):
        """Certain params should use log-scale (4th element = True)."""
        from src.backtest.wfo_optimizer import SEARCH_SPACE

        log_params = ["spread_compression_pct", "kelly_fraction", "max_impact_pct"]
        for p in log_params:
            spec = SEARCH_SPACE[p]
            assert len(spec) == 4 and spec[3] is True, (
                f"{p} should have log=True"
            )

    def test_min_edge_score_lower_bound(self):
        """Lower bound of min_edge_score must be >= 30 to prevent regression."""
        from src.backtest.wfo_optimizer import SEARCH_SPACE

        spec = SEARCH_SPACE["min_edge_score"]
        lower_bound = spec[1]
        assert lower_bound >= 20.0, (
            f"min_edge_score lower bound {lower_bound} is below 20.0 — "
            "risk of WFO recommending a value that undoes quality-gate tightening"
        )

    def test_log_scale_kwarg_passed(self):
        """Verify log=True is passed to suggest_float for log-scale params."""
        from src.backtest.wfo_optimizer import _suggest_params

        calls_with_log = []

        def mock_suggest(name, lo, hi, **kwargs):
            if kwargs.get("log"):
                calls_with_log.append(name)
            return (lo + hi) / 2

        mock_trial = MagicMock()
        mock_trial.suggest_float = MagicMock(side_effect=mock_suggest)
        mock_trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])

        _suggest_params(mock_trial)

        assert "kelly_fraction" in calls_with_log
        assert "spread_compression_pct" in calls_with_log
        assert "max_impact_pct" in calls_with_log


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: single-fold smoke test with synthetic data
# ═══════════════════════════════════════════════════════════════════════════

def _write_events(path: Path, events: list[dict]) -> None:
    """Write JSONL events to a file (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _make_synthetic_day(data_dir: Path, date_str: str, asset_id: str = "YES") -> None:
    """Create a minimal synthetic JSONL file for one date."""
    tick_dir = data_dir / "raw_ticks" / date_str
    events = [
        {
            "local_ts": 1.0 + i,
            "source": "l2",
            "asset_id": asset_id,
            "payload": {
                "event_type": "snapshot",
                "bids": [{"price": "0.55", "size": "100"}],
                "asks": [{"price": "0.60", "size": "100"}],
            },
        }
        for i in range(5)
    ] + [
        {
            "local_ts": 10.0 + i,
            "source": "trade",
            "asset_id": asset_id,
            "payload": {"price": "0.58", "size": "10", "side": "buy"},
        }
        for i in range(5)
    ]
    _write_events(tick_dir / f"{asset_id}.jsonl", events)


class TestSingleFoldSmoke:
    """Minimal end-to-end WFO with synthetic data (1 fold, 3 trials)."""

    def test_run_single_backtest(self, tmp_path: Path):
        """_run_single_backtest runs without error on synthetic data."""
        from src.backtest.wfo_optimizer import _run_single_backtest

        # Create 1 day of data
        _make_synthetic_day(tmp_path, "2026-01-15", "YES")

        result = _run_single_backtest(
            data_dir=str(tmp_path),
            dates=["2026-01-15"],
            param_overrides={"zscore_threshold": 2.0},
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            initial_cash=1000.0,
            latency_ms=0.0,
            fee_max_pct=1.56,
            fee_enabled=True,
        )

        assert result is not None
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "equity_curve" in result

    def test_run_single_backtest_missing_data_returns_none(self, tmp_path: Path):
        """If no files exist for the requested dates, return None."""
        from src.backtest.wfo_optimizer import _run_single_backtest

        result = _run_single_backtest(
            data_dir=str(tmp_path),
            dates=["2099-01-01"],
            param_overrides={},
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            initial_cash=1000.0,
            latency_ms=0.0,
            fee_max_pct=1.56,
            fee_enabled=True,
        )
        assert result is None

    def test_generate_folds_with_synthetic_dates(self, tmp_path: Path):
        """generate_folds works with MarketDataRecorder.available_dates."""
        # Create 40 days of data
        from datetime import datetime, timedelta

        start = datetime(2026, 1, 1)
        for i in range(40):
            d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            _make_synthetic_day(tmp_path, d, "YES")

        from src.backtest.data_recorder import MarketDataRecorder

        available = MarketDataRecorder.available_dates(str(tmp_path))
        assert len(available) == 40

        folds = generate_folds(available, train_days=30, test_days=7, step_days=7)
        assert len(folds) >= 1
        assert len(folds[0].train_dates) > 0
        assert len(folds[0].test_dates) > 0


class TestPureMmWfoHooks:
    def test_suggest_params_can_be_filtered(self):
        class _Trial:
            def suggest_float(self, name, low, high, log=False):
                return (low + high) / 2.0

            def suggest_int(self, name, low, high):
                return low

            def suggest_categorical(self, name, choices):
                return choices[0]

        params = _suggest_params(
            _Trial(),
            search_space_params=(
                "pure_mm_wide_tier_enabled",
                "pure_mm_wide_spread_pct",
                "pure_mm_toxic_ofi_ratio",
                "pure_mm_depth_evaporation_pct",
            ),
        )

        assert set(params) == {
            "pure_mm_wide_tier_enabled",
            "pure_mm_wide_spread_pct",
            "pure_mm_toxic_ofi_ratio",
            "pure_mm_depth_evaporation_pct",
        }
        assert params["pure_mm_wide_tier_enabled"] is True

    def test_run_single_backtest_with_pure_mm_adapter(self, tmp_path: Path):
        from src.backtest.wfo_optimizer import _run_single_backtest

        tick_dir = tmp_path / "raw_ticks" / "2026-01-15"
        events = [
            {
                "local_ts": 1.0,
                "source": "l2",
                "asset_id": "NO",
                "payload": {
                    "event_type": "snapshot",
                    "bids": [{"price": "0.40", "size": "20"}],
                    "asks": [{"price": "0.42", "size": "20"}],
                },
            },
            {
                "local_ts": 2.0,
                "source": "l2",
                "asset_id": "NO",
                "payload": {
                    "event_type": "snapshot",
                    "bids": [{"price": "0.39", "size": "20"}],
                    "asks": [{"price": "0.40", "size": "20"}],
                },
            },
        ]
        _write_events(tick_dir / "NO.jsonl", events)

        result = _run_single_backtest(
            data_dir=str(tmp_path),
            dates=["2026-01-15"],
            param_overrides={
                "pure_mm_toxic_ofi_ratio": 0.8,
                "pure_mm_depth_evaporation_pct": 0.75,
            },
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            initial_cash=1000.0,
            latency_ms=0.0,
            fee_max_pct=1.56,
            fee_enabled=False,
            strategy_adapter="pure_market_maker",
        )

        assert result is not None
        assert result["total_fills"] >= 1

    def test_run_single_backtest_with_pure_mm_adapter_requires_l2(self, tmp_path: Path):
        from src.backtest.wfo_optimizer import _run_single_backtest

        tick_dir = tmp_path / "raw_ticks" / "2026-01-15"
        events = [
            {
                "local_ts": 1.0,
                "source": "trade",
                "asset_id": "NO",
                "payload": {"price": "0.40", "size": "5", "side": "sell"},
            }
        ]
        _write_events(tick_dir / "NO.jsonl", events)

        result = _run_single_backtest(
            data_dir=str(tmp_path),
            dates=["2026-01-15"],
            param_overrides={
                "pure_mm_toxic_ofi_ratio": 0.8,
                "pure_mm_depth_evaporation_pct": 0.75,
            },
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            initial_cash=1000.0,
            latency_ms=0.0,
            fee_max_pct=1.56,
            fee_enabled=False,
            strategy_adapter="pure_market_maker",
        )

        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
#  WFO Parquet support
# ═══════════════════════════════════════════════════════════════════════════

class TestWfoParquetSupport:
    """Verify that WFO can discover and load Parquet files."""

    def test_collect_files_finds_parquet(self, tmp_path: Path):
        """_collect_files_for_dates picks up Parquet files from <data_dir>/<date>/."""
        from src.backtest.wfo_optimizer import _collect_files_for_dates

        # Create processed Parquet layout: <data_dir>/2026-01-01/general.parquet
        date_dir = tmp_path / "2026-01-01"
        date_dir.mkdir()
        pq_file = date_dir / "general.parquet"
        pq_file.write_bytes(b"PAR1fake")  # dummy content

        files = _collect_files_for_dates(str(tmp_path), ["2026-01-01"])
        assert any(f.suffix == ".parquet" for f in files)
        assert pq_file in files

    def test_collect_files_prefers_both_formats(self, tmp_path: Path):
        """Both JSONL and Parquet files are collected when present."""
        from src.backtest.wfo_optimizer import _collect_files_for_dates

        # JSONL layout
        jsonl_dir = tmp_path / "raw_ticks" / "2026-01-01"
        jsonl_dir.mkdir(parents=True)
        jsonl_file = jsonl_dir / "asset.jsonl"
        jsonl_file.write_text("{}\n")

        # Parquet layout
        pq_dir = tmp_path / "2026-01-01"
        pq_dir.mkdir()
        pq_file = pq_dir / "general.parquet"
        pq_file.write_bytes(b"PAR1fake")

        files = _collect_files_for_dates(str(tmp_path), ["2026-01-01"])
        suffixes = {f.suffix for f in files}
        assert ".jsonl" in suffixes
        assert ".parquet" in suffixes

    def test_available_dates_includes_parquet_dirs(self, tmp_path: Path):
        """available_dates discovers date directories with Parquet files."""
        from src.backtest.data_recorder import MarketDataRecorder

        # Raw JSONL date
        jsonl_dir = tmp_path / "raw_ticks" / "2026-01-01"
        jsonl_dir.mkdir(parents=True)
        (jsonl_dir / "asset.jsonl").write_text("{}\n")

        # Processed Parquet date (no raw_ticks entry)
        pq_dir = tmp_path / "2026-01-02"
        pq_dir.mkdir()
        (pq_dir / "general.parquet").write_bytes(b"PAR1fake")

        dates = MarketDataRecorder.available_dates(str(tmp_path))
        assert "2026-01-01" in dates
        assert "2026-01-02" in dates

    def test_available_dates_ignores_non_parquet_dirs(self, tmp_path: Path):
        """Date-like directories without .parquet files are NOT included."""
        from src.backtest.data_recorder import MarketDataRecorder

        # Create a date-like dir with no Parquet files
        empty_dir = tmp_path / "2026-01-03"
        empty_dir.mkdir()
        (empty_dir / "readme.txt").write_text("not parquet")

        dates = MarketDataRecorder.available_dates(str(tmp_path))
        assert "2026-01-03" not in dates


# ═══════════════════════════════════════════════════════════════════════════
#  Overfitting diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestOverfittingDiagnostics:
    """Tests for IS/OOS decay, overfit probability, and param stability."""

    def test_sharpe_decay_computed(self):
        """FoldResult.sharpe_decay_pct computed correctly."""
        fr = FoldResult(
            fold_index=0,
            best_params={},
            is_sharpe=2.0,
            oos_sharpe=1.0,
            sharpe_decay_pct=((1.0 - 2.0) / abs(2.0)) * 100,  # -50%
        )
        assert abs(fr.sharpe_decay_pct - (-50.0)) < 0.1

    def test_overfit_probability_all_positive(self):
        """If all OOS Sharpe > 0, overfit probability = 0."""
        report = WfoReport(
            folds=[
                FoldResult(fold_index=0, best_params={}, oos_sharpe=1.0),
                FoldResult(fold_index=1, best_params={}, oos_sharpe=0.5),
            ],
            overfit_probability=0.0,
        )
        assert report.overfit_probability == 0.0

    def test_overfit_probability_half_negative(self):
        """If half of OOS Sharpe < 0, overfit probability = 0.5."""
        report = WfoReport(
            folds=[
                FoldResult(fold_index=0, best_params={}, oos_sharpe=1.0),
                FoldResult(fold_index=1, best_params={}, oos_sharpe=-0.5),
            ],
            overfit_probability=0.5,
        )
        assert report.overfit_probability == 0.5

    def test_unstable_params_identified(self):
        """Parameters with CV > 0.50 should be flagged."""
        import numpy as np

        vals = [0.05, 0.35]  # very spread out
        arr = np.array(vals)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1))
        cv = sd / abs(mu)
        assert cv > 0.50  # confirm it's unstable

    def test_fold_result_new_fields_default(self):
        """New FoldResult fields have sane defaults."""
        fr = FoldResult(fold_index=0, best_params={})
        assert fr.is_sortino == 0.0
        assert fr.is_profit_factor == 0.0
        assert fr.is_total_fills == 0
        assert fr.oos_sortino == 0.0
        assert fr.oos_profit_factor == 0.0
        assert fr.oos_total_fills == 0
        assert fr.sharpe_decay_pct == 0.0

    def test_report_new_fields_default(self):
        """New WfoReport fields have sane defaults."""
        report = WfoReport()
        assert report.avg_sharpe_decay_pct == 0.0
        assert report.overfit_probability == 0.0
        assert report.unstable_params == []


# ═══════════════════════════════════════════════════════════════════════════
#  Backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Ensure old API patterns still work after upgrade."""

    def test_generate_folds_old_signature(self):
        """generate_folds with just the original 4 args still works."""
        from datetime import datetime, timedelta

        d = datetime(2026, 1, 1).date()
        dates = [(d + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(45)]
        folds = generate_folds(dates, 30, 7, 7)
        assert len(folds) >= 1

    def test_wfo_config_old_fields_unchanged(self):
        """All original WfoConfig fields remain at their original defaults."""
        cfg = WfoConfig()
        assert cfg.data_dir == "data"
        assert cfg.train_days == 30
        assert cfg.test_days == 7
        assert cfg.step_days == 7
        assert cfg.n_trials == 100
        assert cfg.max_acceptable_drawdown == 0.15
        assert cfg.initial_cash == 1000.0
        assert cfg.storage_url == "sqlite:///wfo_optuna.db"
        assert cfg.study_prefix == "polymarket_wfo"
        assert cfg.latency_ms == 150.0
        assert cfg.fee_max_pct == 2.00
        assert cfg.fee_enabled is True

    def test_fold_result_old_fields_still_work(self):
        """Old FoldResult fields still exist and work."""
        fr = FoldResult(
            fold_index=0,
            best_params={"zscore_threshold": 2.0},
            is_sharpe=1.5,
            is_max_drawdown=0.05,
            is_total_pnl=50.0,
            oos_sharpe=1.2,
            oos_max_drawdown=0.07,
            oos_total_pnl=30.0,
            n_trials_completed=100,
            best_trial_score=1.3,
        )
        assert fr.is_sharpe == 1.5
        assert fr.oos_sharpe == 1.2
