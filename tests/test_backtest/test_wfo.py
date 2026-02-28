"""
Tests for the Walk-Forward Optimization (WFO) pipeline.

Covers:
- Time-series fold generation (date windowing, edge cases)
- Objective function (drawdown penalty, score collapse)
- Strategy parameter injection via BotReplayAdapter
- OOS equity-curve stitching
- End-to-end single-fold smoke test with synthetic data
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.backtest.strategy import BotReplayAdapter
from src.backtest.wfo_optimizer import (
    FoldResult,
    WfoConfig,
    WfoReport,
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


# ═══════════════════════════════════════════════════════════════════════════
#  Objective / scoring math
# ═══════════════════════════════════════════════════════════════════════════

class TestWfoScore:
    """Tests for compute_wfo_score — the drawdown-penalized Sharpe objective."""

    def test_zero_drawdown_no_penalty(self):
        """If drawdown is 0, penalty = 1 → score = Sharpe."""
        score = compute_wfo_score(sharpe_ratio=2.0, max_drawdown=0.0, max_acceptable_drawdown=0.15)
        assert score == 2.0

    def test_half_threshold_drawdown(self):
        """If drawdown = half of threshold, penalty = 0.5."""
        score = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.075, max_acceptable_drawdown=0.15
        )
        assert abs(score - 1.0) < 1e-9

    def test_at_threshold_collapses_to_zero(self):
        """If drawdown == threshold, penalty = 0 → score = 0."""
        score = compute_wfo_score(
            sharpe_ratio=5.0, max_drawdown=0.15, max_acceptable_drawdown=0.15
        )
        assert score == 0.0

    def test_exceeds_threshold_collapses_to_zero(self):
        """If drawdown > threshold, score must still be 0."""
        score = compute_wfo_score(
            sharpe_ratio=10.0, max_drawdown=0.25, max_acceptable_drawdown=0.15
        )
        assert score == 0.0

    def test_negative_sharpe_with_drawdown(self):
        """Negative Sharpe × positive penalty → negative score."""
        score = compute_wfo_score(
            sharpe_ratio=-1.5, max_drawdown=0.05, max_acceptable_drawdown=0.15
        )
        expected = -1.5 * (1.0 - 0.05 / 0.15)
        assert abs(score - expected) < 1e-9

    def test_very_small_drawdown(self):
        """Near-zero drawdown: penalty ≈ 1."""
        score = compute_wfo_score(
            sharpe_ratio=3.0, max_drawdown=0.001, max_acceptable_drawdown=0.15
        )
        assert score > 2.9

    def test_custom_threshold(self):
        """Different max_acceptable_drawdown values."""
        score_10 = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.10, max_acceptable_drawdown=0.10
        )
        assert score_10 == 0.0  # exactly at threshold

        score_20 = compute_wfo_score(
            sharpe_ratio=2.0, max_drawdown=0.10, max_acceptable_drawdown=0.20
        )
        assert abs(score_20 - 1.0) < 1e-9  # penalty = 0.5


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
        assert adapter._params.zscore_threshold == StrategyParams().zscore_threshold
        assert adapter._params.kelly_fraction == StrategyParams().kelly_fraction

    def test_custom_params_used(self):
        """Injected StrategyParams override defaults."""
        custom = StrategyParams(zscore_threshold=1.7, kelly_fraction=0.10)
        adapter = BotReplayAdapter(
            market_id="MKT",
            yes_asset_id="YES",
            no_asset_id="NO",
            params=custom,
        )
        assert adapter._params.zscore_threshold == 1.7
        assert adapter._params.kelly_fraction == 0.10

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
                ),
                FoldResult(
                    fold_index=1,
                    best_params={"zscore_threshold": 2.3, "kelly_fraction": 0.12},
                    is_sharpe=2.1,
                    oos_sharpe=1.5,
                    is_max_drawdown=0.04,
                    oos_max_drawdown=0.06,
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
        )

        text = report.summary()
        assert "WALK-FORWARD" in text
        assert "1.35" in text  # aggregate sharpe
        assert "zscore_threshold" in text

    def test_empty_report(self):
        report = WfoReport()
        text = report.summary()
        assert "Folds completed" in text
        assert "0" in text


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
        assert cfg.n_trials == 100
        assert cfg.max_acceptable_drawdown == 0.15
        assert cfg.initial_cash == 1000.0
        assert cfg.max_workers >= 1

    def test_custom_config(self):
        cfg = WfoConfig(
            data_dir="/tmp/data",
            train_days=14,
            test_days=3,
            step_days=3,
            n_trials=50,
            max_acceptable_drawdown=0.10,
        )
        assert cfg.train_days == 14
        assert cfg.n_trials == 50
        assert cfg.max_acceptable_drawdown == 0.10


# ═══════════════════════════════════════════════════════════════════════════
#  Search space
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchSpace:
    """Verify search space configuration."""

    def test_suggest_params_returns_all_keys(self):
        from src.backtest.wfo_optimizer import SEARCH_SPACE, _suggest_params

        # Mock an Optuna trial
        mock_trial = MagicMock()
        mock_trial.suggest_float = MagicMock(side_effect=lambda name, lo, hi: (lo + hi) / 2)

        params = _suggest_params(mock_trial)

        for key in SEARCH_SPACE:
            assert key in params, f"Missing parameter: {key}"

    def test_suggested_params_build_strategy_params(self):
        """Suggested params can be passed to StrategyParams constructor."""
        from src.backtest.wfo_optimizer import _suggest_params

        mock_trial = MagicMock()
        mock_trial.suggest_float = MagicMock(side_effect=lambda name, lo, hi: (lo + hi) / 2)

        params = _suggest_params(mock_trial)
        sp = StrategyParams(**params)

        assert sp.zscore_threshold == (1.5 + 3.5) / 2
        assert sp.kelly_fraction == (0.05 + 0.35) / 2


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
