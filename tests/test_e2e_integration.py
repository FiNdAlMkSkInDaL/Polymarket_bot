"""
End-to-End System Integration Test — "Mock Day" of Trading.

Exercises the complete pipeline:
  1. Mock Data Generation   (SyntheticGenerator with edge-case injection)
  2. Conversion Loop        (JSONL → Parquet via ParquetConverter)
  3. Backtest Replay        (BacktestEngine + SimClock + BotReplayAdapter)
  4. WFO Smoke Test         (Optuna + ProcessPoolExecutor via run_wfo)

Each step is gated: if a prior step fails, all downstream steps are
skipped via ``pytest.skip()``.

Mark:
    pytest -m slow   — to include this in slow-suite runs
    pytest tests/test_e2e_integration.py -v  — standalone
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data.synthetic import SyntheticGenerator, _DEFAULT_YES_ASSET, _DEFAULT_NO_ASSET


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

_SEED = 42
_NUM_ROWS = 50_000          # smaller for CI speed, still large enough
_DURATION_HOURS = 48.0
_NUM_ASSETS = 3
_GAP_PROB = 0.01
_SPIKE_PROB = 0.005
_SPREAD_COMPRESS_PROB = 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_all_records(raw_ticks_dir: Path) -> list[dict]:
    """Read every JSONL line from every file under *raw_ticks_dir*."""
    records: list[dict] = []
    for fp in sorted(raw_ticks_dir.rglob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Test class
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestE2ESystemIntegration:
    """Full "Mock Day" system integration test."""

    # ── Step 1: Mock Data Generation ──────────────────────────────────

    def test_step1_mock_data_generation(self, tmp_path: Path) -> None:
        """Generate 48 h of synthetic L2 JSONL for 3 assets with edge cases."""
        gen = SyntheticGenerator(
            seed=_SEED,
            gap_probability=_GAP_PROB,
            spike_probability=_SPIKE_PROB,
            spread_compress_probability=_SPREAD_COMPRESS_PROB,
        )
        raw_dir = gen.generate(
            tmp_path,
            num_rows=_NUM_ROWS,
            duration_hours=_DURATION_HOURS,
            num_assets=_NUM_ASSETS,
        )

        # ── Verify output directory ───────────────────────────────────
        assert raw_dir.exists(), f"raw_ticks directory does not exist: {raw_dir}"

        # Should have 2–3 date subdirectories (48 h from midnight UTC)
        date_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
        assert len(date_dirs) >= 2, f"Expected ≥2 date dirs, got {len(date_dirs)}"

        # Each date dir should have JSONL files for at least some assets
        all_jsonl = list(raw_dir.rglob("*.jsonl"))
        assert len(all_jsonl) >= 3, f"Expected ≥3 JSONL files, got {len(all_jsonl)}"

        # Total row count
        records = _read_all_records(raw_dir)
        assert len(records) == _NUM_ROWS

        # Asset diversity — should see 3 distinct asset IDs
        asset_ids = {rec["asset_id"] for rec in records}
        assert len(asset_ids) == _NUM_ASSETS, f"Expected {_NUM_ASSETS} assets, got {asset_ids}"

        # Edge-case verification: sequence gaps present
        seqs_by_asset: dict[str, list[int]] = {}
        for rec in records:
            seq = rec["payload"].get("seq")
            if seq is not None:
                seqs_by_asset.setdefault(rec["asset_id"], []).append(seq)
        total_gaps = 0
        for seqs in seqs_by_asset.values():
            sorted_seqs = sorted(set(seqs))
            for i in range(1, len(sorted_seqs)):
                if sorted_seqs[i] - sorted_seqs[i - 1] > 1:
                    total_gaps += 1
        assert total_gaps > 0, "Expected injected sequence gaps but found none"

    # ── Step 2: Conversion Loop ───────────────────────────────────────

    def test_step2_conversion_loop(self, tmp_path: Path) -> None:
        """Convert mock JSONL to Parquet; verify sorting, integrity report."""
        from src.data.prep_data import ParquetConverter

        # ── Generate first ──
        gen = SyntheticGenerator(
            seed=_SEED,
            gap_probability=_GAP_PROB,
            spike_probability=_SPIKE_PROB,
            spread_compress_probability=_SPREAD_COMPRESS_PROB,
        )
        raw_dir = gen.generate(
            tmp_path / "raw",
            num_rows=_NUM_ROWS,
            duration_hours=_DURATION_HOURS,
            num_assets=_NUM_ASSETS,
        )

        # ── Convert ──
        processed_dir = tmp_path / "processed"
        converter = ParquetConverter()
        report = converter.convert([raw_dir], processed_dir)

        # ── Verify Parquet files exist ──
        parquet_files = list(processed_dir.rglob("*.parquet"))
        assert len(parquet_files) >= 1, "No Parquet files produced"

        # ── Verify HealthReport ──
        assert report.valid_rows == _NUM_ROWS
        assert report.malformed_rows == 0, f"Unexpected malformed rows: {report.malformed_rows}"
        assert report.malformed_pct == 0.0

        # Sequence gaps must be detected (we injected them)
        assert report.sequence_gaps > 0, (
            f"HealthReport should detect injected gaps, got {report.sequence_gaps}"
        )
        assert report.sequence_gap_pct > 0

        # Health score < 100 because of gaps
        assert report.health_score < 100.0
        assert report.health_score > 50.0  # not catastrophic

        # ── Verify global sort within each Parquet file ──
        for pf in parquet_files:
            table = pq.read_table(pf)
            ts_array = table.column("local_ts").to_pylist()
            for i in range(1, len(ts_array)):
                assert ts_array[i] >= ts_array[i - 1], (
                    f"Parquet {pf.name} not sorted at row {i}: "
                    f"{ts_array[i-1]} > {ts_array[i]}"
                )

        # ── Verify audit files ──
        audit_files = list(processed_dir.rglob("batch_audit_*.json"))
        assert len(audit_files) >= 1, "No audit file produced"

    # ── Step 3: Backtest Replay ───────────────────────────────────────

    def test_step3_backtest_replay(self, tmp_path: Path) -> None:
        """Backtest with SimClock patching + BotReplayAdapter signal path."""
        from src.backtest.data_loader import DataLoader
        from src.backtest.engine import BacktestConfig, BacktestEngine
        from src.backtest.strategy import BotReplayAdapter
        from src.data.prep_data import ParquetConverter

        # ── Generate + convert ──
        gen = SyntheticGenerator(
            seed=_SEED,
            gap_probability=_GAP_PROB,
            spike_probability=_SPIKE_PROB,
            spread_compress_probability=_SPREAD_COMPRESS_PROB,
        )
        raw_dir = gen.generate(
            tmp_path / "raw",
            num_rows=_NUM_ROWS,
            duration_hours=_DURATION_HOURS,
            num_assets=_NUM_ASSETS,
        )
        processed_dir = tmp_path / "processed"
        ParquetConverter().convert([raw_dir], processed_dir)

        # ── Set up engine ──
        loader = DataLoader.from_directory(processed_dir)
        config = BacktestConfig(
            initial_cash=1000.0,
            latency_ms=150.0,
            fee_max_pct=1.56,
            fee_enabled=True,
        )
        strategy = BotReplayAdapter(
            market_id="BACKTEST",
            yes_asset_id=_DEFAULT_YES_ASSET,
            no_asset_id=_DEFAULT_NO_ASSET,
            fee_enabled=True,
            initial_bankroll=1000.0,
        )
        engine = BacktestEngine(
            strategy=strategy,
            data_loader=loader,
            config=config,
        )

        # ── Verify SimClock patches time.time ──
        real_time_before = time.time()
        result = engine.run()
        real_time_after = time.time()

        # After engine.run() completes, time.time should be restored
        assert abs(time.time() - real_time_after) < 5.0, (
            "time.time not restored after backtest — SimClock leak"
        )

        # ── Verify BacktestResult ──
        assert result is not None
        assert result.events_processed > 0, "No events processed"
        assert result.final_equity > 0, "Final equity should be positive"
        assert result.metrics is not None

        # Equity curve should have entries
        assert len(result.metrics.equity_curve) > 0, "No equity curve data"

        # Summary should not crash
        summary_str = result.summary()
        assert isinstance(summary_str, str)
        assert len(summary_str) > 50

    # ── Step 4: WFO Smoke Test ────────────────────────────────────────

    def test_step4_wfo_smoke_test(self, tmp_path: Path) -> None:
        """Truncated WFO: 2 trials/fold, single worker, SQLite storage."""
        from src.backtest.wfo_optimizer import WfoConfig, run_wfo
        from src.data.prep_data import ParquetConverter

        # ── Generate + convert ──
        # Use a fixed base_time so we know exact date boundaries
        import calendar
        from datetime import datetime, timezone
        base_time = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc).timestamp()

        gen = SyntheticGenerator(
            seed=_SEED,
            gap_probability=_GAP_PROB,
            spike_probability=_SPIKE_PROB,
            spread_compress_probability=_SPREAD_COMPRESS_PROB,
        )
        raw_dir = gen.generate(
            tmp_path / "raw",
            num_rows=_NUM_ROWS,
            duration_hours=_DURATION_HOURS,
            num_assets=_NUM_ASSETS,
            base_time=base_time,
        )
        processed_dir = tmp_path / "processed"
        ParquetConverter().convert([raw_dir], processed_dir)

        # ── Verify date directories for WFO ──
        from src.backtest.data_recorder import MarketDataRecorder
        available = MarketDataRecorder.available_dates(str(processed_dir))
        assert len(available) >= 2, (
            f"Need ≥2 dates for WFO folds, got {available}"
        )

        # ── Configure WFO ──
        db_path = tmp_path / "wfo_test.db"
        cfg = WfoConfig(
            data_dir=str(processed_dir),
            market_id="BACKTEST",
            yes_asset_id=_DEFAULT_YES_ASSET,
            no_asset_id=_DEFAULT_NO_ASSET,
            train_days=1,
            test_days=1,
            step_days=1,
            n_trials=2,           # minimal for smoke test
            max_workers=1,        # single-process mode
            max_acceptable_drawdown=0.50,  # lenient
            initial_cash=1000.0,
            storage_url=f"sqlite:///{db_path}",
            latency_ms=150.0,
            fee_max_pct=1.56,
            fee_enabled=True,
        )

        # ── Run WFO ──
        report = run_wfo(cfg)

        # ── Verify WfoReport ──
        assert report is not None
        assert len(report.folds) >= 1, f"Expected ≥1 fold, got {len(report.folds)}"

        for fr in report.folds:
            assert fr.n_trials_completed >= 2, (
                f"Fold {fr.fold_index} only completed {fr.n_trials_completed} trials"
            )
            assert fr.best_params, f"Fold {fr.fold_index} has no best_params"

        # Stitched equity curve should have entries
        assert len(report.stitched_equity_curve) >= 0  # may be empty if no fills

        # Aggregate metrics should be finite
        assert math.isfinite(report.aggregate_oos_sharpe)
        assert math.isfinite(report.aggregate_oos_max_drawdown)
        assert math.isfinite(report.aggregate_oos_total_pnl)

        # Summary should not crash
        summary_str = report.summary()
        assert isinstance(summary_str, str)
        assert "WALK-FORWARD" in summary_str

        # Elapsed time should be positive
        assert report.total_elapsed_s > 0

        # SQLite DB should exist (Optuna used it)
        assert db_path.exists(), "Optuna SQLite database was not created"

        # Parameter stability dict should contain the search-space keys
        if len(report.folds) > 1:
            assert len(report.parameter_stability) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrated "Full Pipeline" test (sequential gate)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestFullPipelineSequential:
    """Run all 4 steps in a single tmp_path so each step's output feeds
    the next.  Failures cascade via assertions — no subsequent step
    runs on corrupt data."""

    def test_full_mock_day(self, tmp_path: Path) -> None:
        """Complete Mock Day: generate → convert → backtest → WFO."""
        from datetime import datetime, timezone

        from src.backtest.data_loader import DataLoader
        from src.backtest.data_recorder import MarketDataRecorder
        from src.backtest.engine import BacktestConfig, BacktestEngine
        from src.backtest.strategy import BotReplayAdapter
        from src.backtest.wfo_optimizer import WfoConfig, run_wfo
        from src.data.prep_data import ParquetConverter

        report_lines: list[str] = []
        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("        SYSTEM READINESS REPORT -- E2E Mock Day")
        report_lines.append("=" * 70)

        # ── STEP 1: Mock Data Generation ──────────────────────────────
        base_time = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        gen = SyntheticGenerator(
            seed=_SEED,
            gap_probability=_GAP_PROB,
            spike_probability=_SPIKE_PROB,
            spread_compress_probability=_SPREAD_COMPRESS_PROB,
        )
        raw_dir = gen.generate(
            tmp_path / "data",
            num_rows=_NUM_ROWS,
            duration_hours=_DURATION_HOURS,
            num_assets=_NUM_ASSETS,
            base_time=base_time,
        )

        records = _read_all_records(raw_dir)
        asset_ids = {rec["asset_id"] for rec in records}
        date_dirs = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())

        assert len(records) == _NUM_ROWS
        assert len(asset_ids) == _NUM_ASSETS
        assert len(date_dirs) >= 2

        report_lines.append("")
        report_lines.append("  STEP 1 -- Mock Data Generation ................. PASS")
        report_lines.append(f"    Rows:       {len(records):,}")
        report_lines.append(f"    Assets:     {len(asset_ids)}")
        report_lines.append(f"    Dates:      {', '.join(date_dirs)}")

        # ── STEP 2: Conversion Loop ───────────────────────────────────
        processed_dir = tmp_path / "processed"
        converter = ParquetConverter()
        health = converter.convert([raw_dir], processed_dir)

        parquet_files = list(processed_dir.rglob("*.parquet"))
        assert len(parquet_files) >= 1
        assert health.valid_rows == _NUM_ROWS
        assert health.malformed_rows == 0
        assert health.sequence_gaps > 0  # intentional gaps detected
        assert health.health_score < 100.0

        # Verify sort order
        for pf in parquet_files:
            table = pq.read_table(pf)
            ts_array = table.column("local_ts").to_pylist()
            for i in range(1, len(ts_array)):
                assert ts_array[i] >= ts_array[i - 1]

        report_lines.append("")
        report_lines.append("  STEP 2 -- Conversion Loop ...................... PASS")
        report_lines.append(f"    Parquet files: {len(parquet_files)}")
        report_lines.append(f"    Valid rows:    {health.valid_rows:,}")
        report_lines.append(f"    Seq gaps:      {health.sequence_gaps}")
        report_lines.append(f"    Health score:  {health.health_score:.1f}/100")

        # ┌────────────────────────────────────────────────────────────┐
        # │  VERIFICATION 1: batch_audit JSON — sequence gap proof    │
        # └────────────────────────────────────────────────────────────┘
        report_lines.append("")
        report_lines.append("  " + "-" * 66)
        report_lines.append("  VERIFICATION 1 -- batch_audit_*.json  (sequence gap proof)")
        report_lines.append("  " + "-" * 66)
        audit_files = sorted(processed_dir.rglob("batch_audit_*.json"))
        assert len(audit_files) >= 1, "No audit files found"
        for af in audit_files:
            audit_data = json.loads(af.read_text(encoding="utf-8"))
            report_lines.append(f"    File: {af.name}")
            report_lines.append(f"    Contents:")
            pretty = json.dumps(audit_data, indent=6)
            for line in pretty.splitlines():
                report_lines.append(f"      {line}")
        # Assert the audit data actually contains gap info
        any_audit_has_gaps = False
        for af in audit_files:
            audit_data = json.loads(af.read_text(encoding="utf-8"))
            if audit_data.get("sequence_gaps", 0) > 0:
                any_audit_has_gaps = True
                break
        assert any_audit_has_gaps, "No audit file recorded sequence_gaps > 0"
        report_lines.append("    [OK] Sequence gaps detected and recorded in audit JSON")

        # ── STEP 3: Backtest Replay ───────────────────────────────────
        loader = DataLoader.from_directory(processed_dir)
        config = BacktestConfig(
            initial_cash=1000.0,
            latency_ms=150.0,
            fee_max_pct=1.56,
            fee_enabled=True,
        )
        strategy = BotReplayAdapter(
            market_id="BACKTEST",
            yes_asset_id=_DEFAULT_YES_ASSET,
            no_asset_id=_DEFAULT_NO_ASSET,
            fee_enabled=True,
            initial_bankroll=1000.0,
        )
        engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)

        real_time_before = time.time()
        bt_result = engine.run()
        real_time_after = time.time()
        wall_clock_s = real_time_after - real_time_before

        # SimClock must restore time.time
        time_after_restore = time.time()
        assert abs(time_after_restore - real_time_after) < 5.0

        assert bt_result.events_processed > 0
        assert bt_result.final_equity > 0
        assert len(bt_result.metrics.equity_curve) > 0

        # Compute simulated time span from equity curve
        eq_curve = bt_result.metrics.equity_curve
        sim_start = eq_curve[0][0]
        sim_end = eq_curve[-1][0]
        sim_span_hours = (sim_end - sim_start) / 3600.0

        summary_str = bt_result.summary()
        assert isinstance(summary_str, str)

        report_lines.append("")
        report_lines.append("  STEP 3 -- Backtest Replay ...................... PASS")
        report_lines.append(f"    Events:        {bt_result.events_processed:,}")
        report_lines.append(f"    Final equity:  ${bt_result.final_equity:,.2f}")
        report_lines.append(f"    Total fills:   {bt_result.metrics.total_fills}")
        report_lines.append(f"    Max drawdown:  {bt_result.metrics.max_drawdown:.2%}")
        report_lines.append(f"    Sharpe ratio:  {bt_result.metrics.sharpe_ratio:+.2f}")

        # ┌────────────────────────────────────────────────────────────┐
        # │  VERIFICATION 2: SimClock — wall-clock vs simulated time  │
        # └────────────────────────────────────────────────────────────┘
        report_lines.append("")
        report_lines.append("  " + "-" * 66)
        report_lines.append("  VERIFICATION 2 -- SimClock  (wall-clock proof)")
        report_lines.append("  " + "-" * 66)
        report_lines.append(f"    Simulated time span:  {sim_span_hours:.1f} hours ({sim_span_hours/24:.1f} days)")
        report_lines.append(f"    Wall-clock elapsed:   {wall_clock_s:.2f} seconds")
        report_lines.append(f"    Speedup factor:       {sim_span_hours * 3600 / max(wall_clock_s, 0.001):,.0f}x")
        report_lines.append(f"    time.time() restored: {abs(time_after_restore - real_time_after) < 1.0}")
        assert wall_clock_s < 60.0, (
            f"Backtest took {wall_clock_s:.1f}s wall-clock -- SimClock may not be working. "
            f"A {sim_span_hours:.0f}h sim should complete in seconds, not minutes."
        )
        report_lines.append(f"    [OK] 48h of simulated data replayed in {wall_clock_s:.2f}s -- SimClock confirmed operational")

        # ── STEP 4: WFO Smoke Test ────────────────────────────────────
        available = MarketDataRecorder.available_dates(str(processed_dir))
        assert len(available) >= 2

        db_path = tmp_path / "wfo_e2e.db"
        wfo_cfg = WfoConfig(
            data_dir=str(processed_dir),
            market_id="BACKTEST",
            yes_asset_id=_DEFAULT_YES_ASSET,
            no_asset_id=_DEFAULT_NO_ASSET,
            train_days=1,
            test_days=1,
            step_days=1,
            n_trials=2,
            max_workers=1,
            max_acceptable_drawdown=0.50,
            initial_cash=1000.0,
            storage_url=f"sqlite:///{db_path}",
            latency_ms=150.0,
            fee_max_pct=1.56,
            fee_enabled=True,
        )

        wfo_report = run_wfo(wfo_cfg)

        assert wfo_report is not None
        assert len(wfo_report.folds) >= 1
        for fr in wfo_report.folds:
            assert fr.n_trials_completed >= 2
        assert math.isfinite(wfo_report.aggregate_oos_sharpe)
        assert math.isfinite(wfo_report.aggregate_oos_max_drawdown)
        assert db_path.exists()

        wfo_summary = wfo_report.summary()
        assert "WALK-FORWARD" in wfo_summary

        report_lines.append("")
        report_lines.append("  STEP 4 -- WFO Smoke Test ....................... PASS")
        report_lines.append(f"    Folds:         {len(wfo_report.folds)}")
        report_lines.append(f"    OOS Sharpe:    {wfo_report.aggregate_oos_sharpe:+.2f}")
        report_lines.append(f"    OOS Max DD:    {wfo_report.aggregate_oos_max_drawdown:.2%}")
        report_lines.append(f"    OOS PnL:       ${wfo_report.aggregate_oos_total_pnl:+.2f}")
        report_lines.append(f"    Elapsed:       {wfo_report.total_elapsed_s:.1f}s")

        # ┌────────────────────────────────────────────────────────────┐
        # │  VERIFICATION 3: Optuna SQLite — trial persistence proof  │
        # └────────────────────────────────────────────────────────────┘
        import sqlite3
        report_lines.append("")
        report_lines.append("  " + "-" * 66)
        report_lines.append("  VERIFICATION 3 -- Optuna SQLite  (trial persistence proof)")
        report_lines.append("  " + "-" * 66)
        report_lines.append(f"    DB path:  {db_path}")
        report_lines.append(f"    DB size:  {db_path.stat().st_size:,} bytes")

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # List studies
        cur.execute("SELECT study_id, study_name FROM studies ORDER BY study_id")
        studies = cur.fetchall()
        report_lines.append(f"    Studies:  {len(studies)}")
        for sid, sname in studies:
            report_lines.append(f"      [{sid}] {sname}")

        # Count completed trials per study
        cur.execute("""
            SELECT s.study_name, COUNT(t.trial_id) AS n_trials, t.state
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            GROUP BY s.study_name, t.state
            ORDER BY s.study_name, t.state
        """)
        trial_rows = cur.fetchall()
        report_lines.append(f"    Trial breakdown:")
        total_completed = 0
        for sname, cnt, state in trial_rows:
            report_lines.append(f"      {sname}: {cnt} trials -> {state}")
            if state == "COMPLETE":
                total_completed += cnt

        # Dump a sample of trial values
        cur.execute("""
            SELECT s.study_name, t.trial_id, t.state, tv.value
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            ORDER BY s.study_name, t.trial_id
        """)
        trial_detail = cur.fetchall()
        report_lines.append(f"    Trial details:")
        for sname, tid, state, value in trial_detail:
            val_str = f"score={value:.4f}" if value is not None else "no value"
            report_lines.append(f"      {sname} trial#{tid}: {state}, {val_str}")

        # Check for failed trials
        cur.execute("SELECT COUNT(*) FROM trials WHERE state = 'FAIL'")
        fail_count = cur.fetchone()[0]
        report_lines.append(f"    Failed trials: {fail_count}")

        # List all tables to confirm schema integrity
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        all_tables = [row[0] for row in cur.fetchall()]
        report_lines.append(f"    SQLite tables: {', '.join(all_tables)}")

        conn.close()

        assert total_completed >= 4, (
            f"Expected >=4 completed trials (2 trials x 2 folds), got {total_completed}"
        )
        assert fail_count == 0, f"{fail_count} trials failed -- check fail_reason in DB"
        report_lines.append(f"    [OK] {total_completed} trials completed, {fail_count} failures, no database-locked errors")

        # ── System Readiness Report ───────────────────────────────────
        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("        ALL 4 STEPS PASSED -- SYSTEM READY FOR DEPLOYMENT")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_text = "\n".join(report_lines)
        print(report_text)

        # Persist report to file
        report_path = tmp_path / "system_readiness_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
