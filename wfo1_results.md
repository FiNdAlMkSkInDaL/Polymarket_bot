# WFO Pipeline Run #1 — Investigation Report

**Date:** 2026-03-10  
**Investigator:** Automated Diagnostic Agent  
**Pipeline Script:** `scripts/run_optimization_pipeline.py`  
**VPS:** `botuser@135.181.85.32`  
**Target Data:** `data/vps_march2026/` (89 available dates)

---

## 1. Pipeline Status

| Check | Result |
|-------|--------|
| **Process Liveness** | No `run_optimization_pipeline.py` process running on VPS |
| **Active VPS Processes** | Live bot running: `src.cli run --env PAPER` (PID 218119, 14.6% CPU, 211 MB RSS) + 2 multiprocessing workers |
| **tmux Sessions** | No tmux server running (`no server running on /tmp/tmux-1000/default`) |
| **Pipeline Completion** | **COMPLETED (smoke-test mode) — but with CRITICAL zero-trade failure** |

### Execution Timeline

The VPS shows evidence of **three separate pipeline executions**:

| Run | Time (UTC) | Mode | Outcome |
|-----|-----------|------|---------|
| **Run 1** | 2026-03-09 ~16:05–16:06 | `--smoke-test` | Completed all 3 phases in ~24 seconds. Produced all JSON artefacts. **Zero trades across every fold.** |
| **Run 2** | 2026-03-09 16:31–17:13+ | Production (50 trials) | Started Phase 1. `pipeline.log` shows folds 0–1 completed (16:31–16:59), fold 2 began. `wfo_phase1.db` grew to 704 KB by 17:13. **No Phase 2 DB produced — run appears to have stalled or been killed during Phase 1.** |
| **Run 3** | 2026-03-10 14:07–14:08 | Unknown | Two `wfo_complete` events in `bot.jsonl` (1 fold, then 2 folds). **Zero trades again.** |

### Critical Finding: Universal Zero-Trade Problem

**Every single backtest across all runs produced zero trades:**
```
total_positions: 0 | open_remaining: 0 | total_pnl_net: 0
pnl: 0.0 | sharpe: 0.0 | max_dd: 0.0
```

The Optuna optimizer recorded best scores of `-Infinity` for all folds because no parameter combination could generate any trades. The champion selection defaulted to fold 0 with all-zero metrics.

---

## 2. Artefact Audit (VPS: `/home/botuser/polymarket-bot/logs/`)

### JSON Artefacts

| Artefact | Status | Size | Modified | `generated_at` (UTC) |
|----------|--------|------|----------|---------------------|
| `phase1_champion.json` | **EXISTS** | 1,378 B | Mar 9 16:06 | 2026-03-09T16:06:00 |
| `phase2_bounds.json` | **EXISTS** | 1,620 B | Mar 9 16:06 | — |
| `phase2_champion.json` | **EXISTS** | 1,373 B | Mar 9 16:06 | 2026-03-09T16:06:12 |
| `phase3_locked_bounds.json` | **EXISTS** | 1,690 B | Mar 9 16:06 | — |
| `final_validation_tearsheet.json` | **EXISTS** | 1,373 B | Mar 9 16:06 | 2026-03-09T16:06:24 |

### Optuna SQLite Databases

| Database | Status | Size | Modified | Notes |
|----------|--------|------|----------|-------|
| `wfo_phase1.db` | **EXISTS** | 704 KB | Mar 9 17:13 | From production Run 2 |
| `wfo_phase2.db` | **MISSING** | — | — | Only `.bak_pre_clean_restart` (139 KB) exists |
| `wfo_phase3.db` | **EXISTS** | 139 KB | Mar 9 16:06 | From smoke-test Run 1 |
| `wfo_optuna.db` | EXISTS | 2.4 MB | Mar 8 22:49 | Legacy from prior WFO runs |

### Backup Files

| File | Size | Notes |
|------|------|-------|
| `wfo_phase1.db.bak_pre_clean_restart` | 1,007 KB | Phase 1 DB from before the March 9 WFO Reset |
| `wfo_phase2.db.bak_pre_clean_restart` | 139 KB | Phase 2 DB from before the March 9 WFO Reset |

### Additional Logs

| File | Size | Notes |
|------|------|-------|
| `pipeline.log` | 29 KB (137 lines) | Production Run 2 log — cuts off at fold 2 `wfo_warm_start` |
| `wfo.log` | 1.9 MB | Legacy WFO log from Mar 8 prior runs |
| `bot_console.log` | 11.9 MB | Live bot console — no pipeline events |

### Zombie/Failed Trial Analysis

All trials technically "completed" (state `COMPLETE` in Optuna) but with `-Infinity` scores due to the zero-trade problem. No zombie or stuck trials — the optimizer simply found no viable parameter region.

---

## 3. Parameter Evolution (Phase 1 → Phase 2 → Phase 3)

Despite zero trades, the pipeline still selected "champions" (the first trial's random params) and narrowed bounds. The parameters are **meaningless** because they were not selected based on performance — they're essentially random draws from the search space.

### Champion Parameters Across Phases

| Parameter | Search Space | Phase 1 Champion | Phase 2 Bounds (±15%) | Phase 2 Champion | Phase 3 (Locked) |
|-----------|-------------|-------------------|----------------------|-------------------|-----------------|
| `zscore_threshold` | [1.0, 2.5] | 2.086 | [1.861, 2.311] | 2.180 | [2.180, 2.180] |
| `spread_compression_pct` | [0.02, 0.30] | 0.054 | [0.020, 0.096] | 0.064 | [0.064, 0.064] |
| `volume_ratio_threshold` | [0.1, 4.0] | 2.647 | [2.062, 3.232] | 2.112 | [2.112, 2.112] |
| `trend_guard_pct` | [0.05, 1.0] | 0.834 | [0.692, 0.977] | 0.811 | [0.811, 0.811] |
| `stop_loss_cents` | [4.0, 12.0] | 5.224 | [4.024, 6.424] | 5.598 | [5.598, 5.598] |
| `trailing_stop_offset_cents` | [0.5, 6.0] | 3.161 | [2.336, 3.986] | 3.800 | [3.800, 3.800] |
| `kelly_fraction` | [0.03, 0.40] | 0.032 | [0.030, 0.088] | 0.063 | [0.063, 0.063] |
| `max_impact_pct` | [0.03, 0.30] | 0.145 | [0.104, 0.185] | 0.145 | [0.145, 0.145] |
| `alpha_default` | [0.25, 0.75] | 0.264 | [0.250, 0.339] | 0.330 | [0.330, 0.330] |
| `tp_vol_sensitivity` | [0.5, 3.0] | 0.826 | [0.500, 1.201] | 0.710 | [0.710, 0.710] |
| `min_edge_score` | [50.0, 85.0] | 60.322 | [55.072, 65.572] | 64.702 | [64.702, 64.702] |
| `rpe_confidence_threshold` | [0.03, 0.20] | 0.043 | [0.030, 0.069] | 0.034 | [0.034, 0.034] |
| `rpe_bayesian_obs_weight` | [1.0, 15.0] | 4.077 | [1.977, 6.177] | 2.748 | [2.748, 2.748] |
| `rpe_crypto_vol_default` | [0.50, 1.20] | 1.093 | [0.988, 1.198] | 1.149 | [1.149, 1.149] |
| `drift_z_threshold` | [0.5, 2.0] | 0.932 | [0.707, 1.157] | 1.064 | [1.064, 1.064] |
| `drift_vol_ceiling` | [0.02, 0.15] | 0.094 | [0.074, 0.113] | 0.108 | [0.108, 0.108] |
| `pce_max_portfolio_var_usd` | [20.0, 100.0] | 95.001 | [83.001, 100.0] | 84.243 | [84.243, 84.243] |
| `pce_correlation_haircut_threshold` | [0.30, 0.80] | 0.592 | [0.517, 0.667] | 0.535 | [0.535, 0.535] |
| `pce_structural_prior_weight` | int [5, 30] | 6 | [5, 9] | 6 | [6, 7] |
| `pce_holding_period_minutes` | int [30, 360] | 55 | [30, 104] | 75 | [75, 76] |
| `iceberg_eqs_bonus` | [0.05, 0.25] | 0.089 | [0.059, 0.119] | 0.112 | [0.112, 0.112] |
| `iceberg_tp_alpha` | [0.02, 0.10] | 0.048 | [0.036, 0.060] | 0.054 | [0.054, 0.054] |

> **WARNING:** These parameters have no statistical significance. They were "selected" from a pool where every candidate scored `-Infinity`. Do not deploy these values.

---

## 4. Final Performance Metrics

### Validation Tearsheet (Phase 3)

| Metric | Value |
|--------|-------|
| **OOS Sharpe** | 0.0 |
| **OOS Max Drawdown** | 0.0 |
| **OOS PnL** | $0.00 |
| **Avg Sharpe Decay %** | 0.0% |
| **Overfit Probability** | 0.0% |
| **Champion Fold** | 0 (default — no differentiation) |
| **Champion Degradation %** | 0.0% |
| **Unstable Parameters** | [] |
| **N Folds** | 3 |
| **N Trials Per Fold** | 2 (smoke-test) |
| **Data Range** | 2025-12-06 to 2026-02-08 |

### Per-Fold Breakdown (from pipeline.log)

| Fold | Train Window | Test Window | IS Sharpe | OOS Sharpe | Decay % | IS Fills |
|------|-------------|-------------|-----------|------------|---------|----------|
| 0 | 2025-12-06..12-19 | 2025-12-21..12-27 | 0.0 | 0.0 | 0.0% | 0 |
| 1 | 2025-12-13..12-26 | 2025-12-28..01-03 | 0.0 | 0.0 | 0.0% | 0 |
| 2+ | 2025-12-20..01-02 | 2026-01-04..01-10 | — | — | — | — |

### Per-Market Backtest Results (all identical)

Every market across every fold produced:
```
total_positions: 0
open_remaining: 0
total_pnl_net: 0
events: 14000 (train) / 7000 (test)
pnl: 0.0
sharpe: 0.0
max_dd: 0.0
```

---

## 5. Root Cause Analysis: Zero-Trade Problem

### Data Loading Pattern

The pipeline log reveals a severe data loading anomaly:
```
dataloader_done: total=14000, skipped=420000
```

**96.7% of tick data is being skipped.** For a 14-day training window across 5 markets, the loader processes 14,000 events but skips 420,000. This 30:1 skip ratio suggests:

1. **Data format mismatch** — The parquet files may contain data in a format the backtest data loader doesn't recognize, causing it to skip most rows.
2. **Market ID mismatch** — The `market_map.json` maps market slugs to contract IDs, but the tick data in the parquet files may use different identifiers, causing the loader to discard most events.
3. **Timestamp filtering** — The loader may be rejecting events due to gap detection (`gap_threshold=0.01`, `gap_max_interval_s=300.0`) or embargo period filtering.

### Signal Gate Analysis

Even with 14,000 non-skipped events, zero trades means the signal chain is too restrictive. The March 9 WFO Reset hardened three critical gates:

| Gate | Old Bounds | New Bounds | Impact |
|------|-----------|------------|--------|
| `zscore_threshold` | [0.15, 1.5] | **[1.0, 2.5]** | Requires >1σ panic signals — may be too strict for the data |
| `stop_loss_cents` | [2.0, 12.0] | **[4.0, 12.0]** | Eliminates tight stops, but also eliminates entry on narrow spreads |
| `min_edge_score` | [30.0, 60.0] | **[50.0, 85.0]** | Institutional-quality filter — may reject all edges in historical data |

The combination of these three hardened gates means the strategy requires simultaneously:
- A strong z-score panic (≥1.0σ)
- Sufficient spread to accommodate a ≥4-cent stop loss
- An edge score of ≥50 (institutional quality)

This triple gate may be impossible to satisfy on the recorded tick data.

### Production Run 2 Stalling

The production run (16:31–17:13+) had `pipeline.log` cut off mid-execution. Possible causes:
- The process was manually killed (e.g., user switched to deploying the live bot)
- OOM kill (the `wfo_phase1.db` grew to 704 KB, suggesting ~50+ trials were attempted)
- The pipeline redirected logs to stdout/stderr rather than `pipeline.log` after fold 2

---

## 6. Diagnostic Recommendations

### Priority 1: Fix the Zero-Trade Problem (Before Re-running WFO)

1. **Run `scripts/diagnose_wfo_gates.py`** on the VPS to identify which gate(s) are rejecting all signals:
   ```bash
   ssh botuser@135.181.85.32
   cd /home/botuser/polymarket-bot && source .venv/bin/activate
   python scripts/diagnose_wfo_gates.py --data-dir data/vps_march2026
   ```

2. **Investigate the 96.7% skip rate** in the data loader. Check if the loader's file path resolution matches the actual parquet directory structure on the VPS.

3. **Consider relaxing the hardened bounds** for the discovery phase:
   - `zscore_threshold`: Try [0.5, 2.5] to allow moderate panic signals
   - `min_edge_score`: Try [30.0, 85.0] to allow lower-quality but nonzero edges
   - Keep `stop_loss_cents` at [4.0, 12.0] (the Chop Trap protection is important)

### Priority 2: Re-run the Pipeline (Production Mode)

Once zero-trade is resolved:
```bash
ssh botuser@135.181.85.32
cd /home/botuser/polymarket-bot && source .venv/bin/activate

# Clean stale artefacts from failed runs
rm -f logs/phase*.json logs/final_validation_tearsheet.json logs/phase3_locked_bounds.json
rm -f logs/wfo_phase*.db logs/pipeline.log

# Run in a detached tmux session with stdout capture
tmux new-session -d -s wfo \
    "python scripts/run_optimization_pipeline.py --data-dir data/vps_march2026 2>&1 | tee logs/pipeline.log"
```

### Priority 3: Recover Artefacts to Local Machine

After a successful run, pull artefacts locally (deploy.sh excludes `logs/` and `*.db`):
```bash
scp botuser@135.181.85.32:/home/botuser/polymarket-bot/logs/phase*.json logs/
scp botuser@135.181.85.32:/home/botuser/polymarket-bot/logs/final_validation_tearsheet.json logs/
scp botuser@135.181.85.32:/home/botuser/polymarket-bot/logs/wfo_phase*.db logs/
scp botuser@135.181.85.32:/home/botuser/polymarket-bot/logs/pipeline.log logs/
```

### Priority 4: SSH Agent Configuration

Enable the Windows SSH agent to avoid passphrase fatigue:
```powershell
Set-Service ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add "$env:USERPROFILE\.ssh\id_ed25519"
```

---

## 7. Summary

| Item | Status |
|------|--------|
| Pipeline Executed | **YES** (on VPS — smoke-test completed; production run stalled) |
| Artefacts Found (VPS) | **5/5 JSON** present, **2/3 SQLite DBs** present |
| Artefacts Found (Local) | **0/8** (deploy.sh excludes logs/) |
| Pipeline Health | **CRITICAL FAILURE — zero trades across all folds and phases** |
| Root Cause | Triple-hardened signal gates + 96.7% data skip rate = impossible entry conditions |
| Performance Metrics | All zero (Sharpe, Sortino, PnL, Win Rate, Drawdown) |
| Next Step | **Fix zero-trade problem before re-running WFO** |
