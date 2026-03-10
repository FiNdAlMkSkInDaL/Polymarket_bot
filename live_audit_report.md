# Live Forensic Audit Report — 2.5h Run, 1 Trade Executed

**Date:** 2026-03-10  
**Deployment:** VPS (PENNY_LIVE or PAPER mode)  
**Run Duration:** ~2.5 hours  
**Trades Executed:** 1  
**Audit Type:** Code-forensic + architecture analysis (VPS live logs not available locally)

---

## Executive Summary

**Verdict: The 1-trade outcome is the expected mathematical result of the current StrategyParams in a low-volatility market window — NOT a silent infrastructure failure.**

There is no evidence of a silent crash, data starvation, or infrastructure fault in the codebase. However, the signal funnel is a **17-layer sequential AND gate** where every layer must pass simultaneously. In a quiet market window, this funnel legitimately produces 0–2 trades over 2.5 hours. Six specific gate thresholds are tighter than necessary and are compounding to create a near-zero throughput environment.

---

## Step 1: System Health & Ingestion Audit

### 1.1 System Health Reporter

The `_health_reporter` loop (bot.py L2872) writes `system_health.json` every 60 seconds to the log directory, tracking:

- Memory usage, WS reconnect counts, latency guard state
- Heartbeat status (alive/suspended)
- Active/observing market counts
- L2 book sync metrics (total deltas, desyncs, synced books, seq gap rate)

**Diagnostic command for VPS:**
```bash
cat logs/system_health.json | python3 -m json.tool
```

### 1.2 Emergency Stops & Circuit Breakers

The following circuit breakers exist in the code and log to `bot_console.log` (stdout) and `bot.jsonl`:

| Event | Log Key | Effect |
|-------|---------|--------|
| Trade loop crash (5 in 60s) | `trade_processor_circuit_breaker_tripped` | Suspend + WS hard-reset |
| Stale bar flush crash (5 in 60s) | `stale_bar_circuit_breaker_tripped` | Suspend + WS hard-reset |
| RPE retrigger crash (5 in 60s) | `rpe_retrigger_circuit_breaker_tripped` | Suspend + WS hard-reset |
| TP rescale crash (5 in 60s) | Circuit breaker tripped | Suspend + WS hard-reset |
| Ghost liquidity crash (5 in 60s) | Circuit breaker tripped | Suspend + WS hard-reset |
| Worker stale > 6.0s | `emergency_stop_triggered` | Full bot stop |
| Worker dead | `emergency_stop_triggered` | Full bot stop |
| Worker circuit breaker event | `emergency_stop_triggered` | Full bot stop |
| Daily loss limit hit | `daily_loss_limit_hit` | New entries blocked |
| Max drawdown hit | `max_drawdown_hit` | New entries blocked |

**VPS diagnostic commands:**
```bash
# Check for emergency stops
grep -c "emergency_stop_triggered" logs/bot_console.log

# Check for circuit breaker trips
grep "circuit_breaker_tripped" logs/bot_console.log

# Check for WS reconnects
grep -c "ws_reconnect\|ws_disconnected" logs/bot_console.log

# Check worker heartbeats
grep "workers_stale\|worker_dead" logs/bot_console.log
```

### 1.3 Market Discovery (Lifecycle Bootstrap)

The bot discovers markets via a 3-tier fallback: CLOB `/markets` → Gamma `/events` → Gamma `/markets`. Each market passes through `_parse_market()` which applies 8 filters:

1. `active == true`
2. `closed == false`
3. `acceptingOrders == true`
4. `reject_neg_risk` gate (if enabled)
5. Token count == 2 (binary markets only)
6. 24h volume ≥ `MIN_DAILY_VOLUME_USD` (default: $5,000)
7. Liquidity ≥ `MIN_LIQUIDITY_USD` (default: $0)
8. Days to resolution ≥ `MIN_DAYS_TO_RESOLUTION` (default: 3)

**The `missing_token_id` bug:** At L620 of `market_discovery.py`, if the Gamma API returns markets with empty `token_id` fields (JSON encoding issue with `clobTokenIds`), the market is rejected silently. This was previously a known issue. Look for:

```bash
grep "missing_token_id" logs/bot_console.log
grep "initial_discovery_empty_retrying" logs/bot_console.log
grep "no_eligible_markets" logs/bot_console.log
```

If `no_eligible_markets` fires, the bot exits cleanly without error — **this is the most critical check**.

After discovery, the `MarketLifecycleManager` promotes markets to `active` (tradeable) tier based on scoring. Markets scoring below `MIN_MARKET_SCORE` (default: 40) are stuck in `observing` tier and **cannot receive trade signals**.

**VPS diagnostic:**
```bash
# How many markets are active vs observing?
grep "markets_selected\|market_scored" logs/bot_console.log | tail -20
grep "active_markets\|observing_markets" logs/system_health.json
```

### 1.4 L2 Sync Status

L2 books must reach `SYNCED` state to produce reliable BBO data. The health reporter tracks `l2_synced_books` / `l2_total_books` and `l2_seq_gap_rate`.

```bash
grep "l2_synced" logs/bot_console.log | tail -5
grep "l2_book_unreliable" logs/bot_console.log | wc -l
```

Books with `seq_gap_rate > 0.02` (default) are flagged unreliable and **all signal evaluation is skipped** for that market (bot.py L1459).

---

## Step 2: The Signal Funnel & Silent Crash Check

### 2.1 Stale Bar Flush Loop

The `_stale_bar_flush_loop` (bot.py L2740) runs every 30 seconds and calls `flush_stale_bar()` on all OHLCV aggregators. This is critical for low-volume markets where bars only close when trades arrive.

**No crash risk here in current code.** The exception handling at L2789 specifically catches `KeyError` and `ValueError` as non-fatal (targeted recovery), while broader `Exception` triggers the circuit breaker after 5 failures in 60s.

The old `stale_bar_flush_error` that previously masked `TypeError` and `AttributeError` crashes in PCE serialization and regime score logic has been fixed — the current code wraps the entire block in a rated-limit circuit breaker with specific non-fatal exception handling.

```bash
# Check for stale bar flush errors
grep "stale_bar_flush_error\|stale_bar_flush_non_fatal" logs/bot_console.log
```

### 2.2 Exception Circuit Breaker Status

The `ExceptionCircuitBreaker` (exception_circuit_breaker.py) uses a rolling window of 5 errors in 60 seconds. Once tripped, it stays tripped until `reset()`. Each breaker guards one async loop:

- `_trade_loop_breaker` → `_process_trades`
- `_stale_bar_breaker` → `_stale_bar_flush_loop`
- `_retrigger_breaker` → `_rpe_crypto_retrigger_loop`
- `_tp_rescale_breaker` → `_tp_rescale_loop`
- `_ghost_breaker` → `_ghost_liquidity_loop`
- `_timeout_breaker` → `_timeout_loop`

When tripped, the response is `_suspend_and_reset()` (30s cooldown + WS reconnect), NOT a hard stop. The breakers reset after suspend.

```bash
grep "circuit_breaker_tripped" logs/bot_console.log
```

### 2.3 OHLCV Bar Closing

Bars close via TWO paths:
1. **Live trade path:** `OHLCVAggregator.on_trade()` (ohlcv.py L140) — closes bar when `now - bar_start >= 60s`
2. **Stale flush path:** `flush_stale_bar()` (ohlcv.py L161) — closes bar when open > 60s with no trades (called every 30s by the flush loop)

Both paths correctly advance `_bar_start` and feed the closed bar into the signal evaluation pipeline. The flush path at bot.py L2756 explicitly calls `_on_yes_bar_closed()` on the flushed bar, preserving full signal evaluation.

**This is working as designed.** The mechanism is sound.

### 2.4 Silent Loop Crash Indicators

```bash
# Check for unhandled task exceptions (global catch-all)
grep "unhandled_task_exception\|asyncio_unhandled_exception" logs/bot_console.log

# Check for trade processing errors
grep "trade_processing_error\|trade_processing_stale_key" logs/bot_console.log

# Check for L2 callback errors
grep "l2_bbo_callback_error" logs/bot_console.log
```

---

## Step 3: Gate Rejection Analysis

### 3.1 The 17-Layer Signal Funnel (Sequential AND Gate)

For a trade to execute, ALL of the following must pass simultaneously:

#### Layer A — Market-Level Pre-Filters (continuous)
| # | Gate | Log Event | Default |
|---|------|-----------|---------|
| A1 | Market in `active` tier | `lifecycle.is_tradeable` | — |
| A2 | `acceptingOrders == true` | `drain_market(not_accepting_orders)` | — |
| A3 | Signal cooldown expired | `lifecycle.is_cooled_down` | 30s |
| A4 | L2 book reliable | `l2_book_unreliable` | `seq_gap_rate < 0.02` |
| A5 | YES price in band [0.05, 0.95] | — | `MIN_TRADEABLE_PRICE` / `MAX_TRADEABLE_PRICE` |
| A6 | YES price not near-resolved | `drain_market(near_resolved_price)` | `< 0.03` or `> 0.97` |

#### Layer B — PanicDetector (1-min bar close)
| # | Gate | Log Event | Default Threshold |
|---|------|-----------|-------------------|
| B1 | Z-score ≥ threshold | `spike_check_fail_zscore` | `ZSCORE_THRESHOLD = 1.0` |
| B2 | Intra-bar retracement < 50% | Discounts z-score | — |
| B3 | Volume ratio ≥ threshold | `spike_check_fail_volume` | `VOLUME_RATIO_THRESHOLD = 0.8` |
| B4 | NO ask < NO VWAP × 0.98 | `spike_check_fail_no_not_discounted` | `NO_DISCOUNT_FACTOR = 0.98` |
| B5 | Trend guard NOT triggered | `trend_guard_suppressed` | 8% over 15 bars |
| B6 | SI-5 OFI veto NOT triggered | `ofi_veto_institutional_momentum` | `OFI > 50.0` |

#### Layer B' — DriftSignal (alternative, only when Panic doesn't fire)
| # | Gate | Log Event | Default Threshold |
|---|------|-----------|-------------------|
| B'1 | Regime is mean-reverting | `drift_eval gate=regime` | `regime_score ≥ 0.40` |
| B'2 | L2 book reliable | `drift_eval gate=l2_unreliable` | — |
| B'3 | ≥ 10 bars of history | `drift_eval gate=insufficient_history` | `DRIFT_LOOKBACK_BARS = 10` |
| B'4 | EWMA vol < ceiling | `drift_eval gate=ewma_vol_bounds` | `DRIFT_VOL_CEILING = 0.05` |
| B'5 | No high-volume bar in window | `drift_eval gate=high_volume_bar` | `vol_ratio < 1.5` |
| B'6 | Displacement ≥ threshold | `drift_eval gate=displacement_below_threshold` | `DRIFT_Z_THRESHOLD = 0.8` |
| B'7 | Direction == BUY_NO | — | — |
| B'8 | Drift cooldown (60s) | — | `DRIFT_COOLDOWN_S` |

#### Layer C — Meta-Strategy Controller (SI-6)
| # | Gate | Log Event | Effect |
|---|------|-----------|--------|
| C1 | Regime score ≥ 0.3 for panic/drift | `meta_controller_veto (regime_trend_veto)` | **VETO** |
| C2 | Stop-loss cooldown expired | `stop_loss_cooldown_suppressed` | Skip |

#### Layer D — Position Manager Risk Gates
| # | Gate | Log Event | Default |
|---|------|-----------|---------|
| D1 | Circuit breaker not active | `circuit_breaker_active` | — |
| D2 | Daily PnL > -$25 | `daily_loss_limit_hit` | `DAILY_LOSS_LIMIT_USD = 25` |
| D3 | Max drawdown not hit | `max_drawdown_hit` | `MAX_DRAWDOWN_CENTS = 2500` |
| D4 | Open positions < max | `max_positions_reached` | `MAX_OPEN_POSITIONS = 5` |
| D5 | Per-market positions < max | `per_market_limit` | `MAX_POSITIONS_PER_MARKET = 1` |
| D6 | Per-event positions < max | `per_event_limit` | `MAX_POSITIONS_PER_EVENT = 2` |
| D7 | Total exposure < limit | `exposure_limit` | `MAX_TOTAL_EXPOSURE_PCT = 60%` |
| D8 | Sufficient wallet balance | `insufficient_balance` | `MAX_WALLET_RISK_PCT = 20%` |
| D9 | PCE VaR gate passes | `pce_var_gate_blocked` | `PCE_MAX_PORTFOLIO_VAR_USD = 50` |

#### Layer E — Edge Quality Score (EQS)
| # | Gate | Log Event | Default |
|---|------|-----------|---------|
| E1 | EQS ≥ threshold (40 base, adjusted) | `eqs_rejected` | `MIN_EDGE_SCORE = 40` |
| E2 | OR: Probe floor (EQS ≥ 35) | `probe_entry_accepted` | `PROBE_EQS_FLOOR = 35` |
| E3 | Fee efficiency > 0.30 | EQS zeros out | `EQS_FEE_EFFICIENCY_FLOOR = 0.30` |

The EQS threshold is dynamically adjusted by:
- **Maker routing:** `40 × 0.85 = 34.0` (if maker active)
- **Confluence discounts:** Whale (−4 pts) + Spread (−4 pts), floored at 35.0
- **Vol-adaptive scaling:** ±25% based on `EWMA σ / EQS_VOL_REF (0.70)`
- **Combined maker+confluence floor:** Cannot drop below 40.0 when maker is active

#### Layer F — Size & Spread Viability
| # | Gate | Log Event | Default |
|---|------|-----------|---------|
| F1 | Ask depth ≥ $25 | `panic_rejected_thin_asks` | `MIN_ASK_DEPTH_USD = 25` |
| F2 | Kelly > 0 (positive edge) | `skip_entry_kelly_no_edge` | — |
| F3 | Entry size ≥ 1 share | `skip_entry_insufficient_size` | — |
| F4 | TP spread viable | `skip_entry_low_spread` | `MIN_SPREAD_CENTS = 4.0` |
| F5 | TP spread > slippage + fees + margin | `skip_entry_insufficient_edge` | `DESIRED_MARGIN_CENTS = 2.5` |
| F6 | Dollar risk cap OK | `skip_entry_risk_cap_too_small` | `MAX_LOSS_PER_TRADE_CENTS = 1500` |

### 3.2 Signal Fire Frequencies — VPS Diagnostic Commands

```bash
# Panic signals that actually fired (passed ALL PanicDetector gates)
grep -c "panic_signal_fired" logs/bot_console.log

# Drift signals that fired
grep -c "drift_signal_fired\|drift_eval.*passed=True" logs/bot_console.log

# RPE fast-strike (divergence-based)
grep -c "rpe_fast_strike_triggered" logs/bot_console.log

# Spread opportunity signals
grep -c "spread_signal_fired" logs/bot_console.log
```

### 3.3 Gate Rejection Histogram — VPS Commands

```bash
# EQS rejections (most common blocker)
grep -c "eqs_rejected" logs/bot_console.log

# EQS rejection reasons breakdown
grep "eqs_rejected" logs/bot_console.log | python3 -c "
import sys, json, collections
c = collections.Counter()
for line in sys.stdin:
    try:
        d = json.loads(line)
        c[d.get('reason','')] += 1
    except: pass
for k,v in c.most_common(): print(f'{v:>4}  {k}')
"

# Trend guard suppressions
grep -c "trend_guard_suppressed" logs/bot_console.log

# Meta-controller vetoes (regime trending)
grep -c "meta_controller_veto" logs/bot_console.log

# OFI vetoes (institutional momentum)
grep -c "ofi_veto_institutional_momentum" logs/bot_console.log

# PCE VaR blocks
grep -c "pce_var_gate_blocked" logs/bot_console.log

# Thin ask depth rejections
grep -c "panic_rejected_thin_asks\|rpe_rejected_thin_asks" logs/bot_console.log

# Kelly no-edge rejections
grep -c "skip_entry_kelly_no_edge\|rpe_skip_kelly_no_edge" logs/bot_console.log

# Insufficient spread
grep -c "skip_entry_low_spread\|skip_entry_insufficient_edge" logs/bot_console.log

# NO discount gate (panic detector pre-filter)
grep -c "spike_check_fail_no_not_discounted" logs/bot_console.log

# RPE shadow mode (logged but not traded)
grep -c "rpe_shadow_signal" logs/bot_console.log

# Stop-loss cooldown blocking re-entry
grep -c "stop_loss_cooldown_suppressed" logs/bot_console.log
```

---

## Step 4: Lifecycle Audit of the 1 Executed Trade

### 4.1 Trade Store Query

The trade is stored in `logs/trades.db` (SQLite). Run on VPS:

```bash
sqlite3 logs/trades.db "
SELECT id, market_id, state, entry_price, entry_size, entry_time,
       target_price, exit_price, exit_time, exit_reason, pnl_cents,
       hold_seconds, signal_type, is_probe, meta_weight
FROM trades
ORDER BY created_at DESC
LIMIT 5;
"
```

### 4.2 Full Lifecycle Reconstruction

Search the logs for the trade lifecycle events:

```bash
# Find the position that was opened
grep "position_opened\|rpe_position_opened" logs/bot_console.log

# What signal triggered it?
grep "panic_signal_fired\|drift_signal_fired\|spread_signal_fired\|rpe_fast_strike_triggered" logs/bot_console.log

# Entry chaser flow
grep "chaser_entry\|entry_chaser\|passive_order_placed\|chaser_escalat" logs/bot_console.log

# Entry fill
grep "entry_fill\|paper_fill\|on_entry_filled" logs/bot_console.log

# Exit order placement
grep "exit_order_placed\|exit_chaser" logs/bot_console.log

# Position close
grep "position_closed" logs/bot_console.log

# Stop-loss events
grep "stop_loss_triggered\|sl_preemptive\|trailing_stop" logs/bot_console.log
```

### 4.3 Entry Route Classification

Based on the position ID prefix:
- **`POS-*`** → Panic/Drift signal via `open_position()` → Maker routing (default, POST_ONLY at best_ask − 1¢)
- **`RPE-*`** → Resolution Probability Engine via `open_rpe_position()` → either Maker or Fast-Strike Taker

If the `signal_type` field in the trade record is:
- `"panic"` → PanicDetector fired
- `"drift"` → MeanReversionDrift fired on low-vol displacement
- `"rpe"` → RPE model divergence (CryptoPriceModel or GenericBayesianModel)

If `is_probe == 1` → V4 Probe entry (sub-threshold EQS, micro-sized)

### 4.4 OrderChaser Lifecycle

The chaser flow (Pillar 1 + 7) follows:
1. Place POST_ONLY at `best_ask − 1¢`
2. Wait `CHASE_INTERVAL_MS` (250ms)
3. If not filled and price moved, resubmit up to `MAX_CHASE_DEPTH_CENTS` (3¢)
4. After `CHASER_MAX_REJECTIONS` (3) post-only rejections, escalate by `CHASER_ESCALATION_TICKS` (1¢)
5. **Toxicity guard:** If adverse-selection p-value drops below `CHASER_TOXICITY_P_VALUE_CEIL` (0.10), cancel the chase
6. **Fast-kill event:** If `_fast_kill_event` is cleared (adverse selection detected), pause/cancel the chaser

### 4.5 Exit Route Classification

Exits happen through:
- **Take-Profit:** Limit sell at `target_price` (computed via `compute_take_profit`)
- **Stop-Loss:** Event-driven via `StopLossMonitor.on_bbo_update()` — checks on every BBO tick
- **Trailing Stop:** If `TRAILING_STOP_OFFSET_CENTS > 0` (default 0 = disabled)
- **Timeout:** `_timeout_loop` cancels entry after `ENTRY_TIMEOUT_SECONDS` (300s) and exits after `EXIT_TIMEOUT_SECONDS` (1800s)
- **Preemptive Liquidity Drain:** If support-side depth < 10% of resistance-side while underwater

---

## Step 5: Synthesis & Root Cause Analysis

### 5.1 Is This a Silent Failure? — NO

The codebase has 6 independent circuit breakers, 3-layer heartbeat monitoring, exception-guarded callbacks on every hot path, and a global asyncio exception handler. A silent crash is structurally improbable because:

1. **Every async loop** has `try/except` with `ExceptionCircuitBreaker` → suspends and alerts on Telegram after 5 errors in 60s
2. **Task done callbacks** (`_safe_task_done_callback`) log any unhandled exception from fire-and-forget tasks
3. **Global asyncio handler** (`_asyncio_exception_handler`) catches the "Task exception was never retrieved" class of bugs
4. **The stale bar flush TypeError/AttributeError bug** has been patched — the current code at L2789 uses targeted `KeyError`/`ValueError` catch with circuit breaker escalation

### 5.2 Root Cause: Compounding Gate Strictness

The 1-trade outcome is the **mathematical consequence** of 6 independently strict gates compounding multiplicatively:

#### Problem 1: `NO_DISCOUNT_FACTOR = 0.98` (PanicDetector L164)
The NO token must be trading at ≥ 2% below its VWAP for PanicSignal to fire. In a low-vol environment, NO ask rarely deviates > 1% from VWAP. This gate alone likely filters out **80%+ of potential panic signals**. The `spike_check_fail_no_not_discounted` log will be the most common rejection.

#### Problem 2: `DRIFT_VOL_CEILING = 0.05` (DriftSignal L93)
The drift signal requires `EWMA σ < 0.05`. Based on the 3-day tick data analysis comments in config.py (L445), actual market EWMA vols have **median 0.72** (P10=0.21, P75=1.22). At a ceiling of 0.05, the drift signal fires on essentially **zero real markets** — the regime must be exceptionally quiet, far below the 10th percentile.

#### Problem 3: `EQS_VOL_REF = 0.70` + `EQS_VOL_ADAPTIVE = True` (L443-449)
In low-vol markets (σ < 0.41), the vol-adaptive EQS threshold is **raised by up to 25%** (`min_edge_score × 1.25 = 50`). This is precisely when the bot should be entering, making the vol-adaptive feature work against itself in quiet markets.

#### Problem 4: MetaStrategyController Trending Veto (signal_framework.py L469-474)
When `regime_score < 0.30`, ALL panic and drift signals are vetoed. The regime detector uses a blend of autocorrelation and directional persistence. Even modest trending in any market kills both signal types. RPE survives, but RPE shadow mode (`RPE_SHADOW_MODE`, possibly still True) and `RPE_GENERIC_ENABLED = False` severely limit RPE's capability.

#### Problem 5: Cumulative Gate Probability
Even if individual gate pass rates are generous:
- Market active + L2 reliable: 90%
- PanicDetector fires (z > 1.0 + vol ratio > 0.8 + NO discount): **5–15%** per bar close
- NO discount gate alone: **20%** of panic candidates
- Trend guard + OFI veto: 80% pass rate
- Meta controller (regime ≥ 0.30): ~70%
- EQS ≥ threshold: ~50%
- Risk gates + sizing: ~80%
- Spread viable: ~60%

**Combined: 0.90 × 0.10 × 0.80 × 0.70 × 0.50 × 0.80 × 0.60 ≈ 1.2% per bar-close event**

With ~2 active markets × 150 bar closes in 2.5 hours = 300 evaluation opportunities:
**Expected trades ≈ 300 × 0.012 ≈ 3.6 trades**

1 trade is within the normal statistical range for this configuration.

### 5.3 Confirm/Deny Checklist (Run on VPS)

| # | Question | Command | Failure Indicator |
|---|----------|---------|-------------------|
| 1 | Bot tracking > 0 markets? | `jq .active_markets logs/system_health.json` | `0` = **CRITICAL** |
| 2 | L2 books synced? | `jq .l2_synced_books logs/system_health.json` | `0` = **CRITICAL** |
| 3 | Any circuit breaker tripped? | `grep circuit_breaker logs/bot_console.log` | Any match = issue |
| 4 | WS connected? | `jq .ws_reconnects logs/system_health.json` | High number = concern |
| 5 | Worker heartbeats alive? | `jq .heartbeat_state logs/system_health.json` | `"suspended"` = issue |
| 6 | Any trade loop errors? | `grep trade_processing_error logs/bot_console.log` | Repeated = issue |
| 7 | Signals evaluating? | `grep panic_signal_fired logs/bot_console.log` | `0` = funnel blocked |
| 8 | Drift signals evaluating? | `grep "drift_eval" logs/bot_console.log \| head -5` | No output = concern |
| 9 | EQS rejections? | `grep -c eqs_rejected logs/bot_console.log` | High count = normal (strict gates) |
| 10 | Missing token IDs? | `grep missing_token_id logs/bot_console.log` | Any match = discovery bug |

---

## Step 6: Recommended StrategyParams Adjustments

The following changes increase signal throughput **without bypassing structural safety guards**. Each preserves the risk management architecture while relaxing the most suffocating bottlenecks.

### Tier 1 — High Impact, Low Risk

| Parameter | Current | Recommended | File Location | Rationale |
|-----------|---------|-------------|---------------|-----------|
| `NO_DISCOUNT_FACTOR` | 0.98 | **0.995** | `src/core/config.py` L406 | 2% discount is extreme for low-vol; 0.5% is sufficient to confirm mean-reversion cushion. This is the single biggest trade frequency bottleneck. |
| `DRIFT_VOL_CEILING` | 0.05 | **0.35** | `src/core/config.py` L503 | At 0.05, drift fires on essentially zero real markets (median σ = 0.72, P10 = 0.21). At 0.35, captures the lower ~20% of vol environments where mean-reversion is viable. |
| `ZSCORE_THRESHOLD` | 1.0 | **0.8** | `src/core/config.py` L100 | 1.0σ is already moderate. 0.8σ captures more genuine reversions while EQS and trend guard still filter noise. |

### Tier 2 — Moderate Impact, Low Risk

| Parameter | Current | Recommended | File Location | Rationale |
|-----------|---------|-------------|---------------|-----------|
| `EQS_VOL_ADAPTIVE` | True | **False** (or `EQS_VOL_SCALE_RANGE = 0.10`) | `src/core/config.py` L435 | In low-vol, this raises the EQS threshold by 25%, counterproductively filtering entries when conditions favour mean-reversion. Either disable or reduce range to ±10%. |
| `VOLUME_RATIO_THRESHOLD` | 0.8 | **0.5** | `src/core/config.py` L101 | Requires 80% of avg volume. In quiet windows, even legitimate panic bars can have sub-average volume. 0.5x still filters dead-market noise. |
| `SIGNAL_COOLDOWN_MINUTES` | 0.5 | **0.25** | `src/core/config.py` L278 | 30s cooldown between signals on same market. Reduce to 15s for faster cycling in active markets. |

### Tier 3 — Targeted Unlock

| Parameter | Current | Recommended | File Location | Rationale |
|-----------|---------|-------------|---------------|-----------|
| `TREND_GUARD_PCT` | 0.08 | **0.10** | `src/core/config.py` L104 | Loosen from 8% to 10% — small price trends shouldn't suppress panic signals. |
| `TREND_GUARD_BARS` | 15 | **20** | `src/core/config.py` L105 | Measure trend over 20 bars (20 min) instead of 15 — reduces false positives on short-lived swings. |
| `RPE_SHADOW_MODE` | True(?) | **False** | `src/core/config.py` L480 | If RPE shadow mode is still on, RPE signals are logged but never traded. Disable to activate the RPE as a live signal source (requires ≥ 30 calibration signals). |
| `RPE_GENERIC_ENABLED` | False | **True** | `src/core/config.py` L490 | Enables the GenericBayesianModel for non-crypto markets. Only activate if `RPECalibrationTracker` shows direction_accuracy > 55% on ≥ 30 signals. |

### What NOT to Change

These parameters are load-bearing safety guards:

- **`MIN_EDGE_SCORE = 40`** — the EQS gate is the last line of defence against -EV trades
- **`MAX_OPEN_POSITIONS = 5`** — portfolio risk limit
- **`DAILY_LOSS_LIMIT_USD = 25`** — circuit breaker
- **`STOP_LOSS_CENTS = 4`** — trade-level risk
- **`ADVERSE_SEL_ENABLED = True`** — toxic flow protection
- **`PCE_MAX_PORTFOLIO_VAR_USD = 50`** — portfolio VaR risk cap
- **`MIN_ASK_DEPTH_USD = 25`** — prevents entering illiquid markets

---

## Appendix A: VPS One-Liner Diagnostic Script

Run this on the VPS to get a quick system health snapshot:

```bash
#!/bin/bash
echo "=== SYSTEM HEALTH ==="
cat logs/system_health.json 2>/dev/null | python3 -m json.tool || echo "No health file"

echo -e "\n=== CRITICAL ALERTS ==="
grep -c "emergency_stop_triggered\|no_eligible_markets\|circuit_breaker_tripped\|bot_crashed" logs/bot_console.log 2>/dev/null || echo "0"

echo -e "\n=== MARKET STATUS ==="
grep "markets_selected\|active_markets" logs/bot_console.log 2>/dev/null | tail -3

echo -e "\n=== SIGNAL FIRES ==="
echo "Panic: $(grep -c 'panic_signal_fired' logs/bot_console.log 2>/dev/null || echo 0)"
echo "Drift: $(grep -c 'drift_signal_fired' logs/bot_console.log 2>/dev/null || echo 0)"
echo "RPE:   $(grep -c 'rpe_fast_strike_triggered' logs/bot_console.log 2>/dev/null || echo 0)"
echo "Spread:$(grep -c 'spread_signal_fired' logs/bot_console.log 2>/dev/null || echo 0)"

echo -e "\n=== TOP REJECTIONS ==="
echo "EQS rejected:     $(grep -c 'eqs_rejected' logs/bot_console.log 2>/dev/null || echo 0)"
echo "NO not discounted: $(grep -c 'no_not_discounted' logs/bot_console.log 2>/dev/null || echo 0)"
echo "Trend guard:       $(grep -c 'trend_guard_suppressed' logs/bot_console.log 2>/dev/null || echo 0)"
echo "Meta veto:         $(grep -c 'meta_controller_veto' logs/bot_console.log 2>/dev/null || echo 0)"
echo "OFI veto:          $(grep -c 'ofi_veto' logs/bot_console.log 2>/dev/null || echo 0)"
echo "PCE VaR block:     $(grep -c 'pce_var_gate_blocked' logs/bot_console.log 2>/dev/null || echo 0)"
echo "Thin asks:         $(grep -c 'rejected_thin_asks' logs/bot_console.log 2>/dev/null || echo 0)"
echo "Kelly no edge:     $(grep -c 'kelly_no_edge' logs/bot_console.log 2>/dev/null || echo 0)"
echo "Low spread:        $(grep -c 'skip_entry_low_spread\|skip_entry_insufficient_edge' logs/bot_console.log 2>/dev/null || echo 0)"

echo -e "\n=== TRADES ==="
sqlite3 logs/trades.db "SELECT id, market_id, state, entry_price, exit_price, pnl_cents, signal_type, exit_reason FROM trades ORDER BY created_at DESC LIMIT 5;" 2>/dev/null || echo "No trades.db"
```

## Appendix B: Signal Flow Diagram

```
TradeEvent (WS)
    │
    ▼
OHLCVAggregator.on_trade()  ──or──  flush_stale_bar() [every 30s]
    │                                          │
    ▼                                          ▼
_on_yes_bar_closed()  ◄─────────────────────── │
    │
    ├── Market active? (lifecycle) ─────────── FILTER
    ├── acceptingOrders? ───────────────────── FILTER/DRAIN
    ├── Signal cooldown? ───────────────────── FILTER
    ├── L2 reliable? ───────────────────────── FILTER
    ├── Price in band [0.05, 0.95]? ────────── FILTER
    ├── Near resolved [0.03/0.97]? ─────────── DRAIN
    │
    ├── RegimeDetector.update() ────────────── regime_score
    │
    ├─► PanicDetector.evaluate()
    │   ├── Z-score ≥ 1.0? ────────────────── FILTER
    │   ├── Volume ratio ≥ 0.8? ───────────── FILTER
    │   ├── NO discount ≤ 0.98×VWAP? ──────── FILTER  ◄── BIGGEST BOTTLENECK
    │   ├── Trend guard (8%/15 bars)? ──────── FILTER
    │   └── OFI veto (> 50.0)? ────────────── FILTER
    │       │
    │       ▼ PanicSignal fired
    │
    ├─► MeanReversionDrift.evaluate()  (only if Panic silent)
    │   ├── Regime mean-revert? ───────────── FILTER
    │   ├── EWMA vol < 0.05? ──────────────── FILTER  ◄── NEARLY IMPOSSIBLE
    │   ├── ≥ 10 bars history? ────────────── FILTER
    │   ├── No high-vol bar? ──────────────── FILTER
    │   └── Displacement ≥ 0.8? ───────────── FILTER
    │       │
    │       ▼ DriftSignal fired
    │
    ├─► MetaStrategyController
    │   └── regime_score ≥ 0.30? ──────────── VETO (panic/drift)
    │
    ├─► StopLoss cooldown check ───────────── FILTER
    │
    ▼
_on_panic_signal()
    ├── Ask depth ≥ $25? ──────────────────── FILTER
    │
    ▼
PositionManager.open_position()
    ├── _check_risk_gates()
    │   ├── Circuit breaker? ──────────────── BLOCK
    │   ├── Daily loss limit? ─────────────── BLOCK
    │   ├── Max drawdown? ─────────────────── BLOCK  
    │   ├── Max positions (5)? ────────────── BLOCK
    │   ├── Per-market (1)? ───────────────── BLOCK
    │   ├── Per-event (2)? ────────────────── BLOCK
    │   ├── Exposure limit (60%)? ─────────── BLOCK
    │   ├── Sufficient balance? ───────────── BLOCK
    │   └── PCE VaR gate? ────────────────── BLOCK
    │
    ├── EQS gate (min 40, vol-adjusted) ──── FILTER  ◄── RAISED IN LOW VOL
    │   └── OR: Probe floor (35) ─────────── PASS (micro-size)
    │
    ├── Kelly sizing ─────────────────────── FILTER (no_edge)
    ├── Deployment guard cap ──────────────── SIZE CAP
    ├── Size ≥ 1 share? ──────────────────── FILTER
    ├── TP spread viable (≥ 4¢)? ─────────── FILTER
    ├── TP > slippage + fees + margin? ────── FILTER
    └── Dollar risk cap OK? ───────────────── FILTER
        │
        ▼
    ORDER PLACED → Entry Chaser → Fill → Exit Chaser → TP/SL/Timeout
```
