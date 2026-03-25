# Polymarket Bot

Institutional documentation for the current checked-in trading stack.

This repository is no longer documented as a generic mean-reversion market
maker. The repo now contains multiple strategy families, but the approved
live posture is intentionally narrow:

- LIVE OPS: SI-9 combinatorial arbitrage only
- DEPLOYMENT MODE: PAPER
- RETIRED LIVE STRATEGY: PURE_MM passive market making
- NEW DIRECTIONAL RESEARCH STACK: OFI momentum taker built on the refactored
   O(1) OHLCV engine

Use this file as the operator-facing source of truth. Use ARCHITECTURE.md for
the full component-level explanation.

## Current State

### Live Ops

- SI-9 combinatorial arbitrage is the only strategy that should be treated as
   active live deployment.
- The checked-in service and environment are configured for PAPER mode.
- The live posture is intentionally separated from research and backtest-only
   capabilities.

### Backtest Ops

- The replay stack supports both the legacy passive maker path and the newer
   OFI momentum taker path.
- Walk-forward optimization now assumes the OHLCV engine is incremental and
   must be run with file logging disabled to avoid I/O bottlenecks.

## The Pivot

PURE_MM is deprecated as a live strategy.

The retirement reason is not cosmetic. The market split into two losing
regimes for passive quoting:

- Long-tail markets produced too few passive fills to justify capital or
   operational complexity. In practice, they behaved like zero-fill books.
- Top 25 / high-volume markets did fill, but those fills were dominated by
   toxic flow and adverse selection.

The repo still contains the PURE_MM implementation and replay adapter because
they remain useful for research, regression testing, and historical analysis.
That should not be confused with approval for live deployment.

For the current high-volume universe, recent out-of-sample WFO artifacts are
treated as disqualifying evidence for passive deployment. Operationally,
PURE_MM stays retired until the quoting model changes materially.

## Strategy Status Matrix

| Strategy | Status | Operational Meaning |
| --- | --- | --- |
| PURE_MM passive maker | Retired from live ops | Kept in code for replay, research, and historical comparison |
| SI-9 combinatorial arbitrage | Active live strategy | Approved live paper strategy on VPS |
| OFI momentum taker | Research / replay / rollout candidate | Implemented, wired, and documented, but not the primary live engine |
| Legacy panic / drift directional paths | Legacy compatibility | Still present in the repo, not the strategic center of live ops |

## Live Ops

### Approved Production Posture

The live VPS posture is:

- PAPER mode
- SI-9 enabled when running the combinatorial arb deployment
- PURE_MM disabled on the current Top 25 / high-volume target map

This distinction matters. The repo is a transitional system and still carries
legacy and optional code paths. Live approval is narrower than code presence.

### SI-9 Overview

SI-9 trades mutually exclusive negRisk event clusters.

Core idea:

- If the sum of YES best bids across all legs is below $1.00$ by enough
   margin, the event cluster can be bought as a Dutch-book style arb.

Operational guardrails:

- Ghost-town filter: reject clusters when $\sum bids < 0.85$
- Reject implausible edge prints when edge exceeds $\$0.15$
- Require minimum leg depth before sizing
- Enforce max concurrent combos, max total combo exposure, and per-combo
   collateral caps
- Use maker-first routing for the bottleneck leg, then sweep taker legs only
   after the maker leg fills
- If a combo partially fills, the hanging-leg state machine either emergency
   hedges the missing leg or dumps filled legs to eliminate naked exposure

Institutional interpretation:

- SI-9 is a tightly scoped execution strategy, not a generic market-making
   engine.
- Its risk control surface is explicit and leg-aware.

## Momentum Architecture

The repository now includes a new OFIMomentumSignal pipeline designed around
aggressive taker entry rather than passive quote capture.

### What It Does

- Tracks rolling top-of-book volume imbalance over a millisecond window
- Emits a directional momentum signal when rolling VI crosses a configured
   threshold
- Routes execution through an aggressive taker entry path instead of waiting
   for maker fills

### Why It Exists

The strategy is built for the regime where waiting passively is the problem.
If the signal is about fleeting order-flow imbalance, execution must cross the
spread immediately or the edge disappears.

### Hard Brackets

OFI momentum uses hard post-entry bracket logic:

- Take profit: 3.0%
- Stop-loss: 1.5%
- Time-stop: 300 seconds

Those brackets are deliberate guardrails, not optional tuning ideas. The time
stop exists because momentum that does not resolve quickly usually decays into
inventory risk rather than edge.

## Engine Optimization

The OHLCV stack was refactored to remove repeated full-window recomputation.

Current design:

- OHLCV bars update incrementally on each trade
- Rolling VWAP and rolling volatility maintain O(1) state updates
- Return sums, squared-return sums, and rolling volume sums are updated as bars
   enter and leave the window
- The backtest engine skips duplicate aggregation when the strategy manages
   its own trade aggregation

This is the key March 24 engineering change that made momentum WFO practical.
The prior model rebuilt rolling state too often and burned CPU on every tick.

Important replay rule:

- BotReplayAdapter aggregates its own trade stream
- The backtest engine detects that via `self_aggregates_trades = True`
- Engine-side duplicate OHLCV work is skipped

That separation prevents the replay stack from paying for the same bar work
twice.

## Walk-Forward Optimization

This repo now draws a hard operational line between LIVE OPS and BACKTEST OPS.
Do not mix their assumptions.

### Required Operator Rule

Before running WFO, disable file logging:

```powershell
$env:BOT_DISABLE_FILE_LOGGING='1'
```

Reason:

- WFO runs many short child-process backtests
- Structured file logging creates avoidable disk contention
- The repo explicitly supports disabling rotating file output while preserving
   stdout/stderr logging

### Trial Timeout Wrapper

Each trial is wrapped in a hard 60-second wall-clock timeout.

Operational meaning:

- A slow trial is not allowed to stall the study indefinitely
- The optimizer runs each trial in a child process
- If the child does not finish within 60 seconds, it is terminated and the
   trial is marked as timed out

This is a control mechanism, not just a convenience feature.

### Generic WFO Command

```powershell
$env:BOT_DISABLE_FILE_LOGGING='1'
python -m src.cli wfo `
   --data-dir data/vps_march2026 `
   --train-days 35 `
   --test-days 7 `
   --step-days 7 `
   --anchored `
   --embargo-days 1 `
   --n-trials 500 `
   --max-workers -1 `
   --strategy-adapter bot_replay
```

### Top 25 OFI Momentum Wrapper

```powershell
$env:BOT_DISABLE_FILE_LOGGING='1'
python wfo_ofi_momentum.py `
   --train-days 35 `
   --test-days 7 `
   --step-days 7 `
   --embargo-days 1 `
   --n-trials 500 `
   --max-workers -1 `
   --trial-timeout-s 60
```

### WFO Interpretation Rules

- Treat LIVE OPS and BACKTEST OPS as separate operating domains
- PURE_MM backtests require L2 snapshots and deltas; trade-only data is not a
   valid passive-maker evaluation dataset
- OFI momentum WFO is only useful if logging overhead is controlled and the
   OHLCV path stays incremental

## Repository Layout

```text
src/
├── backtest/      Replay engine, matching engine, WFO optimizer
├── core/          Config, logger, guards, process management
├── data/          Discovery, OHLCV, L2 books, adapters
├── execution/     Aggressive execution helpers such as momentum taker brackets
├── monitoring/    Trade store, health reporting, Telegram notifications
├── signals/       SI-9 detector, OFI momentum detector, legacy signal stack
├── strategies/    PURE_MM live strategy implementation
├── trading/       Position manager, stop-loss engine, combo lifecycle
├── bot.py         Runtime orchestrator
└── cli.py         Operator CLI
```

## Checked-In Code Versus Deployment Truth

This repo has experienced runtime-versus-source drift before. Documentation now
separates:

- checked-in architecture
- approved live deployment posture
- historical artifacts and saved runtime observations

If a piece of code still exists, that does not automatically mean it is part of
the approved live deployment.

That distinction is especially important for:

- PURE_MM defaults that still exist in config
- legacy directional code paths still present for compatibility
- SI-9 and OFI components that are wired in source but used differently across
   live paper, replay, and research workflows

## Further Reading

- See `ARCHITECTURE.md` for the full system architecture and rationale
- See `src/backtest/wfo_optimizer.py` for timeout and study orchestration
- See `src/data/ohlcv.py` for the incremental rolling-state implementation
